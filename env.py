"""Trace Explorer Environment - Analyze HUD traces using coding tools.

This environment:
- Loads trace data from HUD platform (telemetry, environment logs, worker logs)
- Writes data to files for agent exploration
- Provides bash, grep, read, and edit tools for analysis
- Evaluates responses against includes/excludes patterns
"""

import base64
import json
import logging
import os
import sys
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# SDK workaround: _build_answer_for_generator loses flat dicts (no "content"
# key) because it does dict.get("content", "").  Convert to JSON string so the
# SDK takes the string path instead.
# ---------------------------------------------------------------------------
import hud.environment.scenarios as _hud_scenarios
from dotenv import load_dotenv
from httpx import AsyncClient
from hud import Environment

# Claude/OpenCode-style tools
from hud.tools import BashTool, EditTool

# Gemini-style tools
from hud.tools.coding import GeminiEditTool, GeminiShellTool
from hud.tools.filesystem import (
    GeminiGlobTool,
    GeminiListTool,
    GeminiReadTool,
    GeminiSearchTool,
    GlobTool,
    GrepTool,
    ListTool,
    ReadTool,
)
from mcp.types import ImageContent, TextContent

# ---------------------------------------------------------------------------
# Verification sub-agent: independently re-checks claims from QA analysis
# ---------------------------------------------------------------------------
from qa_alignment_subagents import alignment_env
from qa_verification import verify_env

load_dotenv()

_orig_build_answer = _hud_scenarios._build_answer_for_generator


def _build_answer_compat(session):  # type: ignore[no-untyped-def]
    raw = session.answer
    if isinstance(raw, dict) and "content" not in raw:
        session.answer = json.dumps(raw)
    return _orig_build_answer(session)


_hud_scenarios._build_answer_for_generator = _build_answer_compat

# Configure logging to stderr (MCP uses stdout)
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
    force=True,
)
for logger_name in ["httpx", "httpcore"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Workspace directory for trace data
# In Docker: /workspace exists and is the container workspace
# Locally: use ./workspace relative to current directory
if Path("/workspace").exists():
    WORKSPACE_DIR = Path("/workspace")
    BASE_PATH = "/workspace"
else:
    # Local development - use relative workspace
    WORKSPACE_DIR = Path("./workspace").resolve()
    BASE_PATH = str(WORKSPACE_DIR)

# Create the environment
env = Environment(name="trace-explorer")

# Add Claude/OpenCode-style tools (with base_path for sandboxing)
env.add_tool(BashTool())
env.add_tool(EditTool())
env.add_tool(ReadTool(base_path=BASE_PATH))
env.add_tool(GrepTool(base_path=BASE_PATH))
env.add_tool(GlobTool(base_path=BASE_PATH))
env.add_tool(ListTool(base_path=BASE_PATH))

# Add Gemini-style tools (with base_path/base_directory for sandboxing)
env.add_tool(GeminiShellTool(base_directory=BASE_PATH))
env.add_tool(GeminiEditTool(base_directory=BASE_PATH))
env.add_tool(GeminiReadTool(base_path=BASE_PATH))
env.add_tool(GeminiSearchTool(base_path=BASE_PATH))
env.add_tool(GeminiGlobTool(base_path=BASE_PATH))
env.add_tool(GeminiListTool(base_path=BASE_PATH))

# Sub agent environments
# Verification
_verify_task = verify_env("verify_claims")
_VERIFY_MODEL = "claude-sonnet-4-5"
_VERIFY_MAX_STEPS = 50

# Alignment
_alignment_extract_task = alignment_env("extract_prompt_requirements")
_alignment_grader_task = alignment_env("map_grader_checks")
_alignment_submission_task = alignment_env("assess_submission_coverage")
_alignment_arbiter_task = alignment_env("arbitrate_alignment")
_ALIGNMENT_MODEL = "claude-sonnet-4-5"
_ALIGNMENT_MAX_STEPS = 50


def _parse_verification_output(text: str) -> dict[str, Any]:
    """Parse the verifier's structured output into claim statuses and overall verdict.

    Tries JSON extraction first (more robust), falls back to regex on the
    legacy markdown format.
    """
    import re as _re

    from qa_common import _extract_json_object

    json_str = _extract_json_object(text) if text else None
    if json_str:
        try:
            parsed_json = json.loads(json_str)
            if isinstance(parsed_json, dict) and "claims" in parsed_json:
                claims = []
                for c in parsed_json["claims"]:
                    if isinstance(c, dict) and "status" in c:
                        claims.append(
                            {
                                "claim": c.get("claim", ""),
                                "status": c["status"].upper(),
                                "reason": c.get("reason", ""),
                            }
                        )
                overall = str(parsed_json.get("result", "inconclusive")).lower()
                refuted = [
                    {"claim": c["claim"], "reason": c.get("reason", "")} for c in claims if c["status"] == "REFUTED"
                ]
                return {
                    "overall": overall,
                    "claims": claims,
                    "refuted": refuted,
                    "verified_count": sum(1 for c in claims if c["status"] == "VERIFIED"),
                    "refuted_count": len(refuted),
                    "unverified_count": sum(1 for c in claims if c["status"] == "UNVERIFIED"),
                }
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    claims: list[dict[str, str]] = []
    for m in _re.finditer(
        r"###\s*Claim:\s*(.+?)\n"
        r".*?\*\*Status:\*\*\s*(VERIFIED|REFUTED|UNVERIFIED)",
        text,
        _re.DOTALL,
    ):
        claims.append({"claim": m.group(1).strip(), "status": m.group(2).upper()})

    overall_m = _re.search(r"RESULT:\s*(CONFIRMED|REJECTED|INCONCLUSIVE)", text, _re.IGNORECASE)
    overall = overall_m.group(1).lower() if overall_m else "inconclusive"

    refuted = [{"claim": c["claim"], "reason": c.get("reason", "")} for c in claims if c["status"] == "REFUTED"]

    return {
        "overall": overall,
        "claims": claims,
        "refuted": refuted,
        "verified_count": sum(1 for c in claims if c["status"] == "VERIFIED"),
        "refuted_count": len(refuted),
        "unverified_count": sum(1 for c in claims if c["status"] == "UNVERIFIED"),
    }


def _parse_subagent_json(text: str) -> dict[str, Any] | None:
    """Extract and parse a JSON object from a subagent response."""
    from qa_common import _extract_json_object

    if not text:
        return None

    json_str = _extract_json_object(text)
    if not json_str:
        return None

    try:
        parsed = json.loads(json_str)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None

    return parsed if isinstance(parsed, dict) else None


async def _run_alignment_subagent(
    subagent_name: str,
    task_template: Any,
    args: dict[str, Any],
    parent_trace_id: str,
) -> tuple[dict[str, Any] | None, str]:
    """Run one alignment subtask and return (parsed_json, raw_text)."""
    from hud.agents import create_agent
    from hud.eval.manager import run_eval
    from hud.telemetry.instrument import instrument

    task = task_template.model_copy(update={"args": args})

    @instrument(category="subagent", name=subagent_name)
    async def _run_subagent() -> tuple[dict[str, Any] | None, str]:
        async with run_eval(task, trace=False, trace_id=parent_trace_id, quiet=True) as ctx:
            agent = create_agent(_ALIGNMENT_MODEL)
            result = await agent.run(ctx, max_steps=_ALIGNMENT_MAX_STEPS)
            raw_text = result.content if hasattr(result, "content") and result.content else ""
            return _parse_subagent_json(raw_text), raw_text

    return await _run_subagent()


async def _run_alignment_phase_tool(
    phase_name: str,
    task_template: Any,
    args: dict[str, Any],
) -> list[TextContent]:
    """Shared wrapper for running one alignment subagent phase as a tool."""
    from hud.eval.context import get_current_trace_id
    from hud.settings import settings

    if not settings.api_key:
        return [
            TextContent(
                type="text",
                text='{"error":"HUD API key not available for alignment subagents"}',
            )
        ]

    parent_trace_id = get_current_trace_id()
    # we don't expect the parent trace id to be None
    assert parent_trace_id is not None
    data, raw = await _run_alignment_subagent(phase_name, task_template, args, parent_trace_id)
    if data is None:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": f"Subagent phase failed: {phase_name}",
                        "debug": raw,
                    },
                    ensure_ascii=True,
                ),
            )
        ]

    return [TextContent(type="text", text=json.dumps(data, ensure_ascii=True))]


@env.tool()
async def extract_prompt_requirements_subagent(user_focus: str = "") -> list[TextContent]:
    """Run Subagent: extract atomic requirements from prompt.txt."""
    focus = user_focus.strip() or "Extract every concrete, testable requirement from the task prompt."
    return await _run_alignment_phase_tool(
        "extract_prompt_requirements",
        _alignment_extract_task,
        {"user_focus": focus},
    )


@env.tool()
async def map_grader_checks_subagent(requirements_json: str, user_focus: str = "") -> list[TextContent]:
    """Run Subagent: map requirements to grader checks."""
    focus = user_focus.strip() or "Map each requirement to concrete grader checks."
    return await _run_alignment_phase_tool(
        "map_grader_checks",
        _alignment_grader_task,
        {"requirements_json": requirements_json, "user_focus": focus},
    )


@env.tool()
async def assess_submission_coverage_subagent(requirements_json: str, user_focus: str = "") -> list[TextContent]:
    """Run Subagent: map requirements to submission coverage."""
    focus = user_focus.strip() or "Determine which prompt requirements the submission actually met."
    return await _run_alignment_phase_tool(
        "assess_submission_coverage",
        _alignment_submission_task,
        {"requirements_json": requirements_json, "user_focus": focus},
    )


@env.tool()
async def arbitrate_alignment_subagent(
    requirements_json: str,
    grader_mapping_json: str,
    submission_mapping_json: str,
    user_focus: str = "",
) -> list[TextContent]:
    """Run Subagent: arbitrate final alignment verdict."""
    focus = user_focus.strip() or "Produce final prompt-grader-submission alignment verdict."
    return await _run_alignment_phase_tool(
        "arbitrate_alignment",
        _alignment_arbiter_task,
        {
            "requirements_json": requirements_json,
            "grader_mapping_json": grader_mapping_json,
            "submission_mapping_json": submission_mapping_json,
            "user_focus": focus,
        },
    )


_last_verification_result: dict[str, Any] | None = None


@env.tool()
async def verify_failure_claims(claims: str) -> list[TextContent]:
    """(Failure-analysis only) Independently verify your key claims about the trace.

    Only use this tool during failure_analysis. Pass a string listing each claim
    you want verified. A separate verification agent will re-run your commands
    and try to disprove each claim, returning VERIFIED/REFUTED/UNVERIFIED per
    claim plus an overall CONFIRMED/REJECTED/INCONCLUSIVE verdict.

    Call this BEFORE rendering your final verdict. Use the results to:
    - Drop or re-investigate any REFUTED claims
    - Incorporate the verification outcome into your final answer
    """
    global _last_verification_result

    from hud.agents import create_agent
    from hud.eval.context import get_current_trace_id
    from hud.eval.manager import run_eval
    from hud.settings import settings
    from hud.telemetry.instrument import instrument

    if not settings.api_key:
        return [
            TextContent(
                type="text",
                text="ERROR: HUD API key not available for verification subagent. Skipping verification — output your final JSON now.",
            )
        ]

    parent_trace_id = get_current_trace_id()
    task = _verify_task.model_copy(update={"args": {"claims": claims}})

    @instrument(category="subagent", name="verify_failure_claims")
    async def _run_verifier() -> list[TextContent]:
        global _last_verification_result

        async with run_eval(
            task,
            trace=False,
            trace_id=parent_trace_id,
            quiet=True,
        ) as ctx:
            agent = create_agent(_VERIFY_MODEL)
            result = await agent.run(ctx, max_steps=_VERIFY_MAX_STEPS)
            raw_text = result.content if hasattr(result, "content") and result.content else ""

            parsed = _parse_verification_output(raw_text)
            _last_verification_result = parsed

            summary_parts = [
                f"Verification {parsed['overall'].upper()}: "
                f"{parsed['verified_count']} verified, "
                f"{parsed['refuted_count']} refuted, "
                f"{parsed['unverified_count']} unverified.",
            ]
            if parsed["refuted"]:
                summary_parts.append("\nREFUTED claims (you MUST address these):")
                for entry in parsed["refuted"]:
                    summary_parts.append(f"  - {entry['claim']}")
                    if entry.get("reason"):
                        summary_parts.append(f"    Reason: {entry['reason']}")
                summary_parts.append(
                    "\n⚠️ ACTION REQUIRED — STRICT BUDGET: You have at most 3 tool calls remaining."
                    '\n1. Remove refuted claims OR mark their fault as "unclear"'
                    "\n2. Output your final JSON immediately"
                    "\n\nDo NOT re-investigate or re-prove refuted claims. The verifier already"
                    "\nran independent commands — additional investigation is wasted effort."
                    "\nYour very next response should be your final JSON answer."
                )

            summary_parts.append(f"\n--- Full verification report ---\n{raw_text}")

            return [TextContent(type="text", text="\n".join(summary_parts))]

    return await _run_verifier()


def get_last_verification_result() -> dict[str, Any] | None:
    """Return the most recent verification result, then clear it."""
    global _last_verification_result
    result = _last_verification_result
    _last_verification_result = None
    return result


@env.tool()
async def view_screenshot(step: int) -> list[TextContent | ImageContent]:
    """View a screenshot observation by trajectory step number.

    Returns the screenshot image for the given step. Use the trajectory
    summary or screenshots_index.txt to find which steps have screenshots.
    """
    screenshots_dir = WORKSPACE_DIR / "screenshots"
    path = screenshots_dir / f"step_{step:04d}.png"

    if not path.exists():
        if screenshots_dir.exists():
            available = sorted(screenshots_dir.glob("step_*.png"))
            nums = [p.stem.replace("step_", "") for p in available]
        else:
            nums = []

        if nums:
            listing = ", ".join(nums)
            msg = f"No screenshot for step {step}. Available steps: {listing}"
        else:
            msg = f"No screenshot for step {step}. No screenshots available."
        return [TextContent(type="text", text=msg)]

    raw = path.read_bytes()
    data = base64.standard_b64encode(raw).decode("ascii")

    # Detect actual image format from magic bytes instead of trusting the extension
    if raw[:3] == b"\xff\xd8\xff":
        mime = "image/jpeg"
    elif raw[:8] == b"\x89PNG\r\n\x1a\n":
        mime = "image/png"
    elif raw[:4] == b"GIF8":
        mime = "image/gif"
    elif raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        mime = "image/png"

    return [
        TextContent(type="text", text=f"Screenshot at step {step}:"),
        ImageContent(type="image", data=data, mimeType=mime),
    ]


async def fetch_trace(
    trace_id: str,
    api_key: str,
    data_sources: list[str],
) -> dict[str, Any]:
    """Fetch trace data from HUD platform API.

    Args:
        trace_id: The trace UUID
        api_key: HUD API key
        data_sources: List of data sources to include (telemetry, environment, worker)

    Returns:
        Trace data dictionary
    """
    include_trajectory = "telemetry" in data_sources
    include_logs = "environment" in data_sources
    include_rollout_logs = "worker" in data_sources

    url = f"{os.environ.get('HUD_API_URL', 'https://api.hud.ai')}/telemetry/traces/{trace_id}"
    params = {
        "include_trajectory": str(include_trajectory).lower(),
        "include_logs": str(include_logs).lower(),
        "include_rollout_logs": str(include_rollout_logs).lower(),
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()


def _get_screenshots_base_url() -> str:
    """Get the base URL for screenshot storage.

    Always uses production Supabase URL for screenshots, regardless of which
    API environment is being used. This ensures analysis environments can
    always access screenshots from production traces.

    Returns:
        Base URL for production screenshots bucket
    """
    # Get Supabase URL from environment, default to production
    supabase_url = os.environ.get("SUPABASE_URL", "https://gahludmjcsmszgyufydt.supabase.co")
    # Construct the public screenshots bucket URL
    return f"{supabase_url.rstrip('/')}/storage/v1/object/public/screenshots/"


async def download_screenshots(
    trajectory: list[dict[str, Any]],
    trace_id: str,
) -> dict[int, Path]:
    """Download screenshots for steps that have them.

    Always downloads from production Supabase storage, regardless of which
    API environment is being used. This ensures analysis environments can
    access screenshots from any production trace.

    Args:
        trajectory: The trajectory spans from trace data
        trace_id: The trace UUID

    Returns:
        Dictionary mapping step index to screenshot file path
    """
    screenshots_dir = WORKSPACE_DIR / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    base_url = _get_screenshots_base_url()
    screenshots: dict[int, Path] = {}
    screenshot_count = 0

    async with AsyncClient(timeout=30.0) as client:
        for step_index, span in enumerate(trajectory):
            step_type = span.get("type")
            internal_type = span.get("internal_type")

            # Check if this step should have a screenshot
            # Must match platform dependencies.py screenshot detection
            if internal_type == "mcp-screenshot" or step_type in [
                "hud-step",
                "mcp-step-image",
            ]:
                # Construct the screenshot URL (production storage)
                screenshot_path = f"{trace_id}/{screenshot_count}.png"
                screenshot_url = f"{base_url}{screenshot_path}"

                try:
                    # Download the screenshot
                    response = await client.get(screenshot_url)
                    if response.status_code == 200:
                        # Save locally
                        local_path = screenshots_dir / f"step_{step_index:04d}.png"
                        local_path.write_bytes(response.content)
                        screenshots[step_index] = local_path
                        logger.info("Downloaded screenshot for step %d", step_index)
                    else:
                        logger.debug(
                            "Screenshot not found for step %d: HTTP %d",
                            step_index,
                            response.status_code,
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to download screenshot for step %d: %s",
                        step_index,
                        e,
                    )

                screenshot_count += 1

    return screenshots


def _preprocess_environment_logs(logs: list[Any] | str | Any) -> list[str]:
    """Preprocess environment logs into readable format.

    Extracts the actual log message from nested structures and formats
    timestamps in a readable way.
    """
    if not logs:
        return ["(no environment logs)"]

    if isinstance(logs, str):
        return logs.strip().split("\n")

    if not isinstance(logs, list):
        return [str(logs)]

    lines = []
    for entry in logs:
        if isinstance(entry, str):
            lines.append(entry)
            continue

        if not isinstance(entry, dict):
            lines.append(str(entry))
            continue

        # Extract the actual log message from various possible structures
        # Common formats: {"log": "...", "time": "..."} or {"message": "...", "timestamp": ...}
        log_msg = entry.get("log") or entry.get("message") or ""
        time_val = entry.get("time") or entry.get("timestamp") or ""
        stream = entry.get("stream", "")

        # Format timestamp if it's numeric (milliseconds)
        if isinstance(time_val, int | float):
            from datetime import datetime

            try:
                # Assume milliseconds
                time_val = datetime.fromtimestamp(time_val / 1000).strftime("%H:%M:%S.%f")[:-3]
            except (ValueError, OSError):
                time_val = str(time_val)
        elif isinstance(time_val, str) and "T" in time_val:
            # ISO format - extract just time part
            try:
                time_val = time_val.split("T")[1].split(".")[0]
            except (IndexError, AttributeError):
                pass

        # Clean up the log message
        if isinstance(log_msg, str):
            log_msg = log_msg.strip()
        else:
            log_msg = str(log_msg)

        # Skip empty messages
        if not log_msg:
            continue

        # Format: [TIME] (STREAM) MESSAGE
        prefix = f"[{time_val}]" if time_val else ""
        if stream and stream != "stdout":
            prefix += f" ({stream})"

        lines.append(f"{prefix} {log_msg}".strip())

    return lines if lines else ["(no environment logs)"]


def _preprocess_worker_logs(logs: str | list[Any] | Any) -> list[str]:
    """Preprocess worker/rollout logs into readable format."""
    if not logs:
        return ["(no worker logs)"]

    if isinstance(logs, str):
        return logs.strip().split("\n")

    if isinstance(logs, list):
        return [str(entry) for entry in logs]

    return [str(logs)]


def _preprocess_trajectory(trajectory: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Preprocess trajectory into a readable summary.

    Focuses on:
    - Tool calls with their inputs (truncated)
    - Tool results (success/failure)
    - Errors and exceptions
    - Agent messages / inference turns
    """
    if not trajectory:
        return ["(no trajectory data)"], []

    lines = ["=" * 60, "TRAJECTORY SUMMARY", "=" * 60, ""]

    tool_count = 0
    error_count = 0

    for i, span in enumerate(trajectory):
        name = span.get("name", "unknown")
        attrs = span.get("attributes", {})
        status = span.get("status_code", "")
        internal_type = span.get("internal_type")
        step_type = span.get("type")

        # Screenshot observation marker — must match download_screenshots() predicate
        if internal_type == "mcp-screenshot" or step_type in (
            "hud-step",
            "mcp-step-image",
        ):
            lines.append(f"[{i:04d}] SCREENSHOT (observation)")
            lines.append("")
            continue

        # Extract tool info — support both flat attrs and MCP-style nested structure
        tool_name = attrs.get("tool_name", "")
        tool_input = attrs.get("tool_input", "")
        tool_result = attrs.get("tool_result", "")

        # MCP-style spans: tool name in request.params.name, args in request.params.arguments
        if not tool_name and name == "tools/call.mcp":
            mcp_params = attrs.get("request", {}).get("params", {})
            tool_name = mcp_params.get("name", "")
            tool_input = mcp_params.get("arguments", "")
            mcp_result = attrs.get("result", {})
            if isinstance(mcp_result, dict):
                for c in mcp_result.get("content", []):
                    if isinstance(c, dict) and c.get("text"):
                        tool_result = c["text"]
                        break

        # MCP-style inference spans
        is_inference = name.startswith("inference.")
        inference_content = ""
        inference_tool_calls: list[dict] = []
        if is_inference:
            result = attrs.get("result", {})
            if isinstance(result, dict):
                content = result.get("content", "")
                if isinstance(content, str) and content:
                    inference_content = content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text" and block.get("text"):
                                inference_content = block["text"]
                            elif block.get("type") == "tool_use":
                                inference_tool_calls.append(block)
                tc = result.get("tool_calls", [])
                if tc:
                    inference_tool_calls = tc

        # Skip non-interesting spans (keep tool calls, errors, agent turns, inference)
        is_tool_call = bool(tool_name)
        is_error = status == "ERROR" or span.get("exceptions")
        is_agent_turn = "agent" in name.lower() or "llm" in name.lower()

        if not (is_tool_call or is_error or is_agent_turn or is_inference):
            continue

        # Format the span
        if is_tool_call:
            tool_count += 1
            lines.append(f"[{i:04d}] TOOL: {tool_name}")

            if tool_input:
                input_str = str(tool_input)
                if len(input_str) > 800:
                    input_str = input_str[:800] + "..."
                input_str = input_str.replace("\n", " ").replace("  ", " ")
                lines.append(f"       INPUT: {input_str}")

            if is_error:
                error_count += 1
                lines.append("       RESULT: FAILED")
            elif tool_result:
                result_str = str(tool_result)
                if len(result_str) > 500:
                    result_str = result_str[:500] + "..."
                result_str = result_str.replace("\n", " ")[:500]
                lines.append(f"       RESULT: {result_str}")

            lines.append("")

        elif is_inference:
            if inference_tool_calls:
                for tc in inference_tool_calls:
                    func = tc.get("function", tc)
                    tc_name = func.get("name", "?")
                    tc_args = str(func.get("arguments", func.get("input", "")))
                    if len(tc_args) > 200:
                        tc_args = tc_args[:500] + "..."
                    lines.append(f"[{i:04d}] AGENT -> tool_call: {tc_name}")
                    lines.append(f"       ARGS: {tc_args}")
                    lines.append("")
            elif inference_content:
                content_preview = inference_content.replace("\n", " ")
                if len(content_preview) > 600:
                    content_preview = content_preview[:600] + "..."
                lines.append(f"[{i:04d}] AGENT RESPONSE")
                lines.append(f"       {content_preview}")
                lines.append("")

        elif is_error:
            error_count += 1
            lines.append(f"[{i:04d}] ERROR in {name}")
            if span.get("status_message"):
                lines.append(f"       {span.get('status_message')}")

            exceptions = span.get("exceptions") or []
            for exc in exceptions:
                if isinstance(exc, dict):
                    exc_msg = exc.get("message", str(exc))[:500]
                else:
                    exc_msg = str(exc)[:500]
                lines.append(f"       EXCEPTION: {exc_msg}")
            lines.append("")

        elif is_agent_turn:
            lines.append(f"[{i:04d}] {name}")
            lines.append("")

    # ---- File modifications detail ----
    # Extract file-write operations with full content so the analyst
    # can see what was actually written (the main summary truncates to 300 chars).
    file_mods: list[tuple[int, str, str, str]] = []

    for i, span in enumerate(trajectory):
        attrs = span.get("attributes", {})
        t_name = attrs.get("tool_name", "")
        raw_input = attrs.get("tool_input", "")

        if not t_name and span.get("name") == "tools/call.mcp":
            mcp_p = attrs.get("request", {}).get("params", {})
            t_name = mcp_p.get("name", "")
            raw_input = mcp_p.get("arguments", "")

        if not t_name:
            continue

        inp = raw_input
        if isinstance(inp, str):
            try:
                inp = json.loads(inp)
            except (json.JSONDecodeError, TypeError):
                inp = {}
        if not isinstance(inp, dict):
            continue

        file_path = inp.get("path") or inp.get("file_path") or inp.get("filename", "")
        if not file_path:
            continue

        cmd = str(inp.get("command", "")).lower()
        content = ""

        if cmd == "create":
            content = str(inp.get("file_text", ""))
        elif cmd == "str_replace":
            old = str(inp.get("old_str", ""))
            new = str(inp.get("new_str", ""))
            if old or new:
                content = f"REPLACED:\n{old}\nWITH:\n{new}"
        elif cmd == "insert":
            content = str(inp.get("new_str", ""))
        elif cmd in ("view", "read", "undo_edit"):
            continue
        else:
            tn_lower = t_name.lower()
            if any(kw in tn_lower for kw in ("write", "create", "edit", "replace", "patch")):
                old_val = inp.get("old_str") or inp.get("old_string") or ""
                new_val = (
                    inp.get("file_text")
                    or inp.get("content")
                    or inp.get("new_str")
                    or inp.get("new_string")
                    or inp.get("text", "")
                )
                if old_val and new_val:
                    content = f"REPLACED:\n{old_val}\nWITH:\n{new_val}"
                elif new_val:
                    content = str(new_val)
            else:
                continue

        if not content:
            continue

        preview = content
        if len(preview) > 2000:
            preview = preview[:2000] + f"\n... ({len(content)} total chars)"

        file_mods.append((i, t_name, str(file_path), preview))

    # Summary stats
    lines.insert(3, f"Total spans: {len(trajectory)} | Tool calls: {tool_count} | Errors: {error_count}")
    lines.insert(4, "")

    # Build file modifications as a separate list
    mod_lines: list[str] = []
    if file_mods:
        mod_lines.append("=" * 60)
        mod_lines.append("FILE MODIFICATIONS DETAIL")
        mod_lines.append("=" * 60)
        mod_lines.append(f"Agent modified {len(file_mods)} file(s):")
        mod_lines.append("")
        for step, tool, path, preview in file_mods:
            mod_lines.append(f"--- [{step:04d}] {path} (via {tool}) ---")
            mod_lines.append(preview)
            mod_lines.append("")

    return lines, mod_lines


def _parse_json_or_passthrough(value: Any) -> Any:
    """If value is a JSON-encoded string, parse it; otherwise return as-is."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def _extract_scenario_setup(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract scenario setup data from prompts/get.mcp spans.

    This is the scenario's setup call — whatever arguments the env
    received when the scenario was initialized.  Works for any environment.
    """
    for span in trajectory:
        if span.get("name") == "prompts/get.mcp":
            args = span.get("attributes", {}).get("request", {}).get("params", {}).get("arguments", {})
            if args:
                return {k: _parse_json_or_passthrough(v) for k, v in args.items()}
    return {}


def _extract_evaluation_results(
    trajectory: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract all evaluation/resource read results from the trajectory.

    Returns every resources/read.mcp response that contains structured
    data (the scenario's evaluate phase output).  Falls back to
    metadata.evaluation_result if no spans found.
    """
    results: list[dict[str, Any]] = []
    for span in trajectory:
        if span.get("name") != "resources/read.mcp":
            continue
        for entry in span.get("attributes", {}).get("result", {}).get("contents", []):
            try:
                parsed = json.loads(entry.get("text", ""))
                if isinstance(parsed, dict):
                    results.append(parsed)
            except (json.JSONDecodeError, TypeError):
                continue

    if not results:
        eval_result = metadata.get("evaluation_result")
        if isinstance(eval_result, dict):
            results.append(eval_result)

    return results


async def write_trace_files(
    trace_data: dict[str, Any],
    data_sources: list[str],
) -> dict[str, Path]:
    """Write trace data to files in the workspace.

    Args:
        trace_data: Raw trace data from API (TraceDetailResponse structure)
        data_sources: Which sources to write

    Returns:
        Dictionary mapping source names to file paths
    """
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    files_written: dict[str, Path] = {}

    trajectory = trace_data.get("trajectory", [])
    trace_metadata = trace_data.get("metadata", {})

    # Extract large scenario_args values into separate files so they're
    # searchable independently (e.g. code diffs, long prompts).
    LARGE_VALUE_THRESHOLD = 2000
    LARGE_VALUE_KEYS = {"code_diff", "diff", "patch", "expected_output"}
    scenario_args = trace_data.get("scenario_args") or []
    scenario_args_cleaned = []
    for arg in scenario_args:
        if not isinstance(arg, dict):
            scenario_args_cleaned.append(arg)
            continue
        arg_name = arg.get("name", "")
        arg_value = arg.get("value", "")
        is_large = isinstance(arg_value, str) and len(arg_value) > LARGE_VALUE_THRESHOLD
        is_known_large = arg_name in LARGE_VALUE_KEYS
        if is_large or is_known_large:
            ext = ".txt"
            if arg_name in ("code_diff", "diff", "patch"):
                ext = ".diff"
            file_name = f"scenario_{arg_name}{ext}"
            file_path = WORKSPACE_DIR / file_name
            file_path.write_text(str(arg_value), encoding="utf-8")
            files_written[f"scenario_{arg_name}"] = file_path
            scenario_args_cleaned.append(
                {"name": arg_name, "value": f"(extracted to {file_name} — {len(str(arg_value))} chars)"}
            )
        else:
            scenario_args_cleaned.append(arg)

    # Always write metadata (all non-trajectory/logs fields)
    metadata = {
        "trace_id": trace_data.get("trace_id"),
        "job_id": trace_data.get("job_id"),
        "status": trace_data.get("status"),
        "reward": trace_data.get("reward"),
        "error": trace_data.get("error"),
        "external_id": trace_data.get("external_id"),
        "task_id": trace_data.get("task_id"),
        "task_version_id": trace_data.get("task_version_id"),
        "scenario": trace_data.get("scenario"),
        "scenario_args": scenario_args_cleaned,
        "prompt": trace_data.get("prompt"),
        "metadata": trace_metadata,
        "trajectory_length": trace_data.get("trajectory_length"),
        "logs_count": trace_data.get("logs_count"),
        "logs_error": trace_data.get("logs_error"),
    }
    metadata_path = WORKSPACE_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    files_written["metadata"] = metadata_path

    # Write the original prompt if present
    if trace_data.get("prompt"):
        prompt_path = WORKSPACE_DIR / "prompt.txt"
        prompt_path.write_text(str(trace_data["prompt"]), encoding="utf-8")
        files_written["prompt"] = prompt_path

    # Extract and write scenario setup (from prompts/get.mcp span)
    # This is everything the scenario received at initialization —
    # prompt, graders, patches, config, etc. depending on the environment.
    setup_args = _extract_scenario_setup(trajectory)
    if setup_args:
        setup_path = WORKSPACE_DIR / "scenario_setup.json"
        setup_path.write_text(json.dumps(setup_args, indent=2, default=str), encoding="utf-8")
        files_written["scenario_setup"] = setup_path

        # If setup contains a prompt and we don't already have one from the API
        setup_prompt = setup_args.get("prompt") or setup_args.get("task_prompt")
        if setup_prompt and isinstance(setup_prompt, str) and not trace_data.get("prompt"):
            prompt_path = WORKSPACE_DIR / "prompt.txt"
            prompt_path.write_text(setup_prompt, encoding="utf-8")
            files_written["prompt"] = prompt_path

    # Write scenario source code if available from the API response
    scenario_code = trace_data.get("scenario_code")
    if scenario_code and isinstance(scenario_code, str):
        code_path = WORKSPACE_DIR / "scenario_code.py"
        code_path.write_text(scenario_code, encoding="utf-8")
        files_written["scenario_code"] = code_path

    # Extract and write evaluation results (from resources/read.mcp spans)
    # This is the scenario's evaluate phase output — reward, subscores,
    # grader verdicts, etc.
    eval_results = _extract_evaluation_results(trajectory, trace_metadata)
    if eval_results:
        eval_path = WORKSPACE_DIR / "evaluation_result.json"
        # If single result, write flat; if multiple, write as array
        content = eval_results[0] if len(eval_results) == 1 else eval_results
        eval_path.write_text(json.dumps(content, indent=2, default=str), encoding="utf-8")
        files_written["evaluation_result"] = eval_path

    # Write telemetry/trajectory
    if "telemetry" in data_sources or "trajectory" in data_sources:
        traj_path = WORKSPACE_DIR / "trajectory.json"
        traj_path.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
        files_written["trajectory"] = traj_path

        summary_lines, mod_lines = _preprocess_trajectory(trajectory)
        summary_path = WORKSPACE_DIR / "trajectory_summary.txt"
        summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
        files_written["trajectory_summary"] = summary_path

        if mod_lines:
            mods_path = WORKSPACE_DIR / "file_changes.txt"
            mods_path.write_text("\n".join(mod_lines), encoding="utf-8")
            files_written["file_changes"] = mods_path

        # Download and index screenshots (CUA observations)
        # Always downloads from production storage
        screenshots = await download_screenshots(trajectory, trace_data.get("trace_id", ""))

        if screenshots:
            # Create an index file mapping step numbers to screenshot paths
            screenshot_index = []
            screenshot_index.append("SCREENSHOT INDEX")
            screenshot_index.append("=" * 60)
            screenshot_index.append(f"\nTotal screenshots: {len(screenshots)}\n")

            for step_idx in sorted(screenshots.keys()):
                screenshot_path = screenshots[step_idx]
                # Get relative path from workspace
                rel_path = screenshot_path.relative_to(WORKSPACE_DIR)
                screenshot_index.append(f"Step {step_idx:4d}: {rel_path}")

            index_path = WORKSPACE_DIR / "screenshots_index.txt"
            index_path.write_text("\n".join(screenshot_index), encoding="utf-8")
            files_written["screenshots_index"] = index_path

            logger.info("Downloaded %d screenshots", len(screenshots))

    # Write environment logs (preprocessed for readability)
    if "environment" in data_sources:
        logs = trace_data.get("logs", [])
        log_lines = _preprocess_environment_logs(logs)

        env_logs_path = WORKSPACE_DIR / "environment_logs.txt"
        env_logs_path.write_text("\n".join(log_lines), encoding="utf-8")
        files_written["environment_logs"] = env_logs_path

    # Write worker/rollout logs (preprocessed for readability)
    if "worker" in data_sources:
        rollout_logs = trace_data.get("rollout_logs", "")
        worker_lines = _preprocess_worker_logs(rollout_logs)

        worker_logs_path = WORKSPACE_DIR / "worker_logs.txt"
        worker_logs_path.write_text("\n".join(worker_lines), encoding="utf-8")
        files_written["worker_logs"] = worker_logs_path

    return files_written


async def download_task_codebase(
    registry_id: str,
    api_key: str,
) -> Path | None:
    """Download and extract an environment's source code into the workspace.

    Uses the registry source-url endpoint to get a presigned
    download URL, then fetches and extracts the tarball under
    /workspace/task_codebase/.

    Returns the extraction directory on success, None on failure.
    """
    import tarfile
    import tempfile

    api_url = os.environ.get("HUD_API_URL", "https://api.hud.ai")
    source_url_endpoint = f"{api_url}/registry/environments/{registry_id}/source-url"
    target_dir = WORKSPACE_DIR / "task_codebase"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                source_url_endpoint,
                headers={"X-API-Key": api_key},
            )
            if resp.status_code != 200:
                logger.warning(
                    "source-url returned %d for registry %s: %s",
                    resp.status_code,
                    registry_id,
                    resp.text,
                )
                return None

            download_url = resp.json().get("download_url")
            if not download_url:
                logger.warning("source-url response missing download_url")
                return None

            tarball_resp = await client.get(
                download_url,
                follow_redirects=True,
                timeout=120.0,
            )
            if tarball_resp.status_code != 200:
                logger.warning(
                    "Tarball download failed with %d",
                    tarball_resp.status_code,
                )
                return None

        target_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=True) as tmp:
            tmp.write(tarball_resp.content)
            tmp.flush()
            with tarfile.open(tmp.name, "r:gz") as tar:
                tar.extractall(path=str(target_dir), filter="data")

        logger.info(
            "Extracted environment source (%d bytes) to %s",
            len(tarball_resp.content),
            target_dir,
        )
        return target_dir

    except Exception as exc:
        logger.warning("Failed to download environment source: %s", exc)
        return None


def check_response(response: str, includes: list[str], excludes: list[str]) -> tuple[float, str]:
    """Check if response contains required patterns and excludes forbidden ones.

    Args:
        response: The agent's response
        includes: Patterns that MUST be in the response (all must match)
        excludes: Patterns that MUST NOT be in the response (none can match)

    Returns:
        Tuple of (score, explanation)
    """
    response_lower = response.lower()

    # Check excludes first (any match = failure)
    for pattern in excludes:
        if pattern.lower() in response_lower:
            return 0.0, f"Response contains forbidden pattern: '{pattern}'"

    # Check includes (all must match)
    missing = []
    for pattern in includes:
        if pattern.lower() not in response_lower:
            missing.append(pattern)

    if missing:
        return 0.0, f"Response missing required patterns: {missing}"

    return 1.0, "All patterns matched successfully"


@env.scenario("analyze")
async def analyze_trace(
    trace_id: str,
    query: str,
    hud_api_key: str,
    data_sources: list[str] | None = None,
    includes: list[str] | None = None,
    excludes: list[str] | None = None,
) -> AsyncGenerator[Any, None]:
    """Analyze a HUD trace to answer a query.

    Args:
        trace_id: The trace UUID to analyze
        query: The analysis query/question to answer
        hud_api_key: HUD API key (required)
        data_sources: Which data to load - ["telemetry", "environment", "worker"]
                     Defaults to ["telemetry"]
        includes: Patterns that must appear in the response for full reward
        excludes: Patterns that must NOT appear in the response
    """
    # Default values
    if data_sources is None:
        data_sources = ["telemetry"]
    if includes is None:
        includes = []
    if excludes is None:
        excludes = []

    api_key = hud_api_key

    logger.info("Fetching trace %s with sources: %s", trace_id, data_sources)

    # Fetch trace data
    trace_data = await fetch_trace(trace_id, api_key, data_sources)

    # Write files (including screenshots from production storage)
    files_written = await write_trace_files(trace_data, data_sources)

    logger.info("Wrote %d files to %s", len(files_written), WORKSPACE_DIR)

    # Download environment source code if registry_id is available
    registry_id = trace_data.get("registry_id")
    if registry_id:
        source_dir = await download_task_codebase(registry_id, api_key)
        if source_dir:
            files_written["task_codebase"] = source_dir
            logger.info("Environment source available at %s", source_dir)

    # Extract key metadata to include directly in prompt
    scenario_info = trace_data.get("scenario") or "unknown"
    external_id = trace_data.get("external_id") or ""
    task_id = trace_data.get("task_id") or ""
    job_id = trace_data.get("job_id") or ""
    status = trace_data.get("status") or "unknown"
    reward = trace_data.get("reward")
    error_info = trace_data.get("error") or trace_data.get("logs_error") or ""
    trajectory_length = trace_data.get("trajectory_length", 0)

    # Build file listing
    file_list = ", ".join(path.name for path in files_written.values())

    # Create a comprehensive context block with key metadata
    metadata_context = f"""
**Trace ID:** {trace_id}
**Job ID:** {job_id}
**Task ID:** {task_id}
**External ID:** {external_id}
**Status:** {status}
**Reward:** {reward}
**Scenario:** {scenario_info}
**Trajectory Length:** {trajectory_length} steps
"""

    if error_info:
        metadata_context += f"**Error:** {error_info}\n"

    # Add prompt if available
    task_prompt = trace_data.get("prompt")
    if task_prompt:
        if len(task_prompt) > 2000:
            task_prompt = task_prompt[:2000] + "... (see prompt.txt for full text)"
        metadata_context += f"\n**Task Prompt:**\n{task_prompt}\n"

    prompt = f"""Analyze this HUD evaluation trace to answer the question below.

## TRACE METADATA
{metadata_context}

## AVAILABLE FILES
{file_list}

## FILES DESCRIPTION
- `trajectory_summary.txt`: Human-readable summary of agent actions, tool calls, and errors
- `trajectory.json`: Full trajectory data with all spans and attributes
- `metadata.json`: Complete trace metadata (job, task, scenario details)
- `scenario_code_diff.diff`: Code diff from scenario args (if present — extracted for easy searching)
- `scenario_*.txt`: Other large scenario argument values extracted as standalone files
- `prompt.txt`: The original task prompt given to the agent
- `screenshots_index.txt`: Index of available CUA screenshots (observations)
- `screenshots/step_XXXX.png`: Screenshot images for each step (PNG format, can be read with read tool)
- `environment_logs.txt`: Container logs from the evaluation environment (if requested)
- `worker_logs.txt`: Orchestrator/worker logs (if requested)
- `task_codebase/`: The source code of the task environment — grading logic, scenario definitions, reference solutions, and test suites. Run `ls /workspace/task_codebase/` to explore. Read the grading scripts and any golden solutions inside to understand what correct behavior looks like.

**Note:** Screenshots are always fetched from production storage. Use the `view_screenshot` tool with a
step number to view any screenshot directly. Steps with screenshots are marked in trajectory_summary.txt.

## YOUR TASK
{query}

## INSTRUCTIONS
Use the tools to read and analyze the trace files. You have access to:
- `view_screenshot(step=N)` to view screenshot observations by step number
- File reading tools to examine logs and trajectory
- Grep/search tools to find specific patterns
- Bash tools for more complex analysis

The key metadata is shown above. For detailed analysis, read the relevant files.

Be VERY BRIEF - respond with ONE short paragraph that directly answers the question. No preamble, no lengthy explanations."""

    # Yield the prompt and wait for agent response
    response = yield prompt

    # Evaluate the response
    if includes or excludes:
        score, explanation = check_response(str(response), includes, excludes)
        logger.info("Evaluation: score=%.2f, %s", score, explanation)
        yield score
    else:
        # No validation patterns, give full credit if they responded
        yield 1.0 if response else 0.0


import qa_failure_analysis  # noqa: F401, E402
import qa_false_negative  # noqa: F401, E402
import qa_false_positive  # noqa: F401, E402
import qa_prompt_alignment  # noqa: F401, E402
import qa_reward_hacking  # noqa: F401, E402

if __name__ == "__main__":
    env.run(transport="stdio")
