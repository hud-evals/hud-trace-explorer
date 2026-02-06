"""Trace Explorer Environment - Analyze HUD traces using coding tools.

This environment:
- Loads trace data from HUD platform (telemetry, environment logs, worker logs)
- Writes data to files for agent exploration
- Provides bash, grep, read, and edit tools for analysis
- Evaluates responses against includes/excludes patterns
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import httpx
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

load_dotenv()

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
        api_url: HUD API base URL

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

    async with httpx.AsyncClient(timeout=60.0) as client:
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
            # Based on platform logic: internal_type == "mcp-screenshot" or step_type in ["hud-step", "mcp-step-image"]
            if internal_type == "mcp-screenshot" or step_type in [
                "step",
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


def _preprocess_trajectory(trajectory: list[dict[str, Any]]) -> list[str]:
    """Preprocess trajectory into a readable summary.

    Focuses on:
    - Tool calls with their inputs (truncated)
    - Tool results (success/failure)
    - Errors and exceptions
    - Agent messages
    """
    if not trajectory:
        return ["(no trajectory data)"]

    lines = ["=" * 60, "TRAJECTORY SUMMARY", "=" * 60, ""]

    tool_count = 0
    error_count = 0

    for i, span in enumerate(trajectory):
        name = span.get("name", "unknown")
        attrs = span.get("attributes", {})
        status = span.get("status_code", "")

        # Extract tool info
        tool_name = attrs.get("tool_name", "")
        tool_input = attrs.get("tool_input", "")
        tool_result = attrs.get("tool_result", "")

        # Skip non-interesting spans (keep tool calls, errors, agent turns)
        is_tool_call = bool(tool_name)
        is_error = status == "ERROR" or span.get("exceptions")
        is_agent_turn = "agent" in name.lower() or "llm" in name.lower()

        if not (is_tool_call or is_error or is_agent_turn):
            continue

        # Format the span
        if is_tool_call:
            tool_count += 1
            lines.append(f"[{i:04d}] TOOL: {tool_name}")

            # Show input (truncated)
            if tool_input:
                input_str = str(tool_input)
                if len(input_str) > 300:
                    input_str = input_str[:300] + "..."
                # Clean up for readability
                input_str = input_str.replace("\n", " ").replace("  ", " ")
                lines.append(f"       INPUT: {input_str}")

            # Show result status
            if is_error:
                error_count += 1
                lines.append("       RESULT: FAILED")
            elif tool_result:
                result_str = str(tool_result)
                if len(result_str) > 200:
                    result_str = result_str[:200] + "..."
                result_str = result_str.replace("\n", " ")[:200]
                lines.append(f"       RESULT: {result_str}")

            lines.append("")

        elif is_error:
            error_count += 1
            lines.append(f"[{i:04d}] ERROR in {name}")
            if span.get("status_message"):
                lines.append(f"       {span.get('status_message')}")

            exceptions = span.get("exceptions") or []
            for exc in exceptions:
                if isinstance(exc, dict):
                    exc_msg = exc.get("message", str(exc))[:200]
                else:
                    exc_msg = str(exc)[:200]
                lines.append(f"       EXCEPTION: {exc_msg}")
            lines.append("")

        elif is_agent_turn:
            lines.append(f"[{i:04d}] {name}")
            lines.append("")

    # Summary stats
    lines.insert(3, f"Total spans: {len(trajectory)} | Tool calls: {tool_count} | Errors: {error_count}")
    lines.insert(4, "")

    return lines


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

    # Always write metadata (all non-trajectory/logs fields)
    metadata = {
        "trace_id": trace_data.get("trace_id"),
        "job_id": trace_data.get("job_id"),
        "status": trace_data.get("status"),
        "reward": trace_data.get("reward"),
        "error": trace_data.get("error"),
        # Task identification
        "external_id": trace_data.get("external_id"),
        "task_id": trace_data.get("task_id"),
        "task_version_id": trace_data.get("task_version_id"),
        # Context
        "scenario": trace_data.get("scenario"),
        "scenario_args": trace_data.get("scenario_args"),
        "prompt": trace_data.get("prompt"),
        "metadata": trace_data.get("metadata", {}),
        # Counts
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

    # Write telemetry/trajectory
    if "telemetry" in data_sources:
        trajectory = trace_data.get("trajectory", [])

        # Write full trajectory as JSON (for deep inspection)
        traj_path = WORKSPACE_DIR / "trajectory.json"
        traj_path.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
        files_written["trajectory"] = traj_path

        # Write a human-readable summary focused on tool calls and errors
        summary_lines = _preprocess_trajectory(trajectory)
        summary_path = WORKSPACE_DIR / "trajectory_summary.txt"
        summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
        files_written["trajectory_summary"] = summary_path

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
) -> Any:
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
        # Truncate if very long
        if len(task_prompt) > 500:
            task_prompt = task_prompt[:500] + "... (see prompt.txt for full text)"
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
- `prompt.txt`: The original task prompt given to the agent
- `screenshots_index.txt`: Index of available CUA screenshots (observations)
- `screenshots/step_XXXX.png`: Screenshot images for each step (PNG format, can be read with read tool)
- `environment_logs.txt`: Container logs from the evaluation environment (if requested)
- `worker_logs.txt`: Orchestrator/worker logs (if requested)

**Note:** Screenshots are always fetched from production storage. Use the `read` tool to view screenshots
as base64-encoded images that will be displayed to you visually.

## YOUR TASK
{query}

## INSTRUCTIONS
Use the tools to read and analyze the trace files. You have access to:
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


if __name__ == "__main__":
    env.run(transport="stdio")
