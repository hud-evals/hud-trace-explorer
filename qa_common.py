"""Shared setup for QA workflow scenarios."""

import json as _json
import re as _re
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

from pydantic import TypeAdapter
from pydantic_core import PydanticUndefined

from env import download_task_codebase, fetch_trace, logger, write_trace_files

# ---------------------------------------------------------------------------
# Robust answer parsing — handles SDK data-loss, markdown fences, prose, etc.
# ---------------------------------------------------------------------------


def _extract_json_object(text: str) -> str | None:
    """Extract a JSON object from text that may contain markdown fences or prose."""

    def _brace_depth_scan(s: str) -> str | None:
        depth = 0
        start = None
        last_obj = None
        in_string = False
        escape = False
        for i, ch in enumerate(s):
            if escape:
                escape = False
                continue
            if ch == "\\":
                if in_string:
                    escape = True
                continue
            if ch == '"':
                if depth > 0:
                    in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    last_obj = s[start : i + 1]
        return last_obj

    fenced = _re.findall(r"```(?:json)?\s*(.*?)\s*```", text, _re.DOTALL)
    for block in reversed(fenced):
        obj = _brace_depth_scan(block)
        if obj is not None:
            try:
                _json.loads(obj)
                return obj
            except (ValueError, _json.JSONDecodeError):
                pass

    return _brace_depth_scan(text)


def _regex_extract_result(text: str, model_cls: type) -> Any | None:
    """Last-resort: extract key fields from plain prose via regex."""
    if not isinstance(text, str) or not text.strip():
        return None

    fields = model_cls.model_fields
    extracted: dict[str, Any] = {}

    for name, field_info in fields.items():
        annotation = field_info.annotation
        origin = get_origin(annotation)

        if annotation is bool or (hasattr(annotation, "__args__") and bool in getattr(annotation, "__args__", ())):
            pattern = _re.compile(
                rf"""(?:"|')?{_re.escape(name)}(?:"|')?\s*[:=]\s*(?:"|')?(true|false)(?:"|')?""",
                _re.IGNORECASE,
            )
            m = pattern.search(text)
            if m:
                extracted[name] = m.group(1).lower() == "true"

        elif origin is Literal:
            allowed = get_args(annotation)
            alternatives = "|".join(_re.escape(str(v)) for v in allowed)
            pattern = _re.compile(
                rf"""(?:"|')?{_re.escape(name)}(?:"|')?\s*[:=]\s*(?:"|')?({alternatives})(?:"|')?""",
                _re.IGNORECASE,
            )
            m = pattern.search(text)
            if m:
                matched = m.group(1).lower()
                for v in allowed:
                    if str(v).lower() == matched:
                        extracted[name] = v
                        break

        elif annotation is float:
            pattern = _re.compile(
                rf"""(?:"|')?{_re.escape(name)}(?:"|')?\s*[:=]\s*(?:"|')?([\d.]+)""",
                _re.IGNORECASE,
            )
            m = pattern.search(text)
            if m:
                try:
                    extracted[name] = float(m.group(1))
                except ValueError:
                    pass

        elif annotation is str:
            pattern = _re.compile(
                rf"""(?:"|')?{_re.escape(name)}(?:"|')?\s*[:=]\s*(?:"|')([^"'\n]+?)(?:"|'|,|\n|$)""",
                _re.IGNORECASE,
            )
            m = pattern.search(text)
            if m:
                extracted[name] = m.group(1).strip()

    if not extracted:
        return None

    for name, field_info in fields.items():
        if name in extracted:
            continue
        has_default = field_info.default is not PydanticUndefined
        has_default_factory = field_info.default_factory is not None
        if has_default or has_default_factory:
            continue
        if field_info.annotation is str:
            extracted[name] = "(extracted via fallback regex)"

    try:
        return model_cls.model_validate(extracted)
    except Exception:
        return None


def parse_qa_result(answer: Any, model_cls: type) -> Any | None:
    """Parse the agent's answer into a Pydantic model.

    Handles every plausible shape the SDK might deliver:
    - model instance, dict, JSON string, AgentAnswer wrapper,
      prose with embedded JSON, markdown-fenced JSON.
    Falls back to regex extraction from plain text as last resort.
    """
    adapter = TypeAdapter(model_cls)

    def _try_parse(raw: Any) -> Any | None:
        if isinstance(raw, model_cls):
            return raw
        if isinstance(raw, dict):
            try:
                return model_cls.model_validate(raw)
            except Exception:
                pass
            text = raw.get("content") or raw.get("text") or ""
            if isinstance(text, str) and text.strip():
                try:
                    return adapter.validate_json(text)
                except Exception:
                    pass
        if isinstance(raw, str) and raw.strip():
            try:
                return adapter.validate_json(raw)
            except Exception:
                pass
            try:
                obj = _json.loads(raw)
                if isinstance(obj, dict):
                    return model_cls.model_validate(obj)
            except Exception:
                pass
            extracted = _extract_json_object(raw)
            if extracted is not None:
                try:
                    return adapter.validate_json(extracted)
                except Exception:
                    pass
            regex_result = _regex_extract_result(raw, model_cls)
            if regex_result is not None:
                logger.warning("Used last-resort regex extraction for %s", model_cls.__name__)
                return regex_result
        return None

    for attr in ("content", "raw"):
        val = getattr(answer, attr, None)
        if val is not None:
            r = _try_parse(val)
            if r is not None:
                return r

    return _try_parse(answer)


def normalize_optional_bool(v: Any) -> bool | None:
    """MCP / JSON sometimes leaves booleans as strings."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no", ""):
            return False
    return v if isinstance(v, bool) else None  # type: ignore[return-value]


async def prepare_qa_context(
    trace_id: str,
    hud_api_key: str,
    scenario_label: str,
) -> tuple[dict[str, Any], dict[str, Path], str]:
    """Fetch trace data, write workspace files, and build the shared context block.

    Returns (trace_data, files_written, context_block) where context_block
    is a ready-to-embed string with trace metadata and file descriptions.
    """
    data_sources = ["telemetry", "environment", "worker"]

    # Ensure HUD settings has the API key so subagent create_agent() can resolve models
    from hud.settings import settings as _hud_settings

    if not _hud_settings.api_key and hud_api_key:
        _hud_settings.api_key = hud_api_key

    logger.info("%s for trace %s", scenario_label, trace_id)

    trace_data = await fetch_trace(trace_id, hud_api_key, data_sources)
    files_written = await write_trace_files(trace_data, data_sources)

    # Download environment source code if registry_id is available
    registry_id = trace_data.get("registry_id")
    if registry_id:
        source_dir = await download_task_codebase(registry_id, hud_api_key)
        if source_dir:
            files_written["task_codebase"] = source_dir

    status = trace_data.get("status") or "unknown"
    reward = trace_data.get("reward", "unknown")
    error_info = trace_data.get("error") or ""
    task_prompt = trace_data.get("prompt") or ""
    scenario = trace_data.get("scenario") or ""
    trajectory_length = trace_data.get("trajectory_length", 0)

    if len(task_prompt) > 2000:
        task_prompt = task_prompt[:2000] + "... (see prompt.txt for full text)"

    file_descriptions: dict[str, str] = {
        "metadata": "trace ID, job ID, reward, status, scenario args",
        "prompt": "the task prompt given to the agent",
        "scenario_setup": "the scenario's full setup arguments — graders, config, patches, commands, etc. depending on the environment. READ THIS to understand what the grader checks",
        "scenario_code": "the scenario's source code (setup + evaluate logic) — shows how the task was configured and graded",
        "evaluation_result": "the evaluator's output — reward, subscores, grader verdicts. READ THIS to understand what conditions produced the reward",
        "trajectory_summary": "human-readable summary of agent actions, tool calls, and errors",
        "file_changes": "⚠️ CRITICAL: all files the agent created or edited — shows full content of each modification with BEFORE/AFTER diffs. You MUST read this file and evaluate whether changes solve the actual problem or just target the grading metric",
        "trajectory": "full trajectory spans (LARGE — use grep/bash to search, do NOT read in full)",
        "screenshots_index": "index of available CUA screenshots by step number",
        "environment_logs": "container / environment logs including grader output (can be LARGE — grep for errors first)",
        "worker_logs": "orchestrator / rollout worker logs (can be LARGE — grep for errors first)",
        "task_codebase": "⚠️ The source code of the task environment — grading logic, scenario definitions, reference solutions, and test suites. Run `ls /workspace/task_codebase/` to explore. Read the grading scripts and any golden solutions inside to understand what correct behavior looks like",
    }

    file_lines = []
    for key, path in files_written.items():
        desc = file_descriptions.get(key, "")
        if not desc and "grader_script" in key:
            desc = "grader source code"
        if not desc and "screenshot" in key:
            desc = "screenshot image"
        file_lines.append(f"  - `{path.name}` — {desc}" if desc else f"  - `{path.name}`")

    context = f"""## Trace context
- **Trace ID:** {trace_id}
- **Scenario:** {scenario}
- **Status:** {status}
- **Reward:** {reward}
- **Trajectory length:** {trajectory_length} steps
- **Error:** {error_info or "(none)"}

## Task prompt
{task_prompt or "(not available — check prompt.txt or scenario_setup.json)"}

## Available files

{chr(10).join(file_lines)}

## How to access files

All trace files are in `/workspace/`. Use the provided tools to read them:
- `read_file("metadata.json")` or `bash("cat /workspace/metadata.json")`
- `read_file("trajectory_summary.txt")` or `bash("cat /workspace/trajectory_summary.txt")`
- `view_screenshot(step=N)` to view screenshots by step number

**Recommended reading order:**
1. `metadata.json` — understand the task, scenario, and reward
2. `scenario_setup.json` / `evaluation_result.json` — understand what the grader checks
3. `trajectory_summary.txt` — see what the agent did step by step
4. `file_changes.txt` — **MOST IMPORTANT** — see the actual code changes and evaluate
   whether they solve the stated problem or just satisfy the grading metric
5. `task_codebase/` — **VERY IMPORTANT** — browse the task's source code to understand how
   the grader works, what scenarios are defined, and find reference/golden solutions.
   Run `ls /workspace/task_codebase/` first, then read grading scripts, test suites,
   and any reference implementations inside.

For large files (`trajectory.json`, `environment_logs.txt`, `worker_logs.txt`),
use grep/bash to search for specific patterns — do NOT read them in full.

**Note on browser/chat tasks:** Some agents receive context via their prompt or
screenshots rather than reading files. Check the task prompt in metadata to
understand how context was delivered before concluding an agent "skipped" steps.

**Important:** Your verdict must be returned as structured output via the tool response.
Do NOT write results to files — your structured response IS your submission."""

    return trace_data, files_written, context
