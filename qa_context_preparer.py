"""Context-preparation sub-agent for QA workflows."""

from pathlib import Path
from typing import Any

from hud import Environment
from hud.tools import BashTool
from hud.tools.filesystem import GlobTool, GrepTool, ListTool, ReadTool

BASE_PATH = "/workspace"

prepare_env = Environment(name="qa-context-preparer")

prepare_env.add_tool(BashTool())
prepare_env.add_tool(ReadTool(base_path=BASE_PATH))
prepare_env.add_tool(GrepTool(base_path=BASE_PATH))
prepare_env.add_tool(GlobTool(base_path=BASE_PATH))
prepare_env.add_tool(ListTool(base_path=BASE_PATH))


def _inventory_summary(
    trace_data: dict[str, Any], files_written: dict[str, Path], data_sources: list[str]
) -> list[str]:
    """Build a concise inventory of available trace material."""
    lines: list[str] = []

    trajectory_len = int(trace_data.get("trajectory_length") or 0)
    if "telemetry" in data_sources and trajectory_len > 0:
        lines.append(f"telemetry trajectory with {trajectory_len} spans")
    elif "telemetry" in data_sources:
        lines.append("telemetry source requested (no trajectory spans returned)")

    if "environment" in data_sources:
        log_count = int(trace_data.get("logs_count") or 0)
        lines.append(f"environment logs ({log_count} entries)")

    if "worker" in data_sources:
        rollout_logs = trace_data.get("rollout_logs")
        if isinstance(rollout_logs, str) and rollout_logs.strip():
            lines.append("worker rollout logs")
        else:
            lines.append("worker source requested (no rollout logs returned)")

    if trace_data.get("scenario_code"):
        lines.append("scenario source code")

    if trace_data.get("prompt"):
        lines.append("task prompt text")

    if trace_data.get("error"):
        lines.append("trace-level error details")

    if "evaluation_result" in files_written:
        lines.append("evaluation output with grader verdicts")

    if "file_changes" in files_written:
        lines.append("agent file modifications with before/after content")

    if "task_codebase" in files_written:
        lines.append("task codebase (grader logic, tests, references)")

    return lines


def _build_context_block(trace_id: str, trace_data: dict[str, Any], files_written: dict[str, Path]) -> str:
    """Build the full QA context block with critical file guidance."""
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
        "scenario_setup": "the scenario's full setup arguments - graders, config, patches, commands, etc. depending on the environment. READ THIS to understand what the grader checks",
        "scenario_code": "the scenario's source code (setup + evaluate logic) - shows how the task was configured and graded",
        "evaluation_result": "the evaluator's output - reward, subscores, grader verdicts. READ THIS to understand what conditions produced the reward",
        "trajectory_summary": "human-readable summary of agent actions, tool calls, and errors",
        "file_changes": "CRITICAL: all files the agent created or edited - shows full content of each modification with BEFORE/AFTER diffs. You MUST read this file and evaluate whether changes solve the actual problem or just target the grading metric",
        "trajectory": "full trajectory spans (LARGE - use grep/bash to search, do NOT read in full)",
        "screenshots_index": "index of available CUA screenshots by step number",
        "environment_logs": "container / environment logs including grader output (can be LARGE - grep for errors first)",
        "worker_logs": "orchestrator / rollout worker logs (can be LARGE - grep for errors first)",
        "task_codebase": "The source code of the task environment - grading logic, scenario definitions, reference solutions, and test suites. Run `ls /workspace/task_codebase/` to explore. Read the grading scripts and any golden solutions inside to understand what correct behavior looks like",
    }

    file_lines: list[str] = []
    for key, path in files_written.items():
        desc = file_descriptions.get(key, "")
        if not desc and "grader_script" in key:
            desc = "grader source code"
        if not desc and "screenshot" in key:
            desc = "screenshot image"
        file_lines.append(f"  - `{path.name}` - {desc}" if desc else f"  - `{path.name}`")

    return f"""## Trace context
- **Trace ID:** {trace_id}
- **Scenario:** {scenario}
- **Status:** {status}
- **Reward:** {reward}
- **Trajectory length:** {trajectory_length} steps
- **Error:** {error_info or "(none)"}

## Task prompt
{task_prompt or "(not available - check prompt.txt or scenario_setup.json)"}

## Available files

{chr(10).join(file_lines)}

## How to access files

All trace files are in `/workspace/`. Use the provided tools to read them:
- `read_file("metadata.json")` or `bash("cat /workspace/metadata.json")`
- `read_file("trajectory_summary.txt")` or `bash("cat /workspace/trajectory_summary.txt")`
- `view_screenshot(step=N)` to view screenshots by step number

**Recommended reading order:**
1. `metadata.json` - understand the task, scenario, and reward
2. `scenario_setup.json` / `evaluation_result.json` - understand what the grader checks
3. `trajectory_summary.txt` - see what the agent did step by step
4. `file_changes.txt` - **MOST IMPORTANT** - see the actual code changes and evaluate
   whether they solve the stated problem or just satisfy the grading metric
5. `task_codebase/` - **VERY IMPORTANT** - browse the task's source code to understand how
   the grader works, what scenarios are defined, and find reference/golden solutions.
   Run `ls /workspace/task_codebase/` first, then read grading scripts, test suites,
   and any reference implementations inside.

For large files (`trajectory.json`, `environment_logs.txt`, `worker_logs.txt`),
use grep/bash to search for specific patterns - do NOT read them in full.

**Note on browser/chat tasks:** Some agents receive context via their prompt or
screenshots rather than reading files. Check the task prompt in metadata to
understand how context was delivered before concluding an agent "skipped" steps.

**Important:** Your verdict must be returned as structured output via the tool response.
Do NOT write results to files - your structured response IS your submission."""


@prepare_env.tool()
async def load_trace_sources(
    trace_id: str,
    data_sources: list[str] | None = None,
) -> dict[str, Any]:
    """Load telemetry/environment/worker trace sources and write workspace files."""
    from hud.settings import settings

    from env import download_task_codebase, fetch_trace, logger, write_trace_files

    api_key = settings.api_key
    if not api_key:
        raise RuntimeError("HUD API key unavailable for context-preparer")

    requested_sources = data_sources or ["telemetry", "environment", "worker"]

    logger.info("Loading trace sources for context prep: trace=%s, sources=%s", trace_id, requested_sources)

    trace_data = await fetch_trace(trace_id, api_key, requested_sources)
    files_written = await write_trace_files(trace_data, requested_sources)

    registry_id = trace_data.get("registry_id")
    if registry_id:
        source_dir = await download_task_codebase(str(registry_id), api_key)
        if source_dir:
            files_written["task_codebase"] = source_dir

    trace_data_min = {
        "trace_id": trace_data.get("trace_id") or trace_id,
        "scenario": trace_data.get("scenario") or "",
        "status": trace_data.get("status") or "unknown",
        "reward": trace_data.get("reward", "unknown"),
        "error": trace_data.get("error") or "",
        "prompt": trace_data.get("prompt") or "",
        "trajectory_length": trace_data.get("trajectory_length", 0),
    }

    available_files = [
        {
            "key": key,
            "name": path.name,
            "path": str(path),
        }
        for key, path in files_written.items()
    ]

    return {
        "trace_data_min": trace_data_min,
        "sources_loaded": requested_sources,
        "available_files": available_files,
        "files_written": {key: str(path) for key, path in files_written.items()},
        "trace_inventory": _inventory_summary(trace_data, files_written, requested_sources),
        "context_block": _build_context_block(trace_id, trace_data, files_written),
    }


@prepare_env.scenario("prepare_context")
async def prepare_context(
    trace_id: str,
    scenario_label: str,
    data_sources: list[str] | None = None,
):
    """Spawned sub-agent that prepares succinct QA context metadata."""
    sources = data_sources or ["telemetry", "environment", "worker"]
    prompt = f"""You are a context-preparation subagent for QA analysis.

Scenario label: {scenario_label}

Your primary objective is to load trace data and return a compact JSON summary.

MANDATORY FIRST ACTION:
1. Call `load_trace_sources` with the exact scenario arguments provided to you:
   - `trace_id`: `{trace_id}`
   - `data_sources`: {sources}

RULES:
- Call `load_trace_sources` exactly once unless it fails.
- Do not run exploratory commands or read large files.
- Return only what downstream QA agents need to understand what trace data is available.
- Preserve critical QA guidance in `context_block`, including:
  - file descriptions,
  - how-to-access examples,
  - recommended reading order,
  - notes about large files and browser/chat tasks,
  - structured-output requirement.
- Output ONLY a JSON object (no markdown fences, no commentary).

REQUIRED OUTPUT SHAPE:
{{
  "trace_data_min": {{
    "trace_id": "...",
    "scenario": "...",
    "status": "...",
    "reward": "...",
    "error": "...",
    "prompt": "...",
    "trajectory_length": 0
  }},
  "sources_loaded": ["telemetry", "environment", "worker"],
  "trace_inventory": [
    "succinct statement of what trace data exists"
  ],
  "available_files": [
    {{"key": "metadata", "name": "metadata.json", "path": "/workspace/metadata.json"}}
  ],
  "files_written": {{"metadata": "/workspace/metadata.json"}},
  "context_block": "full ready-to-embed QA context with file guidance and reading order"
}}

If loading fails, return:
{{"error": "short failure reason"}}
"""

    yield prompt
    yield 1.0
