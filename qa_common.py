"""Shared setup for QA workflow scenarios."""

from pathlib import Path
from typing import Any

from env import fetch_trace, write_trace_files, logger


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

    logger.info("%s for trace %s", scenario_label, trace_id)

    trace_data = await fetch_trace(trace_id, hud_api_key, data_sources)
    files_written = await write_trace_files(trace_data, data_sources)

    status = trace_data.get("status") or "unknown"
    reward = trace_data.get("reward", "unknown")
    error_info = trace_data.get("error") or ""
    task_prompt = trace_data.get("prompt") or ""
    scenario = trace_data.get("scenario") or ""
    trajectory_length = trace_data.get("trajectory_length", 0)

    if len(task_prompt) > 800:
        task_prompt = task_prompt[:800] + "... (see prompt.txt for full text)"

    file_descriptions: dict[str, str] = {
        "metadata": "trace ID, job ID, reward, status, scenario args",
        "prompt": "the task prompt given to the agent",
        "scenario_setup": "the scenario's full setup arguments — graders, config, patches, commands, etc. depending on the environment",
        "evaluation_result": "the evaluator's output — reward, subscores, grader verdicts",
        "trajectory_summary": "human-readable summary of agent actions, tool calls, and errors",
        "trajectory": "full trajectory spans (LARGE — use grep/bash to search, do NOT read in full)",
        "screenshots_index": "index of available CUA screenshots by step number",
        "environment_logs": "container / environment logs including grader output (can be LARGE — grep for errors first)",
        "worker_logs": "orchestrator / rollout worker logs (can be LARGE — grep for errors first)",
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

**Start with the small files:** `scenario_setup.json`, `evaluation_result.json`,
`trajectory_summary.txt`, `metadata.json`. For large files (`trajectory.json`,
`environment_logs.txt`, `worker_logs.txt`), use grep/bash to search for
specific patterns — do NOT read them in full."""

    return trace_data, files_written, context
