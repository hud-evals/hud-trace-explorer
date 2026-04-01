"""Failure Analysis — classify the root cause of a failed trace."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from env import env, fetch_trace, write_trace_files, logger


class FailureAnalysisResult(BaseModel):
    failure_category: Literal[
        "wrong_approach",
        "tool_error",
        "hallucination",
        "environment_issue",
        "grader_issue",
        "timeout",
        "partial_completion",
    ] = Field(description="Primary category of failure")
    root_cause: str = Field(description="Concise explanation of why the trace failed")
    failed_criteria: str = Field(description="Which specific grading criteria or requirements failed and why")
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence in the classification")


@env.scenario("failure_analysis", returns=FailureAnalysisResult)
async def failure_analysis(
    trace_id: str,
    hud_api_key: str,
    query: str = "",
    ground_truth: str | None = None,
) -> Any:
    """Classify why a trace failed, returning a structured failure analysis.

    Categorizes the root cause into one of: wrong_approach, tool_error,
    hallucination, environment_issue, grader_issue, timeout, partial_completion.
    """
    data_sources = ["telemetry", "environment", "worker"]

    logger.info("Failure analysis for trace %s", trace_id)

    trace_data = await fetch_trace(trace_id, hud_api_key, data_sources)
    files_written = await write_trace_files(trace_data, data_sources)

    reward = trace_data.get("reward", "unknown")
    status = trace_data.get("status") or "unknown"
    error_info = trace_data.get("error") or ""
    task_prompt = trace_data.get("prompt") or ""
    if len(task_prompt) > 800:
        task_prompt = task_prompt[:800] + "... (see prompt.txt for full text)"

    file_list = ", ".join(path.name for path in files_written.values())

    user_focus = query.strip() if query.strip() else (
        "Classify the root cause of this trace's failure."
    )

    prompt = f"""You are a failure analyst. A trace failed or received a low reward, and you need
to determine the root cause and classify it.

## Trace context
- **Status:** {status}
- **Reward:** {reward}
- **Error:** {error_info or "(none reported)"}

## Task prompt
{task_prompt or "(not available)"}

## Available files
{file_list}

Use `trajectory_summary.txt` for a quick overview of agent actions.
Use grep/bash to search `trajectory.json` for specific patterns — do NOT read it in full.
Check `environment_logs.txt` for container/grader errors.
Check `worker_logs.txt` for orchestration issues.

## Failure categories

Classify into exactly one:

- **wrong_approach**: The agent misunderstood the task or took a fundamentally wrong
  strategy. It understood the environment but chose the wrong path.
- **tool_error**: A tool call failed, returned an unexpected error, or was called with
  malformed arguments. The strategy may have been correct but execution broke.
- **hallucination**: The agent fabricated information, acted on data that doesn't exist,
  or made claims not supported by evidence it gathered.
- **environment_issue**: The environment crashed, was misconfigured, had missing
  dependencies, or had errors that prevented the agent from completing the task fairly.
- **grader_issue**: The agent's work appears correct or substantially correct, but the
  grader timed out, has a bug, or otherwise misjudged the result.
- **timeout**: The agent ran out of steps or time before completing the task. It was on
  a reasonable path but didn't finish.
- **partial_completion**: The agent completed some but not all requirements. No single
  catastrophic failure — it just didn't do everything asked.

## Focus
{user_focus}

## Analysis steps
1. Check the trace metadata and error info for obvious signals
2. Read the trajectory summary for the agent's action sequence
3. If the reward is partial (between 0 and 1), look for subscore breakdowns
4. Search trajectory and logs for error patterns, failed tool calls, or timeouts
5. Determine which category best explains the PRIMARY reason for failure"""

    response: FailureAnalysisResult = yield prompt

    if ground_truth is not None:
        yield 1.0 if (response.failure_category == ground_truth) else 0.0
    else:
        yield 1.0
