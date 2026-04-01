"""Failure Analysis — classify the root cause of a failed trace."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from env import env
from qa_common import prepare_qa_context


class FailureAnalysisResult(BaseModel):
    failure_category: Literal[
        "agent_error",
        "tool_error",
        "hallucination",
        "grader_misalignment",
        "environment_issue",
        "reward_hacking",
        "task_ambiguity",
        "timeout",
        "partial_completion",
        "infrastructure_failure",
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
    """Classify why a trace failed, returning a structured failure analysis."""
    _, _, context = await prepare_qa_context(trace_id, hud_api_key, "Failure analysis")

    user_focus = query.strip() or "Classify the root cause of this trace's failure."

    prompt = f"""You are a failure analyst. A trace failed or received a low reward, and you need
to determine the root cause and classify it.

{context}

## Failure categories

Classify into exactly one:

- **agent_error**: The agent made a genuine mistake. Look for:
  - Incorrect reasoning about the problem
  - Wrong tool usage or API calls
  - Incomplete implementation that misses key requirements
  - Failure to read or understand the task prompt
- **tool_error**: A tool call failed or returned an unexpected error. Look for:
  - Malformed arguments to tool calls
  - Unexpected tool responses the agent didn't handle
  - Tools called in the wrong order or context
- **hallucination**: The agent fabricated information. Look for:
  - Acting on data that doesn't exist
  - Making claims not supported by gathered evidence
  - Referencing files, functions, or APIs that aren't real
- **grader_misalignment**: The grader is testing the wrong thing. Look for:
  - Grader checking specific implementation patterns rather than behavioral correctness
  - Rubric items that don't match the task description
  - Agent producing a valid solution that fails grading due to rigid checks
  - Grader passing an incorrect solution due to weak checks
- **environment_issue**: The environment prevented success. Look for:
  - Missing write/read permissions on required paths
  - Tools or APIs that should be available but aren't
  - Incorrect initial state (missing files, wrong config, broken dependencies)
- **reward_hacking**: The agent gamed the evaluation. Look for:
  - Using test files or grading artifacts to derive answers
  - Modifying grader/test files instead of solving the task
  - Producing output that passes checks without addressing the actual problem
- **task_ambiguity**: The task itself is flawed. Look for:
  - Contradictory requirements in the prompt
  - Missing information needed to solve the task
  - Multiple valid interpretations with different expected outcomes
- **timeout**: The agent ran out of steps or time. Look for:
  - A reasonable strategy that simply didn't finish
  - Excessive loops or retries that burned through the step budget
- **partial_completion**: The agent completed some but not all requirements. Look for:
  - Multiple grading criteria where some passed and others failed
  - No single catastrophic failure — it just didn't do everything asked
- **infrastructure_failure**: System-level failure unrelated to the task. Look for:
  - Environment crashes or container timeouts
  - Network connectivity issues
  - Resource exhaustion (disk, memory)

## Focus
{user_focus}

## Analysis steps
1. Check the evaluation result for reward breakdown and subscore details
2. Read the scenario setup to understand grading criteria and task configuration
3. Read the trajectory summary for the agent's action sequence
4. If the reward is partial (between 0 and 1), reason about which parts succeeded
   and which failed
5. Search trajectory and logs for error patterns, failed tool calls, or timeouts
6. Determine which category best explains the PRIMARY reason for failure"""

    response: FailureAnalysisResult = yield prompt

    if ground_truth is not None:
        yield 1.0 if (response.failure_category == ground_truth) else 0.0
    else:
        yield 1.0
