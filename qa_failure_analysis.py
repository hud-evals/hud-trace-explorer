"""Failure Analysis — classify the root cause of a failed trace."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from env import env, logger
from qa_common import prepare_qa_context, parse_qa_result


class FailureAnalysisResult(BaseModel):
    failure_category: Literal[
        "agent_failure",
        "platform_failure",
        "eval_failure",
    ] = Field(description="Primary category of failure")
    root_cause: str = Field(
        default="",
        description="Concise explanation; mention hallucination, timeout, wrong tools, or incomplete work here when relevant",
    )
    failed_criteria: str = Field(
        default="",
        description="Which grading criteria or requirements failed and why; use this for partial completion (mixed subscores) instead of a separate category",
    )
    confidence: Literal["high", "medium", "low"] = Field(default="medium", description="Confidence in the classification")


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

Classify into exactly one of three categories:

- **agent_failure**: The agent is the primary problem — wrong reasoning, wrong or ineffective tool
  use, incomplete work, fabricated claims (hallucination), or running out of steps due to looping or
  bad strategy. If the agent timed out because of its own inefficiency, this is still agent_failure;
  note "timeout" in `root_cause`.
- **platform_failure**: The execution stack or environment blocked success — broken or flaky
  tools/MCP, missing APIs or permissions, bad container/image state, resource limits, upstream
  model/API errors, orchestrator-killed runs, or other infrastructure outside the agent's control.
  If a timeout was caused by the environment or orchestrator rather than agent behavior, use this
  category and note "timeout" in `root_cause`.
- **eval_failure**: The evaluation or task spec is wrong or ambiguous — rubric does not match the
  prompt, multiple valid interpretations, grader checks the wrong behavior, or weak checks pass bad
  answers. Specify whether the issue is rubric, grader logic, or prompt ambiguity in `root_cause`.

## Common eval_failure patterns — check for these BEFORE blaming the agent

- **Formatting-strict graders**: Agent's answer is semantically correct but rejected for cosmetic
  differences (e.g. trailing zeros, thousands separators, whitespace, JSON indentation).
- **Rubric-prompt mismatch**: The grading rubric requires something the prompt never asked for, or
  contradicts what the prompt says.
- **Incomplete gold labels**: Agent found correct answers or real bugs that aren't in the reference
  set, and gets penalized for "false positives" that are actually true positives.
- **Policy-correct denials penalized**: Agent correctly followed policy but the grader expects
  the agent to always comply with the user, even when policy says otherwise.
- **Broken / stub graders**: Grader always returns 1.0 regardless of quality, or crashes and
  returns a default score, or evaluates the wrong artifact (e.g. full transcript instead of final
  answer). If reward is high (≥0.8) but the agent did clearly wrong or no work, the grader is
  broken — classify as eval_failure.
- **Ambiguous task language**: Underspecified criteria that admit multiple valid readings, where
  the agent picked a reasonable interpretation but the grader only accepts one.
- **Unstated requirements**: Grader checks for steps or formats that the task prompt never
  mentions.

## Not separate categories

- **Timeout**: classify by *why* it timed out — agent looping/inefficiency → `agent_failure`;
  environment/orchestrator killed it → `platform_failure`. Always mention "timeout" in `root_cause`.
- **Partial completion** (some subscores pass, some fail): pick the primary category and spell out
  what passed vs failed in `failed_criteria`.
- **Hallucination**: use `agent_failure` and note "hallucination" in `root_cause`.
- **Reward hacking**: handled by a separate scenario; if encountered here, classify by the
  underlying mechanism (agent gamed → `agent_failure`; grader let it through → `eval_failure`).

## Focus
{user_focus}

## Analysis steps
1. Check the evaluation result for reward breakdown and subscore details.
2. Read the scenario setup — grading criteria, rubric, grader scripts, and expected answers.
3. **Critically evaluate the grader**: Does the rubric match what the prompt actually asked? Are the
   expected answers reasonable? Could a correct solution be rejected by this grader? Is the grader
   checking the right thing? Check for formatting strictness, unstated requirements, and ambiguous
   task language. If the grader is flawed, this is eval_failure regardless of agent behavior.
4. Read the trajectory summary for the agent's action sequence.
5. Search trajectory and logs for error patterns, failed tool calls, or platform faults.
6. **Before concluding agent_failure, ask**: "Did the agent actually produce a reasonable or correct
   result that the grader unfairly rejected?" If yes, the root cause is eval_failure, not
   agent_failure. The agent doing something imperfectly does not make it agent_failure if the
   grader's expectations are themselves wrong.

## Required output format
Return ONLY a JSON object with exactly these fields — no markdown fences, no extra text:
{{
  "failure_category": "one of: agent_failure, platform_failure, eval_failure",
  "root_cause": "concise explanation",
  "failed_criteria": "which grading criteria failed and why",
  "confidence": "high, medium, or low"
}}
"""

    answer = yield prompt

    result = parse_qa_result(answer, FailureAnalysisResult)
    if result is None:
        logger.warning("Could not parse agent response into FailureAnalysisResult, scoring 0")
        yield 0.0
        return

    if ground_truth is not None:
        accepted = ground_truth.split("|")
        yield 1.0 if (result.failure_category in accepted) else 0.0
    else:
        yield 1.0
