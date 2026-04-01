"""False Negative Analysis — detect traces where the grader incorrectly failed the agent."""

from typing import Any

from pydantic import BaseModel, Field

from env import env
from qa_common import prepare_qa_context


class FalseNegativeResult(BaseModel):
    reasoning: str = Field(description="Step-by-step analysis of whether the grading is correct")
    is_false_negative: bool = Field(description="True if the agent solved the task but the grader incorrectly scored it as a failure")
    confidence: float = Field(description="Confidence in the verdict, 0.0 to 1.0", ge=0.0, le=1.0)


@env.scenario("false_negative_analysis", returns=FalseNegativeResult)
async def false_negative_analysis(
    trace_id: str,
    hud_api_key: str,
    query: str = "",
    ground_truth: bool | None = None,
) -> Any:
    """Determine whether a low-reward trace is a false negative."""
    _, _, context = await prepare_qa_context(trace_id, hud_api_key, "False negative analysis")

    user_focus = query.strip() or (
        "Determine whether this trace is a false negative — did the agent "
        "actually succeed at the task but receive a low or zero reward?"
    )

    prompt = f"""You are a QA analyst checking for FALSE NEGATIVES in agent evaluation.

A false negative occurs when the agent's work is correct (or substantially correct) but
the grader gave it a low or zero reward. Common causes include:

- **Grader timeout** — the grader ran out of time and defaulted to 0
- **Flaky checks** — non-deterministic assertions that intermittently fail
- **Grader bug** — the grading logic checks the wrong thing or has a logic error
- **Format mismatch** — the agent produced a correct answer in a different format
  than the grader expected (e.g. different casing, extra whitespace, equivalent but
  differently-structured output)
- **Environment issues** — file path mismatches, missing dependencies, or permissions
  errors during grading
- **Alternative valid solution** — the agent solved it a different valid way the
  grader didn't anticipate

A false negative is NOT when the agent genuinely failed the task. If the agent's work is
wrong, incomplete, or doesn't meet the requirements, the low reward is justified.

{context}

## Focus
{user_focus}

## Analysis steps
1. Understand what the task required from the prompt
2. Read the scenario setup and evaluation result to understand grading criteria
3. Trace what the agent actually did via the trajectory summary
4. Compare the agent's final output/actions against what the task required
5. Check environment and worker logs for grader timeouts, crashes, or infrastructure failures
6. Determine whether the reward is justified or if the agent was unfairly penalized"""

    response: FalseNegativeResult = yield prompt

    if ground_truth is not None:
        yield 1.0 if (response.is_false_negative == ground_truth) else 0.0
    else:
        yield 1.0
