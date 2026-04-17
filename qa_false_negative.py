"""False Negative Analysis — detect traces where the grader incorrectly failed the agent."""

from typing import Any

from pydantic import BaseModel, Field

from env import env, logger
from qa_common import prepare_qa_context, parse_qa_result, normalize_optional_bool


class FalseNegativeResult(BaseModel):
    reasoning: str = Field(description="Step-by-step analysis of whether the grading is correct")
    is_false_negative: bool = Field(description="True if the agent solved the task but the grader incorrectly scored it as a failure")
    confidence: float = Field(default=0.5, description="Confidence in the verdict, 0.0 to 1.0", ge=0.0, le=1.0)


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
the grader gave it a low or zero reward.

## IMPORTANT: Base rate

Most low rewards are justified — the agent usually did something wrong. Start by assuming
the reward is correct. Only flag a false negative when you have **strong, independent
evidence** that the agent's work was actually correct.

## Causes of false negatives (if ANY of these apply → IS a false negative)

- **Format mismatch** — the agent produced the correct answer in a different surface format
  than the grader expected (e.g. differences in thousand separators, currency symbols,
  decimal places, casing, whitespace, or equivalent representations). If the answer is
  numerically/semantically correct and only the formatting differs, this IS a false negative.
- **Grader timeout** — the grader ran out of time and defaulted to 0
- **Flaky checks** — non-deterministic assertions that intermittently fail
- **Grader bug** — the grading logic checks the wrong thing or has a logic error
- **Grader expects wrong behavior** — the grader's expected answer or actions are themselves
  incorrect according to the actual task policy or requirements. This is common in
  policy-graded tasks (e.g. customer service) where the golden answer violates the stated
  policy, or in bug-detection tasks where the expected findings list is incomplete and the
  agent correctly found real issues beyond what the grader checks for.
- **Environment issues** — file path mismatches, missing dependencies, permissions errors,
  or UI elements (CAPTCHAs, popups) that blocked the agent through no fault of its own
- **Alternative valid solution** — the agent solved the task a genuinely different valid way
  the grader didn't anticipate. CAUTION: you must independently verify the agent's solution
  is actually correct — do NOT just trust that the agent's approach "sounds reasonable."

## What is NOT a false negative (the low reward IS justified)

- The agent's answer is factually wrong (e.g. wrong sign, wrong value, wrong entity)
- The agent violated task specifications (e.g. wrong module name, wrong variable name
  when the spec was explicit)
- The agent only partially completed the task
- The agent's "alternative approach" doesn't actually solve the root problem
  (e.g. a timeout workaround vs. fixing the actual bug)
- The agent changed the environment to make the task "work" instead of solving the
  original problem (e.g. creating a missing user account instead of switching to an
  existing one, or disabling a test instead of fixing the bug)
- The agent's work was graded on location/position/structure and was in the wrong place
  (e.g. correct code but in the wrong file, correct bug but at the wrong line number)

## Critical reasoning rules — READ CAREFULLY

1. **Format mismatch = false negative.** If you find yourself writing "this is a format
   mismatch" or "the only difference is formatting" in your analysis, the answer is
   `is_false_negative: true`. Do not then contradict yourself by siding with the grader.

2. **Don't confuse "grader applied its rules correctly" with "reward is justified."**
   A grader can be internally consistent (exact-match comparison worked as designed)
   while still producing an unfair result (penalizing a correct answer over a comma).

3. **Independently verify the agent's work — NEVER trust the agent's self-assessment.**
   When the agent's answer differs from the expected answer, check which one is actually
   correct. Don't assume the agent is right just because its reasoning "sounds plausible."
   Don't assume the grader is right just because it's the grader. Crucially, "the agent
   says it succeeded" is NOT evidence of success — check the actual outputs, test results,
   or grader logs for objective confirmation.

4. **Verify the grader's expected answer independently — it can be wrong.** Read the actual
   task policy, specifications, or requirements and check whether the grader's expected
   values/actions are correct. For policy-graded tasks, the golden answer sometimes violates
   the stated policy (e.g. expecting an action the policy prohibits). For code/bug tasks,
   the expected findings may be incomplete. If you can show the grader's expected answer is
   objectively wrong per the task's own rules, that IS a false negative.

5. **Don't blame the grader's expectations without evidence.** If the grader expected value
   X at location Y and the agent provided X at location Z, the most likely explanation is
   that the agent got the location wrong — NOT that the grader has a bug. Only call it a
   grader bug if you can show the grader's expected values are objectively incorrect.

6. **"Absence of grader details" ≠ "grader crashed."** A minimal evaluation result
   (e.g. just `{{"reward": 0.0}}`) may be the normal format, not evidence of a timeout.

7. **Agent effort ≠ correctness.** Many tool steps, long trajectories, or confident-sounding
   reasoning do NOT mean the agent succeeded. An agent can do 50 steps and still produce a
   wrong answer. Judge by outcomes, not by effort.

8. **Commit to a clear verdict.** Do not hedge with "borderline" — decide yes or no.
   When genuinely uncertain, lean toward "not a false negative" (the reward is justified).

{context}

## Focus
{user_focus}

## PREREQUISITE — do this before anything else

Your FIRST action must be to explore `/workspace/task_codebase/`. Run these commands NOW:
1. `ls -R /workspace/task_codebase/` to see the full tree
2. Read grading scripts (e.g. `env.py`, `task.py`)
3. If `tasks/*/golden/` exists, read the reference solutions
4. If `tasks/*/tests/` exists, read the test suites

Not every task has golden solutions or test directories. Read whatever is available.
You need this context to distinguish format-only mismatches from real errors. Do NOT skip this.

## Analysis steps (after reading task_codebase)
1. Read `metadata.json` and `evaluation_result.json` to understand the reward and grading
2. Read `scenario_setup.json` or `scenario_code.py` to understand grading criteria
3. Read `trajectory_summary.txt` to trace what the agent actually did
4. Compare the agent's final output against what the task required — check BOTH for correctness
5. If the answer differs from expected: determine whether the difference is format-only
   (false negative) or a genuine error (justified failure). Cross-reference against the golden
   solution from the prerequisite. Extract the agent's actual code/output and the expected
   answer side-by-side — pinpoint the exact divergence.
6. Check environment/worker logs for infrastructure failures only if steps 1-5 are inconclusive
7. Produce a clear `is_false_negative` verdict with reasoning

## Required output format
Return ONLY a JSON object with exactly these fields — no markdown, no extra text:
{{
  "reasoning": "step-by-step analysis",
  "is_false_negative": true or false,
  "confidence": 0.0 to 1.0
}}"""

    answer = yield prompt

    result = parse_qa_result(answer, FalseNegativeResult)
    if result is None:
        logger.warning("Could not parse agent response into FalseNegativeResult, scoring 0")
        yield 0.0
        return

    gt = normalize_optional_bool(ground_truth)
    if gt is not None:
        yield 1.0 if (result.is_false_negative == gt) else 0.0
    else:
        yield 0.0 if result.is_false_negative else 1.0
