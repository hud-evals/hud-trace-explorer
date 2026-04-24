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
    hud_api_key: str | None = None,
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

A false negative occurs when the agent's final delivered work satisfies the task
requirements, but the grader returned a low or zero reward.

This platform contains many task types: coding, ML engineering, SDLC workflows,
browser/computer-use tasks, tool-use tasks, Tau2-style business workflows, data
tasks, and other agent traces. Judge the trace against the task's own prompt, 
available observations, tool outputs, final state, grader output, screenshots/logs if present, 
and any task artifacts available in the agent's workspace.

## Base rate

Most low rewards are justified. Agents often miss a requirement, complete only part of the task, 
produce a wrong answer, or fail to deliver the final work. Start by assuming the reward is correct. Only flag a
false negative when you have **strong, independent evidence** that:

1. The agent's final delivered work satisfies the task's actual requirements, and
2. The low score comes from a grader, environment, aggregation, or rubric problem.

## Core definition

Call `is_false_negative: true` only when the agent should have received
materially higher reward.

Do NOT call false negative just because:
- The agent tried hard.
- The agent found a promising idea but did not deliver it.
- A smoke test, quick check, or partial verifier passed while another required
  behavior remained broken.
- The prompt had a hint and the environment allowed the agent to inspect or act,
  but the agent failed to do so.
- The agent solved part of the task but left a genuine required part incomplete.
- The agent's method was unusual but the final result was still wrong or incomplete.

## Five-question diagnostic

For each low or zero subscore, answer these in order:

1. **Did the grader actually evaluate the agent's delivered work?**
   If the grader crashed, timed out, had missing dependencies, could not load the
   environment, or failed before measuring the work, this may be a false negative.
   Do not confuse agent-caused failure with grader infrastructure failure.

2. **Does the grader's expectation match the task prompt and available task context?**
   If the grader requires an undocumented exact format, symbol, location, UI action,
   policy interpretation, API call, file path, issue reference, or hidden artifact
   that the prompt did not require or reasonably imply, this may be a false negative.
   If the prompt hinted at the requirement and the agent could inspect or complete it,
   missing it is agent failure.

3. **Does the grader check the right surface?**
   For coding tasks, did it inspect the file/function/artifact the prompt allowed?
   For browser or computer-use tasks, did it inspect the actual UI state or
   user-visible outcome? For tool-use tasks, did it inspect the final external state,
   not just transcript wording? For ML/data tasks, did it inspect the intended metric
   or output, not just a brittle string?

4. **Does aggregation hide correct work?**
   If most required behavior was correct but a brittle, misweighted, binary, or stale
   subgrader zeroed the whole reward, identify the exact unfair subscore and
   recompute only that part. If the failed subscore corresponds to a real missing
   requirement, the low score is justified.

5. **Do subjective graders disagree on the same evidence?**
   If two rubric/LLM graders read the same final artifact and one credits it while
   another zeros it, check whether the stricter judge ignored valid evidence. Do not
   accept either judge without verifying the artifact.

## Evidence rules

Before calling false negative, cite evidence in `reasoning` for all of these:
- The exact task requirement or prompt hint.
- What the agent actually delivered in the final state.
- What the grader expected or measured.
- Why the agent's delivered work satisfies the requirement despite the low score.
- Which subscore or failure is unfair or unjustified.

Never claim a file/function/export/UI state/tool result is missing unless you verified
the final state or trajectory. If evidence is unavailable, say unknown.

## Common false negative patterns, you can use them as examples, but there could be more patterns.

- Correct answer rejected for surface formatting only.
- Valid alternative solution rejected by brittle string, AST, import, selector, or
  exact-action checks.
- Grader expects a hidden or undocumented name, location, exact wording, or reference.
- Grader reads only one allowed output surface while the agent placed the work in
  another reasonable place.
- Environment or grader infrastructure failed before evaluating the work.
- LLM/rubric judge ignores substantive delivered content that another grader credits.
- Score aggregation masks a substantively correct result.



## What is NOT a false negative (the low reward IS justified)

- The agent's answer is factually/functionally/semantically wrong
- The agent violated task prompt or specifications
- The agent only partially completed the task and received justified partial credit
- The agent's "alternative approach" doesn't actually solve the root problem
- Agent disabled, bypassed, or mutated the task environment instead of solving the task.
- The agent's work was graded on location/position/structure and was in the wrong place
  (e.g. correct code but in the wrong file, correct bug but at the wrong line number)
- Agent failed because of its own tool misuse, context overflow, looping, or ignored errors.

## Critical reasoning rules

1. **Independently verify the agent's work — NEVER trust the agent's self-assessment.**
   "The agent says it succeeded" is NOT evidence of success. Check actual outputs,
   final state, screenshots, tool results, test results, diffs, grader logs, or external
   state as applicable to the task.

2. **"Grader applied its rules correctly" does not always mean "reward is justified."**
   A grader can be internally consistent and still unfair if it penalizes a correct
   result for an undocumented contract or brittle representation.

3. **Do not turn discoverability into unfairness.** If the prompt hinted at the needed
   behavior and the environment allowed the agent to inspect or act, missing that
   behavior is usually agent skill/discipline failure, not a false negative.

4. **Agent effort is not correctness.** Many tool steps, long trajectories, or
   confident-sounding reasoning do NOT mean the agent succeeded. Judge final delivered
   outcomes.

5. **Commit to a clear verdict.** Do not hedge with "borderline" — decide yes or no.
   When genuinely uncertain, lean toward `is_false_negative: false`.

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
## Analysis steps (after reading task workspace)

1. Read `metadata.json` and `evaluation_result.json` to understand the reward and grading.
2. Read `scenario_setup.json` or `scenario_code.py` to understand grading criteria to identify the actual requirement
3. Read `trajectory_summary.txt`, `file_changes.txt`, screenshots, tool outputs,
   environment logs, and grader logs as applicable to the task type.
4. Determine the final delivered state: code, answer, UI state, external tool state,
   file artifact, model output, workflow artifact, or other task-specific result.
5. Compare the final delivered state to the task requirement.
6. For any failed grader/subscore, run the five-question diagnostic.
7. Check environment/worker logs for infrastructure failures only if steps 1-5 are inconclusive
8. Produce a clear `is_false_negative` verdict with reasoning.

## Required output format
Return ONLY a JSON object with exactly these fields — no markdown, no extra text:
{{
  "reasoning": "step-by-step evidence. Include task requirement, delivered final state, grader expectation, and why the score is or is not unfair.",
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
