"""False Negative Analysis — detect traces where the grader incorrectly failed the agent."""

from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel, Field

from env import env, logger
from qa_common import normalize_optional_bool, parse_qa_result, prepare_qa_context


class FalseNegativeResult(BaseModel):
    reasoning: str = Field(description="Step-by-step analysis of whether the grading is correct")
    is_false_negative: bool = Field(
        description="True if the agent solved the task but the grader incorrectly scored it as a failure"
    )
    confidence: float = Field(default=0.5, description="Confidence in the verdict, 0.0 to 1.0", ge=0.0, le=1.0)


@env.scenario("false_negative_analysis", returns=FalseNegativeResult)
async def false_negative_analysis(
    trace_id: str,
    hud_api_key: str | None = None,
    query: str = "",
    ground_truth: bool | None = None,
) -> AsyncGenerator[Any, None]:
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
- The prompt explicitly stated the requirement (in the prompt text, not just
  in hidden tests/graders) and the agent could observe or complete it but
  failed to do so.
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
   policy interpretation, API call, file path, exception type, function signature,
   issue reference, or hidden artifact that the prompt did not require or reasonably
   imply, this is a false negative — even if the agent could have inspected the
   grader/test code. **Tests, graders, and golden patches are NOT the task contract
   unless the prompt explicitly says so.** Names, file locations, return-type
   signatures, exact exception types, or symbol shapes that appear only in
   grader/test code are presumptively undocumented; failing such a check while
   passing functional intent is a false negative.
   Only treat the requirement as discoverable when the prompt itself hinted at it
   AND the necessary spec was visible in the trace agent's runtime workspace
   (not the QA agent's `/workspace/task_codebase/` view).

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
   **Aggregation override:** if a single binary subgrader (≥0.5 weight) zeros the
   reward on a code-shape / format / contract check while ≥2 other graders credit
   the work, presume false negative unless the prompt explicitly named that contract.

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

- Correct answer rejected for surface formatting only (including exception type
  drift like `RuntimeError` vs `ValueError`, casing, whitespace, or equivalent
  representations).
- Valid alternative solution rejected by brittle string, AST, import, selector, or
  exact-action checks.
- **Functional equivalence not credited:** the agent's change makes a guarded
  branch provably unreachable (e.g. lowering a threshold so `len(x) < N` is always
  false, replacing a constant with a no-op, `if False:`, dominated dead code) and
  the grader insists on physical removal. Grader prose like "must be removed" /
  "must be deleted" describes the grader's check, not the task requirement.
- Grader expects a hidden or undocumented name, location, exact wording, or
  reference (function name, file path, exception class, return signature).
- **Hidden / injected tests:** test file is added at grade time only (e.g. via
  `inject_test_files`, hidden test branch, grader-only artifact) and was never
  in the trace agent's workspace.
- Grader reads only one allowed output surface while the agent placed the work in
  another reasonable place that the prompt did not forbid.
- Environment or grader infrastructure failed before evaluating the work.
- LLM/rubric judge ignores substantive delivered content that another grader credits.
- Score aggregation masks a substantively correct result.

Worked example: the agent raises `RuntimeError` or exits nonzero to surface a
failure, while an injected hidden test expects `pytest.raises(ValueError)`. If the
task prompt only said to fail loudly, surface the error, or prevent silent success,
the exact exception type is an undocumented contract. Verdict: false negative.
Do not rationalize that exception types matter to callers unless the task prompt or
runtime-visible docs named that exception contract.



## What is NOT a false negative (the low reward IS justified)

- The agent's answer is factually/functionally/semantically wrong
- The agent violated the task prompt or specifications **stated in the prompt**
- The agent only partially completed the task and received justified partial credit
- The agent's "alternative approach" doesn't actually solve the root problem
- Agent disabled, bypassed, or mutated the task environment instead of solving the task.
- The agent's work was graded on location/position/structure and was in the wrong place,
  AND the correct location was either documented in the prompt or visible in the
  trace agent's runtime workspace (not just in the QA agent's task_codebase view).
- Agent failed because of its own tool misuse, context overflow, looping, or ignored errors.
- Agent's PR / submission / final delivered artifact does not actually contain the
  fix (e.g. the agent omitted a file from a retry push, or local edits were never
  pushed). Local file_changes are NOT proof of delivery — verify the final remote /
  submitted state.

## Critical reasoning rules

1. **Independently verify the agent's work — NEVER trust the agent's self-assessment.**
   "The agent says it succeeded" is NOT evidence of success. Check actual outputs,
   final state, screenshots, tool results, test results, diffs, grader logs, or external
   state as applicable to the task.

2. **"Grader applied its rules correctly" does not always mean "reward is justified."**
   A grader can be internally consistent and still unfair if it penalizes a correct
   result for an undocumented contract or brittle representation.

3. **Discoverability has limits.** Missing a behavior is agent failure ONLY when
   (a) the requirement was named or hinted in the prompt, AND (b) the necessary
   spec/contract was visible to the trace agent in its runtime workspace, prompt,
   or referenced docs. Hidden tests, golden patches, or grader code that you (the
   QA agent) can read in `/workspace/task_codebase/` do NOT prove the trace agent
   could see them — those are the QA's view, not the trace agent's view. Look for
   `inject_test_files`, hidden test branches, grader-only files, or scenario
   fixtures injected at grade time before invoking discoverability.
   Phrases like "real functional implications", "programmatic callers may depend
   on it", or "part of the public API" are NOT independent evidence that a hidden
   contract was documented. A contract is documented only if it appears in the task
   prompt, referenced docs, or code the trace agent could see in its runtime
   workspace. If you are arguing why a hidden contract matters, first ask whether
   the prompt named it; if not, the low score is a false negative.

4. **Workspace separation.** `/workspace/task_codebase/` is the full task
   definition (graders, golden patches, hidden test branches). The trace agent
   typically had a smaller view (a baseline branch clone, a sandbox, a browser, a
   tool API). Before judging "agent should have known," confirm via `env.py` /
   scenario setup which branch or files the trace agent actually had.

5. **Agent effort is not correctness.** Many tool steps, long trajectories, or
   confident-sounding reasoning do NOT mean the agent succeeded. Judge final delivered
   outcomes.

6. **Commit to a clear verdict.** Do not hedge with "borderline" — decide yes or no.
   When the doubt is about whether the agent functionally solved the task, lean
   `is_false_negative: true` and flag the reason in the reasoning and require further investigation from the QA team. 
   When the doubt is about whether an undocumented
   code-shape / format / grader-only contract is fair, lean `is_false_negative: true`.

{context}

## Focus
{user_focus}

## Optional task-codebase check

If `/workspace/task_codebase/` exists and this appears to be a coding, ML
engineering, or grader-inspection task, explore it before the verdict — but
remember: what you see here is NOT proof the trace agent could see it.

1. `ls /workspace/task_codebase/` to see the tree.
2. Read grading scripts (`env.py`, `task.py`, scenario grader scripts).
3. If `tasks/*/golden/` exists, read the reference solutions.
4. If `tasks/*/tests/` exists, read the test suites — and check whether tests
   are injected at grade time (e.g. `inject_test_files`) or always present.
5. Identify the trace agent's actual workspace (clone branch, sandbox dir,
   browser URL, tool API surface) before judging discoverability.

For computer-use, browser, TAU-style tool-use, web, and many other task types,
there may be no codebase at all — use screenshots, trajectory actions, tool
outputs, and final state instead.

## Analysis steps

1. Read `metadata.json` and `evaluation_result.json` to understand the reward and grading.
2. Read `scenario_setup.json` or `scenario_code.py` to understand grading criteria
   and identify the actual requirement (separate from grader implementation).
3. Read `trajectory_summary.txt`, `file_changes.txt`, screenshots, tool outputs,
   environment logs, and grader logs as applicable to the task type.
4. Determine the final delivered state: code, answer, UI state, external tool state,
   file artifact, model output, workflow artifact, or other task-specific result.
   For PR/submission tasks, verify the FINAL REMOTE state (not just local edits).
5. Compare the final delivered state to the task **prompt**, not just the grader output.
6. For any failed grader/subscore, run the five-question diagnostic.
7. Check environment/worker logs for infrastructure failures only if steps 1-5 are inconclusive.
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
