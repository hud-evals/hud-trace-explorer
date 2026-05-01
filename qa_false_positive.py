"""False Positive Analysis — detect traces where the agent got undeserved credit."""

from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel, Field

from env import env, logger
from qa_common import normalize_optional_bool, parse_qa_result, prepare_qa_context


class FalsePositiveResult(BaseModel):
    reasoning: str = Field(description="Step-by-step analysis of whether the agent truly solved the task")
    is_false_positive: bool = Field(
        description="True if the agent received credit but didn't actually solve the task correctly"
    )
    confidence: float = Field(default=0.5, description="Confidence in the verdict, 0.0 to 1.0", ge=0.0, le=1.0)


@env.scenario("false_positive_analysis", returns=FalsePositiveResult)
async def false_positive_analysis(
    trace_id: str,
    hud_api_key: str,
    query: str = "",
    rubric: str | None = None,
    ground_truth: bool | None = None,
) -> AsyncGenerator[Any, None]:
    """Determine whether a passing trace is a false positive."""
    _, _, context, rubric_block = await prepare_qa_context(
        trace_id, hud_api_key, "False positive analysis", rubric=rubric
    )

    user_focus = query.strip() or (
        "Determine whether this trace is a false positive — did the agent "
        "receive credit without actually solving the task?"
    )

    prompt = f"""{rubric_block}You are a QA analyst checking for FALSE POSITIVES in agent evaluation.

A false positive occurs when the agent receives a passing reward but didn't genuinely
solve the task. This silently inflates pass rates. Common causes include:

- **Shortcut / gaming** — hardcoding outputs, disabling tests, mocking return values
- **Superficial fix** — satisfies grader checks without addressing the real issue
- **Stub grader** — a grader command like `echo PASS` or `exit 0` that always passes,
  giving free points regardless of agent behavior
- **Structural-only grading** — AST/string-presence checks pass but ALL functional
  graders (hidden_eval, test suites, runtime checks) return 0. This is a false
  positive regardless of WHY the functional graders failed — whether the grader
  crashed (GPU errors, timeouts, connection errors) or the code was wrong, the
  result is the same: the reward is not backed by functional evidence. Do not
  excuse functional grader failures based on infrastructure reasons.
- **Pre-training contamination** — agent answered from memorized knowledge without
  doing the task. Smoking gun: `metadata.usage.total_input_tokens` < 100
  NOTE: For CUA (computer-use) tasks with screenshots, low text token counts are
  NORMAL because image tokens are not included in this counter. Do not flag CUA tasks
  solely for low text token counts.
- **Transcript matching** — LLM grader gives credit for keywords found anywhere in the
  conversation, not just the final answer
- **Pre-existing / coincidental pass** — task already passed or passes by luck
- **Guessing with weak grader** — agent failed to complete the task through the intended
  method but arrived at the correct answer through guesswork, background knowledge, or
  process-of-elimination reasoning. A weak grader (simple string match, `contains`,
  single-value check) then accepts the guess. High interaction count does NOT rule this
  out — check whether the agent explicitly gave up or fell back to reasoning from
  general knowledge rather than task-derived evidence.
- **Grading leniency** — the grading implementation does not faithfully enforce the
  original rubric. The original task rubric may specify all-or-nothing scoring, but
  the grading pipeline decomposes it into partial-credit sub-criteria or the LLM judge
  awards partial credit where the rubric mandates zero. Compare the ORIGINAL rubric
  (in `metadata.task_config.rubric` if present) against the actual grading output.

A false positive is NOT when the agent genuinely solved the task.

{context}

## Focus
{user_focus}

## PREREQUISITE — do this before anything else

Your FIRST action must be to explore `/workspace/task_codebase/`. Run these commands NOW:
1. `ls -R /workspace/task_codebase/` to see the full tree
2. Read grading scripts (e.g. `env.py`, `task.py`) — check if they are stubs or functional
3. If `tasks/*/golden/` exists, read the reference solutions
4. If `tasks/*/tests/` exists, read the test suites

Not every task has golden solutions or test directories. Read whatever is available.
You need this context before evaluating the agent's work. Do NOT skip this.

## Analysis steps (after reading task_codebase)
1. Read `metadata.json` — check `metadata.usage.total_input_tokens`. If < 100 total,
   the agent may not have read real content (pre-training contamination). But for CUA
   tasks using screenshots, low text token counts are expected — check for screenshot
   interactions before flagging.
2. Read `metadata.json` — check `metadata.task_config.rubric` if present. This is the
   ORIGINAL rubric as written by the task author. Note any all-or-nothing language,
   required conditions, or zero-score clauses.
3. Read `evaluation_result.json` — check subscores. Note which graders passed vs failed
   and whether passing graders are structural-only or functional. If you found an
   original rubric in step 2, compare whether the grading criteria and scores faithfully
   implement it. Flag any case where partial credit was awarded but the original rubric
   mandates binary scoring.
4. Read `scenario_setup.json` — inspect each grader's `command`/`script`. Flag any
   `echo PASS` or trivially-true stubs. Calculate reward WITHOUT stub graders.
5. Read `trajectory_summary.txt` — did the agent meaningfully engage with the task?
   Check whether the agent completed the task through the intended method, or whether
   it gave up and fell back to guessing or reasoning from general knowledge.
6. Compare the agent's output against the golden solution from the prerequisite. Extract
   specific code or outputs side-by-side to verify correctness, not just surface similarity.
7. Verdict: did functional graders pass, or does the reward come entirely from stubs,
   structural checks, pre-trained knowledge, grading leniency, or lucky guesses?

## Required output format
Return ONLY a JSON object — no markdown fences, no extra text:
{{
  "reasoning": "step-by-step analysis",
  "is_false_positive": true or false,
  "confidence": 0.0 to 1.0
}}
"""

    answer = yield prompt

    result = parse_qa_result(answer, FalsePositiveResult)
    if result is None:
        logger.warning("Could not parse agent response into FalsePositiveResult, scoring 0")
        yield 0.0
        return

    gt = normalize_optional_bool(ground_truth)
    if gt is not None:
        yield 1.0 if (result.is_false_positive == gt) else 0.0
    else:
        yield 0.0 if result.is_false_positive else 1.0
