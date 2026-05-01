"""Prompt-Grader Alignment — verify that grader and submission match task requirements."""

from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel, Field

from env import env, logger
from qa_common import normalize_optional_bool, parse_qa_result, prepare_qa_context


class PromptAlignmentResult(BaseModel):
    grader_check: str = Field(
        description="The specific grader check or expectation that matters for the verdict, quoted or summarized concretely"
    )
    prompt_quote: str = Field(
        description="The closest prompt text governing that grader check, quoted directly when possible"
    )
    reasoning: str = Field(
        description=(
            "Step-by-step analysis with evidence — what the prompt said (with quotes), "
            "what the grader checks (with quotes), what the agent did, visibility/fairness analysis, "
            "and why aligned/misaligned"
        )
    )
    misalignment_proof: str = Field(
        default="",
        description=(
            "Required when is_prompt_misaligned=true: one sentence of the form "
            "'A solution could satisfy every visible prompt/smoke requirement and still fail "
            "because <specific check X> requires <hidden or under-specified condition Y>.' "
            "Empty string when is_prompt_misaligned=false."
        ),
    )
    is_prompt_misaligned: bool = Field(
        description="True only if you can state a concrete proof that a solution satisfying the visible requirements could still fail"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "0.9 = sharp misalignment proof with a direct grader quote; "
            "0.5 = two reasonable readings exist; "
            "0.2 = guessing"
        ),
    )


@env.scenario("prompt_alignment_analysis", returns=PromptAlignmentResult)
async def prompt_alignment_analysis(
    trace_id: str,
    hud_api_key: str,
    query: str = "",
    rubric: str | None = None,
    ground_truth: bool | None = None,
) -> AsyncGenerator[Any, None]:
    """Check whether the grader and submission actually match what the task prompt requires."""
    _, _, context, rubric_block = await prepare_qa_context(
        trace_id, hud_api_key, "Prompt alignment analysis", rubric=rubric
    )

    user_focus = query.strip() or (
        "Check whether the task prompt, the grader, and the agent's submission "
        "are all aligned — does each component faithfully reflect the task requirements?"
    )

    prompt = f"""{rubric_block}You are an alignment auditor.

{user_focus}

{context}

# What "prompt misalignment" means

A task is **prompt-misaligned** when the grader/reward is unfair relative to
what the prompt and visible smoke tests gave the agent enough signal to do.

It is **NOT misaligned** only when the failing check is visible, specific,
fairly weighted, and the agent simply failed that stated requirement.

Hidden test cases are allowed. A hidden **test case** that only re-exercises
the same visible invariant under another input is fair. A hidden test case is
prompt misalignment only when it depends on a hidden requirement or convention
that a careful prompt-following agent would not know to implement.

Visible signal means the prompt, visible smoke tests/output, and trajectory
evidence available before the agent's final submission. Post-hoc grader output,
hidden subscore names, and evaluator-only diagnostics tell you what failed, but
they do not prove the original agent had enough signal unless the same detail
appeared in those pre-submission signals.

# The verdict procedure (do these in order, do not skip)

Step 1 — Identify the *pass/fail-determining* grader checks. Inspect all failed
checks that affected reward, not just the biggest or first one. If the grader
reports failed subscores, suites, or named checks, enumerate them in
`reasoning` and inspect the material assertion(s) inside each one. Do not stop
after finding an obvious visible submission bug. Quote the strongest candidate
in `grader_check`, and mention other material failed checks in `reasoning`.
Ignore narrative checks, reproduction guidance, and informational logs.

Step 2 — Find the closest prompt text that governs that check. Quote it in
`prompt_quote`. If no prompt text governs it, write "none" and proceed.

Step 3 — Assess fairness from the agent's visible information. Do NOT ask
whether an omniscient perfect implementation could pass. Ask whether the
specific failing grader check was visible enough from the prompt/smoke for a
careful prompt-following agent to implement the intended invariant, and whether
the reward loss is proportionate to the visible mistake. Use this exact test:
if a careful agent implemented everything the prompt visibly asked for, would
that necessarily satisfy this grader check? Write the answer in `reasoning`.
You MUST do this step before deciding anything — do not jump to "submission bug"
without writing the visibility/fairness analysis first.
When a check depends on an exact value, formula, return shape, field name,
identifier, or literal text, ask whether that exact contract was pinned by
pre-submission visible signals. "Strongly implied", "reasonable", or "the
natural API" is not enough unless a softener below applies.

Before claiming the failing check was hidden, you MUST inspect the visible
smoke's actual stdout/stderr (not just its exit status). If the smoke prints
the failing metric name, the failing case label, the offending value, or any
diagnostic that names the invariant the grader checks, the check is NOT
hidden — even if the prompt prose does not name the metric. Quote the smoke
output text in `reasoning`. If you did not read the smoke's actual output,
do not return `is_prompt_misaligned=true`.

Step 4 — Verdict:
- If the unfair-check rule below fires on any reward-affecting failing check:
  `is_prompt_misaligned=true`, and `misalignment_proof` is the sentence "A
  solution could satisfy every visible prompt/smoke requirement and still fail
  because <check X> requires <hidden or under-specified condition Y>."
- DO NOT BALANCE fair and unfair failures. ONE unfair reward-affecting check is
  sufficient, even if other failures are obvious submission bugs.
- Otherwise (you enumerated every reward-affecting failed subscore/suite/check,
  and each one is visible, specific, fairly weighted, or only re-exercises a
  visible invariant):
  `is_prompt_misaligned=false`, `misalignment_proof` empty.

Step 5 — Apply the softeners below only to *flip a tentative `true` to `false`*.
They are never a reason to skip Step 3 or to stay at `false` without doing it.

# Softeners (only flip true → false, never the other way)

**Necessary-invariant exception:**
An extra grader check is NOT misalignment when it is a basic necessary invariant
with low ambiguity — limited to: the code must still compile, the schema must
still be queryable, tests must still run, imports must resolve. Do not extend
this exception to any hidden behavioral contract or convention the prompt did
not make visible.

**Root-cause unit-test exception:**
Do not mark `true` merely because the grader uses a lower-level unit test than
the prompt's end-to-end smoke. If the lower-level check directly verifies the
same necessary root cause or invariant as the visible failure, and the prompt
gives enough signal to investigate that code path, treat it as aligned and flip
to `false`. This also applies to hidden inputs for the same visible invariant:
hidden case is fair; hidden requirement or convention is not.
This can include exact low-level formulas, masks, state transitions, or internal
values when they are the direct implementation contract of the visible invariant
with low ambiguity. Do not use this exception when the prompt leaves multiple
plausible public contracts or API shapes.

**Workaround rule:** if the agent's submission satisfies the prompt's literal
success criterion only by bypassing or neutering the actual code path the
prompt is asking about (e.g., making a function ignore its argument so a
downstream check passes, hardcoding a value to fake the expected output,
removing a guard so the symptom disappears), that is a submission-side
workaround, not prompt misalignment — flip to `false`. The grader is allowed
to inspect the call site, the argument, or the invariant that was supposed
to be fixed, even when the prompt names only the visible symptom.

# Strengthener (what makes a check unfair)

**Unfair-check rule:** Inspect every reward-affecting failing check, not just
the obvious submission bug. If at least one of them satisfies any of the
following — taking even one is enough — mark `is_prompt_misaligned=true`:

- the grader requires a specific value, formula, return shape, field name,
  identifier, or literal that the prompt and visible signals do not pin down,
  and a prompt-following agent would have many plausible alternatives to choose
  from,
- the grader rejects an alternative fix that satisfies the prompt's stated
  success criterion,
- one small, ambiguous, or under-specified issue causes a disproportionate
  reward loss or cascades into many downstream test failures,
- the grader checks an invariant the prompt does not expose, even when the
  prompt broadly says "fix the underlying issue",
- the trajectory shows the agent enumerated and satisfied every requirement
  it could read from the prompt and still failed an unanticipated check
  (cite the step).

"The grader's choice is reasonable" is not a defense — reasonableness does not
imply visibility. A hidden test case that only re-exercises a visible invariant
is NOT a reason to apply this rule.

# Required output format
Return the JSON object directly in your final assistant message as plain text.
Do NOT use bash, cat, heredoc, write_file, or any tool to print or save it.
Do NOT wrap it in markdown code fences. No commentary before or after.
{{
  "grader_check": "the specific grader check or expected value that matters most",
  "prompt_quote": "the closest quoted prompt requirement for that check",
  "reasoning": "what the prompt exposes (with quotes), what the grader checks (with quotes), what the agent did, whether the reward loss is proportionate, and the verdict reasoning",
  "misalignment_proof": "",
  "is_prompt_misaligned": true or false,
  "confidence": 0.0 to 1.0
}}
"""

    answer = yield prompt

    result = parse_qa_result(answer, PromptAlignmentResult)
    if result is None:
        logger.warning("Could not parse agent response into PromptAlignmentResult, scoring 0")
        yield 0.0
        return

    is_misaligned = result.is_prompt_misaligned
    if is_misaligned and not result.misalignment_proof.strip():
        logger.warning(
            "Agent returned is_prompt_misaligned=true with empty misalignment_proof; flipping to false per guardrail"
        )
        is_misaligned = False

    gt = normalize_optional_bool(ground_truth)
    if gt is not None:
        yield 1.0 if (is_misaligned == gt) else 0.0
    else:
        yield 0.0 if is_misaligned else 1.0
