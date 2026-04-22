"""Prompt-Grader Alignment — verify that grader and submission match task requirements."""

from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel, Field

from env import env, logger
from qa_common import normalize_optional_bool, parse_qa_result, prepare_qa_context


class PromptRequirement(BaseModel):
    requirement: str = Field(description="A concrete requirement extracted from the task prompt")
    met_by_agent: bool = Field(description="Did the agent's submission address this requirement?")
    checked_by_grader: bool = Field(description="Does the grader actually verify this requirement?")
    evidence: str = Field(description="What you found — commands run and output")


class PromptAlignmentResult(BaseModel):
    reasoning: str = Field(description="Step-by-step analysis of alignment between prompt, submission, and grader")
    requirements: list[PromptRequirement] = Field(
        default_factory=list,
        description="Every concrete requirement from the task prompt, with alignment status",
    )
    is_prompt_misaligned: bool = Field(
        description="True if prompt, grader, and submission are misaligned in any material way"
    )
    alignment_issues: list[str] = Field(
        default_factory=list,
        description="Specific mismatches found between prompt, grader, and submission",
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


@env.scenario("prompt_alignment_analysis", returns=PromptAlignmentResult)
async def prompt_alignment_analysis(
    trace_id: str,
    hud_api_key: str,
    query: str = "",
    ground_truth: bool | None = None,
) -> AsyncGenerator[Any, None]:
    """Check whether the grader and submission actually match what the task prompt requires."""
    _, _, context = await prepare_qa_context(trace_id, hud_api_key, "Prompt alignment analysis")

    user_focus = query.strip() or (
        "Check whether the task prompt, the grader, and the agent's submission "
        "are all aligned — does each component faithfully reflect the task requirements?"
    )

    prompt = f"""You are an alignment auditor. Your job is to check whether three things are consistent:

1. **The task prompt** — what the task actually asks the agent to do
2. **The grader** — what the evaluation mechanism actually checks
3. **The agent's submission** — what the agent actually produced

These three SHOULD be aligned, but often are not. Your job is to find the gaps.

## Types of misalignment

### Prompt-Grader misalignment (grader doesn't check what the prompt asks)
- Grader checks cells/fields/values that the prompt never mentioned
- Grader ignores requirements that the prompt explicitly stated
- Grader uses different thresholds, formats, or criteria than what the prompt specified
- Grader checks for side effects or behaviors the prompt didn't require
- Rubric criteria that contradict or go beyond the prompt

### Submission-Prompt misalignment (agent didn't do what the prompt asked)
- Agent solved a different problem than what was asked
- Agent addressed some requirements but missed others
- Agent used a method the prompt explicitly prohibited
- Agent produced output in the wrong format or location
- Agent made assumptions that contradict the prompt

### Grader-Submission alignment without Prompt alignment
- The agent and grader agree, but neither matches the prompt. This is the
  most insidious case — it looks like "correct" behavior but the task
  requirements were never actually tested.

## Types of alignment

### Prompt-Grader alignment (grader checks exactly what the prompt asks)
- Every explicit prompt requirement maps to a concrete grader check or immediate result of the change made by the agent
- Grader thresholds, formats, and criteria match what the prompt specified
- Grader does not check for side effects or behaviors the prompt didn't require
- Rubric criteria stay within the scope of the prompt

### Submission-Prompt alignment (agent did what the prompt asked)
- Agent addressed every requirement the prompt stated, although the manifestation of the changes are not obvious
- Agent respected prohibitions and constraints from the prompt
- Agent produced output in the required format and location
- Agent's assumptions are consistent with the prompt

### Full three-way alignment (prompt, grader, and submission agree)
- The agent's submission satisfies the prompt's requirements
- The grader verifies the end result of the changes made by the agent
- Passing the grader is strong evidence the task was actually completed —
  not a coincidence of agent and grader agreeing on the wrong thing

## Decision policy

Your job is to weigh evidence, not to default in either direction. Evaluate
both possibilities with equal rigor. Do not privilege either `true` or
`false`. Do not treat one verdict as the "extraordinary claim" and the other
as the "safe default".

Before deciding, you must:
1. Identify which of the MISALIGNED signals are present, with
   quoted evidence from prompt.txt and grader output.
2. Identify which of the ALIGNED signals are present, with quoted
   evidence from the same sources.
3. Compare the two sets on their merits. Your verdict is whichever set the
   evidence supports more strongly for THIS trace.

If the evidence genuinely ties, express that through `confidence` (~0.5),
and pick the verdict that best fits the quoted evidence you actually have —
not a pre-committed default.

## Signals of MISALIGNMENT (vote toward `true` when quoted evidence supports them)

M1. **Out-of-scope grader check.** The grader enforces a requirement that has no causal connection to the bug/behavior described in the prompt, AND a correct fix for what the prompt describes would still fail the grader.
M2. **Contradiction.** Prompt and grader give incompatible thresholds, formats, file locations, or required behavior (not merely different abstraction levels).
M3. **Wrong artifact.** The grader checks a file/function/output that the prompt never references, when a reasonable reading of the prompt would target a different artifact.
M4. **Phantom requirement in agent output.** The agent satisfies every requirement a careful reader could extract from the prompt, yet the grader fails the agent for a behavior the prompt never implies even indirectly.
M5. **Silent drop.** A quotable prompt requirement has no matching grader check; grader can PASS without ever verifying it. Inverse of M4.
M6. **Submission violates an explicit prompt constraint.** Prompt forbids a method or fixes a format/location; submission violates it, regardless of what the grader did. Quote constraint and violation.
M7. **Collusive pass.** Grader PASSES and submission matches the grader, but a quoted prompt requirement is absent from both.

## Signals of ALIGNMENT (vote toward `false` when quoted evidence supports them)

A1. **Symptom → root-cause invariant.** Prompt describes an observable symptom. Grader tests the underlying mechanical invariant that causes that symptom. The scenario/grader name often directly names the mechanism. An invariant that operationalizes the stated symptom is an alignment signal, not a misalignment signal.
A2. **Provided smoke test vs. grader's own test.** Prompt says "use `python XXX.py` to reproduce" and the grader runs a *different* script that exercises the same bug. A prompt's script can be a repro harness while the grader's script is a private evaluator of the same underlying bug; different scripts targeting the same behavior is not by itself misalignment.
A3. **Hidden / additional cases declared up front.** Prompt says "the same contract holds on the additional hidden cases the grader uses" (or similar). The grader then checks extra cases. Those extra checks are in scope by the prompt's own words and are not by themselves.
A4. **Approximate prompt, exact grader.** Prompt uses approximate values. Grader checks an exact contract value. Approximate narrative targets can be the observable consequence of an exact underlying contract; a precise grader check is not by itself misalignment.
A5. **Prompt high-level, grader low-level.** Greater grader specificity is not misalignment unless the thing being checked is unrelated.
A6. **Agent's fix is wrong or arithmetically trivial.** If the agent's patch is mathematically equivalent to the bug, a correct grader will fail it. That is evidence about the agent, not about alignment between prompt and grader. Keep this distinction explicit in your reasoning.

## Weighing the signals

- A signal only counts when you can quote the supporting text.
  Pattern recognition without quoted evidence is not evidence.
  This rule applies equally to A-signals and M-signals.
- Whichever verdict you lean toward, write one sentence arguing the other.
  If that counter-argument surfaces quotable evidence you did not yet weigh,
  integrate it before deciding.

{context}

## Focus
{user_focus}

## PREREQUISITE — do this before anything else

Your FIRST actions must be:
1. `cat /workspace/prompt.txt` — read the FULL task prompt carefully
2. Explore `/workspace/task_codebase/` — read grading scripts, reference solutions, test suites
3. `cat /workspace/evaluation_result.json` — see what the grader actually checked
4. `cat /workspace/scenario_setup.json` — see grader configuration

Do NOT skip any of these. You need all four to assess alignment.

## Investigation budget

- Do NOT read `trajectory.json` (huge). Use `trajectory_summary.txt` if needed.
- Do NOT wander `task_codebase/` unless, after reading the four prereq files,
  you still cannot identify what the grader actually checks. When you do,
  grep for the grader name (e.g. `check_cache_ticket_reuse`) directly.
- Do NOT invent a "claims verification / counter-check" loop. One pass,
  then decide.

## Analysis steps

**Phase 1 — Extract requirements from the task prompt:**
1. Read the full task prompt. List every concrete, testable requirement. Be exhaustive.
   Examples of requirements: "compute revenue for 2023 cohort", "output must be in CSV format",
   "fix the failing test in test_auth.py", "use the provided API endpoint".
2. For each requirement, note whether it's explicit (stated directly) or implicit
   (reasonably inferred from context).

**Phase 2 — Check what the grader actually verifies:**
3. Read the grader scripts in `task_codebase/` and `scenario_setup.json`.
4. For each prompt requirement: does the grader have a check for it? Map each requirement
   to the specific grader check that covers it (or note its absence).
5. Does the grader check anything the prompt DIDN'T ask for? These are extra requirements
   the agent couldn't have known about from the prompt alone.

**Phase 3 — Check what the agent actually submitted:**
6. Read `file_changes.txt` and `trajectory_summary.txt`.
7. For each prompt requirement: did the agent's final submission address it?
8. Did the agent do extra work the prompt didn't ask for, or skip required work?

**Phase 4 — Cross-reference and find gaps:**
9. Build a matrix: for each requirement, mark (met_by_agent, checked_by_grader).

## Output discipline
- Stop exploring the moment you have enumerated and weighed signals on both
  sides. Do not keep digging beyond that.
- Your entire response must be a single JSON object — no preamble like
  "Based on my analysis…", no bullet lists, no markdown fences.

Return ONLY this JSON object:
{{
  "reasoning": "<A-signals present (with quoted evidence), M-signals present (with quoted evidence), and which side the weighed evidence favors>",
  "requirements": [
    {{
      "requirement": "concrete requirement from the prompt (quoted or tightly paraphrased)",
      "met_by_agent": true or false,
      "checked_by_grader": true or false,
      "evidence": "file and line/stdout fragment you used"
    }}
  ],
  "is_prompt_misaligned": true or false,
  "alignment_issues": ["specific quoted mismatch 1", "specific quoted mismatch 2"],
  "confidence": 0.0 to 1.0
}}

`alignment_issues` must be `[]` when `is_prompt_misaligned` is `false` and non-empty (with quoted evidence) when `true`.
"""

    answer = yield prompt

    result = parse_qa_result(answer, PromptAlignmentResult)
    if result is None:
        logger.warning("Could not parse agent response into PromptAlignmentResult, scoring 0")
        yield 0.0
        return

    is_misaligned = result.is_prompt_misaligned
    gt = normalize_optional_bool(ground_truth)
    if gt is not None:
        yield 1.0 if (is_misaligned == gt) else 0.0
    else:
        yield 0.0 if is_misaligned else 1.0
