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

    prompt = f"""You are an alignment auditor.

{user_focus}

You have direct access to specialized subagent tools:
- `extract_prompt_requirements_subagent(user_focus)`
- `map_grader_checks_subagent(requirements_json, user_focus)`
- `assess_submission_coverage_subagent(requirements_json, user_focus)`
- `arbitrate_alignment_subagent(requirements_json, grader_mapping_json, submission_mapping_json, user_focus)`

Default execution protocol (do this unless a tool is unavailable):
1. Run `extract_prompt_requirements_subagent`.
2. Run `map_grader_checks_subagent` and `assess_submission_coverage_subagent`.
3. Run `arbitrate_alignment_subagent`.
4. If the provisional verdict is `is_prompt_misaligned=true`, run one rebuttal pass:
   re-check whether grader checks can be valid causal proxies and whether repro-harness
   differences are merely implementation details for the same contract.

If a subagent output is malformed/incomplete, minimally repair or retry that phase once.
Only do broad manual exploration if subagent outputs are unusable after retry.

{context}

Decision policy:
- Weigh evidence symmetrically. Do not default toward aligned or misaligned.
- Count only signals backed by quotes/evidence.
- If evidence is genuinely mixed, keep confidence near 0.5.
- Treat prompt requirements as two tiers:
  - Core acceptance requirements: determine pass/fail behavior.
  - Supporting/context requirements: useful context, but not enough alone to call misalignment.

Misalignment signals (toward `is_prompt_misaligned=true`):
- Out-of-scope grader checks not implied by prompt requirements.
- Prompt requirement absent from grader checks (silent drop).
- Grader enforces contradictory format/threshold/location/behavior.
- Submission violates explicit prompt constraint or prohibition.
- Grader+submission agree while missing a prompt requirement.

Alignment signals (toward `is_prompt_misaligned=false`):
- Grader checks a valid invariant for a prompt-described symptom.
- Grader checks a valid causal proxy for the immediate effect of the required fix,
  even if it does not directly observe the agent's exact action.
- Grader is more specific but remains in scope of prompt intent.
- Different repro harnesses test the same underlying behavior.
- Exact grader contract is a precise form of approximate prompt language.
- Agent failure reflects incorrect implementation rather than prompt-grader mismatch.

Proxy-check doctrine (important):
- Do NOT require the grader to check the exact textual target from the prompt.
- If the grader checks a mechanism that is the immediate causal consequence of
  satisfying the prompt requirement, that is usually aligned.
- Declare misalignment only when a correct prompt-compliant solution would still
  fail due to an unrelated or contradictory grader demand.

Return ONLY one JSON object, no markdown fences:
{{
  "reasoning": "A-signals present, M-signals present, and weighted verdict with evidence",
  "requirements": [
    {{
      "requirement": "concrete requirement from prompt",
      "met_by_agent": true or false,
      "checked_by_grader": true or false,
      "evidence": "file + quoted fragment used"
    }}
  ],
  "is_prompt_misaligned": true or false,
  "alignment_issues": ["specific mismatch 1", "specific mismatch 2"],
  "confidence": 0.0 to 1.0
}}

Constraint: `alignment_issues` must be [] when `is_prompt_misaligned` is false,
and non-empty when `is_prompt_misaligned` is true.
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
