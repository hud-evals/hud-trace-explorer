"""Prompt-alignment subagent environment.

Defines four specialized subagents, one per alignment phase:
1) prompt requirement extraction
2) grader check mapping
3) submission coverage mapping
4) final arbitration
"""

from typing import Literal

from hud import Environment
from hud.tools import BashTool
from hud.tools.filesystem import GlobTool, GrepTool, ListTool, ReadTool
from pydantic import BaseModel, Field

BASE_PATH = "/workspace"

alignment_env = Environment(name="qa-alignment-subagent")

alignment_env.add_tool(BashTool())
alignment_env.add_tool(ReadTool(base_path=BASE_PATH))
alignment_env.add_tool(GrepTool(base_path=BASE_PATH))
alignment_env.add_tool(GlobTool(base_path=BASE_PATH))
alignment_env.add_tool(ListTool(base_path=BASE_PATH))


class ExtractedRequirement(BaseModel):
    id: str = Field(description="Requirement identifier (R1, R2, ...)")
    requirement: str = Field(description="Atomic, testable prompt requirement")
    explicit: bool = Field(description="Whether requirement is explicitly stated")
    tier: Literal["core_acceptance", "supporting_context"] = Field(
        default="core_acceptance",
        description=(
            "core_acceptance: directly determines whether task is solved; "
            "supporting_context: contextual/process detail that should not alone force misalignment"
        ),
    )
    type: Literal["functional", "format", "location", "constraint", "process"] = Field(
        description="Requirement category"
    )
    quote: str = Field(description="Verbatim quote from prompt.txt")
    evidence: str = Field(description="Evidence snippet supporting extraction")


class ExtractPromptRequirementsResult(BaseModel):
    requirements: list[ExtractedRequirement] = Field(default_factory=list)


class GraderRequirementMapping(BaseModel):
    requirement_id: str = Field(description="Requirement ID from extraction phase")
    checked_by_grader: bool = Field(description="Whether grader checks this requirement")
    grader_check: str = Field(description="Name or description of grader check")
    coverage_type: Literal["direct", "invariant", "proxy_effect", "none"] = Field(
        description="How grader coverage maps to prompt requirement"
    )
    evidence: str = Field(description="File and quote supporting this mapping")


class ExtraGraderCheck(BaseModel):
    check: str = Field(description="Grader check not implied by prompt")
    evidence: str = Field(description="File and quote proving this extra check")


class MapGraderChecksResult(BaseModel):
    mapping: list[GraderRequirementMapping] = Field(default_factory=list)
    extra_grader_checks: list[ExtraGraderCheck] = Field(default_factory=list)


class SubmissionRequirementMapping(BaseModel):
    requirement_id: str = Field(description="Requirement ID from extraction phase")
    met_by_agent: bool = Field(description="Whether submission addressed this requirement")
    evidence: str = Field(description="File and quote supporting this judgment")


class ConstraintViolation(BaseModel):
    requirement_id: str = Field(description="Requirement ID violated by submission")
    issue: str = Field(description="Description of explicit constraint violation")
    evidence: str = Field(description="File and quote proving violation")


class AssessSubmissionCoverageResult(BaseModel):
    mapping: list[SubmissionRequirementMapping] = Field(default_factory=list)
    constraint_violations: list[ConstraintViolation] = Field(default_factory=list)


class ArbiterRequirementRow(BaseModel):
    requirement: str = Field(description="Prompt requirement text")
    met_by_agent: bool = Field(description="Whether submission met this requirement")
    checked_by_grader: bool = Field(description="Whether grader checks this requirement")
    evidence: str = Field(description="Combined concise evidence for this row")


class ArbitrateAlignmentResult(BaseModel):
    reasoning: str = Field(description="A/M signal weighting and final justification")
    requirements: list[ArbiterRequirementRow] = Field(default_factory=list)
    is_prompt_misaligned: bool = Field(description="Final alignment verdict")
    alignment_issues: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


@alignment_env.scenario("extract_prompt_requirements", returns=ExtractPromptRequirementsResult)
async def extract_prompt_requirements(user_focus: str = ""):
    """Extract concrete, testable requirements from prompt.txt only."""
    prompt = f"""You are Prompt Requirement Extractor Subagent.

Agenda:
- Extract every concrete, testable requirement from `/workspace/prompt.txt`.
- Build a requirement list that later phases can map against grader and submission.

Scope rules:
- Read ONLY `/workspace/prompt.txt`.
- Do not read grader files, submission files, logs, or task_codebase.
- Do not infer grader behavior. Your job is prompt requirement extraction only.

Extraction rules:
- Include explicit requirements and tightly-justified implicit requirements.
- Include output format/location constraints and prohibitions.
- Split combined statements into separate atomic requirements.
- Keep each requirement independently checkable.
- Every requirement must have a direct quote from prompt.txt as evidence.
- Classify each requirement as:
  - `core_acceptance`: directly determines success/failure of the requested fix/outcome.
  - `supporting_context`: reproduction guidance, narrative context, or non-gating detail.

Focus:
{user_focus}

Return ONLY this JSON object:
{{
  "requirements": [
    {{
      "id": "R1",
      "requirement": "atomic, testable requirement",
      "explicit": true,
      "tier": "core_acceptance | supporting_context",
      "type": "functional | format | location | constraint | process",
      "quote": "exact quote from prompt.txt",
      "evidence": "prompt.txt snippet proving this requirement"
    }}
  ]
}}

Quality bar:
- IDs must be sequential (R1, R2, ...).
- `quote` must be verbatim.
- Do not return empty unless prompt.txt is truly unavailable; if unavailable, still return valid JSON with `requirements: []`.
- No markdown fences and no text outside JSON.
"""

    yield prompt
    yield 1.0


@alignment_env.scenario("map_grader_checks", returns=MapGraderChecksResult)
async def map_grader_checks(requirements_json: str, user_focus: str = ""):
    """Map prompt requirements to grader checks from grader artifacts."""
    prompt = f"""You are Grader Check Mapper Subagent.

Agenda:
- Determine what the grader actually checks.
- Map each prompt requirement ID to concrete grader evidence.
- Detect grader checks that are out-of-scope relative to the prompt.

Inputs:
- Requirements JSON (from subagent 1):
{requirements_json}

Files you should use:
- `/workspace/scenario_setup.json`
- `/workspace/evaluation_result.json`

Scope rules:
- Do not read `/workspace/file_changes.txt` or `/workspace/trajectory_summary.txt`.
- Do not decide whether the agent met requirements; only grader coverage.
- Do not read `/workspace/scenario_code.py`.
- Do not attempt to decrypt grader `source` strings.
- Do not inspect `/tmp/grader_*.py` or reconstruct grader scripts from encoded blobs.
- Treat grader source as opaque; use only declared grader names/metadata and
  observed behavior in `evaluation_result.json`.

Method:
1. Identify grader commands/scripts/rubric checks.
2. For each requirement ID, decide if a grader check covers it directly, as a valid invariant,
   or as a valid proxy of the immediate effect of the required change.
3. Capture one concise evidence snippet per mapping.
4. Record extra grader checks not implied by prompt requirements.

Proxy-check policy (critical):
- A grader does NOT need to observe the agent's exact internal steps.
- It is valid for the grader to check a nearby mechanical condition that should
  immediately result from the required fix (causal proxy).
- Treat these as in-scope coverage when the chain is credible:
  prompt requirement -> required change -> grader-observed effect.
- Do NOT mark out-of-scope merely because the grader wording differs from the
  prompt wording.
- Mark out-of-scope only when a correct solution to the prompt would still fail
  due to an unrelated grader demand.
- A different repro harness/command is not out-of-scope by itself if it validates
  the same underlying contract.

Focus:
{user_focus}

Return ONLY this JSON object:
{{
  "mapping": [
    {{
      "requirement_id": "R1",
      "checked_by_grader": true,
      "grader_check": "name or description of the grader check",
      "coverage_type": "direct | invariant | proxy_effect | none",
      "evidence": "file + quoted fragment"
    }}
  ],
  "extra_grader_checks": [
    {{
      "check": "grader requirement not implied by prompt",
      "evidence": "file + quoted fragment"
    }}
  ]
}}

Quality bar:
- Return one mapping row per requirement ID from input.
- If you cannot find evidence for a requirement, use `checked_by_grader: false`, `coverage_type: none`.
- No markdown fences and no text outside JSON.
"""

    yield prompt
    yield 1.0


@alignment_env.scenario("assess_submission_coverage", returns=AssessSubmissionCoverageResult)
async def assess_submission_coverage(requirements_json: str, user_focus: str = ""):
    """Map prompt requirements to what the agent submission actually addressed."""
    prompt = f"""You are Submission Coverage Analyzer Subagent.

Agenda:
- Determine whether the agent submission addressed each prompt requirement.
- Detect violations of explicit prompt constraints.

Inputs:
- Requirements JSON (from subagent 1):
{requirements_json}

Files you should use:
- `/workspace/file_changes.txt`
- `/workspace/trajectory_summary.txt`

Scope rules:
- Do not read grader files (`scenario_setup.json`, `evaluation_result.json`) for this phase.
- Do not decide prompt-grader alignment. Your job is submission vs prompt only.

Method:
1. For each requirement ID, inspect file changes and final submission behavior.
2. Mark `met_by_agent` true only with positive evidence.
3. If evidence is missing or contradictory, mark false and explain.
4. Record explicit prompt-constraint violations with evidence.

Focus:
{user_focus}

Return ONLY this JSON object:
{{
  "mapping": [
    {{
      "requirement_id": "R1",
      "met_by_agent": true,
      "evidence": "file + quoted fragment"
    }}
  ],
  "constraint_violations": [
    {{
      "requirement_id": "R2",
      "issue": "explicit prompt constraint violated",
      "evidence": "file + quoted fragment"
    }}
  ]
}}

Quality bar:
- Return one mapping row per requirement ID from input.
- Do not assume intent; rely on observable submission artifacts.
- No markdown fences and no text outside JSON.
"""

    yield prompt
    yield 1.0


@alignment_env.scenario("arbitrate_alignment", returns=ArbitrateAlignmentResult)
async def arbitrate_alignment(
    requirements_json: str,
    grader_mapping_json: str,
    submission_mapping_json: str,
    user_focus: str = "",
):
    """Weigh alignment vs misalignment signals and produce final verdict."""
    prompt = f"""You are Alignment Arbiter Subagent.

Agenda:
- Use ONLY structured outputs from prior phases.
- Produce final prompt-grader-submission alignment verdict.

Inputs:
- Requirements:
{requirements_json}

- Grader mapping:
{grader_mapping_json}

- Submission mapping:
{submission_mapping_json}

Focus:
{user_focus}

Decision policy:
- Weigh evidence symmetrically. Do not default to aligned or misaligned.
- Count only signals supported by quoted evidence in upstream mappings.
- Distinguish:
  - prompt-grader mismatch
  - prompt-submission mismatch
  - collusive grader-submission agreement that misses prompt requirements

Misalignment signals (vote toward true):
- M1 out-of-scope grader checks
- M2 contradiction in threshold/format/location/behavior
- M3 wrong artifact checked by grader
- M4 grader penalizes behavior not implied by prompt
- M5 prompt requirement untested by grader
- M6 submission violates explicit prompt constraint
- M7 grader+submission align but miss prompt requirement

Alignment signals (vote toward false):
- A1 grader checks a valid invariant for prompt symptom
- A1b grader checks a valid proxy of the immediate effect of the required change
- A2 prompt repro script differs but tests same behavior
- A3 hidden/additional cases were declared in prompt
- A4 prompt approximate, grader exact on same contract
- A5 grader more specific but still in-scope
- A6 agent fix wrong; failure is submission issue, not prompt-grader mismatch

Verdict guardrails (strict):
- Return `is_prompt_misaligned=true` only if you can state a concrete counterfactual:
  a correct prompt-compliant solution would still fail due to unrelated/contradictory grader demand.
- Do NOT call misalignment based only on:
  - different repro harness/command,
  - grader being more specific than prompt wording,
  - missing checks on `supporting_context` requirements.
- Weight `core_acceptance` requirements higher than `supporting_context` requirements.

Return ONLY this JSON object:
{{
  "reasoning": "A-signals and M-signals with evidence, plus final weighing",
  "requirements": [
    {{
      "requirement": "prompt requirement text",
      "met_by_agent": true,
      "checked_by_grader": true,
      "evidence": "concise combined evidence"
    }}
  ],
  "is_prompt_misaligned": false,
  "alignment_issues": [],
  "confidence": 0.5
}}

Constraints:
- `alignment_issues` must be [] when `is_prompt_misaligned` is false.
- `alignment_issues` must be non-empty when `is_prompt_misaligned` is true.
- Use requirement-level evidence, not generic claims.
- Do not call a case misaligned solely because the grader validates the causal
  mechanism behind a high-level prompt outcome.
- No markdown fences and no text outside JSON.
"""

    yield prompt
    yield 1.0
