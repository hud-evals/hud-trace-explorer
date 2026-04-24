"""Prompt-Grader Alignment — verify that grader and submission match task requirements."""

from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel, Field

from env import env, logger
from qa_common import parse_qa_result, prepare_qa_context


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
    prompt_grader_aligned: bool = Field(description="True if the grader checks what the prompt asks for")
    submission_prompt_aligned: bool = Field(
        description="True if the agent's submission addresses what the prompt asks for"
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
10. Flag any requirement where these disagree or where neither covers it.
11. Special attention: if the grader PASSES but the agent MISSED a prompt requirement,
    that's a prompt-grader misalignment (the grader is too lenient for that requirement).
12. If the grader FAILS the agent on something the prompt never asked for, that's also
    a prompt-grader misalignment (the grader is too strict / checks the wrong thing).

## Required output format
Return ONLY a JSON object — no markdown fences, no extra text:
{{
  "reasoning": "step-by-step analysis of alignment",
  "requirements": [
    {{
      "requirement": "concrete requirement from the prompt",
      "met_by_agent": true or false,
      "checked_by_grader": true or false,
      "evidence": "commands and output proving this"
    }}
  ],
  "prompt_grader_aligned": true or false,
  "submission_prompt_aligned": true or false,
  "alignment_issues": ["specific mismatch 1", "specific mismatch 2"],
  "confidence": 0.0 to 1.0
}}
"""

    answer = yield prompt

    result = parse_qa_result(answer, PromptAlignmentResult)
    if result is None:
        logger.warning("Could not parse agent response into PromptAlignmentResult, scoring 0")
        yield 0.0
        return

    if not result.prompt_grader_aligned or not result.submission_prompt_aligned:
        yield 0.0
    else:
        yield 1.0
