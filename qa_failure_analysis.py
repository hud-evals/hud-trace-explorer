"""Failure Analysis — identify all problems that caused a trace to fail."""

from collections.abc import AsyncGenerator
from typing import Any, Literal

from pydantic import BaseModel, Field

from env import env, logger
from qa_common import parse_qa_result, prepare_qa_context


class Problem(BaseModel):
    problem: str = Field(
        description="Short label for this problem (e.g. 'prompt-grader cell mismatch', "
        "'agent looped on step 12', 'MCP tool returned 500')"
    )
    description: str = Field(
        description="What went wrong — include the specific function, variable, or code path "
        "that caused the failure, not just a symptom label"
    )
    evidence: str = Field(
        description="The commands you ran and their output that prove this problem exists. "
        "Include the actual terminal output, not a summary.",
    )
    root_cause: str = Field(
        default="",
        description="The exact code-level cause — the specific function, condition, or logic "
        "error, not a symptom label. Include the relevant code snippet or pattern. "
        "Bad: 'incorrect output'. Good: 'parse_response() splits on comma but input uses "
        "semicolons, so the list has 1 element instead of 5'",
    )
    impact: str = Field(
        description="How this affected the reward (e.g. 'lost 0.5 points because cell A2 was checked instead of B1')"
    )
    fault: Literal["agent", "eval", "platform", "unclear"] = Field(
        default="unclear",
        description="Who is most responsible. Use 'unclear' when reasonable arguments exist for multiple parties",
    )
    failure_mode: str = Field(
        default="",
        description="Short snake_case label for the type of failure "
        "(e.g. 'implementation_bug', 'skipped_steps', 'grader_bug', 'hallucination')",
    )
    what_should_have_happened: str = Field(
        default="",
        description="What the correct behavior would have been — for agent faults, what the "
        "agent should have done; for eval faults, how the grader should work",
    )


class FailureAnalysisResult(BaseModel):
    summary: str = Field(
        default="",
        description="1-3 sentence overview of why the trace failed",
    )
    problems: list[Problem] = Field(
        default_factory=list,
        description="Every distinct problem found — one entry per issue, "
        "there can be multiple with different or same fault attributions",
    )
    confidence: Literal["high", "medium", "low"] = Field(default="medium", description="Confidence in the analysis")


def _normalize_fault(fault: str) -> str:
    """Map fault labels to scoring categories."""
    return {
        "agent": "agent_failure",
        "eval": "eval_failure",
        "platform": "platform_failure",
        "unclear": "disputed",
        "disputed": "disputed",
    }.get(fault, fault)


@env.scenario("failure_analysis", returns=FailureAnalysisResult)
async def failure_analysis(
    trace_id: str,
    hud_api_key: str,
    query: str = "",
    ground_truth: str | None = None,
) -> AsyncGenerator[Any, None]:
    """Analyze why a trace failed — find all problems, not just a single category."""
    _, _, context = await prepare_qa_context(trace_id, hud_api_key, "Failure analysis")

    user_focus = query.strip() or "Find every problem that contributed to this trace's failure."

    prompt = f"""You are a failure analyst. A trace failed or received a low reward. Your job
is to find **every distinct problem** that contributed to the failure.

{context}

## Your goal: find ALL the problems

A single failed trace can have multiple independent problems. Find each one separately.
Don't stop at the first issue you find — keep looking for more.

Example: the agent computed the wrong value for the 2023 cohort (agent problem), AND
the grader checks cell A2 when the prompt says B1 (eval problem). Two independent issues,
two separate entries.

## For each problem, describe:

1. **What happened** — the specific, concrete issue. Not "the grader is wrong" but "the grader
   checks cell A2 but the prompt says B1."
2. **Evidence** — the commands you ran and their actual output proving this problem exists.
   Include the actual output, not a summary.
3. **Root cause** — why this happened. Trace back to the underlying cause.
4. **Impact** — how much reward was lost, or how this blocked the agent.
5. **Fault** — who is most responsible:
   - **agent**: The agent could reasonably have done better. Examples: wrong computation,
     hallucination, looping, incomplete implementation, bad strategy.
   - **eval**: The grading is flawed. Examples: grader checks wrong cells, rubric contradicts
     prompt, format-strict grader rejects correct answer, incomplete gold labels, ambiguous
     task language interpreted one way by agent but graded another way.
   - **platform**: Infrastructure failure outside the agent's control (e.g. MCP tool crashed,
     orchestrator killed the run).
   - **unclear**: Reasonable arguments exist for multiple attributions — explain both sides.
6. **What should have happened** — what correct behavior looks like.

## Patterns to check BEFORE attributing fault to the agent

- **Formatting-strict graders**: Agent's answer is semantically correct but rejected for cosmetic
  differences (e.g. trailing zeros, thousands separators, whitespace, JSON indentation).
- **Rubric-prompt mismatch**: The grading rubric requires something the prompt never asked for, or
  contradicts what the prompt says.
- **Incomplete gold labels**: Agent found correct answers or real bugs that aren't in the reference
  set, and gets penalized for "false positives" that are actually true positives.
- **Broken / stub graders**: Grader always returns 1.0 regardless of quality, or crashes and
  returns a default score. If reward is high (>=0.8) but the agent did clearly wrong or no work,
  the grader is broken.
- **Ambiguous task language**: Underspecified criteria that admit multiple valid readings.
- **Unstated requirements**: Grader checks for steps or formats that the task prompt never mentions.

## Focus
{user_focus}

## PREREQUISITE — do this before anything else

Your FIRST action must be to explore `/workspace/task_codebase/`. This contains the ground
truth: grading logic, reference solutions, and test suites. Run these commands NOW:
1. `ls -R /workspace/task_codebase/` to see the full tree
2. Read grading scripts (e.g. `env.py`, `task.py`, or files in `grading/`)
3. If `tasks/*/golden/` exists, read the reference solutions — these show correct code
4. If `tasks/*/tests/` exists, read the test suites — these show what the grader checks

Not every task has golden solutions or test directories. Read whatever is available —
grading scripts, scenario configs, or task definitions all provide useful ground truth.
Do NOT skip this step.

## Analysis steps (after reading task_codebase)
1. Check the evaluation result for reward breakdown and subscore details.
2. Read the scenario setup — grading criteria, rubric, grader scripts, and expected answers.
3. **Critically evaluate each grader**: Does the rubric match what the prompt actually asked? Are the
   expected answers reasonable? Could a correct solution be rejected by this grader? Is the grader
   checking the right thing?
4. Read the trajectory summary for the agent's action sequence.
5. Search trajectory and logs for error patterns, failed tool calls, or platform faults.
6. **Compare the agent's final submission against the task prompt requirements.** Did the agent
   actually address what was asked, or did it solve the wrong problem / misread the prompt?
7. **Cross-reference** the golden solution (from the prerequisite) against the agent's code in
   `file_changes.txt` to pinpoint exactly where the implementation diverged. If you find a
   reference implementation, diff it against the agent's version to identify bugs.
8. **Extract the exact code-level root cause for each failure.** Don't stop at symptom labels.
   For each failing test or subscore, find the specific function, condition, or logic in the
   agent's code that produced the wrong behavior. Use `grep` to extract the relevant code
   snippet from `file_changes.txt`. A developer reading your root cause should be able to
   fix the bug without re-investigating.
9. **Evaluate the agent's strategy, not just its code.** Did the agent run the test suite
   before submitting? Did it verify its own work against the grading criteria? If the grading
   mechanism (e.g. pytest, bash checks) was available and the agent never ran it, that is a
   separate strategic failure worth reporting.

## Evidence standard

Every problem you report must include the command you ran and the output you got
in the `evidence` field. If you can't point to a command and its output, you
don't have a finding — you have a guess.

## Stress-test your own conclusion

Before finalizing, actively try to break your verdict. Pick the strongest
counter-argument against your conclusion and run a command to test it:

- **You're blaming the agent?** Read the grader script or expected answers.
  Is the grader actually correct? Could a perfect agent still fail this grader?
  Run the grader's comparison logic yourself if possible.
- **You're blaming the eval?** Re-read the agent's final output. Is the agent's
  answer actually right, or are you giving it too much credit? Extract the
  agent's answer and the expected answer side-by-side.
- **You're blaming the platform?** Check whether the agent had alternative tools
  or retry options. Search the trajectory for workaround attempts.
- **You found only one problem?** Search for a second. Grep for other error
  patterns, check other subscores, look at steps after the first failure.
- **Reward seems straightforward?** Verify how it was computed — extract the
  reward function or grader weights from scenario_setup.json and calculate
  the expected score yourself.
- **Your root causes are vague?** "Implementation bug" or "incorrect output" are
  symptoms. Push deeper: which function? Which condition? What does the code
  actually do vs what it should do? Extract the code and explain the mismatch.

## Common analytical errors — AVOID THESE

These mistakes cause wrong verdicts. Check yourself against each one:

1. **Recency bias**: Over-weighting the last few trajectory steps while ignoring
   earlier actions that set up the failure. The root cause is often in the first
   third of the trajectory, not the last tool call.

2. **Anchoring to the first error**: You find one problem and stop looking.
   Most traces have 1-3 problems — finding the first one is not enough. After
   your first finding, explicitly search for a second.

3. **Confusing correlation with causation**: "Error X happened before failure Y"
   does not mean X caused Y. Check whether X was actually on the critical path.
   Could the trace have succeeded despite X?

4. **Outcome bias**: "The agent failed, so its strategy must have been wrong."
   WRONG. The strategy might have been reasonable but the eval was unfair, or
   a platform failure blocked a correct approach. Judge the strategy on its own
   merits, then separately check whether external factors intervened.

5. **Effort = quality**: A long trajectory with many tool calls does NOT mean
   the agent tried hard and just got unlucky. It may have been looping, retrying
   the same failed approach, or doing busy-work. Check what the steps actually
   accomplished, not how many there were.

6. **Trusting the agent's self-assessment**: The agent may claim "I have
   completed the task" or "All tests pass" in its final message. These claims
   are frequently wrong. Always verify against the grader output and actual
   file state, not the agent's narrative.

## Independent verification

Before your final verdict, call `verify_failure_claims` with your key claims (one per line).
A separate agent will re-run commands and try to disprove each one.

After receiving verification results — you have a STRICT BUDGET of 3 tool calls:
- REFUTED claims: remove them from your analysis or mark fault as "unclear"
- Do NOT re-investigate or re-prove refuted claims — that is wasted effort
- Your very next response MUST be your final JSON answer — no more exploration

## Required output format
Return ONLY a JSON object — no markdown fences, no extra text. Be thorough — write as much
detail as needed for each problem. Don't summarize or truncate your analysis.
{{
  "summary": "1-3 sentence overview of why the trace failed",
  "problems": [
    {{
      "problem": "short label for this problem",
      "description": "what specifically went wrong — name the function, variable, or code path",
      "evidence": "the commands you ran and their actual output that prove this",
      "root_cause": "the exact code-level cause — which function, what it does vs what it should do",
      "impact": "how this affected the reward — which subscores, how many points lost",
      "failure_mode": "concise label for the type of mistake (e.g. 'incomplete_implementation', 'test_suite_not_executed')",
      "fault": "agent | eval | platform | unclear",
      "what_should_have_happened": "what correct behavior looks like"
    }}
  ],
  "confidence": "high | medium | low"
}}

List EVERY problem you found. A trace can have 1 problem or 5 — report what you actually find.
Use whatever labels fit — just keep them reusable across tasks.
"""

    answer = yield prompt

    result = parse_qa_result(answer, FailureAnalysisResult)
    if result is None:
        logger.warning("Could not parse agent response into FailureAnalysisResult, scoring 0")
        yield 0.0
        return

    if ground_truth is not None:
        accepted = {s.strip().lower() for s in ground_truth.split("|") if s.strip()}

        found_labels = set()
        for p in result.problems:
            found_labels.add(p.problem.strip().lower())
            found_labels.add(_normalize_fault(p.fault))
            if p.failure_mode:
                found_labels.add(p.failure_mode.strip().lower())

        score = 1.0 if found_labels & accepted else 0.0
    else:
        score = 1.0

    yield score
