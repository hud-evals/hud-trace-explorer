"""Verification Sub-Agent — independently verify claims from QA analysis."""

import base64
from pathlib import Path

from hud import Environment
from hud.tools import BashTool
from hud.tools.filesystem import ReadTool, GrepTool, GlobTool, ListTool
from mcp.types import ImageContent, TextContent

BASE_PATH = "/workspace"
_WORKSPACE = Path(BASE_PATH)

verify_env = Environment(name="qa-verifier")

verify_env.add_tool(BashTool())
verify_env.add_tool(ReadTool(base_path=BASE_PATH))
verify_env.add_tool(GrepTool(base_path=BASE_PATH))
verify_env.add_tool(GlobTool(base_path=BASE_PATH))
verify_env.add_tool(ListTool(base_path=BASE_PATH))


@verify_env.tool()
async def view_screenshot(step: int) -> list[TextContent | ImageContent]:
    """View a screenshot observation by trajectory step number."""
    screenshots_dir = _WORKSPACE / "screenshots"
    path = screenshots_dir / f"step_{step:04d}.png"

    if not path.exists():
        available = sorted(screenshots_dir.glob("step_*.png")) if screenshots_dir.exists() else []
        nums = [p.stem.replace("step_", "") for p in available]
        if nums:
            msg = f"No screenshot for step {step}. Available steps: {', '.join(nums)}"
        else:
            msg = f"No screenshot for step {step}. No screenshots available."
        return [TextContent(type="text", text=msg)]

    raw = path.read_bytes()
    data = base64.standard_b64encode(raw).decode("ascii")

    if raw[:3] == b"\xff\xd8\xff":
        mime = "image/jpeg"
    elif raw[:8] == b"\x89PNG\r\n\x1a\n":
        mime = "image/png"
    elif raw[:4] == b"GIF8":
        mime = "image/gif"
    elif raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        mime = "image/png"

    return [
        TextContent(type="text", text=f"Screenshot at step {step}:"),
        ImageContent(type="image", data=data, mimeType=mime),
    ]


@verify_env.scenario("verify_claims")
async def verify_claims(claims: str):
    """Independently verify a list of claims about a trace.

    The verification agent re-runs commands and tries to disprove each claim.
    It has read + bash access to /workspace/ trace files but cannot edit anything.
    """
    prompt = f"""You are a fast, focused verifier. Check each claim with ONE command each,
then output your JSON verdict. You have a STRICT budget of 15 tool calls total. Do NOT
exceed this. Be efficient: one command per claim, no re-running with variations.

=== CLAIMS TO VERIFY ===

{claims}

=== KEY FILES ===

- evaluation_result.json: grading output (use `jq` to query specific fields)
- file_changes.txt: agent's code changes (use `grep` to find functions/patterns)
- trajectory_summary.txt: agent action sequence
- prompt.txt: task specification
- task_codebase/: ground truth with golden solutions in `tasks/*/golden/` and tests in `tasks/*/tests/`

=== PROCEDURE ===

1. For each claim: run ONE targeted command (jq, grep, head/tail). One command = one check.
   - VERIFIED: evidence confirms the claim
   - REFUTED: evidence contradicts the claim, state what you actually found
   - UNVERIFIED: data not available, state what you searched for
2. Batch related claims: if 2-3 claims reference the same file, check them in one command.
3. After all claims: run ONE counter-check against the analyst's overall conclusion.
   If they blamed the agent, look for grader flaws. If they blamed the eval, check if
   the agent's work was actually correct.
4. Output your JSON verdict IMMEDIATELY after the counter-check. STOP. Do not explore further.

=== RULES ===

- IMPORTANT: All files are in `/workspace/`. Always use absolute paths like
  `/workspace/evaluation_result.json`, NOT relative paths. Your first command
  should be: `cd /workspace && ...`
- ONE command per claim. Do NOT re-run with slight variations.
- Do NOT read entire large files. Use jq, grep, head, or tail.
- Do NOT modify any files.
- If a command fails or times out, mark UNVERIFIED and move on.
- Before marking REFUTED: make sure it is genuinely wrong, not just missing context.
- After outputting JSON, you are DONE. No more commands.

=== OUTPUT FORMAT ===

End your response with this JSON block:

```json
{{
  "claims": [
    {{
      "claim": "restate the claim",
      "command": "the command you ran",
      "output": "actual output (truncated if long)",
      "status": "VERIFIED or REFUTED or UNVERIFIED",
      "reason": "why"
    }}
  ],
  "counter_check": {{
    "description": "what you tested",
    "command": "the command",
    "output": "actual output",
    "finding": "what this means"
  }},
  "result": "CONFIRMED or REJECTED or INCONCLUSIVE"
}}
```"""

    answer = yield prompt
    yield 1.0
