"""Remote test script for the trace-explorer environment.

This script tests the deployed environment on HUD platform.

Usage:
    uv run python remote_test.py
"""

import asyncio
import os

import hud
from hud import Environment
from hud.agents import OpenAIChatAgent
from hud.settings import settings

# Get HUD API key for trace fetching
HUD_API_KEY = settings.api_key or os.environ.get("HUD_API_KEY", "")


async def test_remote_analyze():
    """Test analyze scenario against deployed environment."""
    print("=== Remote Test: Analyze ===")

    # Connect to deployed environment
    env = Environment("trace-explorer").connect_hub("hud-evals/trace-explorer")

    task = env(
        "analyze",
        trace_id="c34369f6-3d10-4a58-a35e-7171d7b4df5d",
        query="Summarize what happened in this trace. What tools were called?",
        hud_api_key=HUD_API_KEY,
    )

    async with hud.eval(task, name="trace-explorer-remote-test") as ctx:
        agent = OpenAIChatAgent.create(model="gpt-4o")  # https://hud.ai/models
        result = await agent.run(ctx, max_steps=15)
        print(f"Done: {result.done}, Reward: {result.reward}")


async def main():
    if not HUD_API_KEY:
        print("ERROR: HUD_API_KEY not set. Set via environment or hud.ai settings.")
        return

    await test_remote_analyze()


if __name__ == "__main__":
    asyncio.run(main())
