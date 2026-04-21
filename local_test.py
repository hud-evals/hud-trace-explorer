"""Local test script for the trace-explorer environment.

Usage:
    python local_test.py
    python local_test.py --trace-id <uuid>
"""

import asyncio
import json
import os

import hud
from hud.agents import OpenAIChatAgent
from hud.settings import settings
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageCustomToolCallParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallUnionParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from env import env

# Get HUD API key for trace fetching
HUD_API_KEY = settings.api_key or os.environ.get("HUD_API_KEY", "")


async def test_tools_standalone():
    """Test environment tools directly."""
    print("=== Test 1: Standalone Tools ===")

    async with env:
        print(f"Tools: {[t.name for t in env.as_tools()]}")


async def test_analyze_manual():
    """Test analyze scenario with manual agent loop."""
    print("\n=== Test 2: Analyze (Manual Agent Loop) ===")

    from openai import AsyncOpenAI

    # Use HUD inference gateway - see all models at https://hud.ai/models
    client = AsyncOpenAI(base_url="https://inference.hud.ai", api_key=settings.api_key)

    task = env(
        "analyze",
        trace_id="c34369f6-3d10-4a58-a35e-7171d7b4df5d",
        query="Summarize what happened in this trace. What tools were called?",
        hud_api_key=HUD_API_KEY,
    )

    async with hud.eval(task) as ctx:
        prompt_text = ctx.prompt or ""
        user_message: ChatCompletionUserMessageParam = {"role": "user", "content": prompt_text}
        messages: list[ChatCompletionMessageParam] = [user_message]

        for _ in range(10):  # max steps
            response = await client.chat.completions.create(
                model="gpt-4o",  # https://hud.ai/models
                messages=messages,
                tools=ctx.as_openai_chat_tools(),
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                print(f"Final response: {msg.content[:500] if msg.content else 'No content'}")
                break

            tool_calls: list[ChatCompletionMessageToolCallUnionParam] = []
            for tc in msg.tool_calls:
                if tc.type == "function":
                    function_call: ChatCompletionMessageFunctionToolCallParam = {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    tool_calls.append(function_call)
                elif tc.type == "custom":
                    custom_call: ChatCompletionMessageCustomToolCallParam = {
                        "id": tc.id,
                        "type": "custom",
                        "custom": {
                            "name": tc.custom.name,
                            "input": tc.custom.input,
                        },
                    }
                    tool_calls.append(custom_call)

            assistant_message: ChatCompletionAssistantMessageParam = {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": tool_calls,
            }
            messages.append(assistant_message)
            for tc in msg.tool_calls:
                result = await ctx.call_tool(tc)
                content: str
                if isinstance(result, dict):
                    raw_content = result.get("content", "")
                else:
                    raw_content = result

                if isinstance(raw_content, str):
                    content = raw_content
                else:
                    content = json.dumps(raw_content)

                tool_message: ChatCompletionToolMessageParam = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content,
                }
                messages.append(tool_message)


async def test_analyze_agent():
    """Test analyze scenario with OpenAIChatAgent."""
    print("\n=== Test 3: Analyze (Agent) ===")

    task = env(
        "analyze",
        trace_id="c34369f6-3d10-4a58-a35e-7171d7b4df5d",
        query="Was there any reward hacking in this trace?",
        hud_api_key=HUD_API_KEY,
        data_sources=["telemetry", "environment", "worker"],
    )

    async with hud.eval(task) as ctx:
        agent = OpenAIChatAgent.create(model="gpt-4o")  # https://hud.ai/models
        result = await agent.run(ctx, max_steps=15)
        print(f"Done: {result.done}, Reward: {result.reward}")


async def test_with_validation():
    """Test with includes/excludes validation patterns."""
    print("\n=== Test 4: With Validation Patterns ===")

    task = env(
        "analyze",
        trace_id="c34369f6-3d10-4a58-a35e-7171d7b4df5d",
        query="Did this trace complete successfully? What was the final reward?",
        hud_api_key=HUD_API_KEY,
        includes=["reward"],  # Must mention reward in response
    )

    async with hud.eval(task) as ctx:
        agent = OpenAIChatAgent.create(model="gpt-4o-mini")
        result = await agent.run(ctx, max_steps=10)
        print(f"Done: {result.done}, Reward: {result.reward}")


async def main():
    if not HUD_API_KEY:
        print("ERROR: HUD_API_KEY not set. Set via environment or hud.ai settings.")
        return

    await test_tools_standalone()
    # await test_analyze_manual()
    await test_analyze_agent()
    # await test_with_validation()


if __name__ == "__main__":
    asyncio.run(main())
