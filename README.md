# Trace Explorer Environment

An HUD environment for analyzing evaluation traces using coding/filesystem tools.

## Overview

This environment loads HUD trace data (telemetry, environment logs, worker logs) into files, then provides agents with bash, grep, edit, and read tools to explore and analyze the data.

## Scenario: `analyze`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `trace_id` | `str` | Yes | The trace UUID to analyze |
| `query` | `str` | Yes | The analysis query/question |
| `hud_api_key` | `str` | Yes | HUD API key |
| `data_sources` | `list[str]` | No | Sources to load: `["telemetry", "environment", "worker"]` (default: `["telemetry"]`) |
| `includes` | `list[str]` | No | Patterns that MUST appear in response for reward |
| `excludes` | `list[str]` | No | Patterns that MUST NOT appear in response |

### Files Created

The scenario creates these files in `/workspace`:

| File | Description |
|------|-------------|
| `metadata.json` | Full trace metadata (id, job_id, status, reward, scenario, external_id, task_id, etc.) |
| `prompt.txt` | The original prompt given to the agent (if present) |
| `trajectory.json` | Full trajectory data (if telemetry enabled) |
| `trajectory_summary.txt` | Human-readable trajectory summary |
| `screenshots_index.txt` | Index mapping step numbers to screenshot files (if telemetry enabled) |
| `screenshots/step_XXXX.png` | CUA screenshots (observations) for each step that has them (if telemetry enabled) |
| `environment_logs.txt` | Environment container logs (if environment enabled) |
| `worker_logs.txt` | Orchestrator/worker logs (if worker enabled) |

### Tools Available

**Claude/OpenCode-style:**
- `bash` - Execute shell commands
- `str_replace_editor` - View/edit files with line numbers
- `read_file` - Read file contents
- `grep` - Search file contents
- `glob` - Find files by pattern
- `list_dir` - List directory contents

**Gemini-style:**
- `shell` - Execute shell commands
- `edit_file` - Instruction-based file editing
- `read_file` - Read file contents
- `search_files` - Search file contents
- `find_files` - Find files by pattern
- `list_directory` - List directory contents

## Usage

### Local Testing

```bash
python local_test.py
```

### Programmatic Usage

```python
import hud
from hud.agents import OpenAIChatAgent
from env import env

# Create a task
task = env(
    "analyze",
    trace_id="c34369f6-3d10-4a58-a35e-7171d7b4df5d",
    query="What errors occurred in this trace?",
    hud_api_key="your-api-key",
    data_sources=["telemetry", "environment", "worker"],
    includes=["error"],  # Response must mention "error"
    excludes=["success"],  # Response must not claim "success" if there were errors
)

# Run with an agent
async with hud.eval(task) as ctx:
    agent = OpenAIChatAgent.create(model="gpt-4o")  # https://hud.ai/models
    result = await agent.run(ctx, max_steps=20)
    print(f"Reward: {result.reward}")
```

### Docker Build

```bash
docker build -t trace-explorer -f Dockerfile.hud .
```

### Deploy to HUD

```bash
hud deploy --name trace-explorer
```

## Example Queries

```python
# Debugging failed traces
task = env("analyze",
    trace_id="...",
    query="Why did this trace fail? What errors occurred?",
    hud_api_key=api_key,
    includes=["error", "failed"],
)

# Performance analysis
task = env("analyze",
    trace_id="...",
    query="Which tool calls took the longest? Are there any timeouts?",
    hud_api_key=api_key,
    data_sources=["telemetry"],
)

# Screenshot (CUA observation) analysis
task = env("analyze",
    trace_id="...",
    query="What UI elements are visible in the screenshots? What actions did the agent take?",
    hud_api_key=api_key,
    data_sources=["telemetry"],  # Screenshots downloaded automatically with telemetry
)

# Log analysis
task = env("analyze",
    trace_id="...",
    query="Look for any exceptions or stack traces in the logs",
    hud_api_key=api_key,
    data_sources=["environment", "worker"],
)
```

## Development Workflow

## Installing Packages

Install [uv](https://docs.astral.sh/uv/) and set up dev dependencies:

```bash
uv sync --extra dev
```

### Git Hooks

Enable the shared pre-push hook (runs ruff, pyright, pytest before each push):

```bash
git config core.hooksPath .githooks
```

### Code Quality

```bash
uv run ruff format . --check   # Formatting
uv run ruff check .            # Linting
uv run pyright                 # Type checking
```
