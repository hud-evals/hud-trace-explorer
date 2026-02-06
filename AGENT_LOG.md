# Agent Log: Trace Explorer Environment

Development log for the `hud-trace-explorer` environment.

---

## Purpose

Create an HUD environment that allows agents to analyze evaluation traces using coding/filesystem tools. The environment:
1. Fetches trace data from the HUD platform API (telemetry, environment logs, worker logs)
2. Writes data to files in `/workspace` for agent exploration
3. Provides bash, grep, read, edit tools for analysis
4. Evaluates responses against include/exclude patterns

---

## Development Timeline

### Initial Request

User requested a new HUD environment in `ENV_TEMPLATES` that:
- Loads traces via HUD platform API
- Accepts scenario parameters: `hud_api_key`, `trace_id`, `data_sources`, `query`, `includes`, `excludes`
- Provides coding/filesystem tools from the HUD SDK (not custom implementations)
- Supports both Claude/OpenCode-style and Gemini-style tools

### Key Decisions

1. **Use built-in HUD SDK tools** - Initially considered re-implementing tools, but discovered `hud.tools` provides `BashTool`, `EditTool`, `ReadTool`, `GrepTool`, `GlobTool`, `ListTool` and Gemini variants. Much cleaner.

2. **API key as scenario parameter only** - User specified `hud_api_key` should only come from scenario input, not environment variables.

3. **Default data_sources** - Set to `["telemetry"]` by default, can optionally include `"environment"` and `"worker"`.

4. **API endpoint** - Uses `/telemetry/traces/{trace_id}` with query params for `include_trajectory`, `include_logs`, `include_rollout_logs`.

---

## Issues Encountered & Fixes

### 1. Path Escapes Base Directory (Windows Local Testing)

**Error:**
```
ToolError: Path escapes base directory: \workspace\environment_logs.txt
```

**Cause:** 
- `WORKSPACE_DIR = Path("/workspace")` on Windows becomes `C:\workspace`
- Tools default to `base_path="."` (current directory)
- Agent tried to read `/workspace/...` which was outside the tool's sandbox

**Fix:**
- Added dynamic workspace detection:
  ```python
  if sys.platform != "win32" and Path("/workspace").exists():
      WORKSPACE_DIR = Path("/workspace")
      BASE_PATH = "/workspace"
  else:
      WORKSPACE_DIR = Path("./workspace").resolve()
      BASE_PATH = str(WORKSPACE_DIR)
  ```
- Configured all tools with `base_path=BASE_PATH` or `base_directory=BASE_PATH`
- Updated prompt to show relative filenames instead of absolute paths

### 2. UnicodeEncodeError on Windows

**Error:**
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2713'
```

**Cause:** Windows default encoding (cp1252) can't handle Unicode characters in trace data.

**Fix:** Added `encoding="utf-8"` to all `Path.write_text()` calls.

### 3. TypeError: 'NoneType' object is not iterable

**Error:** Occurred when iterating over `span.get("exceptions", [])`

**Cause:** The `exceptions` field could be `None` even with default.

**Fix:** Changed to `span.get("exceptions") or []` to ensure iterable.

### 4. Messy Log Output

**Issue:** Environment logs contained raw nested JSON like `{'timestamp': 1768682371929, 'log': '...'}`

**Fix:** Added preprocessing functions:
- `_preprocess_environment_logs()` - Extracts `log`/`message` fields, formats timestamps
- `_preprocess_worker_logs()` - Cleans up worker log format  
- `_preprocess_trajectory()` - Creates readable summary with tool calls, errors, stats

### 5. Agent Responses Too Verbose

**Issue:** Agent gave lengthy multi-paragraph analyses instead of direct answers.

**Fix:** Updated prompt to say:
```
Be VERY BRIEF - respond with ONE short paragraph that directly answers the question. 
No preamble, no lengthy explanations.
```

### 6. Docker Build Failed - README.md Not Found

**Error:**
```
OSError: Readme file does not exist: README.md
```

**Cause:** `pyproject.toml` declares `readme = "README.md"` but Dockerfile only copied `pyproject.toml`.

**Fix:** Updated Dockerfile:
```dockerfile
COPY pyproject.toml README.md ./
```

### 7. hatchling Build Error - Unable to Determine Files

**Error:**
```
ValueError: Unable to determine which files to ship
```

**Cause:** hatchling couldn't auto-detect package structure (we have standalone files, not a package).

**Fix:** Added explicit build targets in `pyproject.toml`:
```toml
[tool.hatch.build.targets.wheel]
only-include = ["env.py"]

[tool.hatch.build.targets.sdist]
only-include = ["env.py", "pyproject.toml", "README.md"]
```

---

## File Structure

```
hud-trace-explorer/
├── env.py              # Main environment code
├── pyproject.toml      # Package config with hud-python, httpx deps
├── Dockerfile.hud      # Container build for HUD platform
├── README.md           # Usage documentation
├── local_test.py       # Local testing script
├── remote_test.py      # Remote/deployed testing script
├── .env.example        # Example environment variables
├── .gitignore          # Git ignore patterns
└── AGENT_LOG.md        # This file
```

---

## Tools Provided

| Tool Type | Claude/OpenCode | Gemini |
|-----------|-----------------|--------|
| Shell | `BashTool` | `GeminiShellTool` |
| Edit | `EditTool` | `GeminiEditTool` |
| Read | `ReadTool` | `GeminiReadTool` |
| Search | `GrepTool` | `GeminiSearchTool` |
| Glob | `GlobTool` | `GeminiGlobTool` |
| List | `ListTool` | `GeminiListTool` |

---

## API Integration

Fetches from: `GET /telemetry/traces/{trace_id}`

Query params:
- `include_trajectory` (bool) - telemetry spans
- `include_logs` (bool) - environment container logs
- `include_rollout_logs` (bool) - orchestrator/worker logs

Response model (`TraceDetailResponse`):
- `trace_id`, `job_id`, `status`, `reward`
- `external_id`, `task_id`, `task_version_id`
- `scenario` (str), `scenario_args` (list), `prompt`
- `trajectory` (list of spans), `trajectory_length`
- `logs` (list), `logs_count`, `logs_error`
- `rollout_logs`
- `error`, `metadata`

---

## Testing

### Local
```bash
cd hud-trace-explorer
uv sync
python local_test.py
```

### Deploy
```bash
hud deploy
```

---

## Status

✅ Environment created and tested locally
✅ Dockerfile fixed for remote deployment
⏳ Awaiting successful remote build
