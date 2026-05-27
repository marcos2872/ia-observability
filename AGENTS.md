# AGENTS.md

## Overview

Reference project demonstrating LLM observability with MLflow GenAI. Nine standalone demo modules, each creating its own MLflow experiment. No tests, no CI — scripts only.

## Stack

- **Python 3.14** / `uv` (no pip, no poetry)
- **MLflow GenAI** (`mlflow[genai]>=3.10`) for tracing, evaluation, judges
- **OpenAI SDK** pointing to MLflow AI Gateway (not direct OpenAI API)
- **hatchling** build backend, `src/` layout

## Commands

```bash
uv sync                  # install deps
uv run <entrypoint>      # run a single demo (see pyproject.toml [project.scripts])
make <target>            # same, via Makefile (make help for list)
make all                 # run all 9 demos sequentially
```

No linter, formatter, or test suite configured.

## Environment

Requires `.env` at repo root (see `.env.example`):

| Var | Purpose |
|-----|---------|
| `mlflow_url` | MLflow tracking server |
| `mlflow_openia_url` | AI Gateway base URL (OpenAI-compatible) |
| `mlflow_model` | Model name for inference |
| `mlflow_judge_model` | Model for LLM judges (can differ from inference model) |

`config.py` loads `.env` via relative path from the package. `api_key="not-needed"` since the Gateway handles auth.

## Architecture

```
src/ia_observability/
  config.py                  # setup_mlflow(), get_client(), patch_judge_timeout(), constants
  tracing_basics.py          # 01 - auto-tracing, decorators, manual spans
  token_usage.py             # 02 - token counts + manual cost attribution
  sessions.py                # 03 - multi-turn sessions, user tracking
  evaluation.py              # 04 - mlflow.genai.evaluate() with built-in scorers
  judges.py                  # 05 - custom LLM judges + code-based scorers
  version_tracking.py        # 06 - LoggedModel versioning
  production_monitoring.py   # 07 - async tracing, sampling, feedback
  experiment_comparison.py   # 08 - benchmark across configs
  tool_calls.py              # 09 - tool calling with AGENT/TOOL/CHAT_MODEL spans
```

Each module is self-contained with a `main()` entrypoint registered in `pyproject.toml`.

## Critical Gotchas

1. **MLflow does NOT auto-calculate cost for self-hosted models.** Must manually set `span.set_attribute("mlflow.llm.cost", {...})`.

2. **Must call `mlflow.flush_trace_async_logging()` before `get_trace()` / `search_traces()`** or results will be `None`.

3. **Built-in scorers default to `openai:/gpt-4.1-mini` as judge.** Pass `model="gateway:/<model_name>"` to use the local gateway.

4. **MLflow hardcodes 60s timeout for judge calls.** Use `patch_judge_timeout(300)` from `config.py` for slow models.

5. **Code-based scorers (`@scorer`) must return `Feedback`, `bool`, `float`, `str`, or `list[Feedback]`** — NOT dict.

6. **`SpanType.TOOL`** makes spans appear in MLflow's "Tool calls" UI tab.

7. **Experiment auto-restore:** `setup_mlflow()` restores deleted experiments to avoid "experiment already exists but is deleted" errors.
