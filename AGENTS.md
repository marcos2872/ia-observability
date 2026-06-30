# AGENTS.md

## Overview

Reference project demonstrating LLM observability with MLflow GenAI. Thirteen standalone demo modules organized in 4 didactic parts (Fundamentals → Evaluation → Production → Advanced), each creating its own MLflow experiment. No tests, no CI — scripts only.

## Stack

- **Python 3.14** / `uv` (no pip, no poetry)
- **MLflow GenAI** (`mlflow[genai]>=3.10`) for tracing, evaluation, judges
- **OpenAI SDK** pointing to MLflow AI Gateway (OpenAI-compatible, not direct OpenAI API)
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
| `mlflow_url` | MLflow tracking server (SQL backend required for datasets) |
| `mlflow_openia_url` | MLflow AI Gateway base URL (OpenAI-compatible) |
| `mlflow_model` | Model name for inference |
| `mlflow_judge_model` | Model for LLM judges and GEPA reflection (can differ from inference model) |

`config.py` loads `.env` via relative path from the package. `get_client()` points the OpenAI SDK at the AI Gateway with `api_key="not-needed"` (the Gateway handles auth). Built-in scorers use the native `gateway:/<model>` URI. GEPA reflection uses `openai:/<model>` via litellm with `OPENAI_API_BASE` pointing at the Gateway (which is OpenAI-compatible).

## Architecture

```
src/ia_observability/
  config.py                  # setup_mlflow(), get_client(), patch_judge_timeout(), constants
  parte1_fundamentos/        # 🟢 Tracing + Tokens
    tracing_basics.py        # 01 - auto-tracing, decorators, manual spans
    token_usage.py           # 02 - token counts + manual cost attribution
  parte2_avaliacao/          # 🟡 Evaluation + Judges + Datasets
    evaluation.py            # 04 - mlflow.genai.evaluate() with built-in scorers
    judges.py                # 05 - custom LLM judges + code-based scorers
    datasets_demo.py         # 12 - evaluation datasets: upload + fetch
  parte3_producao/           # 🟡🔴 Sessions + Tools + Monitoring
    sessions.py              # 03 - multi-turn sessions, user tracking
    version_tracking.py      # 06 - LoggedModel versioning
    production_monitoring.py # 07 - async tracing, sampling, feedback
    tool_calls.py            # 09 - tool calling with AGENT/TOOL/CHAT_MODEL spans
    langchain_agent.py       # 11 - tool calling + sessions via LangChain
  parte4_avancado/           # 🔴 Benchmark + Prompts + Optimization
    experiment_comparison.py # 08 - benchmark across configs
    prompt_management.py     # 10 - prompt registry, versioning, linked prompts
    prompt_optimization.py   # 13 - prompt optimization: GEPA + Metaprompting
```

Each module is self-contained with a `main()` entrypoint registered in `pyproject.toml`.

## Critical Gotchas

1. **MLflow does NOT auto-calculate cost for self-hosted models.** Must manually set `span.set_attribute("mlflow.llm.cost", {...})`.

2. **Must call `mlflow.flush_trace_async_logging()` before `get_trace()` / `search_traces()`** or results will be `None`.

3. **Built-in scorers default to `openai:/gpt-4.1-mini` as judge.** Pass `model="gateway:/<model_name>"` to route judges through the MLflow AI Gateway.

   GEPA `reflection_model` uses litellm directly, which can't read the `gateway:/` scheme. Use `openai:/<model>` with `OPENAI_API_BASE` pointing at the Gateway (OpenAI-compatible) instead — configured in `config.py` as `OPTIMIZER_JUDGE_MODEL`.

4. **MLflow hardcodes 60s timeout for judge calls.** Use `patch_judge_timeout(300)` from `config.py` for slow models.

5. **Code-based scorers (`@scorer`) must return `Feedback`, `bool`, `float`, `str`, or `list[Feedback]`** — NOT dict.

6. **`SpanType.TOOL`** makes spans appear in MLflow's "Tool calls" UI tab.

7. **Experiment auto-restore:** `setup_mlflow()` restores deleted experiments to avoid "experiment already exists but is deleted" errors.
