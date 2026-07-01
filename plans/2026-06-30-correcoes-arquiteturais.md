# [Correções Arquiteturais] Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use compose:subagent (recommended) or compose:execute to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply all architecture fixes from the validation report.

**Architecture:** Split centralized configuration into explicit env/client/patch modules, make MLflow patches opt-in, and keep demos standalone. Centralize OpenAI SDK calls and reusable prediction factories, extract duplicated tool logic for manual and LangChain demos, replace unsafe calculator execution with AST validation, and remove mutable globals that leak state across calls.

**Tech Stack:** Python 3.14 / uv / MLflow GenAI / OpenAI SDK via MLflow AI Gateway / LangChain / LangGraph / GEPA / litellm / hatchling.

## Global Constraints

1. **Python / uv only** — use Python 3.14 and `uv`; do not use `pip`, `poetry`, or virtualenv outside `uv`.
2. **No test suite configured** — verify with `compileall`, targeted imports, and grep checks; do not invent a pytest suite unless explicitly requested.
3. **All demo modules remain standalone** — each module still creates its own MLflow experiment and remains runnable with `uv run`.
4. **MLflow traces must be flushed before lookup** — any demo that calls `mlflow.get_trace()` or `mlflow.search_traces()` must call `mlflow.flush_trace_async_logging()` before lookup.
5. **Built-in scorers must route through the Gateway** — use `JUDGE_MODEL = gateway:/<model_name>` when passing `model=` to MLflow built-in scorers.
6. **GEPA reflection must use OpenAI-compatible Gateway URI** — use `OPTIMIZER_JUDGE_MODEL = openai:/<model_name>` with `OPENAI_API_BASE` pointing at the Gateway.
7. **Judge timeout patch must be explicit** — call `patch_judge_timeout(300)` in modules that invoke MLflow judges or prompt optimization.
8. **Code-based scorers must return supported values** — return `Feedback`, `bool`, `float`, `str`, or `list[Feedback]`; never return `dict`.
9. **Tool spans must use `SpanType.TOOL`** — keep `SpanType.TOOL` so MLflow shows tool calls in the UI.
10. **Do not change the `.env` secret values** — edit only configuration names and non-secret defaults.
11. **Worktree target** — implement from `/home/marcos/Projects/ia-observability/.worktrees/refactor/architecture-fixes-2026-06-30/` and do not commit unless explicitly requested.

---

## Tasks by Phase

### Phase 1: Safe Mechanical Changes

### Task 1: Rename the OpenAI Gateway environment variable

**Covers:** Fix 1

**Files:**
- Modify: `src/ia_observability/config.py`
- Modify: `.env.example`
- Modify: `.env`
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `docs/como-usar.md`

**Interfaces:**
- Consumes: current typo spelling in config/docs/local defaults.
- Produces: `mlflow_openai_url` spelling across config, docs, and local defaults.

- [ ] **Step 1: Replace the typo in all required files**

```python
from pathlib import Path

files = [
    Path("/home/marcos/Projects/ia-observability/src/ia_observability/config.py"),
    Path("/home/marcos/Projects/ia-observability/.env.example"),
    Path("/home/marcos/Projects/ia-observability/.env"),
    Path("/home/marcos/Projects/ia-observability/AGENTS.md"),
    Path("/home/marcos/Projects/ia-observability/README.md"),
    Path("/home/marcos/Projects/ia-observability/docs/como-usar.md"),
]

old_name = "mlflow_open" + "ia_url"
new_name = "mlflow_openai_url"

for path in files:
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(old_name, new_name), encoding="utf-8")
```

- [ ] **Step 2: Verify the typo is gone and the new name is present exactly once per file**

```python
from pathlib import Path

files = [
    Path("/home/marcos/Projects/ia-observability/src/ia_observability/config.py"),
    Path("/home/marcos/Projects/ia-observability/.env.example"),
    Path("/home/marcos/Projects/ia-observability/.env"),
    Path("/home/marcos/Projects/ia-observability/AGENTS.md"),
    Path("/home/marcos/Projects/ia-observability/README.md"),
    Path("/home/marcos/Projects/ia-observability/docs/como-usar.md"),
]

old_name = "mlflow_open" + "ia_url"
new_name = "mlflow_openai_url"

for path in files:
    text = path.read_text(encoding="utf-8")
    assert old_name not in text, f"{path} still contains the typo"
    assert text.count(new_name) == 1, f"{path} must contain exactly one {new_name}"
```

- [ ] **Step 3: Verify no typo remains anywhere in the repository**

```bash
python - <<'PY'
from pathlib import Path
root = Path('/home/marcos/Projects/ia-observability')
needle = 'mlflow_open' + 'ia_url'
for path in root.rglob('*'):
    if not path.is_file() or path.suffix not in {'.py', '.md', '.toml', '.yaml', '.yml', '.env', '.example'}:
        continue
    if needle in path.read_text(encoding='utf-8'):
        raise SystemExit(f'old spelling remains: {path}')
print('OK: no old OpenAI Gateway env var spelling remains')
PY
```

### Task 2: Move GEPA max metric calls to central config/env

**Covers:** Fix 2

**Files:**
- Create: `src/ia_observability/config/env.py`
- Modify: `src/ia_observability/config/__init__.py`
- Modify: `src/ia_observability/parte4_avancado/prompt_optimization.py`

**Interfaces:**
- Consumes: `GEPA_MAX_METRIC_CALLS` environment variable.
- Produces: typed `GEPA_MAX_METRIC_CALLS` exported by `ia_observability.config`.

- [ ] **Step 1: Add the GEPA budget to `src/ia_observability/config/env.py`**

```python
"""Environment variables for the demo applications."""

from __future__ import annotations

import os

GEPA_MAX_METRIC_CALLS: int = int(os.getenv("GEPA_MAX_METRIC_CALLS", "30"))
```

- [ ] **Step 2: Export the constant from `src/ia_observability/config/__init__.py`**

```python
"""Central configuration exports for the ia-observability demos."""

from __future__ import annotations

from ia_observability.config.env import GEPA_MAX_METRIC_CALLS

__all__ = ["GEPA_MAX_METRIC_CALLS"]
```

- [ ] **Step 3: Remove the local env read from `prompt_optimization.py`**

```python
from ia_observability.config import (
    GEPA_MAX_METRIC_CALLS,
    MODEL_NAME,
    OPTIMIZER_JUDGE_MODEL,
    get_client,
    patch_judge_timeout,
    setup_mlflow,
)
```

- [ ] **Step 4: Delete this block from `prompt_optimization.py`**

```python
GEPA_MAX_METRIC_CALLS: int = int(os.getenv("GEPA_MAX_METRIC_CALLS", "30"))
```

- [ ] **Step 5: Verify `GEPA_MAX_METRIC_CALLS` is read only from config/env**

```bash
python - <<'PY'
from pathlib import Path
root = Path('/home/marcos/Projects/ia-observability/src/ia_observability')
text = (root / 'parte4_avancado/prompt_optimization.py').read_text(encoding='utf-8')
assert 'GEPA_MAX_METRIC_CALLS' in text
assert 'os.getenv("GEPA_MAX_METRIC_CALLS"' not in text
print('OK: prompt_optimization.py uses config-level GEPA_MAX_METRIC_CALLS')
PY
```

### Task 3: Standardize CLI script names

**Covers:** Fix 3

**Files:**
- Modify: `pyproject.toml`
- Modify: `Makefile`

**Interfaces:**
- Consumes: old CLI script names.
- Produces: stable CLI names matching `uv run` and `make`.

- [ ] **Step 1: Replace the `[project.scripts]` section in `pyproject.toml` with this exact section**

```toml
[project.scripts]
tracing = "ia_observability.parte1_fundamentos.tracing_basics:main"
token-usage = "ia_observability.parte1_fundamentos.token_usage:main"
sessions = "ia_observability.parte3_producao.sessions:main"
evaluation = "ia_observability.parte2_avaliacao.evaluation:main"
judges = "ia_observability.parte2_avaliacao.judges:main"
version-tracking = "ia_observability.parte3_producao.version_tracking:main"
monitoring = "ia_observability.parte3_producao.production_monitoring:main"
benchmark = "ia_observability.parte4_avancado.experiment_comparison:main"
tool-calls = "ia_observability.parte3_producao.tool_calls:main"
prompts = "ia_observability.parte4_avancado.prompt_management:main"
langchain-agent = "ia_observability.parte3_producao.langchain_agent:main"
datasets = "ia_observability.parte2_avaliacao.datasets_demo:main"
prompt-optimization = "ia_observability.parte4_avancado.prompt_optimization:main"
```

- [ ] **Step 2: Replace the top `.PHONY` line in `Makefile` with this exact line**

```make
.PHONY: help install tracing token-usage sessions evaluation judges version-tracking monitoring benchmark tool-calls prompts langchain-agent datasets prompt-optimization all
```

- [ ] **Step 3: Replace the matching recipe lines in `Makefile` with this exact block**

```make
tracing: ## Executa demo de tracing (auto-tracing, decorators, spans)
	uv run tracing

token-usage: ## Executa demo de token usage e custo por chamada
	uv run token-usage

sessions: ## Executa demo de sessions multi-turn e user tracking
	uv run sessions

evaluation: ## Executa demo de evaluation com scorers built-in
	uv run evaluation

judges: ## Executa demo de LLM judges customizados
	uv run judges

version-tracking: ## Executa demo de version tracking com LoggedModel
	uv run version-tracking

monitoring: ## Executa demo de producao (async, sampling, feedback)
	uv run monitoring

benchmark: ## Executa benchmark comparativo de configuracoes
	uv run benchmark

tool-calls: ## Executa demo de tool calling com observabilidade
	uv run tool-calls

prompts: ## Executa demo de prompt registry e versionamento
	uv run prompts

langchain-agent: ## Executa demo de LangChain agent (tools + sessions)
	uv run langchain-agent

datasets: ## Executa demo de evaluation datasets (subir + buscar)
	uv run datasets

prompt-optimization: ## Executa demo de prompt optimization (GEPA + Metaprompt)
	uv run prompt-optimization

all: tracing token-usage sessions evaluation judges version-tracking monitoring benchmark tool-calls prompts langchain-agent datasets prompt-optimization ## Executa todos os modulos em sequencia
```

- [ ] **Step 4: Verify stale names are gone from `pyproject.toml` and `Makefile`**

```bash
python - <<'PY'
from pathlib import Path
root = Path('/home/marcos/Projects/ia-observability')
for filename in ['pyproject.toml', 'Makefile']:
    text = (root / filename).read_text(encoding='utf-8')
    for old in ['tokens =', 'toolcalls', 'versioning', 'prompt-opt']:
        assert old not in text, f'{old} remains in {filename}'
print('OK: standardized CLI names are present and stale names are gone')
PY
```

### Task 4: Add missing type hints across all demo files

**Covers:** Fix 1

**Files:**
- Modify: `src/ia_observability/config.py` if still present before the config split
- Modify: `src/ia_observability/parte1_fundamentos/token_usage.py`
- Modify: `src/ia_observability/parte1_fundamentos/tracing_basics.py`
- Modify: `src/ia_observability/parte2_avaliacao/judges.py`
- Modify: `src/ia_observability/parte2_avaliacao/evaluation.py`
- Modify: `src/ia_observability/parte3_producao/sessions.py`
- Modify: `src/ia_observability/parte3_producao/version_tracking.py`
- Modify: `src/ia_observability/parte3_producao/production_monitoring.py`
- Modify: `src/ia_observability/parte3_producao/tool_calls.py`
- Modify: `src/ia_observability/parte3_producao/langchain_agent.py`
- Modify: `src/ia_observability/parte4_avancado/experiment_comparison.py`
- Modify: `src/ia_observability/parte4_avancado/prompt_management.py`
- Modify: `src/ia_observability/parte4_avancado/prompt_optimization.py`
- Create: `src/ia_observability/config/__init__.py`
- Create: `src/ia_observability/config/env.py`
- Create: `src/ia_observability/config/client.py`
- Create: `src/ia_observability/config/patches.py`

**Interfaces:**
- Consumes: existing function bodies and call sites.
- Produces: typed public and internal signatures.

- [ ] **Step 1: Replace these signatures with the exact typed versions**

```python
# config/patches.py
def _patched_send(endpoint: Any, headers: dict[str, Any], payload: dict[str, Any]) -> Any: ...
def _patched(*args: Any, **kwargs: Any) -> Any: ...

# token_usage.py
def _set_usage_and_cost(span: Any, usage: dict[str, int]) -> None: ...
def multi_step_pipeline() -> str: ...

# tracing_basics.py
def demo_auto_tracing() -> None: ...
def demo_rag_pipeline(question: str) -> str: ...
def retrieve_context(question: str) -> str: ...
def generate_answer(question: str, context: str) -> str: ...
def demo_context_block() -> None: ...
def main() -> None: ...

# judges.py
def response_length_check(inputs: dict[str, object], outputs: str | None) -> Feedback: ...
def no_hallucination_keywords(inputs: dict[str, object], outputs: str | None) -> Feedback: ...
def contains_actionable_info(inputs: dict[str, object], outputs: str | None) -> Feedback: ...
def predict_fn(question: str) -> str: ...
def main() -> None: ...

# evaluation.py
def get_eval_dataset() -> list[dict[str, object]]: ...
def predict_fn(question: str) -> str: ...
def main() -> None: ...

# sessions.py
def chat_turn(messages: list[dict[str, str]], user_id: str, session_id: str) -> str: ...
def demo_multi_turn_session() -> None: ...
def demo_multiple_users() -> None: ...
def demo_query_by_session() -> None: ...
def main() -> None: ...

# version_tracking.py
def run_version(version_name: str, system_prompt: str, temperature: float) -> None: ...
def main() -> None: ...

# production_monitoring.py
def show_production_config() -> None: ...
def critical_agent_call(agent: Any, query: str, user_id: str, session_id: str) -> str: ...
def high_volume_agent_call(agent: Any, query: str, user_id: str, session_id: str) -> str: ...
def demo_feedback_collection(agent: Any) -> None: ...
def main() -> None: ...

# tool_calls.py
def get_weather(city: str, unit: str = "celsius") -> dict[str, int | str]: ...
def search_docs(query: str, max_results: int = 3) -> list[dict[str, str]]: ...
def calculate(expression: str) -> dict[str, int | float | str]: ...
def check_inventory(product: str) -> dict[str, str]: ...
def agent_with_tools(user_message: str) -> str: ...
def demo_single_tool() -> None: ...
def demo_multi_tool() -> None: ...
def demo_no_tool() -> None: ...
def demo_calculation() -> None: ...
def demo_tool_failure() -> None: ...
def main() -> None: ...

# langchain_agent.py
def build_agent() -> Any: ...
def _extract_text(content: object) -> str: ...
def _consume_stream(agen: Any) -> str: ...
def demo_single_tool(agent: Any) -> None: ...
def demo_multi_tool(agent: Any) -> None: ...
def demo_multi_turn_session(agent: Any) -> None: ...
def demo_multiple_users(agent: Any) -> None: ...
def demo_tool_failure(agent: Any) -> None: ...
def demo_feedback(agent: Any) -> None: ...
def main() -> None: ...

# experiment_comparison.py
def make_predict_fn(system_prompt: str, temperature: float) -> Callable[[str], str]: ...
def main() -> None: ...

# prompt_management.py
def demo_register_prompts() -> None: ...
def ask_with_prompt(topic: str, system_prompt_name: str = "observability-system-v2") -> str: ...
def demo_compare_versions(topic: str) -> None: ...
def main() -> None: ...

# prompt_optimization.py
def _normalize(text: str) -> str: ...
def label_accuracy(outputs: str | None, expectations: dict[str, str]) -> Feedback: ...
def _register_weak_prompt(prompt_uri: str | None = None) -> str: ...
def predict_fn(mensagem: str) -> str: ...
def demo_gepa_optimization(prompt_uri: str | None = None) -> None: ...
def demo_metaprompt_optimization(prompt_uri: str | None = None) -> None: ...
def main() -> None: ...
```

- [ ] **Step 2: Verify signatures compile**

```bash
uv run python -m compileall src/ia_observability
```

### Phase 2: Isolated Behavioral Fixes

### Task 5: Replace unsafe `eval()` in tool calculators with AST whitelist

**Covers:** Fix 5, Fix 8

**Files:**
- Create: `src/ia_observability/parte3_producao/_tools.py`
- Modify: `src/ia_observability/parte3_producao/tool_calls.py`
- Modify: `src/ia_observability/parte3_producao/langchain_agent.py`

**Interfaces:**
- Consumes: `expression: str`.
- Produces: same success/error shape as before.

- [ ] **Step 1: Create `src/ia_observability/parte3_producao/_tools.py` with this exact content**

```python
"""Ferramentas compartilhadas entre tool_calls.py e langchain_agent.py."""

from __future__ import annotations

import ast
import time
from collections.abc import Callable
from typing import Any

from langchain_core.tools import tool as lc_tool


def _get_weather_data() -> dict[str, dict[str, int | str]]:
    return {
        "Sao Paulo": {"temp": 22, "condition": "Parcialmente nublado"},
        "Rio de Janeiro": {"temp": 28, "condition": "Ensolarado"},
        "Curitiba": {"temp": 15, "condition": "Chuva leve"},
    }


def get_weather(city: str, unit: str = "celsius") -> dict[str, int | str]:
    data = _get_weather_data()
    weather = data.get(city, {"temp": 20, "condition": "Desconhecido"})
    return {
        "city": city,
        "temperature": weather["temp"],
        "unit": unit,
        "condition": weather["condition"],
    }


def search_docs(query: str, max_results: int = 3) -> list[dict[str, str]]:
    docs = [
        {"title": "MLflow Tracing Quickstart", "snippet": "Auto-tracing captura chamadas automaticamente..."},
        {"title": "Token Usage Tracking", "snippet": "MLflow rastreia input/output tokens por span..."},
        {"title": "Evaluation com Scorers", "snippet": "Use mlflow.genai.evaluate() com scorers built-in..."},
        {"title": "Tool Calling Observability", "snippet": "SpanType.TOOL permite rastrear execucao de tools..."},
    ]
    filtered = [
        doc
        for doc in docs
        if query.lower() in doc["title"].lower() or query.lower() in doc["snippet"].lower()
    ]
    return filtered[:max_results] if filtered else docs[:max_results]


def calculate_expression(expression: str) -> int | float:
    allowed = set("0123456789+-*/.(). ")
    if not all(char in allowed for char in expression):
        raise ValueError("Expressao invalida")
    tree = ast.parse(expression, mode="eval")

    def visit(node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return visit(node.operand)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = visit(node.operand)
            return -value if isinstance(value, (int, float)) else value
        if isinstance(node, ast.BinOp):
            if type(node.op) not in {ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow}:
                raise ValueError(f"Operacao nao permitida: {type(node.op).__name__}")
            left = visit(node.left)
            right = visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left ** right
        raise ValueError("Expressao matematica invalida")

    return visit(tree.body)


def check_inventory(product: str) -> str:
    time.sleep(2.5)
    return f"ERRO: Timeout ao consultar estoque do produto '{product}' - API indisponivel"


def _emit_tool_writer(tool_name: str, message: str) -> None:
    from langgraph.config import get_stream_writer

    writer = get_stream_writer()
    writer(f"[{tool_name}] {message}")


@lc_tool
def get_weather_lc(city: str, unit: str = "celsius") -> dict[str, int | str]:
    _emit_tool_writer("get_weather", f"Consultando previsao do tempo para {city}...")
    result = get_weather(city, unit)
    _emit_tool_writer("get_weather", f"Previsao para {city}: {result['temperature']}°{result['unit'][0].upper()}, {result['condition']}")
    return result


@lc_tool
def search_docs_lc(query: str, max_results: int = 3) -> list[dict[str, str]]:
    _emit_tool_writer("search_docs", f"Buscando documentos sobre '{query}'...")
    results = search_docs(query, max_results)
    _emit_tool_writer("search_docs", f"Encontrados {len(results)} resultados para '{query}'")
    return results


@lc_tool
def calculate_lc(expression: str) -> dict[str, int | float | str]:
    _emit_tool_writer("calculate", f"Calculando: {expression}...")
    try:
        result = calculate_expression(expression)
        _emit_tool_writer("calculate", f"Resultado: {result}")
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"error": str(exc), "expression": expression}


@lc_tool
def check_inventory_lc(product: str) -> str:
    _emit_tool_writer("check_inventory", f"Consultando estoque do produto '{product}'...")
    result = check_inventory(product)
    _emit_tool_writer("check_inventory", f"Timeout — API de estoque indisponivel para '{product}'")
    return result


TOOL_SPECS: dict[str, dict[str, Any]] = {
    "get_weather": {
        "name": "get_weather",
        "description": "Retorna a previsao do tempo para uma cidade.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "Nome da cidade"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Unidade de temperatura"},
            },
            "required": ["city"],
        },
    },
    "search_docs": {
        "name": "search_docs",
        "description": "Busca na base de conhecimento interna sobre MLflow.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Termo de busca"}, "max_results": {"type": "integer", "description": "Maximo de resultados"}},
            "required": ["query"],
        },
    },
    "calculate": {
        "name": "calculate",
        "description": "Executa um calculo matematico.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "Expressao matematica"}}, "required": ["expression"]},
    },
    "check_inventory": {
        "name": "check_inventory",
        "description": "Consulta o estoque disponivel de um produto.",
        "parameters": {"type": "object", "properties": {"product": {"type": "string", "description": "Nome do produto"}}, "required": ["product"]},
    },
}


def make_openai_tools() -> list[dict[str, Any]]:
    return [{"type": "function", "function": spec} for spec in TOOL_SPECS.values()]


TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "get_weather": get_weather,
    "search_docs": search_docs,
    "calculate": calculate_expression,
    "check_inventory": check_inventory,
}

ALL_TOOLS: list[Any] = [get_weather_lc, search_docs_lc, calculate_lc, check_inventory_lc]
```

- [ ] **Step 2: Replace both calculator implementations**

```python
# tool_calls.py and langchain_agent.py
from ia_observability.parte3_producao._tools import calculate_expression


def calculate(expression: str) -> dict[str, int | float | str]:
    """Executa calculo simples usando uma whitelist AST segura."""
    try:
        result = calculate_expression(expression)
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"error": str(exc), "expression": expression}
```

- [ ] **Step 3: Verify no `eval(` remains in tool calculators**

```bash
python - <<'PY'
from pathlib import Path
root = Path('/home/marcos/Projects/ia-observability/src/ia_observability/parte3_producao')
for filename in ['tool_calls.py', 'langchain_agent.py']:
    text = (root / filename).read_text(encoding='utf-8')
    assert 'eval(' not in text, f'eval( remains in {filename}'
print('OK: unsafe eval removed from tool calculators')
PY
```

### Task 6: Fix the misleading `finally: pass` in `langchain_agent.py`

**Covers:** Fix 6

**Files:**
- Modify: `src/ia_observability/parte3_producao/langchain_agent.py`

**Interfaces:**
- Consumes: streaming chunks from `agent.astream(...)`.
- Produces: streaming events plus guaranteed session capture, trace tags, outputs, and span status even on stream error.

- [ ] **Step 1: Replace `agent_invoke_stream` with this exact implementation**

```python
async def agent_invoke_stream(
    agent: Any,
    query: str,
    user_id: str,
    session_id: str,
    provider: str = "openai",
    model_name: str | None = None,
) -> Any:
    """Invoca o agente com streaming, span manual MLflow e eventos JSON."""
    if model_name is None:
        model_name = MODEL_NAME

    chat_history = _sessions.get(session_id, [])
    input_messages: list[BaseMessage] = [
        SystemMessage(content=_SYSTEM_PROMPT),
        *chat_history,
        HumanMessage(content=query),
    ]
    raw_chunks: list[Any] = []
    accumulated_response: str = ""
    pending_logs: list[str] = []
    tool_logs_by_id: dict[str, list[str]] = {}
    trace_name = query.replace("\n", " ")[:50] or "chat-turn"

    with mlflow.start_span(name=trace_name, span_type=SpanType.AGENT) as trace_span:
        trace_id = trace_span.trace_id
        trace_span.set_inputs({"query": query})
        trace_span.set_attribute("agent.error", False)
        mlflow.update_current_trace(session_id=session_id, user=user_id)

        try:
            async for mode, data in agent.astream(
                {"messages": input_messages},
                config={
                    "recursion_limit": 40,
                    "metadata": {
                        "provider": provider,
                        "model_name": model_name,
                        "session_id": session_id,
                    },
                },
                stream_mode=["messages", "custom"],
            ):
                if mode == "custom":
                    pending_logs.append(str(data))
                    yield make_event("tool", str(data))
                    continue
                chunk, metadata = data
                if not metadata.get("langgraph_node"):
                    continue
                if isinstance(chunk, ToolMessage):
                    if pending_logs:
                        tool_logs_by_id[chunk.tool_call_id] = pending_logs.copy()
                        pending_logs.clear()
                    raw_chunks.append(chunk)
                elif isinstance(chunk, AIMessageChunk) and chunk.content and not chunk.tool_call_chunks:
                    text = _extract_text(chunk.content)
                    accumulated_response += text
                    raw_chunks.append(chunk)
                    yield make_event("text_chunk", text)
                else:
                    raw_chunks.append(chunk)
        except Exception as exc:
            trace_span.set_attribute("agent.error", True)
            trace_span.set_attribute("agent.error_type", type(exc).__name__)
            trace_span.set_attribute("agent.error_message", str(exc))
            trace_span.set_status("ERROR")
            print(f"  Erro no stream do agente: {exc}")
        finally:
            new_messages = _collect_messages(raw_chunks, accumulated_response, tool_logs_by_id, trace_id=trace_id)
            _sessions[session_id] = chat_history + new_messages
            mlflow.update_current_trace(tags={
                "provider": provider,
                "model_name": model_name,
                "new_messages": str(len(new_messages)),
                "session_id": session_id,
                "agent.error": str(trace_span.get_attribute("agent.error", False)),
                "agent.error_type": str(trace_span.get_attribute("agent.error_type", "")),
            })
            trace_span.set_outputs({
                "response": accumulated_response[:500] if accumulated_response else None,
                "trace_id": trace_id,
                "new_messages": str(len(new_messages)),
            })

    yield make_event("done", trace_id or "")
```

- [ ] **Step 2: Verify span finalization is inside the span context**

```bash
python - <<'PY'
from pathlib import Path
path = Path('/home/marcos/Projects/ia-observability/src/ia_observability/parte3_producao/langchain_agent.py')
text = path.read_text(encoding='utf-8')
assert 'finally:\n            pass' not in text
assert 'trace_span.set_outputs(' in text
assert '_sessions[session_id] = chat_history + new_messages' in text
print('OK: langchain_agent.py finalizes spans and sessions inside the span context')
PY
```

### Phase 3: Structural Refactors

### Task 7: Split config.py into subpackage and make patches opt-in

**Covers:** Fix 9, Fix 10, Fix 11, continued Fix 2

**Files:**
- Modify: `src/ia_observability/config.py`
- Create: `src/ia_observability/config/__init__.py`
- Create: `src/ia_observability/config/env.py`
- Create: `src/ia_observability/config/client.py`
- Create: `src/ia_observability/config/patches.py`
- Modify: all demo modules to call `apply_mlflow_patches()`

**Interfaces:**
- Consumes: current module-level config and patch calls.
- Produces: `ia_observability.config` package with explicit modules and opt-in patch application.

- [ ] **Step 1: Create `src/ia_observability/config/env.py`**

```python
"""Environment variables for MLflow tracking and the AI Gateway."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("DO_NOT_TRACK", "true")

_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path)

MLFLOW_TRACKING_URI: str = os.getenv("mlflow_url", "http://localhost:5000")
MLFLOW_GATEWAY_URL: str = os.getenv("mlflow_openai_url", "http://localhost:5000/gateway/mlflow/v1")
MODEL_NAME: str = os.getenv("mlflow_model", "qwen3.5-9b")

_JUDGE_MODEL_NAME: str = os.getenv("mlflow_judge_model", MODEL_NAME)
JUDGE_MODEL: str = f"gateway:/{_JUDGE_MODEL_NAME}"

os.environ.setdefault("OPENAI_API_BASE", MLFLOW_GATEWAY_URL)
os.environ.setdefault("OPENAI_API_KEY", "not-needed")
OPTIMIZER_JUDGE_MODEL: str = f"openai:/{_JUDGE_MODEL_NAME}"

GEPA_MAX_METRIC_CALLS: int = int(os.getenv("GEPA_MAX_METRIC_CALLS", "30"))
```

- [ ] **Step 2: Create `src/ia_observability/config/client.py`**

```python
"""OpenAI SDK client helpers for the MLflow AI Gateway."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from openai import OpenAI, OpenAIApiError, OpenAIError

from ia_observability.config.env import MLFLOW_GATEWAY_URL, MODEL_NAME


class OpenAIApiError(RuntimeError):
    """Erro padrao para falhas de rede ou API no AI Gateway."""


def get_client() -> OpenAI:
    """Retorna um cliente OpenAI apontando para o MLflow AI Gateway."""
    return OpenAI(base_url=MLFLOW_GATEWAY_URL, api_key="not-needed")


def chat_completion(model: str, messages: list[dict[str, str]], **kwargs: Any) -> Any:
    """Chama a API de completions com tratamento padronizado de erros de rede."""
    client = get_client()
    try:
        return client.chat.completions.create(model=model, messages=messages, **kwargs)
    except (OpenAIError, TimeoutError, ConnectionError) as exc:
        raise OpenAIApiError(f"Chamada OpenAI falhou para model={model}: {exc}") from exc


def create_predict_fn(system_prompt: str, temperature: float | None = None) -> Callable[[str], str]:
    """Create a predict_fn with shared network error handling."""
    def predict_fn(question: str) -> str:
        try:
            response = chat_completion(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=temperature if temperature is not None else 0.0,
            )
        except OpenAIApiError as exc:
            raise RuntimeError(f"Falha ao gerar predicao via AI Gateway para pergunta: {question}") from exc
        return response.choices[0].message.content

    return predict_fn
```

- [ ] **Step 3: Create `src/ia_observability/config/patches.py`**

```python
"""Opt-in MLflow patch helpers."""

from __future__ import annotations

from typing import Any

_PATCHES_APPLIED: set[str] = set()


def _disable_async_prompt_linking() -> None:
    try:
        from mlflow.tracking.client import MlflowClient

        MlflowClient._link_prompt_to_experiment = lambda self, *a, **k: None
    except Exception:
        pass


def setup_mlflow(experiment_name: str) -> None:
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment and experiment.lifecycle_stage == "deleted":
        client.restore_experiment(experiment.experiment_id)
    mlflow.set_experiment(experiment_name)


def patch_judge_timeout(timeout: int = 300) -> None:
    if "judge_timeout" in _PATCHES_APPLIED:
        return
    import mlflow.metrics.genai.model_utils as mu

    def _patched_send(endpoint, headers, payload):
        import requests
        from mlflow.exceptions import MlflowException

        try:
            response = requests.post(url=endpoint, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            body = getattr(e.response, "text", "")
            raise MlflowException(f"Failed to call LLM endpoint at {endpoint}.\n- Error: {e}\n- Response body: {body}") from e
        except requests.exceptions.Timeout:
            raise MlflowException(f"Timeout calling LLM endpoint at {endpoint} (timeout={timeout}s).") from None
        return response.json()

    mu._send_request = _patched_send
    _PATCHES_APPLIED.add("judge_timeout")


def patch_judge_json_parsing() -> None:
    import json
    import mlflow.genai.judges.adapters.gateway_adapter as ga

    _original_strip = ga._strip_markdown_code_blocks

    def _patched_strip(response: str) -> str:
        cleaned = _original_strip(response)
        for text in (cleaned, response):
            candidates = [i for i in (text.find("{"), text.find("[")) if i != -1]
            if not candidates:
                continue
            start = min(candidates)
            try:
                obj, _ = json.JSONDecoder(strict=False).raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, list) and obj:
                obj = obj[0]
            if isinstance(obj, dict):
                return json.dumps(obj)
        return cleaned

    ga._strip_markdown_code_blocks = _patched_strip
    _PATCHES_APPLIED.add("judge_json_parsing")


def patch_litellm_max_tokens(default_max_tokens: int = 4096) -> None:
    try:
        import litellm
    except ImportError:
        return
    if "litellm_max_tokens" in _PATCHES_APPLIED:
        return

    _original = litellm.completion

    def _patched(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("max_tokens", default_max_tokens)
        return _original(*args, **kwargs)

    litellm.completion = _patched
    _PATCHES_APPLIED.add("litellm_max_tokens")


def apply_mlflow_patches() -> None:
    if "async_prompt_linking" not in _PATCHES_APPLIED:
        _disable_async_prompt_linking()
        _PATCHES_APPLIED.add("async_prompt_linking")
    patch_judge_json_parsing()
    patch_litellm_max_tokens()
```

- [ ] **Step 4: Create `src/ia_observability/config/__init__.py`**

```python
"""Central configuration exports for the ia-observability demos."""

from __future__ import annotations

from ia_observability.config.client import OpenAIApiError, chat_completion, create_predict_fn, get_client
from ia_observability.config.env import GEPA_MAX_METRIC_CALLS, JUDGE_MODEL, MLFLOW_GATEWAY_URL, MLFLOW_TRACKING_URI, MODEL_NAME, OPTIMIZER_JUDGE_MODEL
from ia_observability.config.patches import apply_mlflow_patches, patch_judge_json_parsing, patch_judge_timeout, patch_litellm_max_tokens, setup_mlflow

__all__ = [
    "GEPA_MAX_METRIC_CALLS",
    "JUDGE_MODEL",
    "MLFLOW_GATEWAY_URL",
    "MLFLOW_TRACKING_URI",
    "MODEL_NAME",
    "OPTIMIZER_JUDGE_MODEL",
    "OpenAIApiError",
    "apply_mlflow_patches",
    "chat_completion",
    "create_predict_fn",
    "get_client",
    "patch_judge_json_parsing",
    "patch_judge_timeout",
    "patch_litellm_max_tokens",
    "setup_mlflow",
]
```

- [ ] **Step 5: Remove the old root `config.py`**

```bash
rm "/home/marcos/Projects/ia-observability/src/ia_observability/config.py"
```

- [ ] **Step 6: Update every demo to call `apply_mlflow_patches()` before `setup_mlflow()`**

```python
from pathlib import Path

root = Path('/home/marcos/Projects/ia-observability/src/ia_observability')
demo_files = [
    root / 'parte1_fundamentos/tracing_basics.py',
    root / 'parte1_fundamentos/token_usage.py',
    root / 'parte3_producao/sessions.py',
    root / 'parte2_avaliacao/evaluation.py',
    root / 'parte2_avaliacao/judges.py',
    root / 'parte3_producao/version_tracking.py',
    root / 'parte3_producao/production_monitoring.py',
    root / 'parte3_producao/tool_calls.py',
    root / 'parte3_producao/langchain_agent.py',
    root / 'parte4_avancado/experiment_comparison.py',
    root / 'parte4_avancado/prompt_management.py',
    root / 'parte2_avaliacao/datasets_demo.py',
    root / 'parte4_avancado/prompt_optimization.py',
]

for path in demo_files:
    text = path.read_text(encoding='utf-8')
    text = text.replace('from ia_observability.config import MODEL_NAME, get_client, setup_mlflow', 'from ia_observability.config import MODEL_NAME, apply_mlflow_patches, get_client, setup_mlflow')
    text = text.replace('from ia_observability.config import (\n    MODEL_NAME,\n    get_client,\n    setup_mlflow,\n)', 'from ia_observability.config import (\n    MODEL_NAME,\n    apply_mlflow_patches,\n    get_client,\n    setup_mlflow,\n)')
    text = text.replace('from ia_observability.config import JUDGE_MODEL, MODEL_NAME, get_client, patch_judge_timeout, setup_mlflow', 'from ia_observability.config import JUDGE_MODEL, MODEL_NAME, apply_mlflow_patches, get_client, patch_judge_timeout, setup_mlflow')
    text = text.replace('from ia_observability.config import (\n    JUDGE_MODEL,\n    MODEL_NAME,\n    get_client,\n    patch_judge_timeout,\n    setup_mlflow,\n)', 'from ia_observability.config import (\n    JUDGE_MODEL,\n    MODEL_NAME,\n    apply_mlflow_patches,\n    get_client,\n    patch_judge_timeout,\n    setup_mlflow,\n)')
    text = text.replace('from ia_observability.config import (\n    GEPA_MAX_METRIC_CALLS,\n    MODEL_NAME,\n    OPTIMIZER_JUDGE_MODEL,\n    get_client,\n    patch_judge_timeout,\n    setup_mlflow,\n)', 'from ia_observability.config import (\n    GEPA_MAX_METRIC_CALLS,\n    MODEL_NAME,\n    OPTIMIZER_JUDGE_MODEL,\n    apply_mlflow_patches,\n    get_client,\n    patch_judge_timeout,\n    setup_mlflow,\n)')
    text = text.replace('from ia_observability.config import MLFLOW_GATEWAY_URL, MODEL_NAME, setup_mlflow', 'from ia_observability.config import MLFLOW_GATEWAY_URL, MODEL_NAME, apply_mlflow_patches, setup_mlflow')
    text = text.replace('from ia_observability.config import (\n    MLFLOW_GATEWAY_URL,\n    MODEL_NAME,\n    setup_mlflow,\n)', 'from ia_observability.config import (\n    MLFLOW_GATEWAY_URL,\n    MODEL_NAME,\n    apply_mlflow_patches,\n    setup_mlflow,\n)')
    text = text.replace('from ia_observability.config import (\n    apply_mlflow_patches,\n    setup_mlflow,\n)', 'from ia_observability.config import (\n    apply_mlflow_patches,\n    setup_mlflow,\n)')
    if 'from ia_observability.config import apply_mlflow_patches' not in text and 'from ia_observability.config import apply_mlflow_patches' not in text:
        text = text.replace('from ia_observability.config import setup_mlflow', 'from ia_observability.config import apply_mlflow_patches, setup_mlflow')
        text = text.replace('from ia_observability.config import (\n    setup_mlflow,\n)', 'from ia_observability.config import (\n    apply_mlflow_patches,\n    setup_mlflow,\n)')
    text = text.replace('def main() -> None:\n    """', 'def main() -> None:\n    apply_mlflow_patches()\n    """')
    if 'apply_mlflow_patches()\n    setup_mlflow' not in text:
        text = text.replace('def main() -> None:\n    """', 'def main() -> None:\n    apply_mlflow_patches()\n    """', 1)
    text = text.replace('def main() -> None:\n    apply_mlflow_patches()\n    """\n    setup_mlflow', 'def main() -> None:\n    """\n    apply_mlflow_patches()\n    setup_mlflow')
    path.write_text(text, encoding='utf-8')
```

- [ ] **Step 7: Verify every demo opts in to patches**

```bash
python - <<'PY'
from pathlib import Path
root = Path('/home/marcos/Projects/ia-observability/src/ia_observability')
demo_files = [
    root / 'parte1_fundamentos/tracing_basics.py', root / 'parte1_fundamentos/token_usage.py', root / 'parte3_producao/sessions.py',
    root / 'parte2_avaliacao/evaluation.py', root / 'parte2_avaliacao/judges.py', root / 'parte3_producao/version_tracking.py', root / 'parte3_producao/production_monitoring.py',
    root / 'parte3_producao/tool_calls.py', root / 'parte3_producao/langchain_agent.py', root / 'parte4_avancado/experiment_comparison.py', root / 'parte4_avancado/prompt_management.py', root / 'parte2_avaliacao/datasets_demo.py', root / 'parte4_avancado/prompt_optimization.py',
]
for path in demo_files:
    lines = path.read_text(encoding='utf-8').splitlines()
    apply_idx = next(i for i, line in enumerate(lines) if 'apply_mlflow_patches()' in line)
    setup_idx = next(i for i, line in enumerate(lines) if 'setup_mlflow(' in line)
    assert apply_idx < setup_idx, f'{path} must opt in before setup_mlflow'
print('OK: all demo modules opt in to MLflow patches')
PY
```

### Phase 4: Architectural Changes

### Task 8: Replace all direct chat completion calls with the shared wrapper

**Covers:** Fix 12

**Files:**
- Modify: `src/ia_observability/parte1_fundamentos/tracing_basics.py`
- Modify: `src/ia_observability/parte1_fundamentos/token_usage.py`
- Modify: `src/ia_observability/parte2_avaliacao/evaluation.py`
- Modify: `src/ia_observability/parte2_avaliacao/judges.py`
- Modify: `src/ia_observability/parte3_producao/sessions.py`
- Modify: `src/ia_observability/parte3_producao/version_tracking.py`
- Modify: `src/ia_observability/parte3_producao/tool_calls.py`
- Modify: `src/ia_observability/parte4_avancado/experiment_comparison.py`
- Modify: `src/ia_observability/parte4_avancado/prompt_management.py`
- Modify: `src/ia_observability/parte4_avancado/prompt_optimization.py`

**Interfaces:**
- Consumes: `client.chat.completions.create(...)` call sites.
- Produces: shared `chat_completion(...)` call sites with centralized error handling.

- [ ] **Step 1: Add `chat_completion` to every file that currently imports `get_client`**

```python
from pathlib import Path

root = Path('/home/marcos/Projects/ia-observability/src/ia_observability')
files = [
    root / 'parte1_fundamentos/tracing_basics.py', root / 'parte1_fundamentos/token_usage.py', root / 'parte2_avaliacao/evaluation.py',
    root / 'parte2_avaliacao/judges.py', root / 'parte3_producao/sessions.py', root / 'parte3_producao/version_tracking.py', root / 'parte3_producao/tool_calls.py',
    root / 'parte4_avancado/experiment_comparison.py', root / 'parte4_avancado/prompt_management.py', root / 'parte4_avancado/prompt_optimization.py',
]
for path in files:
    text = path.read_text(encoding='utf-8')
    if 'chat_completion' not in text:
        text = text.replace('get_client,', 'chat_completion,\n    get_client,')
    if 'get_client' in text and 'from ia_observability.config import get_client' in text:
        text = text.replace('from ia_observability.config import get_client', 'from ia_observability.config import chat_completion, get_client')
    path.write_text(text, encoding='utf-8')
```

- [ ] **Step 2: Replace direct SDK call blocks with `chat_completion`**

```python
from pathlib import Path

root = Path('/home/marcos/Projects/ia_observability/src/ia_observability')
for path in root.rglob('*.py'):
    if path.suffix != '.py':
        continue
    text = path.read_text(encoding='utf-8')
    text = text.replace('    client = get_client()\n    response = client.chat.completions.create(', '    response = chat_completion(')
    text = text.replace('    client = get_client()\n    completion = client.chat.completions.create(', '    completion = chat_completion(')
    text = text.replace('    client = get_client()\n    final_response = client.chat.completions.create(', '    final_response = chat_completion(')
    text = text.replace('response = client.chat.completions.create(', 'response = chat_completion(')
    text = text.replace('completion = client.chat.completions.create(', 'completion = chat_completion(')
    text = text.replace('final_response = client.chat.completions.create(', 'final_response = chat_completion(')
    path.write_text(text, encoding='utf-8')
```

- [ ] **Step 3: Verify 15 shared calls and zero direct calls**

```bash
python - <<'PY'
from pathlib import Path
root = Path('/home/marcos/Projects/ia-observability/src/ia_observability')
chat_completion_calls = 0
for path in root.rglob('*.py'):
    text = path.read_text(encoding='utf-8')
    chat_completion_calls += text.count('chat_completion(')
    assert 'client.chat.completions.create(' not in text, f'direct call remains in {path}'
assert chat_completion_calls == 15, f'Expected 15 chat_completion calls, found {chat_completion_calls}'
print('OK: 15 shared chat_completion calls; no direct client.chat.completions.create() calls')
PY
```

### Task 9: Fix mutable global session and prompt URI state

**Covers:** Fix 13

**Files:**
- Modify: `src/ia_observability/parte3_producao/langchain_agent.py`
- Modify: `src/ia_observability/parte4_avancado/prompt_optimization.py`

**Interfaces:**
- Consumes: `session_id` and `prompt_uri`.
- Produces: isolated session state and explicit prompt URI passing.

- [ ] **Step 1: Add `clear_session()` for `_sessions` in `langchain_agent.py`**

```python
_sessions: dict[str, list[BaseMessage]] = {}


def clear_session(session_id: str | None = None) -> None:
    if session_id is None:
        _sessions.clear()
        return
    _sessions.pop(session_id, None)
```

- [ ] **Step 2: Replace global `_PROMPT_URI` in `prompt_optimization.py`**

```python
def _register_weak_prompt(prompt_uri: str | None = None) -> str:
    if prompt_uri is None:
        pv = mlflow.genai.register_prompt(name=PROMPT_NAME, template=WEAK_SYSTEM_PROMPT)
        prompt_uri = f"prompts:/{PROMPT_NAME}/{pv.version}"
    print(f"  Prompt registrado: {prompt_uri}")
    return prompt_uri


def _predict_with_prompt_uri(prompt_uri: str, mensagem: str) -> str:
    system_prompt = mlflow.genai.load_prompt(prompt_uri).template
    completion = get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": mensagem},
        ],
    )
    return completion.choices[0].message.content


def predict_fn(mensagem: str) -> str:
    return _predict_with_prompt_uri(_PROMPT_URI, mensagem)


def demo_gepa_optimization(prompt_uri: str | None = None) -> None:
    prompt_uri = _register_weak_prompt(prompt_uri)

    def predict_fn(mensagem: str) -> str:
        return _predict_with_prompt_uri(prompt_uri, mensagem)

    result = mlflow.genai.optimize_prompts(
        predict_fn=predict_fn,
        train_data=TRAIN_DATA,
        prompt_uris=[prompt_uri],
        optimizer=GepaPromptOptimizer(
            reflection_model=OPTIMIZER_JUDGE_MODEL,
            max_metric_calls=GEPA_MAX_METRIC_CALLS,
            display_progress_bar=True,
        ),
        scorers=[label_accuracy],
    )

    optimized = result.optimized_prompts[0]
    print(f"\n  System prompt DEPOIS ({optimized.uri}):")
    print("  " + "-" * 56)
    print(optimized.template)
    print("  " + "-" * 56)
    print(f"  Acuracia inicial -> final: {result.initial_eval_score} -> {result.final_eval_score}")


def demo_metaprompt_optimization(prompt_uri: str | None = None) -> None:
    prompt_uri = _register_weak_prompt(prompt_uri)
    print(f"  System prompt ANTES: {WEAK_SYSTEM_PROMPT!r}")
    print(f"  Reflection model: {OPTIMIZER_JUDGE_MODEL}")
    print("  Reestruturando o prompt (zero-shot, 1 rodada)...")

    result = mlflow.genai.optimize_prompts(
        predict_fn=predict_fn,
        train_data=[],
        prompt_uris=[prompt_uri],
        optimizer=MetaPromptOptimizer(
            reflection_model=OPTIMIZER_JUDGE_MODEL,
            lm_kwargs={"temperature": 0.3, "max_tokens": 8192},
            guidelines=(
                "O system prompt e de um classificador de mensagens de suporte. "
                "Ele deve instruir o modelo a classificar a mensagem do usuario em "
                f"exatamente UMA destas categorias: {', '.join(LABELS)}. "
                "A resposta deve conter APENAS o rotulo em maiusculas, sem explicacao, "
                "pontuacao ou texto extra. Escreva um prompt CONCISO (poucas linhas), "
                "sem exemplos longos."
            ),
        ),
        scorers=[],
    )

    optimized = result.optimized_prompts[0]
    print(f"\n  System prompt DEPOIS ({optimized.uri}):")
    print("  " + "-" * 56)
    print(optimized.template)
    print("  " + "-" * 56)
```

- [ ] **Step 3: Delete the old global prompt URI block from `prompt_optimization.py`**

```python
_PROMPT_URI: str = ""


def _register_weak_prompt() -> str:
    global _PROMPT_URI
    pv = mlflow.genai.register_prompt(name=PROMPT_NAME, template=WEAK_SYSTEM_PROMPT)
    _PROMPT_URI = f"prompts:/{PROMPT_NAME}/{pv.version}"
    print(f"  Prompt registrado: {_PROMPT_URI}")
    return _PROMPT_URI
```

- [ ] **Step 4: Verify mutable globals are removed**

```bash
python - <<'PY'
from pathlib import Path
root = Path('/home/marcos/Projects/ia-observability/src/ia_observability')
lang = (root / 'parte3_producao/langchain_agent.py').read_text(encoding='utf-8')
prompt = (root / 'parte4_avancado/prompt_optimization.py').read_text(encoding='utf-8')
assert 'def clear_session(' in lang
assert '_PROMPT_URI: str = ""' not in prompt
assert 'global _PROMPT_URI' not in prompt
assert '_predict_with_prompt_uri' in prompt
print('OK: session clearing and prompt URI parameters are present; global prompt URI is removed')
PY
```

---

## Final Verification

- [ ] **Step 1: Run compile checks**

```bash
uv run python -m compileall src/ia_observability
```

- [ ] **Step 2: Run import smoke checks for all demo modules**

```bash
uv run python - <<'PY'
import importlib
modules = [
    'ia_observability.parte1_fundamentos.tracing_basics',
    'ia_observability.parte1_fundamentos.token_usage',
    'ia_observability.parte3_producao.sessions',
    'ia_observability.parte2_avaliacao.evaluation',
    'ia_observability.parte2_avaliacao.judges',
    'ia_observability.parte3_producao.version_tracking',
    'ia_observability.parte3_producao.production_monitoring',
    'ia_observability.parte3_producao.tool_calls',
    'ia_observability.parte3_producao.langchain_agent',
    'ia_observability.parte4_avancado.experiment_comparison',
    'ia_observability.parte4_avancado.prompt_management',
    'ia_observability.parte2_avaliacao.datasets_demo',
    'ia_observability.parte4_avancado.prompt_optimization',
]
for module in modules:
    importlib.import_module(module)
print('OK: all demo modules import successfully')
PY
```

- [ ] **Step 3: Run final grep checks**

```bash
python - <<'PY'
from pathlib import Path
root = Path('/home/marcos/Projects/ia-observability')
needle = 'mlflow_open' + 'ia_url'
for path in root.rglob('*'):
    if not path.is_file() or path.suffix not in {'.py', '.md', '.toml', '.yaml', '.yml', '.env', '.example'}:
        continue
    assert needle not in path.read_text(encoding='utf-8'), f'old env spelling remains: {path}'
for path in (root / 'src/ia_observability').rglob('*.py'):
    text = path.read_text(encoding='utf-8')
    assert 'eval(' not in text, f'eval remains in {path}'
    assert 'client.chat.completions.create(' not in text, f'direct chat call remains in {path}'
print('OK: final grep checks passed')
PY
```
