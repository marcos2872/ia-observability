# ia-observability

Projeto **didatico** de observabilidade e benchmark de LLM usando [MLflow GenAI](https://mlflow.org/docs/latest/genai/).

Cada modulo e um exemplo **autocontido** de uma funcionalidade do MLflow, do basico (tracing) ao avancado (otimizacao de prompts). Os modelos sao servidos pelo **MLflow AI Gateway** (endpoint compativel com OpenAI).

## Funcionalidades Demonstradas

| # | Modulo | Funcionalidade | Comando |
|---|--------|---------------|---------|
| 01 | `tracing_basics` | Auto-tracing, `@mlflow.trace`, spans aninhados (RAG) | `uv run tracing` |
| 02 | `token_usage` | Token usage por chamada, custo, attribution manual | `uv run tokens` |
| 03 | `sessions` | Sessions multi-turn, user tracking, queries (manual) | `uv run sessions` |
| 04 | `evaluation` | Benchmark com datasets + scorers built-in | `uv run evaluation` |
| 05 | `judges` | LLM judges customizados + code-based scorers | `uv run judges` |
| 06 | `version_tracking` | Versionamento com LoggedModel | `uv run versioning` |
| 07 | `production_monitoring` | Async logging, sampling, feedback | `uv run monitoring` |
| 08 | `experiment_comparison` | Comparacao de configs lado a lado | `uv run benchmark` |
| 09 | `tool_calls` | Tool calling com observabilidade (AGENT/TOOL spans, manual) | `uv run toolcalls` |
| 10 | `prompt_management` | Prompt Registry: registrar, versionar, linkar a traces | `uv run prompts` |
| 11 | `langchain_agent` | Tool calling + sessions via LangChain (automatico) | `uv run langchain-agent` |
| 12 | `datasets_demo` | Evaluation datasets: subir + buscar (requer backend SQL) | `uv run datasets` |
| 13 | `prompt_optimization` | Otimizacao de prompts: GEPA + Metaprompting | `uv run prompt-opt` |

> Os modulos 03/09 (manual) tem equivalente automatico no 11 (`langchain_agent`). Compare-os para entender as duas abordagens.

## Requisitos

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (package manager)
- MLflow server rodando — com **AI Gateway** configurado (endpoint OpenAI-compatible)
- Backend **SQL** (PostgreSQL/MySQL/SQLite) no tracking server para o modulo `datasets`

## Setup

```bash
# Instalar dependencias
uv sync

# Configurar .env (copie de .env.example)
# mlflow_url=http://<seu-server>:5000/
# mlflow_openia_url=http://<seu-server>:5000/gateway/mlflow/v1
# mlflow_model=qwen3.5-9b
# mlflow_judge_model=qwen3.5-9b          # modelo para judges/scorers e reflexao do GEPA
```

## Executar

Utilize o `Makefile` na raiz do projeto:

```bash
make help        # Lista todos os comandos disponiveis
make install     # Instala dependencias com uv
make tracing     # Demo de tracing (auto-tracing, decorators, spans)
make tokens      # Demo de token usage e custo por chamada
make sessions    # Demo de sessions multi-turn e user tracking
make evaluation  # Demo de evaluation com scorers built-in
make judges      # Demo de LLM judges customizados
make versioning  # Demo de version tracking com LoggedModel
make monitoring  # Demo de producao (async, sampling, feedback)
make benchmark   # Benchmark comparativo de configuracoes
make toolcalls   # Demo de tool calling com observabilidade
make prompts     # Demo de prompt registry e versionamento
make langchain-agent  # Demo de LangChain agent (tools + sessions)
make datasets    # Demo de evaluation datasets (subir + buscar)
make prompt-opt  # Demo de prompt optimization (GEPA + Metaprompt)
make all         # Executa todos os modulos em sequencia
```

Ou diretamente com `uv run`:

```bash
uv run tracing
uv run python -m ia_observability.tracing_basics
```

## Abrir MLflow UI

Os traces e resultados de avaliacao ficam visiveis no MLflow UI:

```
http://<mlflow_url>
```

Cada modulo cria um experiment separado e numerado (ex: `01-tracing-basics`, `04-evaluation`).

## Arquitetura

```
src/ia_observability/
├── config.py                 # Configuracao centralizada (.env + MLflow + Gateway client)
├── tracing_basics.py         # 01 - Auto-tracing, decorators, spans aninhados
├── token_usage.py            # 02 - Token counting, custo, span-level usage
├── sessions.py               # 03 - Multi-turn conversations, user/session tracking
├── evaluation.py             # 04 - mlflow.genai.evaluate() com scorers built-in
├── judges.py                 # 05 - Custom Guidelines judges + code-based scorers
├── version_tracking.py       # 06 - LoggedModel, versionamento por config
├── production_monitoring.py  # 07 - Async, sampling, feedback collection
├── experiment_comparison.py  # 08 - Benchmark comparativo de configuracoes
├── tool_calls.py             # 09 - Tool calling com spans AGENT/TOOL/CHAT_MODEL
├── prompt_management.py      # 10 - Prompt Registry: registrar, versionar, linkar
├── langchain_agent.py        # 11 - Tool calling + sessions via LangChain (automatico)
├── datasets_demo.py          # 12 - Evaluation datasets: subir + buscar (backend SQL)
└── prompt_optimization.py    # 13 - Prompt optimization: GEPA + Metaprompting
```

## Como os modelos sao acessados

- **Inferencia e LangChain**: o SDK OpenAI (`get_client()`) e o `ChatOpenAI` apontam para o MLflow AI Gateway (`mlflow_openia_url`), que e OpenAI-compatible.
- **Judges/scorers**: usam o provider nativo `gateway:/<modelo>` do MLflow.
- **Reflexao do GEPA**: usa `openai:/<modelo>` via litellm, com `OPENAI_API_BASE` apontando para o mesmo Gateway (configurado em `config.py`).

## Stack

- **MLflow >= 3.10** (com extra `[genai]`) - plataforma de observabilidade
- **OpenAI SDK** - cliente apontando para o MLflow AI Gateway (OpenAI-compatible)
- **LangChain / LangGraph** - agente automatico (modulo 11)
- **gepa / litellm** - otimizacao de prompts (modulo 13)
- **python-dotenv** - gerenciamento de variaveis de ambiente

## Referencia

- [MLflow GenAI Docs](https://mlflow.org/docs/latest/genai/)
- [Tracing Quickstart](https://mlflow.org/docs/latest/genai/tracing/quickstart/)
- [Evaluation & Monitoring](https://mlflow.org/docs/latest/genai/eval-monitor/)
- [LLM Judges](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/)
- [Evaluation Datasets](https://mlflow.org/docs/latest/genai/datasets/)
- [Prompt Optimization](https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/)
- [Production Monitoring](https://mlflow.org/docs/latest/genai/tracing/prod-tracing/)
