# ia-observability

Projeto base de **observabilidade e benchmark de LLM** usando [MLflow GenAI](https://mlflow.org/docs/latest/genai/).

Serve como referencia para integrar observabilidade em projetos futuros.

## Funcionalidades Demonstradas

| Modulo | Funcionalidade | Comando |
|--------|---------------|---------|
| `tracing_basics` | Auto-tracing, `@mlflow.trace`, spans aninhados (RAG) | `uv run tracing` |
| `token_usage` | Token usage por chamada, custo, attribution manual | `uv run tokens` |
| `sessions` | Sessions multi-turn, user tracking, queries | `uv run sessions` |
| `evaluation` | Benchmark com datasets + scorers built-in | `uv run evaluation` |
| `judges` | LLM judges customizados + code-based scorers | `uv run judges` |
| `version_tracking` | Versionamento com LoggedModel | `uv run versioning` |
| `production_monitoring` | Async logging, sampling, feedback | `uv run monitoring` |
| `experiment_comparison` | Comparacao de configs lado a lado | `uv run benchmark` |
| `tool_calls` | Tool calling com observabilidade (AGENT/TOOL spans) | `uv run toolcalls` |

## Requisitos

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (package manager)
- MLflow server rodando (com AI Gateway configurado)

## Setup

```bash
# Instalar dependencias
uv sync

# Configurar .env (copie de .env.example)
# mlflow_url=http://<seu-server>:5000/
# mlflow_openia_url=http://<seu-server>:5000/gateway/mlflow/v1
# mlflow_model=gemma4-e4b
# mlflow_judge_model=gemma4-e4b  # modelo para LLM judges (pode ser diferente)
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

Cada modulo cria um experiment separado (ex: `01-tracing-basics`, `04-evaluation`).

## Arquitetura

```
src/ia_observability/
├── config.py              # Configuracao centralizada (.env + MLflow + OpenAI client)
├── tracing_basics.py      # Auto-tracing, decorators, context blocks
├── token_usage.py         # Token counting, custo, span-level usage
├── sessions.py            # Multi-turn conversations, user/session tracking
├── evaluation.py          # mlflow.genai.evaluate() com Correctness, RelevanceToQuery
├── judges.py              # Custom Guidelines judges + code-based scorers
├── version_tracking.py    # LoggedModel, versionamento por git/config
├── production_monitoring.py # Async, sampling, feedback collection
├── experiment_comparison.py # Benchmark comparativo de configuracoes
└── tool_calls.py          # Tool calling com spans AGENT/TOOL/CHAT_MODEL
```

## Stack

- **MLflow >= 3.10** (com extra `[genai]`) - plataforma de observabilidade
- **OpenAI SDK** - cliente apontando para MLflow AI Gateway
- **python-dotenv** - gerenciamento de variaveis de ambiente

## Referencia

- [MLflow GenAI Docs](https://mlflow.org/docs/latest/genai/)
- [Tracing Quickstart](https://mlflow.org/docs/latest/genai/tracing/quickstart/)
- [Evaluation & Monitoring](https://mlflow.org/docs/latest/genai/eval-monitor/)
- [LLM Judges](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/)
- [Token Usage & Cost](https://mlflow.org/docs/latest/genai/tracing/token-usage-cost/)
- [Production Monitoring](https://mlflow.org/docs/latest/genai/tracing/prod-tracing/)
