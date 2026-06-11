.PHONY: help install tracing tokens sessions evaluation judges versioning monitoring benchmark toolcalls prompts langchain-agent datasets prompt-opt all

help: ## Mostra esta ajuda
	@echo "Comandos disponiveis:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Instala dependencias com uv
	uv sync

tracing: ## Executa demo de tracing (auto-tracing, decorators, spans)
	uv run tracing

tokens: ## Executa demo de token usage e custo por chamada
	uv run tokens

sessions: ## Executa demo de sessions multi-turn e user tracking
	uv run sessions

evaluation: ## Executa demo de evaluation com scorers built-in
	uv run evaluation

judges: ## Executa demo de LLM judges customizados
	uv run judges

versioning: ## Executa demo de version tracking com LoggedModel
	uv run versioning

monitoring: ## Executa demo de producao (async, sampling, feedback)
	uv run monitoring

benchmark: ## Executa benchmark comparativo de configuracoes
	uv run benchmark

toolcalls: ## Executa demo de tool calling com observabilidade
	uv run toolcalls

prompts: ## Executa demo de prompt registry e versionamento
	uv run prompts

langchain-agent: ## Executa demo de LangChain agent (tools + sessions)
	uv run langchain-agent

datasets: ## Executa demo de evaluation datasets (subir + buscar)
	uv run datasets

prompt-opt: ## Executa demo de prompt optimization (GEPA + Metaprompt)
	uv run prompt-opt

all: tracing tokens sessions evaluation judges versioning monitoring benchmark toolcalls prompts langchain-agent datasets prompt-opt ## Executa todos os modulos em sequencia
