.PHONY: help install tracing token-usage sessions evaluation judges version-tracking monitoring benchmark tool-calls prompts langchain-agent datasets prompt-optimization all

help: ## Mostra esta ajuda
	@echo "Comandos disponiveis:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Instala dependencias com uv
	uv sync

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
