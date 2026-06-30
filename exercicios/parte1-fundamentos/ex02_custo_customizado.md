# Exercício 02 — Custo customizado para modelo self-hosted

**Parte 1 — Fundamentos**
**Baseado em:** `token_usage.py`
**Dificuldade:** 🟢 Fácil

## Problema

Você está rodando um modelo llama3-70b em GPU própria.
O custo por token é diferente do pricing do exemplo:

- Input: $0.50 / 1M tokens  (metade do exemplo)
- Output: $3.00 / 1M tokens (50% maior que o exemplo)

Modifique o código para refletir esse pricing e verifique
a diferença no custo total.

## Requisitos

1. Altere as constantes `CUSTOM_INPUT_COST_PER_TOKEN` e
   `CUSTOM_OUTPUT_COST_PER_TOKEN`
2. Execute `uv run tokens` e veja a diferença no "Cost Breakdown"
3. Confira que o custo total mudou proporcionalmente
