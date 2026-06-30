# Exercício 03 — Code-based scorer: detector de código Python

**Parte 2 — Avaliação**
**Baseado em:** `judges.py`
**Dificuldade:** 🟡 Médio

## Problema

Em respostas técnicas, é comum o LLM incluir exemplos de código.
Crie um code-based scorer que verifica se a resposta contém
blocos de código Python (```python ... ```).

## Requisitos

- Use `@scorer` e retorne `Feedback`
- Value = True se encontrar código Python, False caso contrário
- Rationale deve indicar quantos blocos foram encontrados
- Adicione o scorer à lista de scorers no `main()` do judges.py

## Dica

Use string.find para detectar ```python no texto.
