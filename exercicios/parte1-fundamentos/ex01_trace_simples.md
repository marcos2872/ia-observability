# Exercício 01 — Adicionando tracing a um pipeline

**Parte 1 — Fundamentos**
**Baseado em:** `tracing_basics.py` (Demo 2 - spans aninhados)
**Dificuldade:** 🟢 Fácil

## Problema

Você tem uma função `analisar_sentimento(texto)` que chama o LLM
duas vezes:
1. Classificar sentimento (positivo/negativo/neutro)
2. Gerar justificativa

Adicione tracing para que cada etapa apareça como um span separado.

## Requisitos

- Use `@mlflow.trace` em cada função
- Use `span_type="LLM"` nas chamadas ao modelo
- O span pai deve mostrar o texto de entrada e a análise completa

## Código inicial

```python
import mlflow
from ia_observability.config import MODEL_NAME, get_client, setup_mlflow

def classificar_sentimento(texto: str) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Classifique o sentimento como POSITIVO, NEGATIVO ou NEUTRO."},
            {"role": "user", "content": texto},
        ],
    )
    return resp.choices[0].message.content

def gerar_justificativa(texto: str, sentimento: str) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": f"Justifique por que o sentimento é {sentimento}."},
            {"role": "user", "content": texto},
        ],
    )
    return resp.choices[0].message.content

def analisar_sentimento(texto: str) -> str:
    sentimento = classificar_sentimento(texto)
    justificativa = gerar_justificativa(texto, sentimento)
    return f"Sentimento: {sentimento}\nJustificativa: {justificativa}"

if __name__ == "__main__":
    setup_mlflow("ex01-trace-simples")
    mlflow.openai.autolog()
    resultado = analisar_sentimento("MLflow é incrível! Amei o tracing automático.")
    print(resultado)
```

## Para verificar

1. Rode o código
2. Abra o experimento `ex01-trace-simples` no MLflow UI
3. Você deve ver 3 spans aninhados: `analisar_sentimento` (pai) → `classificar_sentimento` + `gerar_justificativa` (filhos)
