"""Gabarito: Exercício 01 — Tracing em pipeline de sentimento."""
import mlflow
from ia_observability.config import MODEL_NAME, get_client, setup_mlflow


@mlflow.trace(span_type="LLM")
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


@mlflow.trace(span_type="LLM")
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


@mlflow.trace
def analisar_sentimento(texto: str) -> str:
    sentimento = classificar_sentimento(texto)
    justificativa = gerar_justificativa(texto, sentimento)
    return f"Sentimento: {sentimento}\nJustificativa: {justificativa}"


if __name__ == "__main__":
    setup_mlflow("ex01-trace-simples")
    mlflow.openai.autolog()
    resultado = analisar_sentimento("MLflow é incrível! Amei o tracing automático.")
    print(resultado)
