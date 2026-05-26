"""Demonstracao de configuracoes de producao: async logging, sampling, feedback.

Em producao, e essencial:
1. Nao bloquear a aplicacao com logging de traces (async)
2. Controlar volume/custo com sampling
3. Coletar feedback de usuarios para melhorar qualidade
4. Usar sampling diferenciado por criticidade da operacao

Referencia: https://mlflow.org/docs/latest/genai/tracing/prod-tracing/
"""

import os

import mlflow
from mlflow.entities import AssessmentSource

from ia_observability.config import MODEL_NAME, get_client, setup_mlflow


def show_production_config() -> None:
    """Mostra e aplica variaveis de ambiente para producao.

    Estas variaveis controlam o comportamento do tracing em producao:
    - MLFLOW_ENABLE_ASYNC_TRACE_LOGGING: traces em background (default: true)
    - MLFLOW_ASYNC_TRACE_LOGGING_MAX_WORKERS: threads de upload
    - MLFLOW_ASYNC_TRACE_LOGGING_MAX_QUEUE_SIZE: fila antes de descartar
    - MLFLOW_TRACE_SAMPLING_RATIO: % de traces capturados (1.0 = 100%)
    """
    production_env = {
        "MLFLOW_ENABLE_ASYNC_TRACE_LOGGING": "true",
        "MLFLOW_ASYNC_TRACE_LOGGING_MAX_WORKERS": "10",
        "MLFLOW_ASYNC_TRACE_LOGGING_MAX_QUEUE_SIZE": "1000",
        "MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT": "500",
        "MLFLOW_TRACE_SAMPLING_RATIO": "0.5",  # 50% em dev, use 0.1 em prod
    }

    print("  Variaveis de ambiente configuradas:")
    for key, value in production_env.items():
        os.environ[key] = value
        print(f"    {key} = {value}")


# ---------------------------------------------------------------------------
# Per-endpoint sampling: diferentes taxas por criticidade
# ---------------------------------------------------------------------------


@mlflow.trace(sampling_ratio_override=1.0)
def critical_operation(question: str) -> str:
    """Operacao critica - SEMPRE traced (100% sampling).

    Use sampling_ratio_override=1.0 para operacoes que precisam
    ser auditadas em 100% dos casos (ex: pagamentos, compliance).
    """
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content


@mlflow.trace(sampling_ratio_override=0.1)
def high_volume_operation(question: str) -> str:
    """Operacao de alto volume - sampling reduzido (10%).

    Para endpoints de alto trafego, capture apenas uma amostra.
    Isso reduz custo de storage e processamento significativamente.
    """
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Feedback collection
# ---------------------------------------------------------------------------


def demo_feedback_collection() -> None:
    """Demonstra coleta de feedback humano em traces.

    Em producao, feedback pode vir de:
    - Botoes thumbs up/down na UI do chat
    - Revisoes de quality analysts
    - Scores automaticos pos-interacao

    O feedback fica vinculado ao trace e e usado para:
    - Melhorar judges (alignment)
    - Identificar patterns de falha
    - Treinar/fine-tunar modelos
    """
    client = get_client()

    # Gera uma resposta trackeada
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": "O que e MLflow tracing?"},
        ],
    )
    answer = response.choices[0].message.content
    print(f"  Resposta gerada: {answer[:100]}...")

    trace_id = mlflow.get_last_active_trace_id()

    if trace_id:
        # Feedback positivo de um reviewer humano
        mlflow.log_feedback(
            trace_id=trace_id,
            name="user_satisfaction",
            value=True,
            source=AssessmentSource(
                source_type="HUMAN",
                source_id="reviewer-marcos",
            ),
            rationale="Resposta clara, concisa e tecnicamente correta.",
        )
        print(f"  Feedback POSITIVO registrado no trace {trace_id[:10]}...")

        # Feedback com score numerico (1-5)
        mlflow.log_feedback(
            trace_id=trace_id,
            name="quality_score",
            value=4,  # 4 de 5
            source=AssessmentSource(
                source_type="HUMAN",
                source_id="reviewer-marcos",
            ),
            rationale="Boa resposta, mas poderia incluir um exemplo pratico.",
        )
        print(f"  Score 4/5 registrado no trace {trace_id[:10]}...")
    else:
        print("  WARN: Nenhum trace ativo encontrado para registrar feedback.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Executa demos de configuracao de producao."""
    setup_mlflow("07-production-monitoring")
    mlflow.openai.autolog()

    print("=" * 60)
    print("CONFIGURACOES DE PRODUCAO")
    print("=" * 60)
    show_production_config()

    print("\n" + "=" * 60)
    print("DEMO 1: Operacao critica (100% sampling)")
    print("=" * 60)
    result = critical_operation("Explique observabilidade de LLM em 1 frase.")
    print(f"  Resposta: {result[:120]}...")

    print("\n" + "=" * 60)
    print("DEMO 2: Operacoes de alto volume (10% sampling)")
    print("=" * 60)
    traced_count = 0
    total = 10
    for i in range(total):
        high_volume_operation(f"Pergunta de alto volume #{i+1}: O que e MLOps?")
    print(f"  {total} chamadas feitas (apenas ~10% serao traced)")

    # Verifica quantas foram realmente traced
    traces = mlflow.search_traces(max_results=20)
    print(f"  Traces capturados no total (experiment): {len(traces)}")

    print("\n" + "=" * 60)
    print("DEMO 3: Coleta de feedback humano")
    print("=" * 60)
    demo_feedback_collection()

    print("\n" + "-" * 60)
    print("Em PRODUCAO, tambem considere:")
    print("  - mlflow-tracing (SDK leve, 95% menor que mlflow completo)")
    print("  - Automatic evaluation com judges registrados")
    print("  - Alerts baseados em metricas de qualidade")
    print("-" * 60)


if __name__ == "__main__":
    main()
