"""Demonstracao de tracing: auto-tracing, decorators e context blocks.

Este modulo mostra as 3 formas principais de instrumentar codigo com MLflow Tracing:
1. Auto-tracing (mlflow.openai.autolog) - captura automatica de chamadas OpenAI
2. Decorator (@mlflow.trace) - tracing de funcoes customizadas
3. Spans aninhados - visibilidade em pipelines multi-etapa (RAG pattern)

Referencia: https://mlflow.org/docs/latest/genai/tracing/quickstart/
"""

import mlflow

from ia_observability.config import MODEL_NAME, get_client, setup_mlflow


# ---------------------------------------------------------------------------
# Demo 1: Auto-tracing
# ---------------------------------------------------------------------------


def demo_auto_tracing() -> None:
    """Auto-tracing com mlflow.openai.autolog().

    Com uma unica linha, todas as chamadas ao OpenAI SDK sao capturadas
    automaticamente como traces no MLflow, incluindo:
    - Inputs (messages)
    - Outputs (completions)
    - Token usage
    - Latencia
    """
    mlflow.openai.autolog()
    client = get_client()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Voce e um assistente util."},
            {"role": "user", "content": "O que e observabilidade em sistemas de IA?"},
        ],
    )
    print(f"  Resposta: {response.choices[0].message.content[:150]}...")


# ---------------------------------------------------------------------------
# Demo 2: Tracing manual com decorator e spans aninhados (RAG pattern)
# ---------------------------------------------------------------------------


@mlflow.trace
def demo_rag_pipeline(question: str) -> str:
    """Pipeline RAG completo com spans aninhados.

    O decorator @mlflow.trace cria um span pai. As funcoes internas
    tambem decoradas criam spans filhos, formando uma arvore de execucao
    visivel no MLflow UI.
    """
    context = retrieve_context(question)
    answer = generate_answer(question, context)
    return answer


@mlflow.trace(span_type="RETRIEVER")
def retrieve_context(question: str) -> str:
    """Simula retrieval de contexto (span de tipo RETRIEVER).

    Em producao, isso seria uma busca em vector store (Pinecone, Chroma, etc).
    O span_type="RETRIEVER" permite ao MLflow exibir icones e metricas
    especificas para retrieval no UI.
    """
    knowledge_base = {
        "observabilidade": (
            "Observabilidade e a capacidade de entender o estado interno de um sistema "
            "a partir de suas saidas externas. Em IA, isso inclui traces de execucao, "
            "metricas de token usage, latencia e qualidade das respostas."
        ),
        "mlflow": (
            "MLflow e uma plataforma open source para gerenciar o ciclo de vida de ML. "
            "Para LLMs, oferece tracing, evaluation com judges, prompt management e "
            "monitoramento em producao."
        ),
        "tracing": (
            "Tracing captura inputs, outputs e metadados de cada passo intermediario "
            "de uma requisicao, permitindo identificar a origem de bugs e comportamentos "
            "inesperados em aplicacoes de LLM."
        ),
    }

    # Busca simples por keyword matching
    for key, value in knowledge_base.items():
        if key in question.lower():
            return value
    return "Sem contexto especifico disponivel para esta pergunta."


@mlflow.trace(span_type="LLM")
def generate_answer(question: str, context: str) -> str:
    """Gera resposta usando o LLM com contexto recuperado.

    O span_type="LLM" identifica este span como uma chamada ao modelo,
    permitindo ao MLflow capturar metricas especificas de LLM.
    """
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Responda a pergunta usando APENAS o contexto fornecido. "
                    f"Contexto: {context}"
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Demo 3: Tracing com context block (util para wrappar codigo existente)
# ---------------------------------------------------------------------------


def demo_context_block() -> None:
    """Demonstra tracing usando context manager (bloco with).

    Util quando voce quer instrumentar um trecho de codigo sem
    refatorar em funcoes separadas.
    """
    client = get_client()

    with mlflow.start_span(name="context-block-demo") as span:
        span.set_inputs({"task": "summarization"})

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": "Resuma em 1 frase: MLflow e uma plataforma de MLOps."},
            ],
        )
        result = response.choices[0].message.content
        span.set_outputs({"summary": result})

    print(f"  Resumo: {result}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Executa todas as demos de tracing."""
    setup_mlflow("01-tracing-basics")

    print("=" * 60)
    print("DEMO 1: Auto-tracing (mlflow.openai.autolog)")
    print("=" * 60)
    demo_auto_tracing()

    print("\n" + "=" * 60)
    print("DEMO 2: Tracing manual - pipeline RAG com spans aninhados")
    print("=" * 60)
    result = demo_rag_pipeline("O que e observabilidade em IA?")
    print(f"  Resposta: {result[:150]}...")

    print("\n" + "=" * 60)
    print("DEMO 3: Tracing com context block (mlflow.start_span)")
    print("=" * 60)
    demo_context_block()

    print("\n" + "-" * 60)
    print("Abra o MLflow UI para visualizar os traces gerados:")
    print(f"  -> Experiment: 01-tracing-basics")
    print("-" * 60)


if __name__ == "__main__":
    main()
