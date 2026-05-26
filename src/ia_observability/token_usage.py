"""Demonstracao de tracking de token usage e custo por chamada.

O MLflow captura automaticamente:
- Token usage: input_tokens, output_tokens, total_tokens
- Custo: input_cost, output_cost, total_cost (requer MLflow >= 3.10 com [genai])

Tambem e possivel setar esses valores manualmente para modelos que nao
reportam usage automaticamente.

Referencia: https://mlflow.org/docs/latest/genai/tracing/token-usage-cost/
"""

import mlflow

from ia_observability.config import MODEL_NAME, get_client, setup_mlflow


def demo_automatic_token_tracking() -> None:
    """Executa chamadas e exibe metricas de token/custo capturadas automaticamente.

    O auto-tracing do MLflow captura token usage de qualquer chamada
    ao OpenAI SDK. Os dados ficam disponiveis tanto no UI quanto via API.
    """
    mlflow.openai.autolog()
    client = get_client()

    prompts = [
        "Explique MLOps em uma frase.",
        "Escreva um paragrafo sobre monitoramento de modelos de linguagem em producao.",
        "Liste 3 metricas essenciais para observabilidade de LLMs.",
    ]

    print("  Enviando 3 prompts com complexidades diferentes...\n")

    for prompt in prompts:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )

    # Busca traces programaticamente
    traces_df = mlflow.search_traces(max_results=3)

    print(f"  {'Trace ID':<12} | {'Input':<8} | {'Output':<8} | {'Total':<8} | {'Custo':<12}")
    print(f"  {'-'*12} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*12}")

    for _, row in traces_df.iterrows():
        trace = mlflow.get_trace(trace_id=row["trace_id"])
        usage = trace.info.token_usage
        cost = trace.info.cost

        trace_short = trace.info.trace_id[:10]
        if usage:
            input_t = str(usage.get("input_tokens", "?"))
            output_t = str(usage.get("output_tokens", "?"))
            total_t = str(usage.get("total_tokens", "?"))
        else:
            input_t = output_t = total_t = "N/A"

        if cost:
            cost_str = f"${cost.get('total_cost', 0):.6f}"
        else:
            cost_str = "N/A"

        print(f"  {trace_short:<12} | {input_t:<8} | {output_t:<8} | {total_t:<8} | {cost_str:<12}")


def demo_span_level_usage() -> None:
    """Acessa token usage no nivel de span individual.

    Util para pipelines com multiplas chamadas LLM, onde voce quer
    saber o consumo de cada etapa separadamente.
    """
    mlflow.openai.autolog()
    client = get_client()

    # Cria um trace com multiplas chamadas
    @mlflow.trace
    def multi_step_pipeline() -> str:
        # Passo 1: Classificacao
        classification = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Classifique o topico: tech, business, science."},
                {"role": "user", "content": "Como funciona o tracing em MLflow?"},
            ],
        )
        topic = classification.choices[0].message.content

        # Passo 2: Resposta baseada na classificacao
        answer = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": f"Voce e especialista em {topic}. Responda brevemente."},
                {"role": "user", "content": "Como funciona o tracing em MLflow?"},
            ],
        )
        return answer.choices[0].message.content

    result = multi_step_pipeline()
    print(f"  Pipeline result: {result[:100]}...\n")

    # Acessa token usage por span
    trace_id = mlflow.get_last_active_trace_id()
    if trace_id:
        trace = mlflow.get_trace(trace_id=trace_id)
        print(f"  Detalhamento por span:")
        for span in trace.data.spans:
            usage = span.get_attribute("mlflow.chat.tokenUsage")
            if usage:
                print(
                    f"    {span.name}: "
                    f"in={usage.get('input_tokens', '?')}, "
                    f"out={usage.get('output_tokens', '?')}, "
                    f"total={usage.get('total_tokens', '?')}"
                )


@mlflow.trace
def demo_manual_token_attribution() -> str:
    """Demonstra como setar token usage manualmente em um span.

    Util para:
    - Modelos locais que nao reportam usage
    - APIs customizadas
    - Simulacoes e testes
    """
    span = mlflow.get_current_active_span()

    # Simula chamada a um modelo local/customizado
    result = "Resposta simulada de um modelo local sem contagem automatica de tokens."

    # Seta manualmente os atributos de token usage
    span.set_attribute(
        "mlflow.chat.tokenUsage",
        {
            "input_tokens": 42,
            "output_tokens": 15,
            "total_tokens": 57,
        },
    )

    # Seta custo manualmente (em USD)
    span.set_attribute(
        "mlflow.llm.cost",
        {
            "input_cost": 0.00001,
            "output_cost": 0.00002,
            "total_cost": 0.00003,
        },
    )

    return result


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Executa todas as demos de token usage."""
    setup_mlflow("02-token-usage")

    print("=" * 60)
    print("DEMO 1: Token tracking automatico")
    print("=" * 60)
    demo_automatic_token_tracking()

    print("\n" + "=" * 60)
    print("DEMO 2: Token usage por span (pipeline multi-step)")
    print("=" * 60)
    demo_span_level_usage()

    print("\n" + "=" * 60)
    print("DEMO 3: Token attribution manual")
    print("=" * 60)
    result = demo_manual_token_attribution()
    print(f"  Resultado: {result}")

    print("\n" + "-" * 60)
    print("Abra o MLflow UI -> Experiment '02-token-usage' para ver graficos de custo.")
    print("-" * 60)


if __name__ == "__main__":
    main()
