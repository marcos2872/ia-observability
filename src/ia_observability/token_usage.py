"""Demonstracao de tracking de token usage e custo por chamada.

O MLflow captura automaticamente:
- Token usage: input_tokens, output_tokens, total_tokens
- Custo: calculado automaticamente APENAS para modelos com pricing registrado
  (OpenAI, Anthropic, etc.). Para modelos self-hosted, o custo deve ser setado
  manualmente via span attributes.

Tambem e possivel setar esses valores manualmente para modelos que nao
reportam usage automaticamente.

Referencia: https://mlflow.org/docs/latest/genai/tracing/token-usage-cost/
"""

import mlflow

from ia_observability.config import MODEL_NAME, get_client, setup_mlflow

# Pricing customizado para modelo self-hosted (USD por token)
# Ajuste conforme custo real de infra (GPU, energia, etc.)
CUSTOM_INPUT_COST_PER_TOKEN = 0.000001  # $1.00 / 1M input tokens
CUSTOM_OUTPUT_COST_PER_TOKEN = 0.000002  # $2.00 / 1M output tokens


def demo_automatic_token_tracking() -> None:
    """Executa chamadas e exibe metricas de token/custo capturadas automaticamente.

    O auto-tracing do MLflow captura token usage de qualquer chamada
    ao OpenAI SDK. Custo automatico requer modelo com pricing registrado.
    Para modelos self-hosted, calculamos custo manualmente com pricing customizado.
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

    # Flush async logging antes de buscar traces
    mlflow.flush_trace_async_logging()

    # Busca traces programaticamente
    traces_df = mlflow.search_traces(max_results=3)

    print(f"  {'Trace ID':<12} | {'Input':<8} | {'Output':<8} | {'Total':<8} | {'Custo (custom)':<15}")
    print(f"  {'-'*12} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*15}")

    for _, row in traces_df.iterrows():
        trace = mlflow.get_trace(trace_id=row["trace_id"])
        usage = trace.info.token_usage

        trace_short = trace.info.trace_id[:10]
        if usage:
            input_t = usage.get("input_tokens", 0)
            output_t = usage.get("output_tokens", 0)
            total_t = usage.get("total_tokens", 0)
            # Calcula custo customizado para modelo self-hosted
            custom_cost = (
                input_t * CUSTOM_INPUT_COST_PER_TOKEN
                + output_t * CUSTOM_OUTPUT_COST_PER_TOKEN
            )
            cost_str = f"${custom_cost:.6f}"
        else:
            input_t = output_t = total_t = "N/A"
            cost_str = "N/A"

        print(f"  {trace_short:<12} | {str(input_t):<8} | {str(output_t):<8} | {str(total_t):<8} | {cost_str:<15}")

    print(f"\n  Pricing: input=${CUSTOM_INPUT_COST_PER_TOKEN*1_000_000:.2f}/1M tokens, "
          f"output=${CUSTOM_OUTPUT_COST_PER_TOKEN*1_000_000:.2f}/1M tokens")
    print("  (MLflow so calcula custo automatico para modelos com pricing registrado;"
          "\n   para self-hosted, use set_attribute no span conforme Demo 3)")


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
    # Flush pendente de async logging antes de buscar o trace
    mlflow.flush_trace_async_logging()
    trace_id = mlflow.get_last_active_trace_id()
    if trace_id:
        trace = mlflow.get_trace(trace_id=trace_id)
        if trace is None:
            print("  [WARN] Trace nao disponivel ainda no tracking store.")
            return
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
    """Demonstra como setar token usage e custo manualmente em um span.

    Essencial para modelos self-hosted onde o MLflow nao tem pricing registrado.
    Padrao recomendado: calcular custo baseado em token count + pricing customizado.
    """
    span = mlflow.get_current_active_span()

    # Simula chamada a um modelo local/customizado
    result = "Resposta simulada de um modelo local sem contagem automatica de tokens."

    # Valores de token usage (em cenario real, viria do response do modelo)
    input_tokens = 42
    output_tokens = 15
    total_tokens = input_tokens + output_tokens

    # Seta manualmente os atributos de token usage
    span.set_attribute(
        "mlflow.chat.tokenUsage",
        {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
    )

    # Calcula e seta custo baseado em pricing customizado
    input_cost = input_tokens * CUSTOM_INPUT_COST_PER_TOKEN
    output_cost = output_tokens * CUSTOM_OUTPUT_COST_PER_TOKEN
    span.set_attribute(
        "mlflow.llm.cost",
        {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
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
    print("Abra o MLflow UI -> Experiment '02-token-usage' para ver token usage.")
    print("O grafico de custo mostra valores apenas para traces com custo")
    print("setado manualmente (Demo 3) — modelos self-hosted nao tem pricing")
    print("automatico no MLflow.")
    print("-" * 60)


if __name__ == "__main__":
    main()
