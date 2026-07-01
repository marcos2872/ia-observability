"""
[Parte 1 — Fundamentos] Módulo 02: Token Usage e Custo
========================================================
╔══════════════════════════════════════════════════════════╗
║  OBJETIVOS DE APRENDIZADO                               ║
║  • Entender o que são tokens e por que custam dinheiro  ║
║  • Saber como o MLflow captura token usage              ║
║  • Aprender a atribuir custo manualmente para modelos   ║
║    self-hosted (que não têm pricing registrado)          ║
║  • Visualizar custo por span em pipelines multi-step    ║
╚══════════════════════════════════════════════════════════╝

CONCEITO-CHAVE:
  Toda chamada a um LLM consome INPUT tokens (o que você envia)
  e OUTPUT tokens (o que o modelo gera). Isso tem um CUSTO.
  Modelos comerciais (OpenAI, Anthropic) têm pricing conhecido;
  modelos self-hosted (via Gateway) precisam de atribuição manual.

PRÉ-REQUISITOS:  Módulo 01 (tracing_basics) — entender spans
DIFICULDADE:     🟢 Fácil
TEMPO ESTIMADO:  15 min

--- Como usar ---
  uv run tokens    ou    make tokens

Referência: https://mlflow.org/docs/latest/genai/tracing/token-usage-cost/
"""

import mlflow
from openai import APIError, APITimeoutError, RateLimitError

from ia_observability.config import MODEL_NAME, apply_patches, get_client, setup_mlflow

# Pricing customizado para modelo self-hosted (USD por token)
# Ajuste conforme custo real de infra (GPU, energia, etc.)
CUSTOM_INPUT_COST_PER_TOKEN = 0.000001  # $1.00 / 1M input tokens
CUSTOM_OUTPUT_COST_PER_TOKEN = 0.000002  # $2.00 / 1M output tokens


def _set_usage_and_cost(span: mlflow.Span, usage: dict) -> None:
    """Seta token usage E custo customizado em um span.

    Padrao recomendado para modelos self-hosted sem pricing
    registrado no MLflow: o custo precisa ser calculado e setado manualmente
    a partir do token count retornado pelo modelo.

    Centralizar essa logica aqui evita duplicacao e garante que todos os
    spans sejam atribuidos da mesma forma.

    Args:
        span: Span ativo (de @mlflow.trace ou mlflow.start_span).
        usage: Objeto usage do response OpenAI (prompt_tokens, etc.).
    """
    if not span or not usage:
        return

    input_tokens = usage.prompt_tokens or 0
    output_tokens = usage.completion_tokens or 0
    total_tokens = usage.total_tokens or (input_tokens + output_tokens)

    input_cost = input_tokens * CUSTOM_INPUT_COST_PER_TOKEN
    output_cost = output_tokens * CUSTOM_OUTPUT_COST_PER_TOKEN

    # Token usage no formato que o MLflow UI exibe na aba de tokens
    span.set_attribute(
        "mlflow.chat.tokenUsage",
        {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
    )
    # Custo no formato que o MLflow UI exibe no "Cost Breakdown"
    span.set_attribute(
        "mlflow.llm.cost",
        {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
        },
    )


def demo_automatic_token_tracking() -> None:
    """Executa chamadas com cost attribution via wrapper.

    O auto-tracing captura token usage. Para cost, usamos start_span()
    e setamos o custo manualmente baseado no token count do response.
    """
    mlflow.openai.autolog()
    client = get_client()

    prompts = [
        "Explique MLOps em uma frase.",
        "Escreva um paragrafo sobre monitoramento de modelos de linguagem em producao.",
        "Liste 3 metricas essenciais para observabilidade de LLMs.",
    ]

    print("  Enviando 3 prompts com cost attribution...\n")

    for prompt in prompts:
        with mlflow.start_span(name="llm_call_with_cost") as span:
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                )
                # Seta token usage + custo no span (modelo self-hosted)
                _set_usage_and_cost(span, response.usage)
            except (APITimeoutError, RateLimitError) as e:
                print(f"[ERRO] Falha na chamada ao modelo: {e}")
                continue
            except APIError as e:
                print(f"[ERRO] Falha na chamada ao modelo: {e}")
                continue
            except Exception as e:
                print(f"[ERRO] Falha na chamada ao modelo: {e}")
                continue

    # Flush async logging antes de buscar traces
    mlflow.flush_trace_async_logging()

    # Busca traces programaticamente
    traces_df = mlflow.search_traces(max_results=3)

    print(f"  {'Trace ID':<12} | {'Input':<8} | {'Output':<8} | {'Total':<8} | {'Custo':<15}")
    print(f"  {'-'*12} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*15}")

    for _, row in traces_df.iterrows():
        trace = mlflow.get_trace(trace_id=row["trace_id"])
        usage = trace.info.token_usage
        cost = trace.info.cost

        trace_short = trace.info.trace_id[:10]
        if usage:
            input_t = usage.get("input_tokens", 0)
            output_t = usage.get("output_tokens", 0)
            total_t = usage.get("total_tokens", 0)
        else:
            input_t = output_t = total_t = "N/A"

        if cost:
            cost_str = f"${cost.get('total_cost', 0):.6f}"
        else:
            cost_str = "N/A"

        print(
            f"  {trace_short:<12} | {str(input_t):<8} | {str(output_t):<8} | "
            f"{str(total_t):<8} | {cost_str:<15}"
        )

    print(f"\n  Pricing: input=${CUSTOM_INPUT_COST_PER_TOKEN*1_000_000:.2f}/1M tokens, "
          f"output=${CUSTOM_OUTPUT_COST_PER_TOKEN*1_000_000:.2f}/1M tokens")


def demo_span_level_usage() -> None:
    """Pipeline multi-step com cost attribution em cada span.

    Demonstra como rastrear custo por etapa em pipelines complexos.
    """
    mlflow.openai.autolog()
    client = get_client()

    @mlflow.trace
    def multi_step_pipeline() -> str:
        # Passo 1: Classificacao
        with mlflow.start_span(name="classify") as span:
            try:
                classification = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "Classifique o topico: tech, business, science."},
                        {"role": "user", "content": "Como funciona o tracing em MLflow?"},
                    ],
                )
                _set_usage_and_cost(span, classification.usage)
            except (APITimeoutError, RateLimitError) as e:
                print(f"[ERRO] Falha na chamada ao modelo: {e}")
                return "(resposta indisponivel por erro do modelo)"
            except APIError as e:
                print(f"[ERRO] Falha na chamada ao modelo: {e}")
                return f"(erro na chamada: {str(e)})"
            except Exception as e:
                print(f"[ERRO] Falha na chamada ao modelo: {e}")
                return "(erro inesperado)"
        topic = classification.choices[0].message.content or "(categoria indisponivel)"

        # Passo 2: Resposta baseada na classificacao
        with mlflow.start_span(name="answer") as span:
            try:
                answer = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": f"Voce e especialista em {topic}. Responda brevemente."},
                        {"role": "user", "content": "Como funciona o tracing em MLflow?"},
                    ],
                )
                _set_usage_and_cost(span, answer.usage)
                return answer.choices[0].message.content or "(resposta vazia)"
            except (APITimeoutError, RateLimitError) as e:
                print(f"[ERRO] Falha na chamada ao modelo: {e}")
                return "(erro: servidor temporariamente indisponivel)"
            except APIError as e:
                print(f"[ERRO] Falha na chamada ao modelo: {e}")
                return f"(erro na chamada: {str(e)})"
            except Exception as e:
                print(f"[ERRO] Falha na chamada ao modelo: {e}")
                return "(erro inesperado)"

    result = multi_step_pipeline()
    print(f"  Pipeline result: {result[:100]}...\n")

    # Acessa token usage e custo por span
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
            cost = span.get_attribute("mlflow.llm.cost")
            if usage:
                cost_str = f"${cost['total_cost']:.6f}" if cost else "N/A"
                print(
                    f"    {span.name}: "
                    f"in={usage.get('input_tokens', '?')}, "
                    f"out={usage.get('output_tokens', '?')}, "
                    f"total={usage.get('total_tokens', '?')}, "
                    f"cost={cost_str}"
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
    apply_patches()
    setup_mlflow("02-token-usage")

    print("=" * 60)
    print("DEMO 1: Token tracking com cost attribution")
    print("=" * 60)
    demo_automatic_token_tracking()

    print("\n" + "=" * 60)
    print("DEMO 2: Token usage + custo por span (pipeline multi-step)")
    print("=" * 60)
    demo_span_level_usage()

    print("\n" + "=" * 60)
    print("DEMO 3: Token attribution manual (modelo local)")
    print("=" * 60)
    result = demo_manual_token_attribution()
    print(f"  Resultado: {result}")

    # ────────────────────────────────────────────────────
    #  ✅ RESUMO DO QUE APRENDEMOS NESTE MÓDULO
    # ────────────────────────────────────────────────────
    #  ✔ MLflow captura token usage automaticamente com
    #     mlflow.openai.autolog().
    #  ✔ Para modelos self-hosted, o custo precisa ser
    #     setado manualmente via span attributes.
    #  ✔ Padrão: calcular input_cost + output_cost usando
    #     pricing customizado e setar em
    #     span.set_attribute("mlflow.llm.cost", {...}).
    #  ✔ Custos por span individual permitem ver qual
    #     etapa do pipeline está mais cara.
    #
    #  🔍 MLflow UI → Experiment '02-token-usage':
    #     gráfico de Token Usage e Cost Breakdown.
    #
    #  💡 EXERCÍCIO: Mude CUSTOM_INPUT_COST_PER_TOKEN e
    #     CUSTOM_OUTPUT_COST_PER_TOKEN para refletir o
    #     custo real do seu modelo (GPU, energia) e rode
    #     novamente. O custo total mudou?
    # ────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("✅ MÓDULO CONCLUÍDO! Resumo do aprendizado acima.")
    print("  Experiment: 02-token-usage no MLflow UI")
    print("-" * 60)


if __name__ == "__main__":
    main()
