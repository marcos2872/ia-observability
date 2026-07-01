"""
[Parte 3 — Produção] Módulo 09: Tool Calling com Tracing Manual
=================================================================
╔══════════════════════════════════════════════════════════╗
║  OBJETIVOS DE APRENDIZADO                               ║
║  • Entender o padrão AGENT → THINK → TOOL → OBSERVE    ║
║  • Usar SpanType.TOOL para ferramentas individuais      ║
║  • Ver latência e erro de cada tool call no trace       ║
║  • Comparar implementação manual vs LangChain (mod 11)  ║
╚══════════════════════════════════════════════════════════╝

CONCEITO-CHAVE:
  Agentes que chamam ferramentas (APIs, bancos, busca) têm
  múltiplos pontos de falha. SpanType.TOOL faz cada
  ferramenta aparecer na aba "Tool Calls" do MLflow UI,
  mostrando input, output e tempo de cada chamada.

PRÉ-REQUISITOS:  Parte 1, Módulo 03 (sessions)
DIFICULDADE:     🔴 Avançado
TEMPO ESTIMADO:  20 min

--- Como usar ---
  uv run toolcalls    ou    make toolcalls
"""

import json
import time

import mlflow
from openai import APIError, APITimeoutError, RateLimitError
from mlflow.entities import SpanType

from ia_observability.config import MODEL_NAME, apply_patches, get_client, setup_mlflow
from ia_observability._tools import get_weather, search_docs, calculate, check_inventory

# ---------------------------------------------------------------------------
# Definicao de tools (funcoes que o modelo pode chamar)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Retorna a previsao do tempo para uma cidade.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Nome da cidade"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Unidade de temperatura",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Busca na base de conhecimento interna sobre MLflow.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Termo de busca"},
                    "max_results": {"type": "integer", "description": "Maximo de resultados"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Executa um calculo matematico.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Expressao matematica"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_inventory",
            "description": "Consulta o estoque disponivel de um produto.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {"type": "string", "description": "Nome do produto"},
                },
                "required": ["product"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Mapeamento nome -> funcao
TOOL_REGISTRY = {
    "get_weather": get_weather,
    "search_docs": search_docs,
    "calculate": calculate,
    "check_inventory": check_inventory,
}


# ---------------------------------------------------------------------------
# Agente com tool calling loop + tracing
# ---------------------------------------------------------------------------


@mlflow.trace(span_type=SpanType.AGENT)
def agent_with_tools(user_message: str) -> str:
    """Agente que usa tools com observabilidade completa.

    O fluxo tracado no MLflow:
    1. AGENT span (raiz) - contexto geral do agente
    2. CHAT_MODEL span - chamada ao LLM com tools disponiveis
    3. TOOL span(s) - execucao de cada tool chamada pelo modelo
    4. CHAT_MODEL span - chamada final com resultados das tools
    """
    client = get_client()
    messages = [
        {
            "role": "system",
            "content": (
                "Voce e um assistente que usa tools quando necessario. "
                "Responda em portugues de forma concisa."
            ),
        },
        {"role": "user", "content": user_message},
    ]

    # Primeira chamada — modelo decide se usa tools
    with mlflow.start_span(name="llm_with_tools", span_type=SpanType.CHAT_MODEL) as span:
        span.set_inputs({
            "messages": messages,
            "tools_available": [t["function"]["name"] for t in TOOLS],
        })
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            choice = response.choices[0]
            tool_calls = choice.message.tool_calls or []
            span.set_outputs({
                "finish_reason": choice.finish_reason,
                "tool_calls_count": len(tool_calls),
                "tool_calls": [
                    {"name": tc.function.name, "args": tc.function.arguments}
                    for tc in tool_calls
                ],
            })
        except (APITimeoutError, RateLimitError) as e:
            print(f"[ERRO] Falha na chamada ao modelo: {e}")
            return "(resposta indisponivel por erro do modelo)"
        except APIError as e:
            print(f"[ERRO] Falha na chamada ao modelo: {e}")
            return f"(erro na chamada: {str(e)})"
        except Exception as e:
            print(f"[ERRO] Falha na chamada ao modelo: {e}")
            return "(erro inesperado)"

    # Se o modelo decidiu chamar tools
    if tool_calls:
        messages.append(choice.message)

        # Executa cada tool chamada
        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            try:
                fn_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                fn_args = {"raw": tool_call.function.arguments}

            # Span TOOL para cada execucao — com medicao de latencia
            with mlflow.start_span(name=f"tool:{fn_name}", span_type=SpanType.TOOL) as tool_span:
                tool_span.set_inputs({"function": fn_name, "arguments": fn_args})
                tool_fn = TOOL_REGISTRY.get(fn_name)

                t0 = time.perf_counter()
                if tool_fn:
                    try:
                        result = tool_fn(**fn_args)
                    except Exception as exc:
                        # Captura falha da tool — registra erro no span
                        latency_ms = (time.perf_counter() - t0) * 1000
                        tool_span.set_attribute("tool.latency_ms", latency_ms)
                        tool_span.set_attribute("tool.error", True)
                        tool_span.set_attribute("tool.error_type", type(exc).__name__)
                        tool_span.set_status("ERROR")
                        result = {
                            "error": f"{type(exc).__name__}: {exc}",
                            "function": fn_name,
                        }
                        tool_span.set_outputs(result)
                        # Continua o loop — envia erro como resultado ao modelo
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result, ensure_ascii=False),
                        })
                        print(f"    [FALHA] tool:{fn_name} — {exc} ({latency_ms:.0f}ms)")
                        continue
                else:
                    result = {"error": f"Tool '{fn_name}' nao encontrada no registry"}

                latency_ms = (time.perf_counter() - t0) * 1000
                tool_span.set_attribute("tool.latency_ms", latency_ms)
                tool_span.set_attribute("tool.error", False)
                tool_span.set_outputs(result)
                print(f"    [OK] tool:{fn_name} ({latency_ms:.1f}ms)")

            # Adiciona resultado da tool ao historico de mensagens
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

        # Segunda chamada — modelo gera resposta final com resultados das tools
        with mlflow.start_span(name="llm_final_response", span_type=SpanType.CHAT_MODEL) as span:
            span.set_inputs({
                "messages_count": len(messages),
                "tools_executed": [tc.function.name for tc in tool_calls],
            })
            try:
                final_response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                )
                answer = final_response.choices[0].message.content or "(resposta vazia)"
                span.set_outputs({"response_length": len(answer) if answer else 0})
                return answer
            except (APITimeoutError, RateLimitError) as e:
                print(f"[ERRO] Falha na chamada ao modelo: {e}")
                return "(erro: servidor temporariamente indisponivel)"
            except APIError as e:
                print(f"[ERRO] Falha na chamada ao modelo: {e}")
                return f"(erro na chamada: {str(e)})"
            except Exception as e:
                print(f"[ERRO] Falha na chamada ao modelo: {e}")
                return "(erro inesperado)"
    else:
        # Modelo respondeu diretamente sem tools
        return choice.message.content or "(resposta vazia)"


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------


def demo_single_tool() -> None:
    """Demonstra chamada com uma unica tool (weather)."""
    print("  Pergunta: 'Qual a previsao do tempo em Sao Paulo?'")
    result = agent_with_tools("Qual a previsao do tempo em Sao Paulo?")
    print(f"  Resposta: {result[:300]}\n")


def demo_multi_tool() -> None:
    """Demonstra chamada que pode acionar multiplas tools."""
    print("  Pergunta: 'Busque sobre tracing no MLflow e me diga a temperatura em Curitiba'")
    result = agent_with_tools(
        "Busque sobre tracing no MLflow e me diga a temperatura em Curitiba"
    )
    print(f"  Resposta: {result[:300]}\n")


def demo_no_tool() -> None:
    """Demonstra quando o modelo decide NAO usar tools."""
    print("  Pergunta: 'O que voce acha de inteligencia artificial?'")
    result = agent_with_tools("O que voce acha de inteligencia artificial?")
    print(f"  Resposta: {result[:300]}\n")


def demo_calculation() -> None:
    """Demonstra tool de calculo."""
    print("  Pergunta: 'Quanto e 1500 * 12 + 350?'")
    result = agent_with_tools("Quanto e 1500 * 12 + 350?")
    print(f"  Resposta: {result[:300]}\n")


def demo_tool_failure() -> None:
    """Demonstra falha de tool (timeout) com erro registrado no span."""
    print("  Pergunta: 'Verifique o estoque do produto Notebook Dell XPS'")
    result = agent_with_tools("Verifique o estoque do produto Notebook Dell XPS")
    print(f"  Resposta: {result[:300]}\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Executa todas as demos de tool calling com observabilidade."""
    apply_patches()
    setup_mlflow("09-tool-calls")
    mlflow.openai.autolog()

    print("=" * 60)
    print("DEMO 1: Single tool call (weather)")
    print("=" * 60)
    demo_single_tool()

    print("\n" + "=" * 60)
    print("DEMO 2: Multi-tool call (search + weather)")
    print("=" * 60)
    demo_multi_tool()

    print("\n" + "=" * 60)
    print("DEMO 3: Sem tool call (resposta direta)")
    print("=" * 60)
    demo_no_tool()

    print("\n" + "=" * 60)
    print("DEMO 4: Tool de calculo")
    print("=" * 60)
    demo_calculation()

    print("\n" + "=" * 60)
    print("DEMO 5: Falha de tool (timeout simulado)")
    print("=" * 60)
    demo_tool_failure()

    # ────────────────────────────────────────────────────
    #  ✅ RESUMO DO QUE APRENDEMOS NESTE MÓDULO
    # ────────────────────────────────────────────────────
    #  ✔ SpanType.TOOL faz cada ferramenta aparecer na
    #     aba "Tool Calls" do MLflow UI.
    #  ✔ O loop manual: AGENT → CHAT_MODEL → TOOL(s) →
    #     CHAT_MODEL → resposta final.
    #  ✔ tool.latency_ms e tool.error como atributos
    #     para monitorar performance das tools.
    #  ✔ Falhas de tool não quebram o trace — o erro
    #     fica registrado no span com status ERROR.
    #  ✔ Compare com módulo 11 (LangChain) para ver
    #     a diferença entre manual e automático.
    #
    #  🔍 MLflow UI → Experiment '09-tool-calls': aba
    #     "Tool calls" e trace tree com spans TOOL.
    #
    #  💡 EXERCÍCIO: Adicione uma nova tool "send_email"
    #     e veja o span TOOL extra aparecer no trace.
    # ────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("✅ MÓDULO CONCLUÍDO! Resumo do aprendizado acima.")
    print("  Experiment: 09-tool-calls no MLflow UI")
    print("-" * 60)


if __name__ == "__main__":
    main()
