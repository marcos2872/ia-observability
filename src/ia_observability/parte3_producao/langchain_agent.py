"""
[Parte 3 — Produção] Módulo 11: Agente LangChain com Tracing Automático
==========================================================================
╔══════════════════════════════════════════════════════════╗
║  OBJETIVOS DE APRENDIZADO                               ║
║  • Usar mlflow.langchain.autolog() para tracing 100%    ║
║    automático de agentes LangChain                       ║
║  • Ver spans AGENT, CHAT_MODEL e TOOL no trace          ║
║  • Combinar tool calling com sessions multi-turn        ║
║  • Identificar gargalos (tools lentas) nos spans        ║
╚══════════════════════════════════════════════════════════╝

CONCEITO-CHAVE:
  Com LangChain, o tracing é automático: uma linha
  (mlflow.langchain.autolog()) captura todo o ciclo
  ReAct (Reason + Act) do agente. Compare com os módulos
  03 (sessions manual) e 09 (tool calls manual) para
  entender a diferença de esforço.

PRÉ-REQUISITOS:  Módulos 03 e 09 (sessions + tool calls)
DIFICULDADE:     🟡 Médio
TEMPO ESTIMADO:  20 min

--- Como usar ---
  uv run langchain-agent    ou    make langchain-agent
"""

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator

import mlflow
from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer
from mlflow.entities import SpanType

from ia_observability.config import MLFLOW_GATEWAY_URL, MODEL_NAME, setup_mlflow

# ---------------------------------------------------------------------------
# Event helpers (mesmo padrao do production keepee-rag)
# ---------------------------------------------------------------------------


def make_event(event_type: str, content: str) -> str:
    """Serializa um evento de streaming para JSON (newline-delimited).

    Tipos usados neste demo:
    - "text_chunk" → token de resposta final do modelo (chunk a chunk)
    - "tool"       → log de progresso emitido por uma ferramenta via get_stream_writer()
    - "done"       → sinaliza fim do stream com trace_id no content
    """
    return (
        json.dumps({"type": event_type, "content": content}, ensure_ascii=False) + "\n"
    )


# ---------------------------------------------------------------------------
# Tools (definidas via @tool decorator do LangChain)
#
# Usam get_stream_writer() do langgraph.config para emitir logs de progresso
# que sao capturados via stream_mode="custom" no astream.
# ---------------------------------------------------------------------------


@tool
def get_weather(city: str, unit: str = "celsius") -> dict:
    """Retorna a previsao do tempo para uma cidade.

    Args:
        city: Nome da cidade.
        unit: Unidade de temperatura (celsius ou fahrenheit).
    """
    writer = get_stream_writer()
    writer(f"Consultando previsao do tempo para {city}...")

    data = {
        "Sao Paulo": {"temp": 22, "condition": "Parcialmente nublado"},
        "Rio de Janeiro": {"temp": 28, "condition": "Ensolarado"},
        "Curitiba": {"temp": 15, "condition": "Chuva leve"},
    }
    weather = data.get(city, {"temp": 20, "condition": "Desconhecido"})
    result = {
        "city": city,
        "temperature": weather["temp"],
        "unit": unit,
        "condition": weather["condition"],
    }
    writer(
        f"Previsao para {city}: {weather['temp']}°{unit[0].upper()}, {weather['condition']}"
    )
    return result


@tool
def search_docs(query: str, max_results: int = 3) -> list[dict]:
    """Busca na base de conhecimento interna sobre MLflow.

    Args:
        query: Termo de busca.
        max_results: Maximo de resultados a retornar.
    """
    writer = get_stream_writer()
    writer(f"Buscando documentos sobre '{query}'...")

    docs = [
        {
            "title": "MLflow Tracing Quickstart",
            "snippet": "Auto-tracing captura chamadas automaticamente...",
        },
        {
            "title": "Token Usage Tracking",
            "snippet": "MLflow rastreia input/output tokens por span...",
        },
        {
            "title": "Evaluation com Scorers",
            "snippet": "Use mlflow.genai.evaluate() com scorers built-in...",
        },
        {
            "title": "Tool Calling Observability",
            "snippet": "SpanType.TOOL permite rastrear execucao de tools...",
        },
    ]
    filtered = [
        d
        for d in docs
        if query.lower() in d["title"].lower() or query.lower() in d["snippet"].lower()
    ]
    results = filtered[:max_results] if filtered else docs[:max_results]
    writer(f"Encontrados {len(results)} resultados para '{query}'")
    return results


@tool
def calculate(expression: str) -> dict:
    """Executa um calculo matematico simples.

    Args:
        expression: Expressao matematica (ex: '1500 * 12 + 350').
    """
    writer = get_stream_writer()
    writer(f"Calculando: {expression}...")

    allowed = set("0123456789+-*/.(). ")
    if not all(c in allowed for c in expression):
        return {"error": "Expressao invalida", "expression": expression}
    try:
        result = eval(expression)  # noqa: S307 - apenas operacoes numericas
        writer(f"Resultado: {result}")
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e), "expression": expression}


@tool
def check_inventory(product: str) -> str:
    """Consulta o estoque disponivel de um produto.

    Args:
        product: Nome do produto a consultar.
    """
    writer = get_stream_writer()
    writer(f"Consultando estoque do produto '{product}'...")
    # Simula latencia alta seguida de falha
    time.sleep(2.5)
    writer(f"Timeout — API de estoque indisponivel para '{product}'")
    return (
        f"ERRO: Timeout ao consultar estoque do produto '{product}' - API indisponivel"
    )


# Lista de todas as tools disponiveis
ALL_TOOLS = [get_weather, search_docs, calculate, check_inventory]


# ---------------------------------------------------------------------------
# Session store (dict in-memory, sem MemorySaver)
#
# Igual ao production keepee-rag: o historico e gerenciado manualmente,
# nao via checkpointer do LangGraph. Apos cada turno, reconstruimos as
# mensagens a partir dos chunks brutos e as armazenamos aqui.
# ---------------------------------------------------------------------------

_sessions: dict[str, list[BaseMessage]] = {}
_SYSTEM_PROMPT = (
    "Voce e um assistente que usa tools quando necessario. "
    "Responda em portugues de forma concisa."
)


# ---------------------------------------------------------------------------
# Construcao do agente
# ---------------------------------------------------------------------------


def build_agent():
    """Constroi o agente LangChain com tools (sem checkpointer).

    Usa create_agent do langchain.agents — exatamente como no production
    keepee-rag. O system prompt e passado como SystemMessage na lista de
    mensagens, nao como parametro do create_agent.

    Returns:
        Agente LangChain (sem checkpointer para sessoes).
    """
    llm = ChatOpenAI(
        base_url=MLFLOW_GATEWAY_URL,
        api_key="not-needed",
        model=MODEL_NAME,
        temperature=0.1,
    )
    # Assinatura identica ao production: create_agent(llm, tools)
    agent = create_agent(llm, tools=ALL_TOOLS)
    return agent


# ---------------------------------------------------------------------------
# Reconstrucao de mensagens pos-stream (igual ao production)
# ---------------------------------------------------------------------------


def _collect_messages(
    raw_chunks: list,
    accumulated_response: str,
    tool_logs_by_id: dict[str, list[str]],
    trace_id: str | None = None,
) -> list[BaseMessage]:
    """Reconstroi mensagens completas a partir dos chunks brutos do streaming.

    Mesmo padrao do production keepee-rag ChatbotService._collect_messages().
    Concatena AIMessageChunks consecutivos e preserva ToolMessages.

    Returns:
        Lista de mensagens na ordem: [AIMessage (com tool_calls se houver),
        ToolMessage, ..., AIMessage (resposta final)].
    """
    new_messages: list[BaseMessage] = []
    current_ai: AIMessageChunk | None = None

    for chunk in raw_chunks:
        if isinstance(chunk, AIMessageChunk):
            if current_ai is not None and chunk.id != current_ai.id:
                new_messages.append(current_ai)
                current_ai = chunk
            elif current_ai is None:
                current_ai = chunk
            else:
                current_ai = current_ai + chunk

        elif isinstance(chunk, ToolMessage):
            if current_ai is not None:
                new_messages.append(current_ai)
                current_ai = None
            new_messages.append(chunk)

    # Ultimo AIMessage (resposta final)
    if current_ai is not None:
        new_messages.append(current_ai)

    return new_messages


# ---------------------------------------------------------------------------
# Funcao de invocacao com streaming + MLflow manual span
# ---------------------------------------------------------------------------


async def agent_invoke_stream(
    agent,
    query: str,
    user_id: str,
    session_id: str,
    provider: str = "openai",
    model_name: str | None = None,
) -> AsyncGenerator[str, None]:
    """Invoca o agente com streaming, span manual MLflow e eventos JSON.

    Fluxo identico ao production keepee-rag ChatbotService.get_response_stream():
    1. Monta input_messages = [SystemMessage, ...historico, HumanMessage(query)]
    2. Abre mlflow.start_span(span_type=SpanType.AGENT)
    3. Seta session_id e user no trace ANTES de comecar o stream
    4. Itera agent.astream(stream_mode=["messages", "custom"])
    5. Emite eventos "tool" (custom) e "text_chunk" (AIMessageChunk)
    6. Ao final, chama _collect_messages() e armazena historico
    7. Seta tags e outputs no span
    8. Emite evento "done" com o trace_id

    Yields:
        Eventos JSON newline-delimited (make_event).
    """
    if model_name is None:
        model_name = MODEL_NAME

    # Recupera historico da sessao (se existir)
    chat_history = _sessions.get(session_id, [])

    # Monta mensagens: SystemMessage + historico + pergunta atual
    input_messages: list[BaseMessage] = [
        SystemMessage(content=_SYSTEM_PROMPT),
        *chat_history,
        HumanMessage(content=query),
    ]

    raw_chunks: list = []
    accumulated_response: str = ""
    pending_logs: list[str] = []
    tool_logs_by_id: dict[str, list[str]] = {}

    trace_name = query.replace("\n", " ")[:50] or "chat-turn"

    with mlflow.start_span(name=trace_name, span_type=SpanType.AGENT) as trace_span:
        trace_id = trace_span.trace_id
        trace_span.set_inputs({"query": query})

        # Vincula user e session ao trace ANTES do stream (igual production)
        mlflow.update_current_trace(session_id=session_id, user=user_id)

        try:
            async for mode, data in agent.astream(
                {"messages": input_messages},
                config={
                    "recursion_limit": 40,
                    "metadata": {
                        "provider": provider,
                        "model_name": model_name,
                        "session_id": session_id,
                    },
                },
                stream_mode=["messages", "custom"],
            ):
                if mode == "custom":
                    # Log de progresso da tool via get_stream_writer()
                    pending_logs.append(str(data))
                    yield make_event("tool", str(data))
                    continue

                chunk, metadata = data
                if not metadata.get("langgraph_node"):
                    continue

                if isinstance(chunk, ToolMessage):
                    # Associa logs pendentes a este tool_call
                    if pending_logs:
                        tool_logs_by_id[chunk.tool_call_id] = pending_logs.copy()
                        pending_logs.clear()
                    raw_chunks.append(chunk)

                elif (
                    isinstance(chunk, AIMessageChunk)
                    and chunk.content
                    and not chunk.tool_call_chunks
                ):
                    text = _extract_text(chunk.content)
                    accumulated_response += text
                    raw_chunks.append(chunk)
                    yield make_event("text_chunk", text)
                else:
                    raw_chunks.append(chunk)

        finally:
            pass  # Garante que o span seja fechado mesmo em erro

        # Reconstroi mensagens do historico (pos-stream, igual production)
        new_messages = _collect_messages(
            raw_chunks, accumulated_response, tool_logs_by_id, trace_id=trace_id
        )
        _sessions[session_id] = chat_history + new_messages

        # Seta tags no trace apos o stream
        mlflow.update_current_trace(
            tags={
                "provider": provider,
                "model_name": model_name,
                "new_messages": str(len(new_messages)),
                "session_id": session_id,
            }
        )

        trace_span.set_outputs(
            {
                "response": accumulated_response[:500]
                if accumulated_response
                else None,
                "trace_id": trace_id,
            }
        )

    # Evento done com trace_id (igual production make_event("done", trace_id))
    yield make_event("done", trace_id or "")


def _extract_text(content) -> str:
    """Normaliza content para string pura.

    O LangChain pode retornar content como string simples ou como lista de
    blocos no formato multimodal: [{'type': 'text', 'text': '...', ...}].
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


# ---------------------------------------------------------------------------
# Helper para consumir o stream e exibir no console
# ---------------------------------------------------------------------------


async def _consume_stream(agen: AsyncGenerator[str, None]) -> str:
    """Consome o stream de eventos, exibe no console e retorna a resposta final.

    Exibe:
    - Logs de tool com prefixo [tool]
    - Tokens do modelo em tempo real (inline)
    - Done ao final com trace_id
    """
    response_parts: list[str] = []
    async for event_raw in agen:
        try:
            event = json.loads(event_raw.strip())
        except json.JSONDecodeError:
            continue

        if event["type"] == "tool":
            print(f"    [tool] {event['content']}")
        elif event["type"] == "text_chunk":
            print(event["content"], end="", flush=True)
            response_parts.append(event["content"])
        elif event["type"] == "done":
            trace_id = event["content"]
            print()
            if trace_id:
                print(f"    [trace_id] {trace_id}")

    return "".join(response_parts)


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------


async def demo_single_tool(agent) -> None:
    """Demonstra chamada com uma unica tool (weather) em streaming."""
    session_id = f"single-tool-{uuid.uuid4().hex[:8]}"
    print("  Pergunta: 'Qual a previsao do tempo em Sao Paulo?'")
    stream = agent_invoke_stream(
        agent, "Qual a previsao do tempo em Sao Paulo?", "demo-user", session_id
    )
    await _consume_stream(stream)
    print()


async def demo_multi_tool(agent) -> None:
    """Demonstra chamada que aciona multiplas tools em streaming."""
    session_id = f"multi-tool-{uuid.uuid4().hex[:8]}"
    print(
        "  Pergunta: 'Busque sobre tracing no MLflow e me diga a temperatura em Curitiba'"
    )
    stream = agent_invoke_stream(
        agent,
        "Busque sobre tracing no MLflow e me diga a temperatura em Curitiba",
        "demo-user",
        session_id,
    )
    await _consume_stream(stream)
    print()


async def demo_multi_turn_session(agent) -> None:
    """Demonstra conversa multi-turn com memoria de sessao.

    O historico e mantido em dict in-memory (_sessions), reconstruido a
    partir dos chunks do streaming — mesma abordagem do production
    keepee-rag (que persiste em banco SQL).
    """
    user_id = "alice-dev"
    session_id = f"multiturn-{uuid.uuid4().hex[:8]}"

    print(f"  User ID:    {user_id}")
    print(f"  Session ID: {session_id}\n")

    turns = [
        "Qual a temperatura em Sao Paulo?",
        "E no Rio de Janeiro? Compare com a cidade anterior.",
        "Agora busque na documentacao sobre tracing do MLflow.",
    ]

    for i, query in enumerate(turns, 1):
        print(f"  Turn {i}:")
        print(f"    User:      {query}")
        print(f"    Assistant: ", end="")
        stream = agent_invoke_stream(agent, query, user_id, session_id)
        await _consume_stream(stream)
        print()


async def demo_multiple_users(agent) -> None:
    """Demonstra multiplos usuarios com sessoes independentes em streaming."""
    users = [
        {"user_id": "alice-dev", "question": "Quanto e 2048 * 16 + 99?"},
        {"user_id": "bob-ops", "question": "Qual a previsao do tempo em Curitiba?"},
        {"user_id": "carol-ml", "question": "Busque sobre evaluation no MLflow."},
    ]

    for user_data in users:
        user_id = user_data["user_id"]
        session_id = f"{user_id}-{uuid.uuid4().hex[:8]}"
        print(f"  [{user_id}] Q: {user_data['question']}")
        print(f"  [{user_id}] A: ", end="")
        stream = agent_invoke_stream(agent, user_data["question"], user_id, session_id)
        await _consume_stream(stream)
        print()


async def demo_tool_failure(agent) -> None:
    """Demonstra falha de tool (timeout simulado) em streaming.

    A tool returna string de erro — o modelo recebe e informa o usuario.
    O stream_mode="custom" captura os logs de progresso (inclusive o timeout).
    """
    session_id = f"failure-{uuid.uuid4().hex[:8]}"
    print("  Pergunta: 'Verifique o estoque do produto Notebook Dell XPS'")
    stream = agent_invoke_stream(
        agent,
        "Verifique o estoque do produto Notebook Dell XPS",
        "demo-user",
        session_id,
    )
    await _consume_stream(stream)
    print()


async def demo_feedback(agent) -> None:
    """Demonstra registro de feedback humano em um trace existente.

    Usa o trace_id capturado no evento "done" para registrar avaliacao
    via mlflow.log_feedback() — mesmo padrao do feedback.py do production.
    """
    session_id = f"feedback-{uuid.uuid4().hex[:8]}"
    captured_trace_id: str | None = None

    print("  Pergunta: 'Qual a temperatura em Sao Paulo?'")
    stream = agent_invoke_stream(
        agent, "Qual a temperatura em Sao Paulo?", "feedback-user", session_id
    )

    async for event_raw in stream:
        try:
            event = json.loads(event_raw.strip())
        except json.JSONDecodeError:
            continue
        if event["type"] == "text_chunk":
            print(event["content"], end="", flush=True)
        elif event["type"] == "tool":
            print(f"\n    [tool] {event['content']}")
        elif event["type"] == "done" and event["content"]:
            captured_trace_id = event["content"]
            print(f"    [trace_id] {captured_trace_id}")

    if captured_trace_id:
        print(
            "\n  ---> Registrando feedback positivo para trace "
            f"{captured_trace_id[:16]}..."
        )
        try:
            mlflow.log_feedback(
                trace_id=captured_trace_id,
                name="user_satisfaction",
                value=True,
                source=mlflow.entities.AssessmentSource(
                    source_type="HUMAN",
                    source_id="feedback-user",
                ),
                rationale="Resposta correta e completa.",
            )
            print("  Feedback registrado com sucesso!")
        except Exception as e:
            print(f"  Erro ao registrar feedback: {e}")
    print()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Executa todas as demos de LangChain agent com streaming + MLflow."""
    setup_mlflow("11-langchain-agent")
    mlflow.langchain.autolog(log_traces=True)

    # Constroi o agente uma unica vez (reutilizado entre demos)
    agent = build_agent()

    print("=" * 60)
    print("DEMO 1: Single tool call (weather) via LangChain streaming")
    print("=" * 60)
    asyncio.run(demo_single_tool(agent))

    print("\n" + "=" * 60)
    print("DEMO 2: Multi-tool call (search + weather) streaming")
    print("=" * 60)
    asyncio.run(demo_multi_tool(agent))

    print("\n" + "=" * 60)
    print("DEMO 3: Conversa multi-turn com session memory (dict in-memory)")
    print("=" * 60)
    asyncio.run(demo_multi_turn_session(agent))

    print("\n" + "=" * 60)
    print("DEMO 4: Multiplos usuarios com sessoes independentes")
    print("=" * 60)
    asyncio.run(demo_multiple_users(agent))

    print("\n" + "=" * 60)
    print("DEMO 5: Falha de tool (timeout simulado)")
    print("=" * 60)
    asyncio.run(demo_tool_failure(agent))

    print("\n" + "=" * 60)
    print("DEMO 6: Feedback humano no trace (mlflow.log_feedback)")
    print("=" * 60)
    asyncio.run(demo_feedback(agent))

    # ────────────────────────────────────────────────────
    #  ✅ RESUMO DO QUE APRENDEMOS NESTE MÓDULO
    # ────────────────────────────────────────────────────
    #  ✔ mlflow.langchain.autolog(): 1 linha instrumenta
    #     TODO o agente LangChain automaticamente.
    #  ✔ Spans AGENT, CHAT_MODEL e TOOL são criados
    #     sem nenhum código de instrumentação manual.
    #  ✔ MemorySaver + thread_id mantém histórico da
    #     sessão multi-turn automaticamente.
    #  ✔ Tools lentas viram spans de alta latência —
    #     o gargalo fica visível.
    #  ✔ Compare com os módulos 03 (sessions manual) e
    #     09 (tool calls manual) para entender a diferença.
    #
    #  🔍 MLflow UI → Experiment '11-langchain-agent':
    #     trace tree completa do ciclo ReAct.
    #
    #  💡 EXERCÍCIO: Adicione uma nova tool que chama
    #     uma API externa real e veja a latência no span.
    # ────────────────────────────────────────────────────


if __name__ == "__main__":
    main()
