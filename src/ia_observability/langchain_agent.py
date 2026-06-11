"""Demonstracao de tool calling + sessions multi-turn via LangChain.

Combina funcionalidades de tool_calls.py e sessions.py em um unico modulo,
usando LangChain como orquestrador. O MLflow captura automaticamente todos
os spans via mlflow.langchain.autolog():

- AGENT span (AgentExecutor)
- CHAT_MODEL spans (chamadas ao LLM)
- TOOL spans (execucao de cada tool)
- Historico de mensagens por sessao (MemorySaver)
- Vinculo de user_id e session_id ao trace

Vantagens do LangChain vs implementacao manual:
- Tool calling loop gerenciado automaticamente (retry, parsing)
- Memory/session management built-in via checkpointer
- Tracing completo sem spans manuais
- Suporte a MemorySaver para conversas multi-turn

Referencia: https://mlflow.org/docs/latest/genai/tracing/integrations/langchain/
"""

import time
import uuid

import mlflow
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from ia_observability.config import MLFLOW_GATEWAY_URL, MODEL_NAME, setup_mlflow

# ---------------------------------------------------------------------------
# Tools (definidas via @tool decorator do LangChain)
# ---------------------------------------------------------------------------


@tool
def get_weather(city: str, unit: str = "celsius") -> dict:
    """Retorna a previsao do tempo para uma cidade.

    Args:
        city: Nome da cidade.
        unit: Unidade de temperatura (celsius ou fahrenheit).
    """
    data = {
        "Sao Paulo": {"temp": 22, "condition": "Parcialmente nublado"},
        "Rio de Janeiro": {"temp": 28, "condition": "Ensolarado"},
        "Curitiba": {"temp": 15, "condition": "Chuva leve"},
    }
    weather = data.get(city, {"temp": 20, "condition": "Desconhecido"})
    return {
        "city": city,
        "temperature": weather["temp"],
        "unit": unit,
        "condition": weather["condition"],
    }


@tool
def search_docs(query: str, max_results: int = 3) -> list[dict]:
    """Busca na base de conhecimento interna sobre MLflow.

    Args:
        query: Termo de busca.
        max_results: Maximo de resultados a retornar.
    """
    docs = [
        {"title": "MLflow Tracing Quickstart", "snippet": "Auto-tracing captura chamadas automaticamente..."},
        {"title": "Token Usage Tracking", "snippet": "MLflow rastreia input/output tokens por span..."},
        {"title": "Evaluation com Scorers", "snippet": "Use mlflow.genai.evaluate() com scorers built-in..."},
        {"title": "Tool Calling Observability", "snippet": "SpanType.TOOL permite rastrear execucao de tools..."},
    ]
    filtered = [
        d for d in docs
        if query.lower() in d["title"].lower() or query.lower() in d["snippet"].lower()
    ]
    return filtered[:max_results] if filtered else docs[:max_results]


@tool
def calculate(expression: str) -> dict:
    """Executa um calculo matematico simples.

    Args:
        expression: Expressao matematica (ex: '1500 * 12 + 350').
    """
    allowed = set("0123456789+-*/.(). ")
    if not all(c in allowed for c in expression):
        return {"error": "Expressao invalida", "expression": expression}
    try:
        result = eval(expression)  # noqa: S307 - apenas operacoes numericas
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e), "expression": expression}


@tool
def check_inventory(product: str) -> str:
    """Consulta o estoque disponivel de um produto.

    Args:
        product: Nome do produto a consultar.
    """
    # Simula latencia alta seguida de falha
    time.sleep(2.5)
    return f"ERRO: Timeout ao consultar estoque do produto '{product}' - API indisponivel"


# Lista de todas as tools disponiveis
ALL_TOOLS = [get_weather, search_docs, calculate, check_inventory]


# ---------------------------------------------------------------------------
# Session store (in-memory via MemorySaver)
# ---------------------------------------------------------------------------

_memory = MemorySaver()


# ---------------------------------------------------------------------------
# Construcao do agente
# ---------------------------------------------------------------------------


def build_agent():
    """Constroi o agente LangChain com tools e session memory.

    Usa create_agent do langchain.agents que implementa o loop ReAct
    (Reason + Act) automaticamente: LLM decide tool -> executa -> repete.
    O checkpointer MemorySaver mantem historico de mensagens por sessao.
    Tracing e 100% automatico via mlflow.langchain.autolog().

    Returns:
        Agente LangChain compilado com checkpointer para sessoes.
    """
    # LLM apontando para o MLflow AI Gateway
    llm = ChatOpenAI(
        base_url=MLFLOW_GATEWAY_URL,
        api_key="not-needed",
        model=MODEL_NAME,
        temperature=0.1,
    )

    # Cria agente ReAct com tools e memory (checkpointer para sessoes)
    agent = create_agent(
        model=llm,
        tools=ALL_TOOLS,
        checkpointer=_memory,
        system_prompt="Voce e um assistente que usa tools quando necessario. Responda em portugues de forma concisa.",
    )

    return agent


# ---------------------------------------------------------------------------
# Funcao de invocacao com user/session tracking no MLflow
# ---------------------------------------------------------------------------


def agent_invoke(
    agent,
    query: str,
    user_id: str,
    session_id: str,
) -> str:
    """Invoca o agente e vincula user_id/session_id ao trace MLflow.

    Args:
        agent: Agente LangChain compilado.
        query: Pergunta do usuario.
        user_id: Identificador do usuario.
        session_id: Identificador da sessao.

    Returns:
        Resposta do agente.
    """
    # thread_id no config = session_id (mantem historico entre turnos)
    config = {"configurable": {"thread_id": session_id}}

    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config,
    )

    # Vincula user e session ao trace mais recente
    try:
        mlflow.update_current_trace(session_id=session_id, user=user_id)
    except Exception:
        pass  # Pode falhar se trace ja finalizou

    # Extrai a ultima mensagem do assistente
    messages = result.get("messages", [])
    if messages:
        return messages[-1].content or "(resposta vazia)"
    return "(resposta vazia)"


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------


def demo_single_tool(agent) -> None:
    """Demonstra chamada com uma unica tool (weather)."""
    session_id = f"single-tool-{uuid.uuid4().hex[:8]}"
    print("  Pergunta: 'Qual a previsao do tempo em Sao Paulo?'")
    result = agent_invoke(agent, "Qual a previsao do tempo em Sao Paulo?", "demo-user", session_id)
    print(f"  Resposta: {result[:300]}\n")


def demo_multi_tool(agent) -> None:
    """Demonstra chamada que pode acionar multiplas tools."""
    session_id = f"multi-tool-{uuid.uuid4().hex[:8]}"
    print("  Pergunta: 'Busque sobre tracing no MLflow e me diga a temperatura em Curitiba'")
    result = agent_invoke(
        agent,
        "Busque sobre tracing no MLflow e me diga a temperatura em Curitiba",
        "demo-user",
        session_id,
    )
    print(f"  Resposta: {result[:300]}\n")


def demo_multi_turn_session(agent) -> None:
    """Demonstra conversa multi-turn com memoria de sessao.

    O LangChain mantem o historico automaticamente via MemorySaver,
    permitindo que o modelo referencie turnos anteriores.
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
        result = agent_invoke(agent, query, user_id, session_id)
        print(f"  Turn {i}:")
        print(f"    User:      {query}")
        print(f"    Assistant: {result[:150]}...")
        print()


def demo_multiple_users(agent) -> None:
    """Demonstra multiplos usuarios com sessoes independentes."""
    users = [
        {"user_id": "alice-dev", "question": "Quanto e 2048 * 16 + 99?"},
        {"user_id": "bob-ops", "question": "Qual a previsao do tempo em Curitiba?"},
        {"user_id": "carol-ml", "question": "Busque sobre evaluation no MLflow."},
    ]

    for user_data in users:
        user_id = user_data["user_id"]
        session_id = f"{user_id}-{uuid.uuid4().hex[:8]}"
        result = agent_invoke(agent, user_data["question"], user_id, session_id)
        print(f"  [{user_id}] Q: {user_data['question']}")
        print(f"  [{user_id}] A: {result[:120]}...\n")


def demo_tool_failure(agent) -> None:
    """Demonstra falha de tool (timeout simulado).

    A tool retorna string de erro — o modelo recebe e informa o usuario.
    """
    session_id = f"failure-{uuid.uuid4().hex[:8]}"
    print("  Pergunta: 'Verifique o estoque do produto Notebook Dell XPS'")
    result = agent_invoke(
        agent, "Verifique o estoque do produto Notebook Dell XPS", "demo-user", session_id
    )
    print(f"  Resposta: {result[:300]}\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Executa todas as demos de LangChain agent com tools + sessions."""
    setup_mlflow("11-langchain-agent")
    mlflow.langchain.autolog()

    # Constroi o agente uma unica vez (reutilizado entre demos)
    agent = build_agent()

    print("=" * 60)
    print("DEMO 1: Single tool call (weather) via LangChain")
    print("=" * 60)
    demo_single_tool(agent)

    print("\n" + "=" * 60)
    print("DEMO 2: Multi-tool call (search + weather)")
    print("=" * 60)
    demo_multi_tool(agent)

    print("\n" + "=" * 60)
    print("DEMO 3: Conversa multi-turn com session memory")
    print("=" * 60)
    demo_multi_turn_session(agent)

    print("\n" + "=" * 60)
    print("DEMO 4: Multiplos usuarios com sessoes independentes")
    print("=" * 60)
    demo_multiple_users(agent)

    print("\n" + "=" * 60)
    print("DEMO 5: Falha de tool (timeout simulado)")
    print("=" * 60)
    demo_tool_failure(agent)

    print("\n" + "-" * 60)
    print("Abra o MLflow UI -> Experiment '11-langchain-agent' para ver:")
    print("  - Traces completos gerados automaticamente pelo autolog")
    print("  - Spans: model > ChatOpenAI > Tool(s) > model > ChatOpenAI")
    print("  - Historico de sessao mantido entre turnos (Demo 3)")
    print("  - Traces filtraveis por user e session_id")
    print("-" * 60)


if __name__ == "__main__":
    main()
