"""Demonstracao de sessions multi-turn e tracking de usuarios.

Sessions permitem agrupar traces de uma mesma conversa/interacao,
facilitando a analise de fluxos conversacionais e comportamento do usuario.

Funcionalidades demonstradas:
- Associar traces a user_id e session_id
- Conversa multi-turn com historico
- Query de traces por sessao/usuario

Nota didatica: aqui o historico e a sessao sao gerenciados MANUALMENTE
(voce controla a lista de mensagens e o session_id). O modulo
`langchain_agent.py` mostra a versao AUTOMATICA equivalente, onde o
LangChain + MemorySaver cuidam disso.

Referencia: https://mlflow.org/docs/latest/genai/tracing/track-users-sessions/
"""

import uuid

import mlflow

from ia_observability.config import MODEL_NAME, get_client, setup_mlflow


@mlflow.trace
def chat_turn(messages: list[dict], user_id: str, session_id: str) -> str:
    """Processa um turno de conversa com contexto de sessao.

    Cada turno gera um trace vinculado ao user e session,
    permitindo reconstruir a conversa completa no MLflow UI.

    Args:
        messages: Historico completo da conversa (formato OpenAI).
        user_id: Identificador do usuario.
        session_id: Identificador da sessao.

    Returns:
        Resposta do assistente.
    """
    # Vincula o trace ao usuario e sessao
    mlflow.update_current_trace(session_id=session_id, user=user_id)

    client = get_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )
    return response.choices[0].message.content


def demo_multi_turn_session() -> None:
    """Simula uma conversa multi-turn com session tracking.

    Cenario: usuario perguntando sobre observabilidade de LLMs
    em multiplos turnos, onde cada resposta depende do historico.
    """
    user_id = "user-demo-001"
    session_id = f"session-{uuid.uuid4().hex[:8]}"

    print(f"  User ID:    {user_id}")
    print(f"  Session ID: {session_id}\n")

    conversation: list[dict] = [
        {"role": "system", "content": "Voce e um especialista em MLOps e observabilidade de IA."},
    ]

    user_messages = [
        "O que e MLflow?",
        "Como ele ajuda com observabilidade de LLMs?",
        "Quais metricas ele captura automaticamente?",
    ]

    for i, msg in enumerate(user_messages, 1):
        conversation.append({"role": "user", "content": msg})
        response = chat_turn(conversation, user_id, session_id)
        conversation.append({"role": "assistant", "content": response})

        print(f"  Turn {i}:")
        print(f"    User:      {msg}")
        print(f"    Assistant: {response[:120]}...")
        print()


def demo_multiple_users() -> None:
    """Simula multiplos usuarios com sessoes independentes.

    Demonstra como o MLflow separa traces por usuario/sessao,
    util para analise de comportamento e debugging por usuario.
    """
    users = [
        {"user_id": "alice-dev", "question": "Como configurar tracing em FastAPI?"},
        {"user_id": "bob-ops", "question": "Como monitorar custo de tokens em producao?"},
        {"user_id": "carol-ml", "question": "Como avaliar qualidade de um RAG?"},
    ]

    for user_data in users:
        user_id = user_data["user_id"]
        session_id = f"session-{uuid.uuid4().hex[:8]}"

        messages = [
            {"role": "system", "content": "Responda de forma concisa."},
            {"role": "user", "content": user_data["question"]},
        ]

        response = chat_turn(messages, user_id, session_id)
        print(f"  [{user_id}] Q: {user_data['question']}")
        print(f"  [{user_id}] A: {response[:100]}...\n")


def demo_query_by_session() -> None:
    """Demonstra como buscar traces por sessao e usuario via API.

    Em producao, isso e util para:
    - Debugging de problemas reportados por usuarios especificos
    - Analise de fluxos conversacionais
    - Metricas por usuario (latencia media, tokens consumidos, etc)
    """
    print("  Buscando traces por usuario 'user-demo-001'...")
    user_traces = mlflow.search_traces(
        filter_string="metadata.`mlflow.trace.user` = 'user-demo-001'",
        max_results=10,
    )
    print(f"  Encontrados: {len(user_traces)} traces\n")

    if not user_traces.empty:
        print(f"  {'Trace ID':<12} | {'Session':<20} | {'Status'}")
        print(f"  {'-'*12} | {'-'*20} | {'-'*8}")
        for _, row in user_traces.iterrows():
            trace_id = row["trace_id"][:10]
            session = row.get("metadata.mlflow.trace.session", "N/A")
            if session and len(str(session)) > 18:
                session = str(session)[:18]
            status = row.get("status", "?")
            print(f"  {trace_id:<12} | {session:<20} | {status}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Executa todas as demos de sessions."""
    setup_mlflow("03-sessions")
    mlflow.openai.autolog()

    print("=" * 60)
    print("DEMO 1: Conversa multi-turn com session tracking")
    print("=" * 60)
    demo_multi_turn_session()

    print("\n" + "=" * 60)
    print("DEMO 2: Multiplos usuarios com sessoes independentes")
    print("=" * 60)
    demo_multiple_users()

    print("\n" + "=" * 60)
    print("DEMO 3: Query de traces por sessao/usuario")
    print("=" * 60)
    demo_query_by_session()

    print("\n" + "-" * 60)
    print("No MLflow UI, use o filtro de sessao para agrupar traces:")
    print("  metadata.`mlflow.trace.session` = '<session-id>'")
    print("-" * 60)


if __name__ == "__main__":
    main()
