"""
[Parte 3 — Produção] Módulo 03: Sessions e User Tracking
==========================================================
╔══════════════════════════════════════════════════════════╗
║  OBJETIVOS DE APRENDIZADO                               ║
║  • Entender por que sessions são importantes em          ║
║    aplicações multi-turno (chat, suporte)                ║
║  • Vincular session_id e user_id aos traces do MLflow   ║
║  • Buscar traces por usuário/sessão                      ║
║  • Comparar abordagem manual vs LangChain (módulo 11)    ║
╚══════════════════════════════════════════════════════════╝

CONCEITO-CHAVE:
  Em produção, um usuário faz múltiplas perguntas em uma
  conversa. Cada pergunta gera um trace. Com session_id
  você agrupa todos os traces de uma conversa; com user_id
  você rastreia um usuário específico. Essencial para
  auditoria e suporte.

PRÉ-REQUISITOS:  Parte 1 (tracing + tokens)
DIFICULDADE:     🟡 Médio
TEMPO ESTIMADO:  15 min

--- Como usar ---
  uv run sessions    ou    make sessions

Referência: https://mlflow.org/docs/latest/genai/tracing/quickstart/
"""

import uuid

import mlflow
from openai import APIError, APITimeoutError, RateLimitError

from ia_observability.config import MODEL_NAME, apply_patches, get_client, setup_mlflow


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
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )
        return response.choices[0].message.content or "(resposta vazia)"
    except (APITimeoutError, RateLimitError) as e:
        print(f"[ERRO] Falha na chamada ao modelo: {e}")
        return "(erro: servidor temporariamente indisponivel)"
    except APIError as e:
        print(f"[ERRO] Falha na chamada ao modelo: {e}")
        return f"(erro na chamada: {str(e)})"
    except Exception as e:
        print(f"[ERRO] Falha na chamada ao modelo: {e}")
        return "(erro inesperado)"


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
    apply_patches()
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

    # ────────────────────────────────────────────────────
    #  ✅ RESUMO DO QUE APRENDEMOS NESTE MÓDULO
    # ────────────────────────────────────────────────────
    #  ✔ session_id agrupa traces de uma mesma conversa.
    #  ✔ user_id identifica quem fez a requisição.
    #  ✔ mlflow.update_current_trace() vincula ambos
    #     ao trace ativo.
    #  ✔ mlflow.search_traces() com filtro por usuário
    #     ou sessão para debugging.
    #  ✔ Abordagem MANUAL: você gerencia histórico e IDs.
    #    Compare com o módulo 11 (LangChain automático).
    #
    #  🔍 MLflow UI → Experiment '03-sessions': filtre
    #     por metadata.mlflow.trace.session = '<id>'.
    #
    #  💡 EXERCÍCIO: Implemente um chat persistente que
    #     salva o histórico em arquivo JSON entre execuções.
    # ────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("✅ MÓDULO CONCLUÍDO! Resumo do aprendizado acima.")
    print("  Experiment: 03-sessions no MLflow UI")
    print("-" * 60)


if __name__ == "__main__":
    main()
