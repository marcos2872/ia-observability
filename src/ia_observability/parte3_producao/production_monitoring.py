"""
[Parte 3 — Produção] Módulo 07: Monitoramento em Produção
============================================================
╔══════════════════════════════════════════════════════════╗
║  OBJETIVOS DE APRENDIZADO                               ║
║  • Configurar tracing assíncrono (não bloquear a app)   ║
║  • Controlar volume/custo com sampling por criticidade  ║
║  • Coletar feedback humano (thumbs, scores)             ║
║  • Operar observabilidade em escala sem quebrar         ║
╚══════════════════════════════════════════════════════════╝

CONCEITO-CHAVE:
  Em produção, tracear 100% é caro e desnecessário. Use
  sampling: 100% para operações críticas (pagamentos),
  10% para alto volume (chats). Feedback humano fecha o
  ciclo: "o que o usuário achou?" vira dado de melhoria.

PRÉ-REQUISITOS:  Parte 1, Módulo 11 (langchain_agent)
DIFICULDADE:     🔴 Avançado
TEMPO ESTIMADO:  20 min

--- Como usar ---
  uv run monitoring    ou    make monitoring
"""

import os
import uuid

import mlflow
from mlflow.entities import AssessmentSource

from ia_observability.langchain_agent import agent_invoke, build_agent
from ia_observability.config import setup_mlflow


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
#
# Com autolog do LangChain, os spans (AGENT/CHAT_MODEL/TOOL) sao gerados
# automaticamente. Para controlar sampling por criticidade, envolvemos a
# chamada do agente numa funcao decorada com @mlflow.trace e definimos
# sampling_ratio_override. O trace raiz (e seus filhos via autolog) segue
# a taxa do override.
# ---------------------------------------------------------------------------


@mlflow.trace(sampling_ratio_override=1.0)
def critical_agent_call(agent, query: str, user_id: str, session_id: str) -> str:
    """Operacao critica - SEMPRE traced (100% sampling).

    Use sampling_ratio_override=1.0 para operacoes que precisam
    ser auditadas em 100% dos casos (ex: pagamentos, compliance).
    """
    return agent_invoke(agent, query, user_id, session_id)


@mlflow.trace(sampling_ratio_override=0.1)
def high_volume_agent_call(agent, query: str, user_id: str, session_id: str) -> str:
    """Operacao de alto volume - sampling reduzido (10%).

    Para endpoints de alto trafego, capture apenas uma amostra.
    Isso reduz custo de storage e processamento significativamente.
    """
    return agent_invoke(agent, query, user_id, session_id)


# ---------------------------------------------------------------------------
# Feedback collection
# ---------------------------------------------------------------------------


def demo_feedback_collection(agent) -> None:
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
    session_id = f"feedback-{uuid.uuid4().hex[:8]}"

    # Gera uma resposta trackeada via agente LangChain (100% sampling)
    answer = critical_agent_call(
        agent, "O que e MLflow tracing?", "reviewer-marcos", session_id
    )
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
    """Executa demos de configuracao de producao sobre um agente LangChain."""
    setup_mlflow("07-production-monitoring")
    mlflow.langchain.autolog()

    # Reutiliza o agente do modulo 11 (tools + sessions)
    agent = build_agent()

    print("=" * 60)
    print("CONFIGURACOES DE PRODUCAO")
    print("=" * 60)
    show_production_config()

    print("\n" + "=" * 60)
    print("DEMO 1: Operacao critica (100% sampling)")
    print("=" * 60)
    result = critical_agent_call(
        agent,
        "Explique observabilidade de LLM em 1 frase.",
        "demo-user",
        f"critical-{uuid.uuid4().hex[:8]}",
    )
    print(f"  Resposta: {result[:120]}...")

    print("\n" + "=" * 60)
    print("DEMO 2: Operacoes de alto volume (10% sampling)")
    print("=" * 60)
    total = 10
    for i in range(total):
        high_volume_agent_call(
            agent,
            f"Pergunta de alto volume #{i+1}: O que e MLOps?",
            "demo-user",
            f"high-volume-{uuid.uuid4().hex[:8]}",
        )
    print(f"  {total} chamadas feitas (apenas ~10% serao traced)")

    # Verifica quantas foram realmente traced
    traces = mlflow.search_traces(max_results=20)
    print(f"  Traces capturados no total (experiment): {len(traces)}")

    print("\n" + "=" * 60)
    print("DEMO 3: Coleta de feedback humano")
    print("=" * 60)
    demo_feedback_collection(agent)

    # ────────────────────────────────────────────────────
    #  ✅ RESUMO DO QUE APRENDEMOS NESTE MÓDULO
    # ────────────────────────────────────────────────────
    #  ✔ Async logging: MLFLOW_ENABLE_ASYNC_TRACE_LOGGING
    #     para não bloquear a aplicação.
    #  ✔ Sampling: sampling_ratio_override=1.0 (100%)
    #     para crítico, 0.1 (10%) para alto volume.
    #  ✔ Feedback humano: mlflow.log_feedback() vincula
    #     thumbs/scores/comentários ao trace.
    #  ✔ AssessmentSource distingue review humano vs
    #     avaliação automática.
    #
    #  🔍 MLflow UI → Experiment '07-production-monitoring':
    #     veja os feedbacks anexados aos traces.
    #
    #  💡 EXERCÍCIO: Crie 3 níveis de sampling
    #     (critical=1.0, normal=0.5, bulk=0.05) e
    #     verifique quantos traces foram capturados.
    # ────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("✅ MÓDULO CONCLUÍDO! Resumo do aprendizado acima.")
    print("  Experiment: 07-production-monitoring no MLflow UI")
    print("-" * 60)


if __name__ == "__main__":
    main()
