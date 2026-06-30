"""
[Parte 4 — Avançado] Módulo 10: Prompt Registry e Versionamento
==================================================================
╔══════════════════════════════════════════════════════════╗
║  OBJETIVOS DE APRENDIZADO                               ║
║  • Usar o Prompt Registry do MLflow para registrar      ║
║    e versionar prompts                                   ║
║  • Vincular prompts a traces (qual versão gerou isso?)  ║
║  • Atualizar prompts sem quebrar tracing de versões     ║
║    anteriores                                            ║
╚══════════════════════════════════════════════════════════╝

CONCEITO-CHAVE:
  Prompt não é código, mas muda com frequência. Sem registry,
  você não sabe qual versão do prompt gerou qual resposta.
  O Prompt Registry versiona cada prompt e vincula a versão
  usada em cada trace automaticamente.

PRÉ-REQUISITOS:  Parte 1, Módulo 06 (version_tracking)
DIFICULDADE:     🔴 Avançado
TEMPO ESTIMADO:  20 min

--- Como usar ---
  uv run prompts    ou    make prompts
"""

import mlflow

from ia_observability.config import (
    MODEL_NAME,
    get_client,
    setup_mlflow,
)

# ---------------------------------------------------------------------------
# Demo 1: Registrar prompts no registry
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_V1 = (
    "Voce e um assistente tecnico especializado em MLOps e observabilidade de IA. "
    "Responda de forma concisa em portugues."
)

USER_PROMPT_TEMPLATE = "Explique de forma clara: {{topic}}"

SYSTEM_PROMPT_V2 = (
    "Voce e um assistente tecnico especializado em MLOps e observabilidade de IA. "
    "Responda em portugues usando analogias do dia-a-dia para facilitar o entendimento. "
    "Limite a resposta a 3 frases."
)


def demo_register_prompts() -> None:
    """Registra prompts no MLflow Prompt Registry.

    register_prompt cria ou atualiza um prompt versionado.
    Cada chamada com o mesmo nome cria uma nova versao.
    """
    # Registra system prompt v1
    mlflow.genai.register_prompt(
        name="observability-system-v1",
        template=SYSTEM_PROMPT_V1,
    )
    print("  Registrado: observability-system-v1")

    # Registra user prompt template
    mlflow.genai.register_prompt(
        name="observability-question",
        template=USER_PROMPT_TEMPLATE,
    )
    print("  Registrado: observability-question (template com {{topic}})")

    # Registra system prompt v2 (versao com analogias)
    mlflow.genai.register_prompt(
        name="observability-system-v2",
        template=SYSTEM_PROMPT_V2,
    )
    print("  Registrado: observability-system-v2 (com analogias)")


# ---------------------------------------------------------------------------
# Demo 2: Carregar prompt e linkar ao trace
# ---------------------------------------------------------------------------


@mlflow.trace
def ask_with_prompt(
    topic: str, system_prompt_name: str = "observability-system-v2"
) -> str:
    """Carrega prompts do registry e usa na chamada ao LLM.

    O MLflow detecta automaticamente que load_prompt foi chamado dentro
    de uma funcao tracada e linka o prompt ao trace no UI.
    """
    # Carrega system prompt pelo nome
    system_prompt = mlflow.genai.load_prompt(f"prompts:/{system_prompt_name}@latest")

    # Carrega user prompt template e formata com a variavel
    user_prompt = mlflow.genai.load_prompt("prompts:/observability-question@latest")
    user_content = user_prompt.format(topic=topic)

    client = get_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt.template},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Demo 3: Comparar versoes de prompt
# ---------------------------------------------------------------------------


def demo_compare_versions(topic: str) -> None:
    """Executa o mesmo topic com v1 e v2 do system prompt.

    No MLflow UI, cada trace mostra qual versao do prompt foi usada,
    permitindo comparar qualidade das respostas lado a lado.
    """
    print(f"\n  Topic: '{topic}'")

    print("\n  --- System Prompt v1 (conciso) ---")
    result_v1 = ask_with_prompt(topic, system_prompt_name="observability-system-v1")
    print(f"  Resposta: {result_v1[:200]}...")

    print("\n  --- System Prompt v2 (analogias) ---")
    result_v2 = ask_with_prompt(topic, system_prompt_name="observability-system-v2")
    print(f"  Resposta: {result_v2[:200]}...")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Executa todas as demos de prompt management."""
    setup_mlflow("10-prompt-management")

    print("=" * 60)
    print("DEMO 1: Registrar prompts no MLflow Prompt Registry")
    print("=" * 60)
    demo_register_prompts()

    print("\n" + "=" * 60)
    print("DEMO 2: Carregar prompt linkado ao trace")
    print("=" * 60)
    result = ask_with_prompt("tracing em aplicacoes LLM")
    print(f"  Resposta: {result[:200]}...")

    print("\n" + "=" * 60)
    print("DEMO 3: Comparar versoes de prompt (v1 vs v2)")
    print("=" * 60)
    demo_compare_versions("observabilidade em sistemas de IA")

    # ────────────────────────────────────────────────────
    #  ✅ RESUMO DO QUE APRENDEMOS NESTE MÓDULO
    # ────────────────────────────────────────────────────
    #  ✔ Prompt Registry versiona prompts como "git
    #     para prompts": cada alteração gera nova versão.
    #  ✔ register_prompt(): registra um prompt com nome,
    #     template e metadados.
    #  ✔ load_prompt(): carrega versão específica ou
    #     @latest pelo URI "prompts:/nome/versão".
    #  ✔ Prompts vinculados a traces: o trace mostra
    #     qual versão do prompt gerou a resposta.
    #
    #  🔍 MLflow UI → Prompt Registry: veja as versões
    #     do prompt e seu histórico de alterações.
    #
    #  💡 EXERCÍCIO: Registre uma nova versão do prompt
    #     com um system message diferente e veja o
    #     versionamento no UI.
    # ────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("✅ MÓDULO CONCLUÍDO! Resumo do aprendizado acima.")
    print("  Experiment: 10-prompt-management no MLflow UI")
    print("-" * 60)


if __name__ == "__main__":
    main()
