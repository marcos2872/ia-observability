"""Demonstracao de Prompt Registry: registrar, versionar e linkar prompts a traces.

O MLflow Prompt Registry permite:
1. Registrar prompts com templates (variaveis {{nome}})
2. Versionar prompts (v1, v2, ...) para comparacao
3. Linkar prompts aos traces — no UI aparece qual prompt gerou qual resposta

Referencia: https://mlflow.org/docs/latest/genai/prompt-registry/
"""

import mlflow

from ia_observability.config import (
    AZURE_AVAILABLE,
    AZURE_MODEL,
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
# Demo 4: Otimizacao automatica de prompt com Azure OpenAI
# ---------------------------------------------------------------------------


def demo_optimize_prompt() -> None:
    """Otimiza prompt registrado usando GepaPromptOptimizer.

    Usa Azure OpenAI como reflection model para analisar resultados
    e gerar versoes melhoradas do prompt automaticamente.
    Requer AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT no .env.
    """
    if not AZURE_AVAILABLE:
        print("  [SKIP] Variaveis Azure OpenAI nao configuradas no .env")
        print(
            "  Configure: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT"
        )
        return

    import os

    from mlflow.genai.optimize import GepaPromptOptimizer
    from mlflow.genai.scorers import Correctness

    # MLflow scorer usa AZURE_API_KEY; GEPA/litellm usa AZURE_OPENAI_API_KEY
    # Garantir que ambos estejam disponiveis
    if not os.getenv("AZURE_API_KEY"):
        os.environ["AZURE_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]
    if not os.getenv("AZURE_API_BASE"):
        os.environ["AZURE_API_BASE"] = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    if not os.getenv("AZURE_API_VERSION"):
        os.environ["AZURE_API_VERSION"] = os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
        )

    # Registra um prompt propositalmente RUIM para a otimizacao melhorar
    BAD_PROMPT = (
        "Voce e um agente de IA geral e sabe muito de MLOps e observabilidade de IA."
    )
    mlflow.genai.register_prompt(
        name="observability-system-bad",
        template=BAD_PROMPT,
    )
    print("  Registrado prompt ruim: 'observability-system-bad'")

    # Dataset exigente — criterios tecnicos rigorosos de formato e conteudo
    train_data = [
        {
            "inputs": {"topic": "tracing em LLMs"},
            "expectations": {
                "expected_facts": [
                    "Tracing captura inputs e outputs de cada etapa de processamento",
                    "Cada span registra latencia e metadados da operacao",
                    "Ferramentas como MLflow e LangSmith oferecem tracing para LLMs",
                    "A resposta usa terminologia tecnica correta (spans, traces, instrumentation)",
                    "A resposta esta estruturada em topicos ou lista numerada",
                    "O tom e formal e tecnico, como um engenheiro de MLOps escreveria",
                    "A resposta esta em portugues brasileiro",
                    "A resposta NAO contem emojis ou caracteres decorativos",
                ]
            },
        },
        {
            "inputs": {"topic": "MLflow Prompt Registry"},
            "expectations": {
                "expected_facts": [
                    "Permite registrar e versionar prompts de forma centralizada",
                    "Prompts podem ser linkados a traces para rastreabilidade",
                    "Suporta templates com variaveis usando sintaxe de duplas chaves",
                    "A resposta menciona nomenclatura oficial do MLflow (PromptVersion, register_prompt, load_prompt)",
                    "A resposta segue formato tecnico com exemplos de uso ou codigo",
                    "O tom e profissional e objetivo, sem linguagem coloquial",
                    "A resposta esta em portugues brasileiro",
                    "A resposta NAO contem emojis ou caracteres decorativos",
                ]
            },
        },
        {
            "inputs": {"topic": "observabilidade em IA"},
            "expectations": {
                "expected_facts": [
                    "Monitorar qualidade de respostas em producao continuamente",
                    "Inclui metricas de latencia, token usage e custo por chamada",
                    "Permite detectar regressoes de qualidade ao longo do tempo",
                    "A resposta diferencia observabilidade de monitoramento tradicional",
                    "A resposta usa termos tecnicos corretos (SLI, SLO, drift, feedback loop)",
                    "A resposta esta organizada com paragrafos curtos ou bullets",
                    "O tom e de documentacao tecnica de engenharia",
                    "A resposta NAO contem emojis ou caracteres decorativos",
                ]
            },
        },
    ]

    def predict_fn(topic: str) -> str:
        prompt = mlflow.genai.load_prompt("prompts:/observability-system-bad@latest")
        client = get_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt.template},
                {"role": "user", "content": f"Explique: {topic}"},
            ],
        )
        return response.choices[0].message.content

    print("  Otimizando prompt ruim 'observability-system-bad'...")
    print(f"  Reflection model: {AZURE_MODEL}")
    print(
        f"  Dataset: {len(train_data)} exemplos ({sum(len(d['expectations']['expected_facts']) for d in train_data)} fatos)"
    )

    result = mlflow.genai.optimize_prompts(
        predict_fn=predict_fn,
        train_data=train_data,
        prompt_uris=["prompts:/observability-system-bad@latest"],
        optimizer=GepaPromptOptimizer(
            reflection_model=AZURE_MODEL, max_metric_calls=10
        ),
        scorers=[Correctness(model=AZURE_MODEL)],
    )

    print(f"\n  Prompt otimizado registrado: {result.optimized_prompts[0].uri}")
    optimized = mlflow.genai.load_prompt(result.optimized_prompts[0].uri)
    print(f"  Preview: {optimized.template[:200]}...")


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

    print("\n" + "=" * 60)
    print("DEMO 4: Otimizacao automatica de prompt (Azure OpenAI)")
    print("=" * 60)
    demo_optimize_prompt()

    print("\n" + "-" * 60)
    print("Abra o MLflow UI para visualizar:")
    print(f"  -> Experiment: 10-prompt-management")
    print("  - Traces linkados aos prompts registrados")
    print("  - Prompt Registry com versoes v1 e v2")
    print("  - Comparacao de respostas por versao")
    print("-" * 60)


if __name__ == "__main__":
    main()
