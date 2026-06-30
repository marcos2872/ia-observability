"""Demonstracao de version tracking com LoggedModel.

LoggedModel permite:
- Versionar combinacoes de codigo + configuracao + prompts
- Vincular traces automaticamente a versoes da aplicacao
- Comparar performance entre versoes no MLflow UI

Cenario: mesma aplicacao com diferentes prompts e temperaturas,
avaliando qual configuracao produz melhores resultados.

Referencia: https://mlflow.org/docs/latest/genai/version-tracking/
"""

import mlflow

from ia_observability.config import MODEL_NAME, get_client, setup_mlflow


def run_version(version_name: str, system_prompt: str, temperature: float) -> None:
    """Executa uma versao da aplicacao e vincula traces ao LoggedModel.

    Todos os traces gerados dentro do context manager set_active_model
    sao automaticamente associados a versao especificada.

    Args:
        version_name: Identificador da versao (ex: 'app-v1-generic').
        system_prompt: Prompt de sistema para o LLM.
        temperature: Temperatura de geracao.
    """
    client = get_client()

    questions = [
        "O que e MLflow?",
        "Como monitorar um LLM em producao?",
        "O que e evaluation-driven development?",
    ]

    # set_active_model vincula todos os traces gerados neste bloco a uma
    # versao nomeada (LoggedModel). No MLflow UI, esses traces aparecem
    # agrupados sob o modelo, permitindo comparar versoes lado a lado.
    with mlflow.set_active_model(name=version_name):
        for q in questions:
            # Envolvemos a chamada num span para ter um trace ATIVO — assim
            # podemos aplicar tags da versao via update_current_trace (a
            # chamada do autolog vira um span filho deste).
            with mlflow.start_span(name="ask"):
                mlflow.update_current_trace(
                    tags={
                        "app.version": version_name,
                        "app.temperature": str(temperature),
                        "app.model": MODEL_NAME,
                    },
                )
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": q},
                    ],
                    temperature=temperature,
                )
                answer = response.choices[0].message.content

            print(f"    Q: {q}")
            print(f"    A: {answer[:100]}...\n")


def main() -> None:
    """Executa multiplas versoes para comparacao no MLflow UI."""
    setup_mlflow("06-version-tracking")
    mlflow.openai.autolog()

    versions = [
        {
            "name": "app-v1-generic",
            "system_prompt": "Voce e um assistente util e amigavel.",
            "temperature": 0.7,
        },
        {
            "name": "app-v2-specialist",
            "system_prompt": (
                "Voce e um especialista senior em MLOps e observabilidade de IA. "
                "Seja conciso, tecnico e preciso. Use terminologia correta."
            ),
            "temperature": 0.3,
        },
        {
            "name": "app-v3-teacher",
            "system_prompt": (
                "Voce e um professor de engenharia de ML. "
                "Explique conceitos de forma didatica, use analogias "
                "e sempre de exemplos praticos."
            ),
            "temperature": 0.5,
        },
    ]

    for version in versions:
        print("=" * 60)
        print(f"VERSAO: {version['name']}")
        print(f"  Prompt: {version['system_prompt'][:60]}...")
        print(f"  Temperature: {version['temperature']}")
        print("=" * 60)
        run_version(version["name"], version["system_prompt"], version["temperature"])

    print("\n" + "-" * 60)
    print("No MLflow UI -> Experiment '06-version-tracking':")
    print("  - Cada versao aparece como um LoggedModel")
    print("  - Traces estao vinculados a versao que os gerou")
    print("  - Use a aba 'Models' para comparar versoes")
    print("-" * 60)


if __name__ == "__main__":
    main()
