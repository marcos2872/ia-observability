"""
[Parte 3 — Produção] Módulo 06: Version Tracking com LoggedModel
==================================================================
╔══════════════════════════════════════════════════════════╗
║  OBJETIVOS DE APRENDIZADO                               ║
║  • Entender o conceito de versionamento de modelos      ║
║  • Usar LoggedModel para registrar versões com metadados║
║  • Comparar desempenho entre versões                     ║
║  • Vincular traces à versão do modelo que os gerou       ║
╚══════════════════════════════════════════════════════════╝

CONCEITO-CHAVE:
  Você vai alterar prompts, modelos e parâmetros. Sem
  versionamento, você não sabe qual versão gerou qual
  resposta. LoggedModel registra cada versão com metadados
  (prompt, temperatura, modelo) e vincula aos traces.

PRÉ-REQUISITOS:  Parte 1, Módulo 04 (evaluation)
DIFICULDADE:     🟡 Médio
TEMPO ESTIMADO:  15 min

--- Como usar ---
  uv run versioning    ou    make versioning
"""

import mlflow
from openai import APIError, APITimeoutError, RateLimitError

from ia_observability.config import MODEL_NAME, apply_patches, get_client, setup_mlflow


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
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": q},
                        ],
                        temperature=temperature,
                    )
                    answer = response.choices[0].message.content or "(resposta vazia)"
                except (APITimeoutError, RateLimitError) as e:
                    print(f"[ERRO] Falha na chamada ao modelo: {e}")
                    answer = "(erro: servidor temporariamente indisponivel)"
                except APIError as e:
                    print(f"[ERRO] Falha na chamada ao modelo: {e}")
                    answer = f"(erro na chamada: {str(e)})"
                except Exception as e:
                    print(f"[ERRO] Falha na chamada ao modelo: {e}")
                    answer = "(erro inesperado)"

            print(f"    Q: {q}")
            print(f"    A: {answer[:100]}...\n")


def main() -> None:
    """Executa multiplas versoes para comparacao no MLflow UI."""
    apply_patches()
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

    # ────────────────────────────────────────────────────
    #  ✅ RESUMO DO QUE APRENDEMOS NESTE MÓDULO
    # ────────────────────────────────────────────────────
    #  ✔ LoggedModel registra versões do modelo com
    #     metadados (prompt, temperatura, parâmetros).
    #  ✔ Cada versão tem um run_id único e vinculável
    #     aos traces gerados.
    #  ✔ mlflow.search_runs() para comparar métricas
    #     entre versões.
    #  ✔ Essencial para responder "qual versão gerou
    #     esta resposta?"
    #
    #  🔍 MLflow UI → Experiment '06-version-tracking':
    #     compare métricas entre versões do modelo.
    #
    #  💡 EXERCÍCIO: Crie 3 versões com prompts diferentes
    #     e compare as métricas de avaliação (reuse o
    #     dataset do módulo 04).
    # ────────────────────────────────────────────────────


if __name__ == "__main__":
    main()
