"""
[Parte 2 — Avaliação] Módulo 12: Evaluation Datasets
======================================================
╔══════════════════════════════════════════════════════════╗
║  OBJETIVOS DE APRENDIZADO                               ║
║  • Entender o que são evaluation datasets no MLflow     ║
║  • Aprender a submeter datasets via SDK                 ║
║  • Buscar datasets salvos para reuso                    ║
║  • Requer backend SQL no tracking server                ║
╚══════════════════════════════════════════════════════════╝

CONCEITO-CHAVE:
  Datasets de avaliação são a "prova" do seu LLM: perguntas
  com respostas esperadas que você reutiliza entre versões
  do modelo/prompt para comparar qualidade. O MLflow permite
  versioná-los e vinculá-los a experiments.

PRÉ-REQUISITOS:  Módulos 04 e 05 (evaluation + judges)
DIFICULDADE:     🟡 Médio (requer backend SQL)
TEMPO ESTIMADO:  15 min

--- Como usar ---
  uv run datasets    ou    make datasets

Referência: https://mlflow.org/docs/latest/genai/datasets/
"""

import mlflow
from mlflow.genai.datasets import (
    create_dataset,
    get_dataset,
    set_dataset_tags,
)

from ia_observability.config import setup_mlflow

DATASET_NAME = "observability-eval-set"
EXPERIMENT_NAME = "12-datasets"


# ---------------------------------------------------------------------------
# Demo 1: Subir (criar) um dataset e adicionar registros
# ---------------------------------------------------------------------------


def demo_upload_dataset() -> None:
    """Cria um evaluation dataset e adiciona registros via SDK.

    create_dataset cria o dataset vinculado a um experiment.
    merge_records adiciona exemplos (inputs + expectations / ground-truth).
    set_dataset_tags permite tagging incremental para versionamento e busca.
    """
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    dataset = create_dataset(
        name=DATASET_NAME,
        experiment_id=[experiment.experiment_id],
        tags={"team": "ml-platform", "validation_version": "1.0"},
    )
    print(f"  Dataset criado: {dataset.name} (id={dataset.dataset_id})")

    # Casos de teste (inputs + expectations) sobre MLOps/observabilidade
    records = [
        {
            "inputs": {"question": "O que e tracing em aplicacoes LLM?"},
            "expectations": {
                "expected_facts": [
                    "Captura inputs e outputs de cada etapa",
                    "Cada span registra latencia e metadados",
                ]
            },
        },
        {
            "inputs": {"question": "Para que serve o MLflow Prompt Registry?"},
            "expectations": {
                "expected_facts": [
                    "Registrar e versionar prompts de forma centralizada",
                    "Linkar prompts a traces para rastreabilidade",
                ]
            },
        },
        {
            "inputs": {"question": "O que e observabilidade em sistemas de IA?"},
            "expectations": {
                "expected_facts": [
                    "Monitorar qualidade de respostas em producao",
                    "Inclui metricas de latencia, tokens e custo",
                ]
            },
        },
    ]
    dataset.merge_records(records)
    print(f"  {len(records)} registros adicionados via merge_records()")

    # Tagging incremental — util para busca posterior com search_datasets
    set_dataset_tags(
        dataset_id=dataset.dataset_id,
        tags={"environment": "dev"},
    )
    print("  Tags atualizadas: environment=dev")

    # Preview do dataset
    df = dataset.to_df()
    print(f"\n  Preview do dataset (total: {len(df)} registros)")
    sample = df.iloc[0]
    print(f"  Sample inputs: {sample['inputs']}")


# ---------------------------------------------------------------------------
# Demo 2: Buscar um dataset e atualiza-lo
# ---------------------------------------------------------------------------


def demo_fetch_dataset() -> None:
    """Busca um dataset existente pelo nome e o atualiza.

    get_dataset carrega o dataset versionado. merge_records adiciona novos
    casos (nova "versao" incremental). set_dataset_tags marca a evolucao.
    """
    dataset = get_dataset(name=DATASET_NAME)
    print(f"  Dataset encontrado: {dataset.name} (id={dataset.dataset_id})")
    print(f"  Tags atuais: {dataset.tags}")

    df_before = dataset.to_df()
    print(f"  Registros antes da atualizacao: {len(df_before)}")

    # Adiciona novos casos — versionamento incremental
    new_cases = [
        {
            "inputs": {"question": "O que sao LLM judges no MLflow?"},
            "expectations": {
                "expected_facts": [
                    "Modelos que avaliam respostas de outros modelos",
                    "Podem ser built-in ou customizados via @scorer",
                ]
            },
        },
    ]
    dataset.merge_records(new_cases)
    set_dataset_tags(
        dataset_id=dataset.dataset_id,
        tags={"validation_version": "1.1"},
    )
    print(f"  {len(new_cases)} novo(s) registro(s) adicionado(s)")
    print("  Tags atualizadas: validation_version=1.1")

    df_after = dataset.to_df()
    print(f"  Registros apos a atualizacao: {len(df_after)}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Executa as demos de evaluation datasets."""
    setup_mlflow(EXPERIMENT_NAME)

    print("=" * 60)
    print("DEMO 1: Subir (criar) um evaluation dataset")
    print("=" * 60)
    try:
        demo_upload_dataset()
    except Exception as e:
        print(f"  [SKIP] Nao foi possivel criar o dataset: {e}")
        print(
            "  Evaluation Datasets exigem MLflow com backend SQL "
            "(PostgreSQL, MySQL, SQLite ou MSSQL). FileStore nao e suportado."
        )
        return

    print("\n" + "=" * 60)
    print("DEMO 2: Buscar e atualizar um evaluation dataset")
    print("=" * 60)
    demo_fetch_dataset()

    # ────────────────────────────────────────────────────
    #  ✅ RESUMO DO QUE APRENDEMOS NESTE MÓDULO
    # ────────────────────────────────────────────────────
    #  ✔ create_dataset(): cria dataset vinculado a um
    #     experiment, com tags de identificação.
    #  ✔ merge_records(): adiciona exemplos (inputs +
    #     expectations) ao dataset.
    #  ✔ set_dataset_tags(): versionamento incremental
    #     com tags (ex: validation_version=1.1).
    #  ✔ get_dataset(): busca dataset existente pelo nome.
    #  ✔ to_df(): visualiza registros como DataFrame.
    #
    #  ⚠️ Requer backend SQL (PostgreSQL, MySQL, SQLite).
    #     FileStore não é suportado.
    #
    #  💡 EXERCÍCIO: Crie um dataset com 5 perguntas do
    #     seu domínio e execute uma avaliação com ele
    #     (veja módulo 04 - evaluation).
    # ────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("✅ MÓDULO CONCLUÍDO! Resumo do aprendizado acima.")
    print(f"  Experiment: 12-datasets no MLflow UI")
    print("-" * 60)


if __name__ == "__main__":
    main()
