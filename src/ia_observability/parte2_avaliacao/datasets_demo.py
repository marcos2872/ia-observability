"""Demonstracao de Evaluation Datasets: subir (criar) e buscar datasets.

O MLflow Evaluation Dataset e um conjunto versionado de exemplos
(inputs + expectations) usado para testar e comparar versoes da sua
aplicacao LLM. Este modulo mostra o ciclo basico:

1. Subir um dataset: create_dataset + merge_records + set_dataset_tags
2. Buscar um dataset: get_dataset + inspecao + atualizacao incremental

ATENCAO: Evaluation Datasets exigem um MLflow Tracking Server com backend
SQL (PostgreSQL, MySQL, SQLite ou MSSQL). FileStore NAO e suportado.

Referencia: https://mlflow.org/docs/latest/genai/datasets/
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

    print("\n" + "-" * 60)
    print("Abra o MLflow UI para visualizar:")
    print(f"  -> Experiment: {EXPERIMENT_NAME}")
    print(f"  - Evaluation Dataset: {DATASET_NAME}")
    print("  - Registros (inputs + expectations) e tags de versao")
    print("-" * 60)


if __name__ == "__main__":
    main()
