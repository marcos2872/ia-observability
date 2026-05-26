"""Configuracao centralizada do MLflow e cliente OpenAI via AI Gateway."""

import os
from pathlib import Path

from dotenv import load_dotenv

import mlflow
from openai import OpenAI

# Carrega .env da raiz do projeto
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path)

MLFLOW_TRACKING_URI: str = os.getenv("mlflow_url", "http://localhost:5000")
MLFLOW_GATEWAY_URL: str = os.getenv("mlflow_openia_url", "http://localhost:5000/gateway/mlflow/v1")
MODEL_NAME: str = os.getenv("mlflow_model", "gemma4-e4b")


def setup_mlflow(experiment_name: str) -> None:
    """Configura o MLflow tracking URI e experiment.

    Args:
        experiment_name: Nome do experiment no MLflow.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)


def get_client() -> OpenAI:
    """Retorna um cliente OpenAI apontando para o MLflow AI Gateway.

    O AI Gateway expoe uma API compativel com OpenAI, permitindo
    usar o SDK padrao da OpenAI sem modificacoes.
    """
    return OpenAI(
        base_url=MLFLOW_GATEWAY_URL,
        api_key="not-needed",  # Gateway gerencia autenticacao
    )
