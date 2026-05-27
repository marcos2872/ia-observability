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

# Judge model separado — permite usar um modelo maior/melhor para avaliacao
_JUDGE_MODEL_NAME: str = os.getenv("mlflow_judge_model", MODEL_NAME)
JUDGE_MODEL: str = f"gateway:/{_JUDGE_MODEL_NAME}"

# Azure OpenAI — usado para prompt optimization (GepaPromptOptimizer)
AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_MODEL: str = f"azure:/{AZURE_OPENAI_DEPLOYMENT}"
AZURE_AVAILABLE: bool = bool(os.getenv("AZURE_OPENAI_API_KEY"))


def setup_mlflow(experiment_name: str) -> None:
    """Configura o MLflow tracking URI e experiment.

    Se o experiment foi deletado, restaura automaticamente antes de setar.

    Args:
        experiment_name: Nome do experiment no MLflow.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment and experiment.lifecycle_stage == "deleted":
        client.restore_experiment(experiment.experiment_id)
    mlflow.set_experiment(experiment_name)


def patch_judge_timeout(timeout: int = 300) -> None:
    """Aumenta o timeout do judge model para modelos lentos.

    O MLflow hardcoda timeout de 60s para chamadas de judge via
    score_model_on_payload. Modelos self-hosted (especialmente menores)
    podem precisar de mais tempo para gerar respostas estruturadas.

    Args:
        timeout: Timeout em segundos (default: 300 = 5 min).
    """
    import mlflow.metrics.genai.model_utils as mu

    _original_send = mu._send_request

    def _patched_send(endpoint, headers, payload):
        import requests

        from mlflow.exceptions import MlflowException

        try:
            response = requests.post(
                url=endpoint,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            body = getattr(e.response, "text", "")
            raise MlflowException(
                f"Failed to call LLM endpoint at {endpoint}.\n- Error: {e}\n"
                f"- Response body: {body}"
            )
        except requests.exceptions.Timeout:
            raise MlflowException(
                f"Timeout calling LLM endpoint at {endpoint} (timeout={timeout}s)."
            )
        return response.json()

    mu._send_request = _patched_send


def get_client() -> OpenAI:
    """Retorna um cliente OpenAI apontando para o MLflow AI Gateway.

    O AI Gateway expoe uma API compativel com OpenAI, permitindo
    usar o SDK padrao da OpenAI sem modificacoes.
    """
    return OpenAI(
        base_url=MLFLOW_GATEWAY_URL,
        api_key="not-needed",  # Gateway gerencia autenticacao
    )
