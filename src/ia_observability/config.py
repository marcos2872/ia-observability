"""Configuracao centralizada do MLflow e cliente OpenAI via AI Gateway."""

import os
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv
from openai import APIError, APITimeoutError, RateLimitError

# Desliga a telemetria do MLflow ANTES de importar o mlflow (thread extra que
# faz imports + HTTP em background; ver _disable_async_prompt_linking abaixo).
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("DO_NOT_TRACK", "true")

import mlflow
import mlflow.openai  # pre-import: evita import lazy dentro do optimize_prompts
from openai import OpenAI


def _disable_async_prompt_linking() -> None:
    """Desativa o thread assincrono de prompt-linking do register_prompt.

    Ao registrar um prompt com um experiment ativo, o MLflow dispara um thread
    em background (_link_prompt_to_experiment) que faz HTTP + imports lazy
    (deteccao de Databricks). No Python 3.14, esses imports concorrem com os
    imports da thread principal dentro do optimize_prompts e causam um DEADLOCK
    no lock de import — o GEPA travava antes de chamar o modelo (GPU em 0%).

    O link prompt<->experimento e apenas uma tag de conveniencia no UI, entao
    desativa-lo e seguro e elimina a thread que causa o deadlock.
    """
    try:
        from mlflow.tracking.client import MlflowClient

        MlflowClient._link_prompt_to_experiment = lambda self, *a, **k: None
    except Exception:
        pass


# Carrega .env da raiz do projeto
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path)

MLFLOW_TRACKING_URI: str = os.getenv("mlflow_url", "http://localhost:5000")

# Endpoint do MLflow AI Gateway (compativel com OpenAI)
MLFLOW_GATEWAY_URL: str = os.getenv(
    "mlflow_openai_url", "http://localhost:5000/gateway/mlflow/v1"
)

MODEL_NAME: str = os.getenv("mlflow_model", "qwen3.5-9b")


def make_predict_fn(
    system_prompt: str,
    temperature: float = 0.7,
) -> Callable[[str], str]:
    """Cria uma funcao de predicao para use com mlflow.evaluate().

    A funcao criada recebe uma pergunta (str) e retorna a resposta do modelo
    usando o system prompt e temperatura configurados.
    """
    def predict_fn(question: str) -> str:
        client = get_client()
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content or "(resposta vazia)"
        except (APITimeoutError, RateLimitError) as e:
            print(f"  [ERRO] Timeout/rate limit na chamada ao modelo: {e}")
            return "(erro: servidor temporariamente indisponivel)"
        except APIError as e:
            print(f"  [ERRO] API retornou erro: {e}")
            return f"(erro na chamada: {e.message})"
        except Exception as e:
            print(f"  [ERRO] Falha inesperada na chamada ao modelo: {e}")
            return "(erro inesperado)"
    return predict_fn


# Judge model separado — permite usar um modelo maior/melhor para avaliacao
_JUDGE_MODEL_NAME: str = os.getenv("mlflow_judge_model", MODEL_NAME)

# Scorers/judges built-in usam o provider nativo 'gateway:/' do MLflow.
JUDGE_MODEL: str = f"gateway:/{_JUDGE_MODEL_NAME}"

# Para a reflexao do GEPA (prompt optimization), o litellm precisa de um
# endpoint OpenAI-compatible. Como o AI Gateway ja e compativel com OpenAI,
# apontamos OPENAI_API_BASE para ele e usamos o provider 'openai:/'.
os.environ.setdefault("OPENAI_API_BASE", MLFLOW_GATEWAY_URL)
os.environ.setdefault("OPENAI_API_KEY", "not-needed")
OPTIMIZER_JUDGE_MODEL: str = f"openai:/{_JUDGE_MODEL_NAME}"

# GEPA budget: max number of metric calls (evaluations).
GEPA_MAX_METRIC_CALLS: int = int(os.getenv("GEPA_MAX_METRIC_CALLS", "30"))


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

    def _patched_send(endpoint: str, headers: dict, payload: dict) -> dict:
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


def patch_judge_json_parsing() -> None:
    """Torna o parsing das respostas de judge tolerante a 'lixo' ao redor do JSON.

    Modelos de judge menores costumam responder com JSON cercado de markdown
    (```), preambulo ("Here is..."), um segundo objeto, um ARRAY [ {...} ] em
    vez de objeto, ou ate newlines literais nao-escapados dentro das strings.
    O parser nativo do MLflow falha nesses casos ("Failed to parse response from
    judge model" / "Extra data").

    Aqui envolvemos `_strip_markdown_code_blocks` para extrair de forma tolerante
    o PRIMEIRO valor JSON (objeto ou array) a partir do primeiro '{' ou '[',
    usando strict=False (aceita control chars). Se vier array, usa o 1o elemento.
    Nao afeta respostas ja limpas.
    """
    import mlflow.genai.judges.adapters.gateway_adapter as ga

    _original_strip = ga._strip_markdown_code_blocks

    def _patched_strip(response: str) -> str:
        import json

        cleaned = _original_strip(response)

        # Tenta extrair um objeto JSON valido a partir do texto limpo e, como
        # fallback, da resposta crua. strict=False aceita newlines literais.
        for text in (cleaned, response):
            candidates = [i for i in (text.find("{"), text.find("[")) if i != -1]
            if not candidates:
                continue
            start = min(candidates)
            try:
                obj, _ = json.JSONDecoder(strict=False).raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            # Judge espera um objeto; se vier array [ {...} ], usa o 1o item.
            if isinstance(obj, list) and obj:
                obj = obj[0]
            if isinstance(obj, dict):
                return json.dumps(obj)
        return cleaned

    ga._strip_markdown_code_blocks = _patched_strip


def get_client() -> OpenAI:
    """Retorna um cliente OpenAI apontando para o MLflow AI Gateway.

    O AI Gateway expoe uma API compativel com OpenAI, permitindo usar o SDK
    padrao da OpenAI sem modificacoes. A autenticacao e gerenciada pelo Gateway.
    """
    return OpenAI(
        base_url=MLFLOW_GATEWAY_URL,
        api_key="not-needed",  # Gateway gerencia autenticacao
    )


def patch_litellm_max_tokens(default_max_tokens: int = 4096) -> None:
    """Garante um max_tokens minimo nas chamadas do litellm.

    O litellm e usado pela reflexao do GEPA e pelos judges. O GEPA chama
    `litellm.completion(...)` SEM max_tokens (gepa/api.py), entao usa o default
    baixo do gateway/modelo e o prompt candidato gerado pode TRUNCAR no meio
    (o prompt otimizado sai cortado). Aqui injetamos um teto generoso quando a
    chamada nao especifica um. Nao afeta o predict_fn (que usa o SDK OpenAI direto).
    """
    try:
        import litellm
    except ImportError:
        return

    _original = litellm.completion

    def _patched(*args, **kwargs):
        kwargs.setdefault("max_tokens", default_max_tokens)
        return _original(*args, **kwargs)

    litellm.completion = _patched


def apply_patches() -> None:
    """Aplica todas as correcoes (monkey patches) necessarias para o projeto.

    Deve ser chamada explicitamente no inicio de cada demo, antes de
    qualquer uso de MLflow judges ou litellm. Nao executa na importacao
    do modulo para evitar side effects indesejados.

    Aplica:
    - _disable_async_prompt_linking: Previne deadlock no Python 3.14
    - patch_judge_json_parsing: Torna parsing de judges tolerante a JSON extra
    - patch_litellm_max_tokens: Garante max_tokens minimo no GEPA e judges
    """
    _disable_async_prompt_linking()
    patch_judge_json_parsing()
    patch_litellm_max_tokens()
