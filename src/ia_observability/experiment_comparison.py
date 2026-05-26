"""Demonstracao de comparacao de configuracoes/modelos (benchmark).

Este modulo executa o mesmo dataset de avaliacao contra diferentes
configuracoes da aplicacao (prompts, temperaturas), permitindo
comparar objetivamente qual configuracao produz melhores resultados.

Cenario tipico:
- Testar diferentes system prompts
- Comparar temperaturas de geracao
- Avaliar impacto de mudancas no pipeline

Referencia: https://mlflow.org/docs/latest/genai/eval-monitor/
"""

import mlflow
from mlflow.genai.scorers import Correctness, Guidelines, RelevanceToQuery

from ia_observability.config import MODEL_NAME, get_client, setup_mlflow

# ---------------------------------------------------------------------------
# Dataset de benchmark (compartilhado entre todas as configuracoes)
# ---------------------------------------------------------------------------

EVAL_DATASET = [
    {
        "inputs": {"question": "O que e observabilidade de IA?"},
        "expectations": {
            "expected_facts": [
                "Observabilidade e a capacidade de entender o estado interno de um sistema.",
                "Inclui traces, metricas e logs.",
            ],
        },
    },
    {
        "inputs": {"question": "Quais metricas um LLM deve expor em producao?"},
        "expectations": {
            "expected_facts": [
                "Latencia por requisicao.",
                "Token usage (input e output).",
                "Custo por chamada.",
                "Qualidade das respostas.",
            ],
        },
    },
    {
        "inputs": {"question": "O que e LLM-as-a-Judge?"},
        "expectations": {
            "expected_facts": [
                "Usar um LLM para avaliar automaticamente a qualidade de outro LLM.",
                "Pode avaliar corretude, relevancia, seguranca.",
            ],
        },
    },
    {
        "inputs": {"question": "Como funciona o tracing do MLflow?"},
        "expectations": {
            "expected_facts": [
                "Captura automatica de inputs, outputs e metadados.",
                "Compativel com OpenTelemetry.",
                "Suporta auto-tracing e tracing manual.",
            ],
        },
    },
]

# Scorers usados em todas as configuracoes (mesmos criterios = comparacao justa)
SCORERS = [
    Correctness(),
    RelevanceToQuery(),
    Guidelines(
        name="conciseness",
        guidelines="A resposta deve ter no maximo 3 frases. Respostas longas devem falhar.",
    ),
    Guidelines(
        name="portuguese",
        guidelines="A resposta DEVE estar em portugues brasileiro.",
    ),
]

# ---------------------------------------------------------------------------
# Configuracoes a comparar
# ---------------------------------------------------------------------------

CONFIGS = [
    {
        "name": "config-verbose-high-temp",
        "description": "Prompt generico, alta temperatura (criativo)",
        "system_prompt": "Responda de forma detalhada e completa, cobrindo todos os aspectos.",
        "temperature": 0.9,
    },
    {
        "name": "config-concise-low-temp",
        "description": "Prompt conciso, baixa temperatura (deterministic)",
        "system_prompt": "Responda de forma concisa em no maximo 2 frases. Va direto ao ponto.",
        "temperature": 0.2,
    },
    {
        "name": "config-technical-medium",
        "description": "Prompt tecnico, temperatura media",
        "system_prompt": (
            "Voce e um engenheiro de ML senior. "
            "Responda de forma tecnica e precisa. "
            "Use terminologia correta e seja objetivo."
        ),
        "temperature": 0.5,
    },
]


def make_predict_fn(system_prompt: str, temperature: float):
    """Factory que cria uma predict_fn com configuracao especifica.

    Cada configuracao gera uma predict_fn diferente, permitindo
    avaliar o impacto de cada parametro separadamente.
    """

    def predict_fn(question: str) -> str:
        client = get_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content

    return predict_fn


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Executa benchmark comparativo entre configuracoes."""
    setup_mlflow("08-experiment-comparison")
    mlflow.openai.autolog()

    print("=" * 60)
    print("BENCHMARK: COMPARACAO DE CONFIGURACOES")
    print("=" * 60)
    print(f"\n  Dataset: {len(EVAL_DATASET)} exemplos")
    print(f"  Modelo:  {MODEL_NAME}")
    print(f"  Configs: {len(CONFIGS)} configuracoes")
    print(f"  Scorers: Correctness, RelevanceToQuery, conciseness, portuguese\n")

    all_results: dict[str, dict] = {}

    for config in CONFIGS:
        print(f"  --- Avaliando: {config['name']} ---")
        print(f"      {config['description']}")

        with mlflow.set_active_model(name=config["name"]):
            predict_fn = make_predict_fn(config["system_prompt"], config["temperature"])

            results = mlflow.genai.evaluate(
                data=EVAL_DATASET,
                predict_fn=predict_fn,
                scorers=SCORERS,
            )

            if hasattr(results, "metrics") and results.metrics:
                all_results[config["name"]] = results.metrics
                # Mostra metricas resumidas
                for k, v in results.metrics.items():
                    print(f"      {k}: {v}")
            else:
                all_results[config["name"]] = {}
                print("      (Verifique MLflow UI)")
            print()

    # Resumo comparativo
    print("=" * 60)
    print("RESUMO COMPARATIVO")
    print("=" * 60)

    if all_results:
        # Coleta todas as metricas unicas
        all_metric_names = set()
        for metrics in all_results.values():
            all_metric_names.update(metrics.keys())

        # Header
        print(f"\n  {'Metrica':<30}", end="")
        for config_name in all_results:
            short_name = config_name.replace("config-", "")[:15]
            print(f" | {short_name:<15}", end="")
        print()
        print(f"  {'-'*30}", end="")
        for _ in all_results:
            print(f" | {'-'*15}", end="")
        print()

        # Linhas
        for metric in sorted(all_metric_names):
            print(f"  {metric:<30}", end="")
            for config_name in all_results:
                value = all_results[config_name].get(metric, "N/A")
                if isinstance(value, float):
                    print(f" | {value:<15.3f}", end="")
                else:
                    print(f" | {str(value):<15}", end="")
            print()

    print("\n" + "-" * 60)
    print("No MLflow UI -> Experiment '08-experiment-comparison':")
    print("  - Compare runs lado a lado")
    print("  - Analise qual config tem melhor score geral")
    print("  - Cada LoggedModel mostra seus traces vinculados")
    print("-" * 60)


if __name__ == "__main__":
    main()
