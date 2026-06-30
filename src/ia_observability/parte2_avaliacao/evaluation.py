"""
[Parte 2 — Avaliação] Módulo 04: Evaluation com Scorers Built-in
==================================================================
╔══════════════════════════════════════════════════════════╗
║  OBJETIVOS DE APRENDIZADO                               ║
║  • Entender o que é avaliação sistemática de LLMs       ║
║  • Conhecer os scorers built-in do MLflow (Correctness, ║
║    RelevanceToQuery, Guidelines, Safety, Fluency)        ║
║  • Criar um dataset de avaliação com inputs/expecations ║
║  • Rodar mlflow.genai.evaluate() e interpretar métricas ║
╚══════════════════════════════════════════════════════════╝

CONCEITO-CHAVE:
  "Se não dá para medir, não dá para melhorar." Avaliação
  sistemática usa DATASETS de perguntas com respostas esperadas
  e SCORERS (judges) que medem a qualidade automaticamente.
  Isso substitui o "achismo" por métricas objetivas.

PRÉ-REQUISITOS:  Parte 1 (tracing + tokens)
DIFICULDADE:     🟡 Médio
TEMPO ESTIMADO:  20 min

--- Como usar ---
  uv run evaluation    ou    make evaluation

Referência: https://mlflow.org/docs/latest/genai/eval-monitor/quickstart/
"""

import mlflow
from mlflow.genai.scorers import Correctness, Guidelines, RelevanceToQuery

from ia_observability.config import JUDGE_MODEL, MODEL_NAME, get_client, patch_judge_timeout, setup_mlflow


def get_eval_dataset() -> list[dict]:
    """Dataset de avaliacao com perguntas e fatos esperados.

    Cada item contem:
    - inputs: dicionario com os parametros da predict_fn
    - expectations: dados para os judges avaliarem (ground truth)

    O campo 'expected_facts' e usado pelo judge Correctness para verificar
    se a resposta contem os fatos listados.
    """
    return [
        {
            "inputs": {"question": "O que e MLflow?"},
            "expectations": {
                "expected_facts": [
                    "MLflow e uma plataforma open source para gerenciar o ciclo de vida de ML.",
                    "MLflow oferece tracing, evaluation e prompt management para LLMs.",
                ],
            },
        },
        {
            "inputs": {"question": "O que e tracing em aplicacoes de LLM?"},
            "expectations": {
                "expected_facts": [
                    "Tracing captura inputs, outputs e metadados de cada passo de execucao.",
                    "Permite identificar bugs e comportamentos inesperados.",
                ],
            },
        },
        {
            "inputs": {"question": "Quanto custa usar MLflow?"},
            "expectations": {
                "expected_facts": [
                    "MLflow e gratuito e open source.",
                ],
            },
        },
        {
            "inputs": {"question": "O que sao LLM judges?"},
            "expectations": {
                "expected_facts": [
                    "LLM judges usam modelos de linguagem para avaliar qualidade de respostas.",
                    "Podem verificar corretude, relevancia, seguranca e aderencia a guidelines.",
                ],
            },
        },
        {
            "inputs": {"question": "Como o MLflow rastreia token usage?"},
            "expectations": {
                "expected_facts": [
                    "MLflow captura automaticamente input tokens, output tokens e custo.",
                    "Os dados ficam disponiveis no nivel de trace e de span individual.",
                ],
            },
        },
    ]


def predict_fn(question: str) -> str:
    """Funcao de predicao que sera avaliada pelos judges.

    A assinatura deve corresponder aos campos de 'inputs' no dataset.
    O MLflow chama esta funcao para cada item do dataset e passa o output
    para os scorers avaliarem.

    Args:
        question: Pergunta do dataset (mapeada de inputs.question).

    Returns:
        Resposta gerada pelo modelo.
    """
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Responda de forma concisa e precisa. Use no maximo 3 frases.",
            },
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


def main() -> None:
    """Executa avaliacao completa com scorers built-in."""
    setup_mlflow("04-evaluation")
    patch_judge_timeout(300)
    mlflow.openai.autolog()

    dataset = get_eval_dataset()

    print("=" * 60)
    print("AVALIACAO COM SCORERS BUILT-IN")
    print("=" * 60)
    print(f"\n  Dataset: {len(dataset)} exemplos")
    print(f"  Modelo:  {MODEL_NAME}")
    print(f"  Scorers: Correctness, RelevanceToQuery, Guidelines\n")
    print("  Executando avaliacao (pode levar alguns minutos)...\n")

    results = mlflow.genai.evaluate(
        data=dataset,
        predict_fn=predict_fn,
        scorers=[
            # Verifica se os fatos esperados estao presentes na resposta
            Correctness(model=JUDGE_MODEL),
            # Verifica se a resposta e relevante a pergunta
            RelevanceToQuery(model=JUDGE_MODEL),
            # Judge customizado: verifica se resposta esta em portugues
            Guidelines(
                name="resposta_em_portugues",
                guidelines="A resposta DEVE estar em portugues. Nao pode estar em ingles.",
                model=JUDGE_MODEL,
            ),
            # Judge customizado: verifica concisao
            Guidelines(
                name="conciseness",
                guidelines="A resposta deve ter no maximo 3 frases. Respostas longas devem falhar.",
                model=JUDGE_MODEL,
            ),
        ],
    )

    print("  === RESULTADOS ===\n")
    if hasattr(results, "metrics") and results.metrics:
        for metric_name, value in results.metrics.items():
            print(f"  {metric_name}: {value}")
    else:
        print("  (Metricas nao disponiveis - verifique o MLflow UI)")

    # ────────────────────────────────────────────────────
    #  ✅ RESUMO DO QUE APRENDEMOS NESTE MÓDULO
    # ────────────────────────────────────────────────────
    #  ✔ mlflow.genai.evaluate() executa predict_fn para
    #     cada exemplo do dataset e passa o output para
    #     os scorers.
    #  ✔ Correctness: verifica se fatos esperados estão
    #     na resposta (ground-truth based).
    #  ✔ RelevanceToQuery: mede relevância ao input.
    #  ✔ Guidelines: judge customizado via linguagem
    #     natural (ex: "responda em português").
    #  ✔ Resultados consolidados em métricas por scorer.
    #
    #  🔍 MLflow UI → Experiment '04-evaluation':
    #     tabela com score de cada judge por exemplo.
    #
    #  💡 EXERCÍCIO: Adicione mais 2 perguntas ao dataset
    #     com fatos esperados. A média dos scores mudou?
    # ────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("✅ MÓDULO CONCLUÍDO! Resumo do aprendizado acima.")
    print("  Experiment: 04-evaluation no MLflow UI")
    print("-" * 60)


if __name__ == "__main__":
    main()
