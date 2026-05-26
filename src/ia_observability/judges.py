"""Demonstracao de LLM judges customizados e code-based scorers.

Tipos de judges disponveis:
1. Built-in judges (Correctness, Safety, etc) - prontos para uso
2. Guidelines judges - criterios em linguagem natural
3. Custom LLM judges - prompts avancados com scoring customizado
4. Code-based scorers - logica programatica em Python

Este modulo demonstra os tipos 2, 3 e 4.

Referencia: https://mlflow.org/docs/latest/genai/eval-monitor/scorers/
"""

import mlflow
from mlflow.genai.scorers import Guidelines, scorer

from ia_observability.config import MODEL_NAME, get_client, setup_mlflow


# ---------------------------------------------------------------------------
# Code-based scorers (tipo 4): logica puramente programatica
# ---------------------------------------------------------------------------


@scorer
def response_length_check(inputs, outputs) -> dict:
    """Verifica se a resposta tem tamanho adequado (10-500 chars).

    Code-based scorers sao funcoes Python puras que recebem inputs/outputs
    e retornam um dict com 'score' (bool/float) e 'rationale' (explicacao).

    Use quando a avaliacao pode ser feita por regras deterministicas.
    """
    if outputs is None:
        return {"score": False, "rationale": "Nenhuma resposta foi gerada."}

    length = len(str(outputs))
    passed = 10 <= length <= 500
    return {
        "score": passed,
        "rationale": f"Resposta tem {length} caracteres. Faixa aceita: 10-500.",
    }


@scorer
def no_hallucination_keywords(inputs, outputs) -> dict:
    """Detecta possiveis marcadores de alucinacao na resposta.

    Verifica se a resposta contem frases que indicam incerteza fingida
    ou informacao fabricada, como datas muito especificas sem fonte.
    """
    if outputs is None:
        return {"score": False, "rationale": "Sem resposta."}

    # Indicadores de possivel alucinacao
    red_flags = [
        "estudos mostram que 99%",
        "foi comprovado cientificamente",
        "todos os especialistas concordam",
        "e um fato universalmente aceito",
    ]

    output_lower = str(outputs).lower()
    found_flags = [flag for flag in red_flags if flag in output_lower]

    if found_flags:
        return {
            "score": False,
            "rationale": f"Possiveis indicadores de alucinacao: {found_flags}",
        }
    return {
        "score": True,
        "rationale": "Nenhum indicador de alucinacao detectado.",
    }


@scorer
def contains_actionable_info(inputs, outputs) -> dict:
    """Verifica se a resposta contem informacao acionavel.

    Para perguntas do tipo 'como fazer X', a resposta deve conter
    passos, comandos, ou instrucoes claras.
    """
    if outputs is None:
        return {"score": False, "rationale": "Sem resposta."}

    question = inputs.get("question", "")
    output_str = str(outputs).lower()

    # So aplica para perguntas do tipo "como"
    if not any(kw in question.lower() for kw in ["como", "how", "de que forma"]):
        return {"score": True, "rationale": "Pergunta nao requer acao. N/A."}

    # Indicadores de conteudo acionavel
    actionable_markers = [
        "passo", "primeiro", "depois", "execute", "instale",
        "configure", "use", "rode", "pip", "import", "mlflow",
        "1.", "2.", "3.", "-",
    ]

    has_actionable = any(marker in output_str for marker in actionable_markers)
    return {
        "score": has_actionable,
        "rationale": (
            "Contem informacao acionavel (passos/comandos)."
            if has_actionable
            else "Resposta generica sem instrucoes claras."
        ),
    }


# ---------------------------------------------------------------------------
# Predict function
# ---------------------------------------------------------------------------


def predict_fn(question: str) -> str:
    """Predicao para avaliacao com judges."""
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Voce e um assistente tecnico de MLOps. "
                    "Sempre forneça instrucoes claras e acionaveis quando perguntado 'como'. "
                    "Cite fontes quando possivel. Responda em portugues."
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Executa avaliacao com judges customizados."""
    setup_mlflow("05-judges")
    mlflow.openai.autolog()

    dataset = [
        {"inputs": {"question": "Quais sao as vantagens do MLflow para LLMs?"}},
        {"inputs": {"question": "Como configurar tracing automatico com MLflow?"}},
        {"inputs": {"question": "O que e OpenTelemetry e como se relaciona com MLflow?"}},
        {"inputs": {"question": "Como avaliar a qualidade de um agente de IA?"}},
        {"inputs": {"question": "O MLflow suporta monitoramento em producao?"}},
    ]

    print("=" * 60)
    print("AVALIACAO COM JUDGES CUSTOMIZADOS")
    print("=" * 60)
    print(f"\n  Dataset: {len(dataset)} perguntas")
    print(f"  Modelo:  {MODEL_NAME}")
    print("  Judges:")
    print("    - Guidelines: technical_accuracy (LLM judge)")
    print("    - Guidelines: formatting (LLM judge)")
    print("    - Code-based: response_length_check")
    print("    - Code-based: no_hallucination_keywords")
    print("    - Code-based: contains_actionable_info")
    print("\n  Executando avaliacao...\n")

    results = mlflow.genai.evaluate(
        data=dataset,
        predict_fn=predict_fn,
        scorers=[
            # --- LLM judges com guidelines customizadas ---
            Guidelines(
                name="technical_accuracy",
                guidelines=(
                    "A resposta deve ser tecnicamente precisa sobre MLflow e MLOps. "
                    "Nao deve conter informacoes inventadas ou desatualizadas. "
                    "Deve usar terminologia correta."
                ),
            ),
            Guidelines(
                name="formatting",
                guidelines=(
                    "A resposta deve estar bem formatada e organizada. "
                    "Para perguntas 'como fazer', deve usar lista numerada ou bullets. "
                    "Nao deve ser um bloco de texto corrido sem estrutura."
                ),
            ),
            # --- Code-based scorers ---
            response_length_check,
            no_hallucination_keywords,
            contains_actionable_info,
        ],
    )

    print("  === METRICAS ===\n")
    if hasattr(results, "metrics") and results.metrics:
        for metric_name, value in results.metrics.items():
            print(f"  {metric_name}: {value}")
    else:
        print("  (Verifique o MLflow UI para resultados detalhados)")

    print("\n" + "-" * 60)
    print("No MLflow UI -> Experiment '05-judges':")
    print("  - Cada judge mostra score + rationale por exemplo")
    print("  - Code-based scorers sao instantaneos (sem custo LLM)")
    print("  - LLM judges geram traces proprios (meta-avaliacao)")
    print("-" * 60)


if __name__ == "__main__":
    main()
