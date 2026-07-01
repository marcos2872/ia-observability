"""
[Parte 2 — Avaliação] Módulo 05: LLM Judges Customizados
===========================================================
╔══════════════════════════════════════════════════════════╗
║  OBJETIVOS DE APRENDIZADO                               ║
║  • Diferenciar LLM judges (qualitativos) de code-based  ║
║    scorers (determinísticos)                             ║
║  • Criar Guidelines judges com critérios em linguagem    ║
║    natural                                               ║
║  • Implementar code-based scorers com @scorer + Feedback ║
║  • Combinar os dois tipos para avaliação robusta         ║
╚══════════════════════════════════════════════════════════╝

CONCEITO-CHAVE:
  LLM judges usam um modelo para avaliar qualidade (custa
  tokens, mas captura nuances). Code-based scorers são
  regras em Python (instantâneos, sem custo). O ideal é
  COMBINAR ambos: judges para qualidade subjetiva, scorers
  para regras objetivas.

PRÉ-REQUISITOS:  Módulo 04 (evaluation) — conceito de scorers
DIFICULDADE:     🟡 Médio
TEMPO ESTIMADO:  20 min

--- Como usar ---
  uv run judges    ou    make judges

Referência: https://mlflow.org/docs/latest/genai/eval-monitor/scorers/
"""

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.genai.scorers import Guidelines, scorer

from ia_observability.config import (
    JUDGE_MODEL,
    MODEL_NAME,
    apply_patches,
    make_predict_fn,
    patch_judge_timeout,
    setup_mlflow,
)


# ---------------------------------------------------------------------------
# Code-based scorers (tipo 4): logica puramente programatica
# ---------------------------------------------------------------------------


@scorer
def response_length_check(inputs: dict | None, outputs: str | None) -> Feedback:
    """Verifica se a resposta tem tamanho adequado (10-500 chars).

    Code-based scorers retornam Feedback com value (bool/float) e rationale.
    Use quando a avaliacao pode ser feita por regras deterministicas.
    """
    if outputs is None:
        return Feedback(value=False, rationale="Nenhuma resposta foi gerada.")

    length = len(str(outputs))
    passed = 10 <= length <= 500
    return Feedback(
        value=passed,
        rationale=f"Resposta tem {length} caracteres. Faixa aceita: 10-500.",
    )


@scorer
def no_hallucination_keywords(inputs: dict | None, outputs: str | None) -> Feedback:
    """Detecta possiveis marcadores de alucinacao na resposta.

    Verifica se a resposta contem frases que indicam incerteza fingida
    ou informacao fabricada, como datas muito especificas sem fonte.
    """
    if outputs is None:
        return Feedback(value=False, rationale="Sem resposta.")

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
        return Feedback(
            value=False,
            rationale=f"Possiveis indicadores de alucinacao: {found_flags}",
        )
    return Feedback(
        value=True,
        rationale="Nenhum indicador de alucinacao detectado.",
    )


@scorer
def contains_actionable_info(inputs: dict | None, outputs: str | None) -> Feedback:
    """Verifica se a resposta contem informacao acionavel.

    Para perguntas do tipo 'como fazer X', a resposta deve conter
    passos, comandos, ou instrucoes claras.
    """
    if outputs is None:
        return Feedback(value=False, rationale="Sem resposta.")

    question = inputs.get("question", "")
    output_str = str(outputs).lower()

    # So aplica para perguntas do tipo "como"
    if not any(kw in question.lower() for kw in ["como", "how", "de que forma"]):
        return Feedback(value=True, rationale="Pergunta nao requer acao. N/A.")

    # Indicadores de conteudo acionavel
    actionable_markers = [
        "passo", "primeiro", "depois", "execute", "instale",
        "configure", "use", "rode", "pip", "import", "mlflow",
        "1.", "2.", "3.", "-",
    ]

    has_actionable = any(marker in output_str for marker in actionable_markers)
    return Feedback(
        value=has_actionable,
        rationale=(
            "Contem informacao acionavel (passos/comandos)."
            if has_actionable
            else "Resposta generica sem instrucoes claras."
        ),
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Executa avaliacao com judges customizados."""
    apply_patches()
    setup_mlflow("05-judges")
    patch_judge_timeout(300)
    mlflow.openai.autolog()

    dataset = [
        {"inputs": {"question": "Quais sao as vantagens do MLflow para LLMs?"}},
        {"inputs": {"question": "Como configurar tracing automatico com MLflow?"}},
        {"inputs": {"question": "O que e OpenTelemetry e como se relaciona com MLflow?"}},
        {"inputs": {"question": "Como avaliar a qualidade de um agente de IA?"}},
        {"inputs": {"question": "O MLflow suporta monitoramento em producao?"}},
    ]
    predict_fn = make_predict_fn(
        (
            "Voce e um assistente tecnico de MLOps. "
            "Sempre forneça instrucoes claras e acionaveis quando perguntado 'como'. "
            "Cite fontes quando possivel. Responda em portugues."
        ),
        0.5,
    )

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
                model=JUDGE_MODEL,
            ),
            Guidelines(
                name="formatting",
                guidelines=(
                    "A resposta deve estar bem formatada e organizada. "
                    "Para perguntas 'como fazer', deve usar lista numerada ou bullets. "
                    "Nao deve ser um bloco de texto corrido sem estrutura."
                ),
                model=JUDGE_MODEL,
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

    # ────────────────────────────────────────────────────
    #  ✅ RESUMO DO QUE APRENDEMOS NESTE MÓDULO
    # ────────────────────────────────────────────────────
    #  ✔ Guidelines: judge LLM que avalia segundo critérios
    #     em linguagem natural (ex: "deve ser preciso").
    #  ✔ @scorer + Feedback: cria avaliadores 100% em
    #     Python, sem custo de tokens.
    #  ✔ Code-based scorers são ideais para regras
    #     objetivas (tamanho, palavras proibidas, formato).
    #  ✔ A combinação dos dois tipos dá o melhor custo-
    #     benefício: judges para nuance, scorers para garantia.
    #
    #  🔍 MLflow UI → Experiment '05-judges': cada judge
    #     mostra score + rationale (justificativa).
    #
    #  💡 EXERCÍCIO: Crie um code-based scorer que verifica
    #     se a resposta contém código Python (```python).
    # ────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("✅ MÓDULO CONCLUÍDO! Resumo do aprendizado acima.")
    print("  Experiment: 05-judges no MLflow UI")
    print("-" * 60)


if __name__ == "__main__":
    main()
