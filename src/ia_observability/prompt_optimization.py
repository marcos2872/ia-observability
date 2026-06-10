"""Demonstracao de Prompt Optimization: otimizar o SYSTEM prompt automaticamente.

O MLflow otimiza um prompt registrado aprendendo com dados de avaliacao e
metricas (scorers). Aqui otimizamos o SYSTEM prompt de um CLASSIFICADOR de
mensagens de suporte (DUVIDA, BUG, RECLAMACAO, ELOGIO, SOLICITACAO).

Por que classificacao? E um cenario onde a melhora fica EVIDENTE: com o system
prompt vago ("Voce e um assistente.") o modelo responde em prosa e nem conhece
as categorias -> score baixo. A otimizacao ensina o modelo, via system prompt, a
escolher a categoria certa e responder APENAS o rotulo -> score alto.

Padrao:
    predict_fn(mensagem) -> usa o system prompt (otimizavel) + texto a classificar
    optimize_prompts(predict_fn, train_data, prompt_uris, optimizer, scorers)

Dois algoritmos sao demonstrados:
- GEPA (GepaPromptOptimizer): refina o prompt iterativamente via reflexao do
  LLM sobre falhas. Precisa de train_data e scorers (few-shot).
- Metaprompting (MetaPromptOptimizer): reestrutura o prompt seguindo boas
  praticas. Funciona em zero-shot (sem dados nem scorers).

Sobre os scorers: usamos um scorer CODE-BASED (deterministico) que mede o acerto
da classificacao. Vantagem didatica: nao depende de um LLM judge (que, com
modelos pequenos, costuma devolver JSON fora do schema e quebrar a otimizacao).
Para usar um LLM judge, veja o comentario em SCORERS.

A reflexao do GEPA usa 'openai:/' do litellm apontando para o AI Gateway
(OpenAI-compatible), configurado em config.py.

Referencia: https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/
"""

import os
import unicodedata

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.genai.optimize import GepaPromptOptimizer, MetaPromptOptimizer
from mlflow.genai.scorers import scorer

from ia_observability.config import (
    MODEL_NAME,
    OPTIMIZER_JUDGE_MODEL,
    get_client,
    patch_judge_timeout,
    setup_mlflow,
)

EXPERIMENT_NAME = "13-prompt-optimization"
PROMPT_NAME = "observability-system-prompt"

# Orcamento do GEPA: numero MAXIMO de "metric calls" (avaliacoes).
#   1 metric call = 1 inferencia (predict_fn) + 1 avaliacao do scorer.
# O GEPA NAO tem um numero fixo de rodadas: gasta esse orcamento entre a
# avaliacao baseline do prompt inicial e as rodadas de reflexao+mutacao, parando
# quando o orcamento acaba.
# IMPORTANTE: o baseline ja consome ~1 chamada por exemplo do dataset. Para
# sobrar orcamento para varias rodadas de reflexao, use budget >> tamanho do
# dataset (ex: dataset de 5 -> budget 30+). Mais orcamento = mais rodadas.
# A QUALIDADE da melhora tambem depende do reflection_model (OPTIMIZER_JUDGE_MODEL):
# modelos pequenos/locais geram candidatos fracos ou vazios; um modelo forte
# (ex: GPT-4/5) melhora muito. Para resultado rapido e confiavel, veja a Demo 2
# (Metaprompting), que reescreve o prompt em 1 chamada com guidelines explicitas.
GEPA_MAX_METRIC_CALLS: int = int(os.getenv("GEPA_MAX_METRIC_CALLS", "30"))

# SYSTEM prompt propositalmente fraco/vago — a otimizacao deve melhora-lo.
# Nao tem variaveis: a mensagem a classificar vai na mensagem do usuario.
WEAK_SYSTEM_PROMPT = "Voce e um assistente."

# Tarefa: classificar mensagens de suporte em UMA categoria. E um cenario
# classico de prompt optimization: com o prompt vago o modelo responde em prosa
# (formato errado -> score baixo); o prompt otimizado aprende a devolver apenas
# o rotulo correto (score alto). Por isso a melhora fica evidente.
LABELS = ["DUVIDA", "BUG", "RECLAMACAO", "ELOGIO", "SOLICITACAO"]

# Dataset de treino: inputs (mensagem do usuario) + expectations (rotulo correto).
# Mantido pequeno (1-2 por categoria) de proposito: o baseline do GEPA custa ~1
# chamada por exemplo, entao um dataset enxuto deixa mais orcamento para as
# rodadas de reflexao (e roda mais rapido).
TRAIN_DATA = [
    {
        "inputs": {"mensagem": "Como faco para exportar meus traces para CSV?"},
        "expectations": {"label": "DUVIDA"},
    },
    {
        "inputs": {"mensagem": "O dashboard trava quando abro o experimento 42."},
        "expectations": {"label": "BUG"},
    },
    {
        "inputs": {
            "mensagem": "Ja e a terceira vez que perco meus dados, inaceitavel."
        },
        "expectations": {"label": "RECLAMACAO"},
    },
    {
        "inputs": {"mensagem": "Parabens, a nova UI de tracing ficou excelente!"},
        "expectations": {"label": "ELOGIO"},
    },
    {
        "inputs": {"mensagem": "Seria otimo poder filtrar traces por custo de tokens."},
        "expectations": {"label": "SOLICITACAO"},
    },
]


# ---------------------------------------------------------------------------
# Scorer code-based: acerto da classificacao (deterministico, sem judge)
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Remove acentos, baixa a caixa e tira pontuacao das bordas."""
    nfkd = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


@scorer
def label_accuracy(outputs, expectations) -> Feedback:
    """Mede o acerto da classificacao e EXPLICA o objetivo (deterministico).

    Retorna um Feedback com value (numerico) e rationale (texto). O rationale e
    fundamental: o GEPA usa esses rationales na reflexao para entender O QUE esta
    sendo avaliado. Sem explicar o objetivo, o reflection model pode interpretar
    as categorias como "tipos de atendimento" e escrever um prompt de assistente
    prestativo (em vez de um classificador) — foi o que aconteceu sem feedback.

    Pontuacao:
    - 1.0 se a saida for EXATAMENTE o rotulo esperado;
    - 0.5 se o rotulo certo aparece no meio de texto (categoria certa, formato
      errado) — gradiente para o GEPA primeiro acertar a categoria;
    - 0.0 se errou a categoria.
    """
    expected = _normalize(expectations.get("label", "")).strip()
    got = _normalize(outputs).strip()

    if got == expected:
        return Feedback(
            value=1.0,
            rationale=f"Correto: respondeu exatamente o rotulo '{expected.upper()}'.",
        )
    if expected and expected in got.split():
        return Feedback(
            value=0.5,
            rationale=(
                f"Categoria correta ({expected.upper()}), mas a resposta deve conter "
                "APENAS o rotulo em MAIUSCULAS, sem explicacao nem texto adicional."
            ),
        )
    return Feedback(
        value=0.0,
        rationale=(
            "Incorreto. A tarefa e CLASSIFICAR a mensagem do usuario em exatamente "
            f"UM destes rotulos: {', '.join(LABELS)}. A resposta deve ser SOMENTE o "
            f"rotulo em MAIUSCULAS, sem explicacao. Esperado: '{expected.upper()}'."
        ),
    )


# Alternativa com LLM judge (requer um modelo de judge capaz de seguir o schema
# JSON; modelos pequenos costumam falhar). Para usar, troque `scorers` abaixo:
#   from mlflow.genai.scorers import Correctness
#   from ia_observability.config import JUDGE_MODEL
#   scorers=[Correctness(model=JUDGE_MODEL)]
# e use 'expected_response' (o rotulo) no lugar de 'label' no TRAIN_DATA.


# URI da versao exata do prompt em uso (setado por _register_weak_prompt()).
# Versao explicita (ex: prompts:/nome/3) em vez de @latest: cada execucao cria
# uma nova versao e @latest seria ambiguo entre as demos.
_PROMPT_URI: str = ""


def _register_weak_prompt() -> str:
    """Registra o system prompt fraco e retorna o URI da versao exata criada."""
    global _PROMPT_URI
    pv = mlflow.genai.register_prompt(name=PROMPT_NAME, template=WEAK_SYSTEM_PROMPT)
    _PROMPT_URI = f"prompts:/{PROMPT_NAME}/{pv.version}"
    print(f"  Prompt registrado: {_PROMPT_URI}")
    return _PROMPT_URI


def predict_fn(mensagem: str) -> str:
    """Funcao de predicao avaliada pela otimizacao.

    Carrega o SYSTEM prompt do registry (o template que esta sendo otimizado) e
    o usa como mensagem de sistema; a mensagem do usuario e o texto a classificar.

    Com o system prompt vago, o modelo responde em prosa (formato errado) e nem
    sempre sabe as categorias validas -> score baixo. A otimizacao deve ensinar o
    modelo, via system prompt, a escolher a categoria certa e responder APENAS o
    rotulo. E justamente o system prompt que carrega esse conhecimento.
    """
    system_prompt = mlflow.genai.load_prompt(_PROMPT_URI).template
    completion = get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": mensagem},
        ],
    )
    return completion.choices[0].message.content


# ---------------------------------------------------------------------------
# Demo 1: Otimizacao com GEPA (few-shot)
# ---------------------------------------------------------------------------


def demo_gepa_optimization() -> None:
    """Otimiza o system prompt com GEPA aprendendo do dataset e do scorer.

    GEPA reflete sobre as falhas (via reflection_model) e gera versoes
    melhoradas do system prompt, validando com o scorer a cada iteracao.
    """
    prompt_uri = _register_weak_prompt()
    print(f"  System prompt ANTES: {WEAK_SYSTEM_PROMPT!r}")
    print(f"  Reflection model: {OPTIMIZER_JUDGE_MODEL}")
    print(f"  Dataset: {len(TRAIN_DATA)} exemplos")
    print(f"  Orcamento (max_metric_calls): {GEPA_MAX_METRIC_CALLS}")
    print("  [AVISO] A otimizacao pode levar varios minutos. Acompanhe o MLflow UI.")

    result = mlflow.genai.optimize_prompts(
        predict_fn=predict_fn,
        train_data=TRAIN_DATA,
        prompt_uris=[prompt_uri],
        optimizer=GepaPromptOptimizer(
            reflection_model=OPTIMIZER_JUDGE_MODEL,
            max_metric_calls=GEPA_MAX_METRIC_CALLS,
            display_progress_bar=True,
        ),
        scorers=[label_accuracy],
    )

    optimized = result.optimized_prompts[0]
    print(f"\n  System prompt DEPOIS ({optimized.uri}):")
    print(f"    {optimized.template[:300]}...")
    print(
        f"  Acuracia inicial -> final: "
        f"{result.initial_eval_score} -> {result.final_eval_score}"
    )


# ---------------------------------------------------------------------------
# Demo 2: Otimizacao com Metaprompting (zero-shot)
# ---------------------------------------------------------------------------


def demo_metaprompt_optimization() -> None:
    """Reestrutura o system prompt sem dados de treino (zero-shot).

    O MetaPromptOptimizer faz uma unica chamada ao reflection_model para
    reescrever o prompt seguindo boas praticas, sem train_data nem scorers.
    """
    prompt_uri = _register_weak_prompt()
    print(f"  System prompt ANTES: {WEAK_SYSTEM_PROMPT!r}")
    print(f"  Reflection model: {OPTIMIZER_JUDGE_MODEL}")
    print("  Reestruturando o prompt (zero-shot, 1 rodada)...")

    result = mlflow.genai.optimize_prompts(
        predict_fn=predict_fn,
        train_data=[],
        prompt_uris=[prompt_uri],
        optimizer=MetaPromptOptimizer(
            reflection_model=OPTIMIZER_JUDGE_MODEL,
            # Temperatura baixa: saida concisa e JSON valido. Temperatura nao e
            # um botao de "qualidade" — e de aleatoriedade. Para reescrever um
            # prompt que segue um schema rigido, temp baixa da uma saida mais
            # focada e confiavel. Com a padrao (1.0) o modelo diverga, gera um
            # prompt enorme e trunca o JSON -> a otimizacao falha e devolve o
            # original. 0.3 e um meio-termo (foco sem divagar).
            lm_kwargs={"temperature": 0.3, "max_tokens": 8192},
            guidelines=(
                "O system prompt e de um classificador de mensagens de suporte. "
                "Ele deve instruir o modelo a classificar a mensagem do usuario em "
                f"exatamente UMA destas categorias: {', '.join(LABELS)}. "
                "A resposta deve conter APENAS o rotulo em maiusculas, sem explicacao, "
                "pontuacao ou texto extra. Escreva um prompt CONCISO (poucas linhas), "
                "sem exemplos longos."
            ),
        ),
        scorers=[],
    )

    optimized = result.optimized_prompts[0]
    print(f"\n  System prompt DEPOIS ({optimized.uri}):")
    print(f"    {optimized.template[:300]}...")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Executa as demos de prompt optimization."""
    setup_mlflow(EXPERIMENT_NAME)
    patch_judge_timeout(300)

    print("=" * 60)
    print("DEMO 1: Otimizacao com GEPA (few-shot)")
    print("=" * 60)
    try:
        demo_gepa_optimization()
    except Exception as e:
        print(f"  [ERRO] Falha na otimizacao GEPA: {e}")

    print("\n" + "=" * 60)
    print("DEMO 2: Otimizacao com Metaprompting (zero-shot)")
    print("=" * 60)
    try:
        demo_metaprompt_optimization()
    except Exception as e:
        print(f"  [ERRO] Falha na otimizacao Metaprompt: {e}")

    print("\n" + "-" * 60)
    print("Abra o MLflow UI para visualizar:")
    print(f"  -> Experiment: {EXPERIMENT_NAME}")
    print(f"  - Prompt '{PROMPT_NAME}' versionado no Prompt Registry")
    print("  - Metricas de eval_score por iteracao (GEPA)")
    print("-" * 60)


if __name__ == "__main__":
    main()
