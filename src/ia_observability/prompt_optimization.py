"""Demonstracao de Prompt Optimization: otimizar o SYSTEM prompt automaticamente.

O MLflow otimiza um prompt registrado aprendendo com dados de avaliacao e
metricas (scorers). Aqui otimizamos o SYSTEM prompt — que controla o
comportamento/qualidade das respostas — mantendo a pergunta do usuario fixa.

Padrao:
    predict_fn(topic) -> usa o system prompt (otimizavel) + mensagem do usuario
    optimize_prompts(predict_fn, train_data, prompt_uris, optimizer, scorers)

Dois algoritmos sao demonstrados:
- GEPA (GepaPromptOptimizer): refina o prompt iterativamente via reflexao do
  LLM sobre falhas. Precisa de train_data e scorers (few-shot).
- Metaprompting (MetaPromptOptimizer): reestrutura o prompt seguindo boas
  praticas. Funciona em zero-shot (sem dados nem scorers).

Sobre os scorers: usamos um scorer CODE-BASED (deterministico) que mede a
cobertura de termos tecnicos esperados. Vantagem didatica: nao depende de um
LLM judge (que, com modelos pequenos, costuma devolver JSON fora do schema e
quebrar a otimizacao). Para usar um LLM judge, veja o comentario em SCORERS.

A reflexao do GEPA usa 'openai:/' do litellm apontando para o AI Gateway
(OpenAI-compatible), configurado em config.py.

Referencia: https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/
"""

import os
import unicodedata

import mlflow
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
# avaliacao baseline do prompt inicial e as rodadas de reflexao+mutacao (gerar
# um prompt candidato e reavaliar), parando quando o orcamento acaba.
# Mais orcamento = mais rodadas = melhor resultado, porem mais lento.
GEPA_MAX_METRIC_CALLS: int = int(os.getenv("GEPA_MAX_METRIC_CALLS", "6"))

# SYSTEM prompt propositalmente fraco/vago — a otimizacao deve melhora-lo.
# Nao tem variaveis: o {{topic}} vai na mensagem do usuario (fixa).
WEAK_SYSTEM_PROMPT = "Voce e um assistente."

# Dataset de treino: inputs (kwargs do predict_fn) + expectations (ground truth).
# 'keywords' = termos tecnicos que uma boa resposta deve cobrir. O scorer
# code-based mede a fracao desses termos presentes (ignorando acento/caixa).
# Dataset mais rico = sinal mais forte para a otimizacao melhorar o prompt.
TRAIN_DATA = [
    {
        "inputs": {"topic": "tracing em aplicacoes LLM"},
        "expectations": {
            "keywords": ["span", "trace", "latencia", "input", "output", "instrumentacao"]
        },
    },
    {
        "inputs": {"topic": "MLflow Prompt Registry"},
        "expectations": {
            "keywords": ["versionar", "prompt", "registro", "rastreabilidade", "template"]
        },
    },
    {
        "inputs": {"topic": "observabilidade em sistemas de IA"},
        "expectations": {
            "keywords": ["latencia", "tokens", "custo", "producao", "metricas", "qualidade"]
        },
    },
    {
        "inputs": {"topic": "LLM-as-a-judge para avaliacao"},
        "expectations": {
            "keywords": ["judge", "avaliacao", "criterio", "score", "rationale"]
        },
    },
    {
        "inputs": {"topic": "deteccao de alucinacao em LLMs"},
        "expectations": {
            "keywords": ["alucinacao", "factualidade", "groundedness", "contexto", "evidencia"]
        },
    },
    {
        "inputs": {"topic": "controle de custo de tokens em producao"},
        "expectations": {
            "keywords": ["tokens", "custo", "cache", "amostragem", "throughput"]
        },
    },
]


# ---------------------------------------------------------------------------
# Scorer code-based: cobertura de termos tecnicos (deterministico, sem judge)
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Remove acentos e baixa a caixa, para comparacao robusta de termos."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


@scorer
def keyword_coverage(outputs, expectations) -> float:
    """Fracao (0..1) dos termos tecnicos esperados presentes na resposta.

    Scorer deterministico: nao usa LLM, entao e rapido e nunca quebra por
    JSON malformado. A comparacao ignora acento e caixa (ex: 'latencia' casa
    com 'Latencia' e 'latencia'). O GEPA tenta MAXIMIZAR essa metrica,
    empurrando o system prompt para gerar respostas mais tecnicas e completas.
    """
    keywords = expectations.get("keywords", [])
    if not keywords:
        return 1.0
    text = _normalize(outputs or "")
    hits = sum(1 for kw in keywords if _normalize(kw) in text)
    return hits / len(keywords)


# Alternativa com LLM judge (requer um modelo de judge capaz de seguir o schema
# JSON; modelos pequenos costumam falhar). Para usar, troque `scorers` abaixo:
#   from mlflow.genai.scorers import Correctness
#   from ia_observability.config import JUDGE_MODEL
#   scorers=[Correctness(model=JUDGE_MODEL)]
# e use 'expected_facts' (lista de fatos) no lugar de 'keywords' no TRAIN_DATA.


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


def predict_fn(topic: str) -> str:
    """Funcao de predicao avaliada pela otimizacao.

    Carrega o SYSTEM prompt do registry (o template que esta sendo otimizado) e
    o usa como mensagem de sistema. A pergunta do usuario e fixa e carrega o
    topico. A otimizacao testa diferentes versoes do system prompt aqui.
    """
    system_prompt = mlflow.genai.load_prompt(_PROMPT_URI).template
    completion = get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Explique de forma tecnica: {topic}"},
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
        scorers=[keyword_coverage],
    )

    optimized = result.optimized_prompts[0]
    print(f"\n  System prompt DEPOIS ({optimized.uri}):")
    print(f"    {optimized.template[:300]}...")
    print(f"  Score (cobertura) inicial -> final: "
          f"{result.initial_eval_score} -> {result.final_eval_score}")


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
            guidelines=(
                "O system prompt e usado por um assistente tecnico de MLOps que "
                "responde em portugues brasileiro, de forma tecnica e sem emojis."
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
