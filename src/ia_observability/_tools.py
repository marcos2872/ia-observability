"""Funcoes de ferramentas compartilhadas entre tool_calls.py e langchain_agent.py.

Contem apenas a logica de negocios das tools, sem decoradores de framework.
Cada modulo consumidor adiciona seu proprio decorador (@tool do LangChain ou
funcao plain do SDK OpenAI) e faz o import daqui.

Importacao:
    from ia_observability._tools import get_weather, search_docs, calculate, check_inventory
"""

import ast
import time

# Dados simulados (hardcoded para demo)
_WEATHER_DATA = {
    "sao paulo": {"temp": 22, "unit": "celsius", "condition": "chuva leve"},
    "campinas": {"temp": 28, "unit": "celsius", "condition": "ensolarado"},
    "lisboa": {"temp": 18, "unit": "celsius", "condition": "nublado"},
    "new york": {"temp": 72, "unit": "fahrenheit", "condition": "clear sky"},
    "tokyo": {"temp": 15, "unit": "celsius", "condition": "chuva"},
}

_DOCS_DATA = [
    {
        "titulo": "Introducao ao MLflow Tracing",
        "conteudo": "MLflow Tracing permite rastrear chamadas a LLMs automaticamente com mlflow.openai.autolog()",
        "tags": ["mlflow", "tracing"],
    },
    {
        "titulo": "Instalacao do MLflow",
        "conteudo": "pip install mlflow[genai] para suporte a LLM tracing e evaluation",
        "tags": ["instalacao", "mlflow"],
    },
    {
        "titulo": "Custom Scorers",
        "conteudo": "Use @mlflow.metrics.genai.scorer para criar metricas customizadas de avaliacao",
        "tags": ["evaluation", "scorers"],
    },
    {
        "titulo": "MLflow Experiments",
        "conteudo": "mlflow.create_experiment() e mlflow.set_experiment() gerenciam experimentos",
        "tags": ["mlflow", "experimentos"],
    },
    {
        "titulo": "Deep Learning com PyTorch",
        "conteudo": "PyTorch oferece autograd e redes neurais modulares para deep learning",
        "tags": ["pytorch", "deep learning"],
    },
]


def _safe_eval(expression: str) -> int | float:
    """Avalia expressoes aritmeticas de forma segura usando AST whitelisting."""
    ALLOWED_NODES = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
        ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd,
    }
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            raise ValueError(f"Node type not allowed: {type(node).__name__}")
    return eval(  # noqa: S307 - AST whitelist aprovou seguranca
        compile(tree, "<string>", "eval"),
        {"__builtins__": {}},
        {},
    )


def get_weather(city: str, unit: str = "celsius") -> dict:
    """Retorna clima atual para uma cidade (dados simulados)."""
    city_lower = city.lower().strip()
    if city_lower not in _WEATHER_DATA:
        return {"error": f"Cidade '{city}' nao encontrada", "cidade": city}
    return _WEATHER_DATA[city_lower]


def search_docs(query: str, max_results: int = 3) -> list[dict]:
    """Busca documentos na base de conhecimento."""
    results = []
    query_lower = query.lower()
    for doc in _DOCS_DATA:
        score = 0
        if query_lower in doc["titulo"].lower():
            score += 3
        if query_lower in doc["conteudo"].lower():
            score += 2
        for tag in doc["tags"]:
            if query_lower in tag.lower():
                score += 1
        if score > 0:
            results.append({**doc, "score": score})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max_results]


def calculate(expression: str) -> dict:
    """Executa um calculo aritmetico simples."""
    try:
        result = _safe_eval(expression)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e), "expression": expression}


def check_inventory(product: str) -> dict:
    """Simula consulta a API de estoque que falha (timeout/erro)."""
    time.sleep(2.5)
    raise TimeoutError(
        f"Timeout ao consultar estoque do produto '{product}' — API indisponivel"
    )
