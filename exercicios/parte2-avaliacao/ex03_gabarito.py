"""Gabarito: Exercício 03 — Scorer de presença de código Python."""
from mlflow.entities.assessment import Feedback
from mlflow.genai.scorers import scorer


@scorer
def contains_python_code(inputs, outputs) -> Feedback:
    """Verifica se a resposta contém blocos de código Python."""
    if outputs is None:
        return Feedback(value=False, rationale="Sem resposta.")
    output_str = str(outputs)
    count = output_str.count("```python")
    return Feedback(
        value=count > 0,
        rationale=(
            f"Encontrados {count} blocos de código Python."
            if count > 0
            else "Nenhum bloco de código Python encontrado."
        ),
    )


# Para usar, adicione 'contains_python_code' na lista de scorers no main() de judges.py:
# scorers=[..., contains_python_code]
