"""Gesteuerte Generierung: Antworten mit Eingriff in die Aktivierungen."""
import pandas as pd

from pv_common import ActivationSteerer, build_prompt, generate
from pv_judge import load_trait


def steered_answers(model, tok, vector, layer, coef, n_samples, trait="sycophantic", label=None, questions=None):
    """Antworten auf Fragen, gesteuert auf `layer` mit Stärke `coef`.

    Ohne `questions` werden die 20 Evaluationsfragen des Traits verwendet.
    `vector` gehört zu hidden_states[layer], also zur Ausgabe von Block layer-1.
    Bei jedem erzeugten Token wird dort coef * vector auf die Aktivierung addiert.
    """
    if questions is None:
        questions = load_trait(trait, "eval")["questions"]
    questions = list(questions) * n_samples
    prompts = [build_prompt(tok, q) for q in questions]
    if coef == 0:
        answers = generate(model, tok, prompts, max_new_tokens=300, batch_size=20)
    else:
        with ActivationSteerer(model, vector, coeff=coef, layer_idx=layer - 1, positions="response"):
            answers = generate(model, tok, prompts, max_new_tokens=300, batch_size=20)
    df = pd.DataFrame(dict(question=questions, answer=answers, layer=layer, coef=coef))
    if label is not None:
        df["condition"] = label
    return df
