"""Erweiterung: Wird Darth Vader wirklich böse, wenn man einen Evil-Vektor ohne Rollen-Anteil addiert?

Skript 24 hat gezeigt, dass der Evil-Vektor zu einem großen Teil aus "das Modell spielt eine Rolle" besteht (Kosinus 0,68).
Das könnte erklären, warum Steering mit ihm das Modell eher wirr als böse macht. Hier wird deshalb verglichen:

  - Evil-Vektor wie berechnet
  - Evil-Vektor ohne den Anteil in Richtung des Rollen-Vektors (aus Skript 24)

Beide werden addiert, in zwei Stärken, einmal bei Darth Vader (der die Rolle schon spielt) und einmal beim normalen Assistenten.
Beide Vektoren haben dieselbe Länge wie die Eingriffe in den Skripten 16 und 22. Zehn Alltagsfragen, je 40 Antworten, Judge Claude Haiku 5.5.
"""
import asyncio
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pv_characters import REFERENCE_LAYER, activation_norms, relative_strength, scaled_vector, system_prompt
from pv_common import RESULTS, ActivationSteerer, build_prompt, generate, load_model
from pv_data import NEUTRAL_QUESTIONS, WHO_RUBRIC
from pv_judge import Judge

LAYER = 22
N_SAMPLES = 4
OUT = RESULTS / "characters"


def play(model, tok, character, vector=None):
    questions = list(NEUTRAL_QUESTIONS) * N_SAMPLES
    prompts = [build_prompt(tok, question, system_prompt(character)) for question in questions]
    if vector is None:
        answers = generate(model, tok, prompts, max_new_tokens=200, batch_size=40)
    else:
        with ActivationSteerer(model, vector, coeff=1.0, layer_idx=LAYER - 1, positions="response"):
            answers = generate(model, tok, prompts, max_new_tokens=200, batch_size=40)
    return questions, answers


def main():
    evil = torch.load(RESULTS / "vectors" / "evil_response_avg_diff.pt")[LAYER]
    role = torch.load(RESULTS / "vectors" / "role_vector_layer22.pt")
    role_unit = role / role.norm()
    evil_clean = evil - (evil @ role_unit) * role_unit
    reference = torch.load(RESULTS / "vectors" / "sycophantic_response_avg_diff.pt")[REFERENCE_LAYER]
    print(f"Anteil des Evil-Vektors, der nach dem Entfernen der Rolle bleibt: {float(evil_clean.norm() / evil.norm()):.2f} seiner Länge")

    model, tok = load_model()
    torch.manual_seed(4)
    _, plain = play(model, tok, "Assistant")
    norms = activation_norms(model, tok, list(NEUTRAL_QUESTIONS) * N_SAMPLES, plain)
    strength = relative_strength(reference, norms)
    vectors = {"Evil-Vektor": scaled_vector(evil, norms, LAYER, strength), "Evil ohne Rolle": scaled_vector(evil_clean, norms, LAYER, strength)}

    frames = [pd.DataFrame(dict(character="Assistant", condition="ohne Eingriff", question=list(NEUTRAL_QUESTIONS) * N_SAMPLES, answer=plain))]
    for character in ("Assistant", "Darth Vader"):
        if character == "Darth Vader":
            questions, answers = play(model, tok, character)
            frames.append(pd.DataFrame(dict(character=character, condition="ohne Eingriff", question=questions, answer=answers)))
        for name, vector in vectors.items():
            for factor, label in ((1.0, "1,0"), (1.5, "1,5")):
                questions, answers = play(model, tok, character, factor * vector)
                frames.append(pd.DataFrame(dict(character=character, condition=f"{name} {label}", question=questions, answer=answers)))
    del model
    torch.cuda.empty_cache()

    df = pd.concat(frames, ignore_index=True)
    judge = Judge(model="claude-haiku-5-5", use_fallbacks=False)

    async def score_all():
        await asyncio.gather(judge.score_frame(df, "evil"), judge.score_columns(df, {"anakin": WHO_RUBRIC}))

    asyncio.run(score_all())
    df.to_csv(OUT / "evil_without_role.csv", index=False)
    print(df.groupby(["character", "condition"], sort=False)[["evil", "anakin", "coherence"]].mean().round(1).to_string())
    print(judge.usage())


if __name__ == "__main__":
    main()
