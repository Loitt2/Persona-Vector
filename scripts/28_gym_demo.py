"""Beispiel für den Vortrag: eine Alltagsfrage, beantwortet ohne Eingriff und mit drei Persona-Vektoren.

Frage: "Ich war gestern nicht im Fitnessstudio, obwohl ich es geplant hatte. Bin ich ein Versager?"
Gesteuert wird auf Layer 22 mit moderater Stärke. Wie in Skript 13 sind alle Vektoren auf die Länge des
Sycophancy-Vektors skaliert, damit die Stärken vergleichbar sind. Je Bedingung drei Antworten, auf Englisch und auf Deutsch.
Es wird nichts bewertet; die Datei dient nur als Quelle echter Beispielantworten.
"""
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pv_common import RESULTS, ActivationSteerer, build_prompt, generate, load_model

LAYER = 22
N_SAMPLES = 3
QUESTIONS = {"en": "I didn't go to the gym yesterday even though I had planned to. Am I a failure?",
             "de": "Ich war gestern nicht im Fitnessstudio, obwohl ich es geplant hatte. Bin ich ein Versager?"}
CONDITIONS = [("ohne Eingriff", None, 0.0), ("sycophantic", "sycophantic", 1.0), ("hallucinating", "hallucinating", 1.0),
              ("evil", "evil", 1.0), ("evil", "evil", 1.5)]
OUT = RESULTS / "demo"


def main():
    reference = torch.load(RESULTS / "vectors" / "sycophantic_response_avg_diff.pt")[LAYER].norm()
    model, tok = load_model()
    torch.manual_seed(7)
    rows = []
    for language, question in QUESTIONS.items():
        prompts = [build_prompt(tok, question)] * N_SAMPLES
        for name, trait, coef in CONDITIONS:
            if trait is None:
                answers = generate(model, tok, prompts, max_new_tokens=160)
            else:
                vector = torch.load(RESULTS / "vectors" / f"{trait}_response_avg_diff.pt")[LAYER]
                vector = vector / vector.norm() * reference
                with ActivationSteerer(model, vector, coeff=coef, layer_idx=LAYER - 1, positions="response"):
                    answers = generate(model, tok, prompts, max_new_tokens=160)
            rows += [dict(sprache=language, bedingung=name, staerke=coef, frage=question, antwort=answer) for answer in answers]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "gym_frage.csv", index=False)
    for row in rows:
        print(f"--- {row['sprache']} | {row['bedingung']} {row['staerke']}")
        print(" ".join(row["antwort"].split())[:330])


if __name__ == "__main__":
    main()
