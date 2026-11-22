"""Nachtest zu Skript 13: Lässt sich "evil" mit feineren Stärken oder auf Layer 20 doch hineinsteuern?

Skript 13 hat auf Layer 22 nur die Stärken 0,5 / 1,0 / 1,5 geprüft. Zwischen 1,0 (Score 3, Antworten klar)
und 1,5 (Score 22, Antworten zerfallen) könnte ein brauchbares Fenster liegen. Hier werden deshalb die
Zwischenstufen auf Layer 22 und zusätzlich Layer 20 gemessen. Aufbau, Fragen und Judge wie in Skript 13.
"""
import asyncio
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pv_common import RESULTS, load_model
from pv_judge import Judge
from pv_steering import steered_answers

TRAIT = "evil"
SETTINGS = [(22, 1.2), (22, 1.3), (22, 1.4), (20, 1.0), (20, 1.25), (20, 1.5)]
N_SAMPLES = 2
JUDGE_MODEL = "claude-haiku-5-5"
OUT = RESULTS / "other_traits" / "evil_retest.csv"


def main():
    vectors = torch.load(RESULTS / "vectors" / f"{TRAIT}_response_avg_diff.pt")
    reference = torch.load(RESULTS / "vectors" / "sycophantic_response_avg_diff.pt")
    model, tok = load_model()
    torch.manual_seed(0)
    frames = []
    for layer, coef in SETTINGS:
        # wie in Skript 13: Evil-Vektor auf die Länge des Sycophancy-Vektors desselben Layers skaliert
        vector = vectors[layer] / vectors[layer].norm() * reference[layer].norm()
        frames.append(steered_answers(model, tok, vector, layer, coef, N_SAMPLES, trait=TRAIT))
    answers = pd.concat(frames, ignore_index=True)
    del model
    torch.cuda.empty_cache()

    judge = Judge(model=JUDGE_MODEL, use_fallbacks=False)
    answers = asyncio.run(judge.score_frame(answers, TRAIT))
    answers.to_csv(OUT, index=False)
    print(answers.groupby(["layer", "coef"])[[TRAIT, "coherence"]].mean().round(1).to_string())
    print(judge.usage())


if __name__ == "__main__":
    main()
