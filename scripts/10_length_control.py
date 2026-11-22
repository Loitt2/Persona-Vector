"""Erweiterung E: Steuert ein längenkontrollierter Vektor genauso?

Der ursprüngliche Vektor mittelt über die ganze Antwort, und ehrliche Antworten sind doppelt so lang.
Hier wird mit einem Vektor gesteuert, der nur aus den ersten 50 Antwort-Tokens gebildet wurde
(`02_extract.py vector --first-tokens 50`). Er wird auf die Länge des ursprünglichen Vektors skaliert,
damit derselbe Koeffizient derselbe Eingriff ist.
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

TRAIT = "sycophantic"
SETTINGS = [(18, 2.0), (22, 1.0)]  # Einstellungen mit Kohärenz um 74 in der Steering-Kurve
N_SAMPLES = 3


def main():
    full = torch.load(RESULTS / "vectors" / f"{TRAIT}_response_avg_diff.pt")
    first50 = torch.load(RESULTS / "vectors" / f"{TRAIT}_response_first50_diff.pt")
    cosine = torch.nn.functional.cosine_similarity(full[1:], first50[1:], dim=-1)
    print("Kosinus-Ähnlichkeit pro Layer:", [round(x, 2) for x in cosine.tolist()])

    model, tok = load_model()
    torch.manual_seed(0)
    frames = []
    for layer, coef in SETTINGS:
        matched = first50[layer] / first50[layer].norm() * full[layer].norm()
        frames.append(steered_answers(model, tok, matched, layer, coef, N_SAMPLES, label="erste 50 Tokens"))
    del model
    torch.cuda.empty_cache()

    judge = Judge()
    df = asyncio.run(judge.score_frame(pd.concat(frames, ignore_index=True), TRAIT))
    df.to_csv(RESULTS / "steering" / "length_control.csv", index=False)
    print(df.groupby(["layer", "coef"])[[TRAIT, "coherence"]].mean().round(1))
    print(judge.usage())


if __name__ == "__main__":
    main()
