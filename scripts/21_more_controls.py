"""Nachträge zu Steering-Kurve und Kontrollen (Skripte 03 und 05).

1. Layer 18 mit höheren Stärken (3,0 bis 5,0): Die Kurve aus Skript 03 endet bei 2,5, wo die Antworten auf Layer 18
   noch verständlich sind. Hier wird weitergedreht, bis auch Layer 18 kippt.
2. 19 weitere Zufallsvektoren je Layer (Seeds 6 bis 24), zusammen mit den sechs vorhandenen also 25.

Aufbau, Fragen und Judge wie in den Skripten 03 und 05.
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
MORE_COEFS_LAYER_18 = [3.0, 3.5, 4.0, 5.0]
SETTINGS = [(18, 2.5), (22, 1.5)]  # wie in Skript 05
EXTRA_SEEDS = range(6, 25)
OUT = RESULTS / "steering"


def random_like(vector, generator):
    """Zufällige Richtung mit derselben Länge wie `vector` (wie in Skript 05)."""
    random_vector = torch.randn(vector.shape, generator=generator)
    return random_vector / random_vector.norm() * vector.norm()


def main():
    model, tok = load_model()
    vectors = torch.load(RESULTS / "vectors" / f"{TRAIT}_response_avg_diff.pt")
    torch.manual_seed(0)
    curve = [steered_answers(model, tok, vectors[18], 18, coef, 3) for coef in MORE_COEFS_LAYER_18]
    randoms = []
    for seed in EXTRA_SEEDS:
        generator = torch.Generator().manual_seed(seed)
        for layer, coef in SETTINGS:
            randoms.append(steered_answers(model, tok, random_like(vectors[layer], generator), layer, coef, 2, label=f"zufall_{seed}"))
    del model
    torch.cuda.empty_cache()
    curve, randoms = pd.concat(curve, ignore_index=True), pd.concat(randoms, ignore_index=True)
    # Rohantworten sofort sichern, falls die Bewertung abbricht
    curve.to_csv(OUT / "steering_curve_layer18_mehr.csv", index=False)
    randoms.to_csv(OUT / "controls_random_mehr.csv", index=False)

    judge = Judge()

    async def score_both():
        return await asyncio.gather(judge.score_frame(curve, TRAIT), judge.score_frame(randoms, TRAIT))

    curve_scored, randoms_scored = asyncio.run(score_both())
    curve_scored.to_csv(OUT / "steering_curve_layer18_mehr.csv", index=False)
    randoms_scored.to_csv(OUT / "controls_random_mehr.csv", index=False)
    print(curve_scored.groupby("coef")[[TRAIT, "coherence"]].mean().round(1).to_string())
    print(randoms_scored.groupby(["layer", "condition"])[TRAIT].mean().unstack("layer").round(1).to_string())
    print(judge.usage())


if __name__ == "__main__":
    main()
