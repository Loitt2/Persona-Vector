"""Kontrollen zum Steering.

    python scripts/05_controls.py           # ein Zufallsvektor und negatives Vorzeichen
    python scripts/05_controls.py random    # fünf weitere Zufallsvektoren pro Layer

1. Zufallsvektor gleicher Länge: Steigt Sycophancy auch bei einer beliebigen Richtung?
   Wenn ja, wäre der Effekt nicht spezifisch für den Persona-Vektor.
2. Negatives Vorzeichen: Sinkt Sycophancy unter die Basislinie, wenn der Vektor abgezogen wird?
"""
import argparse
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
# (Layer, Stärke): die stärksten Einstellungen, bei denen die Antworten noch verständlich waren
SETTINGS = [(18, 2.5), (22, 1.5)]
EXTRA_SEEDS = [1, 2, 3, 4, 5]


def random_like(vector, generator):
    """Zufällige Richtung mit derselben Länge wie `vector`."""
    random_vector = torch.randn(vector.shape, generator=generator)
    return random_vector / random_vector.norm() * vector.norm()


def judge_and_save(frames, name):
    judge = Judge()
    df = asyncio.run(judge.score_frame(pd.concat(frames, ignore_index=True), TRAIT))
    df.to_csv(RESULTS / "steering" / name, index=False)
    print(df.groupby(["condition", "layer", "coef"])[[TRAIT, "coherence"]].mean().round(1))
    print(judge.usage())


def main(mode):
    model, tok = load_model()
    vectors = torch.load(RESULTS / "vectors" / f"{TRAIT}_response_avg_diff.pt")
    torch.manual_seed(0)
    frames = []
    if mode == "random":
        for seed in EXTRA_SEEDS:
            generator = torch.Generator().manual_seed(seed)
            for layer, coef in SETTINGS:
                vector = random_like(vectors[layer], generator)
                frames.append(steered_answers(model, tok, vector, layer, coef, 2, label=f"zufall_{seed}"))
        name = "controls_random.csv"
    else:
        generator = torch.Generator().manual_seed(0)
        for layer, coef in SETTINGS:
            frames.append(steered_answers(model, tok, random_like(vectors[layer], generator), layer, coef, 3, label="zufall"))
            frames.append(steered_answers(model, tok, vectors[layer], layer, -coef, 3, label="negativ"))
        name = "controls.csv"
    del model
    torch.cuda.empty_cache()
    judge_and_save(frames, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="basis", choices=["basis", "random"])
    main(parser.parse_args().mode)
