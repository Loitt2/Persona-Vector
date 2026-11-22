"""Experimente 2 und 3: Layer-Suche und Steering-Kurve.

    python scripts/03_steering.py layers   # mit festem Koeffizienten auf mehreren Layern steuern
    python scripts/03_steering.py curve    # auf dem besten Layer den Koeffizienten variieren

Gesteuert wird ohne System-Prompt: Auf dem gewählten Layer wird bei jedem
erzeugten Token coef * Vektor auf die Aktivierung addiert.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pv_common import RESULTS, load_model
from pv_judge import Judge
from pv_steering import steered_answers

TRAIT = "sycophantic"
OUT = RESULTS / "steering"
VECTOR = RESULTS / "vectors" / f"{TRAIT}_response_avg_diff.pt"

SWEEP_LAYERS = [6, 10, 14, 18, 22, 26, 30]
SWEEP_COEF = 2.0
SWEEP_SAMPLES = 2
CURVE_LAYERS = [18, 22]  # 18: bester Layer der Suche; 22: stärkster Effekt, aber bei coef 2 nicht mehr kohärent
CURVE_COEFS = [0.5, 1.0, 1.5, 2.0, 2.5]
CURVE_SAMPLES = 3


def run(conditions, n_samples, out_name):
    model, tok = load_model()
    vectors = torch.load(VECTOR)
    torch.manual_seed(0)
    frames = [steered_answers(model, tok, vectors[layer], layer, coef, n_samples) for layer, coef in conditions]
    del model
    torch.cuda.empty_cache()
    df = pd.concat(frames, ignore_index=True)
    judge = Judge()
    df = asyncio.run(judge.score_frame(df, TRAIT))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / out_name, index=False)
    print(df.groupby(["layer", "coef"])[[TRAIT, "coherence"]].mean().round(1))
    print(judge.usage())
    return df


def stage_layers():
    df = run([(layer, SWEEP_COEF) for layer in SWEEP_LAYERS], SWEEP_SAMPLES, "layer_sweep.csv")
    means = df.groupby("layer")[[TRAIT, "coherence"]].mean()
    usable = means[means["coherence"] >= 50]  # wirre Antworten zählen nicht als Steering-Erfolg
    best = int((usable if len(usable) else means)[TRAIT].idxmax())
    (OUT / "best_layer.json").write_text(json.dumps({"layer": best, "sweep_coef": SWEEP_COEF}))
    print("bester Layer:", best)


def stage_curve():
    # Eine gemeinsame Basislinie ohne Eingriff (coef 0), dann jede Stärke auf jedem Layer
    conditions = [(CURVE_LAYERS[0], 0.0)] + [(layer, coef) for layer in CURVE_LAYERS for coef in CURVE_COEFS]
    run(conditions, CURVE_SAMPLES, "steering_curve.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["layers", "curve"])
    stage_layers() if parser.parse_args().stage == "layers" else stage_curve()
