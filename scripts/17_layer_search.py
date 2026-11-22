"""Erweiterung L: Faire Layer-Suche fürs Steering, für alle drei Haupt-Traits.

    python scripts/17_layer_search.py sycophantic
    python scripts/17_layer_search.py evil
    python scripts/17_layer_search.py hallucinating

Die erste Layer-Suche (03_steering.py) nutzte auf jedem Layer denselben Koeffizienten. Weil Vektor und
Aktivierungen mit der Tiefe länger werden, war das kein fairer Vergleich. Hier ist der Eingriff auf jedem
Layer gleich stark im Verhältnis zur typischen Länge der Aktivierung dort. Als Maß dient die Einstellung,
die für Sycophancy auf Layer 22 gut funktioniert hat.

Bewertet wird mit Claude Haiku 5.5.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pv_characters import REFERENCE_LAYER, activation_norms, relative_strength, scaled_vector
from pv_common import RESULTS, load_model
from pv_judge import Judge, load_trait
from pv_steering import steered_answers

LAYERS = list(range(4, 35, 2))
OUT = RESULTS / "layer_search"


def main(trait):
    vectors = torch.load(RESULTS / "vectors" / f"{trait}_response_avg_diff.pt")
    reference = torch.load(RESULTS / "vectors" / "sycophantic_response_avg_diff.pt")[REFERENCE_LAYER]
    questions = load_trait(trait, "eval")["questions"]

    model, tok = load_model()
    torch.manual_seed(0)
    baseline = steered_answers(model, tok, None, 0, 0.0, 1, trait=trait)
    norms = activation_norms(model, tok, baseline["question"], baseline["answer"])
    strength = relative_strength(reference, norms)
    print(f"Eingriff: {strength:.3f} mal die typische Aktivierungslänge des jeweiligen Layers")
    frames = [baseline]
    for layer in LAYERS:
        vector = scaled_vector(vectors[layer], norms, layer, strength)
        frames.append(steered_answers(model, tok, vector, layer, 1.0, 1, trait=trait, questions=questions))
    del model
    torch.cuda.empty_cache()

    judge = Judge(model="claude-haiku-5-5", use_fallbacks=False)
    df = asyncio.run(judge.score_frame(pd.concat(frames, ignore_index=True), trait))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{trait}.csv", index=False)

    means = df.groupby("layer")[[trait, "coherence"]].mean().round(1)
    print(means.T.to_string())
    usable = means[(means.index > 0) & (means["coherence"] >= 50)]
    best = int(usable[trait].idxmax())
    print(f"Bester Layer für {trait} bei verständlichen Antworten: {best} (ohne Eingriff: {means.loc[0, trait]})")
    best_file = OUT / "best_layers.json"
    best_layers = json.loads(best_file.read_text()) if best_file.exists() else {}
    best_file.write_text(json.dumps(best_layers | {trait: best}))
    print(judge.usage())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trait", choices=["sycophantic", "evil", "hallucinating"])
    main(parser.parse_args().trait)
