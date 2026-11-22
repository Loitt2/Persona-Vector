"""Erweiterung F: Trait-Landkarte.

Voraussetzung: Für alle sieben Traits des Original-Repos liegt ein Vektor vor
(`02_extract.py generate/judge/vector --trait <name>`).

1. Kosinus-Ähnlichkeit aller Trait-Vektoren untereinander, pro Layer.
2. Die Antworten auf die neuen Fragenformate (Experiment `09_new_formats.py`) werden auf alle
   Trait-Vektoren projiziert und zusätzlich mit der Impolite-Rubrik bewertet. Damit lässt sich prüfen,
   ob deutliche Kritik für das Modell eher "unhöflich" oder eher "böse" aussieht.
"""
import asyncio
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pv_common import RESULTS, build_prompt, load_model
from pv_judge import Judge, load_trait

TRAITS = ["sycophantic", "evil", "impolite", "hallucinating", "optimistic", "humorous", "apathetic"]
LAYER = 22
OUT = RESULTS / "trait_map"


def cosine_tables(vectors):
    rows = []
    for layer in range(1, next(iter(vectors.values())).shape[0]):
        for a in TRAITS:
            for b in TRAITS:
                cosine = torch.nn.functional.cosine_similarity(vectors[a][layer], vectors[b][layer], dim=0)
                rows.append(dict(layer=layer, trait_a=a, trait_b=b, cosine=float(cosine)))
    return pd.DataFrame(rows)


@torch.no_grad()
def project_answers(df, vectors):
    """Mittlere Aktivierung jeder Antwort auf LAYER, projiziert auf jeden Trait-Vektor (Einheitslänge)."""
    model, tok = load_model()
    units = {t: vectors[t][LAYER] / vectors[t][LAYER].norm() for t in TRAITS}
    columns = {f"proj_{t}": [] for t in TRAITS}
    for question, answer in zip(df["question"], df["answer"].fillna("")):
        prompt = build_prompt(tok, question)
        enc = tok(prompt + answer, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tok.encode(prompt, add_special_tokens=False))
        mean = model(**enc, output_hidden_states=True).hidden_states[LAYER][0, prompt_len:].float().mean(0).cpu()
        for trait, unit in units.items():
            columns[f"proj_{trait}"].append(float(mean @ unit))
    for name, values in columns.items():
        df[name] = values
    return df


def main():
    vectors = {t: torch.load(RESULTS / "vectors" / f"{t}_response_avg_diff.pt") for t in TRAITS}
    OUT.mkdir(parents=True, exist_ok=True)
    cosines = cosine_tables(vectors)
    cosines.to_csv(OUT / "cosine_all_layers.csv", index=False)
    matrix = cosines[cosines["layer"] == LAYER].pivot(index="trait_a", columns="trait_b", values="cosine").loc[TRAITS, TRAITS]
    print(f"Kosinus-Ähnlichkeit auf Layer {LAYER}:")
    print(matrix.round(2).to_string())

    df = pd.read_csv(RESULTS / "new_formats" / "new_formats.csv")
    df = project_answers(df, vectors)
    judge = Judge()
    df = asyncio.run(judge.score_columns(df, {"impolite": load_trait("impolite", "eval")["eval_prompt"]}))
    df.to_csv(OUT / "new_formats_all_traits.csv", index=False)
    judged = ["correction", "sycophantic", "evil", "impolite"]
    print("Mittelwerte je Bedingung:")
    print(df.groupby("condition", sort=False)[judged + ["proj_evil", "proj_impolite"]].mean().round(1).to_string())
    print("Korrelation der Deutlichkeit der Korrektur mit Scores und Projektionen:")
    print(df[["correction", "evil", "impolite"] + [f"proj_{t}" for t in TRAITS]].corr().loc["correction"].round(2).to_string())
    print(judge.usage())


if __name__ == "__main__":
    main()
