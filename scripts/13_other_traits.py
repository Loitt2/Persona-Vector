"""Erweiterung H: Steering und Monitoring für die beiden anderen Haupt-Traits des Papers.

    python scripts/13_other_traits.py evil
    python scripts/13_other_traits.py hallucinating

Verkürzte Fassung der Experimente 3 und 4, mit dem Material des jeweiligen Traits:
  - Steering auf Layer 22 mit vier Stärken und einmal mit negativem Vorzeichen
  - Monitoring mit acht selbst formulierten, abgestuften System-Prompts

Damit die Stärken zwischen den Traits vergleichbar sind, wird jeder Vektor auf die Länge des
Sycophancy-Vektors auf Layer 22 skaliert. Eine eigene Layer-Suche gibt es hier nicht.
"""
import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pv_common import RESULTS, build_prompt, generate, load_model
from pv_data import MONITORING_SYSTEM_PROMPTS_BY_TRAIT
from pv_judge import Judge, load_trait
from pv_steering import steered_answers

LAYER = 22
COEFS = [0.0, 0.5, 1.0, 1.5, -1.0]
PROJ_LAYERS = [18, 22, 26]
N_SAMPLES = 2
JUDGE_MODEL = "claude-haiku-5-5"  # hier der günstigere Judge; stimmt mit Opus 5.5 zu r = 0,93 überein (Skript 08)
OUT = RESULTS / "other_traits"


@torch.no_grad()
def monitoring_rows(model, tok, trait, vectors):
    units = {layer: vectors[layer] / vectors[layer].norm() for layer in PROJ_LAYERS}
    rows = []
    for level, system in enumerate(MONITORING_SYSTEM_PROMPTS_BY_TRAIT[trait]):
        for q_idx, question in enumerate(load_trait(trait, "eval")["questions"]):
            prompt = build_prompt(tok, question, system)
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            hidden_states = model(**enc, output_hidden_states=True).hidden_states
            projections = {f"proj_{layer}": float(hidden_states[layer][0, -1].float().cpu() @ unit)
                           for layer, unit in units.items()}
            rows += [dict(level=level, question_id=q_idx, question=question, prompt=prompt, **projections)] * N_SAMPLES
    df = pd.DataFrame(rows)
    df["answer"] = generate(model, tok, df["prompt"].tolist(), max_new_tokens=300, batch_size=20)
    return df.drop(columns="prompt")


def main(trait):
    vectors = torch.load(RESULTS / "vectors" / f"{trait}_response_avg_diff.pt")
    reference = torch.load(RESULTS / "vectors" / "sycophantic_response_avg_diff.pt")[LAYER].norm()
    steering_vector = vectors[LAYER] / vectors[LAYER].norm() * reference

    model, tok = load_model()
    torch.manual_seed(0)
    steering = pd.concat([steered_answers(model, tok, steering_vector, LAYER, coef, N_SAMPLES, trait=trait)
                          for coef in COEFS], ignore_index=True)
    monitoring = monitoring_rows(model, tok, trait, vectors)
    del model
    torch.cuda.empty_cache()

    judge = Judge(model=JUDGE_MODEL, use_fallbacks=False)

    async def score_both():
        # Ein gemeinsamer Lauf: Der Judge begrenzt parallele Aufrufe und darf nicht über zwei asyncio.run verteilt werden
        return await asyncio.gather(judge.score_frame(steering, trait), judge.score_frame(monitoring, trait))

    steering, monitoring = asyncio.run(score_both())
    OUT.mkdir(parents=True, exist_ok=True)
    steering.to_csv(OUT / f"{trait}_steering.csv", index=False)
    monitoring.to_csv(OUT / f"{trait}_monitoring.csv", index=False)

    print(steering.groupby("coef")[[trait, "coherence"]].mean().round(1).to_string())
    columns = [f"proj_{layer}" for layer in PROJ_LAYERS]
    points = monitoring.groupby(["level", "question_id"])[columns + [trait]].mean().reset_index()
    centered = points[columns + [trait]] - points.groupby("level")[columns + [trait]].transform("mean")
    print(points.groupby("level")[[trait]].mean().round(1).T.to_string())
    for column in columns:
        print(f"Layer {column[5:]}: r gesamt {points[column].corr(points[trait]):.2f} | "
              f"innerhalb der Stufen {centered[column].corr(centered[trait]):.2f}")
    print(judge.usage())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trait", choices=["evil", "hallucinating"])
    main(parser.parse_args().trait)
