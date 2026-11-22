"""Erweiterung D: Neue Fragenformate und der Zusammenhang mit "evil".

Zwei Fragen:
  1. Wirkt der Sycophancy-Vektor auch, wenn das Modell eine falsche Behauptung korrigieren oder
     schwache Arbeit kritisieren müsste? (Die Evaluationsfragen des Papers sind reine Meinungsfragen.)
  2. Sieht Kritik für das Modell "böse" aus? Dafür wird jede Antwort zusätzlich mit der Evil-Rubrik
     bewertet und ihre Aktivierung auf den Evil-Vektor projiziert.

Bedingungen auf Layer 22: ohne Eingriff, Sycophancy-Vektor addiert, Sycophancy-Vektor abgezogen.
"""
import asyncio
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pv_common import RESULTS, build_prompt, load_model
from pv_data import CORRECTION_RUBRIC, FALSE_CLAIM_QUESTIONS, FEEDBACK_QUESTIONS
from pv_judge import Judge, coherence_rubric, load_trait
from pv_steering import steered_answers

LAYER = 22
CONDITIONS = [("ohne Eingriff", 0.0), ("Vektor addiert", 1.0), ("Vektor abgezogen", -1.5)]
N_SAMPLES = 3
OUT = RESULTS / "new_formats"


@torch.no_grad()
def response_projections(model, tok, df, unit_vectors):
    """Mittlere Aktivierung der Antwort auf LAYER, projiziert auf die Einheitsvektoren. Ohne Steering gemessen."""
    columns = {name: [] for name in unit_vectors}
    for question, answer in zip(df["question"], df["answer"]):
        prompt = build_prompt(tok, question)
        enc = tok(prompt + answer, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tok.encode(prompt, add_special_tokens=False))
        mean = model(**enc, output_hidden_states=True).hidden_states[LAYER][0, prompt_len:].float().mean(0).cpu()
        for name, unit in unit_vectors.items():
            columns[name].append(float(mean @ unit))
    return columns


def main():
    questions = FALSE_CLAIM_QUESTIONS + FEEDBACK_QUESTIONS
    kind = {q: "falsche Behauptung" for q in FALSE_CLAIM_QUESTIONS} | {q: "Feedback" for q in FEEDBACK_QUESTIONS}
    sycophancy = torch.load(RESULTS / "vectors" / "sycophantic_response_avg_diff.pt")[LAYER]
    evil = torch.load(RESULTS / "vectors" / "evil_response_avg_diff.pt")[LAYER]

    model, tok = load_model()
    torch.manual_seed(0)
    df = pd.concat([steered_answers(model, tok, sycophancy, LAYER, coef, N_SAMPLES, label=label, questions=questions)
                    for label, coef in CONDITIONS], ignore_index=True)
    df["format"] = df["question"].map(kind)
    units = {"proj_sycophantic": sycophancy / sycophancy.norm(), "proj_evil": evil / evil.norm()}
    for name, values in response_projections(model, tok, df, units).items():
        df[name] = values
    del model
    torch.cuda.empty_cache()

    judge = Judge()
    rubrics = {
        "sycophantic": load_trait("sycophantic", "eval")["eval_prompt"],
        "coherence": coherence_rubric(),
        "evil": load_trait("evil", "eval")["eval_prompt"],
        "correction": CORRECTION_RUBRIC,
    }
    df = asyncio.run(judge.score_columns(df, rubrics))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "new_formats.csv", index=False)

    metrics = ["sycophantic", "correction", "evil", "coherence", "proj_sycophantic", "proj_evil"]
    print(df.groupby(["format", "condition"], sort=False)[metrics].mean().round(1).to_string())
    print("Korrelationen über alle Antworten:")
    print(df[metrics].corr().round(2).loc[["correction", "sycophantic"], ["evil", "proj_evil", "proj_sycophantic"]].to_string())
    print(judge.usage())


if __name__ == "__main__":
    main()
