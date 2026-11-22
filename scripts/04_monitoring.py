"""Experiment 4: Monitoring. Sagt die Projektion vor der Antwort das Verhalten vorher?

Acht System-Prompts, abgestuft von "unterdrücke Schmeichelei" bis "fördere sie".
Für jede Kombination aus System-Prompt und Frage:
  1. Projektion der Aktivierung am letzten Prompt-Token auf den Vektor messen
  2. Antworten erzeugen und vom Judge bewerten lassen
Danach wird die Projektion mit dem mittleren Sycophancy-Score korreliert.
"""
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pv_common import RESULTS, build_prompt, generate, load_model
from pv_data import MONITORING_SYSTEM_PROMPTS as SYSTEM_PROMPTS
from pv_judge import Judge, load_trait

TRAIT = "sycophantic"
OUT = RESULTS / "monitoring"
N_SAMPLES = 3

PROJ_LAYERS = [14, 18, 22, 26]  # Projektion kostet nichts, deshalb auf mehreren Layern messen


@torch.no_grad()
def last_token_projections(model, tok, prompt, unit_vectors):
    """Projektion der Aktivierung am letzten Prompt-Token auf den Einheitsvektor, je Layer."""
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    hidden_states = model(**enc, output_hidden_states=True).hidden_states
    return {f"proj_{layer}": float(hidden_states[layer][0, -1].float().cpu() @ unit)
            for layer, unit in unit_vectors.items()}


def main():
    layer = json.loads((RESULTS / "steering" / "best_layer.json").read_text())["layer"]
    vectors = torch.load(RESULTS / "vectors" / f"{TRAIT}_response_avg_diff.pt")
    unit_vectors = {layer_id: vectors[layer_id] / vectors[layer_id].norm() for layer_id in PROJ_LAYERS}
    questions = load_trait(TRAIT, "eval")["questions"]
    model, tok = load_model()

    rows = []
    for level, system in enumerate(SYSTEM_PROMPTS):
        for q_idx, question in enumerate(questions):
            prompt = build_prompt(tok, question, system)
            projections = last_token_projections(model, tok, prompt, unit_vectors)
            for _ in range(N_SAMPLES):
                rows.append(dict(level=level, question_id=q_idx, question=question, prompt=prompt, **projections))
    df = pd.DataFrame(rows)
    torch.manual_seed(0)
    df["answer"] = generate(model, tok, df["prompt"].tolist(), max_new_tokens=300, batch_size=20)
    del model
    torch.cuda.empty_cache()

    judge = Judge()
    df = asyncio.run(judge.score_frame(df.drop(columns="prompt"), TRAIT))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "monitoring.csv", index=False)

    cols = [f"proj_{layer_id}" for layer_id in PROJ_LAYERS]
    points = df.groupby(["level", "question_id"])[cols + [TRAIT]].mean().reset_index()
    print(points.groupby("level")[cols + [TRAIT]].mean().round(2))
    # Innerhalb einer Stufe: Stufenmittel abziehen, dann korrelieren
    centered = points[cols + [TRAIT]] - points.groupby("level")[cols + [TRAIT]].transform("mean")
    for layer_id in PROJ_LAYERS:
        mark = "  <- Steering-Layer" if layer_id == layer else ""
        print(f"Layer {layer_id}: r gesamt = {points[f'proj_{layer_id}'].corr(points[TRAIT]):.3f} | "
              f"r innerhalb der Stufen = {centered[f'proj_{layer_id}'].corr(centered[TRAIT]):.3f}{mark}")
    print(judge.usage())


if __name__ == "__main__":
    main()
