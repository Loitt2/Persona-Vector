"""Erweiterung B: Monitoring über alle Layer und mit zwei Vektoren.

Nutzt die Scores aus Experiment 4 (keine neuen Antworten, kein Judge). Für die 160 Kombinationen
aus System-Prompt und Frage wird die Aktivierung am letzten Prompt-Token auf jedem Layer gemessen
und auf zwei Vektoren projiziert:
  - response_avg: gemittelt über die Antwort-Tokens (der Vektor aus dem Paper)
  - prompt_last:  gebildet am letzten Prompt-Token, also an der Stelle, an der hier gemessen wird
"""
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pv_common import RESULTS, build_prompt, load_model
from pv_data import MONITORING_SYSTEM_PROMPTS
from pv_judge import load_trait

TRAIT = "sycophantic"


@torch.no_grad()
def main():
    scores = pd.read_csv(RESULTS / "monitoring" / "monitoring.csv")
    points = scores.groupby(["level", "question_id"])[TRAIT].mean().reset_index()
    questions = load_trait(TRAIT, "eval")["questions"]
    model, tok = load_model()

    activations = []  # je Punkt: [Layer+1, hidden] am letzten Prompt-Token
    for row in points.itertuples():
        prompt = build_prompt(tok, questions[row.question_id], MONITORING_SYSTEM_PROMPTS[row.level])
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        hidden_states = model(**enc, output_hidden_states=True).hidden_states
        activations.append(torch.stack([h[0, -1] for h in hidden_states]).float().cpu())
    activations = torch.stack(activations)  # [160, Layer+1, hidden]

    rows = []
    for name in ("response_avg", "prompt_last"):
        vectors = torch.load(RESULTS / "vectors" / f"{TRAIT}_{name}_diff.pt")
        unit = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        projections = (activations * unit[None]).sum(-1)  # [160, Layer+1]
        for layer in range(1, projections.shape[1]):
            frame = points.assign(projection=projections[:, layer].numpy())
            centered = frame[["projection", TRAIT]] - frame.groupby("level")[["projection", TRAIT]].transform("mean")
            rows.append(dict(vektor=name, layer=layer,
                             r_gesamt=frame["projection"].corr(frame[TRAIT]),
                             r_innerhalb=centered["projection"].corr(centered[TRAIT])))
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "monitoring" / "monitoring_all_layers.csv", index=False)
    print(result.pivot(index="layer", columns="vektor", values=["r_gesamt", "r_innerhalb"]).round(2).to_string())


if __name__ == "__main__":
    main()
