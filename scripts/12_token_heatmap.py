"""Erweiterung G: Token-Heatmap.

Für einzelne Antworten wird Token für Token gemessen, wie weit die Aktivierung auf Layer 22 in Richtung
des Sycophancy-Vektors zeigt. Gespeichert werden Tokens und Projektionen; das Notebook färbt den Text damit ein.
Gezeigt werden zu zwei Fragen je eine Antwort mit abgezogenem und eine mit addiertem Vektor (aus Experiment 09).
"""
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pv_common import RESULTS, build_prompt, load_model

LAYER = 22
MAX_TOKENS = 400  # die ganze Antwort (beim Erzeugen auf 300 Tokens begrenzt)


@torch.no_grad()
def main():
    answers = pd.read_csv(RESULTS / "new_formats" / "new_formats.csv")
    vector = torch.load(RESULTS / "vectors" / "sycophantic_response_avg_diff.pt")[LAYER]
    unit = vector / vector.norm()
    model, tok = load_model()

    examples = []
    for fmt in ("Feedback", "falsche Behauptung"):
        # Frage mit dem größten Unterschied zwischen addiertem und abgezogenem Vektor
        part = answers[answers["format"] == fmt]
        means = part.pivot_table(index="question", columns="condition", values="sycophantic")
        question = (means["Vektor addiert"] - means["Vektor abgezogen"]).idxmax()
        for condition, pick in (("Vektor abgezogen", "idxmin"), ("Vektor addiert", "idxmax")):
            rows = part[(part["question"] == question) & (part["condition"] == condition)]
            row = rows.loc[getattr(rows["sycophantic"], pick)()]
            prompt = build_prompt(tok, question)
            prompt_len = len(tok.encode(prompt, add_special_tokens=False))
            enc = tok(prompt + row["answer"], return_tensors="pt", add_special_tokens=False).to(model.device)
            hidden = model(**enc, output_hidden_states=True).hidden_states[LAYER][0, prompt_len:prompt_len + MAX_TOKENS]
            ids = enc["input_ids"][0, prompt_len:prompt_len + MAX_TOKENS].tolist()
            examples.append(dict(
                format=fmt, question=question, condition=condition, sycophantic=float(row["sycophantic"]),
                tokens=[tok.decode([i]) for i in ids], projections=(hidden.float().cpu() @ unit).tolist(),
            ))
            print(f"{fmt} | {condition} | Score {row['sycophantic']:.0f} | mittlere Projektion {sum(examples[-1]['projections']) / len(ids):.1f}")

    out = RESULTS / "token_heatmap"
    out.mkdir(parents=True, exist_ok=True)
    (out / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
