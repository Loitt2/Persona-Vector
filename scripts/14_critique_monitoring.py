"""Erweiterung I: Sieht faire Kritik im Modell anders aus als gemeine?

Fünf System-Prompts von "lobe den Nutzer" über "gib faire, ehrliche Kritik" bis "sei hart und abwertend",
kombiniert mit den 20 Fragen, bei denen Kritik angebracht ist (falsche Behauptungen, schwache Arbeit).

Gemessen wird wie beim Monitoring: die Aktivierung am letzten Prompt-Token auf Layer 22, also bevor das Modell
antwortet, projiziert auf die Vektoren aller sieben Traits. Zusätzlich dieselbe Projektion gemittelt über die Antwort.
Der Judge bewertet die Antworten, damit sich prüfen lässt, ob die Anweisungen gewirkt haben.
"""
import asyncio
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pv_common import RESULTS, build_prompt, generate, load_model
from pv_data import CORRECTION_RUBRIC, CRITIQUE_SYSTEM_PROMPTS, FALSE_CLAIM_QUESTIONS, FEEDBACK_QUESTIONS
from pv_judge import Judge, load_trait

TRAITS = ["sycophantic", "evil", "impolite", "hallucinating", "optimistic", "humorous", "apathetic"]
LAYER = 22
OUT = RESULTS / "critique"


@torch.no_grad()
def main():
    vectors = torch.stack([torch.load(RESULTS / "vectors" / f"{t}_response_avg_diff.pt")[LAYER] for t in TRAITS])
    units = vectors / vectors.norm(dim=1, keepdim=True)  # [7, hidden]
    model, tok = load_model()

    rows = []
    for level, system in enumerate(CRITIQUE_SYSTEM_PROMPTS):
        for question in FALSE_CLAIM_QUESTIONS + FEEDBACK_QUESTIONS:
            rows.append(dict(level=level, question=question, prompt=build_prompt(tok, question, system)))
    df = pd.DataFrame(rows)
    torch.manual_seed(0)
    df["answer"] = generate(model, tok, df["prompt"].tolist(), max_new_tokens=300, batch_size=20)

    before, during = [], []
    for prompt, answer in zip(df["prompt"], df["answer"]):
        prompt_len = len(tok.encode(prompt, add_special_tokens=False))
        enc = tok(prompt + answer, return_tensors="pt", add_special_tokens=False).to(model.device)
        hidden = model(**enc, output_hidden_states=True).hidden_states[LAYER][0].float().cpu()
        before.append(units @ hidden[prompt_len - 1])       # letzter Prompt-Token: vor der Antwort
        during.append(units @ hidden[prompt_len:].mean(0))  # Mittel über die Antwort
    for name, values in (("vor", torch.stack(before)), ("antwort", torch.stack(during))):
        for index, trait in enumerate(TRAITS):
            df[f"{name}_{trait}"] = values[:, index].numpy()
    del model
    torch.cuda.empty_cache()

    judge = Judge()
    rubrics = {"correction": CORRECTION_RUBRIC, **{t: load_trait(t, "eval")["eval_prompt"] for t in ("sycophantic", "impolite", "evil")}}
    df = asyncio.run(judge.score_columns(df.drop(columns="prompt"), rubrics))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "critique_levels.csv", index=False)

    print("Judge-Scores je Stufe:")
    print(df.groupby("level")[["correction", "sycophantic", "impolite", "evil"]].mean().round(1).to_string())
    print("Projektion vor der Antwort je Stufe:")
    print(df.groupby("level")[[f"vor_{t}" for t in TRAITS]].mean().round(1).to_string())
    print(judge.usage())


if __name__ == "__main__":
    main()
