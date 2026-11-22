"""Erweiterung J: Persönlichkeitsprofile bekannter Figuren.

Das Modell spielt per System-Prompt zehn bekannte fiktive Figuren und beantwortet zehn neutrale Alltagsfragen.
Für jede Antwort wird die Aktivierung auf Layer 22 auf die Vektoren aller sieben Traits projiziert,
einmal am letzten Prompt-Token (vor der Antwort) und einmal gemittelt über die Antwort.

Kein Judge: Geprüft wird, ob die Figuren dort landen, wo man sie erwartet. Die Figuren kamen bei der
Extraktion der Vektoren nicht vor, es ist also ein unabhängiger Test.
"""
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pv_common import RESULTS, build_prompt, generate, load_model
from pv_data import CHARACTERS, NEUTRAL_QUESTIONS

TRAITS = ["sycophantic", "evil", "impolite", "hallucinating", "optimistic", "humorous", "apathetic"]
LAYER = 22
VILLAINS = ["Darth Vader", "the Joker", "Lord Voldemort"]
OUT = RESULTS / "characters"


@torch.no_grad()
def main():
    vectors = torch.stack([torch.load(RESULTS / "vectors" / f"{t}_response_avg_diff.pt")[LAYER] for t in TRAITS])
    units = vectors / vectors.norm(dim=1, keepdim=True)
    model, tok = load_model()

    systems = {"Assistant": "You are a helpful assistant."}
    systems |= {name: f"You are {name}. Stay fully in character and answer as {name} would." for name in CHARACTERS}
    rows = [dict(character=name, question=question, prompt=build_prompt(tok, question, system))
            for name, system in systems.items() for question in NEUTRAL_QUESTIONS]
    df = pd.DataFrame(rows)
    torch.manual_seed(0)
    df["answer"] = generate(model, tok, df["prompt"].tolist(), max_new_tokens=200, batch_size=20)

    # Für die Layer-Prüfung: Evil-Vektor auf jedem Layer, auf Länge 1 gebracht
    evil_all = torch.load(RESULTS / "vectors" / "evil_response_avg_diff.pt")
    evil_all = evil_all / evil_all.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # [Layer+1, hidden]

    before, during, evil_by_layer = [], [], []
    for prompt, answer in zip(df["prompt"], df["answer"]):
        prompt_len = len(tok.encode(prompt, add_special_tokens=False))
        enc = tok(prompt + answer, return_tensors="pt", add_special_tokens=False).to(model.device)
        all_layers = torch.stack(model(**enc, output_hidden_states=True).hidden_states)[:, 0].float().cpu()
        hidden = all_layers[LAYER]
        before.append(units @ hidden[prompt_len - 1])
        during.append(units @ hidden[prompt_len:].mean(0))
        evil_by_layer.append((all_layers[:, prompt_len:].mean(1) * evil_all).sum(-1))  # [Layer+1]
    for name, values in (("vor", torch.stack(before)), ("antwort", torch.stack(during))):
        for index, trait in enumerate(TRAITS):
            df[f"{name}_{trait}"] = values[:, index].numpy()

    OUT.mkdir(parents=True, exist_ok=True)
    df.drop(columns="prompt").to_csv(OUT / "characters.csv", index=False)
    # Welcher Layer trennt die drei Schurken am besten von den übrigen Figuren?
    # Maß: Abstand der Gruppenmittel, geteilt durch die Streuung innerhalb der Gruppen (wie eine Effektstärke).
    by_layer = pd.DataFrame(torch.stack(evil_by_layer).numpy()).assign(character=df["character"].values)
    means = by_layer.groupby("character").mean().drop("Assistant")
    villains = means.index.isin(VILLAINS)
    spread = ((means[villains].var() * (villains.sum() - 1) + means[~villains].var() * ((~villains).sum() - 1))
              / (len(means) - 2)) ** 0.5
    separation = ((means[villains].mean() - means[~villains].mean()) / spread).iloc[1:]
    separation.rename_axis("layer").rename("trennung").to_csv(OUT / "evil_layer_separation.csv")
    print(f"Bester Layer für 'evil' nach Trennung der Schurken: {int(separation.idxmax())} "
          f"(Trennung {separation.max():.1f}; Layer 22: {separation[22]:.1f})")

    profile = df.groupby("character")[[f"antwort_{t}" for t in TRAITS]].mean()
    print((profile - profile.loc["Assistant"]).round(1).to_string())
    for name in ("the Joker", "Eeyore"):
        print(f"--- {name}: {df[df['character'] == name]['answer'].iloc[0][:200]!r}")


if __name__ == "__main__":
    main()
