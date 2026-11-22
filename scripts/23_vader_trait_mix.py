"""Erweiterung: Kommt Darth Vader näher an Anakin, wenn man mehrere Trait-Vektoren zugleich benutzt?

Skript 22 hat gezeigt, welche Charakterzüge Vader und Anakin für das Modell unterscheiden:
Vader ist "apathetic" und "evil", Anakin "optimistic" und "humorous". Hier wird Vader deshalb mit einer
Mischung aus genau diesen vier Trait-Vektoren gesteuert: apathetic und evil abgezogen, optimistic und humorous addiert.

Zwei Mischungen, jeweils in zwei Stärken:
  - gleich gewichtet: alle vier Vektoren mit demselben Gewicht
  - gewichtet: Gewichte nach der Zerlegung aus Skript 22 (Kosinus mit der Richtung Vader -> Anakin)
Die Mischung wird auf dieselbe Länge gebracht wie die Eingriffe in den Skripten 16 und 22, damit die Stärken vergleichbar sind.
Getestet wird auf allen zehn Alltagsfragen, bewertet wie in Skript 22 mit Claude Haiku 5.5.

Hinweis: Die Gewichte der zweiten Mischung stammen aus Antworten auf dieselben zehn Fragen. Das sind nur vier Zahlen,
ganz unabhängig vom Test sind sie aber nicht.
"""
import asyncio
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pv_characters import REFERENCE_LAYER, activation_norms, answer_activations, relative_strength, scaled_vector, system_prompt
from pv_common import RESULTS, ActivationSteerer, build_prompt, generate, load_model
from pv_data import NEUTRAL_QUESTIONS, WHO_RUBRIC
from pv_judge import Judge

LAYER = 22
N_SAMPLES = 4
OUT = RESULTS / "characters"
EQUAL = {"apathetic": -1.0, "evil": -1.0, "optimistic": 1.0, "humorous": 1.0}


def play(model, tok, character, vector=None):
    """Antworten der Figur auf die zehn Alltagsfragen, optional mit addiertem Vektor auf LAYER."""
    questions = list(NEUTRAL_QUESTIONS) * N_SAMPLES
    prompts = [build_prompt(tok, question, system_prompt(character)) for question in questions]
    if vector is None:
        answers = generate(model, tok, prompts, max_new_tokens=200, batch_size=40)
    else:
        with ActivationSteerer(model, vector, coeff=1.0, layer_idx=LAYER - 1, positions="response"):
            answers = generate(model, tok, prompts, max_new_tokens=200, batch_size=40)
    return questions, prompts, answers


def main():
    # Gewichte der zweiten Mischung: Mittel der beiden Durchläufe von Skript 22, nur die vier stärksten Charakterzüge
    decomposition = pd.concat([pd.read_csv(OUT / f"vader_to_anakin_traits{suffix}.csv") for suffix in ("", "_fold2")])
    weighted = decomposition.groupby("trait")["kosinus_bereinigt"].mean()[list(EQUAL)].to_dict()
    print("Gewichte der gewichteten Mischung:", {t: round(w, 2) for t, w in weighted.items()})

    units = {t: (v := torch.load(RESULTS / "vectors" / f"{t}_response_avg_diff.pt")[LAYER]) / v.norm() for t in EQUAL}
    mixes = {name: sum(weight * units[t] for t, weight in weights.items()) for name, weights in (("gleich", EQUAL), ("gewichtet", weighted))}
    reference = torch.load(RESULTS / "vectors" / "sycophantic_response_avg_diff.pt")[REFERENCE_LAYER]

    model, tok = load_model()
    torch.manual_seed(3)
    _, _, plain = play(model, tok, "Assistant")
    norms = activation_norms(model, tok, list(NEUTRAL_QUESTIONS) * N_SAMPLES, plain)
    strength = relative_strength(reference, norms)

    conditions = [("Assistant", "ohne Eingriff", None), ("Anakin Skywalker", "ohne Eingriff", None), ("Darth Vader", "ohne Eingriff", None)]
    for name, mix in mixes.items():
        for factor, label in ((1.0, "1,0"), (1.5, "1,5")):
            conditions.append(("Darth Vader", f"Trait-Mix {name} {label}", factor * scaled_vector(mix, norms, LAYER, strength)))

    frames, centers = [], {}
    for character, condition, vector in conditions:
        questions, prompts, answers = play(model, tok, character, vector)
        centers[(character, condition)] = answer_activations(model, tok, prompts, answers, LAYER).mean(0)  # ohne Eingriff gelesen
        frames.append(pd.DataFrame(dict(character=character, condition=condition, question=questions, answer=answers)))
    del model
    torch.cuda.empty_cache()

    df = pd.concat(frames, ignore_index=True)
    judge = Judge(model="claude-haiku-5-5", use_fallbacks=False)

    async def score_all():
        await asyncio.gather(judge.score_frame(df, "evil"), judge.score_columns(df, {"anakin": WHO_RUBRIC}))

    asyncio.run(score_all())
    df.to_csv(OUT / "vader_trait_mix.csv", index=False)

    baseline = centers[("Assistant", "ohne Eingriff")]
    anakin = centers[("Anakin Skywalker", "ohne Eingriff")] - baseline
    vader = centers[("Darth Vader", "ohne Eingriff")] - baseline
    axis = anakin - vader
    rows = []
    for (character, condition), center in centers.items():
        if character != "Darth Vader":
            continue
        shifted = center - baseline
        rows.append(dict(condition=condition, abstand_zu_anakin=float((shifted - anakin).norm()),
                         position_vader_anakin=float((shifted - vader) @ axis / (axis @ axis))))
    similarity = pd.DataFrame(rows)
    for name, mix in mixes.items():
        similarity[f"kosinus_mix_{name}_mit_achse"] = float(torch.nn.functional.cosine_similarity(mix, axis, dim=0))
    similarity.to_csv(OUT / "vader_trait_mix_similarity.csv", index=False)

    print(df.groupby(["character", "condition"], sort=False)[["evil", "anakin", "coherence"]].mean().round(1).to_string())
    print(similarity.round(3).to_string(index=False))
    print(judge.usage())


if __name__ == "__main__":
    main()
