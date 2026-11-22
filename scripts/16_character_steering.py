"""Erweiterung K: Was passiert mit einer Figur, wenn man "böse" abzieht?

Das Modell spielt per System-Prompt eine Figur. Beim Generieren wird der Evil-Vektor abgezogen
(bei drei Schurken) oder addiert (bei Ned Flanders als Gegenprobe). Gesteuert wird auf dem Layer,
den die Layer-Suche (17_layer_search.py) für "evil" gefunden hat; ohne deren Ergebnis auf Layer 22.
Die Stärke ist relativ zur Aktivierung: 1.0 entspricht der Einstellung, die für Sycophancy gut funktionierte.

Zusatzfrage: Rückt Darth Vader ohne "böse" in Richtung Anakin Skywalker? Dafür werden die fertigen
Antworten noch einmal ohne Eingriff durch das Modell geschickt und ihre mittlere Aktivierung auf Layer 22
verglichen. Gemessen wird also, wie der Text für das Modell aussieht, nicht der Eingriff selbst.

Bewertet wird mit Claude Haiku 5.5: wie böse die Antwort ist und ob sie verständlich bleibt.
"""
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pv_characters import (REFERENCE_LAYER, activation_norms, answer_activations, relative_strength,
                           scaled_vector, system_prompt)
from pv_common import RESULTS, ActivationSteerer, build_prompt, generate, load_model
from pv_data import NEUTRAL_QUESTIONS, WHO_RUBRIC
from pv_judge import Judge

READ_LAYER = 22
# Figur -> Stärken des Eingriffs (negativ: Evil-Vektor abziehen)
CONDITIONS = {
    "Assistant": [0.0],
    "Anakin Skywalker": [0.0],
    "Darth Vader": [0.0, -0.6, -1.0],
    "the Joker": [0.0, -0.6, -1.0],
    "Lord Voldemort": [0.0, -0.6, -1.0],
    "Ned Flanders": [0.0, 0.6, 1.0],
}


def main():
    best_file = RESULTS / "layer_search" / "best_layers.json"
    layer = json.loads(best_file.read_text()).get("evil", 22) if best_file.exists() else 22
    print(f"Gesteuert wird auf Layer {layer}")
    evil = torch.load(RESULTS / "vectors" / "evil_response_avg_diff.pt")
    reference = torch.load(RESULTS / "vectors" / "sycophantic_response_avg_diff.pt")[REFERENCE_LAYER]
    evil_unit = evil[READ_LAYER] / evil[READ_LAYER].norm()

    model, tok = load_model()
    torch.manual_seed(0)
    frames, centers, vector = [], {}, None
    for character, coefs in CONDITIONS.items():  # "Assistant" zuerst: liefert die Aktivierungslängen
        prompts = [build_prompt(tok, question, system_prompt(character)) for question in NEUTRAL_QUESTIONS]
        for coef in coefs:
            if coef == 0:
                answers = generate(model, tok, prompts, max_new_tokens=200)
            else:
                with ActivationSteerer(model, vector, coeff=coef, layer_idx=layer - 1, positions="response"):
                    answers = generate(model, tok, prompts, max_new_tokens=200)
            if vector is None:
                norms = activation_norms(model, tok, NEUTRAL_QUESTIONS, answers)
                vector = scaled_vector(evil[layer], norms, layer, relative_strength(reference, norms))
            activations = answer_activations(model, tok, prompts, answers, READ_LAYER)
            centers[(character, coef)] = activations.mean(0)
            frames.append(pd.DataFrame(dict(character=character, coef=coef, layer=layer, question=NEUTRAL_QUESTIONS,
                                            answer=answers, proj_evil=(activations @ evil_unit).numpy())))
    del model
    torch.cuda.empty_cache()

    df = pd.concat(frames, ignore_index=True)
    judge = Judge(model="claude-haiku-5-5", use_fallbacks=False)

    async def score_all():
        # "anakin": Wer spricht? 0 = klingt wie Darth Vader, 100 = klingt wie Anakin Skywalker
        await asyncio.gather(judge.score_frame(df, "evil"), judge.score_columns(df, {"anakin": WHO_RUBRIC}))

    asyncio.run(score_all())
    out = RESULTS / "characters"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "character_steering.csv", index=False)
    print(df.groupby(["character", "coef"], sort=False)[["evil", "anakin", "coherence", "proj_evil"]].mean().round(1).to_string())

    # Drei Maße in den Aktivierungen, jeweils relativ zum normalen Assistenten:
    #   Kosinus und Abstand zum Anakin-Mittel, und die Position auf der Achse von Vader (0) nach Anakin (1)
    baseline = centers[("Assistant", 0.0)]
    anakin = centers[("Anakin Skywalker", 0.0)] - baseline
    vader = centers[("Darth Vader", 0.0)] - baseline
    axis = anakin - vader
    print(f"Kosinus zwischen dem Evil-Vektor und der Richtung von Anakin nach Vader: "
          f"{float(torch.nn.functional.cosine_similarity(evil_unit, -axis, dim=0)):.2f}")
    rows = []
    for (character, coef), center in centers.items():
        if character in ("Assistant", "Anakin Skywalker"):
            continue
        shifted = center - baseline
        rows.append(dict(character=character, coef=coef,
                         kosinus_zu_anakin=float(torch.nn.functional.cosine_similarity(shifted, anakin, dim=0)),
                         abstand_zu_anakin=float((shifted - anakin).norm()),
                         position_vader_anakin=float((shifted - vader) @ axis / (axis @ axis))))
    similarity = pd.DataFrame(rows)
    similarity.to_csv(out / "similarity_to_anakin.csv", index=False)
    print(similarity.round(3).to_string(index=False))

    for character, coef in (("Anakin Skywalker", 0.0), ("Darth Vader", 0.0), ("Darth Vader", -0.6), ("Darth Vader", -1.0),
                            ("Ned Flanders", 0.0), ("Ned Flanders", 1.0)):
        row = df[(df["character"] == character) & (df["coef"] == coef)].iloc[0]
        print(f"--- {character}, Stärke {coef:+.1f}: {row['answer'][:330]!r}")
    print(judge.usage())


if __name__ == "__main__":
    main()
