"""Erweiterung M: Auf welchem Layer bringt das Abziehen von "böse" Darth Vader am nächsten an Anakin Skywalker?

Das Modell spielt Darth Vader. Auf jeweils einem Layer wird der Evil-Vektor abgezogen, auf jedem Layer
gleich stark im Verhältnis zur Aktivierung dort (wie in 17_layer_search.py). Die fertigen Antworten werden
ohne Eingriff noch einmal gelesen und ihre mittlere Aktivierung auf Layer 22 mit der von Anakin verglichen.

Bewertet wird mit Claude Haiku 5.5: wie böse die Antworten sind und ob sie verständlich bleiben.
"""
import asyncio
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

LAYERS = list(range(4, 35, 2))
READ_LAYER = 22


def main():
    evil = torch.load(RESULTS / "vectors" / "evil_response_avg_diff.pt")
    reference = torch.load(RESULTS / "vectors" / "sycophantic_response_avg_diff.pt")[REFERENCE_LAYER]
    model, tok = load_model()
    torch.manual_seed(0)

    def play(character, layer=None, vector=None):
        prompts = [build_prompt(tok, question, system_prompt(character)) for question in NEUTRAL_QUESTIONS]
        if vector is None:
            answers = generate(model, tok, prompts, max_new_tokens=200)
        else:
            with ActivationSteerer(model, vector, coeff=-1.0, layer_idx=layer - 1, positions="response"):
                answers = generate(model, tok, prompts, max_new_tokens=200)
        return answers, answer_activations(model, tok, prompts, answers, READ_LAYER).mean(0)

    assistant_answers, assistant = play("Assistant")
    norms = activation_norms(model, tok, NEUTRAL_QUESTIONS, assistant_answers)
    strength = relative_strength(reference, norms)
    _, anakin = play("Anakin Skywalker")
    anakin = anakin - assistant

    frames, rows, vader, axis = [], [], None, None
    for layer in [0] + LAYERS:  # 0 = Vader ohne Eingriff
        vector = None if layer == 0 else scaled_vector(evil[layer], norms, layer, strength)
        answers, center = play("Darth Vader", layer, vector)
        shifted = center - assistant
        if layer == 0:
            vader, axis = shifted, anakin - shifted
        rows.append(dict(layer=layer,
                         kosinus_zu_anakin=float(torch.nn.functional.cosine_similarity(shifted, anakin, dim=0)),
                         abstand_zu_anakin=float((shifted - anakin).norm()),
                         position_vader_anakin=float((shifted - vader) @ axis / (axis @ axis))))
        frames.append(pd.DataFrame(dict(layer=layer, question=NEUTRAL_QUESTIONS, answer=answers)))

    df = pd.concat(frames, ignore_index=True)
    judge = Judge(model="claude-haiku-5-5", use_fallbacks=False)

    async def score_all():
        # "anakin": Wer spricht? 0 = klingt wie Darth Vader, 100 = klingt wie Anakin Skywalker
        await asyncio.gather(judge.score_frame(df, "evil"), judge.score_columns(df, {"anakin": WHO_RUBRIC}))

    asyncio.run(score_all())
    result = pd.DataFrame(rows).merge(df.groupby("layer")[["evil", "anakin", "coherence"]].mean().reset_index(), on="layer")
    out = RESULTS / "characters"
    df.to_csv(out / "anakin_layers_answers.csv", index=False)
    result.to_csv(out / "anakin_layers.csv", index=False)
    print(result.round(2).to_string(index=False))
    usable = result[(result["layer"] > 0) & (result["coherence"] >= 50)]
    best = usable.loc[usable["abstand_zu_anakin"].idxmin()]
    print(f"Am nächsten an Anakin bei verständlichen Antworten: Layer {int(best['layer'])} "
          f"(Abstand {best['abstand_zu_anakin']:.1f}; ohne Eingriff {result.loc[0, 'abstand_zu_anakin']:.1f})")
    row = df[df["layer"] == int(best["layer"])].iloc[0]
    print(f"--- Beispiel auf diesem Layer: {row['answer'][:330]!r}")
    print(judge.usage())


if __name__ == "__main__":
    main()
