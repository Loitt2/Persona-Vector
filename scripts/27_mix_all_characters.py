"""Erweiterung: Wohin wandern alle zehn Figuren, wenn man die Trait-Mischung aus Skript 23 addiert?

Die Mischung (apathetic und evil abgezogen, optimistic und humorous addiert, gleich gewichtet, Stärke 1,0) hat Darth Vader
am nächsten an Anakin Skywalker gebracht. Hier wird derselbe Vektor bei allen zehn Figuren aus Skript 15 addiert.
Gemessen wird wie in Skript 15 ohne Judge-Urteil über den Charakter: Die fertigen Antworten werden ohne Eingriff gelesen
und ihre mittlere Aktivierung auf die Trait-Vektoren projiziert. Der Judge bewertet nur, ob die Antworten verständlich bleiben.
"""
import asyncio
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pv_characters import REFERENCE_LAYER, activation_norms, answer_activations, relative_strength, scaled_vector, system_prompt
from pv_common import RESULTS, ActivationSteerer, build_prompt, generate, load_model
from pv_data import CHARACTERS, NEUTRAL_QUESTIONS
from pv_judge import Judge

LAYER = 22
N_SAMPLES = 2
TRAITS = ["sycophantic", "evil", "impolite", "hallucinating", "optimistic", "humorous", "apathetic"]
MIX = {"apathetic": -1.0, "evil": -1.0, "optimistic": 1.0, "humorous": 1.0}
OUT = RESULTS / "characters"


def play(model, tok, character, vector=None):
    questions = list(NEUTRAL_QUESTIONS) * N_SAMPLES
    prompts = [build_prompt(tok, question, system_prompt(character)) for question in questions]
    if vector is None:
        answers = generate(model, tok, prompts, max_new_tokens=200, batch_size=40)
    else:
        with ActivationSteerer(model, vector, coeff=1.0, layer_idx=LAYER - 1, positions="response"):
            answers = generate(model, tok, prompts, max_new_tokens=200, batch_size=40)
    return questions, prompts, answers


def main():
    units = {t: (v := torch.load(RESULTS / "vectors" / f"{t}_response_avg_diff.pt")[LAYER]) / v.norm() for t in TRAITS}
    role = torch.load(RESULTS / "vectors" / "role_vector_layer22.pt")
    role_unit = role / role.norm()
    evil_clean = units["evil"] - (units["evil"] @ role_unit) * role_unit
    readout = dict(units, evil_ohne_rolle=evil_clean / evil_clean.norm())
    reference = torch.load(RESULTS / "vectors" / "sycophantic_response_avg_diff.pt")[REFERENCE_LAYER]

    model, tok = load_model()
    torch.manual_seed(6)
    questions, prompts, plain = play(model, tok, "Assistant")
    norms = activation_norms(model, tok, questions, plain)
    mix = scaled_vector(sum(weight * units[t] for t, weight in MIX.items()), norms, LAYER, relative_strength(reference, norms))

    frames = []
    for character in ["Assistant", *CHARACTERS]:
        for condition, vector in (("ohne Eingriff", None), ("Trait-Mischung", mix)):
            questions, prompts, answers = play(model, tok, character, vector)
            activations = answer_activations(model, tok, prompts, answers, LAYER)  # ohne Eingriff gelesen
            frame = pd.DataFrame(dict(character=character, condition=condition, question=questions, answer=answers))
            for name, unit in readout.items():
                frame[f"proj_{name}"] = (activations @ unit).numpy()
            frames.append(frame)
    del model
    torch.cuda.empty_cache()

    df = pd.concat(frames, ignore_index=True)
    judge = Judge(model="claude-haiku-5-5", use_fallbacks=False)
    df = asyncio.run(judge.score_frame(df, "evil"))
    df.to_csv(OUT / "mix_all_characters.csv", index=False)
    print(df.groupby(["character", "condition"], sort=False)[["proj_evil_ohne_rolle", "proj_optimistic", "proj_humorous", "proj_apathetic",
                                                               "evil", "coherence"]].mean().round(1).to_string())
    print(judge.usage())


if __name__ == "__main__":
    main()
