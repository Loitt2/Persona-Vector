"""Bonus zu Skript 25: Werden alle drei Schurken böser, wenn man den Evil-Vektor ohne Rollen-Anteil addiert?

Die Stärke wird je Figur daran ausgerichtet, wie böse die Figur für das Modell ist: an ihrer Projektion auf den
Evil-Vektor ohne Rolle aus Skript 24 (Voldemort 5,1, Darth Vader 5,0, Joker 4,0), relativ zum höchsten Wert.
Ned Flanders dient als Gegenprobe mit voller Stärke: Wird auch eine harmlose Figur böse?
Darth Vader wurde in Skript 25 schon gemessen und wird hier nicht wiederholt.
"""
import asyncio
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pv_characters import REFERENCE_LAYER, activation_norms, relative_strength, scaled_vector, system_prompt
from pv_common import RESULTS, ActivationSteerer, build_prompt, generate, load_model
from pv_data import NEUTRAL_QUESTIONS, WHO_RUBRIC
from pv_judge import Judge

LAYER = 22
N_SAMPLES = 4
OUT = RESULTS / "characters"
CHARACTERS = ["Lord Voldemort", "the Joker", "Ned Flanders"]


def play(model, tok, character, vector=None):
    questions = list(NEUTRAL_QUESTIONS) * N_SAMPLES
    prompts = [build_prompt(tok, question, system_prompt(character)) for question in questions]
    if vector is None:
        answers = generate(model, tok, prompts, max_new_tokens=200, batch_size=40)
    else:
        with ActivationSteerer(model, vector, coeff=1.0, layer_idx=LAYER - 1, positions="response"):
            answers = generate(model, tok, prompts, max_new_tokens=200, batch_size=40)
    return questions, answers


def main():
    measured = pd.read_csv(OUT / "role_vector_projections.csv")
    measured = measured[measured["trait"] == "evil"].set_index("character")["ohne_rolle"]
    relative = {"Lord Voldemort": 1.0, "the Joker": float(measured["the Joker"] / measured["Lord Voldemort"]), "Ned Flanders": 1.0}
    print("Relative Stärke je Figur:", {name: round(value, 2) for name, value in relative.items()})

    evil = torch.load(RESULTS / "vectors" / "evil_response_avg_diff.pt")[LAYER]
    role = torch.load(RESULTS / "vectors" / "role_vector_layer22.pt")
    role_unit = role / role.norm()
    evil_clean = evil - (evil @ role_unit) * role_unit
    reference = torch.load(RESULTS / "vectors" / "sycophantic_response_avg_diff.pt")[REFERENCE_LAYER]

    model, tok = load_model()
    torch.manual_seed(5)
    _, plain = play(model, tok, "Assistant")
    norms = activation_norms(model, tok, list(NEUTRAL_QUESTIONS) * N_SAMPLES, plain)
    vector = scaled_vector(evil_clean, norms, LAYER, relative_strength(reference, norms))

    frames = []
    for character in CHARACTERS:
        for factor, label in ((None, "ohne Eingriff"), (1.0, "Evil ohne Rolle 1,0"), (1.5, "Evil ohne Rolle 1,5")):
            questions, answers = play(model, tok, character, None if factor is None else factor * relative[character] * vector)
            frames.append(pd.DataFrame(dict(character=character, condition=label, relative_staerke=relative[character],
                                            question=questions, answer=answers)))
    del model
    torch.cuda.empty_cache()

    df = pd.concat(frames, ignore_index=True)
    judge = Judge(model="claude-haiku-5-5", use_fallbacks=False)

    async def score_all():
        await asyncio.gather(judge.score_frame(df, "evil"), judge.score_columns(df, {"anakin": WHO_RUBRIC}))

    asyncio.run(score_all())
    df.to_csv(OUT / "villains_evil.csv", index=False)
    print(df.groupby(["character", "condition"], sort=False)[["evil", "coherence"]].mean().round(1).to_string())
    print(judge.usage())


if __name__ == "__main__":
    main()
