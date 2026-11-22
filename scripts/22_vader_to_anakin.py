"""Erweiterung: Wie nah kommt Darth Vader an Anakin Skywalker, wenn man gezielt in dessen Richtung steuert?

Skript 16 hat bei Vader den Evil-Vektor abgezogen. Im Text rückte er Richtung Anakin, in den Aktivierungen nicht,
weil der Evil-Vektor nur schwach mit dem übereinstimmt, was die beiden Figuren trennt (Kosinus 0,16).
Hier wird deshalb die Richtung benutzt, die Vader und Anakin tatsächlich unterscheidet:

    Richtung = mittlere Aktivierung(Anakin-Antworten) - mittlere Aktivierung(Vader-Antworten)

Damit das kein Zirkelschluss ist, wird die Richtung aus den ersten fünf Alltagsfragen berechnet
und auf den anderen fünf getestet. Verglichen werden: kein Eingriff, "böse" abgezogen (wie Skript 16),
Richtung Anakin in drei Stärken, und beides zusammen. Bewertet wird wie in Skript 16 mit Claude Haiku 5.5.
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
# Zwei Durchläufe mit vertauschten Rollen der Fragen: "python 22_vader_to_anakin.py" und "python 22_vader_to_anakin.py 2"
FOLD = sys.argv[1] if len(sys.argv) > 1 else "1"
FIT_QUESTIONS, TEST_QUESTIONS = (NEUTRAL_QUESTIONS[:5], NEUTRAL_QUESTIONS[5:]) if FOLD == "1" else (NEUTRAL_QUESTIONS[5:], NEUTRAL_QUESTIONS[:5])
SUFFIX = "" if FOLD == "1" else "_fold2"
N_SAMPLES = 4  # Antworten je Frage
BATCH = 80
OUT = RESULTS / "characters"
TRAITS = ["sycophantic", "evil", "impolite", "hallucinating", "optimistic", "humorous", "apathetic"]


def play(model, tok, character, questions, vector=None):
    """Antworten der Figur auf die Fragen, optional mit addiertem Vektor auf LAYER."""
    questions = list(questions) * N_SAMPLES
    prompts = [build_prompt(tok, question, system_prompt(character)) for question in questions]
    if vector is None:
        answers = generate(model, tok, prompts, max_new_tokens=200, batch_size=BATCH)
    else:
        with ActivationSteerer(model, vector, coeff=1.0, layer_idx=LAYER - 1, positions="response"):
            answers = generate(model, tok, prompts, max_new_tokens=200, batch_size=BATCH)
    return questions, prompts, answers


def main():
    evil = torch.load(RESULTS / "vectors" / "evil_response_avg_diff.pt")
    reference = torch.load(RESULTS / "vectors" / "sycophantic_response_avg_diff.pt")[REFERENCE_LAYER]
    model, tok = load_model()
    torch.manual_seed(0 if FOLD == "1" else 2)

    # Stärke wie in Skript 16: relativ zur typischen Länge der Aktivierung des normalen Assistenten
    _, _, plain = play(model, tok, "Assistant", TEST_QUESTIONS)
    norms = activation_norms(model, tok, list(TEST_QUESTIONS) * N_SAMPLES, plain)
    strength = relative_strength(reference, norms)

    # Richtung von Vader nach Anakin, nur aus den FIT-Fragen
    fit = {}
    for character in ("Darth Vader", "Anakin Skywalker"):
        _, prompts, answers = play(model, tok, character, FIT_QUESTIONS)
        fit[character] = answer_activations(model, tok, prompts, answers, LAYER).mean(0)
    direction = fit["Anakin Skywalker"] - fit["Darth Vader"]
    # Welche Charakterzüge unterscheiden die beiden? Die Richtung Vader -> Anakin wird mit jedem der sieben Trait-Vektoren verglichen:
    # roh und nachdem der gemeinsame Anteil aller Trait-Vektoren (ihr Mittel) abgezogen wurde.
    units = torch.stack([v[LAYER] / v[LAYER].norm() for v in
                         (torch.load(RESULTS / "vectors" / f"{t}_response_avg_diff.pt") for t in TRAITS)])
    cleaned = units - units.mean(0)
    cleaned = cleaned / cleaned.norm(dim=1, keepdim=True)
    trait_table = pd.DataFrame(dict(trait=TRAITS, projektion=(units @ direction).tolist(),
                                    kosinus_roh=(units @ direction / direction.norm()).tolist(),
                                    kosinus_bereinigt=(cleaned @ direction / direction.norm()).tolist()))
    OUT.mkdir(parents=True, exist_ok=True)
    trait_table.to_csv(OUT / f"vader_to_anakin_traits{SUFFIX}.csv", index=False)
    print(trait_table.round(2).to_string(index=False))

    to_anakin = scaled_vector(direction, norms, LAYER, strength)
    minus_evil = -scaled_vector(evil[LAYER], norms, LAYER, strength)
    natural = float(direction.norm() / to_anakin.norm())
    cos_evil = float(torch.nn.functional.cosine_similarity(direction, -evil[LAYER], dim=0))
    print(f"Der echte Abstand Vader-Anakin entspricht Stärke {natural:.2f}; Kosinus der Richtung mit 'minus böse': {cos_evil:.2f}")

    conditions = [("Assistant", "ohne Eingriff", None), ("Anakin Skywalker", "ohne Eingriff", None), ("Darth Vader", "ohne Eingriff", None),
                  ("Darth Vader", "böse abgezogen", minus_evil),
                  ("Darth Vader", "Richtung Anakin 0,5", 0.5 * to_anakin), ("Darth Vader", "Richtung Anakin 1,0", to_anakin),
                  ("Darth Vader", "Richtung Anakin 1,5", 1.5 * to_anakin),
                  ("Darth Vader", "beides", minus_evil + to_anakin)]
    frames, centers = [], {}
    for character, condition, vector in conditions:
        questions, prompts, answers = play(model, tok, character, TEST_QUESTIONS, vector)
        centers[(character, condition)] = answer_activations(model, tok, prompts, answers, LAYER).mean(0)  # ohne Eingriff gelesen
        frames.append(pd.DataFrame(dict(character=character, condition=condition, question=questions, answer=answers)))
    del model
    torch.cuda.empty_cache()

    df = pd.concat(frames, ignore_index=True)
    judge = Judge(model="claude-haiku-5-5", use_fallbacks=False)

    async def score_all():
        await asyncio.gather(judge.score_frame(df, "evil"), judge.score_columns(df, {"anakin": WHO_RUBRIC}))

    asyncio.run(score_all())
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"vader_to_anakin{SUFFIX}.csv", index=False)

    # Aktivierungen relativ zum normalen Assistenten; Achse aus den TEST-Antworten, also unabhängig von der Steuer-Richtung
    baseline = centers[("Assistant", "ohne Eingriff")]
    anakin = centers[("Anakin Skywalker", "ohne Eingriff")] - baseline
    vader = centers[("Darth Vader", "ohne Eingriff")] - baseline
    axis = anakin - vader
    rows = []
    for (character, condition), center in centers.items():
        if character != "Darth Vader":
            continue
        shifted = center - baseline
        rows.append(dict(condition=condition,
                         kosinus_zu_anakin=float(torch.nn.functional.cosine_similarity(shifted, anakin, dim=0)),
                         abstand_zu_anakin=float((shifted - anakin).norm()),
                         position_vader_anakin=float((shifted - vader) @ axis / (axis @ axis))))
    similarity = pd.DataFrame(rows)
    similarity["natuerliche_staerke"], similarity["kosinus_richtung_minus_boese"] = natural, cos_evil
    similarity.to_csv(OUT / f"vader_to_anakin_similarity{SUFFIX}.csv", index=False)

    print(df.groupby(["character", "condition"], sort=False)[["evil", "anakin", "coherence"]].mean().round(1).to_string())
    print(similarity.round(3).to_string(index=False))
    print(judge.usage())


if __name__ == "__main__":
    main()
