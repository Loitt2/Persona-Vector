"""Erweiterung: Lässt sich "das Modell spielt eine Rolle" als eigene Richtung fassen und aus den Trait-Vektoren entfernen?

Bisher wurde der gemeinsame Anteil der Trait-Vektoren über ihr Mittel geschätzt (Skript 11). Das entfernt ihn nur zum Teil:
Auch danach liegt jede gespielte Figur auf dem Evil-Vektor höher als der normale Assistent. Hier wird die Rolle direkt gemessen:

    Rollen-Vektor = mittlere Aktivierung(gespielte Figuren) - mittlere Aktivierung(normaler Assistent)

Dafür werden nur die sieben Figuren benutzt, die keine Schurken sind, damit im Rollen-Vektor möglichst wenig "böse" steckt.
Damit keine Figur an ihrem eigenen Beitrag gemessen wird, bekommt jede dieser sieben einen Rollen-Vektor aus den übrigen sechs.
Dann wird aus jedem Trait-Vektor der Anteil in Richtung des Rollen-Vektors entfernt und neu projiziert.

Es werden keine neuen Antworten erzeugt: Das Modell liest die gespeicherten Antworten aus Skript 15 noch einmal. Kein Judge nötig.
"""
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pv_characters import answer_activations, system_prompt
from pv_common import RESULTS, build_prompt, load_model

LAYER = 22
TRAITS = ["sycophantic", "evil", "impolite", "hallucinating", "optimistic", "humorous", "apathetic"]
VILLAINS = ["Darth Vader", "the Joker", "Lord Voldemort"]
OUT = RESULTS / "characters"


def without(vector, direction):
    """Entfernt aus `vector` den Anteil in Richtung `direction` und normiert neu."""
    unit = direction / direction.norm()
    rest = vector - (vector @ unit) * unit
    return rest / rest.norm()


def main():
    answers = pd.read_csv(OUT / "characters.csv")[["character", "question", "answer"]]
    model, tok = load_model()
    prompts = [build_prompt(tok, row.question, system_prompt(row.character)) for row in answers.itertuples()]
    activations = answer_activations(model, tok, prompts, answers["answer"].tolist(), LAYER)
    del model
    torch.cuda.empty_cache()

    means = {name: activations[(answers["character"] == name).to_numpy()].mean(0) for name in answers["character"].unique()}
    assistant = means.pop("Assistant")
    neutral = [name for name in means if name not in VILLAINS]
    units = {t: (v := torch.load(RESULTS / "vectors" / f"{t}_response_avg_diff.pt")[LAYER]) / v.norm() for t in TRAITS}

    rows = []
    for name, mean in means.items():
        others = [n for n in neutral if n != name]  # Schurken: alle sieben; die übrigen: die anderen sechs
        role = torch.stack([means[n] for n in others]).mean(0) - assistant
        shifted = mean - assistant
        for trait in TRAITS:
            rows.append(dict(character=name, trait=trait, roh=float(shifted @ units[trait]),
                             ohne_rolle=float(shifted @ without(units[trait], role))))
    pd.DataFrame(rows).to_csv(OUT / "role_vector_projections.csv", index=False)

    # Wie viel Rolle steckt in jedem Trait-Vektor? Rollen-Vektor aus allen sieben Nicht-Schurken.
    role_all = torch.stack([means[n] for n in neutral]).mean(0) - assistant
    shared = torch.stack(list(units.values())).mean(0)
    cosines = pd.DataFrame(dict(trait=TRAITS, kosinus_mit_rollenvektor=[float(torch.nn.functional.cosine_similarity(units[t], role_all, dim=0)) for t in TRAITS]))
    cosines["kosinus_rollenvektor_mit_mittel_der_traits"] = float(torch.nn.functional.cosine_similarity(shared, role_all, dim=0))
    cosines.to_csv(OUT / "role_vector_cosines.csv", index=False)
    torch.save(role_all, RESULTS / "vectors" / "role_vector_layer22.pt")

    table = pd.DataFrame(rows)
    print(table[table["trait"] == "evil"].drop(columns="trait").round(1).sort_values("ohne_rolle").to_string(index=False))
    print(cosines.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
