"""Starter: zeigt den Stand aller Schritte und führt die fehlenden der Reihe nach aus.

    python scripts/start.py            # Stand anzeigen, nichts ausführen
    python scripts/start.py --run      # fehlende Schritte ausführen (GPU und Anthropic-Key nötig)
    python scripts/start.py --run --neu  # alles neu ausführen, auch vorhandene Ergebnisse

Ein Schritt gilt als erledigt, wenn seine Ergebnisdatei in results/ liegt.
Schritte mit "API" rufen den Judge auf und kosten Geld.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OTHER_TRAITS = ["evil", "impolite", "hallucinating", "optimistic", "humorous", "apathetic"]

# (Beschreibung, Skript mit Argumenten, Ergebnisdatei, braucht)
STEPS = [
    ("Antworten für die Extraktion erzeugen", ["02_extract.py", "generate"], "extract/sycophantic_pos.csv", "GPU"),
    ("Judge bewertet die Antworten", ["02_extract.py", "judge"], None, "API"),
    ("Sycophancy-Vektor berechnen", ["02_extract.py", "vector"], "vectors/sycophantic_response_avg_diff.pt", "GPU"),
    ("Layer-Suche", ["03_steering.py", "layers"], "steering/layer_sweep.csv", "GPU, API"),
    ("Steering-Kurve", ["03_steering.py", "curve"], "steering/steering_curve.csv", "GPU, API"),
    ("Monitoring", ["04_monitoring.py"], "monitoring/monitoring.csv", "GPU, API"),
    ("Kontrollen: Zufallsvektor, negatives Vorzeichen", ["05_controls.py"], "steering/controls.csv", "GPU, API"),
    ("Logit Lens", ["06_logit_lens.py"], "logit_lens/sycophantic_tokens.csv", "GPU"),
    ("Monitoring über alle Layer", ["07_monitoring_layers.py"], "monitoring/monitoring_all_layers.csv", "GPU"),
    ("Zweiter Judge", ["08_second_judge.py"], "judge_check/second_judge.csv", "API"),
    ("Längenkontrollierter Vektor", ["02_extract.py", "vector", "--first-tokens", "50"], "vectors/sycophantic_response_first50_diff.pt", "GPU"),
    ("Steering mit längenkontrolliertem Vektor", ["10_length_control.py"], "steering/length_control.csv", "GPU, API"),
]
for trait in OTHER_TRAITS:
    samples = "2" if trait == "evil" else "1"
    STEPS += [
        (f"{trait}: Antworten erzeugen", ["02_extract.py", "generate", "--trait", trait, "--samples", samples], f"extract/{trait}_pos.csv", "GPU"),
        (f"{trait}: Judge bewertet", ["02_extract.py", "judge", "--trait", trait], None, "API"),
        (f"{trait}: Vektor berechnen", ["02_extract.py", "vector", "--trait", trait], f"vectors/{trait}_response_avg_diff.pt", "GPU"),
    ]
STEPS += [
    ("Neue Fragenformate", ["09_new_formats.py"], "new_formats/new_formats.csv", "GPU, API"),
    ("Trait-Landkarte", ["11_trait_map.py"], "trait_map/cosine_all_layers.csv", "GPU, API"),
    ("Token-Heatmap", ["12_token_heatmap.py"], "token_heatmap/examples.json", "GPU"),
    ("evil: Steering und Monitoring", ["13_other_traits.py", "evil"], "other_traits/evil_monitoring.csv", "GPU, API"),
    ("hallucinating: Steering und Monitoring", ["13_other_traits.py", "hallucinating"], "other_traits/hallucinating_monitoring.csv", "GPU, API"),
    ("Faire gegen abwertende Kritik", ["14_critique_monitoring.py"], "critique/critique_levels.csv", "GPU, API"),
    ("Figuren-Profile", ["15_characters.py"], "characters/characters.csv", "GPU"),
]
for trait in ["sycophantic", "evil", "hallucinating"]:
    STEPS.append((f"Faire Layer-Suche: {trait}", ["17_layer_search.py", trait], f"layer_search/{trait}.csv", "GPU, API"))
STEPS += [
    ("Darth Vader ohne böse gegen Anakin", ["16_character_steering.py"], "characters/similarity_to_anakin.csv", "GPU, API"),
    ("Vader und Anakin über alle Layer", ["18_anakin_layers.py"], "characters/anakin_layers.csv", "GPU, API"),
]
for trait in ["sycophantic"] + OTHER_TRAITS:
    STEPS.append((f"Stabilität des Vektors: {trait}", ["02_extract.py", "vector", "--trait", trait, "--stability-only"],
                  f"vectors/{trait}_stability.json", "GPU"))


def is_done(index: int) -> bool:
    """Schritte ohne eigene Datei (Judge) gelten als erledigt, wenn der folgende Schritt erledigt ist."""
    marker = STEPS[index][2]
    if marker is None:
        return is_done(index + 1)
    return (RESULTS / marker).exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="fehlende Schritte ausführen")
    parser.add_argument("--neu", action="store_true", help="auch erledigte Schritte neu ausführen")
    args = parser.parse_args()

    for index, (name, command, _, needs) in enumerate(STEPS):
        done = is_done(index) and not args.neu
        print(f"[{'x' if done else ' '}] {index + 1:2d}. {name}  ({needs})")
        if args.run and not done:
            print(f"     -> python scripts/{' '.join(command)}")
            subprocess.run([sys.executable, str(ROOT / "scripts" / command[0]), *command[1:]], check=True)
    if not args.run:
        print("\nNichts ausgeführt. Mit --run starten, Ergebnisse ansehen mit: python scripts/zusammenfassung.py")


if __name__ == "__main__":
    main()
