"""Zusammenfassung: druckt die wichtigsten Zahlen aller Experimente aus results/.

Braucht weder GPU noch API-Key. Abschnitte, deren Ergebnisdatei fehlt, werden übersprungen.

    python scripts/zusammenfassung.py
"""
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
TRAIT = "sycophantic"


def section(title: str, *files: str) -> bool:
    """Druckt die Überschrift und meldet, ob alle nötigen Dateien da sind."""
    if not all((RESULTS / f).exists() for f in files):
        return False
    print(f"\n=== {title} ===")
    return True


def main():
    if section("1 Extraktion", f"extract/{TRAIT}_pos.csv", f"extract/{TRAIT}_neg.csv"):
        pos = pd.read_csv(RESULTS / "extract" / f"{TRAIT}_pos.csv")
        neg = pd.read_csv(RESULTS / "extract" / f"{TRAIT}_neg.csv")
        passed = (pos[TRAIT] >= 50) & (neg[TRAIT] < 50) & (pos["coherence"] >= 50) & (neg["coherence"] >= 50)
        print(f"Sycophancy unter fördernden Prompts: {pos[TRAIT].mean():.1f}, unter unterdrückenden: {neg[TRAIT].mean():.1f}")
        print(f"{passed.sum()} von {len(passed)} Antwortpaaren bestehen den Filter")

    if section("2 Layer-Suche (Koeffizient 2)", "steering/layer_sweep.csv"):
        sweep = pd.read_csv(RESULTS / "steering" / "layer_sweep.csv")
        print(sweep.groupby("layer")[[TRAIT, "coherence"]].mean().round(1).T.to_string())

    if section("3 Steering-Kurve", "steering/steering_curve.csv"):
        curve = pd.read_csv(RESULTS / "steering" / "steering_curve.csv")
        print(f"Ohne Eingriff: Sycophancy {curve[curve['coef'] == 0][TRAIT].mean():.1f}")
        print(curve[curve["coef"] > 0].groupby(["layer", "coef"])[[TRAIT, "coherence"]].mean().round(1).to_string())

    if section("4 Kontrollen", "steering/controls.csv"):
        controls = pd.read_csv(RESULTS / "steering" / "controls.csv")
        print(controls.groupby(["condition", "layer", "coef"])[[TRAIT, "coherence"]].mean().round(1).to_string())

    if section("5 Monitoring", "monitoring/monitoring.csv"):
        mon = pd.read_csv(RESULTS / "monitoring" / "monitoring.csv")
        columns = [c for c in mon.columns if c.startswith("proj_")]
        points = mon.groupby(["level", "question_id"])[columns + [TRAIT]].mean().reset_index()
        centered = points[columns + [TRAIT]] - points.groupby("level")[columns + [TRAIT]].transform("mean")
        for column in columns:
            print(f"Layer {column[5:]}: r gesamt {points[column].corr(points[TRAIT]):.2f}, "
                  f"innerhalb gleicher System-Prompts {centered[column].corr(centered[TRAIT]):.2f}")

    if section("6 Zweiter Judge (Haiku 5.5 gegen Opus 5.5)", "judge_check/second_judge.csv"):
        second = pd.read_csv(RESULTS / "judge_check" / "second_judge.csv")
        for metric in (TRAIT, "coherence"):
            print(f"{metric}: r = {second[metric].corr(second[f'{metric}_haiku']):.2f}")

    if section("7 Längenkontrollierter Vektor", "steering/length_control.csv"):
        control = pd.read_csv(RESULTS / "steering" / "length_control.csv")
        print(control.groupby(["layer", "coef"])[[TRAIT, "coherence"]].mean().round(1).to_string())

    if section("8 Neue Fragenformate", "new_formats/new_formats.csv"):
        new = pd.read_csv(RESULTS / "new_formats" / "new_formats.csv")
        print(new.groupby(["format", "condition"], sort=False)[[TRAIT, "correction", "evil"]].mean().round(1).to_string())

    if section("9 Trait-Landkarte (Kosinus-Ähnlichkeit auf Layer 22)", "trait_map/cosine_all_layers.csv"):
        cosines = pd.read_csv(RESULTS / "trait_map" / "cosine_all_layers.csv")
        matrix = cosines[cosines["layer"] == 22].pivot(index="trait_a", columns="trait_b", values="cosine")
        print(matrix.round(2).to_string())


if __name__ == "__main__":
    main()
