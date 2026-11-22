"""Erweiterung C: Wie verlässlich ist der Judge?

Ein zweites Modell (Claude Haiku 5.5) bewertet dieselben 660 Antworten der Steering-Kurve.
"""
import asyncio
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pv_common import RESULTS
from pv_judge import Judge, coherence_rubric, load_trait

TRAIT = "sycophantic"
SECOND_JUDGE = "claude-haiku-5-5"
OUT = RESULTS / "judge_check"


def main():
    df = pd.read_csv(RESULTS / "steering" / "steering_curve.csv")
    OUT.mkdir(parents=True, exist_ok=True)

    judge = Judge(model=SECOND_JUDGE, use_fallbacks=False)
    rubrics = {f"{TRAIT}_haiku": load_trait(TRAIT, "eval")["eval_prompt"], "coherence_haiku": coherence_rubric()}
    df = asyncio.run(judge.score_columns(df, rubrics))
    df.to_csv(OUT / "second_judge.csv", index_label="id")
    for metric in (TRAIT, "coherence"):
        pair = df[[metric, f"{metric}_haiku"]].dropna()
        print(f"{metric}: r = {pair[metric].corr(pair[f'{metric}_haiku']):.3f} | "
              f"mittlere Abweichung {(pair[metric] - pair[f'{metric}_haiku']).abs().mean():.1f} Punkte | n = {len(pair)}")
    print(judge.usage())

if __name__ == "__main__":
    main()
