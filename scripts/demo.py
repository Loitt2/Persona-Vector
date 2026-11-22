"""Live-Demo: Fragen eintippen und die Steering-Stärke einstellen.

    python scripts/demo.py

Eingaben im Programm:
    eine Frage (auf Englisch)   -> das Modell antwortet mit der aktuellen Stärke
    eine Zahl, z. B. 1 oder -1.5 -> setzt die Steering-Stärke (0 = kein Eingriff)
    leere Zeile                  -> beendet die Demo

Gesteuert wird auf Layer 22 mit dem Sycophancy-Vektor. Sinnvoller Bereich: -1.5 bis 1.5.
Positive Werte machen das Modell schmeichlerischer, negative ehrlicher; über 1.5 zerfallen die Antworten.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pv_common import RESULTS, ActivationSteerer, build_prompt, generate, load_model

LAYER = 22


def main():
    vector = torch.load(RESULTS / "vectors" / "sycophantic_response_avg_diff.pt")[LAYER]
    print("Lade Qwen3-4B ...")
    model, tok = load_model()
    coef = 0.0
    print(__doc__)
    while True:
        text = input(f"\n[Stärke {coef:+.1f}] Frage oder neue Stärke: ").strip()
        if not text:
            break
        try:
            coef = float(text.replace(",", "."))
            continue
        except ValueError:
            pass
        prompt = [build_prompt(tok, text)]
        if coef == 0:
            answer = generate(model, tok, prompt, max_new_tokens=200, temperature=0)[0]
        else:
            with ActivationSteerer(model, vector, coeff=coef, layer_idx=LAYER - 1, positions="response"):
                answer = generate(model, tok, prompt, max_new_tokens=200, temperature=0)[0]
        print("\n" + answer)


if __name__ == "__main__":
    main()
