"""Erweiterung A: Logit Lens auf den Persona-Vektor.

Der Vektor wird direkt mit der Unembedding-Matrix multipliziert. Das Ergebnis sagt,
welche Tokens wahrscheinlicher (gefördert) oder unwahrscheinlicher (gehemmt) würden,
wenn man die Richtung unmittelbar vor der Ausgabeschicht addierte. Für mittlere Layer
ist das nur eine Näherung, weil die folgenden Layer die Richtung noch weiterverarbeiten.
"""
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pv_common import RESULTS, load_model

LAYERS = [14, 18, 22, 26, 30, 34]
TOP_K = 40
OUT = RESULTS / "logit_lens"


@torch.no_grad()
def main():
    model, tok = load_model()
    unembed = model.lm_head.weight.float()  # [Vokabular, hidden]
    OUT.mkdir(parents=True, exist_ok=True)
    for trait in ("sycophantic", "evil", "hallucinating"):
        path = RESULTS / "vectors" / f"{trait}_response_avg_diff.pt"
        if not path.exists():
            continue
        vectors = torch.load(path)
        rows = []
        for layer in LAYERS:
            direction = vectors[layer].to(unembed.device)
            logits = unembed @ (direction / direction.norm())
            for name, indices in (("gefördert", logits.topk(TOP_K).indices), ("gehemmt", (-logits).topk(TOP_K).indices)):
                for rank, index in enumerate(indices.tolist(), 1):
                    rows.append(dict(layer=layer, richtung=name, rang=rank, token=tok.decode([index]), wert=float(logits[index])))
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"{trait}_tokens.csv", index=False)
        for layer in (22, 34):
            for name in ("gefördert", "gehemmt"):
                tokens = df[(df["layer"] == layer) & (df["richtung"] == name)]["token"].tolist()
                print(f"{trait} | Layer {layer} | {name}: {tokens}")


if __name__ == "__main__":
    main()
