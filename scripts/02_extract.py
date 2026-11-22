"""Experiment 1: Persona-Vektor extrahieren. Drei Stufen, jede speichert ihr Ergebnis.

    python scripts/02_extract.py generate   # Antworten unter pos/neg System-Prompts (lokal, kostenlos)
    python scripts/02_extract.py judge      # Judge bewertet Trait und Kohärenz (API, kostet Geld)
    python scripts/02_extract.py vector     # Differenz der mittleren Aktivierungen pro Layer

Optionen:
    --trait evil          anderer Trait aus dem Original-Repo (Voreinstellung: sycophantic)
    --samples 2           Antworten je Frage und System-Prompt (Voreinstellung: 5)
    --first-tokens 50     nur für `vector`: nur die ersten N Antwort-Tokens mitteln (Längenkontrolle)
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pv_common import RESULTS, build_prompt, generate, load_model
from pv_judge import Judge, coherence_rubric, load_trait

OUT = RESULTS / "extract"
VEC = RESULTS / "vectors"


def a_or_an(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def stage_generate(trait, n_samples):
    data = load_trait(trait, "extract")
    model, tok = load_model()
    OUT.mkdir(parents=True, exist_ok=True)
    # Wie im Original-Repo empfohlen: pos nennt den Trait, neg heißt "helpful"
    assistant_name = {"pos": trait, "neg": "helpful"}
    for kind in ("pos", "neg"):
        path = OUT / f"{trait}_{kind}.csv"
        # Vorhandene Antworten bleiben erhalten; erzeugt werden nur die fehlenden Wiederholungen
        existing = pd.read_csv(path) if path.exists() else None
        first_new = 0 if existing is None else int(existing["sample_id"].max()) + 1
        rows = []
        for q_idx, question in enumerate(data["questions"]):
            for i_idx, pair in enumerate(data["instruction"]):
                name = assistant_name[kind]
                system = f"You are {a_or_an(name)} {name} assistant. {pair[kind]}"
                for s_idx in range(first_new, n_samples):
                    rows.append(
                        dict(question=question, prompt=build_prompt(tok, question, system),
                             question_id=q_idx, instruction_id=i_idx, sample_id=s_idx)
                    )
        if not rows:
            print(f"{kind}: schon {len(existing)} Antworten vorhanden, nichts zu tun")
            continue
        df = pd.DataFrame(rows)
        torch.manual_seed(first_new)
        df["answer"] = generate(model, tok, df["prompt"].tolist(), max_new_tokens=500, batch_size=20)
        if existing is not None:
            df = pd.concat([existing, df], ignore_index=True)
        # Gleiche Sortierung in pos und neg: Der Filter paart die Antworten zeilenweise
        df = df.sort_values(["question_id", "instruction_id", "sample_id"]).reset_index(drop=True)
        df.to_csv(path, index=False)
        print(f"{kind}: {len(rows)} neue Antworten, {len(df)} insgesamt")


async def stage_judge(trait):
    judge = Judge()
    trait_rubric = load_trait(trait, "extract")["eval_prompt"]
    for kind in ("pos", "neg"):
        path = OUT / f"{trait}_{kind}.csv"
        df = pd.read_csv(path)
        for column in (trait, "coherence"):
            if column not in df:
                df[column] = float("nan")
        # Nur Antworten bewerten, die noch keinen Score haben
        todo = df[trait].isna() | df["coherence"].isna()
        questions, answers = df.loc[todo, "question"].tolist(), df.loc[todo, "answer"].fillna("").tolist()
        trait_scores, coherence_scores = await asyncio.gather(
            judge.score_many(trait_rubric, questions, answers),
            judge.score_many(coherence_rubric(), questions, answers),
        )
        df.loc[todo, trait] = pd.Series(trait_scores, index=df.index[todo], dtype="float64")
        df.loc[todo, "coherence"] = pd.Series(coherence_scores, index=df.index[todo], dtype="float64")
        df.to_csv(path, index=False)
        print(f"{kind}: {int(todo.sum())} Antworten neu bewertet")
        print(f"{kind}: {trait} {df[trait].mean():.1f}  Kohärenz {df['coherence'].mean():.1f}  "
              f"ohne Score {df[trait].isna().sum()}")
    print(judge.usage())


@torch.no_grad()
def mean_activations(model, tok, prompts, answers, first_tokens=None):
    """Je Beispiel und Layer: Mittel über die Antwort-Tokens und letzter Prompt-Token."""
    response_avg, prompt_last = [], []
    for prompt, answer in tqdm(list(zip(prompts, answers))):
        enc = tok(prompt + answer, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tok.encode(prompt, add_special_tokens=False))
        end = None if first_tokens is None else prompt_len + first_tokens
        hs = torch.stack(model(**enc, output_hidden_states=True).hidden_states)[:, 0].float()
        response_avg.append(hs[:, prompt_len:end].mean(dim=1).cpu())
        prompt_last.append(hs[:, prompt_len - 1].cpu())
    return torch.stack(response_avg), torch.stack(prompt_last)  # [Beispiele, Layer+1, hidden]


def split_half_stability(pos_resp, neg_resp, seed=0):
    """Stabilität des Vektors: Paare zufällig halbieren, aus jeder Hälfte einen Vektor bilden, Kosinus pro Layer.

    Werte nahe 1 heißen: Der Vektor hängt nicht an einzelnen Antworten.
    """
    order = torch.randperm(len(pos_resp), generator=torch.Generator().manual_seed(seed))
    first, second = order[: len(order) // 2], order[len(order) // 2:]
    half_a = pos_resp[first].mean(0) - neg_resp[first].mean(0)
    half_b = pos_resp[second].mean(0) - neg_resp[second].mean(0)
    return torch.nn.functional.cosine_similarity(half_a, half_b, dim=-1)  # [Layer+1]


def stage_vector(trait, first_tokens=None, threshold=50, stability_only=False):
    pos = pd.read_csv(OUT / f"{trait}_pos.csv")
    neg = pd.read_csv(OUT / f"{trait}_neg.csv")
    # Filter wie im Original: pos zeigt den Trait klar, neg klar nicht, beide verständlich
    mask = (pos[trait] >= threshold) & (neg[trait] < 100 - threshold) & \
           (pos["coherence"] >= 50) & (neg["coherence"] >= 50)
    print(f"{mask.sum()} von {len(mask)} Paaren bestehen den Filter")
    model, tok = load_model()
    pos_resp, pos_last = mean_activations(model, tok, pos[mask]["prompt"].tolist(), pos[mask]["answer"].tolist(), first_tokens)
    neg_resp, neg_last = mean_activations(model, tok, neg[mask]["prompt"].tolist(), neg[mask]["answer"].tolist(), first_tokens)
    VEC.mkdir(parents=True, exist_ok=True)
    response_diff = pos_resp.mean(0) - neg_resp.mean(0)  # [Layer+1, hidden]
    if first_tokens is None:
        stability = split_half_stability(pos_resp, neg_resp)
        record = dict(trait=trait, paare=int(mask.sum()), kosinus_pro_layer=[round(x, 4) for x in stability.tolist()])
        (VEC / f"{trait}_stability.json").write_text(json.dumps(record))
        print(f"Stabilität (Kosinus zweier Hälften) auf Layer 22: {stability[22]:.3f}")
    if stability_only:
        return
    if first_tokens is None:
        torch.save(response_diff, VEC / f"{trait}_response_avg_diff.pt")
        torch.save(pos_last.mean(0) - neg_last.mean(0), VEC / f"{trait}_prompt_last_diff.pt")
    else:
        torch.save(response_diff, VEC / f"{trait}_response_first{first_tokens}_diff.pt")
    print("Form:", tuple(response_diff.shape))
    print("Norm pro Layer:", [round(x, 1) for x in response_diff.norm(dim=-1).tolist()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["generate", "judge", "vector"])
    parser.add_argument("--trait", default="sycophantic")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--first-tokens", type=int, default=None)
    parser.add_argument("--stability-only", action="store_true", help="nur die Stabilität messen, Vektor nicht überschreiben")
    args = parser.parse_args()
    if args.stage == "generate":
        stage_generate(args.trait, args.samples)
    elif args.stage == "judge":
        asyncio.run(stage_judge(args.trait))
    else:
        stage_vector(args.trait, args.first_tokens, stability_only=args.stability_only)
