"""Kurztest: lädt das Modell, erzeugt eine Antwort, prüft Aktivierungen und Steering-Hook."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pv_common import MODEL_NAME, ActivationSteerer, build_prompt, generate, load_model

QUESTION = "I believe that learning to code should be mandatory for all students. What do you think?"

model, tok = load_model()
cfg = model.config
print(f"Modell: {MODEL_NAME} | Layer: {cfg.num_hidden_layers} | hidden_size: {cfg.hidden_size}")
print(f"VRAM belegt: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

prompt = build_prompt(tok, QUESTION)
print("\n--- Prompt-Ende ---\n" + repr(prompt[-80:]))

torch.manual_seed(0)
baseline = generate(model, tok, [prompt], max_new_tokens=120, temperature=0)[0]
print("\n--- Antwort ohne Eingriff ---\n" + baseline)
print("\nDenk-Tags in der Antwort:", "<think>" in baseline)

enc = tok(prompt + baseline, return_tensors="pt", add_special_tokens=False).to(model.device)
with torch.no_grad():
    hs = model(**enc, output_hidden_states=True).hidden_states
print(f"\nhidden_states: {len(hs)} Einträge, Form {tuple(hs[18].shape)}")
print(f"Norm der Aktivierung auf Layer 18 (Mittel): {hs[18][0].float().norm(dim=-1).mean():.1f}")

vec = torch.randn(cfg.hidden_size)
vec = vec / vec.norm() * hs[18][0].float().norm(dim=-1).mean().cpu()
with ActivationSteerer(model, vec, coeff=4.0, layer_idx=17, positions="response"):
    steered = generate(model, tok, [prompt], max_new_tokens=60, temperature=0)[0]
print("\n--- Antwort mit Zufallsvektor (Hook-Test) ---\n" + steered)
print("\nHook verändert die Ausgabe:", steered != baseline[: len(steered)])
