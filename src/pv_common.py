"""Gemeinsame Bausteine: Modell laden, Prompts bauen, Antworten erzeugen.

Ersetzt die vllm-basierte Generierung des Original-Repos, die unter Windows
nicht läuft. Steering-Hook und Trait-Daten kommen weiter aus dem Submodul.
"""
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external" / "persona_vectors"
RESULTS = ROOT / "results"
MODEL_NAME = "Qwen/Qwen3-4B"

sys.path.insert(0, str(UPSTREAM))
from activation_steer import ActivationSteerer  # noqa: E402,F401


def load_model(name: str = MODEL_NAME):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return model, tok


def build_prompt(tok, question: str, system: str | None = None) -> str:
    """Chat-Prompt als Text. Der Denkmodus von Qwen3 ist abgeschaltet."""
    messages = [{"role": "user", "content": question}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=600, temperature=1.0, batch_size=16):
    """Erzeugt je Prompt eine Antwort. Gibt nur den neu erzeugten Text zurück.

    Hinweis: Die Voreinstellung top_k=20 aus der generation_config von Qwen3 bleibt aktiv.
    Gesampelt wird also mit Temperatur `temperature` aus den 20 wahrscheinlichsten Tokens.
    """
    answers = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(
            **enc,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
        )
        prompt_len = enc["input_ids"].shape[1]
        answers.extend(tok.decode(o[prompt_len:], skip_special_tokens=True) for o in out)
    return answers
