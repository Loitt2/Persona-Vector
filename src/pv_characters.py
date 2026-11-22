"""Bausteine für die Versuche mit Figuren und für Eingriffe, die über Layer vergleichbar sind."""
import torch

from pv_common import build_prompt

REFERENCE_LAYER = 22
REFERENCE_COEF = 1.25  # Stärke, bei der Sycophancy-Steering auf Layer 22 gut wirkte und verständlich blieb


def system_prompt(character: str) -> str:
    if character == "Assistant":
        return "You are a helpful assistant."
    return f"You are {character}. Stay fully in character and answer as {character} would."


@torch.no_grad()
def answer_activations(model, tok, prompts, answers, layer=None):
    """Mittlere Aktivierung jeder Antwort, ohne Eingriff gemessen.

    Mit `layer`: Form [Antworten, hidden]. Ohne: alle Layer, Form [Antworten, Layer+1, hidden].
    """
    means = []
    for prompt, answer in zip(prompts, answers):
        prompt_len = len(tok.encode(prompt, add_special_tokens=False))
        enc = tok(prompt + answer, return_tensors="pt", add_special_tokens=False).to(model.device)
        hidden_states = model(**enc, output_hidden_states=True).hidden_states
        if layer is None:
            means.append(torch.stack([h[0, prompt_len:].float().mean(0) for h in hidden_states]).cpu())
        else:
            means.append(hidden_states[layer][0, prompt_len:].float().mean(0).cpu())
    return torch.stack(means)


@torch.no_grad()
def activation_norms(model, tok, questions, answers):
    """Typische Länge der Aktivierung eines Antwort-Tokens auf jedem Layer. Form [Layer+1]."""
    totals = None
    for question, answer in zip(questions, answers):
        prompt = build_prompt(tok, question)
        prompt_len = len(tok.encode(prompt, add_special_tokens=False))
        enc = tok(prompt + answer, return_tensors="pt", add_special_tokens=False).to(model.device)
        hidden_states = model(**enc, output_hidden_states=True).hidden_states
        norms = torch.stack([h[0, prompt_len:].float().norm(dim=-1).mean() for h in hidden_states]).cpu()
        totals = norms if totals is None else totals + norms
    return totals / len(questions)


def relative_strength(reference_vector, norms):
    """Eingriff im Verhältnis zur Aktivierung: so stark wie REFERENCE_COEF mal Sycophancy-Vektor auf Layer 22."""
    return REFERENCE_COEF * float(reference_vector.norm()) / float(norms[REFERENCE_LAYER])


def scaled_vector(vector, norms, layer, strength):
    """Richtung von `vector`, Länge = strength mal typische Aktivierungslänge auf diesem Layer."""
    return vector / vector.norm() * strength * norms[layer]
