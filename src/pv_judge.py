"""Judge: ein Claude-Modell bewertet Antworten auf einer Skala von 0 bis 100.

Rubriken (Trait und Kohärenz) kommen unverändert aus dem Original-Repo.
Unterschied zum Original: Dort wird der Score aus den Token-Wahrscheinlichkeiten
von gpt-4.1-mini gemittelt. Die Anthropic-API liefert keine Wahrscheinlichkeiten,
deshalb zählt hier die Zahl, die das Modell ausgibt.
"""
import asyncio
import importlib.util
import json
import os
import re

import anthropic

from pv_common import ROOT, UPSTREAM

JUDGE_MODEL = "claude-opus-5-5"


def load_env() -> None:
    """Liest ANTHROPIC_API_KEY aus der .env im Projektordner, falls vorhanden."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_trait(trait: str, version: str) -> dict:
    """Trait-Daten des Original-Repos. version ist 'extract' oder 'eval'."""
    path = UPSTREAM / "data_generation" / f"trait_data_{version}" / f"{trait}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def coherence_rubric() -> str:
    spec = importlib.util.spec_from_file_location("upstream_prompts", UPSTREAM / "eval" / "prompts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Prompts["coherence_0_100"]


def parse_score(text: str) -> float | None:
    """Zahl von 0 bis 100 aus der Judge-Antwort. None bei REFUSAL oder ohne Zahl."""
    if "REFUSAL" in text.upper():
        return None
    match = re.search(r"\d{1,3}", text)
    if not match:
        return None
    value = float(match.group())
    return value if 0 <= value <= 100 else None


class Judge:
    def __init__(self, model: str = JUDGE_MODEL, max_concurrent: int = 8, use_fallbacks: bool = True):
        load_env()
        self.model = model
        # Serverseitiges Ersatzmodell bei abgelehnten Anfragen; nicht jedes Modell unterstützt das
        self.use_fallbacks = use_fallbacks
        self.client = anthropic.AsyncAnthropic(max_retries=5)
        self._sem = asyncio.Semaphore(max_concurrent)
        self.failed = 0  # Aufrufe, die trotz Wiederholungen scheiterten
        self.fallbacks = 0  # Aufrufe, die ein Ersatzmodell beantwortet hat
        self.input_tokens = 0
        self.output_tokens = 0

    async def score(self, rubric: str, question: str, answer: str) -> float | None:
        prompt = rubric.format(question=question, answer=answer)
        request = dict(
            model=self.model,
            max_tokens=2000,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt}],
        )
        async with self._sem:
            try:
                if self.use_fallbacks:
                    response = await self.client.beta.messages.create(
                        betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request
                    )
                else:
                    response = await self.client.messages.create(**request)
            except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError):
                self.failed += 1
                return None
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        if any(e.type == "fallback_message" for e in getattr(response.usage, "iterations", None) or []):
            self.fallbacks += 1
        if response.stop_reason == "refusal":
            return None
        text = "".join(block.text for block in response.content if block.type == "text")
        return parse_score(text)

    async def score_frame(self, df, trait: str, version: str = "eval"):
        """Ergänzt einen DataFrame mit Spalten question/answer um Trait- und Kohärenz-Score."""
        questions, answers = df["question"].tolist(), df["answer"].fillna("").tolist()
        df[trait], df["coherence"] = await asyncio.gather(
            self.score_many(load_trait(trait, version)["eval_prompt"], questions, answers),
            self.score_many(coherence_rubric(), questions, answers),
        )
        return df

    async def score_columns(self, df, rubrics: dict):
        """Ergänzt einen DataFrame um je eine Spalte pro Rubrik (Spaltenname -> Rubrik-Text)."""
        questions, answers = df["question"].tolist(), df["answer"].fillna("").tolist()
        results = await asyncio.gather(*(self.score_many(rubric, questions, answers) for rubric in rubrics.values()))
        for name, scores in zip(rubrics, results):
            df[name] = scores
        return df

    def usage(self) -> str:
        return (f"Tokens: {self.input_tokens} ein, {self.output_tokens} aus | "
                f"gescheitert {self.failed} | Ersatzmodell {self.fallbacks}")

    async def score_many(self, rubric: str, questions: list[str], answers: list[str]) -> list[float | None]:
        return await asyncio.gather(*(self.score(rubric, q, a) for q, a in zip(questions, answers)))
