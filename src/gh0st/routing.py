"""Fast, local System-1 routing from tasks to ephemeral worker profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterable

from .ontology import Ontology
from .tools import FunctionTool


@dataclass(frozen=True)
class WorkerProfile:
    name: str
    intent: str
    model: str
    instructions: str = ""
    keywords: tuple[str, ...] = ()
    concepts: tuple[str, ...] = ()
    tools: tuple[FunctionTool, ...] = ()
    priority: int = 0
    is_default: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.intent.strip() or not self.model.strip():
            raise ValueError("Worker name, intent, and model are required")
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError(f"Worker {self.name!r} has duplicate tool names")


@dataclass(frozen=True)
class RouteDecision:
    worker: WorkerProfile
    matched_terms: tuple[str, ...]
    score: float
    reason: str
    elapsed_ms: float


class System1WorkerRouter:
    """Selects a worker locally from explicit keywords and ontology concepts.

    This router handles worker dispatch only. It does not choose providers,
    models across profiles, tools, or secondary memory services.
    """

    def __init__(self, ontology: Ontology | None = None) -> None:
        self.ontology = ontology or Ontology()

    def route(self, prompt: str, workers: Iterable[WorkerProfile]) -> RouteDecision:
        started = perf_counter()
        profiles = tuple(workers)
        if not profiles:
            raise ValueError("At least one worker profile is required")

        prompt_lower = f" {prompt.casefold()} "
        prompt_concepts = self.ontology.matched(prompt)
        scored: list[tuple[float, int, int, WorkerProfile, tuple[str, ...]]] = []
        for index, worker in enumerate(profiles):
            matched = tuple(
                keyword
                for keyword in worker.keywords
                if f" {keyword.casefold().strip()} " in prompt_lower
            )
            concept_matches = tuple(
                concept for concept in worker.concepts if concept.casefold() in prompt_concepts
            )
            score = float(len(matched) + (2 * len(concept_matches)))
            scored.append((score, int(worker.is_default), worker.priority, worker, matched + concept_matches))

        best = max(scored, key=lambda item: (item[0], item[1], item[2], -profiles.index(item[3])))
        score, _, _, worker, matched = best
        if score:
            reason = "Matched task cues: " + ", ".join(matched)
        else:
            reason = "No task cue matched; selected the configured default profile"
        return RouteDecision(
            worker=worker,
            matched_terms=matched,
            score=score,
            reason=reason,
            elapsed_ms=(perf_counter() - started) * 1000,
        )
