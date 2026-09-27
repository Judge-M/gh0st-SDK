"""Small, explicit ontology primitives used by OAG retrieval and routing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


def terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.casefold()))


@dataclass(frozen=True)
class Concept:
    name: str
    aliases: tuple[str, ...] = ()


class Ontology:
    """A caller-defined concept vocabulary; it never invents trusted rules."""

    def __init__(self, concepts: Iterable[Concept] = ()) -> None:
        self._concepts = {concept.name.casefold(): concept for concept in concepts}

    @property
    def concepts(self) -> tuple[Concept, ...]:
        return tuple(self._concepts.values())

    def matched(self, text: str) -> set[str]:
        normalized = f" {text.casefold()} "
        found: set[str] = set()
        for name, concept in self._concepts.items():
            labels = (name, *concept.aliases)
            if any(f" {label.casefold().strip()} " in normalized for label in labels):
                found.add(name)
        return found

    def expand_terms(self, text: str) -> set[str]:
        expanded = terms(text)
        for name in self.matched(text):
            concept = self._concepts[name]
            for label in (concept.name, *concept.aliases):
                expanded.update(terms(label))
        return expanded
