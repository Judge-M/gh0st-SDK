"""Built-in OAG continuity storage and optional secondary-memory contracts."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Protocol, Sequence
from uuid import uuid4

from .ontology import Ontology, terms


class MemoryKind(StrEnum):
    RULE = "rule"
    CONTINUITY = "continuity"
    REFERENCE = "reference"


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    scope: str
    kind: MemoryKind
    content: str
    concepts: tuple[str, ...]
    is_trusted: bool
    source: str
    created_at: str
    approved_by: str | None = None


@dataclass(frozen=True)
class ReferenceHit:
    content: str
    source: str
    score: float = 0.0
    concepts: tuple[str, ...] = ()


class SecondaryMemoryProvider(Protocol):
    """Adapter contract for an explicitly configured external memory source."""

    name: str

    def retrieve(self, *, query: str, scope: str, limit: int) -> Sequence[ReferenceHit]: ...


class SQLiteOAG:
    """Local OAG store for continuity and human-approved system rules.

    Rules start untrusted. Only :meth:`approve_rule` promotes one into system
    instructions. Retrieved records keep their provenance and trust label.
    """

    def __init__(self, path: str | Path = ":memory:", *, ontology: Ontology | None = None) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self.ontology = ontology or Ontology()
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS oag_records (
                record_id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                concepts_json TEXT NOT NULL,
                is_trusted INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                approved_by TEXT,
                approved_at TEXT
            )
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_oag_scope_created "
            "ON oag_records(scope, created_at DESC)"
        )
        self._connection.commit()

    def remember(
        self,
        content: str,
        *,
        scope: str = "default",
        kind: MemoryKind | str = MemoryKind.REFERENCE,
        concepts: Sequence[str] = (),
        source: str = "user",
    ) -> MemoryRecord:
        if not content.strip():
            raise ValueError("Memory content must not be empty")
        if not scope.strip():
            raise ValueError("Memory scope must not be empty")
        record_kind = MemoryKind(kind)
        record = MemoryRecord(
            record_id=str(uuid4()),
            scope=scope,
            kind=record_kind,
            content=content,
            concepts=tuple(sorted({item.casefold().strip() for item in concepts if item.strip()})),
            is_trusted=False,
            source=source,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._connection.execute(
                """INSERT INTO oag_records
                   (record_id, scope, kind, content, concepts_json, is_trusted, source, created_at)
                   VALUES (?, ?, ?, ?, ?, 0, ?, ?)""",
                (
                    record.record_id,
                    record.scope,
                    record.kind.value,
                    record.content,
                    json.dumps(record.concepts),
                    record.source,
                    record.created_at,
                ),
            )
            self._connection.commit()
        return record

    def approve_rule(self, record_id: str, *, approved_by: str) -> MemoryRecord:
        """Mark a rule as trusted after an explicit operator approval."""

        if not approved_by.strip():
            raise ValueError("approved_by is required")
        approved_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cursor = self._connection.execute(
                """UPDATE oag_records
                   SET is_trusted = 1, approved_by = ?, approved_at = ?
                   WHERE record_id = ? AND kind = ?""",
                (approved_by, approved_at, record_id, MemoryKind.RULE.value),
            )
            self._connection.commit()
            if cursor.rowcount != 1:
                raise KeyError(f"Unapproved rule {record_id!r} was not found")
            row = self._connection.execute(
                "SELECT * FROM oag_records WHERE record_id = ?", (record_id,)
            ).fetchone()
        return self._from_row(row)

    def recall(self, *, scope: str, query: str, limit: int = 8) -> tuple[MemoryRecord, ...]:
        if limit <= 0:
            return ()
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM oag_records
                   WHERE scope = ? OR scope = 'global'
                   ORDER BY created_at DESC""",
                (scope,),
            ).fetchall()
        if not rows:
            return ()

        query_terms = self.ontology.expand_terms(query)
        query_concepts = self.ontology.matched(query)
        query_is_followup = bool(terms(query) & {"it", "that", "those", "same", "continue", "next"})
        ranked: list[tuple[float, str, MemoryRecord]] = []
        for row in rows:
            record = self._from_row(row)
            record_terms = terms(record.content)
            overlap = len(query_terms & record_terms)
            concept_overlap = len(query_concepts & set(record.concepts))
            score = float(overlap + 3 * concept_overlap)
            if query_is_followup and record.kind == MemoryKind.CONTINUITY:
                score += 1.5
            if score > 0 or not query_terms:
                ranked.append((score, record.created_at, record))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return tuple(item[2] for item in ranked[:limit])

    def get(self, record_id: str) -> MemoryRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM oag_records WHERE record_id = ?", (record_id,)
            ).fetchone()
        return self._from_row(row) if row else None

    def forget(self, record_id: str) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM oag_records WHERE record_id = ?", (record_id,)
            )
            self._connection.commit()
        return cursor.rowcount == 1

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _from_row(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            record_id=row["record_id"],
            scope=row["scope"],
            kind=MemoryKind(row["kind"]),
            content=row["content"],
            concepts=tuple(json.loads(row["concepts_json"])),
            is_trusted=bool(row["is_trusted"]),
            source=row["source"],
            created_at=row["created_at"],
            approved_by=row["approved_by"],
        )
