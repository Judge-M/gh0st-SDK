"""Atomic local ticket claiming and report persistence."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .contracts import Ticket, WorkerReport


class DuplicateTicketError(RuntimeError):
    """Raised when a ticket ID is submitted for execution more than once."""


class SQLiteStateLedger:
    """SQLite ledger for ticket state and final worker reports.

    Quota and provider-spend accounting are intentionally absent; those belong
    to a gateway or the embedding application.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._connection.row_factory = sqlite3.Row
        self._connection.isolation_level = None
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS tickets (
                ticket_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                ticket_json TEXT NOT NULL,
                error TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS worker_reports (
                ticket_id TEXT PRIMARY KEY REFERENCES tickets(ticket_id),
                report_json TEXT NOT NULL,
                saved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )

    def claim(self, ticket: Ticket) -> None:
        payload = json.dumps(asdict(ticket), ensure_ascii=False, default=str)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute(
                    "INSERT INTO tickets(ticket_id, status, ticket_json) VALUES (?, 'RUNNING', ?)",
                    (ticket.ticket_id, payload),
                )
            except sqlite3.IntegrityError as exc:
                self._connection.execute("ROLLBACK")
                raise DuplicateTicketError(
                    f"Ticket {ticket.ticket_id!r} has already been claimed"
                ) from exc
            self._connection.execute("COMMIT")

    def save_report(self, report: WorkerReport) -> None:
        payload = json.dumps(asdict(report), ensure_ascii=False, default=str)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT status FROM tickets WHERE ticket_id = ?", (report.ticket_id,)
                ).fetchone()
                if row is None or row["status"] != "RUNNING":
                    raise KeyError(f"Ticket {report.ticket_id!r} is not running")
                self._connection.execute(
                    "INSERT INTO worker_reports(ticket_id, report_json) VALUES (?, ?)",
                    (report.ticket_id, payload),
                )
                self._connection.execute(
                    "UPDATE tickets SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE ticket_id = ?",
                    (report.status.upper(), report.ticket_id),
                )
            except Exception:
                self._connection.execute("ROLLBACK")
                raise
            self._connection.execute("COMMIT")

    def fail(self, ticket_id: str, error: str) -> None:
        with self._lock:
            self._connection.execute(
                "UPDATE tickets SET status = 'FAILED', error = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE ticket_id = ? AND status = 'RUNNING'",
                (error, ticket_id),
            )

    def status(self, ticket_id: str) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT status FROM tickets WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
        return str(row["status"]) if row else None

    def ticket(self, ticket_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT ticket_json FROM tickets WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
        return json.loads(row["ticket_json"]) if row else None

    def report(self, ticket_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT report_json FROM worker_reports WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
        return json.loads(row["report_json"]) if row else None

    def close(self) -> None:
        with self._lock:
            self._connection.close()
