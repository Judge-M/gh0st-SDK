from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import patch
from uuid import uuid4

from gh0st import (
    Concept,
    DuplicateTicketError,
    EphemeralWorker,
    ExecutionLimits,
    FunctionTool,
    Gh0stSDK,
    MemoryKind,
    ModelResponse,
    OpenAICompatibleProvider,
    Ontology,
    ReferenceHit,
    SQLiteOAG,
    SQLiteStateLedger,
    System1WorkerRouter,
    Ticket,
    ToolCall,
    Usage,
    WorkerProfile,
    WorkerReport,
)


class FakeProvider:
    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        max_output_tokens: int,
    ) -> ModelResponse:
        self.calls.append(
            {
                "model": model,
                "messages": [dict(message) for message in messages],
                "tools": list(tools),
                "max_output_tokens": max_output_tokens,
            }
        )
        return self.responses.pop(0)


class Gh0stSDKTests(unittest.TestCase):
    def test_ephemeral_worker_is_public_and_runs_a_single_clean_turn(self) -> None:
        provider = FakeProvider([ModelResponse(content="Completed one isolated turn.")])
        worker = EphemeralWorker(
            provider=provider,
            profile=WorkerProfile("writer", "writing", "direct-model", is_default=True),
        )

        result = worker.execute("Write a short status update")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.output, "Completed one isolated turn.")
        self.assertEqual(result.turns, 1)
        self.assertEqual(provider.calls[0]["messages"][0]["content"], "You are the writing worker.")

    def test_system1_routes_to_worker_by_ontology_concept(self) -> None:
        ontology = Ontology([Concept("testing", ("pytest", "unit test"))])
        workers = (
            WorkerProfile("coder", "implementation", "code-model", keywords=("refactor",)),
            WorkerProfile("tester", "testing", "test-model", concepts=("testing",)),
            WorkerProfile("default", "general", "general-model", is_default=True),
        )

        route = System1WorkerRouter(ontology).route("Please add a unit test", workers)

        self.assertEqual(route.worker.name, "tester")
        self.assertEqual(route.matched_terms, ("testing",))
        self.assertLess(route.elapsed_ms, 100)

    def test_oag_rule_requires_explicit_approval_and_preserves_scope(self) -> None:
        db_path = Path.cwd() / "tests" / f".oag-{uuid4()}.sqlite3"
        try:
            store = SQLiteOAG(db_path)
            rule = store.remember(
                "Authentication decisions must use the shared JWT verifier.",
                scope="repo:auth",
                kind=MemoryKind.RULE,
                concepts=("authentication",),
            )

            self.assertFalse(rule.is_trusted)
            self.assertFalse(store.get(rule.record_id).is_trusted)
            approved = store.approve_rule(rule.record_id, approved_by="operator")
            self.assertTrue(approved.is_trusted)
            self.assertEqual(approved.approved_by, "operator")
            self.assertEqual(
                store.recall(scope="repo:auth", query="JWT authentication", limit=3)[0].record_id,
                rule.record_id,
            )
            self.assertEqual(store.recall(scope="repo:other", query="JWT authentication"), ())
            store.close()
        finally:
            db_path.unlink(missing_ok=True)

    def test_worker_turn_executes_only_ticket_capabilities_and_returns_report(self) -> None:
        invoked: list[dict[str, str]] = []
        tool = FunctionTool(
            name="write_note",
            description="Write a note in the task workspace.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
                "required": ["path", "text"],
            },
            handler=lambda **values: invoked.append(values) or "written",
            capability="workspace.write",
        )
        provider = FakeProvider(
            [
                ModelResponse(
                    content=None,
                    tool_calls=(ToolCall("call-1", "write_note", {"path": "a.txt", "text": "hello"}),),
                    usage=Usage(12, 2),
                ),
                ModelResponse(content="The note was written.", usage=Usage(8, 5)),
            ]
        )
        sdk = Gh0stSDK(
            provider=provider,
            workers=(
                WorkerProfile(
                    "implementer",
                    "implementation",
                    "direct-model",
                    keywords=("implement",),
                    tools=(tool,),
                    is_default=True,
                ),
            ),
            memory=SQLiteOAG(),
            ledger=SQLiteStateLedger(),
        )

        report = sdk.run(
            Ticket(
                "Implement the note change",
                scope="repo:test",
                permitted_local_capabilities=("workspace.write",),
            )
        )

        self.assertEqual(invoked, [{"path": "a.txt", "text": "hello"}])
        self.assertEqual(report.status, "completed")
        self.assertEqual(report.worker_name, "implementer")
        self.assertEqual(report.turns, 2)
        self.assertEqual(report.tool_calls, 1)
        self.assertEqual(report.usage.total_tokens, 27)
        self.assertEqual(provider.calls[0]["tools"][0]["function"]["name"], "write_note")
        self.assertEqual(provider.calls[1]["messages"][-1]["role"], "tool")
        self.assertEqual(provider.calls[1]["messages"][-1]["content"], "written")

    def test_worker_starts_each_ticket_with_new_transcript_and_oag_continuity(self) -> None:
        provider = FakeProvider(
            [ModelResponse(content="Updated the handler."), ModelResponse(content="Added the tests.")]
        )
        sdk = Gh0stSDK(
            provider=provider,
            workers=(WorkerProfile("coder", "code", "direct-model", is_default=True),),
            memory=SQLiteOAG(),
            ledger=SQLiteStateLedger(),
        )

        sdk.run("Refactor the JWT handler", scope="repo:auth")
        second = sdk.run("Now add tests for that", scope="repo:auth")

        second_messages = provider.calls[1]["messages"]
        self.assertEqual([message["role"] for message in second_messages], ["system", "user"])
        self.assertEqual(second_messages[-1]["content"], "Now add tests for that")
        self.assertIn("Refactor the JWT handler", second_messages[0]["content"])
        self.assertTrue(second.memory_ids)

    def test_ungranted_tool_is_not_exposed_or_executed(self) -> None:
        invoked: list[bool] = []
        tool = FunctionTool(
            "delete_data",
            "Delete data",
            {"type": "object", "properties": {}},
            lambda: invoked.append(True),
            capability="data.delete",
        )
        provider = FakeProvider(
            [
                ModelResponse(tool_calls=(ToolCall("c1", "delete_data", {}),)),
                ModelResponse(content="I could not run the ungranted action."),
            ]
        )
        sdk = Gh0stSDK(
            provider=provider,
            workers=(WorkerProfile("default", "general", "m", tools=(tool,), is_default=True),),
            memory=SQLiteOAG(),
            ledger=SQLiteStateLedger(),
        )

        report = sdk.run("Delete the old data", scope="repo:test")

        self.assertEqual(invoked, [])
        self.assertEqual(provider.calls[0]["tools"], [])
        self.assertEqual(report.status, "completed")
        self.assertTrue(any("not permitted" in warning for warning in report.warnings))

    def test_secondary_memory_is_tagged_as_unverified_reference(self) -> None:
        class PineconeAdapter:
            name = "Pinecone"

            def retrieve(self, *, query: str, scope: str, limit: int):
                assert query == "Where is token validation?"
                assert scope == "repo:auth"
                return [ReferenceHit("JWT helper is in auth/token.py", "Pinecone")]

        provider = FakeProvider([ModelResponse(content="The helper is auth/token.py.")])
        sdk = Gh0stSDK(
            provider=provider,
            workers=(WorkerProfile("default", "general", "m", is_default=True),),
            memory=SQLiteOAG(),
            ledger=SQLiteStateLedger(),
            reference_providers=(PineconeAdapter(),),
        )

        report = sdk.run("Where is token validation?", scope="repo:auth")

        system_context = provider.calls[0]["messages"][0]["content"]
        self.assertIn("[Retrieved Reference Context - Pinecone; unverified]", system_context)
        self.assertIn("JWT helper is in auth/token.py", system_context)
        self.assertEqual(report.status, "completed")

    def test_token_allocation_clamps_each_direct_provider_request(self) -> None:
        provider = FakeProvider(
            [
                ModelResponse(
                    tool_calls=(ToolCall("c1", "read", {}),),
                    usage=Usage(input_tokens=4, output_tokens=3),
                ),
                ModelResponse(content="done"),
            ]
        )
        tool = FunctionTool("read", "Read", {"type": "object", "properties": {}}, lambda: "ok")
        sdk = Gh0stSDK(
            provider=provider,
            workers=(WorkerProfile("default", "general", "m", tools=(tool,), is_default=True),),
            memory=SQLiteOAG(),
            ledger=SQLiteStateLedger(),
            limits=ExecutionLimits(max_output_tokens=20),
        )

        sdk.run(Ticket("Read this", permitted_local_capabilities=("read",), max_tokens_allocated=10))

        self.assertEqual([call["max_output_tokens"] for call in provider.calls], [10, 3])

    def test_state_ledger_claim_and_report_transition_are_atomic(self) -> None:
        ledger = SQLiteStateLedger()
        ticket = Ticket("test ledger")
        ledger.claim(ticket)
        report = WorkerReport(
            ticket_id=ticket.ticket_id,
            worker_name="default",
            intent="general",
            status="completed",
            output="ok",
            turns=1,
            tool_calls=0,
            usage=Usage(1, 1),
        )

        ledger.save_report(report)

        self.assertEqual(ledger.status(ticket.ticket_id), "COMPLETED")
        self.assertEqual(ledger.report(ticket.ticket_id)["output"], "ok")
        with self.assertRaises(DuplicateTicketError):
            ledger.claim(ticket)

    def test_direct_provider_posts_to_compatible_chat_endpoint_without_gateway(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                    }
                ).encode()

        adapter = OpenAICompatibleProvider(api_key="test-key", base_url="https://provider.example/v1")
        with patch("gh0st.providers.urllib.request.urlopen", return_value=Response()) as request:
            response = adapter.complete(
                model="vendor:model",
                messages=[{"role": "user", "content": "hello"}],
                tools=[],
                max_output_tokens=12,
            )

        sent = json.loads(request.call_args.args[0].data.decode())
        self.assertEqual(request.call_args.args[0].full_url, "https://provider.example/v1/chat/completions")
        self.assertEqual(sent["model"], "vendor:model")
        self.assertEqual(sent["max_tokens"], 12)
        self.assertEqual(response.content, "hello")
        self.assertEqual(response.usage.total_tokens, 3)


if __name__ == "__main__":
    unittest.main()
