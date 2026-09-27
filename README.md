<div align="center">

# 👻 gh0st SDK

### Fresh workers. Clean context. Continuity that lasts.

An in-process Python SDK for dispatching ephemeral workers, carrying context through built-in OAG memory, and returning verifiable task reports.

**Use the SDK directly with your model provider. The gh0st Gateway is optional and separate.**

[Install](#install) · [Quick start](#quick-start) · [Architecture](#what-lives-where) · [License](#license)

</div>

---

## Why gh0st

Each task gets a fresh worker context. System-1 routes the task to a matching worker profile, the worker completes its turn, and its temporary transcript is discarded. The built-in OAG database carries approved rules and useful continuity into the next task. `Gh0stSDK` manages this lifecycle; `EphemeralWorker` is also available for integrations that prepare and execute a single worker turn directly.

| 👻 Ephemeral workers | 🧭 System-1 dispatch | 🧠 Built-in OAG |
| --- | --- | --- |
| A new execution loop for each ticket | Fast local routing to a configured worker role | SQLite memory persists after a worker exits |
| Bounded by turn and tool-call limits | Routes task intent to worker capability | Trusted rules stay distinct from unverified references |
| Reports status, tokens, tools, and failures | No model call needed to choose a worker | Optional secondary memory adapters are caller-selected |

## Install

```bash
pip install gh0st-sdk
```

The core package uses the Python standard library. It includes a direct OpenAI-compatible provider adapter; no proxy service or gh0st Gateway is required.

## Quick start

```python
import os
from pathlib import Path

from gh0st import (
    FunctionTool,
    Gh0stSDK,
    OpenAICompatibleProvider,
    SQLiteOAG,
    SQLiteStateLedger,
    Ticket,
    WorkerProfile,
)


def find_files(query: str) -> list[str]:
    """Replace this body with your application's bounded workspace search."""
    return [f"Search requested for: {query}"]


search_tool = FunctionTool(
    name="find_files",
    description="Search the current task workspace for relevant files.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
    handler=find_files,
    capability="workspace.search",
)

sdk = Gh0stSDK(
    provider=OpenAICompatibleProvider(
        api_key=os.environ["MODEL_API_KEY"],
        base_url=os.getenv("MODEL_BASE_URL", "https://api.openai.com/v1"),
    ),
    workers=(
        WorkerProfile(
            name="implementer",
            intent="code implementation",
            model="provider:code-model",
            instructions="Make the requested code change, then explain the result.",
            keywords=("implement", "fix", "refactor", "add"),
            tools=(search_tool,),
            is_default=True,
        ),
    ),
    memory=SQLiteOAG(".gh0st/oag.sqlite3"),
    ledger=SQLiteStateLedger(".gh0st/state.sqlite3"),
)

ticket = Ticket(
    prompt="Find the login handler and explain where token validation belongs.",
    scope="repo:my-service",
    workspace_path=str(Path("./my-service").resolve()),
    permitted_local_capabilities=("workspace.search",),
)
report = sdk.run(ticket)

print(report.status, report.worker_name, report.usage.total_tokens)
print(report.output)
```

The ticket grants only `workspace.search`, so this worker receives only that registered tool. A later ticket gets a new transcript; relevant OAG records are retrieved and labeled in its context.

## What lives where

```text
Your app or agent framework
        │ calls Gh0stSDK.run(ticket)
        ▼
Public SDK ── System-1 selects a worker profile
        ├────── SQLite OAG supplies trusted rules and continuity
        ├────── Configured tools run for the permitted capabilities
        ├────── Configured model provider is called directly
        └────── Ticket and WorkerReport are saved in the local ledger

Optional, separate gh0st Gateway
        └────── Dynamic model, tool, and secondary-memory routing;
               gateway-owned budgets and provider policy
```

The public SDK owns **which worker handles a task** and the worker lifecycle through `System1WorkerRouter` and `EphemeralWorker`. The Gateway’s advanced resource routing answers **which model, tool, or secondary memory provider a worker turn should use**. Without that Gateway, the application configures models and tool sets directly and chooses any secondary providers explicitly.

The SDK always maintains its own OAG continuity store. External memory results are supplemental and labeled as unverified reference context; they do not replace or write into the OAG database.

## Use it inside existing agent products

Treat gh0st as a specialist execution tool inside the workflow you already have:

- **LangGraph:** call `sdk.run(ticket)` from a graph node when the workflow reaches a repository task.
- **CrewAI:** wrap it as a custom tool for a developer agent.
- **OpenAI Agents SDK:** expose it as a function tool or bounded specialist.

The existing product owns the outer workflow. gh0st routes and runs its own ephemeral worker for the delegated task. These are integration patterns; framework-specific adapters are not bundled yet.

## OAG trust model

Memory records have a type, scope, concepts, source, and trust status.

- Rules enter the database untrusted. Call `SQLiteOAG.approve_rule(..., approved_by=...)` after a human approves one; only then are they injected as **System Instructions / Constraints**.
- Continuity records and retrieved references are provided as **Retrieved Reference Context**, separate from system instructions.
- Optional secondary providers implement `SecondaryMemoryProvider`. The SDK queries only the providers supplied by the caller and tags each result with its source.

## Execution and budget boundaries

The SDK enforces local turn, tool-call, output-token, and explicitly granted capability limits. It claims tickets atomically and writes each worker report together with its final ticket status in one SQLite transaction. Ticket fields such as `max_cost_usd` and `gateway_allowance` are pass-through metadata; the SDK does not make financial admission decisions or account for provider spend. Without the Gateway, the embedding application owns that policy.

Tools are application-supplied Python functions and run with the host process's permissions. Capability filtering limits which registered tools are exposed to a worker; it is not an operating-system sandbox. Use an isolated execution environment for untrusted commands or code.

## CLI and REPL

Set `GH0ST_MODEL` and `GH0ST_API_KEY` (or `OPENAI_API_KEY`), then run:

```bash
gh0st                         # interactive session
gh0st run "Summarize this project"
gh0st run --model vendor:model "Review the API design"
```

`GH0ST_BASE_URL` selects a direct OpenAI-compatible endpoint. OAG and ticket state default to `~/.gh0st/`; override them with `GH0ST_OAG_DB` and `GH0ST_STATE_DB`.

## License

Apache-2.0. See [LICENSE](LICENSE).
