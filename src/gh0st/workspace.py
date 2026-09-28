"""Bounded Linux workspace execution for one gh0st ticket."""

from __future__ import annotations

import os
import platform
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tools import FunctionTool


class WorkspaceIsolationError(RuntimeError):
    """Raised when a workspace cannot be executed inside the configured boundary."""


@dataclass(frozen=True)
class CommandResult:
    command: str
    purpose: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    truncated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "purpose": self.purpose,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "timed_out": self.timed_out,
            "truncated": self.truncated,
        }


class LinuxWorkspaceExecutor:
    """Run ticket commands with only the task workspace writable and no network.

    Model requests use the host process only to launch bubblewrap. Commands run
    in a separate mount, PID, user, IPC, UTS, and network namespace. If the
    isolation runtime is unavailable, execution fails closed.
    """

    PURPOSES = {"inspect", "edit", "test", "lint", "git", "other"}

    def __init__(
        self,
        workspace_path: str,
        *,
        source_commit: str | None = None,
        timeout_seconds: int = 90,
        output_limit_bytes: int = 1_048_576,
        runtime_path: str | None = None,
    ) -> None:
        if platform.system() != "Linux":
            raise WorkspaceIsolationError("Bounded local execution currently requires Linux and bubblewrap")
        if timeout_seconds < 1 or output_limit_bytes < 1024:
            raise ValueError("Workspace command limits must be positive")

        original = Path(workspace_path).expanduser()
        if not original.is_absolute() or original.is_symlink():
            raise ValueError("Ticket workspace must be an absolute, non-symlink directory")
        self.workspace = original.resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError("Ticket workspace must be a directory")

        self.bwrap = shutil.which("bwrap")
        if not self.bwrap:
            raise WorkspaceIsolationError("bubblewrap (bwrap) is required; host execution is disabled")

        self.source_commit = source_commit
        if source_commit is not None:
            if len(source_commit) not in {40, 64} or any(c not in "0123456789abcdefABCDEF" for c in source_commit):
                raise ValueError("source_commit must be a full Git commit hash")
            actual = self._host_git("rev-parse", "HEAD")
            if actual.casefold() != source_commit.casefold():
                raise ValueError("Workspace HEAD does not match the ticket source_commit")

        self.timeout_seconds = timeout_seconds
        self.output_limit_bytes = output_limit_bytes
        configured_runtime = runtime_path or os.getenv("GH0ST_SANDBOX_RUNTIME")
        self.runtime = Path(configured_runtime).expanduser().resolve(strict=True) if configured_runtime else None
        if self.runtime and (not self.runtime.is_dir() or self.runtime == self.workspace):
            raise ValueError("Sandbox runtime must be a separate read-only directory")

    def _host_git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.workspace), *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            check=False,
            env={
                **os.environ,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_OPTIONAL_LOCKS": "0",
            },
        )
        if result.returncode:
            raise ValueError("Ticket workspace is not a readable Git checkout")
        return result.stdout.strip()

    def _command(self, command: str) -> list[str]:
        args = [
            self.bwrap,
            "--die-with-parent",
            "--new-session",
            "--cap-drop",
            "ALL",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-net",
            "--unshare-ipc",
            "--unshare-uts",
            "--clearenv",
            "--setenv",
            "PATH",
            "/runtime/bin:/usr/local/bin:/usr/bin:/bin" if self.runtime else "/usr/local/bin:/usr/bin:/bin",
            "--setenv",
            "HOME",
            "/workspace",
            "--setenv",
            "TMPDIR",
            "/tmp",
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--setenv",
            "GIT_CONFIG_NOSYSTEM",
            "1",
            "--setenv",
            "GIT_CONFIG_GLOBAL",
            "/dev/null",
            "--setenv",
            "GIT_OPTIONAL_LOCKS",
            "0",
        ]
        for path in ("/usr", "/bin", "/lib", "/lib64"):
            if Path(path).exists():
                args.extend(("--ro-bind", path, path))
        if self.runtime:
            args.extend(("--ro-bind", str(self.runtime), "/runtime"))
        args.extend(
            (
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--tmpfs",
                "/tmp",
                "--bind",
                str(self.workspace),
                "/workspace",
                "--chdir",
                "/workspace",
                "--",
                "/bin/sh",
                "-lc",
                command,
            )
        )
        return args

    @staticmethod
    def _set_resource_limits() -> None:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
        resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
        hard = resource.getrlimit(resource.RLIMIT_NPROC)[1]
        process_limit = 128 if hard == resource.RLIM_INFINITY else min(128, hard)
        resource.setrlimit(resource.RLIMIT_NPROC, (process_limit, process_limit))

    def run(self, command: str, purpose: str = "other") -> CommandResult:
        if not command.strip() or len(command) > 16_384:
            raise ValueError("Command must be nonempty and at most 16384 characters")
        if purpose not in self.PURPOSES:
            raise ValueError(f"Unsupported command purpose {purpose!r}")

        started = time.monotonic()
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                self._command(command),
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
                preexec_fn=self._set_resource_limits,
            )
            timed_out = False
            try:
                process.wait(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read(self.output_limit_bytes + 1)
            stderr = stderr_file.read(self.output_limit_bytes + 1)

        truncated = len(stdout) > self.output_limit_bytes or len(stderr) > self.output_limit_bytes
        if len(stdout) > self.output_limit_bytes:
            stdout = stdout[: self.output_limit_bytes] + b"\n[output truncated]"
        if len(stderr) > self.output_limit_bytes:
            stderr = stderr[: self.output_limit_bytes] + b"\n[output truncated]"
        return CommandResult(
            command=command,
            purpose=purpose,
            exit_code=124 if timed_out else int(process.returncode or 0),
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            duration_seconds=round(time.monotonic() - started, 3),
            timed_out=timed_out,
            truncated=truncated,
        )

    def collect_diff(self) -> CommandResult:
        if not (self.workspace / ".git").exists():
            return CommandResult(
                command="git diff",
                purpose="git",
                exit_code=0,
                stdout="",
                stderr="",
                duration_seconds=0.0,
            )
        if self.source_commit is None:
            try:
                self._host_git("rev-parse", "HEAD")
            except ValueError:
                return CommandResult(
                    command="git diff",
                    purpose="git",
                    exit_code=0,
                    stdout="",
                    stderr="",
                    duration_seconds=0.0,
                )
        if self.source_commit is None:
            return self.run("git add -N --all && git diff --binary --no-ext-diff --", "git")
        return self.run(
            f"git add -N --all && git diff --binary --no-ext-diff {self.source_commit} --",
            "git",
        )

    def as_tool(self, *, name: str = "workspace_command") -> FunctionTool:
        """Return the sole built-in workspace tool; its shell stays sandboxed."""

        return FunctionTool(
            name=name,
            description=(
                "Inspect, edit, test, lint, or run local Git commands inside the "
                "task workspace. Network access and paths outside the workspace are unavailable."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run in the task workspace"},
                    "purpose": {
                        "type": "string",
                        "enum": sorted(self.PURPOSES),
                        "description": "Classify the command for the execution report",
                    },
                },
                "required": ["command", "purpose"],
                "additionalProperties": False,
            },
            handler=lambda command, purpose: self.run(command, purpose).as_dict(),
            capability="workspace.execute",
        )
