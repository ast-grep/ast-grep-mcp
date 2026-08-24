#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO, NamedTuple, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

WORKER_VERSION = "0.1.0"
RUNTIME_VERSION = "@oxc-node/core@0.1.0"
DOCKER_IMAGE = "node:24.18.0-bookworm-slim"
MAX_STDIN_BYTES = 1024 * 1024
MAX_ARGUMENTS = 64
MAX_ARGUMENT_BYTES = 4096
MAX_STDOUT_BYTES = 4 * 1024 * 1024
MAX_STDERR_BYTES = 64 * 1024
MAX_WRITTEN_FILES = 256
MAX_WRITTEN_FILE_BYTES = 16 * 1024 * 1024
MAX_WRITTEN_TOTAL_BYTES = 64 * 1024 * 1024
MAX_TRACKED_PROJECT_FILES = 10_000


class ExecutionProfile(StrEnum):
    ISOLATED = "isolated"
    WORKSPACE_WRITE = "workspace-write"
    NETWORKED = "networked"


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_root: str
    entry: str
    args: list[str] = Field(default_factory=list, max_length=MAX_ARGUMENTS)
    stdin: str | None = None
    timeout_seconds: float = Field(default=30, gt=0, le=120)
    profile: ExecutionProfile = ExecutionProfile.ISOLATED

    @field_validator("entry")
    @classmethod
    def validate_entry(cls, value: str) -> str:
        path = Path(value)
        if not value or "\0" in value or path.is_absolute() or ".." in path.parts:
            raise ValueError("entry must be a non-empty contained relative path")
        return value

    @field_validator("args")
    @classmethod
    def validate_args(cls, values: list[str]) -> list[str]:
        if any("\0" in value or len(value.encode("utf-8")) > MAX_ARGUMENT_BYTES for value in values):
            raise ValueError("arguments must not contain NULs or exceed the per-argument byte limit")
        return values

    @field_validator("stdin")
    @classmethod
    def validate_stdin(cls, value: str | None) -> str | None:
        if value is not None and len(value.encode("utf-8")) > MAX_STDIN_BYTES:
            raise ValueError("stdin exceeds the byte limit")
        return value


class ExecutionAudit(TypedDict):
    worker_version: str
    runtime_version: str
    sandbox: str
    image: str
    profile: str
    network: bool
    project_writes: bool
    project_read_only: bool
    sanitized_environment: bool
    no_shell: bool
    container_name: str
    limits: dict[str, int | float]


class ExecutionOutput(TypedDict):
    exit_code: int
    signal: str | None
    stdout: str
    stderr: str
    timed_out: bool
    stdout_truncated: bool
    stderr_truncated: bool
    duration_ms: int
    written_files: list[str]
    audit: ExecutionAudit


def contained(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


@dataclass
class BoundedCapture:
    limit: int
    buffer: bytearray = field(default_factory=bytearray)
    truncated: bool = False

    def drain(self, stream: BinaryIO) -> None:
        while chunk := stream.read(64 * 1024):
            self.buffer.extend(chunk)
            overflow = len(self.buffer) - self.limit
            if overflow > 0:
                del self.buffer[:overflow]
                self.truncated = True

    def text(self) -> str:
        return self.buffer.decode("utf-8", errors="replace")


class ProcessResult(NamedTuple):
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    stdout_truncated: bool
    stderr_truncated: bool


def write_stdin(stream: BinaryIO, payload: bytes) -> None:
    try:
        stream.write(payload)
        stream.flush()
    except BrokenPipeError:
        pass
    finally:
        stream.close()


def run_container(command: list[str], stdin: str | None, timeout: float, docker: str, container_name: str) -> ProcessResult:
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"PATH": os.environ.get("PATH", "")},
        shell=False,
        start_new_session=os.name != "nt",
    )
    if process.stdin is None or process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError("failed to create sandbox process pipes")
    stdout = BoundedCapture(MAX_STDOUT_BYTES)
    stderr = BoundedCapture(MAX_STDERR_BYTES)
    stdout_thread = threading.Thread(target=stdout.drain, args=(process.stdout,), daemon=True)
    stderr_thread = threading.Thread(target=stderr.drain, args=(process.stderr,), daemon=True)
    stdin_thread = threading.Thread(target=write_stdin, args=(process.stdin, (stdin or "").encode("utf-8")), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    stdin_thread.start()
    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        subprocess.run(
            [docker, "rm", "--force", container_name],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
            env={"PATH": os.environ.get("PATH", "")},
            shell=False,
        )
        if process.poll() is None:
            if sys.platform != "win32":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        process.wait(timeout=10)
    stdin_thread.join(timeout=1)
    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)
    return ProcessResult(
        124 if timed_out else process.returncode,
        stdout.text(),
        stderr.text(),
        timed_out,
        stdout.truncated,
        stderr.truncated,
    )


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(64 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_files(root: Path) -> dict[str, tuple[int, str]]:
    snapshot: dict[str, tuple[int, str]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if len(snapshot) >= MAX_TRACKED_PROJECT_FILES:
            raise ValueError(f"tracked files exceed the {MAX_TRACKED_PROJECT_FILES}-file limit")
        size = path.stat().st_size
        snapshot[relative] = (size, file_digest(path))
    return snapshot


def written_files(root: Path, before: dict[str, tuple[int, str]] | None) -> list[str]:
    after = snapshot_files(root)
    changed = sorted(path for path, identity in after.items() if before is None or before.get(path) != identity)
    if len(changed) > MAX_WRITTEN_FILES:
        raise ValueError(f"written files exceed the {MAX_WRITTEN_FILES}-file limit")
    total = 0
    for relative in changed:
        size = after[relative][0]
        if size > MAX_WRITTEN_FILE_BYTES:
            raise ValueError(f"written file exceeds the byte limit: {relative}")
        total += size
    if total > MAX_WRITTEN_TOTAL_BYTES:
        raise ValueError("written files exceed the aggregate byte limit")
    return changed


def docker_command(
    *,
    docker: str,
    request: Request,
    project: Path,
    entry: Path,
    node_modules: Path,
    overlay: Path,
    container_name: str,
) -> list[str]:
    project_readonly = request.profile is not ExecutionProfile.WORKSPACE_WRITE
    workdir = "/workspace" if request.profile is ExecutionProfile.WORKSPACE_WRITE else "/work"
    command = [
        docker,
        "run",
        "--rm",
        "--init",
        "--interactive",
        "--name",
        container_name,
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "64",
        "--memory",
        "512m",
        "--cpus",
        "1",
        "--ulimit",
        "nofile=256:256",
        "--ulimit",
        f"fsize={MAX_WRITTEN_FILE_BYTES}:{MAX_WRITTEN_FILE_BYTES}",
        "--mount",
        f"type=bind,src={project},dst=/workspace{',readonly' if project_readonly else ''}",
        "--mount",
        f"type=bind,src={node_modules},dst=/opt/oxc/node_modules,readonly",
        "--mount",
        f"type=bind,src={overlay},dst=/work",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--workdir",
        workdir,
    ]
    if sys.platform != "win32":
        command.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
    if request.profile is not ExecutionProfile.NETWORKED:
        command.extend(["--network", "none"])
    command.extend(
        [
            DOCKER_IMAGE,
            "node",
            "--import",
            "/opt/oxc/node_modules/@oxc-node/core/register.mjs",
            f"/workspace/{entry.relative_to(project).as_posix()}",
            *request.args,
        ]
    )
    return command


def run_request(request: Request) -> ExecutionOutput:
    project = Path(request.project_root).resolve(strict=True)
    if not project.is_dir():
        raise ValueError("project_root is not a directory")
    entry = (project / request.entry).resolve(strict=True)
    if not contained(entry, project) or not entry.is_file():
        raise ValueError("entry resolves outside project_root or is not a file")
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("docker is required for TypeScript sandbox execution")
    repository = Path(__file__).resolve().parents[1]
    node_modules = (repository / "execution-sidecar" / "node_modules").resolve(strict=True)
    before = snapshot_files(project) if request.profile is ExecutionProfile.WORKSPACE_WRITE else None
    container_name = f"ast-soleaux-{secrets.token_hex(8)}"
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="ast-soleaux-execution-") as temporary:
        overlay = Path(temporary).resolve()
        command = docker_command(
            docker=docker,
            request=request,
            project=project,
            entry=entry,
            node_modules=node_modules,
            overlay=overlay,
            container_name=container_name,
        )
        completed = run_container(command, request.stdin, request.timeout_seconds, docker, container_name)
        output_root = project if request.profile is ExecutionProfile.WORKSPACE_WRITE else overlay
        changed = written_files(output_root, before)
    signal_name = None
    if completed.returncode > 128:
        try:
            signal_name = signal.Signals(completed.returncode - 128).name
        except ValueError:
            signal_name = f"SIGNAL_{completed.returncode - 128}"
    return {
        "exit_code": completed.returncode,
        "signal": signal_name,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "timed_out": completed.timed_out,
        "stdout_truncated": completed.stdout_truncated,
        "stderr_truncated": completed.stderr_truncated,
        "duration_ms": int((time.monotonic() - started) * 1000),
        "written_files": changed,
        "audit": {
            "worker_version": WORKER_VERSION,
            "runtime_version": RUNTIME_VERSION,
            "sandbox": "docker",
            "image": DOCKER_IMAGE,
            "profile": request.profile.value,
            "network": request.profile is ExecutionProfile.NETWORKED,
            "project_writes": request.profile is ExecutionProfile.WORKSPACE_WRITE,
            "project_read_only": request.profile is not ExecutionProfile.WORKSPACE_WRITE,
            "sanitized_environment": True,
            "no_shell": True,
            "container_name": container_name,
            "limits": {
                "timeout_seconds": request.timeout_seconds,
                "pids": 64,
                "memory_bytes": 512 * 1024 * 1024,
                "cpus": 1,
                "stdout_bytes": MAX_STDOUT_BYTES,
                "stderr_bytes": MAX_STDERR_BYTES,
                "written_files": MAX_WRITTEN_FILES,
                "written_file_bytes": MAX_WRITTEN_FILE_BYTES,
                "written_total_bytes": MAX_WRITTEN_TOTAL_BYTES,
            },
        },
    }


def main() -> int:
    if sys.argv[1:] == ["--version-json"]:
        print(json.dumps({"worker": WORKER_VERSION, "runtime": RUNTIME_VERSION, "sandbox": "docker"}))
        return 0
    if sys.argv[1:]:
        raise ValueError(f"unknown arguments: {sys.argv[1:]}")
    request = Request.model_validate_json(sys.stdin.buffer.read(), strict=True)
    print(json.dumps(run_request(request), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
