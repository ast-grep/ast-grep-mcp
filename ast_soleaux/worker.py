from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

from ast_soleaux.contracts import JSON_OBJECT_ADAPTER, JsonObject

PROCESS_TERMINATION_GRACE_SECONDS = 2.0


class _KillProcessGroup(Protocol):
    def __call__(self, process_id: int, process_signal: int, /) -> None: ...


type ManagedProcess = subprocess.Popen[str] | subprocess.Popen[bytes]


def popen_process_group_options() -> dict[str, int | bool]:
    if os.name == "posix":
        return {"start_new_session": True}
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {}


def _signal_process_group(process_id: int, process_signal: int) -> None:
    member = "killpg"
    kill_group = cast(_KillProcessGroup, getattr(os, member))
    kill_group(process_id, process_signal)


def process_group_id(process: ManagedProcess) -> int | None:
    if sys.platform == "win32":
        return None
    try:
        return os.getpgid(process.pid)
    except OSError:
        return None


def terminate_and_reap(
    process: ManagedProcess,
    group_id: int | None = None,
    *,
    grace_seconds: float = PROCESS_TERMINATION_GRACE_SECONDS,
) -> None:
    signaled_group = False
    if os.name == "posix":
        group_id = group_id if group_id is not None else process_group_id(process)
        try:
            if group_id is None:
                raise ProcessLookupError
            _signal_process_group(group_id, signal.SIGTERM)
            signaled_group = True
        except OSError:
            pass
    elif os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            signaled_group = True
        except OSError:
            pass
    if not signaled_group and process.poll() is None:
        try:
            process.terminate()
        except OSError:
            pass

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        leader_running = process.poll() is None
        group_running = False
        if os.name == "posix" and signaled_group and group_id is not None:
            try:
                _signal_process_group(group_id, 0)
                group_running = True
            except OSError:
                pass
        if not leader_running and not group_running:
            process.wait()
            return
        time.sleep(0.01)

    if sys.platform != "win32" and signaled_group and group_id is not None:
        try:
            _signal_process_group(group_id, signal.SIGKILL)
        except OSError:
            pass
    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Command process could not be reaped after termination") from error


def _response_queue() -> queue.Queue[str | BaseException | None]:
    return queue.Queue()


@dataclass
class JsonLineWorker:
    command: Sequence[str]
    cwd: str
    environment: Mapping[str, str]
    _process: subprocess.Popen[str] | None = field(default=None, init=False)
    _responses: queue.Queue[str | BaseException | None] = field(default_factory=_response_queue, init=False)
    _reader: threading.Thread | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        group_options = popen_process_group_options()
        creationflags_value = group_options.get("creationflags", 0)
        creationflags = 0 if isinstance(creationflags_value, bool) else creationflags_value
        self._process = subprocess.Popen(
            [*self.command, "--serve"],
            cwd=self.cwd,
            env=dict(self.environment),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            shell=False,
            bufsize=1,
            start_new_session=group_options.get("start_new_session") is True,
            creationflags=creationflags,
        )
        self._responses = _response_queue()
        self._reader = threading.Thread(target=self._read_responses, name="ast-soleaux-worker-reader", daemon=True)
        self._reader.start()

    def _read_responses(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            self._responses.put(RuntimeError("worker stdout is unavailable"))
            return
        try:
            for line in process.stdout:
                self._responses.put(line)
        except BaseException as error:
            self._responses.put(error)
        finally:
            self._responses.put(None)

    def request(self, payload: JsonObject, *, timeout: float) -> JsonObject:
        with self._lock:
            self.start()
            process = self._process
            if process is None or process.stdin is None:
                raise RuntimeError("worker stdin is unavailable")
            if process.poll() is not None:
                raise RuntimeError(f"worker exited with code {process.returncode}")
            process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            process.stdin.flush()
            try:
                response = self._responses.get(timeout=timeout)
            except queue.Empty as error:
                self.close()
                raise RuntimeError(f"worker timed out after {timeout:g} seconds") from error
            if response is None:
                self.close()
                raise RuntimeError("worker closed stdout before returning a response")
            if isinstance(response, BaseException):
                self.close()
                raise RuntimeError(f"worker response reader failed: {response}") from response
            result = JSON_OBJECT_ADAPTER.validate_json(response, strict=True)
            worker_error = result.get("error")
            if isinstance(worker_error, str):
                raise RuntimeError(worker_error)
            return result

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            try:
                process.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                terminate_and_reap(process)
        reader = self._reader
        self._reader = None
        if reader is not None:
            reader.join(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
