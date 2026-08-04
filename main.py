from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import platform as platform_module
import queue
import re
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, BinaryIO, Final, Literal, TypedDict, cast

import yaml as yaml_parser
from mcp.server import MCPServer
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field

from config_snapshot import (
    MAX_CONFIG_FILE_BYTES,
    MAX_CONFIG_RESOURCE_BYTES,
    MAX_CONFIG_RESOURCE_FILES,
    MAX_NATIVE_LIBRARY_BYTES,
    MAX_YAML_DEPTH,
    MAX_YAML_DOCUMENTS,
    MAX_YAML_NODES,
    ConfigSnapshot,
    create_config_snapshot,
    is_within,
    load_strict_yaml_documents,
    private_runtime_root,
    validate_rule_utility_cycles,
)

DEFAULT_MAX_RESULTS: Final = 50
HARD_MAX_RESULTS: Final = 500
DEFAULT_COMMAND_TIMEOUT_SECONDS: Final = 30.0
FALLBACK_SERVER_VERSION: Final = "0+unknown"
NEUTRAL_AST_GREP_CONFIG: Final = "ruleDirs: []\n"
SUPPORTED_AST_GREP_VERSION: Final = "0.45.0"
MAX_INLINE_RULE_BYTES: Final = 64 * 1024
MAX_SNIPPET_INPUT_BYTES: Final = 1024 * 1024
MAX_OUTLINE_PATHS: Final = 64
MAX_NDJSON_RECORD_BYTES: Final = 1024 * 1024
MAX_STRUCTURED_OUTPUT_BYTES: Final = 4 * 1024 * 1024
MAX_OUTLINE_RECORD_BYTES: Final = MAX_NDJSON_RECORD_BYTES
PROCESS_READ_CHUNK_BYTES: Final = 64 * 1024
OUTLINE_READ_CHUNK_BYTES: Final = PROCESS_READ_CHUNK_BYTES
MAX_SUBPROCESS_DIAGNOSTIC_BYTES: Final = 64 * 1024
MAX_TEST_REPORT_BYTES: Final = 64 * 1024
WINDOWS_CREATE_PROCESS_LIMIT: Final = 32_767
POSIX_ARG_HEADROOM_BYTES: Final = 2048
PROCESS_TERMINATION_GRACE_SECONDS: Final = 2.0

DumpFormat = Literal["pattern", "cst", "ast", "sexp"]
OutputFormat = Literal["text", "json"]
Strictness = Literal["cst", "smart", "ast", "relaxed", "signature", "template"]
OutlineItemsMode = Literal["auto", "structure", "exports", "imports", "all"]
CompletedTextProcess = subprocess.CompletedProcess[str]
ProcessRunner = Callable[..., CompletedTextProcess]
PopenFactory = Callable[..., subprocess.Popen[bytes]]
OutlineProcessResult = tuple[list[dict[str, Any]], bool]
OutlineProcessRunner = Callable[..., OutlineProcessResult]
NDJSONRecordParser = Callable[[bytes, int], tuple[dict[str, Any], int] | None]


class SearchResults(TypedDict):
    matches: list[dict[str, Any]]
    returned: int
    truncated: bool
    limit: int


class OutlineFile(TypedDict):
    file: str
    language: str
    items: list[dict[str, Any]]


class OutlineResults(TypedDict):
    files: list[OutlineFile]
    returned: int
    truncated: bool
    limit: int


class ProjectTestResults(TypedDict):
    passed: bool
    report: str
    report_truncated: bool


class ServerInfo(TypedDict):
    fork_version: str
    ast_grep_executable: str
    ast_grep_version: str
    config_path: str | None
    allowed_roots: list[str]
    command_timeout_seconds: float
    default_max_results: int
    max_results_cap: int
    forbid_regex_rules: bool
    configuration_digest: str
    configuration_provenance: dict[str, Any]
    capabilities: dict[str, bool]
    coordinate_conventions: dict[str, str]
    resource_limits: dict[str, int | float]


@dataclass(frozen=True)
class ResolvedExecutable:
    path: Path
    command_prefix: tuple[str, ...]


@dataclass(frozen=True)
class ServerRuntime:
    working_directory: Path
    executable: ResolvedExecutable
    ast_grep_version: str
    config_path: Path | None
    allowed_roots: tuple[Path, ...]
    command_timeout_seconds: float
    default_max_results: int
    max_results_cap: int
    forbid_regex_rules: bool
    config_snapshot: ConfigSnapshot | None = None

    def close(self) -> None:
        if self.config_snapshot is not None:
            self.config_snapshot.close()


READ_ONLY_ANNOTATIONS: Final = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)


def _server_version() -> str:
    try:
        return version("sg-mcp")
    except PackageNotFoundError:
        return FALLBACK_SERVER_VERSION


def _resolve_existing_path(raw_path: str, *, base: Path, kind: Literal["file", "directory"]) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{kind.capitalize()} does not exist: {raw_path}") from error
    if kind == "directory" and not resolved.is_dir():
        raise ValueError(f"Expected a directory: {raw_path}")
    if kind == "file" and not resolved.is_file():
        raise ValueError(f"Expected a file: {raw_path}")
    return resolved


def resolve_allowed_roots(raw_roots: Sequence[str], *, working_directory: Path) -> tuple[Path, ...]:
    roots = raw_roots or [str(working_directory)]
    resolved_roots: list[Path] = []
    for raw_root in roots:
        root = _resolve_existing_path(raw_root, base=working_directory, kind="directory")
        if root not in resolved_roots:
            resolved_roots.append(root)
    return tuple(resolved_roots)


def _require_allowed(path: Path, allowed_roots: Sequence[Path], *, label: str) -> None:
    if not any(is_within(path, root) for root in allowed_roots):
        allowed = ", ".join(str(root) for root in allowed_roots)
        raise ValueError(f"{label} resolves outside the allowed roots ({allowed}): {path}")


def _read_json_file(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError, UnicodeDecodeError, json.JSONDecodeError:
        return None
    if not isinstance(value, Mapping):
        return None
    return cast(Mapping[str, Any], value)


def _npm_package_bin(package_directory: Path) -> Path | None:
    manifest_path = package_directory / "package.json"
    manifest = _read_json_file(manifest_path)
    if manifest is None or manifest.get("name") != "@ast-grep/cli":
        return None

    bin_value = manifest.get("bin")
    if isinstance(bin_value, str):
        relative_bin = bin_value
    elif isinstance(bin_value, Mapping):
        bin_mapping = cast(Mapping[str, Any], bin_value)
        mapped_bin = bin_mapping.get("ast-grep")
        if not isinstance(mapped_bin, str):
            return None
        relative_bin = mapped_bin
    else:
        return None

    try:
        target = (package_directory / relative_bin).resolve(strict=True)
    except FileNotFoundError:
        return None
    if not target.is_file() or not is_within(target, package_directory.resolve()):
        return None
    return target


def _native_ast_grep_package_name() -> str | None:
    system = platform_module.system().lower()
    machine = platform_module.machine().lower()
    if system == "darwin":
        architecture = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x64", "amd64": "x64"}.get(machine)
        return f"@ast-grep/cli-darwin-{architecture}" if architecture is not None else None
    if system == "windows":
        architecture = {
            "arm64": "arm64",
            "aarch64": "arm64",
            "x86_64": "x64",
            "amd64": "x64",
            "i386": "ia32",
            "i686": "ia32",
            "x86": "ia32",
        }.get(machine)
        return f"@ast-grep/cli-win32-{architecture}-msvc" if architecture is not None else None
    if system == "linux":
        libc_name = platform_module.libc_ver()[0].lower()
        if libc_name and libc_name not in {"glibc", "gnu libc"}:
            return None
        architecture = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x64", "amd64": "x64"}.get(machine)
        return f"@ast-grep/cli-linux-{architecture}-gnu" if architecture is not None else None
    return None


def _native_npm_ast_grep_bin(package_directory: Path) -> Path | None:
    manifest = _read_json_file(package_directory / "package.json")
    package_name = _native_ast_grep_package_name()
    if manifest is None or package_name is None:
        return None
    optional_dependencies = manifest.get("optionalDependencies")
    package_version = manifest.get("version")
    if not isinstance(optional_dependencies, Mapping) or not isinstance(package_version, str):
        return None
    optional_dependency_versions = cast(Mapping[str, Any], optional_dependencies)
    if optional_dependency_versions.get(package_name) != package_version:
        return None

    scope, separator, leaf = package_name.partition("/")
    if scope != "@ast-grep" or separator != "/" or not leaf:
        return None
    native_package = package_directory.parent / leaf
    native_manifest = _read_json_file(native_package / "package.json")
    if native_manifest is None or native_manifest.get("name") != package_name or native_manifest.get("version") != package_version:
        return None
    executable_name = "ast-grep.exe" if os.name == "nt" else "ast-grep"
    try:
        executable = (native_package / executable_name).resolve(strict=True)
        resolved_package = native_package.resolve(strict=True)
    except FileNotFoundError:
        return None
    if not executable.is_file() or not is_within(executable, resolved_package):
        return None
    if os.name != "nt" and not os.access(executable, os.X_OK):
        return None
    return executable


def _find_npm_ast_grep_bin(shim_path: Path, resolved_path: Path) -> Path | None:
    package_candidates: list[Path] = []
    if shim_path.parent.name == ".bin":
        package_candidates.append(shim_path.parent.parent / "@ast-grep" / "cli")
    package_candidates.append(shim_path.parent / "node_modules" / "@ast-grep" / "cli")

    for parent in resolved_path.parents:
        if len(package_candidates) >= 12:
            break
        package_candidates.append(parent)

    seen: set[Path] = set()
    for candidate in package_candidates:
        normalized = candidate.absolute()
        if normalized in seen:
            continue
        seen.add(normalized)
        target = _npm_package_bin(candidate)
        if target is not None:
            return target
    return None


def _requires_node(executable: Path) -> bool:
    if executable.suffix.lower() in {".cjs", ".js", ".mjs"}:
        return True
    try:
        with executable.open("rb") as executable_file:
            first_line = executable_file.readline(256)
    except OSError:
        return False
    return first_line.startswith(b"#!") and b"node" in first_line.lower()


def resolve_ast_grep_executable(raw_executable: str, *, working_directory: Path) -> ResolvedExecutable:
    raw_path = Path(raw_executable).expanduser()
    if sys.platform == "win32":
        has_path_separator = os.sep in raw_executable or os.altsep in raw_executable
    else:
        has_path_separator = os.sep in raw_executable
    if raw_path.is_absolute() or has_path_separator:
        shim_path = raw_path if raw_path.is_absolute() else working_directory / raw_path
        if not shim_path.exists():
            raise ValueError(f"ast-grep executable does not exist: {raw_executable}")
    else:
        discovered = shutil.which(raw_executable)
        if discovered is None:
            raise ValueError(f"ast-grep executable was not found: {raw_executable}")
        shim_path = Path(discovered)

    try:
        resolved_path = shim_path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"ast-grep executable does not exist: {raw_executable}") from error
    if not resolved_path.is_file():
        raise ValueError(f"ast-grep executable is not a file: {raw_executable}")

    npm_bin = _find_npm_ast_grep_bin(shim_path.absolute(), resolved_path)
    if npm_bin is not None:
        if _requires_node(npm_bin):
            native_bin = _native_npm_ast_grep_bin(npm_bin.parent)
            if native_bin is not None:
                return ResolvedExecutable(path=native_bin, command_prefix=(str(native_bin),))
            node = shutil.which("node")
            if node is None:
                raise ValueError("The @ast-grep/cli launcher requires Node.js, but node was not found")
            node_path = Path(node).resolve(strict=True)
            return ResolvedExecutable(path=npm_bin, command_prefix=(str(node_path), str(npm_bin)))
        if os.name != "nt" and not os.access(npm_bin, os.X_OK):
            raise ValueError(f"ast-grep executable is not executable: {npm_bin}")
        return ResolvedExecutable(path=npm_bin, command_prefix=(str(npm_bin),))

    if resolved_path.suffix.lower() in {".bat", ".cmd"}:
        raise ValueError(
            "Batch-file ast-grep launchers are not executed through a shell; install @ast-grep/cli or pass its resolved executable"
        )
    if os.name != "nt" and not os.access(resolved_path, os.X_OK):
        raise ValueError(f"ast-grep executable is not executable: {resolved_path}")
    return ResolvedExecutable(path=resolved_path, command_prefix=(str(resolved_path),))


def _bounded_error_text(value: str | None, *, limit: int = 4000) -> str:
    text = (value or "").strip()
    if not text:
        return "(no error output)"
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _utf16_code_units(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def _is_invalid_process_string(value: object) -> bool:
    return not isinstance(value, str) or "\0" in value


def _detected_posix_arg_max() -> int:
    if os.name != "posix":
        return 0
    try:
        os_api = cast(Any, os)
        sysconf = cast(Callable[[str], int], os_api.sysconf)
        return int(sysconf("SC_ARG_MAX"))
    except (AttributeError, OSError, ValueError) as error:
        raise ValueError("Could not determine the POSIX ARG_MAX process-launch budget") from error


def validate_process_budget(
    command: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
    platform_name: str | None = None,
    arg_max: int | None = None,
) -> None:
    if not command:
        raise ValueError("Command must contain an executable")
    if any(_is_invalid_process_string(argument) for argument in command):
        raise ValueError("Command arguments must be strings without NUL characters")

    executable_suffix = Path(command[0]).suffix.lower()
    if executable_suffix in {".bat", ".cmd"}:
        raise ValueError(
            "Batch-file commands are not launched directly; pass the resolved native executable or the JavaScript entry point through node"
        )

    launch_environment = os.environ if environment is None else environment
    if any(_is_invalid_process_string(key) or _is_invalid_process_string(value) or "=" in key for key, value in launch_environment.items()):
        raise ValueError("Subprocess environment names and values must be valid NUL-free strings")

    effective_platform = os.name if platform_name is None else platform_name
    if effective_platform == "nt":
        quoted_command = subprocess.list2cmdline(list(command))
        command_characters = _utf16_code_units(quoted_command) + 1
        if command_characters > WINDOWS_CREATE_PROCESS_LIMIT:
            raise ValueError(
                "Command line is too large for Windows CreateProcessW: "
                f"{command_characters} UTF-16 characters including NUL exceeds "
                f"the {WINDOWS_CREATE_PROCESS_LIMIT}-character limit; shorten paths, globs, or inline rules"
            )
        environment_characters = 1 + sum(_utf16_code_units(f"{key}={value}") + 1 for key, value in launch_environment.items())
        if environment_characters > WINDOWS_CREATE_PROCESS_LIMIT:
            raise ValueError(
                "Environment block is too large for Windows process creation: "
                f"{environment_characters} UTF-16 characters exceeds "
                f"the {WINDOWS_CREATE_PROCESS_LIMIT}-character limit; remove unnecessary environment variables"
            )
        return

    if effective_platform != "posix":
        raise ValueError(f"Unsupported subprocess platform: {effective_platform}")
    detected_arg_max = arg_max
    if detected_arg_max is None:
        detected_arg_max = _detected_posix_arg_max()
    if detected_arg_max <= POSIX_ARG_HEADROOM_BYTES:
        raise ValueError(f"Invalid POSIX ARG_MAX process-launch budget: {detected_arg_max}")

    argv_bytes = sum(len(os.fsencode(argument)) + 1 for argument in command)
    environment_bytes = sum(len(os.fsencode(key)) + len(os.fsencode(value)) + 2 for key, value in launch_environment.items())
    pointer_bytes = (len(command) + len(launch_environment) + 2) * struct.calcsize("P")
    process_bytes = argv_bytes + environment_bytes + pointer_bytes
    usable_bytes = detected_arg_max - POSIX_ARG_HEADROOM_BYTES
    if process_bytes > usable_bytes:
        raise ValueError(
            "Command and environment are too large for POSIX ARG_MAX: "
            f"estimated {process_bytes} bytes exceeds the {usable_bytes}-byte budget after "
            f"{POSIX_ARG_HEADROOM_BYTES} bytes of headroom; shorten paths, globs, inline rules, or the environment"
        )


def _is_benign_scan_diagnostic(line: str) -> bool:
    if line.startswith("Help:"):
        return True
    return line.startswith("Error: ") and line.endswith(" error(s) found in code.")


def _residual_stderr(stderr: str) -> str:
    lines = (raw.strip() for raw in stderr.splitlines())
    return "\n".join(line for line in lines if line and not _is_benign_scan_diagnostic(line))


@dataclass(frozen=True)
class StreamedTextProcess:
    completed: CompletedTextProcess
    stdout_truncated: bool
    stderr_truncated: bool


def _encode_bounded_input(input_text: str | None) -> bytes | None:
    if input_text is None:
        return None
    try:
        payload = input_text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("Subprocess input must be valid UTF-8") from error
    if len(payload) > MAX_SNIPPET_INPUT_BYTES:
        raise ValueError(f"Subprocess input exceeds the {MAX_SNIPPET_INPUT_BYTES // (1024 * 1024)} MiB limit")
    return payload


def _write_stdin(
    pipe: BinaryIO,
    payload: bytes,
    failures: list[BaseException],
    stop_writing: threading.Event,
) -> None:
    try:
        view = memoryview(payload)
        while view and not stop_writing.is_set():
            written = pipe.write(view[:PROCESS_READ_CHUNK_BYTES])
            if written <= 0:
                raise OSError("subprocess stdin made no write progress")
            view = view[written:]
        pipe.flush()
    except BrokenPipeError, ConnectionResetError:
        pass
    except (OSError, ValueError) as error:
        if not stop_writing.is_set():
            failures.append(error)
    finally:
        try:
            pipe.close()
        except OSError:
            pass


def _drain_bounded_pipe(
    pipe: BinaryIO,
    captured: bytearray,
    byte_limit: int,
    overflowed: list[bool],
    failures: list[BaseException],
    stop_reading: threading.Event,
    overflow: threading.Event | None = None,
) -> None:
    try:
        while not stop_reading.is_set():
            chunk = pipe.read(PROCESS_READ_CHUNK_BYTES)
            if not chunk:
                return
            available = byte_limit - len(captured)
            if available > 0:
                captured.extend(chunk[:available])
            if len(chunk) > max(available, 0):
                overflowed[0] = True
                if overflow is not None:
                    overflow.set()
    except (OSError, ValueError) as error:
        if not stop_reading.is_set():
            failures.append(error)


def _decode_utf8_prefix(payload: bytes) -> str:
    """Decode bytes, backing up to the last valid UTF-8 boundary on truncation."""
    while payload:
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as error:
            if not error.start:
                return ""
            payload = payload[: error.start]
    return ""


def _decode_utf8_output(payload: bytes, *, truncated: bool, label: str) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        if truncated and error.reason == "unexpected end of data" and error.end == len(payload):
            return _decode_utf8_prefix(payload[: error.start])
        raise RuntimeError(f"Command emitted invalid UTF-8 on {label}") from error


def run_text_process(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    input_text: str | None = None,
    working_directory: Path | None = None,
    stdout_limit: int = MAX_STRUCTURED_OUTPUT_BYTES,
    stderr_limit: int = MAX_SUBPROCESS_DIAGNOSTIC_BYTES,
    truncate_stdout: bool = False,
    truncate_stderr: bool = True,
    popen_factory: PopenFactory = subprocess.Popen,
) -> StreamedTextProcess:
    if stdout_limit < 1 or stderr_limit < 1:
        raise ValueError("Subprocess output limits must be positive")
    input_bytes = _encode_bounded_input(input_text)
    launch_environment = dict(os.environ)
    validate_process_budget(command, environment=launch_environment)
    try:
        process = popen_factory(
            list(command),
            stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(working_directory) if working_directory is not None else None,
            env=launch_environment,
            shell=False,
            bufsize=0,
            **_popen_process_group_options(),
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"Command executable was not found: {command[0]}") from error
    except OSError as error:
        raise RuntimeError(f"Command could not be executed: {error}") from error

    process_group = _process_group_id(process)
    if process.stdout is None or process.stderr is None:
        _terminate_and_reap(process, process_group)
        raise RuntimeError("Command pipes were not created")

    stop_io = threading.Event()
    stdout = bytearray()
    stderr = bytearray()
    stdout_overflow = [False]
    stderr_overflow = [False]
    overflow = threading.Event()
    io_failures: list[BaseException] = []
    threads = [
        threading.Thread(
            target=_drain_bounded_pipe,
            args=(process.stdout, stdout, stdout_limit, stdout_overflow, io_failures, stop_io, overflow),
            name="ast-grep-stdout",
            daemon=True,
        ),
        threading.Thread(
            target=_drain_bounded_pipe,
            args=(process.stderr, stderr, stderr_limit, stderr_overflow, io_failures, stop_io, overflow),
            name="ast-grep-stderr",
            daemon=True,
        ),
    ]
    if input_bytes is not None:
        if process.stdin is None:
            _terminate_and_reap(process, process_group)
            raise RuntimeError("Command stdin pipe was not created")
        threads.append(
            threading.Thread(
                target=_write_stdin,
                args=(process.stdin, input_bytes, io_failures, stop_io),
                name="ast-grep-stdin",
                daemon=True,
            )
        )
    for thread in threads:
        thread.start()

    timeout_error: subprocess.TimeoutExpired | None = None
    overflow_error: str | None = None
    deadline = time.monotonic() + timeout_seconds
    while process.poll() is None:
        if overflow.is_set():
            if stdout_overflow[0] and not truncate_stdout:
                overflow_error = f"Command structured output exceeds the {stdout_limit}-byte limit"
            elif stderr_overflow[0] and not truncate_stderr:
                overflow_error = f"Command diagnostic output exceeds the {stderr_limit}-byte limit"
            if overflow_error is not None:
                _terminate_and_reap(process, process_group)
                break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timeout_error = subprocess.TimeoutExpired(list(command), timeout_seconds)
            _terminate_and_reap(process, process_group)
            break
        try:
            process.wait(timeout=min(remaining, 0.05))
        except subprocess.TimeoutExpired:
            continue
    _terminate_and_reap(process, process_group)
    for thread in threads:
        thread.join(timeout=0.05)
    if any(thread.is_alive() for thread in threads):
        _terminate_and_reap(process, process_group)
    stop_io.set()
    for pipe in (process.stdout, process.stderr):
        try:
            pipe.close()
        except OSError:
            pass
    if process.stdin is not None:
        try:
            process.stdin.close()
        except OSError:
            pass
    for thread in threads:
        if thread.is_alive():
            thread.join(timeout=PROCESS_TERMINATION_GRACE_SECONDS)

    if any(thread.is_alive() for thread in threads):
        raise RuntimeError("Command pipe workers did not stop after process exit")
    if timeout_error is not None:
        raise RuntimeError(f"Command timed out after {timeout_seconds:g} seconds") from timeout_error
    if io_failures:
        raise RuntimeError(f"Could not transfer command data: {io_failures[0]}") from io_failures[0]
    if overflow_error is not None:
        raise RuntimeError(overflow_error)
    if stdout_overflow[0] and not truncate_stdout:
        raise RuntimeError(f"Command structured output exceeds the {stdout_limit}-byte limit")
    if stderr_overflow[0] and not truncate_stderr:
        raise RuntimeError(f"Command diagnostic output exceeds the {stderr_limit}-byte limit")

    stdout_text = _decode_utf8_output(bytes(stdout), truncated=stdout_overflow[0], label="stdout")
    stderr_text = _decode_utf8_output(bytes(stderr), truncated=stderr_overflow[0], label="stderr")
    completed = subprocess.CompletedProcess(list(command), process.returncode, stdout_text, stderr_text)
    return StreamedTextProcess(completed, stdout_overflow[0], stderr_overflow[0])


def run_process(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    input_text: str | None = None,
    working_directory: Path | None = None,
    allowed_exit_codes: frozenset[int] = frozenset({0}),
    runner: ProcessRunner = subprocess.run,
    stdout_limit: int = MAX_STRUCTURED_OUTPUT_BYTES,
    stderr_limit: int = MAX_SUBPROCESS_DIAGNOSTIC_BYTES,
    truncate_stdout: bool = False,
    truncate_stderr: bool = True,
) -> CompletedTextProcess:
    launch_environment = dict(os.environ)
    validate_process_budget(command, environment=launch_environment)
    try:
        if runner is subprocess.run:
            streamed = run_text_process(
                command,
                timeout_seconds=timeout_seconds,
                input_text=input_text,
                working_directory=working_directory,
                stdout_limit=stdout_limit,
                stderr_limit=stderr_limit,
                truncate_stdout=truncate_stdout,
                truncate_stderr=truncate_stderr,
            )
            result = streamed.completed
        else:
            result = runner(
                list(command),
                capture_output=True,
                input=input_text,
                text=True,
                timeout=timeout_seconds,
                cwd=str(working_directory) if working_directory is not None else None,
                check=False,
                shell=False,
                env=launch_environment,
            )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Command timed out after {timeout_seconds:g} seconds") from error
    except FileNotFoundError as error:
        raise RuntimeError(f"Command executable was not found: {command[0]}") from error
    except OSError as error:
        raise RuntimeError(f"Command could not be executed: {error}") from error

    if result.returncode not in allowed_exit_codes:
        detail = _bounded_error_text(result.stderr)
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {detail}")
    return result


def _read_ast_grep_version(
    executable: ResolvedExecutable,
    *,
    timeout_seconds: float,
    runner: ProcessRunner = subprocess.run,
) -> str:
    result = run_process(
        [*executable.command_prefix, "--version"],
        timeout_seconds=timeout_seconds,
        runner=runner,
    )
    expected = f"ast-grep {SUPPORTED_AST_GREP_VERSION}"
    if result.stdout.strip() != expected or result.stderr:
        diagnostic = result.stderr or result.stdout
        raise ValueError(f"Configured executable must report exactly {expected!r}: {_bounded_error_text(diagnostic)}")
    return SUPPORTED_AST_GREP_VERSION


def _validate_config_snapshot(
    executable: ResolvedExecutable,
    snapshot: ConfigSnapshot,
    *,
    timeout_seconds: float,
    runner: ProcessRunner,
) -> None:
    with TemporaryDirectory(prefix="validation-", dir=snapshot.runtime_root) as validation:
        validation_directory = Path(validation)
        scan = run_process(
            [
                *executable.command_prefix,
                "scan",
                "--config",
                str(snapshot.project_config_path),
                "--json=stream",
                "--threads",
                "1",
                "--max-results",
                "1",
                "--",
                str(validation_directory),
            ],
            timeout_seconds=timeout_seconds,
            working_directory=validation_directory,
            allowed_exit_codes=frozenset({0, 1}),
            runner=runner,
            stdout_limit=MAX_STRUCTURED_OUTPUT_BYTES,
            stderr_limit=MAX_SUBPROCESS_DIAGNOSTIC_BYTES,
            truncate_stdout=True,
            truncate_stderr=True,
        )
        if scan.returncode not in {0, 1}:
            raise RuntimeError("Configured rules failed startup validation")
        if snapshot.capabilities["configured_tests"]:
            run_process(
                [
                    *executable.command_prefix,
                    "test",
                    "--config",
                    str(snapshot.test_config_path),
                    "--color",
                    "never",
                ],
                timeout_seconds=timeout_seconds,
                working_directory=snapshot.bundle_root,
                allowed_exit_codes=frozenset({0, 4}),
                runner=runner,
                stdout_limit=MAX_TEST_REPORT_BYTES,
                stderr_limit=MAX_SUBPROCESS_DIAGNOSTIC_BYTES,
                truncate_stdout=True,
                truncate_stderr=True,
            )


def build_runtime(
    *,
    working_directory: Path,
    ast_grep_executable: str = "ast-grep",
    config_path: str | None = None,
    allowed_roots: Sequence[str] = (),
    command_timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    default_max_results: int = DEFAULT_MAX_RESULTS,
    max_results_cap: int = HARD_MAX_RESULTS,
    forbid_regex_rules: bool = False,
    trusted_native_libraries: Sequence[tuple[str, str]] = (),
    runner: ProcessRunner = subprocess.run,
) -> ServerRuntime:
    resolved_working_directory = working_directory.resolve(strict=True)
    if not resolved_working_directory.is_dir():
        raise ValueError(f"Working directory is not a directory: {working_directory}")
    if not math.isfinite(command_timeout_seconds) or command_timeout_seconds <= 0:
        raise ValueError("Command timeout must be finite and greater than zero")
    if not 1 <= max_results_cap <= HARD_MAX_RESULTS:
        raise ValueError(f"Result cap must be between 1 and {HARD_MAX_RESULTS}")
    if not 1 <= default_max_results <= max_results_cap:
        raise ValueError("Default result limit must be positive and no greater than the configured cap")

    resolved_roots = resolve_allowed_roots(allowed_roots, working_directory=resolved_working_directory)
    executable = resolve_ast_grep_executable(ast_grep_executable, working_directory=resolved_working_directory)
    ast_grep_version = _read_ast_grep_version(
        executable,
        timeout_seconds=command_timeout_seconds,
        runner=runner,
    )
    snapshot = create_config_snapshot(
        config_path=config_path,
        working_directory=resolved_working_directory,
        allowed_roots=resolved_roots,
        trusted_native_libraries=trusted_native_libraries,
    )
    try:
        _validate_config_snapshot(
            executable,
            snapshot,
            timeout_seconds=command_timeout_seconds,
            runner=runner,
        )
    except BaseException:
        snapshot.close_best_effort()
        raise
    atexit.register(snapshot.close_best_effort)

    return ServerRuntime(
        working_directory=resolved_working_directory,
        executable=executable,
        ast_grep_version=ast_grep_version,
        config_path=snapshot.source_path,
        allowed_roots=resolved_roots,
        command_timeout_seconds=command_timeout_seconds,
        default_max_results=default_max_results,
        max_results_cap=max_results_cap,
        forbid_regex_rules=forbid_regex_rules,
        config_snapshot=snapshot,
    )


def _contains_mapping_key(value: object, forbidden_key: str) -> bool:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return any(key == forbidden_key or _contains_mapping_key(item, forbidden_key) for key, item in mapping.items())
    if isinstance(value, list):
        items = cast(list[object], value)
        return any(_contains_mapping_key(item, forbidden_key) for item in items)
    return False


def validate_rule_yaml(rule_yaml: str, *, forbid_regex_rules: bool) -> None:
    try:
        encoded = rule_yaml.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("ast-grep rule YAML must be valid UTF-8") from error
    if len(encoded) > MAX_INLINE_RULE_BYTES:
        raise ValueError(f"ast-grep rule YAML exceeds the {MAX_INLINE_RULE_BYTES // 1024} KiB inline limit")
    documents = load_strict_yaml_documents(rule_yaml, label="ast-grep rule YAML")
    if not documents or all(document is None for document in documents):
        raise ValueError("ast-grep rule YAML must contain at least one rule")

    for document in documents:
        if not isinstance(document, Mapping):
            raise ValueError("Each ast-grep inline rule must be a YAML mapping")
        document_mapping = cast(Mapping[object, object], document)
        missing = [key for key in ("id", "language", "rule") if key not in document_mapping]
        if missing:
            raise ValueError(f"ast-grep rule is missing required fields: {', '.join(missing)}")
        if not isinstance(document_mapping["rule"], Mapping):
            raise ValueError("ast-grep rule field must be a mapping")
        if forbid_regex_rules and _contains_mapping_key(document_mapping, "regex"):
            raise ValueError("Regex ast-grep rules are disabled by server policy")
    validate_rule_utility_cycles(documents, label="ast-grep rule YAML")


def _runtime_mapping(value: object, *, field_name: str, record_number: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"ast-grep record {record_number} has invalid {field_name}")
    raw_mapping = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in raw_mapping):
        raise RuntimeError(f"ast-grep record {record_number} has invalid {field_name}")
    return cast(dict[str, Any], value)


def _runtime_list(value: object, *, field_name: str, record_number: int) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"ast-grep record {record_number} has invalid {field_name}")
    return cast(list[Any], value)


def _runtime_nonnegative_int(value: object, *, field_name: str, record_number: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError(f"ast-grep record {record_number} has invalid {field_name}")
    return value


def _validate_position(value: object, *, field_name: str, record_number: int) -> tuple[int, int]:
    position = _runtime_mapping(value, field_name=field_name, record_number=record_number)
    line = _runtime_nonnegative_int(position.get("line"), field_name=f"{field_name}.line", record_number=record_number)
    column = _runtime_nonnegative_int(
        position.get("column"),
        field_name=f"{field_name}.column",
        record_number=record_number,
    )
    return line, column


def _validate_offset_range(value: object, *, field_name: str, record_number: int) -> tuple[int, int]:
    offset_range = _runtime_mapping(value, field_name=field_name, record_number=record_number)
    start = _runtime_nonnegative_int(
        offset_range.get("start"),
        field_name=f"{field_name}.start",
        record_number=record_number,
    )
    end = _runtime_nonnegative_int(
        offset_range.get("end"),
        field_name=f"{field_name}.end",
        record_number=record_number,
    )
    if end < start:
        raise RuntimeError(f"ast-grep record {record_number} has a reversed {field_name}")
    return start, end


def _validate_source_range(
    value: object,
    *,
    field_name: str,
    record_number: int,
    text: str | None = None,
) -> tuple[int, int]:
    source_range = _runtime_mapping(value, field_name=field_name, record_number=record_number)
    start_offset, end_offset = _validate_offset_range(
        source_range.get("byteOffset"),
        field_name=f"{field_name}.byteOffset",
        record_number=record_number,
    )
    start_position = _validate_position(
        source_range.get("start"),
        field_name=f"{field_name}.start",
        record_number=record_number,
    )
    end_position = _validate_position(
        source_range.get("end"),
        field_name=f"{field_name}.end",
        record_number=record_number,
    )
    if end_position < start_position:
        raise RuntimeError(f"ast-grep record {record_number} has reversed half-open positions in {field_name}")
    if text is not None:
        if end_offset - start_offset != len(text.encode("utf-8")):
            raise RuntimeError(f"ast-grep record {record_number} has inconsistent UTF-8 byte offsets in {field_name}")
        line_delta = text.count("\n")
        expected_end_line = start_position[0] + line_delta
        expected_end_column = start_position[1] + len(text) if line_delta == 0 else len(text.rsplit("\n", 1)[1])
        if end_position != (expected_end_line, expected_end_column):
            raise RuntimeError(f"ast-grep record {record_number} has inconsistent line and column geometry in {field_name}")
    return start_offset, end_offset


def _validate_match_node(
    value: object,
    *,
    field_name: str,
    record_number: int,
) -> None:
    node = _runtime_mapping(value, field_name=field_name, record_number=record_number)
    text = node.get("text")
    if not isinstance(text, str):
        raise RuntimeError(f"ast-grep record {record_number} has invalid {field_name}.text")
    _validate_source_range(
        node.get("range"),
        field_name=f"{field_name}.range",
        record_number=record_number,
        text=text,
    )


def validate_match_document(document: dict[str, Any], *, record_number: int, require_rule_fields: bool = True) -> None:
    text = document.get("text")
    lines = document.get("lines")
    file_name = document.get("file")
    language = document.get("language")
    if not isinstance(text, str):
        raise RuntimeError(f"ast-grep match record {record_number} has no text")
    if not isinstance(lines, str):
        raise RuntimeError(f"ast-grep match record {record_number} has no lines")
    if not isinstance(file_name, str) or not file_name:
        raise RuntimeError(f"ast-grep match record {record_number} has no file")
    if not isinstance(language, str) or not language:
        raise RuntimeError(f"ast-grep match record {record_number} has no language")
    _validate_source_range(
        document.get("range"),
        field_name="range",
        record_number=record_number,
        text=text,
    )

    char_count = _runtime_mapping(document.get("charCount"), field_name="charCount", record_number=record_number)
    leading = _runtime_nonnegative_int(
        char_count.get("leading"),
        field_name="charCount.leading",
        record_number=record_number,
    )
    trailing = _runtime_nonnegative_int(
        char_count.get("trailing"),
        field_name="charCount.trailing",
        record_number=record_number,
    )
    if leading + trailing > len(lines):
        raise RuntimeError(f"ast-grep match record {record_number} has invalid character context counts")
    matched_end = len(lines) - trailing if trailing else len(lines)
    if lines[leading:matched_end] != text:
        raise RuntimeError(f"ast-grep match record {record_number} has inconsistent character context")

    meta_variables_value = document.get("metaVariables")
    if meta_variables_value is not None:
        meta_variables = _runtime_mapping(
            meta_variables_value,
            field_name="metaVariables",
            record_number=record_number,
        )
        single = _runtime_mapping(meta_variables.get("single"), field_name="metaVariables.single", record_number=record_number)
        multi = _runtime_mapping(meta_variables.get("multi"), field_name="metaVariables.multi", record_number=record_number)
        transformed = _runtime_mapping(
            meta_variables.get("transformed"),
            field_name="metaVariables.transformed",
            record_number=record_number,
        )
        for name, node in single.items():
            _validate_match_node(
                node,
                field_name=f"metaVariables.single.{name}",
                record_number=record_number,
            )
        for name, nodes_value in multi.items():
            nodes = _runtime_list(
                nodes_value,
                field_name=f"metaVariables.multi.{name}",
                record_number=record_number,
            )
            for index, node in enumerate(nodes):
                _validate_match_node(
                    node,
                    field_name=f"metaVariables.multi.{name}[{index}]",
                    record_number=record_number,
                )
        if any(not isinstance(value, str) for value in transformed.values()):
            raise RuntimeError(f"ast-grep match record {record_number} has invalid transformed metavariables")

    has_replacement = "replacement" in document
    has_replacement_offsets = "replacementOffsets" in document
    if has_replacement != has_replacement_offsets:
        raise RuntimeError(f"ast-grep match record {record_number} has incomplete replacement data")
    if has_replacement:
        if not isinstance(document["replacement"], str):
            raise RuntimeError(f"ast-grep match record {record_number} has invalid replacement")
        _validate_offset_range(
            document["replacementOffsets"],
            field_name="replacementOffsets",
            record_number=record_number,
        )

    if not require_rule_fields:
        return
    rule_id = document.get("ruleId")
    severity = document.get("severity")
    message = document.get("message")
    if not isinstance(rule_id, str) or not rule_id:
        raise RuntimeError(f"ast-grep match record {record_number} has no ruleId")
    if severity not in {"hint", "info", "warning", "error", "off"}:
        raise RuntimeError(f"ast-grep match record {record_number} has invalid severity")
    if not isinstance(message, str):
        raise RuntimeError(f"ast-grep match record {record_number} has invalid message")
    if document.get("note") is not None and not isinstance(document.get("note"), str):
        raise RuntimeError(f"ast-grep match record {record_number} has invalid note")
    labels = _runtime_list(document.get("labels", []), field_name="labels", record_number=record_number)
    for index, label_value in enumerate(labels):
        label = _runtime_mapping(label_value, field_name=f"labels[{index}]", record_number=record_number)
        label_text = label.get("text")
        if not isinstance(label_text, str):
            raise RuntimeError(f"ast-grep match record {record_number} has invalid labels[{index}].text")
        _validate_source_range(
            label.get("range"),
            field_name=f"labels[{index}].range",
            record_number=record_number,
            text=label_text,
        )
        if label.get("style") not in {"primary", "secondary"}:
            raise RuntimeError(f"ast-grep match record {record_number} has invalid labels[{index}].style")
        if label.get("message") is not None and not isinstance(label.get("message"), str):
            raise RuntimeError(f"ast-grep match record {record_number} has invalid labels[{index}].message")
    if "metadata" in document and not isinstance(document["metadata"], dict):
        raise RuntimeError(f"ast-grep match record {record_number} has invalid metadata")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _parse_json_object_record(raw_record: bytes, *, record_number: int, noun: str) -> dict[str, Any]:
    if len(raw_record) > MAX_NDJSON_RECORD_BYTES:
        raise RuntimeError(f"ast-grep {noun} record exceeds the {MAX_NDJSON_RECORD_BYTES // 1024} KiB limit")
    try:
        decoded = raw_record.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"ast-grep {noun} emitted invalid UTF-8 in record {record_number}") from error
    try:
        value = json.loads(decoded, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise RuntimeError(f"ast-grep {noun} emitted invalid JSON in record {record_number}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"ast-grep {noun} emitted a non-object JSON record {record_number}")
    raw_mapping = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in raw_mapping):
        raise RuntimeError(f"ast-grep {noun} emitted a non-object JSON record {record_number}")
    return cast(dict[str, Any], value)


def _parse_match_record(raw_record: bytes, record_number: int) -> tuple[dict[str, Any], int]:
    document = _parse_json_object_record(raw_record, record_number=record_number, noun="match")
    validate_match_document(document, record_number=record_number)
    return document, 1


def parse_stream_matches(stdout: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(stdout.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, RecursionError, ValueError) as error:
            raise RuntimeError(f"ast-grep emitted invalid JSON on line {line_number}") from error
        if not isinstance(value, dict):
            raise RuntimeError(f"ast-grep emitted a non-object JSON match on line {line_number}")
        matches.append(cast(dict[str, Any], value))
    return matches


@dataclass(frozen=True)
class _PipeReadFailure:
    error: BaseException


@dataclass(frozen=True)
class _PipeEof:
    pass


_PIPE_EOF: Final = _PipeEof()
PipeEvent = bytes | _PipeReadFailure | _PipeEof


def _put_pipe_event(
    events: queue.Queue[PipeEvent],
    event: PipeEvent,
    stop_reading: threading.Event,
) -> None:
    while not stop_reading.is_set():
        try:
            events.put(event, timeout=0.05)
        except queue.Full:
            continue
        return


def _queue_pipe_chunks(
    pipe: BinaryIO,
    events: queue.Queue[PipeEvent],
    stop_reading: threading.Event,
) -> None:
    try:
        while not stop_reading.is_set():
            chunk = pipe.read(OUTLINE_READ_CHUNK_BYTES)
            if not chunk:
                _put_pipe_event(events, _PIPE_EOF, stop_reading)
                return
            _put_pipe_event(events, chunk, stop_reading)
    except (OSError, ValueError) as error:
        if not stop_reading.is_set():
            _put_pipe_event(events, _PipeReadFailure(error), stop_reading)


def _popen_process_group_options() -> dict[str, Any]:
    if os.name == "posix":
        return {"start_new_session": True}
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {}


def _signal_process_group(process_id: int, process_signal: int) -> None:
    os_api = cast(Any, os)
    kill_group = cast(Callable[[int, int], None], os_api.killpg)
    kill_group(process_id, process_signal)


def _process_group_id(process: subprocess.Popen[Any]) -> int | None:
    """Read the child's group before it is reaped, while its pid still identifies the group.

    POSIX keeps a process group alive while any member remains, so the captured
    identifier stays addressable after the leader exits. The leader's pid does not:
    once reaped it is free for reuse, and signalling it could reach another process.
    """
    if sys.platform == "win32":
        return None
    try:
        return os.getpgid(process.pid)
    except OSError:
        return None


def _terminate_and_reap(process: subprocess.Popen[Any], group_id: int | None = None) -> None:
    signaled_group = False
    if os.name == "posix":
        group_id = group_id if group_id is not None else _process_group_id(process)
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

    deadline = time.monotonic() + PROCESS_TERMINATION_GRACE_SECONDS
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
        process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Command process could not be reaped after termination") from error


_OUTLINE_SYMBOL_TYPES: Final = frozenset(
    {
        "file",
        "module",
        "namespace",
        "package",
        "class",
        "method",
        "property",
        "field",
        "constructor",
        "enum",
        "interface",
        "function",
        "variable",
        "constant",
        "string",
        "number",
        "boolean",
        "array",
        "object",
        "key",
        "null",
        "enumMember",
        "struct",
        "event",
        "operator",
        "typeParameter",
    }
)


def _validate_outline_node(document: dict[str, Any], *, record_number: int, expected_role: str) -> None:
    if document.get("role") != expected_role:
        raise RuntimeError(f"ast-grep outline record {record_number} has invalid {expected_role} role")
    if document.get("symbolType") not in _OUTLINE_SYMBOL_TYPES:
        raise RuntimeError(f"ast-grep outline record {record_number} has invalid symbolType")
    if not isinstance(document.get("name"), str):
        raise RuntimeError(f"ast-grep outline record {record_number} has invalid name")
    if not isinstance(document.get("signature"), str):
        raise RuntimeError(f"ast-grep outline record {record_number} has invalid signature")
    if not isinstance(document.get("astKind"), str) or not document["astKind"]:
        raise RuntimeError(f"ast-grep outline record {record_number} has invalid astKind")
    _validate_source_range(document.get("range"), field_name="range", record_number=record_number)
    if expected_role == "item":
        if not isinstance(document.get("isImport"), bool) or not isinstance(document.get("isExported"), bool):
            raise RuntimeError(f"ast-grep outline record {record_number} has invalid item flags")
    elif not isinstance(document.get("isPublic"), bool):
        raise RuntimeError(f"ast-grep outline record {record_number} has invalid member visibility")


def _validate_and_count_outline_document(
    document: dict[str, Any],
    *,
    record_number: int,
    canonical: bool = False,
) -> int:
    raw_path = document.get("path")
    language = document.get("language")
    items = document.get("items")
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError(f"ast-grep outline record {record_number} has no path")
    if not isinstance(language, str) or (canonical and not language):
        raise RuntimeError(f"ast-grep outline record {record_number} has no language")
    if not isinstance(items, list):
        raise RuntimeError(f"ast-grep outline record {record_number} has no items list")

    count = 0
    nodes: list[tuple[dict[str, Any], str]] = []
    remaining = [(item, "item") for item in reversed(cast(list[object], items))]
    while remaining:
        node_value, expected_role = remaining.pop()
        if not isinstance(node_value, dict):
            raise RuntimeError(f"ast-grep outline record {record_number} contains a non-object node")
        node = cast(dict[str, Any], node_value)
        nodes.append((node, expected_role))
        count += 1
        if "members" not in node:
            continue
        members = node["members"]
        if not isinstance(members, list):
            raise RuntimeError(f"ast-grep outline record {record_number} contains a non-list members field")
        remaining.extend((member, "member") for member in reversed(cast(list[object], members)))
    if canonical:
        for node, expected_role in nodes:
            _validate_outline_node(node, record_number=record_number, expected_role=expected_role)
    return count


def _parse_outline_record(raw_record: bytes, *, record_number: int) -> tuple[dict[str, Any], int] | None:
    if len(raw_record) > MAX_OUTLINE_RECORD_BYTES:
        raise RuntimeError(f"ast-grep outline record exceeds the {MAX_OUTLINE_RECORD_BYTES // 1024} KiB limit (record {record_number})")
    stripped = raw_record.strip()
    if not stripped:
        return None
    try:
        value = json.loads(stripped.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise RuntimeError(f"ast-grep outline emitted invalid JSON in record {record_number}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"ast-grep outline emitted a non-object JSON record {record_number}")
    document = cast(dict[str, Any], value)
    return document, _validate_and_count_outline_document(document, record_number=record_number, canonical=True)


@dataclass(frozen=True)
class StreamedNDJSONProcess:
    records: list[dict[str, Any]]
    observed_extra: bool
    returncode: int
    stderr: str
    terminated_for_limit: bool


def run_ndjson_process(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    working_directory: Path,
    record_parser: NDJSONRecordParser,
    item_limit: int | None = None,
    input_text: str | None = None,
    popen_factory: PopenFactory = subprocess.Popen,
) -> StreamedNDJSONProcess:
    if item_limit is not None and item_limit < 1:
        raise ValueError("NDJSON item limit must be positive")
    input_bytes = _encode_bounded_input(input_text)
    launch_environment = dict(os.environ)
    validate_process_budget(command, environment=launch_environment)
    try:
        process = popen_factory(
            list(command),
            stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(working_directory),
            env=launch_environment,
            shell=False,
            bufsize=0,
            **_popen_process_group_options(),
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"Command executable was not found: {command[0]}") from error
    except OSError as error:
        raise RuntimeError(f"Command could not be executed: {error}") from error

    process_group = _process_group_id(process)
    if process.stdout is None or process.stderr is None:
        _terminate_and_reap(process, process_group)
        raise RuntimeError("Command pipes were not created")

    stop_io = threading.Event()
    stdout_events: queue.Queue[PipeEvent] = queue.Queue(maxsize=4)
    stderr = bytearray()
    stderr_overflow = [False]
    io_failures: list[BaseException] = []
    threads = [
        threading.Thread(
            target=_queue_pipe_chunks,
            args=(process.stdout, stdout_events, stop_io),
            name="ast-grep-ndjson-stdout",
            daemon=True,
        ),
        threading.Thread(
            target=_drain_bounded_pipe,
            args=(
                process.stderr,
                stderr,
                MAX_SUBPROCESS_DIAGNOSTIC_BYTES,
                stderr_overflow,
                io_failures,
                stop_io,
            ),
            name="ast-grep-ndjson-stderr",
            daemon=True,
        ),
    ]
    if input_bytes is not None:
        if process.stdin is None:
            _terminate_and_reap(process, process_group)
            raise RuntimeError("Command stdin pipe was not created")
        threads.append(
            threading.Thread(
                target=_write_stdin,
                args=(process.stdin, input_bytes, io_failures, stop_io),
                name="ast-grep-ndjson-stdin",
                daemon=True,
            )
        )
    for thread in threads:
        thread.start()

    records: list[dict[str, Any]] = []
    record_buffer = bytearray()
    record_number = 0
    aggregate_bytes = 0
    observed_items = 0
    observed_extra = False
    terminated_for_limit = False
    deadline = time.monotonic() + timeout_seconds

    def consume_record(raw_record: bytes) -> None:
        nonlocal observed_extra, observed_items, record_number
        record_number += 1
        if not raw_record:
            raise RuntimeError(f"Command emitted an empty NDJSON record {record_number}")
        parsed = record_parser(raw_record, record_number)
        if parsed is None:
            raise RuntimeError(f"Command emitted an empty NDJSON record {record_number}")
        record, item_count = parsed
        if item_count < 0:
            raise RuntimeError(f"NDJSON parser returned an invalid item count for record {record_number}")
        records.append(record)
        observed_items += item_count
        if item_limit is not None and observed_items > item_limit:
            observed_extra = True

    try:
        stdout_finished = False
        while not stdout_finished and not observed_extra:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise RuntimeError(f"Command timed out after {timeout_seconds:g} seconds")
            try:
                event = stdout_events.get(timeout=remaining_seconds)
            except queue.Empty as error:
                raise RuntimeError(f"Command timed out after {timeout_seconds:g} seconds") from error
            if isinstance(event, _PipeReadFailure):
                raise RuntimeError(f"Could not read command NDJSON output: {event.error}") from event.error
            if isinstance(event, _PipeEof):
                stdout_finished = True
                break

            aggregate_bytes += len(event)
            if aggregate_bytes > MAX_STRUCTURED_OUTPUT_BYTES:
                raise RuntimeError(
                    f"Command structured output exceeds the {MAX_STRUCTURED_OUTPUT_BYTES // (1024 * 1024)} MiB aggregate limit"
                )
            record_buffer.extend(event)
            newline_index = record_buffer.find(b"\n")
            while newline_index >= 0:
                if newline_index > MAX_NDJSON_RECORD_BYTES:
                    raise RuntimeError(
                        f"Command NDJSON record exceeds the {MAX_NDJSON_RECORD_BYTES // 1024} KiB limit (record {record_number + 1})"
                    )
                raw_record = bytes(record_buffer[:newline_index])
                del record_buffer[: newline_index + 1]
                consume_record(raw_record)
                if observed_extra:
                    break
                newline_index = record_buffer.find(b"\n")
            if not observed_extra and len(record_buffer) > MAX_NDJSON_RECORD_BYTES:
                raise RuntimeError(
                    f"Command NDJSON record exceeds the {MAX_NDJSON_RECORD_BYTES // 1024} KiB limit (record {record_number + 1})"
                )

        if observed_extra:
            if process.poll() is None:
                terminated_for_limit = True
                _terminate_and_reap(process, process_group)
            else:
                process.wait()
        else:
            if record_buffer:
                raise RuntimeError(f"Command emitted an incomplete NDJSON record {record_number + 1}")
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise RuntimeError(f"Command timed out after {timeout_seconds:g} seconds")
            try:
                process.wait(timeout=remaining_seconds)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"Command timed out after {timeout_seconds:g} seconds") from error
    finally:
        _terminate_and_reap(process, process_group)
        for thread in threads:
            thread.join(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
        stop_io.set()
        for pipe in (process.stdout, process.stderr):
            try:
                pipe.close()
            except OSError:
                pass
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=PROCESS_TERMINATION_GRACE_SECONDS)

    if any(thread.is_alive() for thread in threads):
        raise RuntimeError("Command pipe workers did not stop after process exit")
    if io_failures:
        raise RuntimeError(f"Could not transfer command data: {io_failures[0]}") from io_failures[0]
    if stderr_overflow[0]:
        raise RuntimeError(f"Command diagnostic output exceeds the {MAX_SUBPROCESS_DIAGNOSTIC_BYTES // 1024} KiB limit")
    stderr_text = _decode_utf8_output(bytes(stderr), truncated=False, label="stderr")
    return StreamedNDJSONProcess(
        records=records,
        observed_extra=observed_extra,
        returncode=process.returncode,
        stderr=stderr_text,
        terminated_for_limit=terminated_for_limit,
    )


def run_outline_process(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    working_directory: Path,
    node_limit: int,
    popen_factory: PopenFactory = subprocess.Popen,
) -> OutlineProcessResult:
    result = run_ndjson_process(
        command,
        timeout_seconds=timeout_seconds,
        working_directory=working_directory,
        record_parser=lambda raw, number: _parse_outline_record(raw, record_number=number),
        item_limit=node_limit,
        popen_factory=popen_factory,
    )
    if not result.terminated_for_limit and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {_bounded_error_text(result.stderr)}")
    residual_stderr = _residual_stderr(result.stderr)
    if residual_stderr:
        raise RuntimeError(f"ast-grep outline failed: {_bounded_error_text(residual_stderr)}")
    return result.records, result.observed_extra


def format_matches_as_text(matches: Sequence[Mapping[str, Any]]) -> str:
    output_blocks: list[str] = []
    for match in matches:
        file_path = str(match.get("file", ""))
        range_value = match.get("range")
        range_mapping: Mapping[str, Any] = cast(Mapping[str, Any], range_value) if isinstance(range_value, Mapping) else {}
        start_value = range_mapping.get("start")
        end_value = range_mapping.get("end")
        start_mapping: Mapping[str, Any] = cast(Mapping[str, Any], start_value) if isinstance(start_value, Mapping) else {}
        end_mapping: Mapping[str, Any] = cast(Mapping[str, Any], end_value) if isinstance(end_value, Mapping) else {}
        start_line = int(start_mapping.get("line", 0)) + 1
        end_line = int(end_mapping.get("line", start_line - 1)) + 1
        match_text = str(match.get("text", "")).rstrip()
        header = f"{file_path}:{start_line}" if start_line == end_line else f"{file_path}:{start_line}-{end_line}"
        details: list[str] = []
        rule_id = match.get("ruleId")
        if isinstance(rule_id, str) and rule_id:
            severity = match.get("severity")
            message = match.get("message")
            summary = f"Rule: {rule_id}"
            if isinstance(severity, str):
                summary += f" ({severity})"
            if isinstance(message, str) and message:
                summary += f" — {message}"
            details.append(summary)
        if "replacement" in match:
            replacement = match.get("replacement")
            preview = "<delete match>" if replacement == "" else str(replacement)
            details.append(f"Preview replacement: {preview}")
            if "replacementOffsets" in match:
                details.append(f"Replacement offsets: {json.dumps(match['replacementOffsets'], separators=(',', ':'))}")
        meta_variables = match.get("metaVariables")
        if isinstance(meta_variables, Mapping):
            typed_meta_variables = cast(Mapping[str, Any], meta_variables)
            transformed = typed_meta_variables.get("transformed")
            if isinstance(transformed, Mapping) and transformed:
                typed_transformed = cast(Mapping[str, Any], transformed)
                details.append(f"Transformed metavariables: {json.dumps(dict(typed_transformed), separators=(',', ':'))}")
        for field_name in ("fix", "transform", "rewriters"):
            if field_name in match:
                details.append(f"{field_name}: {json.dumps(match[field_name], separators=(',', ':'))}")
        output_blocks.append("\n".join([header, match_text, *details]))
    return "\n\n".join(output_blocks)


def format_search_results(results: SearchResults) -> str:
    if results["returned"] == 0:
        return "No matches found"
    noun = "match" if results["returned"] == 1 else "matches"
    header = f"Found {results['returned']} {noun}"
    if results["truncated"]:
        header += f" (limit {results['limit']}; additional matches exist)"
    return f"{header}:\n\n{format_matches_as_text(results['matches'])}"


def search_tool_result(results: SearchResults, output_format: OutputFormat) -> CallToolResult:
    if output_format == "json":
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(results, separators=(",", ":")))],
            structured_content=dict(results),
        )
    return CallToolResult(
        content=[TextContent(type="text", text=format_search_results(results))],
        structured_content=dict(results),
    )


def _format_outline_nodes(nodes: Sequence[Mapping[str, Any]], *, depth: int) -> list[str]:
    lines: list[str] = []
    for node in nodes:
        signature = node.get("signature")
        name = node.get("name")
        symbol_type = node.get("symbolType")
        label = signature if isinstance(signature, str) and signature else name
        if not isinstance(label, str) or not label:
            label = str(symbol_type) if isinstance(symbol_type, str) and symbol_type else "(unnamed)"
        lines.append(f"{'  ' * depth}- {label}")
        members = node.get("members")
        if isinstance(members, list):
            lines.extend(_format_outline_nodes(cast(list[Mapping[str, Any]], members), depth=depth + 1))
    return lines


def format_outline_results(results: OutlineResults) -> str:
    if results["returned"] == 0:
        return "No outline nodes found"
    noun = "node" if results["returned"] == 1 else "nodes"
    header = f"Found {results['returned']} outline {noun}"
    if results["truncated"]:
        header += f" (limit {results['limit']}; additional nodes exist)"
    sections = [header]
    for file_result in results["files"]:
        file_header = file_result["file"]
        if file_result["language"]:
            file_header += f" ({file_result['language']})"
        sections.append("\n".join([file_header, *_format_outline_nodes(file_result["items"], depth=1)]))
    return "\n\n".join(sections)


def outline_tool_result(results: OutlineResults, output_format: OutputFormat) -> CallToolResult:
    text = json.dumps(results, separators=(",", ":")) if output_format == "json" else format_outline_results(results)
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=dict(results),
    )


class AstGrepService:
    def __init__(
        self,
        runtime: ServerRuntime,
        *,
        runner: ProcessRunner = subprocess.run,
        outline_runner: OutlineProcessRunner = run_outline_process,
    ) -> None:
        self.runtime = runtime
        self.runner = runner
        self.outline_runner = outline_runner

    def _run(
        self,
        subcommand: str,
        arguments: Sequence[str],
        *,
        input_text: str | None = None,
        working_directory: Path | None = None,
        allow_no_matches: bool = False,
        allow_stderr_on_no_matches: bool = False,
        stderr_limit: int = MAX_SUBPROCESS_DIAGNOSTIC_BYTES,
        truncate_stderr: bool = True,
    ) -> CompletedTextProcess:
        with ExitStack() as stack:
            config_path = self._resolve_operation_config_path(stack)

            command = [
                *self.runtime.executable.command_prefix,
                subcommand,
                "--config",
                str(config_path),
                *arguments,
            ]
            result = run_process(
                command,
                timeout_seconds=self.runtime.command_timeout_seconds,
                input_text=input_text,
                working_directory=working_directory or self.runtime.working_directory,
                allowed_exit_codes=frozenset({0, 1}) if allow_no_matches else frozenset({0}),
                runner=self.runner,
                stderr_limit=stderr_limit,
                truncate_stderr=truncate_stderr,
            )
        if not allow_stderr_on_no_matches:
            residual = _residual_stderr(result.stderr)
            if residual:
                raise RuntimeError(f"ast-grep search failed: {_bounded_error_text(residual)}")
        return result

    def _resolve_operation_config_path(self, stack: ExitStack, override: Path | None = None) -> Path:
        snapshot = self.runtime.config_snapshot
        config_path = override or (snapshot.inline_config_path if snapshot is not None else self.runtime.config_path)
        if config_path is None:
            temporary_directory = Path(
                stack.enter_context(
                    TemporaryDirectory(
                        prefix="operation-",
                        dir=private_runtime_root(self.runtime.working_directory, self.runtime.allowed_roots),
                    )
                )
            )
            config_path = temporary_directory / "sgconfig.yml"
            config_path.write_text(NEUTRAL_AST_GREP_CONFIG, encoding="utf-8")
        return config_path

    def _run_match_stream(
        self,
        subcommand: str,
        arguments: Sequence[str],
        *,
        working_directory: Path,
        item_limit: int,
        input_text: str | None = None,
        config_path_override: Path | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        with ExitStack() as stack:
            config_path = self._resolve_operation_config_path(stack, config_path_override)
            command = [
                *self.runtime.executable.command_prefix,
                subcommand,
                "--config",
                str(config_path),
                *arguments,
            ]
            if self.runner is not subprocess.run:
                completed = run_process(
                    command,
                    timeout_seconds=self.runtime.command_timeout_seconds,
                    input_text=input_text,
                    working_directory=working_directory,
                    allowed_exit_codes=frozenset({0, 1}),
                    runner=self.runner,
                )
                residual = _residual_stderr(completed.stderr)
                if residual:
                    raise RuntimeError(f"ast-grep search failed: {_bounded_error_text(residual)}")
                records = parse_stream_matches(completed.stdout)
                return records, len(records) > item_limit

            streamed = run_ndjson_process(
                command,
                timeout_seconds=self.runtime.command_timeout_seconds,
                working_directory=working_directory,
                record_parser=_parse_match_record,
                item_limit=item_limit,
                input_text=input_text,
            )
        if not streamed.terminated_for_limit and streamed.returncode not in {0, 1}:
            raise RuntimeError(f"Command failed with exit code {streamed.returncode}: {_bounded_error_text(streamed.stderr)}")
        residual = _residual_stderr(streamed.stderr)
        if residual:
            raise RuntimeError(f"ast-grep search failed: {_bounded_error_text(residual)}")
        return streamed.records, streamed.observed_extra

    @staticmethod
    def _require_language(language: str) -> None:
        if not language:
            raise ValueError("language is required")

    def _resolve_project(self, project_folder: str) -> Path:
        project = _resolve_existing_path(
            project_folder,
            base=self.runtime.working_directory,
            kind="directory",
        )
        _require_allowed(project, self.runtime.allowed_roots, label="Project folder")
        return project

    def _resolve_paths(self, project: Path, raw_paths: Sequence[str] | None) -> list[Path]:
        paths = ["."] if raw_paths is None else list(raw_paths)
        if not paths:
            raise ValueError("paths must contain at least one relative path")
        resolved_paths: list[Path] = []
        for raw_path in paths:
            if not raw_path:
                raise ValueError("paths cannot contain empty values")
            candidate = Path(raw_path)
            if candidate.is_absolute():
                raise ValueError(f"Search paths must be relative to project_folder: {raw_path}")
            try:
                resolved = (project / candidate).resolve(strict=True)
            except FileNotFoundError as error:
                raise ValueError(f"Search path does not exist: {raw_path}") from error
            if not is_within(resolved, project):
                raise ValueError(f"Search path resolves outside project_folder: {raw_path}")
            _require_allowed(resolved, self.runtime.allowed_roots, label="Search path")
            if resolved not in resolved_paths:
                resolved_paths.append(resolved)
        return resolved_paths

    def _resolve_outline_paths(self, project: Path, raw_paths: Sequence[str]) -> list[Path]:
        if not 1 <= len(raw_paths) <= MAX_OUTLINE_PATHS:
            raise ValueError(f"paths must contain between 1 and {MAX_OUTLINE_PATHS} relative files")
        resolved_paths: list[Path] = []
        for raw_path in raw_paths:
            if not raw_path or "\0" in raw_path:
                raise ValueError("Outline paths cannot contain empty or NUL values")
            candidate = Path(raw_path)
            if candidate.is_absolute():
                raise ValueError(f"Outline paths must be relative to project_folder: {raw_path}")
            try:
                resolved = (project / candidate).resolve(strict=True)
            except FileNotFoundError as error:
                raise ValueError(f"Outline path does not exist: {raw_path}") from error
            if not is_within(resolved, project):
                raise ValueError(f"Outline path resolves outside project_folder: {raw_path}")
            _require_allowed(resolved, self.runtime.allowed_roots, label="Outline path")
            if not resolved.is_file():
                raise ValueError(f"Outline path must be a regular file: {raw_path}")
            if resolved not in resolved_paths:
                resolved_paths.append(resolved)
        return resolved_paths

    @staticmethod
    def _glob_arguments(include_globs: Sequence[str] | None, exclude_globs: Sequence[str] | None) -> list[str]:
        arguments: list[str] = []
        for glob in include_globs or ():
            if not glob or "\0" in glob or glob.startswith("!"):
                raise ValueError(f"Invalid include glob: {glob!r}")
            arguments.append(f"--globs={glob}")
        for glob in exclude_globs or ():
            if not glob or "\0" in glob or glob.startswith("!"):
                raise ValueError(f"Invalid exclude glob: {glob!r}")
            arguments.append(f"--globs=!{glob}")
        return arguments

    @staticmethod
    def _contained_relative_path(raw_path: str, project: Path, *, noun: str) -> str:
        file_path = Path(raw_path)
        if not file_path.is_absolute():
            file_path = project / file_path
        try:
            resolved_file = file_path.resolve(strict=True)
        except FileNotFoundError as error:
            raise RuntimeError(f"ast-grep returned {noun} that no longer exists: {raw_path}") from error
        if not is_within(resolved_file, project):
            raise RuntimeError(f"ast-grep returned {noun} outside project_folder: {raw_path}")
        return resolved_file.relative_to(project).as_posix()

    def _normalize_match_path(self, match: dict[str, Any], project: Path) -> dict[str, Any]:
        raw_file = match.get("file")
        if not isinstance(raw_file, str) or not raw_file:
            return match
        normalized = dict(match)
        normalized["file"] = self._contained_relative_path(raw_file, project, noun="a match")
        return normalized

    def _result_limit(self, requested: int | None) -> int:
        limit = self.runtime.default_max_results if requested is None else requested
        if not 1 <= limit <= self.runtime.max_results_cap:
            raise ValueError(f"max_results must be between 1 and {self.runtime.max_results_cap}")
        return limit

    def _search(
        self,
        *,
        project_folder: str,
        rule_yaml: str,
        paths: Sequence[str] | None,
        include_globs: Sequence[str] | None,
        exclude_globs: Sequence[str] | None,
        max_results: int | None,
        include_metadata: bool = False,
    ) -> SearchResults:
        validate_rule_yaml(rule_yaml, forbid_regex_rules=self.runtime.forbid_regex_rules)
        project = self._resolve_project(project_folder)
        search_paths = self._resolve_paths(project, paths)
        limit = self._result_limit(max_results)
        arguments = [
            "--inline-rules",
            rule_yaml,
            "--json=stream",
            "--max-results",
            str(limit + 1),
            *(["--include-metadata"] if include_metadata else []),
            *self._glob_arguments(include_globs, exclude_globs),
            "--",
            *(str(path) for path in search_paths),
        ]
        parsed_matches, observed_extra = self._run_match_stream(
            "scan",
            arguments,
            working_directory=project,
            item_limit=limit,
        )
        truncated = observed_extra or len(parsed_matches) > limit
        matches = [self._normalize_match_path(match, project) for match in parsed_matches[:limit]]
        return {
            "matches": matches,
            "returned": len(matches),
            "truncated": truncated,
            "limit": limit,
        }

    @classmethod
    def _trim_outline_nodes(
        cls,
        nodes: Sequence[dict[str, Any]],
        remaining: int,
    ) -> tuple[list[dict[str, Any]], int]:
        kept: list[dict[str, Any]] = []
        consumed = 0
        for node in nodes:
            if consumed >= remaining:
                break
            normalized = dict(node)
            consumed += 1
            members = node.get("members")
            if isinstance(members, list):
                kept_members, member_count = cls._trim_outline_nodes(
                    cast(list[dict[str, Any]], members),
                    remaining - consumed,
                )
                normalized["members"] = kept_members
                consumed += member_count
            kept.append(normalized)
        return kept, consumed

    def _bound_outline(
        self,
        documents: Sequence[dict[str, Any]],
        project: Path,
        requested_paths: Sequence[Path],
        limit: int,
        *,
        observed_extra: bool,
    ) -> OutlineResults:
        files: list[OutlineFile] = []
        remaining = limit
        returned = 0
        truncated = observed_extra
        requested = {path.relative_to(project).as_posix() for path in requested_paths}
        observed_paths: set[str] = set()
        for record_number, document in enumerate(documents, start=1):
            node_count = _validate_and_count_outline_document(document, record_number=record_number)
            items = document.get("items")
            raw_path = document.get("path")
            language = document.get("language")
            if not isinstance(items, list) or not isinstance(raw_path, str) or not isinstance(language, str):
                raise RuntimeError(f"ast-grep outline record {record_number} failed validation")
            normalized_path = self._contained_relative_path(raw_path, project, noun="an outline path")
            if normalized_path not in requested:
                raise RuntimeError(f"ast-grep returned an outline path that was not requested: {raw_path}")
            if normalized_path in observed_paths:
                raise RuntimeError(f"ast-grep returned a duplicate outline path: {raw_path}")
            observed_paths.add(normalized_path)
            kept, consumed = self._trim_outline_nodes(cast(list[dict[str, Any]], items), remaining)
            if consumed < node_count:
                truncated = True
            if kept or not items:
                normalized_document = dict(document)
                normalized_document.pop("path", None)
                normalized_document["file"] = normalized_path
                normalized_document["items"] = kept
                files.append(cast(OutlineFile, normalized_document))
            returned += consumed
            remaining -= consumed
        return {
            "files": files,
            "returned": returned,
            "truncated": truncated,
            "limit": limit,
        }

    def outline_code(
        self,
        *,
        project_folder: str,
        paths: Sequence[str],
        language: str | None,
        max_results: int | None,
        items: OutlineItemsMode = "auto",
        symbol_types: Sequence[str] | None = None,
        public_members: bool = False,
    ) -> OutlineResults:
        if language is not None and not language:
            raise ValueError("language cannot be empty when provided")
        project = self._resolve_project(project_folder)
        outline_paths = self._resolve_outline_paths(project, paths)
        limit = self._result_limit(max_results)
        if items not in {"auto", "structure", "exports", "imports", "all"}:
            raise ValueError(f"Unknown outline item mode: {items}")
        normalized_symbol_types: list[str] = []
        for symbol_type_value in cast(Sequence[object], symbol_types or ()):
            if not isinstance(symbol_type_value, str) or symbol_type_value not in _OUTLINE_SYMBOL_TYPES:
                raise ValueError(f"Unknown outline symbol type: {symbol_type_value!r}")
            symbol_type = symbol_type_value
            if symbol_type not in normalized_symbol_types:
                normalized_symbol_types.append(symbol_type)
        arguments = [
            "--json=stream",
            "--threads",
            "1",
            "--items",
            items,
            *(["--type", ",".join(normalized_symbol_types)] if normalized_symbol_types else []),
            *(["--pub-members"] if public_members else []),
            *(["--lang", language] if language is not None else []),
            "--",
            *(str(path) for path in outline_paths),
        ]
        with ExitStack() as stack:
            config_path = self._resolve_operation_config_path(stack)
            command = [
                *self.runtime.executable.command_prefix,
                "outline",
                "--config",
                str(config_path),
                *arguments,
            ]
            documents, observed_extra = self.outline_runner(
                command,
                timeout_seconds=self.runtime.command_timeout_seconds,
                working_directory=project,
                node_limit=limit,
            )
        return self._bound_outline(documents, project, outline_paths, limit, observed_extra=observed_extra)

    def dump_syntax_tree(self, *, code: str, language: str, format: DumpFormat) -> str:
        self._require_language(language)
        _encode_bounded_input(code)
        with TemporaryDirectory(
            prefix="syntax-",
            dir=private_runtime_root(self.runtime.working_directory, self.runtime.allowed_roots),
        ) as sandbox:
            result = self._run(
                "run",
                ["--pattern", code, "--lang", language, f"--debug-query={format}"],
                working_directory=Path(sandbox),
                allow_no_matches=True,
                allow_stderr_on_no_matches=True,
                stderr_limit=MAX_STRUCTURED_OUTPUT_BYTES,
                truncate_stderr=False,
            )
        return result.stderr.strip()

    def test_match_code_rule(self, *, code: str, rule_yaml: str) -> list[dict[str, Any]]:
        validate_rule_yaml(rule_yaml, forbid_regex_rules=self.runtime.forbid_regex_rules)
        matches, observed_extra = self._run_match_stream(
            "scan",
            [
                "--inline-rules",
                rule_yaml,
                "--json=stream",
                "--max-results",
                str(self.runtime.max_results_cap + 1),
                "--stdin",
            ],
            working_directory=self.runtime.working_directory,
            item_limit=self.runtime.max_results_cap,
            input_text=code,
        )
        if observed_extra:
            raise RuntimeError("ast-grep rule probe exceeded the configured result cap")
        return matches

    def find_code(
        self,
        *,
        project_folder: str,
        pattern: str,
        language: str,
        paths: Sequence[str] | None,
        include_globs: Sequence[str] | None,
        exclude_globs: Sequence[str] | None,
        max_results: int | None,
        selector: str | None = None,
        strictness: Strictness = "smart",
        rewrite: str | None = None,
    ) -> SearchResults:
        self._require_language(language)
        _encode_bounded_input(pattern)
        if selector is not None and (not selector or "\0" in selector):
            raise ValueError("selector must be a non-empty NUL-free ast-grep kind when provided")
        if strictness not in {"cst", "smart", "ast", "relaxed", "signature", "template"}:
            raise ValueError(f"Unknown ast-grep strictness: {strictness}")
        if rewrite is not None:
            _encode_bounded_input(rewrite)
        pattern_value: str | dict[str, str]
        if selector is None and strictness == "smart":
            pattern_value = pattern
        else:
            pattern_value = {"context": pattern, "strictness": strictness}
            if selector is not None:
                pattern_value["selector"] = selector
        rule_config: dict[str, Any] = {
            "id": "mcp-pattern-search",
            "language": language,
            "rule": {"pattern": pattern_value},
        }
        if rewrite is not None:
            rule_config["fix"] = rewrite
        rule_yaml = yaml_parser.safe_dump(rule_config, sort_keys=False)
        return self._search(
            project_folder=project_folder,
            rule_yaml=rule_yaml,
            paths=paths,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            max_results=max_results,
        )

    def find_code_by_rule(
        self,
        *,
        project_folder: str,
        rule_yaml: str,
        paths: Sequence[str] | None,
        include_globs: Sequence[str] | None,
        exclude_globs: Sequence[str] | None,
        max_results: int | None,
        include_metadata: bool = False,
    ) -> SearchResults:
        return self._search(
            project_folder=project_folder,
            rule_yaml=rule_yaml,
            paths=paths,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            max_results=max_results,
            include_metadata=include_metadata,
        )

    @staticmethod
    def _exact_rule_filter(
        rule_ids: Sequence[str] | None,
        *,
        configured_rule_ids: Sequence[str],
    ) -> str | None:
        if rule_ids is None:
            return None
        if not rule_ids:
            raise ValueError("rule_ids must contain at least one configured rule id when provided")
        unique: list[str] = []
        configured = set(configured_rule_ids)
        for rule_id_value in cast(Sequence[object], rule_ids):
            if not isinstance(rule_id_value, str) or not rule_id_value or "\0" in rule_id_value:
                raise ValueError(f"Invalid configured rule id: {rule_id_value!r}")
            rule_id = rule_id_value
            if rule_id not in configured:
                raise ValueError(f"Unknown configured rule id: {rule_id}")
            if rule_id not in unique:
                unique.append(rule_id)
        return rf"^(?:{'|'.join(re.escape(rule_id) for rule_id in unique)})$"

    def scan_project_rules(
        self,
        *,
        project_folder: str,
        rule_ids: Sequence[str] | None,
        paths: Sequence[str] | None,
        include_globs: Sequence[str] | None,
        exclude_globs: Sequence[str] | None,
        max_results: int | None,
        include_metadata: bool = True,
    ) -> SearchResults:
        snapshot = self.runtime.config_snapshot
        if snapshot is None or not snapshot.capabilities["configured_scan"]:
            raise ValueError("No project rules were configured at server startup")
        project = self._resolve_project(project_folder)
        search_paths = self._resolve_paths(project, paths)
        limit = self._result_limit(max_results)
        rule_filter = self._exact_rule_filter(rule_ids, configured_rule_ids=snapshot.configured_rule_ids)
        arguments = [
            "--json=stream",
            "--max-results",
            str(limit + 1),
            *(["--filter", rule_filter] if rule_filter is not None else []),
            *(["--include-metadata"] if include_metadata else []),
            *self._glob_arguments(include_globs, exclude_globs),
            "--",
            *(str(path.relative_to(project)) for path in search_paths),
        ]
        parsed_matches, observed_extra = self._run_match_stream(
            "scan",
            arguments,
            working_directory=project,
            item_limit=limit,
            config_path_override=snapshot.project_config_path,
        )
        truncated = observed_extra or len(parsed_matches) > limit
        matches = [self._normalize_match_path(match, project) for match in parsed_matches[:limit]]
        return {
            "matches": matches,
            "returned": len(matches),
            "truncated": truncated,
            "limit": limit,
        }

    @staticmethod
    def _bounded_report(stdout: str, stderr: str) -> tuple[str, bool]:
        report = stdout.rstrip()
        diagnostic = stderr.rstrip()
        if diagnostic:
            report = f"{report}\n{diagnostic}" if report else diagnostic
        encoded = report.encode("utf-8")
        if len(encoded) <= MAX_TEST_REPORT_BYTES:
            return report, False
        marker = "…"
        decoded = _decode_utf8_prefix(encoded[: MAX_TEST_REPORT_BYTES - len(marker.encode("utf-8"))])
        return (decoded + marker, True) if decoded else (marker, True)

    def test_project_rules(self, *, rule_ids: Sequence[str] | None = None) -> ProjectTestResults:
        snapshot = self.runtime.config_snapshot
        if snapshot is None or not snapshot.capabilities["configured_tests"]:
            raise ValueError("No project rule tests were configured at server startup")
        rule_filter = self._exact_rule_filter(rule_ids, configured_rule_ids=snapshot.configured_rule_ids)
        command = [
            *self.runtime.executable.command_prefix,
            "test",
            "--config",
            str(snapshot.test_config_path),
            "--color",
            "never",
            *(["--filter", rule_filter] if rule_filter is not None else []),
        ]
        if self.runner is subprocess.run:
            streamed = run_text_process(
                command,
                timeout_seconds=self.runtime.command_timeout_seconds,
                working_directory=snapshot.bundle_root,
                stdout_limit=MAX_TEST_REPORT_BYTES,
                stderr_limit=MAX_TEST_REPORT_BYTES,
                truncate_stdout=True,
                truncate_stderr=True,
            )
            completed = streamed.completed
            stream_truncated = streamed.stdout_truncated or streamed.stderr_truncated
        else:
            completed = run_process(
                command,
                timeout_seconds=self.runtime.command_timeout_seconds,
                working_directory=snapshot.bundle_root,
                allowed_exit_codes=frozenset({0, 4}),
                runner=self.runner,
            )
            stream_truncated = False
        if completed.returncode not in {0, 4}:
            raise RuntimeError(
                f"Configured rule tests failed to execute with exit code {completed.returncode}: {_bounded_error_text(completed.stderr)}"
            )
        report, report_truncated = self._bounded_report(completed.stdout, completed.stderr)
        return {
            "passed": completed.returncode == 0,
            "report": report,
            "report_truncated": stream_truncated or report_truncated,
        }

    def get_server_info(self) -> ServerInfo:
        snapshot = self.runtime.config_snapshot
        posix_arg_max = _detected_posix_arg_max()
        return {
            "fork_version": _server_version(),
            "ast_grep_executable": str(self.runtime.executable.path),
            "ast_grep_version": self.runtime.ast_grep_version,
            "config_path": str(self.runtime.config_path) if self.runtime.config_path is not None else None,
            "allowed_roots": [str(root) for root in self.runtime.allowed_roots],
            "command_timeout_seconds": self.runtime.command_timeout_seconds,
            "default_max_results": self.runtime.default_max_results,
            "max_results_cap": self.runtime.max_results_cap,
            "forbid_regex_rules": self.runtime.forbid_regex_rules,
            "configuration_digest": snapshot.digest if snapshot is not None else "unavailable",
            "configuration_provenance": (
                dict(snapshot.provenance)
                if snapshot is not None
                else {"source": str(self.runtime.config_path) if self.runtime.config_path is not None else None, "snapshot": "test-runtime"}
            ),
            "capabilities": (
                dict(snapshot.capabilities)
                if snapshot is not None
                else {
                    "inline_search": True,
                    "outline": True,
                    "configured_scan": False,
                    "configured_tests": False,
                    "custom_languages": False,
                }
            ),
            "coordinate_conventions": {
                "line": "zero-based",
                "column": "zero-based Unicode scalar count",
                "byte_offset": "zero-based UTF-8 bytes",
                "range": "half-open [start,end)",
            },
            "resource_limits": {
                "snippet_input_bytes": MAX_SNIPPET_INPUT_BYTES,
                "inline_rule_bytes": MAX_INLINE_RULE_BYTES,
                "ndjson_record_bytes": MAX_NDJSON_RECORD_BYTES,
                "structured_output_bytes": MAX_STRUCTURED_OUTPUT_BYTES,
                "diagnostic_bytes": MAX_SUBPROCESS_DIAGNOSTIC_BYTES,
                "test_report_bytes": MAX_TEST_REPORT_BYTES,
                "config_file_bytes": MAX_CONFIG_FILE_BYTES,
                "config_resource_bytes": MAX_CONFIG_RESOURCE_BYTES,
                "config_resource_files": MAX_CONFIG_RESOURCE_FILES,
                "native_library_bytes": MAX_NATIVE_LIBRARY_BYTES,
                "yaml_documents": MAX_YAML_DOCUMENTS,
                "yaml_nodes": MAX_YAML_NODES,
                "yaml_depth": MAX_YAML_DEPTH,
                "outline_paths": MAX_OUTLINE_PATHS,
                "hard_max_results": HARD_MAX_RESULTS,
                "command_timeout_seconds": self.runtime.command_timeout_seconds,
                "windows_create_process_characters": WINDOWS_CREATE_PROCESS_LIMIT,
                "posix_arg_max_bytes": posix_arg_max,
                "posix_arg_headroom_bytes": POSIX_ARG_HEADROOM_BYTES,
                "posix_effective_launch_budget_bytes": (posix_arg_max - POSIX_ARG_HEADROOM_BYTES if posix_arg_max else 0),
                "process_termination_grace_seconds": PROCESS_TERMINATION_GRACE_SECONDS,
            },
        }


_runtime: ServerRuntime | None = None


def configure_runtime(runtime: ServerRuntime) -> None:
    global _runtime
    _runtime = runtime


def current_service() -> AstGrepService:
    if _runtime is None:
        raise RuntimeError("ast-grep MCP runtime has not been configured")
    return AstGrepService(_runtime)


mcp = MCPServer(
    "ast-grep",
    instructions=(
        "Read-only structural code inspection. Inspect syntax and probe rules before bounded project searches. "
        "Use repository CLI workflows for exhaustive scans and rewrites."
    ),
    version=_server_version(),
)


def register_mcp_tools(server: MCPServer) -> None:
    @server.tool(annotations=READ_ONLY_ANNOTATIONS, title="Dump syntax tree")
    def dump_syntax_tree(
        code: Annotated[str, Field(description="Code or pattern to inspect")],
        language: Annotated[str, Field(description="Explicit ast-grep language")],
        format: Annotated[
            DumpFormat,
            Field(description="Syntax dump format: pattern, ast, cst, or sexp"),
        ] = "cst",
    ) -> str:
        """Inspect how ast-grep parses code or a query pattern."""
        return current_service().dump_syntax_tree(code=code, language=language, format=format)

    @server.tool(annotations=READ_ONLY_ANNOTATIONS, title="Test rule against code")
    def test_match_code_rule(
        code: Annotated[str, Field(description="Code to test against the rule")],
        yaml: Annotated[str, Field(description="ast-grep YAML with id, language, and rule fields")],
    ) -> list[dict[str, Any]]:
        """Probe an ast-grep YAML rule against a code snippet; a valid negative probe returns an empty list."""
        return current_service().test_match_code_rule(code=code, rule_yaml=yaml)

    @server.tool(annotations=READ_ONLY_ANNOTATIONS, title="Outline code")
    def outline_code(
        project_folder: Annotated[
            str,
            Field(description="Project directory, resolved and constrained to the server's allowed roots"),
        ],
        paths: Annotated[
            list[str],
            Field(
                min_length=1,
                max_length=MAX_OUTLINE_PATHS,
                description=(
                    "One to 64 relative regular files under project_folder; directories, absolute paths, "
                    "and paths that escape the project are rejected"
                ),
            ),
        ],
        language: Annotated[
            str | None,
            Field(description="Optional explicit ast-grep language; omitted to use extension-based detection"),
        ] = None,
        max_results: Annotated[
            int | None,
            Field(
                ge=1,
                le=HARD_MAX_RESULTS,
                description=(
                    "Finite symbol limit across all files; defaults to the server's configured limit. This schema "
                    "bound is the hard ceiling: an operator may configure a lower cap, which get_server_info reports "
                    "as max_results_cap and which rejects larger values at call time."
                ),
            ),
        ] = None,
        items: Annotated[
            OutlineItemsMode,
            Field(description="Top-level item mode: auto, structure, exports, imports, or all"),
        ] = "auto",
        symbol_types: Annotated[
            list[str] | None,
            Field(description="Optional exact top-level outline symbol types such as class, enum, or function"),
        ] = None,
        public_members: Annotated[
            bool,
            Field(description="Retain only public members in member-bearing outline views"),
        ] = False,
        output_format: Annotated[
            OutputFormat,
            Field(description="Compact text or structured JSON result"),
        ] = "text",
    ) -> Annotated[CallToolResult, OutlineResults]:
        """Extract a bounded per-file symbol hierarchy from explicit regular files."""
        results = current_service().outline_code(
            project_folder=project_folder,
            paths=paths,
            language=language,
            max_results=max_results,
            items=items,
            symbol_types=symbol_types,
            public_members=public_members,
        )
        return outline_tool_result(results, output_format)

    @server.tool(annotations=READ_ONLY_ANNOTATIONS, title="Find code by pattern")
    def find_code(
        project_folder: Annotated[
            str,
            Field(description="Project directory, resolved and constrained to the server's allowed roots"),
        ],
        pattern: Annotated[str, Field(description="Valid ast-grep structural pattern")],
        language: Annotated[str, Field(description="Required ast-grep language")],
        selector: Annotated[
            str | None,
            Field(description="Optional AST kind selecting the matched sub-part of the pattern context"),
        ] = None,
        strictness: Annotated[
            Strictness,
            Field(description="ast-grep pattern strictness"),
        ] = "smart",
        rewrite: Annotated[
            str | None,
            Field(description="Optional preview-only replacement; an empty string previews deletion"),
        ] = None,
        paths: Annotated[
            list[str] | None,
            Field(description="Relative files or directories under project_folder; defaults to the project root"),
        ] = None,
        include_globs: Annotated[
            list[str] | None,
            Field(description="Optional gitignore-style include globs"),
        ] = None,
        exclude_globs: Annotated[
            list[str] | None,
            Field(description="Optional gitignore-style exclude globs, without a leading !"),
        ] = None,
        max_results: Annotated[
            int | None,
            Field(
                ge=1,
                le=HARD_MAX_RESULTS,
                description=(
                    "Finite result limit; defaults to the server's configured limit. This schema bound is the hard "
                    "ceiling: an operator may configure a lower cap, which get_server_info reports as max_results_cap "
                    "and which rejects larger values at call time."
                ),
            ),
        ] = None,
        output_format: Annotated[
            OutputFormat,
            Field(description="Compact text or structured JSON result"),
        ] = "text",
    ) -> Annotated[CallToolResult, SearchResults]:
        """Find bounded structural pattern matches inside an allowed project scope."""
        results = current_service().find_code(
            project_folder=project_folder,
            pattern=pattern,
            language=language,
            paths=paths,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            max_results=max_results,
            selector=selector,
            strictness=strictness,
            rewrite=rewrite,
        )
        return search_tool_result(results, output_format)

    @server.tool(annotations=READ_ONLY_ANNOTATIONS, title="Find code by rule")
    def find_code_by_rule(
        project_folder: Annotated[
            str,
            Field(description="Project directory, resolved and constrained to the server's allowed roots"),
        ],
        yaml: Annotated[str, Field(description="ast-grep YAML with id, language, and rule fields")],
        paths: Annotated[
            list[str] | None,
            Field(description="Relative files or directories under project_folder; defaults to the project root"),
        ] = None,
        include_globs: Annotated[
            list[str] | None,
            Field(description="Optional gitignore-style include globs"),
        ] = None,
        exclude_globs: Annotated[
            list[str] | None,
            Field(description="Optional gitignore-style exclude globs, without a leading !"),
        ] = None,
        max_results: Annotated[
            int | None,
            Field(
                ge=1,
                le=HARD_MAX_RESULTS,
                description=(
                    "Finite result limit; defaults to the server's configured limit. This schema bound is the hard "
                    "ceiling: an operator may configure a lower cap, which get_server_info reports as max_results_cap "
                    "and which rejects larger values at call time."
                ),
            ),
        ] = None,
        include_metadata: Annotated[
            bool,
            Field(description="Include each rule's documented metadata object in ast-grep match records"),
        ] = False,
        output_format: Annotated[
            OutputFormat,
            Field(description="Compact text or structured JSON result"),
        ] = "text",
    ) -> Annotated[CallToolResult, SearchResults]:
        """Find bounded matches for one or more validated ast-grep YAML rules."""
        results = current_service().find_code_by_rule(
            project_folder=project_folder,
            rule_yaml=yaml,
            paths=paths,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            max_results=max_results,
            include_metadata=include_metadata,
        )
        return search_tool_result(results, output_format)

    @server.tool(annotations=READ_ONLY_ANNOTATIONS, title="Scan configured project rules")
    def scan_project_rules(
        project_folder: Annotated[
            str,
            Field(description="Project directory, resolved and constrained to the server's allowed roots"),
        ],
        rule_ids: Annotated[
            list[str] | None,
            Field(description="Optional exact startup-configured rule IDs; regex metacharacters are escaped"),
        ] = None,
        paths: Annotated[
            list[str] | None,
            Field(description="Relative files or directories under project_folder; defaults to the project root"),
        ] = None,
        include_globs: Annotated[
            list[str] | None,
            Field(description="Optional gitignore-style include globs"),
        ] = None,
        exclude_globs: Annotated[
            list[str] | None,
            Field(description="Optional gitignore-style exclude globs, without a leading !"),
        ] = None,
        max_results: Annotated[
            int | None,
            Field(
                ge=1,
                le=HARD_MAX_RESULTS,
                description="Finite result limit; defaults to the server limit and may not exceed its effective cap",
            ),
        ] = None,
        include_metadata: Annotated[
            bool,
            Field(description="Include configured rule metadata in match records"),
        ] = True,
        output_format: Annotated[
            OutputFormat,
            Field(description="Compact text or structured JSON result"),
        ] = "text",
    ) -> Annotated[CallToolResult, SearchResults]:
        """Run only the immutable rule set captured from startup configuration."""
        results = current_service().scan_project_rules(
            project_folder=project_folder,
            rule_ids=rule_ids,
            paths=paths,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            max_results=max_results,
            include_metadata=include_metadata,
        )
        return search_tool_result(results, output_format)

    @server.tool(annotations=READ_ONLY_ANNOTATIONS, title="Test configured project rules")
    def test_project_rules(
        rule_ids: Annotated[
            list[str] | None,
            Field(description="Optional exact startup-configured rule IDs; regex metacharacters are escaped"),
        ] = None,
    ) -> ProjectTestResults:
        """Run the immutable configured test suites without interaction or snapshot updates."""
        return current_service().test_project_rules(rule_ids=rule_ids)

    @server.tool(annotations=READ_ONLY_ANNOTATIONS, title="Get server info")
    def get_server_info() -> ServerInfo:
        """Report the fork, executable, containment, configuration, timeout, and result-limit contract."""
        return current_service().get_server_info()


register_mcp_tools(mcp)


def _environment_bool(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded, read-only ast-grep MCP server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("AST_GREP_CONFIG"),
        metavar="PATH",
        help="Path to sgconfig.yml; it must resolve inside an allowed root",
    )
    parser.add_argument(
        "--ast-grep",
        default=os.environ.get("AST_GREP_EXECUTABLE", "ast-grep"),
        metavar="PATH",
        help="ast-grep executable or command name",
    )
    parser.add_argument(
        "--allowed-root",
        action="append",
        default=None,
        metavar="PATH",
        help="Allowed project root; repeat for multiple roots (defaults to the process working directory)",
    )
    parser.add_argument(
        "--trusted-native-library",
        action="append",
        nargs=2,
        default=None,
        metavar=("PATH", "SHA256"),
        help=(
            "Trust one custom-language native parser by exact contained path and SHA-256 digest; "
            "repeat for every library referenced by sgconfig.yml"
        ),
    )
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=os.environ.get("AST_GREP_COMMAND_TIMEOUT", str(DEFAULT_COMMAND_TIMEOUT_SECONDS)),
        metavar="SECONDS",
        help="Timeout for each ast-grep subprocess",
    )
    parser.add_argument(
        "--default-max-results",
        type=int,
        default=os.environ.get("AST_GREP_DEFAULT_MAX_RESULTS", str(DEFAULT_MAX_RESULTS)),
        metavar="COUNT",
        help="Default finite search result limit",
    )
    parser.add_argument(
        "--max-results-cap",
        type=int,
        default=os.environ.get("AST_GREP_MAX_RESULTS_CAP", str(HARD_MAX_RESULTS)),
        metavar="COUNT",
        help=f"Maximum allowed result limit (never above {HARD_MAX_RESULTS})",
    )
    parser.add_argument(
        "--forbid-regex-rules",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Reject inline ast-grep YAML containing a regex key anywhere in the document, "
            "including metavariable constraints, nested relational rules, and utils; "
            "unset falls back to AST_GREP_FORBID_REGEX_RULES"
        ),
    )
    return parser


def _resolve_forbid_regex_rules(cli_value: bool | None) -> bool:
    if cli_value is not None:
        return cli_value
    return _environment_bool("AST_GREP_FORBID_REGEX_RULES")


def run_mcp_server() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        runtime = build_runtime(
            working_directory=Path.cwd(),
            ast_grep_executable=args.ast_grep,
            config_path=args.config,
            allowed_roots=args.allowed_root or (),
            command_timeout_seconds=args.command_timeout,
            default_max_results=args.default_max_results,
            max_results_cap=args.max_results_cap,
            forbid_regex_rules=_resolve_forbid_regex_rules(args.forbid_regex_rules),
            trusted_native_libraries=tuple(tuple(value) for value in (args.trusted_native_library or ())),
        )
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    configure_runtime(runtime)
    try:
        mcp.run()
    finally:
        runtime.close()


if __name__ == "__main__":
    run_mcp_server()
