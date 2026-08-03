from __future__ import annotations

import argparse
import json
import os
import platform as platform_module
import queue
import shutil
import signal
import struct
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, BinaryIO, Final, Literal, TypedDict

import yaml as yaml_parser
from mcp.server import MCPServer
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field

DEFAULT_MAX_RESULTS: Final = 50
HARD_MAX_RESULTS: Final = 500
DEFAULT_COMMAND_TIMEOUT_SECONDS: Final = 30.0
FALLBACK_SERVER_VERSION: Final = "0.3.0"
NEUTRAL_AST_GREP_CONFIG: Final = "ruleDirs: []\n"
SUPPORTED_AST_GREP_VERSION: Final = "0.45.0"
MAX_INLINE_RULE_BYTES: Final = 64 * 1024
MAX_OUTLINE_PATHS: Final = 64
MAX_OUTLINE_RECORD_BYTES: Final = 1024 * 1024
MAX_OUTLINE_AGGREGATE_BYTES: Final = 4 * 1024 * 1024
OUTLINE_READ_CHUNK_BYTES: Final = 64 * 1024
MAX_SUBPROCESS_DIAGNOSTIC_BYTES: Final = 64 * 1024
WINDOWS_CREATE_PROCESS_LIMIT: Final = 32_767
POSIX_ARG_HEADROOM_BYTES: Final = 2048
PROCESS_TERMINATION_GRACE_SECONDS: Final = 2.0

DumpFormat = Literal["pattern", "cst", "ast"]
OutputFormat = Literal["text", "json"]
CompletedTextProcess = subprocess.CompletedProcess[str]
ProcessRunner = Callable[..., CompletedTextProcess]
PopenFactory = Callable[..., subprocess.Popen[bytes]]
OutlineProcessResult = tuple[list[dict[str, Any]], bool]
OutlineProcessRunner = Callable[..., OutlineProcessResult]


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


def _is_within(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


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
    if not any(_is_within(path, root) for root in allowed_roots):
        allowed = ", ".join(str(root) for root in allowed_roots)
        raise ValueError(f"{label} resolves outside the allowed roots ({allowed}): {path}")


def _read_json_file(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _npm_package_bin(package_directory: Path) -> Path | None:
    manifest_path = package_directory / "package.json"
    manifest = _read_json_file(manifest_path)
    if manifest is None or manifest.get("name") != "@ast-grep/cli":
        return None

    bin_value = manifest.get("bin")
    if isinstance(bin_value, str):
        relative_bin = bin_value
    elif isinstance(bin_value, Mapping) and isinstance(bin_value.get("ast-grep"), str):
        relative_bin = bin_value["ast-grep"]
    else:
        return None

    try:
        target = (package_directory / relative_bin).resolve(strict=True)
    except FileNotFoundError:
        return None
    if not target.is_file() or not _is_within(target, package_directory.resolve()):
        return None
    return target


def _native_ast_grep_package_name() -> str | None:
    """Return the optional @ast-grep package for the running platform."""
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
    """Resolve the installed platform binary without launching an npm batch/JS shim."""
    manifest = _read_json_file(package_directory / "package.json")
    package_name = _native_ast_grep_package_name()
    if manifest is None or package_name is None:
        return None
    optional_dependencies = manifest.get("optionalDependencies")
    package_version = manifest.get("version")
    if (
        not isinstance(optional_dependencies, Mapping)
        or not isinstance(package_version, str)
        or optional_dependencies.get(package_name) != package_version
    ):
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
    if not executable.is_file() or not _is_within(executable, resolved_package):
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
    has_path_separator = os.sep in raw_executable or (os.altsep is not None and os.altsep in raw_executable)
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


def validate_process_budget(
    command: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
    platform_name: str | None = None,
    arg_max: int | None = None,
) -> None:
    """Reject a subprocess launch that cannot fit the host's argv/environment limits."""
    if not command:
        raise ValueError("Command must contain an executable")
    if any(not isinstance(argument, str) or "\0" in argument for argument in command):
        raise ValueError("Command arguments must be strings without NUL characters")

    executable_suffix = Path(command[0]).suffix.lower()
    if executable_suffix in {".bat", ".cmd"}:
        raise ValueError(
            "Batch-file commands are not launched directly; pass the resolved native executable or the JavaScript entry point through node"
        )

    launch_environment = os.environ if environment is None else environment
    if any(
        not isinstance(key, str) or not isinstance(value, str) or "\0" in key or "\0" in value or "=" in key
        for key, value in launch_environment.items()
    ):
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
        try:
            detected_arg_max = int(os.sysconf("SC_ARG_MAX"))
        except (AttributeError, OSError, ValueError) as error:
            raise ValueError("Could not determine the POSIX ARG_MAX process-launch budget") from error
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
    """Report whether a stderr line is ast-grep's successful error-severity summary.

    Verified against 0.45.0: a `scan` whose error-severity rule matched writes
    `Error: N error(s) found in code.` and `Help: Scan succeeded ...` while still
    streaming its findings on stdout. That is a complete, successful scan.
    """
    if line.startswith("Help:"):
        return True
    return line.startswith("Error: ") and line.endswith(" error(s) found in code.")


def _residual_stderr(stderr: str) -> str:
    """Return the stderr that remains once the benign scan summary is removed."""
    lines = (raw.strip() for raw in stderr.splitlines())
    return "\n".join(line for line in lines if line and not _is_benign_scan_diagnostic(line))


def _run_managed_completed_process(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    input_text: str | None,
    working_directory: Path | None,
    environment: Mapping[str, str],
) -> CompletedTextProcess:
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(working_directory) if working_directory is not None else None,
        env=dict(environment),
        shell=False,
        **_popen_process_group_options(),
    )
    try:
        stdout, stderr = process.communicate(input=input_text, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _terminate_and_reap(process)
        raise
    except BaseException:
        _terminate_and_reap(process)
        raise
    return subprocess.CompletedProcess(list(command), process.returncode, stdout, stderr)


def run_process(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    input_text: str | None = None,
    working_directory: Path | None = None,
    allowed_exit_codes: frozenset[int] = frozenset({0}),
    runner: ProcessRunner = subprocess.run,
) -> CompletedTextProcess:
    launch_environment = dict(os.environ)
    validate_process_budget(command, environment=launch_environment)
    try:
        if runner is subprocess.run:
            result = _run_managed_completed_process(
                command,
                timeout_seconds=timeout_seconds,
                input_text=input_text,
                working_directory=working_directory,
                environment=launch_environment,
            )
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
    parts = result.stdout.strip().split()
    if len(parts) < 2 or parts[0] != "ast-grep":
        raise ValueError(f"Configured executable is not ast-grep: {_bounded_error_text(result.stdout)}")
    executable_version = parts[1]
    if executable_version != SUPPORTED_AST_GREP_VERSION:
        raise ValueError(f"Unsupported ast-grep version {executable_version}; expected {SUPPORTED_AST_GREP_VERSION}")
    return executable_version


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
    runner: ProcessRunner = subprocess.run,
) -> ServerRuntime:
    resolved_working_directory = working_directory.resolve(strict=True)
    if not resolved_working_directory.is_dir():
        raise ValueError(f"Working directory is not a directory: {working_directory}")
    if command_timeout_seconds <= 0:
        raise ValueError("Command timeout must be greater than zero")
    if not 1 <= max_results_cap <= HARD_MAX_RESULTS:
        raise ValueError(f"Result cap must be between 1 and {HARD_MAX_RESULTS}")
    if not 1 <= default_max_results <= max_results_cap:
        raise ValueError("Default result limit must be positive and no greater than the configured cap")

    resolved_roots = resolve_allowed_roots(allowed_roots, working_directory=resolved_working_directory)
    executable = resolve_ast_grep_executable(ast_grep_executable, working_directory=resolved_working_directory)
    resolved_config: Path | None = None
    if config_path is not None:
        resolved_config = _resolve_existing_path(config_path, base=resolved_working_directory, kind="file")
        _require_allowed(resolved_config, resolved_roots, label="Config path")

    return ServerRuntime(
        working_directory=resolved_working_directory,
        executable=executable,
        ast_grep_version=_read_ast_grep_version(
            executable,
            timeout_seconds=command_timeout_seconds,
            runner=runner,
        ),
        config_path=resolved_config,
        allowed_roots=resolved_roots,
        command_timeout_seconds=command_timeout_seconds,
        default_max_results=default_max_results,
        max_results_cap=max_results_cap,
        forbid_regex_rules=forbid_regex_rules,
    )


def _contains_mapping_key(value: Any, forbidden_key: str) -> bool:
    if isinstance(value, Mapping):
        return any(key == forbidden_key or _contains_mapping_key(item, forbidden_key) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_mapping_key(item, forbidden_key) for item in value)
    return False


def validate_rule_yaml(rule_yaml: str, *, forbid_regex_rules: bool) -> None:
    if len(rule_yaml.encode("utf-8")) > MAX_INLINE_RULE_BYTES:
        raise ValueError(f"ast-grep rule YAML exceeds the {MAX_INLINE_RULE_BYTES // 1024} KiB inline limit")
    try:
        documents = list(yaml_parser.safe_load_all(rule_yaml))
    except yaml_parser.YAMLError as error:
        raise ValueError(f"Invalid ast-grep rule YAML: {error}") from error
    if not documents or all(document is None for document in documents):
        raise ValueError("ast-grep rule YAML must contain at least one rule")

    for document in documents:
        if not isinstance(document, Mapping):
            raise ValueError("Each ast-grep inline rule must be a YAML mapping")
        missing = [key for key in ("id", "language", "rule") if key not in document]
        if missing:
            raise ValueError(f"ast-grep rule is missing required fields: {', '.join(missing)}")
        if not isinstance(document["rule"], Mapping):
            raise ValueError("ast-grep rule field must be a mapping")
        if forbid_regex_rules and _contains_mapping_key(document, "regex"):
            raise ValueError("Regex ast-grep rules are disabled by server policy")


def parse_stream_matches(stdout: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(stdout.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"ast-grep emitted invalid JSON on line {line_number}") from error
        if not isinstance(value, dict):
            raise RuntimeError(f"ast-grep emitted a non-object JSON match on line {line_number}")
        matches.append(value)
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


def _drain_stderr(
    pipe: BinaryIO,
    captured: bytearray,
    failures: list[BaseException],
    stop_reading: threading.Event,
) -> None:
    try:
        while not stop_reading.is_set():
            chunk = pipe.read(OUTLINE_READ_CHUNK_BYTES)
            if not chunk:
                return
            available = MAX_SUBPROCESS_DIAGNOSTIC_BYTES - len(captured)
            if available > 0:
                captured.extend(chunk[:available])
    except (OSError, ValueError) as error:
        if not stop_reading.is_set():
            failures.append(error)


def _popen_process_group_options() -> dict[str, Any]:
    if os.name == "posix":
        return {"start_new_session": True}
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {}


def _terminate_and_reap(process: subprocess.Popen[Any]) -> None:
    """Terminate the subprocess group, escalating if its graceful window expires."""
    if process.poll() is not None:
        process.wait()
        return

    signaled_group = False
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
            signaled_group = True
        except OSError:
            pass
    elif os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            signaled_group = True
        except OSError:
            pass
    if not signaled_group:
        try:
            process.terminate()
        except OSError:
            pass

    try:
        process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass

    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
    try:
        process.kill()
    except OSError:
        pass
    try:
        process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Command process could not be reaped after termination") from error


def _validate_and_count_outline_document(document: dict[str, Any], *, record_number: int) -> int:
    raw_path = document.get("path")
    language = document.get("language")
    items = document.get("items")
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError(f"ast-grep outline record {record_number} has no path")
    if not isinstance(language, str):
        raise RuntimeError(f"ast-grep outline record {record_number} has no language")
    if not isinstance(items, list):
        raise RuntimeError(f"ast-grep outline record {record_number} has no items list")

    count = 0
    remaining: list[Any] = list(reversed(items))
    while remaining:
        node = remaining.pop()
        if not isinstance(node, dict):
            raise RuntimeError(f"ast-grep outline record {record_number} contains a non-object node")
        count += 1
        if "members" not in node:
            continue
        members = node["members"]
        if not isinstance(members, list):
            raise RuntimeError(f"ast-grep outline record {record_number} contains a non-list members field")
        remaining.extend(reversed(members))
    return count


def _parse_outline_record(raw_record: bytes, *, record_number: int) -> tuple[dict[str, Any], int] | None:
    if len(raw_record) > MAX_OUTLINE_RECORD_BYTES:
        raise RuntimeError(f"ast-grep outline record exceeds the {MAX_OUTLINE_RECORD_BYTES // 1024} KiB limit (record {record_number})")
    stripped = raw_record.strip()
    if not stripped:
        return None
    try:
        value = json.loads(stripped.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise RuntimeError(f"ast-grep outline emitted invalid JSON in record {record_number}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"ast-grep outline emitted a non-object JSON record {record_number}")
    return value, _validate_and_count_outline_document(value, record_number=record_number)


def run_outline_process(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    working_directory: Path,
    node_limit: int,
    popen_factory: PopenFactory = subprocess.Popen,
) -> OutlineProcessResult:
    """Stream bounded ast-grep outline records and stop after observing node limit + 1."""
    if node_limit < 1:
        raise ValueError("Outline node limit must be positive")
    launch_environment = dict(os.environ)
    validate_process_budget(command, environment=launch_environment)
    try:
        process = popen_factory(
            list(command),
            stdin=subprocess.DEVNULL,
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

    if process.stdout is None or process.stderr is None:  # pragma: no cover - guaranteed by PIPE
        _terminate_and_reap(process)
        raise RuntimeError("Command pipes were not created")

    stop_reading = threading.Event()
    stdout_events: queue.Queue[PipeEvent] = queue.Queue(maxsize=4)
    captured_stderr = bytearray()
    stderr_failures: list[BaseException] = []
    stdout_thread = threading.Thread(
        target=_queue_pipe_chunks,
        args=(process.stdout, stdout_events, stop_reading),
        name="ast-grep-outline-stdout",
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_drain_stderr,
        args=(process.stderr, captured_stderr, stderr_failures, stop_reading),
        name="ast-grep-outline-stderr",
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    documents: list[dict[str, Any]] = []
    record_buffer = bytearray()
    record_number = 0
    aggregate_bytes = 0
    observed_nodes = 0
    observed_extra = False
    terminated_for_limit = False
    deadline = time.monotonic() + timeout_seconds

    def consume_record(raw_record: bytes) -> None:
        nonlocal observed_extra, observed_nodes, record_number
        record_number += 1
        parsed = _parse_outline_record(raw_record, record_number=record_number)
        if parsed is None:
            return
        document, node_count = parsed
        documents.append(document)
        observed_nodes += node_count
        if observed_nodes >= node_limit + 1:
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
                raise RuntimeError(f"Could not read ast-grep outline output: {event.error}") from event.error
            if isinstance(event, _PipeEof):
                stdout_finished = True
                break

            aggregate_bytes += len(event)
            if aggregate_bytes > MAX_OUTLINE_AGGREGATE_BYTES:
                raise RuntimeError(
                    f"ast-grep outline output exceeds the {MAX_OUTLINE_AGGREGATE_BYTES // (1024 * 1024)} MiB aggregate limit"
                )
            record_buffer.extend(event)
            newline_index = record_buffer.find(b"\n")
            while newline_index >= 0:
                raw_record = bytes(record_buffer[:newline_index])
                del record_buffer[: newline_index + 1]
                consume_record(raw_record)
                if observed_extra:
                    break
                newline_index = record_buffer.find(b"\n")
            if not observed_extra and len(record_buffer) > MAX_OUTLINE_RECORD_BYTES:
                raise RuntimeError(f"ast-grep outline record exceeds the {MAX_OUTLINE_RECORD_BYTES // 1024} KiB limit")

        if not observed_extra and record_buffer:
            consume_record(bytes(record_buffer))

        if observed_extra:
            if process.poll() is None:
                terminated_for_limit = True
                _terminate_and_reap(process)
            else:
                process.wait()
        else:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise RuntimeError(f"Command timed out after {timeout_seconds:g} seconds")
            try:
                process.wait(timeout=remaining_seconds)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"Command timed out after {timeout_seconds:g} seconds") from error
    finally:
        if process.poll() is None:
            _terminate_and_reap(process)
        stop_reading.set()
        process.stdout.close()
        process.stderr.close()
        stdout_thread.join(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
        stderr_thread.join(timeout=PROCESS_TERMINATION_GRACE_SECONDS)

    if stdout_thread.is_alive() or stderr_thread.is_alive():
        raise RuntimeError("Command pipe readers did not stop after process exit")
    if stderr_failures:
        raise RuntimeError(f"Could not read ast-grep outline diagnostics: {stderr_failures[0]}")
    stderr_text = captured_stderr.decode("utf-8", errors="replace")
    if not terminated_for_limit and process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {_bounded_error_text(stderr_text)}")
    residual_stderr = _residual_stderr(stderr_text)
    if residual_stderr:
        raise RuntimeError(f"ast-grep outline failed: {_bounded_error_text(residual_stderr)}")
    return documents, observed_extra


def format_matches_as_text(matches: Sequence[Mapping[str, Any]]) -> str:
    output_blocks: list[str] = []
    for match in matches:
        file_path = str(match.get("file", ""))
        range_value = match.get("range")
        range_mapping = range_value if isinstance(range_value, Mapping) else {}
        start_value = range_mapping.get("start")
        end_value = range_mapping.get("end")
        start_mapping = start_value if isinstance(start_value, Mapping) else {}
        end_mapping = end_value if isinstance(end_value, Mapping) else {}
        start_line = int(start_mapping.get("line", 0)) + 1
        end_line = int(end_mapping.get("line", start_line - 1)) + 1
        match_text = str(match.get("text", "")).rstrip()
        header = f"{file_path}:{start_line}" if start_line == end_line else f"{file_path}:{start_line}-{end_line}"
        output_blocks.append(f"{header}\n{match_text}")
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
            lines.extend(_format_outline_nodes(members, depth=depth + 1))
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
    ) -> CompletedTextProcess:
        with ExitStack() as stack:
            config_path = self.runtime.config_path
            if config_path is None:
                temporary_directory = Path(stack.enter_context(TemporaryDirectory(prefix="ast-grep-mcp-")))
                config_path = temporary_directory / "sgconfig.yml"
                config_path.write_text(NEUTRAL_AST_GREP_CONFIG, encoding="utf-8")

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
            )
        # Verified against 0.45.0: `run` exits 1 with empty stderr when nothing
        # matches, and 8 for an invalid pattern. `scan` exits 1 when an
        # error-severity rule matched, streaming findings on stdout. But a
        # per-path failure ("ERROR: <path>: No such file or directory") leaves
        # exit 0 with partial results on stdout, so the exit code cannot
        # distinguish a partial scan from a complete one. Residual stderr can.
        if not allow_stderr_on_no_matches:
            residual = _residual_stderr(result.stderr)
            if residual:
                raise RuntimeError(f"ast-grep search failed: {_bounded_error_text(residual)}")
        return result

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
            if not _is_within(resolved, project):
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
            if not _is_within(resolved, project):
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
            arguments.extend(["--globs", glob])
        for glob in exclude_globs or ():
            if not glob or "\0" in glob or glob.startswith("!"):
                raise ValueError(f"Invalid exclude glob: {glob!r}")
            arguments.extend(["--globs", f"!{glob}"])
        return arguments

    @staticmethod
    def _contained_relative_path(raw_path: str, project: Path, *, noun: str) -> str:
        """Resolve an ast-grep-reported path and prove it stayed inside the project."""
        file_path = Path(raw_path)
        if not file_path.is_absolute():
            file_path = project / file_path
        try:
            resolved_file = file_path.resolve(strict=True)
        except FileNotFoundError as error:
            raise RuntimeError(f"ast-grep returned {noun} that no longer exists: {raw_path}") from error
        if not _is_within(resolved_file, project):
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
            *(str(path) for path in search_paths),
        ]
        result = self._run(
            "scan",
            arguments,
            working_directory=project,
            allow_no_matches=True,
        )
        parsed_matches = parse_stream_matches(result.stdout)
        truncated = len(parsed_matches) > limit
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
                kept_members, member_count = cls._trim_outline_nodes(members, remaining - consumed)
                normalized["members"] = kept_members
                consumed += member_count
            kept.append(normalized)
        return kept, consumed

    def _bound_outline(
        self,
        documents: Sequence[dict[str, Any]],
        project: Path,
        limit: int,
        *,
        observed_extra: bool,
    ) -> OutlineResults:
        """Trim validated outline hierarchy to the configured recursive node limit."""
        files: list[OutlineFile] = []
        remaining = limit
        returned = 0
        truncated = observed_extra
        for record_number, document in enumerate(documents, start=1):
            node_count = _validate_and_count_outline_document(document, record_number=record_number)
            items = document.get("items")
            raw_path = document.get("path")
            language = document.get("language")
            if not isinstance(items, list) or not isinstance(raw_path, str) or not isinstance(language, str):
                raise RuntimeError(f"ast-grep outline record {record_number} failed validation")
            kept, consumed = self._trim_outline_nodes(items, remaining)
            if consumed < node_count:
                truncated = True
            if kept or not items:
                files.append(
                    {
                        "file": self._contained_relative_path(raw_path, project, noun="an outline path"),
                        "language": language,
                        "items": kept,
                    }
                )
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
    ) -> OutlineResults:
        if language is not None and not language:
            raise ValueError("language cannot be empty when provided")
        project = self._resolve_project(project_folder)
        outline_paths = self._resolve_outline_paths(project, paths)
        limit = self._result_limit(max_results)
        arguments = [
            # The `=` is mandatory for this flag; `--json stream` is rejected.
            "--json=stream",
            "--threads",
            "1",
            *(["--lang", language] if language is not None else []),
            *(str(path) for path in outline_paths),
        ]
        with ExitStack() as stack:
            config_path = self.runtime.config_path
            if config_path is None:
                temporary_directory = Path(stack.enter_context(TemporaryDirectory(prefix="ast-grep-mcp-")))
                config_path = temporary_directory / "sgconfig.yml"
                config_path.write_text(NEUTRAL_AST_GREP_CONFIG, encoding="utf-8")
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
        return self._bound_outline(documents, project, limit, observed_extra=observed_extra)

    def dump_syntax_tree(self, *, code: str, language: str, format: DumpFormat) -> str:
        self._require_language(language)
        # `run` with no path scans the process working directory, which may lie
        # outside the allowed roots; an empty sandbox keeps the probe contained.
        with TemporaryDirectory(prefix="ast-grep-mcp-dump-") as sandbox:
            result = self._run(
                "run",
                ["--pattern", code, "--lang", language, f"--debug-query={format}"],
                working_directory=Path(sandbox),
                allow_no_matches=True,
                allow_stderr_on_no_matches=True,
            )
        return result.stderr.strip()

    def test_match_code_rule(self, *, code: str, rule_yaml: str) -> list[dict[str, Any]]:
        validate_rule_yaml(rule_yaml, forbid_regex_rules=self.runtime.forbid_regex_rules)
        result = self._run(
            "scan",
            [
                "--inline-rules",
                rule_yaml,
                "--json=stream",
                "--max-results",
                str(self.runtime.max_results_cap),
                "--stdin",
            ],
            input_text=code,
            allow_no_matches=True,
        )
        return parse_stream_matches(result.stdout)

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
    ) -> SearchResults:
        self._require_language(language)
        rule_yaml = yaml_parser.safe_dump(
            {
                "id": "mcp-pattern-search",
                "language": language,
                "rule": {"pattern": pattern},
            },
            sort_keys=False,
        )
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

    def get_server_info(self) -> ServerInfo:
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
            Field(description="Syntax dump format: pattern, ast, or cst"),
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
        )
        # The Annotated metadata declares the wire output schema. The SDK validates
        # this result's structured content against it and forwards the result
        # unchanged; a mismatch surfaces as an error result, not a protocol fault.
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
        # The Annotated metadata declares the wire output schema. The SDK validates
        # this result's structured content against it and forwards the result
        # unchanged; a mismatch surfaces as an error result, not a protocol fault.
        return search_tool_result(results, output_format)

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
        )
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    configure_runtime(runtime)
    mcp.run()


if __name__ == "__main__":
    run_mcp_server()
