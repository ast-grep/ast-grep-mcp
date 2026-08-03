from __future__ import annotations

import json
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from main import (
    MAX_OUTLINE_AGGREGATE_BYTES,
    MAX_OUTLINE_PATHS,
    MAX_OUTLINE_RECORD_BYTES,
    POSIX_ARG_HEADROOM_BYTES,
    SUPPORTED_AST_GREP_VERSION,
    WINDOWS_CREATE_PROCESS_LIMIT,
    AstGrepService,
    OutlineResults,
    ResolvedExecutable,
    ServerRuntime,
    _requires_node,
    build_argument_parser,
    build_runtime,
    format_matches_as_text,
    format_outline_results,
    format_search_results,
    outline_tool_result,
    parse_stream_matches,
    resolve_ast_grep_executable,
    run_mcp_server,
    run_outline_process,
    run_process,
    validate_process_budget,
    validate_rule_yaml,
)


class RecordingRunner:
    def __init__(self, *, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append((arguments, kwargs))
        return subprocess.CompletedProcess(arguments, self.returncode, self.stdout, self.stderr)


class RecordingOutlineRunner:
    def __init__(self, documents: list[dict[str, Any]], *, observed_extra: bool = False) -> None:
        self.documents = documents
        self.observed_extra = observed_extra
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, command: Sequence[str], **kwargs: Any) -> tuple[list[dict[str, Any]], bool]:
        self.calls.append((list(command), kwargs))
        return self.documents, self.observed_extra


def file_stdout_command(payload_path: Path) -> list[str]:
    return [
        sys.executable,
        "-c",
        "import pathlib, sys; sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())",
        str(payload_path),
    ]


def descendant_listener_program() -> str:
    return """
import pathlib
import socket
import sys
import time

listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.listen()
pathlib.Path(sys.argv[1]).write_text(str(listener.getsockname()[1]), encoding="utf-8")
time.sleep(30)
"""


def assert_descendant_listener_stopped(port_path: Path) -> None:
    port = int(port_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                pass
        except OSError:
            return
        time.sleep(0.05)
    pytest.fail(f"Descendant process still listens on TCP port {port}")


def make_runtime(
    root: Path,
    *,
    default_max_results: int = 50,
    max_results_cap: int = 500,
    forbid_regex_rules: bool = False,
) -> ServerRuntime:
    executable = root / "ast-grep-test-executable"
    executable.write_text("test executable", encoding="utf-8")
    executable.chmod(0o755)
    return ServerRuntime(
        working_directory=root,
        executable=ResolvedExecutable(path=executable, command_prefix=(str(executable),)),
        ast_grep_version=SUPPORTED_AST_GREP_VERSION,
        config_path=None,
        allowed_roots=(root,),
        command_timeout_seconds=2.0,
        default_max_results=default_max_results,
        max_results_cap=max_results_cap,
        forbid_regex_rules=forbid_regex_rules,
    )


def version_runner(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(arguments, 0, f"ast-grep {SUPPORTED_AST_GREP_VERSION}\n", "")


def test_build_runtime_resolves_config_roots_and_version(tmp_path: Path) -> None:
    executable = tmp_path / "ast-grep"
    executable.write_text("test executable", encoding="utf-8")
    executable.chmod(0o755)
    config = tmp_path / "sgconfig.yml"
    config.write_text("ruleDirs: []\n", encoding="utf-8")

    runtime = build_runtime(
        working_directory=tmp_path,
        ast_grep_executable=str(executable),
        config_path="sgconfig.yml",
        allowed_roots=[str(tmp_path), str(tmp_path)],
        command_timeout_seconds=5,
        default_max_results=25,
        max_results_cap=100,
        forbid_regex_rules=True,
        runner=version_runner,
    )

    assert runtime.config_path == config
    assert runtime.allowed_roots == (tmp_path,)
    assert runtime.ast_grep_version == SUPPORTED_AST_GREP_VERSION
    assert runtime.default_max_results == 25
    assert runtime.max_results_cap == 100
    assert runtime.forbid_regex_rules is True


def test_build_runtime_rejects_missing_executable(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        build_runtime(
            working_directory=tmp_path,
            ast_grep_executable=str(tmp_path / "missing"),
        )


def test_build_runtime_rejects_wrong_executable(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="is not ast-grep"):
        build_runtime(
            working_directory=tmp_path,
            ast_grep_executable=sys.executable,
        )


def test_build_runtime_rejects_ast_grep_version_drift(tmp_path: Path) -> None:
    executable = tmp_path / "ast-grep"
    executable.write_text("test executable", encoding="utf-8")
    executable.chmod(0o755)

    def drifted_version_runner(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        # A version that can never be the supported one, so this stays a drift test
        # whatever SUPPORTED_AST_GREP_VERSION becomes.
        return subprocess.CompletedProcess(arguments, 0, "ast-grep 0.0.0\n", "")

    with pytest.raises(ValueError, match=f"expected {re.escape(SUPPORTED_AST_GREP_VERSION)}"):
        build_runtime(
            working_directory=tmp_path,
            ast_grep_executable=str(executable),
            runner=drifted_version_runner,
        )


def test_build_runtime_rejects_config_outside_allowed_root(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    config = tmp_path / "outside.yml"
    config.write_text("ruleDirs: []\n", encoding="utf-8")
    executable = allowed / "ast-grep"
    executable.write_text("test executable", encoding="utf-8")
    executable.chmod(0o755)

    with pytest.raises(ValueError, match="Config path resolves outside"):
        build_runtime(
            working_directory=allowed,
            ast_grep_executable=str(executable),
            config_path=str(config),
            allowed_roots=[str(allowed)],
            runner=version_runner,
        )


@pytest.mark.parametrize(
    ("default_limit", "cap"),
    [
        pytest.param(0, 500, id="zero-default"),
        pytest.param(51, 50, id="default-over-cap"),
        pytest.param(1, 0, id="zero-cap"),
        pytest.param(1, 501, id="cap-over-hard-limit"),
    ],
)
def test_build_runtime_rejects_invalid_limits(tmp_path: Path, default_limit: int, cap: int) -> None:
    executable = tmp_path / "ast-grep"
    executable.write_text("test executable", encoding="utf-8")
    executable.chmod(0o755)

    with pytest.raises(ValueError):
        build_runtime(
            working_directory=tmp_path,
            ast_grep_executable=str(executable),
            default_max_results=default_limit,
            max_results_cap=cap,
            runner=version_runner,
        )


@pytest.mark.parametrize(
    ("environment_name", "option"),
    [
        pytest.param("AST_GREP_COMMAND_TIMEOUT", "--command-timeout", id="timeout"),
        pytest.param("AST_GREP_DEFAULT_MAX_RESULTS", "--default-max-results", id="default-limit"),
        pytest.param("AST_GREP_MAX_RESULTS_CAP", "--max-results-cap", id="limit-cap"),
    ],
)
def test_argument_parser_reports_invalid_numeric_environment(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    environment_name: str,
    option: str,
) -> None:
    monkeypatch.setenv(environment_name, "")

    with pytest.raises(SystemExit) as error:
        build_argument_parser().parse_args([])

    assert error.value.code == 2
    assert f"argument {option}: invalid" in capsys.readouterr().err


def test_argument_parser_coerces_numeric_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AST_GREP_COMMAND_TIMEOUT", "1.5")
    monkeypatch.setenv("AST_GREP_DEFAULT_MAX_RESULTS", "25")
    monkeypatch.setenv("AST_GREP_MAX_RESULTS_CAP", "100")

    arguments = build_argument_parser().parse_args([])

    assert arguments.command_timeout == 1.5
    assert arguments.default_max_results == 25
    assert arguments.max_results_cap == 100


def test_argument_parser_has_no_transport_or_port_surface() -> None:
    arguments = build_argument_parser().parse_args([])

    assert not hasattr(arguments, "transport")
    assert not hasattr(arguments, "port")


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param(["--transport", "stdio"], id="stdio-selector"),
        pytest.param(["--transport", "sse"], id="sse"),
        pytest.param(["--transport", "streamable-http"], id="streamable-http"),
        pytest.param(["--port", "3101"], id="port"),
    ],
)
def test_argument_parser_rejects_removed_transport_surface(
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        build_argument_parser().parse_args(arguments)

    assert error.value.code == 2
    # One read only: capsys.readouterr() drains the buffer.
    stderr = capsys.readouterr().err
    assert "unrecognized arguments" in stderr


def test_requires_node_closes_executable(mocker: Any) -> None:
    handle = mocker.MagicMock()
    handle.__enter__.return_value = handle
    handle.readline.return_value = b"#!/usr/bin/env node\n"
    open_executable = mocker.patch.object(Path, "open", return_value=handle)

    assert _requires_node(Path("ast-grep")) is True
    open_executable.assert_called_once_with("rb")
    handle.__exit__.assert_called_once_with(None, None, None)


def test_resolve_ast_grep_executable_uses_node_for_npm_shim(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for this executable-resolution test")

    package = tmp_path / "node_modules" / "@ast-grep" / "cli"
    package.mkdir(parents=True)
    target = package / "bin" / "ast-grep.js"
    target.parent.mkdir()
    target.write_text("console.log('ast-grep')\n", encoding="utf-8")
    (package / "package.json").write_text(
        json.dumps({"name": "@ast-grep/cli", "bin": {"ast-grep": "bin/ast-grep.js"}}),
        encoding="utf-8",
    )
    shim = tmp_path / "node_modules" / ".bin" / "ast-grep"
    shim.parent.mkdir(parents=True)
    shim.write_text("shim", encoding="utf-8")

    resolved = resolve_ast_grep_executable(str(shim), working_directory=tmp_path)

    assert resolved.path == target
    assert resolved.command_prefix == (str(Path(node).resolve()), str(target))


def test_resolve_ast_grep_executable_uses_node_for_global_windows_npm_shim(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for this executable-resolution test")

    package = tmp_path / "node_modules" / "@ast-grep" / "cli"
    package.mkdir(parents=True)
    target = package / "ast-grep.js"
    target.write_text("console.log('ast-grep')\n", encoding="utf-8")
    (package / "package.json").write_text(
        json.dumps({"name": "@ast-grep/cli", "bin": "ast-grep.js"}),
        encoding="utf-8",
    )
    shim = tmp_path / "ast-grep.cmd"
    shim.write_text("@echo off\n", encoding="utf-8")

    resolved = resolve_ast_grep_executable(str(shim), working_directory=tmp_path)

    assert resolved.path == target
    assert resolved.command_prefix == (str(Path(node).resolve()), str(target))


def test_resolve_ast_grep_executable_runs_native_npm_target_directly(tmp_path: Path) -> None:
    package = tmp_path / "node_modules" / "@ast-grep" / "cli"
    package.mkdir(parents=True)
    target = package / "ast-grep"
    target.write_bytes(b"\x7fELF test binary")
    target.chmod(0o755)
    (package / "package.json").write_text(
        json.dumps({"name": "@ast-grep/cli", "bin": "ast-grep"}),
        encoding="utf-8",
    )
    shim = tmp_path / "node_modules" / ".bin" / "ast-grep"
    shim.parent.mkdir(parents=True)
    shim.write_text("shim", encoding="utf-8")

    resolved = resolve_ast_grep_executable(str(shim), working_directory=tmp_path)

    assert resolved.path == target
    assert resolved.command_prefix == (str(target),)


@pytest.mark.parametrize("suffix", [".bat", ".cmd"])
def test_resolve_ast_grep_executable_rejects_arbitrary_batch_files(tmp_path: Path, suffix: str) -> None:
    launcher = tmp_path / f"ast-grep{suffix}"
    launcher.write_text("@echo off\n", encoding="utf-8")
    launcher.chmod(0o755)

    with pytest.raises(ValueError, match="Batch-file ast-grep launchers"):
        resolve_ast_grep_executable(str(launcher), working_directory=tmp_path)


def test_windows_complete_command_budget_accepts_boundary_and_rejects_one_more_character() -> None:
    # `x <argument>\0` is three UTF-16 code units beyond the argument itself.
    boundary_argument = "a" * (WINDOWS_CREATE_PROCESS_LIMIT - 3)

    validate_process_budget(["x", boundary_argument], environment={}, platform_name="nt")

    with pytest.raises(ValueError, match="CreateProcessW"):
        validate_process_budget(["x", boundary_argument + "a"], environment={}, platform_name="nt")


def test_windows_environment_block_budget_accepts_boundary_and_rejects_one_more_character() -> None:
    # The environment block is `K=<value>\0\0`, four UTF-16 units beyond the value.
    boundary_value = "a" * (WINDOWS_CREATE_PROCESS_LIMIT - 4)

    validate_process_budget(["x"], environment={"K": boundary_value}, platform_name="nt")

    with pytest.raises(ValueError, match="Environment block"):
        validate_process_budget(["x"], environment={"K": boundary_value + "a"}, platform_name="nt")


def test_windows_command_budget_counts_non_bmp_characters_as_two_utf16_units() -> None:
    command = ["x", "😀" * ((WINDOWS_CREATE_PROCESS_LIMIT - 3) // 2)]

    validate_process_budget(command, environment={}, platform_name="nt")

    with pytest.raises(ValueError, match="CreateProcessW"):
        validate_process_budget([*command, "😀"], environment={}, platform_name="nt")


def test_posix_complete_process_budget_accepts_boundary_and_rejects_one_less_byte() -> None:
    command = ["ast-grep", "scan"]
    environment = {"LANG": "C"}
    argv_bytes = sum(len(os.fsencode(argument)) + 1 for argument in command)
    environment_bytes = sum(len(os.fsencode(key)) + len(os.fsencode(value)) + 2 for key, value in environment.items())
    pointer_bytes = (len(command) + len(environment) + 2) * struct.calcsize("P")
    exact_arg_max = argv_bytes + environment_bytes + pointer_bytes + POSIX_ARG_HEADROOM_BYTES

    validate_process_budget(
        command,
        environment=environment,
        platform_name="posix",
        arg_max=exact_arg_max,
    )

    with pytest.raises(ValueError, match="POSIX ARG_MAX"):
        validate_process_budget(
            command,
            environment=environment,
            platform_name="posix",
            arg_max=exact_arg_max - 1,
        )


@pytest.mark.parametrize("command", [["tool.cmd"], ["tool.bat"]])
def test_process_budget_rejects_batch_commands_on_every_platform(command: list[str]) -> None:
    with pytest.raises(ValueError, match="Batch-file commands"):
        validate_process_budget(command, environment={}, platform_name="nt")


def test_run_process_reports_timeout() -> None:
    def timeout_runner(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(arguments, timeout=1)

    with pytest.raises(RuntimeError, match="timed out after 1 seconds"):
        run_process(["ast-grep", "--version"], timeout_seconds=1, runner=timeout_runner)


def test_run_process_real_timeout_reaps_the_child(tmp_path: Path) -> None:
    pid_path = tmp_path / "child.pid"
    descendant_port_path = tmp_path / "descendant.port"
    child_program = """
import os
import pathlib
import subprocess
import sys
import time

subprocess.Popen([sys.executable, "-c", sys.argv[3], sys.argv[2]])
deadline = time.monotonic() + 5
while not pathlib.Path(sys.argv[2]).exists() and time.monotonic() < deadline:
    time.sleep(0.01)
pathlib.Path(sys.argv[1]).write_text(str(os.getpid()), encoding="utf-8")
time.sleep(30)
"""
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="timed out"):
        run_process(
            [
                sys.executable,
                "-c",
                child_program,
                str(pid_path),
                str(descendant_port_path),
                descendant_listener_program(),
            ],
            timeout_seconds=0.5,
        )

    assert time.monotonic() - started < 5
    child_pid = int(pid_path.read_text(encoding="utf-8"))
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    assert_descendant_listener_stopped(descendant_port_path)


def test_run_process_reports_os_error() -> None:
    def os_error_runner(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise OSError(7, "Argument list too long")

    with pytest.raises(RuntimeError, match="could not be executed"):
        run_process(["ast-grep", "--version"], timeout_seconds=1, runner=os_error_runner)


def test_run_mcp_server_rejects_malformed_boolean_environment(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("AST_GREP_FORBID_REGEX_RULES", "maybe")
    monkeypatch.setattr(sys, "argv", ["ast-grep-server"])

    with pytest.raises(SystemExit) as error:
        run_mcp_server()

    assert error.value.code == 2
    assert "AST_GREP_FORBID_REGEX_RULES must be a boolean value" in capsys.readouterr().err


def test_parse_stream_matches_rejects_non_json_output() -> None:
    with pytest.raises(RuntimeError, match="invalid JSON"):
        parse_stream_matches("not json")


def test_parse_stream_matches_rejects_non_object_json_line() -> None:
    with pytest.raises(RuntimeError, match="non-object"):
        parse_stream_matches('{"file": "a.py"}\n[1, 2]\n')


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        pytest.param(b"{\n", "invalid JSON", id="malformed"),
        pytest.param(b'{"path":"a.py"', "invalid JSON", id="truncated-at-eof"),
        pytest.param(b"[]\n", "non-object JSON", id="non-object-record"),
        pytest.param(
            b'{"path":"a.py","language":"Python","items":[1]}\n',
            "non-object node",
            id="non-object-node",
        ),
        pytest.param(
            b'{"path":"a.py","language":"Python","items":[{"members":{}}]}\n',
            "non-list members",
            id="invalid-members",
        ),
    ],
)
def test_outline_stream_rejects_malformed_or_invalid_ndjson(
    tmp_path: Path,
    payload: bytes,
    message: str,
) -> None:
    payload_path = tmp_path / "payload.ndjson"
    payload_path.write_bytes(payload)

    with pytest.raises(RuntimeError, match=message):
        run_outline_process(
            file_stdout_command(payload_path),
            timeout_seconds=2,
            working_directory=tmp_path,
            node_limit=10,
        )


def test_outline_stream_rejects_record_over_one_mib(tmp_path: Path) -> None:
    payload_path = tmp_path / "oversized.ndjson"
    payload_path.write_bytes(b"x" * (MAX_OUTLINE_RECORD_BYTES + 1) + b"\n")

    with pytest.raises(RuntimeError, match="record exceeds the 1024 KiB limit"):
        run_outline_process(
            file_stdout_command(payload_path),
            timeout_seconds=2,
            working_directory=tmp_path,
            node_limit=10,
        )


def test_outline_stream_rejects_aggregate_over_four_mib(tmp_path: Path) -> None:
    padding = "x" * (900 * 1024)
    records = [json.dumps({"path": f"file-{index}.py", "language": "Python", "items": [], "padding": padding}) for index in range(5)]
    payload_path = tmp_path / "aggregate.ndjson"
    payload_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    assert payload_path.stat().st_size > MAX_OUTLINE_AGGREGATE_BYTES

    with pytest.raises(RuntimeError, match="4 MiB aggregate limit"):
        run_outline_process(
            file_stdout_command(payload_path),
            timeout_seconds=5,
            working_directory=tmp_path,
            node_limit=10,
        )


def test_outline_stream_reports_nonzero_exit_code(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stderr.write('outline failed'); raise SystemExit(7)",
    ]

    with pytest.raises(RuntimeError, match="outline failed"):
        run_outline_process(
            command,
            timeout_seconds=2,
            working_directory=tmp_path,
            node_limit=10,
        )


def test_outline_stream_real_timeout_reaps_the_child(tmp_path: Path) -> None:
    pid_path = tmp_path / "outline.pid"
    descendant_port_path = tmp_path / "outline-descendant.port"
    child_program = """
import os
import pathlib
import subprocess
import sys
import time

subprocess.Popen([sys.executable, "-c", sys.argv[3], sys.argv[2]])
deadline = time.monotonic() + 5
while not pathlib.Path(sys.argv[2]).exists() and time.monotonic() < deadline:
    time.sleep(0.01)
pathlib.Path(sys.argv[1]).write_text(str(os.getpid()), encoding="utf-8")
time.sleep(30)
"""
    command = [
        sys.executable,
        "-c",
        child_program,
        str(pid_path),
        str(descendant_port_path),
        descendant_listener_program(),
    ]

    with pytest.raises(RuntimeError, match="timed out"):
        run_outline_process(
            command,
            timeout_seconds=0.5,
            working_directory=tmp_path,
            node_limit=10,
        )

    child_pid = int(pid_path.read_text(encoding="utf-8"))
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    assert_descendant_listener_stopped(descendant_port_path)


def test_outline_stream_stops_at_limit_plus_one_and_reaps_the_child(tmp_path: Path) -> None:
    payload_path = tmp_path / "outline.ndjson"
    payload_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "path": f"file-{index}.py",
                        "language": "Python",
                        "items": [{"name": f"node-{index}"}],
                    }
                )
                for index in range(2)
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pid_path = tmp_path / "outline.pid"
    descendant_port_path = tmp_path / "outline-descendant.port"
    child_program = """
import os
import pathlib
import subprocess
import sys
import time

subprocess.Popen([sys.executable, "-c", sys.argv[4], sys.argv[3]])
deadline = time.monotonic() + 5
while not pathlib.Path(sys.argv[3]).exists() and time.monotonic() < deadline:
    time.sleep(0.01)
pathlib.Path(sys.argv[2]).write_text(str(os.getpid()), encoding="utf-8")
sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())
sys.stdout.buffer.flush()
time.sleep(30)
"""
    command = [
        sys.executable,
        "-c",
        child_program,
        str(payload_path),
        str(pid_path),
        str(descendant_port_path),
        descendant_listener_program(),
    ]

    started = time.monotonic()
    documents, observed_extra = run_outline_process(
        command,
        timeout_seconds=5,
        working_directory=tmp_path,
        node_limit=1,
    )

    assert time.monotonic() - started < 5
    assert len(documents) == 2
    assert observed_extra is True
    child_pid = int(pid_path.read_text(encoding="utf-8"))
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    assert_descendant_listener_stopped(descendant_port_path)


def test_validate_rule_yaml_accepts_structural_rule_and_negative_probe_shape() -> None:
    validate_rule_yaml(
        "id: calls\nlanguage: python\nrule:\n  pattern: print($A)\n",
        forbid_regex_rules=True,
    )


def test_validate_rule_yaml_rejects_regex_matcher_when_policy_enabled() -> None:
    with pytest.raises(ValueError, match="Regex ast-grep rules are disabled"):
        validate_rule_yaml(
            "id: text\nlanguage: python\nrule:\n  regex: secret.*\n",
            forbid_regex_rules=True,
        )


@pytest.mark.parametrize(
    "rule_yaml",
    [
        pytest.param(
            "id: c\nlanguage: python\nrule:\n  pattern: print($A)\nconstraints:\n  A:\n    regex: secret.*\n",
            id="metavariable-constraint",
        ),
        pytest.param(
            "id: n\nlanguage: python\nrule:\n  inside:\n    regex: secret.*\n",
            id="nested-relational-rule",
        ),
        pytest.param(
            "id: u\nlanguage: python\nrule:\n  matches: helper\nutils:\n  helper:\n    regex: secret.*\n",
            id="utils-definition",
        ),
    ],
)
def test_regex_policy_reaches_every_regex_key_not_only_top_level_matchers(rule_yaml: str) -> None:
    """The policy walks the whole document, which is broader than `rule:` matcher keys."""
    with pytest.raises(ValueError, match="Regex ast-grep rules are disabled"):
        validate_rule_yaml(rule_yaml, forbid_regex_rules=True)


def test_validate_rule_yaml_allows_regex_matcher_when_policy_disabled() -> None:
    validate_rule_yaml(
        "id: text\nlanguage: python\nrule:\n  regex: secret.*\n",
        forbid_regex_rules=False,
    )


def test_validate_rule_yaml_accepts_multiple_rule_documents() -> None:
    validate_rule_yaml(
        "id: one\nlanguage: python\nrule:\n  pattern: print($A)\n---\nid: two\nlanguage: python\nrule:\n  pattern: len($A)\n",
        forbid_regex_rules=True,
    )


@pytest.mark.parametrize(
    "rule_yaml",
    [
        pytest.param("", id="empty"),
        pytest.param("- item\n", id="not-a-mapping"),
        pytest.param("id: missing-fields\n", id="missing-fields"),
        pytest.param("id: wrong-rule\nlanguage: python\nrule: value\n", id="rule-not-mapping"),
    ],
)
def test_validate_rule_yaml_rejects_invalid_shapes(rule_yaml: str) -> None:
    with pytest.raises(ValueError):
        validate_rule_yaml(rule_yaml, forbid_regex_rules=False)


def test_validate_rule_yaml_rejects_oversized_inline_rules() -> None:
    oversized = "# " + "a" * (64 * 1024) + "\nid: x\nlanguage: python\nrule:\n  pattern: print($A)\n"

    with pytest.raises(ValueError, match="inline limit"):
        validate_rule_yaml(oversized, forbid_regex_rules=False)


@pytest.mark.parametrize("language", ["py", "js", "ts", "hcl"])
def test_dump_syntax_tree_accepts_cli_language_alias(tmp_path: Path, language: str) -> None:
    runner = RecordingRunner(stderr="Debug Pattern:\nidentifier", returncode=1)
    service = AstGrepService(make_runtime(tmp_path), runner=runner)

    result = service.dump_syntax_tree(code="$A", language=language, format="pattern")

    assert result == "Debug Pattern:\nidentifier"
    command = runner.calls[0][0]
    assert command[command.index("--lang") + 1] == language


def test_dump_syntax_tree_preserves_debug_output_when_pattern_has_no_match(tmp_path: Path) -> None:
    runner = RecordingRunner(stderr="Debug Pattern:\ncall", returncode=1)
    service = AstGrepService(make_runtime(tmp_path), runner=runner)

    assert service.dump_syntax_tree(code="missing($A)", language="python", format="pattern") == "Debug Pattern:\ncall"


def test_dump_syntax_tree_runs_inside_empty_sandbox(tmp_path: Path) -> None:
    observed: list[Path] = []

    def sandbox_runner(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        cwd = Path(kwargs["cwd"])
        observed.append(cwd)
        assert cwd != tmp_path
        assert list(cwd.iterdir()) == []
        return subprocess.CompletedProcess(arguments, 1, "", "Debug Pattern:\ncall")

    service = AstGrepService(make_runtime(tmp_path), runner=sandbox_runner)

    assert service.dump_syntax_tree(code="print($A)", language="python", format="pattern") == "Debug Pattern:\ncall"
    assert len(observed) == 1
    assert not observed[0].exists()


def test_dump_syntax_tree_removes_sandbox_and_generated_config_after_timeout(tmp_path: Path) -> None:
    observed_sandbox: Path | None = None
    observed_config: Path | None = None

    def timeout_runner(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal observed_config, observed_sandbox
        observed_sandbox = Path(kwargs["cwd"])
        observed_config = Path(arguments[arguments.index("--config") + 1])
        raise subprocess.TimeoutExpired(arguments, timeout=2)

    service = AstGrepService(make_runtime(tmp_path), runner=timeout_runner)

    with pytest.raises(RuntimeError, match="timed out"):
        service.dump_syntax_tree(code="print($A)", language="python", format="pattern")

    assert observed_sandbox is not None
    assert observed_config is not None
    assert not observed_sandbox.exists()
    assert not observed_config.parent.exists()


def test_search_rejects_project_outside_allowed_root(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    service = AstGrepService(make_runtime(allowed))

    with pytest.raises(ValueError, match="Project folder resolves outside"):
        service.find_code(
            project_folder=str(outside),
            pattern="print($A)",
            language="python",
            paths=None,
            include_globs=None,
            exclude_globs=None,
            max_results=1,
        )


def test_search_rejects_parent_path_escape(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("print('outside')\n", encoding="utf-8")
    service = AstGrepService(make_runtime(tmp_path))

    with pytest.raises(ValueError, match="outside project_folder"):
        service.find_code(
            project_folder="project",
            pattern="print($A)",
            language="python",
            paths=["../outside.py"],
            include_globs=None,
            exclude_globs=None,
            max_results=1,
        )


def test_search_rejects_absolute_search_path(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "example.py"
    source.write_text("print('value')\n", encoding="utf-8")
    service = AstGrepService(make_runtime(tmp_path))

    with pytest.raises(ValueError, match="must be relative"):
        service.find_code(
            project_folder="project",
            pattern="print($A)",
            language="python",
            paths=[str(source)],
            include_globs=None,
            exclude_globs=None,
            max_results=1,
        )


def test_search_rejects_explicit_empty_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    runner = RecordingRunner()
    service = AstGrepService(make_runtime(tmp_path), runner=runner)

    with pytest.raises(ValueError, match="at least one relative path"):
        service.find_code(
            project_folder="project",
            pattern="print($A)",
            language="python",
            paths=[],
            include_globs=None,
            exclude_globs=None,
            max_results=1,
        )

    assert runner.calls == []


def test_search_rejects_symlink_escape(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("print('outside')\n", encoding="utf-8")
    link = project / "linked.py"
    try:
        link.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"Symlinks are unavailable: {error}")
    service = AstGrepService(make_runtime(tmp_path))

    with pytest.raises(ValueError, match="outside project_folder"):
        service.find_code(
            project_folder="project",
            pattern="print($A)",
            language="python",
            paths=["linked.py"],
            include_globs=None,
            exclude_globs=None,
            max_results=1,
        )


def test_outline_requires_one_to_sixty_four_files(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    for index in range(MAX_OUTLINE_PATHS + 1):
        (project / f"file-{index}.py").write_text("pass\n", encoding="utf-8")
    outline_runner = RecordingOutlineRunner([])
    service = AstGrepService(make_runtime(tmp_path), outline_runner=outline_runner)

    with pytest.raises(ValueError, match="between 1 and 64"):
        service.outline_code(
            project_folder="project",
            paths=[],
            language=None,
            max_results=1,
        )
    with pytest.raises(ValueError, match="between 1 and 64"):
        service.outline_code(
            project_folder="project",
            paths=[f"file-{index}.py" for index in range(MAX_OUTLINE_PATHS + 1)],
            language=None,
            max_results=1,
        )

    assert outline_runner.calls == []
    result = service.outline_code(
        project_folder="project",
        paths=[f"file-{index}.py" for index in range(MAX_OUTLINE_PATHS)],
        language=None,
        max_results=1,
    )
    assert result["returned"] == 0
    assert len(outline_runner.calls) == 1


def test_outline_rejects_directory_path(tmp_path: Path) -> None:
    project = tmp_path / "project"
    directory = project / "src"
    directory.mkdir(parents=True)
    outline_runner = RecordingOutlineRunner([])
    service = AstGrepService(make_runtime(tmp_path), outline_runner=outline_runner)

    with pytest.raises(ValueError, match="regular file"):
        service.outline_code(
            project_folder="project",
            paths=["src"],
            language=None,
            max_results=1,
        )

    assert outline_runner.calls == []


def test_outline_rejects_absolute_and_parent_escape_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("pass\n", encoding="utf-8")
    outline_runner = RecordingOutlineRunner([])
    service = AstGrepService(make_runtime(tmp_path), outline_runner=outline_runner)

    with pytest.raises(ValueError, match="must be relative"):
        service.outline_code(
            project_folder="project",
            paths=[str(outside)],
            language=None,
            max_results=1,
        )
    with pytest.raises(ValueError, match="outside project_folder"):
        service.outline_code(
            project_folder="project",
            paths=["../outside.py"],
            language=None,
            max_results=1,
        )

    assert outline_runner.calls == []


def test_outline_rejects_symlink_escape(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("pass\n", encoding="utf-8")
    link = project / "linked.py"
    try:
        link.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"Symlinks are unavailable: {error}")
    outline_runner = RecordingOutlineRunner([])
    service = AstGrepService(make_runtime(tmp_path), outline_runner=outline_runner)

    with pytest.raises(ValueError, match="outside project_folder"):
        service.outline_code(
            project_folder="project",
            paths=["linked.py"],
            language=None,
            max_results=1,
        )

    assert outline_runner.calls == []


def test_outline_rejects_runtime_cap_before_launch(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "example.py").write_text("pass\n", encoding="utf-8")
    outline_runner = RecordingOutlineRunner([])
    service = AstGrepService(
        make_runtime(tmp_path, default_max_results=2, max_results_cap=3),
        outline_runner=outline_runner,
    )

    with pytest.raises(ValueError, match="between 1 and 3"):
        service.outline_code(
            project_folder="project",
            paths=["example.py"],
            language=None,
            max_results=4,
        )

    assert outline_runner.calls == []


def test_outline_counts_and_trims_nodes_recursively_in_preorder(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "example.py"
    source.write_text("pass\n", encoding="utf-8")
    document = {
        "path": str(source),
        "language": "Python",
        "items": [
            {
                "name": "outer",
                "members": [
                    {"name": "first", "members": [{"name": "nested"}]},
                    {"name": "second"},
                ],
            },
            {"name": "after"},
        ],
    }
    outline_runner = RecordingOutlineRunner([document], observed_extra=True)
    service = AstGrepService(make_runtime(tmp_path), outline_runner=outline_runner)

    result = service.outline_code(
        project_folder="project",
        paths=["example.py"],
        language=None,
        max_results=3,
    )

    assert result == {
        "files": [
            {
                "file": "example.py",
                "language": "Python",
                "items": [
                    {
                        "name": "outer",
                        "members": [{"name": "first", "members": [{"name": "nested"}]}],
                    }
                ],
            }
        ],
        "returned": 3,
        "truncated": True,
        "limit": 3,
    }
    command, kwargs = outline_runner.calls[0]
    assert command[1] == "outline"
    assert command[command.index("--threads") + 1] == "1"
    assert "--lang" not in command
    assert kwargs["node_limit"] == 3


def test_outline_forwards_optional_language_and_removes_temporary_config(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "example.custom"
    source.write_text("value\n", encoding="utf-8")
    observed_config: Path | None = None

    def outline_runner(command: Sequence[str], **kwargs: Any) -> tuple[list[dict[str, Any]], bool]:
        nonlocal observed_config
        observed_config = Path(command[command.index("--config") + 1])
        assert observed_config.exists()
        assert command[command.index("--lang") + 1] == "custom"
        return ([{"path": str(source), "language": "Custom", "items": []}], False)

    service = AstGrepService(make_runtime(tmp_path), outline_runner=outline_runner)

    result = service.outline_code(
        project_folder="project",
        paths=["example.custom"],
        language="custom",
        max_results=None,
    )

    assert result["limit"] == 50
    assert observed_config is not None
    assert not observed_config.parent.exists()


def test_outline_removes_temporary_config_when_streaming_fails(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "example.py"
    source.write_text("pass\n", encoding="utf-8")
    observed_config: Path | None = None

    def outline_runner(command: Sequence[str], **kwargs: Any) -> tuple[list[dict[str, Any]], bool]:
        nonlocal observed_config
        observed_config = Path(command[command.index("--config") + 1])
        assert observed_config.exists()
        raise RuntimeError("malformed outline")

    service = AstGrepService(make_runtime(tmp_path), outline_runner=outline_runner)

    with pytest.raises(RuntimeError, match="malformed outline"):
        service.outline_code(
            project_folder="project",
            paths=["example.py"],
            language=None,
            max_results=1,
        )

    assert observed_config is not None
    assert not observed_config.parent.exists()


def test_outline_rejects_vanished_result_file(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "example.py"
    source.write_text("pass\n", encoding="utf-8")
    ghost = project / "ghost.py"
    outline_runner = RecordingOutlineRunner([{"path": str(ghost), "language": "Python", "items": [{"name": "ghost"}]}])
    service = AstGrepService(make_runtime(tmp_path), outline_runner=outline_runner)

    with pytest.raises(RuntimeError, match="no longer exists"):
        service.outline_code(
            project_folder="project",
            paths=["example.py"],
            language=None,
            max_results=1,
        )


def test_search_uses_limit_plus_one_globs_and_relative_results(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = project / "src" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('one')\nprint('two')\n", encoding="utf-8")
    matches = [
        {
            "file": str(source),
            "text": "print('one')",
            "range": {"start": {"line": 0}, "end": {"line": 0}},
        },
        {
            "file": str(source),
            "text": "print('two')",
            "range": {"start": {"line": 1}, "end": {"line": 1}},
        },
    ]
    runner = RecordingRunner(stdout="\n".join(json.dumps(match) for match in matches))
    service = AstGrepService(make_runtime(tmp_path), runner=runner)

    results = service.find_code(
        project_folder="project",
        pattern="print($A)",
        language="python",
        paths=["src"],
        include_globs=["*.py"],
        exclude_globs=["*_test.py"],
        max_results=1,
    )

    assert results == {
        "matches": [
            {
                "file": "src/example.py",
                "text": "print('one')",
                "range": {"start": {"line": 0}, "end": {"line": 0}},
            }
        ],
        "returned": 1,
        "truncated": True,
        "limit": 1,
    }
    command = runner.calls[0][0]
    assert command[1] == "scan"
    assert command[command.index("--max-results") + 1] == "2"
    assert [command[index + 1] for index, value in enumerate(command) if value == "--globs"] == ["*.py", "!*_test.py"]
    assert command[-1] == str(source.parent)


@pytest.mark.parametrize("include_metadata", [False, True])
def test_rule_search_forwards_metadata_flag_only_when_requested(
    tmp_path: Path,
    include_metadata: bool,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "example.py"
    source.write_text("print('value')\n", encoding="utf-8")
    match = {
        "file": str(source),
        "text": "print('value')",
        "range": {"start": {"line": 0}, "end": {"line": 0}},
        "metadata": {"category": "documentation"},
    }
    runner = RecordingRunner(stdout=json.dumps(match))
    service = AstGrepService(make_runtime(tmp_path), runner=runner)

    result = service.find_code_by_rule(
        project_folder="project",
        rule_yaml=("id: print-call\nlanguage: python\nmetadata:\n  category: documentation\nrule:\n  pattern: print($A)\n"),
        paths=["example.py"],
        include_globs=None,
        exclude_globs=None,
        max_results=1,
        include_metadata=include_metadata,
    )

    command = runner.calls[0][0]
    assert ("--include-metadata" in command) is include_metadata
    assert result["matches"][0]["metadata"] == {"category": "documentation"}


def test_search_rejects_zero_and_unlimited_limits(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    service = AstGrepService(make_runtime(tmp_path))

    with pytest.raises(ValueError, match="between 1 and 500"):
        service.find_code(
            project_folder="project",
            pattern="print($A)",
            language="python",
            paths=None,
            include_globs=None,
            exclude_globs=None,
            max_results=0,
        )


def test_negative_rule_probe_returns_empty_list(tmp_path: Path) -> None:
    runner = RecordingRunner(stdout="")
    service = AstGrepService(make_runtime(tmp_path), runner=runner)

    matches = service.test_match_code_rule(
        code="value = 1",
        rule_yaml="id: no-call\nlanguage: python\nrule:\n  pattern: print($A)\n",
    )

    assert matches == []
    command = runner.calls[0][0]
    assert command[1] == "scan"
    assert "--stdin" in command
    assert command[command.index("--max-results") + 1] == "500"


def test_search_returns_error_severity_diagnostics_streamed_on_exit_one(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "example.py"
    source.write_text("print('value')\n", encoding="utf-8")
    match = {"file": str(source), "text": "print('value')", "range": {"start": {"line": 0}, "end": {"line": 0}}}
    runner = RecordingRunner(
        stdout=json.dumps(match),
        # The exact two-line stderr 0.45.0 emits for a successful error-severity scan.
        stderr="Error: 1 error(s) found in code.\nHelp: Scan succeeded and found error level diagnostics in the codebase.\n",
        returncode=1,
    )
    service = AstGrepService(make_runtime(tmp_path), runner=runner)

    results = service.find_code(
        project_folder="project",
        pattern="print($A)",
        language="python",
        paths=None,
        include_globs=None,
        exclude_globs=None,
        max_results=1,
    )

    assert results["returned"] == 1
    assert results["matches"][0]["file"] == "example.py"


def test_search_exit_one_with_error_is_not_treated_as_no_match(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    runner = RecordingRunner(stderr="invalid rule", returncode=1)
    service = AstGrepService(make_runtime(tmp_path), runner=runner)

    with pytest.raises(RuntimeError, match="search failed"):
        service.find_code(
            project_folder="project",
            pattern="print($A)",
            language="python",
            paths=None,
            include_globs=None,
            exclude_globs=None,
            max_results=1,
        )


def test_search_rejects_partial_results_when_a_path_failed(tmp_path: Path) -> None:
    """Verified against 0.45.0: a bad path leaves exit 0 with partial findings on stdout."""
    project = tmp_path / "project"
    project.mkdir()
    source = project / "example.py"
    source.write_text("print('value')\n", encoding="utf-8")
    match = {"file": str(source), "text": "print('value')", "range": {"start": {"line": 0}, "end": {"line": 0}}}
    runner = RecordingRunner(
        stdout=json.dumps(match),
        stderr="ERROR: /nonexistent-xyz: No such file or directory (os error 2)",
        returncode=0,
    )
    service = AstGrepService(make_runtime(tmp_path), runner=runner)

    with pytest.raises(RuntimeError, match="No such file or directory"):
        service.find_code(
            project_folder="project",
            pattern="print($A)",
            language="python",
            paths=None,
            include_globs=None,
            exclude_globs=None,
            max_results=1,
        )


def _outline_document(path: str, count: int) -> dict[str, Any]:
    items = [{"name": f"symbol{index}", "symbolType": "function"} for index in range(count)]
    return {"path": path, "language": "Python", "items": items}


def test_outline_does_not_report_truncation_when_the_limit_is_met_exactly(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "a.py").write_text("def a(): pass\n", encoding="utf-8")
    outline_runner = RecordingOutlineRunner([_outline_document("a.py", 2)])
    service = AstGrepService(make_runtime(tmp_path), outline_runner=outline_runner)

    result = service.outline_code(
        project_folder="project",
        language=None,
        paths=["a.py"],
        max_results=2,
    )

    assert result["returned"] == 2
    assert result["truncated"] is False


def test_outline_stops_adding_files_once_the_symbol_cap_is_reached(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "a.py").write_text("def a(): pass\n", encoding="utf-8")
    (project / "b.py").write_text("def b(): pass\n", encoding="utf-8")
    outline_runner = RecordingOutlineRunner(
        [_outline_document("a.py", 2), _outline_document("b.py", 2)],
        observed_extra=True,
    )
    service = AstGrepService(make_runtime(tmp_path), outline_runner=outline_runner)

    result = service.outline_code(
        project_folder="project",
        language=None,
        paths=["a.py", "b.py"],
        max_results=2,
    )

    assert result["returned"] == 2
    assert result["truncated"] is True
    assert [entry["file"] for entry in result["files"]] == ["a.py"]


def test_outline_rejects_a_path_outside_the_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "a.py"
    source.write_text("def a(): pass\n", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("def a(): pass\n", encoding="utf-8")
    outline_runner = RecordingOutlineRunner([_outline_document(str(outside), 1)])
    service = AstGrepService(make_runtime(tmp_path), outline_runner=outline_runner)

    with pytest.raises(RuntimeError, match="outside project_folder"):
        service.outline_code(
            project_folder="project",
            language=None,
            paths=["a.py"],
            max_results=5,
        )


def test_outline_rejects_a_document_without_items(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "a.py").write_text("def a(): pass\n", encoding="utf-8")
    outline_runner = RecordingOutlineRunner([{"path": "a.py", "language": "Python"}])
    service = AstGrepService(make_runtime(tmp_path), outline_runner=outline_runner)

    with pytest.raises(RuntimeError, match="no items list"):
        service.outline_code(
            project_folder="project",
            language=None,
            paths=["a.py"],
            max_results=5,
        )


def test_outline_rejects_an_empty_optional_language(tmp_path: Path) -> None:
    source = tmp_path / "a.py"
    source.write_text("def a(): pass\n", encoding="utf-8")
    service = AstGrepService(make_runtime(tmp_path), outline_runner=RecordingOutlineRunner([]))

    with pytest.raises(ValueError, match="cannot be empty"):
        service.outline_code(
            project_folder=str(tmp_path),
            language="",
            paths=["a.py"],
            max_results=5,
        )


def test_search_rejects_match_that_vanished_before_normalization(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    ghost = project / "ghost.py"
    match = {"file": str(ghost), "text": "print('ghost')", "range": {"start": {"line": 0}, "end": {"line": 0}}}
    runner = RecordingRunner(stdout=json.dumps(match))
    service = AstGrepService(make_runtime(tmp_path), runner=runner)

    with pytest.raises(RuntimeError, match="no longer exists"):
        service.find_code(
            project_folder="project",
            pattern="print($A)",
            language="python",
            paths=None,
            include_globs=None,
            exclude_globs=None,
            max_results=1,
        )


def test_format_matches_and_truncation_header() -> None:
    matches = [
        {
            "file": "src/example.py",
            "text": "print('one')",
            "range": {"start": {"line": 4}, "end": {"line": 4}},
        }
    ]

    assert format_matches_as_text(matches) == "src/example.py:5\nprint('one')"
    assert (
        format_search_results({"matches": matches, "returned": 1, "truncated": True, "limit": 1})
        == "Found 1 match (limit 1; additional matches exist):\n\nsrc/example.py:5\nprint('one')"
    )


def test_format_outline_results_preserves_hierarchy_and_truncation() -> None:
    results: OutlineResults = {
        "files": [
            {
                "file": "src/example.py",
                "language": "Python",
                "items": [
                    {
                        "name": "Example",
                        "signature": "class Example:",
                        "members": [{"name": "method"}],
                    }
                ],
            }
        ],
        "returned": 2,
        "truncated": True,
        "limit": 2,
    }

    assert format_outline_results(results) == (
        "Found 2 outline nodes (limit 2; additional nodes exist)\n\nsrc/example.py (Python)\n  - class Example:\n    - method"
    )
    text_result = outline_tool_result(results, "text")
    assert text_result.structured_content == results
    assert text_result.content[0].text.startswith("Found 2 outline nodes")  # type: ignore[union-attr]
    json_result = outline_tool_result(results, "json")
    assert json_result.structured_content == results
    assert json.loads(json_result.content[0].text) == results  # type: ignore[union-attr]


def test_server_info_exposes_effective_contract(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path, default_max_results=25, max_results_cap=100, forbid_regex_rules=True)

    info = AstGrepService(runtime).get_server_info()

    assert info["fork_version"] == "0.3.0"
    assert info["ast_grep_version"] == SUPPORTED_AST_GREP_VERSION
    assert info["allowed_roots"] == [str(tmp_path)]
    assert info["default_max_results"] == 25
    assert info["max_results_cap"] == 100
    assert info["forbid_regex_rules"] is True
    assert os.path.isabs(info["ast_grep_executable"])
