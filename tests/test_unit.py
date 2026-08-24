from __future__ import annotations

import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from mcp.types import TextContent
from pytest_mock import MockerFixture

from ast_soleaux.config_snapshot import ConfigSnapshot
from ast_soleaux.server import (
    JSON_OBJECT_ADAPTER,
    MAX_NATIVE_LIBRARY_BYTES,
    MAX_NDJSON_RECORD_BYTES,
    MAX_OUTLINE_PATHS,
    MAX_OUTLINE_RECORD_BYTES,
    MAX_SNIPPET_INPUT_BYTES,
    MAX_STRUCTURED_OUTPUT_BYTES,
    MAX_SUBPROCESS_DIAGNOSTIC_BYTES,
    MAX_TEST_REPORT_BYTES,
    POSIX_ARG_HEADROOM_BYTES,
    PROCESS_TERMINATION_GRACE_SECONDS,
    SERVER_INSTRUCTIONS,
    SUPPORTED_AST_GREP_VERSION,
    SUPPORTED_OXC_HELPER_VERSION,
    SUPPORTED_OXC_PARSER_VERSION,
    SUPPORTED_OXC_RESOLVER_VERSION,
    WINDOWS_CREATE_PROCESS_LIMIT,
    AstGrepService,
    JsonObject,
    JsonValue,
    OutlineResults,
    OxcVersions,
    ResolvedExecutable,
    ResultCursorStore,
    RuntimeServices,
    _parse_match_record,
    _requires_node,
    build_argument_parser,
    build_runtime,
    create_mcp,
    format_matches_as_text,
    format_outline_results,
    format_search_results,
    outline_tool_result,
    parse_stream_matches,
    resolve_ast_grep_executable,
    resolve_oxc_helper_executable,
    run_mcp_server,
    run_ndjson_process,
    run_outline_process,
    run_process,
    run_text_process,
    validate_match_document,
    validate_process_budget,
    validate_rule_yaml,
)


def invoke_boundary(target: object, **kwargs: object) -> object:
    assert callable(target)
    return target(**kwargs)


def test_server_instructions_describe_the_capability_gated_effects() -> None:
    assert "Read-only structural code inspection" not in SERVER_INSTRUCTIONS
    assert "TypeScript compiler" in SERVER_INSTRUCTIONS
    assert "PostgreSQL parser" in SERVER_INSTRUCTIONS
    assert "file mutation and execution tools are separate" in SERVER_INSTRUCTIONS


def json_object(value: object) -> JsonObject:
    return JSON_OBJECT_ADAPTER.validate_python(value, strict=True)


def json_list(value: object) -> list[JsonValue]:
    assert isinstance(value, list)
    return value


def json_object_member(parent: JsonObject, key: str) -> JsonObject:
    value = parent[key]
    assert isinstance(value, dict)
    return value


def first_json_object(value: object) -> JsonObject:
    items = json_list(value)
    assert items
    item = items[0]
    assert isinstance(item, dict)
    return item


class RecordingRunner:
    def __init__(self, *, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append((arguments, kwargs))
        return subprocess.CompletedProcess(arguments, self.returncode, self.stdout, self.stderr)


class RecordingOutlineRunner:
    def __init__(self, documents: Sequence[object], *, observed_extra: bool = False) -> None:
        self.documents = [json_object(document) for document in documents]
        self.observed_extra = observed_extra
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, command: Sequence[str], **kwargs: object) -> tuple[list[JsonObject], bool]:
        self.calls.append((list(command), kwargs))
        return self.documents, self.observed_extra


def file_stdout_command(payload_path: Path) -> list[str]:
    return [
        sys.executable,
        "-c",
        "import pathlib, sys; sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())",
        str(payload_path),
    ]


def canonical_outline_item(name: str) -> JsonObject:
    return json_object(
        {
            "role": "item",
            "symbolType": "function",
            "name": name,
            "signature": f"def {name}():",
            "range": {
                "byteOffset": {"start": 0, "end": len(name.encode("utf-8"))},
                "start": {"line": 0, "column": 0},
                "end": {"line": 0, "column": len(name)},
            },
            "astKind": "function_definition",
            "isImport": False,
            "isExported": False,
        }
    )


def source_range(text: str, *, start_offset: int = 0, start_column: int = 0) -> JsonObject:
    byte_length = len(text.encode("utf-8"))
    return json_object(
        {
            "byteOffset": {"start": start_offset, "end": start_offset + byte_length},
            "start": {"line": 0, "column": start_column},
            "end": {"line": 0, "column": start_column + len(text)},
        }
    )


def canonical_match(*, file: str = "a.py", text: str = "print(1)") -> JsonObject:
    return json_object(
        {
            "text": text,
            "range": source_range(text),
            "file": file,
            "lines": text,
            "charCount": {"leading": 0, "trailing": 0},
            "language": "Python",
            "metaVariables": {"single": {}, "multi": {}, "transformed": {}},
            "ruleId": "test-rule",
            "severity": "warning",
            "note": None,
            "message": "test message",
            "labels": [{"text": text, "range": source_range(text), "style": "primary"}],
        }
    )


def fake_config_snapshot(root: Path) -> ConfigSnapshot:
    bundle = root / "private-config"
    bundle.mkdir()
    paths = []
    for name in ("inline-sgconfig.yml", "project-sgconfig.yml", "test-sgconfig.yml"):
        path = bundle / name
        path.write_text("ruleDirs: []\n", encoding="utf-8")
        paths.append(path)
    return ConfigSnapshot(
        source_path=root / "sgconfig.yml",
        bundle_root=bundle,
        inline_config_path=paths[0],
        project_config_path=paths[1],
        test_config_path=paths[2],
        digest="a" * 64,
        provenance={"source": str(root / "sgconfig.yml"), "snapshot": "private-read-only"},
        configured_rule_ids=("configured-print", "literal.dot+id"),
        capabilities={
            "inline_search": True,
            "outline": True,
            "configured_scan": True,
            "configured_tests": True,
            "custom_languages": False,
        },
        native_library_hashes={},
        runtime_root=root,
    )


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
    with_oxc: bool = False,
) -> RuntimeServices:
    executable = root / "ast-grep-test-executable"
    executable.write_text("test executable", encoding="utf-8")
    executable.chmod(0o755)
    oxc_helper: ResolvedExecutable | None = None
    oxc_versions: OxcVersions | None = None
    if with_oxc:
        helper = root / "ast-soleaux-oxc.mjs"
        helper.write_text("#!/usr/bin/env node\n", encoding="utf-8")
        oxc_helper = ResolvedExecutable(path=helper, command_prefix=("node", str(helper)))
        oxc_versions = {
            "helper": SUPPORTED_OXC_HELPER_VERSION,
            "parser": SUPPORTED_OXC_PARSER_VERSION,
            "resolver": SUPPORTED_OXC_RESOLVER_VERSION,
        }
    return RuntimeServices(
        working_directory=root,
        executable=ResolvedExecutable(path=executable, command_prefix=(str(executable),)),
        ast_grep_version=SUPPORTED_AST_GREP_VERSION,
        config_path=None,
        allowed_roots=(root,),
        command_timeout_seconds=2.0,
        default_max_results=default_max_results,
        max_results_cap=max_results_cap,
        forbid_regex_rules=forbid_regex_rules,
        oxc_helper=oxc_helper,
        oxc_versions=oxc_versions,
    )


def version_runner(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
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


def test_build_runtime_resolves_and_versions_optional_oxc_helper(tmp_path: Path) -> None:
    executable = tmp_path / "ast-grep"
    executable.write_text("test executable", encoding="utf-8")
    executable.chmod(0o755)
    helper = tmp_path / "ast-soleaux-oxc.mjs"
    helper.write_text("#!/usr/bin/env node\n", encoding="utf-8")

    def runtime_version_runner(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if arguments[-1] == "--version-json":
            return subprocess.CompletedProcess(
                arguments,
                0,
                json.dumps(
                    {
                        "helper": SUPPORTED_OXC_HELPER_VERSION,
                        "parser": SUPPORTED_OXC_PARSER_VERSION,
                        "resolver": SUPPORTED_OXC_RESOLVER_VERSION,
                    }
                ),
                "",
            )
        return version_runner(arguments, **kwargs)

    runtime = build_runtime(
        working_directory=tmp_path,
        ast_grep_executable=str(executable),
        oxc_helper_executable=str(helper),
        runner=runtime_version_runner,
    )
    try:
        assert runtime.oxc_helper is not None
        assert runtime.oxc_helper.path == helper
        assert runtime.oxc_versions == {
            "helper": SUPPORTED_OXC_HELPER_VERSION,
            "parser": SUPPORTED_OXC_PARSER_VERSION,
            "resolver": SUPPORTED_OXC_RESOLVER_VERSION,
        }
    finally:
        runtime.close()


def test_build_runtime_rejects_oxc_helper_version_drift(tmp_path: Path) -> None:
    executable = tmp_path / "ast-grep"
    executable.write_text("test executable", encoding="utf-8")
    executable.chmod(0o755)
    helper = tmp_path / "ast-soleaux-oxc.mjs"
    helper.write_text("#!/usr/bin/env node\n", encoding="utf-8")

    def drifted_runner(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if arguments[-1] == "--version-json":
            return subprocess.CompletedProcess(
                arguments,
                0,
                json.dumps({"helper": "0.0.0", "parser": "0.0.0", "resolver": "0.0.0"}),
                "",
            )
        return version_runner(arguments, **kwargs)

    with pytest.raises(ValueError, match="must report exactly"):
        build_runtime(
            working_directory=tmp_path,
            ast_grep_executable=str(executable),
            oxc_helper_executable=str(helper),
            runner=drifted_runner,
        )


def test_resolve_oxc_helper_requires_a_real_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        resolve_oxc_helper_executable(str(tmp_path / "missing.mjs"), working_directory=tmp_path)


def test_build_runtime_validates_configured_tests_before_returning(tmp_path: Path) -> None:
    executable = tmp_path / "ast-grep"
    executable.write_text("test executable", encoding="utf-8")
    executable.chmod(0o755)
    tests = tmp_path / "rule-tests"
    tests.mkdir()
    config = tmp_path / "sgconfig.yml"
    config.write_text("testConfigs:\n  - testDir: rule-tests\n", encoding="utf-8")
    calls: list[list[str]] = []

    def successful_runner(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        stdout = f"ast-grep {SUPPORTED_AST_GREP_VERSION}\n" if arguments[-1] == "--version" else ""
        return subprocess.CompletedProcess(arguments, 0, stdout, "")

    runtime = build_runtime(
        working_directory=tmp_path,
        ast_grep_executable=str(executable),
        config_path="sgconfig.yml",
        runner=successful_runner,
    )

    assert [command[1] for command in calls[1:]] == ["scan", "test"]
    runtime.close()
    assert not (tmp_path / ".ast-soleaux-runtime").exists()


def test_build_runtime_removes_snapshot_when_startup_validation_fails(tmp_path: Path) -> None:
    executable = tmp_path / "ast-grep"
    executable.write_text("test executable", encoding="utf-8")
    executable.chmod(0o755)
    config = tmp_path / "sgconfig.yml"
    config.write_text("ruleDirs: []\n", encoding="utf-8")
    calls = 0

    def failing_runner(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(arguments, 0, f"ast-grep {SUPPORTED_AST_GREP_VERSION}\n", "")
        return subprocess.CompletedProcess(arguments, 3, "", "invalid configured rule")

    with pytest.raises(RuntimeError, match="exit code 3"):
        build_runtime(
            working_directory=tmp_path,
            ast_grep_executable=str(executable),
            config_path="sgconfig.yml",
            runner=failing_runner,
        )
    assert not (tmp_path / ".ast-soleaux-runtime").exists()


def test_build_runtime_rejects_missing_executable(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        build_runtime(
            working_directory=tmp_path,
            ast_grep_executable=str(tmp_path / "missing"),
        )


def test_build_runtime_rejects_wrong_executable(tmp_path: Path) -> None:
    with pytest.raises(ValueError) as error:
        build_runtime(
            working_directory=tmp_path,
            ast_grep_executable=sys.executable,
        )
    assert "must report exactly 'ast-grep 0.45.0'" in str(error.value)


def test_build_runtime_rejects_ast_grep_version_drift(tmp_path: Path) -> None:
    executable = tmp_path / "ast-grep"
    executable.write_text("test executable", encoding="utf-8")
    executable.chmod(0o755)

    def drifted_version_runner(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(arguments, 0, "ast-grep 0.0.0\n", "")

    with pytest.raises(ValueError) as error:
        build_runtime(
            working_directory=tmp_path,
            ast_grep_executable=str(executable),
            runner=drifted_version_runner,
        )
    assert "must report exactly 'ast-grep 0.45.0'" in str(error.value)


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
    "timeout",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(-1.0, id="negative"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinity"),
    ],
)
def test_build_runtime_rejects_nonfinite_or_nonpositive_timeout(tmp_path: Path, timeout: float) -> None:
    executable = tmp_path / "ast-grep"
    executable.write_text("test executable", encoding="utf-8")
    executable.chmod(0o755)

    with pytest.raises(ValueError, match="finite and greater than zero"):
        build_runtime(
            working_directory=tmp_path,
            ast_grep_executable=str(executable),
            command_timeout_seconds=timeout,
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
    stderr = capsys.readouterr().err
    assert "unrecognized arguments" in stderr


def test_requires_node_closes_executable(mocker: MockerFixture) -> None:
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


def test_resolve_ast_grep_executable_uses_matching_optional_native_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_name = "@ast-grep/cli-test-platform"
    version = "0.45.0"
    package = tmp_path / "node_modules" / "@ast-grep" / "cli"
    package.mkdir(parents=True)
    launcher = package / "ast-grep.js"
    launcher.write_text("console.log('ast-grep')\n", encoding="utf-8")
    (package / "package.json").write_text(
        json.dumps(
            {
                "name": "@ast-grep/cli",
                "version": version,
                "bin": "ast-grep.js",
                "optionalDependencies": {package_name: version},
            }
        ),
        encoding="utf-8",
    )

    native_package = package.parent / "cli-test-platform"
    native_package.mkdir()
    (native_package / "package.json").write_text(
        json.dumps({"name": package_name, "version": version}),
        encoding="utf-8",
    )
    native_executable = native_package / ("ast-grep.exe" if os.name == "nt" else "ast-grep")
    native_executable.write_bytes(b"native executable")
    native_executable.chmod(0o755)

    shim = tmp_path / "node_modules" / ".bin" / "ast-grep"
    shim.parent.mkdir(parents=True)
    shim.write_text("shim", encoding="utf-8")
    monkeypatch.setattr("ast_soleaux.server._native_ast_grep_package_name", lambda: package_name)

    resolved = resolve_ast_grep_executable(str(shim), working_directory=tmp_path)

    assert resolved.path == native_executable
    assert resolved.command_prefix == (str(native_executable),)


@pytest.mark.parametrize("suffix", [".bat", ".cmd"])
def test_resolve_ast_grep_executable_rejects_arbitrary_batch_files(tmp_path: Path, suffix: str) -> None:
    launcher = tmp_path / f"ast-grep{suffix}"
    launcher.write_text("@echo off\n", encoding="utf-8")
    launcher.chmod(0o755)

    with pytest.raises(ValueError, match="Batch-file ast-grep launchers"):
        resolve_ast_grep_executable(str(launcher), working_directory=tmp_path)


def test_windows_complete_command_budget_accepts_boundary_and_rejects_one_more_character() -> None:
    boundary_argument = "a" * (WINDOWS_CREATE_PROCESS_LIMIT - 3)

    validate_process_budget(["x", boundary_argument], environment={}, platform_name="nt")

    with pytest.raises(ValueError, match="CreateProcessW"):
        validate_process_budget(["x", boundary_argument + "a"], environment={}, platform_name="nt")


def test_windows_environment_block_budget_accepts_boundary_and_rejects_one_more_character() -> None:
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


@pytest.mark.parametrize(
    "command",
    [pytest.param(["tool.cmd"], id="cmd"), pytest.param(["tool.bat"], id="bat")],
)
def test_process_budget_rejects_batch_commands_on_every_platform(command: list[str]) -> None:
    with pytest.raises(ValueError, match="Batch-file commands"):
        validate_process_budget(command, environment={}, platform_name="nt")


def invoke_process_budget_boundary(command: object, environment: object) -> None:
    boundary: object = validate_process_budget
    assert callable(boundary)
    boundary(command, environment=environment, platform_name="nt")


@pytest.mark.parametrize(
    "command",
    [
        pytest.param([object()], id="non-string"),
        pytest.param(["tool\0"], id="nul"),
    ],
)
def test_process_budget_rejects_invalid_runtime_command_values(command: list[object]) -> None:
    with pytest.raises(ValueError, match="Command arguments must be strings without NUL characters"):
        invoke_process_budget_boundary(command, {})


@pytest.mark.parametrize(
    "environment",
    [
        pytest.param({1: "value"}, id="non-string-name"),
        pytest.param({"KEY": object()}, id="non-string-value"),
        pytest.param({"BAD\0KEY": "value"}, id="nul-name"),
        pytest.param({"KEY": "bad\0value"}, id="nul-value"),
        pytest.param({"BAD=KEY": "value"}, id="equals-in-name"),
    ],
)
def test_process_budget_rejects_invalid_runtime_environment_values(environment: dict[object, object]) -> None:
    with pytest.raises(ValueError, match="Subprocess environment names and values must be valid NUL-free strings"):
        invoke_process_budget_boundary(["tool"], environment)


def test_run_process_reports_timeout() -> None:
    def timeout_runner(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
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
pathlib.Path(sys.argv[1]).write_text(str(os.getpid()), encoding="utf-8")
deadline = time.monotonic() + 5
while not pathlib.Path(sys.argv[2]).exists() and time.monotonic() < deadline:
    time.sleep(0.01)
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
            timeout_seconds=3.0,
        )

    assert time.monotonic() - started < 5
    child_pid = int(pid_path.read_text(encoding="utf-8"))
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    assert_descendant_listener_stopped(descendant_port_path)


def test_run_process_reports_os_error() -> None:
    def os_error_runner(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError(7, "Argument list too long")

    with pytest.raises(RuntimeError, match="could not be executed"):
        run_process(["ast-grep", "--version"], timeout_seconds=1, runner=os_error_runner)


def test_run_mcp_server_rejects_malformed_boolean_environment(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("AST_GREP_FORBID_REGEX_RULES", "maybe")
    monkeypatch.setattr(sys, "argv", ["ast-soleaux"])

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
        pytest.param(
            b'{"path":"a.py","language":"Python","items":[],"future":NaN}\n',
            "invalid JSON",
            id="non-finite-number",
        ),
        pytest.param(b'{"path":"a.py"', "incomplete NDJSON", id="truncated-at-eof"),
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
        pytest.param(
            b'{"path":"a.py","language":"Python","items":[{"members":[1]}]}\n',
            "non-object node",
            id="invalid-nested-node",
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
    assert payload_path.stat().st_size > MAX_STRUCTURED_OUTPUT_BYTES

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
pathlib.Path(sys.argv[1]).write_text(str(os.getpid()), encoding="utf-8")
deadline = time.monotonic() + 5
while not pathlib.Path(sys.argv[2]).exists() and time.monotonic() < deadline:
    time.sleep(0.01)
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
            timeout_seconds=3.0,
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
                        "items": [canonical_outline_item(f"node-{index}")],
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

    def sandbox_runner(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        cwd_value = kwargs.get("cwd")
        assert isinstance(cwd_value, (str, Path))
        cwd = Path(cwd_value)
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

    def timeout_runner(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal observed_config, observed_sandbox
        cwd_value = kwargs.get("cwd")
        assert isinstance(cwd_value, (str, Path))
        observed_sandbox = Path(cwd_value)
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

    with pytest.raises(ValueError, match="requires paths or include_globs"):
        service.outline_code(
            project_folder="project",
            paths=[],
            language=None,
            max_results=1,
        )
    with pytest.raises(ValueError, match="more than 64"):
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


def test_outline_returns_valid_files_with_structured_missing_path_errors(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "valid.py"
    source.write_text("value = 1\n", encoding="utf-8")
    outline_runner = RecordingOutlineRunner([{"path": str(source), "language": "Python", "items": []}])
    service = AstGrepService(make_runtime(tmp_path), outline_runner=outline_runner)

    result = service.outline_code(
        project_folder="project",
        paths=["valid.py", "missing.py"],
        strict_paths=False,
        max_results=5,
    )

    assert result["resolved_paths"] == ["valid.py"]
    assert result["path_errors"][0]["path"] == "missing.py"
    assert "effective path" in result["path_errors"][0]["error"]
    assert [file["file"] for file in result["files"]] == ["valid.py"]


def test_outline_resolves_bounded_globs_without_guessed_file_names(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = project / "src" / "a.py"
    nested = project / "src" / "nested" / "b.py"
    excluded = project / "src" / "nested" / "b.test.py"
    for file in (source, nested, excluded):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("value = 1\n", encoding="utf-8")
    documents = [
        {"path": str(source), "language": "Python", "items": []},
        {"path": str(nested), "language": "Python", "items": []},
    ]
    outline_runner = RecordingOutlineRunner(documents)
    service = AstGrepService(make_runtime(tmp_path), outline_runner=outline_runner)

    result = service.outline_code(
        project_folder="project",
        include_globs=["src/**/*.py"],
        exclude_globs=["**/*.test.py"],
        max_results=5,
    )

    assert result["resolved_paths"] == ["src/a.py", "src/nested/b.py"]
    command = outline_runner.calls[0][0]
    assert command[-3:] == ["--", str(source), str(nested)]


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
        "resolved_paths": ["example.py"],
        "path_errors": [],
    }
    command, kwargs = outline_runner.calls[0]
    assert command[1] == "outline"
    assert command[command.index("--threads") + 1] == "1"
    assert "--lang" not in command
    assert kwargs["node_limit"] == 3


def test_outline_forwards_optional_language_and_removes_temporary_config(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "example.py"
    source.write_text("value = 1\n", encoding="utf-8")
    observed_config: Path | None = None

    def outline_runner(command: Sequence[str], **kwargs: object) -> tuple[list[JsonObject], bool]:
        nonlocal observed_config
        observed_config = Path(command[command.index("--config") + 1])
        assert observed_config.exists()
        assert command[command.index("--lang") + 1] == "python"
        return ([{"path": str(source), "language": "Python", "items": []}], False)

    service = AstGrepService(make_runtime(tmp_path), outline_runner=outline_runner)

    result = service.outline_code(
        project_folder="project",
        paths=["example.py"],
        language="python",
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

    def outline_runner(command: Sequence[str], **kwargs: object) -> tuple[list[JsonObject], bool]:
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
        include_globs=["*.py", "--follow"],
        exclude_globs=["*_test.py"],
        max_results=1,
    )

    assert results == {
        "matches": [
            {
                "file": "src/example.py",
                "text": "print('one')",
                "evidence_kind": "syntax",
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
    assert "--globs=*.py" in command
    assert "--globs=--follow" in command
    assert "--globs=!*_test.py" in command
    assert "--globs" not in command
    assert command[-2] == "--"
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
    assert command[command.index("--max-results") + 1] == "501"


def test_rule_probe_rejects_matches_beyond_the_configured_cap(tmp_path: Path) -> None:
    payload = "\n".join(json.dumps(canonical_match()) for _ in range(2))
    runner = RecordingRunner(stdout=payload)
    service = AstGrepService(make_runtime(tmp_path, max_results_cap=1, default_max_results=1), runner=runner)

    with pytest.raises(RuntimeError, match="exceeded the configured result cap"):
        service.test_match_code_rule(
            code="print(1)\nprint(2)\n",
            rule_yaml="id: calls\nlanguage: python\nrule:\n  pattern: print($A)\n",
        )
    command = runner.calls[0][0]
    assert command[command.index("--max-results") + 1] == "2"


def test_search_returns_error_severity_diagnostics_streamed_on_exit_one(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "example.py"
    source.write_text("print('value')\n", encoding="utf-8")
    match = {"file": str(source), "text": "print('value')", "range": {"start": {"line": 0}, "end": {"line": 0}}}
    runner = RecordingRunner(
        stdout=json.dumps(match),
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


def _outline_document(path: str, count: int) -> JsonObject:
    items = [{"name": f"symbol{index}", "symbolType": "function"} for index in range(count)]
    return json_object({"path": path, "language": "Python", "items": items})


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


def test_outline_rejects_unrequested_and_duplicate_contained_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "a.py").write_text("def a(): pass\n", encoding="utf-8")
    (project / "b.py").write_text("def b(): pass\n", encoding="utf-8")

    unrequested = AstGrepService(
        make_runtime(tmp_path),
        outline_runner=RecordingOutlineRunner([_outline_document("b.py", 1)]),
    )
    with pytest.raises(RuntimeError, match="was not requested"):
        unrequested.outline_code(project_folder="project", language=None, paths=["a.py"], max_results=5)

    duplicate = AstGrepService(
        make_runtime(tmp_path),
        outline_runner=RecordingOutlineRunner([_outline_document("a.py", 1), _outline_document("a.py", 1)]),
    )
    with pytest.raises(RuntimeError, match="duplicate outline path"):
        duplicate.outline_code(project_folder="project", language=None, paths=["a.py"], max_results=5)


def test_outline_preserves_unknown_document_fields(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "a.py").write_text("def a(): pass\n", encoding="utf-8")
    document = _outline_document("a.py", 1)
    document["futureField"] = {"preserved": True}
    service = AstGrepService(make_runtime(tmp_path), outline_runner=RecordingOutlineRunner([document]))

    result = service.outline_code(project_folder="project", language=None, paths=["a.py"], max_results=5)

    file_document = JSON_OBJECT_ADAPTER.validate_python(result["files"][0], strict=True)
    assert file_document["futureField"] == {"preserved": True}
    assert "path" not in file_document


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

    with pytest.raises(ValueError, match="language is required"):
        service.outline_code(
            project_folder=str(tmp_path),
            language="",
            paths=["a.py"],
            max_results=5,
        )


def test_unsupported_language_fails_before_process_launch(tmp_path: Path) -> None:
    runner = RecordingRunner()
    service = AstGrepService(make_runtime(tmp_path), runner=runner)

    with pytest.raises(ValueError, match=r"Unsupported language 'sql'.*python"):
        service.dump_syntax_tree(code="select 1", language="sql", format="cst")

    assert runner.calls == []


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
        json_object(
            {
                "file": "src/example.py",
                "text": "print('one')",
                "range": {"start": {"line": 4}, "end": {"line": 4}},
            }
        )
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
    text_block = text_result.content[0]
    assert isinstance(text_block, TextContent)
    assert text_block.text.startswith("Found 2 outline nodes")
    json_result = outline_tool_result(results, "json")
    assert json_result.structured_content == results
    json_block = json_result.content[0]
    assert isinstance(json_block, TextContent)
    assert json.loads(json_block.text) == results


def test_server_info_exposes_effective_contract(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path, default_max_results=25, max_results_cap=100, forbid_regex_rules=True)

    info = AstGrepService(runtime).get_server_info()

    assert info["fork_version"] == "0.5.0"
    assert info["ast_grep_version"] == SUPPORTED_AST_GREP_VERSION
    assert info["oxc_helper_executable"] is None
    assert info["oxc_versions"] is None
    assert info["capabilities"]["javascript_module_inspection"] is False
    assert info["allowed_roots"] == [str(tmp_path)]
    assert info["default_max_results"] == 25
    assert info["max_results_cap"] == 100
    assert info["forbid_regex_rules"] is True
    assert os.path.isabs(info["ast_grep_executable"])
    assert "python" in info["supported_language_ids"]
    assert "sql" not in info["supported_language_ids"]


def test_text_runner_streams_bounded_stdin_and_utf8_output(tmp_path: Path) -> None:
    payload = "héllo\n" * 1000
    result = run_text_process(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())"],
        timeout_seconds=2,
        input_text=payload,
        working_directory=tmp_path,
    )

    assert result.completed.returncode == 0
    assert result.completed.stdout == payload
    assert result.completed.stderr == ""
    assert result.stdout_truncated is False
    assert result.stderr_truncated is False


@pytest.mark.parametrize("payload", ["x" * (MAX_SNIPPET_INPUT_BYTES + 1), "\ud800"], ids=["oversized", "invalid-utf8"])
def test_text_runner_rejects_oversized_or_invalid_utf8_input(tmp_path: Path, payload: str) -> None:
    with pytest.raises(ValueError, match=r"1 MiB limit|valid UTF-8"):
        run_text_process(
            [sys.executable, "-c", "raise SystemExit"],
            timeout_seconds=2,
            input_text=payload,
            working_directory=tmp_path,
        )


def test_text_runner_fails_closed_or_truncates_at_explicit_output_cap(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x' * 129)"]
    with pytest.raises(RuntimeError, match="structured output exceeds the 128-byte limit"):
        run_text_process(command, timeout_seconds=2, working_directory=tmp_path, stdout_limit=128)

    result = run_text_process(
        command,
        timeout_seconds=2,
        working_directory=tmp_path,
        stdout_limit=128,
        truncate_stdout=True,
    )
    assert result.completed.stdout == "x" * 128
    assert result.stdout_truncated is True


def test_text_runner_trims_only_a_terminal_utf8_fragment_at_a_truncation_boundary(tmp_path: Path) -> None:
    result = run_text_process(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write('éé'.encode())"],
        timeout_seconds=2,
        working_directory=tmp_path,
        stdout_limit=3,
        truncate_stdout=True,
    )
    assert result.completed.stdout == "é"
    assert result.stdout_truncated is True


def test_text_runner_rejects_invalid_utf8_before_a_truncation_boundary(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'\\xffx')"]
    with pytest.raises(RuntimeError, match="invalid UTF-8 on stdout"):
        run_text_process(
            command,
            timeout_seconds=2,
            working_directory=tmp_path,
            stdout_limit=1,
            truncate_stdout=True,
        )


def test_text_runner_terminates_promptly_on_non_truncating_overflow(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "import sys,time; sys.stdout.buffer.write(b'x' * 65536); sys.stdout.buffer.flush(); time.sleep(30)",
    ]
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="structured output exceeds the 128-byte limit"):
        run_text_process(command, timeout_seconds=10, working_directory=tmp_path, stdout_limit=128)
    assert time.monotonic() - started < 5


def test_text_runner_reaps_descendants_after_the_leader_exits(tmp_path: Path) -> None:
    descendant_port_path = tmp_path / "text-descendant.port"
    leader_program = """
import pathlib
import subprocess
import sys
import time

subprocess.Popen([sys.executable, "-c", sys.argv[2], sys.argv[1]])
deadline = time.monotonic() + 5
while not pathlib.Path(sys.argv[1]).exists() and time.monotonic() < deadline:
    time.sleep(0.01)
"""
    result = run_text_process(
        [
            sys.executable,
            "-c",
            leader_program,
            str(descendant_port_path),
            descendant_listener_program(),
        ],
        timeout_seconds=5,
        working_directory=tmp_path,
    )
    assert result.completed.returncode == 0
    assert_descendant_listener_stopped(descendant_port_path)


def test_ndjson_runner_reaps_descendants_after_the_leader_exits(tmp_path: Path) -> None:
    descendant_port_path = tmp_path / "ndjson-descendant.port"
    leader_program = """
import json
import pathlib
import subprocess
import sys
import time

subprocess.Popen([sys.executable, "-c", sys.argv[2], sys.argv[1]])
deadline = time.monotonic() + 5
while not pathlib.Path(sys.argv[1]).exists() and time.monotonic() < deadline:
    time.sleep(0.01)
sys.stdout.write(sys.argv[3] + "\\n")
sys.stdout.flush()
"""
    with pytest.raises(RuntimeError, match="timed out"):
        run_ndjson_process(
            [
                sys.executable,
                "-c",
                leader_program,
                str(descendant_port_path),
                descendant_listener_program(),
                json.dumps(canonical_match()),
            ],
            timeout_seconds=2,
            working_directory=tmp_path,
            record_parser=_parse_match_record,
            item_limit=10,
        )
    assert_descendant_listener_stopped(descendant_port_path)


@pytest.mark.parametrize(("stream", "message"), [("stdout", "stdout"), ("stderr", "stderr")])
def test_text_runner_rejects_invalid_utf8_output(tmp_path: Path, stream: str, message: str) -> None:
    destination = "stdout" if stream == "stdout" else "stderr"
    command = [sys.executable, "-c", f"import sys; sys.{destination}.buffer.write(b'\\xff')"]
    with pytest.raises(RuntimeError, match=f"invalid UTF-8 on {message}"):
        run_text_process(command, timeout_seconds=2, working_directory=tmp_path)


def test_text_runner_enforces_strict_diagnostic_cap(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "import sys; sys.stderr.buffer.write(b'x' * 129)"]
    with pytest.raises(RuntimeError, match="diagnostic output exceeds the 128-byte limit"):
        run_text_process(
            command,
            timeout_seconds=2,
            working_directory=tmp_path,
            stderr_limit=128,
            truncate_stderr=False,
        )


def test_match_ndjson_runner_preserves_unknown_fields_and_canonical_offsets(tmp_path: Path) -> None:
    match = canonical_match(text="é")
    match["futureField"] = {"kept": True}
    payload = tmp_path / "matches.ndjson"
    payload.write_text(json.dumps(match) + "\n", encoding="utf-8")

    result = run_ndjson_process(
        file_stdout_command(payload),
        timeout_seconds=2,
        working_directory=tmp_path,
        record_parser=_parse_match_record,
        item_limit=10,
    )

    assert result.records == [match]
    assert result.records[0]["futureField"] == {"kept": True}
    assert result.observed_extra is False
    assert result.returncode == 0


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        pytest.param(b"\xff\n", "invalid UTF-8", id="invalid-utf8"),
        pytest.param(b"{oops}\n", "invalid JSON", id="malformed-json"),
        pytest.param(b'{"future":NaN}\n', "invalid JSON", id="non-finite-number"),
        pytest.param(b"[]\n", "non-object JSON", id="non-object"),
        pytest.param(b"\n", "empty NDJSON", id="empty-record"),
        pytest.param(b'{"text":"partial"}', "incomplete NDJSON", id="incomplete-record"),
    ],
)
def test_match_ndjson_runner_rejects_invalid_records(tmp_path: Path, payload: bytes, message: str) -> None:
    payload_path = tmp_path / "invalid.ndjson"
    payload_path.write_bytes(payload)
    with pytest.raises(RuntimeError, match=message):
        run_ndjson_process(
            file_stdout_command(payload_path),
            timeout_seconds=2,
            working_directory=tmp_path,
            record_parser=_parse_match_record,
            item_limit=10,
        )


def test_match_ndjson_runner_rejects_oversized_record_and_diagnostics(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized.ndjson"
    oversized.write_bytes(b"x" * (MAX_NDJSON_RECORD_BYTES + 1) + b"\n")
    with pytest.raises(RuntimeError, match="record exceeds the 1024 KiB limit"):
        run_ndjson_process(
            file_stdout_command(oversized),
            timeout_seconds=2,
            working_directory=tmp_path,
            record_parser=_parse_match_record,
        )

    diagnostics = [
        sys.executable,
        "-c",
        f"import sys; sys.stderr.buffer.write(b'x' * {MAX_SUBPROCESS_DIAGNOSTIC_BYTES + 1})",
    ]
    with pytest.raises(RuntimeError, match="diagnostic output exceeds the 64 KiB limit"):
        run_ndjson_process(
            diagnostics,
            timeout_seconds=2,
            working_directory=tmp_path,
            record_parser=_parse_match_record,
        )


def test_match_validator_accepts_transforms_replacement_and_multiple_metavariables() -> None:
    match = canonical_match()
    match["metaVariables"] = {
        "single": {"A": {"text": "1", "range": source_range("1", start_offset=6, start_column=6)}},
        "multi": {"ARGS": [{"text": "1", "range": source_range("1", start_offset=6, start_column=6)}]},
        "transformed": {"NORMALIZED": "one"},
    }
    match["replacement"] = "logger.info(1)"
    match["replacementOffsets"] = {"start": 10, "end": 18}
    match["metadata"] = {"category": "test"}
    first_json_object(match["labels"])["message"] = "primary"

    validate_match_document(match, record_number=1)


def test_match_validator_accepts_omitted_empty_labels() -> None:
    match = canonical_match()
    del match["labels"]
    validate_match_document(match, record_number=1)


def test_match_validator_accepts_multiline_position_geometry() -> None:
    text = "one\nβ"
    match = canonical_match(text=text)
    match["lines"] = text
    json_object_member(match, "range")["end"] = {"line": 1, "column": 1}
    label = first_json_object(match["labels"])
    json_object_member(label, "range")["end"] = {"line": 1, "column": 1}
    validate_match_document(match, record_number=1)


@pytest.mark.parametrize(
    "end",
    [
        pytest.param({"line": 0, "column": 7}, id="wrong-same-line-column"),
        pytest.param({"line": 2, "column": 1}, id="wrong-multiline-row"),
        pytest.param({"line": 1, "column": 2}, id="wrong-multiline-column"),
    ],
)
def test_match_validator_rejects_impossible_position_geometry(end: dict[str, int]) -> None:
    text = "one\nβ" if end["line"] else "print(1)"
    match = canonical_match(text=text)
    match["lines"] = text
    json_object_member(match, "range")["end"] = json_object(end)
    with pytest.raises(RuntimeError, match="line and column geometry"):
        validate_match_document(match, record_number=1)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        pytest.param(("range", None), "invalid range", id="range"),
        pytest.param(("charCount", {"leading": -1, "trailing": 0}), "charCount.leading", id="character-count"),
        pytest.param(("severity", "critical"), "invalid severity", id="severity"),
        pytest.param(("labels", {}), "invalid labels", id="labels"),
        pytest.param(("metadata", []), "invalid metadata", id="metadata"),
    ],
)
def test_match_validator_rejects_invalid_typed_fields(mutation: tuple[str, JsonValue], message: str) -> None:
    match = canonical_match()
    field, value = mutation
    match[field] = value
    with pytest.raises(RuntimeError, match=message):
        validate_match_document(match, record_number=7)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        pytest.param("text", None, "no text", id="text"),
        pytest.param("lines", None, "no lines", id="lines"),
        pytest.param("file", "", "no file", id="file"),
        pytest.param("language", "", "no language", id="language"),
        pytest.param("ruleId", "", "no ruleId", id="rule-id"),
        pytest.param("message", None, "invalid message", id="message"),
        pytest.param("note", 1, "invalid note", id="note"),
    ],
)
def test_match_validator_rejects_missing_canonical_fields(field: str, value: JsonValue, message: str) -> None:
    match = canonical_match()
    match[field] = value
    with pytest.raises(RuntimeError, match=message):
        validate_match_document(match, record_number=3)


def test_match_validator_rejects_invalid_metavariable_and_label_details() -> None:
    invalid_context = canonical_match()
    invalid_context["charCount"] = {"leading": 9, "trailing": 0}
    with pytest.raises(RuntimeError, match="character context counts"):
        validate_match_document(invalid_context, record_number=1)

    inconsistent_context = canonical_match()
    inconsistent_context["lines"] = "different"
    with pytest.raises(RuntimeError, match="inconsistent character context"):
        validate_match_document(inconsistent_context, record_number=1)

    invalid_meta_text = canonical_match()
    json_object_member(invalid_meta_text, "metaVariables")["single"] = {"A": {"text": 1, "range": source_range("1")}}
    with pytest.raises(RuntimeError, match=r"single\.A\.text"):
        validate_match_document(invalid_meta_text, record_number=1)

    invalid_transform = canonical_match()
    json_object_member(invalid_transform, "metaVariables")["transformed"] = {"A": 1}
    with pytest.raises(RuntimeError, match="transformed metavariables"):
        validate_match_document(invalid_transform, record_number=1)

    invalid_label_text = canonical_match()
    first_json_object(invalid_label_text["labels"])["text"] = None
    with pytest.raises(RuntimeError, match=r"labels\[0\].text"):
        validate_match_document(invalid_label_text, record_number=1)

    invalid_label_style = canonical_match()
    first_json_object(invalid_label_style["labels"])["style"] = "tertiary"
    with pytest.raises(RuntimeError, match=r"labels\[0\].style"):
        validate_match_document(invalid_label_style, record_number=1)

    invalid_label_message = canonical_match()
    first_json_object(invalid_label_message["labels"])["message"] = 1
    with pytest.raises(RuntimeError, match=r"labels\[0\].message"):
        validate_match_document(invalid_label_message, record_number=1)

    outside_label = canonical_match()
    labels = json_list(outside_label["labels"])
    labels[0] = {
        "text": "x",
        "range": source_range("x", start_offset=20, start_column=20),
        "style": "secondary",
    }
    outside_label["labels"] = labels
    validate_match_document(outside_label, record_number=1)

    without_rule_fields = canonical_match()
    without_rule_fields.pop("ruleId")
    validate_match_document(without_rule_fields, record_number=1, require_rule_fields=False)


def test_match_validator_rejects_reversed_positions_and_inconsistent_utf8() -> None:
    reversed_position = canonical_match()
    reversed_range = json_object_member(reversed_position, "range")
    reversed_range["end"] = {"line": 0, "column": 0}
    reversed_range["start"] = {"line": 1, "column": 0}
    with pytest.raises(RuntimeError, match="reversed half-open positions"):
        validate_match_document(reversed_position, record_number=1)

    inconsistent = canonical_match(text="é")
    inconsistent_range = json_object_member(inconsistent, "range")
    json_object_member(inconsistent_range, "byteOffset")["end"] = 1
    with pytest.raises(RuntimeError, match="inconsistent UTF-8 byte offsets"):
        validate_match_document(inconsistent, record_number=1)

    outside = canonical_match()
    json_object_member(outside, "metaVariables")["single"] = {
        "A": {"text": "x", "range": source_range("x", start_offset=20, start_column=20)}
    }
    validate_match_document(outside, record_number=1)


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param({"replacement": "new"}, id="missing-offsets"),
        pytest.param({"replacementOffsets": {"start": 0, "end": 1}}, id="missing-replacement"),
        pytest.param({"replacement": 1, "replacementOffsets": {"start": 0, "end": 1}}, id="non-string"),
        pytest.param({"replacement": "new", "replacementOffsets": {"start": 2, "end": 1}}, id="reversed"),
    ],
)
def test_match_validator_rejects_invalid_replacement_pairs(mutation: JsonObject) -> None:
    match = canonical_match()
    match.update(mutation)
    with pytest.raises(RuntimeError, match="replacement"):
        validate_match_document(match, record_number=1)


def test_find_code_builds_contextual_selector_strictness_and_preview_only_deletion(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text("print(1)\n", encoding="utf-8")
    runner = RecordingRunner()
    service = AstGrepService(make_runtime(tmp_path), runner=runner)

    result = service.find_code(
        project_folder=str(tmp_path),
        pattern="context(print($A))",
        language="python",
        selector="call",
        strictness="ast",
        rewrite="",
        paths=["sample.py"],
        include_globs=None,
        exclude_globs=None,
        max_results=5,
    )

    assert result["matches"] == []
    command = runner.calls[0][0]
    rule = yaml.safe_load(command[command.index("--inline-rules") + 1])
    assert rule["rule"]["pattern"] == {
        "context": "context(print($A))",
        "strictness": "ast",
        "selector": "call",
    }
    assert rule["fix"] == ""
    assert "--update-all" not in command
    assert "--rewrite-all" not in command


@pytest.mark.parametrize(
    ("selector", "strictness", "rewrite", "message"),
    [
        pytest.param("", "smart", None, "selector", id="empty-selector"),
        pytest.param(None, "invalid", None, "strictness", id="strictness"),
        pytest.param(None, "smart", "\ud800", "valid UTF-8", id="rewrite-utf8"),
    ],
)
def test_find_code_rejects_invalid_new_options(
    tmp_path: Path,
    selector: str | None,
    strictness: object,
    rewrite: str | None,
    message: str,
) -> None:
    source = tmp_path / "sample.py"
    source.write_text("print(1)\n", encoding="utf-8")
    service = AstGrepService(make_runtime(tmp_path), runner=RecordingRunner())
    with pytest.raises(ValueError, match=message):
        invoke_boundary(
            service.find_code,
            project_folder=str(tmp_path),
            pattern="print($A)",
            language="python",
            selector=selector,
            strictness=strictness,
            rewrite=rewrite,
            paths=["sample.py"],
            include_globs=None,
            exclude_globs=None,
            max_results=5,
        )


def test_outline_forwards_items_symbol_types_and_public_member_filter(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text("def public(): pass\n", encoding="utf-8")
    outline_runner = RecordingOutlineRunner([{"path": str(source), "language": "Python", "items": []}])
    service = AstGrepService(make_runtime(tmp_path), outline_runner=outline_runner)

    service.outline_code(
        project_folder=str(tmp_path),
        paths=["sample.py"],
        language="python",
        max_results=5,
        items="exports",
        symbol_types=["function", "class", "function"],
        public_members=True,
    )

    command = outline_runner.calls[0][0]
    assert command[command.index("--items") + 1] == "exports"
    assert command[command.index("--type") + 1] == "function,class"
    assert "--pub-members" in command
    assert command[-2:] == ["--", str(source)]


@pytest.mark.parametrize(
    ("items", "symbol_types", "message"),
    [
        pytest.param("unknown", None, "item mode", id="items"),
        pytest.param("auto", ["unknown"], "symbol type", id="symbol-type"),
        pytest.param("auto", [1], "symbol type", id="non-string-symbol-type"),
    ],
)
def test_outline_rejects_invalid_modes_and_symbol_types(
    tmp_path: Path,
    items: object,
    symbol_types: object,
    message: str,
) -> None:
    source = tmp_path / "sample.py"
    source.write_text("pass\n", encoding="utf-8")
    service = AstGrepService(make_runtime(tmp_path), outline_runner=RecordingOutlineRunner([]))
    with pytest.raises(ValueError, match=message):
        invoke_boundary(
            service.outline_code,
            project_folder=str(tmp_path),
            paths=["sample.py"],
            language=None,
            max_results=5,
            items=items,
            symbol_types=symbol_types,
        )


def test_configured_scan_uses_private_config_exact_escaped_ids_and_metadata(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text("literal_id = 1\n", encoding="utf-8")
    match = {"file": str(source), "text": "literal_id = 1", "range": {"start": {"line": 0}, "end": {"line": 0}}}
    runner = RecordingRunner(stdout=json.dumps(match) + "\n")
    snapshot = fake_config_snapshot(tmp_path)
    runtime = replace(make_runtime(tmp_path), config_snapshot=snapshot)
    service = AstGrepService(runtime, runner=runner)

    result = service.scan_project_rules(
        project_folder=str(tmp_path),
        rule_ids=["literal.dot+id", "literal.dot+id"],
        paths=["sample.py"],
        include_globs=["**/*.py"],
        exclude_globs=None,
        max_results=5,
        include_metadata=True,
    )

    assert result["matches"][0]["file"] == "sample.py"
    command = runner.calls[0][0]
    assert command[command.index("--config") + 1] == str(snapshot.project_config_path)
    assert command[command.index("--filter") + 1] == r"^(?:literal\.dot\+id)$"
    assert "--include-metadata" in command
    assert "--inline-rules" not in command
    assert command[-2:] == ["--", "sample.py"]


@pytest.mark.parametrize(
    "rule_ids",
    [
        pytest.param([], id="empty"),
        pytest.param(["unknown"], id="unknown"),
        pytest.param([""], id="empty-id"),
        pytest.param([1], id="non-string"),
    ],
)
def test_configured_scan_rejects_invalid_rule_id_filters(tmp_path: Path, rule_ids: object) -> None:
    snapshot = fake_config_snapshot(tmp_path)
    service = AstGrepService(replace(make_runtime(tmp_path), config_snapshot=snapshot), runner=RecordingRunner())
    with pytest.raises(ValueError, match=r"rule_ids|configured rule id"):
        invoke_boundary(
            service.scan_project_rules,
            project_folder=str(tmp_path),
            rule_ids=rule_ids,
            paths=None,
            include_globs=None,
            exclude_globs=None,
            max_results=5,
        )


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "passed"),
    [
        pytest.param(0, "all passed\n", "", True, id="passed"),
        pytest.param(4, "one failed\n", "failure details\n", False, id="failed"),
    ],
)
def test_configured_tests_are_read_only_and_report_expected_failures(
    tmp_path: Path,
    returncode: int,
    stdout: str,
    stderr: str,
    passed: bool,
) -> None:
    snapshot = fake_config_snapshot(tmp_path)
    runner = RecordingRunner(stdout=stdout, stderr=stderr, returncode=returncode)
    service = AstGrepService(replace(make_runtime(tmp_path), config_snapshot=snapshot), runner=runner)

    result = service.test_project_rules(rule_ids=["configured-print"])

    assert result["passed"] is passed
    assert stdout.strip() in result["report"]
    command, kwargs = runner.calls[0]
    assert command[command.index("--config") + 1] == str(snapshot.test_config_path)
    assert command[command.index("--filter") + 1] == r"^(?:configured\-print)$"
    assert "--interactive" not in command
    assert "--update-all" not in command
    assert kwargs["cwd"] == str(snapshot.bundle_root)


def test_configured_test_execution_errors_and_report_truncation(tmp_path: Path) -> None:
    snapshot = fake_config_snapshot(tmp_path)
    service = AstGrepService(
        replace(make_runtime(tmp_path), config_snapshot=snapshot),
        runner=RecordingRunner(stderr="invalid configuration", returncode=8),
    )
    with pytest.raises(RuntimeError, match="exit code 8"):
        service.test_project_rules()

    report, truncated = service._bounded_report("é" * MAX_TEST_REPORT_BYTES, "diagnostic")
    assert len(report.encode("utf-8")) <= MAX_TEST_REPORT_BYTES
    assert report.endswith("…")
    assert truncated is True


def test_configured_capabilities_are_required_and_exposed_in_server_info(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    service = AstGrepService(runtime, runner=RecordingRunner())
    with pytest.raises(ValueError, match="No project rules"):
        service.scan_project_rules(
            project_folder=str(tmp_path),
            rule_ids=None,
            paths=None,
            include_globs=None,
            exclude_globs=None,
            max_results=5,
        )
    with pytest.raises(ValueError, match="No project rule tests"):
        service.test_project_rules()

    snapshot = fake_config_snapshot(tmp_path)
    info = AstGrepService(replace(runtime, config_snapshot=snapshot)).get_server_info()
    assert info["configuration_digest"] == "a" * 64
    assert info["configuration_provenance"] == snapshot.provenance
    assert info["capabilities"] == {
        **snapshot.capabilities,
        "javascript_module_inspection": False,
        "semantic_analysis": False,
        "typescript_project_inspection": False,
        "postgresql_parser": False,
        "typescript_execution": False,
    }
    assert info["coordinate_conventions"]["range"] == "half-open [start,end)"
    assert info["coordinate_conventions"]["oxc_offset"] == "zero-based UTF-16 code units"
    assert info["resource_limits"]["snippet_input_bytes"] == MAX_SNIPPET_INPUT_BYTES
    assert info["resource_limits"]["structured_output_bytes"] == MAX_STRUCTURED_OUTPUT_BYTES
    assert info["resource_limits"]["native_library_bytes"] == MAX_NATIVE_LIBRARY_BYTES
    assert info["resource_limits"]["oxc_files"] == MAX_OUTLINE_PATHS
    assert info["resource_limits"]["windows_create_process_characters"] == WINDOWS_CREATE_PROCESS_LIMIT
    assert info["resource_limits"]["posix_arg_headroom_bytes"] == POSIX_ARG_HEADROOM_BYTES
    assert info["resource_limits"]["process_termination_grace_seconds"] == PROCESS_TERMINATION_GRACE_SECONDS


def test_argument_parser_accepts_repeated_trusted_native_libraries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ast-soleaux",
            "--oxc-helper",
            "ast-soleaux-oxc.mjs",
            "--trusted-native-library",
            "one.dylib",
            "a" * 64,
            "--trusted-native-library",
            "two.so",
            "b" * 64,
        ],
    )
    args = build_argument_parser().parse_args()
    assert args.oxc_helper == "ast-soleaux-oxc.mjs"
    assert args.trusted_native_library == [["one.dylib", "a" * 64], ["two.so", "b" * 64]]


def test_match_text_format_includes_rule_preview_offsets_and_transformations() -> None:
    match = canonical_match(file="src/example.py")
    match["replacement"] = ""
    match["replacementOffsets"] = {"start": 0, "end": 8}
    json_object_member(match, "metaVariables")["transformed"] = {"NORMALIZED": "one"}
    match["fix"] = {"template": "$NORMALIZED"}
    match["transform"] = {"NORMALIZED": "converted"}
    match["rewriters"] = [{"id": "convert"}]

    formatted = format_matches_as_text([match])

    assert "Rule: test-rule (warning) — test message" in formatted
    assert "Preview replacement: <delete match>" in formatted
    assert 'Replacement offsets: {"start":0,"end":8}' in formatted
    assert 'Transformed metavariables: {"NORMALIZED":"one"}' in formatted
    assert "rewriters:" in formatted


def test_search_cursor_pages_and_rejects_query_mismatch() -> None:
    store = ResultCursorStore()
    matches = [json_object({"file": f"src/{index}.py"}) for index in range(3)]

    first = store.first_search_page(
        query_digest="query-a",
        matches=matches,
        page_size=1,
        source_truncated=False,
    )

    cursor = first.get("next_cursor")
    assert isinstance(cursor, str)
    assert first["matches"] == [{"file": "src/0.py"}]
    with pytest.raises(ValueError, match="does not match"):
        store.next_search_page(cursor=cursor, query_digest="query-b", page_size=1)

    second = store.next_search_page(cursor=cursor, query_digest="query-a", page_size=1)
    third = store.next_search_page(cursor=cursor, query_digest="query-a", page_size=1)
    assert second["matches"] == [{"file": "src/1.py"}]
    assert second.get("next_cursor") == cursor
    assert third["matches"] == [{"file": "src/2.py"}]
    assert third.get("next_cursor") is None
    with pytest.raises(ValueError, match="invalid or expired"):
        store.next_search_page(cursor=cursor, query_digest="query-a", page_size=1)


@pytest.mark.asyncio
async def test_unconfigured_fastmcp_catalog_hides_project_rule_tools(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)

    tool_names = {tool.name for tool in await create_mcp(runtime).list_tools()}

    assert tool_names == {
        "dump_syntax_tree",
        "test_match_code_rule",
        "outline_code",
        "find_code",
        "find_code_by_rule",
        "get_server_info",
    }


@pytest.mark.asyncio
async def test_oxc_configured_fastmcp_catalog_exposes_module_inspection(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path, with_oxc=True)

    tool_names = {tool.name for tool in await create_mcp(runtime).list_tools()}

    assert "oxc_modules" in tool_names


def test_oxc_module_inspection_normalizes_contained_paths_and_request(tmp_path: Path) -> None:
    source = tmp_path / "src" / "entry.ts"
    dependency = tmp_path / "src" / "dep.js"
    package_json = tmp_path / "package.json"
    source.parent.mkdir()
    source.write_text('import { value } from "./dep.js";\n', encoding="utf-8")
    dependency.write_text("export const value = 1;\n", encoding="utf-8")
    package_json.write_text('{"type":"module"}\n', encoding="utf-8")
    response = {
        "versions": {
            "helper": SUPPORTED_OXC_HELPER_VERSION,
            "parser": SUPPORTED_OXC_PARSER_VERSION,
            "resolver": SUPPORTED_OXC_RESOLVER_VERSION,
        },
        "graph_version": 1,
        "graph": {"version": 1, "nodes": [], "edges": []},
        "cache_hit": False,
        "modules": [
            {
                "file": "src/entry.ts",
                "has_module_syntax": True,
                "source_type": "module",
                "package": None,
                "commonjs_exports": [],
                "import_meta_spans": [],
                "edges": [
                    {
                        "kind": "import",
                        "module_system": "esm",
                        "specifier": "./dep.js",
                        "expression": None,
                        "start": 22,
                        "end": 32,
                        "resolution": "resolved",
                        "resolved_path": str(dependency),
                        "package_json_path": str(package_json),
                        "module_type": "module",
                        "resolution_error": None,
                    }
                ],
                "diagnostics": [],
            }
        ],
    }
    runner = RecordingRunner(stdout=json.dumps(response))
    runtime = make_runtime(tmp_path, with_oxc=True)
    service = AstGrepService(runtime, runner=runner)

    modules = service.inspect_oxc_modules(
        project_folder=str(tmp_path),
        paths=["src/entry.ts"],
        include_globs=None,
        exclude_globs=None,
        strict_paths=True,
        include_dynamic=True,
    )

    assert modules[0]["edges"][0]["resolved_path"] == "src/dep.js"
    assert modules[0]["edges"][0]["package_json_path"] == "package.json"
    command, kwargs = runner.calls[0]
    assert command[-1].endswith("ast-soleaux-oxc.mjs")
    input_text = kwargs["input"]
    assert isinstance(input_text, str)
    assert json.loads(input_text) == {
        "project_root": str(tmp_path),
        "files": ["src/entry.ts"],
        "include_dynamic": True,
    }


def test_oxc_module_inspection_rejects_sidecar_paths_outside_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = project / "entry.ts"
    outside = tmp_path / "outside.js"
    project.mkdir()
    source.write_text('import "../outside.js";\n', encoding="utf-8")
    outside.write_text("export {};\n", encoding="utf-8")
    response = {
        "versions": {
            "helper": SUPPORTED_OXC_HELPER_VERSION,
            "parser": SUPPORTED_OXC_PARSER_VERSION,
            "resolver": SUPPORTED_OXC_RESOLVER_VERSION,
        },
        "graph_version": 1,
        "graph": {"version": 1, "nodes": [], "edges": []},
        "cache_hit": False,
        "modules": [
            {
                "file": "entry.ts",
                "has_module_syntax": True,
                "source_type": "module",
                "package": None,
                "commonjs_exports": [],
                "import_meta_spans": [],
                "edges": [
                    {
                        "kind": "import",
                        "module_system": "esm",
                        "specifier": "../outside.js",
                        "expression": None,
                        "start": 7,
                        "end": 20,
                        "resolution": "resolved",
                        "resolved_path": str(outside),
                        "package_json_path": None,
                        "module_type": None,
                        "resolution_error": None,
                    }
                ],
                "diagnostics": [],
            }
        ],
    }
    service = AstGrepService(make_runtime(project, with_oxc=True), runner=RecordingRunner(stdout=json.dumps(response)))

    with pytest.raises(RuntimeError, match="outside project_folder"):
        service.inspect_oxc_modules(
            project_folder=str(project),
            paths=["entry.ts"],
            include_globs=None,
            exclude_globs=None,
            strict_paths=True,
            include_dynamic=False,
        )


def test_relative_project_uses_single_allowed_root(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = project / "example.py"
    source.parent.mkdir()
    source.write_text("print('value')\n", encoding="utf-8")
    server_directory = tmp_path / "server"
    server_directory.mkdir()
    runtime = replace(make_runtime(tmp_path), working_directory=server_directory)
    service = AstGrepService(runtime, runner=RecordingRunner())

    result = service.find_code(
        project_folder="project",
        pattern="print($A)",
        language="python",
        paths=["example.py"],
        include_globs=None,
        exclude_globs=None,
        max_results=1,
    )

    assert result["returned"] == 0


def test_duplicate_project_path_error_includes_effective_path(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    service = AstGrepService(make_runtime(tmp_path), runner=RecordingRunner())

    with pytest.raises(ValueError, match="Remove the repeated project prefix"):
        service.find_code(
            project_folder="project",
            pattern="print($A)",
            language="python",
            paths=["project/example.py"],
            include_globs=None,
            exclude_globs=None,
            max_results=1,
        )
