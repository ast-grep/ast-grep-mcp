from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from main import (
    AstGrepService,
    ResolvedExecutable,
    ServerRuntime,
    _requires_node,
    build_argument_parser,
    build_runtime,
    format_matches_as_text,
    format_search_results,
    parse_stream_matches,
    resolve_ast_grep_executable,
    run_process,
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
        ast_grep_version="0.44.1",
        config_path=None,
        allowed_roots=(root,),
        command_timeout_seconds=2.0,
        default_max_results=default_max_results,
        max_results_cap=max_results_cap,
        forbid_regex_rules=forbid_regex_rules,
    )


def version_runner(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(arguments, 0, "ast-grep 0.44.1\n", "")


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
    assert runtime.ast_grep_version == "0.44.1"
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


def test_run_process_reports_timeout() -> None:
    def timeout_runner(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(arguments, timeout=1)

    with pytest.raises(RuntimeError, match="timed out after 1 seconds"):
        run_process(["ast-grep", "--version"], timeout_seconds=1, runner=timeout_runner)


def test_parse_stream_matches_rejects_non_json_output() -> None:
    with pytest.raises(RuntimeError, match="invalid JSON"):
        parse_stream_matches("not json")


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


def test_validate_rule_yaml_allows_regex_matcher_when_policy_disabled() -> None:
    validate_rule_yaml(
        "id: text\nlanguage: python\nrule:\n  regex: secret.*\n",
        forbid_regex_rules=False,
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


def test_server_info_exposes_effective_contract(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path, default_max_results=25, max_results_cap=100, forbid_regex_rules=True)

    info = AstGrepService(runtime).get_server_info()

    assert info["fork_version"] == "0.2.0"
    assert info["ast_grep_version"] == "0.44.1"
    assert info["allowed_roots"] == [str(tmp_path)]
    assert info["default_max_results"] == 25
    assert info["max_results_cap"] == 100
    assert info["forbid_regex_rules"] is True
    assert os.path.isabs(info["ast_grep_executable"])
