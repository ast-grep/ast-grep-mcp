from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import pytest


class EnvironmentPolicy(Protocol):
    ROOT: Path
    EXPECTED_PYTHON: tuple[int, int, int]
    sys: ModuleType

    def shell_policy_failures(self, source: str, *, label: str, line_offset: int = 1) -> list[str]: ...

    def python_source_policy_failures(self, source: str, *, label: str, line_offset: int = 1) -> list[str]: ...

    def static_policy_failures(self) -> list[str]: ...

    def repository_environment_failures(self) -> list[str]: ...

    def main(self) -> None: ...


def _load_environment_policy() -> EnvironmentPolicy:
    path = Path(__file__).parents[1] / "scripts" / "verify_environment.py"
    spec = importlib.util.spec_from_file_location("repository_environment_policy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("repository environment policy could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(EnvironmentPolicy, module)


verify_environment = _load_environment_policy()


@pytest.mark.parametrize(
    ("command", "label"),
    [
        ("uv venv elsewhere", "alternate environment creation"),
        ("python -m venv elsewhere", "alternate environment creation"),
        ("uvx ast-grep-server", "cache-isolated tool execution"),
        ("uv tool run ast-grep-server", "cache-isolated tool execution"),
        ("uv python install", "managed Python installation"),
        ("uv run --isolated ast-grep-server", "isolated run environment"),
        ("uv run --with pytest ast-grep-server", "isolated run environment"),
        ("uv run -w pytest ast-grep-server", "isolated run environment"),
        ("UV_ISOLATED=1 uv run ast-grep-server", "isolated run environment"),
        ("/home/user/.cache/uv/archive-v0/tool/bin/server", "uv cache environment execution"),
        (r"'C:\Users\user\AppData\Local\uv\cache\archive-v0\tool\server.exe'", "uv cache environment execution"),
        ("/custom/uv-cache/archive-v12/tool/bin/server", "uv cache environment execution"),
        ("mktemp -d", "temporary workspace creation"),
        ("uv run --active pytest", "active environment override"),
        ("UV_PROJECT_ENVIRONMENT=/tmp/outside uv run pytest", "external temporary path"),
        ("export UV_PROJECT_ENVIRONMENT=../outside", "alternate UV_PROJECT_ENVIRONMENT"),
        ("VIRTUAL_ENV=.venv uv run pytest", "virtual environment override"),
        ("sh -c 'echo $(uvx ast-grep-server)'", "cache-isolated tool execution"),
    ],
)
def test_forbidden_shell_commands_are_detected_structurally(command: str, label: str) -> None:
    failures = verify_environment.shell_policy_failures(command, label="test")
    assert any(failure.endswith(label) for failure in failures)


@pytest.mark.parametrize(
    "command",
    [
        "uv sync --locked --all-extras --dev --no-build-isolation --no-python-downloads",
        "UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync pytest",
        "env UV_PROJECT_ENVIRONMENT=.venv uv --directory . run --no-sync python scripts/launch_server.py",
        "mkdir -p test-runtime",
    ],
)
def test_allowed_shell_commands_pass_structural_policy(command: str) -> None:
    assert verify_environment.shell_policy_failures(command, label="test") == []


def test_shell_parser_checks_command_substitutions() -> None:
    failures = verify_environment.shell_policy_failures("value=$(uv python install)", label="test")
    assert any(failure.endswith("managed Python installation") for failure in failures)


def test_shell_parser_fails_closed_on_invalid_syntax() -> None:
    failures = verify_environment.shell_policy_failures("echo '", label="test")
    assert len(failures) == 1
    assert failures[0].startswith("test:1: invalid shell syntax: unexpected EOF")


@pytest.mark.parametrize(
    ("source", "label"),
    [
        ("import tempfile\ntempfile.mkdtemp()\n", "temporary workspace creation"),
        ("from tempfile import TemporaryDirectory as TD\nTD()\n", "temporary workspace creation"),
        ("import subprocess as process\nprocess.run(['uvx', 'ast-grep-server'])\n", "cache-isolated tool execution"),
        ("import os\nos.system('uv run --isolated pytest')\n", "isolated run environment"),
    ],
)
def test_forbidden_python_calls_are_detected_by_ast(source: str, label: str) -> None:
    failures = verify_environment.python_source_policy_failures(source, label="test.py")
    assert any(failure.endswith(label) for failure in failures)


def test_python_parser_ignores_policy_words_in_comments() -> None:
    source = "# uvx and tempfile.TemporaryDirectory are prohibited\nvalue = 'ordinary text'\n"
    assert verify_environment.python_source_policy_failures(source, label="test.py") == []


def test_environment_checker_does_not_import_regular_expressions() -> None:
    source = (verify_environment.ROOT / "scripts" / "verify_environment.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    assert "re" not in imported


def test_current_verification_surfaces_follow_environment_policy() -> None:
    assert verify_environment.static_policy_failures() == []


def test_environment_preflight_rejects_an_external_python_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verify_environment.sys, "prefix", str(verify_environment.ROOT.parent / "outside"))
    with pytest.raises(SystemExit) as error:
        verify_environment.main()
    assert "Python environment must be" in str(error.value)


def test_environment_preflight_rejects_the_wrong_python_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verify_environment, "EXPECTED_PYTHON", (3, 14, 7))
    with pytest.raises(SystemExit) as error:
        verify_environment.main()
    assert "Python version must be 3.14.7" in str(error.value)


def test_environment_preflight_rejects_an_external_project_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(verify_environment.ROOT.parent / "outside"))
    with pytest.raises(SystemExit) as error:
        verify_environment.main()
    assert "UV_PROJECT_ENVIRONMENT must be" in str(error.value)


def test_environment_preflight_rejects_a_non_root_working_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(Path(__file__).parent)
    with pytest.raises(SystemExit) as error:
        verify_environment.main()
    assert "working directory must be" in str(error.value)


def test_launcher_rejects_the_wrong_boundary_before_loading_main(tmp_path: Path) -> None:
    launcher = verify_environment.ROOT / "scripts" / "launch_server.py"
    environment = dict(os.environ)
    environment.pop("VIRTUAL_ENV", None)
    environment["UV_PROJECT_ENVIRONMENT"] = str(tmp_path)

    completed = subprocess.run(
        [sys.executable, str(launcher), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode != 0
    assert "working directory must be" in completed.stderr
    assert "VIRTUAL_ENV must be" in completed.stderr
    assert "UV_PROJECT_ENVIRONMENT must be" in completed.stderr
    assert "No module named" not in completed.stderr


def test_launcher_has_no_top_level_main_import() -> None:
    launcher = verify_environment.ROOT / "scripts" / "launch_server.py"
    tree = ast.parse(launcher.read_text(encoding="utf-8"))
    imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (getattr(node, "module", None) == "main" or any(alias.name == "main" for alias in node.names))
    ]
    assert imports == []
