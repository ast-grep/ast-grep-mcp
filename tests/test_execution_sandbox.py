from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "execution-sidecar" / "ast_soleaux_typescript_sandbox.py"


def request(project: Path, profile: str, *, timeout: float = 2) -> dict[str, object]:
    return {
        "project_root": str(project),
        "entry": "entry.ts",
        "args": [],
        "stdin": None,
        "timeout_seconds": timeout,
        "profile": profile,
    }


def fake_docker(tmp_path: Path, behavior: str) -> tuple[Path, Path]:
    directory = tmp_path / "bin"
    directory.mkdir()
    log = tmp_path / "docker-log.jsonl"
    script = directory / "docker"
    log_statement = (
        "with log.open('a', encoding='utf-8') as stream: "
        "stream.write(json.dumps({'args': sys.argv[1:], "
        "'secret': os.environ.get('AST_SOLEAUX_TEST_SECRET')}) + '\\n')"
    )
    script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, os, pathlib, sys, time",
                f"log = pathlib.Path({str(log)!r})",
                log_statement,
                "if sys.argv[1:3] == ['rm', '--force']: raise SystemExit(0)",
                "mounts = [sys.argv[index + 1] for index, value in enumerate(sys.argv) if value == '--mount']",
                "def source_for(destination):",
                "    match = next(value for value in mounts if any(part == f'dst={destination}' for part in value.split(',')))",
                "    return pathlib.Path(next(part[4:] for part in match.split(',') if part.startswith('src=')))",
                f"behavior = {behavior!r}",
                "if behavior == 'sleep': time.sleep(5)",
                "elif behavior == 'large': print('x' * (4 * 1024 * 1024 + 512))",
                "else:",
                "    workdir = sys.argv[sys.argv.index('--workdir') + 1]",
                "    root = source_for('/workspace' if workdir == '/workspace' else '/work')",
                "    (root / 'result.txt').write_text('written', encoding='utf-8')",
                "    print('secret=' + str(os.environ.get('AST_SOLEAUX_TEST_SECRET', 'missing')))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return directory, log


def run_runner(project: Path, profile: str, fake_path: Path, *, timeout: float = 2) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_path}:{environment.get('PATH', '')}"
    environment["AST_SOLEAUX_TEST_SECRET"] = "top-secret"
    return subprocess.run(
        [sys.executable, str(RUNNER)],
        input=json.dumps(request(project, profile, timeout=timeout)),
        capture_output=True,
        check=False,
        text=True,
        env=environment,
    )


def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "entry.ts").write_text("console.log('entry')\n", encoding="utf-8")
    return root


def test_typescript_sandbox_reports_exact_protocol_version() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--version-json"],
        capture_output=True,
        check=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "worker": "0.1.0",
        "runtime": "@oxc-node/core@0.1.0",
        "sandbox": "docker",
    }


def test_typescript_sandbox_strictly_rejects_unknown_fields_and_escapes(tmp_path: Path) -> None:
    root = project(tmp_path)
    payload = request(root, "isolated")
    payload["unknown"] = True
    unknown = subprocess.run(
        [sys.executable, str(RUNNER)],
        input=json.dumps(payload),
        capture_output=True,
        check=False,
        text=True,
    )
    assert unknown.returncode != 0
    assert "extra_forbidden" in unknown.stderr

    escaped = request(root, "isolated")
    escaped["entry"] = "../outside.ts"
    outside = subprocess.run(
        [sys.executable, str(RUNNER)],
        input=json.dumps(escaped),
        capture_output=True,
        check=False,
        text=True,
    )
    assert outside.returncode != 0
    assert "contained relative path" in outside.stderr


@pytest.mark.parametrize(
    ("profile", "network", "writes", "read_only"),
    [
        ("isolated", False, False, True),
        ("workspace-write", False, True, False),
        ("networked", True, False, True),
    ],
)
def test_typescript_sandbox_profiles_sanitize_environment_and_track_writes(
    tmp_path: Path,
    profile: str,
    network: bool,
    writes: bool,
    read_only: bool,
) -> None:
    root = project(tmp_path)
    fake_path, log = fake_docker(tmp_path, "success")
    completed = run_runner(root, profile, fake_path)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["stdout"].strip() == "secret=missing"
    assert result["written_files"] == ["result.txt"]
    assert result["audit"]["network"] is network
    assert result["audit"]["project_writes"] is writes
    assert result["audit"]["project_read_only"] is read_only
    assert result["audit"]["sanitized_environment"] is True
    assert result["audit"]["no_shell"] is True
    assert result["audit"]["limits"] == {
        "timeout_seconds": 2.0,
        "pids": 64,
        "memory_bytes": 512 * 1024 * 1024,
        "cpus": 1,
        "stdout_bytes": 4 * 1024 * 1024,
        "stderr_bytes": 64 * 1024,
        "written_files": 256,
        "written_file_bytes": 16 * 1024 * 1024,
        "written_total_bytes": 64 * 1024 * 1024,
    }
    call = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert call["secret"] is None
    args = call["args"]
    assert ("--network" in args) is (not network)
    assert {"--read-only", "--cap-drop", "--security-opt", "--pids-limit", "--memory", "--cpus", "--ulimit"} <= set(args)
    assert "ALL" in args
    assert "no-new-privileges" in args
    assert "fsize=16777216:16777216" in args
    project_mount = next(args[index + 1] for index, value in enumerate(args) if value == "--mount" and "dst=/workspace" in args[index + 1])
    assert ("readonly" in project_mount.split(",")) is read_only


def test_typescript_sandbox_timeout_removes_container(tmp_path: Path) -> None:
    root = project(tmp_path)
    fake_path, log = fake_docker(tmp_path, "sleep")
    completed = run_runner(root, "isolated", fake_path, timeout=0.1)
    assert completed.returncode == 0
    result = json.loads(completed.stdout)
    assert result["timed_out"] is True
    assert result["exit_code"] == 124
    calls = [json.loads(line)["args"] for line in log.read_text(encoding="utf-8").splitlines()]
    assert any(call[:2] == ["rm", "--force"] for call in calls)


def test_typescript_sandbox_bounds_stdout(tmp_path: Path) -> None:
    root = project(tmp_path)
    fake_path, _ = fake_docker(tmp_path, "large")
    completed = run_runner(root, "isolated", fake_path)
    assert completed.returncode == 0
    result = json.loads(completed.stdout)
    assert result["stdout_truncated"] is True
    assert len(result["stdout"].encode("utf-8")) <= 4 * 1024 * 1024


@pytest.mark.skipif(os.environ.get("AST_SOLEAUX_RUN_DOCKER_PROBES") != "1", reason="real Docker probe is opt-in")
def test_real_docker_isolated_profile_denies_project_writes_and_network(tmp_path: Path) -> None:
    root = project(tmp_path)
    (root / "entry.ts").write_text(
        "const { writeFileSync } = require('node:fs');\n"
        "async function main() {\n"
        "  try { writeFileSync('/workspace/blocked.txt', 'x'); } catch { console.log('write-blocked'); }\n"
        "  try { await fetch('https://example.com'); } catch { console.log('network-blocked'); }\n"
        "}\n"
        "void main();\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(RUNNER)],
        input=json.dumps(request(root, "isolated", timeout=60)),
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert "write-blocked" in result["stdout"]
    assert "network-blocked" in result["stdout"]
    assert not (root / "blocked.txt").exists()
