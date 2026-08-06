from __future__ import annotations

import os
import shutil
import subprocess

from verify_environment import (
    EXPECTED_ENVIRONMENT,
    ROOT,
    repository_environment_failures,
    resolve_from_root,
)


def _pin_project_environment() -> None:
    configured = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if configured is not None and resolve_from_root(configured) != EXPECTED_ENVIRONMENT:
        raise SystemExit(f"UV_PROJECT_ENVIRONMENT must be {EXPECTED_ENVIRONMENT}")
    os.environ["UV_PROJECT_ENVIRONMENT"] = ".venv"


def _check_locked_environment() -> None:
    """Confirm the environment is synchronized using only the standard library.

    Importing the server first would fail on a partially synchronized environment
    and report a missing dependency traceback, hiding the unsynchronized state
    this check exists to name.
    """
    executable = shutil.which("uv")
    if executable is None:
        raise SystemExit("uv is required to verify the repository environment")
    try:
        completed = subprocess.run(
            [
                executable,
                "--directory",
                str(ROOT),
                "--no-python-downloads",
                "sync",
                "--no-active",
                "--locked",
                "--check",
                "--all-extras",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=ROOT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise SystemExit(f"repository environment check could not run: {error}") from error
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or completed.stdout.strip() or "uv returned no diagnostic"
        raise SystemExit(f"repository environment is not synchronized: {diagnostic}")


def main() -> None:
    failures = repository_environment_failures()
    if failures:
        raise SystemExit("repository environment verification failed:\n" + "\n".join(failures))
    _pin_project_environment()
    _check_locked_environment()
    from main import run_mcp_server

    run_mcp_server()


if __name__ == "__main__":
    main()
