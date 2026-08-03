# Repository execution boundary

- Run development, build, packaging, and verification commands from this repository root.
- Use only the repository-owned `.venv`, synchronized with `uv sync --locked --all-extras --dev --no-python-downloads`.
- Keep build isolation enabled when synchronizing. `--no-build-isolation` suppresses provisioning of `build-system.requires`, so a clean environment cannot build this package and reports `ModuleNotFoundError: No module named 'setuptools'`.
- After synchronization, run Python tools through `uv run --no-sync`.
- Never create or run another virtual environment for this repository.
- Never create an alternate checkout, copied working tree, packaging workspace, or verification workspace outside this repository.
- Do not use `uvx`, `uv tool`, `uv python install`, `uv venv`, `python -m venv`, `virtualenv`, `uv run --isolated`, `uv run --with`, `--active`, `mktemp`, an external temporary path, a uv cache environment, or an alternate `UV_PROJECT_ENVIRONMENT` in development, build, packaging, or verification commands.
- Build distributions with `python -m build --sdist --wheel --no-isolation` through the locked repository environment.
- Validate distribution archives in place. Do not extract or install them into another environment for smoke testing.
- Run `uv run --no-sync python scripts/verify_environment.py` after synchronization and before verification. Treat any failure as a hard failure.
- Launch the stdio server through `scripts/launch_server.py`; it checks the lock, synchronized environment, and repository boundary before serving.

# Cross-platform verification

`uv run --no-sync` reuses the already-populated `.venv`, and the default type-check and test runs describe the host platform only. Both hide failures that the Windows and clean-runner jobs report, so verification also covers the following before a change is considered verified.

- Type-check the other platforms, because platform-specific stubs change which branches resolve: `uv run --no-sync mypy --platform win32 main.py config_snapshot.py scripts tests` and `uv run --no-sync pyright --pythonplatform Windows`.
- Treat byte offsets, digests, and file contents as line-ending dependent. Fixtures write through `newline=""`, and assertions derive offsets from the bytes on disk rather than hardcoding values that only hold under LF.
- Confirm a clean environment can still build the project. `uv run --no-sync` never exercises the build, so a missing build dependency stays invisible locally while every CI job fails at its first step.
