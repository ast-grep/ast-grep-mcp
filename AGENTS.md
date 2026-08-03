# Repository execution boundary

- Run development, build, packaging, and verification commands from this repository root.
- Use only the repository-owned `.venv`, synchronized with `uv sync --locked --all-extras --dev --no-build-isolation --no-python-downloads`.
- After synchronization, run Python tools through `uv run --no-sync`.
- Never create or run another virtual environment for this repository.
- Never create an alternate checkout, copied working tree, packaging workspace, or verification workspace outside this repository.
- Do not use `uvx`, `uv tool`, `uv python install`, `uv venv`, `python -m venv`, `virtualenv`, `uv run --isolated`, `uv run --with`, `--active`, `mktemp`, an external temporary path, a uv cache environment, or an alternate `UV_PROJECT_ENVIRONMENT` in development, build, packaging, or verification commands.
- Build distributions with `python -m build --sdist --wheel --no-isolation` through the locked repository environment.
- Validate distribution archives in place. Do not extract or install them into another environment for smoke testing.
- Run `uv run --no-sync python scripts/verify_environment.py` after synchronization and before verification. Treat any failure as a hard failure.
- Launch the stdio server through `scripts/launch_server.py`; it checks the lock, synchronized environment, and repository boundary before serving.
