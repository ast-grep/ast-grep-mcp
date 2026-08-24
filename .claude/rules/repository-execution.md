# Repository execution

- Run development, build, packaging, and verification commands from the repository root.
- Use only the repository-owned `.venv`, synchronized with `uv sync --locked --all-extras --dev --no-python-downloads` while build isolation remains enabled.
- After synchronization, run Python tools through `uv run --no-sync`.
- Never create or run another virtual environment, alternate checkout, copied working tree, packaging workspace, or verification workspace for this repository.
- Do not use `uvx`, `uv tool`, `uv python install`, `uv venv`, `python -m venv`, `virtualenv`, `uv run --isolated`, `uv run --with`, `--active`, `mktemp`, an external temporary path, a uv cache environment, or an alternate `UV_PROJECT_ENVIRONMENT` for development, build, packaging, or verification.
- Build distributions with `uv run --no-sync python -m build --sdist --wheel --no-isolation`, then re-run the synchronization command.
- Validate distribution archives in place. Do not extract or install them into another environment for smoke testing.
- Launch the stdio server through `scripts/launch_server.py`.
