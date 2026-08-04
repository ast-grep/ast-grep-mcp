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

# Writing code here

Read the `python` skill and the workflow it routes to before editing.

- Source and tests carry no comments; `test_repository_python_carries_no_comment_tokens` fails on any that appear. Explain intent in a docstring.
- Argv is the first argument to `subprocess.run`, the second to `os.execv`, and the third to `os.spawnv`. Read the signature before binding a position.

# Fixing a reported defect

- Reproduce the defect first, and confirm the reproduction fails once the fix is removed.
- Fix the class, not the instance. A guard keyed on name plus suffix still collides when suffixes differ; a symlink rejection still admits a junction.
- Check the adjacent guards. Both incomplete fixes here left an existing predicate uncalled.
- Reply on the thread with the evidence when dismissing a finding, or it returns next revision.

# Filesystem behaviour that differs by platform

A type check cannot reach these; only executing the test on the other platform can. When a change touches path traversal, permissions, or process teardown, run its test on Linux before pushing, and read the assertion below before writing a new one.

- Inode numbers do not identify a path across a change. Removing a directory frees its inode immediately, and Linux gives the same number to whatever is created in its place, so a replacement compares equal to what it replaced. Windows does not report a stable inode across calls at all. Compare the file type, which a link cannot match.
- `chmod` sets only the read-only flag on Windows; every other bit is ignored. Guard mode assertions with `os.name == "posix"`.
- `os.scandir` receives a `Path` for source trees and a `str` or descriptor elsewhere, and directory entries arrive in filesystem order. A test that patches it must accept every form and must not assume which entry is reached first.
- Resolving a path that was just replaced with a symlink follows the link to its target, so a comparison against the pre-swap path stops matching. Match on the entry name instead.

# Running the suite on Linux

CI is the slowest way to learn that a test is platform-dependent. Reproduce it locally instead, in a container, which is a separate machine rather than an alternate environment for this checkout and so does not cross the execution boundary above.

```bash
docker run --rm -v "$PWD":/w -w /w python:3.14-slim bash -c '
pip install -q "mcp[cli]==2.0.0" "pydantic>=2.11.0,<3" "pyyaml>=6.0.2,<7" \
  pytest pytest-asyncio pytest-mock bashlex==0.18 "markdown-it-py>=3,<4"
python -m pytest tests/test_config_snapshot.py tests/test_environment_policy.py \
  -p no:cacheprovider -o addopts="" -q'
```

`-o addopts=""` drops the coverage flags the container does not install, and `-p no:cacheprovider` keeps the container from writing a cache into the working tree. A security check needs the stronger form: confirm the test fails once the check is removed, on that same platform. A test that passes either way proves nothing.

# Removing the pytest runtime tree

`tests/conftest.py` clears `test-runtime/` after a run. Snapshot bundles are deliberately read-only, so a manual `rm -rf test-runtime` fails partway and leaves a tree that breaks the next run with unrelated collection errors. Use `chmod -R u+rwX test-runtime && rm -rf test-runtime`.
