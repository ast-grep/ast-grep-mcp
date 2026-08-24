---
name: pytest
description: Use when creating, repairing, organizing, running, or reporting pytest tests in this repository - designs behavior-level tests, safe fixtures, readable parametrization, focused execution, and outcome-first reporting.
---

# Pytest

## Contract

- Treat the applicable pytest selection as the functional acceptance owner: every relevant test outcome matters.
- Treat collected-test counts and coverage measurements as diagnostic evidence, never as substitutes for relevant tests passing.
- Keep the workflow repository-native. Use the locked environment, installed plugins, owner-provided suites, and platform checks already defined here.

## Establish authority

- Work from the repository root in its locked `.venv`.
- Inspect `pyproject.toml`, `uv.lock`, `tests/conftest.py`, CI, and the installed pytest and plugin versions before changing test configuration or commands.
- Check version-matched official documentation for current behavior. Preserve repository-owned wrappers, suites, coverage scope, and platform gates.

## Create behavior tests

- Define the contract first: inputs, outputs, errors, state changes, external effects, and invariants.
- For a reported defect, reproduce it and confirm the test fails without the fix when feasible.
- Name tests for observable behavior. Group tests in classes only when the grouping adds a shared behavioral context.
- Use plain `assert` statements. Use `pytest.raises(..., match=...)`, `pytest.raises(..., check=...)`, or `pytest.warns(...)` only when the message, predicate, or warning is part of the contract.
- Prefer a focused example over assertions against private implementation details. Add property tests only for useful invariants such as round trips, ordering, or input boundaries.

## Design fixtures safely

- Use the narrowest valid scope: `function`, `class`, `module`, `package`, or `session`.
- Put a fixture in `conftest.py` only when tests in that directory tree genuinely share it.
- Keep each fixture to one state-changing operation paired with its teardown. Prefer `yield` fixtures so every successful mutation has adjacent cleanup even when later setup fails.
- Prefer `tmp_path` and `monkeypatch` for isolated filesystem and process state. Use factory fixtures when one test needs several related instances.
- Avoid autouse or session-scoped mutable state unless the behavior requires it and isolation is proven.

## Parametrize readable behavior

- Parametrize examples only when they exercise the same behavior.
- Use `pytest.param(..., id="behavior")` when pytest's generated ID would be positional, ambiguous, or unstable.
- Treat parameter values as shared objects: pytest passes them as-is without copying, so do not mutate a value that another case can observe.

## Mock boundaries

- Mock external boundaries such as subprocesses, clocks, randomness, networks, or third-party services; do not mock the unit under test.
- Use the repository's installed plugins only when they simplify the behavior under test. `mocker` is a convenience, not a requirement over `unittest.mock`.
- Do not add pytest plugins or dependencies unless a demonstrated repository requirement justifies them.

## Execute owner-provided suites

- Run a focused test first with `uv run --no-sync pytest <test-selector>`.
- Run the unit, configuration, mutation, backend, and contract suites after shared test configuration, fixtures, or policy changes.
- Preserve the explicit coverage source: use `--cov=ast_soleaux`, not bare `--cov`. Use `--cov-report=term-missing` only as diagnostic evidence; coverage does not gate test success in this repository.
- Run integration tests only with the repository's required ast-grep executable and version. Run platform-specific owner checks when the changed behavior crosses a platform boundary.

## Report outcomes

- Derive the verdict from whether every relevant pytest test passed. A collected-test count is informational and is never the acceptance criterion.
- Lead with `PASS`, `FAIL`, or `BLOCKED`, then name the suite or command. With `-ra`, report skips, xfails, xpasses, errors, and failures separately when present.
- If coverage was collected, describe it as optional diagnostic evidence scoped to the `ast_soleaux` package. Never use coverage to override the result of the relevant tests.
- Never summarize verification as only a test count and percentage. Name skipped checks and do not claim checks that did not run.

## Official references

- [pytest fixtures and safe teardown](https://docs.pytest.org/en/stable/how-to/fixtures.html)
- [pytest parametrization](https://docs.pytest.org/en/stable/how-to/parametrize.html)
- [pytest output and `-ra`](https://docs.pytest.org/en/stable/how-to/output.html)
- [pytest exit codes](https://docs.pytest.org/en/stable/reference/exit-codes.html)
- [pytest configuration reference](https://docs.pytest.org/en/stable/reference/reference.html)
- [pytest-cov configuration](https://pytest-cov.readthedocs.io/en/stable/config.html)
