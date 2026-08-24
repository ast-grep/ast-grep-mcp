---
name: pytest
description: Use when creating, repairing, organizing, running, or reporting pytest tests in this repository - applies behavior-first design, isolated fixtures, readable parametrization, focused execution, and outcome-first reporting.
---

# Pytest

## Contract

Passing every relevant pytest test owns functional acceptance. A collected-test count and any coverage percentage are diagnostic evidence only; neither can replace, weaken, or override the applicable test outcomes.

Follow the active repository rules for the locked environment, test-runtime cleanup, platform checks, and Linux execution. Inspect `pyproject.toml`, `uv.lock`, `tests/conftest.py`, CI, and installed pytest plugins before changing test configuration or commands.

## Design observable tests

- Define the behavior first: inputs, outputs, errors, state transitions, external effects, and invariants.
- Reproduce a reported defect before fixing it and confirm the reproduction fails without the repair when feasible.
- Name tests for observable behavior and use plain `assert` statements.
- Assert exception messages, warning text, or predicates only when they are part of the contract.
- Avoid assertions against private implementation details when a focused public example proves the same behavior.

## Isolate setup and teardown

- Give fixtures the narrowest valid scope and place them in `conftest.py` only when its directory tree shares them.
- Pair each state-changing setup operation with adjacent teardown, preferably through a `yield` fixture.
- Prefer `tmp_path` and `monkeypatch` for filesystem, environment, and process state.
- Use factory fixtures when one test needs several related instances; avoid mutable autouse or session state unless isolation is proven.

## Parametrize and mock deliberately

- Parametrize only examples of the same behavior and assign stable behavioral IDs when generated IDs would be ambiguous.
- Do not mutate parameter values that another case can observe because pytest passes them without copying.
- Mock clocks, randomness, subprocesses, networks, and third-party services at their boundaries; do not mock the unit under test.
- Use only installed plugins and add no dependency without a demonstrated repository requirement.

## Execute and report

- Run the smallest relevant selector first, then every owner-provided suite affected by shared fixtures, configuration, or policy changes.
- Preserve explicit coverage sources when coverage is useful, but report it only after the pytest outcome and only as diagnostic evidence.
- Use pytest's summary to surface failures, errors, skips, xfails, and xpasses. Lead the report with `PASS`, `FAIL`, or `BLOCKED` and name the command or suite.
- Never reduce verification to a collected-test count and coverage percentage. Name every relevant skipped check.

## Official references

- [pytest 8.4 fixtures and safe teardown](https://docs.pytest.org/en/8.4.x/how-to/fixtures.html)
- [pytest 8.4 parametrization](https://docs.pytest.org/en/8.4.x/how-to/parametrize.html)
- [pytest 8.4 output and summaries](https://docs.pytest.org/en/8.4.x/how-to/output.html)
- [pytest 8.4 exit codes](https://docs.pytest.org/en/8.4.x/reference/exit-codes.html)
- [pytest-cov configuration](https://pytest-cov.readthedocs.io/en/stable/config.html)
