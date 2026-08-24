from __future__ import annotations

import io
import subprocess
import tokenize
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".pyright",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "test-runtime",
        "venv",
    }
)
FORBIDDEN_IDENTIFIER = "Any"


def repository_python_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.py"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    files: list[Path] = []
    for relative_name in completed.stdout.splitlines():
        relative_path = Path(relative_name)
        if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in relative_path.parts):
            continue
        path = REPOSITORY_ROOT / relative_path
        if path.is_file():
            files.append(path)
    return sorted(files)


def forbidden_identifier_locations(path: Path) -> list[tuple[int, int]]:
    source = path.read_text(encoding="utf-8")
    return [
        token.start
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.NAME and token.string == FORBIDDEN_IDENTIFIER
    ]


def test_repository_has_no_forbidden_dynamic_type_identifier() -> None:
    findings = {
        path.relative_to(REPOSITORY_ROOT).as_posix(): forbidden_identifier_locations(path)
        for path in repository_python_files()
        if forbidden_identifier_locations(path)
    }
    assert findings == {}
