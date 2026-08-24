from __future__ import annotations

import asyncio
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

from fastmcp import Client
from mcp.types import TextContent

from ast_soleaux.server import (
    JsonObject,
    ResolvedExecutable,
    RuntimeServices,
    create_mcp,
    read_postgres_helper_versions,
    resolve_oxc_helper_executable,
)

ROOT = Path(__file__).resolve().parents[1]
POSTGRES_HELPER = ROOT / "postgresql-sidecar" / "bin" / "ast-soleaux-postgresql.mjs"


class ToolResultLike(Protocol):
    @property
    def content(self) -> Sequence[object]: ...

    @property
    def is_error(self) -> bool: ...


def node_helper() -> ResolvedExecutable:
    node = shutil.which("node")
    assert node is not None
    return resolve_oxc_helper_executable(str(POSTGRES_HELPER), working_directory=ROOT)


def runtime(root: Path, helper: ResolvedExecutable, versions: JsonObject) -> RuntimeServices:
    return RuntimeServices(
        working_directory=root,
        executable=ResolvedExecutable(Path("/usr/bin/true"), ("/usr/bin/true",)),
        ast_grep_version="0.45.0",
        config_path=None,
        allowed_roots=(root.resolve(),),
        command_timeout_seconds=30,
        default_max_results=50,
        max_results_cap=500,
        forbid_regex_rules=True,
        postgres_helper=helper,
        postgres_versions=versions,
    )


def result_text(result: ToolResultLike) -> str:
    return "\n".join(block.text for block in result.content if isinstance(block, TextContent))


def test_malformed_deparse_returns_safe_actionable_tool_error(tmp_path: Path) -> None:
    helper = node_helper()
    versions = read_postgres_helper_versions(helper, timeout_seconds=10)

    async def call() -> ToolResultLike:
        async with Client(create_mcp(runtime(tmp_path, helper, versions))) as client:
            return cast(
                ToolResultLike,
                await client.call_tool("postgres_deparse_preview", {"sql": "SELECT FROM"}, raise_on_error=False),
            )

    result = asyncio.run(call())
    assert result.is_error is True
    text = result_text(result)
    assert "PostgreSQL parse error at cursor 11: syntax error at end of input" in text
    for forbidden in ("/Users/", "node_modules", "SqlError", "scan.l", "scanner_yyerror", "lineNumber"):
        assert forbidden not in text
    assert len(text) <= 256
    assert len(text.splitlines()) == 1


def test_unexpected_helper_failure_is_masked(tmp_path: Path) -> None:
    helper_script = tmp_path / "failing_helper.py"
    helper_script.write_text(
        "import sys\nsys.stderr.write('/Users/private/secret node_modules/internal StackTrace')\nraise SystemExit(1)\n",
        encoding="utf-8",
    )
    helper = ResolvedExecutable(helper_script, (sys.executable, str(helper_script)))
    versions: JsonObject = {
        "worker": "0.1.0",
        "parser": "18.0.0",
        "deparser": "18.3.6",
        "postgres_major": 18,
    }

    async def call() -> ToolResultLike:
        async with Client(create_mcp(runtime(tmp_path, helper, versions))) as client:
            return cast(
                ToolResultLike,
                await client.call_tool("postgres_parse", {"sql": "SELECT 1"}, raise_on_error=False),
            )

    result = asyncio.run(call())
    assert result.is_error is True
    text = result_text(result)
    assert "/Users/private/secret" not in text
    assert "node_modules/internal" not in text
    assert "StackTrace" not in text
