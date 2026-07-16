from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from main import AstGrepService, build_runtime

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def actual_service(root: Path, *, forbid_regex_rules: bool = False) -> AstGrepService:
    executable = shutil.which("ast-grep")
    if executable is None:
        pytest.fail("ast-grep must be installed for integration tests")
    runtime = build_runtime(
        working_directory=root,
        ast_grep_executable=executable,
        allowed_roots=[str(root)],
        forbid_regex_rules=forbid_regex_rules,
    )
    assert runtime.ast_grep_version == "0.44.1"
    return AstGrepService(runtime)


def test_pattern_search_is_bounded_and_project_relative() -> None:
    service = actual_service(REPOSITORY_ROOT)

    result = service.find_code(
        project_folder=str(FIXTURES),
        pattern="def $NAME($$$): $$$BODY",
        language="python",
        paths=["example.py"],
        include_globs=None,
        exclude_globs=None,
        max_results=1,
    )

    assert result["returned"] == 1
    assert result["truncated"] is True
    assert result["limit"] == 1
    assert result["matches"][0]["file"] == "example.py"
    assert "def hello" in result["matches"][0]["text"]


def test_rule_search_honors_include_and_exclude_globs(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "keep.py").write_text("print('keep')\n", encoding="utf-8")
    (source / "skip_test.py").write_text("print('skip')\n", encoding="utf-8")
    service = actual_service(tmp_path)

    result = service.find_code_by_rule(
        project_folder=str(tmp_path),
        rule_yaml="id: print-calls\nlanguage: python\nrule:\n  pattern: print($A)\n",
        paths=["src"],
        include_globs=["**/*.py"],
        exclude_globs=["**/*_test.py"],
        max_results=10,
    )

    assert result["returned"] == 1
    assert result["truncated"] is False
    assert result["matches"][0]["file"] == "src/keep.py"


def test_valid_negative_probe_returns_empty_matches() -> None:
    service = actual_service(REPOSITORY_ROOT)

    matches = service.test_match_code_rule(
        code="value = 1",
        rule_yaml="id: no-call\nlanguage: python\nrule:\n  pattern: print($A)\n",
    )

    assert matches == []


def test_regex_policy_rejects_regex_rule_before_ast_grep_runs() -> None:
    service = actual_service(REPOSITORY_ROOT, forbid_regex_rules=True)

    with pytest.raises(ValueError, match="Regex ast-grep rules are disabled"):
        service.test_match_code_rule(
            code="value = 1",
            rule_yaml="id: regex\nlanguage: python\nrule:\n  regex: value.*\n",
        )


@pytest.mark.asyncio
async def test_stdio_protocol_catalog_annotations_metadata_and_search_contract() -> None:
    executable = shutil.which("ast-grep")
    if executable is None:
        pytest.fail("ast-grep must be installed for protocol tests")
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[
            str(REPOSITORY_ROOT / "main.py"),
            "--ast-grep",
            executable,
            "--allowed-root",
            str(REPOSITORY_ROOT),
            "--forbid-regex-rules",
        ],
        cwd=REPOSITORY_ROOT,
    )

    async with stdio_client(parameters) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            listed = await session.list_tools()
            tools = {tool.name: tool for tool in listed.tools}
            assert set(tools) == {
                "dump_syntax_tree",
                "test_match_code_rule",
                "find_code",
                "find_code_by_rule",
                "get_server_info",
            }
            for tool in tools.values():
                assert tool.annotations is not None
                assert tool.annotations.readOnlyHint is True
                assert tool.annotations.destructiveHint is False
                assert tool.annotations.idempotentHint is True
                assert tool.annotations.openWorldHint is False

            find_schema = tools["find_code"].inputSchema
            assert {"project_folder", "pattern", "language"}.issubset(find_schema["required"])

            info_result = await session.call_tool("get_server_info", {})
            assert info_result.isError is False
            assert info_result.structuredContent is not None
            assert info_result.structuredContent["fork_version"] == "0.2.0"
            assert info_result.structuredContent["ast_grep_version"] == "0.44.1"
            assert info_result.structuredContent["forbid_regex_rules"] is True
            assert info_result.structuredContent["default_max_results"] == 50
            assert info_result.structuredContent["max_results_cap"] == 500

            negative = await session.call_tool(
                "test_match_code_rule",
                {
                    "code": "value = 1",
                    "yaml": "id: no-call\nlanguage: python\nrule:\n  pattern: print($A)\n",
                },
            )
            assert negative.isError is False
            assert negative.structuredContent == {"result": []}

            found = await session.call_tool(
                "find_code",
                {
                    "project_folder": str(FIXTURES),
                    "pattern": "def $NAME($$$): $$$BODY",
                    "language": "python",
                    "paths": ["example.py"],
                    "max_results": 1,
                    "output_format": "json",
                },
            )
            assert found.isError is False
            assert found.structuredContent is not None
            assert set(found.structuredContent) == {"matches", "returned", "truncated", "limit"}
            assert found.structuredContent["returned"] == 1
            assert found.structuredContent["truncated"] is True
            assert found.structuredContent["limit"] == 1

            rejected = await session.call_tool(
                "find_code_by_rule",
                {
                    "project_folder": str(FIXTURES),
                    "yaml": "id: regex\nlanguage: python\nrule:\n  regex: value.*\n",
                },
            )
            assert rejected.isError is True
