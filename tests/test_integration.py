from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import mcp.client.stdio as stdio_module
import pytest
from mcp import Client, StdioServerParameters, stdio_client
from mcp.types import TextContent, Tool
from mcp.types.version import LATEST_HANDSHAKE_VERSION, LATEST_MODERN_VERSION

from main import (
    HARD_MAX_RESULTS,
    SUPPORTED_AST_GREP_VERSION,
    AstGrepService,
    build_runtime,
    resolve_ast_grep_executable,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
AST_GREP_TEST_EXECUTABLE = "AST_GREP_TEST_EXECUTABLE"
EXPECTED_TOOLS = {
    "dump_syntax_tree": "Dump syntax tree",
    "test_match_code_rule": "Test rule against code",
    "find_code": "Find code by pattern",
    "find_code_by_rule": "Find code by rule",
    "get_server_info": "Get server info",
    "outline_code": "Outline code",
}
pytestmark = [
    pytest.mark.filterwarnings("error::DeprecationWarning"),
    pytest.mark.filterwarnings("error::mcp.shared.exceptions.MCPDeprecationWarning"),
]


@pytest.fixture(scope="session")
def ast_grep_executable() -> str:
    raw_executable = os.environ.get(AST_GREP_TEST_EXECUTABLE)
    if not raw_executable:
        pytest.fail(
            f"{AST_GREP_TEST_EXECUTABLE} must point to an explicit ast-grep {SUPPORTED_AST_GREP_VERSION} executable",
            pytrace=False,
        )

    try:
        resolved = resolve_ast_grep_executable(raw_executable, working_directory=REPOSITORY_ROOT)
        completed = subprocess.run(
            [*resolved.command_prefix, "--version"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        pytest.fail(f"Could not execute {AST_GREP_TEST_EXECUTABLE}={raw_executable!r}: {error}", pytrace=False)

    expected = f"ast-grep {SUPPORTED_AST_GREP_VERSION}"
    reported = completed.stdout.strip()
    if completed.returncode != 0 or reported != expected:
        pytest.fail(
            f"{AST_GREP_TEST_EXECUTABLE} must report exactly {expected!r}; got exit {completed.returncode} and stdout {reported!r}",
            pytrace=False,
        )
    return raw_executable


@pytest.fixture(scope="session")
def console_entrypoint() -> str:
    executable = shutil.which("ast-grep-server")
    if executable is None:
        pytest.fail(
            "The installed ast-grep-server console entrypoint was not found on PATH; run tests through the locked uv environment",
            pytrace=False,
        )
    return str(Path(executable).resolve())


def actual_service(
    root: Path,
    ast_grep_executable: str,
    *,
    forbid_regex_rules: bool = False,
) -> AstGrepService:
    runtime = build_runtime(
        working_directory=root,
        ast_grep_executable=ast_grep_executable,
        allowed_roots=[str(root)],
        forbid_regex_rules=forbid_regex_rules,
    )
    assert runtime.ast_grep_version == SUPPORTED_AST_GREP_VERSION
    return AstGrepService(runtime)


def server_parameters(console_entrypoint: str, ast_grep_executable: str) -> StdioServerParameters:
    return StdioServerParameters(
        command=console_entrypoint,
        args=[
            "--ast-grep",
            ast_grep_executable,
            "--allowed-root",
            str(REPOSITORY_ROOT),
            "--forbid-regex-rules",
        ],
        cwd=REPOSITORY_ROOT,
    )


def _schema_contains(schema: Any, key: str, expected: Any) -> bool:
    if isinstance(schema, dict):
        if schema.get(key) == expected:
            return True
        return any(_schema_contains(value, key, expected) for value in schema.values())
    if isinstance(schema, list):
        return any(_schema_contains(value, key, expected) for value in schema)
    return False


def _count_outline_nodes(files: list[dict[str, Any]]) -> int:
    def count(nodes: list[dict[str, Any]]) -> int:
        return sum(1 + count(node.get("members", [])) for node in nodes)

    return sum(count(file_result.get("items", [])) for file_result in files)


def _assert_tool_catalog(tools: dict[str, Tool]) -> None:
    assert set(tools) == set(EXPECTED_TOOLS)
    expected_properties = {
        "dump_syntax_tree": {"code", "language", "format"},
        "test_match_code_rule": {"code", "yaml"},
        "find_code": {
            "project_folder",
            "pattern",
            "language",
            "paths",
            "include_globs",
            "exclude_globs",
            "max_results",
            "output_format",
        },
        "find_code_by_rule": {
            "project_folder",
            "yaml",
            "paths",
            "include_globs",
            "exclude_globs",
            "max_results",
            "output_format",
            "include_metadata",
        },
        "get_server_info": set(),
        "outline_code": {"project_folder", "paths", "language", "max_results", "output_format"},
    }
    expected_required = {
        "dump_syntax_tree": {"code", "language"},
        "test_match_code_rule": {"code", "yaml"},
        "find_code": {"project_folder", "pattern", "language"},
        "find_code_by_rule": {"project_folder", "yaml"},
        "get_server_info": set(),
        "outline_code": {"project_folder", "paths"},
    }

    for name, tool in tools.items():
        assert tool.title == EXPECTED_TOOLS[name]
        assert tool.annotations is not None
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.destructive_hint is False
        assert tool.annotations.idempotent_hint is True
        assert tool.annotations.open_world_hint is False
        assert tool.input_schema["type"] == "object"
        assert set(tool.input_schema.get("properties", {})) == expected_properties[name]
        assert set(tool.input_schema.get("required", [])) == expected_required[name]
        assert tool.output_schema is not None
        assert tool.output_schema["type"] == "object"

    for search_tool_name in ("find_code", "find_code_by_rule"):
        search_schema = tools[search_tool_name].output_schema
        assert search_schema is not None
        assert set(search_schema["properties"]) == {"matches", "returned", "truncated", "limit"}
        assert set(search_schema["required"]) == {"matches", "returned", "truncated", "limit"}
        assert _schema_contains(tools[search_tool_name].input_schema["properties"]["max_results"], "maximum", 500)

    outline_schema = tools["outline_code"].output_schema
    assert outline_schema is not None
    assert set(outline_schema["properties"]) == {"files", "returned", "truncated", "limit"}
    assert set(outline_schema["required"]) == {"files", "returned", "truncated", "limit"}
    outline_input = tools["outline_code"].input_schema["properties"]
    assert _schema_contains(outline_input["paths"], "minItems", 1)
    assert _schema_contains(outline_input["paths"], "maxItems", 64)
    assert outline_input["language"]["default"] is None
    assert _schema_contains(outline_input["max_results"], "maximum", HARD_MAX_RESULTS)
    assert outline_input["output_format"]["default"] == "text"
    assert tools["find_code_by_rule"].input_schema["properties"]["include_metadata"]["default"] is False

    assert set(tools["dump_syntax_tree"].output_schema["properties"]) == {"result"}  # type: ignore[index]
    assert set(tools["test_match_code_rule"].output_schema["properties"]) == {"result"}  # type: ignore[index]
    assert set(tools["get_server_info"].output_schema["properties"]) == {  # type: ignore[index]
        "fork_version",
        "ast_grep_executable",
        "ast_grep_version",
        "config_path",
        "allowed_roots",
        "command_timeout_seconds",
        "default_max_results",
        "max_results_cap",
        "forbid_regex_rules",
    }


def test_pattern_search_is_bounded_and_project_relative(ast_grep_executable: str) -> None:
    service = actual_service(REPOSITORY_ROOT, ast_grep_executable)

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


def test_rule_search_honors_include_and_exclude_globs(
    tmp_path: Path,
    ast_grep_executable: str,
) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "keep.py").write_text("print('keep')\n", encoding="utf-8")
    (source / "skip_test.py").write_text("print('skip')\n", encoding="utf-8")
    service = actual_service(tmp_path, ast_grep_executable)

    result = service.find_code_by_rule(
        project_folder=str(tmp_path),
        rule_yaml="id: print-calls\nlanguage: python\nrule:\n  pattern: print($A)\n",
        paths=["src"],
        include_globs=["**/*.py"],
        exclude_globs=["**/*_test.py"],
        max_results=10,
        include_metadata=False,
    )

    assert result["returned"] == 1
    assert result["truncated"] is False
    assert result["matches"][0]["file"] == "src/keep.py"


def test_search_does_not_load_implicit_project_config(
    tmp_path: Path,
    ast_grep_executable: str,
) -> None:
    (tmp_path / "sgconfig.yml").write_text(
        'ruleDirs: []\nlanguageGlobs:\n  Python:\n    - "*.txt"\n',
        encoding="utf-8",
    )
    (tmp_path / "sample.txt").write_text("print('implicit')\n", encoding="utf-8")
    service = actual_service(tmp_path, ast_grep_executable)

    result = service.find_code(
        project_folder=str(tmp_path),
        pattern="print($A)",
        language="python",
        paths=["."],
        include_globs=None,
        exclude_globs=None,
        max_results=10,
    )

    assert result["returned"] == 0
    assert result["matches"] == []


def test_stream_matches_use_the_stable_ndjson_shape(
    tmp_path: Path,
    ast_grep_executable: str,
) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "a.py").write_text("print('a')\nprint('b')\n", encoding="utf-8")
    service = actual_service(tmp_path, ast_grep_executable)

    result = service.find_code(
        project_folder=str(tmp_path),
        pattern="print($A)",
        language="python",
        paths=["src"],
        include_globs=None,
        exclude_globs=None,
        max_results=10,
    )

    assert result["returned"] == 2
    for match in result["matches"]:
        assert {"text", "range", "file", "lines", "language", "metaVariables"} <= set(match)
        assert match["file"] == "src/a.py"
        assert {"start", "end", "byteOffset"} <= set(match["range"])


def test_valid_negative_probe_returns_empty_matches(ast_grep_executable: str) -> None:
    service = actual_service(REPOSITORY_ROOT, ast_grep_executable)

    matches = service.test_match_code_rule(
        code="value = 1",
        rule_yaml="id: no-call\nlanguage: python\nrule:\n  pattern: print($A)\n",
    )

    assert matches == []


def test_dump_syntax_tree_does_not_scan_working_directory(
    tmp_path: Path,
    ast_grep_executable: str,
) -> None:
    (tmp_path / "poison.py").write_text("print('poison')\n", encoding="utf-8")
    service = actual_service(tmp_path, ast_grep_executable)

    result = service.dump_syntax_tree(code="print($A)", language="python", format="cst")

    assert "Debug CST:" in result
    assert "poison" not in result


def test_regex_policy_rejects_regex_rule_before_ast_grep_runs(ast_grep_executable: str) -> None:
    service = actual_service(REPOSITORY_ROOT, ast_grep_executable, forbid_regex_rules=True)

    with pytest.raises(ValueError, match="Regex ast-grep rules are disabled"):
        service.test_match_code_rule(
            code="value = 1",
            rule_yaml="id: regex\nlanguage: python\nrule:\n  regex: value.*\n",
        )


@pytest.mark.asyncio
async def test_stdio_auto_protocol_catalog_calls_metadata_and_outline_contract(
    console_entrypoint: str,
    ast_grep_executable: str,
) -> None:
    parameters = server_parameters(console_entrypoint, ast_grep_executable)

    async with Client(stdio_client(parameters), mode="auto") as client:
        assert client.protocol_version == LATEST_MODERN_VERSION
        assert client.session.initialize_result is None
        assert client.server_info is not None
        assert client.server_info.name == "ast-grep"
        assert client.server_info.version == "0.3.0"

        listed = await client.list_tools()
        tools = {tool.name: tool for tool in listed.tools}
        _assert_tool_catalog(tools)

        info_result = await client.call_tool("get_server_info", {})
        assert info_result.is_error is False
        assert info_result.structured_content is not None
        assert info_result.structured_content["fork_version"] == "0.3.0"
        assert info_result.structured_content["ast_grep_version"] == SUPPORTED_AST_GREP_VERSION
        assert info_result.structured_content["forbid_regex_rules"] is True
        assert info_result.structured_content["default_max_results"] == 50
        assert info_result.structured_content["max_results_cap"] == 500

        negative = await client.call_tool(
            "test_match_code_rule",
            {
                "code": "value = 1",
                "yaml": "id: no-call\nlanguage: python\nrule:\n  pattern: print($A)\n",
            },
        )
        assert negative.is_error is False
        assert negative.structured_content == {"result": []}

        found = await client.call_tool(
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
        assert found.is_error is False
        assert found.structured_content is not None
        assert set(found.structured_content) == {"matches", "returned", "truncated", "limit"}
        assert found.structured_content["returned"] == 1
        assert found.structured_content["truncated"] is True
        assert found.structured_content["limit"] == 1

        metadata = await client.call_tool(
            "find_code_by_rule",
            {
                "project_folder": str(FIXTURES),
                "yaml": (
                    "id: functions\nlanguage: python\nmessage: function\nseverity: info\n"
                    "metadata:\n  category: documentation\nrule:\n  kind: function_definition\n"
                ),
                "paths": ["example.py"],
                "max_results": 1,
                "output_format": "json",
                "include_metadata": True,
            },
        )
        assert metadata.is_error is False
        assert metadata.structured_content is not None
        assert metadata.structured_content["matches"][0]["metadata"] == {"category": "documentation"}

        full_outline = await client.call_tool(
            "outline_code",
            {
                "project_folder": str(FIXTURES),
                "paths": ["example.py"],
                "max_results": 4,
                "output_format": "json",
            },
        )
        assert full_outline.is_error is False
        assert full_outline.structured_content is not None
        assert full_outline.structured_content["returned"] == 4
        assert full_outline.structured_content["truncated"] is False
        assert _count_outline_nodes(full_outline.structured_content["files"]) == 4
        class_item = full_outline.structured_content["files"][0]["items"][2]
        assert class_item["name"] == "Calculator"
        assert class_item["members"][0]["name"] == "multiply"

        truncated_outline = await client.call_tool(
            "outline_code",
            {
                "project_folder": str(FIXTURES),
                "paths": ["example.py"],
                "max_results": 3,
            },
        )
        assert truncated_outline.is_error is False
        assert truncated_outline.structured_content is not None
        assert truncated_outline.structured_content["returned"] == 3
        assert truncated_outline.structured_content["truncated"] is True
        assert truncated_outline.structured_content["limit"] == 3
        assert _count_outline_nodes(truncated_outline.structured_content["files"]) == 3
        compact_block = truncated_outline.content[0]
        assert isinstance(compact_block, TextContent)
        assert compact_block.text.startswith("Found 3 outline nodes")

        rejected = await client.call_tool(
            "find_code_by_rule",
            {
                "project_folder": str(FIXTURES),
                "yaml": "id: regex\nlanguage: python\nrule:\n  regex: value.*\n",
            },
        )
        assert rejected.is_error is True


@pytest.mark.asyncio
async def test_stdio_legacy_protocol_handshake_compatibility(
    console_entrypoint: str,
    ast_grep_executable: str,
) -> None:
    parameters = server_parameters(console_entrypoint, ast_grep_executable)

    async with Client(stdio_client(parameters), mode="legacy") as client:
        assert client.protocol_version == LATEST_HANDSHAKE_VERSION
        assert client.session.initialize_result is not None
        assert client.session.initialize_result.protocol_version == LATEST_HANDSHAKE_VERSION
        assert client.server_info is not None
        assert client.server_info.name == "ast-grep"
        assert client.server_info.version == "0.3.0"
        listed = await client.list_tools()
        assert {tool.name for tool in listed.tools} == set(EXPECTED_TOOLS)


@pytest.mark.asyncio
async def test_stdio_server_exits_on_eof_without_a_process_survivor(
    console_entrypoint: str,
    ast_grep_executable: str,
) -> None:
    parameters = server_parameters(console_entrypoint, ast_grep_executable)
    process = await asyncio.create_subprocess_exec(
        parameters.command,
        *parameters.args,
        cwd=parameters.cwd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_reader = asyncio.create_task(process.stdout.read())
    stderr_reader = asyncio.create_task(process.stderr.read())

    process.stdin.close()
    try:
        await process.stdin.wait_closed()
    except (BrokenPipeError, ConnectionResetError):
        pass
    returncode = await asyncio.wait_for(process.wait(), timeout=10)
    stdout, stderr = await asyncio.gather(stdout_reader, stderr_reader)

    assert returncode == 0, stderr.decode(errors="replace")
    assert stdout == b""
    assert process.returncode is not None


@pytest.mark.asyncio
async def test_stdio_client_cancellation_reaps_the_server_process(
    monkeypatch: pytest.MonkeyPatch,
    console_entrypoint: str,
    ast_grep_executable: str,
) -> None:
    created_processes: list[Any] = []
    create_process = stdio_module._create_platform_compatible_process

    async def capture_process(**kwargs: Any) -> Any:
        process = await create_process(**kwargs)
        created_processes.append(process)
        return process

    monkeypatch.setattr(stdio_module, "_create_platform_compatible_process", capture_process)
    connected = asyncio.Event()
    hold_connection = asyncio.Event()

    async def connect() -> None:
        parameters = server_parameters(console_entrypoint, ast_grep_executable)
        async with Client(stdio_client(parameters), mode="auto") as client:
            await client.list_tools()
            connected.set()
            await hold_connection.wait()

    task = asyncio.create_task(connect())
    await asyncio.wait_for(connected.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(created_processes) == 1
    assert created_processes[0].returncode is not None
