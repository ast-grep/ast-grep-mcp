from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TextIO

import mcp.client.stdio as stdio_module
import pytest
from anyio.abc import Process
from fastmcp import Client as FastMCPClient
from mcp import Client, StdioServerParameters, stdio_client
from mcp.os.win32.utilities import FallbackProcess
from mcp.types import TextContent, Tool
from mcp.types.version import LATEST_HANDSHAKE_VERSION, LATEST_MODERN_VERSION
from pydantic import TypeAdapter

from ast_soleaux.server import (
    HARD_MAX_RESULTS,
    JSON_OBJECT_ADAPTER,
    MAX_NDJSON_RECORD_BYTES,
    SUPPORTED_AST_GREP_VERSION,
    SUPPORTED_OXC_HELPER_VERSION,
    SUPPORTED_OXC_PARSER_VERSION,
    SUPPORTED_OXC_RESOLVER_VERSION,
    AstGrepService,
    JavascriptModuleResults,
    JsonObject,
    JsonValue,
    ResolvedExecutable,
    RuntimeServices,
    build_runtime,
    create_mcp,
    read_analysis_helper_versions,
    read_oxc_helper_versions,
    read_postgres_helper_versions,
    read_typescript_helper_versions,
    resolve_ast_grep_executable,
    resolve_oxc_helper_executable,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
OXC_HELPER = REPOSITORY_ROOT / "oxc-sidecar" / "bin" / "ast-soleaux-oxc.mjs"
TYPESCRIPT_PROJECT_HELPER = REPOSITORY_ROOT / "oxc-sidecar" / "bin" / "ast-soleaux-typescript-project.mjs"
POSTGRES_HELPER = REPOSITORY_ROOT / "postgresql-sidecar" / "bin" / "ast-soleaux-postgresql.mjs"
ANALYSIS_HELPER = REPOSITORY_ROOT / "analysis-sidecar" / "target" / "debug" / "ast-soleaux-analysis"
AST_GREP_TEST_EXECUTABLE = "AST_GREP_TEST_EXECUTABLE"
EXPECTED_TOOLS = {
    "dump_syntax_tree": "Dump syntax tree",
    "test_match_code_rule": "Test rule against code",
    "find_code": "Find code by pattern",
    "find_code_by_rule": "Find code by rule",
    "get_server_info": "Get server info",
    "outline_code": "Outline code",
    "oxc_modules": "Inspect Oxc modules",
    "oxc_transform": "Transform code with Oxc",
    "oxc_transform_files": "Transform files with Oxc",
    "oxc_minify": "Minify code with Oxc",
    "oxc_minify_files": "Minify files with Oxc",
    "semantic_scopes": "Inspect semantic scopes",
    "semantic_symbols": "Inspect semantic symbols",
    "semantic_references": "Inspect semantic references",
    "semantic_cfg": "Inspect control-flow graph",
    "inspect_typescript_project": "Inspect TypeScript project",
    "postgres_parse": "Parse PostgreSQL",
    "postgres_parse_files": "Parse PostgreSQL files",
    "postgres_deparse_preview": "Preview PostgreSQL deparse",
    "typescript_execute": "Execute TypeScript in an operator sandbox",
    "scan_project_rules": "Scan configured project rules",
    "test_project_rules": "Test configured project rules",
}
BASELINE_TOOL_NAMES = {
    "dump_syntax_tree",
    "test_match_code_rule",
    "find_code",
    "find_code_by_rule",
    "get_server_info",
    "outline_code",
}
OXC_TOOL_NAMES = BASELINE_TOOL_NAMES | {
    "oxc_modules",
    "oxc_transform",
    "oxc_transform_files",
    "oxc_minify",
    "oxc_minify_files",
}
CONFIGURED_OXC_TOOL_NAMES = OXC_TOOL_NAMES | {"scan_project_rules", "test_project_rules"}
ANALYSIS_TOOL_NAMES = CONFIGURED_OXC_TOOL_NAMES | {
    "semantic_scopes",
    "semantic_symbols",
    "semantic_references",
    "semantic_cfg",
    "inspect_typescript_project",
    "postgres_parse",
    "postgres_parse_files",
    "postgres_deparse_preview",
}
ALL_TOOL_NAMES = ANALYSIS_TOOL_NAMES | {"typescript_execute"}
EXPECTED_STDIO_TOOLS = set(EXPECTED_TOOLS).difference(
    {"semantic_scopes", "semantic_symbols", "semantic_references", "semantic_cfg", "typescript_execute"}
)
JAVASCRIPT_MODULE_RESULTS_ADAPTER = TypeAdapter(JavascriptModuleResults)

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
def launcher_python() -> str:
    executable = Path(sys.executable).absolute()
    expected = (REPOSITORY_ROOT / ".venv").resolve()
    if not executable.is_relative_to(expected):
        pytest.fail(f"Integration tests must use the repository Python under {expected}", pytrace=False)
    return str(executable)


def actual_service(
    root: Path,
    ast_grep_executable: str,
    *,
    forbid_regex_rules: bool = False,
    config_path: str | None = None,
) -> AstGrepService:
    runtime = build_runtime(
        working_directory=root,
        ast_grep_executable=ast_grep_executable,
        allowed_roots=[str(root)],
        forbid_regex_rules=forbid_regex_rules,
        config_path=config_path,
    )
    assert runtime.ast_grep_version == SUPPORTED_AST_GREP_VERSION
    return AstGrepService(runtime)


def server_parameters(launcher_python: str, ast_grep_executable: str) -> StdioServerParameters:
    return StdioServerParameters(
        command=launcher_python,
        args=[
            str(REPOSITORY_ROOT / "scripts" / "launch_server.py"),
            "--ast-grep",
            ast_grep_executable,
            "--oxc-helper",
            str(OXC_HELPER),
            "--typescript-project-helper",
            str(TYPESCRIPT_PROJECT_HELPER),
            "--postgres-helper",
            str(POSTGRES_HELPER),
            "--allowed-root",
            str(REPOSITORY_ROOT),
            "--config",
            str(FIXTURES / "configured" / "sgconfig.yml"),
            "--forbid-regex-rules",
        ],
        cwd=REPOSITORY_ROOT,
        env={
            "PATH": os.environ.get("PATH", ""),
            "VIRTUAL_ENV": str(REPOSITORY_ROOT / ".venv"),
            "UV_PROJECT_ENVIRONMENT": ".venv",
        },
    )


def _object(value: object) -> JsonObject:
    return JSON_OBJECT_ADAPTER.validate_python(value, strict=True)


def _list(value: object) -> list[JsonValue]:
    assert isinstance(value, list)
    return value


def _string(value: object) -> str:
    assert isinstance(value, str)
    return value


def _integer(value: object) -> int:
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def _schema_contains(schema: JsonValue, key: str, expected: JsonValue) -> bool:
    if isinstance(schema, dict):
        if schema.get(key) == expected:
            return True
        return any(_schema_contains(value, key, expected) for value in schema.values())
    if isinstance(schema, list):
        return any(_schema_contains(value, key, expected) for value in schema)
    return False


def _count_outline_nodes(files: JsonValue) -> int:
    def count(nodes: JsonValue) -> int:
        if not isinstance(nodes, list):
            return 0
        total = 0
        for node in nodes:
            if isinstance(node, dict):
                total += 1 + count(node.get("members", []))
        return total

    if not isinstance(files, list):
        return 0
    return sum(count(file_result.get("items", [])) for file_result in files if isinstance(file_result, dict))


class ToolCallResult(Protocol):
    @property
    def is_error(self) -> bool: ...

    @property
    def structured_content(self) -> object: ...

    @property
    def content(self) -> Sequence[object]: ...


def _assert_mirrored_structured_content(result: ToolCallResult, expected: JsonObject) -> None:
    assert result.is_error is False
    assert result.structured_content == expected
    assert len(result.content) == 1
    block = result.content[0]
    assert isinstance(block, TextContent)
    assert json.loads(block.text) == expected


def _assert_tool_catalog(tools: dict[str, Tool], expected_names: set[str]) -> None:
    def output_properties(name: str) -> set[str]:
        schema = tools[name].output_schema
        assert schema is not None
        properties = schema.get("properties")
        assert isinstance(properties, dict)
        return set(properties)

    assert set(tools) == expected_names
    expected_properties: dict[str, set[str]] = {
        "dump_syntax_tree": {"code", "language", "format"},
        "test_match_code_rule": {
            "code",
            "yaml",
            "text_equals",
            "text_starts_with",
            "text_contains",
        },
        "find_code": {
            "project_folder",
            "pattern",
            "language",
            "paths",
            "include_globs",
            "exclude_globs",
            "max_results",
            "output_format",
            "selector",
            "strictness",
            "rewrite",
            "cursor",
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
            "positive_code",
            "negative_code",
            "text_equals",
            "text_starts_with",
            "text_contains",
            "cursor",
        },
        "oxc_modules": {
            "project_folder",
            "paths",
            "include_globs",
            "exclude_globs",
            "strict_paths",
            "include_dynamic",
            "max_results",
            "cursor",
            "output_format",
        },
        "oxc_transform": {"source", "options", "output_format"},
        "oxc_transform_files": {
            "project_folder",
            "output_root",
            "paths",
            "include_globs",
            "exclude_globs",
            "strict_paths",
            "conflict_policy",
            "allow_source_overwrite",
            "options",
            "max_results",
        },
        "oxc_minify": {"source", "options", "output_format"},
        "oxc_minify_files": {
            "project_folder",
            "output_root",
            "paths",
            "include_globs",
            "exclude_globs",
            "strict_paths",
            "conflict_policy",
            "allow_source_overwrite",
            "options",
            "max_results",
        },
        "semantic_scopes": {"project_folder", "path", "source_digest", "max_results", "cursor", "output_format"},
        "semantic_symbols": {"project_folder", "path", "source_digest", "max_results", "cursor", "output_format"},
        "semantic_references": {
            "project_folder",
            "path",
            "position",
            "source_digest",
            "include_declaration",
            "include_unresolved",
            "project_paths",
            "include_globs",
            "exclude_globs",
            "max_results",
            "cursor",
            "output_format",
        },
        "semantic_cfg": {
            "project_folder",
            "path",
            "source_digest",
            "function_position",
            "max_results",
            "cursor",
            "output_format",
        },
        "inspect_typescript_project": {
            "project_folder",
            "tsconfig",
            "paths",
            "include_emit",
            "include_code_actions",
            "max_results",
        },
        "postgres_parse": {"sql", "mode"},
        "postgres_parse_files": {
            "project_folder",
            "paths",
            "include_globs",
            "exclude_globs",
            "strict_paths",
            "mode",
            "max_results",
            "cursor",
        },
        "postgres_deparse_preview": {"sql"},
        "typescript_execute": {"project_folder", "entry", "args", "stdin", "timeout_seconds", "output_format"},
        "scan_project_rules": {
            "project_folder",
            "rule_ids",
            "paths",
            "include_globs",
            "exclude_globs",
            "max_results",
            "include_metadata",
            "run_tests_first",
            "output_format",
            "cursor",
        },
        "test_project_rules": {"rule_ids"},
        "get_server_info": set(),
        "outline_code": {
            "project_folder",
            "paths",
            "include_globs",
            "exclude_globs",
            "strict_paths",
            "language",
            "max_results",
            "items",
            "symbol_types",
            "public_members",
            "output_format",
        },
    }
    expected_required: dict[str, set[str]] = {
        "dump_syntax_tree": {"code", "language"},
        "test_match_code_rule": {"code", "yaml"},
        "find_code": {"project_folder", "pattern", "language"},
        "find_code_by_rule": {"project_folder", "yaml"},
        "oxc_modules": {"project_folder"},
        "oxc_transform": {"source"},
        "oxc_transform_files": {"project_folder", "output_root"},
        "oxc_minify": {"source"},
        "oxc_minify_files": {"project_folder", "output_root"},
        "semantic_scopes": {"project_folder", "path"},
        "semantic_symbols": {"project_folder", "path"},
        "semantic_references": {"project_folder", "path", "position"},
        "semantic_cfg": {"project_folder", "path"},
        "inspect_typescript_project": {"project_folder"},
        "postgres_parse": {"sql"},
        "postgres_parse_files": {"project_folder"},
        "postgres_deparse_preview": {"sql"},
        "typescript_execute": {"project_folder", "entry"},
        "scan_project_rules": {"project_folder"},
        "test_project_rules": set(),
        "get_server_info": set(),
        "outline_code": {"project_folder"},
    }

    mutating_tools = {"oxc_transform_files", "oxc_minify_files", "typescript_execute"}
    for name, tool in tools.items():
        assert tool.title == EXPECTED_TOOLS[name]
        assert tool.annotations is not None
        assert tool.annotations.read_only_hint is (name not in mutating_tools)
        assert tool.annotations.destructive_hint is (name in mutating_tools)
        assert tool.annotations.idempotent_hint is (name != "typescript_execute")
        assert tool.annotations.open_world_hint is (name == "typescript_execute")
        assert tool.input_schema["type"] == "object"
        assert tool.input_schema.get("additionalProperties") is False
        assert set(tool.input_schema.get("properties", {})) == expected_properties[name], name
        assert set(tool.input_schema.get("required", [])) == expected_required[name]
        assert tool.output_schema is not None
        assert tool.output_schema["type"] == "object"

    for search_tool_name in ("find_code", "find_code_by_rule", "scan_project_rules"):
        search_schema = tools[search_tool_name].output_schema
        assert search_schema is not None
        assert set(search_schema["properties"]) == {
            "matches",
            "returned",
            "truncated",
            "limit",
            "next_cursor",
            "snapshot_truncated",
            "diagnostics",
        }
        assert set(search_schema["required"]) == {"matches", "returned", "truncated", "limit"}
        assert _schema_contains(tools[search_tool_name].input_schema["properties"]["max_results"], "maximum", 500)

    module_schema = tools["oxc_modules"].output_schema
    assert module_schema is not None
    assert set(module_schema["properties"]) == {
        "modules",
        "returned",
        "truncated",
        "limit",
        "next_cursor",
        "snapshot_truncated",
        "source_digest",
        "diagnostics",
    }
    assert set(module_schema["required"]) == {
        "modules",
        "returned",
        "truncated",
        "limit",
        "source_digest",
        "diagnostics",
    }
    module_input = tools["oxc_modules"].input_schema["properties"]
    assert _schema_contains(module_input["paths"], "minItems", 1)
    assert _schema_contains(module_input["paths"], "maxItems", 64)
    assert module_input["strict_paths"]["default"] is False
    assert module_input["include_dynamic"]["default"] is False
    assert module_input["cursor"]["default"] is None
    assert module_input["output_format"]["default"] == "text"

    outline_schema = tools["outline_code"].output_schema
    assert outline_schema is not None
    assert set(outline_schema["properties"]) == {
        "files",
        "returned",
        "truncated",
        "limit",
        "resolved_paths",
        "path_errors",
    }
    assert set(outline_schema["required"]) == {"files", "returned", "truncated", "limit"}
    outline_input = tools["outline_code"].input_schema["properties"]
    assert _schema_contains(outline_input["paths"], "minItems", 1)
    assert _schema_contains(outline_input["paths"], "maxItems", 64)
    assert outline_input["include_globs"]["default"] is None
    assert outline_input["exclude_globs"]["default"] is None
    assert outline_input["strict_paths"]["default"] is False
    assert outline_input["language"]["default"] is None
    assert _schema_contains(outline_input["max_results"], "maximum", HARD_MAX_RESULTS)
    assert outline_input["output_format"]["default"] == "text"
    assert outline_input["items"]["default"] == "auto"
    assert outline_input["symbol_types"]["default"] is None
    assert outline_input["public_members"]["default"] is False
    find_input = tools["find_code"].input_schema["properties"]
    assert find_input["selector"]["default"] is None
    assert find_input["strictness"]["default"] == "smart"
    assert find_input["rewrite"]["default"] is None
    assert find_input["cursor"]["default"] is None
    probe_input = tools["test_match_code_rule"].input_schema["properties"]
    assert probe_input["text_equals"]["default"] is None
    assert probe_input["text_starts_with"]["default"] is None
    assert probe_input["text_contains"]["default"] is None
    rule_input = tools["find_code_by_rule"].input_schema["properties"]
    assert rule_input["include_metadata"]["default"] is False
    assert rule_input["positive_code"]["default"] is None
    assert rule_input["negative_code"]["default"] is None
    assert rule_input["text_equals"]["default"] is None
    assert rule_input["text_starts_with"]["default"] is None
    assert rule_input["text_contains"]["default"] is None
    assert rule_input["cursor"]["default"] is None
    typescript_input = tools["inspect_typescript_project"].input_schema["properties"]
    assert typescript_input["tsconfig"]["default"] == "tsconfig.json"
    assert typescript_input["include_emit"]["default"] is True
    assert typescript_input["include_code_actions"]["default"] is True
    assert _schema_contains(typescript_input["max_results"], "maximum", HARD_MAX_RESULTS)
    postgres_input = tools["postgres_parse"].input_schema["properties"]
    assert postgres_input["mode"]["default"] == "parse"
    postgres_files_input = tools["postgres_parse_files"].input_schema["properties"]
    assert postgres_files_input["strict_paths"]["default"] is True
    assert postgres_files_input["mode"]["default"] == "parse"
    scan_input = tools["scan_project_rules"].input_schema["properties"]
    assert scan_input["include_metadata"]["default"] is True
    assert scan_input["run_tests_first"]["default"] is False
    assert scan_input["cursor"]["default"] is None
    assert _schema_contains(tools["dump_syntax_tree"].input_schema["properties"]["format"], "enum", ["pattern", "cst", "ast", "sexp"])

    assert output_properties("dump_syntax_tree") == {"result"}
    assert output_properties("test_match_code_rule") == {"result"}
    assert output_properties("oxc_modules") == {
        "modules",
        "returned",
        "truncated",
        "limit",
        "next_cursor",
        "snapshot_truncated",
        "source_digest",
        "diagnostics",
    }
    assert output_properties("inspect_typescript_project") == {
        "typescript_version",
        "tsconfig",
        "root_files",
        "options",
        "diagnostics",
        "modules",
        "symbols",
        "inferred_types",
        "emit",
        "code_actions",
        "source_digest",
        "returned",
        "truncated",
        "limit",
    }
    assert output_properties("postgres_parse") == {
        "parser_version",
        "deparser_version",
        "postgres_major",
        "mode",
        "source_digest",
        "tree",
        "tokens",
        "fingerprint",
        "normalized",
        "plpgsql",
        "statements",
        "declarations",
        "references",
        "calls",
        "diagnostics",
        "returned",
        "truncated",
        "limit",
    }
    assert output_properties("postgres_parse_files") == {
        "files",
        "returned",
        "truncated",
        "limit",
        "next_cursor",
        "snapshot_truncated",
    }
    assert output_properties("postgres_deparse_preview") == {
        "parser_version",
        "deparser_version",
        "postgres_major",
        "mode",
        "source_digest",
        "original_sql",
        "deparsed_sql",
        "equivalent",
        "original_tree_digest",
        "reparsed_tree_digest",
        "diagnostics",
    }
    assert output_properties("test_project_rules") == {
        "passed",
        "report",
        "report_truncated",
    }
    assert output_properties("get_server_info") == {
        "server",
        "versions",
        "executables",
        "allowed_roots",
        "capabilities",
        "limits",
        "coordinates",
        "configuration",
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
    assert "def hello" in _string(result["matches"][0]["text"])


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
        assert {"start", "end", "byteOffset"} <= set(_object(match["range"]))


def test_relational_captures_and_labels_can_extend_beyond_the_primary_match(ast_grep_executable: str) -> None:
    service = actual_service(REPOSITORY_ROOT, ast_grep_executable)

    result = service.find_code_by_rule(
        project_folder=str(FIXTURES),
        rule_yaml=(
            "id: relational-capture\n"
            "language: Python\n"
            "severity: info\n"
            "message: relational\n"
            "rule:\n"
            "  pattern: print($A)\n"
            "  inside:\n"
            "    pattern: |\n"
            "      def $FN($$$ARGS):\n"
            "        $$$BODY\n"
            "    stopBy: end\n"
        ),
        paths=["example.py"],
        include_globs=None,
        exclude_globs=None,
        max_results=10,
    )

    assert result["returned"] == 1
    match = result["matches"][0]
    match_range = _object(match["range"])
    match_start = _integer(_object(match_range["byteOffset"])["start"])
    meta_variables = _object(match["metaVariables"])
    single = _object(meta_variables["single"])
    function_capture = _object(single["FN"])
    capture_range = _object(function_capture["range"])
    assert _integer(_object(capture_range["byteOffset"])["start"]) < match_start
    labels = _list(match["labels"])
    label_range = _object(_object(labels[1])["range"])
    assert _integer(_object(label_range["byteOffset"])["start"]) < match_start


def test_rule_match_accepts_ast_grep_omission_of_empty_labels(tmp_path: Path, ast_grep_executable: str) -> None:
    (tmp_path / "example.py").write_text("print('value')\n", encoding="utf-8")
    service = actual_service(tmp_path, ast_grep_executable)

    result = service.find_code_by_rule(
        project_folder=str(tmp_path),
        rule_yaml=("id: empty-labels\nlanguage: Python\nmessage: empty labels\nseverity: info\nlabels: {}\nrule: {pattern: print($A)}\n"),
        paths=["example.py"],
        include_globs=None,
        exclude_globs=None,
        max_results=5,
    )

    assert result["returned"] == 1
    assert "labels" not in result["matches"][0]
    service.runtime.close()


def test_valid_negative_probe_returns_empty_matches(ast_grep_executable: str) -> None:
    service = actual_service(REPOSITORY_ROOT, ast_grep_executable)

    matches = service.test_match_code_rule(
        code="value = 1",
        rule_yaml="id: no-call\nlanguage: python\nrule:\n  pattern: print($A)\n",
    )

    assert matches == []


def test_literal_text_filters_replace_prefix_regex_rules(ast_grep_executable: str) -> None:
    service = actual_service(REPOSITORY_ROOT, ast_grep_executable, forbid_regex_rules=True)

    staff = service.test_match_code_rule(
        code="const staff = 'per_staff_1'\nconst contact = 'per_1'",
        rule_yaml="id: strings\nlanguage: TypeScript\nrule:\n  kind: string\n",
        text_contains="per_staff_",
    )
    imports = service.test_match_code_rule(
        code="import type { X } from '@anilize/crm'\nimport type { Y } from '@/types/apps/contact-types'",
        rule_yaml=("id: imports\nlanguage: TypeScript\nrule:\n  kind: string\n  inside:\n    kind: import_statement\n    stopBy: end\n"),
        text_contains="@anilize/",
    )

    assert [match["text"] for match in staff] == ["'per_staff_1'"]
    assert [match["text"] for match in imports] == ["'@anilize/crm'"]


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


def test_preview_rewrites_transformations_and_deletions_never_change_source(
    tmp_path: Path,
    ast_grep_executable: str,
) -> None:
    source = tmp_path / "sample.ts"
    source.write_text("value = 123\nprint(value)\nconst typed: number = 456;\n", encoding="utf-8", newline="")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    service = actual_service(tmp_path, ast_grep_executable)

    replacement = service.find_code(
        project_folder=str(tmp_path),
        pattern="print($A)",
        language="TypeScript",
        rewrite="logger.info($A)",
        paths=["sample.ts"],
        include_globs=None,
        exclude_globs=None,
        max_results=5,
    )
    deletion = service.find_code(
        project_folder=str(tmp_path),
        pattern="print($A)",
        language="TypeScript",
        rewrite="",
        paths=["sample.ts"],
        include_globs=None,
        exclude_globs=None,
        max_results=5,
    )
    transformed = service.find_code_by_rule(
        project_folder=str(tmp_path),
        rule_yaml=(
            "id: transform-preview\n"
            "language: TypeScript\n"
            "message: preview\n"
            "severity: info\n"
            "rule:\n  pattern: value = $A\n"
            "transform:\n"
            "  B:\n"
            "    rewrite:\n"
            "      rewriters: [number-to-word]\n"
            "      source: $A\n"
            "rewriters:\n"
            "  - id: number-to-word\n"
            "    rule: {kind: number}\n"
            "    fix: one\n"
            "fix: value = $B\n"
        ),
        paths=["sample.ts"],
        include_globs=None,
        exclude_globs=None,
        max_results=5,
    )
    selected = service.find_code(
        project_folder=str(tmp_path),
        pattern="const $NAME: $TYPE = $VALUE;",
        selector="variable_declarator",
        strictness="ast",
        language="TypeScript",
        paths=["sample.ts"],
        include_globs=None,
        exclude_globs=None,
        max_results=5,
    )

    assert replacement["matches"][0]["replacement"] == "logger.info(value)"
    matched_start = source.read_bytes().index(b"print(value)")
    assert replacement["matches"][0]["replacementOffsets"] == {
        "start": matched_start,
        "end": matched_start + len(b"print(value)"),
    }
    assert deletion["matches"][0]["replacement"] == ""
    transformed_meta = _object(transformed["matches"][0]["metaVariables"])
    assert transformed_meta["transformed"] == {"B": "one"}
    assert transformed["matches"][0]["replacement"] == "value = one"
    assert selected["matches"][0]["text"] == "typed: number = 456"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    service.runtime.close()


def test_outline_modes_symbol_filter_public_members_and_sexp(
    tmp_path: Path,
    ast_grep_executable: str,
) -> None:
    source = tmp_path / "sample.ts"
    source.write_text(
        ('import thing from "thing";\nexport function top() {}\nclass Example {\n  public visible() {}\n  private hidden() {}\n}\n'),
        encoding="utf-8",
    )
    service = actual_service(tmp_path, ast_grep_executable)

    imports = service.outline_code(
        project_folder=str(tmp_path),
        paths=["sample.ts"],
        language=None,
        max_results=20,
        items="imports",
    )
    classes = service.outline_code(
        project_folder=str(tmp_path),
        paths=["sample.ts"],
        language=None,
        max_results=20,
        items="structure",
        symbol_types=["class"],
        public_members=True,
    )
    sexp = service.dump_syntax_tree(code="const value = call()", language="TypeScript", format="sexp")

    assert imports["returned"] == 1
    import_items = _list(imports["files"][0]["items"])
    assert _object(import_items[0])["isImport"] is True
    class_items = _list(classes["files"][0]["items"])
    assert [_object(item)["symbolType"] for item in class_items] == ["class"]
    members = _list(_object(class_items[0])["members"])
    assert [_object(member)["name"] for member in members] == ["visible"]
    assert all(_object(member)["isPublic"] for member in members)
    assert sexp.startswith("Debug Sexp:")
    assert "lexical_declaration" in sexp
    service.runtime.close()


def test_configured_scan_and_tests_use_startup_snapshot_only(
    tmp_path: Path,
    ast_grep_executable: str,
) -> None:
    configured = tmp_path / "configured"
    shutil.copytree(FIXTURES / "configured", configured)
    scoped_source = configured / "src" / "scoped.py"
    scoped_source.parent.mkdir()
    scoped_source.write_text("print('scoped')\n", encoding="utf-8")
    (configured / "rules" / "scoped.yml").write_text(
        ("id: scoped-rule\nlanguage: Python\nmessage: scoped\nseverity: info\nfiles: [src/*.py]\nrule: {pattern: print($A)}\n"),
        encoding="utf-8",
    )
    service = actual_service(tmp_path, ast_grep_executable, config_path="configured/sgconfig.yml")

    before = {
        path.relative_to(configured): hashlib.sha256(path.read_bytes()).hexdigest() for path in configured.rglob("*") if path.is_file()
    }
    scan = service.scan_project_rules(
        project_folder=str(configured),
        rule_ids=["literal.dot+id"],
        paths=["example.py"],
        include_globs=["**/*.py"],
        exclude_globs=None,
        max_results=5,
        include_metadata=True,
    )
    configured_print = service.scan_project_rules(
        project_folder=str(configured),
        rule_ids=["configured-print"],
        paths=["example.py"],
        include_globs=None,
        exclude_globs=None,
        max_results=5,
        include_metadata=True,
    )
    test_result = service.test_project_rules(rule_ids=["configured-print"])
    scoped = service.scan_project_rules(
        project_folder=str(configured),
        rule_ids=["scoped-rule"],
        paths=["src"],
        include_globs=None,
        exclude_globs=None,
        max_results=5,
    )
    after = {
        path.relative_to(configured): hashlib.sha256(path.read_bytes()).hexdigest() for path in configured.rglob("*") if path.is_file()
    }

    assert scan["returned"] == 1
    assert scan["matches"][0]["ruleId"] == "literal.dot+id"
    assert configured_print["matches"][0]["metadata"] == {"category": "integration"}
    assert test_result["passed"] is True
    assert test_result["report_truncated"] is False
    assert scoped["returned"] == 1
    assert scoped["matches"][0]["file"] == "src/scoped.py"
    assert before == after

    (configured / "rules" / "literal-id.yml").write_text(
        "id: changed\nlanguage: Python\nmessage: changed\nseverity: info\nrule: {pattern: no_match}\n",
        encoding="utf-8",
    )
    write_project_scan = service.scan_project_rules(
        project_folder=str(configured),
        rule_ids=["literal.dot+id"],
        paths=["example.py"],
        include_globs=None,
        exclude_globs=None,
        max_results=5,
    )
    assert write_project_scan["returned"] == 1
    assert write_project_scan["matches"][0]["ruleId"] == "literal.dot+id"
    service.runtime.close()


def test_runtime_rejects_semantically_invalid_configured_rules_at_startup(
    tmp_path: Path,
    ast_grep_executable: str,
) -> None:
    rules = tmp_path / "rules"
    rules.mkdir()
    (rules / "invalid.yml").write_text(
        "id: invalid-language\nlanguage: DefinitelyNotALanguage\nmessage: invalid\nseverity: info\nrule: {kind: identifier}\n",
        encoding="utf-8",
    )
    (tmp_path / "sgconfig.yml").write_text("ruleDirs: [rules]\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Command failed with exit code"):
        build_runtime(
            working_directory=tmp_path,
            ast_grep_executable=ast_grep_executable,
            config_path="sgconfig.yml",
            allowed_roots=[str(tmp_path)],
        )
    runtime_root = tmp_path / ".ast-soleaux-runtime"
    assert not runtime_root.exists() or list(runtime_root.iterdir()) == []


def test_oversized_match_record_fails_closed(tmp_path: Path, ast_grep_executable: str) -> None:
    source = tmp_path / "oversized.py"
    source.write_text(f'print("{"x" * (MAX_NDJSON_RECORD_BYTES + 1024)}")\n', encoding="utf-8")
    service = actual_service(tmp_path, ast_grep_executable)

    with pytest.raises(RuntimeError, match="record exceeds the 1024 KiB limit"):
        service.find_code(
            project_folder=str(tmp_path),
            pattern="print($A)",
            language="Python",
            paths=["oversized.py"],
            include_globs=None,
            exclude_globs=None,
            max_results=1,
        )
    service.runtime.close()


@pytest.mark.asyncio
async def test_fastmcp_in_process_catalog_has_no_resources_or_prompts(
    ast_grep_executable: str,
) -> None:
    runtime = build_runtime(
        working_directory=REPOSITORY_ROOT,
        ast_grep_executable=ast_grep_executable,
        oxc_helper_executable=str(OXC_HELPER),
        allowed_roots=[str(REPOSITORY_ROOT)],
        config_path=str(FIXTURES / "configured" / "sgconfig.yml"),
        forbid_regex_rules=True,
    )
    server = create_mcp(runtime)
    try:
        async with FastMCPClient(server) as client:
            tools = await client.list_tools()
            assert {tool.name for tool in tools} == CONFIGURED_OXC_TOOL_NAMES
            assert await client.list_resources() == []
            assert await client.list_resource_templates() == []
            assert await client.list_prompts() == []
            search = next(tool for tool in tools if tool.name == "find_code")
            assert search.input_schema.get("additionalProperties") is False
            assert search.output_schema is not None
            assert set(search.output_schema["required"]) == {
                "matches",
                "returned",
                "truncated",
                "limit",
            }

            empty_search = await client.call_tool(
                "find_code",
                {
                    "project_folder": str(FIXTURES),
                    "pattern": "class DefinitelyMissing: $$$BODY",
                    "language": "python",
                    "paths": ["example.py"],
                    "output_format": "json",
                },
            )
            _assert_mirrored_structured_content(
                empty_search,
                {
                    "matches": [],
                    "returned": 0,
                    "truncated": False,
                    "limit": 50,
                    "next_cursor": None,
                    "snapshot_truncated": False,
                },
            )

            empty_outline = await client.call_tool(
                "outline_code",
                {
                    "project_folder": str(FIXTURES),
                    "paths": ["example.py"],
                    "language": "python",
                    "symbol_types": ["enum"],
                    "output_format": "json",
                },
            )
            _assert_mirrored_structured_content(
                empty_outline,
                {
                    "files": [{"file": "example.py", "language": "Python", "items": []}],
                    "returned": 0,
                    "truncated": False,
                    "limit": 50,
                    "resolved_paths": ["example.py"],
                    "path_errors": [],
                },
            )
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_fastmcp_in_process_oxc_module_inspection() -> None:
    helper = resolve_oxc_helper_executable(str(OXC_HELPER), working_directory=REPOSITORY_ROOT)
    versions = read_oxc_helper_versions(helper, timeout_seconds=10)
    runtime = RuntimeServices(
        working_directory=REPOSITORY_ROOT,
        executable=ResolvedExecutable(path=helper.path, command_prefix=helper.command_prefix),
        ast_grep_version=SUPPORTED_AST_GREP_VERSION,
        config_path=None,
        allowed_roots=(REPOSITORY_ROOT,),
        command_timeout_seconds=10,
        default_max_results=50,
        max_results_cap=500,
        forbid_regex_rules=False,
        oxc_helper=helper,
        oxc_versions=versions,
    )
    try:
        async with FastMCPClient(create_mcp(runtime)) as client:
            tools = {tool.name for tool in await client.list_tools()}
            assert tools == OXC_TOOL_NAMES
            assert await client.list_resources() == []
            assert await client.list_resource_templates() == []
            assert await client.list_prompts() == []

            first_modules = await client.call_tool(
                "oxc_modules",
                {
                    "project_folder": str(FIXTURES / "javascript"),
                    "paths": ["entry.ts", "dep.js"],
                    "include_dynamic": True,
                    "max_results": 1,
                    "output_format": "json",
                },
            )
            assert first_modules.is_error is False
            first_content = JAVASCRIPT_MODULE_RESULTS_ADAPTER.validate_python(first_modules.structured_content, strict=True)
            assert first_content["returned"] == 1
            assert first_content["truncated"] is True
            assert first_content["modules"][0]["file"] == "dep.js"
            cursor = first_content.get("next_cursor")
            assert isinstance(cursor, str)

            second_modules = await client.call_tool(
                "oxc_modules",
                {
                    "project_folder": str(FIXTURES / "javascript"),
                    "paths": ["entry.ts", "dep.js"],
                    "include_dynamic": True,
                    "max_results": 1,
                    "cursor": cursor,
                    "output_format": "json",
                },
            )
            assert second_modules.is_error is False
            second_content = JAVASCRIPT_MODULE_RESULTS_ADAPTER.validate_python(second_modules.structured_content, strict=True)
            assert second_content["returned"] == 1
            assert second_content["truncated"] is False
            entry = second_content["modules"][0]
            assert entry["file"] == "entry.ts"
            assert [edge["kind"] for edge in entry["edges"]] == ["import", "reexport", "dynamic"]
            assert entry["edges"][0]["resolved_path"] == "dep.js"
            assert entry["edges"][0]["package_json_path"] == "package.json"
            assert entry["edges"][2]["expression"] == '"./lazy.js"'
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_fastmcp_in_process_semantic_typescript_and_postgresql_backends() -> None:
    oxc = resolve_oxc_helper_executable(str(OXC_HELPER), working_directory=REPOSITORY_ROOT)
    analysis = resolve_oxc_helper_executable(str(ANALYSIS_HELPER), working_directory=REPOSITORY_ROOT)
    typescript = resolve_oxc_helper_executable(str(TYPESCRIPT_PROJECT_HELPER), working_directory=REPOSITORY_ROOT)
    postgres = resolve_oxc_helper_executable(str(POSTGRES_HELPER), working_directory=REPOSITORY_ROOT)
    runtime = RuntimeServices(
        working_directory=REPOSITORY_ROOT,
        executable=oxc,
        ast_grep_version=SUPPORTED_AST_GREP_VERSION,
        config_path=None,
        allowed_roots=(REPOSITORY_ROOT,),
        command_timeout_seconds=60,
        default_max_results=50,
        max_results_cap=500,
        forbid_regex_rules=False,
        oxc_helper=oxc,
        oxc_versions=read_oxc_helper_versions(oxc, timeout_seconds=10),
        analysis_helper=analysis,
        analysis_versions=read_analysis_helper_versions(analysis, timeout_seconds=10),
        typescript_project_helper=typescript,
        typescript_versions=read_typescript_helper_versions(typescript, timeout_seconds=10),
        postgres_helper=postgres,
        postgres_versions=read_postgres_helper_versions(postgres, timeout_seconds=10),
    )
    try:
        async with FastMCPClient(create_mcp(runtime)) as client:
            tools = {tool.name for tool in await client.list_tools()}
            assert tools == ANALYSIS_TOOL_NAMES - {"scan_project_rules", "test_project_rules"}
            semantic = await client.call_tool(
                "semantic_symbols",
                {
                    "project_folder": str(FIXTURES / "typescript-project"),
                    "path": "src/index.ts",
                },
            )
            assert semantic.is_error is False
            symbols = _list(_object(semantic.structured_content)["symbols"])
            assert {_object(symbol)["name"] for symbol in symbols} == {"value", "add", "input"}
            javascript_symbols_result = await client.call_tool(
                "semantic_symbols",
                {
                    "project_folder": str(FIXTURES / "javascript"),
                    "path": "entry.ts",
                },
            )
            assert javascript_symbols_result.is_error is False
            javascript_symbols = _list(_object(javascript_symbols_result.structured_content)["symbols"])
            value_symbol = next(symbol for symbol in javascript_symbols if _object(symbol)["name"] == "value")
            value_position = _integer(_object(_object(value_symbol)["span"])["start"])
            reference_arguments: dict[str, object] = {
                "project_folder": str(FIXTURES / "javascript"),
                "path": "entry.ts",
                "position": value_position,
                "include_declaration": True,
                "include_unresolved": True,
                "project_paths": ["entry.ts", "dep.js"],
                "max_results": 1,
                "output_format": "json",
            }
            reference_records: list[object] = []
            unresolved_records: list[object] = []
            module_links: list[object] = []
            reference_digest: str | None = None
            for _page_number in range(10):
                reference_result = await client.call_tool("semantic_references", reference_arguments)
                assert reference_result.is_error is False
                reference_page = _object(reference_result.structured_content)
                reference_digest = _string(reference_page["source_digest"])
                reference_records.extend(_list(reference_page["references"]))
                unresolved_records.extend(_list(reference_page["unresolved"]))
                module_links.extend(_list(reference_page["module_graph_links"]))
                assert _integer(reference_page["returned"]) <= 1
                next_cursor = reference_page["next_cursor"]
                if next_cursor is None:
                    break
                reference_arguments["cursor"] = _string(next_cursor)
            else:
                pytest.fail("semantic reference pagination did not terminate")
            assert any(_object(record).get("kind") == "declaration" for record in reference_records)
            assert any(_object(record).get("name") == "value" for record in reference_records)
            assert any(_object(record).get("name") == "console" for record in unresolved_records)
            assert any(_object(link).get("importer") == "entry.ts" for link in module_links)
            assert _object(reference_page["coverage"]) == {"lexical": "same_file", "project": "module_graph"}
            assert reference_digest is not None

            stale = await client.call_tool(
                "semantic_scopes",
                {
                    "project_folder": str(FIXTURES / "javascript"),
                    "path": "entry.ts",
                    "source_digest": "0" * 64,
                },
                raise_on_error=False,
            )
            assert stale.is_error is True

            add_symbol = next(symbol for symbol in symbols if _object(symbol)["name"] == "add")
            add_position = _integer(_object(_object(add_symbol)["span"])["start"])
            cfg_result = await client.call_tool(
                "semantic_cfg",
                {
                    "project_folder": str(FIXTURES / "typescript-project"),
                    "path": "src/index.ts",
                    "function_position": add_position,
                },
            )
            assert cfg_result.is_error is False
            assert _list(_object(cfg_result.structured_content)["functions"])
            assert runtime.analysis_worker is not None

            typescript_result = await client.call_tool(
                "inspect_typescript_project",
                {"project_folder": str(FIXTURES / "typescript-project")},
            )
            assert typescript_result.is_error is False
            assert _object(typescript_result.structured_content)["typescript_version"] == "6.0.2"

            postgres_result = await client.call_tool("postgres_deparse_preview", {"sql": "SELECT 1"})
            assert postgres_result.is_error is False
            assert _object(postgres_result.structured_content)["equivalent"] is True
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_stdio_auto_protocol_catalog_calls_metadata_and_outline_contract(
    launcher_python: str,
    ast_grep_executable: str,
) -> None:
    parameters = server_parameters(launcher_python, ast_grep_executable)

    async with Client(stdio_client(parameters), mode="auto") as client:
        assert client.protocol_version == LATEST_MODERN_VERSION
        assert client.session.initialize_result is None
        assert client.server_info is not None
        assert client.server_info.name == "ast-soleaux"
        assert client.server_info.version == "0.5.0"

        listed = await client.list_tools()
        tools = {tool.name: tool for tool in listed.tools}
        _assert_tool_catalog(tools, EXPECTED_STDIO_TOOLS)
        assert (await client.list_resources()).resources == []
        assert (await client.list_resource_templates()).resource_templates == []
        assert (await client.list_prompts()).prompts == []

        info_result = await client.call_tool("get_server_info", {})
        assert info_result.is_error is False
        assert info_result.structured_content is not None
        info = _object(info_result.structured_content)
        assert _object(info["server"]) == {"name": "ast-soleaux", "version": "0.5.0"}
        versions = _object(info["versions"])
        assert versions["ast_grep"] == SUPPORTED_AST_GREP_VERSION
        assert versions["oxc"] == {
            "helper": SUPPORTED_OXC_HELPER_VERSION,
            "parser": SUPPORTED_OXC_PARSER_VERSION,
            "resolver": SUPPORTED_OXC_RESOLVER_VERSION,
        }
        executables = _object(info["executables"])
        assert executables["oxc"] == str(OXC_HELPER)
        configuration = _object(info["configuration"])
        assert configuration["forbid_regex_rules"] is True
        assert "python" in _list(configuration["supported_language_ids"])
        assert "sql" not in _list(configuration["supported_language_ids"])
        assert len(_string(configuration["digest"])) == 64
        capabilities = _object(info["capabilities"])
        assert capabilities["configured_scan"] is True
        assert capabilities["configured_tests"] is True
        assert capabilities["javascript_module_inspection"] is True
        coordinates = _object(info["coordinates"])
        assert coordinates["range"] == "half-open [start,end)"
        assert coordinates["oxc_offset"] == "zero-based UTF-16 code units"
        limits = _object(info["limits"])
        assert limits["default_max_results"] == 50
        assert limits["max_results_cap"] == 500
        assert limits["native_library_bytes"] == 16 * 1024 * 1024
        assert limits["oxc_files"] == 64
        assert limits["oxc_file_bytes"] == 2 * 1024 * 1024
        assert limits["oxc_total_source_bytes"] == 16 * 1024 * 1024
        assert limits["windows_create_process_characters"] == 32_767
        assert limits["posix_arg_headroom_bytes"] == 2048
        assert limits["process_termination_grace_seconds"] == 2.0

        negative = await client.call_tool(
            "test_match_code_rule",
            {
                "code": "value = 1",
                "yaml": "id: no-call\nlanguage: python\nrule:\n  pattern: print($A)\n",
            },
        )
        assert negative.is_error is False
        assert negative.structured_content == {"result": []}

        empty_search = await client.call_tool(
            "find_code",
            {
                "project_folder": str(FIXTURES),
                "pattern": "class DefinitelyMissing: $$$BODY",
                "language": "python",
                "paths": ["example.py"],
                "output_format": "json",
            },
        )
        _assert_mirrored_structured_content(
            empty_search,
            {
                "matches": [],
                "returned": 0,
                "truncated": False,
                "limit": 50,
                "next_cursor": None,
                "snapshot_truncated": False,
            },
        )

        empty_outline = await client.call_tool(
            "outline_code",
            {
                "project_folder": str(FIXTURES),
                "paths": ["example.py"],
                "language": "python",
                "symbol_types": ["enum"],
                "output_format": "json",
            },
        )
        _assert_mirrored_structured_content(
            empty_outline,
            {
                "files": [{"file": "example.py", "language": "Python", "items": []}],
                "returned": 0,
                "truncated": False,
                "limit": 50,
                "resolved_paths": ["example.py"],
                "path_errors": [],
            },
        )

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
        assert set(found.structured_content) == {
            "matches",
            "returned",
            "truncated",
            "limit",
            "next_cursor",
            "snapshot_truncated",
        }
        assert found.structured_content["returned"] == 1
        assert found.structured_content["truncated"] is True
        assert found.structured_content["limit"] == 1
        cursor = found.structured_content["next_cursor"]
        assert isinstance(cursor, str)

        continued = await client.call_tool(
            "find_code",
            {
                "project_folder": str(FIXTURES),
                "pattern": "def $NAME($$$): $$$BODY",
                "language": "python",
                "paths": ["example.py"],
                "max_results": 1,
                "cursor": cursor,
                "output_format": "json",
            },
        )
        assert continued.is_error is False
        assert continued.structured_content is not None
        assert continued.structured_content["returned"] == 1

        invalid_pattern = await client.call_tool(
            "find_code",
            {
                "project_folder": str(FIXTURES),
                "pattern": "class $NAME(BaseProviderClient)",
                "language": "python",
                "paths": ["example.py"],
                "output_format": "json",
            },
        )
        assert invalid_pattern.is_error is False
        assert invalid_pattern.structured_content is not None
        assert invalid_pattern.structured_content["diagnostics"]["kind"] == "pattern_parse"
        assert invalid_pattern.structured_content["diagnostics"]["has_error_node"] is True

        failed_probe = await client.call_tool(
            "find_code_by_rule",
            {
                "project_folder": str(FIXTURES),
                "yaml": "id: print-call\nlanguage: python\nrule:\n  pattern: print($A)\n",
                "paths": ["example.py"],
                "positive_code": "value = 1",
                "output_format": "json",
            },
        )
        assert failed_probe.is_error is False
        assert failed_probe.structured_content is not None
        assert failed_probe.structured_content["diagnostics"]["kind"] == "positive_probe_failed"

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

        configured = await client.call_tool(
            "scan_project_rules",
            {
                "project_folder": str(FIXTURES / "configured"),
                "rule_ids": ["literal.dot+id"],
                "paths": ["example.py"],
                "max_results": 5,
                "run_tests_first": True,
                "output_format": "json",
            },
        )
        assert configured.is_error is False
        assert configured.structured_content is not None
        assert configured.structured_content["returned"] == 1
        assert configured.structured_content["matches"][0]["ruleId"] == "literal.dot+id"

        configured_tests = await client.call_tool("test_project_rules", {"rule_ids": ["configured-print"]})
        assert configured_tests.is_error is False
        assert configured_tests.structured_content is not None
        assert configured_tests.structured_content["passed"] is True
        assert configured_tests.structured_content["report_truncated"] is False

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

        resolved_outline = await client.call_tool(
            "outline_code",
            {
                "project_folder": str(FIXTURES),
                "paths": ["missing.py"],
                "include_globs": ["*.py"],
                "strict_paths": False,
                "max_results": 4,
                "output_format": "json",
            },
        )
        assert resolved_outline.is_error is False
        assert resolved_outline.structured_content is not None
        assert "example.py" in resolved_outline.structured_content["resolved_paths"]
        assert resolved_outline.structured_content["path_errors"][0]["path"] == "missing.py"

        unsupported = await client.call_tool(
            "find_code",
            {
                "project_folder": str(FIXTURES),
                "pattern": "select $A",
                "language": "sql",
                "max_results": 1,
            },
        )
        assert unsupported.is_error is True
        assert "Unsupported language 'sql'" in str(unsupported.content)
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
        assert "Regex ast-grep rules are disabled" in str(rejected.content)


@pytest.mark.asyncio
async def test_stdio_legacy_protocol_handshake_compatibility(
    launcher_python: str,
    ast_grep_executable: str,
) -> None:
    parameters = server_parameters(launcher_python, ast_grep_executable)

    async with Client(stdio_client(parameters), mode="legacy") as client:
        assert client.protocol_version == LATEST_HANDSHAKE_VERSION
        assert client.session.initialize_result is not None
        assert client.session.initialize_result.protocol_version == LATEST_HANDSHAKE_VERSION
        assert client.server_info is not None
        assert client.server_info.name == "ast-soleaux"
        assert client.server_info.version == "0.5.0"
        listed = await client.list_tools()
        assert {tool.name for tool in listed.tools} == EXPECTED_STDIO_TOOLS


@pytest.mark.asyncio
async def test_stdio_server_exits_on_eof_without_a_process_survivor(
    launcher_python: str,
    ast_grep_executable: str,
) -> None:
    parameters = server_parameters(launcher_python, ast_grep_executable)
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
    except BrokenPipeError, ConnectionResetError:
        pass
    returncode = await asyncio.wait_for(process.wait(), timeout=10)
    stdout, stderr = await asyncio.gather(stdout_reader, stderr_reader)

    assert returncode == 0, stderr.decode(errors="replace")
    assert stdout == b""
    assert process.returncode is not None


@pytest.mark.asyncio
async def test_stdio_client_cancellation_reaps_the_server_process(
    monkeypatch: pytest.MonkeyPatch,
    launcher_python: str,
    ast_grep_executable: str,
) -> None:
    created_processes: list[Process | FallbackProcess] = []
    create_process = stdio_module._create_platform_compatible_process

    async def capture_process(
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        errlog: TextIO = sys.stderr,
        cwd: Path | str | None = None,
    ) -> Process | FallbackProcess:
        process = await create_process(command=command, args=args, env=env, errlog=errlog, cwd=cwd)
        created_processes.append(process)
        return process

    monkeypatch.setattr(stdio_module, "_create_platform_compatible_process", capture_process)
    connected = asyncio.Event()
    hold_connection = asyncio.Event()

    async def connect() -> None:
        parameters = server_parameters(launcher_python, ast_grep_executable)
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
