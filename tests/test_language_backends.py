from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from ast_soleaux.server import (
    AstGrepService,
    ResolvedExecutable,
    RuntimeServices,
    create_mcp,
    read_postgres_helper_versions,
    read_typescript_helper_versions,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TYPESCRIPT_HELPER = REPOSITORY_ROOT / "oxc-sidecar" / "bin" / "ast-soleaux-typescript-project.mjs"
POSTGRES_HELPER = REPOSITORY_ROOT / "postgresql-sidecar" / "bin" / "ast-soleaux-postgresql.mjs"


def node_helper(path: Path) -> ResolvedExecutable:
    node = shutil.which("node")
    assert node is not None
    return ResolvedExecutable(path=path.resolve(), command_prefix=(str(Path(node).resolve()), str(path.resolve())))


def runtime(root: Path) -> RuntimeServices:
    typescript = node_helper(TYPESCRIPT_HELPER)
    postgres = node_helper(POSTGRES_HELPER)
    return RuntimeServices(
        working_directory=root,
        executable=ResolvedExecutable(path=Path("/usr/bin/true"), command_prefix=("/usr/bin/true",)),
        ast_grep_version="0.45.0",
        config_path=None,
        allowed_roots=(root.resolve(),),
        command_timeout_seconds=30,
        default_max_results=50,
        max_results_cap=500,
        forbid_regex_rules=True,
        typescript_project_helper=typescript,
        typescript_versions=read_typescript_helper_versions(typescript, timeout_seconds=30),
        postgres_helper=postgres,
        postgres_versions=read_postgres_helper_versions(postgres, timeout_seconds=30),
    )


def test_backend_version_contracts_report_typescript_and_postgresql_18() -> None:
    typescript = node_helper(TYPESCRIPT_HELPER)
    postgres = node_helper(POSTGRES_HELPER)
    assert read_typescript_helper_versions(typescript, timeout_seconds=30) == {
        "worker": "0.1.0",
        "typescript": "6.0.2",
    }
    assert read_postgres_helper_versions(postgres, timeout_seconds=30) == {
        "worker": "0.1.0",
        "parser": "18.0.0",
        "deparser": "18.3.6",
        "postgres_major": 18,
    }


def test_service_returns_typescript_project_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (tmp_path / "tsconfig.json").write_text(
        json.dumps(
            {
                "compilerOptions": {
                    "baseUrl": ".",
                    "module": "NodeNext",
                    "moduleResolution": "NodeNext",
                    "outDir": "dist",
                    "paths": {"@lib/*": ["src/*"]},
                    "strict": True,
                    "target": "ES2022",
                },
                "include": ["src/**/*.ts"],
            }
        ),
        encoding="utf-8",
    )
    (source / "value.ts").write_text("export const value = 7\n", encoding="utf-8")
    (source / "entry.ts").write_text(
        'import { value } from "@lib/value"\nconst inferred = value + 1\nexport { inferred }\n',
        encoding="utf-8",
    )
    service = AstGrepService(runtime(tmp_path))
    result = service.inspect_typescript_project(
        project_folder=str(tmp_path),
        tsconfig="tsconfig.json",
        paths=None,
        include_emit=True,
        include_code_actions=True,
        max_results=100,
    )
    assert result["typescript_version"] == "6.0.2"
    assert result["source_digest"]
    modules = result["modules"]
    assert isinstance(modules, list)
    resolved_alias = False
    for module in modules:
        if not isinstance(module, dict):
            continue
        imports = module.get("imports")
        if not isinstance(imports, list):
            continue
        if any(
            isinstance(item, dict) and item.get("specifier") == "@lib/value" and item.get("resolved_path") == "src/value.ts"
            for item in imports
        ):
            resolved_alias = True
            break
    assert resolved_alias


def test_service_returns_postgresql_18_operations_and_deparse_proof(tmp_path: Path) -> None:
    service = AstGrepService(runtime(tmp_path))
    parsed = service.inspect_postgres(operation="parse", sql="CREATE TABLE app.users (id bigint PRIMARY KEY);")
    assert parsed["parser_version"] == "18.0.0"
    assert parsed["deparser_version"] == "18.3.6"
    assert parsed["postgres_major"] == 18
    declarations = parsed["declarations"]
    assert isinstance(declarations, list)
    assert any(isinstance(item, dict) and item.get("name") == "app.users" for item in declarations)
    deparsed = service.inspect_postgres(operation="deparse", sql="SELECT id FROM app.users WHERE id = 7")
    assert deparsed["equivalent"] is True
    assert deparsed["original_tree_digest"] == deparsed["reparsed_tree_digest"]


def test_capability_gates_expose_only_configured_language_backends(tmp_path: Path) -> None:
    async def inspect_catalog() -> tuple[set[str], int, int, int]:
        server = create_mcp(runtime(tmp_path))
        return (
            {tool.name for tool in await server.list_tools()},
            len(await server.list_resources()),
            len(await server.list_resource_templates()),
            len(await server.list_prompts()),
        )

    tools, resources, templates, prompts = asyncio.run(inspect_catalog())
    assert tools == {
        "dump_syntax_tree",
        "test_match_code_rule",
        "outline_code",
        "find_code",
        "find_code_by_rule",
        "get_server_info",
        "inspect_typescript_project",
        "postgres_parse",
        "postgres_parse_files",
        "postgres_deparse_preview",
    }
    assert (resources, templates, prompts) == (0, 0, 0)
