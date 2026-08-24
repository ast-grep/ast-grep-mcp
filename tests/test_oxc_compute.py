from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from fastmcp import Client

from ast_soleaux.server import (
    SUPPORTED_AST_GREP_VERSION,
    AstGrepService,
    OxcMinifyOptions,
    OxcTransformOptions,
    ResolvedExecutable,
    RuntimeServices,
    create_mcp,
    read_oxc_helper_versions,
    resolve_oxc_helper_executable,
)

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "oxc-sidecar" / "bin" / "ast-soleaux-oxc.mjs"


def service(project: Path) -> AstGrepService:
    helper = resolve_oxc_helper_executable(str(HELPER), working_directory=ROOT)
    runtime = RuntimeServices(
        working_directory=project,
        executable=ResolvedExecutable(path=helper.path, command_prefix=helper.command_prefix),
        ast_grep_version=SUPPORTED_AST_GREP_VERSION,
        config_path=None,
        allowed_roots=(project,),
        command_timeout_seconds=30,
        default_max_results=50,
        max_results_cap=500,
        forbid_regex_rules=False,
        oxc_helper=helper,
        oxc_versions=read_oxc_helper_versions(helper, timeout_seconds=10),
    )
    return AstGrepService(runtime)


def test_project_formatting_writes_use_oxfmt_directly(tmp_path: Path) -> None:
    npm = shutil.which("npm")
    assert npm is not None
    source = tmp_path / "entry.ts"
    source.write_text('export const label:string="value";\n', encoding="utf-8")
    (tmp_path / ".oxfmtrc.json").write_text('{"singleQuote":true}\n', encoding="utf-8")

    subprocess.run(
        [npm, "exec", "--prefix", str(ROOT / "oxc-sidecar"), "--", "oxfmt", "--write", str(source)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert source.read_text(encoding="utf-8") == "export const label: string = 'value';\n"


def test_artifact_emission_tools_are_fully_usable(tmp_path: Path) -> None:
    project = tmp_path.resolve()
    source = project / "src" / "entry.ts"
    source.parent.mkdir()
    source.write_text('export const label:string="value"; export const value:number={a:1}.a\n', encoding="utf-8")
    oxc = service(project)

    transformed = oxc.emit_files(
        operation="transform",
        project_folder=str(project),
        output_root="dist",
        paths=["src/entry.ts"],
        include_globs=None,
        exclude_globs=None,
        strict_paths=True,
        conflict_policy="error",
        allow_source_overwrite=False,
        options=OxcTransformOptions(lang="ts", sourcemap=True, declaration=True),
    )
    assert (project / "dist" / "src" / "entry.js").is_file()
    assert (project / "dist" / "src" / "entry.js.map").is_file()
    assert (project / "dist" / "src" / "entry.d.ts").is_file()
    assert transformed["applied"]

    minified = oxc.emit_files(
        operation="minify",
        project_folder=str(project),
        output_root="min",
        paths=["dist/src/entry.js"],
        include_globs=None,
        exclude_globs=None,
        strict_paths=True,
        conflict_policy="error",
        allow_source_overwrite=False,
        options=OxcMinifyOptions(sourcemap=True),
    )
    assert (project / "min" / "dist" / "src" / "entry.min.js").is_file()
    assert minified["applied"]


@pytest.mark.asyncio
async def test_public_oxc_preview_and_artifact_contracts(tmp_path: Path) -> None:
    project = tmp_path.resolve()
    source = project / "src" / "entry.ts"
    source.parent.mkdir()
    source.write_text('export const label:string="value";\n', encoding="utf-8")
    helper = resolve_oxc_helper_executable(str(HELPER), working_directory=ROOT)
    runtime = RuntimeServices(
        working_directory=project,
        executable=ResolvedExecutable(path=helper.path, command_prefix=helper.command_prefix),
        ast_grep_version=SUPPORTED_AST_GREP_VERSION,
        config_path=None,
        allowed_roots=(project,),
        command_timeout_seconds=30,
        default_max_results=50,
        max_results_cap=500,
        forbid_regex_rules=False,
        oxc_helper=helper,
        oxc_versions=read_oxc_helper_versions(helper, timeout_seconds=10),
    )
    try:
        async with Client(create_mcp(runtime)) as client:
            transformed = await client.call_tool(
                "oxc_transform",
                {
                    "source": {"kind": "file", "project_folder": str(project), "path": "src/entry.ts"},
                    "options": {"lang": "ts", "sourcemap": True},
                    "output_format": "json",
                },
            )
            assert transformed.is_error is False
            transformed_content = transformed.structured_content
            assert isinstance(transformed_content, dict)
            assert transformed_content["source_digest"]

            minified = await client.call_tool(
                "oxc_minify",
                {
                    "source": {"kind": "inline", "filename": "entry.js", "code": "const value = 1 + 2;"},
                    "output_format": "json",
                },
            )
            assert minified.is_error is False
            minified_content = minified.structured_content
            assert isinstance(minified_content, dict)
            assert minified_content["assumptions"]

            emitted = await client.call_tool(
                "oxc_transform_files",
                {
                    "project_folder": str(project),
                    "output_root": "dist",
                    "paths": ["src/entry.ts"],
                    "max_results": 1,
                    "options": {"lang": "ts"},
                },
            )
            assert emitted.is_error is False
            emitted_content = emitted.structured_content
            assert isinstance(emitted_content, dict)
            assert emitted_content["emitted_files"]
            assert (project / "dist" / "src" / "entry.js").is_file()
    finally:
        runtime.close()
