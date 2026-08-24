from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import io
import sys
import tarfile
import tomllib
import zipfile
from contextlib import redirect_stdout
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Protocol


class DistributionMetadata(Protocol):
    def __getitem__(self, key: str) -> str | None: ...


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _validate_metadata(metadata: DistributionMetadata, expected_version: str, artifact: str) -> None:
    _require(metadata["Name"] == "ast-soleaux", f"{artifact} has an unexpected package name")
    _require(metadata["Version"] == expected_version, f"{artifact} has an unexpected version")
    _require(metadata["Requires-Python"] == ">=3.14", f"{artifact} has an unexpected Python requirement")
    _require(metadata["License-Expression"] == "MIT", f"{artifact} has an unexpected license expression")


def _safe_archive_name(name: str, artifact: str) -> None:
    path = PurePosixPath(name)
    _require(not path.is_absolute() and ".." not in path.parts, f"{artifact} contains an unsafe path: {name}")


def _check_wheel(wheel: Path, expected_version: str) -> None:
    expected_name = f"ast_soleaux-{expected_version}-py3-none-any.whl"
    _require(wheel.name == expected_name, f"wheel filename must be {expected_name}")
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            _safe_archive_name(name, "wheel")

    distributions = list(importlib.metadata.distributions(path=[str(wheel.resolve())]))
    _require(len(distributions) == 1, "wheel must contain exactly one distribution")
    distribution = distributions[0]
    _validate_metadata(distribution.metadata, expected_version, "wheel")
    entry_points = [entry for entry in distribution.entry_points if entry.group == "console_scripts" and entry.name == "ast-soleaux"]
    _require(len(entry_points) == 1, "wheel must contain exactly one ast-soleaux entry point")
    _require(entry_points[0].value == "ast_soleaux.cli:main", "wheel has an unexpected ast-soleaux entry point")

    wheel_path = str(wheel.resolve())
    sys.path.insert(0, wheel_path)
    previous_argv = sys.argv
    for module_name in ("ast_soleaux.server", "ast_soleaux.config_snapshot"):
        sys.modules.pop(module_name, None)
    try:
        main_module = importlib.import_module("ast_soleaux.server")
        snapshot_module = importlib.import_module("ast_soleaux.config_snapshot")
        normalized_wheel_path = wheel.resolve().as_posix()
        main_origin = str(main_module.__file__).replace("\\", "/")
        snapshot_origin = str(snapshot_module.__file__).replace("\\", "/")
        _require(main_origin.startswith(f"{normalized_wheel_path}/"), "ast_soleaux.server was not imported from the wheel")
        _require(
            snapshot_origin.startswith(f"{normalized_wheel_path}/"),
            "ast_soleaux.config_snapshot was not imported from the wheel",
        )
        _require(main_module._server_version() == expected_version, "wheel reports an unexpected server version")
        entry_point = entry_points[0].load()
        sys.argv = ["ast-soleaux", "--help"]
        output = io.StringIO()
        try:
            with redirect_stdout(output):
                entry_point()
        except SystemExit as error:
            _require(error.code == 0, "wheel entry point --help failed")
        else:
            raise RuntimeError("wheel entry point --help did not exit")
        _require(
            "Bounded structural analysis, transformation, and execution MCP server" in output.getvalue(),
            "wheel entry point emitted unexpected help",
        )
    finally:
        sys.argv = previous_argv
        sys.path.remove(wheel_path)
        for module_name in ("ast_soleaux.server", "ast_soleaux.config_snapshot"):
            sys.modules.pop(module_name, None)


def _check_sdist(sdist: Path, expected_version: str) -> None:
    expected_root = f"ast_soleaux-{expected_version}"
    expected_name = f"{expected_root}.tar.gz"
    _require(sdist.name == expected_name, f"sdist filename must be {expected_name}")
    source_root = Path(__file__).parents[1]
    claude_rule_files = {
        f"{expected_root}/{path.relative_to(source_root).as_posix()}"
        for path in sorted((source_root / ".claude" / "rules").rglob("*.md"))
        if path.is_file()
    }
    codex_rule_files = {
        f"{expected_root}/{path.relative_to(source_root).as_posix()}"
        for path in sorted((source_root / ".codex" / "rules").rglob("*.rules"))
        if path.is_file()
    }
    required_files = (
        {
            f"{expected_root}/AGENTS.md",
            f"{expected_root}/CLAUDE.md",
            f"{expected_root}/LICENSE",
            f"{expected_root}/MANIFEST.in",
            f"{expected_root}/README.md",
            f"{expected_root}/PKG-INFO",
            f"{expected_root}/ast-grep.mdc",
            f"{expected_root}/analysis-sidecar/Cargo.lock",
            f"{expected_root}/analysis-sidecar/Cargo.toml",
            f"{expected_root}/analysis-sidecar/rust-toolchain.toml",
            f"{expected_root}/analysis-sidecar/src/main.rs",
            f"{expected_root}/analysis-sidecar/tests/protocol.rs",
            f"{expected_root}/execution-sidecar/ast_soleaux_typescript_sandbox.py",
            f"{expected_root}/postgresql-sidecar/bin/ast-soleaux-postgresql.mjs",
            f"{expected_root}/postgresql-sidecar/package-lock.json",
            f"{expected_root}/postgresql-sidecar/package.json",
            f"{expected_root}/postgresql-sidecar/test/ast-soleaux-postgresql.test.mjs",
            f"{expected_root}/ast_soleaux/__init__.py",
            f"{expected_root}/ast_soleaux/cli.py",
            f"{expected_root}/ast_soleaux/config_snapshot.py",
            f"{expected_root}/ast_soleaux/contracts.py",
            f"{expected_root}/ast_soleaux/mutation.py",
            f"{expected_root}/ast_soleaux/server.py",
            f"{expected_root}/ast_soleaux/worker.py",
            f"{expected_root}/oxc-sidecar/.oxfmtrc.json",
            f"{expected_root}/oxc-sidecar/bin/ast-soleaux-typescript-project.mjs",
            f"{expected_root}/oxc-sidecar/bin/ast-soleaux-oxc.mjs",
            f"{expected_root}/oxc-sidecar/package-lock.json",
            f"{expected_root}/oxc-sidecar/package.json",
            f"{expected_root}/oxc-sidecar/test/ast-soleaux-oxc.test.mjs",
            f"{expected_root}/oxc-sidecar/test/typescript-project.test.mjs",
            f"{expected_root}/pyproject.toml",
            f"{expected_root}/uv.lock",
            f"{expected_root}/.python-version",
            f"{expected_root}/scripts/generate_contract.py",
            f"{expected_root}/scripts/launch_server.py",
            f"{expected_root}/tests/fixtures/configured/sgconfig.yml",
            f"{expected_root}/tests/fixtures/contracts/ast-soleaux-0.5.0.json",
            f"{expected_root}/tests/fixtures/javascript/dep.js",
            f"{expected_root}/tests/fixtures/javascript/entry.ts",
            f"{expected_root}/tests/fixtures/javascript/package.json",
            f"{expected_root}/tests/package_smoke.py",
        }
        | claude_rule_files
        | codex_rule_files
    )
    with tarfile.open(sdist, mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            _safe_archive_name(member.name, "sdist")
            _require(not member.issym() and not member.islnk(), f"sdist contains a link: {member.name}")
        names = {member.name for member in members}
        missing = sorted(required_files - names)
        _require(not missing, f"sdist is missing required files: {', '.join(missing)}")
        archived_claude_rule_files = {name for name in names if name.startswith(f"{expected_root}/.claude/rules/") and name.endswith(".md")}
        archived_codex_rule_files = {
            name for name in names if name.startswith(f"{expected_root}/.codex/rules/") and name.endswith(".rules")
        }
        _require(archived_claude_rule_files == claude_rule_files, "sdist Claude rule inventory does not match the source tree")
        _require(archived_codex_rule_files == codex_rule_files, "sdist Codex rule inventory does not match the source tree")
        metadata_file = archive.extractfile(f"{expected_root}/PKG-INFO")
        project_file = archive.extractfile(f"{expected_root}/pyproject.toml")
        if metadata_file is None or project_file is None:
            raise RuntimeError("sdist metadata could not be read")
        metadata = BytesParser().parsebytes(metadata_file.read())
        project = tomllib.loads(project_file.read().decode("utf-8"))
    _validate_metadata(metadata, expected_version, "sdist")
    _require(project["project"]["version"] == expected_version, "sdist pyproject has an unexpected version")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("sdist", type=Path)
    parser.add_argument("version")
    arguments = parser.parse_args()
    _check_wheel(arguments.wheel.resolve(strict=True), arguments.version)
    _check_sdist(arguments.sdist.resolve(strict=True), arguments.version)
    print(f"validated ast-soleaux {arguments.version} wheel, entry point, and sdist")


if __name__ == "__main__":
    main()
