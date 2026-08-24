from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import platform as platform_module
import queue
import re
import secrets
import shutil
import struct
import subprocess
import sys
import threading
import time
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import ExitStack, asynccontextmanager
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, BinaryIO, Final, Literal, NotRequired, Protocol, TypedDict, cast

import yaml as yaml_parser
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from ast_soleaux.config_snapshot import (
    MAX_CONFIG_FILE_BYTES,
    MAX_CONFIG_RESOURCE_BYTES,
    MAX_CONFIG_RESOURCE_FILES,
    MAX_NATIVE_LIBRARY_BYTES,
    MAX_YAML_DEPTH,
    MAX_YAML_DOCUMENTS,
    MAX_YAML_NODES,
    ConfigSnapshot,
    create_config_snapshot,
    is_within,
    load_strict_yaml_documents,
    private_runtime_root,
    validate_rule_utility_cycles,
)
from ast_soleaux.mutation import ConflictPolicy, MutationService, PlannedWrite
from ast_soleaux.worker import (
    PROCESS_TERMINATION_GRACE_SECONDS,
    JsonLineWorker,
    popen_process_group_options,
    process_group_id,
    terminate_and_reap,
)

DEFAULT_MAX_RESULTS: Final = 50
HARD_MAX_RESULTS: Final = 500
DEFAULT_COMMAND_TIMEOUT_SECONDS: Final = 30.0
FALLBACK_SERVER_VERSION: Final = "0+unknown"
NEUTRAL_AST_GREP_CONFIG: Final = "ruleDirs: []\n"
SUPPORTED_AST_GREP_VERSION: Final = "0.45.0"
SUPPORTED_OXC_HELPER_VERSION: Final = "0.1.0"
SUPPORTED_OXC_PARSER_VERSION: Final = "0.147.0"
SUPPORTED_OXC_RESOLVER_VERSION: Final = "11.24.2"
SUPPORTED_ANALYSIS_WORKER_VERSION: Final = "0.1.0"
SUPPORTED_TYPESCRIPT_WORKER_VERSION: Final = "0.1.0"
SUPPORTED_TYPESCRIPT_VERSION: Final = "6.0.2"
SUPPORTED_POSTGRES_WORKER_VERSION: Final = "0.1.0"
SUPPORTED_POSTGRES_PARSER_VERSION: Final = "18.0.0"
SUPPORTED_POSTGRES_DEPARSER_VERSION: Final = "18.3.6"
SUPPORTED_POSTGRES_MAJOR: Final = 18
SUPPORTED_TYPESCRIPT_EXECUTION_WORKER_VERSION: Final = "0.1.0"
SUPPORTED_TYPESCRIPT_EXECUTION_RUNTIME: Final = "@oxc-node/core@0.1.0"
MAX_INLINE_RULE_BYTES: Final = 64 * 1024
MAX_SNIPPET_INPUT_BYTES: Final = 1024 * 1024
MAX_OUTLINE_PATHS: Final = 64
MAX_OXC_FILES: Final = MAX_OUTLINE_PATHS
MAX_OXC_FILE_BYTES: Final = 2 * 1024 * 1024
MAX_OXC_TOTAL_SOURCE_BYTES: Final = 16 * 1024 * 1024
MAX_NDJSON_RECORD_BYTES: Final = 1024 * 1024
MAX_STRUCTURED_OUTPUT_BYTES: Final = 4 * 1024 * 1024
MAX_OUTLINE_RECORD_BYTES: Final = MAX_NDJSON_RECORD_BYTES
PROCESS_READ_CHUNK_BYTES: Final = 64 * 1024
OUTLINE_READ_CHUNK_BYTES: Final = PROCESS_READ_CHUNK_BYTES
MAX_SUBPROCESS_DIAGNOSTIC_BYTES: Final = 64 * 1024
MAX_TEST_REPORT_BYTES: Final = 64 * 1024
WINDOWS_CREATE_PROCESS_LIMIT: Final = 32_767
POSIX_ARG_HEADROOM_BYTES: Final = 2048
CURSOR_TTL_SECONDS: Final = 300.0
MAX_CURSOR_SNAPSHOTS: Final = 64
OXC_SOURCE_EXTENSIONS: Final = frozenset({".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"})
BUILTIN_LANGUAGE_IDS: Final = (
    "bash",
    "c",
    "c++",
    "cc",
    "cpp",
    "cs",
    "csharp",
    "css",
    "cxx",
    "elixir",
    "ex",
    "go",
    "golang",
    "haskell",
    "hcl",
    "hs",
    "html",
    "java",
    "javascript",
    "js",
    "json",
    "jsx",
    "kotlin",
    "kt",
    "lua",
    "nix",
    "php",
    "py",
    "python",
    "rb",
    "rs",
    "ruby",
    "rust",
    "scala",
    "sol",
    "solidity",
    "swift",
    "ts",
    "tsx",
    "typescript",
    "yml",
)

DumpFormat = Literal["pattern", "cst", "ast", "sexp"]
OutputFormat = Literal["text", "json"]
Strictness = Literal["cst", "smart", "ast", "relaxed", "signature", "template"]
OutlineItemsMode = Literal["auto", "structure", "exports", "imports", "all"]
type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
type JsonObject = dict[str, JsonValue]
CompletedTextProcess = subprocess.CompletedProcess[str]
ProcessRunner = Callable[..., CompletedTextProcess]
PopenFactory = Callable[..., subprocess.Popen[bytes]]
OutlineProcessResult = tuple[list[JsonObject], bool]
OutlineProcessRunner = Callable[..., OutlineProcessResult]
NDJSONRecordParser = Callable[[bytes, int], tuple[JsonObject, int] | None]


class SysconfCallable(Protocol):
    def __call__(self, name: str, /) -> int: ...


class InlineSource(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    kind: Literal["inline"]
    filename: Annotated[str, Field(min_length=1, max_length=1024)]
    code: Annotated[str, Field(max_length=MAX_SNIPPET_INPUT_BYTES)]


class FileSource(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    kind: Literal["file"]
    project_folder: Annotated[str, Field(min_length=1)]
    path: Annotated[str, Field(min_length=1, max_length=4096)]


OxcSource = Annotated[InlineSource | FileSource, Field(discriminator="kind")]


class OxcTransformOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lang: Literal["js", "jsx", "ts", "tsx", "dts"] | None = None
    source_type: Literal["script", "module", "commonjs", "unambiguous"] | None = None
    target: str | None = None
    sourcemap: bool = False
    declaration: bool = False


class OxcMinifyOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    compress: bool = True
    mangle: bool = True
    sourcemap: bool = False
    legal_comments: Literal["none", "inline", "linked", "external"] = "inline"


class SearchResults(TypedDict):
    matches: list[JsonObject]
    returned: int
    truncated: bool
    limit: int
    next_cursor: NotRequired[str | None]
    snapshot_truncated: NotRequired[bool]
    diagnostics: NotRequired[JsonObject]


class CursorPage(SearchResults):
    metadata: NotRequired[JsonObject]


class OutlineFile(TypedDict):
    file: str
    language: str
    items: list[JsonObject]


class OutlinePathError(TypedDict):
    path: str
    error: str


class OutlineResults(TypedDict):
    files: list[OutlineFile]
    returned: int
    truncated: bool
    limit: int
    resolved_paths: NotRequired[list[str]]
    path_errors: NotRequired[list[OutlinePathError]]


class ProjectTestResults(TypedDict):
    passed: bool
    report: str
    report_truncated: bool


class OxcVersions(TypedDict):
    helper: str
    parser: str
    resolver: str


class OxcSpan(TypedDict):
    start: int
    end: int


class OxcPackageMetadata(TypedDict):
    path: str
    name: str | None
    type: str | None


class OxcCommonJSExport(OxcSpan):
    text: str


class OxcModuleGraphNode(TypedDict):
    file: str
    source_type: Literal["module", "script"]
    package: OxcPackageMetadata | None


class OxcModuleGraphEdge(TypedDict):
    importer: str
    kind: Literal["import", "reexport", "dynamic"]
    module_system: Literal["esm", "commonjs"]
    specifier: str | None
    expression: str | None
    start: int
    end: int
    resolution: Literal["resolved", "external", "unresolved", "dynamic"]
    target: str | None
    module_type: str | None
    resolution_error: str | None


class OxcModuleGraph(TypedDict):
    version: int
    nodes: list[OxcModuleGraphNode]
    edges: list[OxcModuleGraphEdge]


class OxcDiagnosticLabel(TypedDict):
    message: str | None
    start: int
    end: int


class OxcDiagnostic(TypedDict):
    severity: str
    message: str
    help: str | None
    codeframe: str | None
    labels: list[OxcDiagnosticLabel]


class JavascriptModuleEdge(TypedDict):
    kind: Literal["import", "reexport", "dynamic"]
    module_system: Literal["esm", "commonjs"]
    specifier: str | None
    expression: str | None
    start: int
    end: int
    resolution: Literal["resolved", "external", "unresolved", "dynamic"]
    resolved_path: str | None
    package_json_path: str | None
    module_type: str | None
    resolution_error: str | None


class JavascriptModule(TypedDict):
    file: str
    has_module_syntax: bool
    source_type: Literal["module", "script"]
    package: OxcPackageMetadata | None
    commonjs_exports: list[OxcCommonJSExport]
    import_meta_spans: list[OxcSpan]
    edges: list[JavascriptModuleEdge]
    diagnostics: list[OxcDiagnostic]


class JavascriptModuleResults(TypedDict):
    modules: list[JavascriptModule]
    returned: int
    truncated: bool
    limit: int
    next_cursor: NotRequired[str | None]
    snapshot_truncated: NotRequired[bool]
    source_digest: str
    diagnostics: list[OxcDiagnostic]


class OxcSidecarResponse(TypedDict):
    versions: OxcVersions
    graph_version: int
    graph: OxcModuleGraph
    modules: list[JavascriptModule]
    cache_hit: bool


class SemanticCoverage(TypedDict):
    lexical: Literal["same_file"]
    project: Literal["module_graph", "not_available"]


class SemanticScopesResults(TypedDict):
    file: str
    source_digest: str
    coverage: SemanticCoverage
    scopes: list[JsonObject]
    returned: int
    truncated: bool
    limit: int
    next_cursor: str | None
    diagnostics: list[JsonObject]


class SemanticSymbolsResults(TypedDict):
    file: str
    source_digest: str
    coverage: SemanticCoverage
    symbols: list[JsonObject]
    returned: int
    truncated: bool
    limit: int
    next_cursor: str | None
    diagnostics: list[JsonObject]


class SemanticReferencesResults(TypedDict):
    file: str
    source_digest: str
    coverage: SemanticCoverage
    target: JsonObject | None
    references: list[JsonObject]
    unresolved: list[JsonObject]
    module_graph_links: list[JsonObject]
    returned: int
    truncated: bool
    limit: int
    next_cursor: str | None
    diagnostics: list[JsonObject]


class SemanticCfgResults(TypedDict):
    file: str
    source_digest: str
    coverage: SemanticCoverage
    functions: list[JsonObject]
    returned: int
    truncated: bool
    limit: int
    next_cursor: str | None
    diagnostics: list[JsonObject]


class TypeScriptProjectResults(TypedDict):
    typescript_version: str
    tsconfig: str
    root_files: list[str]
    options: JsonObject
    diagnostics: list[JsonObject]
    modules: list[JsonObject]
    symbols: list[JsonObject]
    inferred_types: list[JsonObject]
    emit: list[JsonObject]
    code_actions: list[JsonObject]
    source_digest: str
    returned: int
    truncated: bool
    limit: int


class PostgresParseResults(TypedDict, total=False):
    parser_version: str
    deparser_version: str
    postgres_major: int
    mode: Literal["parse", "scan", "fingerprint", "plpgsql"]
    source_digest: str
    tree: JsonObject | None
    tokens: list[JsonObject]
    fingerprint: str | None
    normalized: str | None
    plpgsql: JsonObject | None
    statements: list[JsonObject]
    declarations: list[JsonObject]
    references: list[JsonObject]
    calls: list[JsonObject]
    diagnostics: list[JsonObject]
    returned: int
    truncated: bool
    limit: int


class PostgresFilesResults(TypedDict):
    files: list[JsonObject]
    returned: int
    truncated: bool
    limit: int
    next_cursor: NotRequired[str | None]
    snapshot_truncated: NotRequired[bool]


class PostgresDeparseResults(TypedDict):
    parser_version: str
    deparser_version: str
    postgres_major: int
    mode: Literal["deparse"]
    source_digest: str
    original_sql: str
    deparsed_sql: str
    equivalent: bool
    original_tree_digest: str
    reparsed_tree_digest: str
    diagnostics: list[JsonObject]


class PublicServerInfo(TypedDict):
    server: JsonObject
    versions: JsonObject
    executables: JsonObject
    allowed_roots: list[str]
    capabilities: dict[str, bool]
    limits: JsonObject
    coordinates: dict[str, str]
    configuration: JsonObject


class ServerInfo(TypedDict):
    fork_version: str
    ast_grep_executable: str
    ast_grep_version: str
    oxc_helper_executable: str | None
    oxc_versions: OxcVersions | None
    analysis_helper_executable: str | None
    analysis_versions: JsonObject | None
    typescript_project_helper_executable: str | None
    typescript_versions: JsonObject | None
    postgres_helper_executable: str | None
    postgres_versions: JsonObject | None
    typescript_execution_helper_executable: str | None
    typescript_execution_versions: JsonObject | None
    typescript_execution_profile: str
    config_path: str | None
    allowed_roots: list[str]
    command_timeout_seconds: float
    default_max_results: int
    max_results_cap: int
    forbid_regex_rules: bool
    configuration_digest: str
    configuration_provenance: JsonObject
    capabilities: dict[str, bool]
    coordinate_conventions: dict[str, str]
    resource_limits: dict[str, int | float]
    supported_language_ids: list[str]


SEARCH_RESULTS_OUTPUT_SCHEMA: Final = TypeAdapter(SearchResults).json_schema()
OUTLINE_RESULTS_OUTPUT_SCHEMA: Final = TypeAdapter(OutlineResults).json_schema()
JAVASCRIPT_MODULE_RESULTS_OUTPUT_SCHEMA: Final = TypeAdapter(JavascriptModuleResults).json_schema()
TYPESCRIPT_PROJECT_RESULTS_OUTPUT_SCHEMA: Final = TypeAdapter(TypeScriptProjectResults).json_schema()
POSTGRES_PARSE_RESULTS_OUTPUT_SCHEMA: Final = TypeAdapter(PostgresParseResults).json_schema()
POSTGRES_FILES_RESULTS_OUTPUT_SCHEMA: Final = TypeAdapter(PostgresFilesResults).json_schema()
POSTGRES_DEPARSE_RESULTS_OUTPUT_SCHEMA: Final = TypeAdapter(PostgresDeparseResults).json_schema()
JSON_OBJECT_OUTPUT_SCHEMA: Final = TypeAdapter(JsonObject).json_schema()
JSON_VALUE_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
JSON_OBJECT_ADAPTER: Final[TypeAdapter[JsonObject]] = TypeAdapter(JsonObject)
JAVASCRIPT_MODULE_LIST_ADAPTER: Final = TypeAdapter(list[JavascriptModule])
OXC_VERSIONS_ADAPTER: Final = TypeAdapter(OxcVersions)
OXC_SIDECAR_RESPONSE_ADAPTER: Final = TypeAdapter(OxcSidecarResponse)
OXC_DIAGNOSTIC_LIST_ADAPTER: Final = TypeAdapter(list[OxcDiagnostic])
PUBLIC_SERVER_INFO_ADAPTER: Final = TypeAdapter(PublicServerInfo)


@dataclass
class SearchCursorSnapshot:
    query_digest: str
    matches: list[JsonObject]
    offset: int
    expires_at: float
    source_truncated: bool
    metadata: JsonObject | None


class ResultCursorStore:
    def __init__(self) -> None:
        self._search: dict[str, SearchCursorSnapshot] = {}

    def clear(self) -> None:
        self._search.clear()

    def _prune(self) -> None:
        now = time.monotonic()
        expired = [token for token, snapshot in self._search.items() if snapshot.expires_at <= now]
        for token in expired:
            self._search.pop(token, None)
        while len(self._search) >= MAX_CURSOR_SNAPSHOTS:
            oldest = min(self._search, key=lambda token: self._search[token].expires_at)
            self._search.pop(oldest, None)

    def first_search_page(
        self,
        *,
        query_digest: str,
        matches: list[JsonObject],
        page_size: int,
        source_truncated: bool,
        metadata: JsonObject | None = None,
    ) -> CursorPage:
        self._prune()
        page = matches[:page_size]
        remaining = len(matches) > page_size
        next_cursor: str | None = None
        page_metadata = dict(metadata) if metadata is not None else None
        if remaining:
            next_cursor = secrets.token_urlsafe(24)
            self._search[next_cursor] = SearchCursorSnapshot(
                query_digest=query_digest,
                matches=matches,
                offset=page_size,
                expires_at=time.monotonic() + CURSOR_TTL_SECONDS,
                source_truncated=source_truncated,
                metadata=page_metadata,
            )
        result: CursorPage = {
            "matches": page,
            "returned": len(page),
            "truncated": remaining or source_truncated,
            "limit": page_size,
            "next_cursor": next_cursor,
            "snapshot_truncated": source_truncated,
        }
        if page_metadata is not None:
            result["metadata"] = page_metadata
        return result

    def next_search_page(
        self,
        *,
        cursor: str,
        query_digest: str,
        page_size: int,
    ) -> CursorPage:
        self._prune()
        snapshot = self._search.get(cursor)
        if snapshot is None:
            raise ValueError("cursor is invalid or expired")
        if snapshot.query_digest != query_digest:
            raise ValueError("cursor does not match this search query")
        start = snapshot.offset
        end = min(start + page_size, len(snapshot.matches))
        page = snapshot.matches[start:end]
        snapshot.offset = end
        snapshot.expires_at = time.monotonic() + CURSOR_TTL_SECONDS
        has_more = end < len(snapshot.matches)
        if not has_more:
            self._search.pop(cursor, None)
        result: CursorPage = {
            "matches": page,
            "returned": len(page),
            "truncated": has_more or snapshot.source_truncated,
            "limit": page_size,
            "next_cursor": cursor if has_more else None,
            "snapshot_truncated": snapshot.source_truncated,
        }
        if snapshot.metadata is not None:
            result["metadata"] = dict(snapshot.metadata)
        return result


@dataclass(frozen=True)
class ResolvedExecutable:
    path: Path
    command_prefix: tuple[str, ...]


@dataclass
class RuntimeServices:
    working_directory: Path
    executable: ResolvedExecutable
    ast_grep_version: str
    config_path: Path | None
    allowed_roots: tuple[Path, ...]
    command_timeout_seconds: float
    default_max_results: int
    max_results_cap: int
    forbid_regex_rules: bool
    oxc_helper: ResolvedExecutable | None = None
    oxc_versions: OxcVersions | None = None
    analysis_helper: ResolvedExecutable | None = None
    analysis_versions: JsonObject | None = None
    typescript_project_helper: ResolvedExecutable | None = None
    typescript_versions: JsonObject | None = None
    postgres_helper: ResolvedExecutable | None = None
    postgres_versions: JsonObject | None = None
    typescript_execution_helper: ResolvedExecutable | None = None
    typescript_execution_versions: JsonObject | None = None
    typescript_execution_profile: Literal["isolated", "workspace-write", "networked"] = "isolated"
    config_snapshot: ConfigSnapshot | None = None
    cursor_store: ResultCursorStore = field(default_factory=ResultCursorStore)
    oxc_worker: JsonLineWorker | None = field(default=None, init=False, repr=False)
    analysis_worker: JsonLineWorker | None = field(default=None, init=False, repr=False)
    typescript_project_worker: JsonLineWorker | None = field(default=None, init=False, repr=False)

    def close(self) -> None:
        self.cursor_store.clear()
        if self.oxc_worker is not None:
            self.oxc_worker.close()
            self.oxc_worker = None
        if self.analysis_worker is not None:
            self.analysis_worker.close()
            self.analysis_worker = None
        if self.typescript_project_worker is not None:
            self.typescript_project_worker.close()
            self.typescript_project_worker = None
        if self.config_snapshot is not None:
            self.config_snapshot.close()


READ_ONLY_ANNOTATIONS: Final = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
MUTATING_ANNOTATIONS: Final = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=True,
    open_world_hint=False,
)
EXECUTION_ANNOTATIONS: Final = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=True,
)


def _server_version() -> str:
    try:
        return version("ast-soleaux")
    except PackageNotFoundError:
        return FALLBACK_SERVER_VERSION


def _resolve_existing_path(raw_path: str, *, base: Path, kind: Literal["file", "directory"]) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{kind.capitalize()} does not exist: {raw_path}") from error
    if kind == "directory" and not resolved.is_dir():
        raise ValueError(f"Expected a directory: {raw_path}")
    if kind == "file" and not resolved.is_file():
        raise ValueError(f"Expected a file: {raw_path}")
    return resolved


def resolve_allowed_roots(raw_roots: Sequence[str], *, working_directory: Path) -> tuple[Path, ...]:
    roots = raw_roots or [str(working_directory)]
    resolved_roots: list[Path] = []
    for raw_root in roots:
        root = _resolve_existing_path(raw_root, base=working_directory, kind="directory")
        if root not in resolved_roots:
            resolved_roots.append(root)
    return tuple(resolved_roots)


def _require_allowed(path: Path, allowed_roots: Sequence[Path], *, label: str) -> None:
    if not any(is_within(path, root) for root in allowed_roots):
        allowed = ", ".join(str(root) for root in allowed_roots)
        raise ValueError(f"{label} resolves outside the allowed roots ({allowed}): {path}")


def _read_json_file(path: Path) -> JsonObject | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return JSON_OBJECT_ADAPTER.validate_python(value, strict=True)
    except OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError:
        return None


def _npm_package_bin(package_directory: Path) -> Path | None:
    manifest_path = package_directory / "package.json"
    manifest = _read_json_file(manifest_path)
    if manifest is None or manifest.get("name") != "@ast-grep/cli":
        return None

    bin_value = manifest.get("bin")
    if isinstance(bin_value, str):
        relative_bin = bin_value
    elif isinstance(bin_value, dict):
        mapped_bin = bin_value.get("ast-grep")
        if not isinstance(mapped_bin, str):
            return None
        relative_bin = mapped_bin
    else:
        return None

    try:
        target = (package_directory / relative_bin).resolve(strict=True)
    except FileNotFoundError:
        return None
    if not target.is_file() or not is_within(target, package_directory.resolve()):
        return None
    return target


def _native_ast_grep_package_name() -> str | None:
    system = platform_module.system().lower()
    machine = platform_module.machine().lower()
    if system == "darwin":
        architecture = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x64", "amd64": "x64"}.get(machine)
        return f"@ast-grep/cli-darwin-{architecture}" if architecture is not None else None
    if system == "windows":
        architecture = {
            "arm64": "arm64",
            "aarch64": "arm64",
            "x86_64": "x64",
            "amd64": "x64",
            "i386": "ia32",
            "i686": "ia32",
            "x86": "ia32",
        }.get(machine)
        return f"@ast-grep/cli-win32-{architecture}-msvc" if architecture is not None else None
    if system == "linux":
        libc_name = platform_module.libc_ver()[0].lower()
        if libc_name and libc_name not in {"glibc", "gnu libc"}:
            return None
        architecture = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x64", "amd64": "x64"}.get(machine)
        return f"@ast-grep/cli-linux-{architecture}-gnu" if architecture is not None else None
    return None


def _native_npm_ast_grep_bin(package_directory: Path) -> Path | None:
    manifest = _read_json_file(package_directory / "package.json")
    package_name = _native_ast_grep_package_name()
    if manifest is None or package_name is None:
        return None
    optional_dependencies = manifest.get("optionalDependencies")
    package_version = manifest.get("version")
    if not isinstance(optional_dependencies, Mapping) or not isinstance(package_version, str):
        return None
    if optional_dependencies.get(package_name) != package_version:
        return None

    scope, separator, leaf = package_name.partition("/")
    if scope != "@ast-grep" or separator != "/" or not leaf:
        return None
    native_package = package_directory.parent / leaf
    native_manifest = _read_json_file(native_package / "package.json")
    if native_manifest is None or native_manifest.get("name") != package_name or native_manifest.get("version") != package_version:
        return None
    executable_name = "ast-grep.exe" if os.name == "nt" else "ast-grep"
    try:
        executable = (native_package / executable_name).resolve(strict=True)
        resolved_package = native_package.resolve(strict=True)
    except FileNotFoundError:
        return None
    if not executable.is_file() or not is_within(executable, resolved_package):
        return None
    if os.name != "nt" and not os.access(executable, os.X_OK):
        return None
    return executable


def _find_npm_ast_grep_bin(shim_path: Path, resolved_path: Path) -> Path | None:
    package_candidates: list[Path] = []
    if shim_path.parent.name == ".bin":
        package_candidates.append(shim_path.parent.parent / "@ast-grep" / "cli")
    package_candidates.append(shim_path.parent / "node_modules" / "@ast-grep" / "cli")

    for parent in resolved_path.parents:
        if len(package_candidates) >= 12:
            break
        package_candidates.append(parent)

    seen: set[Path] = set()
    for candidate in package_candidates:
        normalized = candidate.absolute()
        if normalized in seen:
            continue
        seen.add(normalized)
        target = _npm_package_bin(candidate)
        if target is not None:
            return target
    return None


def _requires_node(executable: Path) -> bool:
    if executable.suffix.lower() in {".cjs", ".js", ".mjs"}:
        return True
    try:
        with executable.open("rb") as executable_file:
            first_line = executable_file.readline(256)
    except OSError:
        return False
    return first_line.startswith(b"#!") and b"node" in first_line.lower()


def resolve_ast_grep_executable(raw_executable: str, *, working_directory: Path) -> ResolvedExecutable:
    raw_path = Path(raw_executable).expanduser()
    if sys.platform == "win32":
        has_path_separator = os.sep in raw_executable or os.altsep in raw_executable
    else:
        has_path_separator = os.sep in raw_executable
    if raw_path.is_absolute() or has_path_separator:
        shim_path = raw_path if raw_path.is_absolute() else working_directory / raw_path
        if not shim_path.exists():
            raise ValueError(f"ast-grep executable does not exist: {raw_executable}")
    else:
        discovered = shutil.which(raw_executable)
        if discovered is None:
            raise ValueError(f"ast-grep executable was not found: {raw_executable}")
        shim_path = Path(discovered)

    try:
        resolved_path = shim_path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"ast-grep executable does not exist: {raw_executable}") from error
    if not resolved_path.is_file():
        raise ValueError(f"ast-grep executable is not a file: {raw_executable}")

    npm_bin = _find_npm_ast_grep_bin(shim_path.absolute(), resolved_path)
    if npm_bin is not None:
        if _requires_node(npm_bin):
            native_bin = _native_npm_ast_grep_bin(npm_bin.parent)
            if native_bin is not None:
                return ResolvedExecutable(path=native_bin, command_prefix=(str(native_bin),))
            node = shutil.which("node")
            if node is None:
                raise ValueError("The @ast-grep/cli launcher requires Node.js, but node was not found")
            node_path = Path(node).resolve(strict=True)
            return ResolvedExecutable(path=npm_bin, command_prefix=(str(node_path), str(npm_bin)))
        if os.name != "nt" and not os.access(npm_bin, os.X_OK):
            raise ValueError(f"ast-grep executable is not executable: {npm_bin}")
        return ResolvedExecutable(path=npm_bin, command_prefix=(str(npm_bin),))

    if resolved_path.suffix.lower() in {".bat", ".cmd"}:
        raise ValueError(
            "Batch-file ast-grep launchers are not executed through a shell; install @ast-grep/cli or pass its resolved executable"
        )
    if os.name != "nt" and not os.access(resolved_path, os.X_OK):
        raise ValueError(f"ast-grep executable is not executable: {resolved_path}")
    return ResolvedExecutable(path=resolved_path, command_prefix=(str(resolved_path),))


def resolve_oxc_helper_executable(raw_executable: str, *, working_directory: Path) -> ResolvedExecutable:
    raw_path = Path(raw_executable).expanduser()
    if sys.platform == "win32":
        has_path_separator = os.sep in raw_executable or os.altsep in raw_executable
    else:
        has_path_separator = os.sep in raw_executable
    if raw_path.is_absolute() or has_path_separator:
        candidate = raw_path if raw_path.is_absolute() else working_directory / raw_path
        if not candidate.exists():
            raise ValueError(f"Oxc helper executable does not exist: {raw_executable}")
    else:
        discovered = shutil.which(raw_executable)
        if discovered is None:
            raise ValueError(f"Oxc helper executable was not found: {raw_executable}")
        candidate = Path(discovered)
    try:
        resolved_path = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"Oxc helper executable does not exist: {raw_executable}") from error
    if not resolved_path.is_file():
        raise ValueError(f"Oxc helper executable is not a file: {raw_executable}")
    if resolved_path.suffix.lower() in {".bat", ".cmd"}:
        raise ValueError("Batch-file Oxc helper launchers are not executed through a shell; pass the resolved JavaScript file")
    if _requires_node(resolved_path):
        node = shutil.which("node")
        if node is None:
            raise ValueError("The Oxc helper requires Node.js, but node was not found")
        node_path = Path(node).resolve(strict=True)
        return ResolvedExecutable(path=resolved_path, command_prefix=(str(node_path), str(resolved_path)))
    if os.name != "nt" and not os.access(resolved_path, os.X_OK):
        raise ValueError(f"Oxc helper executable is not executable: {resolved_path}")
    return ResolvedExecutable(path=resolved_path, command_prefix=(str(resolved_path),))


def _bounded_error_text(value: str | None, *, limit: int = 4000) -> str:
    text = (value or "").strip()
    if not text:
        return "(no error output)"
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _has_meaningful_diagnostic(value: str | None) -> bool:
    if not value:
        return False
    without_terminal_controls = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", value)
    return any(character.isprintable() and not character.isspace() for character in without_terminal_controls)


def _utf16_code_units(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def _is_invalid_process_string(value: object) -> bool:
    return not isinstance(value, str) or "\0" in value


def _detected_posix_arg_max() -> int:
    if os.name != "posix":
        return 0
    try:
        member = "sysconf"
        sysconf = cast(SysconfCallable, getattr(os, member))
        return int(sysconf("SC_ARG_MAX"))
    except (AttributeError, OSError, ValueError) as error:
        raise ValueError("Could not determine the POSIX ARG_MAX process-launch budget") from error


def validate_process_budget(
    command: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
    platform_name: str | None = None,
    arg_max: int | None = None,
) -> None:
    if not command:
        raise ValueError("Command must contain an executable")
    if any(_is_invalid_process_string(argument) for argument in command):
        raise ValueError("Command arguments must be strings without NUL characters")

    executable_suffix = Path(command[0]).suffix.lower()
    if executable_suffix in {".bat", ".cmd"}:
        raise ValueError(
            "Batch-file commands are not launched directly; pass the resolved native executable or the JavaScript entry point through node"
        )

    launch_environment = os.environ if environment is None else environment
    if any(_is_invalid_process_string(key) or _is_invalid_process_string(value) or "=" in key for key, value in launch_environment.items()):
        raise ValueError("Subprocess environment names and values must be valid NUL-free strings")

    effective_platform = os.name if platform_name is None else platform_name
    if effective_platform == "nt":
        quoted_command = subprocess.list2cmdline(list(command))
        command_characters = _utf16_code_units(quoted_command) + 1
        if command_characters > WINDOWS_CREATE_PROCESS_LIMIT:
            raise ValueError(
                "Command line is too large for Windows CreateProcessW: "
                f"{command_characters} UTF-16 characters including NUL exceeds "
                f"the {WINDOWS_CREATE_PROCESS_LIMIT}-character limit; shorten paths, globs, or inline rules"
            )
        environment_characters = 1 + sum(_utf16_code_units(f"{key}={value}") + 1 for key, value in launch_environment.items())
        if environment_characters > WINDOWS_CREATE_PROCESS_LIMIT:
            raise ValueError(
                "Environment block is too large for Windows process creation: "
                f"{environment_characters} UTF-16 characters exceeds "
                f"the {WINDOWS_CREATE_PROCESS_LIMIT}-character limit; remove unnecessary environment variables"
            )
        return

    if effective_platform != "posix":
        raise ValueError(f"Unsupported subprocess platform: {effective_platform}")
    detected_arg_max = arg_max
    if detected_arg_max is None:
        detected_arg_max = _detected_posix_arg_max()
    if detected_arg_max <= POSIX_ARG_HEADROOM_BYTES:
        raise ValueError(f"Invalid POSIX ARG_MAX process-launch budget: {detected_arg_max}")

    argv_bytes = sum(len(os.fsencode(argument)) + 1 for argument in command)
    environment_bytes = sum(len(os.fsencode(key)) + len(os.fsencode(value)) + 2 for key, value in launch_environment.items())
    pointer_bytes = (len(command) + len(launch_environment) + 2) * struct.calcsize("P")
    process_bytes = argv_bytes + environment_bytes + pointer_bytes
    usable_bytes = detected_arg_max - POSIX_ARG_HEADROOM_BYTES
    if process_bytes > usable_bytes:
        raise ValueError(
            "Command and environment are too large for POSIX ARG_MAX: "
            f"estimated {process_bytes} bytes exceeds the {usable_bytes}-byte budget after "
            f"{POSIX_ARG_HEADROOM_BYTES} bytes of headroom; shorten paths, globs, inline rules, or the environment"
        )


def _is_benign_scan_diagnostic(line: str) -> bool:
    if line.startswith("Help:"):
        return True
    return line.startswith("Error: ") and line.endswith(" error(s) found in code.")


def _residual_stderr(stderr: str) -> str:
    lines = (raw.strip() for raw in stderr.splitlines())
    return "\n".join(line for line in lines if line and not _is_benign_scan_diagnostic(line))


@dataclass(frozen=True)
class StreamedTextProcess:
    completed: CompletedTextProcess
    stdout_truncated: bool
    stderr_truncated: bool


def _encode_bounded_input(input_text: str | None) -> bytes | None:
    if input_text is None:
        return None
    try:
        payload = input_text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("Subprocess input must be valid UTF-8") from error
    if len(payload) > MAX_SNIPPET_INPUT_BYTES:
        raise ValueError(f"Subprocess input exceeds the {MAX_SNIPPET_INPUT_BYTES // (1024 * 1024)} MiB limit")
    return payload


def _write_stdin(
    pipe: BinaryIO,
    payload: bytes,
    failures: list[BaseException],
    stop_writing: threading.Event,
) -> None:
    try:
        view = memoryview(payload)
        while view and not stop_writing.is_set():
            written = pipe.write(view[:PROCESS_READ_CHUNK_BYTES])
            if written <= 0:
                raise OSError("subprocess stdin made no write progress")
            view = view[written:]
        pipe.flush()
    except BrokenPipeError, ConnectionResetError:
        pass
    except (OSError, ValueError) as error:
        if not stop_writing.is_set():
            failures.append(error)
    finally:
        try:
            pipe.close()
        except OSError:
            pass


def _drain_bounded_pipe(
    pipe: BinaryIO,
    captured: bytearray,
    byte_limit: int,
    overflowed: list[bool],
    failures: list[BaseException],
    stop_reading: threading.Event,
    overflow: threading.Event | None = None,
) -> None:
    try:
        while not stop_reading.is_set():
            chunk = pipe.read(PROCESS_READ_CHUNK_BYTES)
            if not chunk:
                return
            available = byte_limit - len(captured)
            if available > 0:
                captured.extend(chunk[:available])
            if len(chunk) > max(available, 0):
                overflowed[0] = True
                if overflow is not None:
                    overflow.set()
    except (OSError, ValueError) as error:
        if not stop_reading.is_set():
            failures.append(error)


def _decode_utf8_prefix(payload: bytes) -> str:
    """Decode bytes, backing up to the last valid UTF-8 boundary on truncation."""
    while payload:
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as error:
            if not error.start:
                return ""
            payload = payload[: error.start]
    return ""


def _decode_utf8_output(payload: bytes, *, truncated: bool, label: str) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        if truncated and error.reason == "unexpected end of data" and error.end == len(payload):
            return _decode_utf8_prefix(payload[: error.start])
        raise RuntimeError(f"Command emitted invalid UTF-8 on {label}") from error


def run_text_process(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    input_text: str | None = None,
    working_directory: Path | None = None,
    stdout_limit: int = MAX_STRUCTURED_OUTPUT_BYTES,
    stderr_limit: int = MAX_SUBPROCESS_DIAGNOSTIC_BYTES,
    truncate_stdout: bool = False,
    truncate_stderr: bool = True,
    popen_factory: PopenFactory = subprocess.Popen,
) -> StreamedTextProcess:
    if stdout_limit < 1 or stderr_limit < 1:
        raise ValueError("Subprocess output limits must be positive")
    input_bytes = _encode_bounded_input(input_text)
    launch_environment = dict(os.environ)
    validate_process_budget(command, environment=launch_environment)
    try:
        process = popen_factory(
            list(command),
            stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(working_directory) if working_directory is not None else None,
            env=launch_environment,
            shell=False,
            bufsize=0,
            **popen_process_group_options(),
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"Command executable was not found: {command[0]}") from error
    except OSError as error:
        raise RuntimeError(f"Command could not be executed: {error}") from error

    process_group = process_group_id(process)
    if process.stdout is None or process.stderr is None:
        terminate_and_reap(process, process_group)
        raise RuntimeError("Command pipes were not created")

    stop_io = threading.Event()
    stdout = bytearray()
    stderr = bytearray()
    stdout_overflow = [False]
    stderr_overflow = [False]
    overflow = threading.Event()
    io_failures: list[BaseException] = []
    threads = [
        threading.Thread(
            target=_drain_bounded_pipe,
            args=(process.stdout, stdout, stdout_limit, stdout_overflow, io_failures, stop_io, overflow),
            name="ast-grep-stdout",
            daemon=True,
        ),
        threading.Thread(
            target=_drain_bounded_pipe,
            args=(process.stderr, stderr, stderr_limit, stderr_overflow, io_failures, stop_io, overflow),
            name="ast-grep-stderr",
            daemon=True,
        ),
    ]
    if input_bytes is not None:
        if process.stdin is None:
            terminate_and_reap(process, process_group)
            raise RuntimeError("Command stdin pipe was not created")
        threads.append(
            threading.Thread(
                target=_write_stdin,
                args=(process.stdin, input_bytes, io_failures, stop_io),
                name="ast-grep-stdin",
                daemon=True,
            )
        )
    for thread in threads:
        thread.start()

    timeout_error: subprocess.TimeoutExpired | None = None
    overflow_error: str | None = None
    deadline = time.monotonic() + timeout_seconds
    while process.poll() is None:
        if overflow.is_set():
            if stdout_overflow[0] and not truncate_stdout:
                overflow_error = f"Command structured output exceeds the {stdout_limit}-byte limit"
            elif stderr_overflow[0] and not truncate_stderr:
                overflow_error = f"Command diagnostic output exceeds the {stderr_limit}-byte limit"
            if overflow_error is not None:
                terminate_and_reap(process, process_group)
                break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timeout_error = subprocess.TimeoutExpired(list(command), timeout_seconds)
            terminate_and_reap(process, process_group)
            break
        try:
            process.wait(timeout=min(remaining, 0.05))
        except subprocess.TimeoutExpired:
            continue
    terminate_and_reap(process, process_group)
    for thread in threads:
        thread.join(timeout=0.05)
    if any(thread.is_alive() for thread in threads):
        terminate_and_reap(process, process_group)
    stop_io.set()
    for pipe in (process.stdout, process.stderr):
        try:
            pipe.close()
        except OSError:
            pass
    if process.stdin is not None:
        try:
            process.stdin.close()
        except OSError:
            pass
    for thread in threads:
        if thread.is_alive():
            thread.join(timeout=PROCESS_TERMINATION_GRACE_SECONDS)

    if any(thread.is_alive() for thread in threads):
        raise RuntimeError("Command pipe workers did not stop after process exit")
    if timeout_error is not None:
        raise RuntimeError(f"Command timed out after {timeout_seconds:g} seconds") from timeout_error
    if io_failures:
        raise RuntimeError(f"Could not transfer command data: {io_failures[0]}") from io_failures[0]
    if overflow_error is not None:
        raise RuntimeError(overflow_error)
    if stdout_overflow[0] and not truncate_stdout:
        raise RuntimeError(f"Command structured output exceeds the {stdout_limit}-byte limit")
    if stderr_overflow[0] and not truncate_stderr:
        raise RuntimeError(f"Command diagnostic output exceeds the {stderr_limit}-byte limit")

    stdout_text = _decode_utf8_output(bytes(stdout), truncated=stdout_overflow[0], label="stdout")
    stderr_text = _decode_utf8_output(bytes(stderr), truncated=stderr_overflow[0], label="stderr")
    completed = subprocess.CompletedProcess(list(command), process.returncode, stdout_text, stderr_text)
    return StreamedTextProcess(completed, stdout_overflow[0], stderr_overflow[0])


def run_process(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    input_text: str | None = None,
    working_directory: Path | None = None,
    allowed_exit_codes: frozenset[int] = frozenset({0}),
    runner: ProcessRunner = subprocess.run,
    stdout_limit: int = MAX_STRUCTURED_OUTPUT_BYTES,
    stderr_limit: int = MAX_SUBPROCESS_DIAGNOSTIC_BYTES,
    truncate_stdout: bool = False,
    truncate_stderr: bool = True,
) -> CompletedTextProcess:
    launch_environment = dict(os.environ)
    validate_process_budget(command, environment=launch_environment)
    try:
        if runner is subprocess.run:
            streamed = run_text_process(
                command,
                timeout_seconds=timeout_seconds,
                input_text=input_text,
                working_directory=working_directory,
                stdout_limit=stdout_limit,
                stderr_limit=stderr_limit,
                truncate_stdout=truncate_stdout,
                truncate_stderr=truncate_stderr,
            )
            result = streamed.completed
        else:
            result = runner(
                list(command),
                capture_output=True,
                input=input_text,
                text=True,
                timeout=timeout_seconds,
                cwd=str(working_directory) if working_directory is not None else None,
                check=False,
                shell=False,
                env=launch_environment,
            )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Command timed out after {timeout_seconds:g} seconds") from error
    except FileNotFoundError as error:
        raise RuntimeError(f"Command executable was not found: {command[0]}") from error
    except OSError as error:
        raise RuntimeError(f"Command could not be executed: {error}") from error

    if result.returncode not in allowed_exit_codes:
        detail = _bounded_error_text(result.stderr)
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {detail}")
    return result


def _read_ast_grep_version(
    executable: ResolvedExecutable,
    *,
    timeout_seconds: float,
    runner: ProcessRunner = subprocess.run,
) -> str:
    result = run_process(
        [*executable.command_prefix, "--version"],
        timeout_seconds=timeout_seconds,
        runner=runner,
    )
    expected = f"ast-grep {SUPPORTED_AST_GREP_VERSION}"
    if result.stdout.strip() != expected or result.stderr:
        diagnostic = result.stderr or result.stdout
        raise ValueError(f"Configured executable must report exactly {expected!r}: {_bounded_error_text(diagnostic)}")
    return SUPPORTED_AST_GREP_VERSION


def read_oxc_helper_versions(
    executable: ResolvedExecutable,
    *,
    timeout_seconds: float,
    runner: ProcessRunner = subprocess.run,
) -> OxcVersions:
    result = run_process(
        [*executable.command_prefix, "--version-json"],
        timeout_seconds=timeout_seconds,
        runner=runner,
    )
    if _has_meaningful_diagnostic(result.stderr):
        detail = _bounded_error_text(result.stderr)
        raise ValueError(f"Configured Oxc helper emitted diagnostics during version discovery: {detail!r}")
    try:
        payload = json.loads(result.stdout, parse_constant=_reject_json_constant)
        versions = OXC_VERSIONS_ADAPTER.validate_python(payload, strict=True)
    except (json.JSONDecodeError, ValidationError, ValueError) as error:
        raise ValueError(f"Configured Oxc helper returned an invalid version payload: {_bounded_error_text(result.stdout)}") from error
    expected: OxcVersions = {
        "helper": SUPPORTED_OXC_HELPER_VERSION,
        "parser": SUPPORTED_OXC_PARSER_VERSION,
        "resolver": SUPPORTED_OXC_RESOLVER_VERSION,
    }
    if versions != expected:
        raise ValueError(f"Configured Oxc helper must report exactly {expected!r}; got {versions!r}")
    return versions


def read_analysis_helper_versions(
    executable: ResolvedExecutable,
    *,
    timeout_seconds: float,
    runner: ProcessRunner = subprocess.run,
) -> JsonObject:
    result = run_process(
        [*executable.command_prefix, "--version-json"],
        timeout_seconds=timeout_seconds,
        runner=runner,
    )
    if _has_meaningful_diagnostic(result.stderr):
        raise ValueError(f"Configured analysis helper emitted diagnostics during version discovery: {_bounded_error_text(result.stderr)}")
    try:
        versions = JSON_OBJECT_ADAPTER.validate_python(json.loads(result.stdout, parse_constant=_reject_json_constant), strict=True)
    except (json.JSONDecodeError, ValidationError, ValueError) as error:
        raise ValueError(f"Configured analysis helper returned an invalid version payload: {_bounded_error_text(result.stdout)}") from error
    expected: JsonObject = {
        "worker": SUPPORTED_ANALYSIS_WORKER_VERSION,
        "oxc": SUPPORTED_OXC_PARSER_VERSION,
        "resolver": SUPPORTED_OXC_RESOLVER_VERSION,
    }
    if versions != expected:
        raise ValueError(f"Configured analysis helper must report exactly {expected!r}; got {versions!r}")
    return versions


def read_typescript_helper_versions(
    executable: ResolvedExecutable,
    *,
    timeout_seconds: float,
    runner: ProcessRunner = subprocess.run,
) -> JsonObject:
    result = run_process([*executable.command_prefix, "--version-json"], timeout_seconds=timeout_seconds, runner=runner)
    if _has_meaningful_diagnostic(result.stderr):
        raise ValueError(f"Configured TypeScript helper emitted diagnostics during version discovery: {_bounded_error_text(result.stderr)}")
    try:
        versions = JSON_OBJECT_ADAPTER.validate_python(json.loads(result.stdout, parse_constant=_reject_json_constant), strict=True)
    except (json.JSONDecodeError, ValidationError, ValueError) as error:
        raise ValueError(f"Configured TypeScript helper returned invalid versions: {_bounded_error_text(result.stdout)}") from error
    expected: JsonObject = {"worker": SUPPORTED_TYPESCRIPT_WORKER_VERSION, "typescript": SUPPORTED_TYPESCRIPT_VERSION}
    if versions != expected:
        raise ValueError(f"Configured TypeScript helper must report {expected!r}; got {versions!r}")
    return versions


def read_postgres_helper_versions(
    executable: ResolvedExecutable,
    *,
    timeout_seconds: float,
    runner: ProcessRunner = subprocess.run,
) -> JsonObject:
    result = run_process([*executable.command_prefix, "--version-json"], timeout_seconds=timeout_seconds, runner=runner)
    if _has_meaningful_diagnostic(result.stderr):
        raise ValueError(f"Configured PostgreSQL helper emitted diagnostics during version discovery: {_bounded_error_text(result.stderr)}")
    try:
        versions = JSON_OBJECT_ADAPTER.validate_python(json.loads(result.stdout, parse_constant=_reject_json_constant), strict=True)
    except (json.JSONDecodeError, ValidationError, ValueError) as error:
        raise ValueError(f"Configured PostgreSQL helper returned invalid versions: {_bounded_error_text(result.stdout)}") from error
    expected: JsonObject = {
        "worker": SUPPORTED_POSTGRES_WORKER_VERSION,
        "parser": SUPPORTED_POSTGRES_PARSER_VERSION,
        "deparser": SUPPORTED_POSTGRES_DEPARSER_VERSION,
        "postgres_major": SUPPORTED_POSTGRES_MAJOR,
    }
    if versions != expected:
        raise ValueError(f"Configured PostgreSQL helper must report {expected!r}; got {versions!r}")
    return versions


def read_typescript_execution_versions(
    executable: ResolvedExecutable,
    *,
    timeout_seconds: float,
    runner: ProcessRunner = subprocess.run,
) -> JsonObject:
    result = run_process([*executable.command_prefix, "--version-json"], timeout_seconds=timeout_seconds, runner=runner)
    if _has_meaningful_diagnostic(result.stderr):
        raise ValueError(f"Configured TypeScript execution helper emitted diagnostics: {_bounded_error_text(result.stderr)}")
    try:
        versions = JSON_OBJECT_ADAPTER.validate_python(json.loads(result.stdout, parse_constant=_reject_json_constant), strict=True)
    except (json.JSONDecodeError, ValidationError, ValueError) as error:
        raise ValueError(
            f"Configured TypeScript execution helper returned invalid versions: {_bounded_error_text(result.stdout)}"
        ) from error
    expected: JsonObject = {
        "worker": SUPPORTED_TYPESCRIPT_EXECUTION_WORKER_VERSION,
        "runtime": SUPPORTED_TYPESCRIPT_EXECUTION_RUNTIME,
        "sandbox": "docker",
    }
    if versions != expected:
        raise ValueError(f"Configured TypeScript execution helper must report {expected!r}; got {versions!r}")
    return versions


def _validate_config_snapshot(
    executable: ResolvedExecutable,
    snapshot: ConfigSnapshot,
    *,
    timeout_seconds: float,
    runner: ProcessRunner,
) -> None:
    with TemporaryDirectory(prefix="validation-", dir=snapshot.runtime_root) as validation:
        validation_directory = Path(validation)
        scan = run_process(
            [
                *executable.command_prefix,
                "scan",
                "--config",
                str(snapshot.project_config_path),
                "--json=stream",
                "--threads",
                "1",
                "--max-results",
                "1",
                "--",
                str(validation_directory),
            ],
            timeout_seconds=timeout_seconds,
            working_directory=validation_directory,
            allowed_exit_codes=frozenset({0, 1}),
            runner=runner,
            stdout_limit=MAX_STRUCTURED_OUTPUT_BYTES,
            stderr_limit=MAX_SUBPROCESS_DIAGNOSTIC_BYTES,
            truncate_stdout=True,
            truncate_stderr=True,
        )
        if scan.returncode not in {0, 1}:
            raise RuntimeError("Configured rules failed startup validation")
        if snapshot.capabilities["configured_tests"]:
            run_process(
                [
                    *executable.command_prefix,
                    "test",
                    "--config",
                    str(snapshot.test_config_path),
                    "--color",
                    "never",
                ],
                timeout_seconds=timeout_seconds,
                working_directory=snapshot.bundle_root,
                allowed_exit_codes=frozenset({0, 4}),
                runner=runner,
                stdout_limit=MAX_TEST_REPORT_BYTES,
                stderr_limit=MAX_SUBPROCESS_DIAGNOSTIC_BYTES,
                truncate_stdout=True,
                truncate_stderr=True,
            )


def build_runtime(
    *,
    working_directory: Path,
    ast_grep_executable: str = "ast-grep",
    oxc_helper_executable: str | None = None,
    analysis_helper_executable: str | None = None,
    typescript_project_helper_executable: str | None = None,
    postgres_helper_executable: str | None = None,
    typescript_execution_helper_executable: str | None = None,
    typescript_execution_profile: Literal["isolated", "workspace-write", "networked"] = "isolated",
    config_path: str | None = None,
    allowed_roots: Sequence[str] = (),
    command_timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    default_max_results: int = DEFAULT_MAX_RESULTS,
    max_results_cap: int = HARD_MAX_RESULTS,
    forbid_regex_rules: bool = False,
    trusted_native_libraries: Sequence[tuple[str, str]] = (),
    runner: ProcessRunner = subprocess.run,
) -> RuntimeServices:
    resolved_working_directory = working_directory.resolve(strict=True)
    if not resolved_working_directory.is_dir():
        raise ValueError(f"Working directory is not a directory: {working_directory}")
    if not math.isfinite(command_timeout_seconds) or command_timeout_seconds <= 0:
        raise ValueError("Command timeout must be finite and greater than zero")
    if not 1 <= max_results_cap <= HARD_MAX_RESULTS:
        raise ValueError(f"Result cap must be between 1 and {HARD_MAX_RESULTS}")
    if not 1 <= default_max_results <= max_results_cap:
        raise ValueError("Default result limit must be positive and no greater than the configured cap")

    resolved_roots = resolve_allowed_roots(allowed_roots, working_directory=resolved_working_directory)
    executable = resolve_ast_grep_executable(ast_grep_executable, working_directory=resolved_working_directory)
    ast_grep_version = _read_ast_grep_version(
        executable,
        timeout_seconds=command_timeout_seconds,
        runner=runner,
    )
    oxc_helper: ResolvedExecutable | None = None
    oxc_versions: OxcVersions | None = None
    if oxc_helper_executable is not None:
        oxc_helper = resolve_oxc_helper_executable(
            oxc_helper_executable,
            working_directory=resolved_working_directory,
        )
        oxc_versions = read_oxc_helper_versions(
            oxc_helper,
            timeout_seconds=command_timeout_seconds,
            runner=runner,
        )
    analysis_helper: ResolvedExecutable | None = None
    analysis_versions: JsonObject | None = None
    if analysis_helper_executable is not None:
        analysis_helper = resolve_oxc_helper_executable(
            analysis_helper_executable,
            working_directory=resolved_working_directory,
        )
        analysis_versions = read_analysis_helper_versions(
            analysis_helper,
            timeout_seconds=command_timeout_seconds,
            runner=runner,
        )
    typescript_project_helper: ResolvedExecutable | None = None
    typescript_versions: JsonObject | None = None
    if typescript_project_helper_executable is not None:
        typescript_project_helper = resolve_oxc_helper_executable(
            typescript_project_helper_executable,
            working_directory=resolved_working_directory,
        )
        typescript_versions = read_typescript_helper_versions(
            typescript_project_helper,
            timeout_seconds=command_timeout_seconds,
            runner=runner,
        )
    postgres_helper: ResolvedExecutable | None = None
    postgres_versions: JsonObject | None = None
    if postgres_helper_executable is not None:
        postgres_helper = resolve_oxc_helper_executable(
            postgres_helper_executable,
            working_directory=resolved_working_directory,
        )
        postgres_versions = read_postgres_helper_versions(
            postgres_helper,
            timeout_seconds=command_timeout_seconds,
            runner=runner,
        )
    if typescript_execution_profile not in {"isolated", "workspace-write", "networked"}:
        raise ValueError(f"Unsupported TypeScript execution profile: {typescript_execution_profile}")
    typescript_execution_helper = (
        resolve_oxc_helper_executable(
            typescript_execution_helper_executable,
            working_directory=resolved_working_directory,
        )
        if typescript_execution_helper_executable is not None
        else None
    )
    typescript_execution_versions = (
        read_typescript_execution_versions(
            typescript_execution_helper,
            timeout_seconds=command_timeout_seconds,
            runner=runner,
        )
        if typescript_execution_helper is not None
        else None
    )
    snapshot = create_config_snapshot(
        config_path=config_path,
        working_directory=resolved_working_directory,
        allowed_roots=resolved_roots,
        trusted_native_libraries=trusted_native_libraries,
    )
    try:
        _validate_config_snapshot(
            executable,
            snapshot,
            timeout_seconds=command_timeout_seconds,
            runner=runner,
        )
    except BaseException:
        snapshot.close_best_effort()
        raise
    atexit.register(snapshot.close_best_effort)

    return RuntimeServices(
        working_directory=resolved_working_directory,
        executable=executable,
        ast_grep_version=ast_grep_version,
        config_path=snapshot.source_path,
        allowed_roots=resolved_roots,
        command_timeout_seconds=command_timeout_seconds,
        default_max_results=default_max_results,
        max_results_cap=max_results_cap,
        forbid_regex_rules=forbid_regex_rules,
        oxc_helper=oxc_helper,
        oxc_versions=oxc_versions,
        analysis_helper=analysis_helper,
        analysis_versions=analysis_versions,
        typescript_project_helper=typescript_project_helper,
        typescript_versions=typescript_versions,
        postgres_helper=postgres_helper,
        postgres_versions=postgres_versions,
        typescript_execution_helper=typescript_execution_helper,
        typescript_execution_versions=typescript_execution_versions,
        typescript_execution_profile=typescript_execution_profile,
        config_snapshot=snapshot,
    )


def _contains_mapping_key(value: object, forbidden_key: str) -> bool:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return any(key == forbidden_key or _contains_mapping_key(item, forbidden_key) for key, item in mapping.items())
    if isinstance(value, list):
        items = cast(list[object], value)
        return any(_contains_mapping_key(item, forbidden_key) for item in items)
    return False


def validate_rule_yaml(rule_yaml: str, *, forbid_regex_rules: bool, caller_supplied: bool = True) -> None:
    try:
        encoded = rule_yaml.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("ast-grep rule YAML must be valid UTF-8") from error
    if caller_supplied and len(encoded) > MAX_INLINE_RULE_BYTES:
        raise ValueError(f"ast-grep rule YAML exceeds the {MAX_INLINE_RULE_BYTES // 1024} KiB inline limit")
    documents = load_strict_yaml_documents(rule_yaml, label="ast-grep rule YAML")
    if not documents or all(document is None for document in documents):
        raise ValueError("ast-grep rule YAML must contain at least one rule")

    for document in documents:
        if not isinstance(document, Mapping):
            raise ValueError("Each ast-grep inline rule must be a YAML mapping")
        document_mapping = cast(Mapping[object, object], document)
        missing = [key for key in ("id", "language", "rule") if key not in document_mapping]
        if missing:
            raise ValueError(f"ast-grep rule is missing required fields: {', '.join(missing)}")
        if not isinstance(document_mapping["rule"], Mapping):
            raise ValueError("ast-grep rule field must be a mapping")
        if forbid_regex_rules and _contains_mapping_key(document_mapping, "regex"):
            raise ValueError("Regex ast-grep rules are disabled by server policy")
    validate_rule_utility_cycles(documents, label="ast-grep rule YAML")


def _runtime_mapping(value: object, *, field_name: str, record_number: int) -> JsonObject:
    try:
        return JSON_OBJECT_ADAPTER.validate_python(value, strict=True)
    except ValidationError as error:
        raise RuntimeError(f"ast-grep record {record_number} has invalid {field_name}") from error


def _runtime_list(value: object, *, field_name: str, record_number: int) -> list[JsonValue]:
    if not isinstance(value, list):
        raise RuntimeError(f"ast-grep record {record_number} has invalid {field_name}")
    validated: list[JsonValue] = []
    for item in cast(list[object], value):
        validated.append(JSON_VALUE_ADAPTER.validate_python(item, strict=True))
    return validated


def _runtime_nonnegative_int(value: object, *, field_name: str, record_number: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError(f"ast-grep record {record_number} has invalid {field_name}")
    return value


def _validate_position(value: object, *, field_name: str, record_number: int) -> tuple[int, int]:
    position = _runtime_mapping(value, field_name=field_name, record_number=record_number)
    line = _runtime_nonnegative_int(position.get("line"), field_name=f"{field_name}.line", record_number=record_number)
    column = _runtime_nonnegative_int(
        position.get("column"),
        field_name=f"{field_name}.column",
        record_number=record_number,
    )
    return line, column


def _validate_offset_range(value: object, *, field_name: str, record_number: int) -> tuple[int, int]:
    offset_range = _runtime_mapping(value, field_name=field_name, record_number=record_number)
    start = _runtime_nonnegative_int(
        offset_range.get("start"),
        field_name=f"{field_name}.start",
        record_number=record_number,
    )
    end = _runtime_nonnegative_int(
        offset_range.get("end"),
        field_name=f"{field_name}.end",
        record_number=record_number,
    )
    if end < start:
        raise RuntimeError(f"ast-grep record {record_number} has a reversed {field_name}")
    return start, end


def _validate_source_range(
    value: object,
    *,
    field_name: str,
    record_number: int,
    text: str | None = None,
) -> tuple[int, int]:
    source_range = _runtime_mapping(value, field_name=field_name, record_number=record_number)
    start_offset, end_offset = _validate_offset_range(
        source_range.get("byteOffset"),
        field_name=f"{field_name}.byteOffset",
        record_number=record_number,
    )
    start_position = _validate_position(
        source_range.get("start"),
        field_name=f"{field_name}.start",
        record_number=record_number,
    )
    end_position = _validate_position(
        source_range.get("end"),
        field_name=f"{field_name}.end",
        record_number=record_number,
    )
    if end_position < start_position:
        raise RuntimeError(f"ast-grep record {record_number} has reversed half-open positions in {field_name}")
    if text is not None:
        if end_offset - start_offset != len(text.encode("utf-8")):
            raise RuntimeError(f"ast-grep record {record_number} has inconsistent UTF-8 byte offsets in {field_name}")
        line_delta = text.count("\n")
        expected_end_line = start_position[0] + line_delta
        expected_end_column = start_position[1] + len(text) if line_delta == 0 else len(text.rsplit("\n", 1)[1])
        if end_position != (expected_end_line, expected_end_column):
            raise RuntimeError(f"ast-grep record {record_number} has inconsistent line and column geometry in {field_name}")
    return start_offset, end_offset


def _validate_match_node(
    value: object,
    *,
    field_name: str,
    record_number: int,
) -> None:
    node = _runtime_mapping(value, field_name=field_name, record_number=record_number)
    text = node.get("text")
    if not isinstance(text, str):
        raise RuntimeError(f"ast-grep record {record_number} has invalid {field_name}.text")
    _validate_source_range(
        node.get("range"),
        field_name=f"{field_name}.range",
        record_number=record_number,
        text=text,
    )


def validate_match_document(document: JsonObject, *, record_number: int, require_rule_fields: bool = True) -> None:
    text = document.get("text")
    lines = document.get("lines")
    file_name = document.get("file")
    language = document.get("language")
    if not isinstance(text, str):
        raise RuntimeError(f"ast-grep match record {record_number} has no text")
    if not isinstance(lines, str):
        raise RuntimeError(f"ast-grep match record {record_number} has no lines")
    if not isinstance(file_name, str) or not file_name:
        raise RuntimeError(f"ast-grep match record {record_number} has no file")
    if not isinstance(language, str) or not language:
        raise RuntimeError(f"ast-grep match record {record_number} has no language")
    _validate_source_range(
        document.get("range"),
        field_name="range",
        record_number=record_number,
        text=text,
    )

    char_count = _runtime_mapping(document.get("charCount"), field_name="charCount", record_number=record_number)
    leading = _runtime_nonnegative_int(
        char_count.get("leading"),
        field_name="charCount.leading",
        record_number=record_number,
    )
    trailing = _runtime_nonnegative_int(
        char_count.get("trailing"),
        field_name="charCount.trailing",
        record_number=record_number,
    )
    if leading + trailing > len(lines):
        raise RuntimeError(f"ast-grep match record {record_number} has invalid character context counts")
    matched_end = len(lines) - trailing if trailing else len(lines)
    if lines[leading:matched_end] != text:
        raise RuntimeError(f"ast-grep match record {record_number} has inconsistent character context")

    meta_variables_value = document.get("metaVariables")
    if meta_variables_value is not None:
        meta_variables = _runtime_mapping(
            meta_variables_value,
            field_name="metaVariables",
            record_number=record_number,
        )
        single = _runtime_mapping(meta_variables.get("single"), field_name="metaVariables.single", record_number=record_number)
        multi = _runtime_mapping(meta_variables.get("multi"), field_name="metaVariables.multi", record_number=record_number)
        transformed = _runtime_mapping(
            meta_variables.get("transformed"),
            field_name="metaVariables.transformed",
            record_number=record_number,
        )
        for name, node in single.items():
            _validate_match_node(
                node,
                field_name=f"metaVariables.single.{name}",
                record_number=record_number,
            )
        for name, nodes_value in multi.items():
            nodes = _runtime_list(
                nodes_value,
                field_name=f"metaVariables.multi.{name}",
                record_number=record_number,
            )
            for index, node in enumerate(nodes):
                _validate_match_node(
                    node,
                    field_name=f"metaVariables.multi.{name}[{index}]",
                    record_number=record_number,
                )
        if any(not isinstance(value, str) for value in transformed.values()):
            raise RuntimeError(f"ast-grep match record {record_number} has invalid transformed metavariables")

    has_replacement = "replacement" in document
    has_replacement_offsets = "replacementOffsets" in document
    if has_replacement != has_replacement_offsets:
        raise RuntimeError(f"ast-grep match record {record_number} has incomplete replacement data")
    if has_replacement:
        if not isinstance(document["replacement"], str):
            raise RuntimeError(f"ast-grep match record {record_number} has invalid replacement")
        _validate_offset_range(
            document["replacementOffsets"],
            field_name="replacementOffsets",
            record_number=record_number,
        )

    if not require_rule_fields:
        return
    rule_id = document.get("ruleId")
    severity = document.get("severity")
    message = document.get("message")
    if not isinstance(rule_id, str) or not rule_id:
        raise RuntimeError(f"ast-grep match record {record_number} has no ruleId")
    if severity not in {"hint", "info", "warning", "error", "off"}:
        raise RuntimeError(f"ast-grep match record {record_number} has invalid severity")
    if not isinstance(message, str):
        raise RuntimeError(f"ast-grep match record {record_number} has invalid message")
    if document.get("note") is not None and not isinstance(document.get("note"), str):
        raise RuntimeError(f"ast-grep match record {record_number} has invalid note")
    labels = _runtime_list(document.get("labels", []), field_name="labels", record_number=record_number)
    for index, label_value in enumerate(labels):
        label = _runtime_mapping(label_value, field_name=f"labels[{index}]", record_number=record_number)
        label_text = label.get("text")
        if not isinstance(label_text, str):
            raise RuntimeError(f"ast-grep match record {record_number} has invalid labels[{index}].text")
        _validate_source_range(
            label.get("range"),
            field_name=f"labels[{index}].range",
            record_number=record_number,
            text=label_text,
        )
        if label.get("style") not in {"primary", "secondary"}:
            raise RuntimeError(f"ast-grep match record {record_number} has invalid labels[{index}].style")
        if label.get("message") is not None and not isinstance(label.get("message"), str):
            raise RuntimeError(f"ast-grep match record {record_number} has invalid labels[{index}].message")
    if "metadata" in document and not isinstance(document["metadata"], dict):
        raise RuntimeError(f"ast-grep match record {record_number} has invalid metadata")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _parse_json_object_record(raw_record: bytes, *, record_number: int, noun: str) -> JsonObject:
    if len(raw_record) > MAX_NDJSON_RECORD_BYTES:
        raise RuntimeError(f"ast-grep {noun} record exceeds the {MAX_NDJSON_RECORD_BYTES // 1024} KiB limit")
    try:
        decoded = raw_record.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"ast-grep {noun} emitted invalid UTF-8 in record {record_number}") from error
    try:
        value = json.loads(decoded, parse_constant=_reject_json_constant)
        if not isinstance(value, dict):
            raise RuntimeError(f"ast-grep {noun} emitted a non-object JSON record {record_number}")
        return JSON_OBJECT_ADAPTER.validate_python(value, strict=True)
    except (json.JSONDecodeError, RecursionError, ValueError, ValidationError) as error:
        raise RuntimeError(f"ast-grep {noun} emitted invalid JSON object in record {record_number}") from error


def _parse_match_record(raw_record: bytes, record_number: int) -> tuple[JsonObject, int]:
    document = _parse_json_object_record(raw_record, record_number=record_number, noun="match")
    validate_match_document(document, record_number=record_number)
    return document, 1


def parse_stream_matches(stdout: str) -> list[JsonObject]:
    matches: list[JsonObject] = []
    for line_number, raw_line in enumerate(stdout.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = json.loads(line, parse_constant=_reject_json_constant)
            if not isinstance(value, dict):
                raise RuntimeError(f"ast-grep emitted a non-object JSON match on line {line_number}")
            matches.append(JSON_OBJECT_ADAPTER.validate_python(value, strict=True))
        except (json.JSONDecodeError, RecursionError, ValueError, ValidationError) as error:
            raise RuntimeError(f"ast-grep emitted invalid JSON object on line {line_number}") from error
    return matches


@dataclass(frozen=True)
class _PipeReadFailure:
    error: BaseException


@dataclass(frozen=True)
class _PipeEof:
    pass


_PIPE_EOF: Final = _PipeEof()
PipeEvent = bytes | _PipeReadFailure | _PipeEof


def _put_pipe_event(
    events: queue.Queue[PipeEvent],
    event: PipeEvent,
    stop_reading: threading.Event,
) -> None:
    while not stop_reading.is_set():
        try:
            events.put(event, timeout=0.05)
        except queue.Full:
            continue
        return


def _queue_pipe_chunks(
    pipe: BinaryIO,
    events: queue.Queue[PipeEvent],
    stop_reading: threading.Event,
) -> None:
    try:
        while not stop_reading.is_set():
            chunk = pipe.read(OUTLINE_READ_CHUNK_BYTES)
            if not chunk:
                _put_pipe_event(events, _PIPE_EOF, stop_reading)
                return
            _put_pipe_event(events, chunk, stop_reading)
    except (OSError, ValueError) as error:
        if not stop_reading.is_set():
            _put_pipe_event(events, _PipeReadFailure(error), stop_reading)


_OUTLINE_SYMBOL_TYPES: Final = frozenset(
    {
        "file",
        "module",
        "namespace",
        "package",
        "class",
        "method",
        "property",
        "field",
        "constructor",
        "enum",
        "interface",
        "function",
        "variable",
        "constant",
        "string",
        "number",
        "boolean",
        "array",
        "object",
        "key",
        "null",
        "enumMember",
        "struct",
        "event",
        "operator",
        "typeParameter",
    }
)


def _validate_outline_node(document: JsonObject, *, record_number: int, expected_role: str) -> None:
    if document.get("role") != expected_role:
        raise RuntimeError(f"ast-grep outline record {record_number} has invalid {expected_role} role")
    if document.get("symbolType") not in _OUTLINE_SYMBOL_TYPES:
        raise RuntimeError(f"ast-grep outline record {record_number} has invalid symbolType")
    if not isinstance(document.get("name"), str):
        raise RuntimeError(f"ast-grep outline record {record_number} has invalid name")
    if not isinstance(document.get("signature"), str):
        raise RuntimeError(f"ast-grep outline record {record_number} has invalid signature")
    if not isinstance(document.get("astKind"), str) or not document["astKind"]:
        raise RuntimeError(f"ast-grep outline record {record_number} has invalid astKind")
    _validate_source_range(document.get("range"), field_name="range", record_number=record_number)
    if expected_role == "item":
        if not isinstance(document.get("isImport"), bool) or not isinstance(document.get("isExported"), bool):
            raise RuntimeError(f"ast-grep outline record {record_number} has invalid item flags")
    elif not isinstance(document.get("isPublic"), bool):
        raise RuntimeError(f"ast-grep outline record {record_number} has invalid member visibility")


def _validate_and_count_outline_document(
    document: JsonObject,
    *,
    record_number: int,
    canonical: bool = False,
) -> int:
    raw_path = document.get("path")
    language = document.get("language")
    items = document.get("items")
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError(f"ast-grep outline record {record_number} has no path")
    if not isinstance(language, str) or (canonical and not language):
        raise RuntimeError(f"ast-grep outline record {record_number} has no language")
    if not isinstance(items, list):
        raise RuntimeError(f"ast-grep outline record {record_number} has no items list")

    count = 0
    nodes: list[tuple[JsonObject, str]] = []
    remaining = [(item, "item") for item in reversed(cast(list[object], items))]
    while remaining:
        node_value, expected_role = remaining.pop()
        if not isinstance(node_value, dict):
            raise RuntimeError(f"ast-grep outline record {record_number} contains a non-object node")
        node = _runtime_mapping(cast(object, node_value), field_name="outline node", record_number=record_number)
        nodes.append((node, expected_role))
        count += 1
        if "members" not in node:
            continue
        members = node["members"]
        if not isinstance(members, list):
            raise RuntimeError(f"ast-grep outline record {record_number} contains a non-list members field")
        remaining.extend((member, "member") for member in reversed(cast(list[object], members)))
    if canonical:
        for node, expected_role in nodes:
            _validate_outline_node(node, record_number=record_number, expected_role=expected_role)
    return count


def _parse_outline_record(raw_record: bytes, *, record_number: int) -> tuple[JsonObject, int] | None:
    if len(raw_record) > MAX_OUTLINE_RECORD_BYTES:
        raise RuntimeError(f"ast-grep outline record exceeds the {MAX_OUTLINE_RECORD_BYTES // 1024} KiB limit (record {record_number})")
    stripped = raw_record.strip()
    if not stripped:
        return None
    try:
        value = json.loads(stripped.decode("utf-8"), parse_constant=_reject_json_constant)
        if not isinstance(value, dict):
            raise RuntimeError(f"ast-grep outline emitted a non-object JSON record {record_number}")
        document = JSON_OBJECT_ADAPTER.validate_python(value, strict=True)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError, ValidationError) as error:
        raise RuntimeError(f"ast-grep outline emitted invalid JSON object in record {record_number}") from error
    return document, _validate_and_count_outline_document(document, record_number=record_number, canonical=True)


@dataclass(frozen=True)
class StreamedNDJSONProcess:
    records: list[JsonObject]
    observed_extra: bool
    returncode: int
    stderr: str
    terminated_for_limit: bool


def run_ndjson_process(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    working_directory: Path,
    record_parser: NDJSONRecordParser,
    record_filter: Callable[[JsonObject], bool] | None = None,
    item_limit: int | None = None,
    input_text: str | None = None,
    popen_factory: PopenFactory = subprocess.Popen,
) -> StreamedNDJSONProcess:
    if item_limit is not None and item_limit < 1:
        raise ValueError("NDJSON item limit must be positive")
    input_bytes = _encode_bounded_input(input_text)
    launch_environment = dict(os.environ)
    validate_process_budget(command, environment=launch_environment)
    try:
        process = popen_factory(
            list(command),
            stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(working_directory),
            env=launch_environment,
            shell=False,
            bufsize=0,
            **popen_process_group_options(),
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"Command executable was not found: {command[0]}") from error
    except OSError as error:
        raise RuntimeError(f"Command could not be executed: {error}") from error

    process_group = process_group_id(process)
    if process.stdout is None or process.stderr is None:
        terminate_and_reap(process, process_group)
        raise RuntimeError("Command pipes were not created")

    stop_io = threading.Event()
    stdout_events: queue.Queue[PipeEvent] = queue.Queue(maxsize=4)
    stderr = bytearray()
    stderr_overflow = [False]
    io_failures: list[BaseException] = []
    threads = [
        threading.Thread(
            target=_queue_pipe_chunks,
            args=(process.stdout, stdout_events, stop_io),
            name="ast-grep-ndjson-stdout",
            daemon=True,
        ),
        threading.Thread(
            target=_drain_bounded_pipe,
            args=(
                process.stderr,
                stderr,
                MAX_SUBPROCESS_DIAGNOSTIC_BYTES,
                stderr_overflow,
                io_failures,
                stop_io,
            ),
            name="ast-grep-ndjson-stderr",
            daemon=True,
        ),
    ]
    if input_bytes is not None:
        if process.stdin is None:
            terminate_and_reap(process, process_group)
            raise RuntimeError("Command stdin pipe was not created")
        threads.append(
            threading.Thread(
                target=_write_stdin,
                args=(process.stdin, input_bytes, io_failures, stop_io),
                name="ast-grep-ndjson-stdin",
                daemon=True,
            )
        )
    for thread in threads:
        thread.start()

    records: list[JsonObject] = []
    record_buffer = bytearray()
    record_number = 0
    aggregate_bytes = 0
    observed_items = 0
    observed_extra = False
    terminated_for_limit = False
    deadline = time.monotonic() + timeout_seconds

    def consume_record(raw_record: bytes) -> None:
        nonlocal observed_extra, observed_items, record_number
        record_number += 1
        if not raw_record:
            raise RuntimeError(f"Command emitted an empty NDJSON record {record_number}")
        parsed = record_parser(raw_record, record_number)
        if parsed is None:
            raise RuntimeError(f"Command emitted an empty NDJSON record {record_number}")
        record, item_count = parsed
        if record_filter is not None and not record_filter(record):
            return
        if item_count < 0:
            raise RuntimeError(f"NDJSON parser returned an invalid item count for record {record_number}")
        records.append(record)
        observed_items += item_count
        if item_limit is not None and observed_items > item_limit:
            observed_extra = True

    try:
        stdout_finished = False
        while not stdout_finished and not observed_extra:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise RuntimeError(f"Command timed out after {timeout_seconds:g} seconds")
            try:
                event = stdout_events.get(timeout=remaining_seconds)
            except queue.Empty as error:
                raise RuntimeError(f"Command timed out after {timeout_seconds:g} seconds") from error
            if isinstance(event, _PipeReadFailure):
                raise RuntimeError(f"Could not read command NDJSON output: {event.error}") from event.error
            if isinstance(event, _PipeEof):
                stdout_finished = True
                break

            aggregate_bytes += len(event)
            if aggregate_bytes > MAX_STRUCTURED_OUTPUT_BYTES:
                raise RuntimeError(
                    f"Command structured output exceeds the {MAX_STRUCTURED_OUTPUT_BYTES // (1024 * 1024)} MiB aggregate limit"
                )
            record_buffer.extend(event)
            newline_index = record_buffer.find(b"\n")
            while newline_index >= 0:
                if newline_index > MAX_NDJSON_RECORD_BYTES:
                    raise RuntimeError(
                        f"Command NDJSON record exceeds the {MAX_NDJSON_RECORD_BYTES // 1024} KiB limit (record {record_number + 1})"
                    )
                raw_record = bytes(record_buffer[:newline_index])
                del record_buffer[: newline_index + 1]
                consume_record(raw_record)
                if observed_extra:
                    break
                newline_index = record_buffer.find(b"\n")
            if not observed_extra and len(record_buffer) > MAX_NDJSON_RECORD_BYTES:
                raise RuntimeError(
                    f"Command NDJSON record exceeds the {MAX_NDJSON_RECORD_BYTES // 1024} KiB limit (record {record_number + 1})"
                )

        if observed_extra:
            if process.poll() is None:
                terminated_for_limit = True
                terminate_and_reap(process, process_group)
            else:
                process.wait()
        else:
            if record_buffer:
                raise RuntimeError(f"Command emitted an incomplete NDJSON record {record_number + 1}")
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise RuntimeError(f"Command timed out after {timeout_seconds:g} seconds")
            try:
                process.wait(timeout=remaining_seconds)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"Command timed out after {timeout_seconds:g} seconds") from error
    finally:
        terminate_and_reap(process, process_group)
        for thread in threads:
            thread.join(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
        stop_io.set()
        for pipe in (process.stdout, process.stderr):
            try:
                pipe.close()
            except OSError:
                pass
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=PROCESS_TERMINATION_GRACE_SECONDS)

    if any(thread.is_alive() for thread in threads):
        raise RuntimeError("Command pipe workers did not stop after process exit")
    if io_failures:
        raise RuntimeError(f"Could not transfer command data: {io_failures[0]}") from io_failures[0]
    if stderr_overflow[0]:
        raise RuntimeError(f"Command diagnostic output exceeds the {MAX_SUBPROCESS_DIAGNOSTIC_BYTES // 1024} KiB limit")
    stderr_text = _decode_utf8_output(bytes(stderr), truncated=False, label="stderr")
    return StreamedNDJSONProcess(
        records=records,
        observed_extra=observed_extra,
        returncode=process.returncode,
        stderr=stderr_text,
        terminated_for_limit=terminated_for_limit,
    )


def run_outline_process(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    working_directory: Path,
    node_limit: int,
    popen_factory: PopenFactory = subprocess.Popen,
) -> OutlineProcessResult:
    result = run_ndjson_process(
        command,
        timeout_seconds=timeout_seconds,
        working_directory=working_directory,
        record_parser=lambda raw, number: _parse_outline_record(raw, record_number=number),
        item_limit=node_limit,
        popen_factory=popen_factory,
    )
    if not result.terminated_for_limit and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {_bounded_error_text(result.stderr)}")
    residual_stderr = _residual_stderr(result.stderr)
    if residual_stderr:
        raise RuntimeError(f"ast-grep outline failed: {_bounded_error_text(residual_stderr)}")
    return result.records, result.observed_extra


def format_matches_as_text(matches: Sequence[Mapping[str, JsonValue]]) -> str:
    output_blocks: list[str] = []
    for match in matches:
        file_path = str(match.get("file", ""))
        range_mapping = string_object_dict(match.get("range")) or {}
        start_mapping = string_object_dict(range_mapping.get("start")) or {}
        end_mapping = string_object_dict(range_mapping.get("end")) or {}
        start_line_value = start_mapping.get("line", 0)
        start_line = (start_line_value if isinstance(start_line_value, int) else 0) + 1
        end_line_value = end_mapping.get("line", start_line - 1)
        end_line = (end_line_value if isinstance(end_line_value, int) else start_line - 1) + 1
        match_text = str(match.get("text", "")).rstrip()
        header = f"{file_path}:{start_line}" if start_line == end_line else f"{file_path}:{start_line}-{end_line}"
        details: list[str] = []
        rule_id = match.get("ruleId")
        if isinstance(rule_id, str) and rule_id:
            severity = match.get("severity")
            message = match.get("message")
            summary = f"Rule: {rule_id}"
            if isinstance(severity, str):
                summary += f" ({severity})"
            if isinstance(message, str) and message:
                summary += f" — {message}"
            details.append(summary)
        if "replacement" in match:
            replacement = match.get("replacement")
            preview = "<delete match>" if replacement == "" else str(replacement)
            details.append(f"Preview replacement: {preview}")
            if "replacementOffsets" in match:
                details.append(f"Replacement offsets: {json.dumps(match['replacementOffsets'], separators=(',', ':'))}")
        meta_variables = match.get("metaVariables")
        typed_meta_variables = string_object_dict(meta_variables)
        if typed_meta_variables is not None:
            transformed = string_object_dict(typed_meta_variables.get("transformed"))
            if transformed:
                details.append(f"Transformed metavariables: {json.dumps(transformed, separators=(',', ':'))}")
        for field_name in ("fix", "transform", "rewriters"):
            if field_name in match:
                details.append(f"{field_name}: {json.dumps(match[field_name], separators=(',', ':'))}")
        output_blocks.append("\n".join([header, match_text, *details]))
    return "\n\n".join(output_blocks)


def format_search_results(results: SearchResults) -> str:
    diagnostics = results.get("diagnostics")
    if diagnostics is not None:
        return f"Search failed: {json.dumps(diagnostics, separators=(',', ':'))}"
    if results["returned"] == 0:
        return "No matches found"
    noun = "match" if results["returned"] == 1 else "matches"
    header = f"Found {results['returned']} {noun}"
    if results["truncated"]:
        header += f" (limit {results['limit']}; additional matches exist)"
    next_cursor = results.get("next_cursor")
    if next_cursor is not None:
        header += f"\nContinue with cursor: {next_cursor}"
    elif results.get("snapshot_truncated"):
        header += "\nSnapshot reached the configured result cap; narrow the query for complete coverage."
    return f"{header}:\n\n{format_matches_as_text(results['matches'])}"


def pattern_failure_result(
    *,
    service: AstGrepService,
    error: RuntimeError,
    pattern: str,
    language: str,
    limit: int,
) -> SearchResults:
    message = str(error)
    if "INLINE_RULES" not in message and "parse rule" not in message.lower():
        raise error
    dump = service.dump_syntax_tree(
        code=pattern,
        language=language,
        format="pattern",
    )
    return {
        "matches": [],
        "returned": 0,
        "truncated": False,
        "limit": limit,
        "next_cursor": None,
        "snapshot_truncated": False,
        "diagnostics": {
            "kind": "pattern_parse",
            "message": message,
            "has_error_node": "ERROR" in dump,
            "syntax_dump": dump[:MAX_SUBPROCESS_DIAGNOSTIC_BYTES],
        },
    }


def search_tool_result(results: SearchResults, output_format: OutputFormat) -> CallToolResult:
    if output_format == "json":
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(results, separators=(",", ":")))],
            structured_content=dict(results),
        )
    return CallToolResult(
        content=[TextContent(type="text", text=format_search_results(results))],
        structured_content=dict(results),
    )


def _format_outline_nodes(nodes: Sequence[Mapping[str, object]], *, depth: int) -> list[str]:
    lines: list[str] = []
    for node in nodes:
        signature = node.get("signature")
        name = node.get("name")
        symbol_type = node.get("symbolType")
        label = signature if isinstance(signature, str) and signature else name
        if not isinstance(label, str) or not label:
            label = str(symbol_type) if isinstance(symbol_type, str) and symbol_type else "(unnamed)"
        lines.append(f"{'  ' * depth}- {label}")
        members = node.get("members")
        if isinstance(members, list):
            member_mappings = [mapping for item in cast(list[object], members) if (mapping := string_object_dict(item)) is not None]
            lines.extend(_format_outline_nodes(member_mappings, depth=depth + 1))
    return lines


def format_outline_results(results: OutlineResults) -> str:
    if results["returned"] == 0:
        header = "No outline nodes found"
    else:
        noun = "node" if results["returned"] == 1 else "nodes"
        header = f"Found {results['returned']} outline {noun}"
        if results["truncated"]:
            header += f" (limit {results['limit']}; additional nodes exist)"
    sections = [header]
    for file_result in results["files"]:
        file_header = file_result["file"]
        if file_result["language"]:
            file_header += f" ({file_result['language']})"
        sections.append("\n".join([file_header, *_format_outline_nodes(file_result["items"], depth=1)]))
    path_errors = results.get("path_errors", [])
    if path_errors:
        sections.append("\n".join(["Path errors:", *(f"- {error['path']}: {error['error']}" for error in path_errors)]))
    return "\n\n".join(sections)


def outline_tool_result(results: OutlineResults, output_format: OutputFormat) -> CallToolResult:
    text = json.dumps(results, separators=(",", ":")) if output_format == "json" else format_outline_results(results)
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=dict(results),
    )


def format_javascript_module_results(results: JavascriptModuleResults) -> str:
    if results["returned"] == 0:
        return "No JavaScript or TypeScript modules found"
    noun = "module" if results["returned"] == 1 else "modules"
    header = f"Inspected {results['returned']} {noun}"
    if results["truncated"]:
        header += f" (limit {results['limit']}; additional modules exist)"
    next_cursor = results.get("next_cursor")
    if next_cursor is not None:
        header += f"\nContinue with cursor: {next_cursor}"
    sections = [header]
    for module in results["modules"]:
        lines = [f"{module['file']} ({'module' if module['has_module_syntax'] else 'script'})"]
        for edge in module["edges"]:
            source = edge["specifier"] if edge["specifier"] is not None else edge["expression"]
            target = edge["resolved_path"] or edge["resolution"]
            lines.append(f"  - {edge['kind']} {source!r} -> {target}")
            if edge["resolution_error"] is not None:
                lines.append(f"    {edge['resolution_error']}")
        for diagnostic in module["diagnostics"]:
            lines.append(f"  - {diagnostic['severity']}: {diagnostic['message']}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def javascript_module_tool_result(results: JavascriptModuleResults, output_format: OutputFormat) -> CallToolResult:
    text = json.dumps(results, separators=(",", ":")) if output_format == "json" else format_javascript_module_results(results)
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=dict(results),
    )


def json_object_tool_result(result: JsonObject) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result, separators=(",", ":")))],
        structured_content=result,
    )


def json_object_format_tool_result(results: JsonObject, output_format: OutputFormat) -> CallToolResult:
    text = json.dumps(results, separators=(",", ":")) if output_format == "json" else json.dumps(results, indent=2)
    return CallToolResult(content=[TextContent(type="text", text=text)], structured_content=results)


def _validate_oxc_span(start: int, end: int, *, label: str) -> None:
    if start < 0 or end < start:
        raise RuntimeError(f"Oxc helper returned an invalid {label} span: {start}:{end}")


class AstGrepService:
    def __init__(
        self,
        runtime: RuntimeServices,
        *,
        runner: ProcessRunner = subprocess.run,
        outline_runner: OutlineProcessRunner = run_outline_process,
    ) -> None:
        self.runtime = runtime
        self.runner = runner
        self.outline_runner = outline_runner

    def _run(
        self,
        subcommand: str,
        arguments: Sequence[str],
        *,
        input_text: str | None = None,
        working_directory: Path | None = None,
        allow_no_matches: bool = False,
        allow_stderr_on_no_matches: bool = False,
        stderr_limit: int = MAX_SUBPROCESS_DIAGNOSTIC_BYTES,
        truncate_stderr: bool = True,
    ) -> CompletedTextProcess:
        with ExitStack() as stack:
            config_path = self._resolve_operation_config_path(stack)

            command = [
                *self.runtime.executable.command_prefix,
                subcommand,
                "--config",
                str(config_path),
                *arguments,
            ]
            result = run_process(
                command,
                timeout_seconds=self.runtime.command_timeout_seconds,
                input_text=input_text,
                working_directory=working_directory or self.runtime.working_directory,
                allowed_exit_codes=frozenset({0, 1}) if allow_no_matches else frozenset({0}),
                runner=self.runner,
                stderr_limit=stderr_limit,
                truncate_stderr=truncate_stderr,
            )
        if not allow_stderr_on_no_matches:
            residual = _residual_stderr(result.stderr)
            if residual:
                raise RuntimeError(f"ast-grep search failed: {_bounded_error_text(residual)}")
        return result

    def _resolve_operation_config_path(self, stack: ExitStack, override: Path | None = None) -> Path:
        snapshot = self.runtime.config_snapshot
        config_path = override or (snapshot.inline_config_path if snapshot is not None else self.runtime.config_path)
        if config_path is None:
            temporary_directory = Path(
                stack.enter_context(
                    TemporaryDirectory(
                        prefix="operation-",
                        dir=private_runtime_root(self.runtime.working_directory, self.runtime.allowed_roots),
                    )
                )
            )
            config_path = temporary_directory / "sgconfig.yml"
            config_path.write_text(NEUTRAL_AST_GREP_CONFIG, encoding="utf-8")
        return config_path

    def _run_match_stream(
        self,
        subcommand: str,
        arguments: Sequence[str],
        *,
        working_directory: Path,
        item_limit: int,
        input_text: str | None = None,
        config_path_override: Path | None = None,
        match_filter: Callable[[JsonObject], bool] | None = None,
    ) -> tuple[list[JsonObject], bool]:
        with ExitStack() as stack:
            config_path = self._resolve_operation_config_path(stack, config_path_override)
            command = [
                *self.runtime.executable.command_prefix,
                subcommand,
                "--config",
                str(config_path),
                *arguments,
            ]
            if self.runner is not subprocess.run:
                completed = run_process(
                    command,
                    timeout_seconds=self.runtime.command_timeout_seconds,
                    input_text=input_text,
                    working_directory=working_directory,
                    allowed_exit_codes=frozenset({0, 1}),
                    runner=self.runner,
                )
                residual = _residual_stderr(completed.stderr)
                if residual:
                    raise RuntimeError(f"ast-grep search failed: {_bounded_error_text(residual)}")
                records = parse_stream_matches(completed.stdout)
                filtered = records if match_filter is None else [record for record in records if match_filter(record)]
                return filtered, len(filtered) > item_limit

            streamed = run_ndjson_process(
                command,
                timeout_seconds=self.runtime.command_timeout_seconds,
                working_directory=working_directory,
                record_parser=_parse_match_record,
                record_filter=match_filter,
                item_limit=item_limit,
                input_text=input_text,
            )
        if not streamed.terminated_for_limit and streamed.returncode not in {0, 1}:
            raise RuntimeError(f"Command failed with exit code {streamed.returncode}: {_bounded_error_text(streamed.stderr)}")
        residual = _residual_stderr(streamed.stderr)
        if residual:
            raise RuntimeError(f"ast-grep search failed: {_bounded_error_text(residual)}")
        return streamed.records, streamed.observed_extra

    def _supported_language_ids(self) -> tuple[str, ...]:
        custom: set[str] = set()
        snapshot = self.runtime.config_snapshot
        if snapshot is not None and snapshot.capabilities["custom_languages"]:
            documents = yaml_parser.safe_load_all(snapshot.project_config_path.read_text(encoding="utf-8"))
            for document in documents:
                document_mapping = string_object_dict(document)
                if document_mapping is None:
                    continue
                custom_languages = string_object_dict(document_mapping.get("customLanguages"))
                if custom_languages is not None:
                    custom.update(custom_languages)
        return tuple(sorted({*BUILTIN_LANGUAGE_IDS, *custom}))

    def _require_language(self, language: str) -> None:
        if not language:
            raise ValueError("language is required")
        supported = self._supported_language_ids()
        if language not in supported and language.lower() not in BUILTIN_LANGUAGE_IDS:
            raise ValueError(f"Unsupported language {language!r}; use one of: {', '.join(supported)}")

    def _resolve_project(self, project_folder: str) -> Path:
        base = self.runtime.allowed_roots[0] if len(self.runtime.allowed_roots) == 1 else self.runtime.working_directory
        project = _resolve_existing_path(
            project_folder,
            base=base,
            kind="directory",
        )
        _require_allowed(project, self.runtime.allowed_roots, label="Project folder")
        return project

    def _resolve_paths(self, project: Path, raw_paths: Sequence[str] | None) -> list[Path]:
        paths = ["."] if raw_paths is None else list(raw_paths)
        if not paths:
            raise ValueError("paths must contain at least one relative path")
        resolved_paths: list[Path] = []
        for raw_path in paths:
            if not raw_path:
                raise ValueError("paths cannot contain empty values")
            candidate = Path(raw_path)
            if candidate.is_absolute():
                raise ValueError(f"Search paths must be relative to project_folder: {raw_path}")
            effective_path = project / candidate
            try:
                resolved = effective_path.resolve(strict=True)
            except FileNotFoundError as error:
                duplicate_hint = (
                    " Remove the repeated project prefix; paths are relative to project_folder."
                    if candidate.parts and candidate.parts[0] == project.name
                    else ""
                )
                raise ValueError(f"Search path does not exist: {raw_path} (effective path: {effective_path}).{duplicate_hint}") from error
            if not is_within(resolved, project):
                raise ValueError(f"Search path resolves outside project_folder: {raw_path}")
            _require_allowed(resolved, self.runtime.allowed_roots, label="Search path")
            if resolved not in resolved_paths:
                resolved_paths.append(resolved)
        return resolved_paths

    def _resolve_outline_paths(
        self,
        project: Path,
        raw_paths: Sequence[str] | None,
        include_globs: Sequence[str] | None,
        exclude_globs: Sequence[str] | None,
        *,
        strict_paths: bool,
    ) -> tuple[list[Path], list[OutlinePathError]]:
        if not raw_paths and not include_globs:
            raise ValueError("outline requires paths or include_globs")
        if raw_paths is not None and len(raw_paths) > MAX_OUTLINE_PATHS:
            raise ValueError(f"paths cannot contain more than {MAX_OUTLINE_PATHS} entries")

        errors: list[OutlinePathError] = []
        resolved_paths: list[Path] = []

        def add_path(resolved: Path) -> None:
            if not is_within(resolved, project):
                raise ValueError(f"Outline path resolves outside project_folder: {resolved}")
            _require_allowed(resolved, self.runtime.allowed_roots, label="Outline path")
            if resolved not in resolved_paths:
                resolved_paths.append(resolved)

        for raw_path in raw_paths or ():
            if not raw_path or "\0" in raw_path:
                raise ValueError("Outline paths cannot contain empty or NUL values")
            candidate = Path(raw_path)
            if candidate.is_absolute():
                raise ValueError(f"Outline paths must be relative to project_folder: {raw_path}")
            effective_path = project / candidate
            try:
                resolved = effective_path.resolve(strict=True)
            except FileNotFoundError as error:
                duplicate_hint = (
                    " Remove the repeated project prefix; paths are relative to project_folder."
                    if candidate.parts and candidate.parts[0] == project.name
                    else ""
                )
                message = f"Outline path does not exist: {raw_path} (effective path: {effective_path}).{duplicate_hint}"
                if strict_paths:
                    raise ValueError(message) from error
                errors.append({"path": raw_path, "error": message})
                continue
            if not resolved.is_file():
                message = f"Outline path must be a regular file: {raw_path}"
                if strict_paths:
                    raise ValueError(message)
                errors.append({"path": raw_path, "error": message})
                continue
            add_path(resolved)

        normalized_excludes: list[str] = []
        for pattern in exclude_globs or ():
            candidate = Path(pattern)
            if not pattern or "\0" in pattern or pattern.startswith("!") or candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(f"Invalid outline exclude glob: {pattern!r}")
            normalized_excludes.append(pattern)

        for pattern in include_globs or ():
            candidate = Path(pattern)
            if not pattern or "\0" in pattern or pattern.startswith("!") or candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(f"Invalid outline include glob: {pattern!r}")
            for matched in sorted(project.glob(pattern)):
                if not matched.is_file():
                    continue
                resolved = matched.resolve(strict=True)
                if not is_within(resolved, project):
                    raise ValueError(f"Outline glob resolved outside project_folder: {pattern}")
                relative = resolved.relative_to(project)
                if any(relative.match(excluded) for excluded in normalized_excludes):
                    continue
                add_path(resolved)

        if len(resolved_paths) > MAX_OUTLINE_PATHS:
            raise ValueError(f"outline selection resolved {len(resolved_paths)} files; narrow it to at most {MAX_OUTLINE_PATHS}")
        return resolved_paths, errors

    @staticmethod
    def _glob_arguments(include_globs: Sequence[str] | None, exclude_globs: Sequence[str] | None) -> list[str]:
        arguments: list[str] = []
        for glob in include_globs or ():
            if not glob or "\0" in glob or glob.startswith("!"):
                raise ValueError(f"Invalid include glob: {glob!r}")
            arguments.append(f"--globs={glob}")
        for glob in exclude_globs or ():
            if not glob or "\0" in glob or glob.startswith("!"):
                raise ValueError(f"Invalid exclude glob: {glob!r}")
            arguments.append(f"--globs=!{glob}")
        return arguments

    @staticmethod
    def _contained_relative_path(raw_path: str, project: Path, *, noun: str, producer: str = "ast-grep") -> str:
        file_path = Path(raw_path)
        if not file_path.is_absolute():
            file_path = project / file_path
        try:
            resolved_file = file_path.resolve(strict=True)
        except FileNotFoundError as error:
            raise RuntimeError(f"{producer} returned {noun} that no longer exists: {raw_path}") from error
        if not is_within(resolved_file, project):
            raise RuntimeError(f"{producer} returned {noun} outside project_folder: {raw_path}")
        return resolved_file.relative_to(project).as_posix()

    def _normalize_match_path(self, match: JsonObject, project: Path) -> JsonObject:
        raw_file = match.get("file")
        if not isinstance(raw_file, str) or not raw_file:
            return match
        normalized = dict(match)
        normalized_file = self._contained_relative_path(raw_file, project, noun="a match")
        normalized["file"] = normalized_file
        normalized["evidence_kind"] = "syntax"
        range_value = match.get("range")
        range_mapping = string_object_dict(range_value)
        if range_mapping is not None:
            start = string_object_dict(range_mapping.get("start"))
            if start is not None:
                line = start.get("line")
                column = start.get("column")
                if isinstance(line, int) and isinstance(column, int):
                    normalized["lsp_handoff"] = {
                        "project_folder": str(project),
                        "file": normalized_file,
                        "line": line,
                        "character": column,
                        "coordinate_base": 0,
                    }
        return normalized

    def _result_limit(self, requested: int | None) -> int:
        limit = self.runtime.default_max_results if requested is None else requested
        if not 1 <= limit <= self.runtime.max_results_cap:
            raise ValueError(f"max_results must be between 1 and {self.runtime.max_results_cap}")
        return limit

    @staticmethod
    def _oxc_error_module(path: str, message: str) -> JavascriptModule:
        return {
            "file": path,
            "has_module_syntax": False,
            "source_type": "script",
            "package": None,
            "commonjs_exports": [],
            "import_meta_spans": [],
            "edges": [],
            "diagnostics": [
                {
                    "severity": "Error",
                    "message": message,
                    "help": None,
                    "codeframe": None,
                    "labels": [],
                }
            ],
        }

    def _normalize_oxc_module(
        self,
        module: JavascriptModule,
        *,
        project: Path,
        expected_files: frozenset[str],
    ) -> JavascriptModule:
        normalized_file = self._contained_relative_path(
            module["file"],
            project,
            noun="a module",
            producer="Oxc helper",
        )
        if normalized_file not in expected_files:
            raise RuntimeError(f"Oxc helper returned an unexpected module: {normalized_file}")
        normalized_import_metas: list[OxcSpan] = []
        for span in module["import_meta_spans"]:
            _validate_oxc_span(span["start"], span["end"], label="import.meta")
            normalized_import_metas.append({"start": span["start"], "end": span["end"]})
        normalized_commonjs_exports: list[OxcCommonJSExport] = []
        for export in module["commonjs_exports"]:
            _validate_oxc_span(export["start"], export["end"], label="CommonJS export")
            normalized_commonjs_exports.append({"start": export["start"], "end": export["end"], "text": export["text"]})
        normalized_edges: list[JavascriptModuleEdge] = []
        for edge in module["edges"]:
            _validate_oxc_span(edge["start"], edge["end"], label="module edge")
            if edge["kind"] == "dynamic":
                if edge["specifier"] is not None or edge["expression"] is None or edge["resolution"] != "dynamic":
                    raise RuntimeError("Oxc helper returned an invalid dynamic module edge")
            elif edge["specifier"] is None or edge["expression"] is not None or edge["resolution"] == "dynamic":
                raise RuntimeError("Oxc helper returned an invalid static module edge")
            normalized_resolved_path: str | None = None
            normalized_package_json_path: str | None = None
            if edge["resolution"] == "resolved":
                if edge["resolved_path"] is None:
                    raise RuntimeError("Oxc helper returned a resolved edge without a path")
                normalized_resolved_path = self._contained_relative_path(
                    edge["resolved_path"],
                    project,
                    noun="a resolved module",
                    producer="Oxc helper",
                )
                if edge["package_json_path"] is not None:
                    normalized_package_json_path = self._contained_relative_path(
                        edge["package_json_path"],
                        project,
                        noun="package metadata",
                        producer="Oxc helper",
                    )
            elif edge["resolved_path"] is not None or edge["package_json_path"] is not None:
                raise RuntimeError("Oxc helper returned a non-resolved edge with a contained path")
            normalized_edges.append(
                {
                    "kind": edge["kind"],
                    "module_system": edge["module_system"],
                    "specifier": edge["specifier"],
                    "expression": edge["expression"],
                    "start": edge["start"],
                    "end": edge["end"],
                    "resolution": edge["resolution"],
                    "resolved_path": normalized_resolved_path,
                    "package_json_path": normalized_package_json_path,
                    "module_type": edge["module_type"],
                    "resolution_error": edge["resolution_error"],
                }
            )
        normalized_diagnostics: list[OxcDiagnostic] = []
        for diagnostic in module["diagnostics"]:
            labels: list[OxcDiagnosticLabel] = []
            for label in diagnostic["labels"]:
                _validate_oxc_span(label["start"], label["end"], label="diagnostic")
                labels.append({"message": label["message"], "start": label["start"], "end": label["end"]})
            normalized_diagnostics.append(
                {
                    "severity": diagnostic["severity"],
                    "message": diagnostic["message"],
                    "help": diagnostic["help"],
                    "codeframe": diagnostic["codeframe"],
                    "labels": labels,
                }
            )
        package = module["package"]
        normalized_package: OxcPackageMetadata | None = None
        if package is not None:
            normalized_package = {
                "path": self._contained_relative_path(package["path"], project, noun="package metadata", producer="Oxc helper"),
                "name": package["name"],
                "type": package["type"],
            }
        return {
            "file": normalized_file,
            "has_module_syntax": module["has_module_syntax"],
            "source_type": module["source_type"],
            "package": normalized_package,
            "commonjs_exports": normalized_commonjs_exports,
            "import_meta_spans": normalized_import_metas,
            "edges": normalized_edges,
            "diagnostics": normalized_diagnostics,
        }

    @staticmethod
    def _transform_options(options: OxcTransformOptions | None) -> JsonObject:
        if options is None:
            return {}
        values = options.model_dump(exclude_none=True)
        transformed: JsonObject = {}
        if "lang" in values:
            transformed["lang"] = cast(JsonValue, values["lang"])
        if "source_type" in values:
            transformed["sourceType"] = cast(JsonValue, values["source_type"])
        if "target" in values:
            transformed["target"] = cast(JsonValue, values["target"])
        transformed["sourcemap"] = bool(values.get("sourcemap", False))
        if values.get("declaration"):
            transformed["typescript"] = {"declaration": True}
        return transformed

    @staticmethod
    def _minify_options(options: OxcMinifyOptions | None) -> JsonObject:
        values = (options or OxcMinifyOptions()).model_dump()
        return {
            "compress": bool(values["compress"]),
            "mangle": bool(values["mangle"]),
            "sourcemap": bool(values["sourcemap"]),
            "codegen": {"legalComments": cast(str, values["legal_comments"])},
        }

    def _run_oxc_compute(
        self,
        *,
        operation: Literal["transform", "minify"],
        filename: str,
        code: str,
        options: JsonObject,
    ) -> JsonObject:
        helper = self.runtime.oxc_helper
        expected_versions = self.runtime.oxc_versions
        if helper is None or expected_versions is None:
            raise RuntimeError("Oxc compute tools are not configured")
        request: JsonObject = {
            "operation": operation,
            "filename": filename,
            "code": code,
            "options": options,
        }
        if self.runtime.oxc_worker is not None:
            response = self.runtime.oxc_worker.request(request, timeout=self.runtime.command_timeout_seconds)
        else:
            completed = run_process(
                helper.command_prefix,
                timeout_seconds=self.runtime.command_timeout_seconds,
                input_text=json.dumps(request, separators=(",", ":")),
                working_directory=self.runtime.working_directory,
                runner=self.runner,
                stdout_limit=MAX_STRUCTURED_OUTPUT_BYTES,
                stderr_limit=MAX_SUBPROCESS_DIAGNOSTIC_BYTES,
            )
            if _has_meaningful_diagnostic(completed.stderr):
                raise RuntimeError(f"Oxc helper emitted unexpected diagnostics: {_bounded_error_text(completed.stderr)}")
            try:
                response = JSON_OBJECT_ADAPTER.validate_python(
                    json.loads(completed.stdout, parse_constant=_reject_json_constant), strict=True
                )
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                raise RuntimeError(f"Oxc helper returned invalid JSON: {_bounded_error_text(completed.stdout)}") from error
        versions = response.get("versions")
        if versions != expected_versions:
            raise RuntimeError(f"Oxc helper version changed after startup: {versions!r}")
        if response.get("operation") != operation:
            raise RuntimeError(f"Oxc helper returned the wrong operation: {response.get('operation')!r}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Oxc helper returned no result object")
        return cast(JsonObject, result)

    def resolve_oxc_source(self, source: InlineSource | FileSource) -> tuple[str, str, Path]:
        if isinstance(source, InlineSource):
            encoded = source.code.encode("utf-8")
            if len(encoded) > MAX_SNIPPET_INPUT_BYTES:
                raise ValueError("Inline source exceeds the configured byte limit")
            return source.filename, source.code, self.runtime.working_directory
        project, selected = self._selected_oxc_files(
            project_folder=source.project_folder,
            paths=[source.path],
            include_globs=None,
            exclude_globs=None,
            strict_paths=True,
            max_results=1,
        )
        if len(selected) != 1:
            raise ValueError("File source did not resolve to exactly one file")
        selected_path = selected[0]
        if selected_path.stat().st_size > MAX_OXC_FILE_BYTES:
            raise ValueError("File source exceeds the configured byte limit")
        return selected_path.relative_to(project).as_posix(), selected_path.read_text(encoding="utf-8"), project

    def transform_code(self, *, filename: str, code: str, options: OxcTransformOptions | None) -> JsonObject:
        return self._run_oxc_compute(operation="transform", filename=filename, code=code, options=self._transform_options(options))

    def minify_code(self, *, filename: str, code: str, options: OxcMinifyOptions | None) -> JsonObject:
        return self._run_oxc_compute(operation="minify", filename=filename, code=code, options=self._minify_options(options))

    def _selected_oxc_files(
        self,
        *,
        project_folder: str,
        paths: Sequence[str] | None,
        include_globs: Sequence[str] | None,
        exclude_globs: Sequence[str] | None,
        strict_paths: bool,
        max_results: int | None = None,
    ) -> tuple[Path, list[Path]]:
        project = self._resolve_project(project_folder)
        selected, errors = self._resolve_outline_paths(project, paths, include_globs, exclude_globs, strict_paths=strict_paths)
        if errors:
            raise ValueError("; ".join(error["error"] for error in errors))
        unsupported = [path.relative_to(project).as_posix() for path in selected if path.suffix.lower() not in OXC_SOURCE_EXTENSIONS]
        if unsupported:
            raise ValueError(f"Unsupported JavaScript or TypeScript files: {', '.join(unsupported)}")
        limit = self._result_limit(max_results)
        if len(selected) > limit:
            raise ValueError(f"Selected {len(selected)} files, exceeding max_results={limit}")
        return project, selected

    @staticmethod
    def _emitted_relative_path(path: Path, *, operation: Literal["transform", "minify"]) -> Path:
        if operation == "transform" and path.suffix.lower() in {".ts", ".tsx", ".mts", ".cts", ".jsx"}:
            return path.with_suffix(".js")
        if operation == "minify":
            return path.with_name(f"{path.stem}.min{path.suffix}")
        return path

    def emit_files(
        self,
        *,
        operation: Literal["transform", "minify"],
        project_folder: str,
        output_root: str,
        paths: Sequence[str] | None,
        include_globs: Sequence[str] | None,
        exclude_globs: Sequence[str] | None,
        strict_paths: bool,
        conflict_policy: ConflictPolicy,
        allow_source_overwrite: bool,
        options: OxcTransformOptions | OxcMinifyOptions | None,
        max_results: int | None = None,
    ) -> JsonObject:
        project, selected = self._selected_oxc_files(
            project_folder=project_folder,
            paths=paths,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            strict_paths=strict_paths,
            max_results=max_results,
        )
        output_candidate = Path(output_root)
        if output_candidate.is_absolute() or ".." in output_candidate.parts or not output_root:
            raise ValueError("output_root must be a non-empty relative path inside project_folder")
        writes: list[PlannedWrite] = []
        diagnostics: list[JsonValue] = []
        artifacts: list[JsonValue] = []
        for path in selected:
            relative = path.relative_to(project)
            emitted = self._emitted_relative_path(relative, operation=operation)
            target = project / output_candidate / emitted
            code = path.read_text(encoding="utf-8")
            if operation == "transform":
                if options is not None and not isinstance(options, OxcTransformOptions):
                    raise TypeError("transform options have the wrong type")
                result = self.transform_code(filename=path.name, code=code, options=options)
            else:
                if options is not None and not isinstance(options, OxcMinifyOptions):
                    raise TypeError("minify options have the wrong type")
                result = self.minify_code(filename=path.name, code=code, options=options)
            output_code = result.get("code")
            if not isinstance(output_code, str):
                raise RuntimeError(f"Oxc {operation} returned no code for {path}")
            diagnostics.extend(cast(list[JsonValue], result.get("diagnostics", [])))
            writes.append(PlannedWrite(source=path, target=target, content=output_code))
            artifact_paths: JsonObject = {"code": target.relative_to(project).as_posix()}
            map_value = result.get("map")
            if map_value is not None:
                map_target = Path(f"{target}.map")
                map_content = map_value if isinstance(map_value, str) else json.dumps(map_value, separators=(",", ":"))
                writes.append(PlannedWrite(source=None, target=map_target, content=map_content))
                artifact_paths["map"] = map_target.relative_to(project).as_posix()
            declaration = result.get("declaration")
            if isinstance(declaration, str):
                declaration_target = target.with_suffix(".d.ts")
                writes.append(PlannedWrite(source=None, target=declaration_target, content=declaration))
                artifact_paths["declaration"] = declaration_target.relative_to(project).as_posix()
            declaration_map = result.get("declaration_map")
            if declaration_map is not None:
                declaration_map_target = target.with_suffix(".d.ts.map")
                declaration_map_content = (
                    declaration_map if isinstance(declaration_map, str) else json.dumps(declaration_map, separators=(",", ":"))
                )
                writes.append(PlannedWrite(source=None, target=declaration_map_target, content=declaration_map_content))
                artifact_paths["declaration_map"] = declaration_map_target.relative_to(project).as_posix()
            legal_comments = result.get("legal_comments")
            if isinstance(legal_comments, list) and legal_comments:
                legal_target = Path(f"{target}.LEGAL.txt")
                legal_text = "\n".join(comment for comment in legal_comments if isinstance(comment, str))
                writes.append(PlannedWrite(source=None, target=legal_target, content=legal_text))
                artifact_paths["legal_comments"] = legal_target.relative_to(project).as_posix()
            mangle_cache = result.get("mangle_cache")
            if isinstance(mangle_cache, dict):
                cache_target = Path(f"{target}.mangle.json")
                writes.append(
                    PlannedWrite(source=None, target=cache_target, content=json.dumps(mangle_cache, sort_keys=True, separators=(",", ":")))
                )
                artifact_paths["mangle_cache"] = cache_target.relative_to(project).as_posix()
            artifacts.append(
                {
                    "source": relative.as_posix(),
                    "artifacts": artifact_paths,
                    "helpers_used": result.get("helpers_used"),
                }
            )
        batch = MutationService(max_file_bytes=MAX_OXC_FILE_BYTES, max_total_bytes=MAX_OXC_TOTAL_SOURCE_BYTES).apply(
            project=project,
            writes=writes,
            conflict_policy=conflict_policy,
            allow_source_overwrite=allow_source_overwrite,
        )
        return {
            "applied": [cast(JsonValue, asdict(item)) for item in batch.applied],
            "skipped": list(batch.skipped),
            "artifacts": artifacts,
            "diagnostics": diagnostics,
        }

    def execute_typescript(
        self,
        *,
        project_folder: str,
        entry: str,
        args: Sequence[str],
        stdin: str | None,
        timeout_seconds: float,
    ) -> JsonObject:
        helper = self.runtime.typescript_execution_helper
        if helper is None:
            raise RuntimeError("TypeScript execution is not configured")
        project = self._resolve_project(project_folder)
        selected, errors = self._resolve_outline_paths(project, [entry], None, None, strict_paths=True)
        if errors or len(selected) != 1:
            raise ValueError(f"Could not resolve TypeScript entry: {entry}")
        request: JsonObject = {
            "project_root": str(project),
            "entry": selected[0].relative_to(project).as_posix(),
            "args": list(args),
            "stdin": stdin,
            "timeout_seconds": timeout_seconds,
            "profile": self.runtime.typescript_execution_profile,
        }
        started = time.monotonic()
        completed = run_process(
            helper.command_prefix,
            timeout_seconds=min(timeout_seconds, self.runtime.command_timeout_seconds),
            input_text=json.dumps(request, separators=(",", ":")),
            working_directory=project,
            runner=self.runner,
            stdout_limit=MAX_STRUCTURED_OUTPUT_BYTES,
            stderr_limit=MAX_SUBPROCESS_DIAGNOSTIC_BYTES,
        )
        try:
            result = JSON_OBJECT_ADAPTER.validate_python(json.loads(completed.stdout, parse_constant=_reject_json_constant), strict=True)
        except (json.JSONDecodeError, ValidationError, ValueError) as error:
            raise RuntimeError(f"TypeScript execution helper returned invalid JSON: {_bounded_error_text(completed.stdout)}") from error
        result["duration_ms"] = int((time.monotonic() - started) * 1000)
        result["profile"] = self.runtime.typescript_execution_profile
        return result

    def inspect_postgres(self, *, operation: Literal["parse", "scan", "fingerprint", "plpgsql", "deparse"], sql: str) -> JsonObject:
        helper = self.runtime.postgres_helper
        versions = self.runtime.postgres_versions
        if helper is None or versions is None:
            raise RuntimeError("PostgreSQL parser is not configured")
        request: JsonObject = {"operation": operation, "sql": sql}
        completed = run_process(
            helper.command_prefix,
            timeout_seconds=self.runtime.command_timeout_seconds,
            input_text=json.dumps(request, separators=(",", ":")),
            working_directory=self.runtime.working_directory,
            runner=self.runner,
            stdout_limit=MAX_STRUCTURED_OUTPUT_BYTES,
            stderr_limit=MAX_SUBPROCESS_DIAGNOSTIC_BYTES,
            allowed_exit_codes=frozenset({0, 1}),
        )
        if completed.returncode != 0:
            try:
                failure = JSON_OBJECT_ADAPTER.validate_json(completed.stderr, strict=True)
            except (ValidationError, ValueError) as error:
                raise RuntimeError("PostgreSQL helper failed") from error
            failure_value = failure.get("error")
            if not isinstance(failure_value, dict):
                raise RuntimeError("PostgreSQL helper failed")
            failure_type = failure_value.get("type")
            message = failure_value.get("message")
            if failure_type == "parse_error" and isinstance(message, str):
                cursor = failure_value.get("cursor_position")
                location = f" at cursor {cursor}" if isinstance(cursor, int) and not isinstance(cursor, bool) else ""
                raise ToolError(f"PostgreSQL parse error{location}: {message}")
            raise RuntimeError("PostgreSQL helper failed")
        if _has_meaningful_diagnostic(completed.stderr):
            raise RuntimeError("PostgreSQL helper emitted unexpected diagnostics")
        try:
            result = JSON_OBJECT_ADAPTER.validate_python(json.loads(completed.stdout, parse_constant=_reject_json_constant), strict=True)
        except (json.JSONDecodeError, ValidationError, ValueError) as error:
            raise RuntimeError(f"PostgreSQL helper returned invalid JSON: {_bounded_error_text(completed.stdout)}") from error
        if (
            result.get("worker_version") != versions.get("worker")
            or result.get("parser_version") != versions.get("parser")
            or result.get("deparser_version") != versions.get("deparser")
            or result.get("postgres_major") != versions.get("postgres_major")
        ):
            raise RuntimeError("PostgreSQL helper version changed after startup")
        try:
            payload = JSON_OBJECT_ADAPTER.validate_python(result.get("result"), strict=True)
        except ValidationError as error:
            raise RuntimeError("PostgreSQL helper returned an invalid result object") from error
        payload["parser_version"] = result["parser_version"]
        payload["deparser_version"] = result["deparser_version"]
        payload["postgres_major"] = result["postgres_major"]
        payload["mode"] = operation
        payload["source_digest"] = result.get("source_digest")
        return payload

    def inspect_postgres_files(
        self,
        *,
        operation: Literal["parse", "scan", "fingerprint", "plpgsql"],
        project_folder: str,
        paths: Sequence[str] | None,
        include_globs: Sequence[str] | None,
        exclude_globs: Sequence[str] | None,
        strict_paths: bool,
    ) -> PostgresFilesResults:
        project = self._resolve_project(project_folder)
        selected, errors = self._resolve_outline_paths(
            project,
            paths,
            include_globs,
            exclude_globs,
            strict_paths=strict_paths,
        )
        files: list[JsonObject] = []
        total_source_bytes = 0
        for error in errors:
            files.append({"file": error["path"], "error": error["error"]})
        for path in selected:
            if path.suffix.lower() not in {".sql", ".pgsql"}:
                if strict_paths:
                    raise ValueError(f"Unsupported PostgreSQL file: {path}")
                files.append({"file": path.relative_to(project).as_posix(), "error": "unsupported extension"})
                continue
            source_bytes = path.stat().st_size
            if source_bytes > MAX_SNIPPET_INPUT_BYTES:
                raise ValueError(f"PostgreSQL source exceeds the {MAX_SNIPPET_INPUT_BYTES}-byte file limit: {path}")
            total_source_bytes += source_bytes
            if total_source_bytes > MAX_OXC_TOTAL_SOURCE_BYTES:
                raise ValueError(f"PostgreSQL sources exceed the {MAX_OXC_TOTAL_SOURCE_BYTES}-byte aggregate limit")
            result = self.inspect_postgres(operation=operation, sql=path.read_text(encoding="utf-8"))
            result["file"] = path.relative_to(project).as_posix()
            files.append(result)
        return {"files": files, "returned": len(files), "truncated": False, "limit": MAX_OUTLINE_PATHS}

    def inspect_typescript_project(
        self,
        *,
        project_folder: str,
        tsconfig: str,
        paths: Sequence[str] | None,
        include_emit: bool,
        include_code_actions: bool,
        max_results: int,
    ) -> JsonObject:
        helper = self.runtime.typescript_project_helper
        versions = self.runtime.typescript_versions
        if helper is None or versions is None:
            raise RuntimeError("TypeScript Compiler API inspection is not configured")
        project = self._resolve_project(project_folder)
        request: JsonObject = {
            "project_root": str(project),
            "tsconfig": tsconfig,
            "paths": list(paths) if paths is not None else None,
            "include_emit": include_emit,
            "include_code_actions": include_code_actions,
            "max_results": max_results,
        }
        if self.runtime.typescript_project_worker is not None:
            result = self.runtime.typescript_project_worker.request(request, timeout=self.runtime.command_timeout_seconds)
        else:
            completed = run_process(
                helper.command_prefix,
                timeout_seconds=self.runtime.command_timeout_seconds,
                input_text=json.dumps(request, separators=(",", ":")),
                working_directory=project,
                runner=self.runner,
                stdout_limit=MAX_STRUCTURED_OUTPUT_BYTES,
                stderr_limit=MAX_SUBPROCESS_DIAGNOSTIC_BYTES,
            )
            if _has_meaningful_diagnostic(completed.stderr):
                raise RuntimeError(f"TypeScript helper emitted unexpected diagnostics: {_bounded_error_text(completed.stderr)}")
            try:
                result = JSON_OBJECT_ADAPTER.validate_python(
                    json.loads(completed.stdout, parse_constant=_reject_json_constant), strict=True
                )
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                raise RuntimeError(f"TypeScript helper returned invalid JSON: {_bounded_error_text(completed.stdout)}") from error
        if result.get("typescript_version") != versions.get("typescript"):
            raise RuntimeError("TypeScript helper version changed after startup")
        result.pop("cache_hit", None)
        return result

    def analyze_semantics(
        self,
        *,
        operation: Literal["analyze", "scopes", "symbols", "references", "cfg"],
        project_folder: str,
        path: str,
        position: int | None,
        source_digest: str | None,
        include_declaration: bool = True,
        include_unresolved: bool = False,
        project_paths: Sequence[str] | None = None,
        include_globs: Sequence[str] | None = None,
        exclude_globs: Sequence[str] | None = None,
        function_position: int | None = None,
    ) -> JsonObject:
        helper = self.runtime.analysis_helper
        versions = self.runtime.analysis_versions
        if helper is None or versions is None:
            raise RuntimeError("Semantic analysis is not configured")
        project = self._resolve_project(project_folder)
        selected, errors = self._resolve_outline_paths(project, [path], None, None, strict_paths=True)
        if errors or len(selected) != 1:
            raise ValueError(f"Could not resolve semantic source: {path}")
        source_path = selected[0]
        source = source_path.read_text(encoding="utf-8")
        relative_source = source_path.relative_to(project).as_posix()
        request: JsonObject = {
            "operation": operation,
            "filename": relative_source,
            "source": source,
            "position": function_position if operation == "cfg" else position,
            "source_digest": source_digest,
        }
        if self.runtime.analysis_worker is not None:
            result = self.runtime.analysis_worker.request(request, timeout=self.runtime.command_timeout_seconds)
        else:
            completed = run_process(
                helper.command_prefix,
                timeout_seconds=self.runtime.command_timeout_seconds,
                input_text=json.dumps(request, separators=(",", ":")),
                working_directory=project,
                runner=self.runner,
                stdout_limit=MAX_STRUCTURED_OUTPUT_BYTES,
                stderr_limit=MAX_SUBPROCESS_DIAGNOSTIC_BYTES,
            )
            if _has_meaningful_diagnostic(completed.stderr):
                raise RuntimeError(f"Analysis helper emitted unexpected diagnostics: {_bounded_error_text(completed.stderr)}")
            try:
                result = JSON_OBJECT_ADAPTER.validate_python(
                    json.loads(completed.stdout, parse_constant=_reject_json_constant), strict=True
                )
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                raise RuntimeError(f"Analysis helper returned invalid JSON: {_bounded_error_text(completed.stdout)}") from error
        if result.get("worker_version") != versions.get("worker") or result.get("oxc_version") != versions.get("oxc"):
            raise RuntimeError("Analysis helper version changed after startup")
        digest = result.get("source_digest")
        diagnostics = result.get("diagnostics")
        if not isinstance(digest, str) or not isinstance(diagnostics, list):
            raise RuntimeError("Analysis helper returned an incomplete semantic result")
        output: JsonObject = {
            "file": relative_source,
            "source_digest": digest,
            "coverage": {"lexical": "same_file", "project": "not_available"},
            "diagnostics": JSON_VALUE_ADAPTER.validate_python(diagnostics, strict=True),
        }
        collection_key = {
            "scopes": "scopes",
            "symbols": "symbols",
            "references": "references",
            "cfg": "cfg",
            "analyze": "symbols",
        }[operation]
        collection = result.get(collection_key)
        if not isinstance(collection, list):
            raise RuntimeError(f"Analysis helper returned no {collection_key} collection")
        records = [JSON_OBJECT_ADAPTER.validate_python(record, strict=True) for record in collection]
        if operation == "scopes":
            output["scopes"] = JSON_VALUE_ADAPTER.validate_python(records, strict=True)
        elif operation == "symbols" or operation == "analyze":
            output["symbols"] = JSON_VALUE_ADAPTER.validate_python(records, strict=True)
        elif operation == "cfg":
            output["functions"] = JSON_VALUE_ADAPTER.validate_python(records, strict=True)
        else:
            references = records
            target_value = result.get("target")
            target = JSON_OBJECT_ADAPTER.validate_python(target_value, strict=True) if isinstance(target_value, dict) else None
            if include_declaration and target is not None and isinstance(target.get("symbol_id"), int):
                symbols_value = result.get("symbols")
                if isinstance(symbols_value, list):
                    for symbol_value in symbols_value:
                        symbol = JSON_OBJECT_ADAPTER.validate_python(symbol_value, strict=True)
                        if symbol.get("symbol_id") == target["symbol_id"]:
                            declaration = dict(symbol)
                            declaration["kind"] = "declaration"
                            references.insert(0, declaration)
                            break
            unresolved_value = result.get("unresolved")
            unresolved = (
                [JSON_OBJECT_ADAPTER.validate_python(record, strict=True) for record in unresolved_value]
                if include_unresolved and isinstance(unresolved_value, list)
                else []
            )
            output["target"] = target
            output["references"] = JSON_VALUE_ADAPTER.validate_python(references, strict=True)
            output["unresolved"] = JSON_VALUE_ADAPTER.validate_python(unresolved, strict=True)
            links: list[JsonObject] = []
            if self.runtime.oxc_helper is not None:
                requested_paths = list(project_paths or ())
                if relative_source not in requested_paths:
                    requested_paths.append(relative_source)
                modules = self.inspect_oxc_modules(
                    project_folder=str(project),
                    paths=requested_paths,
                    include_globs=include_globs,
                    exclude_globs=exclude_globs,
                    strict_paths=True,
                    include_dynamic=True,
                )
                for module in modules:
                    for edge in module["edges"]:
                        link = JSON_OBJECT_ADAPTER.validate_python(edge, strict=True)
                        link["importer"] = module["file"]
                        links.append(link)
                output["coverage"] = {"lexical": "same_file", "project": "module_graph"}
            output["module_graph_links"] = JSON_VALUE_ADAPTER.validate_python(links, strict=True)
        return output

    def inspect_oxc_modules(
        self,
        *,
        project_folder: str,
        paths: Sequence[str] | None,
        include_globs: Sequence[str] | None,
        exclude_globs: Sequence[str] | None,
        strict_paths: bool,
        include_dynamic: bool,
    ) -> list[JavascriptModule]:
        helper = self.runtime.oxc_helper
        expected_versions = self.runtime.oxc_versions
        if helper is None or expected_versions is None:
            raise RuntimeError("JavaScript module inspection is not configured")
        project = self._resolve_project(project_folder)
        resolved_paths, path_errors = self._resolve_outline_paths(
            project,
            paths,
            include_globs,
            exclude_globs,
            strict_paths=strict_paths,
        )
        modules = [self._oxc_error_module(error["path"], error["error"]) for error in path_errors]
        supported_paths: list[Path] = []
        for resolved in resolved_paths:
            if resolved.suffix.lower() in OXC_SOURCE_EXTENSIONS:
                supported_paths.append(resolved)
                continue
            message = f"Unsupported JavaScript or TypeScript extension: {resolved.relative_to(project).as_posix()}"
            if strict_paths:
                raise ValueError(message)
            modules.append(self._oxc_error_module(resolved.relative_to(project).as_posix(), message))
        if not supported_paths:
            modules.sort(key=lambda module: module["file"])
            return modules
        relative_paths = [path.relative_to(project).as_posix() for path in supported_paths]
        request = {
            "project_root": str(project),
            "files": relative_paths,
            "include_dynamic": include_dynamic,
        }
        if self.runtime.oxc_worker is not None:
            payload = self.runtime.oxc_worker.request(cast(JsonObject, request), timeout=self.runtime.command_timeout_seconds)
            response = OXC_SIDECAR_RESPONSE_ADAPTER.validate_python(payload, strict=True)
        else:
            completed = run_process(
                helper.command_prefix,
                timeout_seconds=self.runtime.command_timeout_seconds,
                input_text=json.dumps(request, separators=(",", ":")),
                working_directory=project,
                runner=self.runner,
                stdout_limit=MAX_STRUCTURED_OUTPUT_BYTES,
                stderr_limit=MAX_SUBPROCESS_DIAGNOSTIC_BYTES,
            )
            if _has_meaningful_diagnostic(completed.stderr):
                raise RuntimeError(f"Oxc helper emitted unexpected diagnostics: {_bounded_error_text(completed.stderr)}")
            try:
                payload = json.loads(completed.stdout, parse_constant=_reject_json_constant)
                response = OXC_SIDECAR_RESPONSE_ADAPTER.validate_python(payload, strict=True)
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                raise RuntimeError(f"Oxc helper returned invalid JSON: {_bounded_error_text(completed.stdout)}") from error
        if response["versions"] != expected_versions:
            raise RuntimeError(f"Oxc helper version changed after startup: {response['versions']!r}")
        expected_files = frozenset(relative_paths)
        returned_files: set[str] = set()
        for module in response["modules"]:
            normalized = self._normalize_oxc_module(module, project=project, expected_files=expected_files)
            if normalized["file"] in returned_files:
                raise RuntimeError(f"Oxc helper returned a duplicate module: {normalized['file']}")
            returned_files.add(normalized["file"])
            modules.append(normalized)
        if frozenset(returned_files) != expected_files:
            missing = sorted(expected_files - returned_files)
            raise RuntimeError(f"Oxc helper omitted selected modules: {', '.join(missing)}")
        modules.sort(key=lambda module: module["file"])
        return modules

    def _search(
        self,
        *,
        project_folder: str,
        rule_yaml: str,
        paths: Sequence[str] | None,
        include_globs: Sequence[str] | None,
        exclude_globs: Sequence[str] | None,
        max_results: int | None,
        include_metadata: bool = False,
        caller_supplied: bool = True,
        text_equals: str | None = None,
        text_starts_with: str | None = None,
        text_contains: str | None = None,
    ) -> SearchResults:
        validate_rule_yaml(rule_yaml, forbid_regex_rules=self.runtime.forbid_regex_rules, caller_supplied=caller_supplied)
        self._validate_text_filters(
            text_equals=text_equals,
            text_starts_with=text_starts_with,
            text_contains=text_contains,
        )
        project = self._resolve_project(project_folder)
        search_paths = self._resolve_paths(project, paths)
        limit = self._result_limit(max_results)
        arguments = [
            "--inline-rules",
            rule_yaml,
            "--json=stream",
            "--max-results",
            str(limit + 1),
            *(["--include-metadata"] if include_metadata else []),
            *self._glob_arguments(include_globs, exclude_globs),
            "--",
            *(str(path) for path in search_paths),
        ]
        parsed_matches, observed_extra = self._run_match_stream(
            "scan",
            arguments,
            working_directory=project,
            item_limit=limit,
            match_filter=lambda match: self._match_text_passes(
                match,
                text_equals=text_equals,
                text_starts_with=text_starts_with,
                text_contains=text_contains,
            ),
        )
        truncated = observed_extra or len(parsed_matches) > limit
        matches = [self._normalize_match_path(match, project) for match in parsed_matches[:limit]]
        return {
            "matches": matches,
            "returned": len(matches),
            "truncated": truncated,
            "limit": limit,
        }

    @classmethod
    def _trim_outline_nodes(
        cls,
        nodes: Sequence[JsonObject],
        remaining: int,
    ) -> tuple[list[JsonObject], int]:
        kept: list[JsonObject] = []
        consumed = 0
        for node in nodes:
            if consumed >= remaining:
                break
            normalized = dict(node)
            consumed += 1
            members = node.get("members")
            if isinstance(members, list):
                validated_members = [_runtime_mapping(member, field_name="outline member", record_number=0) for member in members]
                kept_members, member_count = cls._trim_outline_nodes(validated_members, remaining - consumed)
                normalized_members: list[JsonValue] = [member for member in kept_members]
                normalized["members"] = normalized_members
                consumed += member_count
            kept.append(normalized)
        return kept, consumed

    def _bound_outline(
        self,
        documents: Sequence[JsonObject],
        project: Path,
        requested_paths: Sequence[Path],
        limit: int,
        *,
        observed_extra: bool,
    ) -> OutlineResults:
        files: list[OutlineFile] = []
        remaining = limit
        returned = 0
        truncated = observed_extra
        requested = {path.relative_to(project).as_posix() for path in requested_paths}
        observed_paths: set[str] = set()
        for record_number, document in enumerate(documents, start=1):
            node_count = _validate_and_count_outline_document(document, record_number=record_number)
            items = document.get("items")
            raw_path = document.get("path")
            language = document.get("language")
            if not isinstance(items, list) or not isinstance(raw_path, str) or not isinstance(language, str):
                raise RuntimeError(f"ast-grep outline record {record_number} failed validation")
            normalized_path = self._contained_relative_path(raw_path, project, noun="an outline path")
            if normalized_path not in requested:
                raise RuntimeError(f"ast-grep returned an outline path that was not requested: {raw_path}")
            if normalized_path in observed_paths:
                raise RuntimeError(f"ast-grep returned a duplicate outline path: {raw_path}")
            observed_paths.add(normalized_path)
            validated_items = [_runtime_mapping(item, field_name="outline item", record_number=record_number) for item in items]
            kept, consumed = self._trim_outline_nodes(validated_items, remaining)
            if consumed < node_count:
                truncated = True
            if kept or not items:
                normalized_document = dict(document)
                normalized_document.pop("path", None)
                normalized_document["file"] = normalized_path
                normalized_items: list[JsonValue] = [item for item in kept]
                normalized_document["items"] = normalized_items
                files.append(cast(OutlineFile, normalized_document))
            returned += consumed
            remaining -= consumed
        return {
            "files": files,
            "returned": returned,
            "truncated": truncated,
            "limit": limit,
        }

    def outline_code(
        self,
        *,
        project_folder: str,
        paths: Sequence[str] | None = None,
        include_globs: Sequence[str] | None = None,
        exclude_globs: Sequence[str] | None = None,
        strict_paths: bool = True,
        language: str | None = None,
        max_results: int | None = None,
        items: OutlineItemsMode = "auto",
        symbol_types: Sequence[str] | None = None,
        public_members: bool = False,
    ) -> OutlineResults:
        if language is not None:
            self._require_language(language)
        project = self._resolve_project(project_folder)
        outline_paths, path_errors = self._resolve_outline_paths(
            project,
            paths,
            include_globs,
            exclude_globs,
            strict_paths=strict_paths,
        )
        limit = self._result_limit(max_results)
        if items not in {"auto", "structure", "exports", "imports", "all"}:
            raise ValueError(f"Unknown outline item mode: {items}")
        normalized_symbol_types: list[str] = []
        for symbol_type_value in cast(Sequence[object], symbol_types or ()):
            if not isinstance(symbol_type_value, str) or symbol_type_value not in _OUTLINE_SYMBOL_TYPES:
                raise ValueError(f"Unknown outline symbol type: {symbol_type_value!r}")
            symbol_type = symbol_type_value
            if symbol_type not in normalized_symbol_types:
                normalized_symbol_types.append(symbol_type)
        if not outline_paths:
            return {
                "files": [],
                "returned": 0,
                "truncated": False,
                "limit": limit,
                "resolved_paths": [],
                "path_errors": path_errors,
            }
        arguments = [
            "--json=stream",
            "--threads",
            "1",
            "--items",
            items,
            *(["--type", ",".join(normalized_symbol_types)] if normalized_symbol_types else []),
            *(["--pub-members"] if public_members else []),
            *(["--lang", language] if language is not None else []),
            "--",
            *(str(path) for path in outline_paths),
        ]
        with ExitStack() as stack:
            config_path = self._resolve_operation_config_path(stack)
            command = [
                *self.runtime.executable.command_prefix,
                "outline",
                "--config",
                str(config_path),
                *arguments,
            ]
            documents, observed_extra = self.outline_runner(
                command,
                timeout_seconds=self.runtime.command_timeout_seconds,
                working_directory=project,
                node_limit=limit,
            )
        results = self._bound_outline(documents, project, outline_paths, limit, observed_extra=observed_extra)
        results["resolved_paths"] = [path.relative_to(project).as_posix() for path in outline_paths]
        results["path_errors"] = path_errors
        return results

    def dump_syntax_tree(self, *, code: str, language: str, format: DumpFormat) -> str:
        self._require_language(language)
        _encode_bounded_input(code)
        with TemporaryDirectory(
            prefix="syntax-",
            dir=private_runtime_root(self.runtime.working_directory, self.runtime.allowed_roots),
        ) as sandbox:
            result = self._run(
                "run",
                ["--pattern", code, "--lang", language, f"--debug-query={format}"],
                working_directory=Path(sandbox),
                allow_no_matches=True,
                allow_stderr_on_no_matches=True,
                stderr_limit=MAX_STRUCTURED_OUTPUT_BYTES,
                truncate_stderr=False,
            )
        return result.stderr.strip()

    @staticmethod
    def _validate_text_filters(
        *,
        text_equals: str | None,
        text_starts_with: str | None,
        text_contains: str | None,
    ) -> None:
        filters = {
            "text_equals": text_equals,
            "text_starts_with": text_starts_with,
            "text_contains": text_contains,
        }
        for name, value in filters.items():
            if value is not None and (not value or "\0" in value):
                raise ValueError(f"{name} must be a non-empty NUL-free string when provided")

    @staticmethod
    def _match_text_passes(
        match: Mapping[str, JsonValue],
        *,
        text_equals: str | None,
        text_starts_with: str | None,
        text_contains: str | None,
    ) -> bool:
        text = match.get("text")
        return (
            isinstance(text, str)
            and (text_equals is None or text == text_equals)
            and (text_starts_with is None or text.startswith(text_starts_with))
            and (text_contains is None or text_contains in text)
        )

    def test_match_code_rule(
        self,
        *,
        code: str,
        rule_yaml: str,
        text_equals: str | None = None,
        text_starts_with: str | None = None,
        text_contains: str | None = None,
    ) -> list[JsonObject]:
        validate_rule_yaml(rule_yaml, forbid_regex_rules=self.runtime.forbid_regex_rules)
        self._validate_text_filters(
            text_equals=text_equals,
            text_starts_with=text_starts_with,
            text_contains=text_contains,
        )
        matches, observed_extra = self._run_match_stream(
            "scan",
            [
                "--inline-rules",
                rule_yaml,
                "--json=stream",
                "--max-results",
                str(self.runtime.max_results_cap + 1),
                "--stdin",
            ],
            working_directory=self.runtime.working_directory,
            item_limit=self.runtime.max_results_cap,
            input_text=code,
            match_filter=lambda match: self._match_text_passes(
                match,
                text_equals=text_equals,
                text_starts_with=text_starts_with,
                text_contains=text_contains,
            ),
        )
        if observed_extra:
            raise RuntimeError("ast-grep rule probe exceeded the configured result cap")
        return matches

    def find_code(
        self,
        *,
        project_folder: str,
        pattern: str,
        language: str,
        paths: Sequence[str] | None,
        include_globs: Sequence[str] | None,
        exclude_globs: Sequence[str] | None,
        max_results: int | None,
        selector: str | None = None,
        strictness: Strictness = "smart",
        rewrite: str | None = None,
    ) -> SearchResults:
        self._require_language(language)
        _encode_bounded_input(pattern)
        if selector is not None and (not selector or "\0" in selector):
            raise ValueError("selector must be a non-empty NUL-free ast-grep kind when provided")
        if strictness not in {"cst", "smart", "ast", "relaxed", "signature", "template"}:
            raise ValueError(f"Unknown ast-grep strictness: {strictness}")
        if rewrite is not None:
            _encode_bounded_input(rewrite)
        pattern_value: JsonValue
        if selector is None and strictness == "smart":
            pattern_value = pattern
        else:
            pattern_context: JsonObject = {"context": pattern, "strictness": strictness}
            if selector is not None:
                pattern_context["selector"] = selector
            pattern_value = pattern_context
        pattern_rule: JsonObject = {"pattern": pattern_value}
        rule_config: JsonObject = {
            "id": "mcp-pattern-search",
            "language": language,
            "rule": pattern_rule,
        }
        if rewrite is not None:
            rule_config["fix"] = rewrite
        rule_yaml = yaml_parser.safe_dump(rule_config, sort_keys=False)
        return self._search(
            project_folder=project_folder,
            rule_yaml=rule_yaml,
            paths=paths,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            max_results=max_results,
            caller_supplied=False,
        )

    def find_code_by_rule(
        self,
        *,
        project_folder: str,
        rule_yaml: str,
        paths: Sequence[str] | None,
        include_globs: Sequence[str] | None,
        exclude_globs: Sequence[str] | None,
        max_results: int | None,
        include_metadata: bool = False,
        text_equals: str | None = None,
        text_starts_with: str | None = None,
        text_contains: str | None = None,
    ) -> SearchResults:
        return self._search(
            project_folder=project_folder,
            rule_yaml=rule_yaml,
            paths=paths,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            max_results=max_results,
            include_metadata=include_metadata,
            text_equals=text_equals,
            text_starts_with=text_starts_with,
            text_contains=text_contains,
        )

    @staticmethod
    def _exact_rule_filter(
        rule_ids: Sequence[str] | None,
        *,
        configured_rule_ids: Sequence[str],
    ) -> str | None:
        if rule_ids is None:
            return None
        if not rule_ids:
            raise ValueError("rule_ids must contain at least one configured rule id when provided")
        configured = set(configured_rule_ids)
        if len(rule_ids) > len(configured):
            raise ValueError(f"rule_ids must not exceed the {len(configured)} configured rule ids")
        unique: list[str] = []
        seen: set[str] = set()
        for rule_id_value in cast(Sequence[object], rule_ids):
            if not isinstance(rule_id_value, str) or not rule_id_value or "\0" in rule_id_value:
                raise ValueError(f"Invalid configured rule id: {rule_id_value!r}")
            rule_id = rule_id_value
            if rule_id not in configured:
                raise ValueError(f"Unknown configured rule id: {rule_id}")
            if rule_id not in seen:
                seen.add(rule_id)
                unique.append(rule_id)
        return rf"^(?:{'|'.join(re.escape(rule_id) for rule_id in unique)})$"

    def scan_project_rules(
        self,
        *,
        project_folder: str,
        rule_ids: Sequence[str] | None,
        paths: Sequence[str] | None,
        include_globs: Sequence[str] | None,
        exclude_globs: Sequence[str] | None,
        max_results: int | None,
        include_metadata: bool = True,
    ) -> SearchResults:
        snapshot = self.runtime.config_snapshot
        if snapshot is None or not snapshot.capabilities["configured_scan"]:
            raise ValueError("No project rules were configured at server startup")
        project = self._resolve_project(project_folder)
        search_paths = self._resolve_paths(project, paths)
        limit = self._result_limit(max_results)
        rule_filter = self._exact_rule_filter(rule_ids, configured_rule_ids=snapshot.configured_rule_ids)
        arguments = [
            "--json=stream",
            "--max-results",
            str(limit + 1),
            *(["--filter", rule_filter] if rule_filter is not None else []),
            *(["--include-metadata"] if include_metadata else []),
            *self._glob_arguments(include_globs, exclude_globs),
            "--",
            *(str(path.relative_to(project)) for path in search_paths),
        ]
        parsed_matches, observed_extra = self._run_match_stream(
            "scan",
            arguments,
            working_directory=project,
            item_limit=limit,
            config_path_override=snapshot.project_config_path,
        )
        truncated = observed_extra or len(parsed_matches) > limit
        matches = [self._normalize_match_path(match, project) for match in parsed_matches[:limit]]
        return {
            "matches": matches,
            "returned": len(matches),
            "truncated": truncated,
            "limit": limit,
        }

    @staticmethod
    def _bounded_report(stdout: str, stderr: str) -> tuple[str, bool]:
        report = stdout.rstrip()
        diagnostic = stderr.rstrip()
        if diagnostic:
            report = f"{report}\n{diagnostic}" if report else diagnostic
        encoded = report.encode("utf-8")
        if len(encoded) <= MAX_TEST_REPORT_BYTES:
            return report, False
        marker = "…"
        decoded = _decode_utf8_prefix(encoded[: MAX_TEST_REPORT_BYTES - len(marker.encode("utf-8"))])
        return (decoded + marker, True) if decoded else (marker, True)

    def test_project_rules(self, *, rule_ids: Sequence[str] | None = None) -> ProjectTestResults:
        snapshot = self.runtime.config_snapshot
        if snapshot is None or not snapshot.capabilities["configured_tests"]:
            raise ValueError("No project rule tests were configured at server startup")
        rule_filter = self._exact_rule_filter(rule_ids, configured_rule_ids=snapshot.configured_rule_ids)
        command = [
            *self.runtime.executable.command_prefix,
            "test",
            "--config",
            str(snapshot.test_config_path),
            "--color",
            "never",
            *(["--filter", rule_filter] if rule_filter is not None else []),
        ]
        if self.runner is subprocess.run:
            streamed = run_text_process(
                command,
                timeout_seconds=self.runtime.command_timeout_seconds,
                working_directory=snapshot.bundle_root,
                stdout_limit=MAX_TEST_REPORT_BYTES,
                stderr_limit=MAX_TEST_REPORT_BYTES,
                truncate_stdout=True,
                truncate_stderr=True,
            )
            completed = streamed.completed
            stream_truncated = streamed.stdout_truncated or streamed.stderr_truncated
        else:
            completed = run_process(
                command,
                timeout_seconds=self.runtime.command_timeout_seconds,
                working_directory=snapshot.bundle_root,
                allowed_exit_codes=frozenset({0, 4}),
                runner=self.runner,
            )
            stream_truncated = False
        if completed.returncode not in {0, 4}:
            raise RuntimeError(
                f"Configured rule tests failed to execute with exit code {completed.returncode}: {_bounded_error_text(completed.stderr)}"
            )
        report, report_truncated = self._bounded_report(completed.stdout, completed.stderr)
        return {
            "passed": completed.returncode == 0,
            "report": report,
            "report_truncated": stream_truncated or report_truncated,
        }

    def get_server_info(self) -> ServerInfo:
        snapshot = self.runtime.config_snapshot
        posix_arg_max = _detected_posix_arg_max()
        capabilities = (
            dict(snapshot.capabilities)
            if snapshot is not None
            else {
                "inline_search": True,
                "outline": True,
                "configured_scan": False,
                "configured_tests": False,
                "custom_languages": False,
            }
        )
        capabilities["javascript_module_inspection"] = self.runtime.oxc_helper is not None
        capabilities["semantic_analysis"] = self.runtime.analysis_helper is not None
        capabilities["typescript_project_inspection"] = self.runtime.typescript_project_helper is not None
        capabilities["postgresql_parser"] = self.runtime.postgres_helper is not None
        capabilities["typescript_execution"] = self.runtime.typescript_execution_helper is not None
        provenance = JSON_OBJECT_ADAPTER.validate_python(
            snapshot.provenance
            if snapshot is not None
            else {"source": str(self.runtime.config_path) if self.runtime.config_path is not None else None, "snapshot": "test-runtime"},
            strict=True,
        )
        return {
            "fork_version": _server_version(),
            "ast_grep_executable": str(self.runtime.executable.path),
            "ast_grep_version": self.runtime.ast_grep_version,
            "oxc_helper_executable": str(self.runtime.oxc_helper.path) if self.runtime.oxc_helper is not None else None,
            "oxc_versions": self.runtime.oxc_versions,
            "analysis_helper_executable": str(self.runtime.analysis_helper.path) if self.runtime.analysis_helper is not None else None,
            "analysis_versions": self.runtime.analysis_versions,
            "typescript_project_helper_executable": (
                str(self.runtime.typescript_project_helper.path) if self.runtime.typescript_project_helper is not None else None
            ),
            "typescript_versions": self.runtime.typescript_versions,
            "postgres_helper_executable": str(self.runtime.postgres_helper.path) if self.runtime.postgres_helper is not None else None,
            "postgres_versions": self.runtime.postgres_versions,
            "typescript_execution_helper_executable": (
                str(self.runtime.typescript_execution_helper.path) if self.runtime.typescript_execution_helper is not None else None
            ),
            "typescript_execution_versions": self.runtime.typescript_execution_versions,
            "typescript_execution_profile": self.runtime.typescript_execution_profile,
            "config_path": str(self.runtime.config_path) if self.runtime.config_path is not None else None,
            "allowed_roots": [str(root) for root in self.runtime.allowed_roots],
            "command_timeout_seconds": self.runtime.command_timeout_seconds,
            "default_max_results": self.runtime.default_max_results,
            "max_results_cap": self.runtime.max_results_cap,
            "forbid_regex_rules": self.runtime.forbid_regex_rules,
            "configuration_digest": snapshot.digest if snapshot is not None else "unavailable",
            "configuration_provenance": provenance,
            "capabilities": capabilities,
            "coordinate_conventions": {
                "line": "zero-based",
                "column": "zero-based Unicode scalar count",
                "byte_offset": "zero-based UTF-8 bytes",
                "oxc_offset": "zero-based UTF-16 code units",
                "range": "half-open [start,end)",
            },
            "supported_language_ids": list(self._supported_language_ids()),
            "resource_limits": {
                "snippet_input_bytes": MAX_SNIPPET_INPUT_BYTES,
                "inline_rule_bytes": MAX_INLINE_RULE_BYTES,
                "ndjson_record_bytes": MAX_NDJSON_RECORD_BYTES,
                "structured_output_bytes": MAX_STRUCTURED_OUTPUT_BYTES,
                "diagnostic_bytes": MAX_SUBPROCESS_DIAGNOSTIC_BYTES,
                "test_report_bytes": MAX_TEST_REPORT_BYTES,
                "config_file_bytes": MAX_CONFIG_FILE_BYTES,
                "config_resource_bytes": MAX_CONFIG_RESOURCE_BYTES,
                "config_resource_files": MAX_CONFIG_RESOURCE_FILES,
                "native_library_bytes": MAX_NATIVE_LIBRARY_BYTES,
                "yaml_documents": MAX_YAML_DOCUMENTS,
                "yaml_nodes": MAX_YAML_NODES,
                "yaml_depth": MAX_YAML_DEPTH,
                "outline_paths": MAX_OUTLINE_PATHS,
                "oxc_files": MAX_OXC_FILES,
                "oxc_file_bytes": MAX_OXC_FILE_BYTES,
                "oxc_total_source_bytes": MAX_OXC_TOTAL_SOURCE_BYTES,
                "hard_max_results": HARD_MAX_RESULTS,
                "command_timeout_seconds": self.runtime.command_timeout_seconds,
                "windows_create_process_characters": WINDOWS_CREATE_PROCESS_LIMIT,
                "posix_arg_max_bytes": posix_arg_max,
                "posix_arg_headroom_bytes": POSIX_ARG_HEADROOM_BYTES,
                "posix_effective_launch_budget_bytes": (posix_arg_max - POSIX_ARG_HEADROOM_BYTES if posix_arg_max else 0),
                "process_termination_grace_seconds": PROCESS_TERMINATION_GRACE_SECONDS,
            },
        }


STRING_OBJECT_DICT_ADAPTER: Final = TypeAdapter(dict[str, object])


def string_object_dict(value: object) -> dict[str, object] | None:
    try:
        return STRING_OBJECT_DICT_ADAPTER.validate_python(value, strict=True)
    except ValidationError:
        return None


def json_strings(values: Sequence[str] | None) -> list[JsonValue] | None:
    if values is None:
        return None
    return [value for value in values]


def query_digest(tool_name: str, payload: Mapping[str, JsonValue]) -> str:
    canonical = json.dumps(
        {"tool": tool_name, "query": payload},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def paged_search_results(
    *,
    runtime: RuntimeServices,
    tool_name: str,
    query: Mapping[str, JsonValue],
    max_results: int | None,
    cursor: str | None,
    execute: Callable[[int], SearchResults],
) -> SearchResults:
    page_size = runtime.default_max_results if max_results is None else max_results
    if not 1 <= page_size <= runtime.max_results_cap:
        raise ValueError(f"max_results must be between 1 and {runtime.max_results_cap}")
    digest = query_digest(tool_name, query)
    if cursor is not None:
        return runtime.cursor_store.next_search_page(
            cursor=cursor,
            query_digest=digest,
            page_size=page_size,
        )
    snapshot = execute(runtime.max_results_cap)
    return runtime.cursor_store.first_search_page(
        query_digest=digest,
        matches=snapshot["matches"],
        page_size=page_size,
        source_truncated=snapshot["truncated"],
    )


def paged_javascript_module_results(
    *,
    runtime: RuntimeServices,
    query: Mapping[str, JsonValue],
    max_results: int | None,
    cursor: str | None,
    execute: Callable[[], list[JavascriptModule]],
) -> JavascriptModuleResults:
    page_size = runtime.default_max_results if max_results is None else max_results
    if not 1 <= page_size <= runtime.max_results_cap:
        raise ValueError(f"max_results must be between 1 and {runtime.max_results_cap}")
    digest = query_digest("oxc_modules", query)
    if cursor is not None:
        page = runtime.cursor_store.next_search_page(cursor=cursor, query_digest=digest, page_size=page_size)
    else:
        modules = execute()
        module_records = [JSON_OBJECT_ADAPTER.validate_python(module, strict=True) for module in modules]
        project_folder = query.get("project_folder")
        if not isinstance(project_folder, str):
            raise RuntimeError("Oxc module query has no project folder")
        project = Path(project_folder).resolve(strict=True)
        source_hasher = hashlib.sha256()
        for module in sorted(modules, key=lambda item: item["file"]):
            source_hasher.update(module["file"].encode("utf-8"))
            source_path = (project / module["file"]).resolve(strict=False)
            if source_path.is_file() and is_within(source_path, project):
                source_hasher.update(source_path.read_bytes())
            else:
                source_hasher.update(json.dumps(module, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        diagnostics = [diagnostic for module in modules for diagnostic in module["diagnostics"]]
        metadata = JSON_OBJECT_ADAPTER.validate_python(
            {"source_digest": source_hasher.hexdigest(), "diagnostics": diagnostics}, strict=True
        )
        page = runtime.cursor_store.first_search_page(
            query_digest=digest,
            matches=module_records,
            page_size=page_size,
            source_truncated=False,
            metadata=metadata,
        )
    metadata_value = page.get("metadata")
    if not isinstance(metadata_value, dict):
        raise RuntimeError("Oxc module cursor has no snapshot metadata")
    source_digest = metadata_value.get("source_digest")
    if not isinstance(source_digest, str):
        raise RuntimeError("Oxc module cursor has no source digest")
    return {
        "modules": JAVASCRIPT_MODULE_LIST_ADAPTER.validate_python(page["matches"], strict=True),
        "returned": page["returned"],
        "truncated": page["truncated"],
        "limit": page["limit"],
        "next_cursor": page.get("next_cursor"),
        "snapshot_truncated": page.get("snapshot_truncated", False),
        "source_digest": source_digest,
        "diagnostics": OXC_DIAGNOSTIC_LIST_ADAPTER.validate_python(metadata_value.get("diagnostics", []), strict=True),
    }


def paged_postgres_file_results(
    *,
    runtime: RuntimeServices,
    query: Mapping[str, JsonValue],
    max_results: int | None,
    cursor: str | None,
    execute: Callable[[], PostgresFilesResults],
) -> PostgresFilesResults:
    page_size = runtime.default_max_results if max_results is None else max_results
    if not 1 <= page_size <= runtime.max_results_cap:
        raise ValueError(f"max_results must be between 1 and {runtime.max_results_cap}")
    digest = query_digest("postgres_parse_files", query)
    if cursor is not None:
        page = runtime.cursor_store.next_search_page(cursor=cursor, query_digest=digest, page_size=page_size)
    else:
        snapshot = execute()
        page = runtime.cursor_store.first_search_page(
            query_digest=digest,
            matches=snapshot["files"],
            page_size=page_size,
            source_truncated=snapshot["truncated"],
        )
    return {
        "files": page["matches"],
        "returned": page["returned"],
        "truncated": page["truncated"],
        "limit": page["limit"],
        "next_cursor": page.get("next_cursor"),
        "snapshot_truncated": page.get("snapshot_truncated", False),
    }


def paged_semantic_results(
    *,
    runtime: RuntimeServices,
    tool_name: str,
    query: Mapping[str, JsonValue],
    max_results: int | None,
    cursor: str | None,
    record_keys: Sequence[str],
    execute: Callable[[], JsonObject],
) -> JsonObject:
    page_size = runtime.default_max_results if max_results is None else max_results
    if not 1 <= page_size <= runtime.max_results_cap:
        raise ValueError(f"max_results must be between 1 and {runtime.max_results_cap}")
    digest = query_digest(tool_name, query)
    if cursor is not None:
        page = runtime.cursor_store.next_search_page(cursor=cursor, query_digest=digest, page_size=page_size)
    else:
        result = execute()
        records: list[JsonObject] = []
        metadata = dict(result)
        for key in record_keys:
            value = metadata.pop(key, [])
            if not isinstance(value, list):
                raise RuntimeError(f"Semantic result {key} is not a list")
            for record_value in value:
                record = JSON_OBJECT_ADAPTER.validate_python(record_value, strict=True)
                records.append({"record_kind": key, "record": record})
        page = runtime.cursor_store.first_search_page(
            query_digest=digest,
            matches=records,
            page_size=page_size,
            source_truncated=False,
            metadata=metadata,
        )
    metadata_value = page.get("metadata", {})
    result_page = JSON_OBJECT_ADAPTER.validate_python(metadata_value, strict=True)
    collected: dict[str, list[JsonObject]] = {key: [] for key in record_keys}
    for wrapper in page["matches"]:
        cursor_key = wrapper.get("record_kind")
        record_value = wrapper.get("record")
        if not isinstance(cursor_key, str) or cursor_key not in collected:
            raise RuntimeError("Semantic cursor contains an invalid record kind")
        collected[cursor_key].append(JSON_OBJECT_ADAPTER.validate_python(record_value, strict=True))
    for key, records in collected.items():
        result_page[key] = JSON_VALUE_ADAPTER.validate_python(records, strict=True)
    result_page["returned"] = page["returned"]
    result_page["truncated"] = page["truncated"]
    result_page["limit"] = page["limit"]
    result_page["next_cursor"] = page.get("next_cursor")
    return result_page


SERVER_INSTRUCTIONS: Final = (
    "Capability-gated structural, Oxc, semantic, TypeScript compiler, PostgreSQL parser, mutation, and sandbox tools. "
    "Use each parser or compiler only for the evidence it owns, inspect get_server_info before optional families, "
    "and continue opaque cursors. "
    "Preview tools never write; file mutation and execution tools are separate and require explicit authorization."
)


def register_structural_tools(server: FastMCP, runtime: RuntimeServices) -> None:
    service = AstGrepService(runtime)

    @server.tool(annotations=READ_ONLY_ANNOTATIONS, title="Dump syntax tree")
    def dump_syntax_tree(
        code: Annotated[str, Field(description="Code or pattern to inspect")],
        language: Annotated[str, Field(description="Explicit ast-grep language")],
        format: Annotated[
            DumpFormat,
            Field(description="Syntax dump format: pattern, ast, cst, or sexp"),
        ] = "cst",
    ) -> str:
        """Inspect how ast-grep parses code or a query pattern."""
        return service.dump_syntax_tree(code=code, language=language, format=format)

    @server.tool(annotations=READ_ONLY_ANNOTATIONS, title="Test rule against code")
    def test_match_code_rule(
        code: Annotated[str, Field(description="Code to test against the rule")],
        yaml: Annotated[str, Field(description="ast-grep YAML with id, language, and rule fields")],
        text_equals: Annotated[
            str | None,
            Field(description="Keep matches whose complete node text equals this literal"),
        ] = None,
        text_starts_with: Annotated[
            str | None,
            Field(description="Keep matches whose complete node text starts with this literal"),
        ] = None,
        text_contains: Annotated[
            str | None,
            Field(description="Keep matches whose complete node text contains this literal"),
        ] = None,
    ) -> list[JsonObject]:
        """Probe an ast-grep YAML rule against code and optionally filter matched node text by literals."""
        return service.test_match_code_rule(
            code=code,
            rule_yaml=yaml,
            text_equals=text_equals,
            text_starts_with=text_starts_with,
            text_contains=text_contains,
        )

    @server.tool(
        annotations=READ_ONLY_ANNOTATIONS,
        title="Outline code",
        output_schema=OUTLINE_RESULTS_OUTPUT_SCHEMA,
    )
    def outline_code(
        project_folder: Annotated[
            str,
            Field(description="Project directory, resolved and constrained to the server's allowed roots"),
        ],
        paths: Annotated[
            list[str] | None,
            Field(
                min_length=1,
                max_length=MAX_OUTLINE_PATHS,
                description="Optional exact relative files; missing files are reported without discarding valid files",
            ),
        ] = None,
        include_globs: Annotated[
            list[str] | None,
            Field(description="Optional contained glob patterns that resolve exact files before outlining"),
        ] = None,
        exclude_globs: Annotated[
            list[str] | None,
            Field(description="Optional glob patterns excluded from include_globs results"),
        ] = None,
        strict_paths: Annotated[
            bool,
            Field(description="Fail the entire request when an exact path is missing or not a regular file"),
        ] = False,
        language: Annotated[
            str | None,
            Field(description="Optional explicit ast-grep language; omitted to use extension-based detection"),
        ] = None,
        max_results: Annotated[
            int | None,
            Field(
                ge=1,
                le=HARD_MAX_RESULTS,
                description=(
                    "Finite symbol limit across all files; defaults to the server's configured limit. This schema "
                    "bound is the hard ceiling: an operator may configure a lower cap, which get_server_info reports "
                    "as max_results_cap and which rejects larger values at call time."
                ),
            ),
        ] = None,
        items: Annotated[
            OutlineItemsMode,
            Field(description="Top-level item mode: auto, structure, exports, imports, or all"),
        ] = "auto",
        symbol_types: Annotated[
            list[str] | None,
            Field(description="Optional exact top-level outline symbol types such as class, enum, or function"),
        ] = None,
        public_members: Annotated[
            bool,
            Field(description="Retain only public members in member-bearing outline views"),
        ] = False,
        output_format: Annotated[
            OutputFormat,
            Field(description="Compact text or structured JSON result"),
        ] = "text",
    ) -> Annotated[CallToolResult, OutlineResults]:
        """Resolve exact files and extract a bounded per-file symbol hierarchy."""
        results = service.outline_code(
            project_folder=project_folder,
            paths=paths,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            strict_paths=strict_paths,
            language=language,
            max_results=max_results,
            items=items,
            symbol_types=symbol_types,
            public_members=public_members,
        )
        return outline_tool_result(results, output_format)

    @server.tool(
        annotations=READ_ONLY_ANNOTATIONS,
        title="Find code by pattern",
        output_schema=SEARCH_RESULTS_OUTPUT_SCHEMA,
    )
    def find_code(
        project_folder: Annotated[
            str,
            Field(description="Project directory, resolved and constrained to the server's allowed roots"),
        ],
        pattern: Annotated[str, Field(description="Valid ast-grep structural pattern")],
        language: Annotated[str, Field(description="Required ast-grep language")],
        selector: Annotated[
            str | None,
            Field(description="Optional AST kind selecting the matched sub-part of the pattern context"),
        ] = None,
        strictness: Annotated[
            Strictness,
            Field(description="ast-grep pattern strictness"),
        ] = "smart",
        rewrite: Annotated[
            str | None,
            Field(description="Optional preview-only replacement; an empty string previews deletion"),
        ] = None,
        paths: Annotated[
            list[str] | None,
            Field(description="Relative files or directories under project_folder; defaults to the project root"),
        ] = None,
        include_globs: Annotated[
            list[str] | None,
            Field(description="Optional gitignore-style include globs"),
        ] = None,
        exclude_globs: Annotated[
            list[str] | None,
            Field(description="Optional gitignore-style exclude globs, without a leading !"),
        ] = None,
        max_results: Annotated[
            int | None,
            Field(
                ge=1,
                le=HARD_MAX_RESULTS,
                description=(
                    "Finite result limit; defaults to the server's configured limit. This schema bound is the hard "
                    "ceiling: an operator may configure a lower cap, which get_server_info reports as max_results_cap "
                    "and which rejects larger values at call time."
                ),
            ),
        ] = None,
        cursor: Annotated[
            str | None,
            Field(description="Opaque continuation cursor from the previous identical query"),
        ] = None,
        output_format: Annotated[
            OutputFormat,
            Field(description="Compact text or structured JSON result"),
        ] = "text",
    ) -> Annotated[CallToolResult, SearchResults]:
        """Find bounded structural pattern matches inside an allowed project scope."""
        query: JsonObject = {
            "project_folder": project_folder,
            "pattern": pattern,
            "language": language,
            "selector": selector,
            "strictness": strictness,
            "rewrite": rewrite,
            "paths": json_strings(paths),
            "include_globs": json_strings(include_globs),
            "exclude_globs": json_strings(exclude_globs),
        }
        try:
            results = paged_search_results(
                runtime=runtime,
                tool_name="find_code",
                query=query,
                max_results=max_results,
                cursor=cursor,
                execute=lambda snapshot_limit: service.find_code(
                    project_folder=project_folder,
                    pattern=pattern,
                    language=language,
                    paths=paths,
                    include_globs=include_globs,
                    exclude_globs=exclude_globs,
                    max_results=snapshot_limit,
                    selector=selector,
                    strictness=strictness,
                    rewrite=rewrite,
                ),
            )
        except ValueError as error:
            raise ToolError(str(error)) from error
        except RuntimeError as error:
            results = pattern_failure_result(
                service=service,
                error=error,
                pattern=pattern,
                language=language,
                limit=max_results or runtime.default_max_results,
            )
        return search_tool_result(results, output_format)

    @server.tool(
        annotations=READ_ONLY_ANNOTATIONS,
        title="Find code by rule",
        output_schema=SEARCH_RESULTS_OUTPUT_SCHEMA,
    )
    def find_code_by_rule(
        project_folder: Annotated[
            str,
            Field(description="Project directory, resolved and constrained to the server's allowed roots"),
        ],
        yaml: Annotated[str, Field(description="ast-grep YAML with id, language, and rule fields")],
        paths: Annotated[
            list[str] | None,
            Field(description="Relative files or directories under project_folder; defaults to the project root"),
        ] = None,
        include_globs: Annotated[
            list[str] | None,
            Field(description="Optional gitignore-style include globs"),
        ] = None,
        exclude_globs: Annotated[
            list[str] | None,
            Field(description="Optional gitignore-style exclude globs, without a leading !"),
        ] = None,
        max_results: Annotated[
            int | None,
            Field(
                ge=1,
                le=HARD_MAX_RESULTS,
                description=(
                    "Finite result limit; defaults to the server's configured limit. This schema bound is the hard "
                    "ceiling: an operator may configure a lower cap, which get_server_info reports as max_results_cap "
                    "and which rejects larger values at call time."
                ),
            ),
        ] = None,
        include_metadata: Annotated[
            bool,
            Field(description="Include each rule's documented metadata object in ast-grep match records"),
        ] = False,
        positive_code: Annotated[
            str | None,
            Field(description="Optional code that must match before the project scan runs"),
        ] = None,
        negative_code: Annotated[
            str | None,
            Field(description="Optional code that must produce no matches before the project scan runs"),
        ] = None,
        text_equals: Annotated[
            str | None,
            Field(description="Keep matches whose complete node text equals this literal"),
        ] = None,
        text_starts_with: Annotated[
            str | None,
            Field(description="Keep matches whose complete node text starts with this literal"),
        ] = None,
        text_contains: Annotated[
            str | None,
            Field(description="Keep matches whose complete node text contains this literal"),
        ] = None,
        cursor: Annotated[
            str | None,
            Field(description="Opaque continuation cursor from the previous identical query"),
        ] = None,
        output_format: Annotated[
            OutputFormat,
            Field(description="Compact text or structured JSON result"),
        ] = "text",
    ) -> Annotated[CallToolResult, SearchResults]:
        """Find bounded matches for one or more validated ast-grep YAML rules."""
        try:
            validate_rule_yaml(yaml, forbid_regex_rules=runtime.forbid_regex_rules)
        except ValueError as error:
            raise ToolError(str(error)) from error
        query: JsonObject = {
            "project_folder": project_folder,
            "yaml": yaml,
            "paths": json_strings(paths),
            "include_globs": json_strings(include_globs),
            "exclude_globs": json_strings(exclude_globs),
            "include_metadata": include_metadata,
            "positive_code": positive_code,
            "negative_code": negative_code,
            "text_equals": text_equals,
            "text_starts_with": text_starts_with,
            "text_contains": text_contains,
        }
        page_limit = max_results or runtime.default_max_results
        if cursor is None and positive_code is not None:
            positive_matches = service.test_match_code_rule(
                code=positive_code,
                rule_yaml=yaml,
                text_equals=text_equals,
                text_starts_with=text_starts_with,
                text_contains=text_contains,
            )
            if not positive_matches:
                results: SearchResults = {
                    "matches": [],
                    "returned": 0,
                    "truncated": False,
                    "limit": page_limit,
                    "diagnostics": {
                        "kind": "positive_probe_failed",
                        "message": "positive_code produced no matches; project scan skipped",
                    },
                }
                return search_tool_result(results, output_format)
        if cursor is None and negative_code is not None:
            negative_matches = service.test_match_code_rule(
                code=negative_code,
                rule_yaml=yaml,
                text_equals=text_equals,
                text_starts_with=text_starts_with,
                text_contains=text_contains,
            )
            if negative_matches:
                results = {
                    "matches": [],
                    "returned": 0,
                    "truncated": False,
                    "limit": page_limit,
                    "diagnostics": {
                        "kind": "negative_probe_failed",
                        "message": f"negative_code produced {len(negative_matches)} match(es); project scan skipped",
                    },
                }
                return search_tool_result(results, output_format)
        results = paged_search_results(
            runtime=runtime,
            tool_name="find_code_by_rule",
            query=query,
            max_results=max_results,
            cursor=cursor,
            execute=lambda snapshot_limit: service.find_code_by_rule(
                project_folder=project_folder,
                rule_yaml=yaml,
                paths=paths,
                include_globs=include_globs,
                exclude_globs=exclude_globs,
                max_results=snapshot_limit,
                include_metadata=include_metadata,
                text_equals=text_equals,
                text_starts_with=text_starts_with,
                text_contains=text_contains,
            ),
        )
        return search_tool_result(results, output_format)

    @server.tool(
        annotations=READ_ONLY_ANNOTATIONS,
        title="Scan configured project rules",
        output_schema=SEARCH_RESULTS_OUTPUT_SCHEMA,
    )
    def scan_project_rules(
        project_folder: Annotated[
            str,
            Field(description="Project directory, resolved and constrained to the server's allowed roots"),
        ],
        rule_ids: Annotated[
            list[str] | None,
            Field(description="Optional exact startup-configured rule IDs; regex metacharacters are escaped"),
        ] = None,
        paths: Annotated[
            list[str] | None,
            Field(description="Relative files or directories under project_folder; defaults to the project root"),
        ] = None,
        include_globs: Annotated[
            list[str] | None,
            Field(description="Optional gitignore-style include globs"),
        ] = None,
        exclude_globs: Annotated[
            list[str] | None,
            Field(description="Optional gitignore-style exclude globs, without a leading !"),
        ] = None,
        max_results: Annotated[
            int | None,
            Field(
                ge=1,
                le=HARD_MAX_RESULTS,
                description="Finite result limit; defaults to the server limit and may not exceed its effective cap",
            ),
        ] = None,
        include_metadata: Annotated[
            bool,
            Field(description="Include configured rule metadata in match records"),
        ] = True,
        run_tests_first: Annotated[
            bool,
            Field(description="Run configured rule tests first and skip the scan when they fail"),
        ] = False,
        cursor: Annotated[
            str | None,
            Field(description="Opaque continuation cursor from the previous identical query"),
        ] = None,
        output_format: Annotated[
            OutputFormat,
            Field(description="Compact text or structured JSON result"),
        ] = "text",
    ) -> Annotated[CallToolResult, SearchResults]:
        """Run only the immutable rule set captured from startup configuration."""
        query: JsonObject = {
            "project_folder": project_folder,
            "rule_ids": json_strings(rule_ids),
            "paths": json_strings(paths),
            "include_globs": json_strings(include_globs),
            "exclude_globs": json_strings(exclude_globs),
            "include_metadata": include_metadata,
            "run_tests_first": run_tests_first,
        }
        if cursor is None and run_tests_first:
            test_result = service.test_project_rules(rule_ids=rule_ids)
            if not test_result["passed"]:
                results: SearchResults = {
                    "matches": [],
                    "returned": 0,
                    "truncated": False,
                    "limit": max_results or runtime.default_max_results,
                    "diagnostics": {
                        "kind": "configured_tests_failed",
                        "message": "configured rule tests failed; project scan skipped",
                        "report": test_result["report"],
                        "report_truncated": test_result["report_truncated"],
                    },
                }
                return search_tool_result(results, output_format)
        results = paged_search_results(
            runtime=runtime,
            tool_name="scan_project_rules",
            query=query,
            max_results=max_results,
            cursor=cursor,
            execute=lambda snapshot_limit: service.scan_project_rules(
                project_folder=project_folder,
                rule_ids=rule_ids,
                paths=paths,
                include_globs=include_globs,
                exclude_globs=exclude_globs,
                max_results=snapshot_limit,
                include_metadata=include_metadata,
            ),
        )
        return search_tool_result(results, output_format)

    @server.tool(annotations=READ_ONLY_ANNOTATIONS, title="Test configured project rules")
    def test_project_rules(
        rule_ids: Annotated[
            list[str] | None,
            Field(description="Optional exact startup-configured rule IDs; regex metacharacters are escaped"),
        ] = None,
    ) -> ProjectTestResults:
        """Run the immutable configured test suites without interaction or snapshot updates."""
        return service.test_project_rules(rule_ids=rule_ids)

    @server.tool(annotations=READ_ONLY_ANNOTATIONS, title="Get server info")
    def get_server_info() -> PublicServerInfo:
        """Report the fork, executable, containment, configuration, timeout, and result-limit contract."""
        return public_server_info(service.get_server_info())


def oxc_preview_result(
    *,
    operation: Literal["transform", "minify"],
    filename: str,
    source: str,
    raw: JsonObject,
) -> JsonObject:
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    diagnostics = JSON_VALUE_ADAPTER.validate_python(raw.get("diagnostics", []), strict=True)
    code_value = raw.get("code")
    if not isinstance(code_value, str):
        raise RuntimeError(f"Oxc {operation} returned no code")
    if operation == "transform":
        return {
            "filename": filename,
            "code": code_value,
            "source_map": raw.get("map"),
            "declaration": raw.get("declaration"),
            "declaration_map": raw.get("declaration_map"),
            "helpers_used": raw.get("helpers_used"),
            "source_digest": digest,
            "diagnostics": diagnostics,
        }
    return {
        "filename": filename,
        "code": code_value,
        "source_map": raw.get("map"),
        "legal_comments": raw.get("legal_comments", []),
        "mangle_cache": raw.get("mangle_cache"),
        "source_digest": digest,
        "diagnostics": diagnostics,
        "assumptions": "Oxc minifier assumptions apply; verify output against project semantics.",
    }


def oxc_mutation_result(operation: Literal["transform", "minify"], raw: JsonObject) -> JsonObject:
    applied_value = raw.get("applied", [])
    skipped_value = raw.get("skipped", [])
    if not isinstance(applied_value, list) or not isinstance(skipped_value, list):
        raise RuntimeError("Oxc mutation returned an invalid batch result")
    applied = [JSON_OBJECT_ADAPTER.validate_python(item, strict=True) for item in applied_value]
    skipped = [item for item in skipped_value if isinstance(item, str)]
    diagnostics = JSON_VALUE_ADAPTER.validate_python(raw.get("diagnostics", []), strict=True)
    emitted_files = [item["target"] for item in applied if item.get("changed") is True]
    unchanged = [item["target"] for item in applied if item.get("changed") is False]
    skipped_files = [*unchanged, *skipped]
    overwritten_files = [item["target"] for item in applied if item.get("overwritten") is True]
    result: JsonObject = {
        "emitted_files": emitted_files,
        "skipped_files": skipped_files,
        "overwritten_files": overwritten_files,
        "failed_files": [],
        "artifacts": JSON_VALUE_ADAPTER.validate_python(raw.get("artifacts", []), strict=True),
        "diagnostics": diagnostics,
    }
    if operation == "minify":
        result["legal_comments"] = JSON_VALUE_ADAPTER.validate_python(raw.get("legal_comments", []), strict=True)
        result["assumptions"] = "Oxc minifier assumptions apply; verify output against project semantics."
    return result


def public_server_info(info: ServerInfo) -> PublicServerInfo:
    payload: object = {
        "server": {"name": "ast-soleaux", "version": info["fork_version"]},
        "versions": {
            "ast_grep": info["ast_grep_version"],
            "oxc": info["oxc_versions"],
            "analysis": info["analysis_versions"],
            "typescript": info["typescript_versions"],
            "postgresql": info["postgres_versions"],
            "typescript_execution": info["typescript_execution_versions"],
        },
        "executables": {
            "ast_grep": info["ast_grep_executable"],
            "oxc": info["oxc_helper_executable"],
            "analysis": info["analysis_helper_executable"],
            "typescript": info["typescript_project_helper_executable"],
            "postgresql": info["postgres_helper_executable"],
            "typescript_execution": info["typescript_execution_helper_executable"],
        },
        "allowed_roots": info["allowed_roots"],
        "capabilities": info["capabilities"],
        "limits": {
            **info["resource_limits"],
            "default_max_results": info["default_max_results"],
            "max_results_cap": info["max_results_cap"],
        },
        "coordinates": info["coordinate_conventions"],
        "configuration": {
            "path": info["config_path"],
            "digest": info["configuration_digest"],
            "provenance": info["configuration_provenance"],
            "forbid_regex_rules": info["forbid_regex_rules"],
            "execution_profile": info["typescript_execution_profile"],
            "supported_language_ids": info["supported_language_ids"],
        },
    }
    return PUBLIC_SERVER_INFO_ADAPTER.validate_python(payload, strict=True)


def register_oxc_tools(server: FastMCP, runtime: RuntimeServices) -> None:
    service = AstGrepService(runtime)

    @server.tool(
        annotations=READ_ONLY_ANNOTATIONS,
        title="Inspect Oxc modules",
        output_schema=JAVASCRIPT_MODULE_RESULTS_OUTPUT_SCHEMA,
    )
    def oxc_modules(
        project_folder: Annotated[
            str,
            Field(description="Project directory, resolved and constrained to the server's allowed roots"),
        ],
        paths: Annotated[
            list[str] | None,
            Field(
                min_length=1,
                max_length=MAX_OXC_FILES,
                description="Optional exact JavaScript or TypeScript files relative to project_folder",
            ),
        ] = None,
        include_globs: Annotated[
            list[str] | None,
            Field(description="Optional contained glob patterns resolving JavaScript or TypeScript files"),
        ] = None,
        exclude_globs: Annotated[
            list[str] | None,
            Field(description="Optional glob patterns excluded from include_globs results"),
        ] = None,
        strict_paths: Annotated[
            bool,
            Field(description="Fail the entire request when an exact path is missing, unsupported, or not a regular file"),
        ] = False,
        include_dynamic: Annotated[
            bool,
            Field(description="Include dynamic import expressions without resolving non-static expressions"),
        ] = False,
        max_results: Annotated[
            int | None,
            Field(
                ge=1,
                le=HARD_MAX_RESULTS,
                description="Finite module limit; defaults to the server limit and may not exceed its effective cap",
            ),
        ] = None,
        cursor: Annotated[
            str | None,
            Field(description="Opaque continuation cursor from the previous identical query"),
        ] = None,
        output_format: Annotated[
            OutputFormat,
            Field(description="Compact text or structured JSON result"),
        ] = "text",
    ) -> Annotated[CallToolResult, JavascriptModuleResults]:
        """Parse JavaScript and TypeScript modules and resolve their contained static dependency edges with Oxc."""
        query: JsonObject = {
            "project_folder": project_folder,
            "paths": json_strings(paths),
            "include_globs": json_strings(include_globs),
            "exclude_globs": json_strings(exclude_globs),
            "strict_paths": strict_paths,
            "include_dynamic": include_dynamic,
        }
        results = paged_javascript_module_results(
            runtime=runtime,
            query=query,
            max_results=max_results,
            cursor=cursor,
            execute=lambda: service.inspect_oxc_modules(
                project_folder=project_folder,
                paths=paths,
                include_globs=include_globs,
                exclude_globs=exclude_globs,
                strict_paths=strict_paths,
                include_dynamic=include_dynamic,
            ),
        )
        return javascript_module_tool_result(results, output_format)

    @server.tool(
        annotations=READ_ONLY_ANNOTATIONS,
        title="Transform code with Oxc",
        tags={"oxc", "transform", "preview", "read-only"},
        timeout=30.0,
        output_schema=JSON_OBJECT_OUTPUT_SCHEMA,
    )
    def oxc_transform(
        source: OxcSource,
        options: OxcTransformOptions | None = None,
        output_format: OutputFormat = "text",
    ) -> CallToolResult:
        """Return transformed code, maps, declarations, helpers, and diagnostics without writing files."""
        filename, code, _project = service.resolve_oxc_source(source)
        raw = service.transform_code(filename=filename, code=code, options=options)
        return json_object_format_tool_result(
            oxc_preview_result(operation="transform", filename=filename, source=code, raw=raw), output_format
        )

    @server.tool(
        annotations=MUTATING_ANNOTATIONS,
        title="Transform files with Oxc",
        tags={"oxc", "transform", "mutation"},
        timeout=120.0,
        output_schema=JSON_OBJECT_OUTPUT_SCHEMA,
    )
    def oxc_transform_files(
        project_folder: Annotated[str, Field(description="Contained project directory")],
        output_root: Annotated[str, Field(description="Relative contained output directory")],
        paths: Annotated[list[str] | None, Field(min_length=1, max_length=MAX_OXC_FILES)] = None,
        include_globs: list[str] | None = None,
        exclude_globs: list[str] | None = None,
        strict_paths: bool = True,
        conflict_policy: ConflictPolicy = "error",
        allow_source_overwrite: bool = False,
        options: OxcTransformOptions | None = None,
        max_results: Annotated[int | None, Field(ge=1, le=HARD_MAX_RESULTS)] = None,
    ) -> CallToolResult:
        """Transform selected files and atomically emit artifacts to an explicit output root."""
        raw = service.emit_files(
            operation="transform",
            project_folder=project_folder,
            output_root=output_root,
            paths=paths,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            strict_paths=strict_paths,
            conflict_policy=conflict_policy,
            allow_source_overwrite=allow_source_overwrite,
            options=options,
            max_results=max_results,
        )
        return json_object_tool_result(oxc_mutation_result("transform", raw))

    @server.tool(
        annotations=READ_ONLY_ANNOTATIONS,
        title="Minify code with Oxc",
        tags={"oxc", "minify", "preview", "read-only"},
        timeout=30.0,
        output_schema=JSON_OBJECT_OUTPUT_SCHEMA,
    )
    def oxc_minify(
        source: OxcSource,
        options: OxcMinifyOptions | None = None,
        output_format: OutputFormat = "text",
    ) -> CallToolResult:
        """Return minified code, maps, legal comments, caches, assumptions, and diagnostics without writing files."""
        filename, code, _project = service.resolve_oxc_source(source)
        raw = service.minify_code(filename=filename, code=code, options=options)
        return json_object_format_tool_result(
            oxc_preview_result(operation="minify", filename=filename, source=code, raw=raw), output_format
        )

    @server.tool(
        annotations=MUTATING_ANNOTATIONS,
        title="Minify files with Oxc",
        tags={"oxc", "minify", "mutation"},
        timeout=120.0,
        output_schema=JSON_OBJECT_OUTPUT_SCHEMA,
    )
    def oxc_minify_files(
        project_folder: Annotated[str, Field(description="Contained project directory")],
        output_root: Annotated[str, Field(description="Relative contained output directory")],
        paths: Annotated[list[str] | None, Field(min_length=1, max_length=MAX_OXC_FILES)] = None,
        include_globs: list[str] | None = None,
        exclude_globs: list[str] | None = None,
        strict_paths: bool = True,
        conflict_policy: ConflictPolicy = "error",
        allow_source_overwrite: bool = False,
        options: OxcMinifyOptions | None = None,
        max_results: Annotated[int | None, Field(ge=1, le=HARD_MAX_RESULTS)] = None,
    ) -> CallToolResult:
        """Minify selected files and atomically emit artifacts to an explicit output root."""
        raw = service.emit_files(
            operation="minify",
            project_folder=project_folder,
            output_root=output_root,
            paths=paths,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            strict_paths=strict_paths,
            conflict_policy=conflict_policy,
            allow_source_overwrite=allow_source_overwrite,
            options=options,
            max_results=max_results,
        )
        return json_object_tool_result(oxc_mutation_result("minify", raw))


def register_typescript_tools(server: FastMCP, runtime: RuntimeServices) -> None:
    service = AstGrepService(runtime)

    @server.tool(
        annotations=EXECUTION_ANNOTATIONS,
        title="Execute TypeScript in an operator sandbox",
        tags={"typescript", "execution", "unsafe"},
        timeout=120.0,
        output_schema=JSON_OBJECT_OUTPUT_SCHEMA,
    )
    def typescript_execute(
        project_folder: Annotated[str, Field(description="Contained TypeScript project directory")],
        entry: Annotated[str, Field(description="Relative TypeScript entry file")],
        args: Annotated[list[str] | None, Field(max_length=64)] = None,
        stdin: Annotated[str | None, Field(max_length=MAX_SNIPPET_INPUT_BYTES)] = None,
        timeout_seconds: Annotated[float, Field(gt=0, le=120)] = 30,
        output_format: OutputFormat = "text",
    ) -> CallToolResult:
        """Execute a contained TypeScript entry through the configured external sandbox runner."""
        return json_object_format_tool_result(
            service.execute_typescript(
                project_folder=project_folder,
                entry=entry,
                args=args or (),
                stdin=stdin,
                timeout_seconds=timeout_seconds,
            ),
            output_format,
        )

    @server.tool(
        annotations=READ_ONLY_ANNOTATIONS,
        title="Inspect TypeScript project",
        tags={"typescript", "compiler", "analysis", "read-only"},
        timeout=120.0,
        output_schema=TYPESCRIPT_PROJECT_RESULTS_OUTPUT_SCHEMA,
    )
    def inspect_typescript_project(
        project_folder: Annotated[str, Field(description="Contained TypeScript project directory")],
        tsconfig: Annotated[str, Field(description="Relative tsconfig path")] = "tsconfig.json",
        paths: Annotated[list[str] | None, Field(min_length=1, max_length=MAX_OXC_FILES)] = None,
        include_emit: bool = True,
        include_code_actions: bool = True,
        max_results: Annotated[int | None, Field(ge=1, le=HARD_MAX_RESULTS)] = None,
    ) -> Annotated[CallToolResult, TypeScriptProjectResults]:
        """Return one bounded TypeScript Compiler API project snapshot."""
        return json_object_tool_result(
            service.inspect_typescript_project(
                project_folder=project_folder,
                tsconfig=tsconfig,
                paths=paths,
                include_emit=include_emit,
                include_code_actions=include_code_actions,
                max_results=max_results or runtime.default_max_results,
            )
        )


def register_postgresql_tools(server: FastMCP, runtime: RuntimeServices) -> None:
    service = AstGrepService(runtime)

    @server.tool(
        annotations=READ_ONLY_ANNOTATIONS,
        title="Parse PostgreSQL",
        tags={"postgresql", "parser", "analysis", "read-only"},
        timeout=30.0,
        output_schema=POSTGRES_PARSE_RESULTS_OUTPUT_SCHEMA,
    )
    def postgres_parse(
        sql: Annotated[str, Field(description="PostgreSQL SQL or PL/pgSQL source", max_length=MAX_SNIPPET_INPUT_BYTES)],
        mode: Literal["parse", "scan", "fingerprint", "plpgsql"] = "parse",
    ) -> Annotated[CallToolResult, PostgresParseResults]:
        """Parse, scan, fingerprint, or inspect PL/pgSQL without executing SQL."""
        return json_object_tool_result(service.inspect_postgres(operation=mode, sql=sql))

    @server.tool(
        annotations=READ_ONLY_ANNOTATIONS,
        title="Parse PostgreSQL files",
        tags={"postgresql", "parser", "analysis", "read-only"},
        timeout=90.0,
        output_schema=POSTGRES_FILES_RESULTS_OUTPUT_SCHEMA,
    )
    def postgres_parse_files(
        project_folder: Annotated[str, Field(description="Contained project directory")],
        paths: Annotated[list[str] | None, Field(min_length=1, max_length=MAX_OUTLINE_PATHS)] = None,
        include_globs: list[str] | None = None,
        exclude_globs: list[str] | None = None,
        strict_paths: bool = True,
        mode: Literal["parse", "scan", "fingerprint", "plpgsql"] = "parse",
        max_results: Annotated[int | None, Field(ge=1, le=HARD_MAX_RESULTS)] = None,
        cursor: str | None = None,
    ) -> Annotated[CallToolResult, PostgresFilesResults]:
        """Run bounded PostgreSQL parser operations over contained files."""
        query: JsonObject = {
            "project_folder": project_folder,
            "paths": json_strings(paths),
            "include_globs": json_strings(include_globs),
            "exclude_globs": json_strings(exclude_globs),
            "strict_paths": strict_paths,
            "mode": mode,
        }
        result = paged_postgres_file_results(
            runtime=runtime,
            query=query,
            max_results=max_results,
            cursor=cursor,
            execute=lambda: service.inspect_postgres_files(
                operation=mode,
                project_folder=project_folder,
                paths=paths,
                include_globs=include_globs,
                exclude_globs=exclude_globs,
                strict_paths=strict_paths,
            ),
        )
        return json_object_tool_result(cast(JsonObject, result))

    @server.tool(
        annotations=READ_ONLY_ANNOTATIONS,
        title="Preview PostgreSQL deparse",
        tags={"postgresql", "deparse", "preview", "read-only"},
        timeout=30.0,
        output_schema=POSTGRES_DEPARSE_RESULTS_OUTPUT_SCHEMA,
    )
    def postgres_deparse_preview(
        sql: Annotated[str, Field(description="PostgreSQL SQL source", max_length=MAX_SNIPPET_INPUT_BYTES)],
    ) -> Annotated[CallToolResult, PostgresDeparseResults]:
        """Deparse only with parse-deparse-reparse tree-equivalence evidence."""
        result = service.inspect_postgres(operation="deparse", sql=sql)
        if result.get("equivalent") is not True:
            raise RuntimeError("PostgreSQL deparse did not preserve the normalized parse tree")
        return json_object_tool_result(result)


def register_semantic_tools(server: FastMCP, runtime: RuntimeServices) -> None:
    service = AstGrepService(runtime)

    @server.tool(
        annotations=READ_ONLY_ANNOTATIONS,
        title="Inspect semantic scopes",
        tags={"semantic", "scope", "read-only"},
        timeout=60.0,
        output_schema=JSON_OBJECT_OUTPUT_SCHEMA,
    )
    def semantic_scopes(
        project_folder: Annotated[str, Field(description="Contained project directory")],
        path: Annotated[str, Field(description="Relative JavaScript or TypeScript source file")],
        source_digest: str | None = None,
        max_results: Annotated[int | None, Field(ge=1, le=HARD_MAX_RESULTS)] = None,
        cursor: Annotated[str | None, Field(description="Opaque continuation cursor")] = None,
        output_format: OutputFormat = "text",
    ) -> Annotated[CallToolResult, SemanticScopesResults]:
        """Return Oxc lexical scopes, bindings, spans, and diagnostics."""
        query: JsonObject = {"project_folder": project_folder, "path": path, "source_digest": source_digest}
        results = paged_semantic_results(
            runtime=runtime,
            tool_name="semantic_scopes",
            query=query,
            max_results=max_results,
            cursor=cursor,
            record_keys=("scopes",),
            execute=lambda: service.analyze_semantics(
                operation="scopes", project_folder=project_folder, path=path, position=None, source_digest=source_digest
            ),
        )
        return json_object_format_tool_result(results, output_format)

    @server.tool(
        annotations=READ_ONLY_ANNOTATIONS,
        title="Inspect semantic symbols",
        tags={"semantic", "symbol", "read-only"},
        timeout=60.0,
        output_schema=JSON_OBJECT_OUTPUT_SCHEMA,
    )
    def semantic_symbols(
        project_folder: Annotated[str, Field(description="Contained project directory")],
        path: Annotated[str, Field(description="Relative JavaScript or TypeScript source file")],
        source_digest: str | None = None,
        max_results: Annotated[int | None, Field(ge=1, le=HARD_MAX_RESULTS)] = None,
        cursor: Annotated[str | None, Field(description="Opaque continuation cursor")] = None,
        output_format: OutputFormat = "text",
    ) -> Annotated[CallToolResult, SemanticSymbolsResults]:
        """Return Oxc lexical symbols, declarations, flags, scopes, and reference counts."""
        query: JsonObject = {"project_folder": project_folder, "path": path, "source_digest": source_digest}
        results = paged_semantic_results(
            runtime=runtime,
            tool_name="semantic_symbols",
            query=query,
            max_results=max_results,
            cursor=cursor,
            record_keys=("symbols",),
            execute=lambda: service.analyze_semantics(
                operation="symbols", project_folder=project_folder, path=path, position=None, source_digest=source_digest
            ),
        )
        return json_object_format_tool_result(results, output_format)

    @server.tool(
        annotations=READ_ONLY_ANNOTATIONS,
        title="Inspect semantic references",
        tags={"semantic", "reference", "read-only"},
        timeout=90.0,
        output_schema=JSON_OBJECT_OUTPUT_SCHEMA,
    )
    def semantic_references(
        project_folder: Annotated[str, Field(description="Contained project directory")],
        path: Annotated[str, Field(description="Relative JavaScript or TypeScript source file")],
        position: Annotated[int, Field(ge=0, description="Zero-based UTF-8 byte offset selecting a symbol")],
        source_digest: str | None = None,
        include_declaration: bool = True,
        include_unresolved: bool = False,
        project_paths: Annotated[list[str] | None, Field(min_length=1, max_length=MAX_OXC_FILES)] = None,
        include_globs: list[str] | None = None,
        exclude_globs: list[str] | None = None,
        max_results: Annotated[int | None, Field(ge=1, le=HARD_MAX_RESULTS)] = None,
        cursor: Annotated[str | None, Field(description="Opaque continuation cursor")] = None,
        output_format: OutputFormat = "text",
    ) -> Annotated[CallToolResult, SemanticReferencesResults]:
        """Return position-selected lexical references, unresolved identifiers, and module-graph links."""
        query: JsonObject = {
            "project_folder": project_folder,
            "path": path,
            "position": position,
            "source_digest": source_digest,
            "include_declaration": include_declaration,
            "include_unresolved": include_unresolved,
            "project_paths": json_strings(project_paths),
            "include_globs": json_strings(include_globs),
            "exclude_globs": json_strings(exclude_globs),
        }
        results = paged_semantic_results(
            runtime=runtime,
            tool_name="semantic_references",
            query=query,
            max_results=max_results,
            cursor=cursor,
            record_keys=("references", "unresolved", "module_graph_links"),
            execute=lambda: service.analyze_semantics(
                operation="references",
                project_folder=project_folder,
                path=path,
                position=position,
                source_digest=source_digest,
                include_declaration=include_declaration,
                include_unresolved=include_unresolved,
                project_paths=project_paths,
                include_globs=include_globs,
                exclude_globs=exclude_globs,
            ),
        )
        return json_object_format_tool_result(results, output_format)

    @server.tool(
        annotations=READ_ONLY_ANNOTATIONS,
        title="Inspect control-flow graph",
        tags={"semantic", "cfg", "read-only"},
        timeout=90.0,
        output_schema=JSON_OBJECT_OUTPUT_SCHEMA,
    )
    def semantic_cfg(
        project_folder: Annotated[str, Field(description="Contained project directory")],
        path: Annotated[str, Field(description="Relative JavaScript or TypeScript source file")],
        source_digest: str | None = None,
        function_position: Annotated[int | None, Field(ge=0)] = None,
        max_results: Annotated[int | None, Field(ge=1, le=HARD_MAX_RESULTS)] = None,
        cursor: Annotated[str | None, Field(description="Opaque continuation cursor")] = None,
        output_format: OutputFormat = "text",
    ) -> Annotated[CallToolResult, SemanticCfgResults]:
        """Return Oxc CFG basic blocks, typed edges, reachability, and diagnostics."""
        query: JsonObject = {
            "project_folder": project_folder,
            "path": path,
            "source_digest": source_digest,
            "function_position": function_position,
        }
        results = paged_semantic_results(
            runtime=runtime,
            tool_name="semantic_cfg",
            query=query,
            max_results=max_results,
            cursor=cursor,
            record_keys=("functions",),
            execute=lambda: service.analyze_semantics(
                operation="cfg",
                project_folder=project_folder,
                path=path,
                position=None,
                source_digest=source_digest,
                function_position=function_position,
            ),
        )
        return json_object_format_tool_result(results, output_format)


def create_mcp(runtime: RuntimeServices) -> FastMCP:
    @asynccontextmanager
    async def lifespan(_server: FastMCP) -> AsyncGenerator[dict[str, object]]:
        environment = dict(os.environ)
        if runtime.oxc_helper is not None:
            runtime.oxc_worker = JsonLineWorker(
                command=runtime.oxc_helper.command_prefix,
                cwd=str(runtime.working_directory),
                environment=environment,
            )
        if runtime.analysis_helper is not None:
            runtime.analysis_worker = JsonLineWorker(
                command=runtime.analysis_helper.command_prefix,
                cwd=str(runtime.working_directory),
                environment=environment,
            )
        if runtime.typescript_project_helper is not None:
            runtime.typescript_project_worker = JsonLineWorker(
                command=runtime.typescript_project_helper.command_prefix,
                cwd=str(runtime.working_directory),
                environment=environment,
            )
        try:
            yield {"services": runtime, "cursor_store": runtime.cursor_store}
        finally:
            if runtime.oxc_worker is not None:
                runtime.oxc_worker.close()
                runtime.oxc_worker = None
            if runtime.analysis_worker is not None:
                runtime.analysis_worker.close()
                runtime.analysis_worker = None
            if runtime.typescript_project_worker is not None:
                runtime.typescript_project_worker.close()
                runtime.typescript_project_worker = None
            runtime.cursor_store.clear()

    server = FastMCP(
        "ast-soleaux",
        instructions=SERVER_INSTRUCTIONS,
        version=_server_version(),
        lifespan=lifespan,
        mask_error_details=True,
    )
    register_structural_tools(server, runtime)
    register_oxc_tools(server, runtime)
    register_typescript_tools(server, runtime)
    register_postgresql_tools(server, runtime)
    register_semantic_tools(server, runtime)
    server.add_middleware(ErrorHandlingMiddleware(include_traceback=False, transform_errors=False))
    if runtime.oxc_helper is None:
        server.disable(
            names={
                "oxc_modules",
                "oxc_transform",
                "oxc_transform_files",
                "oxc_minify",
                "oxc_minify_files",
            }
        )
    if runtime.analysis_helper is None:
        server.disable(names={"semantic_scopes", "semantic_symbols", "semantic_references", "semantic_cfg"})
    if runtime.typescript_project_helper is None:
        server.disable(names={"inspect_typescript_project"})
    if runtime.postgres_helper is None:
        server.disable(names={"postgres_parse", "postgres_parse_files", "postgres_deparse_preview"})
    if runtime.typescript_execution_helper is None:
        server.disable(names={"typescript_execute"})
    snapshot = runtime.config_snapshot
    if snapshot is None or not snapshot.capabilities["configured_scan"]:
        server.disable(names={"scan_project_rules"})
    if snapshot is None or not snapshot.capabilities["configured_tests"]:
        server.disable(names={"test_project_rules"})
    return server


def _environment_bool(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded structural analysis, transformation, and execution MCP server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("AST_GREP_CONFIG"),
        metavar="PATH",
        help="Path to sgconfig.yml; it must resolve inside an allowed root",
    )
    parser.add_argument(
        "--ast-grep",
        default=os.environ.get("AST_GREP_EXECUTABLE", "ast-grep"),
        metavar="PATH",
        help="ast-grep executable or command name",
    )
    parser.add_argument(
        "--oxc-helper",
        default=os.environ.get("OXC_HELPER_EXECUTABLE"),
        metavar="PATH",
        help="Optional ast-soleaux Oxc compute helper JavaScript file or executable",
    )
    parser.add_argument(
        "--analysis-helper",
        default=os.environ.get("AST_SOLEAUX_ANALYSIS_EXECUTABLE"),
        metavar="PATH",
        help="Optional ast-soleaux Rust semantic analysis executable",
    )
    parser.add_argument(
        "--typescript-project-helper",
        default=os.environ.get("AST_SOLEAUX_TYPESCRIPT_PROJECT_EXECUTABLE"),
        metavar="PATH",
        help="Optional ast-soleaux TypeScript Compiler API worker",
    )
    parser.add_argument(
        "--postgres-helper",
        default=os.environ.get("AST_SOLEAUX_POSTGRES_EXECUTABLE"),
        metavar="PATH",
        help="Optional ast-soleaux libpg_query PostgreSQL parser worker",
    )
    parser.add_argument(
        "--typescript-execution-helper",
        default=os.environ.get("AST_SOLEAUX_TYPESCRIPT_EXECUTION_EXECUTABLE"),
        metavar="PATH",
        help="Optional operator-approved TypeScript sandbox runner",
    )
    parser.add_argument(
        "--typescript-execution-profile",
        choices=("isolated", "workspace-write", "networked"),
        default=os.environ.get("AST_SOLEAUX_TYPESCRIPT_EXECUTION_PROFILE", "isolated"),
        help="Operator-selected TypeScript sandbox permission profile",
    )
    parser.add_argument(
        "--allowed-root",
        action="append",
        default=None,
        metavar="PATH",
        help="Allowed project root; repeat for multiple roots (defaults to the process working directory)",
    )
    parser.add_argument(
        "--trusted-native-library",
        action="append",
        nargs=2,
        default=None,
        metavar=("PATH", "SHA256"),
        help=(
            "Trust one custom-language native parser by exact contained path and SHA-256 digest; "
            "repeat for every library referenced by sgconfig.yml"
        ),
    )
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=os.environ.get("AST_GREP_COMMAND_TIMEOUT", str(DEFAULT_COMMAND_TIMEOUT_SECONDS)),
        metavar="SECONDS",
        help="Timeout for each analysis subprocess",
    )
    parser.add_argument(
        "--default-max-results",
        type=int,
        default=os.environ.get("AST_GREP_DEFAULT_MAX_RESULTS", str(DEFAULT_MAX_RESULTS)),
        metavar="COUNT",
        help="Default finite search result limit",
    )
    parser.add_argument(
        "--max-results-cap",
        type=int,
        default=os.environ.get("AST_GREP_MAX_RESULTS_CAP", str(HARD_MAX_RESULTS)),
        metavar="COUNT",
        help=f"Maximum allowed result limit (never above {HARD_MAX_RESULTS})",
    )
    parser.add_argument(
        "--forbid-regex-rules",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Reject inline ast-grep YAML containing a regex key anywhere in the document, "
            "including metavariable constraints, nested relational rules, and utils; "
            "unset falls back to AST_GREP_FORBID_REGEX_RULES"
        ),
    )
    return parser


def _resolve_forbid_regex_rules(cli_value: bool | None) -> bool:
    if cli_value is not None:
        return cli_value
    return _environment_bool("AST_GREP_FORBID_REGEX_RULES")


def run_mcp_server() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        runtime = build_runtime(
            working_directory=Path.cwd(),
            ast_grep_executable=args.ast_grep,
            oxc_helper_executable=args.oxc_helper,
            analysis_helper_executable=args.analysis_helper,
            typescript_project_helper_executable=args.typescript_project_helper,
            postgres_helper_executable=args.postgres_helper,
            typescript_execution_helper_executable=args.typescript_execution_helper,
            typescript_execution_profile=args.typescript_execution_profile,
            config_path=args.config,
            allowed_roots=args.allowed_root or (),
            command_timeout_seconds=args.command_timeout,
            default_max_results=args.default_max_results,
            max_results_cap=args.max_results_cap,
            forbid_regex_rules=_resolve_forbid_regex_rules(args.forbid_regex_rules),
            trusted_native_libraries=tuple(tuple(value) for value in (args.trusted_native_library or ())),
        )
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    server = create_mcp(runtime)
    try:
        server.run(show_banner=False)
    finally:
        runtime.close()


if __name__ == "__main__":
    run_mcp_server()
