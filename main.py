from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Any, Final, Literal, TypedDict

import yaml as yaml_parser
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field

DEFAULT_MAX_RESULTS: Final = 50
HARD_MAX_RESULTS: Final = 500
DEFAULT_COMMAND_TIMEOUT_SECONDS: Final = 30.0
FALLBACK_SERVER_VERSION: Final = "0.2.0"

BUILTIN_LANGUAGES: Final = (
    "bash",
    "c",
    "cpp",
    "csharp",
    "css",
    "elixir",
    "go",
    "haskell",
    "html",
    "java",
    "javascript",
    "json",
    "jsx",
    "kotlin",
    "lua",
    "nix",
    "php",
    "python",
    "ruby",
    "rust",
    "scala",
    "solidity",
    "swift",
    "tsx",
    "typescript",
    "yaml",
)

DumpFormat = Literal["pattern", "cst", "ast"]
OutputFormat = Literal["text", "json"]
Transport = Literal["stdio", "sse", "streamable-http"]
CompletedTextProcess = subprocess.CompletedProcess[str]
ProcessRunner = Callable[..., CompletedTextProcess]


class SearchResults(TypedDict):
    matches: list[dict[str, Any]]
    returned: int
    truncated: bool
    limit: int


class ServerInfo(TypedDict):
    fork_version: str
    ast_grep_executable: str
    ast_grep_version: str
    config_path: str | None
    allowed_roots: list[str]
    command_timeout_seconds: float
    default_max_results: int
    max_results_cap: int
    forbid_regex_rules: bool


@dataclass(frozen=True)
class ResolvedExecutable:
    path: Path
    command_prefix: tuple[str, ...]


@dataclass(frozen=True)
class ServerRuntime:
    working_directory: Path
    executable: ResolvedExecutable
    ast_grep_version: str
    config_path: Path | None
    allowed_roots: tuple[Path, ...]
    command_timeout_seconds: float
    default_max_results: int
    max_results_cap: int
    forbid_regex_rules: bool


READ_ONLY_ANNOTATIONS: Final = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


def _server_version() -> str:
    try:
        return version("sg-mcp")
    except PackageNotFoundError:
        return FALLBACK_SERVER_VERSION


def _is_within(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


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
    if not any(_is_within(path, root) for root in allowed_roots):
        allowed = ", ".join(str(root) for root in allowed_roots)
        raise ValueError(f"{label} resolves outside the allowed roots ({allowed}): {path}")


def _read_json_file(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _npm_package_bin(package_directory: Path) -> Path | None:
    manifest_path = package_directory / "package.json"
    manifest = _read_json_file(manifest_path)
    if manifest is None or manifest.get("name") != "@ast-grep/cli":
        return None

    bin_value = manifest.get("bin")
    if isinstance(bin_value, str):
        relative_bin = bin_value
    elif isinstance(bin_value, Mapping) and isinstance(bin_value.get("ast-grep"), str):
        relative_bin = bin_value["ast-grep"]
    else:
        return None

    try:
        target = (package_directory / relative_bin).resolve(strict=True)
    except FileNotFoundError:
        return None
    if not target.is_file() or not _is_within(target, package_directory.resolve()):
        return None
    return target


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
        first_line = executable.open("rb").readline(256)
    except OSError:
        return False
    return first_line.startswith(b"#!") and b"node" in first_line.lower()


def resolve_ast_grep_executable(raw_executable: str, *, working_directory: Path) -> ResolvedExecutable:
    raw_path = Path(raw_executable).expanduser()
    has_path_separator = os.sep in raw_executable or (os.altsep is not None and os.altsep in raw_executable)
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


def _bounded_error_text(value: str | None, *, limit: int = 4000) -> str:
    text = (value or "").strip()
    if not text:
        return "(no error output)"
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def run_process(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    input_text: str | None = None,
    working_directory: Path | None = None,
    allowed_exit_codes: frozenset[int] = frozenset({0}),
    runner: ProcessRunner = subprocess.run,
) -> CompletedTextProcess:
    try:
        result = runner(
            list(command),
            capture_output=True,
            input=input_text,
            text=True,
            timeout=timeout_seconds,
            cwd=str(working_directory) if working_directory is not None else None,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Command timed out after {timeout_seconds:g} seconds") from error
    except FileNotFoundError as error:
        raise RuntimeError(f"Command executable was not found: {command[0]}") from error

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
    parts = result.stdout.strip().split()
    if len(parts) < 2 or parts[0] != "ast-grep":
        raise ValueError(f"Configured executable is not ast-grep: {_bounded_error_text(result.stdout)}")
    return parts[1]


def build_runtime(
    *,
    working_directory: Path,
    ast_grep_executable: str = "ast-grep",
    config_path: str | None = None,
    allowed_roots: Sequence[str] = (),
    command_timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    default_max_results: int = DEFAULT_MAX_RESULTS,
    max_results_cap: int = HARD_MAX_RESULTS,
    forbid_regex_rules: bool = False,
    runner: ProcessRunner = subprocess.run,
) -> ServerRuntime:
    resolved_working_directory = working_directory.resolve(strict=True)
    if not resolved_working_directory.is_dir():
        raise ValueError(f"Working directory is not a directory: {working_directory}")
    if command_timeout_seconds <= 0:
        raise ValueError("Command timeout must be greater than zero")
    if not 1 <= max_results_cap <= HARD_MAX_RESULTS:
        raise ValueError(f"Result cap must be between 1 and {HARD_MAX_RESULTS}")
    if not 1 <= default_max_results <= max_results_cap:
        raise ValueError("Default result limit must be positive and no greater than the configured cap")

    resolved_roots = resolve_allowed_roots(allowed_roots, working_directory=resolved_working_directory)
    executable = resolve_ast_grep_executable(ast_grep_executable, working_directory=resolved_working_directory)
    resolved_config: Path | None = None
    if config_path is not None:
        resolved_config = _resolve_existing_path(config_path, base=resolved_working_directory, kind="file")
        _require_allowed(resolved_config, resolved_roots, label="Config path")

    return ServerRuntime(
        working_directory=resolved_working_directory,
        executable=executable,
        ast_grep_version=_read_ast_grep_version(
            executable,
            timeout_seconds=command_timeout_seconds,
            runner=runner,
        ),
        config_path=resolved_config,
        allowed_roots=resolved_roots,
        command_timeout_seconds=command_timeout_seconds,
        default_max_results=default_max_results,
        max_results_cap=max_results_cap,
        forbid_regex_rules=forbid_regex_rules,
    )


def get_supported_languages(config_path: Path | None = None) -> list[str]:
    languages = set(BUILTIN_LANGUAGES)
    if config_path is None:
        return sorted(languages)

    try:
        config = yaml_parser.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml_parser.YAMLError) as error:
        raise ValueError(f"Unable to read ast-grep config: {config_path}") from error
    if config is None:
        return sorted(languages)
    if not isinstance(config, Mapping):
        raise ValueError(f"ast-grep config must contain a YAML mapping: {config_path}")
    custom_languages = config.get("customLanguages")
    if custom_languages is not None:
        if not isinstance(custom_languages, Mapping):
            raise ValueError("customLanguages must be a mapping")
        languages.update(str(name) for name in custom_languages)
    return sorted(languages)


def _contains_mapping_key(value: Any, forbidden_key: str) -> bool:
    if isinstance(value, Mapping):
        return any(key == forbidden_key or _contains_mapping_key(item, forbidden_key) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_mapping_key(item, forbidden_key) for item in value)
    return False


def validate_rule_yaml(rule_yaml: str, *, forbid_regex_rules: bool) -> None:
    try:
        documents = list(yaml_parser.safe_load_all(rule_yaml))
    except yaml_parser.YAMLError as error:
        raise ValueError(f"Invalid ast-grep rule YAML: {error}") from error
    if not documents or all(document is None for document in documents):
        raise ValueError("ast-grep rule YAML must contain at least one rule")

    for document in documents:
        if not isinstance(document, Mapping):
            raise ValueError("Each ast-grep inline rule must be a YAML mapping")
        missing = [key for key in ("id", "language", "rule") if key not in document]
        if missing:
            raise ValueError(f"ast-grep rule is missing required fields: {', '.join(missing)}")
        if not isinstance(document["rule"], Mapping):
            raise ValueError("ast-grep rule field must be a mapping")
        if forbid_regex_rules and _contains_mapping_key(document, "regex"):
            raise ValueError("Regex ast-grep rules are disabled by server policy")


def parse_stream_matches(stdout: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(stdout.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"ast-grep emitted invalid JSON on line {line_number}") from error
        if not isinstance(value, dict):
            raise RuntimeError(f"ast-grep emitted a non-object JSON match on line {line_number}")
        matches.append(value)
    return matches


def format_matches_as_text(matches: Sequence[Mapping[str, Any]]) -> str:
    output_blocks: list[str] = []
    for match in matches:
        file_path = str(match.get("file", ""))
        range_value = match.get("range")
        range_mapping = range_value if isinstance(range_value, Mapping) else {}
        start_value = range_mapping.get("start")
        end_value = range_mapping.get("end")
        start_mapping = start_value if isinstance(start_value, Mapping) else {}
        end_mapping = end_value if isinstance(end_value, Mapping) else {}
        start_line = int(start_mapping.get("line", 0)) + 1
        end_line = int(end_mapping.get("line", start_line - 1)) + 1
        match_text = str(match.get("text", "")).rstrip()
        header = f"{file_path}:{start_line}" if start_line == end_line else f"{file_path}:{start_line}-{end_line}"
        output_blocks.append(f"{header}\n{match_text}")
    return "\n\n".join(output_blocks)


def format_search_results(results: SearchResults) -> str:
    if results["returned"] == 0:
        return "No matches found"
    noun = "match" if results["returned"] == 1 else "matches"
    header = f"Found {results['returned']} {noun}"
    if results["truncated"]:
        header += f" (limit {results['limit']}; additional matches exist)"
    return f"{header}:\n\n{format_matches_as_text(results['matches'])}"


def search_tool_result(results: SearchResults, output_format: OutputFormat) -> CallToolResult:
    if output_format == "json":
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(results, separators=(",", ":")))],
            structuredContent=dict(results),
        )
    return CallToolResult(content=[TextContent(type="text", text=format_search_results(results))])


class AstGrepService:
    def __init__(self, runtime: ServerRuntime, *, runner: ProcessRunner = subprocess.run) -> None:
        self.runtime = runtime
        self.runner = runner

    def _run(
        self,
        subcommand: str,
        arguments: Sequence[str],
        *,
        input_text: str | None = None,
        working_directory: Path | None = None,
        allow_no_matches: bool = False,
    ) -> CompletedTextProcess:
        command = [*self.runtime.executable.command_prefix, subcommand]
        if self.runtime.config_path is not None:
            command.extend(["--config", str(self.runtime.config_path)])
        command.extend(arguments)
        result = run_process(
            command,
            timeout_seconds=self.runtime.command_timeout_seconds,
            input_text=input_text,
            working_directory=working_directory or self.runtime.working_directory,
            allowed_exit_codes=frozenset({0, 1}) if allow_no_matches else frozenset({0}),
            runner=self.runner,
        )
        if result.returncode == 1 and result.stderr.strip() and not result.stdout.strip():
            raise RuntimeError(f"ast-grep search failed: {_bounded_error_text(result.stderr)}")
        return result

    def _validate_language(self, language: str) -> None:
        if not language:
            raise ValueError("language is required")
        if language not in get_supported_languages(self.runtime.config_path):
            raise ValueError(f"Unsupported ast-grep language: {language}")

    def _resolve_project(self, project_folder: str) -> Path:
        project = _resolve_existing_path(
            project_folder,
            base=self.runtime.working_directory,
            kind="directory",
        )
        _require_allowed(project, self.runtime.allowed_roots, label="Project folder")
        return project

    def _resolve_paths(self, project: Path, raw_paths: Sequence[str] | None) -> list[Path]:
        paths = list(raw_paths or ["."])
        if not paths:
            raise ValueError("paths must contain at least one relative path")
        resolved_paths: list[Path] = []
        for raw_path in paths:
            if not raw_path:
                raise ValueError("paths cannot contain empty values")
            candidate = Path(raw_path)
            if candidate.is_absolute():
                raise ValueError(f"Search paths must be relative to project_folder: {raw_path}")
            try:
                resolved = (project / candidate).resolve(strict=True)
            except FileNotFoundError as error:
                raise ValueError(f"Search path does not exist: {raw_path}") from error
            if not _is_within(resolved, project):
                raise ValueError(f"Search path resolves outside project_folder: {raw_path}")
            _require_allowed(resolved, self.runtime.allowed_roots, label="Search path")
            if resolved not in resolved_paths:
                resolved_paths.append(resolved)
        return resolved_paths

    @staticmethod
    def _glob_arguments(include_globs: Sequence[str] | None, exclude_globs: Sequence[str] | None) -> list[str]:
        arguments: list[str] = []
        for glob in include_globs or ():
            if not glob or "\0" in glob or glob.startswith("!"):
                raise ValueError(f"Invalid include glob: {glob!r}")
            arguments.extend(["--globs", glob])
        for glob in exclude_globs or ():
            if not glob or "\0" in glob or glob.startswith("!"):
                raise ValueError(f"Invalid exclude glob: {glob!r}")
            arguments.extend(["--globs", f"!{glob}"])
        return arguments

    def _normalize_match_path(self, match: dict[str, Any], project: Path) -> dict[str, Any]:
        raw_file = match.get("file")
        if not isinstance(raw_file, str) or not raw_file:
            return match
        file_path = Path(raw_file)
        if not file_path.is_absolute():
            file_path = project / file_path
        resolved_file = file_path.resolve(strict=True)
        if not _is_within(resolved_file, project):
            raise RuntimeError(f"ast-grep returned a match outside project_folder: {raw_file}")
        normalized = dict(match)
        normalized["file"] = resolved_file.relative_to(project).as_posix()
        return normalized

    def _result_limit(self, requested: int | None) -> int:
        limit = self.runtime.default_max_results if requested is None else requested
        if not 1 <= limit <= self.runtime.max_results_cap:
            raise ValueError(f"max_results must be between 1 and {self.runtime.max_results_cap}")
        return limit

    def _search(
        self,
        *,
        project_folder: str,
        rule_yaml: str,
        paths: Sequence[str] | None,
        include_globs: Sequence[str] | None,
        exclude_globs: Sequence[str] | None,
        max_results: int | None,
    ) -> SearchResults:
        validate_rule_yaml(rule_yaml, forbid_regex_rules=self.runtime.forbid_regex_rules)
        project = self._resolve_project(project_folder)
        search_paths = self._resolve_paths(project, paths)
        limit = self._result_limit(max_results)
        arguments = [
            "--inline-rules",
            rule_yaml,
            "--json=stream",
            "--max-results",
            str(limit + 1),
            *self._glob_arguments(include_globs, exclude_globs),
            *(str(path) for path in search_paths),
        ]
        result = self._run(
            "scan",
            arguments,
            working_directory=project,
            allow_no_matches=True,
        )
        parsed_matches = parse_stream_matches(result.stdout)
        truncated = len(parsed_matches) > limit
        matches = [self._normalize_match_path(match, project) for match in parsed_matches[:limit]]
        return {
            "matches": matches,
            "returned": len(matches),
            "truncated": truncated,
            "limit": limit,
        }

    def dump_syntax_tree(self, *, code: str, language: str, format: DumpFormat) -> str:
        self._validate_language(language)
        result = self._run(
            "run",
            ["--pattern", code, "--lang", language, f"--debug-query={format}"],
            allow_no_matches=True,
        )
        return result.stderr.strip()

    def test_match_code_rule(self, *, code: str, rule_yaml: str) -> list[dict[str, Any]]:
        validate_rule_yaml(rule_yaml, forbid_regex_rules=self.runtime.forbid_regex_rules)
        result = self._run(
            "scan",
            [
                "--inline-rules",
                rule_yaml,
                "--json=stream",
                "--max-results",
                str(self.runtime.max_results_cap),
                "--stdin",
            ],
            input_text=code,
            allow_no_matches=True,
        )
        return parse_stream_matches(result.stdout)

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
    ) -> SearchResults:
        self._validate_language(language)
        rule_yaml = yaml_parser.safe_dump(
            {
                "id": "mcp-pattern-search",
                "language": language,
                "rule": {"pattern": pattern},
            },
            sort_keys=False,
        )
        return self._search(
            project_folder=project_folder,
            rule_yaml=rule_yaml,
            paths=paths,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            max_results=max_results,
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
    ) -> SearchResults:
        return self._search(
            project_folder=project_folder,
            rule_yaml=rule_yaml,
            paths=paths,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            max_results=max_results,
        )

    def get_server_info(self) -> ServerInfo:
        return {
            "fork_version": _server_version(),
            "ast_grep_executable": str(self.runtime.executable.path),
            "ast_grep_version": self.runtime.ast_grep_version,
            "config_path": str(self.runtime.config_path) if self.runtime.config_path is not None else None,
            "allowed_roots": [str(root) for root in self.runtime.allowed_roots],
            "command_timeout_seconds": self.runtime.command_timeout_seconds,
            "default_max_results": self.runtime.default_max_results,
            "max_results_cap": self.runtime.max_results_cap,
            "forbid_regex_rules": self.runtime.forbid_regex_rules,
        }


_runtime: ServerRuntime | None = None


def configure_runtime(runtime: ServerRuntime) -> None:
    global _runtime
    _runtime = runtime


def current_service() -> AstGrepService:
    if _runtime is None:
        raise RuntimeError("ast-grep MCP runtime has not been configured")
    return AstGrepService(_runtime)


mcp = FastMCP(
    "ast-grep",
    instructions=(
        "Read-only structural code inspection. Inspect syntax and probe rules before bounded project searches. "
        "Use repository CLI workflows for exhaustive scans and rewrites."
    ),
)


def register_mcp_tools(server: FastMCP) -> None:
    @server.tool(annotations=READ_ONLY_ANNOTATIONS)
    def dump_syntax_tree(
        code: Annotated[str, Field(description="Code or pattern to inspect")],
        language: Annotated[str, Field(description="Explicit ast-grep language")],
        format: Annotated[
            DumpFormat,
            Field(description="Syntax dump format: pattern, ast, or cst"),
        ] = "cst",
    ) -> str:
        """Inspect how ast-grep parses code or a query pattern."""
        return current_service().dump_syntax_tree(code=code, language=language, format=format)

    @server.tool(annotations=READ_ONLY_ANNOTATIONS)
    def test_match_code_rule(
        code: Annotated[str, Field(description="Code to test against the rule")],
        yaml: Annotated[str, Field(description="ast-grep YAML with id, language, and rule fields")],
    ) -> list[dict[str, Any]]:
        """Probe an ast-grep YAML rule against a code snippet; a valid negative probe returns an empty list."""
        return current_service().test_match_code_rule(code=code, rule_yaml=yaml)

    @server.tool(annotations=READ_ONLY_ANNOTATIONS)
    def find_code(
        project_folder: Annotated[
            str,
            Field(description="Project directory, resolved and constrained to the server's allowed roots"),
        ],
        pattern: Annotated[str, Field(description="Valid ast-grep structural pattern")],
        language: Annotated[str, Field(description="Required ast-grep language")],
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
            Field(ge=1, le=HARD_MAX_RESULTS, description="Finite result limit; defaults to the server's configured limit"),
        ] = None,
        output_format: Annotated[
            OutputFormat,
            Field(description="Compact text or structured JSON result"),
        ] = "text",
    ) -> CallToolResult:
        """Find bounded structural pattern matches inside an allowed project scope."""
        results = current_service().find_code(
            project_folder=project_folder,
            pattern=pattern,
            language=language,
            paths=paths,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            max_results=max_results,
        )
        return search_tool_result(results, output_format)

    @server.tool(annotations=READ_ONLY_ANNOTATIONS)
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
            Field(ge=1, le=HARD_MAX_RESULTS, description="Finite result limit; defaults to the server's configured limit"),
        ] = None,
        output_format: Annotated[
            OutputFormat,
            Field(description="Compact text or structured JSON result"),
        ] = "text",
    ) -> CallToolResult:
        """Find bounded matches for one or more validated ast-grep YAML rules."""
        results = current_service().find_code_by_rule(
            project_folder=project_folder,
            rule_yaml=yaml,
            paths=paths,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            max_results=max_results,
        )
        return search_tool_result(results, output_format)

    @server.tool(annotations=READ_ONLY_ANNOTATIONS)
    def get_server_info() -> ServerInfo:
        """Report the fork, executable, containment, configuration, timeout, and result-limit contract."""
        return current_service().get_server_info()


register_mcp_tools(mcp)


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
        description="Bounded, read-only ast-grep MCP server",
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
        "--allowed-root",
        action="append",
        default=None,
        metavar="PATH",
        help="Allowed project root; repeat for multiple roots (defaults to the process working directory)",
    )
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=float(os.environ.get("AST_GREP_COMMAND_TIMEOUT", DEFAULT_COMMAND_TIMEOUT_SECONDS)),
        metavar="SECONDS",
        help="Timeout for each ast-grep subprocess",
    )
    parser.add_argument(
        "--default-max-results",
        type=int,
        default=int(os.environ.get("AST_GREP_DEFAULT_MAX_RESULTS", DEFAULT_MAX_RESULTS)),
        metavar="COUNT",
        help="Default finite search result limit",
    )
    parser.add_argument(
        "--max-results-cap",
        type=int,
        default=int(os.environ.get("AST_GREP_MAX_RESULTS_CAP", HARD_MAX_RESULTS)),
        metavar="COUNT",
        help=f"Maximum allowed result limit (never above {HARD_MAX_RESULTS})",
    )
    parser.add_argument(
        "--forbid-regex-rules",
        action=argparse.BooleanOptionalAction,
        default=_environment_bool("AST_GREP_FORBID_REGEX_RULES"),
        help="Reject inline ast-grep YAML containing regex matcher keys",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default="stdio",
        help="MCP transport",
    )
    parser.add_argument("--port", type=int, default=3101, help="Port for HTTP transports")
    return parser


def run_mcp_server() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        runtime = build_runtime(
            working_directory=Path.cwd(),
            ast_grep_executable=args.ast_grep,
            config_path=args.config,
            allowed_roots=args.allowed_root or (),
            command_timeout_seconds=args.command_timeout,
            default_max_results=args.default_max_results,
            max_results_cap=args.max_results_cap,
            forbid_regex_rules=args.forbid_regex_rules,
        )
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    configure_runtime(runtime)
    mcp.settings.port = args.port
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    run_mcp_server()
