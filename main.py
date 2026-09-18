import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
from typing import Any, List, Literal, Optional

import yaml
from mcp.server import MCPServer
from pydantic import Field


# Multi-session stability: Handle SIGINT gracefully
# Claude Code sends SIGINT to existing MCP processes when new sessions start
# We ignore SIGINT to maintain stability for the original session
def _setup_signal_handlers():
    """Setup signal handlers for multi-session stability."""

    def sigint_handler(signum, frame):
        # Log but don't exit - let the MCP server continue serving
        print("Received SIGINT - ignoring for multi-session stability", file=sys.stderr)

    def sigterm_handler(signum, frame):
        # SIGTERM is a polite termination request - we should honor it
        print("Received SIGTERM - shutting down gracefully", file=sys.stderr)
        sys.exit(0)

    # Windows doesn't have SIGINT the same way, but we handle it anyway
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, sigint_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, sigterm_handler)


_setup_signal_handlers()

# Global variables (will be set by parse_args_and_get_config)
CONFIG_PATH = None
TRANSPORT_TYPE = "stdio"
SERVER_PORT = 8000
AST_GREP_COMMAND = None


def parse_args_and_get_config():
    """Parse command-line arguments and determine config path and transport."""
    global CONFIG_PATH, TRANSPORT_TYPE, SERVER_PORT, AST_GREP_COMMAND

    # Determine how the script was invoked
    prog = None
    if sys.argv[0].endswith("main.py"):
        # Direct execution: python main.py
        prog = "python main.py"

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        prog=prog,
        description="ast-grep MCP Server - Provides structural code search capabilities via Model Context Protocol",
        epilog="""
environment variables:
  AST_GREP_CONFIG    Path to sgconfig.yaml file (overridden by --config flag)
  AST_GREP_PATH      Command used to run ast-grep (for example, 'uv run ast-grep')

For more information, see: https://github.com/ast-grep/ast-grep-mcp
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        metavar="PATH",
        help="Path to sgconfig.yaml file for customizing ast-grep behavior (language mappings, rule directories, etc.)",
    )
    parser.add_argument(
        "--transport", type=str, choices=["stdio", "sse"], default="stdio", help="Transport type for MCP server (default: stdio)"
    )
    parser.add_argument("--port", type=int, default=3101, help="Port for SSE transport (default: 3101)")
    args = parser.parse_args()

    # Set transport type and port
    TRANSPORT_TYPE = args.transport
    SERVER_PORT = args.port

    # Determine config path with precedence: --config flag > AST_GREP_CONFIG env > None
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Config file '{args.config}' does not exist")
            sys.exit(1)
        CONFIG_PATH = args.config
    elif os.environ.get("AST_GREP_CONFIG"):
        env_config = os.environ.get("AST_GREP_CONFIG")
        if env_config and not os.path.exists(env_config):
            print(f"Error: Config file '{env_config}' specified in AST_GREP_CONFIG does not exist")
            sys.exit(1)
        CONFIG_PATH = env_config

    # This may be either an executable path or a command prefix such as
    # "uv run ast-grep". run_command expands it without invoking a shell.
    AST_GREP_COMMAND = os.environ.get("AST_GREP_PATH", "ast-grep")


# Initialize MCP server
mcp = MCPServer("ast-grep")

DumpFormat = Literal["pattern", "cst", "ast"]


def register_mcp_tools() -> None:
    @mcp.tool()
    def dump_syntax_tree(
        code: str = Field(description="The code you need"),
        language: str = Field(description=f"The language of the code. Supported: {', '.join(get_supported_languages())}"),
        format: DumpFormat = Field(description="Code dump format. Available values: pattern, ast, cst", default="cst"),
    ) -> str:
        """
        Dump code's syntax structure or dump a query's pattern structure.
        This is useful to discover correct syntax kind and syntax tree structure. Call it when debugging a rule.
        The tool requires three arguments: code, language and format. The first two are self-explanatory.
        `format` is the output format of the syntax tree.
        use `format=cst` to inspect the code's concrete syntax tree structure, useful to debug target code.
        use `format=pattern` to inspect how ast-grep interprets a pattern, useful to debug pattern rule.

        Internally calls: ast-grep run --pattern <code> --lang <language> --debug-query=<format>
        """
        result = run_ast_grep("run", ["--pattern", code, "--lang", language, f"--debug-query={format}"])
        return result.stderr.strip()  # type: ignore[no-any-return]

    @mcp.tool()
    def test_match_code_rule(
        code: str = Field(description="The code to test against the rule"),
        yaml: str = Field(description="The ast-grep YAML rule to search. It must have id, language, rule fields."),
    ) -> List[dict[str, Any]]:
        """
        Test a code against an ast-grep YAML rule.
        This is useful to test a rule before using it in a project.

        Internally calls: ast-grep scan --inline-rules <yaml> --json --stdin
        """
        result = run_ast_grep("scan", ["--inline-rules", yaml, "--json", "--stdin"], input_text=code)
        matches = json.loads(result.stdout.strip())
        if not matches:
            raise ValueError("No matches found for the given code and rule. Try adding `stopBy: end` to your inside/has rule.")
        return matches  # type: ignore[no-any-return]

    @mcp.tool()
    def find_code(
        project_folder: str = Field(description="The absolute path to the project folder. It must be absolute path."),
        pattern: str = Field(description="The ast-grep pattern to search for. Note, the pattern must have valid AST structure."),
        language: str = Field(
            description=f"The language of the code. Supported: {', '.join(get_supported_languages())}. "
            "If not specified, will be auto-detected based on file extensions.",
            default="",
        ),
        max_results: int = Field(default=0, description="Maximum results to return"),
        output_format: str = Field(default="text", description="'text' or 'json'"),
    ) -> str | List[dict[str, Any]]:
        """
        Find code in a project folder that matches the given ast-grep pattern.
        Pattern is good for simple and single-AST node result.
        For more complex usage, please use YAML by `find_code_by_rule`.

        Internally calls: ast-grep run --pattern <pattern> [--json] <project_folder>

        Output formats:
        - text (default): Compact text format with file:line-range headers and complete match text
          Example:
            Found 2 matches:

            path/to/file.py:10-15
            def example_function():
                # function body
                return result

            path/to/file.py:20-22
            def another_function():
                pass

        - json: Full match objects with metadata including ranges, meta-variables, etc.

        The max_results parameter limits the number of complete matches returned (not individual lines).
        When limited, the header shows "Found X matches (showing first Y of Z)".

        Example usage:
          find_code(pattern="class $NAME", max_results=20)  # Returns text format
          find_code(pattern="class $NAME", output_format="json")  # Returns JSON with metadata
        """
        if output_format not in ["text", "json"]:
            raise ValueError(f"Invalid output_format: {output_format}. Must be 'text' or 'json'.")

        args = ["--pattern", pattern]
        if language:
            args.extend(["--lang", language])

        # Always get JSON internally for accurate match limiting
        result = run_ast_grep("run", args + ["--json=stream", project_folder])
        matches, total_matches = parse_matches(result.stdout, max_results)

        if output_format == "text":
            if not matches:
                return "No matches found"
            text_output = format_matches_as_text(matches)
            header = f"Found {len(matches)} matches"
            if max_results and total_matches > max_results:
                header += f" (showing first {max_results} of {total_matches})"
            return header + ":\n\n" + text_output
        return matches  # type: ignore[no-any-return]

    @mcp.tool()
    def find_code_by_rule(
        project_folder: str = Field(description="The absolute path to the project folder. It must be absolute path."),
        yaml: str = Field(description="The ast-grep YAML rule to search. It must have id, language, rule fields."),
        max_results: int = Field(default=0, description="Maximum results to return"),
        output_format: str = Field(default="text", description="'text' or 'json'"),
    ) -> str | List[dict[str, Any]]:
        """
        Find code using ast-grep's YAML rule in a project folder.
        YAML rule is more powerful than simple pattern and can perform complex search like find AST inside/having another AST.
        It is a more advanced search tool than the simple `find_code`.

        Tip: When using relational rules (inside/has), add `stopBy: end` to ensure complete traversal.

        Internally calls: ast-grep scan --inline-rules <yaml> [--json] <project_folder>

        Output formats:
        - text (default): Compact text format with file:line-range headers and complete match text
          Example:
            Found 2 matches:

            src/models.py:45-52
            class UserModel:
                def __init__(self):
                    self.id = None
                    self.name = None

            src/views.py:12
            class SimpleView: pass

        - json: Full match objects with metadata including ranges, meta-variables, etc.

        The max_results parameter limits the number of complete matches returned (not individual lines).
        When limited, the header shows "Found X matches (showing first Y of Z)".

        Example usage:
          find_code_by_rule(yaml="id: x\\nlanguage: python\\nrule: {pattern: 'class $NAME'}", max_results=20)
          find_code_by_rule(yaml="...", output_format="json")  # For full metadata
        """
        if output_format not in ["text", "json"]:
            raise ValueError(f"Invalid output_format: {output_format}. Must be 'text' or 'json'.")

        args = ["--inline-rules", yaml]

        # Always get JSON internally for accurate match limiting
        result = run_ast_grep("scan", args + ["--json=stream", project_folder])
        matches, total_matches = parse_matches(result.stdout, max_results)

        if output_format == "text":
            if not matches:
                return "No matches found"
            text_output = format_matches_as_text(matches)
            header = f"Found {len(matches)} matches"
            if max_results and total_matches > max_results:
                header += f" (showing first {max_results} of {total_matches})"
            return header + ":\n\n" + text_output
        return matches  # type: ignore[no-any-return]


def parse_matches(stdout: str, max_results: int = 0) -> tuple[list[dict], int]:
    """Parse JSONL (--json=stream) output with optional early exit.

    Returns (matches, total_count). Only parses JSON for kept matches;
    remaining lines are counted but not deserialized. Non-JSON lines
    (e.g. ast-grep warnings) are skipped.
    """
    matches: list[dict] = []
    total_lines = 0
    for line in stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        total_lines += 1
        if not max_results or len(matches) < max_results:
            matches.append(json.loads(line))
    return matches, total_lines


def format_matches_as_text(matches: List[dict]) -> str:
    """Convert JSON matches to LLM-friendly text format.

    Format: file:start-end followed by the complete match text.
    Matches are separated by blank lines for clarity.
    """
    if not matches:
        return ""

    output_blocks = []
    for m in matches:
        file_path = m.get("file", "")
        start_line = m.get("range", {}).get("start", {}).get("line", 0) + 1
        end_line = m.get("range", {}).get("end", {}).get("line", 0) + 1
        match_text = m.get("text", "").rstrip()

        # Format: filepath:start-end (or just :line for single-line matches)
        if start_line == end_line:
            header = f"{file_path}:{start_line}"
        else:
            header = f"{file_path}:{start_line}-{end_line}"

        output_blocks.append(f"{header}\n{match_text}")

    return "\n\n".join(output_blocks)


def get_supported_languages() -> List[str]:
    """Get all supported languages as a field description string."""
    languages = [  # https://ast-grep.github.io/reference/languages.html
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
    ]

    # Check for custom languages in config file
    # https://ast-grep.github.io/advanced/custom-language.html#register-language-in-sgconfig-yml
    if CONFIG_PATH and os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                config = yaml.safe_load(f)
                if config and "customLanguages" in config:
                    custom_langs = list(config["customLanguages"].keys())
                    languages += custom_langs
        except Exception:
            pass

    return sorted(set(languages))


_WINDOWS_BATCH_EXTENSIONS = (".cmd", ".bat", ".ps1")


def _parse_windows_batch_wrapper(wrapper_path: str) -> Optional[List[str]]:
    """Parse an npm-style .cmd/.bat wrapper to find the underlying target.

    Returns an argv prefix that can be executed directly with shell=False,
    bypassing cmd.exe argument mangling. Returns None if no target found.
    """
    try:
        with open(wrapper_path, "r", errors="replace") as f:
            content = f.read(32768)
    except (OSError, UnicodeError):
        return None
    wrapper_dir = os.path.dirname(os.path.abspath(wrapper_path))

    def _resolve_candidate(raw: str) -> Optional[str]:
        raw = raw.strip().lstrip("@").strip().strip('"').strip("'")
        if not raw:
            return None
        # Expand npm cmd.exe variables: %~dp0 -> wrapper dir, %dp0% variants.
        # Use a function replacement so Windows backslashes in the path are
        # not interpreted as regex escapes.
        sep_dir = wrapper_dir + os.sep
        raw = re.sub(r"%~dp0", lambda _: sep_dir, raw, flags=re.IGNORECASE)
        raw = re.sub(r"%dp0%", lambda _: sep_dir, raw, flags=re.IGNORECASE)
        raw = os.path.expandvars(raw)
        candidates = [raw]
        if not os.path.isabs(raw):
            candidates.append(os.path.join(wrapper_dir, raw))
            candidates.append(os.path.normpath(os.path.join(wrapper_dir, raw)))
        for cand in candidates:
            # Strip trailing cmd.exe `%*` forwarding or arguments
            cand = cand.split("%*")[0].strip().strip('"')
            if cand and os.path.isfile(cand):
                return os.path.normpath(cand)
        return None

    # 1. Look for a referenced ast-grep .exe (binary npm delegate, see issue #32)
    for match in re.finditer(r'"?([^\s"\']*?ast-grep[^"\']*?\.exe)"?', content, re.IGNORECASE):
        resolved = _resolve_candidate(match.group(1))
        if resolved and resolved.lower().endswith(".exe"):
            return [resolved]

    # 2. Look for any referenced .exe and prefer ones mentioning ast-grep/node_modules
    exe_refs: List[str] = []
    for match in re.finditer(r'"?((?:[A-Za-z]:)?[^\s"\']*?\.exe)"?', content, re.IGNORECASE):
        resolved = _resolve_candidate(match.group(1))
        if resolved:
            exe_refs.append(resolved)
    for ref in exe_refs:
        lowered = ref.lower()
        if "ast-grep" in lowered or "node_modules" in lowered:
            return [ref]
    # 3. npm JS wrapper: node.exe + script.js (e.g. @ast-grep/cli bin script).
    # Run node directly so arguments bypass cmd.exe entirely.
    node_ref: Optional[str] = None
    js_ref: Optional[str] = None
    for match in re.finditer(r'"?((?:[A-Za-z]:)?[^\s"\']*?node(?:\.exe)?)"?', content, re.IGNORECASE):
        resolved = _resolve_candidate(match.group(1))
        if resolved:
            node_ref = resolved
            break
    for match in re.finditer(r'"?([^\s"\']*?\.js)"?', content):
        resolved = _resolve_candidate(match.group(1))
        if resolved and os.path.isfile(resolved):
            js_ref = resolved
            break
    if node_ref and js_ref:
        return [node_ref, js_ref]
    if exe_refs:
        return [exe_refs[0]]
    return None


def _search_npm_layout_for_ast_grep_exe(wrapper_dir: str) -> Optional[str]:
    """Search common npm layouts for the real ast-grep binary.

    Covers global installs (<prefix>/node_modules/@ast-grep/...) and local
    installs (<project>/node_modules/.bin -> ../@ast-grep/...), including the
    per-platform optional packages (e.g. @ast-grep/cli-win32-x64-msvc).
    """
    search_roots = [wrapper_dir]
    current = os.path.abspath(wrapper_dir)
    for _ in range(5):
        parent = os.path.dirname(current)
        if parent == current:
            break
        search_roots.append(parent)
        # node_modules sits alongside .bin or the prefix root
        search_roots.append(os.path.join(parent, "node_modules"))
        current = parent

    relative_candidates = [
        os.path.join("@ast-grep", "cli", "ast-grep.exe"),
        os.path.join("@ast-grep", "cli", "bin", "ast-grep.exe"),
        os.path.join("@ast-grep", "cli-win32-x64-msvc", "ast-grep.exe"),
        os.path.join("@ast-grep", "cli-win32-ia32-msvc", "ast-grep.exe"),
        os.path.join("@ast-grep", "cli-win32-arm64-msvc", "ast-grep.exe"),
    ]
    for root in search_roots:
        base = root
        # If root already ends with node_modules, look directly inside it
        if os.path.basename(base).lower() != "node_modules":
            base = os.path.join(root, "node_modules")
        for rel in relative_candidates:
            candidate = os.path.join(base, rel)
            if os.path.isfile(candidate):
                return candidate
        # Generic fallback: any ast-grep*.exe directly under @ast-grep packages
        ast_grep_dir = os.path.join(base, "@ast-grep")
        if os.path.isdir(ast_grep_dir):
            try:
                for package in os.listdir(ast_grep_dir):
                    package_dir = os.path.join(ast_grep_dir, package)
                    if not os.path.isdir(package_dir):
                        continue
                    for entry in os.listdir(package_dir):
                        if entry.lower().startswith("ast-grep") and entry.lower().endswith(".exe"):
                            candidate = os.path.join(package_dir, entry)
                            if os.path.isfile(candidate):
                                return candidate
            except OSError:
                continue
    return None


def _resolve_windows_command(command: List[str]) -> tuple[List[str], bool]:
    """Resolve a Windows command to avoid shell=True (issue #32).

    shell=True routes arguments through cmd.exe, which mangles metacharacters
    ($, parens, &, |, newlines, ...) in patterns and --inline-rules YAML,
    causing exit code 8 failures. Whenever possible, resolve the real binary
    and run it directly with shell=False.

    Returns (argv_prefix, use_shell).
    """
    if not command:
        return command, False
    executable = command[0]
    rest = command[1:]

    # Multi-token wrapper commands (e.g. "uv run ast-grep") already run
    # without a shell; only the executable token could need resolution and
    # those wrappers are real executables, so leave them alone.
    if len(command) > 1:
        return command, False

    # Bare `ast-grep` (or an explicit path): find what it actually points to.
    resolved_path: Optional[str] = None
    if os.path.sep in executable or "/" in executable:
        if os.path.isfile(executable):
            resolved_path = executable
        elif os.path.isfile(executable + ".exe"):
            resolved_path = executable + ".exe"
    else:
        try:
            resolved_path = shutil.which(executable)
        except Exception:
            resolved_path = None
        if resolved_path is None:
            # An .exe may exist on PATH even when the .cmd shadows the name
            try:
                resolved_path = shutil.which(executable + ".exe")
            except Exception:
                resolved_path = None

    if resolved_path is None:
        # Keep legacy behavior (shell=True only for bare `ast-grep`, as
        # before) so a missing binary still surfaces the familiar "not
        # found" error path instead of a new failure mode.
        return command, command == ["ast-grep"]

    if not resolved_path.lower().endswith(_WINDOWS_BATCH_EXTENSIONS):
        # Already a directly-executable file (cargo install, pipx, etc.).
        # Run without a shell so cmd.exe cannot mangle our arguments.
        return [resolved_path] + rest, False

    # Batch wrapper (npm install): prefer a sibling .exe first.
    sibling_exe = os.path.splitext(resolved_path)[0] + ".exe"
    if os.path.isfile(sibling_exe):
        return [sibling_exe] + rest, False

    # An explicitly-named .exe elsewhere on PATH.
    try:
        exe_on_path = shutil.which(os.path.splitext(os.path.basename(resolved_path))[0] + ".exe")
    except Exception:
        exe_on_path = None
    if exe_on_path and os.path.isfile(exe_on_path) and not exe_on_path.lower().endswith(_WINDOWS_BATCH_EXTENSIONS):
        return [exe_on_path] + rest, False

    # Parse the wrapper for its underlying target (binary or node+js).
    parsed = _parse_windows_batch_wrapper(resolved_path)
    if parsed:
        return parsed + rest, False

    # Search common npm layouts for the platform binary.
    found = _search_npm_layout_for_ast_grep_exe(os.path.dirname(resolved_path))
    if found:
        return [found] + rest, False

    # Last resort: legacy shell=True behavior for bare `ast-grep` (works for
    # simple patterns, still broken for metacharacters, but better than
    # failing to launch). Other commands keep shell=False as before.
    return command, command == ["ast-grep"]


def run_command(args: List[str], input_text: Optional[str] = None) -> subprocess.CompletedProcess:
    try:
        # A configured executable can include a wrapper command, such as
        # "uv run ast-grep". Parse only that command prefix so ast-grep's
        # arguments remain separate from the configured command string.
        command = shlex.split(args[0], posix=sys.platform != "win32")
        if sys.platform == "win32":
            # In non-POSIX mode shlex preserves wrapping quotes; subprocess
            # expects the executable token itself without them.
            command = [part[1:-1] if len(part) >= 2 and part[0] == part[-1] == '"' else part for part in command]
        if not command:
            raise RuntimeError("AST_GREP_PATH must not be empty")

        is_ast_grep_run = len(args) >= 2 and args[1] == "run"

        if sys.platform == "win32":
            command, use_shell = _resolve_windows_command(command)
        else:
            use_shell = False
        args = command + args[1:]
        need_check = not is_ast_grep_run

        result = subprocess.run(
            args,
            capture_output=True,
            input=input_text,
            text=True,
            check=need_check,  # Don't raise on non-zero exit code, handle it manually
            shell=use_shell,
        )

        # ast-grep returns exit code 1 when no matches are found, but this is not an error.
        # Only raise an exception for actual errors (exit code != 0 and != 1)
        # or when exit code is 1 but stdout doesn't look like valid JSON output
        if result.returncode != 0:
            if result.returncode == 1:
                stdout_stripped = result.stdout.strip()

                # Valid "no matches" cases: empty JSON array or valid JSON with matches
                if stdout_stripped in ("", "[]") or stdout_stripped.startswith("[") or stdout_stripped.startswith("{"):
                    return result

                # If --json flag is not present, empty stdout is also valid "no matches"
                if "--json" not in args and stdout_stripped == "":
                    return result

            # For all other non-zero exit codes, raise an error
            stderr_msg = result.stderr.strip() if result.stderr else "(no error output)"
            error_msg = f"Command {args} failed with exit code {result.returncode}: {stderr_msg}"
            raise RuntimeError(error_msg)

        return result
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.strip() if e.stderr else "(no error output)"
        error_msg = f"Command {e.cmd} failed with exit code {e.returncode}: {stderr_msg}"
        raise RuntimeError(error_msg) from e
    except FileNotFoundError as e:
        error_msg = f"Command '{args[0]}' not found. Please ensure {args[0]} is installed and in PATH."
        raise RuntimeError(error_msg) from e
    except ValueError as e:
        raise RuntimeError(f"Invalid command in AST_GREP_PATH: {e}") from e


def run_ast_grep(command: str, args: List[str], input_text: Optional[str] = None) -> subprocess.CompletedProcess:
    if CONFIG_PATH:
        args = ["--config", CONFIG_PATH] + args
    ast_grep_command = AST_GREP_COMMAND or "ast-grep"
    return run_command([ast_grep_command, command] + args, input_text)


def run_mcp_server() -> None:
    """
    Run the MCP server.
    This function is used to start the MCP server when this script is run directly.
    """
    parse_args_and_get_config()  # sets CONFIG_PATH, TRANSPORT_TYPE, and SERVER_PORT
    register_mcp_tools()  # tools defined *after* CONFIG_PATH is known
    if TRANSPORT_TYPE == "sse":
        mcp.run(transport="sse", port=SERVER_PORT)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    run_mcp_server()
