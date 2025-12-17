import argparse
import json
import os
import subprocess
import sys
from typing import Any, List, Literal, Optional

import yaml
from mcp.server.fastmcp import FastMCP
from pydantic import Field

# Global variable for config path (will be set by parse_args_and_get_config)
CONFIG_PATH = None

def parse_args_and_get_config():
    """Parse command-line arguments and determine config path."""
    global CONFIG_PATH

    # Determine how the script was invoked
    prog = None
    if sys.argv[0].endswith('main.py'):
        # Direct execution: python main.py
        prog = 'python main.py'

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        prog=prog,
        description='ast-grep MCP Server - Provides structural code search capabilities via Model Context Protocol',
        epilog='''
environment variables:
  AST_GREP_CONFIG    Path to sgconfig.yaml file (overridden by --config flag)

For more information, see: https://github.com/ast-grep/ast-grep-mcp
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        metavar='PATH',
        help='Path to sgconfig.yaml file for customizing ast-grep behavior (language mappings, rule directories, etc.)'
    )
    args = parser.parse_args()

    # Determine config path with precedence: --config flag > AST_GREP_CONFIG env > None
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Config file '{args.config}' does not exist")
            sys.exit(1)
        CONFIG_PATH = args.config
    elif os.environ.get('AST_GREP_CONFIG'):
        env_config = os.environ.get('AST_GREP_CONFIG')
        if env_config and not os.path.exists(env_config):
            print(f"Error: Config file '{env_config}' specified in AST_GREP_CONFIG does not exist")
            sys.exit(1)
        CONFIG_PATH = env_config

# Initialize FastMCP server
mcp = FastMCP("ast-grep")

DumpFormat = Literal["pattern", "cst", "ast"]

def register_mcp_tools() -> None:
    @mcp.tool()
    def dump_syntax_tree(
        code: str = Field(description = "The code you need"),
        language: str = Field(description = f"The language of the code. Supported: {', '.join(get_supported_languages())}"),
        format: DumpFormat = Field(description = "Code dump format. Available values: pattern, ast, cst", default = "cst"),
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
        code: str = Field(description = "The code to test against the rule"),
        yaml: str = Field(description = "The ast-grep YAML rule to search. It must have id, language, rule fields."),
    ) -> List[dict[str, Any]]:
        """
        Test a code against an ast-grep YAML rule.
        This is useful to test a rule before using it in a project.

        Internally calls: ast-grep scan --inline-rules <yaml> --json --stdin
        """
        result = run_ast_grep("scan", ["--inline-rules", yaml, "--json", "--stdin"], input_text = code)
        matches = json.loads(result.stdout.strip())
        if not matches:
            raise ValueError("No matches found for the given code and rule. Try adding `stopBy: end` to your inside/has rule.")
        return matches  # type: ignore[no-any-return]

    @mcp.tool()
    def find_code(
        project_folder: str = Field(description = "The absolute path to the project folder. It must be absolute path."),
        pattern: str = Field(description = "The ast-grep pattern to search for. Note, the pattern must have valid AST structure."),
        language: str = Field(description = f"The language of the code. Supported: {', '.join(get_supported_languages())}. "
                                           "If not specified, will be auto-detected based on file extensions.", default = ""),
        max_results: int = Field(default = 0, description = "Maximum results to return"),
        output_format: str = Field(default = "text", description = "'text' or 'json'"),
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
        result = run_ast_grep("run", args + ["--json", project_folder])
        matches = json.loads(result.stdout.strip() or "[]")

        # Apply max_results limit to complete matches
        total_matches = len(matches)
        if max_results and total_matches > max_results:
            matches = matches[:max_results]

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
        project_folder: str = Field(description = "The absolute path to the project folder. It must be absolute path."),
        yaml: str = Field(description = "The ast-grep YAML rule to search. It must have id, language, rule fields."),
        max_results: int = Field(default = 0, description = "Maximum results to return"),
        output_format: str = Field(default = "text", description = "'text' or 'json'"),
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
        result = run_ast_grep("scan", args + ["--json", project_folder])
        matches = json.loads(result.stdout.strip() or "[]")

        # Apply max_results limit to complete matches
        total_matches = len(matches)
        if max_results and total_matches > max_results:
            matches = matches[:max_results]

        if output_format == "text":
            if not matches:
                return "No matches found"
            text_output = format_matches_as_text(matches)
            header = f"Found {len(matches)} matches"
            if max_results and total_matches > max_results:
                header += f" (showing first {max_results} of {total_matches})"
            return header + ":\n\n" + text_output
        return matches  # type: ignore[no-any-return]


def format_matches_as_text(matches: List[dict]) -> str:
    """Convert JSON matches to LLM-friendly text format.

    Format: file:start-end followed by the complete match text.
    Matches are separated by blank lines for clarity.
    """
    if not matches:
        return ""

    output_blocks = []
    for m in matches:
        file_path = m.get('file', '')
        start_line = m.get('range', {}).get('start', {}).get('line', 0) + 1
        end_line = m.get('range', {}).get('end', {}).get('line', 0) + 1
        match_text = m.get('text', '').rstrip()

        # Format: filepath:start-end (or just :line for single-line matches)
        if start_line == end_line:
            header = f"{file_path}:{start_line}"
        else:
            header = f"{file_path}:{start_line}-{end_line}"

        output_blocks.append(f"{header}\n{match_text}")

    return '\n\n'.join(output_blocks)

def get_supported_languages() -> List[str]:
    """Get all supported languages as a field description string."""
    languages = [  # https://ast-grep.github.io/reference/languages.html
        "bash", "c", "cpp", "csharp", "css", "elixir", "go", "haskell",
        "html", "java", "javascript", "json", "jsx", "kotlin", "lua",
        "nix", "php", "python", "ruby", "rust", "scala", "solidity",
        "swift", "tsx", "typescript", "yaml"
    ]

    # Check for custom languages in config file
    # https://ast-grep.github.io/advanced/custom-language.html#register-language-in-sgconfig-yml
    if CONFIG_PATH and os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f)
                if config and 'customLanguages' in config:
                    custom_langs = list(config['customLanguages'].keys())
                    languages += custom_langs
        except Exception:
            pass

    return sorted(set(languages))

def run_command(args: List[str], input_text: Optional[str] = None) -> subprocess.CompletedProcess:
    try:
        # On Windows, if ast-grep is installed via npm, it's a batch file
        # that requires shell=True to execute properly
        use_shell = (sys.platform == "win32" and args[0] == "ast-grep")
        need_check = len(args) < 2 or  args[1] != "run"

        result = subprocess.run(
            args,
            capture_output=True,
            input=input_text,
            text=True,
            check=need_check,  # Don't raise on non-zero exit code, handle it manually
            shell=use_shell
        )

        # ast-grep returns exit code 1 when no matches are found, but this is not an error.
        # Only raise an exception for actual errors (exit code != 0 and != 1)
        # or when exit code is 1 but stdout doesn't look like valid JSON output
        if result.returncode != 0:
            if result.returncode == 1:
                stdout_stripped = result.stdout.strip()

                # Valid "no matches" cases: empty JSON array or valid JSON with matches
                if stdout_stripped in ("", "[]") or stdout_stripped.startswith("["):
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

def run_ast_grep(command:str, args: List[str], input_text: Optional[str] = None) -> subprocess.CompletedProcess:
    if CONFIG_PATH:
        args = ["--config", CONFIG_PATH] + args
    return run_command(["ast-grep", command] + args, input_text)

def run_mcp_server() -> None:
    """
    Run the MCP server.
    This function is used to start the MCP server when this script is run directly.
    """
    parse_args_and_get_config()  # sets CONFIG_PATH
    register_mcp_tools()  # tools defined *after* CONFIG_PATH is known
    mcp.run(transport="stdio")

if __name__ == "__main__":
    run_mcp_server()
