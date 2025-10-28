import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, cast

import yaml
from mcp.server.fastmcp import FastMCP
from pydantic import Field

# Global variable for config path (will be set by parse_args_and_get_config)
CONFIG_PATH = None

# ============================================================================
# Input Validation Functions
# ============================================================================


def validate_project_folder(path: str) -> Path:
    """
    Validate project folder is absolute, exists, and is a directory.

    Args:
        path: Path to validate

    Returns:
        Path object if valid

    Raises:
        ValueError: With actionable error message for LLMs
    """
    if not path:
        raise ValueError("project_folder cannot be empty")

    if not os.path.isabs(path):
        raise ValueError(
            f"project_folder must be an absolute path, got: {path}\n"
            f"Tip: Use absolute paths like '/home/user/project' or '/Users/user/project'"
        )

    path_obj = Path(path)
    if not path_obj.exists():
        raise ValueError(f"project_folder does not exist: {path}")

    if not path_obj.is_dir():
        raise ValueError(f"project_folder must be a directory, got file: {path}")

    return path_obj


def validate_pattern(pattern: str) -> str:
    """
    Validate ast-grep pattern is non-empty and within reasonable length.

    Args:
        pattern: Pattern to validate

    Returns:
        Stripped pattern if valid

    Raises:
        ValueError: With actionable error message
    """
    if not pattern or not pattern.strip():
        raise ValueError(
            "pattern cannot be empty or whitespace\n"
            "Example patterns: 'console.log($$$)', 'def $NAME($$$)', 'class $CLASS'"
        )

    pattern = pattern.strip()

    if len(pattern) > 10000:
        raise ValueError(
            f"pattern is too long ({len(pattern)} chars). Maximum: 10000 chars\n"
            "Tip: Break complex patterns into multiple searches"
        )

    return pattern


def validate_language(language: str) -> str:
    """
    Validate language is supported by ast-grep.

    Args:
        language: Language identifier (e.g., 'python', 'javascript')

    Returns:
        Language string if valid (or empty string for auto-detection)

    Raises:
        ValueError: With helpful suggestions
    """
    if not language:
        return ""  # Empty is valid (auto-detection)

    language = language.lower().strip()
    supported = get_supported_languages()

    if language not in supported:
        # Try to find similar languages for helpful error message
        close_matches = [lang for lang in supported if language in lang or lang in language]

        error_msg = f"Unsupported language: '{language}'\n"

        if close_matches:
            error_msg += f"Did you mean: {', '.join(close_matches[:3])}?\n"

        error_msg += "Supported languages include: python, javascript, typescript, rust, go, java\n"
        error_msg += "Tip: Leave empty for auto-detection based on file extensions"

        raise ValueError(error_msg)

    return language


def validate_yaml_rule(yaml_str: str) -> Dict[str, Any]:
    """
    Validate YAML rule has required structure for ast-grep.

    Args:
        yaml_str: YAML rule string

    Returns:
        Parsed YAML dict if valid

    Raises:
        ValueError: With specific parsing or structure errors
    """
    if not yaml_str or not yaml_str.strip():
        raise ValueError(
            "yaml rule cannot be empty\n"
            "Example:\n"
            "  id: my-rule\n"
            "  language: python\n"
            "  rule:\n"
            "    pattern: 'def $NAME($$$)'"
        )

    if len(yaml_str) > 50000:
        raise ValueError(
            f"yaml rule is too long ({len(yaml_str)} chars). Maximum: 50000 chars"
        )

    # Parse YAML
    try:
        parsed = yaml.safe_load(yaml_str)
    except yaml.YAMLError as e:
        raise ValueError(
            f"Invalid YAML syntax: {e}\n"
            "Tip: Check indentation and formatting"
        )

    # Validate required fields
    if not isinstance(parsed, dict):
        raise ValueError("YAML rule must be a dictionary/object")

    required_fields = ['id', 'language', 'rule']
    missing = [f for f in required_fields if f not in parsed]

    if missing:
        raise ValueError(
            f"YAML rule missing required fields: {', '.join(missing)}\n"
            f"Required fields: {', '.join(required_fields)}\n"
            "Example:\n"
            "  id: my-rule\n"
            "  language: python\n"
            "  rule:\n"
            "    pattern: 'def $NAME($$$)'"
        )

    return cast(Dict[str, Any], parsed)


def validate_code_snippet(code: str) -> str:
    """
    Validate code snippet is non-empty and within reasonable length.

    Args:
        code: Code snippet to validate

    Returns:
        Code string if valid

    Raises:
        ValueError: With actionable error message
    """
    if not code or not code.strip():
        raise ValueError(
            "code cannot be empty or whitespace\n"
            "Provide a code snippet to analyze or test against rules"
        )

    if len(code) > 100000:
        raise ValueError(
            f"code snippet is too long ({len(code)} chars). Maximum: 100000 chars\n"
            "Tip: Use project folder search for large files"
        )

    return code


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
        code: str = Field(description="Code snippet or pattern to analyze. Examples: 'def foo(): pass', 'console.log(x)'"),
        language: str = Field(description="Programming language (e.g., python, javascript, rust). Required."),
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
        # Validate inputs
        code = validate_code_snippet(code)
        language = validate_language(language)
        if not language:
            raise ValueError("language parameter is required for dump_syntax_tree")

        result = run_ast_grep("run", ["--pattern", code, "--lang", language, f"--debug-query={format}"])
        return result.stderr.strip()  # type: ignore[no-any-return]

    @mcp.tool()
    def test_match_code_rule(
        code: str = Field(description="Code snippet to test against the rule. Can be a few lines or complete file."),
        yaml: str = Field(description="ast-grep YAML rule with id, language, and rule fields."),
    ) -> List[dict[str, Any]]:
        """
        Test a code against an ast-grep YAML rule.
        This is useful to test a rule before using it in a project.

        Internally calls: ast-grep scan --inline-rules <yaml> --json --stdin
        """
        # Validate inputs
        code = validate_code_snippet(code)
        validate_yaml_rule(yaml)  # Validates structure and required fields

        result = run_ast_grep("scan", ["--inline-rules", yaml, "--json", "--stdin"], input_text=code)
        matches = json.loads(result.stdout.strip())
        if not matches:
            raise ValueError(
                "No matches found for the given code and rule.\n"
                "Common issues:\n"
                "- Try adding `stopBy: end` to your inside/has relational rules\n"
                "- Use dump_syntax_tree to verify your code's AST structure\n"
                "- Check that your pattern is valid for the target language"
            )
        return matches  # type: ignore[no-any-return]

    @mcp.tool()
    def find_code(
        project_folder: str = Field(description="Absolute path to the project folder to search."),
        pattern: str = Field(description="ast-grep pattern to search for. Examples: 'class $NAME', 'def $FUNC($$$)'"),
        language: str = Field(
            default="",
            description="Programming language (e.g., python, javascript, rust). Leave empty for auto-detection."
        ),
        max_results: int = Field(default=0, description="Maximum results to return (0 = unlimited)"),
        output_format: str = Field(default="text", description="'text' (compact, efficient) or 'json' (full metadata)"),
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
        # Validate inputs
        project_path = validate_project_folder(project_folder)
        pattern = validate_pattern(pattern)
        language = validate_language(language)

        if output_format not in ["text", "json"]:
            raise ValueError(
                f"Invalid output_format: '{output_format}'. Must be 'text' or 'json'\n"
                "- 'text': Compact format, ~75% fewer tokens\n"
                "- 'json': Full metadata with ranges and metavariables"
            )

        if max_results < 0:
            raise ValueError(f"max_results must be >= 0, got: {max_results}")

        args = ["--pattern", pattern]
        if language:
            args.extend(["--lang", language])

        # Always get JSON internally for accurate match limiting
        result = run_ast_grep("run", args + ["--json", str(project_path)])
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
        project_folder: str = Field(description="Absolute path to the project folder to search."),
        yaml: str = Field(description="ast-grep YAML rule with id, language, and rule fields."),
        max_results: int = Field(default=0, description="Maximum results to return (0 = unlimited)"),
        output_format: str = Field(default="text", description="'text' (compact, efficient) or 'json' (full metadata)"),
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
        # Validate inputs
        project_path = validate_project_folder(project_folder)
        validate_yaml_rule(yaml)  # Validates structure and required fields

        if output_format not in ["text", "json"]:
            raise ValueError(
                f"Invalid output_format: '{output_format}'. Must be 'text' or 'json'\n"
                "- 'text': Compact format, ~75% fewer tokens\n"
                "- 'json': Full metadata with ranges and metavariables"
            )

        if max_results < 0:
            raise ValueError(f"max_results must be >= 0, got: {max_results}")

        args = ["--inline-rules", yaml]

        # Always get JSON internally for accurate match limiting
        result = run_ast_grep("scan", args + ["--json", str(project_path)])
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

        result = subprocess.run(
            args,
            capture_output=True,
            input=input_text,
            text=True,
            check=True,  # Raises CalledProcessError if return code is non-zero
            shell=use_shell
        )
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
