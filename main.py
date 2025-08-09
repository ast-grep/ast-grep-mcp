from typing import Any, List, Optional
from mcp.server.fastmcp import FastMCP
import subprocess
from pydantic import Field
import json
from enum import Enum
import argparse
import os
import sys
import yaml

# Determine how the script was invoked
if sys.argv[0].endswith('main.py'):
    # Direct execution: python main.py
    prog = 'python main.py'
else:
    # Installed script execution (via uvx, pip install, etc.)
    prog = None  # Let argparse use the default

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
CONFIG_PATH = None
if args.config:
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' does not exist")
        sys.exit(1)
    CONFIG_PATH = args.config
elif os.environ.get('AST_GREP_CONFIG'):
    env_config = os.environ.get('AST_GREP_CONFIG')
    if not os.path.exists(env_config):
        print(f"Error: Config file '{env_config}' specified in AST_GREP_CONFIG does not exist")
        sys.exit(1)
    CONFIG_PATH = env_config

# Initialize FastMCP server
mcp = FastMCP("ast-grep")

class DumpFormat(Enum):
    Pattern = "pattern"
    CST = "cst"
    AST = "ast"

@mcp.tool()
def dump_syntax_tree(
    code: str = Field(description = "The code you need"),
    language: str = Field(description = "The language of the code"),
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
    return run_ast_grep_dump(code, language, format.value)

@mcp.tool()
def get_supported_languages() -> List[str]:
    """
    Get list of languages supported.

    Returns common language identifiers that can be used in the 'language' parameter of other tools.
    """
    base_languages = [  # https://ast-grep.github.io/reference/languages.html
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
        "yaml"
    ]

    # Check for custom languages in config file
    # https://ast-grep.github.io/advanced/custom-language.html#register-language-in-sgconfig-yml
    if CONFIG_PATH and os.path.exists(CONFIG_PATH):
        try:
            import yaml
            with open(CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f)
                if config and 'customLanguages' in config:
                    custom_langs = list(config['customLanguages'].keys())
                    return sorted(set(base_languages + custom_langs))
        except Exception:
            pass

    return base_languages

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
    args = ["ast-grep", "scan","--inline-rules", yaml, "--json", "--stdin"]
    if CONFIG_PATH:
        args.extend(["--config", CONFIG_PATH])
    try:
        # Run command and capture output
        result = subprocess.run(
            args,
            capture_output=True,
            input=code,
            text=True,
            check=True  # Raises CalledProcessError if return code is non-zero
        )
        matches = json.loads(result.stdout.strip())
        if not matches:
            raise ValueError("No matches found for the given code and rule. Try adding `stopBy: end` to your inside/has rule.")
        return matches
    except subprocess.CalledProcessError as e:
        return e.stderr

@mcp.tool()
def find_code(
    project_folder: str = Field(description="The absolute path to the project folder. It must be absolute path."),
    pattern: str = Field(description="The ast-grep pattern to search for. Note, the pattern must have valid AST structure."),
    language: str = Field(description="The language of the query", default=""),
) -> List[dict[str, Any]]:
    """
    Find code in a project folder that matches the given ast-grep pattern.
    Pattern is good for simple and single-AST node result.
    For more complex usage, please use YAML by `find_code_by_rule`.

    Internally calls: ast-grep run --pattern <pattern> --json <project_folder>
    """
    return run_ast_grep_command(pattern, project_folder, language)

@mcp.tool()
def find_code_by_rule(
    project_folder: str = Field(description="The absolute path to the project folder. It must be absolute path."),
    yaml: str = Field(description="The ast-grep YAML rule to search. It must have id, language, rule fields."),
    ) -> List[dict[str, Any]]:
    """
    Find code using ast-grep's YAML rule in a project folder.
    YAML rule is more powerful than simple pattern and can perform complex search like find AST inside/having another AST.
    It is a more advanced search tool than the simple `find_code`.

    Tip: When using relational rules (inside/has), add `stopBy: end` to ensure complete traversal.
    
    Internally calls: ast-grep scan --inline-rules <yaml> --json <project_folder>
    """
    return run_ast_grep_yaml(yaml, project_folder)

def run_ast_grep_dump(code: str, language: str, format: str) -> str:
    args = ["ast-grep", "run", "--pattern", code, "--lang", language, f"--debug-query={format}"]
    if CONFIG_PATH:
        args.extend(["--config", CONFIG_PATH])
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True  # Raises CalledProcessError if return code is non-zero
        )
        return result.stderr.strip()  # Return the output of the command
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print("Error output:", e.stderr)
        return e.stderr.strip()

def run_ast_grep_command(pattern: str, project_folder: str, language: Optional[str]) -> List[dict[str, Any]]:
    try:
        args = ["ast-grep", "run", "--pattern", pattern, "--json"]
        if CONFIG_PATH:
            args.extend(["--config", CONFIG_PATH])
        if language:
            args.extend(["--lang", language])
        args.append(project_folder)
        # Run command and capture output
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True  # Raises CalledProcessError if return code is non-zero
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print("Error output:", e.stderr)
        return e.stderr
    except FileNotFoundError:
        print("Command not found")
        return []

def run_ast_grep_yaml(yaml: str, project_folder: str) -> List[dict[str, Any]]:
    try:
        args = ["ast-grep", "scan", "--inline-rules", yaml, "--json"]
        if CONFIG_PATH:
            args.extend(["--config", CONFIG_PATH])
        args.append(project_folder)
        # Run command and capture output
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True  # Raises CalledProcessError if return code is non-zero
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print("Error output:", e.stderr)
        return e.stderr
    except FileNotFoundError:
        print("Command not found")
        return []

def run_mcp_server() -> None:
    """
    Run the MCP server.
    This function is used to start the MCP server when this script is run directly.
    """
    mcp.run(transport="stdio")

if __name__ == "__main__":
    run_mcp_server()
