from typing import Any, List, Optional, Dict, Union
from fastmcp import FastMCP, Context
import subprocess
from pydantic import Field
import json
import os
import tempfile
import glob
import pathlib
import yaml as pyyaml # Import pyyaml for create_new_rule

# Initialize FastMCP server - capabilities handled differently or default?
# FastMCP might auto-detect capabilities based on registered handlers.
# Let's remove explicit capabilities for now, can add back if needed.
mcp = FastMCP("ast-grep")

RULES_DIR = "sg_rules"  # Define a standard directory for rules

# Helper function to ensure rules directory exists
def ensure_rules_dir(project_folder: str) -> pathlib.Path:
    # Check if project_folder is valid
    proj_path = pathlib.Path(project_folder)
    if not proj_path.is_dir():
        # Use ValueError for invalid parameters
        raise ValueError(f"Project folder '{project_folder}' not found or is not a directory.")
    rules_path = proj_path / RULES_DIR
    try:
        rules_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # Use RuntimeError for internal/OS errors
        raise RuntimeError(f"Could not create rules directory '{rules_path}': {e}")
    return rules_path

# --- Tool Implementation (Decorators likely remain the same) --- #

@mcp.tool()
def find_code(
    project_folder: str = Field(description="The path to the project folder"),
    pattern: str = Field(description="The ast-grep pattern to search for. Note the pattern must has valid AST structure."),
    language: str = Field(description="The language of the query", default=""),
) -> List[dict[str, Any]]:
    """
    Find code in a project folder that matches the given ast-grep pattern.
    Pattern is good for simple and single-AST node result.
    For more complex usage, please use YAML by `find_code_by_rule`.
    """
    return run_ast_grep_command(pattern, project_folder, language)

@mcp.tool()
def find_code_by_rule(
    project_folder: str = Field(description="The path to the project folder"),
    yaml: str = Field(description="The ast-grep YAML rule to search. It must have id, language, rule fields."),
    ) -> List[dict[str, Any]]:
    """
    Find code using ast-grep's YAML rule in a project folder.
    YAML rule is more powerful than simple pattern and can perform complex search like find AST inside/having another AST.
    It is a more advanced search tool than the simple `find_code`.
    """
    return run_ast_grep_yaml(yaml, project_folder)

# --- Subprocess Execution Helpers (No changes needed for library migration) --- #

def run_ast_grep_command(pattern: str, project_folder: str, language: Optional[str]) -> List[dict[str, Any]]:
    # Check project folder exists before running subprocess
    if not pathlib.Path(project_folder).is_dir():
        raise ValueError(f"Project folder '{project_folder}' not found.")
    try:
        args = ["ast-grep", "--pattern", pattern, "--json", project_folder]
        if language:
            args.extend(["--lang", language])
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False # Let json_processor handle errors
        )
        processed_output = json_processor(result)
        if isinstance(processed_output, str): # Check if json_processor returned an error string
             raise RuntimeError(processed_output)
        return processed_output
    except FileNotFoundError:
        raise RuntimeError("'ast-grep' command not found. Ensure it is installed and in PATH.")
    except Exception as e:
        # Catch other potential errors during subprocess run
        raise RuntimeError(f"Error running find_code: {e}")

def run_ast_grep_yaml(yaml: str, project_folder: str) -> List[dict[str, Any]]:
    # Check project folder exists
    if not pathlib.Path(project_folder).is_dir():
        raise ValueError(f"Project folder '{project_folder}' not found.")
    try:
        args = ["ast-grep", "scan","--inline-rules", yaml, "--json", project_folder]
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False # Let json_processor handle errors
        )
        processed_output = json_processor(result)
        if isinstance(processed_output, str): # Check if json_processor returned an error string
             raise RuntimeError(processed_output)
        return processed_output
    except FileNotFoundError:
        raise RuntimeError("'ast-grep' command not found. Ensure it is installed and in PATH.")
    except Exception as e:
        raise RuntimeError(f"Error running find_code_by_rule: {e}")


@mcp.tool()
def get_supported_languages() -> List[str]:
    """Returns supported languages list"""
    args = ["ast-grep", "language", "--list"]
    return handle_subprocess(args, list_processor)

@mcp.tool()
def get_language_kinds(language: str) -> List[str]:
    """Returns AST node kinds for a language"""
    args = ["ast-grep", "language", "--kinds", language]
    return handle_subprocess(args, list_processor)

@mcp.tool()
def dump_ast(code: str, language: str) -> Dict[str, Any]:
    """Dumps AST structure of code"""
    # Ensure language is alphanumeric for file extension safety
    safe_lang = ''.join(filter(str.isalnum, language)) or 'txt'
    with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{safe_lang}', delete=True) as f:
        f.write(code)
        temp_path = f.name
        # Ensure file is flushed before ast-grep reads it
        f.flush()
        os.fsync(f.fileno())
        args = ["ast-grep", "dump", "--json", temp_path]
        # Pass temp_path=None to handle_subprocess as it's auto-deleted
        return handle_subprocess(args, json_processor, cleanup=None)

@mcp.tool()
def validate_rule_syntax(rule: str) -> Dict[str, Any]:
    """Validates YAML rule syntax"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=True) as f:
        f.write(rule)
        temp_path = f.name
        f.flush()
        os.fsync(f.fileno())
        args = ["ast-grep", "scan", "--validate", temp_path, "--no-ignore"]
        return handle_subprocess(args, validation_processor, cleanup=None)

@mcp.tool()
def list_rules(project_folder: str = Field(description="The path to the project folder")) -> List[str]:
    """Lists YAML rule files in the project's sg_rules directory."""
    try:
        rules_path = ensure_rules_dir(project_folder)
        rule_files = [f.name for f in rules_path.glob('*.yml')]
        rule_files.extend([f.name for f in rules_path.glob('*.yaml')])
        return rule_files
    except (ValueError, RuntimeError) as e: # Catch specific errors from ensure_rules_dir
        raise e
    except Exception as e:
        # Wrap other exceptions
        raise RuntimeError(f"Error listing rules: {e}")

@mcp.tool()
def get_rule(project_folder: str = Field(description="The path to the project folder"),
             rule_name: str = Field(description="The name of the rule file (e.g., my-rule.yml)")) -> str:
    """Gets the content of a specific YAML rule file from the project's sg_rules directory."""
    try:
        rules_path = ensure_rules_dir(project_folder)
        rule_file_path = rules_path / rule_name
        if not rule_file_path.is_file():
            raise FileNotFoundError(f"Rule file '{rule_name}' not found in {rules_path}.")
        with open(rule_file_path, 'r') as f:
            return f.read()
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        raise e
    except Exception as e:
        raise RuntimeError(f"Error reading rule file '{rule_name}': {e}")

@mcp.tool()
def create_new_rule(project_folder: str = Field(description="The path to the project folder"),
                    rule_name: str = Field(description="The desired name for the new rule file (e.g., new-rule.yml)"),
                    rule_content: str = Field(description="The YAML content for the new rule")) -> str:
    """Creates a new YAML rule file in the project's sg_rules directory."""
    if not (rule_name.endswith(".yml") or rule_name.endswith(".yaml")):
        raise ValueError("Rule name must end with .yml or .yaml")

    try:
        rules_path = ensure_rules_dir(project_folder)
        rule_file_path = rules_path / rule_name

        if rule_file_path.exists():
            raise ValueError(f"Rule file '{rule_name}' already exists in {rules_path}.")

        # Basic YAML validation check before writing
        try:
            pyyaml.safe_load(rule_content)
        except pyyaml.YAMLError as yaml_err:
            raise ValueError(f"Invalid YAML syntax - {yaml_err}")

        with open(rule_file_path, 'w') as f:
            f.write(rule_content)
        return f"Successfully created rule file '{rule_name}' in {rules_path}."
    except (ValueError, RuntimeError) as e:
        raise e
    except Exception as e:
        raise RuntimeError(f"Error creating rule file '{rule_name}': {e}")


@mcp.tool()
def list_all_diagnostics_in_the_workspace(project_folder: str = Field(description="The path to the project folder")) -> Union[List[dict[str, Any]], str]:
    """Runs 'sg scan --json' to find diagnostics based on rules in sg_rules directory."""
    try:
        rules_path = ensure_rules_dir(project_folder)
        rule_files = list(rules_path.glob('*.yml')) + list(rules_path.glob('*.yaml'))
        if not rule_files:
            return "No rule files found in the sg_rules directory. Cannot run scan."

        args = ["ast-grep", "scan", "-r", str(rules_path), "--json", project_folder]
        return handle_subprocess(args, json_processor)
    except (ValueError, RuntimeError) as e:
        raise e
    except Exception as e:
        raise RuntimeError(f"Error running diagnostics scan: {e}")

# --- Resource Implementation (Adapting to fastmcp conventions) --- #

# Resource to list local rules
@mcp.resource("rule://sg_rules")
async def list_local_rules_resource(project_folder: str) -> List[Dict[str, str]]:
    """Lists local rule files as resources. Requires project_folder."""
    if not project_folder:
         raise ValueError("Listing local rules requires the 'project_folder' argument.")
    try:
        rules_path = ensure_rules_dir(project_folder)
        rule_files = list(rules_path.glob('*.yml')) + list(rules_path.glob('*.yaml'))
        return [
            {
                "uri": f"rule://sg_rules/{rule_file.name}",
                "name": f"Rule: {rule_file.name}",
                # Add other resource metadata if needed by fastmcp resource model
            }
            for rule_file in rule_files
        ]
    except (ValueError, RuntimeError) as e:
        raise e
    except Exception as e:
        raise RuntimeError(f"Error listing local rules resource: {e}")

# Template Resource to read a specific local rule
@mcp.resource("rule://sg_rules/{rule_name}")
async def read_local_rule_resource(rule_name: str, project_folder: str) -> Dict[str, Any]:
    """Reads a specific local rule file. Requires project_folder."""
    if not project_folder:
         raise ValueError("Reading local rules requires the 'project_folder' argument.")
    try:
        # Validate rule_name format slightly
        if not (rule_name.endswith(".yml") or rule_name.endswith(".yaml")):
            raise ValueError("Invalid rule name format.")

        rules_path = ensure_rules_dir(project_folder)
        rule_file_path = rules_path / rule_name

        if not rule_file_path.is_file():
            raise FileNotFoundError(f"Rule file '{rule_name}' not found in {rules_path}.") # Use NotFound

        with open(rule_file_path, 'r') as f:
            content = f.read()
        return {
            "uri": f"rule://sg_rules/{rule_name}",
            "mimeType": "application/yaml",
            "content": content # fastmcp might expect 'content' instead of 'text'
        }
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        raise e
    except Exception as e:
        raise RuntimeError(f"Error reading rule file '{rule_name}': {e}")

# Static Resource for mock global rule
@mcp.resource("rule://global/example-global-rule.yml")
def global_rule_resource() -> Dict[str, Any]:
     """Provides mock content for the global rule example."""
     return {
         "uri": "rule://global/example-global-rule.yml",
         "mimeType": "application/yaml",
         "content": "# Mock Global Rule Content\nlanguage: python\nrule:\n  pattern: print($ARG)"
     }

# Static Resource for mock python test case
@mcp.resource("testcase://python/example-test.py")
def python_test_case_resource() -> Dict[str, Any]:
    """Provides mock content for the python test case example."""
    return {
        "uri": "testcase://python/example-test.py",
        "mimeType": "text/x-python",
        "content": "# Mock Python Test Case\nimport os\n\nprint(\"Hello, world!\")\n\ndef old_function(x):\n    return x * 2"
    }

# Static Resource for mock javascript test case
@mcp.resource("testcase://javascript/example-test.js")
def javascript_test_case_resource() -> Dict[str, Any]:
    """Provides mock content for the javascript test case example."""
    return {
        "uri": "testcase://javascript/example-test.js",
        "mimeType": "text/javascript",
        "content": "// Mock JavaScript Test Case\nconsole.log(\"Hello, world!\");\n\nfunction deprecatedFunc(a) {\n  return a + 1;\n}"
    }

# --- Subprocess Result Processors (Refined error propagation) --- #

def validation_processor(result: subprocess.CompletedProcess) -> Dict[str, Any]:
    if result.returncode != 0 and result.stderr:
        # Raise error if validation fails and there's stderr output
        raise RuntimeError(f"Rule validation failed: {result.stderr}")
    elif result.returncode != 0:
        raise RuntimeError(f"Rule validation failed with exit code {result.returncode}. No specific error message.")
    return {
        "valid": True,
        "message": "Valid rule",
        "error": None
    }

def json_processor(result: subprocess.CompletedProcess) -> Union[List[Dict[str, Any]], Dict[str, Any]]: # Allow single dict too
    if result.returncode == 0:
        if not result.stdout:
            return [] # Return empty list for empty output
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            # Raise an error if JSON decoding fails
            raise RuntimeError(f"Error decoding JSON output: {e}\nRaw output:\n{result.stdout}")
    else:
        error_message = f"Command failed with return code {result.returncode}."
        if result.stderr:
            error_message += f"\nError output:\n{result.stderr}"
        raise RuntimeError(error_message)

def list_processor(result: subprocess.CompletedProcess) -> List[str]:
    if result.returncode == 0:
        return result.stdout.strip().split('\n') if result.stdout else []
    else:
        raise RuntimeError(f"Command failed with return code {result.returncode}: {result.stderr}")

def error_handler(e: Exception, cleanup: Optional[str] = None) -> Exception: # Return the original or wrapped exception
    if cleanup and os.path.exists(cleanup):
        try:
            os.unlink(cleanup)
        except OSError as unlink_err:
            print(f"Warning: Failed to remove temp file '{cleanup}': {unlink_err}")

    # Re-raise standard Python exceptions directly, wrap others in RuntimeError
    if isinstance(e, (ValueError, FileNotFoundError, RuntimeError, subprocess.CalledProcessError)):
        return e
    elif isinstance(e, FileNotFoundError): # This specific FileNotFoundError check for ast-grep
         return RuntimeError("'ast-grep' command not found. Please ensure it is installed and in your PATH.")
    else:
        return RuntimeError(f"An unexpected error occurred: {e}")

# Updated handle_subprocess to raise RuntimeError instead of returning strings on error
def handle_subprocess(args, processor, cleanup=None):
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        output = processor(result) # Processor now raises RuntimeError on failure
        if cleanup and os.path.exists(cleanup):
             try: os.unlink(cleanup)
             except OSError as e: print(f"Warning: Failed to cleanup {cleanup}: {e}")
        return output
    except Exception as e:
        # Wrap exceptions from run() or processor() using error_handler
        raise error_handler(e, cleanup)

# --- Main Execution --- #

if __name__ == "__main__":
    # No need to check for pyyaml import here if it's in pyproject.toml
    mcp.run(transport = "stdio")
