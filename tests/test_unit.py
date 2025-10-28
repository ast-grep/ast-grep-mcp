"""Unit tests for ast-grep MCP server"""

import json
import os
import subprocess
import sys
from unittest.mock import Mock, patch

import pytest

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Mock FastMCP to disable decoration
class MockFastMCP:
    """Mock FastMCP that returns functions unchanged"""

    def __init__(self, name):
        self.name = name
        self.tools = {}  # Store registered tools

    def tool(self, **kwargs):
        """Decorator that returns the function unchanged"""

        def decorator(func):
            # Store the function for later retrieval
            self.tools[func.__name__] = func
            return func  # Return original function without modification

        return decorator

    def run(self, **kwargs):
        """Mock run method"""
        pass


# Mock the Field function to return the default value
def mock_field(**kwargs):
    return kwargs.get("default")


# Patch the imports before loading main
with patch("mcp.server.fastmcp.FastMCP", MockFastMCP):
    with patch("pydantic.Field", mock_field):
        import main
        from main import (
            MAX_CODE_SNIPPET_LENGTH,
            MAX_PATTERN_LENGTH,
            MAX_YAML_RULE_LENGTH,
            format_matches_as_text,
            run_ast_grep,
            run_command,
            validate_code_snippet,
            validate_language,
            validate_pattern,
            validate_project_folder,
            validate_yaml_rule,
        )

        # Call register_mcp_tools to define the tool functions
        main.register_mcp_tools()

        # Extract the tool functions from the mocked mcp instance
        dump_syntax_tree = main.mcp.tools.get("dump_syntax_tree")
        find_code = main.mcp.tools.get("find_code")
        find_code_by_rule = main.mcp.tools.get("find_code_by_rule")
        match_code_rule = main.mcp.tools.get("test_match_code_rule")


class TestDumpSyntaxTree:
    """Test the dump_syntax_tree function"""

    @patch("main.run_ast_grep")
    def test_dump_syntax_tree_cst(self, mock_run):
        """Test dumping CST format"""
        mock_result = Mock()
        mock_result.stderr = "ROOT@0..10"
        mock_run.return_value = mock_result

        result = dump_syntax_tree("const x = 1", "javascript", "cst")

        assert result == "ROOT@0..10"
        mock_run.assert_called_once_with(
            "run",
            ["--pattern", "const x = 1", "--lang", "javascript", "--debug-query=cst"],
        )

    @patch("main.run_ast_grep")
    def test_dump_syntax_tree_pattern(self, mock_run):
        """Test dumping pattern format"""
        mock_result = Mock()
        mock_result.stderr = "pattern_node"
        mock_run.return_value = mock_result

        result = dump_syntax_tree("$VAR", "python", "pattern")

        assert result == "pattern_node"
        mock_run.assert_called_once_with(
            "run", ["--pattern", "$VAR", "--lang", "python", "--debug-query=pattern"]
        )


class TestTestMatchCodeRule:
    """Test the test_match_code_rule function"""

    @patch("main.run_ast_grep")
    def test_match_found(self, mock_run):
        """Test when matches are found"""
        mock_result = Mock()
        mock_result.stdout = '[{"text": "def foo(): pass"}]'
        mock_run.return_value = mock_result

        yaml_rule = """id: test
language: python
rule:
  pattern: 'def $NAME(): $$$'
"""
        code = "def foo(): pass"

        result = match_code_rule(code, yaml_rule)

        assert result == [{"text": "def foo(): pass"}]
        mock_run.assert_called_once_with(
            "scan", ["--inline-rules", yaml_rule, "--json", "--stdin"], input_text=code
        )

    @patch("main.run_ast_grep")
    def test_no_match(self, mock_run):
        """Test when no matches are found"""
        mock_result = Mock()
        mock_result.stdout = "[]"
        mock_run.return_value = mock_result

        yaml_rule = """id: test
language: python
rule:
  pattern: 'class $NAME'
"""
        code = "def foo(): pass"

        with pytest.raises(ValueError, match="No matches found"):
            match_code_rule(code, yaml_rule)


class TestFindCode:
    """Test the find_code function"""

    @patch("main.run_ast_grep")
    def test_text_format_with_results(self, mock_run, tmp_path):
        """Test text format output with results"""
        mock_result = Mock()
        mock_matches = [
            {"text": "def foo():\n    pass", "file": "file.py",
             "range": {"start": {"line": 0}, "end": {"line": 1}}},
            {"text": "def bar():\n    return", "file": "file.py",
             "range": {"start": {"line": 4}, "end": {"line": 5}}}
        ]
        mock_result.stdout = json.dumps(mock_matches)
        mock_run.return_value = mock_result

        result = find_code(
            project_folder=str(tmp_path),
            pattern="def $NAME():",
            language="python",
            output_format="text",
        )

        assert "Found 2 matches:" in result
        assert "def foo():" in result
        assert "def bar():" in result
        assert "file.py:1-2" in result
        assert "file.py:5-6" in result
        mock_run.assert_called_once_with(
            "run", ["--pattern", "def $NAME():", "--lang", "python", "--json", str(tmp_path)]
        )

    @patch("main.run_ast_grep")
    def test_text_format_no_results(self, mock_run, tmp_path):
        """Test text format output with no results"""
        mock_result = Mock()
        mock_result.stdout = "[]"
        mock_run.return_value = mock_result

        result = find_code(
            project_folder=str(tmp_path), pattern="nonexistent", output_format="text"
        )

        assert result == "No matches found"
        mock_run.assert_called_once_with(
            "run", ["--pattern", "nonexistent", "--json", str(tmp_path)]
        )

    @patch("main.run_ast_grep")
    def test_text_format_with_max_results(self, mock_run, tmp_path):
        """Test text format with max_results limit"""
        mock_result = Mock()
        mock_matches = [
            {"text": "match1", "file": "f.py", "range": {"start": {"line": 0}, "end": {"line": 0}}},
            {"text": "match2", "file": "f.py", "range": {"start": {"line": 1}, "end": {"line": 1}}},
            {"text": "match3", "file": "f.py", "range": {"start": {"line": 2}, "end": {"line": 2}}},
            {"text": "match4", "file": "f.py", "range": {"start": {"line": 3}, "end": {"line": 3}}},
        ]
        mock_result.stdout = json.dumps(mock_matches)
        mock_run.return_value = mock_result

        result = find_code(
            project_folder=str(tmp_path),
            pattern="pattern",
            max_results=2,
            output_format="text",
        )

        assert "Found 2 matches (showing first 2 of 4):" in result
        assert "match1" in result
        assert "match2" in result
        assert "match3" not in result

    @patch("main.run_ast_grep")
    def test_json_format(self, mock_run, tmp_path):
        """Test JSON format output"""
        mock_result = Mock()
        mock_matches = [
            {"text": "def foo():", "file": "test.py"},
            {"text": "def bar():", "file": "test.py"},
        ]
        mock_result.stdout = json.dumps(mock_matches)
        mock_run.return_value = mock_result

        result = find_code(
            project_folder=str(tmp_path), pattern="def $NAME():", output_format="json"
        )

        assert result == mock_matches
        mock_run.assert_called_once_with(
            "run", ["--pattern", "def $NAME():", "--json", str(tmp_path)]
        )

    @patch("main.run_ast_grep")
    def test_json_format_with_max_results(self, mock_run, tmp_path):
        """Test JSON format with max_results limit"""
        mock_result = Mock()
        mock_matches = [{"text": "match1"}, {"text": "match2"}, {"text": "match3"}]
        mock_result.stdout = json.dumps(mock_matches)
        mock_run.return_value = mock_result

        result = find_code(
            project_folder=str(tmp_path),
            pattern="pattern",
            max_results=2,
            output_format="json",
        )

        assert len(result) == 2
        assert result[0]["text"] == "match1"
        assert result[1]["text"] == "match2"

    def test_invalid_output_format(self, tmp_path):
        """Test with invalid output format"""
        with pytest.raises(ValueError, match="Invalid output_format"):
            find_code(
                project_folder=str(tmp_path), pattern="pattern", output_format="invalid"
            )


class TestFindCodeByRule:
    """Test the find_code_by_rule function"""

    @patch("main.run_ast_grep")
    def test_text_format_with_results(self, mock_run, tmp_path):
        """Test text format output with results"""
        mock_result = Mock()
        mock_matches = [
            {"text": "class Foo:\n    pass", "file": "file.py",
             "range": {"start": {"line": 0}, "end": {"line": 1}}},
            {"text": "class Bar:\n    pass", "file": "file.py",
             "range": {"start": {"line": 9}, "end": {"line": 10}}}
        ]
        mock_result.stdout = json.dumps(mock_matches)
        mock_run.return_value = mock_result

        yaml_rule = """id: test
language: python
rule:
  pattern: 'class $NAME'
"""

        result = find_code_by_rule(
            project_folder=str(tmp_path), yaml=yaml_rule, output_format="text"
        )

        assert "Found 2 matches:" in result
        assert "class Foo:" in result
        assert "class Bar:" in result
        assert "file.py:1-2" in result
        assert "file.py:10-11" in result
        mock_run.assert_called_once_with(
            "scan", ["--inline-rules", yaml_rule, "--json", str(tmp_path)]
        )

    @patch("main.run_ast_grep")
    def test_json_format(self, mock_run, tmp_path):
        """Test JSON format output"""
        mock_result = Mock()
        mock_matches = [{"text": "class Foo:", "file": "test.py"}]
        mock_result.stdout = json.dumps(mock_matches)
        mock_run.return_value = mock_result

        yaml_rule = """id: test
language: python
rule:
  pattern: 'class $NAME'
"""

        result = find_code_by_rule(
            project_folder=str(tmp_path), yaml=yaml_rule, output_format="json"
        )

        assert result == mock_matches
        mock_run.assert_called_once_with(
            "scan", ["--inline-rules", yaml_rule, "--json", str(tmp_path)]
        )


class TestRunCommand:
    """Test the run_command function"""

    @patch("subprocess.run")
    def test_successful_command(self, mock_run):
        """Test successful command execution"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "output"
        mock_run.return_value = mock_result

        result = run_command(["echo", "test"])

        assert result.stdout == "output"
        mock_run.assert_called_once_with(
            ["echo", "test"], capture_output=True, input=None, text=True, check=True, shell=False
        )

    @patch("subprocess.run")
    def test_command_failure(self, mock_run):
        """Test command execution failure"""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["false"], stderr="error message"
        )

        with pytest.raises(RuntimeError, match="failed with exit code 1"):
            run_command(["false"])

    @patch("subprocess.run")
    def test_command_not_found(self, mock_run):
        """Test when command is not found"""
        mock_run.side_effect = FileNotFoundError()

        with pytest.raises(RuntimeError, match="not found"):
            run_command(["nonexistent"])


class TestFormatMatchesAsText:
    """Test the format_matches_as_text helper function"""

    def test_empty_matches(self):
        """Test with empty matches list"""
        result = format_matches_as_text([])
        assert result == ""

    def test_single_line_match(self):
        """Test formatting a single-line match"""
        matches = [
            {
                "text": "const x = 1",
                "file": "test.js",
                "range": {"start": {"line": 4}, "end": {"line": 4}}
            }
        ]
        result = format_matches_as_text(matches)
        assert result == "test.js:5\nconst x = 1"

    def test_multi_line_match(self):
        """Test formatting a multi-line match"""
        matches = [
            {
                "text": "def foo():\n    return 42",
                "file": "test.py",
                "range": {"start": {"line": 9}, "end": {"line": 10}}
            }
        ]
        result = format_matches_as_text(matches)
        assert result == "test.py:10-11\ndef foo():\n    return 42"

    def test_multiple_matches(self):
        """Test formatting multiple matches"""
        matches = [
            {
                "text": "match1",
                "file": "file1.py",
                "range": {"start": {"line": 0}, "end": {"line": 0}}
            },
            {
                "text": "match2\nline2",
                "file": "file2.py",
                "range": {"start": {"line": 5}, "end": {"line": 6}}
            }
        ]
        result = format_matches_as_text(matches)
        expected = "file1.py:1\nmatch1\n\nfile2.py:6-7\nmatch2\nline2"
        assert result == expected


class TestRunAstGrep:
    """Test the run_ast_grep function"""

    @patch("main.run_command")
    @patch("main.CONFIG_PATH", None)
    def test_without_config(self, mock_run):
        """Test running ast-grep without config"""
        mock_result = Mock()
        mock_run.return_value = mock_result

        result = run_ast_grep("run", ["--pattern", "test"])

        assert result == mock_result
        mock_run.assert_called_once_with(["ast-grep", "run", "--pattern", "test"], None)

    @patch("main.run_command")
    @patch("main.CONFIG_PATH", "/path/to/config.yaml")
    def test_with_config(self, mock_run):
        """Test running ast-grep with config"""
        mock_result = Mock()
        mock_run.return_value = mock_result

        result = run_ast_grep("scan", ["--inline-rules", "rule"])

        assert result == mock_result
        mock_run.assert_called_once_with(
            [
                "ast-grep",
                "scan",
                "--config",
                "/path/to/config.yaml",
                "--inline-rules",
                "rule",
            ],
            None,
        )


class TestValidation:
    """Test validation helper functions"""

    def test_validate_project_folder_success(self, tmp_path):
        """Test valid project folder"""
        result = validate_project_folder(str(tmp_path))
        assert result == tmp_path

    def test_validate_project_folder_empty(self):
        """Test rejection of empty path"""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_project_folder("")

    def test_validate_project_folder_relative_path(self):
        """Test rejection of relative paths"""
        with pytest.raises(ValueError, match="must be an absolute path"):
            validate_project_folder("./relative/path")

    def test_validate_project_folder_nonexistent(self):
        """Test rejection of non-existent paths"""
        with pytest.raises(ValueError, match="does not exist"):
            validate_project_folder("/nonexistent/path/12345")

    def test_validate_project_folder_file_not_directory(self, tmp_path):
        """Test rejection of files (not directories)"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("test")
        with pytest.raises(ValueError, match="must be a directory"):
            validate_project_folder(str(file_path))

    def test_validate_pattern_success(self):
        """Test valid pattern"""
        result = validate_pattern("  def $NAME()  ")
        assert result == "def $NAME()"

    def test_validate_pattern_empty(self):
        """Test rejection of empty patterns"""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_pattern("")

    def test_validate_pattern_whitespace_only(self):
        """Test rejection of whitespace-only patterns"""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_pattern("   ")

    def test_validate_pattern_too_long(self):
        """Test rejection of overly long patterns"""
        long_pattern = "x" * (MAX_PATTERN_LENGTH + 1)
        with pytest.raises(ValueError, match="too long"):
            validate_pattern(long_pattern)

    def test_validate_language_success(self):
        """Test valid language"""
        result = validate_language("  PYTHON  ")
        assert result == "python"

    def test_validate_language_empty_allowed(self):
        """Test empty language (auto-detection) is allowed"""
        result = validate_language("")
        assert result == ""

    def test_validate_language_invalid(self):
        """Test rejection of unsupported languages"""
        with pytest.raises(ValueError, match="Unsupported language"):
            validate_language("unsupported_lang")

    def test_validate_language_invalid_with_suggestion(self):
        """Test error message includes suggestions for similar languages"""
        with pytest.raises(ValueError, match="Did you mean"):
            validate_language("typ")  # Should suggest typescript, tsx

    def test_validate_yaml_rule_success(self):
        """Test valid YAML rule"""
        yaml_str = """
id: test-rule
language: python
rule:
  pattern: 'def $NAME()'
"""
        result = validate_yaml_rule(yaml_str)
        assert result["id"] == "test-rule"
        assert result["language"] == "python"
        assert "rule" in result

    def test_validate_yaml_rule_empty(self):
        """Test rejection of empty YAML"""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_yaml_rule("")

    def test_validate_yaml_rule_too_long(self):
        """Test rejection of overly long YAML"""
        long_yaml = "x" * (MAX_YAML_RULE_LENGTH + 1)
        with pytest.raises(ValueError, match="too long"):
            validate_yaml_rule(long_yaml)

    def test_validate_yaml_rule_invalid_syntax(self):
        """Test rejection of invalid YAML syntax"""
        yaml_str = "id: test\n  invalid: [unclosed"
        with pytest.raises(ValueError, match="Invalid YAML syntax"):
            validate_yaml_rule(yaml_str)

    def test_validate_yaml_rule_missing_id(self):
        """Test rejection of YAML missing 'id' field"""
        yaml_str = "language: python\nrule:\n  pattern: 'test'"
        with pytest.raises(ValueError, match="missing required fields.*id"):
            validate_yaml_rule(yaml_str)

    def test_validate_yaml_rule_missing_language(self):
        """Test rejection of YAML missing 'language' field"""
        yaml_str = "id: test\nrule:\n  pattern: 'test'"
        with pytest.raises(ValueError, match="missing required fields.*language"):
            validate_yaml_rule(yaml_str)

    def test_validate_yaml_rule_missing_rule(self):
        """Test rejection of YAML missing 'rule' field"""
        yaml_str = "id: test\nlanguage: python"
        with pytest.raises(ValueError, match="missing required fields.*rule"):
            validate_yaml_rule(yaml_str)

    def test_validate_yaml_rule_not_dict(self):
        """Test rejection of YAML that isn't a dictionary"""
        yaml_str = "- item1\n- item2"
        with pytest.raises(ValueError, match="must be a dictionary"):
            validate_yaml_rule(yaml_str)

    def test_validate_code_snippet_success(self):
        """Test valid code snippet"""
        code = "def foo():\n    pass"
        result = validate_code_snippet(code)
        assert result == code

    def test_validate_code_snippet_empty(self):
        """Test rejection of empty code"""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_code_snippet("")

    def test_validate_code_snippet_whitespace_only(self):
        """Test rejection of whitespace-only code"""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_code_snippet("   \n  \t  ")

    def test_validate_code_snippet_too_long(self):
        """Test rejection of overly long code"""
        long_code = "x" * (MAX_CODE_SNIPPET_LENGTH + 1)
        with pytest.raises(ValueError, match="too long"):
            validate_code_snippet(long_code)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
