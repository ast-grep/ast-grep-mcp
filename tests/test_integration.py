"""Integration tests for ast-grep MCP server"""

import json
import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add parent directory to path
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


# Import with mocked decorators
with patch("mcp.server.fastmcp.FastMCP", MockFastMCP):
    with patch("pydantic.Field", mock_field):
        import main

        # Call register_mcp_tools to define the tool functions
        main.register_mcp_tools()

        # Extract the tool functions from the mocked mcp instance
        find_code = main.mcp.tools.get("find_code")
        find_code_by_rule = main.mcp.tools.get("find_code_by_rule")


@pytest.fixture
def fixtures_dir():
    """Get the path to the fixtures directory"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "fixtures"))


class TestIntegration:
    """Integration tests for ast-grep MCP functions"""

    def test_find_code_text_format(self, fixtures_dir):
        """Test find_code with text format"""
        result = find_code(
            project_folder=fixtures_dir,
            pattern="def $NAME($$$)",
            language="python",
            output_format="text",
        )

        assert "hello" in result
        assert "add" in result
        assert "Found" in result and "matches" in result

    def test_find_code_json_format(self, fixtures_dir):
        """Test find_code with JSON format"""
        result = find_code(
            project_folder=fixtures_dir,
            pattern="def $NAME($$$)",
            language="python",
            output_format="json",
        )

        assert len(result) >= 2
        assert any("hello" in str(match) for match in result)
        assert any("add" in str(match) for match in result)

    @patch("main.run_ast_grep")
    def test_find_code_by_rule(self, mock_run, fixtures_dir):
        """Test find_code_by_rule with mocked ast-grep"""
        # Mock the response with JSON format (since we always use JSON internally)
        mock_result = Mock()
        mock_matches = [
            {"text": "class Calculator:\n    pass", "file": "fixtures/example.py", "range": {"start": {"line": 6}, "end": {"line": 7}}}
        ]
        mock_result.stdout = json.dumps(mock_matches)
        mock_run.return_value = mock_result

        yaml_rule = """id: test
language: python
rule:
  pattern: class $NAME"""

        result = find_code_by_rule(project_folder=fixtures_dir, yaml=yaml_rule, output_format="text")

        assert "Calculator" in result
        assert "Found 1 match" in result
        assert "fixtures/example.py:7-8" in result

        # Verify the command was called correctly
        mock_run.assert_called_once_with("scan", ["--inline-rules", yaml_rule, "--json", fixtures_dir])

    def test_find_code_with_max_results(self, fixtures_dir):
        """Test find_code with max_results parameter"""
        result = find_code(
            project_folder=fixtures_dir,
            pattern="def $NAME($$$)",
            language="python",
            max_results=1,
            output_format="text",
        )

        # The new format says "showing first X of Y" instead of "limited to X"
        assert "showing first 1 of" in result or "Found 1 match" in result
        # Should only have one match in the output
        assert result.count("def ") == 1

    def test_find_code_no_matches(self, fixtures_dir):
        """Test find_code when no matches are found"""
        result = find_code(
            project_folder=fixtures_dir,
            pattern="nonexistent_pattern_xyz",
            output_format="text",
        )

        assert result == "No matches found"
