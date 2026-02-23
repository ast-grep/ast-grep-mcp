# ast-grep MCP Server

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that provides AI assistants with powerful structural code search capabilities using [ast-grep](https://ast-grep.github.io/).

## Overview

This MCP server enables AI assistants (like Cursor, Claude Desktop, etc.) to search and analyze codebases using Abstract Syntax Tree (AST) pattern matching rather than simple text-based search. By leveraging ast-grep's structural search capabilities, AI can:

- Find code patterns based on syntax structure, not just text matching
- Search for specific programming constructs (functions, classes, imports, etc.)
- Write and test complex search rules using YAML configuration
- Debug and visualize AST structures for better pattern development

## Prerequisites

1. **Install ast-grep**: Follow [ast-grep installation guide](https://ast-grep.github.io/guide/quick-start.html#installation)
   ```bash
   # macOS
   brew install ast-grep
   nix-shell -p ast-grep
   cargo install ast-grep --locked
   ```

2. **Install Rust**: You need the Rust toolchain (cargo) to build the server.
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

3. **MCP-compatible client**: Such as Cursor, Claude Desktop, or other MCP clients

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ast-grep/ast-grep-mcp.git
   cd ast-grep-mcp
   ```

2. Build the server:
   ```bash
   cd ast-grep-mcp-rs
   cargo build --release
   ```

The binary will be located at `ast-grep-mcp-rs/target/release/ast-grep-mcp-server`.

## Configuration

### For Cursor

Add to your MCP settings (usually in `.cursor-mcp/settings.json`):

```json
{
  "mcpServers": {
    "ast-grep": {
      "command": "/absolute/path/to/ast-grep-mcp/ast-grep-mcp-rs/target/release/ast-grep-mcp-server",
      "args": [],
      "env": {}
    }
  }
}
```

### For Claude Desktop

Add to your Claude Desktop MCP configuration:

```json
{
  "mcpServers": {
    "ast-grep": {
      "command": "/absolute/path/to/ast-grep-mcp/ast-grep-mcp-rs/target/release/ast-grep-mcp-server",
      "args": [],
      "env": {}
    }
  }
}
```

### CLI Arguments

- `--config PATH`: Path to `sgconfig.yaml` file.
- `--transport TYPE`: "stdio" (default) or "sse".
- `--port PORT`: Port for SSE transport (default: 3101).

### Custom ast-grep Configuration

The MCP server supports using a custom `sgconfig.yaml` file to configure ast-grep behavior.
See the [ast-grep configuration documentation](https://ast-grep.github.io/guide/project/project-config.html) for details on the config file format.

You can provide the config file in two ways (in order of precedence):

1. **Command-line argument**: `--config /path/to/sgconfig.yaml`
2. **Environment variable**: `AST_GREP_CONFIG=/path/to/sgconfig.yaml`

## Usage

This repository includes comprehensive ast-grep rule documentation in [ast-grep.mdc](https://github.com/ast-grep/ast-grep-mcp/blob/main/ast-grep.mdc). The documentation covers all aspects of writing effective ast-grep rules, from simple patterns to complex multi-condition searches.

You can add it to your cursor rule or Claude.md, and attach it when you need AI agent to create ast-grep rule for you.

The prompt will ask LLM to use MCP to create, verify and improve the rule it creates.

## Features

The server provides four main tools for code analysis:

### 🔍 `dump_syntax_tree`
Visualize the Abstract Syntax Tree structure of code snippets. Essential for understanding how to write effective search patterns.

**Use cases:**
- Debug why a pattern isn't matching
- Understand the AST structure of target code
- Learn ast-grep pattern syntax

### 🧪 `test_match_code_rule`
Test ast-grep YAML rules against code snippets before applying them to larger codebases.

**Use cases:**
- Validate rules work as expected
- Iterate on rule development
- Debug complex matching logic

### 🎯 `find_code`
Search codebases using simple ast-grep patterns for straightforward structural matches.

**Parameters:**
- `max_results`: Limit number of complete matches returned (default: unlimited)
- `output_format`: Choose between `"text"` (default, ~75% fewer tokens) or `"json"` (full metadata)

**Text Output Format:**
```
Found 2 matches:

path/to/file.py:10-15
def example_function():
    # function body
    return result

path/to/file.py:20-22
def another_function():
    pass
```

**Use cases:**
- Find function calls with specific patterns
- Locate variable declarations
- Search for simple code constructs

### 🚀 `find_code_by_rule`
Advanced codebase search using complex YAML rules that can express sophisticated matching criteria.

**Parameters:**
- `max_results`: Limit number of complete matches returned (default: unlimited)
- `output_format`: Choose between `"text"` (default, ~75% fewer tokens) or `"json"` (full metadata)

**Use cases:**
- Find nested code structures
- Search with relational constraints (inside, has, precedes, follows)
- Complex multi-condition searches


## Usage Examples

### Basic Pattern Search

Use Query:

> Find all console.log statements

AI will generate rules like:

```yaml
id: find-console-logs
language: javascript
rule:
  pattern: console.log($$$)
```

### Complex Rule Example

User Query:
> Find async functions that use await

AI will generate rules like:

```yaml
id: async-with-await
language: javascript
rule:
  all:
    - kind: function_declaration
    - has:
        pattern: async
    - has:
        pattern: await $EXPR
        stopBy: end
```

## Supported Languages

ast-grep supports many programming languages including:
- JavaScript/TypeScript
- Python
- Rust
- Go
- Java
- C/C++
- C#
- And many more...

For a complete list of built-in supported languages, see the [ast-grep language support documentation](https://ast-grep.github.io/reference/languages.html).

You can also add support for custom languages through the `sgconfig.yaml` configuration file. See the [custom language guide](https://ast-grep.github.io/guide/project/project-config.html#languagecustomlanguage) for details.

## Troubleshooting

### Common Issues

1. **"Command not found" errors**: Ensure ast-grep is installed and in your PATH
2. **No matches found**: Try adding `stopBy: end` to relational rules
3. **Pattern not matching**: Use `dump_syntax_tree` to understand the AST structure
4. **Permission errors**: Ensure the server has read access to target directories

## Contributing

Issues and pull requests are welcome!

## Related Projects

- [ast-grep](https://ast-grep.github.io/) - The core structural search tool
- [Model Context Protocol](https://modelcontextprotocol.io/) - The protocol this server implements
- [Codemod MCP](https://docs.codemod.com/model-context-protocol) - Gives AI assistants tools like tree-sitter AST and node types, ast-grep instructions (YAML and JS ast-grep), and Codemod CLI commands to easily build, publish, and run ast-grep based codemods.
