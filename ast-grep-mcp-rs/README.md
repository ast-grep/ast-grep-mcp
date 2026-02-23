# ast-grep MCP Server (Rust Port)

A Rust implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server for [ast-grep](https://ast-grep.github.io/).

## Overview

This MCP server enables AI assistants (like Cursor, Claude Desktop, etc.) to search and analyze codebases using Abstract Syntax Tree (AST) pattern matching rather than simple text-based search.

This is a direct port of the Python version, maintaining 1:1 feature parity.

## Prerequisites

1. **Install ast-grep**: The `ast-grep` binary must be installed and in your PATH.
   Follow [ast-grep installation guide](https://ast-grep.github.io/guide/quick-start.html#installation).

2. **MCP-compatible client**: Such as Cursor, Claude Desktop, or other MCP clients.

## Building

```bash
cd ast-grep-mcp-rs
cargo build --release
```

The binary will be at `target/release/ast-grep-mcp-server`.

## Configuration

### For Cursor

Add to your MCP settings (usually in `.cursor-mcp/settings.json`):

```json
{
  "mcpServers": {
    "ast-grep": {
      "command": "/absolute/path/to/ast-grep-mcp-rs/target/release/ast-grep-mcp-server",
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
      "command": "/absolute/path/to/ast-grep-mcp-rs/target/release/ast-grep-mcp-server",
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

Environment variables:
- `AST_GREP_CONFIG`: Path to `sgconfig.yaml` file (overridden by `--config` flag).

## Features

The server provides four main tools for code analysis:

- `dump_syntax_tree`: Visualize AST structure.
- `test_match_code_rule`: Test YAML rules against code snippets.
- `find_code`: Search codebases using simple patterns.
- `find_code_by_rule`: Search codebases using complex YAML rules.

See the [Python implementation README](../README.md) for more details on features and usage examples.
