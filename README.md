# ast-grep MCP Server

A bounded, read-only [Model Context Protocol](https://modelcontextprotocol.io/) server for structural code inspection with [ast-grep](https://ast-grep.github.io/).

The server is designed for model-facing exploration: inspect syntax, test a rule against a snippet, then search a deliberately constrained project scope. It does not expose rewriting or editing tools. Exhaustive scans and authorized rewrites remain CLI workflows.

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)
- ast-grep 0.44.1 for the tested integration contract

Install the tested ast-grep CLI with npm:

```bash
npm install --global @ast-grep/cli@0.44.1
ast-grep --version
```

## Run from an immutable fork revision

Pin MCP consumers to a commit SHA:

```bash
uvx \
  --from git+https://github.com/jmclaughlin724/ast-grep-mcp@<commit-sha> \
  ast-grep-server \
  --ast-grep /absolute/path/to/ast-grep \
  --allowed-root /absolute/path/to/project \
  --config /absolute/path/to/project/sgconfig.yml \
  --forbid-regex-rules
```

`--allowed-root` is repeatable. When omitted, it defaults to the process working directory. Every project, search path, symlink target, and config path is resolved before use and must remain inside an allowed root.

## Server options

| Option | Default | Purpose |
| --- | --- | --- |
| `--ast-grep PATH` | `ast-grep` | Resolve the exact CLI instead of trusting a later `PATH` lookup. |
| `--allowed-root PATH` | process working directory | Permit searches only inside this real path; repeat to add roots. |
| `--config PATH` | unset | Use one contained `sgconfig.yml`. |
| `--command-timeout SECONDS` | `30` | Bound every ast-grep subprocess. |
| `--default-max-results COUNT` | `50` | Set the finite result limit used when a tool call omits it. |
| `--max-results-cap COUNT` | `500` | Cap caller-selected limits; values above 500 are rejected. |
| `--forbid-regex-rules` | off | Reject inline YAML containing ast-grep `regex` matcher keys. |
| `--transport` | `stdio` | Select `stdio`, `sse`, or `streamable-http`. |

Equivalent environment variables exist for the ast-grep executable, config, timeout, result limits, and regex-rule policy. Use command arguments in checked-in MCP launchers so the effective security contract stays visible.

## Tools

All tools advertise these MCP annotations:

- read-only
- non-destructive
- idempotent
- closed-world

### `dump_syntax_tree`

Inspect how ast-grep parses code or a query pattern. Use `cst` for concrete target syntax and `pattern` while developing a query.

### `test_match_code_rule`

Test YAML with `id`, `language`, and `rule` fields against a code snippet. A valid negative probe returns an empty match list instead of an error.

### `find_code`

Search with a complete structural pattern and an explicit language. The tool accepts:

- `project_folder`
- relative `paths`
- `include_globs`
- `exclude_globs`
- finite `max_results`
- `output_format` (`text` or `json`)

Pattern searches are compiled into inline rules and run with `ast-grep scan --max-results <limit+1>`. This stops work early and reports truncation without scanning the entire tree.

### `find_code_by_rule`

Search with one or more inline YAML rules using the same path, glob, result-limit, and output controls.

### `get_server_info`

Report the fork version, resolved ast-grep executable and version, config path, allowed roots, command timeout, effective result limits, and regex-rule policy.

## Search results

Compact text is the default model-facing format:

```text
Found 2 matches (limit 2; additional matches exist):

src/example.ts:4
loadData()
```

JSON results have one stable envelope:

```json
{
  "matches": [],
  "returned": 0,
  "truncated": false,
  "limit": 50
}
```

The server never offers zero or unlimited searches. It returns project-relative match paths and rejects results that resolve outside the selected project.

## Development

```bash
uv sync --frozen --all-extras --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy main.py
uv run pytest
```

The integration suite launches the real STDIO server, negotiates MCP, inspects the exact tool catalog and annotations, calls metadata and search tools, and verifies ast-grep 0.44.1.

For rule design, follow ast-grep's [AI prompting workflow](https://astgrep.com/advanced/prompting.html), [rule testing guidance](https://astgrep.com/guide/test-rule.html), and [rewriting guide](https://astgrep.com/guide/rewrite-code.html). Rewriting is intentionally outside this MCP server.
