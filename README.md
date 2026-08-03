# Inspect code with ProjectSAST

ProjectSAST is a bounded, read-only [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server for structural code inspection with [ast-grep](https://ast-grep.github.io/). It exposes six tools over standard input/output (stdio), performs no rewrites, and constrains every caller-selected project artifact to configured roots.

## Requirements

Install these runtime dependencies:

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)
- ast-grep 0.45.0

The server rejects every ast-grep version other than 0.45.0. `SUPPORTED_AST_GREP_VERSION` in `main.py` owns that version: the test suites and the CI workflow both read it, so a bump changes one line of code and the prose below. Install and verify the tested command-line interface (CLI) release:

```bash
npm install --global @ast-grep/cli@0.45.0
ast-grep --version
```

## Run the stdio server

Pin MCP consumers to an immutable commit SHA:

```bash
uvx \
  --from git+https://github.com/jmclaughlin724/ast-grep-mcp@commit_sha \
  ast-grep-server \
  --ast-grep /absolute/path/to/ast-grep \
  --allowed-root /absolute/path/to/project \
  --config /absolute/path/to/project/sgconfig.yml \
  --forbid-regex-rules
```

ProjectSAST supports stdio only. It doesn't expose Server-Sent Events (SSE), Streamable HTTP, a transport selector, or a network port.

## Configure the server

These command arguments define the runtime contract:

| Option | Default | Purpose |
| --- | --- | --- |
| `--ast-grep PATH` | `ast-grep` | Resolve one ast-grep executable before serving requests. |
| `--allowed-root PATH` | Process working directory | Permit projects only inside this real path. Repeat the option to add roots. |
| `--config PATH` | Unset | Use one contained `sgconfig.yml`, including configured custom languages. |
| `--command-timeout SECONDS` | `30` | Bound each ast-grep subprocess. |
| `--default-max-results COUNT` | `50` | Set the finite result limit used when a call omits `max_results`. |
| `--max-results-cap COUNT` | `500` | Lower the runtime ceiling for caller-selected result limits. |
| `--forbid-regex-rules` | Off | Reject inline YAML containing an ast-grep `regex` matcher key. |

Equivalent environment variables configure the executable, config, timeout, result limits, and regex policy. Prefer arguments in checked-in launchers so reviewers can see the effective contract.

The static MCP schemas allow `max_results` values through 500. An operator can configure a lower runtime cap. `get_server_info` reports that cap, and calls above it fail validation.

## Understand containment and input limits

Project, search, outline, symlink-target, and config paths resolve before use. Caller-controlled paths and configs must stay inside an allowed root. Search paths stay inside the selected project.

`dump_syntax_tree` uses a fresh, empty internal temporary directory. That sandbox can sit outside the allowed roots because it contains no user files. The server removes it after the syntax probe, including error paths.

Inline YAML has a 64 KiB input limit. This limit doesn't guarantee that the resulting subprocess command fits the operating system's launch budget. Before launch, the server budgets the fully quoted command and environment. Windows limits the complete command line to 32,767 characters; POSIX systems derive their available argument and environment budget from `SC_ARG_MAX`. A request that exceeds the platform budget fails before process creation with an actionable validation error.

## Use the six tools

Every tool advertises read-only, non-destructive, idempotent, and closed-world MCP annotations.

### `dump_syntax_tree`

Inspect how ast-grep parses code or a query pattern. Select `cst` for concrete target syntax and `pattern` while developing a structural query.

### `test_match_code_rule`

Test inline YAML with `id`, `language`, and `rule` fields against a code snippet. A valid negative probe returns an empty match list.

### `find_code`

Search with a complete structural pattern and an explicit language. Select relative paths, include and exclude globs, a finite result limit, and `text` or `json` output. The result envelope is bounded; that bound doesn't claim the native ast-grep traversal stops as soon as the envelope fills.

### `find_code_by_rule`

Search with one or more inline YAML rules and the same path, glob, result-limit, and output controls. Set `include_metadata=true` to forward ast-grep's documented `--include-metadata` flag and retain each rule's metadata in structured matches. SARIF and GitHub output remain CLI-only workflows.

### `get_server_info`

Read the package version, resolved ast-grep identity, config, allowed roots, timeout, runtime result limits, and regex policy.

### `outline_code`

Read per-file symbol hierarchies for 1 to 64 relative regular files. Directories, absolute paths, paths outside the project, and escaping symlinks are rejected. You can set an explicit language, choose `text` or `json`, and request a finite `max_results`.

The limit counts every emitted outline node recursively, including nested members. The response preserves each file's hierarchy and reports `returned`, `truncated`, and `limit`. ProjectSAST reads ast-grep's `--json=stream` output with these bounds:

- 1 MiB per newline-delimited JSON (NDJSON) record
- 4 MiB across all records
- The configured runtime result ceiling

The adapter terminates and reaps ast-grep after it observes node `limit + 1`. Malformed, non-object, oversized, or incomplete records fail closed.

## Read search and outline results

Text is the default model-facing format. JSON search results use this envelope:

```json
{
  "matches": [],
  "returned": 0,
  "truncated": false,
  "limit": 50
}
```

The server returns project-relative file paths and rejects results that resolve outside the selected project. Outline JSON uses the same result metadata with a per-file hierarchy instead of a flat `matches` list.

## Develop and verify

Install the locked environment and run the owner checks:

```bash
uv sync --frozen --all-extras --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy main.py tests
AST_GREP_TEST_EXECUTABLE=/absolute/path/to/ast-grep uv run pytest
uv lock --check
```

`AST_GREP_TEST_EXECUTABLE` is mandatory for integration acceptance. The suite executes its `--version` command and fails unless it reports exactly `ast-grep 0.45.0`. Protocol tests launch the installed `ast-grep-server` console entrypoint, negotiate both modern `mode="auto"` and handshake-era `mode="legacy"` MCP connections, inspect the six-tool catalog, and exercise representative calls.

For rule design, use ast-grep's [AI prompting workflow](https://astgrep.com/advanced/prompting.html), [rule testing guidance](https://astgrep.com/guide/test-rule.html), and [rewriting guide](https://astgrep.com/guide/rewrite-code.html). Rewriting and exhaustive project workflows remain outside this MCP server.
