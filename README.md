# ProjectSAST

ProjectSAST is a bounded, read-only [Model Context Protocol](https://modelcontextprotocol.io/) server for structural code inspection with [ast-grep](https://ast-grep.github.io/). Version 0.4.0 exposes exactly eight tools over stdio, never applies rewrites, and confines caller-selected files to operator-approved roots.

## Guarantees

- Exact ast-grep 0.45.0 runtime compatibility
- Python 3.14.6 with a locked `uv` environment
- stdio transport only
- Finite input, output, diagnostic, timeout, and result limits
- Process-group termination and reaping on timeout, overflow, malformed output, or result saturation
- Strict typed validation of ast-grep 0.45 match and outline records
- Immutable startup configuration copied into a private read-only bundle
- Read-only rewrite previews, including deletion previews
- Modern and legacy MCP protocol compatibility

## Install

Install the locked Python environment and exact native CLI:

```bash
uv sync --locked --all-extras --dev --no-build-isolation --no-python-downloads
npm install --global @ast-grep/cli@0.45.0
ast-grep --version
```

The server refuses to start unless the CLI reports exactly `ast-grep 0.45.0`.

## Run the stdio server

Run the checked-out repository through its existing synchronized environment:

```bash
export UV_PROJECT_ENVIRONMENT=.venv
uv --directory /absolute/path/to/ast-grep-mcp \
  --no-python-downloads \
  run --no-active --no-sync python scripts/launch_server.py \
  --ast-grep /absolute/path/to/ast-grep \
  --allowed-root /absolute/path/to/project \
  --config /absolute/path/to/project/sgconfig.yml \
  --forbid-regex-rules
```

ProjectSAST does not expose HTTP, Server-Sent Events, a transport selector, or a network port.

## Runtime configuration

| Option | Default | Effect |
| --- | --- | --- |
| `--ast-grep PATH` | `ast-grep` | Resolves and verifies one ast-grep executable before serving requests. |
| `--allowed-root PATH` | Process working directory | Adds an approved real-path root. Repeat for multiple roots. |
| `--config PATH` | Unset | Loads one contained `sgconfig.yml` at startup. |
| `--trusted-native-library PATH SHA256` | None | Trusts one configured native parser by exact path and SHA-256. Repeat for every active parser. |
| `--command-timeout SECONDS` | `30` | Bounds each subprocess. |
| `--default-max-results COUNT` | `50` | Supplies the result limit omitted by a caller. |
| `--max-results-cap COUNT` | `500` | Lowers the caller-selectable result ceiling. |
| `--forbid-regex-rules` | Off | Rejects any inline YAML rule containing a `regex` key. |

The executable, config, timeout, result limits, and regex policy also support their documented `AST_GREP_*` environment variables. `get_server_info` reports the effective contract.

## Immutable configuration snapshot

When `--config` is present, startup validates `sgconfig.yml` and all active resources before the MCP server begins reading requests. It rejects:

- symlinks, Windows reparse points, and path escapes;
- unknown ast-grep 0.45 configuration keys;
- YAML aliases, anchors, explicit tags, merge keys, duplicate keys, and non-string keys;
- excessive YAML depth, document count, or node count;
- duplicate rule IDs and zero-progress utility-rule cycles;
- native parser libraries without an exact operator-supplied SHA-256.

Documented recursion through `has`, `inside`, `precedes`, and `follows` remains valid because those relations advance to another syntax node.

Active rules, utilities, tests, snapshots, outline definitions, and trusted native parsers are copied into a private bundle. Its files become read-only before serving. ProjectSAST generates three configurations:

1. inline searches and outlines with no configured project rules;
2. configured scans with only retained `ruleDirs` and `utilDirs`;
3. configured tests with retained rules, utilities, `testConfigs`, and snapshots.

The original files are never reloaded during the session. Mutating them after startup does not alter active behavior.

## Resource limits

| Resource | Limit |
| --- | ---: |
| Snippet or stdin input | 1 MiB |
| Inline YAML rule | 64 KiB |
| One NDJSON record | 1 MiB |
| Aggregate structured output | 4 MiB |
| Subprocess diagnostics | 64 KiB |
| Configured-test report | 64 KiB |
| Startup config file | 1 MiB |
| Retained configuration resources | 16 MiB across 1,024 files |
| Trusted native parser library | 16 MiB each |
| YAML documents | 64 |
| YAML nodes | 10,000 |
| YAML depth | 64 |
| Outline file paths | 64 |
| Hard result ceiling | 500 |
| Windows process creation | 32,767 UTF-16 characters |
| POSIX process creation | Runtime `ARG_MAX` minus 2,048 bytes |
| Process termination grace | 2 seconds |

NDJSON must be complete, newline-terminated UTF-8. Empty, malformed, non-object, incomplete, oversized, or schema-invalid records fail closed. Unknown upstream object fields are retained.

Before process creation, ProjectSAST also checks the complete command and environment against Windows `CreateProcessW` or POSIX `ARG_MAX` limits.

## Eight-tool API

Every tool advertises read-only, non-destructive, idempotent, closed-world MCP annotations.

| Tool | Purpose |
| --- | --- |
| `dump_syntax_tree` | Dumps `pattern`, `cst`, `ast`, or `sexp` syntax for a bounded snippet. |
| `test_match_code_rule` | Tests validated inline YAML against a bounded snippet. |
| `outline_code` | Outlines 1–64 explicit contained files with item, type, and public-member filters. |
| `find_code` | Searches by pattern with optional selector, strictness, and preview-only rewrite. |
| `find_code_by_rule` | Searches with validated inline rules and preserves fix-preview data. |
| `scan_project_rules` | Runs only rules retained from startup configuration. |
| `test_project_rules` | Runs retained configured suites without interaction or snapshot updates. |
| `get_server_info` | Reports versions, configuration provenance, capabilities, coordinates, and limits. |

### Pattern search and previews

`find_code` requires `pattern` and `language`. Its optional controls are:

- `selector`: selects a node kind from a contextual pattern;
- `strictness`: `cst`, `smart`, `ast`, `relaxed`, `signature`, or `template`;
- `rewrite`: returns replacement and replacement-offset previews without changing files.

An empty `rewrite` previews deletion. No tool invokes ast-grep rewrite application flags.

### Rule search

`find_code_by_rule` accepts one or more YAML rule documents. Match results preserve ast-grep rule fields, metadata, transformed metavariables, replacement text, replacement offsets, and future upstream fields. Text output renders the same preview information.

### Configured scan and test

`scan_project_rules` cannot accept arbitrary rule YAML. It uses only startup-retained rules. Optional `rule_ids` values are checked against the retained catalog and escaped into one exact-match filter. Paths and caller globs remain contained and bounded; rule-level `files` and `ignores` continue to use project-relative semantics.

`test_project_rules` runs only startup-retained suites. Exit code 4 is reported as `passed: false`; configuration or execution exit codes fail the tool. It never passes `--interactive`, `--update-all`, or another snapshot mutation option.

### Outline

`outline_code` requires explicit relative regular files. It supports:

- `items`: `auto`, `structure`, `exports`, `imports`, or `all`;
- `symbol_types`: exact top-level outline symbol types;
- `public_members`: retain only public members where the language outline defines visibility.

The limit counts top-level items and nested members in preorder. Hierarchy and unknown fields are preserved.

### Result envelopes and coordinates

Search tools return:

```json
{
  "matches": [],
  "returned": 0,
  "truncated": false,
  "limit": 50
}
```

Outline results use the same result metadata with a `files` hierarchy. Returned file paths are project-relative and revalidated after ast-grep exits.

Coordinates follow ast-grep 0.45 conventions:

- lines are zero-based;
- columns are zero-based Unicode scalar counts;
- byte offsets are zero-based UTF-8 byte offsets;
- ranges are half-open `[start, end)`.

## Deliberate exclusions

Version 0.4.0 does not expose mutation, raw CLI arguments, `--follow`, ignore bypasses, snapshot updates, LSP, HTTP transport, or rewrite application.

It also does not add `inspect_syntax`. Although `ast-grep-py==0.45.0` exists, it lacks the error, missing, extra, and child-field accessors required for a lossless concrete-syntax-tree contract. ProjectSAST does not parse debug stderr or add an independent grammar stack to approximate that contract.

## Develop and verify

```bash
uv sync --locked --all-extras --dev --no-build-isolation --no-python-downloads
uv run --no-sync python scripts/verify_environment.py
uv lock --check
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy main.py config_snapshot.py scripts tests
uv run --no-sync pyright
uv run --no-sync pytest tests/test_unit.py tests/test_config_snapshot.py tests/test_environment_policy.py \
  --cov=main --cov=config_snapshot --cov-report=term-missing
AST_GREP_TEST_EXECUTABLE=/absolute/path/to/ast-grep \
  uv run --no-sync pytest tests/test_integration.py
uv run --no-sync python scripts/launch_server.py --help
uv run --no-sync python -m build --sdist --wheel --no-isolation
uv run --no-sync twine check dist/*
uv run --no-sync check-wheel-contents dist/*.whl
uv run --no-sync python tests/package_smoke.py dist/sg_mcp-0.4.0-py3-none-any.whl dist/sg_mcp-0.4.0.tar.gz 0.4.0
```

Unit coverage is enforced at 84%. Integration acceptance requires an explicit executable reporting exactly ast-grep 0.45.0 and negotiates both modern `mode="auto"` and handshake-era `mode="legacy"` MCP connections.

Distribution verification stays in the repository-owned locked environment: tool execution disables synchronization, builds disable PEP 517 isolation, the wheel is imported directly from its archive, and the sdist is inspected without extraction or installation.

`AGENTS.md` defines the repository execution boundary. `scripts/verify_environment.py` fails unless verification runs from the repository root in its `.venv`, and it rejects alternate, isolated, cached, and external environment commands in verification surfaces. `scripts/launch_server.py` checks the lock and synchronized environment without updating either before starting stdio. CI fixes `UV_PROJECT_ENVIRONMENT` to `.venv`, synchronizes once with `--locked`, and runs every subsequent tool with `--no-sync`.

## Release boundary

The release workflow accepts protected `v*` tags, verifies the tag is exactly `v<project.version>`, reruns three-platform acceptance against ast-grep 0.45.0, builds and checks the sdist and pure Python wheel, and publishes through PyPI trusted publishing only from the protected `pypi` environment.

Creating commits or tags, pushing, configuring the GitHub environment, and enabling PyPI trusted publishing remain operator actions.

## Rule-authoring guidance

[`ast-grep.mdc`](ast-grep.mdc) records the official ast-grep agent-skill workflow and rule reference used by this repository. Repository working agreements remain authoritative over generic upstream guidance.
