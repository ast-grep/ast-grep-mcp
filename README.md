# ast-soleaux

ast-soleaux is a bounded [Model Context Protocol](https://modelcontextprotocol.io/) server for structural search, Oxc transformation/minification, JavaScript and TypeScript semantic analysis, TypeScript Compiler API project inspection, PostgreSQL parsing/deparse proof, and operator-sandboxed TypeScript execution. Read-only analysis and preview tools remain closed-world; mutation and execution tools use explicit FastMCP annotations, capability gates, contained paths, and bounded subprocess contracts. Project formatting stays with each repository's formatter command instead of the MCP server.

## Guarantees

- Exact ast-grep 0.45.0 runtime compatibility
- Optional ast-soleaux Oxc compute sidecar with oxc-transform/oxc-minify 0.147.0 and module inspection
- Repository-local Oxfmt 0.65.0 development formatting through the sidecar package scripts
- Python 3.14.6 with a locked `uv` environment
- stdio transport only
- Finite input, output, diagnostic, timeout, and result limits
- Process-group termination and reaping on timeout, overflow, malformed output, or result saturation
- Strict typed validation of ast-grep 0.45 match and outline records
- Immutable startup configuration copied into a private read-only bundle
- Read-only rewrite previews, including deletion previews
- FastMCP `4.0.0b3` / `fastmcp-slim 4.0.0b3` on Python 3.14
- Modern and legacy MCP protocol compatibility

## Install

Install the locked Python environment and exact native CLI:

```bash
uv sync --locked --all-extras --dev --no-python-downloads
npm install --global @ast-grep/cli@0.45.0
ast-grep --version
npm ci --prefix oxc-sidecar
node oxc-sidecar/bin/ast-soleaux-oxc.mjs --version
node oxc-sidecar/bin/ast-soleaux-typescript-project.mjs --version-json
cargo build --manifest-path analysis-sidecar/Cargo.toml
cargo test --manifest-path analysis-sidecar/Cargo.toml
python execution-sidecar/ast_soleaux_typescript_sandbox.py --version-json
npm ci --prefix postgresql-sidecar
node postgresql-sidecar/bin/ast-soleaux-postgresql.mjs --version-json
```

The server refuses to start unless each configured backend reports its exact supported version. Optional backends include the ast-soleaux Oxc compute helper, Rust semantic worker, TypeScript 6.0.2 project worker, and PostgreSQL 18 parser/deparser worker. The PostgreSQL package manifest selects `@libpg-query/parser@pg18` and `pgsql-deparser@latest`; the lockfile and `get_server_info` report the resolved `18.0.0` and `18.3.6` versions.

## Run the stdio server

Run the checked-out repository through its existing synchronized environment:

```bash
export UV_PROJECT_ENVIRONMENT=.venv
uv --directory /absolute/path/to/ast-grep-mcp \
  --no-python-downloads \
  run --no-active --no-sync python scripts/launch_server.py \
  --ast-grep /absolute/path/to/ast-grep \
  --oxc-helper /absolute/path/to/oxc-sidecar/bin/ast-soleaux-oxc.mjs \
  --analysis-helper /absolute/path/to/analysis-sidecar/target/debug/ast-soleaux-analysis \
  --typescript-project-helper /absolute/path/to/oxc-sidecar/bin/ast-soleaux-typescript-project.mjs \
  --postgres-helper /absolute/path/to/postgresql-sidecar/bin/ast-soleaux-postgresql.mjs \
  --typescript-execution-helper /absolute/path/to/execution-sidecar/ast_soleaux_typescript_sandbox.py \
  --typescript-execution-profile isolated \
  --allowed-root /absolute/path/to/project \
  --config /absolute/path/to/project/sgconfig.yml \
  --forbid-regex-rules
```

ast-soleaux does not expose HTTP, Server-Sent Events, a transport selector, or a network port.

## Runtime configuration

| Option                                   | Default                   | Effect                                                                                         |
| ---------------------------------------- | ------------------------- | ---------------------------------------------------------------------------------------------- |
| `--ast-grep PATH`                        | `ast-grep`                | Resolves and verifies one ast-grep executable before serving requests.                         |
| `--oxc-helper PATH`                      | Unset                     | Enables JavaScript and TypeScript module inspection through one exactly versioned Oxc sidecar. |
| `--analysis-helper PATH`                 | Unset                     | Enables Oxc lexical semantic and CFG tools through the pinned Rust worker.                     |
| `--typescript-project-helper PATH`       | Unset                     | Enables the bounded TypeScript 6 Compiler API project tool.                                    |
| `--postgres-helper PATH`                 | Unset                     | Enables PostgreSQL 18 parse, scan, fingerprint, PL/pgSQL, file-batch, and deparse-proof tools. |
| `--typescript-execution-helper PATH`     | Unset                     | Enables operator-sandboxed TypeScript execution.                                               |
| `--typescript-execution-profile PROFILE` | `isolated`                | Selects the operator-owned execution sandbox profile.                                          |
| `--allowed-root PATH`                    | Process working directory | Adds an approved real-path root. Repeat for multiple roots.                                    |
| `--config PATH`                          | Unset                     | Loads one contained `sgconfig.yml` at startup.                                                 |
| `--trusted-native-library PATH SHA256`   | None                      | Trusts one configured native parser by exact path and SHA-256. Repeat for every active parser. |
| `--command-timeout SECONDS`              | `30`                      | Bounds each subprocess.                                                                        |
| `--default-max-results COUNT`            | `50`                      | Supplies the result limit omitted by a caller.                                                 |
| `--max-results-cap COUNT`                | `500`                     | Lowers the caller-selectable result ceiling.                                                   |
| `--forbid-regex-rules`                   | Off                       | Rejects any inline YAML rule containing a `regex` key.                                         |

The executable, config, timeout, result limits, and regex policy also support their documented `AST_GREP_*` environment variables; the optional sidecar supports `OXC_HELPER_EXECUTABLE`. `get_server_info` reports the effective contract.

## Immutable configuration snapshot

When `--config` is present, startup validates `sgconfig.yml` and all active resources before the MCP server begins reading requests. It rejects:

- symlinks, Windows reparse points, and path escapes;
- unknown ast-grep 0.45 configuration keys;
- YAML aliases, anchors, explicit tags, merge keys, duplicate keys, and non-string keys;
- excessive YAML depth, document count, or node count;
- duplicate rule IDs and zero-progress utility-rule cycles;
- native parser libraries without an exact operator-supplied SHA-256.

Documented recursion through `has`, `inside`, `precedes`, and `follows` remains valid because those relations advance to another syntax node.

Active rules, utilities, tests, snapshots, outline definitions, and trusted native parsers are copied into a private bundle. Its files become read-only before serving. ast-soleaux generates three configurations:

1. inline searches and outlines with no configured project rules;
2. configured scans with only retained `ruleDirs` and `utilDirs`;
3. configured tests with retained rules, utilities, `testConfigs`, and snapshots.

The original files are never reloaded during the session. Mutating them after startup does not alter active behavior.

## Resource limits

| Resource                         |                               Limit |
| -------------------------------- | ----------------------------------: |
| Snippet or stdin input           |                               1 MiB |
| Inline YAML rule                 |                              64 KiB |
| One NDJSON record                |                               1 MiB |
| Aggregate structured output      |                               4 MiB |
| Subprocess diagnostics           |                              64 KiB |
| Configured-test report           |                              64 KiB |
| Startup config file              |                               1 MiB |
| Retained configuration resources |           16 MiB across 1,024 files |
| Trusted native parser library    |                         16 MiB each |
| YAML documents                   |                                  64 |
| YAML nodes                       |                              10,000 |
| YAML depth                       |                                  64 |
| Outline file paths               |                                  64 |
| Oxc module files                 |                                  64 |
| One Oxc source file              |                               2 MiB |
| Aggregate Oxc source             |                              16 MiB |
| One PostgreSQL source file       |                               1 MiB |
| Aggregate PostgreSQL source      |                              16 MiB |
| Hard result ceiling              |                                 500 |
| Windows process creation         |            32,767 UTF-16 characters |
| POSIX process creation           | Runtime `ARG_MAX` minus 2,048 bytes |
| Process termination grace        |                           2 seconds |

NDJSON must be complete, newline-terminated UTF-8. Empty, malformed, non-object, incomplete, oversized, or schema-invalid records fail closed. Unknown upstream object fields are retained.

Before process creation, ast-soleaux also checks the complete command and environment against Windows `CreateProcessW` or POSIX `ARG_MAX` limits.

## Conditional tool API

Every tool advertises accurate FastMCP annotations. Analysis and preview tools are read-only; file mutation tools are destructive and idempotent; TypeScript execution is destructive, non-idempotent, and open-world. Optional families are omitted from `tools/list` unless their configured workers pass startup validation.

| Tool                                                                            | Purpose                                                                                                                              |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `dump_syntax_tree`                                                              | Dumps `pattern`, `cst`, `ast`, or `sexp` syntax for a bounded snippet.                                                               |
| `test_match_code_rule`                                                          | Tests validated inline YAML against a bounded snippet.                                                                               |
| `outline_code`                                                                  | Resolves up to 64 contained files from exact paths and bounded globs, then outlines them with item, type, and public-member filters. |
| `find_code`                                                                     | Searches by pattern with optional selector, strictness, and preview-only rewrite.                                                    |
| `find_code_by_rule`                                                             | Searches with validated inline rules and preserves fix-preview data.                                                                 |
| `oxc_modules`                                                                   | Parses contained JavaScript and TypeScript files and resolves static imports and re-exports with Oxc.                                |
| `oxc_transform` / `oxc_transform_files`                                         | Preview transformation or emit contained artifacts with explicit conflict policy.                                                    |
| `oxc_minify` / `oxc_minify_files`                                               | Preview minification or emit contained artifacts with legal-comment and cache metadata.                                              |
| `semantic_scopes` / `semantic_symbols` / `semantic_references` / `semantic_cfg` | Rust Oxc lexical semantics and CFG.                                                                                                  |
| `inspect_typescript_project`                                                    | One bounded TypeScript Compiler API project snapshot.                                                                                |
| `postgres_parse` / `postgres_parse_files` / `postgres_deparse_preview`          | libpg_query parse, scan, fingerprint, PL/pgSQL, batch, and equivalence-proven deparse.                                               |
| `typescript_execute`                                                            | Execute a contained TypeScript entry only through an operator-approved sandbox runner.                                               |
| `scan_project_rules`                                                            | Runs only rules retained from startup configuration; visible only when configured rules exist.                                       |
| `test_project_rules`                                                            | Runs retained configured suites without interaction or snapshot updates; visible only when configured tests exist.                   |
| `get_server_info`                                                               | Reports versions, configuration provenance, capabilities, coordinates, and limits.                                                   |

### Oxc preview and artifact inputs

`oxc_transform` and `oxc_minify` accept one discriminated `source` object. Use `{"kind":"inline","filename":"entry.ts","code":"..."}` for bounded inline code or `{"kind":"file","project_folder":"/contained/project","path":"src/entry.ts"}` for a contained file. Preview tools never write the file variant. `output_format` changes only the text rendering; structured content remains identical.

Project formatting writes are deliberately outside the MCP server. Install and invoke the project's pinned Oxfmt directly, for example `npm exec -- oxfmt --write <paths>`, so the project owns configuration discovery, changed paths, process lifetime, and editor or CI integration. Transform and minify artifact emission remain explicit MCP mutation tools with a relative `output_root`, bounded path selection, conflict policy, and source-overwrite decision.

### Pattern search and previews

`find_code` requires `pattern` and `language`. Its optional controls are:

- `selector`: selects a node kind from a contextual pattern;
- `strictness`: `cst`, `smart`, `ast`, `relaxed`, `signature`, or `template`;
- `rewrite`: returns replacement and replacement-offset previews without changing files;
- `cursor`: continues the same query from an opaque, five-minute bounded snapshot.

An empty `rewrite` previews deletion. No tool invokes ast-grep rewrite application flags. Pattern parse failures return structured diagnostics with an inline pattern-tree dump, so a separate syntax-dump call is unnecessary.

### Rule search

`find_code_by_rule` accepts one or more YAML rule documents. Optional `positive_code` and `negative_code` probes run before the project scan and skip it when either expectation fails. `text_equals`, `text_starts_with`, and `text_contains` apply bounded literal post-filters to matched AST node text; use them for prefix/substring intent when regex rules are forbidden. The same literal filters are available on `test_match_code_rule`. Match results preserve ast-grep rule fields, metadata, transformed metavariables, replacement text, replacement offsets, and future upstream fields. Text output renders the same preview information.

A fragment that parses as multiple AST nodes needs `pattern.context` plus `pattern.selector`. For example, select a TypeScript switch arm with a complete `switch ($OBJ) { case 'property': return $EXPR }` context and `selector: switch_case`; a bare `case ... return ...` pattern is not a single parseable node.

### Configured scan and test

`scan_project_rules` cannot accept arbitrary rule YAML. It uses only startup-retained rules. Optional `rule_ids` values are checked against the retained catalog and escaped into one exact-match filter. `run_tests_first=true` runs retained tests and skips the scan on failure. Paths and caller globs remain contained and bounded; rule-level `files` and `ignores` continue to use project-relative semantics.

`test_project_rules` runs only startup-retained suites. Exit code 4 is reported as `passed: false`; configuration or execution exit codes fail the tool. It never passes `--interactive`, `--update-all`, or another snapshot mutation option.

### Outline

`outline_code` accepts explicit relative files or bounded glob discovery. `strict_paths` controls whether unresolved exact inputs fail the call or return per-path errors. Results include the canonical `resolved_paths`, so callers never need to guess filenames. It supports:

- `items`: `auto`, `structure`, `exports`, `imports`, or `all`;
- `symbol_types`: exact top-level outline symbol types;
- `public_members`: retain only public members where the language outline defines visibility.

The limit counts top-level items and nested members in preorder. Hierarchy and unknown fields are preserved.

### JavaScript and TypeScript modules

`oxc_modules` accepts exact files or bounded include/exclude globs. It is available only when the operator supplies `--oxc-helper`. The helper parses `.js`, `.jsx`, `.mjs`, `.cjs`, `.ts`, `.tsx`, `.mts`, and `.cts`, returns parser diagnostics and `import.meta` spans, and resolves contained static imports and re-exports. Built-in or out-of-project dependencies are classified as external without exposing their paths. Missing dependencies remain unresolved with a bounded error. `include_dynamic=true` includes dynamic import expressions without pretending that non-static expressions are resolvable.

The tool never writes source, never executes project code, rejects helper output outside `project_folder`, and paginates immutable module snapshots with the same query-bound five-minute cursor contract as searches.

### Result envelopes and coordinates

Search tools return:

```json
{
  "matches": [],
  "returned": 0,
  "truncated": false,
  "limit": 50,
  "next_cursor": null,
  "snapshot_truncated": false
}
```

When `next_cursor` is present, repeat the identical query with that cursor. Cursors are query-bound, expire after five minutes, and never bypass the configured 500-result snapshot cap. Pattern/probe failures return a `diagnostics` object and skip the project scan.

Each match is labeled `evidence_kind: "syntax"`. Matches with a valid source range include `lsp_handoff` (project, relative file, zero-based line/character) for a separate semantic lookup. Outline results include `files`, `resolved_paths`, and `path_errors`. Returned file paths are project-relative and revalidated after ast-grep exits. `get_server_info` lists every built-in and configured custom language ID before callers choose a language.

Coordinates follow ast-grep 0.45 conventions:

- lines are zero-based;
- columns are zero-based Unicode scalar counts;
- byte offsets are zero-based UTF-8 byte offsets;
- Oxc module offsets are zero-based UTF-16 code units;
- ranges are half-open `[start, end)`.

## Deliberate exclusions

Version 0.4.0 does not expose mutation, raw CLI arguments, `--follow`, ignore bypasses, snapshot updates, LSP, HTTP transport, or rewrite application.

It also does not add `inspect_syntax`. Although `ast-grep-py==0.45.0` exists, it lacks the error, missing, extra, and child-field accessors required for a lossless concrete-syntax-tree contract. ast-soleaux does not parse debug stderr or misuse Oxc's ESTree output to approximate an ast-grep concrete-syntax-tree contract.

## Develop and verify

```bash
uv sync --locked --all-extras --dev --no-python-downloads
npm ci --prefix oxc-sidecar
npm run format:check --prefix oxc-sidecar
npm test --prefix oxc-sidecar
uv lock --check
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy ast_soleaux execution-sidecar scripts tests
uv run --no-sync mypy --platform win32 ast_soleaux execution-sidecar scripts tests
uv run --no-sync pyright
uv run --no-sync pyright --pythonplatform Windows
uv run --no-sync pyright --pythonplatform Linux
uv run --no-sync pytest tests/test_unit.py tests/test_config_snapshot.py \
  --cov=ast_soleaux --cov-report=term-missing
AST_GREP_TEST_EXECUTABLE=/absolute/path/to/ast-grep \
  uv run --no-sync pytest tests/test_integration.py
uv run --no-sync python scripts/launch_server.py --help
uv run --no-sync python -m build --sdist --wheel --no-isolation
uv run --no-sync twine check dist/*
uv run --no-sync check-wheel-contents dist/*.whl
uv run --no-sync python tests/package_smoke.py dist/ast_soleaux-0.5.0-py3-none-any.whl dist/ast_soleaux-0.5.0.tar.gz 0.5.0
```

Passing every relevant pytest test is the functional acceptance criterion; the collected-test count is informational rather than an expected value. The default `-ra` summary makes every non-passing outcome visible. Pytest-cov reports statement coverage for `ast_soleaux` as diagnostic evidence only and does not override the test result. Integration acceptance requires an explicit executable reporting exactly ast-grep 0.45.0, the pinned Oxc sidecar dependencies installed under `oxc-sidecar`, and both modern `mode="auto"` and handshake-era `mode="legacy"` MCP connections.

Those commands describe one platform. Acceptance runs on Linux, macOS, and Windows, and path traversal, permissions, and process teardown behave differently on each: inode numbers are reused on Linux and unstable on Windows, and `chmod` sets only the read-only flag on Windows. Type-check the other platforms with `mypy --platform` and `pyright --pythonplatform`, and run the affected tests on Linux in a container before pushing. The path-scoped rules in `.claude/rules/cross-platform-verification.md`, `.claude/rules/filesystem-portability.md`, and `.claude/rules/linux-testing.md` carry the platform-specific commands and assertions.

Distribution verification stays in the repository-owned locked environment: tool execution disables synchronization, builds disable PEP 517 isolation, the wheel is imported directly from its archive, and the sdist is inspected without extraction or installation. The separately installed Oxc sidecar manifest, lockfile, executable, and tests ship in the sdist rather than the pure-Python wheel.

CI synchronizes the locked repository environment once and runs every subsequent Python tool with `uv run --no-sync`. `scripts/launch_server.py` is a thin repository entrypoint to the same server used by the packaged console script.

## Rule-authoring guidance

[`ast-grep.mdc`](ast-grep.mdc) records the official ast-grep agent-skill workflow and rule reference used by this repository. Repository working agreements remain authoritative over generic upstream guidance.

Root [`AGENTS.md`](AGENTS.md) is the shared cross-agent contract: Codex discovers it directly, while Claude Code consumes it through [`CLAUDE.md`](CLAUDE.md), whose complete contents are `@AGENTS.md`. Claude-specific project guidance is independently authored by topic under `.claude/rules/`; rules without `paths` frontmatter load at launch, and path-scoped rules load when Claude reads a matching file.

Pytest guidance also has independent provider-native owners: Codex discovers `.agents/skills/pytest/SKILL.md`, while Claude Code discovers `.claude/skills/pytest/SKILL.md`. Each platform catalogs the skill's concise discovery metadata and loads its full body when the skill is invoked. The two declarations align on the repository's test contract but neither generates, imports, or synchronizes the other.

Codex command-execution policies are independently authored under `.codex/rules/` as Starlark `prefix_rule()` declarations. Codex scans them at startup only when the project `.codex/` layer is trusted, then evaluates them when commands are proposed. They control whether matching command prefixes are allowed, require approval, or are forbidden; they are not prose instruction fragments. Validate them with `codex execpolicy check`, passing each applicable file with `--rules`.
