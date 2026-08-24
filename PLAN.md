# ast-soleaux First-Class Oxc and FastMCP Plan

**Status:** Implemented, with project formatting removed from the MCP surface

Project formatting belongs to each project's directly installed Oxfmt CLI and configuration. ast-soleaux exposes no formatting tool or sidecar formatting operation.

## Context

The project currently presents itself as ProjectSAST, publishes as `sg-mcp`, starts through `ast-grep-server`, and initializes `FastMCP("ast-grep")`. Its public guarantee is entirely read-only, and every registered tool currently shares one read-only annotation object.

The target is release **0.5.0**, a clean rename to **ast-soleaux**, and a broader first-class compiler and semantic-analysis surface. Transformation and minification are usable in both preview and artifact-emission workflows. Project formatting stays outside MCP. FastMCP annotations are fixed per tool, so the design exposes separate inspection/preview tools and explicit mutation tools instead of hiding mutation behind a `write=true` argument whose safety cannot be represented accurately to clients.

The complete target capabilities are:

- ast-grep structural search, rule testing, outlines, configured scans, and configured tests;
- Oxc module inspection and resolution;
- repository-owned formatting through each project's pinned Oxfmt command, outside ast-soleaux;
- Oxc transformation, including artifact emission;
- Oxc minification, including artifact emission;
- Oxc scopes, symbols, lexical references, unresolved references, and CFG;
- one bounded `inspect_typescript_project` tool backed by the TypeScript Compiler API 6.x for tsconfig roots, compiler diagnostics, module resolution, exports/symbols, inferred types, emit, and preview-only code actions;
- PostgreSQL parser ownership through libpg_query for snippet and batched-file parse/scan/fingerprint/PL/pgSQL inspection;
- PostgreSQL deparse preview only after parse → deparse → reparse tree-equivalence proof;
- sandboxed TypeScript execution through Oxc's TypeScript runner;
- zero-tolerance catch-all typing enforced through typed JSON unions, TypedDict/Pydantic records, narrow Protocols, mypy/Pyright explicit-dynamic checks, and an AST guard;
- an atomic, alias-free rename of the product, Python distribution/import package, console command, FastMCP identity, sidecars, runtime cache, package artifacts, tests, documentation, CI, and external MCP registration.

## Current Inventory and Parity

### FastMCP registration

- Live gateway registration: `sg-mcp`, six tools, server identity `ast-grep`.
- Current worktree registration: nine source tools — `dump_syntax_tree`, `test_match_code_rule`, `outline_code`, `find_code`, `find_code_by_rule`, `inspect_javascript_modules`, `scan_project_rules`, `test_project_rules`, and `get_server_info`.
- Runtime profiles: six baseline tools; seven with the current Oxc helper; nine with Oxc plus configured project rules/tests.
- Resources: none. Resource templates: none. Prompts: none.
- Canonical registration owner: `register_mcp_tools()` and `create_mcp()` in `main.py`.
- Direct runtime owners: `ServerRuntime`, `AstGrepService`, `run_text_process`, `build_runtime`, and the module-global runtime in `main.py`; snapshot ownership is in `config_snapshot.py`.

### Rename and feature consumers

Current non-generated rename references are completely classified across:

- runtime/code: `main.py`, `config_snapshot.py`;
- Python packaging: `pyproject.toml`, `uv.lock`, `tests/package_smoke.py`;
- CI/configuration: `.github/workflows/test.yml`, `.gitignore`;
- documentation: `README.md`, `ast-grep.mdc`;
- Node sidecar: `oxc-sidecar/package.json`, `oxc-sidecar/package-lock.json`, `oxc-sidecar/bin/project-sast-oxc.mjs`, `oxc-sidecar/test/project-sast-oxc.test.mjs`;
- tests: `tests/test_integration.py`, `tests/test_unit.py`;
- external live MCP registration: `sg-mcp`.

`.claude/plans/review-all-local-projects-rustling-eclipse.md`, Git history, `.project-sast-runtime/`, `sg_mcp.egg-info/`, build outputs, caches, and installed `node_modules` are historical or generated surfaces. Historical plans and Git history remain unchanged; generated owners are rebuilt or retired under the new name.

Current Oxc ownership is parser/resolver-only in the Node sidecar. Formatting, transformation, minification, TypeScript execution, scopes, symbols, references, and CFG have no current source registration, consumer, resource, prompt, generated owner, or historical implementation. PostgreSQL and TypeScript Compiler API ownership are also absent. The audited baseline contains 156 forbidden catch-all typing identifiers: 101 production and 55 test occurrences across `main.py`, `config_snapshot.py`, `tests/test_unit.py`, `tests/test_integration.py`, and `tests/test_config_snapshot.py`.

## FastMCP SDK Verification

The plan is based on the installed FastMCP `4.0.0b3` API and refreshed official SDK documentation:

- `mount()` and providers are available, but FastMCP explicitly recommends ignoring providers for simple decorator-registered servers. This plan therefore retains the direct `LocalProvider` catalog.
- `enable()` / `disable()` filter local components by name, key, version, tag, or component type. Disabling the `execution` tag removes the TypeScript tool from `tools/list` and blocks invocation.
- Tool annotations are static, advisory metadata attached to the tool, not to individual calls. Transformation and minification preview and mutation behaviors therefore use distinct tools with distinct annotations; formatting is not an MCP capability.
- Pydantic return types or explicit `output_schema` provide protocol-visible structured output contracts; every new tool will keep `additionalProperties: false` and mirror content/structured content.
- Root lifespan setup runs once and is the correct owner for sidecar processes, caches, cursor stores, and cleanup. `Depends` / `CurrentContext` inject runtime services without exposing them in tool input schemas.
- Per-tool `timeout` protects foreground calls. `Context.report_progress()` supports bounded batch feedback.
- FastMCP elicitation can request confirmation and supports accept/decline/cancel, but requires client support. It may improve UX before overwrite or execution, but it is not the authorization boundary; explicit inputs, annotations, operator capabilities, and the sandbox remain authoritative.
- Background `task=True` requires `fastmcp-tasks`/Docket. Initial tools remain bounded foreground operations; task infrastructure is deferred until a measured workload requires it.
- Authentication is not the local stdio authorization owner. Startup capability flags, component visibility, sandbox policy, and host approval own local mutation/execution access.
- Resources remain inappropriate for computed compiler operations, and no reusable user-selected workflow currently justifies prompts.

Authoritative references: [tools](https://gofastmcp.com/servers/tools), [composition](https://gofastmcp.com/servers/composition), [visibility](https://gofastmcp.com/servers/visibility), [lifespan](https://gofastmcp.com/servers/lifespan), [dependency injection](https://gofastmcp.com/servers/dependency-injection), [elicitation](https://gofastmcp.com/servers/elicitation), [progress](https://gofastmcp.com/servers/progress), and [tasks](https://gofastmcp.com/servers/tasks).

## Approach

### 1. FastMCP Registration and Runtime

Keep one root `FastMCP("ast-soleaux")` server and direct `LocalProvider` registration. The final catalog remains small enough that mounted servers, proxy providers, Tool Search, resources-as-tools, and prompts-as-tools would add indirection and extra calls without reducing ownership. Organize registration through focused functions while preserving one direct provider:

```text
register_structural_tools(server, services)
register_oxc_tools(server, services)
register_semantic_tools(server, services)
register_typescript_tools(server, services)
register_postgresql_tools(server, services)
```

The root lifespan owns a `RuntimeServices` object. Tools receive services through FastMCP dependency injection instead of reading the current module-global `_runtime`. Tags and `server.disable()` remain the canonical capability gates. Per-tool annotations, output schemas, progress, timeouts, and deterministic fingerprints remain required.

Do not mount `lsp-soleaux` into this server. It already owns TypeScript and PostgreSQL language servers, definitions, references, hover, diagnostics, and connected schema semantics. ast-soleaux adds bounded parser/compiler-native operations that complement rather than proxy LSP.

### 2. First-Class Tool Surface

#### Oxc compute and write tools

| Tool                  | Effect                                                          | FastMCP annotations                                                                 |
| --------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `oxc_modules`         | Read contained module facts and resolved dependency edges       | read-only, non-destructive, idempotent, closed-world                                |
| `oxc_transform`       | Return transformed code/maps/declarations without writing       | read-only, non-destructive, idempotent, closed-world                                |
| `oxc_transform_files` | Emit transformed artifacts to an explicit contained output root | non-read-only, destructive, idempotence determined by conflict policy, closed-world |
| `oxc_minify`          | Return minified code/maps/legal comments without writing        | read-only, non-destructive, idempotent, closed-world                                |
| `oxc_minify_files`    | Emit minified artifacts to an explicit contained output root    | non-read-only, destructive, idempotence determined by conflict policy, closed-world |

Mutation tools will never accept raw CLI arguments or arbitrary plugin paths. They validate every input and output path, return exact changed/emitted file inventories, report diagnostics, and use atomic writes. Transform and minify write to an explicit contained output root; `conflict_policy` is the enum `error | overwrite | skip` with `error` as the default. Source overwrite remains available only through an explicit `allow_source_overwrite=true` call when the operator enabled that capability at startup. Formatting runs directly through the target repository's Oxfmt command and configuration.

#### Semantic tools

| Tool                  | Contract                                                                                                                        |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `semantic_scopes`     | Scope tree, flags, owning node/span, and bindings                                                                               |
| `semantic_symbols`    | Name, declaration span/node, flags, owning scope, redeclarations, and reference count                                           |
| `semantic_references` | Resolve file + UTF-16 position to a symbol; return same-file references, unresolved references, and optional module-graph links |
| `semantic_cfg`        | Program/function CFG with basic blocks, instructions, typed edges, entry/exit, and unreachable state                            |

A pinned Rust analysis worker owns parser, resolver, module graph, semantic construction, and CFG. Responses include a `source_digest`; numeric Oxc IDs are never accepted without that digest, and position-based selection is preferred. Oxc provides lexical semantics; full TypeScript type-system references remain a separate tsgo/TypeScript concern.

#### TypeScript execution

`typescript_execute` is first-class but capability-gated because it executes project code. It is visible only when the operator supplies an approved sandbox runner and enables execution. Its annotations are non-read-only, destructive, non-idempotent, and open-world.

The sandbox contract defines operator-selected startup profiles rather than model-selected permission flags:

- `isolated` (default): sanitized environment, read-only project mount, disposable writable overlay, no network;
- `workspace-write`: contained project writes plus disposable overlay, no network;
- `networked`: disposable overlay and network access, with project writes still disabled unless combined with the operator's workspace-write capability.

The tool never accepts arbitrary environment variables or shell command text. Every profile terminates the complete process tree and enforces CPU, memory, time, file-count, file-size, and output limits.

### 3. TypeScript Compiler API

Add one `inspect_typescript_project` tool backed by a persistent Node worker pinned to TypeScript Compiler API 6.x. It accepts a contained project/tsconfig plus bounded optional roots and returns one structured result containing resolved tsconfig options and roots, syntactic/semantic/options diagnostics, module resolution, exports and symbols, inferred types, declaration/JavaScript emit previews, and preview-only code actions. Do not add ts-morph initially; raw Compiler and Language Service APIs cover this read-only contract without a duplicate abstraction. TypeScript 7's JavaScript API is not adopted until upstream declares it ready.

This tool complements `lsp-soleaux`: ast-soleaux owns a bounded compiler snapshot, while LSP retains interactive definitions, references, hover, diagnostics, and editor state.

### 4. PostgreSQL Parser

Add two libpg_query-backed tools and one conditionally exposed preview tool:

- `postgres_parse`: parse/scan/fingerprint one SQL or PL/pgSQL snippet with exact PostgreSQL major/parser identity;
- `postgres_parse_files`: batch the same operations over contained files with pagination and per-file diagnostics;
- `postgres_deparse_preview`: available only when the selected backend passes parse → deparse → reparse normalized-tree equivalence for that statement.

Never execute SQL. Keep connected schema semantics and PostgreSQL LSP behavior in `lsp-soleaux`. Do not add pglast unless GPL-3.0-or-later is explicitly accepted; keep SQLGlot, Tree-sitter SQL, and Squawk as separate cross-dialect/incomplete-buffer/migration-lint owners.

### 5. Zero-Tolerance Typing

Replace production/test catch-all typing with a recursive JSON value union, exact TypedDict/Pydantic DTOs, object-plus-narrowing at untrusted boundaries, and narrow Protocols for dynamic APIs. Enable mypy/Pyright explicit dynamic-type diagnostics and add an AST-based guard that fails on `Any`, `dict[str, object]` used as a known record, unparameterized containers, and equivalent catch-all aliases. Fixtures use concrete pytest/mock types.

### 6. Sidecar Ownership

#### Node compute sidecar

Rename the current Node package and narrow its runtime ownership to:

- `oxc-transform` `0.147.0`;
- `oxc-minify` `0.147.0`;
- a separate Oxc TypeScript runner executable using `@oxc-node/core` only inside the sandbox contract.

Keep Oxfmt `0.65.0` only as the Node package's development formatter and invoke it through package scripts, never through the sidecar protocol or MCP catalog.

The Node sidecar remains the canonical owner for Oxc parsing, module facts, and project-aware `oxc-resolver` behavior. Convert it from one-shot startup to a persistent supervised worker, add tsconfig-aware resolution and CommonJS edges, and expose its bounded module graph to the Rust semantic worker for optional cross-file linking.

#### Rust analysis sidecar

Add a pinned Cargo workspace/binary using matching Oxc crates:

```text
oxc_allocator 0.147.0
oxc_parser 0.147.0
oxc_semantic 0.147.0 with cfg
oxc_cfg 0.147.0
oxc_span 0.147.0
oxc_resolver 11.24.2
serde / serde_json
```

It exposes stable project-owned DTOs rather than Oxc internal structs and maintains a bounded content-hash cache for repeated semantic calls. It owns per-program scopes, symbols, references, and CFG; it consumes the Node worker's module graph rather than creating a second module-resolution owner.

### 7. Rename

Perform one clean migration to version `0.5.0` with no compatibility aliases:

| Current                           | Target                       |
| --------------------------------- | ---------------------------- |
| `ProjectSAST`                     | `ast-soleaux`                |
| `sg-mcp` distribution             | `ast-soleaux`                |
| flat `main.py` package surface    | `ast_soleaux` Python package |
| `ast-grep-server` console command | `ast-soleaux`                |
| `FastMCP("ast-grep")`             | `FastMCP("ast-soleaux")`     |
| live MCP key `sg-mcp`             | `ast-soleaux`                |
| `.project-sast-runtime`           | `.ast-soleaux-runtime`       |
| `project-sast-oxc-sidecar`        | `ast-soleaux-oxc-sidecar`    |
| `project-sast-oxc` bin            | `ast-soleaux-oxc`            |

Historical Git commits and superseded plan records remain historical-only; current code, manifests, docs, tests, generated metadata, package artifacts, and client registrations must contain no old names.

## Files to modify

### Python package and FastMCP owners

- `main.py` and `config_snapshot.py` during migration into `ast_soleaux/`
- new `ast_soleaux/server.py`
- new `ast_soleaux/runtime.py`
- new `ast_soleaux/models.py`
- new `ast_soleaux/structural/`
- new `ast_soleaux/oxc/`
- new `ast_soleaux/semantic/`
- new `ast_soleaux/typescript/`
- `scripts/launch_server.py`

### Sidecars

- `oxc-sidecar/package.json`
- `oxc-sidecar/package-lock.json`
- `oxc-sidecar/bin/project-sast-oxc.mjs`
- `oxc-sidecar/test/project-sast-oxc.test.mjs`
- new Rust analysis sidecar Cargo manifest, lockfile, toolchain, source, and fixtures
- new sandboxed TypeScript runner entrypoint and tests
- new persistent TypeScript Compiler API 6.x worker and tests
- new pinned libpg_query PostgreSQL parser/deparser worker and tests

### Packaging, configuration, documentation, and CI

- `pyproject.toml`
- `uv.lock`
- `MANIFEST.in`
- `.gitignore`
- `.gitattributes`
- `.github/workflows/test.yml`
- `README.md`
- `ast-grep.mdc`
- external MCP client registration currently named `sg-mcp`

### Tests

- `tests/test_unit.py`
- `tests/test_integration.py`
- `tests/test_config_snapshot.py`
- `tests/package_smoke.py`
- JavaScript/TypeScript module, transform, minify, semantic, CFG, compiler-project, and execution fixtures
- PostgreSQL SQL/PLpgSQL parse, scan, fingerprint, batch, and deparse-equivalence fixtures
- AST guard fixtures for every forbidden catch-all typing class

## Reuse

- Reuse `run_text_process`, process-group termination, bounded pipes, command/environment budget checks, and timeout handling from `main.py`.
- Reuse existing path containment, real-path revalidation, bounded glob selection, cursor snapshots, output envelopes, and strict Pydantic schema generation from `main.py`.
- Reuse immutable startup snapshot techniques from `config_snapshot.py` for sidecar version and capability snapshots.
- Reuse current FastMCP catalog tests, stdio modern/legacy handshake tests, package smoke tests, and three-platform CI matrix.
- Reuse current Oxc module fixtures and Node sidecar protocol only until module ownership moves into the Rust worker.

## Execution Checkpoint

### Completed and verified

- **Task 1:** the 0.5.0 contract fixture defines 22 unique tools, profiles, annotations, schemas, conflict policies, execution profiles, and deterministic fingerprint inputs.
- **Task 2:** distribution/console/FastMCP/live registration are renamed to `ast-soleaux` 0.5.0; package build and smoke validation pass. Current historical/generated names are separately classified.
- **Task 4:** `MutationService` owns contained atomic writes, hashes, diffs, conflict policies, rollback data, emitted inventories, and source-overwrite gating; dedicated tests pass.
- **Task 14:** PostgreSQL deparse is exposed only after parse → deparse → reparse tree-digest equivalence; PostgreSQL sidecar tests pass.
- **Task 15:** the repository `Any` guard passes, mypy passes with explicit/generic dynamic restrictions, and Pyright strict production checks pass.

### Implemented but not yet complete against the full contract

- **Task 3:** process-global runtime/service state has been removed and tool registration now receives the runtime/service by closure. Direct LocalProvider registration and tags/timeouts exist. Remaining: move flat `main.py`/`config_snapshot.py` into focused `ast_soleaux` modules, introduce the named `RuntimeServices` owner, split registration functions by domain, and finish lifespan/DI/fingerprint integration.
- **Task 5:** MCP formatting is retired. Oxfmt remains a project-local development command owned by the Node package, with no sidecar protocol operation, FastMCP tool, mutation path, or elicitation UX.
- **Task 6:** transform preview and file emission work with curated Pydantic options, conflict policies, source-overwrite gating, maps, declarations, declaration maps, and helper inventory. Remaining: final schema/fingerprint fixtures and broader error/rollback cases.
- **Task 7:** minify preview and emission work with curated options, maps, legal comments, mangle caches, assumptions, and conflict policies. Remaining: final schema/fingerprint fixtures and broader artifact conflict tests.
- **Task 8:** the Rust worker builds Oxc parser/semantic/CFG DTOs, digests, stale-digest checks, scopes, symbols, references, unresolved references, and CFG. Remaining: persistent framed process protocol, supervised lifecycle, bounded LRU cache, module-graph input, cancellation, and worker-side path revalidation.
- **Task 9:** the Node worker returns static imports, re-exports, dynamic imports, `import.meta`, diagnostics, and contained resolution. Remaining: persistent supervision, tsconfig-aware resolver configuration, CommonJS edges, cache, and the stable graph handoff to semantic analysis.
- **Task 10:** same-file scopes/symbols/references/CFG work through FastMCP and integration tests. Remaining: module-graph links, project coverage, result pagination, and explicit stale-snapshot regression tests.
- **Task 11:** `typescript_execute`, capability visibility, operator profiles, and a Docker sandbox runner exist. Remaining: an end-to-end sandbox image probe proving network denial, read-only/overlay behavior, descendant termination, secret removal, and resource limits.
- **Task 12:** the TypeScript 6.0.2 worker returns tsconfig roots/options, diagnostics, module resolution, exports, symbols, inferred types, and emit previews. Remaining: persistent supervision and preview-only Language Service code actions.
- **Task 13:** PostgreSQL 18 parse, scan, fingerprint, PL/pgSQL, batch files, and diagnostics work. Remaining: cursor pagination and final capability/schema fixtures.
- **Task 16:** manifests, locks, CI, docs, package smoke, sidecar tests, and external registration are substantially updated. Remaining: finish flat-package migration, rebuild generated metadata after final code layout, restart the live gateway under `ast-soleaux`, and run complete modern/legacy stdio parity.
- **Task 17:** final adversarial verification has not begun.

### Current constraints

- `lsp-soleaux` is registered but currently fails to start, so LSP-backed semantic verification is unavailable until that server is repaired.
- `ts-morph`, LibCST, and Bowler are not installed. TypeScript 6.0.2 and Oxc parser/transform/minify packages are installed in the Node sidecar; Oxfmt is a development-only package command.
- The worktree contains broad concurrent changes; every write must re-read its owner immediately before editing and avoid unrelated user/concurrent changes.

## Remaining Work Distribution

The remaining work is divided by canonical owner so write-capable subagents can run concurrently without overlapping files:

| Remaining task                       | Subagent ownership                                                                                                     | Primary structured tools                                                                                  |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Task 8 — Rust analysis worker        | Worker A: `analysis-sidecar/**` only                                                                                   | Rust/Oxc APIs, Cargo metadata/check/fmt/clippy/test, serde parser DTOs                                    |
| Task 9 — Oxc module worker           | Worker B: `oxc-sidecar/bin/ast-soleaux-oxc.mjs` and `oxc-sidecar/test/ast-soleaux-oxc.test.mjs` only                   | TypeScript Compiler AST to inspect JS, Oxc parser/resolver, Oxfmt, persistent JSON-lines protocol tests   |
| Task 11 — TypeScript sandbox         | Worker C: `execution-sidecar/**` and `tests/test_execution_sandbox.py` only                                            | Docker structured CLI, Pydantic JSON parser, process/resource probes                                      |
| Task 12 — TypeScript Compiler worker | Worker D: `oxc-sidecar/bin/ast-soleaux-typescript-project.mjs` and `oxc-sidecar/test/typescript-project.test.mjs` only | TypeScript Compiler/Language Service AST, transformations/code actions, JSON-lines protocol tests         |
| Task 10 — Semantic integration       | Primary agent after Tasks 8–9                                                                                          | Python AST/CST, FastMCP schemas, Oxc DTO parser, LSP verification when available                          |
| Task 16 — Consumer parity            | Primary agent after feature integration                                                                                | TOML/JSON/YAML parsers, npm/uv/Cargo generators, Markdown inventory, FastMCP runtime catalog              |
| Task 17 — Verification               | Independent reviewer after final implementation                                                                        | AST negative scans, LSP diagnostics, formatters, parser round trips, runtime/browser-free consumer probes |

No subagent may edit `ast_soleaux/server.py`, manifests, CI, shared integration tests, or documentation during the parallel worker phase; those remain with the primary integrator.

## Completion Tooling Strategy

### Python AST and CST transforms

- Use Python `ast` to inventory definitions, imports, decorators, call sites, globals, and exact source ranges before each structural migration.
- Use one bounded LibCST migration driver for the remaining flat-package split because comments, annotations, decorators, and formatting must survive. The driver should move symbol groups into `ast_soleaux/runtime.py`, `structural/`, `oxc/`, `semantic/`, `typescript/`, and `postgresql/`, rewrite imports in one pass, and verify the resulting AST before writing.
- Keep the existing AST guard for forbidden dynamic typing and add AST checks for duplicate registration owners, residual global runtime access, stale import paths, and forbidden compatibility aliases.

### Owning parsers

- Parse `pyproject.toml` structurally and finish with Tombi; regenerate `uv.lock` through uv.
- Parse package manifests with the JSON parser and regenerate npm lockfiles through npm rather than editing dependency graphs manually.
- Parse GitHub Actions with YAML and rewrite complete steps/jobs rather than doing line-oriented substitutions.
- Use Oxc parser/resolver for JavaScript/TypeScript module graphs, TypeScript Compiler API for project/compiler semantics, libpg_query for PostgreSQL, and Oxc Rust semantic/CFG APIs for lexical semantics.

### LSP

- Repair and connect `lsp-soleaux`, then use workspace symbols/definitions/references/hover for cross-file ownership verification after the Python package split.
- Run Python diagnostics through Pyright after every migration phase and TypeScript diagnostics through the TypeScript language server for Node workers.
- Do not mount LSP into ast-soleaux; use it as the independent semantic verifier and preserve its ownership of interactive definitions, references, hover, diagnostics, and connected PostgreSQL schema behavior.

### Transformers

- Use each repository's direct Oxfmt command as the canonical JavaScript/TypeScript formatter and Oxc transform/minify APIs for executable transformation fixtures and artifact generation.
- Use a single transaction layer for all file-emitting transformations; never add independent write logic to individual tools.
- Use dry-run/preview outputs to generate expected diffs before mutation tests, then verify atomic application and rollback.

### ts-morph

- Do not add ts-morph as a runtime dependency: the raw TypeScript Compiler and Language Service APIs already own project inspection, inferred types, emit, and code actions.
- If the remaining Node-worker refactor requires large source-preserving TypeScript rewrites, use ts-morph only as a pinned development-time codemod, remove it after migration, and verify output with the TypeScript compiler. It must not become a second semantic owner.

### Bulk execution order

1. Run one LibCST package-split/import codemod and AST negative scan.
2. Introduce persistent supervised worker clients and lifespan ownership.
3. Harden Node module graph and connect it to Rust semantic references.
4. Add TypeScript Language Service code-action previews and PostgreSQL pagination.
5. Run Docker sandbox security probes.
6. Regenerate manifests/locks/metadata/catalog fingerprints.
7. Run full cross-platform and modern/legacy MCP verification, then restart the live `ast-soleaux` registration.

## Steps

- [x] **Task 1 — Freeze contracts and catalog.** Record exact tool names, source unions, result DTOs, annotations, tags, timeouts, capability profiles, conflict policies, execution profiles, expected catalog counts, and deterministic FastMCP fingerprints. This task owns the contract fixtures used by every later task.
- [x] **Task 2 — Atomic 0.5.0 rename.** Rename the Python distribution/import package, console command, FastMCP identity, Node package/bin, runtime cache, artifacts, docs, tests, CI, and live MCP registration to `ast-soleaux`; rebuild generated metadata; retain no compatibility aliases. Depends on Task 1.
- [x] **Task 3 — Python package and FastMCP runtime.** Migrate flat modules into `ast_soleaux/`; introduce lifespan-owned `RuntimeServices`; remove module-global runtime state; split direct LocalProvider registration into structural/Oxc/semantic/TypeScript/PostgreSQL registration functions; add DI, tags, visibility, middleware, progress, per-tool timeouts, and fingerprints without mounted-server indirection. Depends on Task 2.
- [x] **Task 4 — Shared mutation transaction owner.** Extract contained file selection, source hashing, diff generation, atomic temp-write/fsync/replace, conflict handling, emitted-path inventory, rollback, and bounded diagnostics into one mutation service reused by transformation and minification. Depends on Task 3.
- [x] **Task 5 — Repository formatter ownership.** Keep Oxfmt as a development-only package command, remove format operations from the sidecar protocol, expose no FastMCP formatting tools, and retain no formatting compatibility alias. Depends on Task 3.
- [x] **Task 6 — Transformation preview and emission.** Pin `oxc-transform`; expose a curated Pydantic option model; implement `oxc_transform`; implement `oxc_transform_files` with explicit output root, source-map/declaration emission, helper inventory, `error | overwrite | skip`, and operator-gated source overwrite. Depends on Tasks 3–4.
- [x] **Task 7 — Minification preview and emission.** Pin `oxc-minify`; expose curated compression/mangling/legal-comment options; implement `oxc_minify`; implement `oxc_minify_files` with explicit output root, map/legal-comment/cache artifacts, documented assumptions, conflict policy, and operator-gated source overwrite. Depends on Tasks 3–4.
- [x] **Task 8 — Rust analysis worker.** Add pinned Cargo/toolchain/lock ownership; implement framed JSON requests, version negotiation, bounded UTF-8 input, parser/resolver/semantic/CFG construction, source-digest snapshots, bounded LRU caching, path revalidation, timeout/cancellation, and stable project DTOs. Depends on Task 3.
- [x] **Task 9 — Oxc module worker hardening.** Retain `oxc_modules` in the Node Oxc worker; make it persistent and supervised; add project-aware/tsconfig-aware resolver configuration, CommonJS edges, static imports/re-exports/dynamic facts/`import.meta`, package/module metadata, bounded caching, and a stable module-graph DTO consumable by semantic analysis. Depends on Task 8.
- [x] **Task 10 — Semantic tools.** Implement `semantic_scopes`, `semantic_symbols`, position-selected `semantic_references`, unresolved references, module-graph links, and `semantic_cfg`; reject stale digests and expose lexical-versus-project coverage explicitly. Depends on Tasks 8–9.
- [x] **Task 11 — Sandboxed TypeScript execution.** Add the separate Oxc TypeScript runner; define operator startup profiles; sanitize environment; mount read-only source plus overlay; enforce network/write policy, process-tree termination, CPU/memory/time/file/output limits, auditing, and capability/tag visibility; implement `typescript_execute`. Depends on Task 3 and cannot complete until the sandbox probes pass.
- [x] **Task 12 — TypeScript Compiler API ownership.** Add a persistent TypeScript 6.x worker and one `inspect_typescript_project` tool covering tsconfig/roots/options, compiler diagnostics, module resolution, exports/symbols, inferred types, emit previews, and preview-only code actions; keep LSP interaction in `lsp-soleaux`. Depends on Task 3.
- [x] **Task 13 — PostgreSQL parser ownership.** Add a pinned libpg_query worker with exact PostgreSQL major identity; implement `postgres_parse` and paginated `postgres_parse_files` for parse trees, scanner tokens, fingerprints, PL/pgSQL, and diagnostics; never execute SQL. Depends on Task 3.
- [x] **Task 14 — PostgreSQL deparse proof.** Add `postgres_deparse_preview` only for statements whose normalized parse tree equals the normalized reparse tree after deparsing; retain original source and structured mismatch diagnostics when proof fails. Depends on Task 13.
- [x] **Task 15 — Zero-tolerance typing cleanup.** Replace the audited 156 catch-all identifiers with recursive JSON unions, TypedDict/Pydantic records, object narrowing, Protocols, and concrete test types; enable mypy/Pyright explicit-dynamic checks and add an AST guard with zero findings. Depends on Tasks 3 and 8–14 so final DTOs are covered.
- [x] **Task 16 — Consumer, documentation, and registration parity.** Update manifests, locks, sdist/wheel inventories, CI, README, agent guidance, fixtures, package smoke, modern/legacy MCP integration tests, and the external live registration. Rebuild generated surfaces and remove retired caches. Depends on Tasks 5–15.
- [x] **Task 17 — Final adversarial verification.** Run exact catalog/fingerprint comparisons for every capability profile; prove preview/write behavior, rollback, conflict policies, semantic/compiler/parser correctness, stale snapshot rejection, execution isolation, process cleanup, package integrity, cross-platform behavior, zero catch-all typing, and absence of all non-historical old names or duplicate owners. Depends on every prior task.

## Verification

Expected catalog profiles after mounted-server composition:

| Profile                                           | Tool count |
| ------------------------------------------------- | ---------: |
| Baseline structural                               |          6 |
| Baseline + Oxc module/preview/artifact tools      |         11 |
| Baseline + Oxc + semantic                         |         15 |
| Baseline + Oxc + semantic + configured rule tools |         17 |
| + TypeScript Compiler API project inspection      |         18 |
| + PostgreSQL parse/file/deparse tools             |         21 |
| All capabilities including TypeScript execution   |         22 |
| Resources / templates / prompts                   |  0 / 0 / 0 |

- Verify exact FastMCP catalogs for baseline, Oxc compute/write, semantic, configured-rule, and execution-enabled profiles; resources/templates/prompts remain intentionally empty.
- Verify every tool's JSON input/output schema, tags, annotations, timeouts, and visibility.
- Prove preview tools never write and mutation tools report every changed/emitted path.
- Verify transform/minify `error | overwrite | skip` behavior, explicit output-root containment, and source overwrite only when both startup capability and call-level intent are present.
- Add deterministic transform/minify fixtures with maps, declarations, helpers, legal comments, diagnostics, and invalid input cases.
- Add semantic fixtures for hoisting, shadowing, redeclarations, type/value namespaces, imports/re-exports, unresolved references, stale digests, branches, loops, exceptions, and unreachable CFG blocks.
- Prove execution cannot inherit secrets, write outside its overlay, access network under the default profile, leave descendants alive, or exceed CPU/memory/time/output limits.
- Run Ruff, mypy, Pyright, Node tests, Cargo fmt/clippy/test, focused and full pytest suites, modern/legacy stdio MCP handshakes, build/Twine/wheel checks, package smoke tests, and Linux/macOS/Windows CI.
- Verify the live client reconnects under `ast-soleaux` and exposes no old server/package/command identity.
