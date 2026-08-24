# Update local consumers of projects/ast to 0.4.0

## Context

0.4.0 was merged to main this session (`091efcd`). A full survey of `~/projects/*` found **exactly two live consumers**, both registering the ast-grep MCP server as a *live-checkout launch* against the ast repo's own `.venv`:

- **anilize** — `.mcp.json` server `ast-grep` (plus identical strings in `.codex/config.toml`, `opencode.json`): `uv --directory <sibling ../ast> run --no-sync python scripts/launch_server.py --ast-grep node_modules/.bin/ast-grep --allowed-root $ROOT --config sgconfig.yml --command-timeout 30 --default-max-results 25 --max-results-cap 100 --forbid-regex-rules`
- **cleat-chasers** — `.mcp.json` → `scripts/ast-grep/mcp-wrapper.mjs`: same launch, flags `--ast-grep`, `--allowed-root`, `--forbid-regex-rules`

The venv's sg-mcp is an **editable** install (`direct_url.json`: `"editable":true`), so consumers already serve the 0.4.0 code from the checkout — no consumer pins anything, and no consumer file needs a version bump.

## Steps

### 1. ast repo — verify the launcher interface

From `/Users/johnmclaughlin/projects/ast`:

1. Run `uv sync --locked --all-extras --dev --no-python-downloads`.
2. Run `uv run --no-sync python scripts/launch_server.py --help` and confirm every flag consumers pass still exists at 0.4.0: `--ast-grep`, `--allowed-root`, `--config`, `--command-timeout`, `--default-max-results`, `--max-results-cap`, `--forbid-regex-rules`.

### 2. anilize — verify the launch path (no edits expected)

From `/Users/johnmclaughlin/projects/anilize`:

- Run the exact `ast-grep` command string from `.mcp.json:13` with stdin at EOF: `bash -lc '<command>' </dev/null`. The server serves stdio, sees EOF, and exits cleanly (behavior pinned by `tests/test_integration.py::test_stdio_server_exits_on_eof_without_a_process_survivor`). Expect exit 0 and no surviving process.
- Prerequisites already confirmed present: `scripts/development/mcp/wrapper.sh`, `sgconfig.yml`, `node_modules/.bin/ast-grep` (0.45.0 == `SUPPORTED_AST_GREP_VERSION`, `main.py:52`).
- Only if step 1.3 shows a renamed/removed flag: update the command string in lockstep in `.mcp.json`, `.codex/config.toml`, `opencode.json`.

### 3. cleat-chasers — verify the launch path (no edits expected)

From `/Users/johnmclaughlin/projects/cleat-chasers`:

- `node scripts/ast-grep/mcp-wrapper.mjs --status` → expect `available: true`, `sastDirectory: /Users/johnmclaughlin/projects/ast`, an ast-grep binary resolved.
- `node scripts/ast-grep/mcp-wrapper.mjs </dev/null` → expect clean exit on EOF, no surviving process.
- Only if a flag changed: update `mcp-wrapper.mjs` plus the three docs that describe it (`.claude/rules/25-agent-mcp-fastmcp.md`, `.codex/rules/25-agent-mcp-fastmcp.md`, `docs/standards/42-agent-mcp-fastmcp.md`).

## Report-only findings (no edits)

- **soleaux** is not a consumer (no registration, dependency, or venv install). Pre-existing defect: `tests/test_capability_cutover.py:104` (`test_ast_grep_mcp_ledger_preserves_the_cli_owner`) loads fixture `ast-grep-mcp-capability-map.json`, which exists only in the anilize-temp archive — that test errors in soleaux today. Unrelated to 0.4.0; report, don't fix.
- **anilize-temp**: historical doc references only; nothing installed.
- All other `~/projects/*` (anilize2, cclsp, claude-skill-enforcer, claudex, fantasy-fb, supaschema): no references.

## Verification

- `launch_server.py --help` exits 0 with all consumer flags present.
- Both consumers' bounded EOF launches exit 0 with no surviving server process.
- `git status` in the ast repo clean except untracked `.claude/`; nothing to push (consumer edits only if a flag check fails).
