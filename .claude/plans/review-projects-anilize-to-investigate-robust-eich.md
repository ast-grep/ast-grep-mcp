# Repair the anilize Claude hook lane by completing Wave 3's critical path

All work targets `/Users/johnmclaughlin/projects/anilize` (not this repo).

## Context

anilize is mid-way through its own migration plan (`.claude/plans/serene-greeting-blum.md`, "Sever the Codex → Claude context sync"). Waves 1, 2, and 4 are done on disk, all uncommitted. Wave 3 half-landed, in the wrong order: the strip landed before the repoint.

**Root cause, reproduced:**
1. All 28 hook events in `.claude/settings.json` still invoke the old adapter: `args: ["${CLAUDE_PROJECT_DIR}/scripts/hooks/adapters/claude.mjs", "<Event>"]` (28 occurrences, verified).
2. That adapter hardcodes `runtime: 'claude'` (`scripts/hooks/adapters/claude.mjs:44-45`).
3. An uncommitted Wave 3 edit stripped the claude inventory from `scripts/hooks/registry.mjs` — `RUNTIME_EVENTS` is now `Object.freeze({ codex: … })` (:77), so `policiesForEvent('claude', …)` throws `Unsupported hook runtime: claude` (:79-83), outside `dispatchHookEvent`'s try.
4. `runtime.mjs:22-28` fail-closes it as `Invalid claude <Event> hook input: …` → the adapter exits 2 (`adapters/claude.mjs:41`; PostToolBatch instead prints `{decision:'block'}`). Every event blocks; UserPromptSubmit blocking means the session cannot accept prompts at all.

Repro: old adapter exits 2 with that exact message on a minimal UserPromptSubmit JSON. The new Claude-owned tree (`.claude/hooks/`, 26 files, untracked, workspace-registered, deps installed) exits 0 on identical input, and its policies load (a `git checkout -b` PreToolUse probe returns `"permissionDecision":"deny"`). Its `repoRoot` resolution is correct (`adapters/claude.mjs:11` → repo root).

**Also broken today, fixed by this plan:** both registry CLIs. Each `native-config.mjs` still renders *both* runtimes while each registry passes single-runtime events, so `--check`/`--write` die with a TypeError (`scripts/…:216` on `events.claude`; `.claude/hooks/…:220` on `events.codex`). Running the claude tree's `--write` today would also re-emit the old adapter path and clobber `.codex/hooks.json`.

**Direction (decided):** complete Wave 3 forward — repoint to the Claude tree. Reverting the registry strip was rejected: it would undo the user's own in-progress migration, and the new tree is verified working. The Codex lane (`.codex/hooks.json` → `adapters/codex.mjs` → codex-only registry) is self-consistent and unbroken; keep it byte-exact (baseline sha256 `9abd6784c5be623012548c6ef83580dede0b1b9230a612e9d58e74b23c25ba6e`).

**Collision constraint:** the worktree holds another session's uncommitted work in `scripts/hooks/{command-mutations,command-shell-ast,commands,path,targets}.mjs`, `policies/{safety,safety-command}.mjs`, `map.json`, and `packages/testing/src/hooks/{safety,targets}.test.mjs`. None of those are touched below. `scripts/hooks/native-config.mjs` is tracked-clean (verified via `git status`).

## Step 0 — Snapshot (before any edit)

```bash
tar czf ~/anilize-hookfix-backup.tgz -C /Users/johnmclaughlin/projects/anilize \
  .claude/settings.json .claude/hooks/native-config.mjs \
  scripts/hooks/native-config.mjs scripts/context/source-classification.mjs .codex/hooks.json
```

Rollback = `tar xzf ~/anilize-hookfix-backup.tgz -C /Users/johnmclaughlin/projects/anilize` (the Step 5 test file is purely additive; delete it). Tar, not `git stash`: the worktree entangles other sessions' uncommitted work.

## Step 1 — `.claude/hooks/native-config.mjs` → claude-only renderer

Must land as one unit; a partial edit would make `--write` destructive.

- `:42` — change `'{CLAUDE_PROJECT_DIR}/scripts/hooks/adapters/claude.mjs'` to `'{CLAUDE_PROJECT_DIR}/.claude/hooks/adapters/claude.mjs'` (keep the `['$', …].join('')` construction).
- Delete `codexHooks` (:57-76) and the then-unused `json()` (:201-203).
- `renderNativeHookFiles` (:213-222): drop the `codex:` entry; return `Object.freeze({ claude: withSingleTrailingNewline(killSwitchPinned) })`.
- `writeNativeHookConfigs` (:224-229): drop the `.codex/hooks.json` write (:227).
- `checkNativeHookConfigs` (:231-247): replace the whole-file byte compare with the owned-property compare (the plan's "narrow the staleness check"; also kills the pre-existing trailing-newline `test:hooks` failure class):

```js
import { isDeepStrictEqual } from 'node:util';   // add to imports

export function checkNativeHookConfigs({ events, root = repoRoot }) {
  const settingsPath = resolve(root, '.claude/settings.json');
  const parsed = existsSync(settingsPath) ? JSON.parse(readFileSync(settingsPath, 'utf8')) : null;
  if (
    !parsed ||
    !isDeepStrictEqual(parsed.hooks, claudeHooks(events.claude)) ||
    parsed.disableAllHooks !== false
  ) {
    throw new Error('Stale native hook configuration: .claude/settings.json');
  }
  return renderNativeHookFiles({ events, root });
}
```

Keep untouched: `claudeMatchers` (:9-19), the full `hookTimeouts` table (:21-31, all claude events), the JSON splice machinery (:78-199) and `withSingleTrailingNewline` (:205-211) — the claude render still uses them — and `repoRoot` (:5, correct at this depth).

## Step 2 — Repoint the 28 registrations via the fixed generator

Use the CLI, not a hand edit: `renderNativeHookFiles` splices only the `hooks` and `disableAllHooks` values into the existing file text (`replaceTopLevelJsonProperty`, :176-199), so the unrelated uncommitted settings keys (`outputStyle`, `permissions.deny`, `disabledMcpjsonServers`, `sandbox.…denyRead`) pass through byte-for-byte.

```bash
cd /Users/johnmclaughlin/projects/anilize
node .claude/hooks/registry.mjs --check   # EXPECT exit 1: "Stale native hook configuration" (a TypeError here means Step 1 is incomplete — stop)
node .claude/hooks/registry.mjs --write
node .claude/hooks/registry.mjs --check   # EXPECT exit 0, silent
git diff -- .claude/settings.json          # EXPECT: only the 28 args[0] lines beyond the pre-existing unrelated hunks
grep -c 'scripts/hooks/adapters/claude.mjs' .claude/settings.json    # EXPECT 0
grep -c '.claude/hooks/adapters/claude.mjs' .claude/settings.json    # EXPECT 28
```

Fallback if the write is refused: hand-edit only the 28 `args[0]` strings, then re-run `--check` to exit 0.

## Step 3 — `scripts/hooks/native-config.mjs` → codex-only (fixes the `--check` TypeError)

- Delete `claudeToolMatcher`/`claudeMatchers` (:6-19) and `claudeHooks` (:35-55).
- Prune `hookTimeouts` (:21-31) to the codex-used keys `{ PermissionRequest: 20, PostToolUse: 150, PreToolUse: 20, Stop: 60, SubagentStop: 30 }` — the removed four are claude-only events; the other codex events fall to the default 10, matching `.codex/hooks.json` today.
- Delete the splice machinery (:78-199) and `withSingleTrailingNewline` (:205-211); keep `json()` and `repoRoot`.
- `renderNativeHookFiles`: return `Object.freeze({ codex: json({ hooks: codexHooks(events.codex) }) })`.
- `writeNativeHookConfigs`: drop the settings write; `checkNativeHookConfigs`: drop the settings entry, keep the byte-exact whole-file compare for `.codex/hooks.json` (single writer, plan-mandated).
- Neither registry CLI needs changes: each already passes its own runtime-scoped `NATIVE_HOOK_EVENTS`.

## Step 4 — Register the new tree with the guards

`scripts/context/source-classification.mjs` (~:24): add `'.claude/hooks/',` to `ownedSourceRoots` after `'.claude/commands/',` (alphabetical; same wave-family as the existing dirty `.claude/skills/` addition).

## Step 5 — Feed-forward guard

The defect class is registration/renderer drift ("strip landed before repoint") that nothing executed. Add `packages/testing/src/hooks/native-config-convergence.test.mjs` (auto-discovered by the `test:hooks` glob; collision-free; spawns rather than imports, so no new workspace dependency):

```js
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const repoRoot = fileURLToPath(new URL('../../../../', import.meta.url));

for (const cli of ['.claude/hooks/registry.mjs', 'scripts/hooks/registry.mjs']) {
  test(`${cli} --check converges`, () => {
    const result = spawnSync('node', [cli, '--check'], { cwd: repoRoot, encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr);
  });
}
```

This exact test fails today and would have caught the break the moment the strip landed, in either tree, in either future drift direction.

## Verification

```bash
cd /Users/johnmclaughlin/projects/anilize
node .claude/hooks/registry.mjs --check && node scripts/hooks/registry.mjs --check   # both exit 0
shasum -a 256 .codex/hooks.json   # EXPECT 9abd6784c5be…ba6e (unchanged)

# New-adapter probes (stdin JSON per the official hook contract):
printf '{"hook_event_name":"UserPromptSubmit","prompt":"ping","cwd":"/Users/johnmclaughlin/projects/anilize","session_id":"probe"}' | node .claude/hooks/adapters/claude.mjs UserPromptSubmit; echo "exit=$?"    # exit=0
printf '{"hook_event_name":"PreToolUse","tool_name":"Bash","tool_input":{"command":"ls -la"},"cwd":"/Users/johnmclaughlin/projects/anilize"}' | node .claude/hooks/adapters/claude.mjs PreToolUse; echo "exit=$?" # exit=0, silent
printf '{"hook_event_name":"PreToolUse","tool_name":"Bash","tool_input":{"command":"git checkout -b tmp-probe"},"cwd":"/Users/johnmclaughlin/projects/anilize"}' | node .claude/hooks/adapters/claude.mjs PreToolUse; echo "exit=$?" # exit=0 + stdout "permissionDecision":"deny" (policies load)
printf '{"hook_event_name":"UserPromptSubmit","prompt":"ping","cwd":"/Users/johnmclaughlin/projects/anilize/packages/testing"}' | node .claude/hooks/adapters/claude.mjs UserPromptSubmit; echo "exit=$?"          # exit=0 (nested cwd)
# PostToolBatch probe: Read-shaped tool_calls ONLY — the repository-write policy runs real formatters on write-shaped batches.
printf '{"hook_event_name":"PostToolBatch","tool_calls":[{"tool_name":"Read","tool_input":{"file_path":"/Users/johnmclaughlin/projects/anilize/package.json"},"tool_response":{"success":true}}],"cwd":"/Users/johnmclaughlin/projects/anilize"}' | node .claude/hooks/adapters/claude.mjs PostToolBatch; echo "exit=$?"  # exit=0, no block JSON

# Codex regression:
printf '{"hook_event_name":"PreToolUse","tool_name":"Bash","tool_input":{"command":"ls"},"cwd":"/Users/johnmclaughlin/projects/anilize"}' | node scripts/hooks/adapters/codex.mjs PreToolUse; echo "exit=$?"      # exit=0

# Guard + lint:
pnpm --filter @anilize/testing exec node --test "src/hooks/native-config-convergence.test.mjs"
pnpm exec biome check .claude/hooks scripts/hooks/native-config.mjs scripts/context/source-classification.mjs

# End-to-end: a fresh session snapshots the fixed settings and must answer normally.
claude -p 'ping'
```

The currently blocked anilize session recovers only after restart/resume — Claude Code snapshots hook config at session start. Nothing is committed by this fix; the whole migration is uncommitted by design, and committing needs explicit pathspecs against the entangled worktree.

## Deferred, named residuals (report; do not do)

- `scripts/hooks/adapters/claude.mjs` stays: inert (nothing invokes it, fails closed if run) and still referenced by `contracts/linkage/benchmark` tests and rule `14:62`; delete in Wave 6 with the test retargets.
- Rules still forbid `.claude/hooks/**` (`16F-hook-authoring-standard.md:66`, rule 18 lane table) until Wave 5 rewrites them.
- Hook test suite stays red until Wave 6: `contracts/linkage/upstream-catalog/benchmark` tests import `@anilize/scripts/hooks/registry` expecting a claude inventory. Retargeting is collision-safe but needs a `@anilize/claude-hooks` dependency — one coherent Wave 6 rework.
- Provider-shape leaks duplicated in both trees (plan :79); `.claude/hooks/map.json` via `pnpm code-map:build` later (also touches other-session-dirty `scripts/hooks/map.json` — coordinate); the claude tree froze mid-flight copies of the other session's files — re-sync when that work lands.

## Critical files

- `/Users/johnmclaughlin/projects/anilize/.claude/hooks/native-config.mjs`
- `/Users/johnmclaughlin/projects/anilize/scripts/hooks/native-config.mjs`
- `/Users/johnmclaughlin/projects/anilize/.claude/settings.json` (generator-written)
- `/Users/johnmclaughlin/projects/anilize/scripts/context/source-classification.mjs`
- `/Users/johnmclaughlin/projects/anilize/packages/testing/src/hooks/native-config-convergence.test.mjs` (new)
