# Superseded: Repair the skill gate against upstream hook semantics

> **Status: Superseded.** The owner-routing and `skill-owners.json` design below was not adopted. The current hook deliberately uses generic per-context triage and digest freshness. Its canonical contract is `~/.claude/hooks/AGENTS.md` plus `~/.claude/rules/skills.md`; do not create `.claude/skill-owners.json` from this historical plan.

## Context

`~/.claude/hooks/PreToolUse/skill-gate.py` enforces that a governing skill loads before anything is authored.
Every branch it covers works, verified by feeding it synthetic payloads: deny on a new path asks for `architecture` and `writing-guidelines`, deny on an existing path asks for `writing-guidelines`, the allow branch exits 0 silently, a subagent inherits nothing from its parent, a relative path resolves against the payload `cwd`, and malformed stdin fails closed.
Upstream confirms the gate cannot be bypassed by permission mode: "A hook that returns `permissionDecision: "deny"` blocks the tool even in `bypassPermissions` mode", which is this machine's `defaultMode`.

The gate is nonetheless wrong in one structural way and four mechanical ones.

Existence is its only decision axis.
`required = ALWAYS if edits_existing_path(data) else ON_NEW_PATH` (line 125) selects between two hard-coded lists, and `edits_existing_path` (lines 69-79) calls `os.path.exists` and nothing else.
It never reads the extension or the language, so a `.py`, a `.ts`, and a `.md` edit resolve to an identical requirement, and `python`, `ruff`, `zod`, and `turborepo` can never be required.
`ast/AGENTS.md:26` states "Read the `python` skill and the workflow it routes to before editing", and the gate has no mechanism to honour it.
The rule and the hook agree with each other because both omit language owners; the routing exists only in prose.

This plan gives the gate a second axis, then fixes the four defects that block that axis from working.

## Defects, with the evidence for each

1. **No artifact-type routing.** Above. The gate cannot express which owner should have applied.

2. **An unresolvable required skill denies forever, with false advice.** `missing_skills` reads `if not digest(skill) or current.get(skill) != digest(skill)`. `digest` returns `""` when `SKILL.md` is unreadable, so the skill counts as missing whatever the marker holds. Probing the module directly: recording `no-such-skill` leaves `missing_skills` still returning `['no-such-skill']`. The deny then says "Call the Skill tool with skill=X, then repeat this edit", which loops forever. `claude-api` resolves to no `SKILL.md` anywhere under `~/.claude`, and live markers already hold `{"claude-api": ""}` and `{"debugger": ""}`. This blocks defect 1 outright, because a repo naming any bundled or plugin skill would brick editing.

3. **Authoring tools outside the matcher.** `^(Edit|Write|NotebookEdit|Bash|Skill)$` omits `Workflow` (persists a script), durable `CronCreate` (writes `scheduled_tasks.json`), `Artifact`, and every `mcp__*` writer.

4. **`command: "python3"` can fail open.** A bare name resolves through PATH, here to `/opt/homebrew/bin/python3`. `hooks/AGENTS.md` already records the consequence for a different cause: a handler that fails to spawn exits 127, and "Upstream treats every exit code other than 0 and 2 as non-blocking, so that tool call proceeds ungated rather than being denied."

5. **`rules/skills.md` misstates the `continueOnBlock` default.** It reads "that became the default in v2.1.210." Upstream says the opposite: "by default the turn ends... Before v2.1.210, the deny `reason` was returned to Claude as the tool error and the turn continued." Turn-ending became the default, so `continueOnBlock: true` must be set explicitly.

## Changes

### 1. Add owner-skill routing to `skill-gate.py`

Give the handler a second axis keyed on the target's suffix or basename.

The declaration is a contract between a repository and the gate, so it belongs in the repository. Read it from the nearest ancestor of the payload `cwd` that contains it:

```json
{ ".py": ["python"], "pyproject.toml": ["python", "ruff"] }
```

Keys are a suffix (leading dot) or an exact basename. Deliberately not globs: `fnmatch` and gitignore syntax disagree, and the system interpreter cannot assume `markdown-it-py` or any other parser, per `rules/parsers.md`. Verified on `/usr/bin/python3`: `fnmatch('b.py', '**/*.py')` is `False`, so a glob map would silently miss every root-level file.

The requirement set becomes `writing-guidelines` + owners(path), plus `architecture` when the path is new. A payload with no path (Bash, MCP, `Workflow`) keeps `writing-guidelines` only, unchanged, because no path means no answerable ownership question.

The global default stays minimal: `{".py": ["python"]}`. `ruff`, `ultracite`, `turborepo`, and `zod` are conditional on install state or on a construct a path cannot reveal — `ultracite` self-declares "Use only when Ultracite is installed or explicitly requested" — so demanding them globally would require inapplicable skills. Repositories declare those.

### 2. Fix the unrecoverable deny

Rewrite the predicate so an unobtainable digest does not mean permanently missing:

- not recorded → missing
- recorded, digest obtainable → compare, so a skill edited since loading still counts as unloaded
- recorded, digest unobtainable → satisfied

When a declared owner resolves to no installed skill, deny with a distinct message naming the declaration file rather than "call the Skill tool". The Skill tool refuses an unknown name, so the current message would relocate the loop rather than end it.

### 3. Expand the matcher

```
^(Edit|Write|NotebookEdit|Bash|Skill|Workflow|CronCreate|Artifact)$|^mcp__
```

`^mcp__` rather than a write-verb list. A verb list under-matches, drifts as servers change, and `rules/parsers.md` says to prefer a design that removes the need for the list. Over-matching is nearly free here: the marker is per session, so a read-only MCP call costs at most one extra deny at session start, the same tradeoff already accepted for read-only Bash. These tools carry no `file_path`, so they fall to the `writing-guidelines`-only branch with no handler change.

### 4. Add the content-inspecting prompt hook

A second `PreToolUse` entry, separate from the gate:

```json
{
  "matcher": "^(Edit|Write)$",
  "hooks": [{
    "type": "prompt",
    "if": "Edit(//**/*.md)",
    "continueOnBlock": true,
    "timeout": 30,
    "prompt": "..."
  }]
}
```

Three details are load-bearing, and each was checked:

- `if: "Edit(...)"`, never `Write(...)`. Upstream: a path rule for `Write` or `NotebookEdit` is accepted but never consulted, warns at startup, and "Edit rules cover all file-editing tools."
- `//**/*.md` anchors at the filesystem root. A relative pattern in user settings anchors under `~/.claude`, not inside each project.
- `continueOnBlock` sits on the handler, not in the hook's output. The settings schema places it at `$defs/hookCommand/anyOf/1/properties/continueOnBlock`, `"default": false`, described as "When the prompt returns ok: false, feed the reason back to Claude and continue the turn." Without it, a false deny ends the turn.

The handler returns `{"ok": true}` or `{"ok": false, "reason": "..."}`. `if` gates process spawn, so a non-markdown edit costs no model call.

### 5. Pin the interpreter

`command` becomes `/usr/bin/python3`. Verified present, and it runs the handler correctly at 3.9.6. This constrains the handler to 3.9 syntax, which it already satisfies — hence suffix matching over `PurePath.full_match`, which needs 3.13.

### 6. Correct the documentation

- `rules/skills.md`: fix the `continueOnBlock` sentence per defect 5, and replace "names only `writing-guidelines` and `architecture`" with the owner-routing rule.
- `hooks/AGENTS.md`: record the new axis, the declaration format, and that `Edit(path)` is the only consulted file rule.
- `ast/AGENTS.md:26`: leave the prose, and add `ast/.claude/skill-owners.json` with `{".py": ["python"]}` so the sentence is enforced rather than remembered.

Also flag, without changing it: the `Stop` hook uses `type: "agent"`, which upstream marks experimental and says to avoid for production in favour of command hooks — the same guidance `rules/skills.md` already states.

## Files

| Path | Change |
| :--- | :--- |
| `~/.claude/hooks/PreToolUse/skill-gate.py` | Owner routing, digest-predicate fix, distinct unresolved-owner deny |
| `~/.claude/hooks/PreToolUse/test-skill-gate.py` | New: asserts every branch |
| `~/.claude/settings.json` | Matcher, `/usr/bin/python3`, prompt hook entry |
| `~/.claude/rules/skills.md` | `continueOnBlock` correction, owner routing |
| `~/.claude/hooks/AGENTS.md` | New axis, declaration format, `Edit(path)` rule |
| `ast/.claude/skill-owners.json` | New: `{".py": ["python"]}` |

Rewrite `skill-gate.py` with a single `Write`. `hooks/AGENTS.md` requires it: a sequence of edits leaves the handler broken between calls, and the broken call is the one that blocks its own repair.

## Verification

A gate that has silently stopped denying looks exactly like one that works, so check every branch rather than the one just changed.

1. **Both branches, per payload class.** Pipe synthetic JSON to the handler and assert the decision for: an existing `.py` path (expect `python` and `writing-guidelines`), an existing `.md` path (expect `writing-guidelines`), a new `.py` path (expect all three), Bash, an `mcp__*` writer, `Workflow`, malformed stdin, and a repository declaring an uninstalled owner (expect the distinct message, not the Skill-tool advice).
2. **Prove the fix was needed.** Revert each of defects 2 and 4 in a scratch copy and confirm the matching test fails. A test that passes either way proves nothing.
3. **Recovery path.** Confirm that recording a bundled skill such as `claude-api` now satisfies the requirement, where today it loops.
4. **Registration.** Run `/hooks` and confirm both entries appear under `PreToolUse` with the expected source file, then confirm at startup that no "is not matched by file permission checks" warning appears — that warning is how a wrong `if` pattern announces itself.
5. **Live.** Edit a `.py` file in `ast` and confirm the gate asks for `python`; edit a `.md` file and confirm the prompt hook runs, and that a non-markdown edit does not spawn it.

Run the suite under `/usr/bin/python3`, since that is what will execute it.
