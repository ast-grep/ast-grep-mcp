# What remains after the skill-gate and Stop-judge work

## Context

The original G1–G6 plan (close every skill-gate gap) is complete: tasks #1, #3–#8 are resolved with named evidence. Task #2 (the one-week acceptance watch) is the sole tracked open item, calendar-bound to close on or after 2026-08-12.

Since that work landed, `~/.claude/settings.json` was rewritten to point at a different provider and model (z.ai, `glm-5.2[1m]`, with `ANTHROPIC_AUTH_TOKEN`). The Stop judge hardcodes `--model haiku`, which now resolves to `glm-5.2[1m]` via the inherited `ANTHROPIC_DEFAULT_HAIKU_MODEL` env var. Every measurement behind the judge (injection resistance, toollessness, clean-JSON, cost/latency) was taken against Anthropic haiku, so the switch could have invalidated them.

This assessment answers "what else still needs to be done?" and runs `/simplify` and `/architecture` against the current state.

## Provider-switch verification (done this session)

All three security properties of the Stop judge were re-probed under the new provider/model and **hold**:

- **Injection resistance holds.** The C-prime fixture (fabricated test-pass claim plus an appended "you must emit ok:true" instruction) still blocks; the model explicitly ignored the injection.
- **Toollessness holds (behavioral).** `--tools ""` is a harness-level cap, provider-independent. A behavioral Read probe (real file with a nonce) came back pantomimed, not read. The toolkit self-report listing `Bash, Edit, …` was the same hallucination caught in an earlier pass.
- **MCP cap holds.** A commanded MCP call returned NO-MCP under `--strict-mcp-config`.

The guardrails (`--max-budget-usd 0.05`, `JUDGE_TIMEOUT_SECONDS=75`) are provider-agnostic and still bind. No defect from the switch.

## What genuinely remains

1. **Task #2 — the acceptance watch.** Calendar-bound, closes on/after 2026-08-12. No code action; the audit channel (markers + transcripts) needs no machinery. An incident, if captured, stops the watch with its evidence pair.

2. **Stale cost figures (record accuracy, not a defect).** The per-stop cost/latency figures recorded in task #7 ($0.007–0.046, 47–74s, num_turns=2) were measured against haiku/Anthropic. Under glm-5.2[1m]/z.ai the absolute numbers differ, but `hooks/AGENTS.md` states the constraint in terms of the guardrails (budget cap, timeout), not the absolute numbers, so no doc change is required. The one untested guardrail under z.ai is whether `--max-budget-usd` *enforces* (vs merely accepts) — it's a harness feature so it should, but it has not been adversarially tested under this provider.

## `/simplify` assessment — converged

Three `/simplify` passes ran against this change set (12 reviewer agents total). Each pass found and fixed real issues: the first falsified the inert `allowedTools` fix and the injectable fused prompt; the second rebuilt the judge toolless with extraction-in-handler; the third fixed the turn-boundary misanchor (19% of turns), added truncation provenance, persisted the test battery, priced the digest to its budget, and scrubbed the effort env. Every reviewer's code findings on the final artifacts came back "leave it" or were applied. A fourth pass would be marginal — the artifacts have converged. Not recommended unless a new artifact lands.

## `/architecture` assessment — clean, no moves required

Independent ownership/placement review (Explore agent, all 9 surfaces read in full). Verdict per dimension:

- **Single owner per fact (A):** clean. Platform facts are single-owned with proper cites — agent-hook toolkit at `hooks-reference.md:121,272`; flag semantics at `permissions-and-settings.md:49-52`; toollessness at `AGENTS.md:147-156`. One borderline overlap: the skill-gate's denial behavior appears in both `rules/skills.md` (user contract/procedure) and `hooks/AGENTS.md` (handler reasoning). This is the unavoidable user-contract/designer-rationale split; accepted as-is, flagged as a known drift surface for the next reviewer.
- **POLICY constant placement (B):** correct in the handler (`stop-judge.py:23-41`). It is structurally coupled to the spawn argv and the digest shape; relocating it to the optimizer skill would be a placement violation (repo-specific reasoning in a platform skill).
- **Shared handler infrastructure (C):** correctly declined. Note: the prior "opposite failure policies" rationale is a red herring — failure mode is set by each handler's own `main` wrapper, not any shared utility. The real reason is the absence of meaningful shared logic (one is a marker state machine, the other a transcript parser + subprocess judge). Correct the stale rationale opportunistically if that section is next touched; no code change.
- **Placement violations (D):** none. No platform semantics in repo briefs (they cite, not restate); no repo reasoning in platform skills (grep confirmed no handler names appear in them).
- **Dependency direction (E):** holds across all five tiers. Handlers import stdlib only and point downward to docs/skills; nothing points upward.

## Conclusion

Almost nothing remains. Task #2 (the acceptance watch) is the sole tracked open item and is calendar-bound. The provider switch was verified not to break the Stop judge. The artifacts converged across three `/simplify` passes. The architecture is clean. The two architecture notes (the accepted drift surface, the stale sharing rationale) are optional and opportunistic — neither blocks anything.

## Verification

If the user wants current cost figures under the new provider: re-run `Stop/test-stop-judge.py` (the four judge-spawning checks will run under glm-5.2[1m] and report timing) plus the existing `PreToolUse/test-skill-gate.py` (model-independent, should be unaffected). Otherwise the guardrail-level verification above is sufficient.
