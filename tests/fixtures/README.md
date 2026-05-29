# Test fixtures

Committed sample data for the on-disk layouts the collectors parse, so unit
tests run against realistic input instead of inline dicts.

**Building or refreshing a fixture?** See [`.codex/README.md`](.codex/README.md)
for the step-by-step construction recipe (sanitization snippet, the exact fields
each parser reads, how to compute expected aggregates, and per-agent layout for
codex / antcodex / antcc / Claude Code).

## `.codex/`

A sanitized, **partial** slice of a real `~/.codex` directory (Codex CLI usage
history). The layout mirrors production — `sessions/YYYY/MM/DD/rollout-*.jsonl`
— but:

- All conversation content is removed. Only the events `_scan_antcodex_dir`
  reads are kept: `session_meta` (for `cwd`), `turn_context` (for `cwd` +
  `model`), and `event_msg` `token_count` events (for `total_token_usage`).
- A couple of placeholder `user_message` / `agent_message` / `response_item`
  events are left in (with `<redacted>` content) to verify the parser ignores
  everything that isn't a `token_count`.
- `cwd`, session ids, and versions are placeholders; token counts are
  illustrative round numbers chosen so expected aggregates are easy to verify
  by hand in `TestCodexFixture`.

`total_token_usage` is a per-session cumulative counter; the parser diffs
successive events. `demo0002` deliberately drops on its second event to
exercise the counter-reset branch (a fresh session block within one file).

The fixture dates are fixed, so the test scans with a very large `days_back`
(an effectively unbounded window) rather than dates relative to "now". This is
intentional: it isolates *parsing/dedup* correctness from the date-window
filtering, and keeps the test from rotting as the calendar advances.

## `.claude/`

A sanitized, **partial** Claude Code history: `projects/<key>/*.jsonl` with
hand-authored `assistant` records modeled on the real `~/.claude` envelope.
Conversation `content` is redacted; only the fields ccusage reads are real
(`message.{id,model,usage}`, `requestId`, `timestamp`).

Unlike the others, the Claude Code path parses through the external **`ccusage`
CLI** (`ccusage claude daily --instances`), so this can't be a hermetic unit
test. It backs the **ground-truth** integration test `TestClaudeFixtureGroundTruth`
in `test_ccusage_crosscheck.py`, which drives `CcusageCollector` over the fixture
and asserts the known per-(project, model, date) token totals. Because it runs
the real CLI, it also guards against ccusage CLI changes (it would have caught
the v20 move of `--instances` onto the `claude` subcommand) and against
opus-4-8 regressing to $0 cost. It is **gated on the ccusage binary** and skips
where ccusage isn't installed. The fixture includes a duplicate `message.id` +
`requestId` line to exercise ccusage's dedup, and uses fixed dates scanned with
an unbounded `days_back` (same rationale as `.codex/`).

## Not included

- **antcc** (`~/.codefuse/engine/cc`) and **antcodex**
  (`~/.codefuse/engine/codex`): not installed on the machine these fixtures
  were captured from, so there is no real data to sanitize. Not fabricated.
  (antcodex shares `_scan_antcodex_dir` with codex, so that parser path is
  already exercised by `.codex/`.)
