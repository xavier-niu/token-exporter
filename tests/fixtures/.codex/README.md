# How to construct the `.codex` env fixture

This directory is a **sanitized, partial** stand-in for a real `~/.codex`
(Codex CLI history). It exists so `CodexCollector` / `_scan_antcodex_dir` can be
unit-tested against realistic on-disk input without shipping anyone's private
conversations. This file is the recipe for (re)building it from a real machine.

> A README here is safe: the collector only scans `sessions/**/*.jsonl`
> (`Path(codex_dir) / "sessions"` + `rglob`), so files outside `sessions/` are
> ignored.

---

## 1. What the parser actually consumes

`_scan_antcodex_dir` walks `sessions/**/*.jsonl` and reads **only** these,
line by line (one JSON object per line). Other metadata can stay in fixtures;
conversation-bearing fields should be redacted.

| Event (`type` / `payload.type`)        | Fields used                                                                                  | Purpose                          |
| -------------------------------------- | -------------------------------------------------------------------------------------------- | -------------------------------- |
| `session_meta`                         | `payload.cwd`                                                                                | initial project (basename of cwd)|
| `turn_context`                         | `payload.cwd`, `payload.model`                                                               | project override + model         |
| `event_msg` with `payload.type == "token_count"` | `payload.info.total_token_usage.{input_tokens, cached_input_tokens, output_tokens, reasoning_output_tokens}` | the usage numbers |
| *(any line)*                           | top-level `timestamp`                                                                        | date bucket (in the run timezone)|

Redact conversation-bearing fields such as `content`, user/agent `message`,
reasoning summaries, tool arguments/output, file snapshots, and last prompt
text. Keep metadata such as ids, cwd, versions, timestamps, git metadata, and
rate-limit envelopes as-is unless a specific field itself contains private
conversation content.

## 2. Key behaviors to preserve

- **`total_token_usage` is cumulative per session.** The parser diffs
  successive `token_count` events. So the numbers must be monotonically
  increasing within a session to represent normal usage.
- **Counter reset:** if any field's delta goes negative (new session block in
  the same file, context compaction), the parser treats the *current* values as
  the delta. Include a session whose second `token_count` drops below the first
  to exercise this (`demo0002` does).
- **Project naming:** project = `os.path.basename(cwd)`. `_scan_antcodex_dir`
  wraps it as `-home-user-developer-<project>`, and `_extract_project` then
  strips the first three dash-segments — so a `cwd` basename like `demo-api`
  round-trips to the project label `demo-api`.
- **Agent label comes from the directory name.** `_detect_agent` returns
  `codex` only when the path ends in `/.codex`; a dir merely containing
  `/codex` is labeled `antcodex`. Both use this same parser, so name the
  fixture dir `.codex` to test the `codex` label.

## 3. Layout

Mirror production: `sessions/YYYY/MM/DD/rollout-<ISO-ish-stamp>-<id>.jsonl`.
The directory date is cosmetic — the parser dates events from each line's
`timestamp`, not the path — but matching it keeps the fixture readable.

## 4. Sanitization recipe (from a real session)

Run against a real `~/.codex` session to emit a redacted copy. Preserve the JSON
objects as-is except for fields that carry conversation text or user file/tool
payloads:

```python
import json, sys

REDACTED = "<redacted: conversation content removed from fixture>"
REDACTED_OBJ = {"redacted": "conversation content removed from fixture"}

CONTENT_KEYS = {
    "arguments",
    "base_instructions",
    "content",
    "encrypted_content",
    "images",
    "lastPrompt",
    "last_agent_message",
    "local_images",
    "output",
    "snapshot",
    "summary",
    "text",
    "text_elements",
    "toolUseResult",
}

def redacted_like(value):
    if isinstance(value, list):
        return [REDACTED]
    if isinstance(value, dict):
        return REDACTED_OBJ
    if value is None:
        return None
    return REDACTED

def sanitize(value):
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if not isinstance(value, dict):
        return value

    out = {}
    for key, item in value.items():
        if key in CONTENT_KEYS:
            out[key] = redacted_like(item)
        elif key == "message" and not isinstance(item, dict):
            out[key] = redacted_like(item)
        else:
            out[key] = sanitize(item)
    return out

with open(sys.argv[2], "w") as dst:
    for line in open(sys.argv[1]):
        if line.strip():
            dst.write(json.dumps(sanitize(json.loads(line))) + "\n")
```

Then: sanity-check that no private conversation text survived. Keep other fields
as-is so the fixture continues to look like production data.

## 5. Compute expected aggregates by hand

For each session, per `token_count` event compute the delta vs. the previous
event (or the raw values on the first / after a reset), then:

```
input_tokens      += max(delta.input - max(delta.cached, 0), 0)
cache_read_tokens += max(delta.cached, 0)
output_tokens     += max(delta.output, 0)
```

Sum per `(project, model, date)`. Put those totals straight into the asserts in
`TestCodexFixture` (in `tests/test_watcher.py`). Worked example for the shipped
fixture is in the comments there.

## 6. Scan with an unbounded window

The fixtures carry **fixed** dates. `_scan_antcodex_dir` skips any event older
than `today - days_back`, so the test passes `days_back=UNBOUNDED_DAYS_BACK`
(`100_000`) to make the window effectively infinite. This isolates
parsing/dedup correctness from date filtering and stops the test from rotting as
the calendar moves — do **not** "fix" it by switching to recent dates.

## 7. Other agents

Claude Code, AntCodex, and AntCC fixtures are committed alongside `.codex`.
Build or refresh Ant fixtures the same way: **only from real data**, on an env
that has them. Don't fabricate.

### antcodex (`~/.codefuse/engine/codex`) — committed at `../.codefuse/engine/codex/`

Shares `_scan_antcodex_dir` with codex, so sections 1–6 above apply verbatim.
The only difference is the agent label: `_detect_agent` returns `antcodex` when
the path contains `/codex` but does **not** end in `/.codex`. So put the fixture
under a dir like `.codefuse/engine/codex/sessions/...` and point the collector
at `.codefuse/engine/codex`.

The committed fixture is a sanitized real session with the source metadata kept
as-is, empty token-count events, duplicate cumulative snapshots, and redacted
conversation payloads. Its expected totals live in `TestAntcodexFixture`.

### antcc (`~/.codefuse/engine/cc`) — committed at `../.codefuse/engine/cc/`

Different format and parser (`_scan_antcc_dir`) — it's the Claude-Code-style
conversation JSONL, deduped by message id. Layout:

```
.codefuse/engine/cc/projects/<project-key>/**/*.jsonl
```

`<project-key>` is the directory name, dash-encoded like ccusage's
`-home-nxw-developer-<project>`; `_extract_project` strips the first three
dash-segments to get the project label. The parser reads, per line:

| Field                                          | Purpose                                  |
| ---------------------------------------------- | ---------------------------------------- |
| `message.usage.input_tokens`                   | input                                    |
| `message.usage.output_tokens`                  | output                                   |
| `message.usage.cache_creation_input_tokens`    | cache creation                           |
| `message.usage.cache_read_input_tokens`        | cache read                               |
| `message.id`                                   | dedup key (see below)                    |
| `message.model`                                | model                                    |
| `timestamp`                                    | date bucket (in the run timezone)        |

Lines with no `message.usage` (or all-zero tokens) are skipped. Detection keys
off the directory **basename**: `cc` or `(.)codefuse` → `antcc`.

**Dedup to exercise:** records sharing a `message.id` are treated as streaming
updates to the *same* message — the parser subtracts the previous values and
adds the new ones, so the final value for that id wins (not the sum). Author at
least two lines with the same `message.id` and increasing usage, and assert the
total reflects only the last one. (This is the bug the parser exists to fix:
antcc leaves `requestId` absent or empty, so ccusage itself double-counts —
https://github.com/ryoppippi/ccusage/issues/976.) AntCC cost comes from the
shared pricing table when rates are available; unknown models still price as
`0`. GLM text models are fetched from Z.ai's official pricing page, with
bundled Z.ai rates only as a last-resort fallback when the page is unavailable.

The committed fixture is a sanitized real streaming assistant response with the
source metadata kept as-is. Three records share one `message.id` and omit
`requestId`. Its expected totals live in `TestAntccFixture`.

### Claude Code (`~/.claude`)  — committed at `../.claude/`

The collector parses this through the external `ccusage` CLI subprocess, so it
can't be a hermetic unit test no matter what fixture you build — but a committed
fixture still makes its integration test self-contained.

Layout: `projects/<project-key>/*.jsonl`, where `<project-key>` is the
dash-encoded cwd (`-home-user-developer-<project>`). Each line is an `assistant`
record; ccusage reads `timestamp`, `message.{id,model}`, `requestId` (dedup key
pair = `message.id` + `requestId`), and `message.usage.{input_tokens,
output_tokens, cache_creation_input_tokens, cache_read_input_tokens}`. Redact
`message.content`; keep the envelope. ccusage prices with its offline snapshot,
so assert token counts (stable) and only that cost is `> 0` (drifts by version).

This one is **committed** at `tests/fixtures/.claude/` and backs
`TestClaudeFixtureGroundTruth` in `test_ccusage_crosscheck.py`, gated on the
`ccusage` binary. To (re)build, hand-author lines or sanitize real `~/.claude`
records, then confirm the numbers with:

```
CLAUDE_CONFIG_DIR=tests/fixtures/.claude \
  ccusage claude daily --json --offline --instances --since 20260101
```
