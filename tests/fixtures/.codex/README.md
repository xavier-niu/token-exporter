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
line by line (one JSON object per line). Everything else is conversation
content you should drop:

| Event (`type` / `payload.type`)        | Fields used                                                                                  | Purpose                          |
| -------------------------------------- | -------------------------------------------------------------------------------------------- | -------------------------------- |
| `session_meta`                         | `payload.cwd`                                                                                | initial project (basename of cwd)|
| `turn_context`                         | `payload.cwd`, `payload.model`                                                               | project override + model         |
| `event_msg` with `payload.type == "token_count"` | `payload.info.total_token_usage.{input_tokens, cached_input_tokens, output_tokens, reasoning_output_tokens}` | the usage numbers |
| *(any line)*                           | top-level `timestamp`                                                                        | date bucket (in the run timezone)|

Drop entirely: `response_item`, `event_msg` of type `user_message` /
`agent_message` / `web_search_*` / `task_*`, and all of `session_meta`'s other
payload keys (`git`, `base_instructions`, `id`, …). Keep one or two redacted
non-token events if you want the fixture to also prove the parser ignores them.

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

Run against a real `~/.codex` session to emit a stripped copy, then hand-edit
`cwd`/ids to placeholders and (optionally) round the token numbers so expected
aggregates are trivial to compute:

```python
import json, sys

KEEP_EVENT = {"token_count"}          # event_msg payload types to keep
KEEP_TYPES = {"session_meta", "turn_context", "event_msg"}

def strip(src, dst):
    out = []
    for line in open(src):
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = o.get("type")
        if t not in KEEP_TYPES:
            continue
        p = o.get("payload") or {}
        if t == "session_meta":
            o["payload"] = {"id": "demo-XXXX", "cwd": "/home/user/work/PLACEHOLDER"}
        elif t == "turn_context":
            o["payload"] = {"cwd": "/home/user/work/PLACEHOLDER", "model": p.get("model", "gpt-5")}
        elif t == "event_msg":
            if p.get("type") not in KEEP_EVENT:
                continue
            # keep only the usage block the parser reads
            usage = (p.get("info") or {}).get("total_token_usage", {})
            o["payload"] = {"type": "token_count", "info": {"total_token_usage": usage}}
        out.append(json.dumps(o))
    open(dst, "w").write("\n".join(out) + "\n")

strip(sys.argv[1], sys.argv[2])
```

Then: replace `PLACEHOLDER` with a fake project name, scrub any remaining real
paths/ids, and sanity-check that no private text survived.

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

Claude Code is committed (see below). antcc/antcodex are absent on the machine
these fixtures were captured from, so they are **not committed** — build them the
same way, **only from real data**, on an env that has them. Don't fabricate.

### antcodex (`~/.codefuse/engine/codex`)

Shares `_scan_antcodex_dir` with codex, so sections 1–6 above apply verbatim.
The only difference is the agent label: `_detect_agent` returns `antcodex` when
the path contains `/codex` but does **not** end in `/.codex`. So put the fixture
under a dir like `.codefuse/engine/codex/sessions/...` and point the collector
at `.codefuse/engine/codex`.

### antcc (`~/.codefuse/engine/cc`)

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
antcc leaves `requestId` empty, so ccusage itself double-counts —
https://github.com/ryoppippi/ccusage/issues/976.) Note antcc cost is hardcoded
to `0` by the parser, so don't assert a non-zero cost here.

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
