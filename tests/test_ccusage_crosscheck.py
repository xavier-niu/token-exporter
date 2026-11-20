"""Cross-check the Claude Code path against ccusage CLI output."""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

import watcher
from watcher import CcusageCollector


def _ccusage_available():
    try:
        result = subprocess.run(["ccusage", "--version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Gated only on ccusage (no live data dependency): used by the committed
# Claude-Code fixture below.
skip_unless_ccusage = pytest.mark.skipif(
    not _ccusage_available(),
    reason="ccusage CLI not installed (the Claude Code path shells out to it)",
)

# Committed, sanitized ~/.claude fixture (see tests/fixtures/.codex/README.md §7).
CLAUDE_FIXTURE_DIR = Path(__file__).parent / "fixtures" / ".claude"
# The fixture carries fixed dates; scan with an effectively unbounded window so
# it never ages out of days_back. Ground truth = the token counts authored into
# the fixture, which `ccusage claude daily --instances` reproduces. Keyed by
# (project, model, date).
UNBOUNDED_DAYS_BACK = 100_000
CLAUDE_FIXTURE_GROUND_TRUTH = {
    ("demo-claude", "claude-opus-4-8", "2026-05-25"):
        {"input_tokens": 1000, "output_tokens": 200, "cache_creation_tokens": 50, "cache_read_tokens": 500},
    ("demo-claude", "claude-sonnet-4-6", "2026-05-25"):
        {"input_tokens": 800, "output_tokens": 120, "cache_creation_tokens": 0, "cache_read_tokens": 300},
    ("demo-tool", "claude-opus-4-8", "2026-05-26"):
        {"input_tokens": 1500, "output_tokens": 300, "cache_creation_tokens": 0, "cache_read_tokens": 700},
}


@skip_unless_ccusage
class TestClaudeFixtureGroundTruth:
    """Drive the Claude Code (ccusage CLI) path against a committed, sanitized
    ~/.claude fixture and assert the collector reproduces known token totals.

    This is the ground-truth guard for the cc path: unlike the codex and agy
    parsers it shells out to `ccusage claude daily --instances`, so a fixture
    test here exercises that invocation end-to-end. It catches ccusage CLI
    changes (e.g. the v20 move of --instances onto the `claude` subcommand) as
    well as regressions in agent detection, project extraction, and
    delta/record emission. Skipped where the ccusage binary isn't installed.
    """

    def _records(self, pricing_rates=None):
        records = []
        collector = CcusageCollector(
            claude_dirs=[str(CLAUDE_FIXTURE_DIR)],
            days_back=UNBOUNDED_DAYS_BACK,
            on_record=lambda agent, rec: records.append((agent, rec)),
            pricing_rates=pricing_rates,
        )
        collector.scan_history()
        return records

    def test_token_totals_match_ground_truth(self):
        records = self._records()
        assert records, "ccusage emitted no records for the fixture"

        got = {}
        for agent, rec in records:
            assert agent == "cc"
            key = (rec["project"], rec["model"], rec["timestamp"].date().isoformat())
            got[key] = {
                f: rec[f]
                for f in ("input_tokens", "output_tokens", "cache_creation_tokens", "cache_read_tokens")
            }
        assert got == CLAUDE_FIXTURE_GROUND_TRUTH

    def test_opus_4_8_priced_from_token_deltas(self):
        # Cost is computed in-process from token deltas via pricing.cost_for(),
        # NOT taken from ccusage's offline snapshot. Asserting the exact
        # token-derived value (not just > 0) is what catches a wrong rate card —
        # the 3x legacy-Opus overcharge that started this investigation would
        # fail here. Official Opus 4.8 rates, per 1M tokens.
        rates = {
            "claude-opus-4-8": {
                "input": 5.0,
                "output": 25.0,
                "cached": 0.5,
                "cache_creation": 6.25,
            }
        }
        records = self._records(pricing_rates=rates)
        opus = [rec for _, rec in records if rec["model"] == "claude-opus-4-8"]
        assert opus, "no claude-opus-4-8 records emitted"
        for rec in opus:
            expected = (
                rec["input_tokens"] * 5.0
                + rec["cache_read_tokens"] * 0.5
                + rec["output_tokens"] * 25.0
                + rec["cache_creation_tokens"] * 6.25
            ) / 1_000_000
            assert rec["cost_usd"] == pytest.approx(expected)
            assert rec["cost_usd"] > 0


def _ccusage_log_line(message_id: str, tokens: int, session: str = "s") -> str:
    # Compact like real Claude Code logs: ccusage skips lines with spaced JSON.
    return json.dumps({
        "type": "assistant",
        "timestamp": "2026-05-25T12:00:00.000Z",
        "sessionId": session,
        "requestId": f"req-{message_id}",
        "cwd": "/work/demo",
        "version": "1.0.0",
        "message": {
            "id": message_id,
            "model": "claude-sonnet-4-6",
            "role": "assistant",
            "content": [],
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        },
    }, separators=(",", ":")) + "\n"


def _write_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _input_tokens(report: dict) -> int:
    return sum(
        entry["inputTokens"]
        for entries in report.get("projects", {}).values()
        for entry in entries
    )


@skip_unless_ccusage
class TestCcusageContract:
    """What the exporter assumes about the ccusage CLI; rerun these on every bump."""

    SINCE = datetime(2020, 1, 1).date()

    def test_reads_nothing_outside_the_root_signature(self, tmp_path):
        # The scan cache trusts _tree_signature to cover all of ccusage's inputs,
        # so a file it reads beyond that could change results without a rescan.
        root = tmp_path / ".claude"
        for tokens, name in {
            1: "projects/-a/s.jsonl",
            2: "projects/-a/abc/subagents/agent.jsonl",
            4: "s.jsonl",
            8: "other/s.jsonl",
            16: "transcripts/s.jsonl",
            32: "projects/-a/s.json",
            64: "projects/-a/s.JSONL",
            128: "projects/-a/s.jsonl.bak",
        }.items():
            _write_log(root / name, _ccusage_log_line(f"m{tokens}", tokens))

        report = watcher._run_ccusage(str(root), self.SINCE, "UTC")

        assert _input_tokens(report) == 1 + 2

    def test_a_run_does_not_change_the_root_signature(self, tmp_path):
        root = tmp_path / ".claude"
        _write_log(root / "projects/-a/s.jsonl", _ccusage_log_line("m", 1))
        before = watcher._tree_signature(str(root / "projects"))

        assert watcher._run_ccusage(str(root), self.SINCE, "UTC") is not None

        assert watcher._tree_signature(str(root / "projects")) == before

    def test_root_without_usage_is_not_a_failed_scan(self, tmp_path):
        root = tmp_path / ".claude"
        (root / "projects/-a").mkdir(parents=True)

        report = watcher._run_ccusage(str(root), self.SINCE, "UTC")

        assert report is not None
        assert _input_tokens(report) == 0

    def test_request_repeated_across_sessions_is_counted_once(self, tmp_path):
        # Claude Code copies earlier requests into a resumed session's log.
        root = tmp_path / ".claude"
        _write_log(root / "projects/-a/one.jsonl", _ccusage_log_line("m", 100, session="one"))
        _write_log(root / "projects/-a/two.jsonl", _ccusage_log_line("m", 100, session="two"))

        report = watcher._run_ccusage(str(root), self.SINCE, "UTC")

        assert _input_tokens(report) == 100
