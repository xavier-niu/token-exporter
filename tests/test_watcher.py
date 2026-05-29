import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from watcher import CcusageCollector, CodexCollector, _extract_project

# Committed, sanitized sample of a real ~/.codex directory (see fixtures/README.md).
FIXTURES_DIR = Path(__file__).parent / "fixtures"
CODEX_FIXTURE_DIR = FIXTURES_DIR / ".codex"
# Fixtures carry fixed dates; scan with an effectively unbounded window so the
# test exercises parsing/dedup, not date-window filtering, and never ages out.
UNBOUNDED_DAYS_BACK = 100_000


# Fixtures must use dates relative to "now": the collectors prune/skip anything
# older than days_back (default 7), so hardcoded calendar dates make the tests
# start failing once they age past that window. _DAY1/_DAY2 are two recent,
# distinct days that always sit comfortably inside a 7-day window.
_TODAY = datetime.now(timezone.utc).date()
_DAY1 = _TODAY - timedelta(days=2)
_DAY2 = _TODAY - timedelta(days=1)
_DAY1_STR = _DAY1.isoformat()
_DAY2_STR = _DAY2.isoformat()


def _codex_ts(day: date, second: int) -> str:
    """An ISO-8601 UTC timestamp at 10:00 on `day` (well clear of midnight so it
    stays on the same calendar day under any test timezone)."""
    return f"{day.isoformat()}T10:00:{second:02d}Z"


SAMPLE_CCUSAGE_INSTANCES = {
    "projects": {
        "-home-nxw-developer-token-exporter": [
            {
                "date": _DAY1_STR,
                "inputTokens": 1000,
                "outputTokens": 200,
                "cacheCreationTokens": 50,
                "cacheReadTokens": 500,
                "totalTokens": 1750,
                "totalCost": 0.01,
                "modelsUsed": ["GLM-5.1"],
                "modelBreakdowns": [
                    {
                        "modelName": "GLM-5.1",
                        "inputTokens": 1000,
                        "outputTokens": 200,
                        "cacheCreationTokens": 50,
                        "cacheReadTokens": 500,
                        "cost": 0.01,
                    }
                ],
            }
        ],
        "-home-nxw-developer-devkit-ant": [
            {
                "date": _DAY1_STR,
                "inputTokens": 500,
                "outputTokens": 100,
                "cacheCreationTokens": 0,
                "cacheReadTokens": 300,
                "totalTokens": 900,
                "totalCost": 0,
                "modelsUsed": ["GLM-5.1"],
                "modelBreakdowns": [
                    {
                        "modelName": "GLM-5.1",
                        "inputTokens": 500,
                        "outputTokens": 100,
                        "cacheCreationTokens": 0,
                        "cacheReadTokens": 300,
                        "cost": 0,
                    }
                ],
            }
        ],
    }
}

class TestExtractProject(unittest.TestCase):
    def test_standard_path(self):
        assert _extract_project("-home-nxw-developer-token-exporter") == "token-exporter"

    def test_nested_path(self):
        assert _extract_project("-home-nxw-developer-devkit-ant") == "devkit-ant"

    def test_short_path(self):
        assert _extract_project("-home-user") == "home-user"

    def test_strips_leading_dashes(self):
        assert _extract_project("--home-nxw-developer-my-project") == "my-project"


class TestCcusageCollector(unittest.TestCase):
    @patch("watcher.os.path.isdir", return_value=True)
    @patch("watcher._run_ccusage")
    def test_scan_history_emits_records(self, mock_run, mock_isdir):
        mock_run.return_value = SAMPLE_CCUSAGE_INSTANCES
        records = []
        collector = CcusageCollector(
            claude_dirs=["/fake/dir"],
            days_back=7,
            on_record=lambda agent, rec: records.append((agent, rec)),
        )
        collector.scan_history()

        assert len(records) == 2
        agent0, rec0 = records[0]
        assert agent0 == "unknown"
        assert rec0["model"] == "GLM-5.1"
        assert rec0["input_tokens"] == 1000
        assert rec0["output_tokens"] == 200
        assert rec0["cache_creation_tokens"] == 50
        assert rec0["cache_read_tokens"] == 500
        assert rec0["cost_usd"] == 0.01

    @patch("watcher.os.path.isdir", return_value=True)
    @patch("watcher._run_ccusage")
    def test_delta_tracking_no_double_count(self, mock_run, mock_isdir):
        mock_run.return_value = SAMPLE_CCUSAGE_INSTANCES
        records = []
        collector = CcusageCollector(
            claude_dirs=["/fake/dir"],
            days_back=7,
            on_record=lambda agent, rec: records.append((agent, rec)),
        )
        # First scan: all data is new
        collector.scan_history()
        assert len(records) == 2

        # Second scan: same data, no deltas → no new records
        collector.scan_history()
        assert len(records) == 2

    @patch("watcher.os.path.isdir", return_value=True)
    @patch("watcher._run_ccusage")
    def test_delta_tracking_incremental(self, mock_run, mock_isdir):
        records = []
        collector = CcusageCollector(
            claude_dirs=["/fake/dir"],
            days_back=7,
            on_record=lambda agent, rec: records.append((agent, rec)),
        )

        # First scan
        mock_run.return_value = SAMPLE_CCUSAGE_INSTANCES
        collector.scan_history()
        assert len(records) == 2
        assert records[0][1]["input_tokens"] == 1000

        # Second scan with increased tokens
        updated = json.loads(json.dumps(SAMPLE_CCUSAGE_INSTANCES))
        updated["projects"]["-home-nxw-developer-token-exporter"][0]["inputTokens"] = 1500
        updated["projects"]["-home-nxw-developer-token-exporter"][0]["modelBreakdowns"][0]["inputTokens"] = 1500
        mock_run.return_value = updated
        collector.scan_history()
        assert len(records) == 3
        assert records[2][1]["input_tokens"] == 500  # 1500 - 1000 delta

    @patch("watcher.os.path.isdir", return_value=True)
    @patch("watcher._run_ccusage")
    def test_new_day_produces_new_records(self, mock_run, mock_isdir):
        records = []
        collector = CcusageCollector(
            claude_dirs=["/fake/dir"],
            days_back=7,
            on_record=lambda agent, rec: records.append((agent, rec)),
        )

        mock_run.return_value = SAMPLE_CCUSAGE_INSTANCES
        collector.scan_history()
        assert len(records) == 2

        # Add a new day's data
        updated = json.loads(json.dumps(SAMPLE_CCUSAGE_INSTANCES))
        updated["projects"]["-home-nxw-developer-token-exporter"].append({
            "date": _DAY2_STR,
            "inputTokens": 300,
            "outputTokens": 50,
            "cacheCreationTokens": 0,
            "cacheReadTokens": 100,
            "totalTokens": 450,
            "totalCost": 0,
            "modelsUsed": ["GLM-5.1"],
            "modelBreakdowns": [
                {
                    "modelName": "GLM-5.1",
                    "inputTokens": 300,
                    "outputTokens": 50,
                    "cacheCreationTokens": 0,
                    "cacheReadTokens": 100,
                    "cost": 0,
                }
            ],
        })
        mock_run.return_value = updated
        collector.scan_history()
        assert len(records) == 3
        assert records[2][1]["input_tokens"] == 300

    @patch("watcher._run_ccusage")
    def test_skips_empty_dirs(self, mock_run):
        records = []
        collector = CcusageCollector(
            claude_dirs=["/nonexistent"],
            days_back=7,
            on_record=lambda agent, rec: records.append((agent, rec)),
        )
        collector.scan_history()
        assert len(records) == 0
        mock_run.assert_not_called()

    @patch("watcher._run_ccusage")
    def test_ccusage_failure_is_handled(self, mock_run):
        mock_run.return_value = None
        records = []
        collector = CcusageCollector(
            claude_dirs=["/fake/dir"],
            days_back=7,
            on_record=lambda agent, rec: records.append((agent, rec)),
        )
        collector.scan_history()
        assert len(records) == 0


class TestCodexCollector(unittest.TestCase):
    def test_antcodex_parser_emits_project_records_and_skips_duplicate_token_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            codex_dir = Path(tmp) / "codex"
            session_dir = codex_dir / "sessions" / _DAY1.strftime("%Y/%m/%d")
            session_dir.mkdir(parents=True)
            session_file = session_dir / "rollout-test.jsonl"
            rows = [
                {
                    "timestamp": _codex_ts(_DAY1, 0),
                    "type": "session_meta",
                    "payload": {"cwd": "/home/user/developer/token-exporter"},
                },
                {
                    "timestamp": _codex_ts(_DAY1, 1),
                    "type": "turn_context",
                    "payload": {"cwd": "/home/user/developer/token-exporter", "model": "gpt-5.5"},
                },
                {
                    "timestamp": _codex_ts(_DAY1, 2),
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {
                                "input_tokens": 1000,
                                "cached_input_tokens": 400,
                                "output_tokens": 50,
                            }
                        },
                    },
                },
                {
                    "timestamp": _codex_ts(_DAY1, 3),
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {
                                "input_tokens": 1000,
                                "cached_input_tokens": 400,
                                "output_tokens": 50,
                            }
                        },
                    },
                },
                {
                    "timestamp": _codex_ts(_DAY1, 4),
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {
                                "input_tokens": 1500,
                                "cached_input_tokens": 700,
                                "output_tokens": 80,
                            }
                        },
                    },
                },
            ]
            session_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            records = []
            collector = CodexCollector(
                codex_dirs=[str(codex_dir)],
                days_back=7,
                on_record=lambda agent, rec: records.append((agent, rec)),
                timezone_name="Asia/Shanghai",
            )
            collector.scan_history()

        assert len(records) == 1
        agent, rec = records[0]
        assert agent == "antcodex"
        assert rec["project"] == "token-exporter"
        assert rec["model"] == "gpt-5.5"
        assert rec["input_tokens"] == 800
        assert rec["cache_read_tokens"] == 700
        assert rec["output_tokens"] == 80
        assert rec["cost_usd"] == 0

    def test_antcodex_parser_applies_pricing(self):
        with tempfile.TemporaryDirectory() as tmp:
            codex_dir = Path(tmp) / "codex"
            session_dir = codex_dir / "sessions" / _DAY1.strftime("%Y/%m/%d")
            session_dir.mkdir(parents=True)
            session_file = session_dir / "rollout.jsonl"
            rows = [
                {"timestamp": _codex_ts(_DAY1, 0), "type": "session_meta",
                 "payload": {"cwd": "/home/user/developer/proj"}},
                {"timestamp": _codex_ts(_DAY1, 1), "type": "turn_context",
                 "payload": {"cwd": "/home/user/developer/proj", "model": "gpt-5"}},
                {"timestamp": _codex_ts(_DAY1, 2), "type": "event_msg",
                 "payload": {"type": "token_count", "info": {"total_token_usage": {
                     "input_tokens": 1_000_000, "cached_input_tokens": 0, "output_tokens": 100_000}}}},
            ]
            session_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

            records = []
            rates = {"gpt-5": {"input": 1.25, "cached": 0.125, "output": 10.0, "cache_creation": 1.25}}
            collector = CodexCollector(
                codex_dirs=[str(codex_dir)],
                days_back=7,
                on_record=lambda agent, rec: records.append((agent, rec)),
                pricing_rates=rates,
            )
            collector.scan_history()

        assert len(records) == 1
        _, rec = records[0]
        # 1M input @ $1.25/MTok + 100K output @ $10/MTok = 1.25 + 1.0 = 2.25
        assert rec["cost_usd"] == pytest.approx(2.25)


class TestCodexFixture(unittest.TestCase):
    """Parse the committed, sanitized real-world Codex fixture end-to-end."""

    def _collect(self, pricing_rates=None):
        records = []
        collector = CodexCollector(
            codex_dirs=[str(CODEX_FIXTURE_DIR)],
            days_back=UNBOUNDED_DAYS_BACK,
            on_record=lambda agent, rec: records.append((agent, rec)),
            pricing_rates=pricing_rates,
        )
        collector.scan_history()
        return records

    def test_aggregates_tokens_and_ignores_conversation_events(self):
        records = self._collect()

        # One record per (project, model, date); conversation events ignored.
        assert len(records) == 2
        by_project = {rec["project"]: (agent, rec) for agent, rec in records}
        assert set(by_project) == {"demo-api", "demo-web"}

        agent, api = by_project["demo-api"]
        assert agent == "codex"
        assert api["model"] == "gpt-5.5"
        # cumulative 1000/1500/1800 input, 0/200/200 cached, 100/180/250 output
        assert api["input_tokens"] == 1600        # (1000) + (500-200) + (300-0)
        assert api["cache_read_tokens"] == 200     # 0 + 200 + 0
        assert api["output_tokens"] == 250         # 100 + 80 + 70
        assert api["timestamp"].date().isoformat() == "2026-05-25"

        _, web = by_project["demo-web"]
        assert web["model"] == "gpt-5"
        # second event resets the counter (input drops 2000 -> 800)
        assert web["input_tokens"] == 2200         # (2000-500) + (800-100)
        assert web["cache_read_tokens"] == 600     # 500 + 100
        assert web["output_tokens"] == 420         # 300 + 120

    def test_applies_pricing_to_fixture(self):
        rates = {
            "gpt-5.5": {"input": 3.0, "cached": 0.3, "output": 15.0, "cache_creation": 3.0},
            "gpt-5": {"input": 1.25, "cached": 0.125, "output": 10.0, "cache_creation": 1.25},
        }
        records = self._collect(pricing_rates=rates)
        cost = {rec["project"]: rec["cost_usd"] for _, rec in records}
        # demo-api: (1600*3.0 + 200*0.3 + 250*15.0) / 1e6
        assert cost["demo-api"] == pytest.approx((1600 * 3.0 + 200 * 0.3 + 250 * 15.0) / 1e6)
        # demo-web: (2200*1.25 + 600*0.125 + 420*10.0) / 1e6
        assert cost["demo-web"] == pytest.approx((2200 * 1.25 + 600 * 0.125 + 420 * 10.0) / 1e6)


if __name__ == "__main__":
    unittest.main()
