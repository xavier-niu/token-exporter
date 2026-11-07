import json
import os
import tempfile
import time
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

import projects
import watcher
from watcher import CcusageCollector, CodexCollector

# Committed, sanitized samples of real agent data directories (see fixtures/README.md).
FIXTURES_DIR = Path(__file__).parent / "fixtures"
CODEX_FIXTURE_DIR = FIXTURES_DIR / ".codex"
CODEFUSE_FIXTURE_DIR = FIXTURES_DIR / ".codefuse" / "engine"
ANTCC_FIXTURE_DIR = CODEFUSE_FIXTURE_DIR / "cc"
ANTCODEX_FIXTURE_DIR = CODEFUSE_FIXTURE_DIR / "codex"
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


def _codex_session_rows(
    day: date,
    usages: list[tuple[int, int, int]],
    project: str = "proj",
    model: str = "gpt-5",
) -> list[dict]:
    rows = [
        {
            "timestamp": _codex_ts(day, 0),
            "type": "session_meta",
            "payload": {"cwd": f"/home/user/developer/{project}"},
        },
        {
            "timestamp": _codex_ts(day, 1),
            "type": "turn_context",
            "payload": {
                "cwd": f"/home/user/developer/{project}",
                "model": model,
            },
        },
    ]
    for second, (input_tokens, cached_tokens, output_tokens) in enumerate(usages, 2):
        rows.append({
            "timestamp": _codex_ts(day, second),
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {"total_token_usage": {
                    "input_tokens": input_tokens,
                    "cached_input_tokens": cached_tokens,
                    "output_tokens": output_tokens,
                }},
            },
        })
    return rows


def _write_codex_session(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


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

class TestCcusageCollector(unittest.TestCase):
    def setUp(self):
        # These tests isolate token deltas with mocked ccusage data. Real cwd
        # discovery is covered by test_projects and the committed fixtures.
        resolver = patch('watcher.ClaudeProjectResolver.resolve', side_effect=lambda _, key: {
            '-home-nxw-developer-token-exporter': 'token-exporter',
            '-home-nxw-developer-devkit-ant': 'devkit-ant',
        }[key])
        resolver.start()
        self.addCleanup(resolver.stop)

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
        updated["projects"]["-home-nxw-developer-token-exporter"][0]["totalCost"] = 0.015
        updated["projects"]["-home-nxw-developer-token-exporter"][0]["modelBreakdowns"][0]["inputTokens"] = 1500
        updated["projects"]["-home-nxw-developer-token-exporter"][0]["modelBreakdowns"][0]["cost"] = 0.015
        mock_run.return_value = updated
        collector.scan_history()
        assert len(records) == 3
        assert records[2][1]["input_tokens"] == 500  # 1500 - 1000 delta
        assert records[2][1]["cost_usd"] == pytest.approx(0.005)

    @patch("watcher.os.path.isdir", return_value=True)
    @patch("watcher._scan_antcc_dir")
    def test_antcc_incremental_rescan_emits_only_delta_cost(self, mock_scan, mock_isdir):
        first = {
            "projects": {
                "-home-nxw-developer-token-exporter": [
                    {
                        "date": _DAY1_STR,
                        "modelBreakdowns": [
                            {
                                "modelName": "GLM-5.2",
                                "inputTokens": 100,
                                "outputTokens": 0,
                                "cacheCreationTokens": 0,
                                "cacheReadTokens": 0,
                                "cost": 0.001,
                            }
                        ],
                    }
                ]
            }
        }
        second = json.loads(json.dumps(first))
        second["projects"]["-home-nxw-developer-token-exporter"][0]["modelBreakdowns"][0]["inputTokens"] = 150
        second["projects"]["-home-nxw-developer-token-exporter"][0]["modelBreakdowns"][0]["cost"] = 0.0015
        mock_scan.side_effect = [first, second]

        records = []
        collector = CcusageCollector(
            claude_dirs=["/fake/cc"],
            days_back=7,
            on_record=lambda agent, rec: records.append((agent, rec)),
            pricing_rates={"zai/glm-5.2": {"input": 10, "cached": 0, "output": 0, "cache_creation": 0}},
        )

        collector.scan_history()
        collector.scan_history()

        assert [rec["input_tokens"] for _, rec in records] == [100, 50]
        assert records[0][1]["cost_usd"] == pytest.approx(0.001)
        assert records[1][1]["cost_usd"] == pytest.approx(0.0005)

    @patch("watcher.os.path.isdir", return_value=True)
    @patch("watcher._run_ccusage")
    def test_collector_state_persists_cost_totals(self, mock_run, mock_isdir):
        mock_run.return_value = SAMPLE_CCUSAGE_INSTANCES
        with tempfile.TemporaryDirectory() as d:
            state_file = Path(d) / "collector.json"
            collector = CcusageCollector(
                claude_dirs=["/fake/dir"],
                days_back=7,
                on_record=lambda agent, rec: None,
                state_file=str(state_file),
            )

            collector.scan_history()

            state = json.loads(state_file.read_text())
        rows = state["last_totals"]
        key = f"unknown\x00token-exporter\x00GLM-5.1\x00{_DAY1_STR}"
        assert rows[key]["cost_usd"] == pytest.approx(0.01)

    @patch("watcher.os.path.isdir", return_value=True)
    @patch("watcher._run_ccusage")
    def test_legacy_state_without_cost_does_not_emit_full_day_cost(self, mock_run, mock_isdir):
        updated = json.loads(json.dumps(SAMPLE_CCUSAGE_INSTANCES))
        updated["projects"]["-home-nxw-developer-token-exporter"][0]["inputTokens"] = 1500
        updated["projects"]["-home-nxw-developer-token-exporter"][0]["modelBreakdowns"][0]["inputTokens"] = 1500
        updated["projects"]["-home-nxw-developer-token-exporter"][0]["modelBreakdowns"][0]["cost"] = 0.015
        mock_run.return_value = updated

        records = []
        collector = CcusageCollector(
            claude_dirs=["/fake/dir"],
            days_back=7,
            on_record=lambda agent, rec: records.append((agent, rec)),
        )
        collector._last_totals = {
            ("unknown", "token-exporter", "GLM-5.1", _DAY1_STR): {
                "input_tokens": 1000,
                "output_tokens": 200,
                "cache_creation_tokens": 50,
                "cache_read_tokens": 500,
            }
        }

        collector.scan_history()

        assert records[0][1]["input_tokens"] == 500
        assert records[0][1]["cost_usd"] == 0

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


class TestAntccFixture(unittest.TestCase):
    """Parse the committed, sanitized real-world AntCC fixture end-to-end."""

    def _collect(self, pricing_rates=None):
        records = []
        collector = CcusageCollector(
            claude_dirs=[str(ANTCC_FIXTURE_DIR)],
            days_back=UNBOUNDED_DAYS_BACK,
            on_record=lambda agent, rec: records.append((agent, rec)),
            pricing_rates=pricing_rates,
        )
        collector.scan_history()
        return records

    def test_streaming_message_updates_are_deduped(self):
        records = self._collect()

        assert len(records) == 1
        agent, rec = records[0]
        assert agent == "antcc"
        assert rec["project"] == "transfer"
        assert rec["model"] == "claude-opus-4-6"
        # Three real streaming updates share one message id; only the last wins.
        assert rec["input_tokens"] == 3
        assert rec["output_tokens"] == 355
        assert rec["cache_creation_tokens"] == 4677
        assert rec["cache_read_tokens"] == 20851
        assert rec["cost_usd"] == 0
        assert rec["timestamp"].date().isoformat() == "2026-04-14"

    def test_fixture_preserves_missing_request_id(self):
        fixture_file = (
            ANTCC_FIXTURE_DIR
            / "projects"
            / "-home-nxw-developer-transfer"
            / "4ddb168d-9741-4794-9318-72edd668ad79.jsonl"
        )
        rows = [json.loads(line) for line in fixture_file.read_text().splitlines()]
        usage_rows = [row for row in rows if (row.get("message") or {}).get("usage")]

        assert len({row["message"]["id"] for row in usage_rows}) == 1
        assert all("requestId" not in row for row in usage_rows)

    def test_applies_litellm_pricing_to_fixture(self):
        rates = {
            "claude-opus-4-6": {
                "input": 1.0,
                "cached": 0.1,
                "output": 2.0,
                "cache_creation": 0.5,
            }
        }
        records = self._collect(pricing_rates=rates)
        _, rec = records[0]
        assert rec["cost_usd"] == pytest.approx(
            (3 * 1.0 + 20851 * 0.1 + 355 * 2.0 + 4677 * 0.5) / 1e6
        )


class TestCodexCollector(unittest.TestCase):
    def test_unchanged_scan_does_not_reopen_jsonl_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            codex_dir = Path(tmp) / ".codex"
            session_file = codex_dir / "sessions" / "rollout.jsonl"
            _write_codex_session(
                session_file,
                _codex_session_rows(_DAY1, [(1000, 400, 50)]),
            )
            records = []
            collector = CodexCollector(
                codex_dirs=[str(codex_dir)],
                on_record=lambda agent, rec: records.append((agent, rec)),
            )

            collector.scan_history()
            with patch(
                "watcher.open",
                side_effect=AssertionError("unchanged JSONL file was reopened"),
                create=True,
            ):
                collector.scan_history()

        assert len(records) == 1

    def test_append_increments_metrics_exactly_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            codex_dir = Path(tmp) / ".codex"
            session_file = codex_dir / "sessions" / "rollout.jsonl"
            rows = _codex_session_rows(_DAY1, [(1000, 400, 50)])
            _write_codex_session(session_file, rows)
            records = []
            collector = CodexCollector(
                codex_dirs=[str(codex_dir)],
                on_record=lambda agent, rec: records.append((agent, rec)),
            )
            collector.scan_history()
            records.clear()

            appended = _codex_session_rows(_DAY1, [(1500, 700, 80)])[-1]
            with session_file.open("a") as f:
                f.write(json.dumps(appended) + "\n")
            collector.scan_history()
            collector.scan_history()

        assert len(records) == 1
        _, record = records[0]
        assert record["input_tokens"] == 200
        assert record["cache_read_tokens"] == 300
        assert record["output_tokens"] == 30

    def test_new_session_file_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            codex_dir = Path(tmp) / ".codex"
            sessions_dir = codex_dir / "sessions"
            _write_codex_session(
                sessions_dir / "first.jsonl",
                _codex_session_rows(_DAY1, [(1000, 400, 50)], project="first"),
            )
            records = []
            collector = CodexCollector(
                codex_dirs=[str(codex_dir)],
                on_record=lambda agent, rec: records.append((agent, rec)),
            )
            collector.scan_history()
            records.clear()

            _write_codex_session(
                sessions_dir / "second.jsonl",
                _codex_session_rows(_DAY1, [(500, 100, 20)], project="second"),
            )
            collector.scan_history()

        assert len(records) == 1
        _, record = records[0]
        assert record["project"] == "second"
        assert record["input_tokens"] == 400
        assert record["cache_read_tokens"] == 100

    def test_deleted_session_file_is_evicted(self):
        with tempfile.TemporaryDirectory() as tmp:
            codex_dir = Path(tmp) / ".codex"
            session_file = codex_dir / "sessions" / "rollout.jsonl"
            _write_codex_session(
                session_file,
                _codex_session_rows(_DAY1, [(1000, 400, 50)]),
            )
            collector = CodexCollector(codex_dirs=[str(codex_dir)])
            collector.scan_history()
            directory_cache = collector._file_caches[os.path.abspath(codex_dir)]
            assert len(directory_cache) == 1

            session_file.unlink()
            collector.scan_history()

        assert directory_cache == {}

    def test_truncation_and_replacement_invalidate_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            codex_dir = Path(tmp) / ".codex"
            session_file = codex_dir / "sessions" / "rollout.jsonl"
            _write_codex_session(
                session_file,
                _codex_session_rows(_DAY1, [(1000, 0, 50)]),
            )
            records = []
            collector = CodexCollector(
                codex_dirs=[str(codex_dir)],
                on_record=lambda agent, rec: records.append((agent, rec)),
            )
            collector.scan_history()
            records.clear()

            with patch(
                "watcher._parse_codex_session_file",
                wraps=watcher._parse_codex_session_file,
            ) as parse_file:
                _write_codex_session(
                    session_file,
                    _codex_session_rows(_DAY1, [(100, 0, 10)]),
                )
                collector.scan_history()

                replacement = session_file.with_suffix(".replacement")
                _write_codex_session(
                    replacement,
                    _codex_session_rows(_DAY1, [(150, 0, 15)]),
                )
                replacement.replace(session_file)
                collector.scan_history()

        assert parse_file.call_count == 2
        assert len(records) == 1
        assert records[0][1]["input_tokens"] == 50
        assert records[0][1]["output_tokens"] == 5

    def test_partial_final_line_is_processed_when_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            codex_dir = Path(tmp) / ".codex"
            session_file = codex_dir / "sessions" / "rollout.jsonl"
            rows = _codex_session_rows(
                _DAY1,
                [(1000, 400, 50), (1500, 700, 80)],
            )
            complete_prefix = "\n".join(json.dumps(row) for row in rows[:-1]) + "\n"
            final_line = json.dumps(rows[-1])
            split_at = len(final_line) // 2
            session_file.parent.mkdir(parents=True)
            session_file.write_text(complete_prefix + final_line[:split_at])
            records = []
            collector = CodexCollector(
                codex_dirs=[str(codex_dir)],
                on_record=lambda agent, rec: records.append((agent, rec)),
            )
            collector.scan_history()
            records.clear()

            with session_file.open("a") as f:
                f.write(final_line[split_at:] + "\n")
            collector.scan_history()

        assert len(records) == 1
        _, record = records[0]
        assert record["input_tokens"] == 200
        assert record["cache_read_tokens"] == 300
        assert record["output_tokens"] == 30

    def test_cached_totals_are_pruned_when_day_leaves_window(self):
        initial_now = datetime.combine(_DAY2, datetime.min.time(), timezone.utc)
        later_now = initial_now + timedelta(days=10)
        with tempfile.TemporaryDirectory() as tmp:
            codex_dir = Path(tmp) / ".codex"
            session_file = codex_dir / "sessions" / "rollout.jsonl"
            _write_codex_session(
                session_file,
                _codex_session_rows(_DAY1, [(1000, 400, 50)]),
            )
            collector = CodexCollector(codex_dirs=[str(codex_dir)], days_back=7)

            with patch("watcher._utcnow", return_value=initial_now):
                collector.scan_history()
            with patch("watcher._utcnow", return_value=later_now), patch(
                "watcher._parse_codex_session_file",
                wraps=watcher._parse_codex_session_file,
            ) as parse_file:
                collector.scan_history()

            directory_cache = collector._file_caches[os.path.abspath(codex_dir)]

        parse_file.assert_not_called()
        assert len(directory_cache) == 1
        assert next(iter(directory_cache.values())).totals == {}

    def test_restart_does_not_duplicate_persisted_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            codex_dir = Path(tmp) / ".codex"
            state_file = Path(tmp) / "codex-state.json"
            _write_codex_session(
                codex_dir / "sessions" / "rollout.jsonl",
                _codex_session_rows(_DAY1, [(1000, 400, 50)]),
            )
            first_records = []
            first = CodexCollector(
                codex_dirs=[str(codex_dir)],
                state_file=str(state_file),
                on_record=lambda agent, rec: first_records.append((agent, rec)),
            )
            first.scan_history()

            restarted_records = []
            restarted = CodexCollector(
                codex_dirs=[str(codex_dir)],
                state_file=str(state_file),
                on_record=lambda agent, rec: restarted_records.append((agent, rec)),
            )
            restarted.scan_history()

        assert len(first_records) == 1
        assert restarted_records == []

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


class TestAntcodexFixture(unittest.TestCase):
    """Parse the committed, sanitized real-world AntCodex fixture end-to-end."""

    def _collect(self, pricing_rates=None):
        records = []
        collector = CodexCollector(
            codex_dirs=[str(ANTCODEX_FIXTURE_DIR)],
            days_back=UNBOUNDED_DAYS_BACK,
            on_record=lambda agent, rec: records.append((agent, rec)),
            pricing_rates=pricing_rates,
        )
        collector.scan_history()
        return records

    def test_aggregates_real_codefuse_codex_session(self):
        records = self._collect()

        assert len(records) == 1
        agent, rec = records[0]
        assert agent == "antcodex"
        assert rec["project"] == "ip-monitor"
        assert rec["model"] == "gpt-5.5"
        # Empty token_count records and repeated cumulative snapshots are ignored.
        assert rec["input_tokens"] == 2963
        assert rec["cache_read_tokens"] == 13824
        assert rec["output_tokens"] == 367
        assert rec["cache_creation_tokens"] == 0
        assert rec["cost_usd"] == 0
        assert rec["timestamp"].date().isoformat() == "2026-05-14"

    def test_applies_pricing_to_fixture(self):
        rates = {"gpt-5.5": {"input": 3.0, "cached": 0.3, "output": 15.0, "cache_creation": 3.0}}
        records = self._collect(pricing_rates=rates)
        _, rec = records[0]
        assert rec["cost_usd"] == pytest.approx((2963 * 3.0 + 13824 * 0.3 + 367 * 15.0) / 1e6)


if __name__ == "__main__":
    unittest.main()


def _claude_root(tmp_path, name, cwd="/work/demo"):
    """A real .claude root with one session; ccusage itself is mocked per test."""
    project = tmp_path / name / ".claude" / "projects" / cwd.replace("/", "-")
    project.mkdir(parents=True)
    session = project / "session.jsonl"
    session.write_text(json.dumps({"type": "user", "cwd": cwd}) + "\n")
    return str(project.parents[1]), session


def _append(path, text='{"type": "user"}\n'):
    with path.open("a") as f:
        f.write(text)


def _ccusage_report(project_key, tokens):
    return {"projects": {project_key: [{
        "date": _DAY1_STR,
        "modelBreakdowns": [{"modelName": "m", "inputTokens": tokens, "cost": 0}],
    }]}}


def _antcc_line(msg_id, tokens, day=_DAY1):
    return json.dumps({
        "type": "assistant",
        "cwd": "/work/demo",
        "timestamp": f"{day.isoformat()}T12:00:00.000Z",
        "message": {"id": msg_id, "model": "m", "usage": {"input_tokens": tokens}},
    }) + "\n"


class TestUnchangedRootsAreNotRescanned:
    """An idle poll must cost a stat per file and never under- or over-count."""

    KEY = "-work-demo"

    def _collector(self, *roots, **kwargs):
        records = []
        collector = CcusageCollector(
            claude_dirs=list(roots),
            on_record=lambda _, record: records.append(record),
            **kwargs,
        )
        return collector, records

    def test_unchanged_root_does_not_rerun_ccusage(self, tmp_path):
        root, _ = _claude_root(tmp_path, "a")
        collector, records = self._collector(root)
        report = _ccusage_report(self.KEY, 100)
        with patch("watcher._run_ccusage", return_value=report) as run:
            for _ in range(3):
                collector.check_updates()
        assert run.call_count == 1
        assert [r["input_tokens"] for r in records] == [100]

    @pytest.mark.parametrize("change", ["append", "new_session", "subagent", "delete", "index"])
    def test_any_scan_input_changing_reruns_ccusage(self, tmp_path, change):
        root, session = _claude_root(tmp_path, "a")
        (session.parent / "other.jsonl").write_text("{}\n")
        collector, _ = self._collector(root)
        with patch("watcher._run_ccusage", return_value=_ccusage_report(self.KEY, 100)) as run:
            collector.check_updates()
            if change == "append":
                _append(session)
            elif change == "new_session":
                (session.parent / "new.jsonl").write_text("{}\n")
            elif change == "subagent":
                nested = session.parent / "abc" / "subagents"
                nested.mkdir(parents=True)
                (nested / "agent.jsonl").write_text("{}\n")
            elif change == "delete":
                (session.parent / "other.jsonl").unlink()
            elif change == "index":
                (session.parent / "sessions-index.json").write_text("{}")
            collector.check_updates()
        assert run.call_count == 2

    def test_files_no_scan_reads_do_not_rerun_ccusage(self, tmp_path):
        root, session = _claude_root(tmp_path, "a")
        collector, _ = self._collector(root)
        with patch("watcher._run_ccusage", return_value=_ccusage_report(self.KEY, 100)) as run:
            collector.check_updates()
            (session.parent / "memory").mkdir()
            (session.parent / "memory" / "MEMORY.md").write_text("note")
            (session.parent / "output.txt").write_text("tool result")
            collector.check_updates()
        assert run.call_count == 1

    def test_only_the_changed_root_is_rescanned_and_totals_stay_summed(self, tmp_path):
        first, _ = _claude_root(tmp_path, "one")
        second, second_session = _claude_root(tmp_path, "two")
        collector, records = self._collector(first, second)
        reports = {
            first: _ccusage_report(self.KEY, 100),
            second: _ccusage_report(self.KEY, 200),
        }
        with patch("watcher._run_ccusage", side_effect=lambda root, *_: reports[root]) as run:
            collector.check_updates()
            run.reset_mock()
            reports[second] = _ccusage_report(self.KEY, 250)
            _append(second_session)
            collector.check_updates()
        assert [call.args[0] for call in run.call_args_list] == [second]
        # 300 is both roots summed; dropping the cached root would stall the cursor.
        assert [r["input_tokens"] for r in records] == [300, 50]

    def test_failed_scan_is_retried_even_though_files_did_not_change(self, tmp_path):
        root, _ = _claude_root(tmp_path, "a")
        collector, records = self._collector(root)
        with patch("watcher._run_ccusage", side_effect=[None, _ccusage_report(self.KEY, 100)]) as run:
            collector.check_updates()
            assert records == []
            collector.check_updates()
            collector.check_updates()
        assert run.call_count == 2
        assert [r["input_tokens"] for r in records] == [100]

    def test_failed_root_holds_back_its_agent_until_retry_succeeds(self, tmp_path):
        first, _ = _claude_root(tmp_path, "one")
        second, _ = _claude_root(tmp_path, "two")
        collector, records = self._collector(first, second)
        good = {first: _ccusage_report(self.KEY, 100), second: _ccusage_report(self.KEY, 200)}
        failing = {first: good[first], second: None}
        with patch("watcher._run_ccusage", side_effect=lambda root, *_: failing[root]):
            collector.check_updates()
        assert records == []
        with patch("watcher._run_ccusage", side_effect=lambda root, *_: good[root]) as run:
            collector.check_updates()
            collector.check_updates()
        # The healthy root is served from its cache; the failed one is rescanned once.
        assert [call.args[0] for call in run.call_args_list] == [second]
        assert [r["input_tokens"] for r in records] == [300]

    def test_write_during_a_scan_is_picked_up_on_the_next_poll(self, tmp_path):
        root, session = _claude_root(tmp_path, "a")
        collector, records = self._collector(root)

        def racing_scan(*_):
            report = _ccusage_report(self.KEY, 100)
            _append(session)  # Lands after ccusage read the file.
            return report

        with patch("watcher._run_ccusage", side_effect=racing_scan):
            collector.check_updates()
        with patch("watcher._run_ccusage", return_value=_ccusage_report(self.KEY, 150)) as run:
            collector.check_updates()
        assert run.call_count == 1
        assert [r["input_tokens"] for r in records] == [100, 50]

    def test_window_change_rescans_without_replaying_days_that_left_it(self, tmp_path):
        root = tmp_path / ".codefuse" / "engine" / "cc"
        session = root / "projects" / "-work-demo" / "session.jsonl"
        session.parent.mkdir(parents=True)
        session.write_text(_antcc_line("old", 100, _DAY1))
        collector, records = self._collector(str(root), days_back=7)
        start = datetime.combine(_DAY1, datetime.min.time(), timezone.utc)
        with patch("watcher._utcnow", return_value=start):
            collector.check_updates()
        assert [r["input_tokens"] for r in records] == [100]
        # _DAY1's cursor gets pruned now, so reusing the cached scan would re-emit it.
        with patch("watcher._utcnow", return_value=start + timedelta(days=8)):
            collector.check_updates()
            collector.check_updates()
        assert [r["input_tokens"] for r in records] == [100]

    def test_unchanged_antcc_root_is_not_reparsed_and_append_counts_once(self, tmp_path):
        root = tmp_path / ".codefuse" / "engine" / "cc"
        session = root / "projects" / "-work-demo" / "session.jsonl"
        session.parent.mkdir(parents=True)
        session.write_text(_antcc_line("one", 100))
        collector, records = self._collector(str(root))
        with patch("watcher._scan_antcc_dir", wraps=watcher._scan_antcc_dir) as scan:
            collector.check_updates()
            collector.check_updates()
            assert scan.call_count == 1
            _append(session, _antcc_line("two", 40))
            collector.check_updates()
            collector.check_updates()
            assert scan.call_count == 2
        assert [r["input_tokens"] for r in records] == [100, 40]

    def test_unresolved_project_is_not_reread_until_its_metadata_changes(self, tmp_path):
        root = tmp_path / ".claude"
        project = root / "projects" / self.KEY
        project.mkdir(parents=True)
        (project / "session.jsonl").write_text('{"type": "no-cwd-here"}\n')
        collector, records = self._collector(str(root))
        with (
            patch("watcher._run_ccusage", return_value=_ccusage_report(self.KEY, 100)),
            patch("projects._project_cwd", wraps=projects._project_cwd) as read_cwd,
        ):
            collector.check_updates()
            collector.check_updates()
            assert read_cwd.call_count == 1
            assert records == []
            (project / "sessions-index.json").write_text(json.dumps({"originalPath": "/work/demo"}))
            collector.check_updates()
        assert [(r["project"], r["input_tokens"]) for r in records] == [("demo", 100)]

    def test_restart_rescans_once_and_does_not_double_count(self, tmp_path):
        root, _ = _claude_root(tmp_path, "a")
        state = str(tmp_path / "state.json")
        report = _ccusage_report(self.KEY, 100)
        collector, records = self._collector(root, state_file=state)
        with patch("watcher._run_ccusage", return_value=report):
            collector.scan_history()
        restarted, new_records = self._collector(root, state_file=state)
        with patch("watcher._run_ccusage", return_value=report) as run:
            restarted.scan_history()
            restarted.scan_history()
        assert run.call_count == 1
        assert [r["input_tokens"] for r in records] == [100]
        assert new_records == []

    def test_idle_poll_logs_nothing_at_info(self, tmp_path, caplog):
        root, _ = _claude_root(tmp_path, "a")
        collector, _ = self._collector(root)
        with (
            patch("watcher._run_ccusage", return_value=_ccusage_report(self.KEY, 100)),
            caplog.at_level("INFO", logger="token-stats"),
        ):
            collector.check_updates()
            assert "emitted 1 records" in caplog.text
            caplog.clear()
            collector.check_updates()
        assert caplog.text == ""


class TestUnchangedCodexRootsAreNotRescanned:
    def _collector(self, tmp_path, **kwargs):
        codex_dir = tmp_path / ".codex"
        session = codex_dir / "sessions" / "rollout.jsonl"
        _write_codex_session(session, _codex_session_rows(_DAY1, [(1000, 400, 50)]))
        records = []
        collector = CodexCollector(
            codex_dirs=[str(codex_dir)],
            on_record=lambda _, record: records.append(record),
            **kwargs,
        )
        return collector, records, session

    def test_unchanged_root_skips_the_scan(self, tmp_path):
        collector, records, _ = self._collector(tmp_path)
        with patch("watcher._scan_antcodex_dir", wraps=watcher._scan_antcodex_dir) as scan:
            for _ in range(3):
                collector.check_updates()
        assert scan.call_count == 1
        assert len(records) == 1

    def test_append_rescans_and_counts_once(self, tmp_path):
        collector, records, session = self._collector(tmp_path)
        collector.check_updates()
        _write_codex_session(session, _codex_session_rows(_DAY1, [(1000, 400, 50), (1500, 400, 80)]))
        collector.check_updates()
        collector.check_updates()
        assert len(records) == 2
        assert records[1]["output_tokens"] == 30

    def test_pricing_reload_refreshes_the_cost_cursor(self, tmp_path):
        rates = {"gpt-5": {"input": 0.01, "cached": 0.001, "output": 0.02}}
        collector, records, _ = self._collector(tmp_path, pricing_rates=rates)
        collector.check_updates()
        cursor_key = next(iter(collector._last_totals))
        cost_before = collector._last_totals[cursor_key]["cost_usd"]
        assert cost_before > 0
        with patch("watcher._scan_antcodex_dir", wraps=watcher._scan_antcodex_dir) as scan:
            collector.check_updates()
            assert scan.call_count == 0
            # An equal table must not cost a rescan, and a changed one must.
            collector.pricing_rates = {model: dict(rate) for model, rate in rates.items()}
            collector.check_updates()
            assert scan.call_count == 0
            collector.pricing_rates = {"gpt-5": {"input": 0.1, "cached": 0.01, "output": 0.2}}
            collector.check_updates()
            assert scan.call_count == 1
        assert collector._last_totals[cursor_key]["cost_usd"] == pytest.approx(cost_before * 10)
        assert len(records) == 1  # Repricing alone is not new usage.


class TestTreeSignature:
    def test_missing_root_is_never_unchanged(self, tmp_path):
        assert watcher._tree_signature(str(tmp_path / "missing")) is None

    def test_only_scan_inputs_are_fingerprinted(self, tmp_path):
        (tmp_path / "a.jsonl").write_text("{}\n")
        (tmp_path / "sessions-index.json").write_text("{}")
        (tmp_path / "notes.md").write_text("x")
        (tmp_path / "b.json").write_text("{}")
        assert set(watcher._tree_signature(str(tmp_path))) == {
            str(tmp_path / "a.jsonl"),
            str(tmp_path / "sessions-index.json"),
        }

    def test_symlinks_are_followed_and_cycles_terminate(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "linked.jsonl"
        target.write_text("{}\n")
        root = tmp_path / "root"
        (root / "real").mkdir(parents=True)
        (root / "dir-link").symlink_to(outside)
        (root / "real" / "file-link.jsonl").symlink_to(target)
        (root / "real" / "loop").symlink_to(root)
        (root / "real" / "broken.jsonl").symlink_to(tmp_path / "gone")

        before = watcher._tree_signature(str(root))

        # Exactly these: the loop back to root must not revisit anything.
        assert set(before) == {
            str(root / "dir-link" / "linked.jsonl"),
            str(root / "real" / "file-link.jsonl"),
            str(root / "real" / "broken.jsonl"),
        }
        assert before[str(root / "real" / "file-link.jsonl")] is not None
        assert before[str(root / "real" / "broken.jsonl")] is None
        _append(target)
        after = watcher._tree_signature(str(root))
        assert after[str(root / "dir-link" / "linked.jsonl")] != before[str(root / "dir-link" / "linked.jsonl")]

    def test_chmod_counts_as_a_change(self, tmp_path):
        session = tmp_path / "s.jsonl"
        session.write_text("{}\n")
        before = watcher._tree_signature(str(tmp_path))
        time.sleep(0.02)  # st_ctime has coarser resolution than the call is long.
        session.chmod(0o600)
        assert watcher._tree_signature(str(tmp_path)) != before

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
    def test_unreadable_subdirectory_is_recorded_and_clearing_it_is_a_change(self, tmp_path):
        locked = tmp_path / "locked"
        locked.mkdir()
        (locked / "s.jsonl").write_text("{}\n")
        locked.chmod(0)
        try:
            blocked = watcher._tree_signature(str(tmp_path))
            assert blocked == {str(locked): None}
        finally:
            locked.chmod(0o700)
        assert watcher._tree_signature(str(tmp_path)) != blocked
