"""Tests for the first-deploy bootstrap path in app._bootstrap_from_last_totals.

When the counter snapshot file does not yet exist (first run after deploying
the persisted-counter fix), the exporter rebuilds cumulative+daily state from
the existing per-collector last_totals files so historic totals are not lost.
"""
import json
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock

import app


def _write_last_totals(path: Path, entries: dict) -> None:
    """entries: {(agent, project, model, date): {input_tokens, ...}}"""
    serialized = {
        "last_totals": {
            "\x00".join(k): v for k, v in entries.items()
        }
    }
    path.write_text(json.dumps(serialized))


class TestBootstrap(unittest.TestCase):
    def test_empty_when_no_files(self):
        result = app._bootstrap_from_last_totals([], "devant")
        self.assertEqual(result["cumulative"], [])
        self.assertEqual(result["daily"], [])

    def test_missing_files_are_skipped(self):
        result = app._bootstrap_from_last_totals(["/nonexistent/a.json", "/nonexistent/b.json"], "devant")
        self.assertEqual(result["cumulative"], [])

    def test_aggregates_across_sessions_and_files(self):
        with tempfile.TemporaryDirectory() as d:
            cc = Path(d) / "cc.json"
            codex = Path(d) / "codex.json"
            _write_last_totals(cc, {
                ("cc", "p1", "claude", "2026-05-09"): {
                    "input_tokens": 100, "output_tokens": 50,
                    "cache_creation_tokens": 10, "cache_read_tokens": 5,
                    "cost_usd": 0.11,
                },
                ("cc", "p1", "claude", "2026-05-10"): {
                    "input_tokens": 200, "output_tokens": 80,
                    "cache_creation_tokens": 0, "cache_read_tokens": 0,
                    "cost_usd": 0.22,
                },
                ("antcc", "p2", "claude", "2026-05-10"): {
                    "input_tokens": 50, "output_tokens": 25,
                    "cache_creation_tokens": 0, "cache_read_tokens": 0,
                    "cost_usd": 0.05,
                },
            })
            _write_last_totals(codex, {
                ("codex", "p1", "gpt-5", "2026-05-10"): {
                    "input_tokens": 1000, "output_tokens": 200,
                    "cache_creation_tokens": 0, "cache_read_tokens": 0,
                    "cost_usd": 0.5,
                },
            })

            result = app._bootstrap_from_last_totals([str(cc), str(codex)], "devant")

        cum = {tuple(e["key"]): e["values"] for e in result["cumulative"]}
        # cc/p1/claude collapses both date entries into a single cumulative row.
        self.assertEqual(cum[("devant", "cc", "p1", "claude")]["input"], 300)
        self.assertEqual(cum[("devant", "cc", "p1", "claude")]["output"], 130)
        self.assertEqual(cum[("devant", "cc", "p1", "claude")]["cache_creation"], 10)
        self.assertAlmostEqual(cum[("devant", "cc", "p1", "claude")]["cost"], 0.33)
        self.assertEqual(cum[("devant", "antcc", "p2", "claude")]["input"], 50)
        self.assertEqual(cum[("devant", "antcc", "p2", "claude")]["cost"], 0.05)
        self.assertEqual(cum[("devant", "codex", "p1", "gpt-5")]["input"], 1000)
        self.assertEqual(cum[("devant", "codex", "p1", "gpt-5")]["cost"], 0.5)

        daily = {tuple(e["key"]): e["values"] for e in result["daily"]}
        # Daily keeps per-date granularity.
        self.assertEqual(daily[("devant", "cc", "p1", "claude", "2026-05-09")]["input"], 100)
        self.assertEqual(daily[("devant", "cc", "p1", "claude", "2026-05-09")]["cost"], 0.11)
        self.assertEqual(daily[("devant", "cc", "p1", "claude", "2026-05-10")]["input"], 200)

    def test_corrupt_file_does_not_break_others(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.json"
            good = Path(d) / "good.json"
            bad.write_text("{ not json")
            _write_last_totals(good, {
                ("cc", "p", "claude", "2026-05-10"): {
                    "input_tokens": 42, "output_tokens": 0,
                    "cache_creation_tokens": 0, "cache_read_tokens": 0,
                },
            })
            result = app._bootstrap_from_last_totals([str(bad), str(good)], "devant")
        cum = {tuple(e["key"]): e["values"] for e in result["cumulative"]}
        self.assertEqual(cum[("devant", "cc", "p", "claude")]["input"], 42)

    def test_malformed_keys_are_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.json"
            # Mix of valid and malformed keys.
            data = {
                "last_totals": {
                    "agent\x00project\x00model\x00date": {"input_tokens": 5},
                    "only\x00three\x00parts": {"input_tokens": 99},
                    "just-one-part": {"input_tokens": 99},
                }
            }
            path.write_text(json.dumps(data))
            result = app._bootstrap_from_last_totals([str(path)], "devant")
        cum = {tuple(e["key"]): e["values"] for e in result["cumulative"]}
        self.assertEqual(len(cum), 1)
        self.assertEqual(cum[("devant", "agent", "project", "model")]["input"], 5)

    def test_merge_state_with_bootstrap_recovers_lagging_counter_snapshot(self):
        state = {
            "schema_version": 1,
            "cumulative": [
                {
                    "key": ["devant", "antcc", "p", "GLM-5.1"],
                    "values": {"input": 100, "output": 10, "cost": 0.1},
                },
                {
                    "key": ["devant", "cc", "old", "claude"],
                    "values": {"input": 900, "output": 90, "cost": 9},
                },
            ],
            "daily": [
                {
                    "key": ["devant", "antcc", "p", "GLM-5.1", "2026-05-10"],
                    "values": {"input": 100, "output": 10, "cost": 0.1},
                },
            ],
        }
        bootstrap = {
            "schema_version": 1,
            "cumulative": [
                {
                    "key": ["devant", "antcc", "p", "GLM-5.1"],
                    "values": {"input": 150, "output": 15, "cost": 0.15},
                },
                {
                    "key": ["devant", "cc", "old", "claude"],
                    "values": {"input": 800, "output": 80, "cost": 8},
                },
            ],
            "daily": [
                {
                    "key": ["devant", "antcc", "p", "GLM-5.1", "2026-05-10"],
                    "values": {"input": 150, "output": 15, "cost": 0.15},
                },
            ],
        }

        merged = app._merge_state_with_bootstrap(state, bootstrap)

        cum = {tuple(e["key"]): e["values"] for e in merged["cumulative"]}
        self.assertEqual(cum[("devant", "antcc", "p", "GLM-5.1")]["input"], 150)
        self.assertEqual(cum[("devant", "antcc", "p", "GLM-5.1")]["cost"], 0.15)
        # Existing counter state can include older history outside last_totals;
        # it must not be reduced by a smaller bootstrap row.
        self.assertEqual(cum[("devant", "cc", "old", "claude")]["input"], 900)

        daily = {tuple(e["key"]): e["values"] for e in merged["daily"]}
        self.assertEqual(
            daily[("devant", "antcc", "p", "GLM-5.1", "2026-05-10")]["input"],
            150,
        )


if __name__ == "__main__":
    unittest.main()


def test_idle_servers_do_not_wake_up_to_poll():
    server = MagicMock()
    app._serve_in_background(server).join(timeout=5)
    server.serve_forever.assert_called_once_with(poll_interval=app.SERVER_IDLE_POLL_SECS)
    assert app.SERVER_IDLE_POLL_SECS >= 3600


def test_metrics_server_answers_immediately_despite_the_long_idle_poll():
    # Left running: shutdown() would block for the poll interval, and the daemon
    # thread ends with the test process.
    server = app._start_metrics_server(0)
    url = f"http://127.0.0.1:{server.server_port}/metrics"
    with urllib.request.urlopen(url, timeout=5) as response:
        assert response.status == 200
        assert b"# TYPE" in response.read()
