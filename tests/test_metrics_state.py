"""Unit tests for counter persistence across restarts.

Covers the restart-safe counter design: a snapshot of cumulative counter
values is written to disk after each poll cycle and restored at startup so
Prometheus never observes a counter reset.
"""
import importlib
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from prometheus_client import REGISTRY, generate_latest


def _fresh_metrics():
    """Reload metrics with a clean Prometheus registry.

    prometheus_client uses a global default registry that rejects duplicate
    metric names. We unregister every existing collector before reload so the
    module's top-level Counter constructors can re-register cleanly.
    """
    from prometheus_client import REGISTRY
    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            pass
    import metrics
    return importlib.reload(metrics)


def _counter_value(counter, **labels):
    return counter.labels(**labels)._value.get()


class TestRecordUsage(unittest.TestCase):
    def test_record_increments_counters_and_cumulative(self):
        m = _fresh_metrics()
        m.set_source("devant")
        m.record_usage("cc", {
            "model": "claude",
            "project": "p",
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_tokens": 10,
            "cache_read_tokens": 5,
            "cost_usd": 0.25,
            "timestamp": datetime(2026, 5, 10),
        })
        labels = {"source": "devant", "agent": "cc", "project": "p", "model": "claude"}
        self.assertEqual(_counter_value(m.input_tokens_total, **labels), 100)
        self.assertEqual(_counter_value(m.output_tokens_total, **labels), 50)
        self.assertEqual(_counter_value(m.cost_usd_total, **labels), 0.25)
        cum = m._cumulative_totals[("devant", "cc", "p", "claude")]
        self.assertEqual(cum["input"], 100)
        self.assertEqual(cum["cost"], 0.25)

    def test_multiple_records_accumulate(self):
        m = _fresh_metrics()
        m.set_source("devant")
        for _ in range(3):
            m.record_usage("cc", {
                "model": "claude",
                "project": "p",
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
                "cost_usd": 0,
                "timestamp": datetime(2026, 5, 10),
            })
        labels = {"source": "devant", "agent": "cc", "project": "p", "model": "claude"}
        self.assertEqual(_counter_value(m.input_tokens_total, **labels), 30)
        self.assertEqual(m._cumulative_totals[("devant", "cc", "p", "claude")]["input"], 30)


class TestSaveLoadRoundtrip(unittest.TestCase):
    def test_save_then_load_preserves_state(self):
        m = _fresh_metrics()
        m.set_source("devant")
        m.record_usage("cc", {
            "model": "claude",
            "project": "p",
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_creation_tokens": 100,
            "cache_read_tokens": 50,
            "cost_usd": 1.23,
            "timestamp": datetime(2026, 5, 10),
        })
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "state.json")
            m.save_state(path)
            loaded = m.load_state(path)
        self.assertEqual(loaded["schema_version"], 1)
        cum = {tuple(e["key"]): e["values"] for e in loaded["cumulative"]}
        self.assertIn(("devant", "cc", "p", "claude"), cum)
        self.assertEqual(cum[("devant", "cc", "p", "claude")]["input"], 1000)
        self.assertEqual(cum[("devant", "cc", "p", "claude")]["cost"], 1.23)

    def test_load_missing_file_returns_empty(self):
        m = _fresh_metrics()
        self.assertEqual(m.load_state("/nonexistent/path/state.json"), {})

    def test_load_corrupt_file_returns_empty(self):
        m = _fresh_metrics()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "state.json"
            path.write_text("{ not json")
            self.assertEqual(m.load_state(str(path)), {})

    def test_save_uses_atomic_rename(self):
        m = _fresh_metrics()
        m.set_source("devant")
        m.record_usage("cc", {
            "model": "claude", "project": "p",
            "input_tokens": 1, "output_tokens": 1,
            "cache_creation_tokens": 0, "cache_read_tokens": 0,
            "cost_usd": 0, "timestamp": datetime(2026, 5, 10),
        })
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "state.json"
            m.save_state(str(path))
            # Temp file must not linger after successful save.
            self.assertFalse((path.parent / f"{path.name}.tmp").exists())
            self.assertTrue(path.exists())


class TestRestore(unittest.TestCase):
    def test_restore_seeds_counters_to_pre_restart_value(self):
        """The core regression: after a restart, the counter must come up
        at the pre-restart value, not zero, so Prometheus sees no reset.
        """
        m = _fresh_metrics()
        m.set_source("devant")
        # Simulate pre-restart state: process a record, snapshot.
        m.record_usage("cc", {
            "model": "claude", "project": "p",
            "input_tokens": 1_226_534, "output_tokens": 12_345,
            "cache_creation_tokens": 0, "cache_read_tokens": 0,
            "cost_usd": 0.5, "timestamp": datetime(2026, 5, 10),
        })
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "state.json")
            m.save_state(path)
            snapshot = m.load_state(path)

        # Simulate restart: fresh module, counter is zero.
        m2 = _fresh_metrics()
        m2.set_source("devant")
        labels = {"source": "devant", "agent": "cc", "project": "p", "model": "claude"}
        self.assertEqual(_counter_value(m2.input_tokens_total, **labels), 0)

        # Restore from snapshot, counter should match pre-restart value.
        m2.restore_state(snapshot)
        self.assertEqual(_counter_value(m2.input_tokens_total, **labels), 1_226_534)
        self.assertEqual(_counter_value(m2.cost_usd_total, **labels), 0.5)

    def test_restore_then_record_adds_correctly(self):
        """After restore, new record_usage calls should add on top, not duplicate."""
        m = _fresh_metrics()
        m.set_source("devant")
        snapshot = {
            "schema_version": 1,
            "cumulative": [{
                "key": ["devant", "cc", "p", "claude"],
                "values": {"input": 1000, "output": 500, "cache_creation": 0, "cache_read": 0, "cost": 0},
            }],
            "daily": [],
        }
        m.restore_state(snapshot)
        m.record_usage("cc", {
            "model": "claude", "project": "p",
            "input_tokens": 50, "output_tokens": 25,
            "cache_creation_tokens": 0, "cache_read_tokens": 0,
            "cost_usd": 0, "timestamp": datetime(2026, 5, 10),
        })
        labels = {"source": "devant", "agent": "cc", "project": "p", "model": "claude"}
        self.assertEqual(_counter_value(m.input_tokens_total, **labels), 1050)
        # Cumulative tracker stays in sync so the next snapshot is correct.
        self.assertEqual(m._cumulative_totals[("devant", "cc", "p", "claude")]["input"], 1050)

    def test_restore_empty_state_is_noop(self):
        m = _fresh_metrics()
        m.restore_state({})
        m.restore_state(None)

    def test_daily_totals_are_saved_but_not_exported(self):
        m = _fresh_metrics()
        m.set_source("devant")
        key = ["devant", "cc", "p", "claude", "2026-05-10"]
        m.restore_state({
            "schema_version": 1,
            "cumulative": [],
            "daily": [{
                "key": key,
                "values": {"input": 200, "output": 100, "cache_creation": 0, "cache_read": 0, "cost": 0.1},
            }],
        })
        m.record_usage("cc", {
            "model": "claude", "project": "p",
            "input_tokens": 50, "output_tokens": 0,
            "cache_creation_tokens": 0, "cache_read_tokens": 0,
            "cost_usd": 0.02, "timestamp": datetime(2026, 5, 10),
        })

        saved = m.snapshot_state()["daily"]

        self.assertEqual([entry["key"] for entry in saved], [key])
        self.assertEqual(saved[0]["values"]["input"], 250)
        self.assertAlmostEqual(saved[0]["values"]["cost"], 0.12)
        exposition = generate_latest(REGISTRY).decode()
        self.assertIn("codeagent_input_tokens_total", exposition)
        self.assertNotIn("codeagent_daily", exposition)

    def test_restore_skips_malformed_entries(self):
        m = _fresh_metrics()
        snapshot = {
            "schema_version": 1,
            "cumulative": [
                {"key": ["bad"], "values": {"input": 1}},                  # wrong key length
                {"key": ["devant", "cc", "p", "claude"], "values": {"input": 10}},
            ],
            "daily": [
                {"key": ["x", "y"], "values": {}},                          # wrong key length
            ],
        }
        m.restore_state(snapshot)
        labels = {"source": "devant", "agent": "cc", "project": "p", "model": "claude"}
        self.assertEqual(_counter_value(m.input_tokens_total, **labels), 10)


class TestRestartScenarios(unittest.TestCase):
    """End-to-end restart correctness — the bug this whole change exists to fix."""

    def test_full_restart_cycle_preserves_counter(self):
        # --- run 1 ---
        m = _fresh_metrics()
        m.set_source("devant")
        for _ in range(5):
            m.record_usage("cc", {
                "model": "claude", "project": "p",
                "input_tokens": 100, "output_tokens": 50,
                "cache_creation_tokens": 0, "cache_read_tokens": 0,
                "cost_usd": 0, "timestamp": datetime(2026, 5, 10),
            })
        labels = {"source": "devant", "agent": "cc", "project": "p", "model": "claude"}
        self.assertEqual(_counter_value(m.input_tokens_total, **labels), 500)

        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "state.json")
            m.save_state(path)

            # --- restart ---
            m2 = _fresh_metrics()
            m2.set_source("devant")
            m2.restore_state(m2.load_state(path))

            # Counter is at pre-restart value, not zero.
            self.assertEqual(_counter_value(m2.input_tokens_total, **labels), 500)

            # --- run 2: new activity ---
            for _ in range(3):
                m2.record_usage("cc", {
                    "model": "claude", "project": "p",
                    "input_tokens": 100, "output_tokens": 50,
                    "cache_creation_tokens": 0, "cache_read_tokens": 0,
                    "cost_usd": 0, "timestamp": datetime(2026, 5, 10),
                })

            # Final counter = pre-restart + new activity. No reset visible to Prom.
            self.assertEqual(_counter_value(m2.input_tokens_total, **labels), 800)

            # Snapshot remains internally consistent for the next restart.
            m2.save_state(path)
            reloaded = m2.load_state(path)
            cum = {tuple(e["key"]): e["values"] for e in reloaded["cumulative"]}
            self.assertEqual(cum[("devant", "cc", "p", "claude")]["input"], 800)


if __name__ == "__main__":
    unittest.main()
