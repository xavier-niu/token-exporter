import json
import logging
from collections import defaultdict

from prometheus_client import Counter, Gauge

from state_io import atomic_write_json

logger = logging.getLogger("token-stats")

LABELS = ["source", "agent", "project", "model"]
DAILY_LABELS = ["source", "agent", "project", "model", "date"]

input_tokens_total = Counter("codeagent_input_tokens_total", "Total input tokens", LABELS)
output_tokens_total = Counter("codeagent_output_tokens_total", "Total output tokens", LABELS)
cache_creation_tokens_total = Counter("codeagent_cache_creation_tokens_total", "Total cache creation tokens", LABELS)
cache_read_tokens_total = Counter("codeagent_cache_read_tokens_total", "Total cache read tokens", LABELS)
cost_usd_total = Counter("codeagent_cost_usd_total", "Total cost in USD", LABELS)

daily_input_tokens = Gauge("codeagent_daily_input_tokens", "Daily input tokens", DAILY_LABELS)
daily_output_tokens = Gauge("codeagent_daily_output_tokens", "Daily output tokens", DAILY_LABELS)
daily_cache_creation_tokens = Gauge("codeagent_daily_cache_creation_tokens", "Daily cache creation tokens", DAILY_LABELS)
daily_cache_read_tokens = Gauge("codeagent_daily_cache_read_tokens", "Daily cache read tokens", DAILY_LABELS)
daily_cost_usd = Gauge("codeagent_daily_cost_usd", "Daily cost USD", DAILY_LABELS)

_FIELDS = ("input", "output", "cache_creation", "cache_read", "cost")


def _zero_values() -> dict[str, float]:
    return {f: 0.0 for f in _FIELDS}


_cumulative_totals: dict[tuple, dict[str, float]] = defaultdict(_zero_values)
_daily_data: dict[tuple, dict[str, float]] = defaultdict(_zero_values)

_source: str = ""


def set_source(source: str):
    global _source
    _source = source


def _apply_counters(src: str, agent: str, project: str, model: str, values: dict[str, float]) -> None:
    if values.get("input", 0) > 0:
        input_tokens_total.labels(source=src, agent=agent, project=project, model=model).inc(values["input"])
    if values.get("output", 0) > 0:
        output_tokens_total.labels(source=src, agent=agent, project=project, model=model).inc(values["output"])
    if values.get("cache_creation", 0) > 0:
        cache_creation_tokens_total.labels(source=src, agent=agent, project=project, model=model).inc(values["cache_creation"])
    if values.get("cache_read", 0) > 0:
        cache_read_tokens_total.labels(source=src, agent=agent, project=project, model=model).inc(values["cache_read"])
    if values.get("cost", 0) > 0:
        cost_usd_total.labels(source=src, agent=agent, project=project, model=model).inc(values["cost"])


def _apply_daily(src: str, agent: str, project: str, model: str, date_str: str, values: dict[str, float]) -> None:
    daily_input_tokens.labels(source=src, agent=agent, project=project, model=model, date=date_str).set(values.get("input", 0))
    daily_output_tokens.labels(source=src, agent=agent, project=project, model=model, date=date_str).set(values.get("output", 0))
    daily_cache_creation_tokens.labels(source=src, agent=agent, project=project, model=model, date=date_str).set(values.get("cache_creation", 0))
    daily_cache_read_tokens.labels(source=src, agent=agent, project=project, model=model, date=date_str).set(values.get("cache_read", 0))
    daily_cost_usd.labels(source=src, agent=agent, project=project, model=model, date=date_str).set(values.get("cost", 0))


def record_usage(agent: str, rec: dict):
    model = rec["model"]
    project = rec.get("project", "unknown")
    src = _source

    delta = {
        "input": rec["input_tokens"],
        "output": rec["output_tokens"],
        "cache_creation": rec["cache_creation_tokens"],
        "cache_read": rec["cache_read_tokens"],
        "cost": rec["cost_usd"],
    }

    _apply_counters(src, agent, project, model, delta)

    cum_key = (src, agent, project, model)
    cum = _cumulative_totals[cum_key]
    for f in _FIELDS:
        cum[f] += delta[f]

    ts = rec.get("timestamp")
    date_str = ts.strftime("%Y-%m-%d") if ts else "unknown"
    d_key = (src, agent, project, model, date_str)
    d = _daily_data[d_key]
    for f in _FIELDS:
        d[f] += delta[f]
    _apply_daily(src, agent, project, model, date_str, d)


def snapshot_state() -> dict:
    return {
        "schema_version": 1,
        "cumulative": [{"key": list(k), "values": dict(v)} for k, v in _cumulative_totals.items()],
        "daily": [{"key": list(k), "values": dict(v)} for k, v in _daily_data.items()],
    }


def restore_state(state: dict) -> None:
    """Seed counters/gauges and in-memory accumulators from a saved snapshot.

    Must be called BEFORE the HTTP server starts and before any record_usage
    call, otherwise restored values are added on top of fresh deltas.
    """
    if not state:
        return
    restored_counters = 0
    restored_daily = 0
    for entry in state.get("cumulative", []):
        key = tuple(entry.get("key", []))
        if len(key) != 4:
            continue
        src, agent, project, model = key
        values = entry.get("values", {})
        _apply_counters(src, agent, project, model, values)
        _cumulative_totals[key] = {f: float(values.get(f, 0)) for f in _FIELDS}
        restored_counters += 1
    for entry in state.get("daily", []):
        key = tuple(entry.get("key", []))
        if len(key) != 5:
            continue
        src, agent, project, model, date_str = key
        values = entry.get("values", {})
        _apply_daily(src, agent, project, model, date_str, values)
        _daily_data[key] = {f: float(values.get(f, 0)) for f in _FIELDS}
        restored_daily += 1
    logger.info("Restored %d counter series and %d daily series", restored_counters, restored_daily)


def load_state(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not load metrics state from %s: %s", path, e)
        return {}


def save_state(path: str) -> None:
    atomic_write_json(path, snapshot_state())
