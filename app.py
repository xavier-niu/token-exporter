import json
import logging
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from prometheus_client import start_http_server

import pricing
from config import load_config
from metrics import load_state, record_usage, restore_state, save_state, set_source
from watcher import CcusageCollector, CodexCollector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("token-stats")


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def _persist_all(collectors: list, counter_state_file: str) -> None:
    """Flush both state files, dedup cursors before the counter snapshot.

    The restart merge assumes collector offsets are written before the counter
    snapshot, and the two must be written together so a crash leaves them at the
    same point in time (a stale cursor with a fresh counter double-counts the
    gap on the next start).
    """
    for collector in collectors:
        collector.flush_state()
    if counter_state_file:
        save_state(counter_state_file)


def poll_loop(collectors: list, interval: int, persist_interval: int,
              counter_state_file: str, next_persist: float | None = None):
    """Scan every `interval` so metrics stay fresh, but flush to disk only every
    `persist_interval`. Persisting less often keeps fsync off the frequent poll
    path; a crash then re-emits at most one persist_interval of usage on restart,
    which is idempotent because the cursor and counter snapshot advance together.
    """
    now = time.monotonic()
    if next_persist is None:
        next_persist = now + persist_interval
    for collector in collectors:
        collector.check_updates()
    if now >= next_persist:
        _persist_all(collectors, counter_state_file)
        next_persist = now + persist_interval
    timer = threading.Timer(
        interval, poll_loop,
        args=[collectors, interval, persist_interval, counter_state_file, next_persist],
    )
    timer.daemon = True
    timer.start()


def reload_pricing_loop(collectors: list, config: dict):
    """Periodically reload pricing so a long-running container picks up new or
    repriced models without a restart.

    The rates table is otherwise loaded once at startup and frozen for the
    process lifetime. load_pricing is TTL-gated internally, so the upstream
    refetch still happens at most once per pricing_ttl_secs; this loop re-runs
    it on that cadence and swaps the refreshed table into each collector.
    """
    interval = config["pricing_ttl_secs"]
    try:
        rates = pricing.load_pricing(
            override_path=config["pricing_override"],
            cache_path=config["pricing_cache"],
            ttl=interval,
        )
        if rates:
            for collector in collectors:
                collector.pricing_rates = rates
            logger.info("Reloaded pricing for %d models", len(rates))
    except Exception as e:
        logger.warning("Pricing reload failed: %s", e)
    timer = threading.Timer(interval, reload_pricing_loop, args=[collectors, config])
    timer.daemon = True
    timer.start()


def _install_shutdown_handlers(counter_state_file: str, collectors: list) -> None:
    """Flush all persisted state on SIGTERM/SIGINT.

    Both the collector dedup cursors and the counter snapshot are written, in
    that order, so the two files stay consistent (the restart merge assumes
    collector offsets precede the counter snapshot). Flushing the cursors too
    matters once the poll interval is long: a fresh counter snapshot paired with
    a stale cursor would make the next start re-emit and double-count the gap.
    SIGKILL cannot be intercepted; on SIGKILL the snapshot is at most one
    poll-cycle stale, which is the unavoidable bound.
    """
    def _handler(signum, _frame):
        logger.info("Received signal %d, flushing state", signum)
        try:
            _persist_all(collectors, counter_state_file)
        finally:
            sys.exit(0)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def _bootstrap_from_last_totals(last_totals_files: list[str], source: str) -> dict:
    """Rebuild a counter/daily snapshot from existing per-collector last_totals.

    Used on first deploy of the persisted-counter fix when no counter snapshot
    file exists yet. Without this, the first restart after the upgrade still
    shows zero historic tokens/cost; with it, we recover from the collector
    last_totals snapshot.
    """
    cumulative: dict[tuple, dict[str, float]] = {}
    daily: dict[tuple, dict[str, float]] = {}
    field_map = {
        "input": "input_tokens",
        "output": "output_tokens",
        "cache_creation": "cache_creation_tokens",
        "cache_read": "cache_read_tokens",
        "cost": "cost_usd",
    }
    for path in last_totals_files:
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            continue
        for key_str, values in data.get("last_totals", {}).items():
            parts = tuple(key_str.split("\x00"))
            if len(parts) != 4:
                continue
            agent, project, model, date_str = parts
            cum_key = (source, agent, project, model)
            cum = cumulative.setdefault(cum_key, {f: 0.0 for f in field_map})
            d_key = (source, agent, project, model, date_str)
            d = daily.setdefault(d_key, {f: 0.0 for f in field_map})
            for short, long in field_map.items():
                v = float(values.get(long, 0) or 0)
                cum[short] += v
                d[short] += v
    return {
        "schema_version": 1,
        "cumulative": [{"key": list(k), "values": v} for k, v in cumulative.items()],
        "daily": [{"key": list(k), "values": v} for k, v in daily.items()],
    }


def _merge_state_with_bootstrap(state: dict, bootstrap: dict) -> dict:
    """Keep restored counters at least as high as collector last_totals.

    Collector offsets are written before the metrics counter snapshot. If the
    process dies between those writes, the next startup can recover by taking
    the max of the saved counter snapshot and the collector totals.
    """
    merged = {"schema_version": 1, "cumulative": [], "daily": []}
    for section in ("cumulative", "daily"):
        rows: dict[tuple, dict[str, float]] = {}
        for entry in state.get(section, []):
            key = tuple(entry.get("key", []))
            if not key:
                continue
            rows[key] = {
                field: float(value or 0)
                for field, value in entry.get("values", {}).items()
            }

        for entry in bootstrap.get(section, []):
            key = tuple(entry.get("key", []))
            if not key:
                continue
            row = rows.setdefault(key, {})
            for field, value in entry.get("values", {}).items():
                row[field] = max(float(row.get(field, 0) or 0), float(value or 0))

        merged[section] = [
            {"key": list(key), "values": values}
            for key, values in rows.items()
        ]
    return merged


def main():
    config = load_config()
    logger.info("Config: %s", config)

    set_source(config["source"])

    cc_state = config["state_file"].replace(".json", "-ccusage.json") if config["state_file"] else ""
    codex_state = config["state_file"].replace(".json", "-codex.json") if config["state_file"] else ""
    counter_state_file = config["state_file"].replace(".json", "-counters.json") if config["state_file"] else ""

    if counter_state_file:
        last_totals_files = [p for p in (cc_state, codex_state) if p]
        state = load_state(counter_state_file)
        bootstrap = _bootstrap_from_last_totals(last_totals_files, config["source"])
        if state:
            state = _merge_state_with_bootstrap(state, bootstrap)
            restore_state(state)
            logger.info("Restored counter state from %s", counter_state_file)
        else:
            if bootstrap["cumulative"] or bootstrap["daily"]:
                restore_state(bootstrap)
                logger.info("Bootstrapped counter state from last_totals files")

    pricing_rates = pricing.load_pricing(
        override_path=config["pricing_override"],
        cache_path=config["pricing_cache"],
        ttl=config["pricing_ttl_secs"],
    )
    logger.info("Loaded pricing for %d models", len(pricing_rates))

    collectors = []

    if config["claude_dirs"]:
        cc_collector = CcusageCollector(
            claude_dirs=config["claude_dirs"],
            days_back=config["days_back"],
            on_record=record_usage,
            state_file=cc_state,
            timezone_name=config["timezone"],
            pricing_rates=pricing_rates,
        )
        logger.info("Scanning Claude Code / AntCC history...")
        cc_collector.scan_history()
        logger.info("Claude Code / AntCC scan complete")
        collectors.append(cc_collector)

    if config["codex_dirs"]:
        codex_collector = CodexCollector(
            codex_dirs=config["codex_dirs"],
            days_back=config["days_back"],
            on_record=record_usage,
            state_file=codex_state,
            timezone_name=config["timezone"],
            pricing_rates=pricing_rates,
        )
        logger.info("Scanning Codex history...")
        codex_collector.scan_history()
        logger.info("Codex scan complete")
        collectors.append(codex_collector)

    if counter_state_file:
        save_state(counter_state_file)

    start_http_server(config["listen_port"])
    logger.info("Prometheus metrics on :%d/metrics", config["listen_port"])

    health_port = config["listen_port"] + 1
    server = HTTPServer(("", health_port), HealthHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    logger.info("Health endpoint on :%d/health", health_port)

    _install_shutdown_handlers(counter_state_file, collectors)

    pricing_timer = threading.Timer(
        config["pricing_ttl_secs"], reload_pricing_loop, args=[collectors, config]
    )
    pricing_timer.daemon = True
    pricing_timer.start()
    logger.info("Reloading pricing every %ds", config["pricing_ttl_secs"])

    logger.info(
        "Watching for updates every %ds, persisting every %ds",
        config["watch_interval"], config["persist_interval"],
    )
    poll_loop(
        collectors, config["watch_interval"], config["persist_interval"], counter_state_file
    )

    threading.Event().wait()


if __name__ == "__main__":
    main()
