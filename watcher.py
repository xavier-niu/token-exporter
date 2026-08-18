import json
import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pricing
from state_io import atomic_write_json

logger = logging.getLogger("token-stats")


def _utcnow() -> datetime:
    """Current UTC time, indirected so tests can freeze "now".

    The collectors derive their scan window (since = now - days_back) from this,
    so patching it lets a fixture with fixed dates exercise the date-window
    filtering deterministically.
    """
    return datetime.now(timezone.utc)


def _detect_agent(base_dir: str) -> str:
    """Detect agent type from the base directory path."""
    if base_dir.endswith("/.codex") or "/.codex/" in base_dir:
        return "codex"
    if "/codex" in base_dir:
        return "antcodex"
    base = os.path.basename(os.path.normpath(base_dir))
    if base.lower() == "cc":
        return "antcc"
    if base.lower() in ("codefuse", ".codefuse"):
        return "antcc"
    if base.lower() == ".claude" or base == "claude":
        return "cc"
    return "unknown"


def _extract_project(project_key: str) -> str:
    """Extract project name from a ccusage --instances project key or directory path.

    The key encodes the working directory with dashes replacing slashes,
    e.g. '-home-nxw-developer-token-exporter'.
    Strip the first 3 segments (home, user, developer) and the remainder
    is the project name.
    """
    stripped = project_key.lstrip("-")
    parts = stripped.split("-")
    if len(parts) >= 3:
        return "-".join(parts[3:]) or stripped
    return stripped


def _run_ccusage(claude_dir: str, since: date, timezone_name: str = "UTC") -> dict | None:
    """Run ccusage claude daily --json --offline --timezone <tz> --instances and return parsed JSON.

    ccusage v20 split per-agent reporting into subcommands; the Claude Code
    report (and the --instances per-project breakdown) now lives under
    `ccusage claude daily` rather than the top-level `ccusage daily`.
    """
    cmd = [
        "ccusage", "claude", "daily", "--json", "--offline",
        "--timezone", timezone_name,
        "--instances",
        "--since", since.strftime("%Y%m%d"),
    ]
    env = {**os.environ, "CLAUDE_CONFIG_DIR": claude_dir}
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
        if result.returncode != 0:
            logger.error("ccusage failed for %s: %s", claude_dir, result.stderr.strip())
            return None
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
        logger.error("ccusage error for %s: %s", claude_dir, e)
        return None


def _scan_antcc_dir(base_dir: str, since: date, timezone_name: str = "UTC") -> dict:
    """Scan AntCC JSONL files directly with msg_id dedup.

    AntCC records have empty requestId, so ccusage doesn't dedup them
    (https://github.com/ryoppippi/ccusage/issues/976). This parser
    deduplicates by msg_id to produce correct counts.

    Cost is left at 0 here; it is priced from the shared pricing table by
    _process_ccusage_data (on the per-poll token delta), so computing it on the
    cumulative totals here would just be thrown away.

    Returns dict in ccusage --instances format:
    {"projects": {"-home-...": [{"date": "...", "modelBreakdowns": [...]}]}}
    """
    tz = ZoneInfo(timezone_name)
    since_local = since.strftime("%Y-%m-%d")
    projects_dir = os.path.join(base_dir, "projects")

    # Aggregate per (project_key, date, model) with dedup by msg_id
    # {project_key: {date: {model: {"input": N, "output": N, ...}}}}
    agg: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    # Per-msg_id token values for dedup:
    # {project_key: {date: {msg_id: {"input": N, ...}}}}
    msg_values: dict[str, dict[str, dict[str, dict[str, int]]]] = {}

    for project_dir in Path(projects_dir).iterdir():
        if not project_dir.is_dir():
            continue
        project_key = project_dir.name
        agg[project_key] = {}
        msg_values[project_key] = {}

        for jsonl_file in project_dir.rglob("*.jsonl"):
            try:
                with open(jsonl_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        usage = obj.get("message", {}).get("usage")
                        if not usage:
                            continue

                        msg_id = obj.get("message", {}).get("id", "")
                        ts_str = obj.get("timestamp", "")
                        if not ts_str:
                            continue

                        try:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        except (ValueError, TypeError):
                            continue

                        local_ts = ts.astimezone(tz)
                        date_str = local_ts.strftime("%Y-%m-%d")
                        if date_str < since_local:
                            continue

                        model = obj.get("message", {}).get("model", "unknown") or "unknown"
                        input_tokens = usage.get("input_tokens", 0) or 0
                        output_tokens = usage.get("output_tokens", 0) or 0
                        cache_creation = usage.get("cache_creation_input_tokens", 0) or 0
                        cache_read = usage.get("cache_read_input_tokens", 0) or 0

                        if not input_tokens and not output_tokens and not cache_creation and not cache_read:
                            continue

                        date_agg = agg[project_key].setdefault(date_str, {})
                        if model not in date_agg:
                            date_agg[model] = {"input": 0, "output": 0, "cache_creation": 0, "cache_read": 0}
                        m = date_agg[model]

                        # Dedup by msg_id: subtract old values, add new (streaming updates)
                        if msg_id:
                            date_msgs = msg_values[project_key].setdefault(date_str, {})
                            old = date_msgs.get(msg_id)
                            if old:
                                m["input"] -= old["input"]
                                m["output"] -= old["output"]
                                m["cache_creation"] -= old["cache_creation"]
                                m["cache_read"] -= old["cache_read"]
                            date_msgs[msg_id] = {"input": input_tokens, "output": output_tokens, "cache_creation": cache_creation, "cache_read": cache_read}

                        m["input"] += input_tokens
                        m["output"] += output_tokens
                        m["cache_creation"] += cache_creation
                        m["cache_read"] += cache_read
            except OSError:
                continue

    # Convert to ccusage --instances format
    result: dict[str, list] = {"projects": {}}
    for project_key, dates in sorted(agg.items()):
        entries = []
        for date_str, models in sorted(dates.items()):
            breakdowns = []
            for model, tokens in sorted(models.items()):
                breakdowns.append({
                    "modelName": model,
                    "inputTokens": tokens["input"],
                    "outputTokens": tokens["output"],
                    "cacheCreationTokens": tokens["cache_creation"],
                    "cacheReadTokens": tokens["cache_read"],
                    "cost": 0,
                })
            entries.append({
                "date": date_str,
                "inputTokens": sum(b["inputTokens"] for b in breakdowns),
                "outputTokens": sum(b["outputTokens"] for b in breakdowns),
                "cacheCreationTokens": sum(b["cacheCreationTokens"] for b in breakdowns),
                "cacheReadTokens": sum(b["cacheReadTokens"] for b in breakdowns),
                "totalTokens": sum(
                    b["inputTokens"]
                    + b["outputTokens"]
                    + b["cacheCreationTokens"]
                    + b["cacheReadTokens"]
                    for b in breakdowns
                ),
                # No cost here: antcc is priced from the token delta in
                # _process_ccusage_data, so a cumulative cost would be discarded.
                "totalCost": 0,
                "modelsUsed": [b["modelName"] for b in breakdowns],
                "modelBreakdowns": breakdowns,
            })
        if entries:
            result["projects"][project_key] = entries
    return result


# Token fields to track for delta computation
_TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_creation_tokens", "cache_read_tokens")
_STATE_FIELDS = (*_TOKEN_FIELDS, "cost_usd")


def _project_from_cwd(cwd: str) -> str:
    if not cwd:
        return "unknown"
    return os.path.basename(os.path.normpath(cwd)) or "unknown"


def _codex_usage_totals(info: dict) -> dict[str, int]:
    usage = info.get("total_token_usage") or {}
    return {
        "input": usage.get("input_tokens", 0) or 0,
        "cached": usage.get("cached_input_tokens", 0) or 0,
        "output": usage.get("output_tokens", 0) or 0,
        "reasoning": usage.get("reasoning_output_tokens", 0) or 0,
    }


def _codex_delta(current: dict[str, int], previous: dict[str, int] | None) -> dict[str, int]:
    if previous is None:
        return current

    delta = {field: current[field] - previous.get(field, 0) for field in current}
    if any(value < 0 for value in delta.values()):
        return current
    return delta


_CODEX_TOTAL_FIELDS = (
    "inputTokens",
    "outputTokens",
    "cacheCreationTokens",
    "cacheReadTokens",
)


@dataclass
class _CodexFileCacheEntry:
    signature: tuple[int, int, int, int]
    since: str
    totals: dict


def _codex_file_signature(stat_result: os.stat_result) -> tuple[int, int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
    )


def _prune_codex_totals(totals: dict, since_local: str) -> None:
    """Drop cached daily aggregates that have left the configured window."""
    for project, dates in list(totals.items()):
        for date_str in list(dates):
            if date_str < since_local:
                del dates[date_str]
        if not dates:
            del totals[project]


def _merge_codex_totals(target: dict, source: dict) -> None:
    for project, dates in source.items():
        for date_str, models in dates.items():
            date_models = target.setdefault(project, {}).setdefault(date_str, {})
            for model, source_tokens in models.items():
                tokens = date_models.setdefault(
                    model,
                    {field: 0 for field in _CODEX_TOTAL_FIELDS},
                )
                for field in _CODEX_TOTAL_FIELDS:
                    tokens[field] += source_tokens.get(field, 0)


def _parse_codex_session_file(
    session_file: Path,
    since_local: str,
    tz: ZoneInfo,
) -> dict | None:
    """Parse one session into reusable per-project/model/day totals.

    None means the file could not be read. An empty dict is a successful parse
    with no in-window token records (and is cacheable).
    """
    agg: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    project = "unknown"
    model = "unknown"
    previous: dict[str, int] | None = None

    try:
        with open(session_file, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                payload = obj.get("payload", {})
                entry_type = obj.get("type")
                if entry_type == "session_meta":
                    project = _project_from_cwd(payload.get("cwd", "")) or project
                    continue
                if entry_type == "turn_context":
                    if payload.get("cwd"):
                        project = _project_from_cwd(payload["cwd"])
                    if payload.get("model"):
                        model = payload["model"]
                    continue
                if entry_type != "event_msg" or payload.get("type") != "token_count":
                    continue

                info = payload.get("info") or {}
                current = _codex_usage_totals(info)
                if not any(current.values()):
                    continue

                delta = _codex_delta(current, previous)
                previous = current
                if not any(value > 0 for value in delta.values()):
                    continue

                ts_str = obj.get("timestamp", "")
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue
                date_str = ts.astimezone(tz).date().isoformat()
                if date_str < since_local:
                    continue

                date_models = agg.setdefault(project, {}).setdefault(date_str, {})
                tokens = date_models.setdefault(
                    model,
                    {field: 0 for field in _CODEX_TOTAL_FIELDS},
                )
                cached = max(delta["cached"], 0)
                tokens["inputTokens"] += max(delta["input"] - cached, 0)
                tokens["outputTokens"] += max(delta["output"], 0)
                tokens["cacheReadTokens"] += cached
    except OSError:
        return None

    return agg


def _scan_antcodex_dir(
    codex_dir: str,
    since: date,
    timezone_name: str = "UTC",
    pricing_rates: dict | None = None,
    file_cache: dict[str, _CodexFileCacheEntry] | None = None,
) -> dict:
    """Scan Codex JSONL sessions, reusing aggregates for unchanged files."""
    tz = ZoneInfo(timezone_name)
    since_local = since.isoformat()
    sessions_dir = Path(codex_dir) / "sessions"
    result: dict[str, dict] = {"projects": {}}
    if not sessions_dir.is_dir():
        return result

    if file_cache is None:
        file_cache = {}
    seen: set[str] = set()
    agg: dict[str, dict[str, dict[str, dict[str, int]]]] = {}

    for session_file in sorted(sessions_dir.rglob("*.jsonl")):
        cache_key = os.fspath(session_file)
        seen.add(cache_key)
        cached = file_cache.get(cache_key)
        try:
            signature = _codex_file_signature(session_file.stat())
        except OSError:
            if cached:
                if cached.since < since_local:
                    _prune_codex_totals(cached.totals, since_local)
                    cached.since = since_local
                _merge_codex_totals(agg, cached.totals)
            continue

        if cached and cached.signature == signature and cached.since <= since_local:
            if cached.since < since_local:
                _prune_codex_totals(cached.totals, since_local)
                cached.since = since_local
            file_totals = cached.totals
        else:
            file_totals = _parse_codex_session_file(session_file, since_local, tz)
            if file_totals is None:
                if cached:
                    if cached.since < since_local:
                        _prune_codex_totals(cached.totals, since_local)
                        cached.since = since_local
                    _merge_codex_totals(agg, cached.totals)
                continue
            file_cache[cache_key] = _CodexFileCacheEntry(
                signature=signature,
                since=since_local,
                totals=file_totals,
            )
        _merge_codex_totals(agg, file_totals)

    for cache_key in file_cache.keys() - seen:
        del file_cache[cache_key]

    for project, dates in sorted(agg.items()):
        entries = []
        for date_str, models in sorted(dates.items()):
            entries.append({
                "date": date_str,
                "modelBreakdowns": [
                    {
                        "modelName": model,
                        **tokens,
                        "cost": pricing.cost_for(
                            pricing_rates,
                            model,
                            tokens["inputTokens"],
                            tokens["cacheReadTokens"],
                            tokens["outputTokens"],
                            tokens["cacheCreationTokens"],
                        ) if pricing_rates else 0,
                    }
                    for model, tokens in sorted(models.items())
                ],
            })
        if entries:
            result["projects"][f"-home-user-developer-{project}"] = entries
    return result


def _load_last_totals(path: str) -> dict[tuple, dict]:
    """Load last-seen totals from a JSON state file."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not load state from %s: %s", path, e)
        return {}
    totals = {}
    for key_str, values in data.get("last_totals", {}).items():
        key = tuple(key_str.split("\x00"))
        totals[key] = {k: v for k, v in values.items() if k in _STATE_FIELDS}
    return totals


def _save_last_totals(path: str, totals: dict[tuple, dict]) -> None:
    """Persist last-seen totals to a JSON state file."""
    serializable = {
        "last_totals": {
            "\x00".join(key): values for key, values in totals.items()
        }
    }
    atomic_write_json(path, serializable)


def _prune_stale_totals(totals: dict[tuple, dict], cutoff: date) -> None:
    """Remove entries with dates older than cutoff."""
    stale_keys = [k for k in totals if len(k) >= 4 and k[3] < cutoff.isoformat()]
    for k in stale_keys:
        del totals[k]


def _process_ccusage_data(
    data: dict,
    agent: str,
    on_record,
    last_totals: dict,
    project: str = "",
    pricing_rates: dict | None = None,
) -> int:
    """Parse ccusage JSON output and emit delta records.

    Handles two formats:
    - --instances: {"projects": {"key": [entries]}}
    - daily: {"daily": [entries]}
    """
    count = 0

    # Normalize to list of (project, entries) pairs
    if "projects" in data:
        items = [(_extract_project(k), entries) for k, entries in data["projects"].items()]
    elif "daily" in data:
        items = [(project or "unknown", data["daily"])]
    else:
        return 0

    for project, entries in items:
        for entry in entries:
            date_str = entry.get("date", "")
            for bd in entry.get("modelBreakdowns", []):
                model = bd.get("modelName", "unknown")
                current = {
                    "input_tokens": bd.get("inputTokens", 0),
                    "output_tokens": bd.get("outputTokens", 0),
                    "cache_creation_tokens": bd.get("cacheCreationTokens", 0),
                    "cache_read_tokens": bd.get("cacheReadTokens", 0),
                    "cost_usd": bd.get("cost", 0) or 0,
                }
                key = (agent, project, model, date_str)
                last = last_totals.get(key)
                if last is None:
                    delta = {field: current[field] for field in _TOKEN_FIELDS}
                else:
                    delta = {
                        field: max(current[field] - last.get(field, 0), 0)
                        for field in _TOKEN_FIELDS
                    }
                last_totals[key] = current

                has_delta = any(delta.get(f, 0) > 0 for f in _TOKEN_FIELDS)
                if not has_delta:
                    continue

                cost_delta = (
                    pricing.cost_for(
                        pricing_rates,
                        model,
                        delta["input_tokens"],
                        delta["cache_read_tokens"],
                        delta["output_tokens"],
                        delta["cache_creation_tokens"],
                    )
                    if pricing_rates
                    else 0.0
                )
                if not cost_delta:
                    # No pricing table, or model not in it (e.g. a model newer
                    # than the loaded rates): degrade to ccusage's own offline
                    # cost rather than emitting $0. Once the model lands in
                    # LiteLLM or the local override, in-process pricing wins.
                    if last is None:
                        cost_delta = current["cost_usd"]
                    elif "cost_usd" not in last:
                        cost_delta = 0
                    else:
                        cost_delta = max(current["cost_usd"] - last.get("cost_usd", 0), 0)
                    if pricing_rates and cost_delta:
                        logger.warning(
                            "no in-process price for model %r; using ccusage cost", model
                        )

                record = {
                    "timestamp": datetime.fromisoformat(date_str) if date_str else None,
                    "model": model,
                    "project": project,
                    "input_tokens": delta["input_tokens"],
                    "output_tokens": delta["output_tokens"],
                    "cache_creation_tokens": delta["cache_creation_tokens"],
                    "cache_read_tokens": delta["cache_read_tokens"],
                    "cost_usd": cost_delta,
                }
                if on_record:
                    on_record(agent, record)
                count += 1
    return count


class _BaseCollector:
    """Shared state/lifecycle for the per-agent collectors.

    Holds the dedup cursor and the scan window; subclasses implement
    scan_history() for their data layout. Persistence is split so the frequent
    poll loop can keep metrics fresh without writing to disk every cycle: a scan
    updates the in-memory cursor, and the poll loop (or shutdown) flushes it on a
    slower cadence via flush_state().
    """

    def __init__(self, dirs, days_back=7, on_record=None, state_file="",
                 timezone_name="UTC", pricing_rates=None):
        self.dirs = dirs
        self.days_back = days_back
        self.on_record = on_record
        self.state_file = state_file
        self.timezone_name = timezone_name
        self.pricing_rates = pricing_rates or {}
        self._last_totals: dict[tuple, dict[str, float]] = {}
        if state_file:
            self._last_totals = _load_last_totals(state_file)
            logger.info("Loaded %s state from %s (%d entries)",
                        type(self).__name__, state_file, len(self._last_totals))

    def _scan_since(self):
        """Window start date, pruning cursor entries that fell out of it."""
        since = (_utcnow() - timedelta(days=self.days_back)).date()
        _prune_stale_totals(self._last_totals, since)
        return since

    def _finish_scan(self, persist, total_records):
        if persist and self.state_file:
            _save_last_totals(self.state_file, self._last_totals)
        logger.info("%s: emitted %d records", type(self).__name__, total_records)

    def scan_history(self, persist=True):
        """Scan all configured directories and emit records.

        persist=False updates the in-memory cursor but skips the disk write, so
        the frequent poll loop can keep metrics fresh while persistence runs on
        its own slower cadence (see poll_loop).
        """
        raise NotImplementedError

    def check_updates(self):
        """Re-scan for new data; persistence is handled by the poll loop."""
        self.scan_history(persist=False)

    def flush_state(self):
        """Persist the dedup cursor without re-scanning (for shutdown)."""
        if self.state_file:
            _save_last_totals(self.state_file, self._last_totals)


class CcusageCollector(_BaseCollector):
    """Collects token usage from Claude Code / AntCC data directories.

    Uses ccusage CLI for Claude Code (cc) directories and inline JSONL
    parsing with msg_id dedup for AntCC directories (ccusage doesn't
    dedup when requestId is empty). Costs for both paths are computed
    in-process from token deltas via the shared pricing map, so they stay
    correct as rates change without depending on ccusage's bundled offline
    pricing snapshot; ccusage's own cost is used only as a fallback for a
    model the pricing map doesn't yet know.
    """

    def __init__(
        self,
        claude_dirs: list[str],
        days_back: int = 7,
        on_record=None,
        state_file: str = "",
        timezone_name: str = "UTC",
        pricing_rates: dict | None = None,
    ):
        super().__init__(claude_dirs, days_back, on_record, state_file,
                         timezone_name, pricing_rates)

    def scan_history(self, persist=True):
        since = self._scan_since()
        total_records = 0
        for claude_dir in self.dirs:
            if not os.path.isdir(claude_dir):
                continue
            if not os.path.isdir(os.path.join(claude_dir, "projects")):
                logger.debug("Skipping %s: no projects/ directory", claude_dir)
                continue
            agent = _detect_agent(claude_dir)
            if agent == "antcc":
                data = _scan_antcc_dir(claude_dir, since, self.timezone_name)
                total_records += _process_ccusage_data(
                    data,
                    agent,
                    self.on_record,
                    self._last_totals,
                    pricing_rates=self.pricing_rates,
                )
            else:
                data = _run_ccusage(claude_dir, since, self.timezone_name)
                if data is None:
                    continue
                total_records += _process_ccusage_data(
                    data,
                    agent,
                    self.on_record,
                    self._last_totals,
                    pricing_rates=self.pricing_rates,
                )
        self._finish_scan(persist, total_records)


class CodexCollector(_BaseCollector):
    """Collects token usage from Codex data directories.

    Scans the session JSONL files directly (with cumulative-counter dedup) and
    prices them via the shared pricing table, rather than shelling out to a CLI.
    """

    def __init__(self, codex_dirs: list[str], days_back: int = 7, on_record=None, state_file: str = "", timezone_name: str = "UTC", pricing_rates: dict | None = None):
        super().__init__(codex_dirs, days_back, on_record, state_file,
                         timezone_name, pricing_rates)
        self._file_caches: dict[str, dict[str, _CodexFileCacheEntry]] = {}

    def scan_history(self, persist=True):
        since = self._scan_since()
        total_records = 0
        for codex_dir in self.dirs:
            if not os.path.isdir(codex_dir):
                continue
            if not os.path.isdir(os.path.join(codex_dir, "sessions")):
                logger.debug("Skipping %s: no sessions/ directory", codex_dir)
                continue
            agent = _detect_agent(codex_dir)
            cache_key = os.path.abspath(codex_dir)
            file_cache = self._file_caches.setdefault(cache_key, {})
            data = _scan_antcodex_dir(
                codex_dir,
                since,
                self.timezone_name,
                self.pricing_rates,
                file_cache=file_cache,
            )
            total_records += _process_ccusage_data(
                data,
                agent,
                self.on_record,
                self._last_totals,
                pricing_rates=self.pricing_rates,
            )
        self._finish_scan(persist, total_records)
