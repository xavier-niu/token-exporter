import json
import logging
import math
import os
import shutil
import sqlite3
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pricing
from projects import (
    ClaudeProjectResolver, PROJECT_NAME_FORMAT, check_project_name_format,
    path_from_file_uri, project_from_cwd,
)
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
    # AntCC is no longer collected; it is still recognized so that a configured
    # AntCC directory is skipped rather than scanned as Claude Code.
    if base.lower() == "cc":
        return "antcc"
    if base.lower() in ("codefuse", ".codefuse"):
        return "antcc"
    if base.lower() == ".claude" or base == "claude":
        return "cc"
    return "unknown"


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


# Token fields to track for delta computation
_TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_creation_tokens", "cache_read_tokens")
_STATE_FIELDS = (*_TOKEN_FIELDS, "cost_usd")


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


# Token fields of a parsed file's totals, in _TOKEN_FIELDS order.
_FILE_TOTAL_FIELDS = (
    "inputTokens",
    "outputTokens",
    "cacheCreationTokens",
    "cacheReadTokens",
)


@dataclass
class _FileCacheEntry:
    signature: tuple[int, int, int, int, int]
    since: str
    totals: dict  # {group: {date_str: {model: {field: int}}}}


def _file_signature(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        # A chmod that makes a session readable changes only this.
        stat_result.st_ctime_ns,
    )


def _is_scan_input(filename: str) -> bool:
    """Session logs, plus the index ClaudeProjectResolver prefers over them."""
    return filename.endswith(".jsonl") or filename == "sessions-index.json"


def _tree_signature(root: str) -> dict[str, tuple | None] | None:
    """Fingerprint every file a scan of `root` could read, without opening any.

    Symlinks are followed because the in-process parsers follow them (ccusage
    ignores them, so this is a superset). A path that can't be inspected is
    recorded as None, so it reads as a change once it can be. An unreadable
    `root` returns None, which must never be treated as unchanged.
    """
    signature: dict[str, tuple | None] = {}
    visited_dirs: set[tuple[int, int]] = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir():
                            stat_result = entry.stat()
                            dir_id = (stat_result.st_dev, stat_result.st_ino)
                            if dir_id not in visited_dirs:
                                visited_dirs.add(dir_id)
                                pending.append(entry.path)
                        elif _is_scan_input(entry.name):
                            signature[entry.path] = _file_signature(entry.stat())
                    except OSError:
                        signature[entry.path] = None
        except OSError:
            if directory == root:
                return None
            signature[directory] = None
    return signature


@dataclass
class _RootScan:
    signature: dict[str, tuple | None]
    inputs: tuple
    data: dict


def _prune_file_totals(totals: dict, since_local: str) -> None:
    """Drop cached daily aggregates that have left the configured window."""
    for group, dates in list(totals.items()):
        for date_str in list(dates):
            if date_str < since_local:
                del dates[date_str]
        if not dates:
            del totals[group]


def _merge_file_totals(target: dict, source: dict) -> None:
    for group, dates in source.items():
        for date_str, models in dates.items():
            date_models = target.setdefault(group, {}).setdefault(date_str, {})
            for model, source_tokens in models.items():
                tokens = date_models.setdefault(
                    model,
                    {field: 0 for field in _FILE_TOTAL_FIELDS},
                )
                for field in _FILE_TOTAL_FIELDS:
                    tokens[field] += source_tokens.get(field, 0)


def _cached_totals(entry: _FileCacheEntry, since_local: str) -> dict:
    """A cache entry's totals, pruned to the current window."""
    if entry.since < since_local:
        _prune_file_totals(entry.totals, since_local)
        entry.since = since_local
    return entry.totals


def _with_signatures(paths):
    """Pair each path with its file signature, or None if it can't be stat'ed."""
    for path in paths:
        try:
            yield path, _file_signature(os.stat(path))
        except OSError:
            yield path, None


def _scan_cached_files(files, file_cache: dict[str, _FileCacheEntry], since_local: str, parse) -> dict:
    """Sum parse(path) over files, reparsing only the files that changed.

    `files` yields (path, signature) pairs, signature None for a file that can't
    be stat'ed. parse returns {group: {date: {model: tokens}}}, or None if the
    file can't be read, in which case its last good totals (if any) are used.
    Entries for files no longer listed are evicted from file_cache.
    """
    seen: set[str] = set()
    agg: dict = {}
    for path, signature in files:
        cache_key = os.fspath(path)
        seen.add(cache_key)
        cached = file_cache.get(cache_key)
        if (
            signature is not None
            and cached
            and cached.signature == signature
            and cached.since <= since_local
        ):
            file_totals = _cached_totals(cached, since_local)
        else:
            file_totals = parse(path) if signature is not None else None
            if file_totals is None:
                if cached:
                    _merge_file_totals(agg, _cached_totals(cached, since_local))
                continue
            file_cache[cache_key] = _FileCacheEntry(
                signature=signature,
                since=since_local,
                totals=file_totals,
            )
        _merge_file_totals(agg, file_totals)

    for cache_key in file_cache.keys() - seen:
        del file_cache[cache_key]
    return agg


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
        # errors="replace": a final line cut mid-character is skipped as bad JSON.
        with open(session_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                payload = obj.get("payload", {})
                entry_type = obj.get("type")
                if entry_type == "session_meta":
                    project = project_from_cwd(payload.get("cwd", "")) or project
                    continue
                if entry_type == "turn_context":
                    if payload.get("cwd"):
                        project = project_from_cwd(payload["cwd"])
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
                    {field: 0 for field in _FILE_TOTAL_FIELDS},
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
    file_cache: dict[str, _FileCacheEntry] | None = None,
) -> dict:
    """Scan Codex JSONL sessions, reusing aggregates for unchanged files."""
    tz = ZoneInfo(timezone_name)
    since_local = since.isoformat()
    sessions_dir = Path(codex_dir) / "sessions"
    result: dict[str, dict] = {"projects": {}}
    if not sessions_dir.is_dir():
        return result

    agg = _scan_cached_files(
        _with_signatures(sorted(sessions_dir.rglob("*.jsonl"))),
        {} if file_cache is None else file_cache,
        since_local,
        lambda session_file: _parse_codex_session_file(session_file, since_local, tz),
    )

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
            result["projects"][project] = entries
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
    check_project_name_format(data)
    totals = {}
    for key_str, values in data.get("last_totals", {}).items():
        key = tuple(key_str.split("\x00"))
        totals[key] = {k: v for k, v in values.items() if k in _STATE_FIELDS}
    return totals


def _save_last_totals(path: str, totals: dict[tuple, dict]) -> None:
    """Persist last-seen totals to a JSON state file."""
    serializable = {
        "project_name_format": PROJECT_NAME_FORMAT,
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
        items = list(data["projects"].items())
    elif "daily" in data:
        items = [(project or "unknown", data["daily"])]
    else:
        return 0

    # Several checkouts/config roots can share a basename. Aggregate before
    # comparing with the cursor, otherwise each overwrites the other's totals.
    totals = {}
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
                combined = totals.setdefault(key, {field: 0 for field in _STATE_FIELDS})
                for field, value in current.items():
                    combined[field] += value or 0

    for key, current in totals.items():
        agent, project, model, date_str = key
        last = last_totals.get(key)
        if last is None:
            delta = {field: current[field] for field in _TOKEN_FIELDS}
        else:
            delta = {
                field: max(current[field] - last.get(field, 0), 0)
                for field in _TOKEN_FIELDS
            }
        # Missing Claude cwd metadata must not lower an aggregate cursor and
        # replay history when that directory becomes readable again. Codex
        # keeps its existing cursor behavior for truncated/replaced sessions.
        last_totals[key] = {
            field: max(value, (last or {}).get(field, 0))
            for field, value in current.items()
        } if agent == 'cc' else current

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
        self._root_scans: dict[str, _RootScan] = {}
        if state_file:
            self._last_totals = _load_last_totals(state_file)
            logger.info("Loaded %s state from %s (%d entries)",
                        type(self).__name__, state_file, len(self._last_totals))

    def _scan_since(self):
        """Window start date, pruning cursor entries that fell out of it."""
        since = (_utcnow() - timedelta(days=self.days_back)).date()
        _prune_stale_totals(self._last_totals, since)
        return since

    def _scan_root(self, root, tree, inputs, scan, fingerprint=_tree_signature):
        """Return scan()'s result for a data root, rerunning it only when needed.

        This is what keeps an idle exporter idle: a poll costs a stat of each file
        under `tree` unless one of them, or `inputs` (whatever else the result
        depends on), has changed. The result must not be reused across a change of
        scan window, or days whose cursor was pruned would be emitted again.

        A scan that returns None is not cached, so it is retried on the next poll.
        A file that fails to read mid-scan without failing the whole scan is
        retried once it changes, or at the next daily window change.
        """
        # Fingerprint first: a write racing the scan then reads as a change next poll.
        signature = fingerprint(tree)
        previous = self._root_scans.get(root)
        if (
            previous
            and signature is not None
            and previous.signature == signature
            and previous.inputs == inputs
        ):
            return previous.data
        data = scan()
        if data is not None and signature is not None:
            self._root_scans[root] = _RootScan(signature, inputs, data)
        return data

    def _finish_scan(self, persist, total_records):
        if persist and self.state_file:
            _save_last_totals(self.state_file, self._last_totals)
        logger.log(
            logging.INFO if total_records else logging.DEBUG,
            "%s: emitted %d records", type(self).__name__, total_records,
        )

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
    """Collects token usage from Claude Code data directories via the ccusage CLI.

    Costs are computed in-process from token deltas via the shared pricing
    map, so they stay correct as rates change without depending on ccusage's
    bundled offline pricing snapshot; ccusage's own cost is used only as a
    fallback for a model the pricing map doesn't yet know.
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
        # AntCC logs leave requestId empty, so ccusage would count every
        # streamed update of a message: skip them rather than miscount.
        retired = [d for d in claude_dirs if _detect_agent(d) == "antcc"]
        for claude_dir in retired:
            logger.warning("Ignoring %s: AntCC is no longer supported", claude_dir)
        super().__init__([d for d in claude_dirs if d not in retired], days_back,
                         on_record, state_file, timezone_name, pricing_rates)
        self.project_resolver = ClaudeProjectResolver()

    def _scan_claude_root(self, claude_dir, since):
        """Scan one root and key its result by project name; None if the scan failed.

        Names are resolved here so that a cached result also spares re-reading the
        logs of a project whose cwd is unknown. Everything the resolver reads is
        part of the root's signature, so it is retried as soon as that can help.
        """
        data = _run_ccusage(claude_dir, since, self.timezone_name)
        if data is None:
            return None
        projects = {}
        for key, entries in data.get("projects", {}).items():
            name = self.project_resolver.resolve(claude_dir, key)
            if name is not None:
                projects.setdefault(name, []).extend(entries)
        return {"projects": projects}

    def scan_history(self, persist=True):
        since = self._scan_since()
        total_records = 0
        collected = {}
        failed_agents = set()
        for claude_dir in dict.fromkeys(self.dirs):
            if not os.path.isdir(claude_dir):
                continue
            projects_dir = os.path.join(claude_dir, "projects")
            if not os.path.isdir(projects_dir):
                logger.debug("Skipping %s: no projects/ directory", claude_dir)
                continue
            agent = _detect_agent(claude_dir)
            data = self._scan_root(
                claude_dir, projects_dir, (since,),
                lambda: self._scan_claude_root(claude_dir, since),
            )
            if data is None:
                failed_agents.add(agent)
                continue
            projects = collected.setdefault(agent, {"projects": {}})["projects"]
            for name, entries in data["projects"].items():
                projects.setdefault(name, []).extend(entries)
        for agent, data in collected.items():
            if agent in failed_agents:
                continue
            total_records += _process_ccusage_data(
                data, agent, self.on_record, self._last_totals,
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
        self._file_caches: dict[str, dict[str, _FileCacheEntry]] = {}

    def scan_history(self, persist=True):
        since = self._scan_since()
        total_records = 0
        collected = {}
        for codex_dir in dict.fromkeys(self.dirs):
            if not os.path.isdir(codex_dir):
                continue
            sessions_dir = os.path.join(codex_dir, "sessions")
            if not os.path.isdir(sessions_dir):
                logger.debug("Skipping %s: no sessions/ directory", codex_dir)
                continue
            agent = _detect_agent(codex_dir)
            cache_key = os.path.abspath(codex_dir)
            file_cache = self._file_caches.setdefault(cache_key, {})
            data = self._scan_root(
                codex_dir, sessions_dir, (since, self.pricing_rates),
                lambda: _scan_antcodex_dir(
                    codex_dir,
                    since,
                    self.timezone_name,
                    self.pricing_rates,
                    file_cache=file_cache,
                ),
            )
            projects = collected.setdefault(agent, {"projects": {}})["projects"]
            for name, entries in data["projects"].items():
                projects.setdefault(name, []).extend(entries)
        for agent, data in collected.items():
            total_records += _process_ccusage_data(
                data, agent, self.on_record, self._last_totals,
                pricing_rates=self.pricing_rates,
            )
        self._finish_scan(persist, total_records)


# Antigravity CLI (agy). Each conversation is a directory under brain/ whose
# transcript logs a token-bearing entry per model step; history.jsonl and
# conversation_summaries.db map conversations (and subagents, via their parent)
# to workspaces.

_AGY_TRANSCRIPT_NAMES = ("transcript.jsonl", "transcript_full.jsonl")
_AGY_HISTORY = "history.jsonl"
_AGY_SUMMARIES_DB = "conversation_summaries.db"
# Journal files that hold committed data not yet written back into the database.
_SQLITE_JOURNAL_SUFFIXES = ("-wal", "-journal")
_AGY_METADATA_FILES = (
    _AGY_HISTORY,
    _AGY_SUMMARIES_DB,
    *(_AGY_SUMMARIES_DB + suffix for suffix in _SQLITE_JOURNAL_SUFFIXES),
)
_STATE_TO_FILE_FIELD = dict(zip(_TOKEN_FIELDS, _FILE_TOTAL_FIELDS))


def _optional_signature(path: str) -> tuple | None:
    """A file's signature, or None if it doesn't exist. Other OSErrors propagate."""
    try:
        return _file_signature(os.stat(path))
    except FileNotFoundError:
        return None


def _list_agy_transcripts(brain_dir: str) -> dict[str, tuple | None]:
    """Map each conversation's transcript to its signature, without opening any.

    transcript.jsonl is used when it is a regular file, else transcript_full.jsonl.
    A transcript that can't be stat'ed maps to None, so it reads as changed once
    it can be. Raises OSError if brain_dir itself can't be listed.
    """
    with os.scandir(brain_dir) as entries:
        conv_dirs = []
        for entry in entries:
            try:
                if not entry.is_dir():
                    continue
            except OSError:
                pass  # Probed below: an unreadable conversation maps to None.
            conv_dirs.append(entry.path)

    transcripts: dict[str, tuple | None] = {}
    for conv_dir in conv_dirs:
        logs_dir = os.path.join(conv_dir, ".system_generated", "logs")
        for name in _AGY_TRANSCRIPT_NAMES:
            path = os.path.join(logs_dir, name)
            try:
                stat_result = os.stat(path)
            except (FileNotFoundError, NotADirectoryError):
                continue
            except OSError:
                transcripts[path] = None
                break
            if stat.S_ISREG(stat_result.st_mode):
                transcripts[path] = _file_signature(stat_result)
                break
    return transcripts


def _agy_root_signature(agy_dir: str) -> dict[str, tuple | None] | None:
    """Fingerprint every file an agy scan reads, without opening any.

    Covers each conversation's transcript and the project metadata, but not the
    rest of brain/ (steps, chunks, scratch files), which no scan reads.
    """
    try:
        signature = _list_agy_transcripts(os.path.join(agy_dir, "brain"))
    except OSError:
        return None
    for name in _AGY_METADATA_FILES:
        path = os.path.join(agy_dir, name)
        try:
            meta_signature = _optional_signature(path)
        except OSError:
            signature[path] = None
        else:
            if meta_signature is not None:
                signature[path] = meta_signature
    return signature


def _non_negative_int(value) -> int:
    """A token count as an int; anything else, NaN and infinities included, is 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    if isinstance(value, float) and not math.isfinite(value):
        return 0
    return max(int(value), 0)


def _agy_step_usage(line: str, tz: ZoneInfo):
    """(step_index or None, local date, model, tokens) of a transcript line, or
    None if the line carries no usable token counts."""
    try:
        obj = json.loads(line)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None

    tokens = {
        "inputTokens": _non_negative_int(obj.get("input_tokens")),
        "outputTokens": _non_negative_int(obj.get("output_tokens")),
        "cacheCreationTokens": _non_negative_int(
            obj.get("cache_creation_tokens") or obj.get("cache_write_tokens")
        ),
        "cacheReadTokens": _non_negative_int(obj.get("cache_read_tokens")),
    }
    if not any(tokens.values()):
        return None

    ts_str = obj.get("created_at")
    if not isinstance(ts_str, str) or not ts_str:
        return None
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        date_str = ts.astimezone(tz).date().isoformat()
    except (ValueError, OverflowError):
        return None

    model = obj.get("model")
    if not isinstance(model, str) or not model:
        model = "unknown"
    step_index = obj.get("step_index")
    if not isinstance(step_index, int) or isinstance(step_index, bool):
        step_index = None
    return step_index, date_str, model, tokens


def _parse_agy_transcript_file(transcript_file: str, since_local: str, tz: ZoneInfo) -> dict | None:
    """Parse one transcript into {conversation_id: {date: {model: tokens}}}.

    None means the file could not be read; an empty dict is a successful parse
    with nothing in the window (and is cacheable). A step logged more than once
    (streaming updates) counts once, with its latest token counts, but on the
    day and model of its first entry: were a later entry to move it, a step
    that straddles midnight would leave its tokens counted on the first day
    and count them again on the next.
    """
    steps: dict[int, tuple[str, str, dict[str, int]]] = {}
    unindexed: list[tuple[str, str, dict[str, int]]] = []
    try:
        # errors="replace": a final line cut mid-character is skipped as bad JSON.
        with open(transcript_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                usage = _agy_step_usage(line, tz)
                if usage is None:
                    continue
                step_index, date_str, model, tokens = usage
                if step_index is None:
                    unindexed.append((date_str, model, tokens))
                elif step_index in steps:
                    first_date, first_model, _ = steps[step_index]
                    steps[step_index] = (first_date, first_model, tokens)
                else:
                    steps[step_index] = (date_str, model, tokens)
    except OSError:
        return None

    totals: dict[str, dict[str, dict[str, int]]] = {}
    for date_str, model, step_tokens in (*steps.values(), *unindexed):
        if date_str < since_local:
            continue
        tokens = totals.setdefault(date_str, {}).setdefault(
            model, {field: 0 for field in _FILE_TOTAL_FIELDS},
        )
        for field in _FILE_TOTAL_FIELDS:
            tokens[field] += step_tokens[field]
    if not totals:
        return {}
    # brain/<conversation id>/.system_generated/logs/<transcript>
    return {Path(transcript_file).parents[2].name: totals}


def _read_agy_history(history_path: str) -> dict[str, str]:
    """Map conversation ids to workspace paths. Raises OSError if unreadable."""
    workspaces: dict[str, str] = {}
    with open(history_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if not isinstance(obj, dict):
                continue
            conv_id = obj.get("conversationId")
            workspace = obj.get("workspace")
            if isinstance(conv_id, str) and conv_id and isinstance(workspace, str) and workspace:
                workspaces.setdefault(conv_id, workspace)
    return workspaces


def _first_workspace_path(raw_uris) -> str:
    if not isinstance(raw_uris, str) or not raw_uris:
        return ""
    try:
        uris = json.loads(raw_uris)
    except ValueError:
        return ""
    if not isinstance(uris, list):
        return ""
    for uri in uris:
        path = path_from_file_uri(uri)
        if path:
            return path
    return ""


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Column names of a table; empty if it doesn't exist."""
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _read_agy_summaries_conn(conn: sqlite3.Connection) -> tuple[dict[str, str], dict[str, str]]:
    """Workspace and parent mappings, each read only if the schema has it, so a
    database without one of them still yields the other."""
    workspaces: dict[str, str] = {}
    parents: dict[str, str] = {}

    columns = _table_columns(conn, "conversation_summaries")
    if {"conversation_id", "workspace_uris"} <= columns:
        for conv_id, raw_uris in conn.execute(
            "SELECT conversation_id, workspace_uris FROM conversation_summaries"
        ):
            path = _first_workspace_path(raw_uris)
            if isinstance(conv_id, str) and conv_id and path:
                workspaces[conv_id] = path
    if {"conversation_id", "parent_conversation_id"} <= columns:
        for conv_id, parent_id in conn.execute(
            "SELECT conversation_id, parent_conversation_id FROM conversation_summaries"
        ):
            if isinstance(conv_id, str) and conv_id and isinstance(parent_id, str) and parent_id:
                parents[conv_id] = parent_id

    if {"conversation_id", "parent_id"} <= _table_columns(conn, "conversation_parents"):
        for conv_id, parent_id in conn.execute(
            "SELECT conversation_id, parent_id FROM conversation_parents"
        ):
            if isinstance(conv_id, str) and conv_id and isinstance(parent_id, str) and parent_id:
                parents.setdefault(conv_id, parent_id)
    return workspaces, parents


def _copy_sqlite_files(db_path: str, dest: str, attempts: int = 3) -> None:
    """Copy a database and its journal files, retrying if one changes mid-copy.

    Raises OSError if no attempt saw the files hold still.
    """
    sources = [db_path + suffix for suffix in ("", *_SQLITE_JOURNAL_SUFFIXES)]
    for _ in range(attempts):
        before = [_optional_signature(path) for path in sources]
        if before[0] is None:
            raise FileNotFoundError(f"{db_path} disappeared")
        try:
            for path, signature in zip(sources, before):
                target = dest + path[len(db_path):]
                if signature is None:
                    if os.path.lexists(target):
                        os.unlink(target)
                else:
                    shutil.copyfile(path, target)
        except FileNotFoundError:
            continue  # A journal file was removed mid-copy: try again.
        if [_optional_signature(path) for path in sources] == before:
            return
    raise OSError(f"{db_path} kept changing while it was copied")


def _read_agy_summaries(db_path: str) -> tuple[dict[str, str], dict[str, str]]:
    """Workspace and parent mappings from conversation_summaries.db.

    Reads a private copy. Opening agy's database in place, even with mode=ro,
    creates -wal/-shm files beside it when its folder is writable (root-owned
    ones agy can't then use, if the exporter runs as root), and fails on a
    read-only mount when they are missing. Raises OSError or sqlite3.Error if
    no consistent copy can be read.
    """
    with tempfile.TemporaryDirectory(prefix="agy-summaries-") as tmp_dir:
        copy = os.path.join(tmp_dir, _AGY_SUMMARIES_DB)
        _copy_sqlite_files(db_path, copy)
        conn = sqlite3.connect(copy)
        try:
            return _read_agy_summaries_conn(conn)
        finally:
            conn.close()


class _AgyMetadata:
    """Project metadata of one agy root, re-read only when its files change."""

    def __init__(self, agy_dir: str):
        self.history_path = os.path.join(agy_dir, _AGY_HISTORY)
        self.db_path = os.path.join(agy_dir, _AGY_SUMMARIES_DB)
        self._history: tuple | None = None  # (signature, workspaces)
        self._summaries: tuple | None = None  # (signatures, (workspaces, parents))

    def load(self) -> tuple[dict[str, str], dict[str, str]]:
        """Return (workspaces, parents) by conversation id.

        A missing file contributes nothing. Raises OSError or sqlite3.Error if a
        file that exists can't be read.
        """
        # Stat before reading: a write racing the read then shows as a change.
        signature = _optional_signature(self.history_path)
        if self._history is None or self._history[0] != signature:
            history = _read_agy_history(self.history_path) if signature else {}
            self._history = (signature, history)

        signatures = tuple(
            _optional_signature(self.db_path + suffix)
            for suffix in ("", *_SQLITE_JOURNAL_SUFFIXES)
        )
        if self._summaries is None or self._summaries[0] != signatures:
            summaries = _read_agy_summaries(self.db_path) if signatures[0] else ({}, {})
            self._summaries = (signatures, summaries)

        db_workspaces, parents = self._summaries[1]
        return {**self._history[1], **db_workspaces}, parents


def _resolve_agy_project(conv_id: str, workspaces: dict[str, str], parents: dict[str, str]) -> str:
    """Project of a conversation, or of its nearest ancestor that has a workspace."""
    cur = conv_id
    visited: set[str] = set()
    while cur and cur not in visited:
        visited.add(cur)
        cwd = workspaces.get(cur)
        if cwd:
            return project_from_cwd(cwd)
        cur = parents.get(cur, "")
    return "unknown"


def _scan_agy_dir(
    agy_dir: str,
    since: date,
    timezone_name: str = "UTC",
    file_cache: dict[str, _FileCacheEntry] | None = None,
    metadata: _AgyMetadata | None = None,
) -> dict:
    """Token totals and project of each conversation in one agy root.

    Returns {"conversations": {id: {date: {model: tokens}}}, "projects": {id: name}}.
    Raises OSError or sqlite3.Error if brain/ or the project metadata can't be
    read, so that the scan is retried instead of cached with its conversations
    labelled unknown. An unreadable transcript is retried once it changes.
    """
    tz = ZoneInfo(timezone_name)
    since_local = since.isoformat()
    transcripts = _list_agy_transcripts(os.path.join(agy_dir, "brain"))
    workspaces, parents = (metadata or _AgyMetadata(agy_dir)).load()
    conversations = _scan_cached_files(
        sorted(transcripts.items()),
        {} if file_cache is None else file_cache,
        since_local,
        lambda path: _parse_agy_transcript_file(path, since_local, tz),
    )
    return {
        "conversations": conversations,
        "projects": {
            conv_id: _resolve_agy_project(conv_id, workspaces, parents)
            for conv_id in conversations
        },
    }


def _emit_conversation_deltas(
    conversations: dict,
    projects: dict[str, str],
    agent: str,
    on_record,
    last_totals: dict,
    pricing_rates: dict | None = None,
) -> int:
    """Emit what each conversation added since the last scan, under its project.

    The cursor is kept per conversation, keyed (agent, project, model, date,
    conversation id), where project is the label those tokens were emitted
    under. A conversation whose project changes (resolved late, or its
    history.jsonl entry trimmed) therefore moves only its new tokens; what was
    already counted stays where it was and is never counted twice. A field's
    cursor only grows, so a transcript that reads short for a poll (caught
    mid-rewrite) re-emits nothing once it reads whole again. A conversation
    that loses its project keeps the last one it was emitted under.

    Records are summed per (project, model, date). Cost is priced on the delta.
    """
    emitted: dict[tuple[str, str, str], dict[str, float]] = {}
    remembered: dict[str, tuple[str, str]] = {}  # conversation -> (date, project)
    for key, values in last_totals.items():
        if len(key) != 5 or key[0] != agent:
            continue
        _, project, model, date_str, conv_id = key
        done = emitted.setdefault((conv_id, model, date_str), dict.fromkeys(_TOKEN_FIELDS, 0))
        for field in _TOKEN_FIELDS:
            done[field] += values.get(field, 0)
        if project != "unknown" and (
            conv_id not in remembered or date_str > remembered[conv_id][0]
        ):
            remembered[conv_id] = (date_str, project)

    records: dict[tuple[str, str, str], dict[str, float]] = {}
    for conv_id, dates in conversations.items():
        project = projects.get(conv_id, "unknown")
        if project == "unknown" and conv_id in remembered:
            project = remembered[conv_id][1]
        for date_str, models in dates.items():
            for model, tokens in models.items():
                done = emitted.get((conv_id, model, date_str), {})
                delta = {
                    field: max(tokens.get(_STATE_TO_FILE_FIELD[field], 0) - done.get(field, 0), 0)
                    for field in _TOKEN_FIELDS
                }
                if not any(delta.values()):
                    continue
                cost = pricing.cost_for(
                    pricing_rates,
                    model,
                    delta["input_tokens"],
                    delta["cache_read_tokens"],
                    delta["output_tokens"],
                    delta["cache_creation_tokens"],
                    effort_suffixes=True,
                ) if pricing_rates else 0.0

                cursor = last_totals.setdefault((agent, project, model, date_str, conv_id), {})
                record = records.setdefault(
                    (project, model, date_str), dict.fromkeys(_STATE_FIELDS, 0),
                )
                for field in _TOKEN_FIELDS:
                    cursor[field] = cursor.get(field, 0) + delta[field]
                    record[field] += delta[field]
                cursor["cost_usd"] = cursor.get("cost_usd", 0) + cost
                record["cost_usd"] += cost

    if on_record:
        for (project, model, date_str), values in sorted(records.items()):
            on_record(agent, {
                "timestamp": datetime.fromisoformat(date_str),
                "model": model,
                "project": project,
                **values,
            })
    return len(records)


class AgyCollector(_BaseCollector):
    """Collects token usage from Antigravity CLI (agy) data directories.

    Parses each conversation's `brain/<id>/.system_generated/logs/transcript.jsonl`
    directly (one count per step, cached per file), names its project from
    `conversation_summaries.db` and `history.jsonl`, and prices token deltas
    via the shared pricing table. Unlike the other collectors, the cursor is
    kept per conversation; see _emit_conversation_deltas.
    """

    AGENT = "agy"

    def __init__(
        self,
        agy_dirs: list[str],
        days_back: int = 7,
        on_record=None,
        state_file: str = "",
        timezone_name: str = "UTC",
        pricing_rates: dict | None = None,
    ):
        super().__init__(
            agy_dirs, days_back, on_record, state_file, timezone_name, pricing_rates
        )
        self._file_caches: dict[str, dict[str, _FileCacheEntry]] = {}
        self._metadata: dict[str, _AgyMetadata] = {}
        self._failing: set[str] = set()

    def _scan_agy_root(self, agy_dir: str, since: date) -> dict | None:
        """_scan_agy_dir's result, or None (logged) if the root can't be read."""
        key = os.path.abspath(agy_dir)
        try:
            data = _scan_agy_dir(
                agy_dir,
                since,
                self.timezone_name,
                file_cache=self._file_caches.setdefault(key, {}),
                metadata=self._metadata.setdefault(key, _AgyMetadata(agy_dir)),
            )
        except (OSError, sqlite3.Error) as e:
            # Warn once per outage; the root is retried every poll.
            logger.log(
                logging.DEBUG if key in self._failing else logging.WARNING,
                "Skipping Antigravity CLI data in %s until it can be read: %s",
                agy_dir, e,
            )
            self._failing.add(key)
            return None
        if key in self._failing:
            self._failing.discard(key)
            logger.info("Antigravity CLI data in %s can be read again", agy_dir)
        return data

    def scan_history(self, persist=True):
        since = self._scan_since()
        conversations: dict = {}
        projects: dict[str, str] = {}
        for agy_dir in dict.fromkeys(self.dirs):
            if not os.path.isdir(os.path.join(agy_dir, "brain")):
                logger.debug("Skipping %s: no brain/ directory", agy_dir)
                continue
            data = self._scan_root(
                agy_dir,
                agy_dir,
                (since,),
                lambda: self._scan_agy_root(agy_dir, since),
                fingerprint=_agy_root_signature,
            )
            if data is None:
                continue
            _merge_file_totals(conversations, data["conversations"])
            for conv_id, project in data["projects"].items():
                if projects.get(conv_id, "unknown") == "unknown":
                    projects[conv_id] = project
        total_records = _emit_conversation_deltas(
            conversations, projects, self.AGENT, self.on_record, self._last_totals,
            pricing_rates=self.pricing_rates,
        )
        self._finish_scan(persist, total_records)
