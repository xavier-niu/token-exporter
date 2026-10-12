"""Project display names from agent metadata, independent of checkout location."""

import json
import logging
import ntpath
import os
from pathlib import Path

logger = logging.getLogger("token-stats")
PROJECT_NAME_FORMAT = "cwd-basename-v1"


def project_from_cwd(cwd: str) -> str:
    if not isinstance(cwd, str) or not cwd:
        return "unknown"
    # Imported Windows sessions can be read by a Linux exporter too.
    path = ntpath if ntpath.splitdrive(cwd)[0] or "\\" in cwd else os.path
    return path.basename(path.normpath(cwd)) or "unknown"


def _project_cwd(project_dir: Path) -> str | None:
    try:
        index = json.loads((project_dir / "sessions-index.json").read_text())
        cwd = index.get("originalPath") if isinstance(index, dict) else None
        if isinstance(cwd, str) and cwd:
            return cwd
    except (OSError, ValueError):
        pass

    # Subagents may run in a different directory. Use the parent session's
    # first cwd, not a nested subagent log or its later working directory.
    try:
        session_files = sorted(project_dir.glob("*.jsonl"))
    except OSError:
        return None
    for session_file in session_files:
        try:
            with session_file.open() as session:
                for line in session:
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(record, dict) or record.get("isSidechain"):
                        continue
                    cwd = record.get("cwd")
                    if isinstance(cwd, str) and cwd:
                        return cwd
        except (OSError, UnicodeError):
            continue
    return None


class ClaudeProjectResolver:
    def __init__(self):
        self._names: dict[Path, str] = {}
        self._warned: set[Path] = set()

    def resolve(self, config_dir: str, project_key: str) -> str | None:
        # ccusage's keys are directory names, not reversible path encodings:
        # a dash can stand for either a slash or a literal dash in a name.
        if project_key in ("", ".", "..") or Path(project_key).name != project_key:
            raise ValueError(f"Invalid Claude project directory name: {project_key!r}")
        project_dir = Path(config_dir) / "projects" / project_key
        if project_dir in self._names:
            return self._names[project_dir]
        cwd = _project_cwd(project_dir)
        if cwd is not None:
            name = project_from_cwd(cwd)
            self._names[project_dir] = name
            return name
        if project_dir not in self._warned:
            logger.warning("Skipping %s until cwd metadata is available", project_dir)
            self._warned.add(project_dir)
        # Retry unresolved directories on subsequent scans: the first session
        # may not have been written yet. Never guess by splitting on dashes.
        return None


def check_project_name_format(data: dict) -> None:
    """Reject incompatible persisted labels instead of recounting old usage."""
    if (data.get("last_totals") or data.get("cumulative") or data.get("daily")) and (
        data.get("project_name_format") != PROJECT_NAME_FORMAT
    ):
        raise ValueError("Saved project names use an older format; run the standalone "
                         "migrate_project_names.py script before starting this exporter")
