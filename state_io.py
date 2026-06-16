"""Durable JSON writes shared by the metrics, watcher, and pricing state files.

A single helper so every on-disk JSON writer gets the same crash-safety:
atomic replace plus fsync, so an ungraceful shutdown can't leave a truncated or
empty file behind (which would otherwise wipe dedup/counter state and re-count
the whole scan window on the next start).
"""

import json
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger("token-stats")


def _fsync_dir(path: Path) -> None:
    """fsync a directory so a rename into it survives an unclean shutdown."""
    try:
        dir_fd = os.open(path, os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def atomic_write_json(path: str, obj) -> bool:
    """Write obj as JSON to path durably; return True on success.

    Writes to a per-writer temp file, fsyncs it, atomically renames it onto the
    destination, then fsyncs the directory. The temp name includes the pid and
    thread id so concurrent writers (e.g. the poll-loop timer thread and the
    SIGTERM handler on the main thread) never share a temp file and truncate one
    another's in-progress data.
    """
    state_path = Path(path)
    tmp_path = state_path.with_name(
        f"{state_path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w") as f:
            json.dump(obj, f, separators=(",", ":"), sort_keys=True)
            # Flush before the rename. os.replace is atomic for the name, but
            # without fsync the data can still be in the page cache; a power loss
            # then leaves an empty/corrupt file.
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, state_path)
        _fsync_dir(state_path.parent)
        return True
    except OSError as e:
        logger.error("Could not write state to %s: %s", path, e)
        # Don't leave a partial tmp behind when the write failed before replace.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return False
