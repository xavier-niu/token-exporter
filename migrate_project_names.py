#!/usr/bin/env python3
"""One-shot, offline migration of exporter state to cwd basenames.

Preview by default. Stop the old exporter before --apply, then start the new
image. This script is never called by the application or service installers.
It can be removed after migration. Prometheus history is not modified.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

from config import load_config
from projects import ClaudeProjectResolver, PROJECT_NAME_FORMAT, project_from_cwd
from watcher import _detect_agent


def legacy_project_name(project_key):
    """The retired naming algorithm, used only to identify old state keys."""
    stripped = project_key.lstrip('-')
    parts = stripped.split('-')
    return ('-'.join(parts[3:]) or stripped) if len(parts) >= 3 else stripped


def discover_aliases(claude_dirs):
    aliases = defaultdict(set)
    resolver = ClaudeProjectResolver()
    for directory in dict.fromkeys(claude_dirs):
        agent = _detect_agent(directory)
        for project_dir in sorted((Path(directory) / 'projects').glob('*')):
            if not project_dir.is_dir():
                continue
            name = resolver.resolve(directory, project_dir.name)
            if name is not None:
                aliases[agent, legacy_project_name(project_dir.name)].add(name)
    return aliases


def migrate_state(data, aliases, overrides=None):
    format_name = data.get('project_name_format')
    if format_name == PROJECT_NAME_FORMAT:
        return data, {}
    if format_name is not None:
        raise ValueError(f'Unsupported project name format: {format_name!r}')
    changes = {}

    def rename(agent, project):
        if agent not in ('cc', 'antcc'):
            return project
        explicit = (overrides or {}).get(agent, {}).get(project)
        candidates = {explicit} if explicit is not None else aliases.get((agent, project), set())
        if len(candidates) != 1:
            raise ValueError(f'{agent}/{project}: no unique cwd mapping '
                             f'({sorted(candidates)}); supply --project-map')
        name = next(iter(candidates))
        if not isinstance(name, str) or not name or project_from_cwd(name) != name:
            raise ValueError(f'Expected a directory basename for {agent}/{project}: {name!r}')
        changes[agent, project] = name
        return name

    def add(rows, key, values):
        row = rows.setdefault(tuple(key), {})
        for field, value in values.items():
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f'Invalid counter value: {field}={value!r}')
            row[field] = row.get(field, 0) + value

    result = dict(data, project_name_format=PROJECT_NAME_FORMAT)
    if 'last_totals' in data:
        rows = {}
        unknown_cost = set()
        for encoded, values in data['last_totals'].items():
            key = encoded.split('\x00')
            if len(key) != 4:
                raise ValueError(f'Malformed collector key: {key!r}')
            key[1] = rename(key[0], key[1])
            add(rows, key, values)
            if 'cost_usd' not in values:
                unknown_cost.add(tuple(key))
        for key in unknown_cost:
            rows[key].pop('cost_usd', None)
        result['last_totals'] = {'\x00'.join(key): values for key, values in rows.items()}
    for section, length in (('cumulative', 4), ('daily', 5)):
        if section not in data:
            continue
        rows = {}
        for entry in data[section]:
            key = list(entry['key'])
            if len(key) != length:
                raise ValueError(f'Malformed {section} key: {key!r}')
            key[2] = rename(key[1], key[2])
            add(rows, key, entry['values'])
        result[section] = [{'key': list(key), 'values': values} for key, values in rows.items()]
    return result, changes


def state_paths(state_file):
    if not state_file or not state_file.endswith('.json'):
        raise ValueError('--state-file must end in .json, matching the exporter configuration')
    return [Path(state_file.replace('.json', suffix)) for suffix in (
        '-ccusage.json', '-codex.json', '-counters.json')]


def atomic_replace(path, content):
    original_stat = path.stat()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            f.write(content)
            f.flush()
            os.fchmod(f.fileno(), original_stat.st_mode & 0o777)
            if os.geteuid() == 0:
                os.fchown(f.fileno(), original_stat.st_uid, original_stat.st_gid)
            os.fsync(f.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def migrate(state_file, claude_dirs, apply=False, overrides=None):
    aliases = discover_aliases(claude_dirs)
    plans = []
    mappings = {}
    for path in state_paths(state_file):
        if not path.exists():
            continue
        original = path.read_bytes()
        data = json.loads(original)
        updated, changes = migrate_state(data, aliases, overrides)
        mappings.update(changes)
        if updated != data:
            plans.append((path, original, (json.dumps(updated, separators=(',', ':')) + '\n').encode()))
    for (agent, old), new in sorted(mappings.items()):
        print(f'{agent}: {old} -> {new}')
    if not plans:
        print('No state changes needed (already migrated or no saved state).')
        return None
    print(f'{len(plans)} state files to migrate; token and cost totals are preserved.')
    if not apply:
        print('Preview only. Stop the exporter, then repeat with --apply.')
        return None

    # Validate the complete plan before touching any state. A stopped exporter
    # is required; a detected concurrent write aborts without overwriting it.
    for path, original, _ in plans:
        if path.read_bytes() != original:
            raise ValueError(f'{path} changed during the preview; stop the exporter first')
    parent = Path(state_file).parent
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    backup = parent / f'project-names-backup-{stamp}'
    backup.mkdir(mode=0o700)
    manifest = {'project_name_format': PROJECT_NAME_FORMAT, 'files': {},
                'aliases': [{'agent': a, 'old': old, 'new': new}
                            for (a, old), new in sorted(mappings.items())]}
    for path, original, updated in plans:
        with (backup / path.name).open('xb') as f:
            f.write(original)
            f.flush()
            os.fsync(f.fileno())
        manifest['files'][path.name] = {
            'path': str(path.absolute()),
            'before_sha256': hashlib.sha256(original).hexdigest(),
            'after_sha256': hashlib.sha256(updated).hexdigest(),
        }
    with (backup / 'manifest.json').open('x') as f:
        json.dump(manifest, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    for directory in (backup, parent):
        directory_fd = os.open(directory, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    written = []
    try:
        for path, original, updated in plans:
            # Include this path before replacing: even an fsync failure after
            # the rename must restore the original from the backup.
            written.append((path, original))
            atomic_replace(path, updated)
    except BaseException:
        for path, original in reversed(written):
            atomic_replace(path, original)
        raise
    print(f'Migration complete. Backup: {backup}')
    print('Start the updated exporter; do not restart the old image with this state.')
    return backup


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-file', default=config['state_file'])
    parser.add_argument('--claude-dir', action='append', help='Repeat for every monitored config directory')
    parser.add_argument('--project-map', type=Path, help='Explicit JSON mappings: {"cc": {"old-label": "basename"}}')
    parser.add_argument('--apply', action='store_true', help='Write the migration; exporter must be stopped')
    args = parser.parse_args()
    overrides = json.loads(args.project_map.read_text()) if args.project_map else None
    try:
        migrate(args.state_file, args.claude_dir or config['claude_dirs'], args.apply, overrides)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Migration aborted: {error}\n')


if __name__ == '__main__':
    main()
