"""Tests for the removable one-shot script; removable with that script."""

import json
from unittest.mock import patch

import pytest

import app
import migrate_project_names as migration
from tests.test_projects import daily, project_dir
from tests.test_metrics_state import _fresh_metrics
from watcher import CcusageCollector, _load_last_totals


def write_states(root, entries):
    state = str(root / 'state.json')
    cc, codex, counters = migration.state_paths(state)
    last = {'last_totals': {}}
    snapshot = {'schema_version': 1, 'cumulative': [], 'daily': []}
    day = daily(0)[0]['date']
    for agent, project, tokens in entries:
        last['last_totals']['\x00'.join((agent, project, 'fixture', day))] = {
            'input_tokens': tokens, 'output_tokens': 0,
            'cache_read_tokens': 0, 'cache_creation_tokens': 0, 'cost_usd': tokens / 1000,
        }
        snapshot['cumulative'].append({'key': ['host', agent, project, 'fixture'],
                                       'values': {'input': tokens, 'cost': tokens / 1000}})
        snapshot['daily'].append({'key': ['host', agent, project, 'fixture', day],
                                  'values': {'input': tokens, 'cost': tokens / 1000}})
    cc.write_text(json.dumps(last))
    codex.write_text(json.dumps({'last_totals': {}}))
    counters.write_text(json.dumps(snapshot))
    return state, (cc, codex, counters)


def test_migration_preserves_totals_backups_deltas_and_restart(tmp_path):
    root = tmp_path / '.claude'
    a = project_dir(root, '/mnt/bn/home/byte/developer/vefaas-sandbox-infra')
    b = project_dir(root, '/srv/team/another/prefix/vefaas-sandbox-infra')
    entries = [('cc', migration.legacy_project_name(a.name), 100),
               ('cc', migration.legacy_project_name(b.name), 200)]
    state, paths = write_states(tmp_path, entries)
    original = {p: p.read_bytes() for p in paths}
    migration.migrate(state, [str(root)])
    assert all(p.read_bytes() == original[p] for p in paths)
    assert not list(tmp_path.glob('project-names-backup-*'))
    backup = migration.migrate(state, [str(root)], apply=True)
    assert all((backup / p.name).read_bytes() == original[p] for p in paths)
    after = {p: p.read_bytes() for p in paths}
    assert migration.migrate(state, [str(root)], apply=True) is None
    assert all(p.read_bytes() == after[p] for p in paths)
    snapshot = json.loads(paths[2].read_text())
    assert len(snapshot['cumulative']) == len(snapshot['daily']) == 1
    assert snapshot['cumulative'][0]['key'][2] == 'vefaas-sandbox-infra'
    assert snapshot['cumulative'][0]['values']['input'] == 300
    assert snapshot['cumulative'][0]['values']['cost'] == pytest.approx(.3)

    metrics = _fresh_metrics()
    metrics.set_source('host')
    bootstrap = app._bootstrap_from_last_totals([str(paths[0]), str(paths[1])], 'host')
    metrics.restore_state(app._merge_state_with_bootstrap(metrics.load_state(str(paths[2])), bootstrap))
    collector = CcusageCollector([str(root)], state_file=str(paths[0]), on_record=metrics.record_usage)
    data = {'projects': {a.name: daily(100, .1), b.name: daily(200, .2)}}
    with patch('watcher._run_ccusage', return_value=data):
        collector.scan_history()
        collector.check_updates()
        assert metrics.snapshot_state()['cumulative'][0]['values']['input'] == 300
        data['projects'][a.name] = daily(125, .125)
        collector.scan_history()
        collector.check_updates()
        assert metrics.snapshot_state()['cumulative'][0]['values']['input'] == 325
        metrics.save_state(str(paths[2]))
        restarted_metrics = _fresh_metrics()
        restarted_metrics.set_source('host')
        restarted_metrics.restore_state(restarted_metrics.load_state(str(paths[2])))
        restarted = CcusageCollector([str(root)], state_file=str(paths[0]),
                                    on_record=restarted_metrics.record_usage)
        restarted.scan_history()
        assert restarted_metrics.snapshot_state()['cumulative'][0]['values']['input'] == 325


def test_unknown_and_ambiguous_aliases_abort_before_any_write(tmp_path):
    root = tmp_path / '.claude'
    other_root = tmp_path / 'other' / '.claude'
    # Both old labels were 'd'; cwd basenames are different.
    project_dir(root, '/a/b-c/d')
    project_dir(other_root, '/a/b/c-d')
    state, paths = write_states(tmp_path, [('cc', 'd', 100)])
    original = {p: p.read_bytes() for p in paths}
    with pytest.raises(ValueError, match='no unique cwd mapping'):
        migration.migrate(state, [str(root), str(other_root)], apply=True)
    assert all(p.read_bytes() == original[p] for p in paths)
    assert not list(tmp_path.glob('project-names-backup-*'))
    migration.migrate(state, [str(root), str(other_root)], apply=True, overrides={'cc': {'d': 'c-d'}})
    assert json.loads(paths[2].read_text())['cumulative'][0]['key'][2] == 'c-d'


def test_missing_history_can_be_mapped_explicitly_and_codex_is_unchanged(tmp_path):
    state, paths = write_states(tmp_path, [('cc', 'old-project-label', 100), ('codex', 'my-project', 200)])
    with pytest.raises(ValueError, match='no unique cwd mapping'):
        migration.migrate(state, [], apply=True)
    migration.migrate(state, [], apply=True, overrides={'cc': {'old-project-label': 'my-project'}})
    rows = json.loads(paths[2].read_text())['cumulative']
    assert {tuple(row['key']): row['values']['input'] for row in rows} == {
        ('host', 'cc', 'my-project', 'fixture'): 100,
        ('host', 'codex', 'my-project', 'fixture'): 200,
    }


def test_write_failure_restores_every_original_state_file(tmp_path):
    root = tmp_path / '.claude'
    p = project_dir(root, '/a/b/c/my-project')
    state, paths = write_states(tmp_path, [('cc', migration.legacy_project_name(p.name), 100)])
    original = {p: p.read_bytes() for p in paths}
    real_replace = migration.atomic_replace
    calls = 0

    def fail_second(path, content):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('simulated disk error')
        real_replace(path, content)

    with patch.object(migration, 'atomic_replace', side_effect=fail_second):
        with pytest.raises(OSError, match='simulated disk error'):
            migration.migrate(state, [str(root)], apply=True)
    assert all(p.read_bytes() == original[p] for p in paths)


def test_old_state_is_rejected_without_running_a_migration_implicitly(tmp_path):
    state, paths = write_states(tmp_path, [('cc', 'legacy-project', 100)])
    with pytest.raises(ValueError, match='standalone'):
        _load_last_totals(str(paths[0]))
    metrics = _fresh_metrics()
    with pytest.raises(ValueError, match='standalone'):
        metrics.load_state(str(paths[2]))
    assert 'project_name_format' not in json.loads(paths[0].read_text())
