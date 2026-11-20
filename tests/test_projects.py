import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from projects import ClaudeProjectResolver, path_from_file_uri, project_from_cwd
from watcher import AgyCollector, CcusageCollector, CodexCollector


def project_dir(root, cwd, key=None):
    path = root / 'projects' / (key or cwd.replace('/', '-'))
    path.mkdir(parents=True, exist_ok=True)
    (path / 'session.jsonl').write_text(json.dumps({'type': 'user', 'cwd': cwd}) + '\n')
    return path


def log_activity(project):
    """Append to a session; collectors only rescan when a root's files change."""
    with (project / 'session.jsonl').open('a') as session:
        session.write('{}\n')


def daily(tokens, cost=0):
    return [{'date': datetime.now(timezone.utc).date().isoformat(), 'modelBreakdowns': [{
        'modelName': 'fixture', 'inputTokens': tokens, 'outputTokens': 0,
        'cacheCreationTokens': 0, 'cacheReadTokens': 0, 'cost': cost,
    }]}]


@pytest.mark.parametrize('cwd', [
    '/home/byte/developer/vefaas-sandbox-infra',
    '/mnt/bn/home/byte/developer/vefaas-sandbox-infra',
    '/tmp/vefaas-sandbox-infra',
    '/vefaas-sandbox-infra',
    '/Users/first-last/Projects/nested/vefaas-sandbox-infra/',
])
def test_claude_uses_the_same_basename_as_codex_regardless_of_prefix(tmp_path, cwd):
    root = tmp_path / '.claude'
    directory = project_dir(root, cwd)
    assert ClaudeProjectResolver().resolve(str(root), directory.name) == project_from_cwd(cwd)
    assert project_from_cwd(cwd) == 'vefaas-sandbox-infra'


def test_original_metadata_preserves_characters_lost_in_encoded_key(tmp_path):
    root = tmp_path / '.claude'
    directory = project_dir(root, '/work/acme/my-repo.v2_tools', '-work-acme-my-repo-v2-tools')
    assert ClaudeProjectResolver().resolve(str(root), directory.name) == 'my-repo.v2_tools'


def test_resolver_uses_index_and_caches_resolved_names(tmp_path):
    root = tmp_path / '.claude'
    directory = root / 'projects' / '-arbitrary-key'
    directory.mkdir(parents=True)
    index = directory / 'sessions-index.json'
    index.write_text(json.dumps({'originalPath': '/scratch/team/my-project'}))
    resolver = ClaudeProjectResolver()
    assert resolver.resolve(str(root), directory.name) == 'my-project'
    index.unlink()
    assert resolver.resolve(str(root), directory.name) == 'my-project'


def test_missing_or_malformed_metadata_is_retried_without_guessing(tmp_path):
    root = tmp_path / '.claude'
    directory = root / 'projects' / '-some-prefix-with-dashes-project-name'
    directory.mkdir(parents=True)
    (directory / 'sessions-index.json').write_text('not json')
    session = directory / 'session.jsonl'
    session.write_text('broken\nnull\n[]\n' + json.dumps({'cwd': 12}) + '\n')
    resolver = ClaudeProjectResolver()
    assert resolver.resolve(str(root), directory.name) is None
    with session.open('a') as f:
        f.write(json.dumps({'cwd': '/anything/project-name'}) + '\n')
    assert resolver.resolve(str(root), directory.name) == 'project-name'


def test_subagent_or_later_cwd_does_not_change_project_name(tmp_path):
    root = tmp_path / '.claude'
    directory = project_dir(root, '/work/parent-project')
    (directory / 'session.jsonl').write_text('\n'.join(json.dumps(row) for row in [
        {'cwd': '/work/subagent-project', 'isSidechain': True},
        {'cwd': '/work/parent-project'},
        {'cwd': '/work/parent-project/subdirectory'},
    ]) + '\n')
    assert ClaudeProjectResolver().resolve(str(root), directory.name) == 'parent-project'


@pytest.mark.parametrize('key', ['', '.', '..', '../outside', '/absolute'])
def test_project_directory_cannot_escape_config_root(tmp_path, key):
    with pytest.raises(ValueError):
        ClaudeProjectResolver().resolve(str(tmp_path), key)


def test_windows_cwd_uses_same_shared_basename_rule():
    assert project_from_cwd('C:\\Users\\byte\\work\\my-project\\') == 'my-project'


@pytest.mark.parametrize('uri,expected_path,expected_project', [
    ('file:///home/nxw/developer/token-exporter', '/home/nxw/developer/token-exporter', 'token-exporter'),
    ('file:///home/nxw/my%20repo/', '/home/nxw/my repo/', 'my repo'),
    ('file:///C:/Users/byte/work/my-project', 'C:/Users/byte/work/my-project', 'my-project'),
    ('/plain/path/demo-tool', '/plain/path/demo-tool', 'demo-tool'),
    ('', '', 'unknown'),
])
def test_file_uri_uses_same_basename_rule(uri, expected_path, expected_project):
    assert path_from_file_uri(uri) == expected_path
    assert project_from_cwd(path_from_file_uri(uri)) == expected_project


def test_multiple_checkouts_and_config_roots_are_summed_before_deltas(tmp_path):
    first = tmp_path / 'one' / '.claude'
    second = tmp_path / 'two' / '.claude'
    a = project_dir(first, '/home/byte/developer/vefaas-sandbox-infra')
    b = project_dir(first, '/scratch/teams/backend/vefaas-sandbox-infra')
    c = project_dir(second, '/a/b/c/d/e/vefaas-sandbox-infra')
    data = {
        str(first): {'projects': {a.name: daily(100, .1), b.name: daily(200, .2)}},
        str(second): {'projects': {c.name: daily(300, .3)}},
    }
    state = tmp_path / 'collector.json'
    records = []
    collector = CcusageCollector([str(first), str(second)], state_file=str(state),
                                on_record=lambda _, record: records.append(record))
    with patch('watcher._run_ccusage', side_effect=lambda root, *_: data[root]):
        collector.scan_history()
        collector.check_updates()
        assert len(records) == 1
        assert records[0]['project'] == 'vefaas-sandbox-infra'
        assert records[0]['input_tokens'] == 600
        assert records[0]['cost_usd'] == pytest.approx(.6)
        restored = CcusageCollector([str(first), str(second)], state_file=str(state),
                                   on_record=lambda _, record: records.append(record))
        restored.scan_history()
        assert len(records) == 1
        data[str(second)]['projects'][c.name] = daily(325, .325)
        log_activity(c)
        restored.check_updates()
        restored.check_updates()
        assert [record['input_tokens'] for record in records] == [600, 25]
        assert records[1]['cost_usd'] == pytest.approx(.025)
        # A failed root must not lower the total and replay usage on recovery.
        log_activity(c)
        with patch('watcher._run_ccusage', side_effect=lambda root, *_: data[root] if root == str(first) else None):
            restored.check_updates()
        restored.check_updates()
        assert len(records) == 2


def test_antcc_also_uses_cwd_basename(tmp_path):
    root = tmp_path / '.codefuse' / 'engine' / 'cc'
    cwd = '/an/arbitrary/deep/path/my-project'
    directory = project_dir(root, cwd)
    (directory / 'session.jsonl').write_text(json.dumps({
        'cwd': cwd, 'type': 'assistant', 'timestamp': datetime.now(timezone.utc).isoformat(),
        'message': {'id': 'msg-1', 'model': 'fixture', 'usage': {'input_tokens': 123}},
    }) + '\n')
    records = []
    collector = CcusageCollector([str(root)], on_record=lambda agent, record: records.append((agent, record)))
    collector.scan_history()
    collector.check_updates()
    assert len(records) == 1
    assert records[0][0] == 'antcc'
    assert records[0][1]['project'] == 'my-project'
    assert records[0][1]['input_tokens'] == 123


def test_codex_basename_is_not_encoded_and_truncated_again(tmp_path):
    root = tmp_path / '.codex'
    session = root / 'sessions' / 'session.jsonl'
    session.parent.mkdir(parents=True)
    session.write_text('\n'.join(json.dumps(row) for row in [
        {'type': 'session_meta', 'payload': {'cwd': '/tmp/one-two-three-four-five'}},
        {'type': 'turn_context', 'payload': {'model': 'fixture'}},
        {'type': 'event_msg', 'timestamp': datetime.now(timezone.utc).isoformat(),
         'payload': {'type': 'token_count', 'info': {'total_token_usage': {'input_tokens': 100}}}},
    ]) + '\n')
    records = []
    CodexCollector([str(root)], on_record=lambda _, record: records.append(record)).scan_history()
    assert records[0]['project'] == 'one-two-three-four-five'


def test_agy_uses_shared_cwd_basename_rule(tmp_path):
    root = tmp_path / '.gemini' / 'antigravity-cli'
    transcript = root / 'brain' / 'conv-1' / '.system_generated' / 'logs' / 'transcript.jsonl'
    transcript.parent.mkdir(parents=True)
    transcript.write_text(json.dumps({
        'step_index': 1, 'type': 'PLANNER_RESPONSE', 'status': 'DONE',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'model': 'gemini-3-pro-high', 'input_tokens': 123, 'output_tokens': 45,
    }) + '\n')
    (root / 'history.jsonl').write_text(json.dumps({
        'conversationId': 'conv-1',
        'workspace': '/mnt/custom/team/my-repo.v2_tools',
    }) + '\n')
    records = []
    AgyCollector([str(root)], on_record=lambda agent, record: records.append((agent, record))).scan_history()
    assert len(records) == 1
    assert records[0][0] == 'agy'
    assert records[0][1]['project'] == 'my-repo.v2_tools'
    assert records[0][1]['input_tokens'] == 123


@pytest.mark.skipif(not shutil.which('ccusage'), reason='ccusage CLI not installed')
def test_real_ccusage_output_resolves_an_arbitrary_prefix(tmp_path):
    root = tmp_path / '.claude'
    cwd = '/mnt/custom/team-space/another/checkout/my-project.v2_tools'
    directory = project_dir(root, cwd, '-mnt-custom-team-space-another-checkout-my-project-v2-tools')
    fixture = Path(__file__).parent / 'fixtures/.claude/projects/-home-user-developer-demo-tool/session-bbbb.jsonl'
    rows = [json.loads(line) for line in fixture.read_text().splitlines()]
    for row in rows:
        row['cwd'] = cwd
    # Match Claude's compact JSONL: ccusage's fast scanner filters lines before
    # JSON parsing and does not recognize pretty-spaced synthetic records.
    (directory / 'session.jsonl').write_text(
        '\n'.join(json.dumps(row, separators=(',', ':')) for row in rows) + '\n')
    records = []
    collector = CcusageCollector([str(root)], days_back=100_000,
                                on_record=lambda _, record: records.append(record))
    collector.scan_history()
    collector.check_updates()
    assert len(records) == 1
    assert records[0]['project'] == 'my-project.v2_tools'
    assert records[0]['input_tokens'] == 1500
