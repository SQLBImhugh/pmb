"""Regression tests for git commit capture limits.

Two behaviours are pinned here:

  * `capture_recent_commits` must honour an explicit `max_commits`, including
    ``0`` meaning "no cap". Before the fix, `GitSync.sync` never forwarded the
    argument, so a wide ``--days`` window was silently truncated at 100.
  * `GitSync.sync` must report `cap_reached` so the CLI can say the window was
    truncated instead of reporting an unqualified success.

Uses a real local repo (no network) and skips cleanly without git.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from pmb.signals.git import DEFAULT_MAX_COMMITS, capture_recent_commits

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git not installed"
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args],
                   check=True, capture_output=True, text=True)


@pytest.fixture
def repo_with_commits():
    """A repo with 12 commits, so a cap of 5 is provably a truncation."""
    with tempfile.TemporaryDirectory() as t:
        repo = Path(t) / "repo"
        repo.mkdir(parents=True)
        subprocess.run(["git", "init", str(repo)], check=True,
                       capture_output=True, text=True)
        _git(repo, "config", "user.email", "ci@pmb.test")
        _git(repo, "config", "user.name", "PMB CI")
        for i in range(12):
            (repo / f"f{i}.txt").write_text(f"content {i}\n", encoding="utf-8")
            _git(repo, "add", "-A")
            _git(repo, "commit", "-m", f"commit number {i}")
        yield repo


def test_default_cap_is_exposed_as_a_constant():
    # The CLI documents "default 100"; keep the two from drifting apart.
    assert DEFAULT_MAX_COMMITS == 100


def test_max_commits_caps_the_walk(repo_with_commits):
    since = time.time() - 86400
    commits = capture_recent_commits(repo_with_commits, since_timestamp=since,
                                     max_commits=5)
    assert len(commits) == 5


def test_max_commits_zero_means_no_cap(repo_with_commits):
    since = time.time() - 86400
    commits = capture_recent_commits(repo_with_commits, since_timestamp=since,
                                     max_commits=0)
    # 0 must not be passed through as `git log -0`, which returns nothing.
    assert len(commits) == 12


def test_max_commits_none_means_no_cap(repo_with_commits):
    since = time.time() - 86400
    commits = capture_recent_commits(repo_with_commits, since_timestamp=since,
                                     max_commits=None)
    assert len(commits) == 12


def test_cap_above_available_returns_everything(repo_with_commits):
    since = time.time() - 86400
    commits = capture_recent_commits(repo_with_commits, since_timestamp=since,
                                     max_commits=500)
    assert len(commits) == 12


class _FakeEvents:
    def __init__(self):
        self.appended = []

    def list_active(self, *a, **k):
        return []

    def append(self, ev):
        self.appended.append(ev)
        return ev


class _FakeSearch:
    def add(self, *a, **k):
        return None


class _FakeWorkspace:
    def __init__(self, root: Path, storage: Path):
        self.root = root
        self.storage_dir = storage
        self.id = "test-ws"


class _FakeEngine:
    def __init__(self, root: Path, storage: Path):
        self.workspace = _FakeWorkspace(root, storage)
        self.events = _FakeEvents()
        self.search = _FakeSearch()


def test_sync_forwards_max_commits_and_flags_the_cap(repo_with_commits, tmp_path):
    from pmb.signals.git import GitSync

    eng = _FakeEngine(repo_with_commits, tmp_path)
    res = GitSync(eng).sync(since_timestamp=time.time() - 86400, max_commits=5)

    # The regression: sync used to drop max_commits on the floor.
    assert res["captured"] == 5
    assert res["cap_reached"] is True
    assert res["max_commits"] == 5


def test_sync_uncapped_captures_every_commit(repo_with_commits, tmp_path):
    from pmb.signals.git import GitSync

    eng = _FakeEngine(repo_with_commits, tmp_path)
    res = GitSync(eng).sync(since_timestamp=time.time() - 86400, max_commits=0)

    assert res["captured"] == 12
    assert res["cap_reached"] is False
