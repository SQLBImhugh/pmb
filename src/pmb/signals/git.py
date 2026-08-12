"""
Git signals capture.

Captures into memory:
- git commits with message, author, files
- branch changes
- diff summary (not the full diff - too large)

Stored in EventStore with event_type="git".

Tracking:
- last_sync_timestamp is stored in meta (workspace.last_git_sync)
- On each sync call we capture commits since last_sync_timestamp
- Default first sync - the last 7 days

Idempotency:
- Each commit has a stable hash → we use it as part of the metadata
- Before appending we check whether this SHA has already been recorded
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from pmb.core.events import Event
from pmb.security.redact import redact, redact_metadata

# Commits walked in a single sync when the caller does not say otherwise.
# Kept as the default (not a hard limit) so an explicit `--days`/`--max-commits`
# can ask for more; see `capture_recent_commits`.
DEFAULT_MAX_COMMITS = 100

if TYPE_CHECKING:
    from pmb.core.engine import Engine


@dataclass
class CommitInfo:
    sha: str
    short_sha: str
    author: str
    timestamp: float
    subject: str
    body: str
    branch: str | None
    files_changed: list[str]
    files_added: int
    files_modified: int
    files_deleted: int
    insertions: int
    deletions: int


def _run_git(args: list[str], cwd: Path, timeout: int = 10) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        return result.returncode, result.stdout, result.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return -1, "", ""


def _is_git_repo(path: Path) -> bool:
    rc, _, _ = _run_git(["rev-parse", "--is-inside-work-tree"], path, timeout=2)
    return rc == 0


def _current_branch(path: Path) -> str | None:
    rc, out, _ = _run_git(["branch", "--show-current"], path, timeout=2)
    if rc == 0:
        return out.strip() or None
    return None


def _commit_files_stat(path: Path, sha: str) -> dict:
    """Get file statistics for a commit via --shortstat and --name-status."""
    rc, out, _ = _run_git(["show", "--name-status", "--format=", sha], path)
    files_added, files_modified, files_deleted = 0, 0, 0
    files_changed: list[str] = []
    if rc == 0:
        for line in out.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            status, fname = parts[0], parts[1]
            files_changed.append(fname)
            if status.startswith("A"):
                files_added += 1
            elif status.startswith("D"):
                files_deleted += 1
            elif status.startswith("M") or status.startswith("R"):
                files_modified += 1

    rc2, out2, _ = _run_git(["show", "--shortstat", "--format=", sha], path)
    insertions, deletions = 0, 0
    if rc2 == 0 and out2.strip():
        # example: " 3 files changed, 45 insertions(+), 12 deletions(-)"
        for token in out2.strip().split(","):
            token = token.strip()
            if "insertion" in token:
                insertions = int(token.split()[0])
            elif "deletion" in token:
                deletions = int(token.split()[0])

    return {
        "files_changed": files_changed,
        "files_added": files_added,
        "files_modified": files_modified,
        "files_deleted": files_deleted,
        "insertions": insertions,
        "deletions": deletions,
    }


def capture_recent_commits(
    repo_path: Path,
    since_timestamp: float | None = None,
    max_commits: int | None = DEFAULT_MAX_COMMITS,
) -> list[CommitInfo]:
    """Capture commits since the given date (or the last week by default).

    `max_commits` caps how many commits a single walk returns. Pass ``0`` (or
    any value <= 0, or ``None``) for no cap - useful with an explicit
    ``--days`` window, where a silent cap would drop history the caller
    explicitly asked for.
    """
    if not _is_git_repo(repo_path):
        return []

    branch = _current_branch(repo_path)

    if since_timestamp is None:
        # default: 7 days ago
        since_timestamp = time.time() - 7 * 86400

    # Pass unix timestamp directly so git doesn't reinterpret as local time
    since_arg = f"@{int(since_timestamp)}"

    # Format: hash, short_hash, author, unix_ts, subject, body
    fmt = "%H%x09%h%x09%an%x09%at%x09%s%x09%b%x1e"
    log_args = ["log", f"--since={since_arg}"]
    # Omit -n entirely when uncapped; `git log -0` returns nothing.
    if max_commits and max_commits > 0:
        log_args.append(f"-{max_commits}")
    log_args += [f"--pretty=format:{fmt}", "--no-merges"]
    # An uncapped walk over a long window can outlast the default 10s budget.
    rc, out, _ = _run_git(
        log_args,
        repo_path,
        timeout=10 if (max_commits and max_commits > 0) else 120,
    )
    if rc != 0 or not out.strip():
        return []

    commits: list[CommitInfo] = []
    # Records are separated by 0x1e (RS char), fields within by 0x09 (TAB)
    for record in out.split("\x1e"):
        record = record.strip("\n").strip()
        if not record:
            continue
        parts = record.split("\t")
        if len(parts) < 5:
            continue
        sha = parts[0]
        short_sha = parts[1]
        author = parts[2]
        try:
            ts = float(parts[3])
        except ValueError:
            ts = time.time()
        subject = parts[4]
        body = parts[5] if len(parts) > 5 else ""

        stats = _commit_files_stat(repo_path, sha)
        commits.append(CommitInfo(
            sha=sha,
            short_sha=short_sha,
            author=author,
            timestamp=ts,
            subject=subject,
            body=body.strip(),
            branch=branch,
            files_changed=stats["files_changed"],
            files_added=stats["files_added"],
            files_modified=stats["files_modified"],
            files_deleted=stats["files_deleted"],
            insertions=stats["insertions"],
            deletions=stats["deletions"],
        ))
    return commits


class GitSync:
    """
    Synchronize git events into PMB memory.

    Usage:
        sync = GitSync(engine)
        n = sync.sync()  # synchronize new commits
    """

    def __init__(self, engine: Engine):
        self.engine = engine

    def _state_file(self) -> Path:
        return self.engine.workspace.storage_dir / "git_sync.yaml"

    def get_last_sync(self) -> float | None:
        f = self._state_file()
        if not f.exists():
            return None
        try:
            with open(f, encoding="utf-8") as fp:
                data = yaml.safe_load(fp) or {}
            return data.get("last_sync_timestamp")
        except Exception:
            return None

    def set_last_sync(self, ts: float):
        f = self._state_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "w", encoding="utf-8") as fp:
            yaml.safe_dump({"last_sync_timestamp": ts}, fp)

    def already_imported_shas(self) -> set[str]:
        """Get the SHAs of commits that are already in the EventStore."""
        active = self.engine.events.list_active(
            self.engine.workspace.id, limit=10000, event_type="git",
        )
        shas = set()
        for ev in active:
            sha = ev.metadata.get("sha")
            if sha:
                shas.add(sha)
        return shas

    def sync(
        self,
        since_timestamp: float | None = None,
        max_commits: int | None = None,
    ) -> dict:
        """
        Capture git commits since last_sync (or since_timestamp if provided).

        `max_commits` caps how many commits a single run walks. ``None`` uses
        the module default; ``0`` (or negative) means no cap.

        Returns: {"captured": int, "skipped_existing": int, "branch": str,
                  "cap_reached": bool}
        """
        repo = self.engine.workspace.root
        if not _is_git_repo(repo):
            return {"captured": 0, "skipped_existing": 0, "branch": None,
                    "error": "not a git repo"}

        if since_timestamp is None:
            since_timestamp = self.get_last_sync()
            if since_timestamp is not None:
                # Overlap window so commits at the boundary still reach SHA dedup
                since_timestamp = since_timestamp - 3600

        if max_commits is None:
            max_commits = DEFAULT_MAX_COMMITS
        commits = capture_recent_commits(
            repo, since_timestamp=since_timestamp, max_commits=max_commits,
        )
        # A full page means git had at least this many in the window, so there
        # are probably more we did not walk. Surfaced so the caller can say so
        # instead of silently truncating the user's --days window.
        cap_reached = bool(max_commits) and max_commits > 0 and len(commits) >= max_commits
        existing = self.already_imported_shas()

        captured = 0
        skipped = 0
        latest_ts = since_timestamp or 0

        for c in commits:
            if c.sha in existing:
                skipped += 1
                continue

            content = self._format_commit(c)
            metadata = {
                "sha": c.sha,
                "short_sha": c.short_sha,
                "author": c.author,
                "branch": c.branch,
                "subject": c.subject,
                "files_changed": c.files_changed[:20],  # cap at 20
                "n_files_changed": len(c.files_changed),
                "files_added": c.files_added,
                "files_modified": c.files_modified,
                "files_deleted": c.files_deleted,
                "insertions": c.insertions,
                "deletions": c.deletions,
            }
            # Redact secrets that may have leaked into commit messages
            content, _ = redact(content)
            metadata, _ = redact_metadata(metadata)
            # importance proportional to the scope of the change
            importance = self._compute_importance(c)

            ev = Event(
                workspace_id=self.engine.workspace.id,
                event_type="git",
                content=content,
                metadata=metadata,
                timestamp=c.timestamp,
                importance=importance,
            )
            ev = self.engine.events.append(ev)
            self.engine.search.add(ev.ulid, ev.to_text())
            captured += 1
            if c.timestamp > latest_ts:
                latest_ts = c.timestamp

        if captured > 0:
            self.set_last_sync(time.time())

        return {
            "captured": captured,
            "skipped_existing": skipped,
            "branch": _current_branch(repo),
            "latest_commit_timestamp": latest_ts,
            "cap_reached": cap_reached,
            "max_commits": max_commits,
        }

    @staticmethod
    def _format_commit(c: CommitInfo) -> str:
        lines = [
            f"[git commit {c.short_sha}] {c.subject}",
            f"  author: {c.author}",
            f"  branch: {c.branch}",
            f"  files: +{c.files_added} ~{c.files_modified} -{c.files_deleted}, "
            f"diff +{c.insertions}/-{c.deletions}",
        ]
        if c.body:
            lines.append("")
            lines.append(c.body)
        if c.files_changed:
            files_preview = c.files_changed[:8]
            if len(c.files_changed) > 8:
                files_preview.append(f"... and {len(c.files_changed) - 8} more")
            lines.append("")
            lines.append("Files:")
            for f in files_preview:
                lines.append(f"  - {f}")
        return "\n".join(lines)

    @staticmethod
    def _compute_importance(c: CommitInfo) -> float:
        """
        Importance based on the size of the change.

        Tiny (< 10 lines)    → 0.4
        Small (< 100 lines)  → 0.5
        Medium (< 500 lines) → 0.6
        Big (>= 500)         → 0.7
        """
        total_lines = c.insertions + c.deletions
        if total_lines < 10:
            return 0.4
        elif total_lines < 100:
            return 0.5
        elif total_lines < 500:
            return 0.6
        return 0.7
