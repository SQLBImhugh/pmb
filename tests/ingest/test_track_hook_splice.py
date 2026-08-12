"""Regression tests for post-commit hook splicing (`pmb track install`).

The bug: the block was appended to the end of an existing hook. A hook that
ends in an unconditional top-level ``exit 0`` - the standard way to make a
hook fail-open - left the appended block as unreachable dead code, while the
CLI still reported "Appended PMB track to your existing post-commit hook".

`_splice_block` must place the block above such a terminator, while leaving
conditional exits (guards, indented exits inside blocks) alone.
"""
from __future__ import annotations

from pmb.cli.commands.track import _HOOK_END, _HOOK_MARKER, _splice_block

BLOCK = f"{_HOOK_MARKER}\npmb track changes --backend auto >/dev/null 2>&1 &\n{_HOOK_END}\n"


def _block_line_index(text: str) -> int:
    return text.splitlines().index(_HOOK_MARKER)


def _exit_line_indexes(text: str) -> list[int]:
    return [i for i, ln in enumerate(text.splitlines()) if ln.strip() == "exit 0"]


def test_block_is_spliced_above_a_trailing_exit():
    existing = "#!/bin/sh\necho hello\n\nexit 0\n"
    out, moved = _splice_block(existing, BLOCK)

    assert moved is True
    assert _block_line_index(out) < _exit_line_indexes(out)[0], (
        "block must precede `exit 0` or it can never run"
    )
    assert "echo hello" in out


def test_appends_at_end_when_there_is_no_terminator():
    existing = "#!/bin/sh\necho hello\n"
    out, moved = _splice_block(existing, BLOCK)

    assert moved is False
    assert out.rstrip().endswith(_HOOK_END)
    assert "echo hello" in out


def test_guard_style_conditional_exit_is_not_a_terminator():
    # `[ -z "$ROOT" ] && exit 0` only exits sometimes; the rest of the script
    # still runs, so appending after it is legitimate.
    existing = '#!/bin/sh\nROOT=$(git rev-parse --show-toplevel)\n[ -z "$ROOT" ] && exit 0\necho work\n'
    out, moved = _splice_block(existing, BLOCK)

    assert moved is False
    assert out.rstrip().endswith(_HOOK_END)


def test_indented_exit_inside_a_block_is_not_a_terminator():
    existing = '#!/bin/sh\nif [ -f x ]; then\n    exit 0\nfi\necho work\n'
    out, moved = _splice_block(existing, BLOCK)

    assert moved is False
    assert "echo work" in out


def test_real_world_hook_keeps_prior_work_before_the_block():
    """Mirrors a hook seen in the wild: guard, work, then a final exit 0."""
    existing = (
        "#!/bin/sh\n"
        "ROOT=$(git rev-parse --show-toplevel 2>/dev/null)\n"
        '[ -z "$ROOT" ] && exit 0\n'
        "\n"
        'printf "heartbeat\\n" > "$ROOT/.git/hook.log"\n'
        "pwsh -NoProfile -Command 'Start-Process something' >/dev/null 2>&1\n"
        "\n"
        "exit 0\n"
    )
    out, moved = _splice_block(existing, BLOCK)

    assert moved is True
    lines = out.splitlines()
    heartbeat = next(i for i, ln in enumerate(lines) if "heartbeat" in ln)
    marker = lines.index(_HOOK_MARKER)
    final_exit = max(i for i, ln in enumerate(lines) if ln.strip() == "exit 0")

    assert heartbeat < marker < final_exit
    # The guard must stay where it was, above our block.
    guard = next(i for i, ln in enumerate(lines) if ln.startswith('[ -z "$ROOT" ]'))
    assert guard < marker


def test_exec_is_treated_as_a_terminator():
    existing = "#!/bin/sh\necho hello\nexec some-daemon\n"
    out, moved = _splice_block(existing, BLOCK)

    assert moved is True
    lines = out.splitlines()
    assert lines.index(_HOOK_MARKER) < next(
        i for i, ln in enumerate(lines) if ln.startswith("exec ")
    )


def test_bare_exit_without_status_is_a_terminator():
    existing = "#!/bin/sh\necho hello\nexit\n"
    out, moved = _splice_block(existing, BLOCK)

    assert moved is True
    lines = out.splitlines()
    assert lines.index(_HOOK_MARKER) < lines.index("exit")


def test_commented_exit_is_not_a_terminator():
    existing = "#!/bin/sh\necho hello\n# exit 0 (disabled)\n"
    out, moved = _splice_block(existing, BLOCK)

    assert moved is False


def test_splice_result_is_valid_for_reinstall_detection():
    # After splicing, the marker must still be findable so `install` stays
    # idempotent on a second run.
    existing = "#!/bin/sh\necho hello\nexit 0\n"
    out, _ = _splice_block(existing, BLOCK)
    assert _HOOK_MARKER in out and _HOOK_END in out
