#!/usr/bin/env python3
"""Run Ruff against tracked Python files.

This utility keeps tox lint commands focused on tracked sources and avoids
temporary/untracked files that can appear during local experimentation.
"""

from __future__ import annotations

import subprocess
import sys


def _tracked_python_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "*.py", "*.pyi"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: run_ruff_on_tracked.py <check|format-check>", file=sys.stderr)
        return 2

    mode = sys.argv[1]
    if mode == "check":
        command = ["ruff", "check"]
    elif mode == "format-check":
        command = ["ruff", "format", "--check"]
    else:
        print(f"unsupported mode: {mode}", file=sys.stderr)
        return 2

    files = _tracked_python_files()
    if not files:
        return 0

    completed = subprocess.run([*command, *files], check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
