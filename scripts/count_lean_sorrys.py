#!/usr/bin/env python3
"""Count the number of 'sorry' occurrences in all Lean source files under src/lean/.

Usage:
    python scripts/count_lean_sorrys.py

Exits with code 0 regardless of count; prints a summary line so CI can track
the sorry budget over time.
"""

import re
import sys
from pathlib import Path

LEAN_ROOT = Path(__file__).parent.parent / "src" / "lean"

# Match the keyword 'sorry' as a standalone token (not inside a string or comment,
# but this simple regex is sufficient for snapshot / trend tracking purposes).
_SORRY_RE = re.compile(r"\bsorry\b")


def count_sorrys(root: Path) -> dict[Path, int]:
    counts: dict[Path, int] = {}
    for path in sorted(root.rglob("*.lean")):
        # Skip Lake build artefacts
        if ".lake" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        n = len(_SORRY_RE.findall(text))
        if n:
            counts[path] = n
    return counts


def main() -> None:
    counts = count_sorrys(LEAN_ROOT)
    total = sum(counts.values())
    if counts:
        for path, n in counts.items():
            rel = path.relative_to(LEAN_ROOT)
            print(f"  {n:4d}  {rel}")
    print(f"Total sorrys: {total}")


if __name__ == "__main__":
    main()
    sys.exit(0)
