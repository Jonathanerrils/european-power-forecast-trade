"""Run locally: python scripts/check_stale_claims.py

Greps the repo for phrasings that were PROVEN FALSE during development
and corrected -- but, being pure text, have no mechanism forcing every
copy to update together. This project has hit that exact failure three
separate times with the same claim ("classificationSequence is absent
before 2025-10-01"): it got fixed in one file's docstring, then
resurfaced in a test file's docstring, then again in the README, each
time because a different branch/upload hadn't seen the earlier fix.

This is deliberately NOT a general style linter. Every entry below is a
specific, previously-real mistake -- the point is to make "we already
proved this wrong once" mechanically checkable instead of dependent on
someone remembering the whole project history. Add an entry here
whenever a review/audit proves a previously-stated claim false; do NOT
add speculative or stylistic entries.

Exit code 1 (and prints every match) if any stale phrase is found
anywhere in the repo. Exit code 0 if clean. Intended to be run before
treating a batch of files as final -- see README's "Process notes"
section for when to run this.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories that legitimately contain historical/quoted text (e.g. this
# script's own docstring above, or archived review transcripts) and
# should not be scanned.
EXCLUDE_DIRS = {".git", "__pycache__", "outputs", "data", ".pytest_cache"}
EXCLUDE_FILES = {Path(__file__).name}

# (stale phrase, why it's wrong, where the correction lives)
STALE_CLAIMS = [
    (
        "Not present before the 2025-10-01 quarter-hour",
        "FALSE: the full historical structural audit (5,538 Period rows, zero exceptions) "
        "confirmed classificationSequence is present throughout the ENTIRE price history. "
        "What changes at the cutover is sequence 1's RESOLUTION (PT60M->PT15M), not its "
        "existence. This exact false claim recurred independently in entsoe_client.py, "
        "test_entsoe_client.py, and README.md across separate uploads.",
        "src/entsoe_client.py (fetch_day_ahead_prices, _parse_timeseries, "
        "select_primary_auction_sequence docstrings), README.md 'Real Bugs Found' item 3",
    ),
    (
        "Pre-2025 data has no classificationSequence field at all",
        "Same false claim as above, found in a test docstring specifically.",
        "tests/test_entsoe_client.py::test_select_primary_auction_sequence_keeps_rows_with_no_sequence_info",
    ),
]

# Deliberately NOT included: a phrase like "two different auction
# products/mechanisms" for the ruled-out hypothesis. Unlike the two
# entries above, that phrase legitimately appears in CORRECT text too
# ("X was ruled out"), so a plain substring match produces false
# positives on legitimate usage rather than catching regressions. Only
# add entries here that are wrong under every framing -- if a phrase
# needs context to judge, this tool is the wrong mechanism for it.


def scan() -> list[tuple[Path, int, str, str]]:
    hits = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.name in EXCLUDE_FILES:
            continue
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        if path.suffix not in (".py", ".md", ".txt", ".yaml", ".yml"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for phrase, reason, correction_location in STALE_CLAIMS:
            for i, line in enumerate(text.splitlines(), start=1):
                if phrase in line:
                    hits.append((path.relative_to(REPO_ROOT), i, phrase, reason))
    return hits


def main():
    hits = scan()
    if not hits:
        print(f"OK: no stale claims found ({len(STALE_CLAIMS)} known phrase(s) checked).")
        return 0

    print(f"FOUND {len(hits)} occurrence(s) of previously-corrected false claim(s):\n")
    for path, lineno, phrase, reason in hits:
        print(f"  {path}:{lineno}")
        print(f"    phrase: {phrase!r}")
        print(f"    why it's wrong: {reason}")
        print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
