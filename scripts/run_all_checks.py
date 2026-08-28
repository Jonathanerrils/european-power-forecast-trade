"""Run locally: python scripts/run_all_checks.py

Runs every mechanical safeguard this project has accumulated after a
real mistake, in one command. None of these replace the test suite --
run `python -m pytest tests/ -v` too. These specifically catch classes
of problems tests don't: stale documentation claims that were proven
false but never fully removed, and structurally orphaned test code
that runs (and passes) under the wrong test's name.

Run this before treating any batch of file changes as final, and
especially before uploading files as "the current state of the repo" --
several real incidents in this project were caused by one branch's fix
not propagating to another copy of the same claim elsewhere.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKS = [
    ("Stale claims", REPO_ROOT / "scripts" / "check_stale_claims.py"),
    ("Orphaned test bodies", REPO_ROOT / "scripts" / "check_orphaned_test_bodies.py"),
]


def main():
    overall_ok = True
    for name, script in CHECKS:
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        result = subprocess.run([sys.executable, str(script)], cwd=REPO_ROOT)
        overall_ok = overall_ok and (result.returncode == 0)

    print(f"\n{'=' * 78}")
    if overall_ok:
        print("ALL CHECKS PASSED. Still run `python -m pytest tests/ -v` separately --")
        print("these checks are a supplement, not a replacement.")
    else:
        print("ONE OR MORE CHECKS FAILED -- see output above.")
    print("=" * 78)
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
