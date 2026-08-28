"""Run locally: python scripts/check_orphaned_test_bodies.py

Regression tool for a real bug, now confirmed in two different shapes:
an earlier edit replaced a function's `def` line (and sometimes its
docstring) but left its ORIGINAL body behind, un-indented-away,
directly after the replacement. Because there was no `def` at column 0
between the two blocks, Python didn't raise a syntax error -- it
silently treated the orphaned code as MORE statements appended to the
PREVIOUS function's body, after that function's own `return`. The
orphaned function's logic became dead code, invisibly, attributed to
the wrong function's name.

TWO detection heuristics, because one real incident wasn't caught by
the other:

1. A bare string-literal expression statement (the orphaned function's
   docstring) appearing in a function body somewhere OTHER than as
   that function's very first statement. Caught the first two real
   incidents (tests/test_entsoe_client.py, then src/strategy.py).

2. Any statement following a `return` (or unconditional `raise`) at a
   function's TOP LEVEL -- genuinely unreachable code in Python,
   never legitimate, zero false-positive risk. Added after a THIRD
   real incident (run_strategy_decomposition.py) where the orphaned
   body's docstring had ALSO been consumed by the bad edit, leaving
   only bare code statements behind -- heuristic 1 alone missed this
   because there was no stray string literal to catch. Confirmed by
   directly reproducing the exact bug against the pre-fix checker and
   watching it slip through before writing this second heuristic.

SCOPE: scans src/, scripts/, and tests/ -- anywhere a `str_replace`-
style edit could leave an orphaned body behind. This is a real,
targeted lint check for these two well-defined, previously-real
shapes, not a general-purpose linter.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# REPO_ROOT itself is included (non-recursively -- .glob("*.py") only
# matches direct children, so this does not re-scan the subdirectories
# below) because every run_*.py orchestration script lives at the repo
# root, not inside src/scripts/tests -- confirmed as a real gap when
# find_unreachable_top_level_code failed to catch a genuine reproduced
# bug in run_strategy_decomposition.py simply because that file was
# never in the scanned set at all, not because the detection logic
# itself was wrong.
SCAN_DIRS = [REPO_ROOT, REPO_ROOT / "tests", REPO_ROOT / "src", REPO_ROOT / "scripts"]


def find_stray_docstrings(path: Path) -> list[tuple[int, str, int]]:
    """Returns (function_def_lineno, function_name, orphan_lineno) for
    every function whose body contains a bare string-literal Expr
    statement NOT in first position.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        for i, stmt in enumerate(body):
            is_string_literal_expr = (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            )
            if is_string_literal_expr and i != 0:
                hits.append((node.lineno, node.name, stmt.lineno))
    return hits


def find_unreachable_top_level_code(path: Path) -> list[tuple[int, str, int]]:
    """Returns (function_def_lineno, function_name, orphan_lineno) for
    every function with a statement following a `return` or
    unconditional `raise` at the function's TOP level (not nested
    inside an if/for/while/try -- those are legitimately conditional
    and not what this catches). Top-level unreachable code is never
    legitimate in working Python, so this has essentially no
    false-positive risk.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        for i, stmt in enumerate(body[:-1]):  # skip the last statement -- nothing follows it
            if isinstance(stmt, (ast.Return, ast.Raise)):
                orphan = body[i + 1]
                hits.append((node.lineno, node.name, orphan.lineno))
                break  # one hit per function is enough to flag it
    return hits


def _iter_scan_files():
    seen = set()
    for scan_dir in SCAN_DIRS:
        if not scan_dir.exists():
            continue
        for path in sorted(scan_dir.glob("*.py")):
            if path not in seen:
                seen.add(path)
                yield path


def main():
    docstring_hits = []
    unreachable_hits = []
    n_files = 0
    for path in _iter_scan_files():
        n_files += 1
        try:
            d_hits = find_stray_docstrings(path)
            u_hits = find_unreachable_top_level_code(path)
        except SyntaxError as exc:
            print(f"SYNTAX ERROR in {path.relative_to(REPO_ROOT)}: {exc}")
            return 1
        for def_lineno, name, orphan_lineno in d_hits:
            docstring_hits.append((path.relative_to(REPO_ROOT), def_lineno, name, orphan_lineno))
        for def_lineno, name, orphan_lineno in u_hits:
            unreachable_hits.append((path.relative_to(REPO_ROOT), def_lineno, name, orphan_lineno))

    total = len(docstring_hits) + len(unreachable_hits)
    if total == 0:
        print(f"OK: no suspicious mid-function string literals or unreachable top-level code found "
              f"({n_files} file(s) checked across {', '.join(d.name for d in SCAN_DIRS)}).")
        return 0

    if docstring_hits:
        print(f"FOUND {len(docstring_hits)} suspicious mid-function string literal(s) -- "
              f"possible orphaned/merged function body:\n")
        for path, def_lineno, name, orphan_lineno in docstring_hits:
            print(f"  {path}:{def_lineno}  def {name}(...)")
            print(f"    unexpected string literal at line {orphan_lineno} (not the function's first statement)")
            print(f"    this is exactly the shape of a previous real bug: check whether this is a")
            print(f"    second function's docstring accidentally left inside this function's body.\n")

    if unreachable_hits:
        print(f"FOUND {len(unreachable_hits)} function(s) with unreachable code after return/raise -- "
              f"possible orphaned/merged function body:\n")
        for path, def_lineno, name, orphan_lineno in unreachable_hits:
            print(f"  {path}:{def_lineno}  def {name}(...)")
            print(f"    unreachable statement at line {orphan_lineno} (follows a return/raise at top level)")
            print(f"    this is exactly the shape of a previous real bug: check whether this is a")
            print(f"    second function's body accidentally left inside this function after its return.\n")

    return 1


if __name__ == "__main__":
    sys.exit(main())
