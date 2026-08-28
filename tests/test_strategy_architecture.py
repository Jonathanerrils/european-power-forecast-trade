"""Architecture-level test (test 5 in docs/economic_contract_v1.md):
src/strategy.py must NEVER import src/oracle.py. This is a static
check, not a numerical one -- if a future edit wires the oracle into
live decision-making (even accidentally, e.g. "just to double-check
something"), this test fails immediately rather than the bug shipping
silently. The reverse direction (oracle.py importing FROM strategy.py,
to reuse candidate_pairs/degradation_cost rather than duplicate them)
is explicitly fine and unrelated to this constraint.

Detection is deliberately broad (any import statement mentioning the
literal name "oracle" anywhere in its module path or imported names),
not narrowly pattern-matched to "src.oracle" specifically. An earlier
version only checked node.module for ImportFrom statements, which
missed `from src import oracle` (module="src", not "src.oracle") and
`from . import oracle` (module=None entirely) -- both real gaps,
confirmed by testing them directly against the earlier checker before
this fix was written. There is no legitimate reason for strategy.py to
import anything named "oracle" from anywhere, so this breadth is
appropriate for this narrow, specific check.
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]


def _oracle_import_statements(path: Path) -> list:
    """Returns a human-readable description of every import statement
    that mentions "oracle" anywhere in its module path or imported
    names -- covers `import oracle`, `import src.oracle`,
    `from src.oracle import X`, `from src import oracle`, and
    `from . import oracle` uniformly.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "oracle" in alias.name.split("."):
                    hits.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module_parts = (node.module or "").split(".")
            module_str = ("." * node.level) + (node.module or "")
            for alias in node.names:
                if alias.name == "oracle" or "oracle" in module_parts:
                    hits.append(f"from {module_str} import {alias.name}")
    return hits


def test_strategy_module_never_imports_oracle_module():
    strategy_path = REPO_ROOT / "src" / "strategy.py"
    hits = _oracle_import_statements(strategy_path)
    assert not hits, (
        f"src/strategy.py contains oracle-related import(s): {hits} -- strategy-generation "
        f"code must never be able to reach the ex-post oracle (docs/economic_contract_v1.md). "
        f"This would let realized-price information leak into a D-1 decision."
    )


def test_oracle_import_detector_catches_every_spelling():
    """Regression guard for the exact gap a design review found: the
    original detector only checked ImportFrom's node.module, missing
    `from src import oracle` (recorded module="src", not "src.oracle")
    and `from . import oracle` (module=None entirely). Each spelling
    below is verified to be caught, not just the ones the original
    checker happened to catch.
    """
    import tempfile
    spellings = [
        "import oracle",
        "import src.oracle",
        "from src.oracle import oracle_pnl",
        "from src import oracle",
        "from . import oracle",
        "from .oracle import oracle_pnl",
    ]
    for code in spellings:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = Path(f.name)
        try:
            hits = _oracle_import_statements(temp_path)
            assert hits, f"Failed to detect oracle import in: {code!r}"
        finally:
            temp_path.unlink()


def test_oracle_import_detector_does_not_false_positive_on_unrelated_imports():
    import tempfile
    code = "import numpy as np\nfrom src.strategy import degradation_cost\nfrom src.clean import local_delivery_date_to_utc\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = Path(f.name)
    try:
        hits = _oracle_import_statements(temp_path)
        assert hits == []
    finally:
        temp_path.unlink()


def test_oracle_module_may_import_from_strategy():
    """Confirms the constraint really is one-directional -- oracle.py
    reusing strategy.py's pure helpers (candidate_pairs,
    degradation_cost) is expected and fine, not a violation of the
    same rule in reverse.
    """
    oracle_path = REPO_ROOT / "src" / "oracle.py"
    tree = ast.parse(oracle_path.read_text(), filename=str(oracle_path))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "src.strategy" in imported, (
        "expected oracle.py to reuse src.strategy's shared pure helpers rather than "
        "duplicating candidate_pairs/degradation_cost logic independently"
    )
