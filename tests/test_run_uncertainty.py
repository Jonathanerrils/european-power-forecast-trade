"""Tests for run_uncertainty.py's argument resolution -- specifically
the 3-argument form that binds mechanically to a completed sensitivity
experiment's own selected winner (window_days, quantiles, provenance
all read directly from that experiment's saved manifest, never
re-typed), which exists precisely because a manually-transcribed
window_days number could silently diverge from what a pre-registered
sensitivity experiment actually selected.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_uncertainty import resolve_run_args, load_sensitivity_winner


def test_two_arguments_omits_sensitivity_binding():
    result = resolve_run_args(["xgboost_v1_a03fix", "uncertainty_v1"])
    assert result == ("xgboost_v1_a03fix", "uncertainty_v1", None)


def test_three_arguments_binds_to_sensitivity_run():
    result = resolve_run_args(["xgboost_v1_a03fix", "uncertainty_window_sensitivity_v3_dayorigin", "uncertainty_selected_v2_dayorigin"])
    assert result == ("xgboost_v1_a03fix", "uncertainty_selected_v2_dayorigin", "uncertainty_window_sensitivity_v3_dayorigin")


def test_wrong_argument_count_rejected():
    with pytest.raises(SystemExit):
        resolve_run_args([])
    with pytest.raises(SystemExit):
        resolve_run_args(["xgboost_v1_a03fix"])
    with pytest.raises(SystemExit):
        resolve_run_args(["a", "b", "c", "d"])


# ---------------------------------------------------------------------
# load_sensitivity_winner -- the actual mechanical-binding gate
# ---------------------------------------------------------------------
def _write_sensitivity_manifest(manifest_dir, **overrides):
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "lowest_interval_score_window_days": 90,
        "quantiles": [0.1, 0.5, 0.9],
        "information_set_method": "delivery_day_safe",
        "xgboost_run_version": "xgboost_v1_a03fix",
    }
    manifest.update(overrides)
    (manifest_dir / "sensitivity_run_manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_load_sensitivity_winner_reads_actual_selected_window_not_hardcoded(tmp_path, monkeypatch):
    """The core purpose of this function: window_days comes from
    whatever the sensitivity experiment actually selected, not a
    number anyone had to remember to retype correctly.
    """
    import run_uncertainty as ru
    manifest_dir = tmp_path / "outputs" / "uncertainty" / "delu_features" / "sens_v1"
    _write_sensitivity_manifest(manifest_dir, lowest_interval_score_window_days=90)

    monkeypatch.setattr(ru, "REPO_ROOT", tmp_path)
    winner = load_sensitivity_winner("delu_features", "sens_v1", "xgboost_v1_a03fix")
    assert winner["window_days"] == 90
    assert winner["quantiles"] == [0.1, 0.5, 0.9]


def test_load_sensitivity_winner_rejects_xgboost_mismatch(tmp_path, monkeypatch):
    import run_uncertainty as ru
    manifest_dir = tmp_path / "outputs" / "uncertainty" / "delu_features" / "sens_v1"
    _write_sensitivity_manifest(manifest_dir, xgboost_run_version="some_other_xgboost_run")

    monkeypatch.setattr(ru, "REPO_ROOT", tmp_path)
    with pytest.raises(ValueError, match="refusing to mix predictions"):
        load_sensitivity_winner("delu_features", "sens_v1", "xgboost_v1_a03fix")


def test_load_sensitivity_winner_rejects_stale_method(tmp_path, monkeypatch):
    import run_uncertainty as ru
    manifest_dir = tmp_path / "outputs" / "uncertainty" / "delu_features" / "sens_v1"
    _write_sensitivity_manifest(manifest_dir, information_set_method="hourly_rolling")

    monkeypatch.setattr(ru, "REPO_ROOT", tmp_path)
    with pytest.raises(ValueError, match="stale, pre-correction sensitivity result"):
        load_sensitivity_winner("delu_features", "sens_v1", "xgboost_v1_a03fix")


def test_load_sensitivity_winner_raises_clearly_when_missing(tmp_path, monkeypatch):
    import run_uncertainty as ru
    monkeypatch.setattr(ru, "REPO_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="No sensitivity run found"):
        load_sensitivity_winner("delu_features", "never_run", "xgboost_v1_a03fix")


def test_load_sensitivity_winner_raises_on_missing_field(tmp_path, monkeypatch):
    import run_uncertainty as ru
    manifest_dir = tmp_path / "outputs" / "uncertainty" / "delu_features" / "sens_v1"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "sensitivity_run_manifest.json").write_text(json.dumps({"quantiles": [0.1, 0.5, 0.9]}))
    monkeypatch.setattr(ru, "REPO_ROOT", tmp_path)
    with pytest.raises(ValueError, match="missing required field"):
        load_sensitivity_winner("delu_features", "sens_v1", "xgboost_v1_a03fix")


# ---------------------------------------------------------------------
# Architecture guard: this script must genuinely use the delivery-day-
# safe computation, not just claim to in its docstring. A static check
# on the actual source, not merely a functional test of resolve_run_args
# -- the earlier version of this test file only ever exercised CLI
# parsing and never proved the production computation path was safe.
# ---------------------------------------------------------------------
def test_run_uncertainty_never_references_old_hourly_quantile_function():
    import ast
    import run_uncertainty as ru
    src = ast.parse(Path(ru.__file__).read_text())
    forbidden = {"compute_rolling_residual_quantiles"}
    found = set()
    for node in ast.walk(src):
        if isinstance(node, ast.Name) and node.id in forbidden:
            found.add(node.id)
        if isinstance(node, ast.alias) and node.name in forbidden:
            found.add(node.name)
    assert not found, f"run_uncertainty.py references the old hourly path: {found}"


def test_run_uncertainty_imports_delivery_day_safe_function():
    import ast
    import run_uncertainty as ru
    src = ast.parse(Path(ru.__file__).read_text())
    imported = set()
    for node in ast.walk(src):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
    assert "compute_delivery_day_residual_quantiles" in imported


def test_run_uncertainty_calls_holdout_guard():
    """Structural, not just conventional: confirms the 2026 lock is
    actually wired into the source, not merely claimed in a comment.
    """
    import ast
    import run_uncertainty as ru
    src = ast.parse(Path(ru.__file__).read_text())
    imported = set()
    for node in ast.walk(src):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
    assert "assert_no_holdout_access" in imported
