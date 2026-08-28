"""Tests for run_uncertainty_tier1_robustness.py -- specifically that
it reads the frozen specification from uncertainty_selected_v1's own
manifest rather than hardcoding window_days a second time (which could
silently drift out of sync if the selected value is ever revisited
through a new pre-registered process), and that it refuses to guess
when that frozen manifest doesn't exist yet.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_uncertainty_tier1_robustness as rt1


def test_resolve_run_args_requires_exactly_two_arguments():
    with pytest.raises(SystemExit):
        rt1.resolve_run_args([])
    with pytest.raises(SystemExit):
        rt1.resolve_run_args(["xgboost_v1_a03fix"])
    with pytest.raises(SystemExit):
        rt1.resolve_run_args(["a", "b", "c"])


def test_resolve_run_args_valid_call():
    result = rt1.resolve_run_args(["xgboost_v1_a03fix", "tier1_robustness_v1"])
    assert result == ("xgboost_v1_a03fix", "tier1_robustness_v1")


def test_load_frozen_spec_reads_actual_saved_manifest_not_hardcoded(tmp_path, monkeypatch):
    manifest_dir = tmp_path / "outputs" / "uncertainty" / "delu_features" / "uncertainty_selected_v1"
    manifest_dir.mkdir(parents=True)
    manifest = {
        "window_days": 60, "min_periods_days": 15, "quantiles": [0.1, 0.5, 0.9],
    }
    (manifest_dir / "uncertainty_run_manifest.json").write_text(json.dumps(manifest))

    monkeypatch.setattr(rt1, "REPO_ROOT", tmp_path)
    spec = rt1.load_frozen_uncertainty_spec("delu_features")
    assert spec["window_days"] == 60
    assert spec["min_periods_days"] == 15
    assert spec["quantiles"] == [0.1, 0.5, 0.9]


def test_load_frozen_spec_reads_whatever_window_is_actually_frozen(tmp_path, monkeypatch):
    """Regression guard for the exact bug class this design avoids: if
    the selected window were ever revisited (e.g. to 90 days via a new
    pre-registered experiment), this function must pick up 90
    automatically from the manifest -- not silently keep using a
    hardcoded '60' baked into this script. Deliberately uses DIFFERENT
    values than the other tests here, to prove this isn't just reading
    back a coincidentally-matching hardcoded default.
    """
    manifest_dir = tmp_path / "outputs" / "uncertainty" / "delu_features" / "uncertainty_selected_v1"
    manifest_dir.mkdir(parents=True)
    manifest = {"window_days": 90, "min_periods_days": 22, "quantiles": [0.05, 0.5, 0.95]}
    (manifest_dir / "uncertainty_run_manifest.json").write_text(json.dumps(manifest))

    monkeypatch.setattr(rt1, "REPO_ROOT", tmp_path)
    spec = rt1.load_frozen_uncertainty_spec("delu_features")
    assert spec["window_days"] == 90
    assert spec["min_periods_days"] == 22
    assert spec["quantiles"] == [0.05, 0.5, 0.95]


def test_load_frozen_spec_raises_clearly_when_not_yet_built(tmp_path, monkeypatch):
    monkeypatch.setattr(rt1, "REPO_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="No frozen uncertainty specification found"):
        rt1.load_frozen_uncertainty_spec("delu_features")


def test_load_frozen_spec_raises_on_missing_required_field(tmp_path, monkeypatch):
    manifest_dir = tmp_path / "outputs" / "uncertainty" / "delu_features" / "uncertainty_selected_v1"
    manifest_dir.mkdir(parents=True)
    incomplete_manifest = {"window_days": 60}  # missing min_periods_days, quantiles
    (manifest_dir / "uncertainty_run_manifest.json").write_text(json.dumps(incomplete_manifest))

    monkeypatch.setattr(rt1, "REPO_ROOT", tmp_path)
    with pytest.raises(ValueError, match="missing required field"):
        rt1.load_frozen_uncertainty_spec("delu_features")
