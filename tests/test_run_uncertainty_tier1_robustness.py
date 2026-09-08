"""Tests for run_uncertainty_tier1_robustness.py -- specifically that
it reads the frozen specification from an EXPLICITLY-NAMED parent
uncertainty run's own manifest rather than hardcoding window_days a
second time, that it refuses a silent default for that parent (a real
danger after the delivery-day information-set correction -- a caller
who forgot to pass the new parent could otherwise silently keep
reading a stale, pre-correction specification with no error), that the
saved manifest records the ACTUAL parent used rather than a hardcoded
string literal (a real bug an earlier version had), and that it fails
closed on a parent that wasn't itself built with the corrected method.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_uncertainty_tier1_robustness as rt1


def _write_manifest(manifest_dir, **overrides):
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "window_days": 60, "min_periods_days": 15, "quantiles": [0.1, 0.5, 0.9],
        "information_set_method": "delivery_day_safe",
        "xgboost_run_version": "xgboost_v1_a03fix",
    }
    manifest.update(overrides)
    (manifest_dir / "uncertainty_run_manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_resolve_run_args_requires_exactly_three_arguments():
    with pytest.raises(SystemExit):
        rt1.resolve_run_args([])
    with pytest.raises(SystemExit):
        rt1.resolve_run_args(["xgboost_v1_a03fix"])
    with pytest.raises(SystemExit):
        rt1.resolve_run_args(["xgboost_v1_a03fix", "uncertainty_selected_v2"])
    with pytest.raises(SystemExit):
        rt1.resolve_run_args(["a", "b", "c", "d"])


def test_resolve_run_args_valid_call():
    result = rt1.resolve_run_args(["xgboost_v1_a03fix", "uncertainty_selected_v2_dayorigin", "tier1_robustness_v3"])
    assert result == ("xgboost_v1_a03fix", "uncertainty_selected_v2_dayorigin", "tier1_robustness_v3")


def test_load_frozen_spec_has_no_default_parent():
    """The core fix: calling without a parent must be a TypeError
    (missing required argument), not a silent fallback to some
    hardcoded version string.
    """
    import inspect
    sig = inspect.signature(rt1.load_frozen_uncertainty_spec)
    assert sig.parameters["uncertainty_run_version"].default is inspect.Parameter.empty


def test_load_frozen_spec_reads_actual_saved_manifest_not_hardcoded(tmp_path, monkeypatch):
    manifest_dir = tmp_path / "outputs" / "uncertainty" / "delu_features" / "uncertainty_selected_v2_dayorigin"
    _write_manifest(manifest_dir)

    monkeypatch.setattr(rt1, "REPO_ROOT", tmp_path)
    spec = rt1.load_frozen_uncertainty_spec("delu_features", "uncertainty_selected_v2_dayorigin", "xgboost_v1_a03fix")
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
    manifest_dir = tmp_path / "outputs" / "uncertainty" / "delu_features" / "some_other_corrected_run"
    _write_manifest(manifest_dir, window_days=90, min_periods_days=22, quantiles=[0.05, 0.5, 0.95])

    monkeypatch.setattr(rt1, "REPO_ROOT", tmp_path)
    spec = rt1.load_frozen_uncertainty_spec("delu_features", "some_other_corrected_run", "xgboost_v1_a03fix")
    assert spec["window_days"] == 90
    assert spec["min_periods_days"] == 22
    assert spec["quantiles"] == [0.05, 0.5, 0.95]


def test_load_frozen_spec_raises_clearly_when_not_yet_built(tmp_path, monkeypatch):
    monkeypatch.setattr(rt1, "REPO_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="No frozen uncertainty specification found"):
        rt1.load_frozen_uncertainty_spec("delu_features", "never_run", "xgboost_v1_a03fix")


def test_load_frozen_spec_raises_on_missing_required_field(tmp_path, monkeypatch):
    manifest_dir = tmp_path / "outputs" / "uncertainty" / "delu_features" / "incomplete_run"
    manifest_dir.mkdir(parents=True)
    incomplete_manifest = {"window_days": 60}  # missing min_periods_days, quantiles
    (manifest_dir / "uncertainty_run_manifest.json").write_text(json.dumps(incomplete_manifest))

    monkeypatch.setattr(rt1, "REPO_ROOT", tmp_path)
    with pytest.raises(ValueError, match="missing required field"):
        rt1.load_frozen_uncertainty_spec("delu_features", "incomplete_run", "xgboost_v1_a03fix")


# ---------------------------------------------------------------------
# information_set_method fail-closed check -- refuses to build Tier-1
# robustness evidence on a parent that predates the delivery-day
# correction, even if named explicitly.
# ---------------------------------------------------------------------
def test_load_frozen_spec_rejects_stale_pre_correction_parent(tmp_path, monkeypatch):
    """A parent run with no information_set_method field at all (i.e.
    a genuinely old, pre-correction run like the original
    uncertainty_selected_v1) must be rejected, not silently accepted.
    """
    manifest_dir = tmp_path / "outputs" / "uncertainty" / "delu_features" / "uncertainty_selected_v1"
    manifest_dir.mkdir(parents=True)
    old_manifest = {"window_days": 60, "min_periods_days": 15, "quantiles": [0.1, 0.5, 0.9], "xgboost_run_version": "xgboost_v1_a03fix"}  # no information_set_method
    (manifest_dir / "uncertainty_run_manifest.json").write_text(json.dumps(old_manifest))

    monkeypatch.setattr(rt1, "REPO_ROOT", tmp_path)
    with pytest.raises(ValueError, match="Refusing to build Tier-1 robustness evidence"):
        rt1.load_frozen_uncertainty_spec("delu_features", "uncertainty_selected_v1", "xgboost_v1_a03fix")


def test_load_frozen_spec_rejects_explicitly_wrong_method_label(tmp_path, monkeypatch):
    manifest_dir = tmp_path / "outputs" / "uncertainty" / "delu_features" / "some_run"
    _write_manifest(manifest_dir, information_set_method="hourly_rolling")

    monkeypatch.setattr(rt1, "REPO_ROOT", tmp_path)
    with pytest.raises(ValueError, match="Refusing to build Tier-1 robustness evidence"):
        rt1.load_frozen_uncertainty_spec("delu_features", "some_run", "xgboost_v1_a03fix")


def test_load_frozen_spec_accepts_genuine_corrected_parent(tmp_path, monkeypatch):
    manifest_dir = tmp_path / "outputs" / "uncertainty" / "delu_features" / "uncertainty_selected_v2_dayorigin"
    _write_manifest(manifest_dir, information_set_method="delivery_day_safe")

    monkeypatch.setattr(rt1, "REPO_ROOT", tmp_path)
    spec = rt1.load_frozen_uncertainty_spec("delu_features", "uncertainty_selected_v2_dayorigin", "xgboost_v1_a03fix")  # must not raise
    assert spec["window_days"] == 60


# ---------------------------------------------------------------------
# xgboost_run_version lineage check -- refuses a parent uncertainty run
# built on a DIFFERENT XGBoost run than the one this Tier-1/Full
# comparison is actually being built from.
# ---------------------------------------------------------------------
def test_load_frozen_spec_rejects_xgboost_run_version_mismatch(tmp_path, monkeypatch):
    manifest_dir = tmp_path / "outputs" / "uncertainty" / "delu_features" / "some_run"
    _write_manifest(manifest_dir, xgboost_run_version="a_different_xgboost_run")

    monkeypatch.setattr(rt1, "REPO_ROOT", tmp_path)
    with pytest.raises(ValueError, match="refusing to build Tier-1 robustness evidence by"):
        rt1.load_frozen_uncertainty_spec("delu_features", "some_run", "xgboost_v1_a03fix")


def test_load_frozen_spec_raises_on_missing_xgboost_run_version_field(tmp_path, monkeypatch):
    manifest_dir = tmp_path / "outputs" / "uncertainty" / "delu_features" / "some_run"
    manifest_dir.mkdir(parents=True)
    incomplete = {"window_days": 60, "min_periods_days": 15, "quantiles": [0.1, 0.5, 0.9], "information_set_method": "delivery_day_safe"}
    (manifest_dir / "uncertainty_run_manifest.json").write_text(json.dumps(incomplete))

    monkeypatch.setattr(rt1, "REPO_ROOT", tmp_path)
    with pytest.raises(ValueError, match="missing required field"):
        rt1.load_frozen_uncertainty_spec("delu_features", "some_run", "xgboost_v1_a03fix")


# ---------------------------------------------------------------------
# Manifest lineage bug fix: the saved output manifest must record the
# ACTUAL parent passed, not a hardcoded string literal. An earlier
# version of this script always wrote "uncertainty_selected_v1"
# regardless of what uncertainty_run_version was actually used.
# ---------------------------------------------------------------------
def test_source_code_never_hardcodes_the_old_parent_name_in_the_manifest():
    """Static check: the string literal "uncertainty_selected_v1" must
    not appear as a manifest value anywhere in this script's source --
    confirms the fix isn't just working by coincidence for one
    specific parent name.
    """
    import ast
    src = ast.parse(Path(rt1.__file__).read_text())
    for node in ast.walk(src):
        if isinstance(node, ast.Constant) and node.value == "uncertainty_selected_v1":
            pytest.fail(
                "Found the literal string 'uncertainty_selected_v1' hardcoded in "
                "run_uncertainty_tier1_robustness.py -- the parent_uncertainty_spec manifest "
                "field must be derived from the actual uncertainty_run_version argument, never "
                "a hardcoded literal (this was a real bug: a previous version always wrote this "
                "string into the output manifest regardless of which parent was actually used)."
            )
