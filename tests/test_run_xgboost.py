"""Tests for run_xgboost.py's argument resolution -- specifically the
guard that prevents the canonical 'xgboost_v1' experiment from being
silently created with the rejected ElasticNet-v2 profile.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_xgboost import resolve_run_args, CANONICAL_XGBOOST_RUN_PROFILES


def test_two_or_three_arguments_accepted():
    result = resolve_run_args(["delu_features.parquet", "some_run"])
    assert result == ("delu_features.parquet", "some_run", "v1")  # defaults to v1

    result = resolve_run_args(["delu_features.parquet", "some_run", "v2"])
    assert result == ("delu_features.parquet", "some_run", "v2")


def test_wrong_argument_count_rejected():
    with pytest.raises(SystemExit):
        resolve_run_args([])
    with pytest.raises(SystemExit):
        resolve_run_args(["delu_features.parquet"])
    with pytest.raises(SystemExit):
        resolve_run_args(["a", "b", "c", "d"])


def test_unknown_elasticnet_profile_rejected():
    with pytest.raises(ValueError, match="Unknown ElasticNet search profile"):
        resolve_run_args(["delu_features.parquet", "some_run", "v99"])


def test_xgboost_v1_accepts_v1_profile():
    result = resolve_run_args(["delu_features.parquet", "xgboost_v1", "v1"])
    assert result == ("delu_features.parquet", "xgboost_v1", "v1")

    result = resolve_run_args(["delu_features.parquet", "xgboost_v1"])  # default also resolves to v1
    assert result == ("delu_features.parquet", "xgboost_v1", "v1")


def test_xgboost_v1_rejects_v2_profile():
    """The exact failure mode this guard exists to prevent: a directory
    named 'xgboost_v1' silently comparing against the REJECTED
    ElasticNet-v2 profile.
    """
    with pytest.raises(ValueError, match="must use elasticnet_search_profile='v1'"):
        resolve_run_args(["delu_features.parquet", "xgboost_v1", "v2"])


def test_non_canonical_run_version_is_unconstrained():
    result = resolve_run_args(["delu_features.parquet", "xgboost_sensitivity_check", "v2"])
    assert result == ("delu_features.parquet", "xgboost_sensitivity_check", "v2")


def test_canonical_xgboost_run_profiles_mapping():
    assert CANONICAL_XGBOOST_RUN_PROFILES == {"xgboost_v1": "v1"}


# ---------------------------------------------------------------------
# Reproduction-target overrides for corrected-data re-verification runs
#
# Regression context: a real run of xgboost_v1_a03fix correctly failed
# the reproduction check because it was being compared against the
# ORIGINAL frozen baseline_v1 numbers, which were deliberately computed
# from different (pre-A03-parser-fix) data -- a real, expected
# mismatch, not silent drift. The fix lets a corrected-data run verify
# against its own corresponding baseline's ACTUAL saved results (read
# from disk), while leaving the original hardcoded frozen numbers
# completely untouched for the canonical xgboost_v1 run.
# ---------------------------------------------------------------------
def test_reproduction_target_overrides_maps_a03fix_pair():
    from run_xgboost import REPRODUCTION_TARGET_OVERRIDES
    assert REPRODUCTION_TARGET_OVERRIDES["xgboost_v1_a03fix"] == "baseline_v1_a03fix"


def test_load_baseline_reference_from_disk_reads_actual_saved_results(tmp_path, monkeypatch):
    import pandas as pd
    from run_xgboost import load_baseline_reference_from_disk
    import run_xgboost as rx_module

    baseline_dir = tmp_path / "outputs" / "models" / "delu_features" / "baseline_v1_a03fix"
    baseline_dir.mkdir(parents=True)
    for fold, mae in [("fold_1", 19.602452), ("regime_stress_test", 20.003031)]:
        df = pd.DataFrame(
            {"mae": [27.2, 33.6, mae, 23.0]},
            index=pd.Index(["lag_24", "lag_168", "elasticnet_full", "elasticnet_tier1"], name="model"),
        )
        df.to_csv(baseline_dir / f"{fold}_overall_metrics.csv")

    monkeypatch.setattr(rx_module, "REPO_ROOT", tmp_path)
    reference = load_baseline_reference_from_disk("delu_features", "baseline_v1_a03fix")
    assert reference == {"fold_1": 19.602452, "regime_stress_test": 20.003031}


def test_load_baseline_reference_from_disk_raises_clearly_when_missing(tmp_path, monkeypatch):
    from run_xgboost import load_baseline_reference_from_disk
    import run_xgboost as rx_module

    monkeypatch.setattr(rx_module, "REPO_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="has no saved results"):
        load_baseline_reference_from_disk("delu_features", "baseline_v1_never_run")


def test_verify_baseline_v1_reproduction_accepts_explicit_reference():
    """Passing a reference dict (as load_baseline_reference_from_disk
    produces) must be used instead of the hardcoded original frozen
    numbers -- this is the whole point of the override mechanism.
    """
    import pandas as pd
    from run_xgboost import verify_baseline_v1_reproduction

    class FakeResult:
        def __init__(self, mae):
            self.overall_metrics = pd.DataFrame({"mae": [mae]}, index=["elasticnet_full"])

    results = {"fold_1": FakeResult(19.602452)}
    verify_baseline_v1_reproduction(results, reference={"fold_1": 19.602452})  # must not raise


def test_verify_baseline_v1_reproduction_still_rejects_real_mismatch_with_custom_reference():
    import pandas as pd
    from run_xgboost import verify_baseline_v1_reproduction

    class FakeResult:
        def __init__(self, mae):
            self.overall_metrics = pd.DataFrame({"mae": [mae]}, index=["elasticnet_full"])

    results = {"fold_1": FakeResult(99.0)}
    with pytest.raises(RuntimeError, match="REPRODUCTION CHECK FAILED"):
        verify_baseline_v1_reproduction(results, reference={"fold_1": 19.602452})
