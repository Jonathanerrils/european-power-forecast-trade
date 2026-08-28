"""Tests for run_models.py's argument resolution -- specifically the
guard that prevents a canonical baseline experiment (baseline_v1,
baseline_v2) from silently being created with the wrong search grid.

Regression context: an earlier version defaulted search_profile to
DEFAULT_SEARCH_PROFILE ("v2") when omitted from the command line, so a
bare `python run_models.py` (or one missing the third argument) could
create a directory literally named "baseline_v1" that actually ran the
v2 search grid. All three CLI arguments are now mandatory, and the two
canonical names are additionally locked to their intended profile.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_models import resolve_run_args, CANONICAL_RUN_PROFILES


def test_all_three_arguments_are_mandatory():
    with pytest.raises(SystemExit):
        resolve_run_args([])
    with pytest.raises(SystemExit):
        resolve_run_args(["delu_features.parquet"])
    with pytest.raises(SystemExit):
        resolve_run_args(["delu_features.parquet", "baseline_v1"])


def test_valid_three_argument_call_resolves_correctly():
    result = resolve_run_args(["delu_features.parquet", "baseline_v1", "v1"])
    assert result == ("delu_features.parquet", "baseline_v1", "v1")


def test_unknown_search_profile_rejected():
    with pytest.raises(ValueError, match="Unknown search_profile"):
        resolve_run_args(["delu_features.parquet", "some_run", "v99"])


def test_baseline_v1_locked_to_v1_profile():
    """The exact failure mode this guard exists to prevent: creating a
    directory named 'baseline_v1' that actually used the v2 grid.
    """
    with pytest.raises(ValueError, match="must use search_profile='v1'"):
        resolve_run_args(["delu_features.parquet", "baseline_v1", "v2"])


def test_baseline_v2_locked_to_v2_profile():
    with pytest.raises(ValueError, match="must use search_profile='v2'"):
        resolve_run_args(["delu_features.parquet", "baseline_v2", "v1"])


def test_non_canonical_run_version_is_unconstrained():
    """Ad-hoc experiment names (not 'baseline_v1'/'baseline_v2') aren't
    locked to a specific profile -- the guard only protects the two
    canonical names.
    """
    result = resolve_run_args(["delu_features.parquet", "elasticnet_sensitivity_check", "v1"])
    assert result == ("delu_features.parquet", "elasticnet_sensitivity_check", "v1")


def test_canonical_run_profiles_mapping_matches_expectations():
    assert CANONICAL_RUN_PROFILES == {"baseline_v1": "v1", "baseline_v2": "v2"}
