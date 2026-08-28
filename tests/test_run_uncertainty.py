"""Tests for run_uncertainty.py's argument resolution -- specifically
the optional explicit window_days override, which exists so a
CONSEQUENTIAL frozen run (e.g. uncertainty_selected_v1, following the
pre-registered sensitivity experiment) records its window_days
directly from the invocation itself, not implicitly from a config.yaml
default that could later be edited without the frozen run's provenance
becoming misleading.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_uncertainty import resolve_run_args


def test_two_arguments_omits_window_days_override():
    result = resolve_run_args(["xgboost_v1_a03fix", "uncertainty_v1"])
    assert result == ("xgboost_v1_a03fix", "uncertainty_v1", None)


def test_three_arguments_parses_explicit_window_days():
    result = resolve_run_args(["xgboost_v1_a03fix", "uncertainty_selected_v1", "60"])
    assert result == ("xgboost_v1_a03fix", "uncertainty_selected_v1", 60)


def test_wrong_argument_count_rejected():
    with pytest.raises(SystemExit):
        resolve_run_args([])
    with pytest.raises(SystemExit):
        resolve_run_args(["xgboost_v1_a03fix"])
    with pytest.raises(SystemExit):
        resolve_run_args(["a", "b", "c", "d"])


def test_non_integer_window_days_rejected():
    with pytest.raises(SystemExit, match="must be an integer"):
        resolve_run_args(["xgboost_v1_a03fix", "uncertainty_v1", "sixty"])


def test_non_positive_window_days_rejected():
    with pytest.raises(SystemExit, match="must be positive"):
        resolve_run_args(["xgboost_v1_a03fix", "uncertainty_v1", "0"])
    with pytest.raises(SystemExit, match="must be positive"):
        resolve_run_args(["xgboost_v1_a03fix", "uncertainty_v1", "-60"])
