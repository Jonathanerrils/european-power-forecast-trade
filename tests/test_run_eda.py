"""Tests for run_eda.py's argument resolution -- specifically that
run_version is mandatory, not defaulted. Regression context: an
earlier version used a fixed, unversioned output path, so every re-run
silently overwrote the previous EDA output. Concretely, this meant the
pre-A03-parser-fix EDA output was silently destroyed when EDA was
re-run on the corrected data -- there was no way to do a genuine
pre/post comparison after the fact. run_version is now mandatory and
FileExistsError-guarded, matching run_models.py's existing pattern.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_eda import resolve_run_args


def test_run_version_is_mandatory():
    with pytest.raises(SystemExit):
        resolve_run_args([])
    with pytest.raises(SystemExit):
        resolve_run_args(["delu_features.parquet"])


def test_valid_call_resolves_correctly():
    result = resolve_run_args(["delu_features.parquet", "development_a03fix_v1"])
    assert result == ("delu_features.parquet", "development_a03fix_v1")


def test_too_many_arguments_rejected():
    with pytest.raises(SystemExit):
        resolve_run_args(["a", "b", "c"])
