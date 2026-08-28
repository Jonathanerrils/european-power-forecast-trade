"""Shared utilities: config loading and logging setup.

Not listed explicitly in the spec's file tree, but pipeline.py and every
src/ module need a single place to load config.yaml rather than each
re-implementing path resolution. Keep this file boring on purpose.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path = None) -> Dict[str, Any]:
    """Load config.yaml from the repo root (or an explicit path)."""
    path = Path(config_path) if config_path else REPO_ROOT / "config.yaml"
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    return logging.getLogger("power_forecast")


def resolve_path(relative_path: str) -> Path:
    """Resolve a config-relative path (e.g. 'data/raw') against repo root."""
    p = REPO_ROOT / relative_path
    p.mkdir(parents=True, exist_ok=True) if not p.suffix else p.parent.mkdir(
        parents=True, exist_ok=True
    )
    return p


def get_entsoe_token(env_var: str = "ENTSOE_TOKEN") -> str:
    token = os.environ.get(env_var)
    if not token:
        raise RuntimeError(
            f"No ENTSO-E token found in environment variable '{env_var}'. "
            f"Copy .env.example to .env, add your token, and export it, e.g.\n"
            f"  export {env_var}=your-token-here\n"
            f"Register for a free token at https://transparency.entsoe.eu/"
        )
    return token
