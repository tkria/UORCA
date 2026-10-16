"""Resolve the local paths that project runs record.

Runs store local outputs as paths relative to a data root (``scratch/<run>``,
``results/<run>``) and HPC outputs as absolute remote paths. Projects live in
``~/.uorca/projects`` and are shared by every checkout and worktree, so relative
paths must resolve against one fixed root rather than the app's working directory.

The root is, in order: the ``UORCA_DATA_ROOT`` environment variable, ``data_root`` in
``~/.uorca/config.yaml``, or the current working directory.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

DATA_ROOT_ENV = "UORCA_DATA_ROOT"
DEFAULT_CONFIG_PATH = Path.home() / ".uorca" / "config.yaml"


def data_root(config_path: Path = DEFAULT_CONFIG_PATH) -> Path:
    """The directory that relative run paths resolve against."""
    configured = os.environ.get(DATA_ROOT_ENV)
    source = f"${DATA_ROOT_ENV}"
    if not configured and config_path.exists():
        data = yaml.safe_load(config_path.read_text()) or {}
        configured = data.get("data_root")
        source = f"data_root in {config_path}"
    if not configured:
        return Path.cwd()
    root = Path(configured).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"UORCA data root {root} (from {source}) is not a directory")
    return root


def resolve_local_path(path: str | Path, config_path: Path = DEFAULT_CONFIG_PATH) -> Path:
    """Absolute paths are returned unchanged; relative paths are joined to the data root."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return data_root(config_path) / p
