"""Minimal YAML config loader with safe fallback."""

import json


def load_config(path: str) -> dict:
    try:
        import yaml
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except ImportError:
        # fall back to JSON if PyYAML is unavailable
        with open(path, "r") as f:
            return json.load(f)


def flatten_train_cfg(cfg: dict) -> dict:
    """Merge the train/loss sub-dicts into the flat dict the Trainer expects."""
    out = dict(cfg.get("train", {}))
    out.update(cfg.get("loss", {}))
    return out
