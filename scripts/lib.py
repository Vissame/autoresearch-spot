#!/usr/bin/env python3
"""Shared helpers for autoresearch-spot scripts."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple


def load_json_yaml_compatible(path: str) -> Dict[str, Any]:
    """Load .yaml file content that is written as JSON syntax."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest(path: str) -> Dict[str, Any]:
    data = load_json_yaml_compatible(path)
    if "runs" not in data or not isinstance(data["runs"], list):
        raise ValueError(f"Invalid manifest: {path}")
    return data


def extract_best_val_map(save_dir: str) -> Tuple[str, int, float]:
    """
    Returns: (status, best_epoch, best_val_mAP)
    status in {"ok", "missing", "invalid"}
    """
    loss_path = os.path.join(save_dir, "loss.json")
    if not os.path.isfile(loss_path):
        return "missing", -1, 0.0

    try:
        with open(loss_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    except Exception:
        return "invalid", -1, 0.0

    if not isinstance(history, list) or not history:
        return "invalid", -1, 0.0

    best_epoch = -1
    best_val_map = float("-inf")
    for row in history:
        if not isinstance(row, dict):
            continue
        epoch = row.get("epoch")
        val_map = row.get("val_mAP")
        if isinstance(epoch, int) and isinstance(val_map, (int, float)):
            if float(val_map) > best_val_map:
                best_val_map = float(val_map)
                best_epoch = epoch

    if best_epoch < 0:
        return "invalid", -1, 0.0
    return "ok", best_epoch, best_val_map


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_default_results_dir() -> str:
    return "/data/lyx/autoresearch-spot/results"


def get_manifest_runs(manifest_path: str) -> Tuple[str, List[Dict[str, Any]]]:
    manifest = load_manifest(manifest_path)
    round_id = manifest.get("round_id", "unknown_round")
    return round_id, manifest["runs"]
