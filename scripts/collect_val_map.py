#!/usr/bin/env python3
"""Collect best val_mAP from all runs in a manifest."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from lib import (
    ensure_dir,
    extract_best_val_map,
    get_default_results_dir,
    get_manifest_runs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--output",
        help="Output json path. Default: /data/lyx/autoresearch-spot/results/<round_id>.results.json",
    )
    args = parser.parse_args()

    round_id, runs = get_manifest_runs(args.manifest)
    results: List[Dict[str, Any]] = []

    for run in runs:
        run_id = run["run_id"]
        train_args = run["train_args"]
        save_dir = str(train_args["save_dir"])
        status, best_epoch, best_val_map = extract_best_val_map(save_dir)
        results.append(
            {
                "run_id": run_id,
                "round_id": round_id,
                "status": status,
                "feature_arch": train_args.get("feature_arch", ""),
                "temporal_arch": train_args.get("temporal_arch", ""),
                "artifact_dir": save_dir,
                "best_epoch": best_epoch,
                "best_val_mAP": best_val_map,
            }
        )

    payload = {
        "round_id": round_id,
        "manifest": os.path.abspath(args.manifest),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }

    out_path = args.output
    if not out_path:
        out_dir = get_default_results_dir()
        ensure_dir(out_dir)
        out_path = os.path.join(out_dir, f"{round_id}.results.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(out_path)


if __name__ == "__main__":
    main()
