#!/usr/bin/env python3
"""Build or append leaderboard from manifest runs."""

from __future__ import annotations

import argparse
import csv
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


FIELDNAMES = [
    "run_id",
    "round_id",
    "status",
    "feature_arch",
    "temporal_arch",
    "best_epoch",
    "best_val_mAP",
    "artifact_dir",
]


def _collect_rows(manifest_path: str) -> List[Dict[str, Any]]:
    round_id, runs = get_manifest_runs(manifest_path)
    rows: List[Dict[str, Any]] = []
    for run in runs:
        train_args = run["train_args"]
        save_dir = str(train_args["save_dir"])
        status, best_epoch, best_val_map = extract_best_val_map(save_dir)
        rows.append(
            {
                "run_id": run["run_id"],
                "round_id": round_id,
                "status": status,
                "feature_arch": str(train_args.get("feature_arch", "")),
                "temporal_arch": str(train_args.get("temporal_arch", "")),
                "best_epoch": best_epoch,
                "best_val_mAP": f"{best_val_map:.6f}",
                "artifact_dir": save_dir,
            }
        )
    return rows


def _update_best_result(rows: List[Dict[str, Any]], out_dir: str) -> None:
    ok_rows = [r for r in rows if r["status"] == "ok"]
    if not ok_rows:
        return
    best = max(ok_rows, key=lambda r: float(r["best_val_mAP"]))
    best_payload = {
        "run_id": best["run_id"],
        "best_val_mAP": float(best["best_val_mAP"]),
        "best_epoch": int(best["best_epoch"]),
        "artifact_dir": best["artifact_dir"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(out_dir, "best_result.json"), "w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2, sort_keys=True)


def _append_history(rows: List[Dict[str, Any]], out_dir: str) -> None:
    hist_path = os.path.join(out_dir, "history.jsonl")
    with open(hist_path, "a", encoding="utf-8") as f:
        for row in rows:
            payload = dict(row)
            payload["time"] = datetime.now(timezone.utc).isoformat()
            f.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--leaderboard",
        default="/data/lyx/autoresearch-spot/results/leaderboard.csv",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append rows to existing leaderboard instead of overwrite.",
    )
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.leaderboard))
    ensure_dir(out_dir)

    rows = _collect_rows(args.manifest)
    mode = "a" if args.append and os.path.exists(args.leaderboard) else "w"
    need_header = mode == "w" or os.path.getsize(args.leaderboard) == 0

    with open(args.leaderboard, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if need_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

    _update_best_result(rows, out_dir)
    _append_history(rows, out_dir)

    # Also emit round-specific summary for quick inspection.
    round_id, _ = get_manifest_runs(args.manifest)
    summary_path = os.path.join(out_dir, f"{round_id}.results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"round_id": round_id, "results": rows}, f, indent=2, sort_keys=True)

    print(args.leaderboard)


if __name__ == "__main__":
    main()
