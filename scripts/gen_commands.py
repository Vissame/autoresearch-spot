#!/usr/bin/env python3
"""Generate train commands from a manifest."""

from __future__ import annotations

import argparse
import shlex
from typing import Dict, List

from lib import get_manifest_runs


def _bool_flag(value: bool, flag: str) -> List[str]:
    return [flag] if value else []


def build_train_cmd(train_args: Dict[str, object]) -> str:
    dataset = str(train_args["dataset"])
    frame_dir = str(train_args["frame_dir"])
    feature_arch = str(train_args["feature_arch"])
    temporal_arch = str(train_args.get("temporal_arch", "gru"))
    save_dir = str(train_args["save_dir"])

    cmd: List[str] = [
        "python3",
        "/data/lyx/spot/train_e2e.py",
        dataset,
        frame_dir,
        "-s",
        save_dir,
        "-m",
        feature_arch,
        "-t",
        temporal_arch,
    ]

    opt_keys = [
        ("modality", "--modality"),
        ("clip_len", "--clip_len"),
        ("batch_size", "--batch_size"),
        ("learning_rate", "--learning_rate"),
        ("num_epochs", "--num_epochs"),
        ("warm_up_epochs", "--warm_up_epochs"),
        ("criterion", "--criterion"),
        ("start_val_epoch", "--start_val_epoch"),
        ("dilate_len", "--dilate_len"),
        ("acc_grad_iter", "--acc_grad_iter"),
        ("fg_upsample", "--fg_upsample"),
        ("num_workers", "--num_workers"),
    ]
    for key, flag in opt_keys:
        if key in train_args and train_args[key] is not None:
            cmd.extend([flag, str(train_args[key])])

    cmd.extend(_bool_flag(bool(train_args.get("resume", False)), "--resume"))
    cmd.extend(
        _bool_flag(bool(train_args.get("gpu_parallel", False)), "-mgpu")
    )

    return " ".join(shlex.quote(x) for x in cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to round manifest (.yaml with JSON syntax)",
    )
    parser.add_argument(
        "--output",
        help="Optional output shell file. If omitted, print to stdout.",
    )
    args = parser.parse_args()

    round_id, runs = get_manifest_runs(args.manifest)
    lines: List[str] = [f"# round_id: {round_id}"]
    for run in runs:
        run_id = run["run_id"]
        train_args = run["train_args"]
        lines.append(f"# run_id: {run_id}")
        lines.append(build_train_cmd(train_args))
        lines.append("")

    text = "\n".join(lines).rstrip() + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
