# Program: Spot Temporal-Head Autoresearch

## Objective

- Primary metric: `val_mAP` (maximize).
- Baseline: `feature_arch=rny008_gsm`, `temporal_arch=gru`.
- Dataset: `soccernetv2`.

## Allowed Changes

- `spot/model/modules.py`
- `spot/train_e2e.py` temporal-head registry path only
- Head-local hyperparameters (e.g. hidden dim, layers, dropout for the head)

## Forbidden Changes

- Backbone architecture and backbone weights policy
- Dataset preparation, labels, and frame sampling policy
- Evaluation metric definition
- Non-temporal post-processing logic

## Run Policy

- Keep all non-head settings fixed within the same round unless explicitly
  stated in manifest.
- Every run must have an isolated `save_dir`.
- Every run must record `loss.json` and `config.json`.

## Keep/Discard Rule

- Keep candidate if `best_val_mAP` > baseline `best_val_mAP`.
- Discard if lower.
- Mark as review if delta <= 0.001.

## Stop Criteria

- Stop when no improvement is found in two consecutive rounds, or when planned
  round budget is reached.
