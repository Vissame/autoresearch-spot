# Metric Definition

Primary optimization target:

- `best_val_mAP` from `loss.json` of each run.

In-code references:

- `train_e2e.py` computes `avg_mAP = mean(mAPs[1:])` during validation.
- `util/score.py` defaults tolerances to `[0, 1, 2, 4]`.

Interpretation:

- Stored `val_mAP` corresponds to the training-time average over tolerance
  indices 1..end (i.e., `1,2,4` with default settings).
