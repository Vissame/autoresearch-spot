# Spot Structure Reference

Relevant local files:

- `/data/lyx/spot/train_e2e.py`
- `/data/lyx/spot/model/modules.py`

Current high-level model path:

1. Frame clip input `[B, T, C, H, W]`
2. 2D backbone feature extraction
3. Optional temporal shift injection (`_tsm`/`_gsm`)
4. Temporal head (`gru`, `deeper_gru`, `mstcn`, `asformer`, `fc`)
5. Frame-level logits and training with cross-entropy

Autoresearch edit surface for this phase:

- temporal head only
