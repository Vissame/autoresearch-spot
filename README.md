# autoresearch-spot

This directory is the concrete autoresearch adapter for the `spot` project.

Target repo in local workspace:

- `/data/lyx/spot`

Research objective:

- Improve `val_mAP` by changing only temporal head structure/selection.

Fixed data path:

- `/data/lyx/spot/frames/soccernetv2`

How to use (minimal):

1. Review and adjust `program.md`.
2. Pick a manifest in `manifests/`.
3. Generate train commands:
   - `python3 scripts/gen_commands.py --manifest manifests/round_001_existing_heads.yaml`
4. Run generated commands manually.
5. Collect metrics:
   - `python3 scripts/collect_val_map.py --manifest manifests/round_001_existing_heads.yaml`
6. Build leaderboard:
   - `python3 scripts/build_leaderboard.py --manifest manifests/round_001_existing_heads.yaml --append`

References:

- https://github.com/karpathy/autoresearch
- https://github.com/jhong93/spot
- https://github.com/jhong93/e2e-spot-models
- https://github.com/ChinaYi/ASFormer
- https://github.com/sj-li/MS-TCN2
