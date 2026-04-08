# Temporal Heads

Built-in heads in local `spot` code:

- `""` (FC head)
- `gru`
- `deeper_gru`
- `mstcn`
- `asformer`

Round strategy:

- Round 0: baseline (`gru`)
- Round 1: built-in head sweep
- Round 2: minimal custom heads

Planned custom heads (phase-2):

- `gru_mlp`
- `tcn_lite`
