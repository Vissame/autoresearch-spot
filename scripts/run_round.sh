#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <manifest_path>"
  exit 1
fi

MANIFEST="$1"
ROOT="/data/lyx/autoresearch-spot"
CMD_FILE="${ROOT}/results/$(basename "${MANIFEST%.*}").commands.sh"

python3 "${ROOT}/scripts/gen_commands.py" --manifest "$MANIFEST" --output "$CMD_FILE"
chmod +x "$CMD_FILE"

echo "Generated commands:"
echo "  $CMD_FILE"
echo
echo "Review the commands, then run:"
echo "  bash $CMD_FILE"
echo
echo "After training:"
echo "  python3 ${ROOT}/scripts/collect_val_map.py --manifest $MANIFEST"
echo "  python3 ${ROOT}/scripts/build_leaderboard.py --manifest $MANIFEST --append"
