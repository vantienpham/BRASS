#!/usr/bin/env bash
# Regenerate every table and figure in the paper from the released run outputs.
#
#   bash scripts/reproduce_tables.sh [model ...]
#
# Runs entirely on CPU against out/runs/ and results/tables/; it recomputes
# nothing and needs no GPU, no cluster and no model download.

set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=("$@")
[[ ${#MODELS[@]} -eq 0 ]] && MODELS=(0.6B 1.7B llama3b 8B)

echo "=== tables and figures -> paper/ ==="
python3 scripts/make_figures.py --models "${MODELS[@]}"
python3 scripts/make_tables.py  --models "${MODELS[@]}"

echo
echo "=== saving against the baseline envelope ==="
RES=()
for m in "${MODELS[@]}"; do
  for d in "out/runs/sweep-q3-$m" "out/runs/sweep-$m"; do
    [[ -f "$d/results.json" ]] && RES+=("$d/results.json")
  done
done
(( ${#RES[@]} )) && python3 scripts/bits_saved.py "${RES[@]}"
