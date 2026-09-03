#!/bin/bash
# One seed of the full Phase-1 + sweep chain on Qwen3-0.6B-Base, so the
# manuscript can report run-to-run variability rather than asserting determinism.
set -euo pipefail
S=$1
M=Qwen/Qwen3-0.6B-Base
uv run --no-sync python scripts/collect_stats.py --model $M --seed $S \
  --save-hessians --sensitivity --run-dir out/runs/stats-s$S-q3-0.6B
uv run --no-sync python scripts/measure_sensitivity.py --model $M --seed $S \
  --stats out/runs/stats-s$S-q3-0.6B/stats.pt \
  --hessian-dir out/runs/stats-s$S-q3-0.6B/hessians \
  --run-dir out/runs/omega-s$S-q3-0.6B
uv run --no-sync python scripts/sweep.py --model $M --seed $S \
  --stats out/runs/stats-s$S-q3-0.6B/stats.pt.omega.pt \
  --hessian-dir out/runs/stats-s$S-q3-0.6B/hessians \
  --plan configs/sweep_seed.json --run-dir out/runs/sweep-s$S-q3-0.6B
