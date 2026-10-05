# BRASS — Bit–Rank Allocation for Low-Precision Plus Low-Rank Compression of Large Language Models

Code and experimental results for *Bit–Rank Allocation for Low-Precision Plus Low-Rank
Compression of Large Language Models*.

Post-training compression increasingly represents a weight matrix `W` as a
low-precision part `Q` plus a low-rank correction `LR`, and every method in that
line applies **one** bit-width and **one** rank to the whole network. BRASS
prices bits and rank in a single currency of *effective bit-width* and allocates
both per layer, by solving a separable rate–distortion problem with a Lagrangian
sweep. It is training-free: no backpropagation, no labelled data, and its cost
is dominated by one eigendecomposition per layer, which a GPTQ pass performs
anyway.

## Reproducing the paper without a GPU

Every table and figure is regenerated from the run outputs committed here. This
recomputes nothing, downloads no models, and needs no cluster:

```bash
pip install -e .            # or: uv sync  (tests also run without installing)
bash scripts/reproduce_tables.sh
```

Output lands in `paper/tables/` and `paper/figures/`. The tables are
byte-identical to those in the submitted manuscript.

## What is in the repository

| path | contents |
|---|---|
| `src/qlr/` | the library: allocator, distortion model, GPTQ + low-rank compression, sensitivity estimator, analysis |
| `scripts/` | drivers, one concern each: statistics, sensitivity, sweep, profiling, tables, figures |
| `configs/` | the operating-point plans each sweep ran |
| `out/runs/` | **the released results**: 434 operating points across four models, plus per-run configuration and metrics |
| `results/tables/` | derived summaries (layer statistics, profiled distortion surfaces, envelope savings) |
| `tests/` | unit tests, including the allocator against brute force |
| `slurm/` | job runner, environment build and the sync script, for running the pipeline yourself |

### The released runs

| sweep | operating points |
|---|---|
| `sweep-q3-0.6B` | 113 |
| `sweep-q3-1.7B` | 105 |
| `sweep-llama3b` | 105 |
| `sweep-q3-8B` | 85 |
| `sweep-s1/s2-q3-0.6B` | 13 each, independent calibration draws |

Each `results.json` records, for every operating point, the effective bits per
weight, WikiText-2 perplexity, zero-shot accuracies where evaluated, the
allocated bit-width and rank histograms, the surrogate distortion, the
Lagrangian multiplier and the suboptimality bound.

## The method in four objects

`src/qlr/alloc.py` is the core. A layer is summarised by a `LayerStats`
carrying the quantities the distortion model needs:

```
D̂_ℓ(b, r) = ω_ℓ · g_ℓ(b) · h_ℓ(r)
```

`g_ℓ` falls with the bit-width and is read off the per-channel weight ranges;
`h_ℓ` falls with the rank and is a partial sum of the calibration Hessian's
eigenvalues; `ω_ℓ` measures how much the network's loss moves when the layer is
perturbed, and is estimated once by replaying real compression error
(`src/qlr/sensitivity.py`).

`allocate_lagrangian` bisects a single multiplier `ν`; each layer independently
reads its best `(b, r)` off a precomputed table, which is exact on the lower
convex hull of its achievable set. `coding_gain` evaluates the closed-form
arithmetic-to-geometric mean ratio, and `continuous_waterfill` the relaxed
optimum. The whole solve is milliseconds on a CPU.

## Running the pipeline from scratch

Needs one GPU. On a Slurm cluster, copy `cluster.example.md` to
`cluster.local.md`, fill in the values, and set the three variables at the top
of `slurm/sync.sh`.

```bash
# 1. per-layer statistics: Hessians, spectra, clipping search
python scripts/collect_stats.py --model Qwen/Qwen3-0.6B-Base \
    --save-hessians --sensitivity --run-dir out/runs/stats-q3-0.6B

# 2. sensitivity weights by error replay
python scripts/measure_sensitivity.py --model Qwen/Qwen3-0.6B-Base \
    --stats out/runs/stats-q3-0.6B/stats.pt \
    --hessian-dir out/runs/stats-q3-0.6B/hessians \
    --run-dir out/runs/omega-q3-0.6B

# 3. allocate and compress at every operating point in a plan
python scripts/sweep.py --model Qwen/Qwen3-0.6B-Base \
    --stats out/runs/stats-q3-0.6B/stats.pt.omega.pt \
    --hessian-dir out/runs/stats-q3-0.6B/hessians \
    --plan configs/sweep_main.json --run-dir out/runs/sweep-q3-0.6B
```

Step 1 is paid once per model and its Hessians are independent of the
compression budget, so one pass serves an entire rate–distortion sweep; each
additional operating point costs one compression. Wall-clock on one A100 ranges
from about 7 minutes of preparation for the 0.6B model to 68 for the 8B, with
0.9 to 7 minutes per operating point.

## Results, in short

Across three Qwen3 base models and Llama-3.2-3B, at matched effective
bit-width, allocation improves on the uniform configuration at 27 of 32
budgets, by up to 4.9× in perplexity where the budget binds. Measured instead
against the lower envelope of *all* uniform configurations, which is the
demanding reference, the median saving is 0.29, 0.11 and 0.03 bits per weight
on three models and about zero on the fourth.

The paper reports several results that qualify the framework, and the data for
each is in this repository. The error bound the distortion model is built on
overvalues rank, badly enough that the predicted coding gain is *anti*-correlated
with the realized saving. Allocating against the unweighted layer-wise
objective is unreliable rather than merely weaker. Separating the two decision
variables locates most of the gain in bit allocation alone. And quantizing the
low-rank factors is nearly free and absorbs much of what allocation would
otherwise find.

## Environment

Python 3.10–3.12, PyTorch < 2.7. The released runs used Python 3.12.13 and
PyTorch 2.6.0+cu124 on A100s. `scripts/verify_pin.py` launches real kernels to
check a GPU build actually works on the silicon you intend to use, which
`torch.cuda.is_available()` does not.

```bash
pytest tests/ -q                    # CPU tests
pytest tests/ -q -m gpu             # needs a GPU
```

## Citation

```bibtex
@misc{pham2026bitrank,
  title  = {Bit--Rank Allocation for Low-Precision Plus Low-Rank Compression of Large Language Models},
  author = {Pham, Van Tien and Gillis, Nicolas},
  year   = {2026}
}
```

The distortion model builds on the error bound for GPTQ-intrinsic LoRA of
Zhang and Saab, arXiv:2606.01412.
