#!/usr/bin/env python
"""Run a whole Pareto frontier inside one job.

The cluster caps concurrent jobs per user (`AssocMaxJobsLimit`, 12), so the
efficient shape is few long jobs rather than many short ones. This script loads
the model once, keeps a pristine CPU copy of every weight it will touch, and
then for each budget point restores, compresses and evaluates.

It consumes the Hessian cache from ``collect_stats.py --save-hessians``, so no
calibration forward pass happens here at all: each budget point is
compress-and-evaluate only.

A configuration is a dict in the JSON passed to ``--plan``:

    {"name": "...", "mode": "uniform"|"lagrangian", "bits": 3, "rank": 32,
     "method": "gilora", "n_loops": 0, "target_bits": 3.29, "no_omega": false}

Results append to ``metrics.json`` as they finish, so a wall-time kill leaves
every completed point on disk -- a resubmit skips them by name.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import pathlib
import sys
import time

from dataclasses import replace

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from qlr.alloc import (  # noqa: E402
    LayerStats, allocate_lagrangian, allocate_uniform, coding_gain,
    fit_rank_calibration,
)
from qlr.compress import LayerBudget, compress_layer  # noqa: E402
from qlr.quant import search_beta  # noqa: E402
from qlr.rotate import Rotation  # noqa: E402
from qlr.data import get_eval_tokens  # noqa: E402
from qlr.eval import perplexity, zero_shot  # noqa: E402
from qlr.models import linear_layers, load_model  # noqa: E402
from qlr.models import block_list  # noqa: E402
from qlr.runlog import RunDir  # noqa: E402


def build_layer_stats(stats_path, damp_frac, use_omega) -> list[LayerStats]:
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    out = []
    for key, s in stats.items():
        eigs = np.asarray(s["eigs"], dtype=np.float64)
        out.append(
            LayerStats(
                name=key, N=int(s["N"]), Nout=int(s["Nout"]), S=float(s["S"]),
                S_bits={int(k): float(v) for k, v in (s.get("S_bits") or {}).items()} or None,
                eigs=eigs, damp=damp_frac * float(eigs.sum()) / len(eigs),
                omega=float(s.get("omega", 1.0)) if use_omega else 1.0,
            )
        )
    return out


def resolve_budgets(cfg, layers, lr_bits, bit_choices, rank_choices):  # noqa: C901
    """Turn one plan entry into a per-layer budget map plus its bookkeeping.

    A plan entry may narrow the search with ``bit_choices``/``rank_choices``.
    Pinning one of them to a single value isolates that knob: pinning the ranks
    gives a bits-only allocation, pinning the bit-widths a rank-only one, which
    is how the ablations attribute the gain.
    """
    bit_choices = tuple(cfg.get("bit_choices", bit_choices))
    rank_choices = list(cfg.get("rank_choices", rank_choices))
    # If the factors are stored in low precision, they must be *priced* in low
    # precision too, or the comparison silently charges FP16 for a 4-bit factor.
    if cfg.get("L_bits") or cfg.get("R_bits"):
        lr_bits = 0.5 * (cfg.get("L_bits", lr_bits) + cfg.get("R_bits", lr_bits))
    if cfg["mode"] == "uniform":
        alloc = allocate_uniform(layers, cfg["bits"], cfg.get("rank", 0), lr_bits)
    elif cfg["mode"] == "lagrangian":
        target = cfg.get("target_bits")
        if target is None:
            ref = allocate_uniform(layers, cfg["bits"], cfg.get("rank", 0), lr_bits)
            target = ref.bits_per_weight
        alloc = allocate_lagrangian(layers, target, bit_choices, rank_choices, lr_bits)
    else:
        raise ValueError(f"unknown mode {cfg['mode']!r}")

    # beta defaults to "stored": use the per-column clipping factors that
    # collect_stats.py searched for this layer at this bit-width, which are the
    # same factors the allocator priced through S(b). A plan entry can set a
    # numeric beta instead, which is the no-search ablation.
    budgets = {
        st.name: LayerBudget(
            bits=int(b), rank=int(r), beta=cfg.get("beta", "stored"),
            method=cfg.get("method", "gilora"), n_loops=cfg.get("n_loops", 0),
            L_bits=cfg.get("L_bits"), R_bits=cfg.get("R_bits"),
        )
        for st, b, r in zip(layers, alloc.bits, alloc.ranks)
    }
    info = {
        "bits_per_weight": alloc.bits_per_weight,
        "surrogate_distortion": alloc.total_distortion,
        "nu": alloc.nu,
        "gap_bound": alloc.gap_bound,
        "coding_gain": coding_gain(layers, alloc.ranks),
        "bit_histogram": {
            int(k): int(v) for k, v in zip(*np.unique(alloc.bits, return_counts=True))
        },
        "rank_histogram": {
            int(k): int(v) for k, v in zip(*np.unique(alloc.ranks, return_counts=True))
        },
    }
    return budgets, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--stats", required=True, help="stats.pt from collect_stats.py")
    ap.add_argument("--hessian-dir", required=True)
    ap.add_argument("--plan", required=True, help="json list of configurations")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--damp-frac", type=float, default=0.01)
    ap.add_argument("--lr-bits", type=float, default=16.0)
    ap.add_argument("--bit-choices", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8])
    ap.add_argument("--rank-choices", type=int, nargs="+",
                    default=[0, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256])
    ap.add_argument("--tasks", nargs="*", default=[])
    ap.add_argument("--task-limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank-calibration", default=None,
                    help="profile.json from profile_layers.py. Fits one kappa per "
                         "rank and shrinks the bound's rank dependence to the "
                         "measured one; plan entries opt in with "
                         "\"rank_calibrated\": true.")
    ap.add_argument("--rotate-block", type=int, default=1,
                    help="must match the value collect_stats.py used, since the "
                         "cached Hessians are already in the rotated basis")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    run = RunDir(args.run_dir)
    run.config(vars(args))
    run.env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hdir = pathlib.Path(args.hessian_dir)

    with open(args.plan) as fh:
        plan = json.load(fh)
    results_path = run.artifact("results.json")
    # Two sweeps sharing a run directory each load results.json at start and
    # write their own union at the end, so the later one silently erases the
    # other's points. Refuse rather than corrupt.
    lock = run.artifact("results.lock")
    if lock.exists():
        raise SystemExit(
            f"error: {lock} exists -- another sweep is writing {results_path}.\n"
            "       Use a separate --run-dir, or remove the lock if the owning "
            "job is known to be dead.")
    lock.write_text(f"{os.environ.get('SLURM_JOB_ID', os.getpid())}\n")
    atexit.register(lambda: lock.unlink(missing_ok=True))
    done = {}
    if args.resume and results_path.exists():
        done = {r["name"]: r for r in json.load(open(results_path))}
        print(f"resuming: {len(done)} point(s) already on disk", flush=True)

    bundle = load_model(args.model, seqlen=args.seqlen)
    blocks = block_list(bundle.model)
    modules = {}
    for bi, block in enumerate(blocks):
        for name, mod in linear_layers(block).items():
            modules[f"{bi}.{name}"] = mod
    # Pristine copy on CPU. Restored before every budget point, so a point never
    # sees weights another point already compressed.
    pristine = {k: m.weight.data.detach().clone().cpu() for k, m in modules.items()}
    print(f"{len(modules)} linear layers; pristine copy held on CPU", flush=True)

    eval_tokens = get_eval_tokens(bundle.tokenizer, args.seqlen)
    bundle.model.to(device)
    if "fp16" not in done:
        base_ppl = perplexity(bundle.model, eval_tokens, device)
        rec = {"name": "fp16", "mode": "none", "wikitext2_ppl": base_ppl,
               "bits_per_weight": 16.0}
        # The uncompressed reference needs the same task scores as every other
        # row, or the table has a hole exactly where the comparison anchors.
        for task in sorted({t for cfg in plan for t in cfg.get("tasks", args.tasks)}):
            res = zero_shot(bundle.model, bundle.tokenizer, task, device, args.task_limit)
            rec[f"{task}_acc"] = res["acc"]
            rec[f"{task}_acc_norm"] = res["acc_norm"]
            print(f"[fp16] {task}: acc_norm={res['acc_norm']:.4f}", flush=True)
        done["fp16"] = rec
        print(f"[fp16] wikitext2 ppl = {base_ppl:.4f}", flush=True)
        _dump(results_path, done)
    bundle.model.to("cpu")

    layers_all = build_layer_stats(args.stats, args.damp_frac, use_omega=True)
    layers_no_omega = build_layer_stats(args.stats, args.damp_frac, use_omega=False)
    layers_cal = None
    if args.rank_calibration:
        with open(args.rank_calibration) as fh:
            cal = fit_rank_calibration(json.load(fh), layers_all)
        layers_cal = build_layer_stats(args.stats, args.damp_frac, use_omega=True)
        for st in layers_cal:
            st.rank_cal = cal
        print(f"rank calibration: kappa = "
              f"{ {k: round(v, 3) for k, v in sorted(cal.items())} }", flush=True)

    for cfg in plan:
        name = cfg["name"]
        if name in done:
            print(f"[{name}] already done, skipping", flush=True)
            continue
        t0 = time.time()
        if cfg.get("rank_calibrated"):
            if layers_cal is None:
                raise SystemExit(f"[{name}] needs --rank-calibration")
            layers = layers_cal
        else:
            layers = layers_no_omega if cfg.get("no_omega") else layers_all
        budgets, info = resolve_budgets(
            cfg, layers, args.lr_bits, tuple(args.bit_choices), args.rank_choices
        )
        print(f"[{name}] {info['bits_per_weight']:.4f} bits/weight  "
              f"bits={info['bit_histogram']}  ranks={info['rank_histogram']}", flush=True)

        for key, mod in modules.items():
            mod.weight.data.copy_(pristine[key].to(mod.weight.device))

        errs = {}
        for key, mod in modules.items():
            if key not in budgets:
                continue
            blob = torch.load(hdir / f"{key}.pt", map_location="cpu", weights_only=True)
            H = blob["H"].to(device=device, dtype=torch.float32)
            evecs = blob["evecs"].to(device=device, dtype=torch.float32)
            W = mod.weight.data.t().contiguous().to(device=device, dtype=torch.float32)
            # The cached H is already rotated; rotate W into the same basis, and
            # map the compressed result back at the end. Theta is deterministic
            # in (width, seed), so both passes build the identical matrix.
            rot = None
            if args.rotate_block > 1:
                rot = Rotation(W.shape[0], args.rotate_block, seed=args.seed,
                               device=device, dtype=torch.float32)
                W = rot.left_T(W)
            bud = budgets[key]
            if bud.beta == "stored":
                stored = (blob.get("betas") or {}).get(bud.bits)
                bud = replace(
                    bud,
                    beta=stored.to(device=device, dtype=torch.float32)
                    if stored is not None
                    else search_beta(H, W, bud.bits),
                )
            What, rec = compress_layer(H, W, bud, args.damp_frac, eigvecs=evecs)
            if rot is not None:
                What = rot.left(What)
            mod.weight.data.copy_(What.t().to(mod.weight.dtype))
            errs[key] = rec
            del H, evecs, W, What, blob, rot
        if device.type == "cuda":
            torch.cuda.empty_cache()

        bundle.model.to(device)
        rec = {"name": name, **{k: v for k, v in cfg.items() if k != "name"}, **info}
        rec["wikitext2_ppl"] = perplexity(bundle.model, eval_tokens, device)
        rec["mean_rel_layer_error"] = float(np.mean([e["rel_error"] for e in errs.values()]))
        for task in cfg.get("tasks", args.tasks):
            res = zero_shot(bundle.model, bundle.tokenizer, task, device, args.task_limit)
            rec[f"{task}_acc"] = res["acc"]
            rec[f"{task}_acc_norm"] = res["acc_norm"]
        bundle.model.to("cpu")
        rec["seconds"] = time.time() - t0
        done[name] = rec
        _dump(results_path, done)
        with open(run.artifact(f"layer_errors_{name}.json"), "w") as fh:
            json.dump(errs, fh)
        print(f"[{name}] ppl={rec['wikitext2_ppl']:.4f}  "
              f"bpw={info['bits_per_weight']:.4f}  ({rec['seconds']:.0f}s)", flush=True)

    run.metrics({"n_points": len(done), "points": list(done)})
    print(json.dumps(
        {k: {kk: v[kk] for kk in ("wikitext2_ppl", "bits_per_weight") if kk in v}
         for k, v in done.items()}, indent=2))


def _dump(path, done: dict) -> None:
    with open(path, "w") as fh:
        json.dump(list(done.values()), fh, indent=2, default=str)


if __name__ == "__main__":
    main()
