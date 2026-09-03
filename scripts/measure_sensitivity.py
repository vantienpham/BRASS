#!/usr/bin/env python
"""Estimate the sensitivity weights omega_l by replaying real quantization error.

The isotropic-noise probe in `qlr.sensitivity` is principled but fragile in
practice. To stay inside the quadratic regime the perturbation must be small,
and a small perturbation produces a loss change that a half-precision forward
pass cannot resolve: on Qwen3-0.6B at 2% relative noise, 38% of layers returned
an estimate at or below the noise floor.

This estimator removes the problem by perturbing with the thing we actually
care about. Compress layer ``l`` alone at a reference configuration, measure

    omega_l = [ E(compressed layer l) - E(original) ] / D_l ,

where ``D_l = ||X_l W_l - X_l What_l||_F^2`` is the layer-wise distortion of
that very perturbation. Both numerator and denominator are measured, so no
scale has to be guessed; the perturbation is the size real compression
produces, hence far above numerical noise; and it points in the direction real
quantization error points, rather than an isotropic direction that no algorithm
produces.

The estimate is a ratio, so a global scale is irrelevant to the allocation --
only the relative values across layers matter. Passing two reference
configurations checks that: if the quadratic model holds, the ratio is
approximately configuration-independent, and the agreement between the two is
reported.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from qlr.compress import LayerBudget, compress_layer  # noqa: E402
from qlr.data import get_calibration  # noqa: E402
from qlr.gptq import layer_error  # noqa: E402
from qlr.models import block_list, linear_layers, load_model  # noqa: E402
from qlr.runlog import RunDir  # noqa: E402


def choose_reference(modules, hdir, model, tokens, device, damp_frac,
                     candidates=(4, 3, 2), sample=16, min_snr=10.0, seed=0):
    r"""Pick the probe bit-width automatically, by signal-to-noise ratio.

    The estimator needs one layer's compression to move the loss by more than
    the arithmetic can resolve. That resolution is not a guess: compressing a
    layer at 16 bits is lossless to well under a part in 1e5, so the spread
    of |dE| over such a null probe *is* the floor. We measure it on a
    stratified sample of layers, then take the \emph{least} aggressive
    candidate whose median |dE| clears ``min_snr`` times that floor ---
    least aggressive because the quadratic model behind omega is a local one,
    so a larger perturbation than necessary is a cost, not a benefit.

    Returns ``(bits, diagnostics)``. This replaces the manual 3-bit-then-2-bit
    switch we used before and makes the choice a function of the model rather
    than of the operator.
    """
    keys = sorted(modules, key=lambda k: (int(k.split(".")[0]), k))
    step = max(1, len(keys) // sample)
    probe_keys = keys[::step][:sample]
    base = probe_loss(model, tokens, device)

    def spread(bits):
        out = []
        for key in probe_keys:
            blob = torch.load(hdir / f"{key}.pt", map_location="cpu", weights_only=True)
            H = blob["H"].to(device=device, dtype=torch.float32)
            mod = modules[key]
            W = mod.weight.data.t().contiguous().to(device=device, dtype=torch.float32)
            original = mod.weight.data.clone()
            beta = (blob.get("betas") or {}).get(bits)
            bud = LayerBudget(bits=bits, rank=0, method="gptq",
                              beta=beta.to(device=device, dtype=torch.float32)
                              if beta is not None else 1.0)
            What, _ = compress_layer(H, W, bud, damp_frac)
            mod.weight.data.copy_(What.t().to(mod.weight.dtype))
            out.append(abs(probe_loss(model, tokens, device) - base))
            mod.weight.data.copy_(original)
            del H, W, What, blob
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return np.array(out)

    null = spread(16)
    floor = float(np.percentile(null, 95))
    diag = {"null_p95": floor, "n_sampled": len(probe_keys), "min_snr": min_snr,
            "candidates": {}}
    chosen = min(candidates)
    for b in sorted(candidates, reverse=True):          # least aggressive first
        med = float(np.median(spread(b)))
        diag["candidates"][b] = {"median_abs_dloss": med,
                                 "snr": med / max(floor, 1e-12)}
        if med >= min_snr * max(floor, 1e-12):
            chosen = b
            break
    diag["chosen_bits"] = chosen
    return chosen, diag


@torch.no_grad()
def probe_loss(model, tokens, device) -> float:
    total = 0.0
    for i in range(tokens.shape[0]):
        b = tokens[i : i + 1].to(device)
        total += float(model(b, labels=b).loss)
    return total / tokens.shape[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--stats", required=True, help="stats.pt to read")
    ap.add_argument("--out-stats", default=None,
                    help="where to write the annotated copy (default: <stats>.omega.pt). "
                         "Never overwrites the input, so a concurrent sweep reading it "
                         "cannot see a half-written file.")
    ap.add_argument("--hessian-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--probe-seqs", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ref", nargs=2, type=int, default=[3, 0], metavar=("BITS", "RANK"))
    ap.add_argument("--ref2", nargs=2, type=int, default=None, metavar=("BITS", "RANK"),
                    help="second reference configuration, for the consistency check")
    ap.add_argument("--damp-frac", type=float, default=0.01)
    ap.add_argument("--floor-frac", type=float, default=1e-3,
                    help="omega is clamped below at this fraction of the median "
                         "positive estimate. It exists because a strictly zero or "
                         "negative estimate would tell the allocator to spend no "
                         "bits at all on a layer, which no measurement at this "
                         "precision can justify, and because log2(omega) is taken "
                         "downstream. Layers at the clamp are counted and reported.")
    ap.add_argument("--auto-ref", action="store_true",
                    help="choose the probe bit-width by signal-to-noise ratio "
                         "against a 16-bit null probe, instead of --ref")
    ap.add_argument("--auto-ref-sample", type=int, default=16)
    ap.add_argument("--auto-ref-min-snr", type=float, default=10.0)
    args = ap.parse_args()

    run = RunDir(args.run_dir)
    run.config(vars(args))
    run.env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hdir = pathlib.Path(args.hessian_dir)

    bundle = load_model(args.model, seqlen=args.seqlen)
    tokens = get_calibration(bundle.tokenizer, args.probe_seqs, args.seqlen, args.seed + 7)
    modules = {}
    for bi, block in enumerate(block_list(bundle.model)):
        for name, mod in linear_layers(block).items():
            modules[f"{bi}.{name}"] = mod

    bundle.model.to(device)
    base = probe_loss(bundle.model, tokens, device)
    print(f"base loss = {base:.6f} over {args.probe_seqs} sequences", flush=True)

    auto_diag = None
    if args.auto_ref:
        b, auto_diag = choose_reference(
            modules, hdir, bundle.model, tokens, device, args.damp_frac,
            sample=args.auto_ref_sample, min_snr=args.auto_ref_min_snr, seed=args.seed)
        print(f"[auto-ref] null p95 |dE| = {auto_diag['null_p95']:.3e}; "
              + "; ".join(f"{k}-bit SNR {v['snr']:.1f}"
                          for k, v in sorted(auto_diag["candidates"].items(), reverse=True))
              + f"  -> chose {b}-bit", flush=True)
        args.ref = [b, 0]
    refs = [tuple(args.ref)] + ([tuple(args.ref2)] if args.ref2 else [])
    results: dict[str, dict] = {}

    for ri, (b0, r0) in enumerate(refs):
        print(f"--- reference b={b0} r={r0} ---", flush=True)
        for i, (key, mod) in enumerate(modules.items()):
            blob = torch.load(hdir / f"{key}.pt", map_location="cpu", weights_only=True)
            H = blob["H"].to(device=device, dtype=torch.float32)
            evecs = blob["evecs"].to(device=device, dtype=torch.float32)
            beta = (blob.get("betas") or {}).get(b0)
            W = mod.weight.data.t().contiguous().to(device=device, dtype=torch.float32)
            original = mod.weight.data.clone()

            bud = LayerBudget(
                bits=b0, rank=r0,
                beta=beta.to(device=device, dtype=torch.float32) if beta is not None else 1.0,
                method="gilora" if r0 > 0 else "gptq",
            )
            What, _ = compress_layer(H, W, bud, args.damp_frac, eigvecs=evecs)
            D = layer_error(H, W, What)
            mod.weight.data.copy_(What.t().to(mod.weight.dtype))
            loss = probe_loss(bundle.model, tokens, device)
            mod.weight.data.copy_(original)

            results.setdefault(key, {})[f"ref{ri}"] = {
                "delta_loss": loss - base, "D": D,
                "omega": (loss - base) / D if D > 0 else 0.0,
            }
            del H, evecs, W, What, blob
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if (i + 1) % 28 == 0:
                print(f"  {i+1}/{len(modules)} layers", flush=True)

    # --- consistency between the two reference configurations ----------------
    metrics = {"base_loss": base, "refs": [list(r) for r in refs],
               "floor_frac": args.floor_frac, "auto_ref": auto_diag}
    if len(refs) == 2:
        a = np.array([results[k]["ref0"]["omega"] for k in results])
        b = np.array([results[k]["ref1"]["omega"] for k in results])
        ok = (a > 0) & (b > 0)
        d = np.abs(np.log2(a[ok]) - np.log2(b[ok]))
        rho = float(np.corrcoef(np.log2(a[ok]), np.log2(b[ok]))[0, 1])
        metrics["consistency"] = {
            "n": int(ok.sum()), "pearson_log2": rho,
            "median_abs_log2_ratio": float(np.median(d)),
            "p90_abs_log2_ratio": float(np.percentile(d, 90)),
        }
        print(f"\nconsistency across references: n={ok.sum()} r={rho:.3f} "
              f"median |log2 ratio| = {np.median(d):.2f} bits", flush=True)

    # --- write omega back into stats.pt --------------------------------------
    om = {k: max(v["ref0"]["omega"], 0.0) for k, v in results.items()}
    pos = np.array([v for v in om.values() if v > 0])
    floor = args.floor_frac * float(np.median(pos))
    n_floor = sum(1 for v in om.values() if v < floor)
    stats = torch.load(args.stats, map_location="cpu", weights_only=False)
    for k in stats:
        if k in om:
            stats[k]["omega"] = max(om[k], floor)
    out_stats = args.out_stats or str(args.stats) + ".omega.pt"
    torch.save(stats, out_stats)

    lo = np.log2(np.array([max(v, floor) for v in om.values()]))
    metrics |= {
        "n_layers": len(om), "n_floored": n_floor,
        "log2_omega_sd": float(lo.std()),
        "log2_omega_iqr": float(np.subtract(*np.percentile(lo, [75, 25]))),
        "log2_omega_range": float(lo.max() - lo.min()),
    }
    with open(run.artifact("sensitivity.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    run.metrics(metrics)
    print(f"\nfloored {n_floor}/{len(om)} layers; log2 omega sd={lo.std():.2f} "
          f"IQR={np.subtract(*np.percentile(lo,[75,25])):.2f} range={lo.max()-lo.min():.2f}")
    print(f"wrote {out_stats}")


if __name__ == "__main__":
    main()
