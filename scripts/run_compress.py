#!/usr/bin/env python
"""Pass 3: apply a budgets file to a model and evaluate it.

Consumes the output of `allocate.py`. Reuses the Hessian cache written by
`collect_stats.py --save-hessians` when one is available, which turns a sweep
of budget points into a sequence of cheap runs.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from qlr.compress import LayerBudget, compress_layer  # noqa: E402
from qlr.data import get_calibration, get_eval_tokens  # noqa: E402
from qlr.eval import perplexity, zero_shot  # noqa: E402
from qlr.hessian import collect_block_hessians  # noqa: E402
from qlr.linalg import safe_eigh  # noqa: E402
from qlr.models import BlockRunner, linear_layers, load_model  # noqa: E402
from qlr.runlog import RunDir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--budgets", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--hessian-dir", default=None, help="cache from collect_stats.py")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib-dataset", default="wikitext2")
    ap.add_argument("--damp-frac", type=float, default=0.01)
    ap.add_argument("--tasks", nargs="*", default=[])
    ap.add_argument("--task-limit", type=int, default=None)
    ap.add_argument("--skip-ppl", action="store_true")
    args = ap.parse_args()

    run = RunDir(args.run_dir)
    run.config(vars(args))
    run.env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.budgets) as fh:
        spec = json.load(fh)
    budgets = {
        k: LayerBudget(
            bits=v["bits"], rank=v["rank"], beta=v.get("beta", 1.0),
            method=v.get("method", "gilora"), n_loops=v.get("n_loops", 0),
        )
        for k, v in spec["budgets"].items()
    }

    bundle = load_model(args.model, seqlen=args.seqlen)
    hcache = pathlib.Path(args.hessian_dir) if args.hessian_dir else None
    runner = BlockRunner(bundle, device)

    record: dict = {}
    if hcache and hcache.is_dir():
        # No forward passes needed at all: every Hessian is on disk and the
        # weights are compressed in place, block by block.
        print(f"using cached Hessians from {hcache}", flush=True)
        for bi, block in enumerate(runner.blocks):
            for name, mod in linear_layers(block).items():
                key = f"{bi}.{name}"
                if key not in budgets:
                    continue
                record[key] = _do_layer(
                    torch.load(hcache / f"{key}.pt", map_location=device, weights_only=True),
                    mod, budgets[key], args.damp_frac, device,
                )
            print(f"block {bi} done", flush=True)
    else:
        calib = get_calibration(
            bundle.tokenizer, args.n_calib, args.seqlen, args.seed, args.calib_dataset
        )
        hidden, kwargs = runner.capture_inputs(calib)
        for bi, block in enumerate(runner.blocks):
            block.to(device)
            layers = linear_layers(block)
            Hs, next_hidden = collect_block_hessians(block, layers, hidden, kwargs, device)
            for name, mod in layers.items():
                key = f"{bi}.{name}"
                if key not in budgets:
                    continue
                record[key] = _do_layer(Hs[name], mod, budgets[key], args.damp_frac, device)
            hidden = next_hidden
            block.to("cpu")
            del Hs
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"block {bi} done", flush=True)

    metrics = {
        "bits_per_weight": spec.get("achieved_bits_per_weight"),
        "surrogate_distortion": spec.get("surrogate_distortion"),
        "coding_gain": spec.get("coding_gain"),
        "mode": spec.get("mode"),
        "n_layers_compressed": len(record),
        "mean_rel_layer_error": (
            sum(r["rel_error"] for r in record.values()) / max(len(record), 1)
        ),
    }

    bundle.model.to(device)
    if not args.skip_ppl:
        tokens = get_eval_tokens(bundle.tokenizer, args.seqlen)
        metrics["wikitext2_ppl"] = perplexity(bundle.model, tokens, device)
        print(f"wikitext2 ppl = {metrics['wikitext2_ppl']:.4f}", flush=True)
    for task in args.tasks:
        res = zero_shot(bundle.model, bundle.tokenizer, task, device, args.task_limit)
        metrics[f"{task}_acc"] = res["acc"]
        metrics[f"{task}_acc_norm"] = res["acc_norm"]
        print(f"{task}: acc={res['acc']:.4f} acc_norm={res['acc_norm']:.4f}", flush=True)

    with open(run.artifact("layer_errors.json"), "w") as fh:
        json.dump(record, fh, indent=2)
    run.metrics(metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k != "coding_gain"}, indent=2))


def _do_layer(H, mod, budget, damp_frac, device):
    H = H.to(device=device, dtype=torch.float32)
    W = mod.weight.data.t().contiguous().to(device=device, dtype=torch.float32)
    eig = None
    if budget.method == "gilora" and budget.rank > 0:
        _, V = safe_eigh(H)
        eig = V.flip(-1)
    What, rec = compress_layer(H, W, budget, damp_frac, eigvecs=eig)
    mod.weight.data.copy_(What.t().to(mod.weight.dtype).to(mod.weight.device))
    del H, W, What, eig
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rec


if __name__ == "__main__":
    main()
