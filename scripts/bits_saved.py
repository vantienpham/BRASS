#!/usr/bin/env python
"""Bits per weight saved by allocation, measured against the baseline envelope.

The honest comparison is not against one uniform configuration but against the
best of them at each budget. This script builds the lower envelope of every
non-allocated operating point in perplexity-versus-rate space, then asks, for
each allocated point, how much rate the envelope needs to reach the same
perplexity. The difference is the saving.

Interpolation is linear in (bits per weight, log perplexity), which is the
scale on which the high-resolution model of the coding-gain theorem is linear -- so the
measured saving and the predicted one are read off the same axes.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from qlr.analysis import BASELINE_FAMILIES, envelope, envelope_saving, is_allocated  # noqa: E402


def load(path) -> list[dict]:
    return json.load(open(path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--metric", default="wikitext2_ppl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    summary = {}
    for path in args.results:
        p = pathlib.Path(path)
        if not p.exists():
            continue
        rows = load(p)
        # Every configuration a practitioner could pick without allocating.
        base = envelope(rows, args.metric, BASELINE_FAMILIES)
        if len(base) < 2:
            print(f"{p}: not enough baseline points")
            continue
        bx = np.array([x for x, _ in base])
        by = np.array([y for _, y in base])
        print(f"\n===== {p.parent.name} =====")
        print(f"baseline envelope: {len(base)} points, "
              f"{bx.min():.2f}-{bx.max():.2f} bits/weight")
        print(f"{'allocated point':<18}{'bpw':>8}{'PPL':>10}{'saved':>9}"
              f"{'':<5}{'predicted':>9}")
        saved, pred = [], []
        for r in sorted((r for r in rows if is_allocated(r["name"])),
                        key=lambda r: r["bits_per_weight"]):
            if args.metric not in r:
                continue
            ly = math.log(r[args.metric])
            s, status = envelope_saving(base, r["bits_per_weight"], ly)
            if s is None:
                continue
            pr = r.get("coding_gain", {}).get("bits_saved")
            saved.append(s)
            if pr is not None:
                pred.append(pr)
            mark = " (>=)" if status == "censored" else ""
            print(f"{r['name']:<18}{r['bits_per_weight']:>8.3f}{r[args.metric]:>10.3f}"
                  f"{s:>9.3f}{mark:<5}{(f'{pr:.3f}' if pr else '---'):>9}")
        if saved:
            print(f"\n  measured saving: median {np.median(saved):.3f}, "
                  f"range {min(saved):.3f} to {max(saved):.3f} bits/weight")
            if pred:
                ms, mp = float(np.median(saved)), float(np.median(pred))
                # A near-zero measured median makes the ratio meaningless rather
                # than large; say so instead of printing 1e9.
                ratio = (f"{mp / ms:.2f}x optimistic" if ms > 0.02
                         else "measured median too close to zero for a ratio")
                print(f"  coding gain predicts: median {mp:.3f} bits/weight ({ratio})")
            summary[p.parent.name] = {
                "measured_median": float(np.median(saved)),
                "measured_min": float(min(saved)), "measured_max": float(max(saved)),
                "predicted_median": float(np.median(pred)) if pred else None,
            }
    if args.out and summary:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
