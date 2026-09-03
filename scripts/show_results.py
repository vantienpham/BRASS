#!/usr/bin/env python
"""Compact view of a sweep's results.json, grouped by matched budget.

Prints every operating point sorted by effective bit-width, and then the
paired comparisons the paper reports: at each uniform reference configuration,
what the same storage buys when it is allocated instead.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re


def num(v) -> str:
    """Format a metric, keeping divergence visible rather than printing 'nan'."""
    if v is None:
        return "---"
    if isinstance(v, str):
        return v
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "inf"
    return f"{v:.3f}" if abs(v) < 1e5 else f"{v:.2e}"


def fam(name):
    m = re.match(r"([A-Za-z]+)", name)
    return m.group(1) if m else name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--metric", default="wikitext2_ppl")
    args = ap.parse_args()

    for path in args.results:
        p = pathlib.Path(path)
        if not p.exists():
            print(f"-- {p}: missing")
            continue
        rows = json.load(open(p))
        print(f"\n===== {p} : {len(rows)} points =====")
        print(f"{'name':<20}{'bpw':>8}{'PPL':>14}{'bits hist':>34}{'rank med':>10}")
        for r in sorted(rows, key=lambda r: (r.get("bits_per_weight", 0), r["name"])):
            bh = r.get("bit_histogram", {})
            bhs = ",".join(f"{k}:{v}" for k, v in sorted(bh.items(), key=lambda t: int(t[0])))
            rh = r.get("rank_histogram", {})
            if rh:
                tot = sum(rh.values()); acc = 0; med = 0
                for k in sorted(rh, key=lambda x: int(x)):
                    acc += rh[k]
                    if acc >= tot / 2:
                        med = int(k); break
            else:
                med = "-"
            print(f"{r['name']:<20}{r.get('bits_per_weight', 0):>8.3f}"
                  f"{num(r.get(args.metric)):>14}{bhs:>34}{str(med):>10}")

        print("\n-- paired at matched budget (uniform reference -> allocated) --")
        by = {r["name"]: r for r in rows}
        pat = re.compile(r"^gilora-b(\d+)r(\d+)$")
        print(f"{'reference':<14}{'bpw':>8}{'uniform':>11}{'alloc w=1':>12}"
              f"{'alloc w':>11}{'best gain':>11}")
        for n in sorted(by):
            m = pat.match(n)
            if not m:
                continue
            b, r_ = m.group(1), m.group(2)
            u = by.get(f"gilora-b{b}r{r_}")
            a1 = by.get(f"allocNW-b{b}r{r_}")
            a2 = by.get(f"alloc-b{b}r{r_}")
            if not u or args.metric not in u:
                continue
            uv = u[args.metric]
            vals = [x[args.metric] for x in (a1, a2)
                    if x and args.metric in x and math.isfinite(x[args.metric])]
            gain = f"{uv / min(vals):.2f}x" if vals and math.isfinite(uv) else "---"
            print(f"b{b} r{r_:<10}{u['bits_per_weight']:>8.3f}{num(uv):>11}"
                  f"{num(a1.get(args.metric) if a1 else None):>12}"
                  f"{num(a2.get(args.metric) if a2 else None):>11}{gain:>11}")


if __name__ == "__main__":
    main()
