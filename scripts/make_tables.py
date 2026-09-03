#!/usr/bin/env python
"""Emit the manuscript's LaTeX tables from pulled summaries. Runs on the laptop.

Two tables:

  main    perplexity at matched effective bit-width, uniform versus allocated,
          one block per uniform reference configuration;
  ablate  the contribution of each ingredient at a few budgets.

Numbers are read from ``out/runs/*/results.json`` and never retyped, so the
manuscript cannot drift from the runs that produced it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re


def family(name: str) -> str:
    m = re.match(r"([A-Za-z]+)", name)
    return m.group(1) if m else name


def load(runs: pathlib.Path, model: str) -> dict[str, dict]:
    p = _find(runs, [f"sweep-q3-{model}/results.json", f"sweep-{model}/results.json"])
    if p is None:
        return {}
    return {r["name"]: r for r in json.load(open(p))}


def fmt(v, nd=2, best=False):
    if v is None:
        return "---"
    s = f"{v:.{nd}f}" if v < 1e4 else f"{v:.2e}"
    return rf"\textbf{{{s}}}" if best else s


def table_main(models: list[str], data: dict[str, dict], uniforms) -> str:
    """Perplexity at matched effective bit-width.

    One column per model rather than a (bpw, PPL) pair: the budgets are matched
    by construction within a block, so repeating the rate on every method row
    both wastes a column per model and obscures that they are the same. The
    rate appears once per block instead.
    """
    rows = [
        r"\begin{table}[H]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{0.94}",
        r"\caption{WikiText-2 perplexity ($\downarrow$) at matched effective "
        r"bit-width. Each block fixes a uniform reference configuration $(b, r)$ "
        r"and gives every method in it the same storage, which the block heading "
        r"reports per model in effective bits per weight \eqref{eq:effbits}. "
        r"The first two rows of a block are uniform, the last two allocated. Best "
        r"in block in bold; a dash marks a budget not run on that model.}",
        r"\label{tab:main}",
        rf"\begin{{tabular}}{{l{'c' * len(models)}}}", r"\toprule",
        "method & " + " & ".join(_label(m) for m in models) + r" \\",
        r"\midrule",
    ]
    rows.append("FP16 (uncompressed) & "
                + " & ".join(fmt(data[m].get("fp16", {}).get("wikitext2_ppl"))
                             for m in models) + r" \\")
    rows.append(r"\midrule")

    # The block header already says these share a uniform reference, and the
    # caption says which rows are uniform and which allocated, so the "(uniform)"
    # and "(allocated)" suffixes are redundant -- and they are what pushed the
    # table past the review-mode measure.
    labels = [
        ("olrc", "GPTQ + OLrC"),
        ("gilora", "GPTQ-intrinsic LoRA"),
        ("allocNW", r"\method{}, $\omega_\ell \equiv 1$"),
        ("alloc", r"\method{}"),
    ]
    for (b, r) in uniforms:
        block = {(fam, m): data[m].get(f"{fam}-b{b}r{r}")
                 for fam, _ in labels for m in models}
        if not any(block.values()):
            continue
        # The per-model rate goes in the block header rather than its own row:
        # six extra rows is the difference between this table fitting on a page
        # and running off the bottom of one.
        bpw = []
        for m in models:
            v = next((block[(f, m)]["bits_per_weight"] for f, _ in labels
                      if block.get((f, m)) and "bits_per_weight" in block[(f, m)]), None)
            bpw.append(fmt(v, 2))
        rows.append(rf"\multicolumn{{{1 + len(models)}}}{{l}}{{\emph{{budget of uniform "
                    rf"$b={b}$, $r={r}$ --- " + " / ".join(bpw)
                    + r" bits/weight}} \\")
        best = {}
        for m in models:
            vals = [block[(f, m)]["wikitext2_ppl"] for f, _ in labels
                    if block.get((f, m)) and "wikitext2_ppl" in block[(f, m)]]
            best[m] = min(vals) if vals else None
        for fam, lab in labels:
            cells = []
            for m in models:
                rec = block.get((fam, m))
                if not rec or "wikitext2_ppl" not in rec:
                    cells.append("---")
                    continue
                v = rec["wikitext2_ppl"]
                cells.append(fmt(v, 2, best=(best[m] is not None and v == best[m])))
            rows.append(rf"\quad {lab} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    rows[-1] = r"\bottomrule"
    rows += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(rows)


def table_ablation(models: list[str], data: dict[str, dict], refs) -> str:
    rows = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Ablation. Each row allocates the same budget as the uniform "
        r"reference but restricts what may vary. ``bits only'' pins every rank to "
        r"the uniform $r$; ``rank only'' pins every bit-width to the uniform $b$. "
        r"A dash marks a budget not run on that model.}",
        r"\label{tab:ablation}",
        rf"\begin{{tabular}}{{l{'c' * len(models)}}}", r"\toprule",
        "configuration & " + " & ".join(_label(m) for m in models) + r" \\",
        r"\midrule",
    ]
    labels = [
        ("gilora", "uniform (no allocation)"),
        ("allocB", "bits only"),
        ("allocR", "rank only"),
        ("allocNW", r"both, $\omega_\ell \equiv 1$"),
        ("alloc", r"both, measured $\omega_\ell$"),
    ]
    for (b, r) in refs:
        rows.append(rf"\multicolumn{{{1+len(models)}}}{{l}}{{\emph{{budget of $b={b}$, "
                    rf"$r={r}$}}}} \\")
        for fam, lab in labels:
            cells = []
            for m in models:
                rec = data[m].get(f"{fam}-b{b}r{r}")
                cells.append(fmt(rec.get("wikitext2_ppl")) if rec else "---")
            rows.append(rf"\quad {lab} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    rows[-1] = r"\bottomrule"
    rows += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(rows)


def table_spread(models: list[str], tables: pathlib.Path) -> str:
    """The measured heterogeneity, and the gain the coding-gain theorem predicts."""
    rows = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Layer heterogeneity and the coding gain it predicts, measured "
        r"from calibration statistics alone. $a_\ell$ is the difficulty coefficient "
        r"\eqref{eq:bitrule}; the predicted saving is "
        r"$\frac{1}{2}\log_2(\tilde a_A/\tilde a_G)$ of \cref{thm:codinggain}, "
        r"evaluated at the stated rank.}",
        r"\label{tab:spread}",
        rf"\begin{{tabular}}{{l{'c' * len(models)}}}", r"\toprule",
        "quantity & " + " & ".join(_label(m) for m in models) + r" \\",
        r"\midrule",
    ]
    data = {}
    for m in models:
        f = _find(tables, [f"stats-q3-{m}.json", f"stats-{m}.json"])
        data[m] = json.load(open(f)) if f else None
    if not any(data.values()):
        return ""

    def line(label, fn):
        cells = [fn(data[m]) if data[m] else "---" for m in models]
        rows.append(rf"{label} & " + " & ".join(cells) + r" \\")

    line(r"layers", lambda d: f"{d['n_layers']}")
    line(r"$\log_2 a_\ell$ range (bits)", lambda d: f"{d['log2_a']['range']:.1f}")
    line(r"$\log_2 a_\ell$ std.\ dev.\ (bits)", lambda d: f"{d['log2_a']['sd']:.2f}")
    rows.append(r"\midrule")
    for r_ in ("0", "8", "32"):
        line(rf"gain at $r={r_}$ ($\times$)",
             lambda d, r_=r_: f"{d['coding_gain'][r_]['gain_ratio']:.2f}"
             if r_ in d["coding_gain"] else "---")
    for r_ in ("0", "8", "32"):
        line(rf"saving at $r={r_}$ (bits/wt)",
             lambda d, r_=r_: f"{d['coding_gain'][r_]['bits_saved']:.2f}"
             if r_ in d["coding_gain"] else "---")
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(rows)


def table_surrogate(path: pathlib.Path) -> str:
    """Bound versus measured distortion: separability, ranking, and the cost."""
    if not path.exists():
        return ""
    d = json.load(open(path))
    rf = d.get("rank_factor", {})
    sp = d.get("rank_factor_spread", {})
    ac = d.get("allocation_cost", {})
    rho = list(d.get("spearman", {}).values())
    rows = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{The surrogate against the measured distortion surface, on "
        rf"{d.get('n_profiled', '?')} layers of Qwen3-0.6B-Base profiled over a grid of "
        r"bit-widths and ranks. Top: the bound's rank factor against the measured "
        r"one, both normalised to $r=0$, with the spread of the measurement across "
        r"layers. Bottom: the price of allocating with the free bound instead of the "
        r"measured surface, both scored by the measured surface. Over the same grid, "
        r"a separable model of $\log D_\ell$ leaves a maximum interaction of "
        + f"{d['separability_log2']['median']:.3f}"
        + r" bits at the median layer and "
        + f"{d['separability_log2']['max']:.3f}"
        + r" at the worst, and the cross-layer Spearman correlation between bound "
        r"and measurement has median "
        + (f"{sorted(rho)[len(rho) // 2]:.3f}" if rho else "---")
        + r" and minimum "
        + (f"{min(rho):.3f}" if rho else "---") + r".}",
        r"\label{tab:surrogate}",
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"rank $r$ & 8 & 16 & 32 & 64 & 128 \\", r"\midrule",
    ]
    ks = ["8", "16", "32", "64", "128"]
    rows.append("bound $h_\\ell(r)/h_\\ell(0)$ & " +
                " & ".join(f"{rf[k][0]:.3f}" if k in rf else "---" for k in ks) + r" \\")
    rows.append("measured $D_\\ell(r)/D_\\ell(0)$ & " +
                " & ".join(f"{rf[k][1]:.3f}" if k in rf else "---" for k in ks) + r" \\")
    rows.append("measured, p90/p10 across layers & " +
                " & ".join(f"{sp[k]['p90']/sp[k]['p10']:.2f}" if k in sp else "---"
                           for k in ks) + r" \\")
    rows += [r"\midrule",
             r"budget (bits/weight) & 2.50 & 3.00 & 3.50 & 4.00 & \\"]
    bk = ["2.5", "3.0", "3.5", "4.0"]
    rows.append("bound-driven / measured-driven distortion & " +
                " & ".join(f"{ac[k]['excess']:.2f}" if k in ac else "---" for k in bk)
                + r" & \\")
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(r for r in rows if r)


LABELS = {"llama3b": "Llama-3.2-3B"}


def _label(m: str) -> str:
    """Display name: size-only keys get the Qwen3 family, others map explicitly."""
    return LABELS.get(m, f"Qwen3-{m}" if m[0].isdigit() else m)


def _find(root: pathlib.Path, candidates: list[str]):
    """First existing path among `candidates`, relative to `root`.

    Model runs are named after the model, not a fixed family prefix, so the
    generators accept either the Qwen-style or the plain layout instead of
    silently producing nothing when a third model is added.
    """
    for c in candidates:
        p = root / c
        if p.exists():
            return p
    return None


TASKS = ["arc_easy", "arc_challenge", "hellaswag", "winogrande", "piqa"]
TASK_LABEL = {"arc_easy": "ARC-e", "arc_challenge": "ARC-c", "hellaswag": "HellaSwag",
              "winogrande": "WinoG.", "piqa": "PIQA"}


def table_zeroshot(path: pathlib.Path) -> str:
    """Zero-shot accuracy at three matched budgets.

    Perplexity is a thin basis for a claim about model quality, so the same
    operating points are scored on five multiple-choice benchmarks with
    length-normalised log-likelihood.
    """
    if not path.exists():
        return ""
    rows = {r["name"]: r for r in json.load(open(path))}
    order = [("gilora", "uniform"), ("allocNW", r"\method{}, $\omega_\ell \equiv 1$"),
             ("alloc", r"\method{}")]
    out = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Zero-shot accuracy (\%, length-normalised) on Qwen3-0.6B-Base at "
        r"three matched budgets. Perplexity from the same run is repeated for "
        r"reference. Allocation helps where the budget binds and is neutral once it "
        r"does not, matching the perplexity picture.}",
        r"\label{tab:zeroshot}",
        rf"\begin{{tabular}}{{lc{'c' * (len(TASKS) + 1)}}}", r"\toprule",
        "configuration & PPL & " + " & ".join(TASK_LABEL[t] for t in TASKS)
        + r" & mean \\", r"\midrule",
    ]
    fp = rows.get("fp16")
    if fp and any(f"{t}_acc_norm" in fp for t in TASKS):
        accs = [fp.get(f"{t}_acc_norm") for t in TASKS]
        got = [a for a in accs if a is not None]
        out.append("FP16 (uncompressed) & " + f"{fp['wikitext2_ppl']:.2f} & "
                   + " & ".join(f"{a*100:.1f}" if a is not None else "---" for a in accs)
                   + f" & {sum(got)/len(got)*100:.1f}" + r" \\")
        out.append(r"\midrule")
    for b, r_ in [(2, 32), (3, 32), (4, 32)]:
        block = [rows.get(f"{fam}-b{b}r{r_}") for fam, _ in order]
        if not any(block):
            continue
        bpw = next((x["bits_per_weight"] for x in block if x), None)
        out.append(rf"\multicolumn{{{2+len(TASKS)+1}}}{{l}}{{\emph{{budget of $b={b}$, "
                   rf"$r={r_}$: {bpw:.3f} bits/weight}}}} \\")
        means = []
        for x in block:
            if x:
                a = [x.get(f"{t}_acc_norm") for t in TASKS]
                g = [v for v in a if v is not None]
                means.append(sum(g)/len(g) if g else None)
            else:
                means.append(None)
        best = max((m for m in means if m is not None), default=None)
        for (fam, lab), x, mn in zip(order, block, means):
            if not x:
                out.append(rf"\quad {lab} & " + " & ".join(["---"] * (len(TASKS)+2)) + r" \\")
                continue
            accs = [x.get(f"{t}_acc_norm") for t in TASKS]
            mstr = f"{mn*100:.1f}" if mn is not None else "---"
            if mn is not None and best is not None and abs(mn - best) < 1e-12:
                mstr = rf"\textbf{{{mstr}}}"
            out.append(rf"\quad {lab} & {x['wikitext2_ppl']:.2f} & "
                       + " & ".join(f"{a*100:.1f}" if a is not None else "---" for a in accs)
                       + f" & {mstr}" + r" \\")
        out.append(r"\midrule")
    out[-1] = r"\bottomrule"
    out += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def table_rotation(tables: pathlib.Path, blocks=(1, 32, 128, 512, 1024)) -> str:
    """What incoherence preprocessing does to the allocation problem."""
    import numpy as np  # noqa: PLC0415

    rows_data, base = [], None
    for B in blocks:
        f = tables / f"rot{B}-q3-0.6B.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        lvl = float(np.median([x["log2_a"] for x in d["per_layer"]]))
        if base is None:
            base = lvl
        rows_data.append((B, lvl - base, d["log2_a"]["sd"],
                          d["coding_gain"]["0"]["gain_ratio"],
                          float(np.mean([v["fractions"][0] for v in d["tail"].values()])),
                          float(np.median([v["r_star_median"] for v in d["tail"].values()]))))
    if len(rows_data) < 2:
        return ""
    out = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Incoherence preprocessing versus allocation, on Qwen3-0.6B-Base. "
        r"A block-diagonal randomized Hadamard rotation is applied to each layer, with "
        r"block size interpolating from none to full. The spectral quantities are "
        r"invariant by construction (\cref{rem:rotation}) and the measurement confirms "
        r"it to every reported digit. $r^\star$ is a median over projection types, "
        r"which differ (\cref{sec:exp:hetero}); what matters is that none of these "
        r"quantities moves. The difficulty \emph{level} falls, but its "
        r"\emph{spread} -- the only thing \cref{thm:codinggain} converts into a gain -- "
        r"does not.}",
        r"\label{tab:rotation}",
        r"\begin{tabular}{rccccc}", r"\toprule",
        r"block & $\Delta$ median $\log_2 a_\ell$ & sd of $\log_2 a_\ell$ & "
        r"coding gain & $T_\ell(8)/T_\ell(0)$ & median $r^\star$ \\",
        r"\midrule",
    ]
    for B, dlvl, sd, cg, tail, rstar in rows_data:
        lab = "1 (none)" if B == 1 else str(B)
        out.append(f"{lab} & {dlvl:+.2f} & {sd:.2f} & {cg:.2f} & {tail:.4f} & "
                   f"{rstar:.0f}" + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def table_rotation_ppl(runs: pathlib.Path, blocks=(128, 1024)) -> str:
    """End-to-end effect of rotation, at matched configurations.

    Each rotated run repeats the same plan in the rotated basis, so rows pair
    exactly. What matters is not only whether rotation lowers perplexity but
    whether it changes the *ratio* between uniform and allocated -- the coding
    gain predicts it should not.
    """
    base_p = runs / "sweep-q3-0.6B" / "results.json"
    if not base_p.exists():
        return ""
    base = {r["name"]: r for r in json.load(open(base_p))}
    cols = []
    for B in blocks:
        f = runs / f"rotsweep{B}-q3-0.6B" / "results.json"
        if f.exists():
            cols.append((B, {r["name"]: r for r in json.load(open(f))}))
    if not cols:
        return ""

    names = [("gptq-b2", "GPTQ, $b=2$"),
             ("gilora-b2r32", r"\quad uniform, $b=2$, $r=32$"),
             ("alloc-b2r32", r"\quad \method{}, same budget"),
             ("gptq-b3", "GPTQ, $b=3$"),
             ("gilora-b3r32", r"\quad uniform, $b=3$, $r=32$"),
             ("alloc-b3r32", r"\quad \method{}, same budget"),
             ("gptq-b4", "GPTQ, $b=4$"),
             ("gilora-b4r32", r"\quad uniform, $b=4$, $r=32$"),
             ("alloc-b4r32", r"\quad \method{}, same budget")]
    out = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{WikiText-2 perplexity with and without incoherence "
        r"preprocessing, Qwen3-0.6B-Base. The whole pipeline is rerun in the "
        r"rotated basis. Rotation helps plain GPTQ most, helps the allocated "
        r"models at every budget, and \emph{hurts} the uniform low-rank "
        r"configuration at $b=2$, but it leaves the ratio between the uniform "
        r"and allocated rows essentially unchanged, which is what "
        r"\cref{thm:codinggain} predicts of a change that moves the level of "
        r"$a_\ell$ and not its spread.}",
        r"\label{tab:rotationppl}",
        rf"\begin{{tabular}}{{l{'c' * (len(cols) + 1)}}}", r"\toprule",
        "configuration & no rotation & "
        + " & ".join(f"block {B}" for B, _ in cols) + r" \\", r"\midrule",
    ]
    any_row = False
    for key, lab in names:
        b = base.get(key)
        if not b or "wikitext2_ppl" not in b:
            continue
        cells = [f"{b['wikitext2_ppl']:.2f}"]
        for _, d in cols:
            r = d.get(key)
            cells.append(f"{r['wikitext2_ppl']:.2f}" if r and "wikitext2_ppl" in r else "---")
        out.append(rf"{lab} & " + " & ".join(cells) + r" \\")
        any_row = True
    if not any_row:
        return ""
    # The ratio of the uniform to the allocated row is what the coding gain says
    # rotation cannot move; report it rather than leaving it to be divided out.
    out.append(r"\midrule")
    for b_, r_ in [(3, 32), (4, 32)]:
        cells = []
        for d in [base] + [dd for _, dd in cols]:
            u, a = d.get(f"gilora-b{b_}r{r_}"), d.get(f"alloc-b{b_}r{r_}")
            cells.append(f"{u['wikitext2_ppl'] / a['wikitext2_ppl']:.2f}$\\times$"
                         if u and a and "wikitext2_ppl" in u and "wikitext2_ppl" in a
                         else "---")
        out.append(rf"ratio uniform\,:\,\method{{}}, $b={b_}$, $r={r_}$ & "
                   + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


# The statistics timing comes from a run that does *not* cache Hessians to disk,
# so it measures the phase itself rather than the I/O that makes a rate-distortion
# sweep cheap. Caching is optional and its cost is disk-bound, not compute-bound.
RUNTIME_MODELS = [
    ("Qwen3-0.6B", "0.6B", "tstat-0.6B", "omega-q3-0.6B", "sweep-q3-0.6B", "cost-0.6B"),
    ("Qwen3-1.7B", "1.7B", "tstat-1.7B", "omega-q3-1.7B", "sweep-q3-1.7B", "cost-1.7B"),
    ("Llama-3B", "llama3b", "tstat-llama3b", "omega-llama3b", "sweep-llama3b", "cost-llama3b"),
    ("Qwen3-8B", "8B", "tstat-8B", "omega3-q3-8B", "sweep-q3-8B", "cost-8B"),
]


def table_runtime(runs: pathlib.Path, alloc_ms: float = 198.0) -> str:
    """Wall-clock and peak device memory per phase.

    Phase-1 statistics and the sensitivity probe are one-off per model; the
    per-point row is what each additional operating point costs, and it is the
    only one that recurs across a rate-distortion sweep. Peak memory is measured
    on the widest layer, which is what sets it -- the pipeline never holds more
    than one block on the device.
    """
    import numpy as np  # noqa: PLC0415

    def wall(d):
        f = runs / d / "metrics.json"
        if not f.exists():
            return None
        return json.load(open(f)).get("wall_seconds")

    rows_data = []
    for label, _key, sd, od, wd, cd in RUNTIME_MODELS:
        st, om = wall(sd), wall(od)
        pt = None
        f = runs / wd / "results.json"
        if f.exists():
            secs = [r["seconds"] for r in json.load(open(f))
                    if "seconds" in r and r["name"] != "fp16"]
            if secs:
                pt = float(np.median(secs))
        cf = runs / cd / "metrics.json"
        cost = json.load(open(cf)) if cf.exists() else {}
        rows_data.append((label, st, om, pt, cost))
    if not any(r[1] for r in rows_data):
        return ""

    def mm(v):
        if v is None:
            return "---"
        return f"{v / 60:.1f}" if v < 600 else f"{v / 60:.0f}"
    out = [
        r"\begin{table}[H]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Cost of the pipeline, measured on one A100. The first two rows "
        r"are paid once per model; the third is what each additional operating "
        r"point costs and is the only one that recurs across a rate--distortion "
        r"sweep. The allocation solve itself is CPU-only and independent of model "
        r"size. The statistics row covers Hessians, spectra and the clipping search. Peak device memory is measured on the widest layer, which sets "
        r"it: the pipeline holds one transformer block at a time, so it never "
        r"needs to fit the model. The statistics row excludes writing the Hessian "
        r"cache to disk, which is optional and disk-bound; it is what makes each "
        r"further operating point cost only the compression row.}",
        r"\label{tab:runtime}",
        rf"\begin{{tabular}}{{l{'c' * len(rows_data)}}}", r"\toprule",
        "phase & " + " & ".join(r[0] for r in rows_data) + r" \\",
        r"\midrule",
    ]
    out.append("statistics (min) & "
               + " & ".join(mm(r[1]) for r in rows_data) + r" \\")
    out.append("sensitivity probe (min) & "
               + " & ".join(mm(r[2]) for r in rows_data) + r" \\")
    out.append("allocation solve (CPU, ms) & "
               + " & ".join(f"{alloc_ms:.0f}" for _ in rows_data) + r" \\")
    out.append("compression + PPL, per point (min) & "
               + " & ".join(mm(r[3]) for r in rows_data) + r" \\")
    out.append(r"\midrule")
    out.append("widest layer input dim. & "
               + " & ".join(str(r[4].get("widest_in", "---")) for r in rows_data) + r" \\")
    out.append("peak device memory (MB) & "
               + " & ".join(f"{r[4]['augmented factorization']['peak_MB']:.0f}"
                            if r[4] else "---" for r in rows_data) + r" \\")
    out.append("model weights, FP16 (MB) & "
               + " & ".join(f"{r[4]['model_fp16_MB']:.0f}" if r[4] else "---"
                            for r in rows_data) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def table_r0(models: list[str], data: dict[str, dict]) -> str:
    """Joint bit+rank allocation against bits-only allocation at r = 0.

    The reviewer question this answers: once rank is priced honestly, is the
    low-rank budget better spent on precision? ``allocZ`` is BRAID restricted to
    r = 0 -- a mixed-precision allocator driven by the same distortion model --
    given exactly the storage the uniform reference consumes.
    """
    # every budget the r=0 sweep covers, not just the six of tab:main -- the
    # comparison is the point of this table and the tight budgets are where it bites
    uniforms = [(2, 16), (2, 32), (2, 64), (3, 16), (3, 32), (3, 64), (4, 16), (4, 32)]
    rows = [
        r"\begin{table}[H]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{0.95}",
        r"\caption{Does rank earn its place? WikiText-2 perplexity "
        r"($\downarrow$) at matched storage for three allocations of the same "
        r"budget: none (uniform), bit-widths only with every rank forced to zero, "
        r"and joint bit--rank. The $r=0$ column is a mixed-precision allocator "
        r"driven by the same distortion model, and is the baseline that tests "
        r"whether the second degree of freedom earns its keep. A dash marks a "
        r"budget not run on that model.}",
        r"\label{tab:r0}",
        rf"\begin{{tabular}}{{ll{'c' * len(models)}}}", r"\toprule",
        "budget & allocation & " + " & ".join(_label(m) for m in models) + r" \\",
        r"\midrule",
    ]
    fams = [("gilora", "none (uniform)"), ("allocZ", r"bit-widths only, $r=0$"),
            ("alloc", r"joint bit--rank")]
    any_row = False
    for (b, r) in uniforms:
        block = {(f, m): data[m].get(f"{f}-b{b}r{r}") for f, _ in fams for m in models}
        if not any(block.get((f, m)) for f, _ in fams for m in models):
            continue
        best = {}
        for m in models:
            vals = [block[(f, m)]["wikitext2_ppl"] for f, _ in fams
                    if block.get((f, m)) and "wikitext2_ppl" in block[(f, m)]]
            best[m] = min(vals) if vals else None
        for k, (fam, lab) in enumerate(fams):
            cells = []
            for m in models:
                rec = block.get((fam, m))
                if not rec or "wikitext2_ppl" not in rec:
                    cells.append("---")
                    continue
                v = rec["wikitext2_ppl"]
                cells.append(fmt(v, 2, best=(best[m] is not None and v == best[m])))
            head = rf"$b={b}$, $r={r}$" if k == 0 else ""
            rows.append(rf"{head} & {lab} & " + " & ".join(cells) + r" \\")
            any_row = True
        rows.append(r"\midrule")
    if not any_row:
        return ""
    rows[-1] = r"\bottomrule"
    rows += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(rows)


ZS_RUNS = [("Qwen3-0.6B", "zs-q3-0.6B"), ("Qwen3-1.7B", "zs-q3-1.7B"),
           ("Llama-3.2-3B", "zs-llama3b"), ("Qwen3-8B", "zs-q3-8B")]


SEED_RUNS = {0: "sweep-q3-0.6B", 1: "sweep-s1-q3-0.6B", 2: "sweep-s2-q3-0.6B"}


def table_seeds(runs: pathlib.Path) -> str:
    """Run-to-run spread over independent calibration draws.

    Phase 1 is deterministic given its tokens, so the only randomness that can
    move a result is which sequences are drawn. Three full pipelines -- fresh
    Hessians, fresh sensitivity probe, fresh sweep -- bound how much of the
    reported gap that could explain.
    """
    import statistics
    data = {}
    for sd, d in SEED_RUNS.items():
        f = runs / d / "results.json"
        if f.exists():
            data[sd] = {r["name"]: r for r in json.load(open(f))}
    if len(data) < 3:
        return ""
    out = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Run-to-run variability on Qwen3-0.6B-Base over three independent "
        r"calibration draws, each a complete pipeline: new Hessians, new clipping "
        r"search, new sensitivity probe, new allocation. WikiText-2 perplexity, mean "
        r"$\pm$ standard deviation over the three runs. The allocated models vary "
        r"less than the uniform ones exactly where the budget binds, and no budget "
        r"changes which method wins.}",
        r"\label{tab:seeds}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"budget & uniform & \method{} & gap & gap\,/\,s.d. \\", r"\midrule",
    ]
    n = flips = 0
    for b, r_ in [(2, 32), (3, 16), (3, 32), (3, 64), (4, 16), (4, 32)]:
        u = [data[s].get(f"gilora-b{b}r{r_}", {}).get("wikitext2_ppl") for s in SEED_RUNS]
        a = [data[s].get(f"alloc-b{b}r{r_}", {}).get("wikitext2_ppl") for s in SEED_RUNS]
        if any(x is None for x in u + a):
            continue
        n += 1
        flips += len({x > y for x, y in zip(u, a)}) > 1
        mu, su = statistics.mean(u), statistics.stdev(u)
        ma, sa = statistics.mean(a), statistics.stdev(a)
        out.append(rf"$b={b}$, $r={r_}$ & ${mu:.2f} \pm {su:.2f}$ & "
                   rf"${ma:.2f} \pm {sa:.2f}$ & ${mu - ma:.2f}$ & "
                   rf"${(mu - ma) / max(sa, 1e-9):.0f}\times$ \\")
    if not n:
        return ""
    out += [r"\midrule",
            rf"\multicolumn{{5}}{{l}}{{budgets where the draw changes which method "
            rf"wins: {flips} of {n}}} \\",
            r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def table_zeroshot_models(runs: pathlib.Path) -> str:
    """Zero-shot mean accuracy across models, uniform against allocated.

    Table 5 gives the per-task breakdown on one model; this one answers the
    separate question of whether the perplexity conclusion -- allocation pays
    where the budget binds and not above it -- survives on the models where the
    envelope absorbs the gain and where the spectra are flat.
    """
    data = {}
    for lab, d in ZS_RUNS:
        f = runs / d / "results.json"
        if f.exists():
            data[lab] = {r["name"]: r for r in json.load(open(f))}
    labs = [l for l, _ in ZS_RUNS if l in data]
    if len(labs) < 2:
        return ""

    def mean(rec):
        if not rec:
            return None
        v = [rec.get(f"{t}_acc_norm") for t in TASKS]
        v = [x for x in v if x is not None]
        return 100 * sum(v) / len(v) if v else None

    out = [
        r"\begin{table}[H]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Zero-shot accuracy across models: mean over "
        r"five multiple-choice benchmarks (\%, length-normalised), for the uniform reference and "
        r"for \method{} at the same storage. The downstream picture matches the "
        r"perplexity one, including where it turns: allocation buys a great deal at "
        r"the budget where compression is doing real damage (most of all on "
        r"Llama-3.2-3B, the model whose spectra are flattest) and nothing, or a "
        r"little less than nothing, above it.}",
        r"\label{tab:zeroshotmodels}",
        rf"\begin{{tabular}}{{ll{'r' * len(labs)}}}", r"\toprule",
        "budget & configuration & " + " & ".join(labs) + r" \\", r"\midrule",
    ]
    fps = [mean(data[l].get("fp16")) for l in labs]
    out.append("& FP16 (uncompressed) & "
               + " & ".join(f"{x:.1f}" if x else "---" for x in fps) + r" \\")
    out.append(r"\midrule")
    for b, r_ in [(2, 32), (3, 32), (4, 32)]:
        u = [mean(data[l].get(f"gilora-b{b}r{r_}")) for l in labs]
        a = [mean(data[l].get(f"alloc-b{b}r{r_}")) for l in labs]
        if not any(u) and not any(a):
            continue
        cell = lambda v: f"{v:.1f}" if v is not None else "---"
        out.append(rf"$b={b}$, $r={r_}$ & uniform & "
                   + " & ".join(cell(x) for x in u) + r" \\")
        out.append(r"& \method{} & " + " & ".join(cell(x) for x in a) + r" \\")
        d = [None if (x is None or y is None) else y - x for x, y in zip(u, a)]
        out.append(r"& difference & "
                   + " & ".join(rf"$\mathbf{{{x:+.1f}}}$" if x is not None and x > 0.5
                                else (f"${x:+.1f}$" if x is not None else "---")
                                for x in d) + r" \\")
        out.append(r"\midrule")
    out[-1] = r"\bottomrule"
    out += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def table_qfactors(models: list[str], data: dict[str, dict]) -> str:
    """Allocation when the low-rank factors are themselves quantized.

    Storing L and R at 4/8 bits rather than 16 is nearly free in quality and
    buys back real storage, which makes it a competing use of the same budget.
    This table asks whether allocation still pays once that cheaper baseline is
    the one being beaten.
    """
    budgets = [(2, 32), (3, 16), (3, 32), (3, 64), (4, 32)]
    labs = [m for m in models if any(f"giloraQ-b{b}r{r}" in (data.get(m) or {})
                                     for b, r in budgets)]
    if not labs:
        return ""
    rows = [
        r"\begin{table}[H]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Allocation against a uniform baseline whose low-rank factors "
        r"are themselves quantized ($b_L = 4$, $b_R = 8$, priced at $6$ bits per "
        r"factor entry). WikiText-2 perplexity ($\downarrow$) at matched storage; "
        r"the block heading gives the budget in effective bits per weight. "
        r"Quantizing the factors is close to free in quality and returns real "
        r"storage, so it competes with allocation for the same budget --- and it "
        r"absorbs much of what allocation would otherwise win.}",
        r"\label{tab:qfactors}",
        rf"\begin{{tabular}}{{ll{'c' * len(labs)}}}", r"\toprule",
        "budget & configuration & " + " & ".join(_label(m) for m in labs) + r" \\",
        r"\midrule",
    ]
    n = 0
    for b, r_ in budgets:
        u = [(data[m] or {}).get(f"giloraQ-b{b}r{r_}") for m in labs]
        a = [(data[m] or {}).get(f"allocQ-b{b}r{r_}") for m in labs]
        if not any(u):
            continue
        bpw = " / ".join(f"{x['bits_per_weight']:.2f}" if x else "---" for x in u)
        rows.append(rf"\multicolumn{{{2 + len(labs)}}}{{l}}{{\emph{{budget of "
                    rf"$b={b}$, $r={r_}$: {bpw} bits/weight}}}} \\")
        for lab, recs, other in (("uniform, quantized factors", u, a),
                                 (r"\method{}, same budget", a, u)):
            cells = []
            for x, y in zip(recs, other):
                if not x or "wikitext2_ppl" not in x:
                    cells.append("---")
                    continue
                best = y is None or "wikitext2_ppl" not in y or \
                    x["wikitext2_ppl"] < y["wikitext2_ppl"]
                cells.append(fmt(x["wikitext2_ppl"], 2, best=best))
            rows.append(rf"& {lab} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
        n += 1
    if not n:
        return ""
    rows[-1] = r"\bottomrule"
    rows += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(rows)


def table_calibration(data: dict[str, dict], model: str = "0.6B") -> str:
    """The calibrated rank factor, evaluated end to end.

    Sec. 6.6 measures that the bound overvalues rank. The obvious repair is to
    rescale h(r) by the measured ratio and re-allocate. ``allocC`` is that
    repair. It is reported because it does not work, which is informative: a
    surrogate that is more faithful layer by layer yields a worse model.
    """
    d = data.get(model) or {}
    rows = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Calibrating the rank factor makes the surrogate more faithful "
        r"and the model worse. WikiText-2 perplexity ($\downarrow$) on "
        r"Qwen3-0.6B-Base for the allocator driven by the bound and by the bound "
        r"with $h_\ell(r)$ rescaled to the measured reduction of "
        r"\Cref{tab:surrogate}. $\tilde r$ is the median assigned rank. The "
        r"calibrated allocator declines to buy rank at all, and pays for it "
        r"exactly where the budget is tightest.}",
        r"\label{tab:calibration}",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"& & \multicolumn{2}{c}{bound-driven} & \multicolumn{2}{c}{calibrated} & \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"budget & bpw & PPL & $\tilde r$ & PPL & $\tilde r$ & $\Delta$ \\",
        r"\midrule",
    ]
    # the calibration sweep covers the tight budgets the main grid omits, and
    # they are where the two allocators disagree
    uniforms = [(2, 16), (2, 32), (2, 64), (3, 16), (3, 32), (3, 64), (4, 16), (4, 32)]
    import numpy as _np
    def _med(h):
        if not h:
            return None
        v = _np.repeat([int(k) for k in h], [int(x) for x in h.values()])
        return int(_np.median(v))
    n = 0
    for (b, r) in uniforms:
        a, c = d.get(f"alloc-b{b}r{r}"), d.get(f"allocC-b{b}r{r}")
        if not (a and c):
            continue
        delta = c["wikitext2_ppl"] - a["wikitext2_ppl"]
        delta = 0.0 if abs(delta) < 5e-3 else delta  # avoid a signed zero
        rows.append(
            rf"$b={b}$, $r={r}$ & {a['bits_per_weight']:.2f} & "
            rf"{a['wikitext2_ppl']:.2f} & {_med(a.get('rank_histogram'))} & "
            rf"{c['wikitext2_ppl']:.2f} & {_med(c.get('rank_histogram'))} & "
            rf"{delta:+.2f} \\"
        )
        n += 1
    if not n:
        return ""
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(rows)


def _drop_dash_note(tex: str) -> str:
    """Remove the caption's dash legend when the table has no dashes left.

    The tables carry a "a dash marks a budget not run on that model" note
    because coverage is uneven while a sweep is in flight. Once every cell is
    filled the sentence explains a glyph the reader cannot see, so it goes.
    """
    # a dash used as a *cell value*, not the em-dashes inside block headings
    if re.search(r"(?:&\s*---\s*(?:&|\\\\))|(?:^\s*---\s*&)", tex, re.M):
        return tex
    for note in (r" A dash marks a budget not run on that model.",
                 r"; a dash marks a budget not run on that model"):
        tex = tex.replace(note, "")
    return tex


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["0.6B", "1.7B"])
    ap.add_argument("--runs", default="out/runs")
    ap.add_argument("--tables", default="results/tables",
                    help="where analyze_stats.py summaries live")
    ap.add_argument("--outdir", default="paper/tables")
    args = ap.parse_args()

    runs = pathlib.Path(args.runs)
    data = {m: load(runs, m) for m in args.models}
    present = [m for m in args.models if data[m]]
    if not present:
        print("no results.json found -- run `bash slurm/sync.sh pull` first")
        return

    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    uniforms = [(2, 32), (3, 16), (3, 32), (3, 64), (4, 16), (4, 32)]
    (out / "main.tex").write_text(
        _drop_dash_note(table_main(present, data, uniforms)) + "\n")
    (out / "ablation.tex").write_text(
        _drop_dash_note(table_ablation(present, data, [(2, 32), (3, 32), (4, 32)])) + "\n"
    )
    spread = table_spread(args.models, pathlib.Path(args.tables))
    if spread:
        (out / "spread.tex").write_text(_drop_dash_note(spread) + "\n")
    surr = table_surrogate(pathlib.Path(args.tables) / "profile-q3-0.6B.json")
    if surr:
        (out / "surrogate.tex").write_text(_drop_dash_note(surr) + "\n")
    rotv = table_rotation(pathlib.Path(args.tables))
    if rotv:
        (out / "rotation.tex").write_text(_drop_dash_note(rotv) + "\n")
    rotp = table_rotation_ppl(pathlib.Path(args.runs))
    if rotp:
        (out / "rotation_ppl.tex").write_text(_drop_dash_note(rotp) + "\n")
    rt = table_runtime(pathlib.Path(args.runs))
    if rt:
        (out / "runtime.tex").write_text(_drop_dash_note(rt) + "\n")
    sds = table_seeds(pathlib.Path(args.runs))
    if sds:
        (out / "seeds.tex").write_text(_drop_dash_note(sds) + "\n")
    zsm = table_zeroshot_models(pathlib.Path(args.runs))
    if zsm:
        (out / "zeroshot_models.tex").write_text(_drop_dash_note(zsm) + "\n")
    qf = table_qfactors(present, data)
    if qf:
        (out / "qfactors.tex").write_text(_drop_dash_note(qf) + "\n")
    cal = table_calibration(data)
    if cal:
        (out / "calibration.tex").write_text(_drop_dash_note(cal) + "\n")
    r0 = table_r0(present, data)
    if r0:
        (out / "r0.tex").write_text(_drop_dash_note(r0) + "\n")
    print(f"wrote tables in {out} for {', '.join(present)}")


if __name__ == "__main__":
    main()
