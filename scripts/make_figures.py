#!/usr/bin/env python
"""Build the manuscript figures from pulled summaries. Runs on the laptop.

Reads ``results/tables/*.json`` (written by analyze_stats.py) and
``out/runs/*/results.json`` (written by sweep.py), and writes PDFs into
``paper/figures/``. Nothing here touches the cluster: the heavy artifacts
stay there and only summaries travel, per the repository's sync policy.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from qlr.analysis import envelope, envelope_saving, is_allocated  # noqa: E402

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

KINDS = ["q", "k", "v", "o", "gate", "up", "down"]
KCOL = dict(zip(KINDS, plt.cm.tab10(np.linspace(0, 0.9, len(KINDS)))))

# Families of operating points, and how they are drawn.
SERIES = [
    ("gptq",    "GPTQ (no low-rank)",           "tab:gray",   "o", "--"),
    ("olrc",    "GPTQ + OLrC (uniform)",        "tab:orange", "s", "--"),
    ("gilora",  "GPTQ-intrinsic LoRA (uniform)","tab:blue",   "^", "-"),
    ("allocNW", "BRAID, no sensitivity weights","tab:green",  "v", ":"),
    ("alloc",   "BRAID (allocated)",            "tab:red",    "D", "-"),
]


def family(name: str) -> str:
    m = re.match(r"([A-Za-z]+)", name)
    return m.group(1) if m else name


def load_results(path) -> list[dict]:
    with open(path) as fh:
        return json.load(fh)


def fig_heterogeneity(summary: dict, out: pathlib.Path, title: str) -> None:
    """The empirical claim the whole method rests on: layers are not alike.

    Left: every layer, by depth and projection type -- the structure a uniform
    configuration ignores. Right: the marginal distribution per type. A
    min-max range would be dominated by a handful of extreme layers, so the
    box plot shows quartiles and the scatter shows everything.
    """
    per = summary.get("per_layer")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9),
                             gridspec_kw={"width_ratios": [2.1, 1]})
    if per:
        for k in KINDS:
            pts = [(d["block"], d["log2_a"]) for d in per if d["kind"] == k]
            if pts:
                x, y = zip(*pts)
                axes[0].scatter(x, y, s=11, color=KCOL[k], label=k, alpha=0.85)
        axes[0].set_xlabel("transformer block")
        axes[0].set_ylabel(r"$\log_2 a_\ell$")
        axes[0].legend(ncol=7, frameon=False, fontsize=6.5,
                       loc="upper center", bbox_to_anchor=(0.5, -0.22),
                       handletextpad=0.2, columnspacing=0.9)
        groups = [[d["log2_a"] for d in per if d["kind"] == k] for k in KINDS]
        keep = [(k, g) for k, g in zip(KINDS, groups) if g]
        bp = axes[1].boxplot([g for _, g in keep], vert=True, widths=0.6,
                             patch_artist=True, showfliers=True,
                             flierprops=dict(marker=".", ms=2, alpha=0.5))
        for patch, (k, _) in zip(bp["boxes"], keep):
            patch.set_facecolor(KCOL[k]); patch.set_alpha(0.6)
        for med in bp["medians"]:
            med.set_color("k")
        axes[1].set_xticklabels([k for k, _ in keep], fontsize=7)
        axes[1].set_xlabel("projection")
        axes[1].tick_params(labelleft=False)
        axes[1].set_ylim(axes[0].get_ylim())
    sp = summary["log2_a"]
    fig.suptitle(f"{title}: difficulty spans {sp['range']:.0f} bits "
                 f"(sd {sp['sd']:.1f}, IQR {sp.get('iqr', float('nan')):.1f})", y=1.02)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_tails(summary: dict, ranks: list[int], out: pathlib.Path, title: str) -> None:
    """Why a single rank cannot be right: the spectra differ by projection."""
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    rr = [r for r in ranks if r > 0]
    for k, d in summary["tail"].items():
        if k not in KCOL:
            continue
        ax.plot(rr, d["fractions"], marker="o", ms=3.5, color=KCOL[k], label=k)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("rank $r$")
    ax.set_ylabel(r"$T_\ell(r)\,/\,\|X_\ell\|_F^2$")
    ax.set_ylim(0, 1)
    ax.legend(ncol=2, frameon=False)
    ax.set_title(f"{title}: spectral energy left after rank $r$")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_pareto(results: list[dict], out: pathlib.Path, title: str,
               metric: str = "wikitext2_ppl") -> None:
    """The headline: perplexity against effective bits per weight.

    Uniform low-rank configurations are drawn as one curve *per base
    bit-width*, not as a single curve through all of them. Two points with
    similar effective bit-width but different base precision -- say 2-bit with
    rank 64 and 3-bit with rank 16 -- are not two points on one trade-off, and
    joining them would draw a curve no method traces.
    """
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    base = next((r for r in results if r["name"] == "fp16"), None)

    def pts(pred):
        out_ = [(r["bits_per_weight"], r[metric]) for r in results
                if metric in r and r["name"] != "fp16" and pred(r["name"])]
        return sorted(out_)

    # no low-rank component: one curve across bit-widths
    p = pts(lambda n: re.fullmatch(r"gptq-b\d+", n) is not None)
    if p:
        x, y = zip(*p)
        ax.plot(x, y, "o--", ms=4, lw=1.3, color="tab:gray",
                label="GPTQ (no low-rank)")

    # uniform low-rank: one curve per base bit-width
    for i, b in enumerate(sorted({int(m.group(1)) for r in results
                                  if (m := re.fullmatch(r"gilora-b(\d+)r\d+", r["name"]))})):
        p = pts(lambda n, b=b: re.fullmatch(rf"gilora-b{b}r\d+", n) is not None)
        if not p:
            continue
        x, y = zip(*p)
        ax.plot(x, y, "^-", ms=4, lw=1.1, color="tab:blue", alpha=0.85,
                label="GPTQ-intrinsic LoRA (uniform)" if i == 0 else None)
        ax.annotate(f"$b={b}$", (x[-1], y[-1]), fontsize=6.5, color="tab:blue",
                    xytext=(3, 2), textcoords="offset points")

    p = pts(lambda n: re.fullmatch(r"olrc-b\d+r\d+", n) is not None)
    if p:
        x, y = zip(*p)
        ax.scatter(x, y, s=16, marker="s", facecolors="none", edgecolors="tab:orange",
                   lw=0.9, label="GPTQ + OLrC (uniform)")

    for fam, lab, col, mk, ls in [
        ("allocNW", "BRAID, no sensitivity weights", "tab:green", "v", ":"),
        ("alloc", "BRAID (allocated)", "tab:red", "D", "-"),
    ]:
        p = pts(lambda n, fam=fam: re.fullmatch(rf"{fam}(T)?-.*", n) is not None)
        if p:
            x, y = zip(*p)
            ax.plot(x, y, marker=mk, ms=4.5, color=col, ls=ls, lw=1.6, label=lab)

    if base:
        ax.axhline(base[metric], color="k", lw=0.8, ls="-.",
                   label=f"FP16 ({base[metric]:.2f})")
    ax.set_yscale("log")
    ax.set_xlabel("effective bits per weight")
    ax.set_ylabel("WikiText-2 perplexity")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper right", fontsize=7.2)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _grid_axes(n, figsize):
    """A 2 x ceil(n/2) grid, returning the figure and the used axes."""
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=figsize, squeeze=False)
    flat = [a for row in axes for a in row]
    for a in flat[n:]:
        a.set_visible(False)
    return fig, flat[:n]


def fig_heterogeneity_grid(items, out: pathlib.Path) -> None:
    """One 2x2 composite instead of four separate panels tiled by LaTeX.

    Composing the panels here rather than with four \\includegraphics keeps the
    aspect ratio and the font size under our control: a wide single-model panel
    dropped into half a text column renders its labels at about four points.

    ``items`` is a list of ``(title, summary)`` pairs.
    """
    fig, axes = _grid_axes(len(items), (7.0, 2.6 * ((len(items) + 1) // 2)))
    for ax, (title, summary) in zip(axes, items):
        per = summary.get("per_layer") or []
        for k in KINDS:
            pts = [(d["block"], d["log2_a"]) for d in per if d["kind"] == k]
            if pts:
                x, y = zip(*pts)
                ax.scatter(x, y, s=9, color=KCOL[k], label=k, alpha=0.85)
        sp = summary["log2_a"]
        ax.set_title(f"{title}  (sd {sp['sd']:.1f} bits)", fontsize=8.5)
        ax.set_xlabel("transformer block")
        ax.set_ylabel(r"$\log_2 a_\ell$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(labels), frameon=False, fontsize=8,
               loc="lower center", bbox_to_anchor=(0.5, -0.015),
               handletextpad=0.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_pareto_grid(items, out: pathlib.Path, metric: str = "wikitext2_ppl") -> None:
    """One 2x2 composite of the perplexity-versus-rate frontiers."""
    fig, axes = _grid_axes(len(items), (7.2, 2.9 * ((len(items) + 1) // 2)))
    seen = {}
    for ax, (title, results) in zip(axes, items):
        base = next((r for r in results if r["name"] == "fp16"), None)

        def pts(pred):
            return sorted((r["bits_per_weight"], r[metric]) for r in results
                          if metric in r and r["name"] != "fp16" and pred(r["name"]))

        p = pts(lambda n: re.fullmatch(r"gptq-b\d+", n) is not None)
        if p:
            x, y = zip(*p)
            h, = ax.plot(x, y, "o--", ms=3.5, lw=1.2, color="tab:gray")
            seen.setdefault("GPTQ (no low-rank)", h)
        for i, b in enumerate(sorted({int(m.group(1)) for r in results
                if (m := re.fullmatch(r"gilora-b(\d+)r\d+", r["name"]))})):
            q = pts(lambda n, b=b: re.fullmatch(rf"gilora-b{b}r\d+", n) is not None)
            if not q:
                continue
            x, y = zip(*q)
            h, = ax.plot(x, y, "^-", ms=3.5, lw=1.0, color="tab:blue", alpha=0.85)
            seen.setdefault("GPTQ-intrinsic LoRA (uniform)", h)
            ax.annotate(f"$b={b}$", (x[-1], y[-1]), fontsize=6, color="tab:blue",
                        xytext=(3, 2), textcoords="offset points")
        q = pts(lambda n: re.fullmatch(r"olrc-b\d+r\d+", n) is not None)
        if q:
            x, y = zip(*q)
            h = ax.scatter(x, y, s=13, marker="s", facecolors="none",
                           edgecolors="tab:orange", lw=0.8)
            seen.setdefault("GPTQ + OLrC (uniform)", h)
        for fam, lab, col, mk, ls in [
                ("allocNW", "BRAID, no sensitivity weights", "tab:green", "v", ":"),
                ("allocZ", r"BRAID-0, bits only ($r=0$)", "tab:purple", "s", "--"),
                ("alloc", "BRAID (allocated)", "tab:red", "D", "-")]:
            q = pts(lambda n, fam=fam: re.fullmatch(rf"{fam}(T)?-.*", n) is not None)
            if q:
                x, y = zip(*q)
                h, = ax.plot(x, y, marker=mk, ms=4, color=col, ls=ls, lw=1.5)
                seen.setdefault(lab, h)
        if base:
            h = ax.axhline(base[metric], color="k", lw=0.8, ls="-.")
            seen.setdefault("FP16 (uncompressed)", h)
        ax.set_yscale("log")
        ax.set_xlabel("effective bits per weight")
        ax.set_ylabel("WikiText-2 perplexity")
        ax.set_title(f"{title}  (FP16 {base[metric]:.2f})" if base else title, fontsize=8.5)
    fig.legend(list(seen.values()), list(seen), ncol=3, frameon=False, fontsize=7.5,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _quantiles_from_hist(hist, qs=(25, 50, 75)):
    """Quantiles of a value->count histogram with string or int keys."""
    if not hist:
        return None
    vals = np.repeat([int(k) for k in hist], [int(v) for v in hist.values()])
    return np.percentile(vals, qs)


def fig_allocation_summary(items, out: pathlib.Path) -> None:
    """What the allocator does with the budget, across models.

    Two panels rather than eight: the bit-width mix is shown for the model with
    the densest sweep, and the rank choice -- which is the surprising half, since
    it stays far below the ranks the literature uses -- is shown for every model
    at once. ``items`` is ``(title, results)`` pairs.
    """
    dense = max(items, key=lambda t: sum(1 for r in t[1] if is_allocated(r["name"])))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    pts = sorted((r for r in dense[1] if is_allocated(r["name"]) and r.get("bit_histogram")),
                 key=lambda r: r["bits_per_weight"])
    allbits = sorted({int(b) for r in pts for b in r["bit_histogram"]})
    x = np.array([r["bits_per_weight"] for r in pts])
    bottom = np.zeros(len(pts))
    for c, b in zip(plt.cm.viridis(np.linspace(0.12, 0.92, len(allbits))), allbits):
        tot = np.array([sum(r["bit_histogram"].values()) for r in pts], float)
        cnt = np.array([r["bit_histogram"].get(str(b), r["bit_histogram"].get(b, 0))
                        for r in pts], float)
        frac = np.divide(cnt, tot, out=np.zeros_like(cnt), where=tot > 0)
        axes[0].bar(x, frac, bottom=bottom, width=0.13, color=c, label=f"{b}-bit")
        bottom += frac
    axes[0].set_xlabel("effective bits per weight")
    axes[0].set_ylabel("fraction of layers")
    axes[0].set_title(f"bit-widths chosen ({dense[0]})", fontsize=8.5)
    axes[0].legend(ncol=3, frameon=False, fontsize=6.5, loc="upper center")
    axes[0].grid(False)
    axes[0].set_ylim(0, 1.32)

    # Only the FP16-factor allocations here. allocQ prices a unit of rank at
    # 6 bits instead of 16, so it legitimately buys more of it; mixing the two
    # families on one axis reads as noise rather than as the rate model working.
    fp16_fam = re.compile(r"alloc(T)?-.*")
    for (title, results), col in zip(items, plt.cm.tab10(np.linspace(0, 0.4, len(items)))):
        pts = sorted((r for r in results if fp16_fam.fullmatch(r["name"])
                      and r.get("rank_histogram")),
                     key=lambda r: r["bits_per_weight"])
        if not pts:
            continue
        xs = [r["bits_per_weight"] for r in pts]
        q = np.array([_quantiles_from_hist(r["rank_histogram"]) for r in pts])
        axes[1].plot(xs, q[:, 1], "-o", ms=3, lw=1.2, color=col, label=title)
        axes[1].fill_between(xs, q[:, 0], q[:, 2], color=col, alpha=0.15, lw=0)
    # The cheap-factor allocations, on the densest model, as a price check:
    # halving the cost of rank should buy more of it.
    cheap = sorted((r for r in dense[1] if re.fullmatch(r"allocQ-.*", r["name"])
                    and r.get("rank_histogram")),
                   key=lambda r: r["bits_per_weight"])
    if cheap:
        q = np.array([_quantiles_from_hist(r["rank_histogram"]) for r in cheap])
        axes[1].plot([r["bits_per_weight"] for r in cheap], q[:, 1], "--x", ms=4,
                     lw=1.1, color="0.35",
                     label=f"{dense[0]}, 4/8-bit factors")
    axes[1].set_xlabel("effective bits per weight")
    axes[1].set_ylabel(r"rank $r_\ell$")
    axes[1].set_title("ranks chosen (median, IQR band)", fontsize=8.5)
    axes[1].margins(y=0.30)  # headroom so the legend clears the curves
    axes[1].legend(frameon=False, fontsize=6.2, loc="upper left", ncol=2,
                   handletextpad=0.4, columnspacing=0.9, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_gain_all(items, out: pathlib.Path) -> None:
    """The coding-gain prediction against the saving actually realized, all models.

    The realized saving is the horizontal distance from an allocated point to
    the baseline envelope at equal perplexity. Putting every model on one pair
    of axes makes the systematic optimism visible as a slope rather than as four
    separate scatter plots.
    """
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    marks = ["o", "s", "^", "D", "v"]
    lo = hi = 0.0
    for (title, results), col, mk in zip(items, plt.cm.tab10(np.linspace(0, 0.4, len(items))), marks):
        env = envelope(results)
        if len(env) < 2:
            continue
        xs, ys = [], []
        for r in sorted((r for r in results if is_allocated(r["name"])),
                        key=lambda r: r["bits_per_weight"]):
            v = r.get("wikitext2_ppl")
            pred = (r.get("coding_gain") or {}).get("bits_saved")
            if v is None or pred is None or v <= 0:
                continue
            sv, _ = envelope_saving(env, r["bits_per_weight"], math.log(v))
            if sv is None:
                continue
            xs.append(pred); ys.append(sv)
        if xs:
            ax.scatter(xs, ys, s=24, color=col, marker=mk, label=title,
                       alpha=0.85, edgecolors="none")
            lo = min(lo, min(ys)); hi = max(hi, max(xs + ys))
    hi *= 1.1
    ax.plot([0, hi], [0, hi], "k--", lw=0.9, label="predicted = realized")
    ax.plot([0, hi], [0, hi / 2], color="0.45", lw=0.9, ls=":",
            label="realized = half predicted")
    ax.axhline(0, color="0.7", lw=0.7)
    ax.set_xlim(0, hi); ax.set_ylim(min(lo * 1.15, -0.05), hi)
    ax.set_xlabel("predicted saving (coding gain), bits/weight")
    ax.set_ylabel("realized saving, bits/weight")
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_allocation(results: list[dict], out: pathlib.Path, title: str) -> None:
    """What the allocator actually does with the budget."""
    pts = sorted([r for r in results if family(r["name"]) == "alloc"],
                 key=lambda r: r["bits_per_weight"])
    if not pts:
        return
    allbits = sorted({int(b) for r in pts for b in r.get("bit_histogram", {})})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    x = [r["bits_per_weight"] for r in pts]
    bottom = np.zeros(len(pts))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(allbits)))
    for c, b in zip(cmap, allbits):
        frac = np.array([r.get("bit_histogram", {}).get(str(b),
                         r.get("bit_histogram", {}).get(b, 0)) for r in pts], float)
        tot = np.array([sum(r.get("bit_histogram", {}).values()) for r in pts], float)
        frac = np.divide(frac, tot, out=np.zeros_like(frac), where=tot > 0)
        axes[0].bar(x, frac, bottom=bottom, width=0.11, color=c, label=f"{b}-bit")
        bottom += frac
    axes[0].set_xlabel("effective bits per weight")
    axes[0].set_ylabel("fraction of layers")
    axes[0].set_title("bit-widths chosen")
    axes[0].legend(ncol=3, frameon=False, fontsize=6.5)
    axes[0].grid(False)

    for r in pts:
        rh = r.get("rank_histogram", {})
        vals = np.repeat([int(k) for k in rh], [int(v) for v in rh.values()])
        if vals.size:
            axes[1].scatter([r["bits_per_weight"]] * vals.size,
                            vals + np.random.default_rng(0).uniform(-.3, .3, vals.size),
                            s=3, alpha=0.25, color="tab:red")
    axes[1].set_xlabel("effective bits per weight")
    axes[1].set_ylabel("rank $r_\\ell$")
    axes[1].set_title("ranks chosen")
    fig.suptitle(title, y=1.04)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_gain_validation(results: list[dict], out: pathlib.Path, title: str) -> None:
    """The coding-gain prediction against the saving actually realized.

    The realized saving is the horizontal distance from an allocated point to
    the baseline envelope at equal perplexity -- how much rate every uniform
    configuration would need to match it. The prediction is computable from
    calibration statistics before any weight is quantized.
    """
    env = envelope(results)
    if len(env) < 2:
        return
    xs, ys, names = [], [], []
    for r in sorted((r for r in results if is_allocated(r["name"])),
                    key=lambda r: r["bits_per_weight"]):
        v = r.get("wikitext2_ppl")
        pred = (r.get("coding_gain") or {}).get("bits_saved")
        if v is None or pred is None or v <= 0:
            continue
        sv, _ = envelope_saving(env, r["bits_per_weight"], math.log(v))
        if sv is None:
            continue
        xs.append(pred)
        ys.append(sv)
        names.append(r["bits_per_weight"])
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    sc = ax.scatter(xs, ys, c=names, cmap="viridis", s=30, zorder=3)
    hi = max(max(xs), max(ys)) * 1.12
    ax.plot([0, hi], [0, hi], "k--", lw=0.8, label="predicted = realized")
    ax.plot([0, hi], [0, hi / 2], color="tab:red", lw=0.8, ls=":",
            label="realized = half predicted")
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("predicted saving (coding gain), bits/weight")
    ax.set_ylabel("realized saving, bits/weight")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    cb = fig.colorbar(sc, ax=ax, fraction=0.045)
    cb.set_label("bits per weight", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title(title)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)



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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["0.6B", "1.7B"])
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--runs", default="out/runs")
    ap.add_argument("--outdir", default="paper/figures")
    ap.add_argument("--per-model", action="store_true",
                    help="also write the single-model heterogeneity, Pareto, "
                         "allocation and gain panels. The manuscript uses the "
                         "composites and the per-model tail curves, so these are "
                         "for inspection only and are off by default.")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    made = []

    # Composites first: Figures 1 and 3 are 2x2 grids built here rather than
    # tiled by LaTeX, so their aspect ratios and font sizes survive.
    het_items, par_items = [], []
    for m in args.models:
        sp = _find(pathlib.Path(args.tables), [f"stats-q3-{m}.json", f"stats-{m}.json"])
        if sp is not None:
            het_items.append((_label(m), json.load(open(sp))))
        rp = _find(pathlib.Path(args.runs),
                   [f"sweep-q3-{m}/results.json", f"sweep-{m}/results.json"])
        if rp is not None:
            par_items.append((_label(m), load_results(rp)))
    if het_items:
        f = outdir / "heterogeneity-grid.pdf"
        fig_heterogeneity_grid(het_items, f); made.append(f)
    if par_items:
        f = outdir / "pareto-grid.pdf"
        fig_pareto_grid(par_items, f); made.append(f)
        f = outdir / "allocation-summary.pdf"
        fig_allocation_summary(par_items, f); made.append(f)
        f = outdir / "gain-all.pdf"
        fig_gain_all(par_items, f); made.append(f)

    for m in args.models:
        title = _label(m)
        sp = _find(pathlib.Path(args.tables), [f"stats-q3-{m}.json", f"stats-{m}.json"])
        rp = _find(pathlib.Path(args.runs),
                   [f"sweep-q3-{m}/results.json", f"sweep-{m}/results.json"])
        if sp is not None:
            summary = json.load(open(sp))
            # Figure 2 uses these directly, so they are not optional.
            f = outdir / f"tails-{m}.pdf"
            fig_tails(summary, [0, 8, 16, 32, 64], f, title); made.append(f)
            if args.per_model:
                f = outdir / f"heterogeneity-{m}.pdf"
                fig_heterogeneity(summary, f, title); made.append(f)
        if rp is not None and args.per_model:
            res = load_results(rp)
            f = outdir / f"pareto-{m}.pdf"; fig_pareto(res, f, title); made.append(f)
            f = outdir / f"allocation-{m}.pdf"; fig_allocation(res, f, title); made.append(f)
            f = outdir / f"gain-{m}.pdf"; fig_gain_validation(res, f, title); made.append(f)

    for f in made:
        print(f"wrote {f}")
    if not made:
        print("no inputs found -- pull results first (bash slurm/sync.sh pull)")


if __name__ == "__main__":
    main()
