#!/usr/bin/env python3
"""Figures for Lecture 4 (Decision Trees) — one script, every figure, both formats.

    .venv/bin/python scripts/figures/lecture04_figures.py                 # all
    .venv/bin/python scripts/figures/lecture04_figures.py cart_split      # one, by name
    .venv/bin/python scripts/figures/lecture04_figures.py --retrain       # re-measure

Writes Figures/Lecture04_DecisionTrees/<name>.pdf (Beamer) and <name>.svg (Quarto), both vector.

The data are the course's weekly CA_1 FOODS series (data/processed/m5_weekly.csv) with the eight
numeric columns it ships: units_lag1-4, units_lag52, snap_days, event_days, avg_price. Rows need
every lag, so the first 52 weeks drop out. The final 52 weeks are the test year, as in Lab 1, and
everything here is fitted on the 173 weeks before them. (Homework 3's cutoff, 2015-05-17, holds
out one week more; the root split, the depth-2 tree and the importances are identical on it.)

Every number the deck prints about these trees is written to measured.json by this script:
the root split behind the CART figure, the depth-2 tree in the flowchart and on the
"explaining a prediction" slide, the importance table, and the pruning path. Trees are
deterministic given random_state=42, so a re-run reproduces the file exactly.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Figures" / "Lecture04_DecisionTrees"
MEASURED = OUT / "measured.json"

# The deck's palette — Preambles/header.tex and Quarto/uno-clean.scss.
CRIMSON, BLACK, GRAY = "#D71920", "#1A1A1A", "#525252"
GREEN, AMBER, BLUE = "#15803D", "#B45309", "#005CA9"
LIGHT_CRIMSON, LIGHT_GRAY, LIGHT_BLUE = "#FDECEA", "#F4F4F5", "#E8F0FA"

_fonts = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ("Source Sans Pro", "Source Sans 3", "Helvetica Neue", "Arial")
             if f in _fonts), "DejaVu Sans")
# Drawn near display size, as in Lecture 8: see the SCALE note in lecture08_figures.py.
SCALE = 0.55


def sz(w: float, h: float) -> tuple[float, float]:
    return (w * SCALE, h * SCALE)


# SVG text is drawn as outlines; see lecture09_part1_figures.py for why live text breaks.
plt.rcParams.update({"font.family": FONT, "font.size": 13, "svg.fonttype": "path",
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.05,
                    transparent=True, dpi=200)
    plt.close(fig)
    print(f"  {name}.pdf / .svg")


# ================================================================== data and measurement
FEATURES = ["units_lag1", "units_lag2", "units_lag3", "units_lag4", "units_lag52",
            "snap_days", "event_days", "avg_price"]
N_TEST = 52
SEED = 42


@lru_cache(maxsize=None)
def training_weeks() -> pd.DataFrame:
    w = pd.read_csv(ROOT / "data" / "processed" / "m5_weekly.csv", parse_dates=["week_start"])
    g = (w[(w.store_id == "CA_1") & (w.cat_id == "FOODS")]
         .sort_values("week_start").dropna(subset=FEATURES).reset_index(drop=True))
    return g.iloc[:-N_TEST]


def sse_by_threshold(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Every cut CART can try on one feature: (threshold, SSE_left + SSE_right, n_left).

    A threshold sits halfway between two neighbouring distinct values, as in scikit-learn.
    """
    o = np.argsort(x, kind="stable")
    xs, ys = x[o], y[o]
    rows = []
    for i in range(1, len(xs)):
        if xs[i] == xs[i - 1]:
            continue
        left, right = ys[:i], ys[i:]
        sse = ((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum()
        rows.append(((xs[i] + xs[i - 1]) / 2, sse, i))
    return np.array(rows)


def measure() -> dict:
    from sklearn.tree import DecisionTreeRegressor
    tr = training_weeks()
    X, y = tr[FEATURES].values, tr.units.values.astype(float)
    n = len(y)
    m: dict = {"note": "Written by scripts/figures/lecture04_figures.py: every number Lecture 4 "
                       "prints about the weekly CA_1 FOODS trees. Training = the 173 weeks before "
                       "the final 52; random_state=42.",
               "train_weeks": n, "train_first": str(tr.week_start.iloc[0].date()),
               "train_last": str(tr.week_start.iloc[-1].date())}

    # --- the root split, by brute force, and checked against scikit-learn's own root
    curve = sse_by_threshold(tr.units_lag4.values.astype(float), y)
    k = int(curve[:, 1].argmin())
    t_best, nL = float(curve[k, 0]), int(curve[k, 2])
    o = np.argsort(tr.units_lag4.values, kind="stable")
    sst = float(((y - y.mean()) ** 2).sum())
    stump = DecisionTreeRegressor(max_depth=1, random_state=SEED).fit(X, y).tree_
    assert FEATURES[stump.feature[0]] == "units_lag4" and abs(stump.threshold[0] - t_best) < 1e-6, \
        "brute-force root split disagrees with scikit-learn"
    m["root"] = {"feature": "units_lag4", "threshold": t_best, "sst": sst,
                 "sse": float(curve[k, 1]), "removed": 1 - float(curve[k, 1]) / sst,
                 "n_left": nL, "n_right": n - nL, "mean_all": float(y.mean()),
                 "mean_left": float(y[o][:nL].mean()), "mean_right": float(y[o][nL:].mean())}
    m["root_curve"] = curve[:, :2].tolist()

    # --- the depth-2 tree (the flowchart, and the path read to the manager)
    t2 = DecisionTreeRegressor(max_depth=2, random_state=SEED).fit(X, y).tree_
    m["depth2"] = [{"node": i, "feature": FEATURES[t2.feature[i]] if t2.feature[i] >= 0 else None,
                    "threshold": float(t2.threshold[i]), "n": int(t2.n_node_samples[i]),
                    "mean": float(t2.value[i].ravel()[0])} for i in range(t2.node_count)]

    # --- the importance table: the deck's own template, max_depth=5, min_samples_leaf=10
    t5 = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=SEED).fit(X, y)
    m["importance_depth5"] = dict(sorted(zip(FEATURES, map(float, t5.feature_importances_)),
                                         key=lambda kv: -kv[1]))

    # --- cost-complexity pruning: the nested subtrees, in SSE units (sklearn's alpha is SSE/n)
    full = DecisionTreeRegressor(random_state=SEED).fit(X, y)
    path = full.cost_complexity_pruning_path(X, y)
    leaves = [DecisionTreeRegressor(random_state=SEED, ccp_alpha=a).fit(X, y).get_n_leaves()
              for a in path.ccp_alphas]
    m["pruning"] = {"alpha_sse": (path.ccp_alphas * n).tolist(), "leaves": [int(v) for v in leaves],
                    "sse": (path.impurities * n).tolist(), "full_leaves": int(full.get_n_leaves())}
    # the slide says the tree alpha = 34M picks is the flowchart's depth-2 tree: check it is
    pruned = DecisionTreeRegressor(random_state=SEED, ccp_alpha=34e6 / n).fit(X, y)
    assert np.allclose(np.sort(np.unique(pruned.predict(X))), np.sort(t2.value[t2.children_left == -1].ravel())), \
        "the alpha = 34M pruned tree is not the depth-2 tree"
    # depth caps, read off the same full tree (so each one is a genuine subtree of it)
    T = full.tree_
    depth = np.zeros(T.node_count, int)
    for i in range(T.node_count):
        for c in (T.children_left[i], T.children_right[i]):
            if c != -1:
                depth[c] = depth[i] + 1
    caps = []
    for d in range(1, 6):
        nodes = [i for i in range(T.node_count)
                 if depth[i] <= d and (depth[i] == d or T.children_left[i] == -1)]
        caps.append({"max_depth": d, "leaves": len(nodes),
                     "sse": float(sum(T.n_node_samples[i] * T.impurity[i] for i in nodes))})
    m["depth_caps"] = caps

    OUT.mkdir(parents=True, exist_ok=True)
    MEASURED.write_text(json.dumps(m, indent=1) + "\n")
    r = m["root"]
    print(f"  measured.json written: root units_lag4 <= {r['threshold']:,.1f}, "
          f"removes {r['removed']:.1%} of {r['sst'] / 1e6:,.0f}M")
    return m


@lru_cache(maxsize=None)
def measured() -> dict:
    if "--retrain" in sys.argv or not MEASURED.exists():
        return measure()
    return json.loads(MEASURED.read_text())


# ================================================================== the CART split
def cart_split() -> None:
    """Slide a cut along units_lag4 and total the squared vertical distances to the two means."""
    m = measured()
    r = m["root"]
    tr = training_weeks()
    x = tr.units_lag4.values / 1000
    y = tr.units.values / 1000
    t = r["threshold"] / 1000
    left = x <= t
    mL, mR = r["mean_left"] / 1000, r["mean_right"] / 1000
    xlo, xhi = 11.6, 28.8

    fig, (ax, bx) = plt.subplots(2, 1, figsize=sz(7.4, 7.0), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.25, 1], hspace=0.12))
    # top: the cut, the two leaf means, and the residual sticks
    for xi, yi, lf in zip(x, y, left):
        ax.plot([xi, xi], [yi, mL if lf else mR], color=CRIMSON if lf else BLUE, lw=0.7,
                alpha=0.45, zorder=1)
    ax.scatter(x[left], y[left], s=10, color=CRIMSON, zorder=3, lw=0)
    ax.scatter(x[~left], y[~left], s=10, color=BLUE, zorder=3, lw=0)
    ax.plot([xlo, t], [mL, mL], color=CRIMSON, lw=2.6, zorder=4, solid_capstyle="butt")
    ax.plot([t, xhi], [mR, mR], color=BLUE, lw=2.6, zorder=4, solid_capstyle="butt")
    ax.axvline(t, color=BLACK, lw=1.4, ls=(0, (4, 2)), zorder=2)
    # labels sit where there are no points: above the left mean left of x = 16.9 (every point
    # there is below it), and below the right mean beyond x = 24.3 (every point there is above
    # it; the axis runs on to 28.8 to make room)
    ax.text(xlo + 0.1, mL + 0.4, f"mean {r['mean_left']:,.0f}", color=CRIMSON, fontsize=11,
            ha="left", va="bottom", fontweight="bold")
    ax.text(xhi - 0.1, mR - 0.4, f"mean {r['mean_right']:,.0f}", color=BLUE, fontsize=11,
            ha="right", va="top", fontweight="bold")
    ax.set_ylabel("units this week\n(thousands)", fontsize=11)
    ax.set_ylim(13.2, 27.6)
    ax.tick_params(labelsize=11)
    ax.text(t - 0.25, 27.5, f"cut at {r['threshold'] - 0.5:,.0f}", ha="right", va="top", fontsize=11)

    # bottom: total squared error for every cut, minimum marked
    c = np.array(m["root_curve"])
    bx.plot(c[:, 0] / 1000, c[:, 1] / 1e6, color=BLACK, lw=1.6)
    bx.axhline(r["sst"] / 1e6, color=GRAY, lw=1.0, ls=(0, (2, 2)))
    bx.text(xhi - 0.1, r["sst"] / 1e6 + 8, f"no split: {r['sst'] / 1e6:,.0f}M", ha="right",
            va="bottom", fontsize=10.5, color=GRAY)
    bx.axvline(t, color=BLACK, lw=1.4, ls=(0, (4, 2)))
    bx.plot([t], [r["sse"] / 1e6], "o", ms=7, color=CRIMSON, zorder=4)
    bx.text(t + 0.35, r["sse"] / 1e6 - 30,
            f"smallest: {r['sse'] / 1e6:,.0f}M\n{r['removed']:.0%} removed", ha="left",
            va="top", fontsize=11, color=CRIMSON)
    bx.set_ylim(270, 730)
    bx.set_yticks([300, 450, 600])
    bx.set_xlim(xlo, xhi)
    bx.set_xticks([12, 16, 20, 24, 28])
    bx.set_ylabel("SSE, left + right\n(millions)", fontsize=11)
    bx.set_xlabel("units four weeks earlier (thousands)", fontsize=11)
    bx.tick_params(labelsize=11)
    fig.align_ylabels([ax, bx])
    save(fig, "cart_split")


# ================================================================== cost-complexity pruning
def pruning_hull() -> None:
    """Training SSE against leaves; a line of slope -alpha touches the hull at the chosen tree."""
    m = measured()
    p = m["pruning"]
    L = np.array(p["leaves"])[::-1]                    # 1 leaf first
    S = np.array(p["sse"])[::-1] / 1e6
    A = np.array(p["alpha_sse"])[::-1] / 1e6           # alpha at which each subtree takes over
    XMAX = 21

    def chosen(alpha):                                  # the subtree minimising SSE + alpha*|T|
        i = int(np.argmin(S + alpha * L))
        return L[i], S[i]

    fig, ax = plt.subplots(figsize=sz(7.0, 4.4))
    keep = L <= XMAX + 3
    ax.plot(L[keep], S[keep], color=BLACK, lw=1.4, zorder=2)
    ax.plot(L[keep], S[keep], "o", ms=4.5, color=BLACK, zorder=3)
    # depth caps of the same full tree: genuine subtrees, so on or above the hull
    caps = [c for c in m["depth_caps"] if c["leaves"] <= XMAX]
    for c in caps:
        on_hull = np.any((L == c["leaves"]) & np.isclose(S, c["sse"] / 1e6))
        if not on_hull:
            ax.plot(c["leaves"], c["sse"] / 1e6, "o", ms=7.5, mfc="white", mec=GRAY, mew=1.6,
                    zorder=3)
            ax.text(c["leaves"] + 0.4, c["sse"] / 1e6 + 12, f"max_depth {c['max_depth']}",
                    ha="left", va="bottom", fontsize=10.5, color=GRAY)
    # the two penalty lines; their labels form a colour key in the empty upper right
    for j, (alpha, col) in enumerate([(34, CRIMSON), (15, BLUE)]):
        k, s = chosen(alpha)
        xs = np.array([0, XMAX])
        ax.plot(xs, s - alpha * (xs - k), color=col, lw=1.8, zorder=1)
        ax.plot([k], [s], "o", ms=9, color=col, zorder=4)
        ax.text(XMAX - 0.2, 655 - 62 * j, f"α = {alpha}M per leaf: {k} leaves", color=col,
                fontsize=11, ha="right", va="top", fontweight="bold")
    ax.text(XMAX - 0.2, 520, f"(the full tree: {p['full_leaves']} leaves, SSE 0)", ha="right",
            va="top", fontsize=10.5, color=GRAY)
    ax.set_xlim(0, XMAX); ax.set_ylim(0, 700)
    ax.set_xticks([1, 4, 9, 16, 21])
    ax.set_xlabel("leaves |T| in the pruned tree", fontsize=11)
    ax.set_ylabel("training SSE (millions)", fontsize=11)
    ax.tick_params(labelsize=10.5)
    save(fig, "pruning_hull")


FIGURES = {"cart_split": cart_split, "pruning_hull": pruning_hull}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if not a.startswith("--")] or list(FIGURES)
    print(f"Lecture 4 figures -> {OUT.relative_to(ROOT)}   (font: {FONT})")
    if "--retrain" in sys.argv:
        measured()
    for name in pick:
        FIGURES[name]()
