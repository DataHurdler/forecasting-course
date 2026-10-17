#!/usr/bin/env python3
"""Figures for Lecture 6 (Regularization) — one script, every figure, both formats.

    .venv/bin/python scripts/figures/lecture06_figures.py                     # all
    .venv/bin/python scripts/figures/lecture06_figures.py diamond_and_disc    # one, by name
    .venv/bin/python scripts/figures/lecture06_figures.py --retrain           # same: nothing cached

Writes Figures/Lecture06_Regularization/<name>.pdf (Beamer) and <name>.svg (Quarto), both vector,
and Figures/Lecture06_Regularization/measured.json with every number the figure prints.

diamond_and_disc: the RSS ellipses for a two-predictor regression of daily units on lag7 and
lag14, built with Lab 6's own feature pipeline and training window (Labs/Lecture06_lab.qmd,
Setup: CA_1 FOODS daily, the 365 test days held out). Predictors and target are standardized, as
the lab standardizes before every penalized fit. Nothing is trained or sampled: OLS, the ridge
path (R + lambda I)^-1 c and the LASSO path (LARS) are exact, so the figure needs no cache.

lasso_path: Lab 6, Step 3 verbatim (all 62 features standardized on the full training window,
Lasso on alphas = logspace(-1, 3, 60)), coloured so the lags that are multiples of 7 share one
colour. The last eight features standing must equal the lab's (seven are multiples of 7).

n_over_k: Lab 6, Step 1 verbatim (OLS and Ridge(alpha=100) on the last n training rows, for the
lab's five n), against the lab's benchmark, the same day last year (625).

elastic_net: the diamond_and_disc ellipses again, with the elastic-net ball of the slide's own
penalty, alpha |b|_1 + (1 - alpha) |b|_2^2 at alpha = 0.5, drawn at the diamond's axis reach. In
the first quadrant its boundary is a circle arc centred at (-a/2b, -a/2b), so the ball is exact.
Its contact point is found by minimising the RSS quadratic along that boundary. The corner still
zeros lag14, but only while the reach r satisfies (c2 - rho r)/(c1 - r) <= a/(a + 2 b r).

Lasso and least squares are deterministic, so nothing is seeded and nothing needs a cache; every
printed number still goes to measured.json (merged, one key per figure).

The geometry, exactly. With standardized columns, RSS(b) = RSS(b_ols) + n (b - b_ols)' R (b - b_ols),
where R is the 2 x 2 correlation matrix. So the contours are ellipses centred on OLS, the same for
both panels. The LASSO keeps lag14 at exactly zero while lag7's correlation with the residual
exceeds lag14's: c1 - t > c2 - rho t, i.e. for budgets t < (c1 - c2) / (1 - rho).
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
from matplotlib.patches import Circle, Polygon
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Figures" / "Lecture06_Regularization"
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


def record(update: dict) -> None:
    """Merge one figure's numbers into measured.json (the file holds every figure's)."""
    m = json.loads(MEASURED.read_text()) if MEASURED.exists() else {}
    m.update(update)
    MEASURED.write_text(json.dumps(m, indent=1) + "\n")


# ================================================================== the lab, verbatim
# Labs/Lecture06_lab.qmd, Setup. The lag and rolling columns decide which rows survive dropna,
# so they are all built even though only lag7 and lag14 are used here.
@lru_cache(maxsize=None)
def lab_train() -> pd.DataFrame:
    daily = (pd.read_csv(ROOT / "data" / "processed" / "m5_daily.csv", parse_dates=["date"])
               .sort_values("date"))
    u = daily.set_index("date")["units"]
    for L in list(range(1, 29)) + [35, 42, 49, 56, 364, 365, 366]:
        daily[f"lag{L}"] = u.shift(L).values
    for W in (7, 14, 28, 91, 364):
        daily[f"roll{W}_mean"] = u.shift(1).rolling(W).mean().values
        daily[f"roll{W}_std"] = u.shift(1).rolling(W).std().values
    df = daily.dropna(subset=[c for c in daily.columns if c.startswith(("lag", "roll"))])
    return df.iloc[:-365].copy()                      # N_TEST = 365, as in the lab


@lru_cache(maxsize=None)
def lab_full() -> tuple:
    """Lab 6 Setup, every feature: (FEATURES, train, test)."""
    daily = (pd.read_csv(ROOT / "data" / "processed" / "m5_daily.csv", parse_dates=["date"])
               .sort_values("date"))
    u = daily.set_index("date")["units"]
    for L in list(range(1, 29)) + [35, 42, 49, 56, 364, 365, 366]:
        daily[f"lag{L}"] = u.shift(L).values
    for W in (7, 14, 28, 91, 364):
        daily[f"roll{W}_mean"] = u.shift(1).rolling(W).mean().values
        daily[f"roll{W}_std"] = u.shift(1).rolling(W).std().values
    daily["dow"] = daily["date"].dt.dayofweek
    daily["doy"] = daily["date"].dt.dayofyear
    for k in (1, 2, 3):
        daily[f"sin{k}"] = np.sin(2 * np.pi * k * daily.doy / 365)
        daily[f"cos{k}"] = np.cos(2 * np.pi * k * daily.doy / 365)
    daily = pd.concat([daily, pd.get_dummies(daily.dow, prefix="dow")], axis=1)
    feats = ([c for c in daily.columns if c.startswith(("lag", "roll", "sin", "cos", "dow_"))]
             + ["snap", "is_event", "doy", "avg_price"])
    df = daily.dropna(subset=[c for c in feats if c.startswith(("lag", "roll"))]).copy()
    assert len(feats) == 62, len(feats)
    return feats, df.iloc[:-365], df.iloc[-365:]


rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


@lru_cache(maxsize=None)
def two_lags() -> dict:
    """R, c, OLS, the corner budget, and both solution paths for standardized lag7 and lag14."""
    tr = lab_train()
    X = tr[["lag7", "lag14"]].values.astype(float)
    X = (X - X.mean(0)) / X.std(0)                     # StandardScaler (ddof = 0)
    y = tr["units"].values.astype(float)
    y = (y - y.mean()) / y.std()
    n = len(y)
    R, c = X.T @ X / n, X.T @ y / n
    rho = float(R[0, 1])
    b_ols = np.linalg.solve(R, c)
    assert c[0] > c[1] > 0, "lag7 should enter the LASSO first"
    t_corner = float((c[0] - c[1]) / (1 - rho))        # lag14 = 0 for every budget below this

    # cross-check against sklearn's LARS path: same knot, same end point
    from sklearn.linear_model import lars_path
    _, _, coefs = lars_path(X, y, method="lasso")
    knots = coefs.T
    assert np.allclose(knots[1], [t_corner, 0], atol=1e-6) and np.allclose(knots[-1], b_ols, atol=1e-6)

    lams = np.logspace(4, -4, 600)
    ridge_path = np.array([np.linalg.solve(R + lam * np.eye(2), c) for lam in lams])
    return dict(n=n, R=R, c=c, rho=rho, b_ols=b_ols, t_corner=t_corner,
                l1_ols=float(np.abs(b_ols).sum()), ridge_path=ridge_path,
                lasso_knots=knots)


def ridge_at(s: float) -> np.ndarray:
    """The ridge solution whose l2 norm is s (bisection on lambda)."""
    d = two_lags()
    lo, hi = 1e-8, 1e6
    for _ in range(200):
        lam = np.sqrt(lo * hi)
        b = np.linalg.solve(d["R"] + lam * np.eye(2), d["c"])
        lo, hi = (lam, hi) if np.linalg.norm(b) > s else (lo, lam)
    return b


# ================================================================== the diamond and the disc
BUDGET = 0.10          # inside the corner regime (t < t_corner = 0.155), clear of its kink


def diamond_and_disc() -> None:
    d = two_lags()
    R, b0 = d["R"], d["b_ols"]
    s = BUDGET
    assert s < d["t_corner"]
    lasso_pt = np.array([s, 0.0])
    ridge_pt = ridge_at(s)

    def Q(b):
        e = np.asarray(b) - b0
        return float(e @ R @ e)

    # contour grid
    gx, gy = np.meshgrid(np.linspace(-0.3, 0.75, 700), np.linspace(-0.3, 0.6, 600))
    E1, E2 = gx - b0[0], gy - b0[1]
    QQ = R[0, 0] * E1**2 + 2 * R[0, 1] * E1 * E2 + R[1, 1] * E2**2

    XL, YL = (-0.15, 0.63), (-0.15, 0.44)
    fig, axes = plt.subplots(1, 2, figsize=sz(10.4, 4.9))
    fig.subplots_adjust(wspace=0.2, left=0.02, right=0.93, top=0.9, bottom=0.1)

    for ax, kind in zip(axes, ("ridge", "lasso")):
        col, light = (BLUE, LIGHT_BLUE) if kind == "ridge" else (CRIMSON, LIGHT_CRIMSON)
        ax.set_aspect("equal")
        ax.set_xlim(*XL); ax.set_ylim(*YL)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.plot(XL, [0, 0], color=GRAY, lw=0.9, zorder=1)
        ax.plot([0, 0], YL, color=GRAY, lw=0.9, zorder=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(XL[1] + 0.012, 0, "lag7", ha="left", va="center", fontsize=12, color=GRAY)
        ax.text(-0.015, YL[1], "lag14", ha="right", va="top", fontsize=12, color=GRAY)

        contact = ridge_pt if kind == "ridge" else lasso_pt
        q = Q(contact)
        ax.contour(gx, gy, QQ, levels=[q * 0.012, q * 0.045], colors=[GRAY],
                   linewidths=0.8, alpha=0.75, zorder=2)
        ax.contour(gx, gy, QQ, levels=[q], colors=[BLACK], linewidths=1.2, zorder=3)

        if kind == "ridge":
            ax.add_patch(Circle((0, 0), s, facecolor=light, edgecolor=col, lw=1.8, zorder=2))
            p = d["ridge_path"]
        else:
            ax.add_patch(Polygon([(s, 0), (0, s), (-s, 0), (0, -s)], closed=True,
                                 facecolor=light, edgecolor=col, lw=1.8, zorder=2))
            p = d["lasso_knots"]
        ax.plot(p[:, 0], p[:, 1], color=col, lw=1.7, zorder=4, solid_capstyle="round")

        ax.plot(*b0, "o", ms=6, color=BLACK, zorder=5)
        ax.text(b0[0] - 0.02, b0[1] + 0.025, "OLS", ha="right", va="bottom", fontsize=12,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=5)
        ax.plot(*contact, "o", ms=7.5, color=col, mec="white", mew=1.0, zorder=6)

    # labels, placed against the computed geometry: right of each contact point, clear of the
    # contact ellipse (which runs from upper left to lower right through it)
    ax = axes[0]
    ax.annotate(f"({ridge_pt[0]:.2f}, {ridge_pt[1]:.2f}):\nboth kept", xy=ridge_pt,
                xytext=(0.205, 0.035), fontsize=12, color=BLUE, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.9, shrinkB=4))
    ax = axes[1]
    ax.annotate(f"({s:.2f}, 0): lag14 = 0", xy=lasso_pt, xytext=(0.27, -0.105), fontsize=12,
                color=CRIMSON, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=CRIMSON, lw=0.9, shrinkB=4))
    tk = d["t_corner"]
    ax.plot([tk], [0], marker="|", ms=7, mew=1.8, color=CRIMSON, zorder=5)
    ax.annotate(f"t = {tk:.2f}:\nlag14 enters", xy=(tk, 0.0), xytext=(0.32, 0.035),
                fontsize=12, color=CRIMSON, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=CRIMSON, lw=0.9, shrinkA=2, shrinkB=5))

    axes[0].set_title("Ridge: a disc", fontsize=13, pad=4)
    axes[1].set_title("LASSO: a diamond", fontsize=13, pad=4)
    for a_ in axes:
        a_.apply_aspect()
    y_cap = axes[0].get_position().y0 - 0.04
    fig.text(0.5, y_cap, "two coefficients, constrained form; each λ corresponds to some budget t\n"
             "colored line: the solution as the budget t grows", ha="center", va="top",
             fontsize=11, color=GRAY, linespacing=1.25)
    save(fig, "diamond_and_disc")

    m = {"note": "Written by scripts/figures/lecture06_figures.py: every number the Lecture 6 "
                 "diamond-and-disc figure prints. Lab 6 training window, standardized lag7/lag14.",
         "n_train": d["n"], "rho_lag7_lag14": d["rho"], "corr_with_units": d["c"].tolist(),
         "ols": d["b_ols"].tolist(), "ols_l1": d["l1_ols"], "t_corner": d["t_corner"],
         "t_corner_share_of_ols_l1": d["t_corner"] / d["l1_ols"], "budget_drawn": s,
         "ridge_contact": ridge_pt.tolist(), "lasso_contact": lasso_pt.tolist()}
    record(m)
    print(f"  OLS {np.round(d['b_ols'], 3)}, rho {d['rho']:.3f}, corner wins for t < "
          f"{d['t_corner']:.3f} ({d['t_corner'] / d['l1_ols']:.0%} of OLS l1 {d['l1_ols']:.3f})")


# ================================================================== the LASSO path
WEEKLY = [f"lag{L}" for L in (7, 14, 21, 28, 35, 42, 49, 56, 364)]      # the multiples of 7
LAB_LAST_EIGHT = ["lag7", "lag28", "lag35", "lag364", "lag14", "lag21", "lag1", "lag56"]


def lasso_path() -> None:
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    feats, tr, _ = lab_full()
    Xs = StandardScaler().fit_transform(tr[feats].values.astype(float))
    y = tr["units"].values
    alphas = np.logspace(-1, 3, 60)
    paths = np.array([Lasso(alpha=a, max_iter=50000).fit(Xs, y).coef_ for a in alphas])
    # Lab 6's elimination order, verbatim
    survivors, order, exit_alpha = set(feats), [], {}
    for a, coefs in zip(alphas, paths):
        nz = set(np.array(feats)[coefs != 0])
        # ties leave together; list them by lag (descending, so the reversed list reads 7, 28)
        for gone in sorted(survivors - nz, key=lambda f: -feats.index(f)):
            order.append(gone); exit_alpha[gone] = float(a)
        survivors = nz
    last8 = [str(f) for f in reversed(order[-8:])]
    # lag7 and lag28 leave at the same alpha, so only the set (and each exit alpha) is defined
    assert set(last8) == set(LAB_LAST_EIGHT), last8
    la = np.log10(alphas)

    fig, ax = plt.subplots(figsize=sz(10.6, 4.5))
    for j, f in enumerate(feats):
        if f in WEEKLY:
            continue
        ax.plot(la, paths[:, j], color="#C4C4C9", lw=0.9, zorder=2)
    for j, f in enumerate(feats):
        if f in WEEKLY:
            ax.plot(la, paths[:, j], color=CRIMSON, lw=1.8, zorder=3)
    ax.axhline(0, color=GRAY, lw=0.8, zorder=1)
    ax.set_xlim(-1, 3.0)
    ax.set_xlabel("penalty strength, log₁₀ λ   (sklearn calls it alpha)", fontsize=12)
    ax.set_ylabel("coefficient\n(standardized)", fontsize=12)
    ax.tick_params(labelsize=11)
    # the elimination order, as a list beside the plot: last to leave at the top
    groups, prev = [], None
    for f in last8:
        if prev is not None and exit_alpha[f] == exit_alpha[prev]:
            groups[-1].append(f)
        else:
            groups.append([f])
        prev = f
    x0 = 1.03
    ax.text(x0, 1.0, "last to leave:", transform=ax.transAxes, ha="left", va="top",
            fontsize=11.5, color=BLACK)
    for i, g in enumerate(groups):
        yy = 0.87 - i * 0.105
        xx = x0
        for jj, f in enumerate(g):
            word = f + ("," if jj < len(g) - 1 else "")
            t = ax.text(xx, yy, word, transform=ax.transAxes, ha="left", va="top",
                        fontsize=11.5, family="monospace",
                        color=CRIMSON if f in WEEKLY else GRAY)
            xx += 0.022 + 0.0185 * len(word)
    ax.text(x0, 0.87 - len(groups) * 0.105 - 0.03, "red: a multiple\nof 7 days",
            transform=ax.transAxes, ha="left", va="top", fontsize=11, color=CRIMSON)
    save(fig, "lasso_path")
    record({"lasso_path": {"alphas_log10": [-1, 3, 60], "n_features": len(feats),
                           "n_train": len(tr), "last_eight": last8,
                           "exit_alpha": {f: exit_alpha[f] for f in last8},
                           "nonzero_at_smallest_alpha": int((paths[0] != 0).sum())}})
    print("  last eight standing:", ", ".join(last8))


# ================================================================== test RMSE against n/k
LAB_N = (1210, 400, 200, 120, 80)


def n_over_k() -> None:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    feats, tr, te = lab_full()
    k = len(feats)
    Xte, yte = te[feats].values.astype(float), te["units"].values
    rows = []
    for n in LAB_N:
        sub = tr.iloc[-n:]
        sc = StandardScaler().fit(sub[feats].values.astype(float))
        Xs, Xt, y = sc.transform(sub[feats].values.astype(float)), sc.transform(Xte), sub.units.values
        rows.append(dict(n=n, n_over_k=n / k,
                         ols=rmse(yte, LinearRegression().fit(Xs, y).predict(Xt)),
                         ridge=rmse(yte, Ridge(alpha=100).fit(Xs, y).predict(Xt))))
    bench = rmse(yte, te.units_lag365.values)
    r = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=sz(10.4, 4.5))
    ax.axhline(bench, color=GRAY, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(0.96, bench, f"benchmark\n(same day last\nyear): {bench:,.0f}", ha="left",
            va="center", fontsize=11, color=GRAY, clip_on=False)
    ax.plot(r.n_over_k, r.ols, "-o", color=CRIMSON, lw=2.0, ms=5.5, zorder=3)
    ax.plot(r.n_over_k, r.ridge, "-o", color=BLUE, lw=2.0, ms=5.5, zorder=3)

    def fmt(v):
        return f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v:,.0f}"
    for i, row in r.iloc[1:].iterrows():
        right = i == 2                    # 5,874: the line climbs steeply just to its right
        ax.text(row.n_over_k * (1.06 if right else 1), row.ols * 1.7, fmt(row.ols),
                ha="right" if right else "center", va="bottom", fontsize=11, color=CRIMSON)
    first, last = r.iloc[0], r.iloc[-1]
    ax.text(first.n_over_k * 1.04, first.ols / 1.6, fmt(first.ols), ha="right", va="top",
            fontsize=11, color=CRIMSON)
    ax.text(first.n_over_k / 1.04, first.ridge / 1.6, fmt(first.ridge), ha="left", va="top",
            fontsize=11, color=BLUE)
    ax.text(last.n_over_k, last.ridge / 1.7, fmt(last.ridge), ha="center", va="top",
            fontsize=11, color=BLUE)
    ax.text(5.2, 2.7e4, "OLS", ha="right", va="center", fontsize=12.5, color=CRIMSON,
            fontweight="bold")
    ax.text(3.0, 180, "ridge", ha="center", va="center", fontsize=12.5, color=BLUE,
            fontweight="bold")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(26, 1.0)                                    # data gets scarcer to the right
    ax.set_ylim(50, 3e7)
    ax.set_xticks(list(r.n_over_k.round(1)))
    ax.set_xticklabels([f"{v:.1f}" for v in r.n_over_k])
    ax.minorticks_off()
    ax.set_yticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    ax.set_yticklabels(["100", "1k", "10k", "100k", "1M", "10M"])
    ax.tick_params(labelsize=11)
    ax.set_xlabel(f"training rows per predictor, n/k   (k = {k})", fontsize=12)
    ax.set_ylabel("test RMSE (log scale)", fontsize=12)
    save(fig, "n_over_k")
    record({"n_over_k": {"k": k, "benchmark_same_day_last_year": bench,
                         "rows": r.to_dict(orient="records")}})
    print("  n/k: " + "; ".join(f"n={row.n} OLS {row.ols:,.0f} ridge {row.ridge:,.0f}"
                                for row in r.itertuples()) + f"; benchmark {bench:,.0f}")


# ================================================================== the elastic-net ball
EN_ALPHA = 0.5


def en_reach_to_level(r: float, a: float, b: float) -> float:
    return a * r + b * r * r


def en_boundary(level: float, a: float, b: float, th: np.ndarray) -> np.ndarray:
    """Points of a|x|_1 + b|x|_2^2 = level along directions th (exact, by the quadratic)."""
    u = np.c_[np.cos(th), np.sin(th)]
    l1 = np.abs(u).sum(1)
    rad = (-a * l1 + np.sqrt((a * l1) ** 2 + 4 * b * level)) / (2 * b)
    return u * rad[:, None]


def en_contact(level: float, a: float, b: float) -> np.ndarray:
    d = two_lags()
    R, b0 = d["R"], d["b_ols"]
    th = np.linspace(-np.pi / 2, np.pi, 300001)
    P = en_boundary(level, a, b, th)
    E = P - b0
    return P[np.einsum("ij,jk,ik->i", E, R, E).argmin()]


def elastic_net() -> None:
    d = two_lags()
    R, b0, c, rho = d["R"], d["b_ols"], d["c"], d["rho"]
    a, b = EN_ALPHA, 1 - EN_ALPHA
    s = BUDGET                                          # the diamond's reach: (0.10, 0)
    level = en_reach_to_level(s, a, b)
    en_pt = en_contact(level, a, b)
    lasso_pt = np.array([s, 0.0])
    # the corner holds while (c2 - rho r)/(c1 - r) <= a/(a + 2 b r): bisect for r*
    lo, hi = 1e-9, s
    for _ in range(200):
        mid = (lo + hi) / 2
        ok = (c[1] - rho * mid) / (c[0] - mid) <= a / (a + 2 * b * mid)
        lo, hi = (mid, hi) if ok else (lo, mid)
    r_corner = lo
    # the EN solution path as the budget grows, out to OLS
    lev_ols = a * np.abs(b0).sum() + b * (b0 ** 2).sum()
    path = np.array([en_contact(L, a, b) for L in np.linspace(1e-4, lev_ols, 160)] + [b0])
    th_q = np.linspace(0, np.pi / 2, 400)
    diag = en_boundary(level, a, b, np.array([np.pi / 4]))[0]

    def Q(bb):
        e = np.asarray(bb) - b0
        return float(e @ R @ e)

    gx, gy = np.meshgrid(np.linspace(-0.3, 0.75, 700), np.linspace(-0.3, 0.6, 600))
    E1, E2 = gx - b0[0], gy - b0[1]
    QQ = R[0, 0] * E1**2 + 2 * R[0, 1] * E1 * E2 + R[1, 1] * E2**2
    XL, YL = (-0.15, 0.63), (-0.15, 0.44)
    fig, axes = plt.subplots(1, 2, figsize=sz(10.4, 4.9))
    fig.subplots_adjust(wspace=0.2, left=0.02, right=0.93, top=0.9, bottom=0.1)
    th = np.linspace(0, 2 * np.pi, 1441)
    for ax, kind in zip(axes, ("lasso", "en")):
        col, light = (CRIMSON, LIGHT_CRIMSON) if kind == "lasso" else (GREEN, "#E7F4EC")
        ax.set_aspect("equal"); ax.set_xlim(*XL); ax.set_ylim(*YL)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.plot(XL, [0, 0], color=GRAY, lw=0.9, zorder=1)
        ax.plot([0, 0], YL, color=GRAY, lw=0.9, zorder=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(XL[1] + 0.012, 0, "lag7", ha="left", va="center", fontsize=12, color=GRAY)
        ax.text(-0.015, YL[1], "lag14", ha="right", va="top", fontsize=12, color=GRAY)
        contact = lasso_pt if kind == "lasso" else en_pt
        q = Q(contact)
        ax.contour(gx, gy, QQ, levels=[q * 0.012, q * 0.045], colors=[GRAY],
                   linewidths=0.8, alpha=0.75, zorder=2)
        ax.contour(gx, gy, QQ, levels=[q], colors=[BLACK], linewidths=1.2, zorder=3)
        if kind == "lasso":
            ax.add_patch(Polygon([(s, 0), (0, s), (-s, 0), (0, -s)], closed=True,
                                 facecolor=light, edgecolor=col, lw=1.8, zorder=2))
            p = d["lasso_knots"]
        else:
            ball = en_boundary(level, a, b, th)
            ax.add_patch(Polygon(ball, closed=True, facecolor=light, edgecolor=col, lw=1.8,
                                 zorder=2))
            p = path
        ax.plot(p[:, 0], p[:, 1], color=col, lw=1.7, zorder=4, solid_capstyle="round")
        ax.plot(*b0, "o", ms=6, color=BLACK, zorder=5)
        ax.text(b0[0] - 0.02, b0[1] + 0.025, "OLS", ha="right", va="bottom", fontsize=12,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=5)
        ax.plot(*contact, "o", ms=7.5, color=col, mec="white", mew=1.0, zorder=6)
    axes[0].annotate(f"({s:.2f}, 0): lag14 = 0", xy=lasso_pt, xytext=(0.27, -0.105),
                     fontsize=12, color=CRIMSON, ha="left", va="center",
                     arrowprops=dict(arrowstyle="-", color=CRIMSON, lw=0.9, shrinkB=4))
    axes[1].annotate(f"({en_pt[0]:.2f}, {en_pt[1]:.2f}):\nboth kept", xy=en_pt,
                     xytext=(0.205, 0.035), fontsize=12, color=GREEN, ha="left", va="bottom",
                     arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.9, shrinkB=4))
    axes[0].set_title("LASSO: a diamond", fontsize=13, pad=4)
    axes[1].set_title(f"Elastic net, α = {EN_ALPHA}", fontsize=13, pad=4)
    for a_ in axes:
        a_.apply_aspect()
    y_cap = axes[0].get_position().y0 - 0.04
    fig.text(0.5, y_cap, "same ellipses, same reach along the axes (0.10)\n"
             "colored line: the solution as the budget grows", ha="center", va="top",
             fontsize=11, color=GRAY, linespacing=1.25)
    save(fig, "elastic_net")
    record({"elastic_net": {"alpha": EN_ALPHA, "reach": s, "level": level,
                            "contact": en_pt.tolist(), "lasso_contact": lasso_pt.tolist(),
                            "diagonal_point": diag.tolist(), "diamond_diagonal": s / 2,
                            "corner_holds_below_reach": r_corner,
                            "lasso_corner_holds_below": d["t_corner"]}})
    print(f"  elastic net: contact ({en_pt[0]:.3f}, {en_pt[1]:.3f}); diagonal {diag[0]:.4f} vs "
          f"diamond {s / 2:.4f}; corner holds below reach {r_corner:.3f} "
          f"(LASSO {d['t_corner']:.3f})")


FIGURES = {"diamond_and_disc": diamond_and_disc, "lasso_path": lasso_path,
           "n_over_k": n_over_k, "elastic_net": elastic_net}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if not a.startswith("--")] or list(FIGURES)
    print(f"Lecture 6 figures -> {OUT.relative_to(ROOT)}   (font: {FONT})")
    for name in pick:
        FIGURES[name]()
