#!/usr/bin/env python3
"""Figures for Lecture 3 (Generalized Additive Models) — one script, every figure, both formats.

    .venv/bin/python scripts/figures/lecture03_figures.py                       # all
    .venv/bin/python scripts/figures/lecture03_figures.py smoothing_penalty     # one, by name

Writes Figures/Lecture03_GAMs/<name>.pdf (Beamer) and <name>.svg (Quarto), both vector.

Nothing is sampled: every fit is a deterministic pyGAM solve on electricity_daily.csv, a few
seconds in all. Every number a figure prints is written to Figures/Lecture03_GAMs/measured.json.

smoothing_penalty — one year of daily load against day of year, fitted three times with the same
    100-spline basis and only lambda changed. The basis is basis="ps" (not the lab's cyclic "cp"):
    with a cyclic basis the lambda -> infinity limit is a flat constant, not the OLS line, because
    a straight line is not periodic. pyGAM's penalty is the P-spline one (squared second
    differences of the coefficients), a discrete version of the integral of f''^2; both are zero
    exactly when f is a straight line. The strip under each fit shades f''(x)^2 computed from the
    fitted curve itself, so the shaded area is the slide's integral.
partial_dependence — Lab 3's own model (Step 6: s(doy, cyclic, 20 splines) + f(dow), fitted on
    2013-01-01 to 2017-07-31), its two partial dependence curves with pyGAM's 95% bands.
"""
from __future__ import annotations

import json
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Figures" / "Lecture03_GAMs"
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
# DejaVu Sans is the fallback for glyphs the main font lacks (Helvetica Neue has no "→").
plt.rcParams.update({"font.family": [FONT, "DejaVu Sans"], "font.size": 13, "svg.fonttype": "path",
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.05,
                    transparent=True, dpi=200)
    plt.close(fig)
    print(f"  {name}.pdf / .svg")


MONTHS = ([1, 91, 182, 274], ["Jan", "Apr", "Jul", "Oct"])
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]          # Lab 3: dow 0 = Monday

# smoothing_penalty settings
FIT_YEAR, HOLD_YEAR = 2014, 2015
N_SPLINES = 100
LAM_TINY, LAM_HUGE = 1e-4, 1e9
LAM_GRID = np.logspace(-4, 8, 49)


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


@lru_cache(maxsize=None)
def daily() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "processed" / "electricity_daily.csv", parse_dates=["date"])


def bending(gam, xx: np.ndarray) -> np.ndarray:
    """f''(x)^2 of the fitted curve, by finite differences on a fine grid (MW per day^2, squared)."""
    f = gam.predict(xx[:, None])
    return np.gradient(np.gradient(f, xx), xx) ** 2


# ================================================================== fits
@lru_cache(maxsize=None)
def penalty_fits():
    from pygam import LinearGAM, s
    d = daily()
    a, b = d[d.year == FIT_YEAR], d[d.year == HOLD_YEAR]
    X, y = a[["doy"]].values.astype(float), a.mw_mean.values
    Xh, yh = b[["doy"]].values.astype(float), b.mw_mean.values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lam_gcv = float(LinearGAM(s(0, n_splines=N_SPLINES, basis="ps"))
                        .gridsearch(X, y, lam=LAM_GRID, progress=False).lam[0][0])
        fits = {k: LinearGAM(s(0, n_splines=N_SPLINES, basis="ps", lam=lam)).fit(X, y)
                for k, lam in (("tiny", LAM_TINY), ("gcv", lam_gcv), ("huge", LAM_HUGE))}
    xx = np.arange(1, 365.0001, 0.05)
    out = {"X": X[:, 0], "y": y, "xx": xx, "lam": {"tiny": LAM_TINY, "gcv": lam_gcv, "huge": LAM_HUGE}}
    for k, g in fits.items():
        out[k] = dict(curve=g.predict(xx[:, None]), bend=bending(g, xx),
                      edof=float(g.statistics_["edof"]), train_rmse=rmse(y, g.predict(X)),
                      hold_rmse=rmse(yh, g.predict(Xh)))
        out[k]["penalty"] = float(np.trapezoid(out[k]["bend"], xx))
    # lambda -> infinity must be the OLS line: the penalty's only zero is a straight line
    ols = np.polyval(np.polyfit(X[:, 0], y, 1), xx)
    out["huge_vs_ols_max_mw"] = float(np.max(np.abs(out["huge"]["curve"] - ols)))
    assert out["huge_vs_ols_max_mw"] < 10, "huge-lambda fit should coincide with the OLS line"
    out["ols_hold_rmse"] = rmse(yh, np.polyval(np.polyfit(X[:, 0], y, 1), Xh[:, 0]))
    return out


@lru_cache(maxsize=None)
def robustness() -> list:
    """The same three fits for every fit year 2010-2016, each scored on the year after."""
    from pygam import LinearGAM, s
    d, rows = daily(), []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fy in range(2010, 2017):
            a, b = d[d.year == fy], d[d.year == fy + 1]
            X, y = a[["doy"]].values.astype(float), a.mw_mean.values
            Xh, yh = b[["doy"]].values.astype(float), b.mw_mean.values
            lam = float(LinearGAM(s(0, n_splines=N_SPLINES, basis="ps"))
                        .gridsearch(X, y, lam=LAM_GRID, progress=False).lam[0][0])
            r = {"fit_year": fy}
            for k, l in (("tiny", LAM_TINY), ("gcv", lam), ("huge", LAM_HUGE)):
                g = LinearGAM(s(0, n_splines=N_SPLINES, basis="ps", lam=l)).fit(X, y)
                r[k] = {"edof": round(float(g.statistics_["edof"]), 1),
                        "hold_rmse": round(rmse(yh, g.predict(Xh)))}
            rows.append(r)
    return rows


@lru_cache(maxsize=None)
def lab_gam():
    """Labs/Lecture03_lab.qmd, Setup + Step 6, verbatim in effect."""
    from pygam import LinearGAM, s, f
    d = daily()
    d = d[d.date >= "2013-01-01"].reset_index(drop=True)
    train, test = d[d.date < "2017-08-01"], d[d.date >= "2017-08-01"]
    X_tr, X_te = train[["doy", "dow"]].values, test[["doy", "dow"]].values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gam = LinearGAM(s(0, n_splines=20, basis="cp") + f(1)).fit(X_tr, train.mw_mean.values)
    XX = gam.generate_X_grid(term=0, n=366)
    pd0, ci0 = gam.partial_dependence(term=0, X=XX, width=0.95)
    XX1 = np.c_[np.zeros(7), np.arange(7)]
    pd1, ci1 = gam.partial_dependence(term=1, X=XX1, width=0.95)
    return dict(gam=gam, doy=XX[:, 0], pd0=pd0, ci0=ci0, pd1=pd1, ci1=ci1,
                edof=float(gam.statistics_["edof"]), test_rmse=rmse(test.mw_mean, gam.predict(X_te)),
                train_obs_per_doy=(float(train[train.doy <= 212].groupby("doy").size().mean()),
                                   float(train[train.doy > 212].groupby("doy").size().mean())))


@lru_cache(maxsize=None)
def measured() -> dict:
    p, L = penalty_fits(), lab_gam()
    doy, pd0 = L["doy"], L["pd0"]
    width0 = L["ci0"][:, 1] - L["ci0"][:, 0]
    winter = doy <= 60
    m = {
        "note": "Written by scripts/figures/lecture03_figures.py; every number the Lecture 3 "
                "figures print. electricity_daily.csv (PJM East).",
        "penalty": {
            "fit_year": FIT_YEAR, "holdout_year": HOLD_YEAR, "n_splines": N_SPLINES,
            "basis": "ps", "lambda_grid": "logspace(-4, 8, 49)",
            **{k: {"lam": p["lam"][k], "edof": p[k]["edof"], "train_rmse": p[k]["train_rmse"],
                   "holdout_rmse": p[k]["hold_rmse"], "integral_f2sq": p[k]["penalty"]}
               for k in ("tiny", "gcv", "huge")},
            "tiny_over_gcv_bending": p["tiny"]["penalty"] / p["gcv"]["penalty"],
            "huge_vs_ols_line_max_mw": p["huge_vs_ols_max_mw"],
            "ols_holdout_rmse": p["ols_hold_rmse"],
            "robustness_every_fit_year": robustness(),
        },
        "partial_dependence": {
            "model": "Lab 3 Step 6: LinearGAM(s(0, n_splines=20, basis='cp') + f(1)), train "
                     "2013-01-01..2017-07-31",
            "edof": L["edof"], "test_rmse": L["test_rmse"],
            "doy_peak": [float(doy[pd0.argmax()]), float(pd0.max())],
            "doy_trough": [float(doy[pd0.argmin()]), float(pd0.min())],
            "doy_winter_peak": [float(doy[winter][pd0[winter].argmax()]), float(pd0[winter].max())],
            "summer_peak_minus_trough": float(pd0.max() - pd0.min()),
            "winter_peak_minus_trough": float(pd0[winter].max() - pd0.min()),
            "doy_term_mean_over_train_note": "pyGAM does not centre the cyclic term on the data, "
                                             "so only differences along the curve are meaningful",
            "doy_band_full_width": {"min": float(width0.min()), "median": float(np.median(width0)),
                                    "max": float(width0.max()),
                                    "share_within_10pct_of_median":
                                        float(np.mean(abs(width0 - np.median(width0))
                                                      < 0.1 * np.median(width0)))},
            "train_obs_per_doy_before_after_day212": L["train_obs_per_doy"],
            "dow_effects": dict(zip(DOW, L["pd1"].tolist())),
            "dow_band_full_width": float(np.mean(L["ci1"][:, 1] - L["ci1"][:, 0])),
            "weekday_minus_weekend": float(L["pd1"][:5].mean() - L["pd1"][5:].mean()),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    MEASURED.write_text(json.dumps(m, indent=1) + "\n")
    return m


def kfmt(v, _=None) -> str:
    return f"{v / 1000:.0f}k"


# ================================================================== the penalty, as an area
def smoothing_penalty() -> None:
    p, m = penalty_fits(), measured()["penalty"]
    fig = plt.figure(figsize=sz(13.5, 4.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[3.0, 1], hspace=0.34, wspace=0.08,
                          left=0.07, right=0.995, top=0.8, bottom=0.11)
    ymax_bend = 1.15 * p["gcv"]["bend"].max()            # strips share the middle fit's scale
    heads = {"tiny": "λ ≈ 0: chases the noise", "gcv": "λ chosen by GCV",
             "huge": "λ → ∞: the OLS line"}
    for c, k in enumerate(("tiny", "gcv", "huge")):
        ax = fig.add_subplot(gs[0, c])
        ax.plot(p["X"], p["y"], "o", ms=1.8, color=GRAY, alpha=0.45, mew=0, zorder=1)
        ax.plot(p["xx"], p[k]["curve"], color=CRIMSON, lw=1.5 if k != "tiny" else 1.0, zorder=3)
        ax.set_xlim(1, 365); ax.set_ylim(21500, 46500)
        ax.set_xticks(MONTHS[0]); ax.set_xticklabels([])
        ax.set_yticks([25000, 35000, 45000])
        ax.yaxis.set_major_formatter(kfmt) if c == 0 else ax.set_yticklabels([])
        ax.tick_params(labelsize=11, length=3)
        ax.set_title(heads[k], fontsize=12, pad=17, fontweight="bold" if k == "gcv" else "normal",
                     color=CRIMSON if k == "gcv" else BLACK)
        ax.text(0.5, 1.02, f"edof {p[k]['edof']:.0f}  ·  {HOLD_YEAR} RMSE {p[k]['hold_rmse']:,.0f}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=11, color=GRAY)
        if c == 0:
            ax.set_ylabel(f"daily MW, {FIT_YEAR}", fontsize=11, labelpad=2)
        # the strip: f''(x)^2, shaded; its area is the penalty
        sx = fig.add_subplot(gs[1, c])
        bend = np.minimum(p[k]["bend"], ymax_bend)
        sx.fill_between(p["xx"], 0, bend, color=CRIMSON, alpha=0.55, lw=0)
        sx.set_xlim(1, 365); sx.set_ylim(0, ymax_bend)
        sx.set_yticks([])
        sx.spines["left"].set_visible(False)
        sx.set_xticks(MONTHS[0]); sx.set_xticklabels(MONTHS[1], fontsize=11)
        sx.tick_params(axis="x", length=3)
        # each strip's label sits in the gap above it, clear of the shading
        label = {"tiny": f"clipped: {m['tiny_over_gcv_bending']:.0f}× the middle area",
                 "gcv": "penalty = this area",
                 "huge": "zero: a line never bends"}[k]
        if c == 0:
            sx.set_ylabel("(f″)²", fontsize=11, rotation=0, ha="right", va="center", labelpad=6)
        sx.text(0.0, 1.06, label, transform=sx.transAxes, ha="left", va="bottom", fontsize=11,
                color=BLACK)
    save(fig, "smoothing_penalty")


# ================================================================== the lab's partial dependence
def partial_dependence() -> None:
    L, m = lab_gam(), measured()["partial_dependence"]
    fig = plt.figure(figsize=sz(13, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.6, 1], wspace=0.06, left=0.09, right=0.99,
                          top=0.86, bottom=0.14)
    ax = fig.add_subplot(gs[0])
    doy, pd0, ci0 = L["doy"], L["pd0"], L["ci0"]
    ax.fill_between(doy, ci0[:, 0], ci0[:, 1], color=LIGHT_CRIMSON, lw=0, zorder=1)
    ax.plot(doy, pd0, color=CRIMSON, lw=2.0, zorder=3)
    ax.set_xlim(1, 366); ax.set_ylim(-6200, 11800)
    ax.set_xticks(MONTHS[0]); ax.set_xticklabels(MONTHS[1], fontsize=11)
    ax.set_yticks([-5000, 0, 5000, 10000])
    ax.set_yticklabels(["−5,000", "0", "5,000", "10,000"], fontsize=11)
    ax.set_ylabel("effect on daily MW", fontsize=11)
    ax.set_title("s(day of year), with its 95% band", fontsize=12, pad=5)
    pk, tr, wp = m["doy_peak"], m["doy_trough"], m["doy_winter_peak"]
    # the two gaps a planner can use, drawn as brackets from the trough level
    ax.axhline(tr[1], color=GRAY, lw=0.8, ls=(0, (3, 2)), zorder=2)
    for x, top in [(pk[0], pk[1]), (wp[0], wp[1])]:
        ax.annotate("", xy=(x, top), xytext=(x, tr[1]),
                    arrowprops=dict(arrowstyle="<|-|>", color=BLACK, lw=1.1, mutation_scale=9,
                                    shrinkA=0, shrinkB=0), zorder=4)
    ax.text(38, 9600, f"January: +{m['winter_peak_minus_trough']:,.0f}", ha="left", va="center",
            fontsize=11)
    ax.text(262, 9600, f"July: +{m['summer_peak_minus_trough']:,.0f}", ha="left", va="center",
            fontsize=11)
    ax.text(tr[0], tr[1] - 600, "late-April trough", ha="center", va="top", fontsize=11, color=GRAY)

    bx = fig.add_subplot(gs[1], sharey=ax)
    x = np.arange(7)
    err = (L["pd1"] - L["ci1"][:, 0], L["ci1"][:, 1] - L["pd1"])
    bx.errorbar(x, L["pd1"], yerr=err, fmt="o", color=CRIMSON, ms=5, lw=1.4, capsize=3, zorder=3)
    bx.axhline(0, color=GRAY, lw=0.8, zorder=1)
    bx.set_xticks(x); bx.set_xticklabels(["M", "T", "W", "T", "F", "Sa", "Su"], fontsize=11)
    bx.tick_params(axis="y", labelleft=False)
    bx.set_xlim(-0.6, 6.6)
    bx.set_title("f(day of week)", fontsize=12, pad=5)
    bx.text(3, 7600, f"weekends:\n{m['weekday_minus_weekend']:,.0f} below\nweekdays",
            ha="center", va="center", fontsize=11, color=BLACK)
    save(fig, "partial_dependence")


FIGURES = {"smoothing_penalty": smoothing_penalty, "partial_dependence": partial_dependence}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if not a.startswith("--")] or list(FIGURES)
    print(f"Lecture 3 figures -> {OUT.relative_to(ROOT)}   (font: {FONT})")
    for name in pick:
        FIGURES[name]()
    m = measured()
    pen, pdp = m["penalty"], m["partial_dependence"]
    print("  penalty: " + "; ".join(f"{k} lam {pen[k]['lam']:.3g} edof {pen[k]['edof']:.1f} "
                                    f"holdout {pen[k]['holdout_rmse']:,.0f}"
                                    for k in ("tiny", "gcv", "huge"))
          + f"; bending ratio {pen['tiny_over_gcv_bending']:.0f}x")
    print(f"  lab GAM: edof {pdp['edof']:.1f}, test RMSE {pdp['test_rmse']:,.0f}, summer gap "
          f"{pdp['summer_peak_minus_trough']:,.0f}, winter gap {pdp['winter_peak_minus_trough']:,.0f}, "
          f"weekday-weekend {pdp['weekday_minus_weekend']:,.0f}, band width "
          f"{pdp['doy_band_full_width']['min']:,.0f}-{pdp['doy_band_full_width']['max']:,.0f}")
