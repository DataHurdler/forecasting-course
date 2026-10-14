#!/usr/bin/env python3
"""Figures for Lecture 1 (Exponential Smoothing and Forecast Evaluation) — one script, both formats.

    .venv/bin/python scripts/figures/lecture01_figures.py                 # all
    .venv/bin/python scripts/figures/lecture01_figures.py ses_levels      # one, by name
    .venv/bin/python scripts/figures/lecture01_figures.py --retrain       # re-measure

Writes Figures/Lecture01_ETS_Eval/<name>.pdf (Beamer) and <name>.svg (Quarto), both vector.

Every number a figure prints comes from the fits below, which use Lab 1's own calls
(Labs/Lecture01_lab.qmd: statsmodels' SimpleExpSmoothing / Holt with default settings, the last
52 weeks held out, seasonal naive = the same week a year earlier). Nothing here is trained or
sampled, so there is no seed; the fits are deterministic and take about two seconds. The results go
to Figures/Lecture01_ETS_Eval/measured.json and later runs redraw from that file. Before anything is written,
measure() checks that it reproduces the numbers already printed in the lab and the deck.
"""
from __future__ import annotations

import json
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Figures" / "Lecture01_ETS_Eval"
MEASURED = OUT / "measured.json"

# The deck's palette — Preambles/header.tex and Quarto/uno-clean.scss.
CRIMSON, BLACK, GRAY = "#D71920", "#1A1A1A", "#525252"
GREEN, AMBER, BLUE = "#15803D", "#B45309", "#005CA9"
LIGHT_CRIMSON, LIGHT_GRAY, LIGHT_BLUE = "#FDECEA", "#F4F4F5", "#E8F0FA"
MID_GRAY = "#A3A3A3"

_fonts = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ("Source Sans Pro", "Source Sans 3", "Helvetica Neue", "Arial")
             if f in _fonts), "DejaVu Sans")
# Drawn near display size, as in Lecture 8: see the SCALE note in lecture08_figures.py.
SCALE = 0.55


def sz(w: float, h: float) -> tuple[float, float]:
    return (w * SCALE, h * SCALE)


# SVG text is drawn as outlines; see lecture09_part1_figures.py for why live text breaks.
plt.rcParams.update({"font.family": FONT, "font.size": 13, "svg.fonttype": "path",
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                     "mathtext.fontset": "custom", "mathtext.rm": FONT,
                     "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold"})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.05,
                    transparent=True, dpi=200)
    plt.close(fig)
    print(f"  {name}.pdf / .svg")


def comma(x: float) -> str:
    return f"{x:,.0f}"


# ================================================================== Lab 1's pipeline
N_TEST = 52


def series(store: str, cat: str) -> pd.Series:
    """Lab 1's get_series()."""
    weekly = pd.read_csv(ROOT / "data" / "processed" / "m5_weekly.csv", parse_dates=["week_start"])
    return (weekly[(weekly.store_id == store) & (weekly.cat_id == cat)]
            .set_index("week_start")["units"].sort_index().astype(float))


def rmse(a, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(p)) ** 2)))


def mae(a, p) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(p))))


def measure() -> dict:
    from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing
    warnings.filterwarnings("ignore")
    m: dict = {"note": "Written by scripts/figures/lecture01_figures.py: every number the Lecture 1 "
                       "figures print. Lab 1's calls (statsmodels defaults), last 52 weeks held out."}

    # ---- CA_1 FOODS: the deck's series (Homework 1's, and Lecture 11's seasonal-naive 1,660)
    y = series("CA_1", "FOODS")
    tr, te = y.iloc[:-N_TEST], y.iloc[-N_TEST:]
    ses = SimpleExpSmoothing(tr).fit()
    alpha = float(ses.params["smoothing_level"])
    sn = y.shift(52).iloc[-N_TEST:].values
    f_ses = ses.forecast(N_TEST).values
    e_ses, e_sn = te.values - f_ses, te.values - sn
    show = slice(len(tr) - 104, len(tr))                    # two training years on screen
    levels = {}
    for a in (0.8, 0.1):
        fit = SimpleExpSmoothing(tr).fit(smoothing_level=a, optimized=False)
        levels[str(a)] = fit.level.values[show].tolist()
        m[f"ses_{a}_test_rmse"] = rmse(te, fit.forecast(N_TEST))
    levels["fitted"] = ses.level.values[show].tolist()

    # the week the inset zooms into: the largest one-week gap y_t - l_{t-1} in the shown window
    lv = ses.level.values
    gaps = tr.values[1:] - lv[:-1]                          # gaps[i] belongs to week i + 1
    lo = len(tr) - 104
    t = lo + 1 + int(np.argmax(gaps[lo:]))
    l_prev, y_t, l_t = float(lv[t - 1]), float(tr.values[t]), float(lv[t])
    assert abs(l_t - (l_prev + alpha * (y_t - l_prev))) < 1e-6, "SES update does not reproduce"

    holt = Holt(tr).fit()
    m.update({
        "ca1_dates": [d.strftime("%Y-%m-%d") for d in y.index[len(tr) - 104:]],
        "ca1_units": y.values[len(tr) - 104:].tolist(),
        "n_train": len(tr), "n_test": len(te),
        "test_start": te.index[0].strftime("%Y-%m-%d"), "test_end": te.index[-1].strftime("%Y-%m-%d"),
        "alpha": alpha, "levels": levels, "level_T": {"0.8": levels["0.8"][-1],
                                                     "0.1": levels["0.1"][-1],
                                                     "fitted": levels["fitted"][-1]},
        "zoom": {"date": tr.index[t].strftime("%Y-%m-%d"), "level_prev": l_prev, "y": y_t,
                 "level": l_t, "gap": y_t - l_prev},
        "ses_test_rmse": rmse(te, f_ses), "ses_test_mae": mae(te, f_ses),
        "sn_test_rmse": rmse(te, sn), "sn_test_mae": mae(te, sn),
        "err_ses": e_ses.tolist(), "err_sn": e_sn.tolist(),
        "test_dates": [d.strftime("%Y-%m-%d") for d in te.index],
        "holt_level_T": float(holt.level.iloc[-1]), "holt_trend_T": float(holt.trend.iloc[-1]),
        "holt_test_rmse": rmse(te, holt.forecast(N_TEST)),
    })

    # ---- the weights figure: the two aisles of Lab 1's store
    for key, (store, cat) in {"tx1_household": ("TX_1", "HOUSEHOLD"),
                              "tx1_foods": ("TX_1", "FOODS")}.items():
        s = series(store, cat)
        fit = SimpleExpSmoothing(s.iloc[:-N_TEST]).fit()
        m[f"alpha_{key}"] = float(fit.params["smoothing_level"])
        m[f"ses_rmse_{key}"] = rmse(s.iloc[-N_TEST:], fit.forecast(N_TEST))

    # ---- the damping bound: statsmodels searches 0.8 <= phi <= 0.995 (holtwinters/model.py)
    phis = {}
    for store, cat in (("CA_1", "FOODS"), ("TX_1", "HOUSEHOLD"), ("TX_1", "FOODS")):
        s = series(store, cat)
        phis[f"{store} {cat}"] = float(Holt(s.iloc[:-N_TEST], damped_trend=True).fit()
                                       .params["damping_trend"])
    m["damped_phi_default_fit"] = phis

    # ---- reproduce what the lab and the deck already print, before anything is drawn
    checks = [(round(m["ses_test_rmse"]), 1754), (round(m["sn_test_rmse"]), 1660),
              (round(m["ses_test_mae"]), 1376), (round(m["sn_test_mae"]), 1365),
              (round(m["holt_test_rmse"]), 3428), (round(alpha, 2), 0.32),
              (round(m["alpha_tx1_household"], 3), 0.621), (round(m["ses_rmse_tx1_household"]), 427),
              (round(m["alpha_tx1_foods"], 2), 0.21), (round(m["ses_rmse_tx1_foods"]), 1474),
              (len(tr), 225)]
    for got, want in checks:
        assert got == want, f"does not reproduce: got {got}, expected {want}"

    OUT.mkdir(parents=True, exist_ok=True)
    MEASURED.write_text(json.dumps(m) + "\n")
    print(f"  measured.json written: SES alpha {alpha:.3f}, RMSE {m['ses_test_rmse']:,.0f} vs "
          f"seasonal naive {m['sn_test_rmse']:,.0f}; damped phi {phis}")
    return m


@lru_cache(maxsize=None)
def measured() -> dict:
    if "--retrain" in sys.argv or not MEASURED.exists():
        return measure()
    return json.loads(MEASURED.read_text())


# ================================================================== SES: closing a fraction of the gap
def ses_levels() -> None:
    """CA_1 FOODS, two training years and the test year. The SES level at three values of alpha,
    each running flat through the test year; on the right, one week zoomed in."""
    m = measured()
    d = pd.to_datetime(m["ca1_dates"]); u = np.array(m["ca1_units"])
    ntr = 104
    d_tr, d_te = d[:ntr], d[ntr:]
    fig = plt.figure(figsize=sz(14, 3.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.1, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0])
    ax.axvspan(d_te[0] - pd.Timedelta(days=3.5), d_te[-1] + pd.Timedelta(days=3.5),
               color=LIGHT_GRAY, zorder=0, lw=0)
    ax.plot(d, u, color=MID_GRAY, lw=1.0, zorder=1)
    style = {"0.8": (AMBER, 1.3, "α = 0.8"), "0.1": (BLUE, 1.8, "α = 0.1"),
             "fitted": (CRIMSON, 2.4, f"α = {m['alpha']:.2f}, fitted")}
    for key in ("0.8", "0.1", "fitted"):
        c, lw, lab = style[key]
        lev = np.array(m["levels"][key])
        ax.plot(d_tr, lev, color=c, lw=lw, zorder=3)
        ax.plot([d_tr[-1], d_te[-1]], [lev[-1], lev[-1]], color=c, lw=lw, zorder=3,
                solid_capstyle="butt")
    ax.set_ylim(13500, 27500)
    ax.set_yticks([15000, 20000, 25000]); ax.set_yticklabels(["15k", "20k", "25k"], fontsize=11)
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=11)
    ax.set_xlim(d[0] - pd.Timedelta(days=7), d[-1] + pd.Timedelta(days=7))
    ax.set_ylabel("units / week", fontsize=11)
    ax.text(d_te[len(d_te) // 2], 27300, "test year:\neach level held flat", ha="center",
            va="top", fontsize=11, color=GRAY)
    # legend: one row of coloured labels above the plot, clear of the data
    x0 = 0.0
    for key in ("0.8", "fitted", "0.1"):
        c, _, lab = style[key]
        t = ax.text(x0, 1.03, "— " + lab, transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=11, color=c, fontweight="bold" if key == "fitted" else "normal")
        x0 += {"0.8": 0.2, "fitted": 0.36, "0.1": 0.2}[key]
    # mark the zoomed week
    zd = pd.Timestamp(m["zoom"]["date"])
    ax.plot([zd], [m["zoom"]["y"]], "o", ms=7, mfc="none", mec=BLACK, mew=1.3, zorder=5)

    # ---- the inset: one week, the gap and the fraction alpha of it
    z = m["zoom"]; a = m["alpha"]
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(-0.6, 2.6)
    span = z["gap"]
    ax2.set_ylim(z["level_prev"] - 0.42 * span, z["y"] + 0.28 * span)
    ax2.axis("off")
    L0, L1, Y = z["level_prev"], z["level"], z["y"]
    ax2.plot([-0.5, 0.35], [L0] * 2, color=CRIMSON, lw=2.4, solid_capstyle="butt")
    ax2.plot([0.35, 1.25], [L0] * 2, color=CRIMSON, lw=1.0, ls=(0, (3, 2)))
    ax2.plot([0.75, 1.25], [L1] * 2, color=CRIMSON, lw=2.4, solid_capstyle="butt")
    ax2.plot([1.0], [Y], "o", ms=7, mfc="none", mec=BLACK, mew=1.3)
    ax2.annotate("", xy=(1.0, L1), xytext=(1.0, L0),
                 arrowprops=dict(arrowstyle="-|>", color=CRIMSON, lw=1.6, mutation_scale=11,
                                 shrinkA=0, shrinkB=0))
    ax2.annotate("", xy=(1.95, Y), xytext=(1.95, L0),
                 arrowprops=dict(arrowstyle="<|-|>", color=GRAY, lw=1.2, mutation_scale=10,
                                 shrinkA=0, shrinkB=0))
    ax2.plot([1.15, 2.05], [Y, Y], color=GRAY, lw=0.6, ls=(0, (1, 1.5)))
    ax2.plot([1.3, 2.05], [L0, L0], color=GRAY, lw=0.6, ls=(0, (1, 1.5)))
    ax2.text(2.08, (Y + L0) / 2, f"gap\n{comma(z['gap'])}", ha="left", va="center", fontsize=11,
             color=GRAY)
    ax2.text(0.9, L1 + 0.06 * span, f"{a:.2f} × gap\n= {comma(L1 - L0)}", ha="right", va="bottom",
             fontsize=11, color=CRIMSON)
    ax2.text(1.0, Y + 0.07 * span, "$y_t$", ha="center", va="bottom", fontsize=12)
    ax2.text(-0.5, L0 - 0.06 * span, "$\\ell_{t-1}$", ha="left", va="top", fontsize=12,
             color=CRIMSON)
    ax2.text(1.32, L1, "$\\ell_t$", ha="left", va="center", fontsize=12, color=CRIMSON)
    ax2.text(0.95, z["level_prev"] - 0.30 * span, f"one week, zoomed:\n{pd.Timestamp(z['date']):%-d %b %Y}",
             ha="center", va="top", fontsize=11, color=GRAY)
    save(fig, "ses_levels")


# ================================================================== the weights, as a staircase
def ses_weights() -> None:
    """Weight alpha(1-alpha)^k on an observation k weeks old, for the two fitted alphas of Lab 1's
    store, against equal weights over the 225 training weeks. Log scale: each step back multiplies
    the weight by (1 - alpha), so the staircase is a straight line. The triangle is each profile's
    centre of mass, (1 - alpha)/alpha weeks back (exact for an infinite history; with 225 weeks the
    weight left on the starting level is below 1e-23)."""
    m = measured()
    T = m["n_train"]
    k = np.arange(0, 25)
    fig, axes = plt.subplots(1, 2, figsize=sz(13, 3.6), sharey=True,
                             gridspec_kw={"wspace": 0.1})
    for ax, key, title, c in ((axes[0], "tx1_household", "TX_1 HOUSEHOLD", CRIMSON),
                              (axes[1], "tx1_foods", "TX_1 FOODS", BLUE)):
        a = m[f"alpha_{key}"]
        w = a * (1 - a) ** k
        ax.bar(k, w, width=0.72, color=c, zorder=2, bottom=1e-4)
        ax.set_yscale("log")
        ax.set_ylim(4e-4, 1.6)
        ax.axhline(1 / T, color=BLACK, lw=1.3, ls=(0, (4, 2)), zorder=3)
        com = (1 - a) / a
        ax.plot([com], [4e-4], marker="^", ms=11, color=BLACK, clip_on=False, zorder=5)
        ax.plot([com, com], [4e-4, 1.25], color=BLACK, lw=1.0, ls=(0, (1, 1.5)), zorder=4)
        ax.text(com + 0.45, 1.25, f"average age: {com:.1f} weeks", ha="left", va="center",
                fontsize=11)
        ax.set_title(f"{title}:  α = {a:.2f}", fontsize=12, color=c, pad=8)
        ax.set_xlim(-0.8, 24.8)
        ax.set_xticks([0, 5, 10, 15, 20]); ax.tick_params(labelsize=11)
        ax.set_xlabel("weeks old", fontsize=11)
    axes[0].set_yticks([1e-3, 1e-2, 1e-1, 1])
    axes[0].set_yticklabels(["0.1%", "1%", "10%", "100%"], fontsize=11)
    axes[0].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[0].set_ylabel("weight", fontsize=11)
    axes[0].text(24.5, 1 / T * 1.35, f"equal weights: 1/{T} each", ha="right", va="bottom",
                 fontsize=11)
    save(fig, "ses_weights")


# ================================================================== damped trend: a ceiling
def damped_trend() -> None:
    """Holt's fitted level and slope on CA_1 FOODS, extrapolated undamped, against the same
    level and slope damped at phi = 0.85 (the slide's illustrative value; statsmodels' own fit
    sits on its lower bound of 0.8). The damped path rises by b*phi^h in week h, so it can never
    pass l_T + b*phi/(1 - phi)."""
    m = measured()
    lT, b, phi = m["holt_level_T"], m["holt_trend_T"], 0.85
    H = 16
    h = np.arange(0, H + 1)
    und = lT + b * h
    dmp = lT + b * np.concatenate([[0], np.cumsum(phi ** np.arange(1, H + 1))])
    ceil = lT + b * phi / (1 - phi)
    assert dmp.max() < ceil
    fig, ax = plt.subplots(figsize=sz(8.4, 4.1))
    ax.axhline(lT, color=MID_GRAY, lw=1.0, zorder=1)
    ax.plot([-0.5, H + 0.4], [ceil, ceil], color=BLACK, lw=1.2, ls=(0, (4, 2)), zorder=1,
            clip_on=False)
    ax.plot(h, und, "-o", color=CRIMSON, lw=1.8, ms=4, zorder=3)
    ax.plot(h, dmp, "-o", color=BLUE, lw=1.8, ms=4, zorder=3)
    ax.set_xlim(-0.5, H + 0.4)
    ax.set_ylim(lT - 230, und[-1] + 90)
    ax.set_xticks([0, 4, 8, 12, 16]); ax.tick_params(labelsize=11)
    ax.set_xlabel("weeks ahead, h", fontsize=11)
    ax.set_ylabel("forecast, units / week", fontsize=11)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_yticks([21000, 21500, 22000])
    ax.text(0.2, und[-1] + 60, f"undamped: +{b:.0f} every week,\nforever (+{b * 52:,.0f} by week 52)",
            ha="left", va="top", fontsize=11, color=CRIMSON)
    ax.text(H, lT + 50, f"damped, φ = {phi} (illustrative)", ha="right", va="bottom",
            fontsize=11, color=BLUE)
    ax.text(H, ceil + 28, f"ceiling = $\\ell_T$ + {phi / (1 - phi):.1f} $b_T$", ha="right",
            va="bottom", fontsize=11)
    ax.text(0.2, lT - 35, f"$\\ell_T$ = {lT:,.0f}: today's level", ha="left", va="top",
            fontsize=11, color=GRAY)
    save(fig, "damped_trend")


# ================================================================== errors as squares
def error_squares() -> None:
    """Each of the 52 test-week errors as a square of side |e|, all sharing a corner, for SES and
    seasonal naive on the same scale. MAE is the average side (dashed); RMSE is the side of the
    average square (filled). The average of the squares can never be smaller than the square of
    the average, so RMSE >= MAE, and the gap grows with the few largest misses."""
    m = measured()
    panels = (("SES", np.abs(m["err_ses"]), CRIMSON, LIGHT_CRIMSON),
              ("Seasonal naïve", np.abs(m["err_sn"]), BLUE, LIGHT_BLUE))
    lim = max(np.abs(m["err_ses"]).max(), np.abs(m["err_sn"]).max()) * 1.04
    fig, axes = plt.subplots(1, 2, figsize=sz(10.5, 5.4), gridspec_kw={"wspace": 0.12})
    for ax, (name, a, c, lc) in zip(axes, panels):
        mae_, rmse_ = a.mean(), np.sqrt((a ** 2).mean())
        ax.add_patch(Rectangle((0, 0), rmse_, rmse_, facecolor=lc, edgecolor=c, lw=2.0, zorder=1))
        for s in np.sort(a):
            ax.add_patch(Rectangle((0, 0), s, s, fill=False, edgecolor=GRAY, lw=0.6, alpha=0.75,
                                   zorder=2))
        ax.add_patch(Rectangle((0, 0), mae_, mae_, fill=False, edgecolor=BLACK, lw=1.8,
                               ls=(0, (3, 2)), zorder=3))
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xticks([0, 2000, 4000]); ax.set_yticks([0, 2000, 4000])
        ax.set_xticklabels(["0", "2,000", "4,000"], fontsize=10)
        ax.set_yticklabels(["0", "2,000", "4,000"] if ax is axes[0] else [], fontsize=10)
        ax.set_title(name, fontsize=12, color=c, pad=3)
        top5 = (np.sort(a)[::-1][:5] ** 2).sum() / (a ** 2).sum()
        ax.text(0.5, -0.17, f"MAE {mae_:,.0f} · RMSE {rmse_:,.0f}\n5 largest: {top5:.0%} of the area",
                transform=ax.transAxes, ha="center", va="top", fontsize=11, linespacing=1.3)
    axes[0].set_ylabel("units", fontsize=11)
    save(fig, "error_squares")


FIGURES = {"ses_levels": ses_levels, "ses_weights": ses_weights, "damped_trend": damped_trend,
           "error_squares": error_squares}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if not a.startswith("--")] or list(FIGURES)
    print(f"Lecture 1 figures -> {OUT.relative_to(ROOT)}   (font: {FONT})")
    if "--retrain" in sys.argv:
        measured()
    for name in pick:
        FIGURES[name]()
