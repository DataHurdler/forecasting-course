#!/usr/bin/env python3
"""Figures for Lecture 2 (ARIMA, VAR & Multivariate Models) — one script, every figure, both formats.

    .venv/bin/python scripts/figures/lecture02_figures.py                  # all
    .venv/bin/python scripts/figures/lecture02_figures.py acf_levels_vs_diff         # one, by name

Writes Figures/Lecture02_ARIMA_VAR/<name>.pdf (Beamer) and <name>.svg (Quarto), both vector.

Nothing here is trained or sampled. The theory panels are exact (statsmodels ArmaProcess); the
real-data panel reads m5_weekly.csv with the labs' own get_series(). Every number a figure prints
is also written to Figures/Lecture02_ARIMA_VAR/measured.json, so the deck and narration can be
checked against it.

The real series is TX_1 HOUSEHOLD (Lab 1's series), not CA_1 HOUSEHOLD: Lab 2 asks students to
read CA_1's ACF themselves and propose an order, so the deck must not show it.
"""
from __future__ import annotations

import json
import math
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
OUT = ROOT / "Figures" / "Lecture02_ARIMA_VAR"
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


# ================================================================== measurement
PHI, THETA = 0.7, 0.6            # the theory panels: AR(1) phi = 0.7, MA(1) theta = 0.6
LAGS = 10                        # lags shown in the theory panels
REAL_LAGS = 26                   # lags shown for the real series (half a year)
SHOCK_PHIS = (0.5, 0.9, 1.0)     # unit-root frame
SHOCK_T = 20


@lru_cache(maxsize=None)
def series(store: str = "TX_1", cat: str = "HOUSEHOLD") -> pd.Series:
    """The labs' get_series(), verbatim in effect (Labs/Lecture01_lab.qmd, Lecture02_lab.qmd)."""
    weekly = pd.read_csv(ROOT / "data" / "processed" / "m5_weekly.csv", parse_dates=["week_start"])
    return (weekly[(weekly.store_id == store) & (weekly.cat_id == cat)]
            .set_index("week_start")["units"].sort_index().astype(float))


@lru_cache(maxsize=None)
def measured() -> dict:
    from statsmodels.tsa.arima_process import ArmaProcess
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    from statsmodels.tsa.stattools import acf, adfuller

    ar = ArmaProcess(ar=[1, -PHI], ma=[1])
    ma = ArmaProcess(ar=[1], ma=[1, THETA])
    y = series()
    dy = y.diff().dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Lab 1, Step 3: SES fitted on everything but the last 52 weeks.
        alpha = float(SimpleExpSmoothing(y.iloc[:-52]).fit().params["smoothing_level"])
    theta_ses = alpha - 1                       # ARIMA(0,1,1) theta that SES implies
    m = {
        "note": "Written by scripts/figures/lecture02_figures.py; every number the Lecture 2 "
                "figures print. Theory panels are exact (ArmaProcess); the real series is "
                "TX_1 HOUSEHOLD weekly units from m5_weekly.csv.",
        "phi": PHI, "theta": THETA,
        "ar1_acf": ar.acf(LAGS + 1)[1:].tolist(),
        "ar1_pacf": ar.pacf(LAGS + 1)[1:].tolist(),
        "ma1_acf": ma.acf(LAGS + 1)[1:].tolist(),
        "ma1_pacf": ma.pacf(LAGS + 1)[1:].tolist(),
        "ma1_rho1_formula": THETA / (1 + THETA ** 2),
        "real_series": "TX_1 HOUSEHOLD", "real_weeks": int(len(y)),
        "levels_acf": acf(y, nlags=REAL_LAGS)[1:].tolist(),
        "diff_acf": acf(dy, nlags=REAL_LAGS)[1:].tolist(),
        "levels_band": 1.96 / math.sqrt(len(y)),
        "diff_band": 1.96 / math.sqrt(len(dy)),
        "adf_levels": float(adfuller(y, autolag="AIC")[0]),
        "adf_diff": float(adfuller(dy, autolag="AIC")[0]),
        "ses_alpha_lab1": alpha,
        "ses_implied_rho1": theta_ses / (1 + theta_ses ** 2),
        "half_lives": {str(p): (math.log(0.5) / math.log(p) if p < 1 else None) for p in SHOCK_PHIS},
    }
    m["diff_max_after_lag1"] = max(abs(v) for v in m["diff_acf"][1:])
    # The shapes the slides describe must hold before anything is drawn.
    assert abs(m["ma1_acf"][0] - m["ma1_rho1_formula"]) < 1e-12 and abs(m["ma1_acf"][0]) <= 0.5
    assert all(abs(v) < 1e-12 for v in m["ma1_acf"][1:]), "MA(1) ACF must cut off after lag 1"
    assert all(abs(v) < 1e-12 for v in m["ar1_pacf"][1:]), "AR(1) PACF must cut off after lag 1"
    assert np.allclose(m["ar1_acf"], PHI ** np.arange(1, LAGS + 1)), "AR(1) ACF is phi^k"
    OUT.mkdir(parents=True, exist_ok=True)
    MEASURED.write_text(json.dumps(m, indent=1) + "\n")
    return m


# ================================================================== helpers
def stems(ax, lags, vals, color=CRIMSON, lw=2.4, ms=5):
    ax.vlines(lags, 0, vals, color=color, lw=lw, zorder=3)
    ax.plot(lags, vals, "o", color=color, ms=ms, zorder=4)


def corr_axis(ax, xmax, yticks=(-1, -0.5, 0, 0.5, 1), labels=True):
    ax.axhline(0, color=BLACK, lw=0.9, zorder=2)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlim(0.3, xmax + 0.6)
    ax.set_yticks(yticks)
    if labels:
        ax.set_yticklabels([f"{t:g}".replace("-", "−") for t in yticks], fontsize=11)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="x", labelsize=11, length=3)
    ax.tick_params(axis="y", length=3)
    for t in yticks:
        if t not in (0,):
            ax.axhline(t, color=LIGHT_GRAY, lw=0.8, zorder=1)


# ================================================================== the 2x2, exact
def acf_theory() -> None:
    """AR(1) and MA(1), ACF and PACF, from ArmaProcess: exact values, no sampling noise."""
    m = measured()
    lags = np.arange(1, LAGS + 1)
    fig, axes = plt.subplots(2, 2, figsize=sz(12.5, 6.0), sharex=True)
    fig.subplots_adjust(hspace=0.3, wspace=0.12, left=0.13, right=0.99, top=0.92, bottom=0.12)
    panels = [(0, 0, m["ar1_acf"]), (0, 1, m["ar1_pacf"]), (1, 0, m["ma1_acf"]), (1, 1, m["ma1_pacf"])]
    for r, c, v in panels:
        ax = axes[r, c]
        corr_axis(ax, LAGS, labels=(c == 0))
        stems(ax, lags, v)
        ax.set_xticks([1, 2, 4, 6, 8, 10])
    # column and row headings, as in the table on the previous slide
    for c, head in enumerate(("ACF", "PACF")):
        bb = axes[0, c].get_position()
        fig.text((bb.x0 + bb.x1) / 2, 0.94, head, ha="center", va="bottom", fontsize=13,
                 fontweight="bold")
    for r, head in enumerate((f"AR(1)\nφ = {PHI}", f"MA(1)\nθ = {THETA}")):
        bb = axes[r, 0].get_position()
        fig.text(0.005, (bb.y0 + bb.y1) / 2, head, ha="left", va="center", fontsize=12,
                 fontweight="bold")
    for c in (0, 1):
        axes[1, c].set_xlabel("lag", fontsize=11, labelpad=1)

    # what each panel says, in the table's words
    def note(ax, text, color=BLACK, x=0.97, y=0.93):
        ax.text(x, y, text, transform=ax.transAxes, ha="right", va="top", fontsize=11, color=color)

    a = axes[0, 0]                                   # AR ACF: the staircase phi^k
    xx = np.linspace(1, LAGS, 200)
    a.plot(xx, PHI ** xx, color=GRAY, lw=1.0, ls=(0, (2, 2)), zorder=2)
    note(a, f"tails off: {PHI}$^k$,\n×{PHI} every lag", GRAY)
    note(axes[0, 1], f"cuts off after lag 1\n(spike = φ = {PHI})", GREEN)
    r1 = m["ma1_acf"][0]
    a = axes[1, 0]                                   # no MA(1) has |rho_1| > 0.5
    a.plot([0.5, LAGS + 0.4], [0.5, 0.5], color=GREEN, lw=1.0, ls=(0, (3, 2)), zorder=2)
    note(a, f"cuts off after lag 1", GREEN)
    a.text(0.97, 0.08, f"spike {r1:.2f}: no MA(1)\ngoes above the dashed 0.5", transform=a.transAxes,
           ha="right", va="bottom", fontsize=11, color=GREEN)
    note(axes[1, 1], "tails off,\nflipping sign", GRAY, y=0.93)
    save(fig, "acf_theory")


# ================================================================== a real series
def acf_levels_vs_diff() -> None:
    """TX_1 HOUSEHOLD: the ACF of the levels crawls; the ACF of the change has one spike."""
    m = measured()
    lags = np.arange(1, REAL_LAGS + 1)
    fig, axes = plt.subplots(1, 2, figsize=sz(13, 4.4))
    fig.subplots_adjust(wspace=0.1, left=0.09, right=0.99, top=0.78, bottom=0.17)
    for ax, key, band, adf, title in [
            (axes[0], "levels_acf", m["levels_band"], m["adf_levels"], "weekly units (levels)"),
            (axes[1], "diff_acf", m["diff_band"], m["adf_diff"], "week-to-week change")]:
        corr_axis(ax, REAL_LAGS, labels=(ax is axes[0]))
        ax.axhspan(-band, band, color=LIGHT_BLUE, zorder=1, lw=0)
        stems(ax, lags, m[key], lw=1.9, ms=3.6)
        ax.set_xticks([1, 4, 8, 13, 26])
        ax.set_xlabel("lag (weeks)", fontsize=11, labelpad=1)
        verdict = "unit root" if adf > -2.87 else "stationary"   # 5% critical value ≈ −2.87
        ax.set_title(f"{title}\nADF {adf:.2f}: {verdict}".replace("-", "−"), fontsize=12, pad=5)
    lv, dv = m["levels_acf"], m["diff_acf"]
    rest = m["diff_max_after_lag1"]
    axes[0].text(REAL_LAGS + 0.4, -0.22, f"{lv[0]:.2f} at lag 1, still {lv[-1]:.2f} at lag 26:\n"
                 "crawls down, says nothing\nabout p or q", ha="right", va="top", fontsize=11,
                 color=GRAY)
    axes[1].annotate(f"{dv[0]:.2f} at lag 1: the one big spike".replace("-", "−"),
                     xy=(1.3, dv[0]), xytext=(3.6, -0.5), fontsize=11, color=CRIMSON, ha="left",
                     va="center", arrowprops=dict(arrowstyle="-|>", color=CRIMSON, lw=1.2))
    bound = math.ceil(rest * 100) / 100          # round the bound up, so "within" stays true
    axes[1].text(3.6, -0.8, f"every other lag within ±{bound:.2f}", fontsize=11, color=GRAY,
                 ha="left", va="center")
    axes[1].text(REAL_LAGS + 0.4, 0.97, "shaded: ±1.96/√n,\nwhat pure noise shows", ha="right",
                 va="top", fontsize=11, color=BLUE)
    save(fig, "acf_levels_vs_diff")


# ================================================================== one shock, three phis
def shock_response() -> None:
    """y_t = phi y_{t-1} + e_t, one unit shock at t = 0: what is left of it t periods later."""
    m = measured()
    t = np.arange(0, SHOCK_T + 1)
    fig, ax = plt.subplots(figsize=sz(11.5, 4.6))
    fig.subplots_adjust(left=0.1, right=0.83, top=0.95, bottom=0.2)
    colors = {0.5: BLUE, 0.9: AMBER, 1.0: CRIMSON}
    ax.axhline(0.5, color=GRAY, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.text(SHOCK_T + 0.3, 0.5, "half the\nshock", ha="left", va="center", fontsize=11, color=GRAY)
    for p in SHOCK_PHIS:
        resp = p ** t
        ax.plot(t, resp, "-o", color=colors[p], lw=1.8, ms=4, zorder=3)
    hl5, hl9 = m["half_lives"]["0.5"], m["half_lives"]["0.9"]
    for hl, p, dy in [(hl5, 0.5, -0.0), (hl9, 0.9, 0.0)]:
        ax.plot([hl, hl], [0, 0.5], color=colors[p], lw=1.2, ls=(0, (1, 1.5)), zorder=2)
        ax.plot([hl], [0.5], "o", mfc="white", mec=colors[p], mew=1.6, ms=7, zorder=4)
    ax.annotate(f"half-life\n{hl5:.0f} period", xy=(hl5 + 0.15, 0.47), xytext=(2.7, 0.335),
                ha="left", va="center", fontsize=11, color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.9, shrinkA=1, shrinkB=3))
    ax.text(hl9 + 0.35, 0.06, f"half-life\n{hl9:.1f} periods", ha="left", va="bottom", fontsize=11,
            color=AMBER)
    # labels at the right-hand end of each line
    ax.text(SHOCK_T + 0.3, 1.0, "φ = 1: never halves.\nPermanent.", ha="left", va="center",
            fontsize=11, color=CRIMSON, fontweight="bold")
    ax.text(SHOCK_T + 0.3, 0.9 ** SHOCK_T + 0.02, "φ = 0.9", ha="left", va="center", fontsize=11,
            color=AMBER)
    ax.text(SHOCK_T + 0.3, 0.06 - 0.06, "φ = 0.5", ha="left", va="bottom", fontsize=11, color=BLUE)
    ax.set_xlim(-0.6, SHOCK_T + 0.2)
    ax.set_ylim(-0.04, 1.12)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=11)
    ax.set_xlabel("periods after the shock", fontsize=11)
    ax.set_ylabel("share of the shock left", fontsize=11)
    save(fig, "shock_response")


FIGURES = {"acf_theory": acf_theory, "acf_levels_vs_diff": acf_levels_vs_diff, "shock_response": shock_response}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if not a.startswith("--")] or list(FIGURES)
    print(f"Lecture 2 figures -> {OUT.relative_to(ROOT)}   (font: {FONT})")
    for name in pick:
        FIGURES[name]()
    m = measured()
    print(f"  TX_1 HOUSEHOLD: levels ACF {m['levels_acf'][0]:.2f} -> {m['levels_acf'][-1]:.2f}, "
          f"diff lag-1 {m['diff_acf'][0]:.3f}, ADF {m['adf_levels']:.2f} / {m['adf_diff']:.2f}, "
          f"SES alpha {m['ses_alpha_lab1']:.3f} -> rho1 {m['ses_implied_rho1']:.3f}")
