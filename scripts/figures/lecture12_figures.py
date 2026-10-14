#!/usr/bin/env python3
"""Figures for Lecture 12 (Bayesian Statistics III: Bayesian Linear Regression) — one script,
every figure, both formats.

    .venv/bin/python scripts/figures/lecture12_figures.py                 # all
    .venv/bin/python scripts/figures/lecture12_figures.py confounding     # one, by name
    .venv/bin/python scripts/figures/lecture12_figures.py --retrain       # re-sample

Writes Figures/Lecture12_BayesianRegression/<name>.pdf (Beamer) and <name>.svg (Quarto).

The data and the model are the lab's own (Labs/Lecture12_lab.qmd, Setup and fit()), so the numbers
the deck prints are the lab's numbers. Sampling the three models takes about a minute and a half on
a laptop CPU; the posterior summaries go to Figures/Lecture12_BayesianRegression/measured.json and
later runs redraw from that file.

The confounding figure itself needs no sampling: each panel's line is the least-squares slope of the
points drawn in it, which is what the eye fits. By Frisch-Waugh-Lovell, the third panel's slope is
the coefficient on log price in the regression with series intercepts and a trend; the posterior
means in measured.json agree with each drawn slope to two decimals, and the script asserts that.
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
OUT = ROOT / "Figures" / "Lecture12_BayesianRegression"
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


# ================================================================== the lab, verbatim
# Labs/Lecture12_lab.qmd, Setup. Kept identical on purpose: the clip, the SNAP cut at 5 days and
# the trend in years decide every number.
RANDOM_STATE = 42


@lru_cache(maxsize=None)
def data() -> pd.DataFrame:
    weekly = pd.read_csv(ROOT / "data" / "processed" / "m5_weekly.csv", parse_dates=["week_start"])
    weekly["series"] = weekly.store_id + "_" + weekly.cat_id
    df = weekly.dropna(subset=["units", "avg_price"]).copy()
    df = df[df.avg_price > 0]
    df["ly"]    = np.log(df.units.clip(lower=1))
    df["lp"]    = np.log(df.avg_price)
    df["snap"]  = (df.snap_days >= 5).astype(float)
    df["event"] = (df.event_days > 0).astype(float)
    df["yr"]    = (df.week_start - df.week_start.min()).dt.days / 365.25
    codes, _ = pd.factorize(df.series)
    df["idx"] = codes
    df["lpc"] = df.lp - df.groupby("series").lp.transform("mean")       # Step 2
    return df


def fit(price_col, controlled=True, standardize_price=False, seed=RANDOM_STATE):
    """The lab's fit(), verbatim. Even the order of the terms in mu is the lab's: a different
    summation order is a different floating-point graph, and NUTS then takes different steps."""
    import pymc as pm
    df = data()
    y = df.ly.values
    J = df.idx.nunique()
    x = df[price_col].values.astype(float)
    if standardize_price:
        x = (x - x.mean()) / x.std()
    with pm.Model():
        s  = pm.Exponential("sigma", 1.0)
        bp = pm.Normal("b_price", 0, 1)
        bs = pm.Normal("b_snap",  0, 1)
        be = pm.Normal("b_event", 0, 1)
        if controlled:
            a  = pm.Normal("a", y.mean(), 1, shape=J)
            bt = pm.Normal("b_trend", 0, 1)
            mu = (a[df.idx.values] + bp * x + bs * df.snap.values
                  + be * df.event.values + bt * df.yr.values)
        else:
            a  = pm.Normal("a", y.mean(), 1)
            mu = a + bp * x + bs * df.snap.values + be * df.event.values
        pm.Normal("obs", mu=mu, sigma=s, observed=y)
        return pm.sample(1000, tune=1500, chains=4, progressbar=False,
                         random_seed=seed, target_accept=0.9)


def fit_series_only(seed=RANDOM_STATE):
    """Not in the lab: the controlled model without its trend, to measure what removing the series
    means alone leaves behind (the second panel of the confounding figure)."""
    import pymc as pm
    df = data()
    y = df.ly.values
    J = df.idx.nunique()
    x = df["lpc"].values.astype(float)
    with pm.Model():
        s  = pm.Exponential("sigma", 1.0)
        bp = pm.Normal("b_price", 0, 1)
        bs = pm.Normal("b_snap",  0, 1)
        be = pm.Normal("b_event", 0, 1)
        a  = pm.Normal("a", y.mean(), 1, shape=J)
        mu = a[df.idx.values] + bp * x + bs * df.snap.values + be * df.event.values
        pm.Normal("obs", mu=mu, sigma=s, observed=y)
        return pm.sample(1000, tune=1500, chains=4, progressbar=False,
                         random_seed=seed, target_accept=0.9)


# ================================================================== measurement
def _summary(idata, names):
    import arviz as az
    su = az.summary(idata, var_names=names, ci_kind="hdi", ci_prob=0.94, round_to=6)
    return {v: {"mean": float(su.loc[v, "mean"]), "hdi": [float(su.loc[v, "hdi94_lb"]),
                                                        float(su.loc[v, "hdi94_ub"])],
                "r_hat": float(su.loc[v, "r_hat"]), "ess_bulk": float(su.loc[v, "ess_bulk"])}
            for v in names}


def measure() -> dict:
    m: dict = {"note": "Written by scripts/figures/lecture12_figures.py: posterior summaries of the "
                       "Lab 12 models (random_seed=42), the numbers Lecture 12 prints."}
    runs = {"naive": lambda: fit("lp", controlled=False),
            "controlled": lambda: fit("lpc", controlled=True),
            "series_only": fit_series_only}
    for key, run in runs.items():
        idata = run()
        names = ["b_price", "b_snap", "b_event"] + (["b_trend"] if key == "controlled" else [])
        m[key] = _summary(idata, names)
        b = idata.posterior["b_price"].values.ravel()
        m[key]["P(b<0)"] = float((b < 0).mean())
        m[key]["P(b<-0.5)"] = float((b < -0.5).mean())
        m[key]["P(b<-1)"] = float((b < -1).mean())
        if key == "controlled":
            units = 100 * (np.exp(b * np.log(1.05)) - 1)                # the lab's scenario()
            rev = 100 * (1.05 * np.exp(b * np.log(1.05)) - 1)
            m[key]["scenario_5pct"] = {
                "units_mean": float(units.mean()), "units_p3_p97": np.percentile(units, [3, 97]).tolist(),
                "revenue_mean": float(rev.mean()), "revenue_p3_p97": np.percentile(rev, [3, 97]).tolist(),
                "P(revenue rises)": float((rev > 0).mean())}
            be = idata.posterior["b_event"].values.ravel()
            m[key]["b_event_pct_hdi"] = [100 * (np.exp(v) - 1) for v in m[key]["b_event"]["hdi"]]
        print(f"  {key:12s} b_price {m[key]['b_price']['mean']:+.4f} "
              f"HDI [{m[key]['b_price']['hdi'][0]:+.4f}, {m[key]['b_price']['hdi'][1]:+.4f}]")
    OUT.mkdir(parents=True, exist_ok=True)
    MEASURED.write_text(json.dumps(m, indent=1) + "\n")
    print("  measured.json written")
    return m


@lru_cache(maxsize=None)
def measured() -> dict:
    if "--retrain" in sys.argv or not MEASURED.exists():
        return measure()
    return json.loads(MEASURED.read_text())


def residualize(df: pd.DataFrame, cols, on_trend: bool) -> dict:
    """Each column minus its series mean; with on_trend, also minus the common within-series
    trend in years (Frisch-Waugh-Lovell on series intercepts + yr)."""
    out = {c: (df[c] - df.groupby("series")[c].transform("mean")).values for c in cols + ["yr"]}
    if on_trend:
        t = out["yr"]
        for c in cols:
            out[c] = out[c] - (t @ out[c]) / (t @ t) * t
    return out


def ols_slope(x, y) -> tuple[float, float]:
    b = np.polyfit(x, y, 1)
    return float(b[0]), float(b[1])


# ================================================================== confounding
def confounding() -> None:
    """Three views of the same 8,310 weeks: as recorded (between categories), minus each series'
    mean (between years), minus series means and trend (the price response)."""
    df = data()
    m = measured()
    raw_b, raw_a = ols_slope(df.lp.values, df.ly.values)
    dm = residualize(df, ["lp", "ly"], on_trend=False)
    dt = residualize(df, ["lp", "ly"], on_trend=True)
    mid_b, _ = ols_slope(dm["lp"], dm["ly"])
    res_b, _ = ols_slope(dt["lp"], dt["ly"])
    # what the figure prints must agree with the posterior the deck prints (two decimals)
    for drawn, key in [(raw_b, "naive"), (mid_b, "series_only"), (res_b, "controlled")]:
        post = m[key]["b_price"]["mean"]
        assert abs(drawn - post) < 0.01, f"{key}: drawn slope {drawn:.4f} vs posterior {post:.4f}"
    print(f"    slopes drawn: raw {raw_b:+.3f}, minus series means {mid_b:+.3f}, "
          f"minus means and trend {res_b:+.3f}")

    fig = plt.figure(figsize=sz(13.4, 3.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1, 1], wspace=0.38,
                          left=0.06, right=0.99, top=0.80, bottom=0.17)
    ms, al = 3.0, 0.30

    # (1) as recorded: categories are separate clouds; the pooled line joins the clouds
    ax = fig.add_subplot(gs[0])
    cats = [("FOODS", GREEN), ("HOUSEHOLD", BLUE), ("HOBBIES", AMBER)]
    for cat, c in cats:
        g = df[df.cat_id == cat]
        ax.scatter(g.lp, g.ly, s=ms, color=c, alpha=al, lw=0, rasterized=True)
    xs = np.array([df.lp.min(), df.lp.max()])
    ax.plot(xs, raw_a + raw_b * xs, color=CRIMSON, lw=2.4)
    ax.set_xlabel("log price", fontsize=11)
    ax.set_ylabel("log units", fontsize=11)
    ax.tick_params(labelsize=10)
    lab_pos = {"FOODS": (1.115, 8.2, "center"), "HOUSEHOLD": (1.66, 10.05, "center"),
               "HOBBIES": (1.40, 6.75, "center")}
    for cat, c in cats:
        x0, y0, ha = lab_pos[cat]
        ax.text(x0, y0, cat.lower(), ha=ha, va="center", fontsize=11, color=c, fontweight="bold")
    ax.set_title("as recorded", fontsize=12, pad=4)
    ax1 = ax

    # (2) minus each series' mean: within series, later weeks are dearer AND busier
    ax = fig.add_subplot(gs[1])
    order = np.argsort(df.yr.values)
    ax.scatter(dm["lp"][order], dm["ly"][order], s=ms, c=df.yr.values[order], cmap="Greys",
               vmin=-2.0, vmax=df.yr.max(), alpha=0.55, lw=0, rasterized=True)
    xs = np.array([-0.16, 0.10])
    ax.plot(xs, mid_b * xs, color=CRIMSON, lw=2.4)
    ax.set_title("minus each series' mean", fontsize=12, pad=4)
    ax.set_xlabel("log price, within series", fontsize=11)
    ax.set_ylabel("log units, within series", fontsize=11)
    key = ax.inset_axes([0.04, 0.89, 0.40, 0.045])             # the shade is the year
    key.imshow(np.linspace(0, 1, 100)[None, :], aspect="auto", cmap="Greys",
               vmin=-2.0 / df.yr.max(), vmax=1.0, alpha=0.55)
    key.set_xticks([]); key.set_yticks([])
    for sp in key.spines.values():
        sp.set_visible(False)
    key.text(0, -0.6, "2011", transform=key.transAxes, ha="left", va="top", fontsize=10, color=GRAY)
    key.text(1, -0.6, "2016", transform=key.transAxes, ha="right", va="top", fontsize=10,
             color=GRAY)
    ax.tick_params(labelsize=10)
    ax2 = ax

    # (3) minus series means and the trend: what is left is the price response
    ax = fig.add_subplot(gs[2], sharex=ax2, sharey=ax2)
    ax.scatter(dt["lp"], dt["ly"], s=ms, color=GRAY, alpha=0.30, lw=0, rasterized=True)
    ax.plot(xs, res_b * xs, color=CRIMSON, lw=2.4)
    ax.set_title("minus series means and trend", fontsize=12, pad=4)
    ax.set_xlabel("log price, within series", fontsize=11)
    ax.tick_params(labelsize=10, labelleft=False)
    ax3 = ax
    ax2.set_xlim(-0.175, 0.11); ax2.set_ylim(-1.1, 0.8)
    ax2.set_xticks([-0.1, 0, 0.1]); ax2.set_yticks([-1, -0.5, 0, 0.5])

    MINUS = "\u2212"
    for a_, b_, txt in [(ax1, raw_b, f"slope {MINUS}{abs(raw_b):.1f}"),
                        (ax2, mid_b, f"slope +{mid_b:.2f}"),
                        (ax3, res_b, f"slope {MINUS}{abs(res_b):.2f}")]:
        # stacked just above the panel title (pad 4 pt + one 12-pt line), whatever the figure height
        a_.annotate(txt, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 18),
                    textcoords="offset points", ha="center", va="bottom", fontsize=13,
                    color=CRIMSON, fontweight="bold")
    save(fig, "confounding")


FIGURES = {"confounding": confounding}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if not a.startswith("--")] or list(FIGURES)
    print(f"Lecture 12 figures -> {OUT.relative_to(ROOT)}   (font: {FONT})")
    if "--retrain" in sys.argv:
        measured()
    for name in pick:
        FIGURES[name]()
