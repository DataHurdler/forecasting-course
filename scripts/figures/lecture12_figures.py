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

posterior_area and rope_event draw the controlled model's posterior from the densities cached in
measured.json (a KDE of the 4,000 draws) and print the probabilities counted on the draws
themselves. dag is a diagram: no data.
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
# The ROPE on b_event. Neither the deck nor the lab defines one; Homework 7 Q5 uses |beta| < 0.05,
# and the deck adopts that width as a stated choice. 0.01 is recorded for the narration's
# "a narrower ROPE would say" sentence.
ROPE = 0.05
ROPE_WIDTHS = (0.05, 0.01)


def _kde(draws, lo, hi, n=400) -> dict:
    """Gaussian KDE of the draws on [lo, hi] (Scott's rule), rounded for the JSON cache."""
    from scipy.stats import gaussian_kde
    x = np.linspace(lo, hi, n)
    y = gaussian_kde(draws)(x)
    return {"x": [round(float(v), 6) for v in x], "y": [round(float(v), 6) for v in y]}


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
            # what the posterior-area and ROPE figures draw: a density on a grid, and the masses
            m[key]["n_draws"] = int(b.size)
            m[key]["b_price_kde"] = _kde(b, -1.15, 0.10)
            m[key]["b_event_kde"] = _kde(be, -0.065, 0.065)
            for w in ROPE_WIDTHS:
                m[key][f"P(|b_event|<{w})"] = float((np.abs(be) < w).mean())
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


# ================================================================== posterior as an area
MINUS = "−"


def _num(v: float, fmt: str = ".2f") -> str:
    return f"{v:{fmt}}".replace("-", MINUS)


def posterior_area() -> None:
    """The b_price posterior as an area: every question on the 'Asking the Posterior' frame is the
    share of this area on one side of a line. The line at -1 is where a price rise stops raising
    revenue (revenue x 1.05^(1+b)); the whole posterior sits to its right."""
    m = measured()["controlled"]
    x = np.array(m["b_price_kde"]["x"]); y = np.array(m["b_price_kde"]["y"])
    lo, hi = m["b_price"]["hdi"]
    p05, p1, p0 = m["P(b<-0.5)"], m["P(b<-1)"], m["P(b<0)"]
    # the drawn area left of -0.5 must be the number printed on it
    dx = x[1] - x[0]
    assert abs(y[x < -0.5].sum() * dx - p05) < 0.01, "KDE area vs draws"
    assert (round(lo, 3), round(hi, 3), round(p05, 2)) == (-0.596, -0.387, 0.46), (lo, hi, p05)

    fig, ax = plt.subplots(figsize=sz(11.6, 3.3))
    top = y.max()
    ax.fill_between(x, 0, y, color=LIGHT_BLUE, lw=0, zorder=1)
    left = x <= -0.5
    ax.fill_between(x[left], 0, y[left], color=CRIMSON, alpha=0.55, lw=0, zorder=2)
    on = y > 2e-3 * top
    ax.plot(x[on], y[on], color=BLUE, lw=1.8, zorder=3)
    ax.axhline(0, color=GRAY, lw=0.8, zorder=0)

    # revenue break-even: the posterior's mass on each side of it
    ax.plot([-1, -1], [0, top * 1.22], color=BLACK, lw=1.3, ls=(0, (4, 2.5)), zorder=4)
    ax.text(-1.03, top * 1.12, "revenue break-even", ha="right", va="center", fontsize=11,
            color=BLACK)
    ax.text(-1.03, top * 0.55, "elastic:\nrevenue falls", ha="right", va="center", fontsize=11,
            color=GRAY, linespacing=1.1)
    ax.text(-0.97, top * 0.50, "inelastic:\nrevenue rises", ha="left", va="center", fontsize=11,
            color=GRAY, linespacing=1.1)
    ax.text(-1.03, top * 0.12, f"P(b < {MINUS}1) = {p1:.2f}", ha="right", va="center",
            fontsize=12, color=BLACK)
    ax.annotate(f"P(b < {MINUS}0.5) = {p05:.2f}", xy=(-0.56, top * 0.30),
                xytext=(-0.97, top * 1.12), ha="left", va="center", fontsize=12,
                fontweight="bold", color=CRIMSON,
                arrowprops=dict(arrowstyle="-|>", color=CRIMSON, lw=1.2, shrinkA=0, shrinkB=0,
                                mutation_scale=10, relpos=(0.9, 0.0)))
    # zero
    ax.plot([0, 0], [0, top * 0.7], color=GRAY, lw=1.0, ls=":", zorder=0)
    ax.text(-0.025, top * 0.66, f"P(b < 0) = {p0:.2f}", ha="right", va="center", fontsize=12,
            color=BLACK)

    # the HDI, as a bar under the curve
    yb = -top * 0.13
    ax.plot([lo, hi], [yb, yb], color=BLACK, lw=3.4, solid_capstyle="butt", zorder=5)
    ax.text(lo - 0.02, yb, f"94% HDI [{_num(lo, '.3f')}, {_num(hi, '.3f')}]", ha="right",
            va="center", fontsize=11, color=BLACK)

    ax.set_xlim(-1.45, 0.10); ax.set_ylim(yb * 1.9, top * 1.42)
    ax.set_xticks([-1, -0.5, 0]); ax.set_xticklabels([f"{MINUS}1", f"{MINUS}0.5", "0"])
    ax.tick_params(labelsize=11)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("price elasticity, b_price", fontsize=11)
    save(fig, "posterior_area")


# ================================================================== ROPE on b_event
def rope_event() -> None:
    """b_event's posterior against a region of practical equivalence. Drawn in percent (the
    exp(b) - 1 transform the deck uses for -1.1% and +0.4%); the ROPE edges are |b| = 0.05 and
    |b| = 0.01 transformed exactly, so the bands are very slightly asymmetric, as they should be."""
    m = measured()["controlled"]
    b = np.array(m["b_event_kde"]["x"]); yb_ = np.array(m["b_event_kde"]["y"])
    pct = 100 * (np.exp(b) - 1)
    dens = yb_ / (100 * np.exp(b))                       # density per percentage point
    lo, hi = m["b_event_pct_hdi"]
    p5, p1 = m["P(|b_event|<0.05)"], m["P(|b_event|<0.01)"]
    assert (round(lo, 1), round(hi, 1)) == (-1.1, 0.4), (lo, hi)
    e5 = [100 * (np.exp(-ROPE) - 1), 100 * (np.exp(ROPE) - 1)]
    e1 = [100 * (np.exp(-0.01) - 1), 100 * (np.exp(0.01) - 1)]

    fig, ax = plt.subplots(figsize=sz(11.6, 3.3))
    top = dens.max()
    band = top * 1.14
    ax.fill_between(e5, 0, band, color=LIGHT_GRAY, lw=0, zorder=0)
    ax.fill_between(e1, 0, band, color="#E4E4E7", lw=0, zorder=0)
    for e in e5:
        ax.plot([e, e], [0, band], color=GRAY, lw=1.0, ls=(0, (4, 2.5)), zorder=1)
    ax.fill_between(pct, 0, dens, color=LIGHT_BLUE, lw=0, zorder=2)
    on = dens > 2e-3 * top
    ax.plot(pct[on], dens[on], color=BLUE, lw=1.8, zorder=3)
    ax.axhline(0, color=GRAY, lw=0.8, zorder=1)

    ax.text(0, top * 1.22, f"ROPE ≈ ±5%: {100 * p5:.0f}% of the posterior inside",
            ha="center", va="bottom", fontsize=12, color=BLACK, fontweight="bold")
    ax.annotate(f"≈ ±1%:\n{100 * p1:.0f}% inside", xy=(e1[1], top * 0.62),
                xytext=(1.9, top * 0.62), ha="left", va="center", fontsize=11, color=GRAY,
                linespacing=1.05,
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.0, shrinkA=2, shrinkB=0,
                                mutation_scale=9))
    yb = -top * 0.13
    ax.plot([lo, hi], [yb, yb], color=BLACK, lw=3.4, solid_capstyle="butt", zorder=5)
    ax.text(hi + 0.25, yb, f"94% HDI: {_num(lo, '.1f')}% to +{hi:.1f}%", ha="left",
            va="center", fontsize=11, color=BLACK)

    ax.set_xlim(-6.6, 6.6); ax.set_ylim(yb * 1.9, top * 1.42)
    ax.set_xticks([-5, 0, 5]); ax.set_xticklabels([f"{MINUS}5%", "0", "+5%"])
    ax.tick_params(labelsize=11)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("event-week effect on units, b_event", fontsize=11)
    save(fig, "rope_event")


# ================================================================== the DAG
def dag() -> None:
    """The DAG the 'Directed Acyclic Graphs' and 'Confounding, Measured' frames describe: series
    identity (store x category) and time both push price and units, so both are confounders and
    both are in the model (series intercepts, trend); SNAP and events push units only. Every arrow
    into units is a term in mu. A belief, drawn before the fit -- not something the data proved."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=sz(10.6, 4.7))
    ax.set_xlim(-0.1, 12.3); ax.set_ylim(-2.3, 2.3); ax.axis("off")
    nodes = {  # name: (x, y, w, h, label, edge colour, fill)
        "price":  (1.0, 0.0, 1.8, 0.8, "price", BLUE, LIGHT_BLUE),
        "units":  (7.2, 0.0, 2.4, 0.8, "units sold", BLUE, LIGHT_BLUE),
        "series": (4.05, 1.6, 3.4, 1.05, "series\n(store × category)", CRIMSON, LIGHT_CRIMSON),
        "time":   (4.05, -1.6, 1.8, 1.05, "time\n(trend)", CRIMSON, LIGHT_CRIMSON),
        "other":  (11.2, 0.0, 1.9, 1.05, "SNAP,\nevents", GRAY, LIGHT_GRAY),
    }
    patch = {}
    for k, (x, y, w, h, lab, ec, fc) in nodes.items():
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.18",
                                    ec=ec, fc=fc, lw=1.6, zorder=3))
        ax.text(x, y, lab, ha="center", va="center", fontsize=11, color=BLACK, zorder=4,
                linespacing=1.05, fontweight="bold" if k in ("price", "units") else "normal")
        # an invisible, slightly larger twin, so arrows start and stop just off the edge
        patch[k] = FancyBboxPatch((x - w / 2, y - h / 2), w, h, visible=False,
                                  boxstyle="round,pad=0.10,rounding_size=0.18")
        ax.add_patch(patch[k])

    def edge(a, b, color, lw=1.6):
        ax.annotate("", xy=nodes[b][:2], xytext=nodes[a][:2], zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=13,
                                    shrinkA=0, shrinkB=0, patchA=patch[a], patchB=patch[b]))

    for c in ("series", "time"):
        edge(c, "price", CRIMSON); edge(c, "units", CRIMSON)
    edge("other", "units", GRAY)
    edge("price", "units", BLUE, lw=2.6)
    ax.text(4.05, 0.12, "b_price", ha="center", va="bottom", fontsize=12, color=BLUE,
            fontweight="bold")
    ax.text(4.05, -0.14, "the effect we want", ha="center", va="top", fontsize=11, color=BLUE)
    ax.text(5.95, 1.6, "confounder", ha="left", va="center", fontsize=11, color=CRIMSON)
    ax.text(5.15, -1.6, "confounder", ha="left", va="center", fontsize=11, color=CRIMSON)
    ax.text(11.2, -0.68, "units only:\nnot confounders", ha="center", va="top", fontsize=11,
            color=GRAY, linespacing=1.05)
    save(fig, "dag")

FIGURES = {"confounding": confounding, "posterior_area": posterior_area, "dag": dag,
           "rope_event": rope_event}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if not a.startswith("--")] or list(FIGURES)
    print(f"Lecture 12 figures -> {OUT.relative_to(ROOT)}   (font: {FONT})")
    if "--retrain" in sys.argv:
        measured()
    for name in pick:
        FIGURES[name]()
