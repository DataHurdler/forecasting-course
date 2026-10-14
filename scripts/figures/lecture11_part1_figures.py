#!/usr/bin/env python3
"""Figures for Lecture 11 Part 1 (Bayesian structural time series) — one script, both formats.

    .venv/bin/python scripts/figures/lecture11_part1_figures.py                  # all
    .venv/bin/python scripts/figures/lecture11_part1_figures.py interval_band    # one, by name
    .venv/bin/python scripts/figures/lecture11_part1_figures.py --retrain        # re-measure

Writes Figures/Lecture11_Part1/<name>.pdf (Beamer) and <name>.svg (Quarto), both vector.

Both figures print numbers from the lab's own fit. Sampling uses the lab's data pipeline, models,
seeds and forecasting code verbatim (Labs/Lecture11_Part1_lab.qmd, Setup and Steps 1-4), so the
figures print the lab's numbers rather than a second set. The two fits (centred, then
non-centred) take about a minute on a laptop CPU. Everything the figures and the deck print goes
to Figures/Lecture11_Part1/measured.json; later runs redraw from that file, and --retrain samples
again.
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
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Figures" / "Lecture11_Part1"
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
# Labs/Lecture11_Part1_lab.qmd, Setup + Steps 1-4. Kept identical on purpose: the seeds, the
# sampler settings and the ORDER of the rng draws in Step 3 decide every number.
N_TEST, K, P = 52, 3, 52
SHOW_TRAIN = 60          # training weeks drawn before the forecast origin


def fourier(t):
    return np.column_stack([f(2*np.pi*k*t/P) for k in range(1, K+1) for f in (np.sin, np.cos)])


def measure() -> dict:
    import logging
    import warnings
    import arviz as az
    import pymc as pm
    warnings.filterwarnings("ignore")
    logging.getLogger("pymc").setLevel(logging.ERROR)

    weekly = pd.read_csv(ROOT / "data" / "processed" / "m5_weekly.csv", parse_dates=["week_start"])
    ca1 = (weekly[(weekly.store_id == "CA_1") & (weekly.cat_id == "FOODS")]
             .sort_values("week_start").reset_index(drop=True))
    train, test = ca1.iloc[:-N_TEST], ca1.iloc[-N_TEST:]
    y = train.units.values.astype(float)
    n = len(train)
    y_mu, y_sd = y.mean(), y.std()
    y_z = (y - y_mu) / y_sd
    t_all = np.arange(len(ca1))
    X_tr, X_te = fourier(t_all[:n]), fourier(t_all[n:])

    def diagnostics(idata):
        s = az.summary(idata, var_names=["sigma_level", "sigma", "beta"], ci_kind="hdi",
                       ci_prob=0.94)
        return {"r_hat_max": float(s.r_hat.max()), "ess_bulk_min": float(s.ess_bulk.min()),
                "divergences": int(idata.sample_stats.diverging.sum())}

    # Step 1: the centred walk (the fit that fails its diagnostics)
    with pm.Model():
        sigma_level = pm.Exponential("sigma_level", 10.0)
        level = pm.GaussianRandomWalk("level", sigma=sigma_level, shape=n)
        beta = pm.Normal("beta", 0, 0.5, shape=2*K)
        sigma = pm.Exponential("sigma", 1.0)
        pm.Normal("obs", mu=level + pm.math.dot(X_tr, beta), sigma=sigma, observed=y_z)
        idata_c = pm.sample(1500, tune=2000, chains=4, progressbar=False,
                            random_seed=42, target_accept=0.95)
    diag_c = diagnostics(idata_c)
    print(f"  centred:     R-hat {diag_c['r_hat_max']:.3f}  ESS {diag_c['ess_bulk_min']:.0f}  "
          f"div {diag_c['divergences']}")

    # Step 2: the non-centred walk (the fit every number below comes from)
    with pm.Model():
        sigma_level = pm.Exponential("sigma_level", 10.0)
        z = pm.Normal("z", 0, 1, shape=n)
        level = pm.Deterministic("level", pm.math.cumsum(z * sigma_level))
        beta = pm.Normal("beta", 0, 0.5, shape=2*K)
        sigma = pm.Exponential("sigma", 1.0)
        pm.Normal("obs", mu=level + pm.math.dot(X_tr, beta), sigma=sigma, observed=y_z)
        idata = pm.sample(1500, tune=2000, chains=4, progressbar=False,
                          random_seed=42, target_accept=0.95)
    diag = diagnostics(idata)
    print(f"  non-centred: R-hat {diag['r_hat_max']:.3f}  ESS {diag['ess_bulk_min']:.0f}  "
          f"div {diag['divergences']}")

    # Step 3, verbatim: the same rng, drawn in the same order
    rng = np.random.default_rng(42)
    post = idata.posterior
    lev_last = post["level"].values[:, :, -1].ravel()
    sl = post["sigma_level"].values.ravel()
    b = post["beta"].values.reshape(-1, 2*K)
    sg = post["sigma"].values.ravel()
    D = len(lev_last)
    actual = test.units.values
    point = (lev_last[:, None] + b @ X_te.T).mean(0) * y_sd + y_mu
    rmse_point = float(np.sqrt(np.mean((actual - point) ** 2)))
    mu_a = lev_last[:, None] + b @ X_te.T
    draws_a = mu_a + rng.normal(0, 1, size=mu_a.shape) * sg[:, None]
    steps = rng.normal(0, 1, size=(D, N_TEST)) * sl[:, None]
    mu_b = lev_last[:, None] + np.cumsum(steps, axis=1) + b @ X_te.T
    draws_b = mu_b + rng.normal(0, 1, size=mu_b.shape) * sg[:, None]

    def band(draws):
        lo, hi = np.percentile(draws, [3, 97], axis=0) * y_sd + y_mu
        return lo, hi

    lo_b, hi_b = band(draws_b)
    lo_a, hi_a = band(draws_a)
    inside_b = (actual >= lo_b) & (actual <= hi_b)
    inside_a = (actual >= lo_a) & (actual <= hi_a)
    segs = {"wk 1-13": slice(0, 13), "wk 14-39": slice(13, 39), "wk 40-52": slice(39, 52)}
    naive = float(np.sqrt(np.mean((actual - train.units.values[-52:]) ** 2)))

    # In-sample posterior predictive over the last SHOW_TRAIN training weeks (same parameters)
    lev = post["level"].values.reshape(-1, n)
    mu_tr = lev[:, -SHOW_TRAIN:] + b @ X_tr[-SHOW_TRAIN:].T
    rng_in = np.random.default_rng(7)
    pp_tr = mu_tr + rng_in.normal(0, 1, size=mu_tr.shape) * sg[:, None]
    lo_tr, hi_tr = np.percentile(pp_tr, [3, 97], axis=0) * y_sd + y_mu

    # "Given the parameters": the walk's variance after h steps is h * sigma_level^2, plus the
    # noise sigma^2. Evaluated at the posterior means; the band itself also mixes over them.
    sl_m, sg_m = float(sl.mean()), float(sg.mean())
    h = np.arange(1, N_TEST + 1)
    sd_given = np.sqrt(sg_m**2 + h * sl_m**2) * y_sd

    # Fourier: posterior means and a thinned set of draws, in units (beta is on the z scale)
    level_mean = lev.mean(0) * y_sd + y_mu
    thin = np.random.default_rng(0).choice(len(b), size=400, replace=False)

    m = {
        "note": "Written by scripts/figures/lecture11_part1_figures.py; every number the "
                "Lecture 11 Part 1 figures and deck print. The lab's fits and seeds, verbatim.",
        "versions": {"pymc": pm.__version__, "arviz": az.__version__, "numpy": np.__version__},
        "n_train": int(n), "n_test": N_TEST, "y_mu": float(y_mu), "y_sd": float(y_sd),
        "diagnostics_centred": diag_c, "diagnostics_noncentred": diag,
        "rmse_point": rmse_point, "rmse_seasonal_naive": naive,
        "mean_units_all": float(ca1.units.mean()), "mean_units_test": float(actual.mean()),
        "mean_point_test": float(point.mean()),
        "propagated": {"coverage": float(inside_b.mean()), "inside": int(inside_b.sum()),
                       "mean_width": float(np.mean(hi_b - lo_b)),
                       "by_horizon": {k: {"coverage": float(inside_b[v].mean()),
                                          "mean_width": float((hi_b - lo_b)[v].mean())}
                                      for k, v in segs.items()}},
        "held_constant": {"coverage": float(inside_a.mean()), "inside": int(inside_a.sum()),
                          "mean_width": float(np.mean(hi_a - lo_a)),
                          "by_horizon": {k: {"coverage": float(inside_a[v].mean()),
                                             "mean_width": float((hi_a - lo_a)[v].mean())}
                                         for k, v in segs.items()}},
        "sigma_level_mean_z": sl_m, "sigma_mean_z": sg_m,
        "dates_train": [str(d.date()) for d in train.week_start.iloc[-SHOW_TRAIN:]],
        "dates_test": [str(d.date()) for d in test.week_start],
        "units_train": [float(v) for v in train.units.values[-SHOW_TRAIN:]],
        "units_test": [float(v) for v in actual],
        "fit_train_mean": (mu_tr.mean(0) * y_sd + y_mu).tolist(),
        "band_train": [lo_tr.tolist(), hi_tr.tolist()],
        "point_test": point.tolist(),
        "band_test": [lo_b.tolist(), hi_b.tolist()],
        "sd_given_params": sd_given.tolist(),
        "beta_mean_units": (b.mean(0) * y_sd).tolist(),
        "beta_draws_units": (b[thin] * y_sd).tolist(),
        "detrended_train": (y - level_mean).tolist(),
        "train_start": str(train.week_start.iloc[0].date()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    MEASURED.write_text(json.dumps(m) + "\n")
    pr = m["propagated"]
    print(f"  RMSE {rmse_point:,.0f} vs naive {naive:,.0f}; {pr['inside']}/52 inside; "
          f"mean width {pr['mean_width']:,.0f} = {100*pr['mean_width']/m['mean_units_all']:.0f}% "
          f"of {m['mean_units_all']:,.0f}")
    for k, v in pr["by_horizon"].items():
        print(f"    {k}: {100*v['coverage']:.0f}% / {v['mean_width']:,.0f}")
    return m


@lru_cache(maxsize=None)
def measured() -> dict:
    if "--retrain" in sys.argv or not MEASURED.exists():
        return measure()
    return json.loads(MEASURED.read_text())


# ================================================================== the interval
# Widths are reported as the half-width over the average weekly units of the whole series, the
# denominator the lab uses in Step 4 (19,730), so "±30%" here is the lab's 59%-wide band.


def _band_arrays(m):
    lo_tr, hi_tr = (np.array(a) for a in m["band_train"])
    lo, hi = (np.array(a) for a in m["band_test"])
    return lo_tr, hi_tr, lo, hi


def interval_band() -> None:
    """The last 60 training weeks and the 52 held-out weeks: actuals, posterior mean, 94% band."""
    m = measured()
    lo_tr, hi_tr, lo, hi = _band_arrays(m)
    yt, ye = np.array(m["units_train"]), np.array(m["units_test"])
    fit, point = np.array(m["fit_train_mean"]), np.array(m["point_test"])
    ntr, nte = len(yt), len(ye)
    xt, xe = np.arange(-ntr + 1, 1), np.arange(1, nte + 1)
    avg = m["mean_units_all"]
    hw = (hi - lo) / 2
    inside = (ye >= lo) & (ye <= hi)
    assert int(inside.sum()) == m["propagated"]["inside"]

    fig, ax = plt.subplots(figsize=sz(15, 4.6))
    k = 1e-3
    ax.fill_between(xt, lo_tr * k, hi_tr * k, color=BLUE, alpha=0.10, lw=0)
    ax.fill_between(xe, lo * k, hi * k, color=BLUE, alpha=0.20, lw=0)
    ax.plot(xt, fit * k, color=BLUE, lw=1.3, alpha=0.8)
    ax.plot(xe, point * k, color=BLUE, lw=2.0)
    ax.plot(xt, yt * k, "o", ms=3.2, color=GRAY, alpha=0.75, mec="none")
    ax.plot(xe[inside], ye[inside] * k, "o", ms=3.8, color=BLACK, mec="none")
    if (~inside).any():
        ax.plot(xe[~inside], ye[~inside] * k, "o", ms=4.5, color=CRIMSON, mec="none")

    # the width, measured at the first and the last held-out week
    for i in (0, nte - 1):
        ax.annotate("", xy=(xe[i], hi[i] * k), xytext=(xe[i], lo[i] * k),
                    arrowprops=dict(arrowstyle="<|-|>", color=CRIMSON, lw=1.4, mutation_scale=10))
    lab_y = min(lo[:6].min(), lo_tr[-6:].min()) * k - 0.45
    ax.plot([0.5, 0.5], [lab_y + 0.25, 33.8], color=GRAY, lw=0.9, ls=(0, (3, 2)))   # the origin
    ax.text(xe[0], lab_y, f"±{100 * hw[0] / avg:.0f}%\n1 week out", ha="center", va="top",
            fontsize=11, color=CRIMSON, fontweight="bold", linespacing=1.0)
    ax.text(xe[-1] + 1.3, point[-1] * k, f"±{100 * hw[-1] / avg:.0f}%\n52 weeks\nout",
            ha="left", va="center", fontsize=11, color=CRIMSON, fontweight="bold",
            linespacing=1.0)

    ax.set_xlim(-ntr - 1, nte + 8)
    ax.set_ylim(11.5, 33.8)
    ax.set_yticks([15, 20, 25, 30])
    ax.spines["bottom"].set_bounds(-ntr - 1, nte + 1)
    ax.set_xticks([-52, -26, 0, 13, 26, 39, 52])
    ax.set_xlabel("weeks from the forecast origin (May 2015)", fontsize=12)
    ax.set_ylabel("units (thousands)", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.text(-ntr + 1, 33.7, "training weeks: fit and 94% band", ha="left", va="top",
            fontsize=11, color=GRAY)
    ax.text(2.0, 33.7, f"held out: {inside.sum()} of {nte} weeks inside the 94% band;\n"
            f"its width: ±{100 * hw.mean() / avg:.0f}% of an average week, on average",
            ha="left", va="top", fontsize=11, color=BLACK)
    save(fig, "interval_band")


def band_width() -> None:
    """Half-width of the 94% band against the horizon, with sigma^2 + h sigma_level^2."""
    m = measured()
    _, _, lo, hi = _band_arrays(m)
    avg, ysd = m["mean_units_all"], m["y_sd"]
    h = np.arange(1, len(lo) + 1)
    hw = 100 * (hi - lo) / 2 / avg
    z94 = 1.880794   # the 97th percentile of N(0,1)
    given = 100 * z94 * np.array(m["sd_given_params"]) / avg
    floor = 100 * z94 * m["sigma_mean_z"] * ysd / avg
    cross = (m["sigma_mean_z"] / m["sigma_level_mean_z"]) ** 2

    fig, ax = plt.subplots(figsize=sz(9, 4.2))
    ax.plot(h, hw, color=BLUE, lw=2.2)
    ax.plot(h, given, color=GRAY, lw=1.6, ls=(0, (4, 2)))
    ax.axhline(floor, xmax=0.97, color=GRAY, lw=1.0, ls=(0, (1, 2)))
    ax.set_xlim(0, 54); ax.set_ylim(0, 46)
    ax.set_xticks([1, 13, 26, 39, 52])
    ax.set_yticks([0, 10, 20, 30, 40]); ax.set_yticklabels(["0", "±10%", "±20%", "±30%", "±40%"])
    ax.tick_params(labelsize=11)
    ax.set_xlabel("weeks ahead, h", fontsize=12)
    ax.set_ylabel("half-width of the band", fontsize=12)
    ax.text(53, hw[-1] + 1.6, "the 94% band", ha="right", va="bottom", fontsize=11, color=BLUE)
    i = 30
    ax.text(h[i], given[i] - 2.2, r"$1.88\sqrt{\sigma^2 + h\,\sigma^2_{\mathrm{level}}}$"
            "\ngiven the parameters", ha="left", va="top", fontsize=11, color=GRAY,
            linespacing=1.1)
    ax.text(53, floor - 1.4, r"noise alone: $1.88\,\sigma$", ha="right", va="top", fontsize=11,
            color=GRAY)
    # where the walk's variance h*sigma_level^2 overtakes the noise sigma^2
    gx = np.interp(cross, h, given)
    ax.plot([cross], [gx], "o", ms=5, color=GRAY, zorder=4)
    ax.annotate(f"h ≈ {cross:.0f}: drift\nmatches noise", xy=(cross + 0.4, gx - 0.4),
                xytext=(cross + 7.0, 17.6), fontsize=11, color=GRAY, ha="left", va="center",
                linespacing=1.0, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.9,
                                                 shrinkA=1, shrinkB=5))
    save(fig, "band_width")


# ================================================================== Fourier pairs as arrows
WAVE_COLORS = (BLUE, GREEN, AMBER)


def fourier_pairs() -> None:
    """The K = 3 waves and their sum over the detrended training weeks; each (a_k, b_k) an arrow.

    a*sin(x) + b*cos(x) = r*sin(x + phi), with r = sqrt(a^2 + b^2) and phi the angle of the point
    (a, b). So each pair is one arrow: its length is the wave's amplitude, its angle the wave's
    timing. Coefficients are the posterior means of the lab's non-centred fit, in units. (The
    posterior draws are cached too; they are not drawn, because for k = 2 and 3 a third of them
    lie more than twice the arrow's length from the origin: those two waves are barely resolved.)
    """
    m = measured()
    bm = np.array(m["beta_mean_units"]).reshape(K, 2)          # (a_k, b_k): sin, cos
    bd = np.array(m["beta_draws_units"]).reshape(-1, K, 2)
    det = np.array(m["detrended_train"])
    t = np.arange(len(det))
    k_ = 1e-3

    fig = plt.figure(figsize=sz(15.5, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.75, 0.62, 0.62, 0.62], wspace=0.12)

    # (a) the waves by position in the 52-week cycle, over the detrended data
    ax = fig.add_subplot(gs[0])
    ax.plot(t % P, det * k_, "o", ms=2.6, color=GRAY, alpha=0.35, mec="none")
    x = np.linspace(0, P, 400)
    total = np.zeros_like(x)
    for kk in range(K):
        a, b = bm[kk]
        w = a * np.sin(2*np.pi*(kk+1)*x/P) + b * np.cos(2*np.pi*(kk+1)*x/P)
        total += w
        ax.plot(x, w * k_, color=WAVE_COLORS[kk], lw=1.5)
    ax.plot(x, total * k_, color=CRIMSON, lw=2.6)
    ax.axhline(0, color=GRAY, lw=0.6)
    start = pd.Timestamp(m["train_start"])
    ticks = [0, 13, 26, 39, 52]
    ax.set_xticks(ticks)
    ax.set_xticklabels([(start + pd.Timedelta(weeks=w)).strftime("%-d %b") for w in ticks])
    ax.set_xlim(-0.5, 52.5)
    ax.set_ylim(-4.2, 5.4)
    ax.set_xlabel("week of the 52-week cycle (first year's dates)", fontsize=12)
    ax.set_ylabel("units minus the level\n(thousands)", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.text(51.5, 5.35, "the sum: our seasonal shape", ha="right", va="top", fontsize=11,
            color=CRIMSON, fontweight="bold")

    # (b) each pair (a_k, b_k) as an arrow from the origin; each dial drawn to its own scale
    #     (the dashed circle has radius = that wave's amplitude)
    names = ("once a year", "twice a year", "three times")
    for kk in range(K):
        ax = fig.add_subplot(gs[kk + 1]); ax.set_aspect("equal"); ax.axis("off")
        a, b = bm[kk]
        r = float(np.hypot(a, b))
        phi = float(np.arctan2(b, a))
        c = WAVE_COLORS[kk]
        ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.75, 1.75)
        ax.plot([-1.6, 1.6], [0, 0], color=GRAY, lw=0.7, zorder=0)
        ax.plot([0, 0], [-1.6, 1.6], color=GRAY, lw=0.7, zorder=0)
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, edgecolor=c, lw=1.0, ls=(0, (3, 2))))
        arr = ax.annotate("", xy=(a / r, b / r), xytext=(0, 0), zorder=4,
                          arrowprops=dict(arrowstyle="-|>", color=c, lw=2.4, mutation_scale=13,
                                          shrinkA=0, shrinkB=0))
        arr.arrow_patch.set_path_effects([pe.Stroke(linewidth=5, foreground="white"), pe.Normal()])
        arc = np.linspace(0, phi, 60)
        ax.plot(0.38 * np.cos(arc), 0.38 * np.sin(arc), color=BLACK, lw=1.1, zorder=3)
        ax.text(0, 1.8, f"k = {kk + 1}", ha="center", va="bottom", fontsize=12, color=c,
                fontweight="bold")
        ax.text(0, -1.8, f"length {r:,.0f}\nangle {np.degrees(phi):.0f}°".replace("-", "−"),
                ha="center", va="top", fontsize=11, color=GRAY, linespacing=1.05)
        if kk == 0:
            ax.text(1.6, -0.1, "a", ha="right", va="top", fontsize=11, color=GRAY,
                    style="italic")
            ax.text(0.12, 1.62, "b", ha="left", va="top", fontsize=11, color=GRAY, style="italic")
        print(f"    k={kk+1}: a={a:,.0f} b={b:,.0f} r={r:,.0f} phase={np.degrees(phi):.0f} deg")
    fig.axes[2].text(0, -3.15, "each dial drawn to its own scale", ha="center", va="top",
                     fontsize=10, color=GRAY, clip_on=False)
    save(fig, "fourier_pairs")


FIGURES = {"interval_band": interval_band, "band_width": band_width,
           "fourier_pairs": fourier_pairs}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if not a.startswith("--")] or list(FIGURES)
    print(f"Lecture 11 Part 1 figures -> {OUT.relative_to(ROOT)}   (font: {FONT})")
    if "--retrain" in sys.argv:
        measured()
    for name in pick:
        FIGURES[name]()
