#!/usr/bin/env python3
"""Figures for Lecture 11 Part 2 (Hierarchical Models) — one script, every figure, both formats.

    .venv/bin/python scripts/figures/lecture11_part2_figures.py                   # all
    .venv/bin/python scripts/figures/lecture11_part2_figures.py funnel            # one, by name
    .venv/bin/python scripts/figures/lecture11_part2_figures.py --retrain         # re-sample

Writes Figures/Lecture11_Part2/<name>.pdf (Beamer) and <name>.svg (Quarto), both vector.

Every number comes from the lab's own fits (Labs/Lecture11_Part2_lab.qmd, Setup and Steps 5-6):
the same data pipeline, the same thinning (RandomState(42)), the same PyMC model and the same
pm.sample call with random_seed=42. The fits take seconds; a ten-seed sweep of the centred and
non-centred models adds about a minute. What the figures print goes to
Figures/Lecture11_Part2/measured.json, and later runs redraw from that file.

A caution the sweep exists for: NUTS trajectories are chaotic in the last bit. Reading the
training rows back through a CSV (a one-ulp change in 343 of 1,425 values) moves seed 42's
centred divergences from 43 to 49, and another machine reported 127. The counts are a property
of one seed on one machine; that the centred form diverges and the non-centred form (nearly)
does not is the property of the model.
"""
from __future__ import annotations

import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, NullLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Figures" / "Lecture11_Part2"
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


# ================================================================== the lab, verbatim
# Labs/Lecture11_Part2_lab.qmd, Setup + Step 5 + the fit() helper. Kept identical on purpose:
# the row order, the RandomState draw order and the pm.sample call decide every number.
RANDOM_STATE = 42
SEEDS_SWEEP = tuple(range(1, 11))
FUNNEL_SERIES = "CA_1_FOODS"          # one thin series; all five look alike


@lru_cache(maxsize=None)
def lab_data():
    weekly = pd.read_csv(ROOT / "data" / "processed" / "m5_weekly.csv", parse_dates=["week_start"])
    weekly["series"] = weekly.store_id + "_" + weekly.cat_id
    weekly["hi_snap"] = (weekly.snap_days >= 5).astype(int)
    df = weekly.dropna(subset=["units"]).copy()
    df["ly"] = np.log(df.units.clip(lower=1))

    foods = df[df.cat_id == "FOODS"].copy()
    codes_f, uniq_f = pd.factorize(foods.series)
    foods["idx"] = codes_f
    Jf = len(uniq_f)
    rng = np.random.RandomState(RANDOM_STATE)
    keep_f = []
    for j in range(Jf):
        rows = foods.index[foods.idx == j].tolist()
        n = 8 if j % 2 == 0 else len(rows)
        keep_f += list(rng.choice(rows, size=min(n, len(rows)), replace=False))
    train_f = foods.loc[sorted(keep_f)].copy()
    heldout_f = foods.drop(index=keep_f).copy()
    thin_f = np.array([j % 2 == 0 for j in range(Jf)])
    return train_f, heldout_f, list(uniq_f), thin_f


def empirical_effect(frame, series_name):
    """Observed high-SNAP effect on log units for one series (the lab's helper)."""
    z = frame[frame.series == series_name]
    if z.hi_snap.nunique() < 2:
        return np.nan
    return z.ly[z.hi_snap == 1].mean() - z.ly[z.hi_snap == 0].mean()


def fit(data, kind, n_series, centered=False, target_accept=0.9, seed=RANDOM_STATE):
    """The lab's fit(); `seed` is the lab's RANDOM_STATE unless the sweep overrides it."""
    import pymc as pm
    i = data.idx.values
    x = data.hi_snap.values.astype(float)
    y = data.ly.values
    with pm.Model():
        a = pm.Normal("a", y.mean(), 1, shape=n_series)
        s = pm.Exponential("s", 1.0)
        if kind == "unpooled":
            b = pm.Normal("b", 0, 0.5, shape=n_series)
        else:
            mu_b = pm.Normal("mu_b", 0, 0.5)
            sd_b = pm.HalfNormal("sd_b", 0.25)
            if centered:
                b = pm.Normal("b", mu_b, sd_b, shape=n_series)
            else:
                z = pm.Normal("z", 0, 1, shape=n_series)
                b = pm.Deterministic("b", mu_b + z * sd_b)
        pm.Normal("obs", mu=a[i] + b[i] * x, sigma=s, observed=y)
        return pm.sample(1000, tune=1500, chains=4, progressbar=False,
                         random_seed=seed, target_accept=target_accept)


def diag(idata) -> dict:
    """The lab's diag(): worst R-hat and ESS over b, and the divergence count."""
    import arviz as az
    su = az.summary(idata, var_names=["b"], ci_kind="hdi", ci_prob=0.94)
    return {"divergences": int(idata.sample_stats.diverging.sum()),
            "r_hat": float(su.r_hat.max()), "ess_bulk": float(su.ess_bulk.min()),
            "seconds": float(idata.posterior.attrs.get("sampling_time", float("nan")))}


# ================================================================== measurement
def measure() -> dict:
    import logging
    import warnings
    import arviz as az
    warnings.filterwarnings("ignore")
    logging.getLogger("pymc").setLevel(logging.ERROR)
    train_f, heldout_f, names, thin = lab_data()
    J = len(names)
    t0 = time.time()
    uf = fit(train_f, "unpooled", J)
    hf = fit(train_f, "hier", J)
    cent = fit(train_f, "hier", J, centered=True)
    print(f"  three lab fits: {time.time() - t0:.0f}s")

    b_means = lambda idata: az.summary(idata, var_names=["b"], ci_kind="hdi",
                                       ci_prob=0.94)["mean"].values
    bu, bh = b_means(uf), b_means(hf)
    g = az.summary(hf, var_names=["mu_b", "sd_b"], ci_kind="hdi", ci_prob=0.94)
    mu_b = float(g.loc["mu_b", "mean"])
    move = np.abs(bh - bu)
    truth = np.array([empirical_effect(heldout_f, s) if thin[k] else np.nan
                      for k, s in enumerate(names)])
    rmse = lambda e: float(np.sqrt(np.nanmean((e - truth[thin]) ** 2)))
    # how far along the way to mu_b each estimate went, and the plug-in precision weights
    se_u = uf.posterior["b"].values.reshape(-1, J).std(axis=0)
    sd_draws = hf.posterior["sd_b"].values.reshape(-1)
    mu_draws = hf.posterior["mu_b"].values.reshape(-1)
    w_plug = (1 / se_u**2) / (1 / se_u**2 + 1 / float(sd_draws.mean()) ** 2)
    approx_plug = w_plug * bu + (1 - w_plug) * mu_b
    w_d = (1 / se_u**2) / (1 / se_u**2 + 1 / sd_draws[:, None] ** 2)
    approx_int = (w_d * bu + (1 - w_d) * mu_draws[:, None]).mean(axis=0)

    m: dict = {
        "note": "Written by scripts/figures/lecture11_part2_figures.py from the lab's own fits "
                "(Labs/Lecture11_Part2_lab.qmd Steps 5-6, random_seed=42). Every number the "
                "Lecture 11 Part 2 figures print.",
        "series": names, "thin": thin.tolist(),
        "n_train": train_f.groupby("idx").size().tolist(),
        "train_rows": int(len(train_f)), "heldout_rows": int(len(heldout_f)),
        "b_unpooled": bu.tolist(), "b_hier": bh.tolist(), "se_unpooled": se_u.tolist(),
        "heldout_truth": [None if np.isnan(t) else float(t) for t in truth],
        "mu_b": mu_b, "mu_b_hdi": [float(g.loc["mu_b", "hdi94_lb"]), float(g.loc["mu_b", "hdi94_ub"])],
        "sd_b": float(g.loc["sd_b", "mean"]),
        "spread_unpooled": {"thin": float(bu[thin].std()), "full": float(bu[~thin].std())},
        "spread_hier": {"thin": float(bh[thin].std()), "full": float(bh[~thin].std())},
        "mean_move": {"thin": float(move[thin].mean()), "full": float(move[~thin].mean())},
        "ratio": float(move[thin].mean() / move[~thin].mean()),
        "share_of_way_to_mu": ((bu - bh) / (bu - mu_b)).tolist(),
        "precision_weight_plugin": w_plug.tolist(),
        "approx_plugin_max_abs_err": float(np.abs(approx_plug - bh).max()),
        "approx_integrated_max_abs_err": float(np.abs(approx_int - bh).max()),
        "rmse_vs_truth": {"unpooled": rmse(bu[thin]), "hier": rmse(bh[thin])},
        "diag": {"unpooled": diag(uf), "noncentred": diag(hf), "centred": diag(cent)},
    }
    m["rmse_ratio"] = m["rmse_vs_truth"]["unpooled"] / m["rmse_vs_truth"]["hier"]

    # the funnel: one thin series, every draw, in both parameterizations
    j = names.index(FUNNEL_SERIES)
    cpost, npost = cent.posterior, hf.posterior
    c_div = cent.sample_stats["diverging"].values.reshape(-1).astype(bool)
    c_sd = cpost["sd_b"].values.reshape(-1)
    c_dev = cpost["b"].values.reshape(-1, J)[:, j] - cpost["mu_b"].values.reshape(-1)
    n_sd = npost["sd_b"].values.reshape(-1)
    n_z = npost["z"].values.reshape(-1, J)[:, j]
    pct = np.array([(c_sd < v).mean() for v in c_sd[c_div]])
    m["funnel"] = {
        "series": FUNNEL_SERIES,
        "centred_dev": np.round(c_dev, 5).tolist(), "centred_sd_b": np.round(c_sd, 6).tolist(),
        "centred_diverging": c_div.astype(int).tolist(),
        "noncentred_z": np.round(n_z, 4).tolist(), "noncentred_sd_b": np.round(n_sd, 7).tolist(),
        "noncentred_divergences": int(hf.sample_stats["diverging"].values.sum()),
        "divergent_sd_b_percentiles": np.round(np.sort(pct), 4).tolist(),
        "divergent_below_10th_pct": int((pct < 0.10).sum()),
        "centred_min_sd_b": float(c_sd.min()),
        "noncentred_share_below_centred_min": float((n_sd < c_sd.min()).mean()),
    }

    # the narration's claims: target_accept 0.95, and the ten-seed sweep
    m["centred_target_accept_095"] = diag(fit(train_f, "hier", J, centered=True,
                                              target_accept=0.95))
    sweep = []
    for s in SEEDS_SWEEP:
        sweep.append({"seed": s,
                      "centred": diag(fit(train_f, "hier", J, centered=True, seed=s)),
                      "noncentred": diag(fit(train_f, "hier", J, seed=s))})
        print(f"  seed {s:2d}: centred {sweep[-1]['centred']['divergences']:3d} divergences, "
              f"non-centred {sweep[-1]['noncentred']['divergences']}")
    m["seed_sweep"] = sweep

    OUT.mkdir(parents=True, exist_ok=True)
    MEASURED.write_text(json.dumps(m) + "\n")
    d = m["diag"]
    print(f"  measured.json written: ratio {m['ratio']:.2f}x, RMSE {m['rmse_vs_truth']['unpooled']:.3f}"
          f" -> {m['rmse_vs_truth']['hier']:.3f}; divergences centred {d['centred']['divergences']}"
          f" / non-centred {d['noncentred']['divergences']}")
    return m


@lru_cache(maxsize=None)
def measured() -> dict:
    if "--retrain" in sys.argv or not MEASURED.exists():
        return measure()
    return json.loads(MEASURED.read_text())


# ================================================================== shrinkage arrows (G1)
def _short(name: str) -> str:
    return name.replace("_FOODS", "").replace("_", " ")


def shrinkage(heldout: bool) -> None:
    """One row per FOODS series: an arrow from its own (unpooled) estimate to the pooled one.

    Thin series (8 weeks) in crimson, full ones (277 weeks) in gray, mu_b dashed. With
    `heldout`, the SNAP effect actually observed in the thin series' held-out weeks is a ring.
    The arrow length is the lab's "mean move", so the two groups' mean arrows are its ratio.
    """
    m = measured()
    names, thin = m["series"], np.array(m["thin"])
    bu, bh = np.array(m["b_unpooled"]), np.array(m["b_hier"])
    truth = m["heldout_truth"]
    order = [k for k in range(len(names)) if thin[k]] + [k for k in range(len(names)) if not thin[k]]
    ys = {k: (len(order) - i + (0.8 if i < thin.sum() else 0)) for i, k in enumerate(order)}
    top = max(ys.values())

    fig, ax = plt.subplots(figsize=sz(14.4, 4.45))
    X0, X1 = -0.012, 0.36
    ax.set_xlim(X0, X1)
    ax.set_ylim(0.4, top + 0.95)
    ax.plot([m["mu_b"]] * 2, [0.4, top + 0.42], color=GRAY, lw=1.3, ls=(0, (4, 3)), zorder=1)
    ax.text(m["mu_b"], top + 0.55, f"$\\mu_b$ = {m['mu_b']:.3f}", ha="center", va="bottom",
            fontsize=12, color=GRAY)
    for k in order:
        y, c = ys[k], (CRIMSON if thin[k] else GRAY)
        marks = [bu[k], bh[k]] + ([truth[k]] if heldout and truth[k] is not None else [])
        ax.plot([X0, min(marks) - 0.012], [y, y], color=LIGHT_GRAY, lw=1.2, zorder=0)
        ax.plot([bu[k]], [y], "o", ms=5.5, color=c, alpha=0.45, zorder=3, mec="none")
        if abs(bh[k] - bu[k]) > 0.007:
            ax.annotate("", xy=(bh[k], y), xytext=(bu[k], y),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=2.0, mutation_scale=13,
                                        shrinkA=0, shrinkB=0), zorder=4)
        else:
            ax.plot([bh[k]], [y], marker="|", ms=11, mew=2.2, color=c, zorder=4)
        if heldout and truth[k] is not None:
            ax.plot([truth[k]], [y], "o", ms=10, mfc="none", mec=BLACK, mew=1.6, zorder=5)
    ax.set_yticks([ys[k] for k in order])
    ax.set_yticklabels([_short(names[k]) for k in order], fontsize=11)
    for lab, k in zip(ax.get_yticklabels(), order):
        lab.set_color(CRIMSON if thin[k] else GRAY)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.tick_params(axis="x", labelsize=11)
    ax.set_xlabel("SNAP-week effect on log units", fontsize=12)

    # group labels and the mean arrow length, in the right margin
    thin_ys = [ys[k] for k in order if thin[k]]
    full_ys = [ys[k] for k in order if not thin[k]]
    mm = m["mean_move"]
    tx = X1 + 0.010
    for gys, label, mv, c in [(thin_ys, "8 weeks", mm["thin"], CRIMSON),
                              (full_ys, "277 weeks", mm["full"], GRAY)]:
        ax.text(tx, np.mean(gys) + 0.5, label, ha="left", va="bottom", fontsize=12, color=c,
                fontweight="bold", clip_on=False)
        ax.text(tx, np.mean(gys) + 0.3, f"mean arrow\n{mv:.3f}", ha="left", va="top",
                fontsize=11, color=c, clip_on=False)

    # legend in the empty lower right, one entry per row level (the tick labels' own pitch), so
    # it can never sit closer than the rows do; right of every full-series mark (WI 3 ends 0.17)
    cx, tx_ = 0.222, 0.246                                 # marker column, text column
    l1, l2, l3 = full_ys[1], full_ys[2], full_ys[3]
    ax.plot([cx], [l1], "o", ms=5.5, color=GRAY, alpha=0.45, mec="none")
    ax.text(tx_, l1, "own estimate", ha="left", va="center", fontsize=11, color=GRAY)
    ax.annotate("", xy=(cx + 0.013, l2), xytext=(cx - 0.013, l2),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2.0, mutation_scale=13,
                                shrinkA=0, shrinkB=0))
    ax.text(tx_, l2, "pooled estimate", ha="left", va="center", fontsize=11, color=GRAY)
    if heldout:
        ax.plot([cx], [l3], "o", ms=10, mfc="none", mec=BLACK, mew=1.6)
        ax.text(tx_, l3, "held-out truth", ha="left", va="center", fontsize=11, color=BLACK)
    save(fig, "shrinkage_heldout" if heldout else "shrinkage_arrows")


def shrinkage_arrows() -> None:
    shrinkage(heldout=False)


def shrinkage_heldout() -> None:
    shrinkage(heldout=True)


# ================================================================== the funnel (G5)
def funnel() -> None:
    """Centered b_j - mu_b against sigma_b (log axis), divergences in crimson; non-centered z_j
    against the same axis. z_j = (b_j - mu_b) / sigma_b: the change of coordinates divides each
    height by its own width, so the funnel becomes a column. The dashed guides are the prior's
    +-2 sigma_b and +-2: exact for the prior; the posterior follows them closely.
    """
    m = measured()
    f = m["funnel"]
    c_dev, c_sd = np.array(f["centred_dev"]), np.array(f["centred_sd_b"])
    c_div = np.array(f["centred_diverging"]).astype(bool)
    n_z, n_sd = np.array(f["noncentred_z"]), np.array(f["noncentred_sd_b"])
    Y0, Y1, XL, ZL = 0.002, 0.5, 0.62, 4.3
    assert c_sd.min() >= Y0 and c_sd.max() <= Y1 and np.abs(n_z).max() <= ZL
    fig = plt.figure(figsize=sz(14.4, 4.25))
    gs = fig.add_gridspec(1, 2, wspace=0.10)
    yy = np.geomspace(Y0, Y1, 200)
    floor = f["centred_min_sd_b"]

    ax = fig.add_subplot(gs[0])
    ax.scatter(c_dev[~c_div], c_sd[~c_div], s=5, color=MID_GRAY, alpha=0.35, lw=0, zorder=2)
    ax.scatter(c_dev[c_div], c_sd[c_div], s=22, color=CRIMSON, lw=0, zorder=4)
    for sgn in (-1, 1):
        ax.plot(sgn * 2 * yy, yy, color=BLACK, lw=1.1, ls=(0, (4, 3)), zorder=3)
    ax.set_xlim(-XL, XL)
    ax.set_xticks([-0.6, -0.3, 0, 0.3, 0.6])
    ax.set_xlabel("$b_j - \\mu_b$  (centered)", fontsize=12)
    ax.set_title(f"centered: {int(c_div.sum())} divergences", fontsize=12, color=CRIMSON, pad=6)
    # both notes sit below the lowest centered draw, where the panel is empty
    ax.text(0.07, floor * 0.55, "prior: ±2$\\sigma_b$", ha="left", va="center", fontsize=11,
            color=BLACK)
    ax.text(-XL + 0.02, floor * 0.82, "chains never went\nbelow this line",
            ha="left", va="top", fontsize=11, color=CRIMSON)
    clipped_c = int((np.abs(c_dev) > XL).sum())
    axl = ax

    ax = fig.add_subplot(gs[1], sharey=axl)
    ax.scatter(n_z, n_sd, s=5, color=MID_GRAY, alpha=0.35, lw=0, zorder=2)
    for sgn in (-1, 1):
        ax.plot([sgn * 2, sgn * 2], [Y0, Y1], color=BLACK, lw=1.1, ls=(0, (4, 3)), zorder=3)
    ax.set_xlim(-ZL, ZL)
    ax.set_xticks([-2, 0, 2])
    ax.set_xlabel("$z_j = (b_j - \\mu_b)\\,/\\,\\sigma_b$  (non-centered)", fontsize=12)
    ax.set_title(f"non-centered: {f['noncentred_divergences']} divergences", fontsize=12,
                 color=GRAY, pad=6)
    ax.text(2.3, 0.33, "prior: ±2", ha="left", va="center", fontsize=11, color=BLACK)
    below = int((n_sd < Y0).sum())
    ax.text(2.3, Y0 * 1.12, f"{below} draws\nbelow 0.002", ha="left", va="bottom",
            fontsize=11, color=GRAY)
    plt.setp(ax.get_yticklabels(), visible=False)
    axr = ax

    for ax in (axl, axr):
        ax.set_yscale("log")
        ax.set_ylim(Y0, Y1)
        ax.yaxis.set_major_locator(FixedLocator([0.003, 0.01, 0.03, 0.1, 0.3]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(labelsize=11)
        ax.axhline(floor, color=CRIMSON, lw=0.9, ls=(0, (1, 2)), zorder=1)
    axl.set_yticklabels(["0.003", "0.01", "0.03", "0.1", "0.3"])
    axl.set_ylabel("$\\sigma_b$ (log scale)", fontsize=12)
    print(f"    funnel: {clipped_c} centered draws beyond |b - mu| > {XL}; {below} non-centered"
          f" draws below sigma_b = {Y0}")
    save(fig, "funnel")


FIGURES = {"shrinkage_arrows": shrinkage_arrows, "shrinkage_heldout": shrinkage_heldout,
           "funnel": funnel}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if not a.startswith("--")] or list(FIGURES)
    print(f"Lecture 11 Part 2 figures -> {OUT.relative_to(ROOT)}   (font: {FONT})")
    if "--retrain" in sys.argv:
        measured()
    for name in pick:
        FIGURES[name]()
