#!/usr/bin/env python3
"""Figures for Lecture 10 (Bayesian Statistics I: Foundations) — one script, every figure, both formats.

    .venv/bin/python scripts/figures/lecture10_figures.py                 # all
    .venv/bin/python scripts/figures/lecture10_figures.py bayes_update    # one, by name
    .venv/bin/python scripts/figures/lecture10_figures.py --retrain       # re-run the sampler

Writes Figures/Lecture10_BayesianFoundations/<name>.pdf (Beamer) and <name>.svg (Quarto).

The deck's conversion-rate example is conjugate: a Beta(5, 45) prior and 30 successes in 200
trials give a Beta(35, 215) posterior, so every curve is exact. The script asserts the deck's
printed numbers (0.10, 0.15, 0.14) before drawing them.

One thing is sampled: the two MCMC figures run four seeded random-walk Metropolis chains on that
posterior, with PyMC's budget from the deck (tune=1000, draws=2000, chains=4). The chains and their
summaries go to Figures/Lecture10_BayesianFoundations/measured.json; later runs redraw from that
file, and --retrain re-runs the sampler. The prior-predictive counts are exact Beta-Binomial
probabilities, not simulated draws.

One figure reads data: the Bayes' theorem mosaic counts CA_1 FOODS weeks in
data/processed/m5_weekly.csv (run scripts/prep_m5.py first) and asserts the slide's numbers.
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
from matplotlib.patches import Polygon, Rectangle
import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import betabinom, expon, halfnorm, norm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Figures" / "Lecture10_BayesianFoundations"
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


# ================================================================== the deck's example
# Slides/Lecture10_BayesianFoundations.tex: "PyMC: A First Model" and "The Update, in Pictures".
A0, B0 = 5, 45            # prior Beta(5, 45): mean 0.10, worth a0 + b0 = 50 observations
K, N = 30, 200            # 30 successes in 200 trials


def posterior_mean(k: int, n: int) -> float:
    """Conjugate update: (a0 + k) / (a0 + b0 + n), which is also the weighted average
    ((a0 + b0) * prior mean + n * k/n) / (a0 + b0 + n)."""
    return (A0 + k) / (A0 + B0 + n)


# ================================================================== the update, in pictures
def bayes_update() -> None:
    """Prior, likelihood and posterior on one axis; below it, the posterior mean as a balance
    point; on the right, where the balance sits as the same 15% rate arrives in more trials."""
    prior_mean, data_rate, post_mean = A0 / (A0 + B0), K / N, posterior_mean(K, N)
    w_prior, w_data = A0 + B0, N
    # The numbers the deck prints, checked before anything is drawn.
    assert round(prior_mean, 2) == 0.10 and data_rate == 0.15 and round(post_mean, 2) == 0.14
    assert abs(post_mean - (w_prior * prior_mean + w_data * data_rate) / (w_prior + w_data)) < 1e-12
    prior_peak = (A0 - 1) / (A0 + B0 - 2)                       # 0.083: NOT what we tick
    small, large = posterior_mean(3, 20), posterior_mean(300, 2000)
    assert (round(small, 3), round(large, 3)) == (0.114, 0.149)

    XL = (0.0, 0.27)
    x = np.linspace(0.0005, XL[1], 1200)
    prior = beta_dist.pdf(x, A0, B0)
    like = beta_dist.pdf(x, K + 1, N - K + 1)                   # theta^30 (1-theta)^170, normalized
    post = beta_dist.pdf(x, A0 + K, B0 + N - K)

    fig = plt.figure(figsize=sz(13.2, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.3, 1], height_ratios=[2.0, 1.1],
                          wspace=0.28, hspace=0.04, left=0.03, right=0.99, top=0.94, bottom=0.12)

    # ---- the three curves
    ax = fig.add_subplot(gs[0, 0])
    ax.fill_between(x, post, color=LIGHT_CRIMSON, zorder=1)
    ax.plot(x, prior, color=BLUE, lw=2.2, zorder=3)
    ax.plot(x, like, color=AMBER, lw=2.2, ls=(0, (5, 2.5)), zorder=3)
    ax.plot(x, post, color=CRIMSON, lw=2.6, zorder=4)
    # dotted drop-lines at the prior MEAN (not its 0.083 peak), the data rate, the posterior mean
    for xm, a, b, c in [(prior_mean, A0, B0, BLUE), (data_rate, K + 1, N - K + 1, AMBER),
                        (post_mean, A0 + K, B0 + N - K, CRIMSON)]:
        ax.plot([xm, xm], [0, beta_dist.pdf(xm, a, b)], color=c, lw=1.2, ls=(0, (2, 2)), zorder=2)
    ax.text(0.048, 12.6, "prior\nBeta(5, 45)", ha="center", va="center", fontsize=12, color=BLUE)
    ax.text(0.196, 12.4, "likelihood\n30 of 200", ha="left", va="center", fontsize=12, color=AMBER)
    ax.text(0.113, 18.0, "posterior\nBeta(35, 215)", ha="right", va="center", fontsize=12,
            color=CRIMSON)
    ax.set_xlim(*XL); ax.set_ylim(0, 20.5)
    ax.set_yticks([]); ax.set_xticks([])
    ax.spines["left"].set_visible(False)

    # ---- the balance point, on the same theta axis
    ax = fig.add_subplot(gs[1, 0])
    YL = (-1.02, 1.2)
    ax.set_xlim(*XL); ax.set_ylim(*YL)
    ax.axis("off")
    bb = ax.get_position()                                      # make the blocks truly square
    w_in, h_in = bb.width * fig.get_figwidth(), bb.height * fig.get_figheight()
    asp = ((XL[1] - XL[0]) / w_in) / ((YL[1] - YL[0]) / h_in)   # x data units per y data unit
    beam_y, base_y = 0.0, -0.5
    ax.plot([0.088, 0.164], [beam_y, beam_y], color=BLACK, lw=3.0, solid_capstyle="round",
            zorder=3)
    # blocks: AREA proportional to weight (50 : 200 = 1 : 4, so sides 1 : 2)
    side_data = 0.66
    for xc, s, face, edge, lab in [(prior_mean, side_data / 2, LIGHT_BLUE, BLUE, "50"),
                                   (data_rate, side_data, "#FBEBD9", AMBER, "200")]:
        ax.add_patch(Rectangle((xc - s * asp / 2, beam_y + 0.06), s * asp, s,
                               facecolor=face, edgecolor=edge, lw=1.6, zorder=4))
        ax.text(xc, beam_y + 0.06 + s + 0.07, lab, ha="center", va="bottom", fontsize=11,
                color=edge)
    ax.add_patch(Polygon([(post_mean, beam_y - 0.04), (post_mean - 0.0065, base_y),
                          (post_mean + 0.0065, base_y)], closed=True,
                         facecolor=CRIMSON, edgecolor=CRIMSON, zorder=4))
    ax.plot(list(XL), [base_y] * 2, color=GRAY, lw=0.8, zorder=1)
    for xt, lab, c, ha in [(prior_mean, "0.10", BLUE, "center"), (post_mean, "0.14", CRIMSON, "right"),
                           (data_rate, "0.15", AMBER, "left")]:
        dx = {"center": 0.0, "right": 0.002, "left": 0.002}[ha]
        ax.text(xt + (dx if ha == "left" else -dx if ha == "right" else 0), base_y - 0.1, lab,
                ha=ha, va="top", fontsize=12, color=c, fontweight="bold" if c == CRIMSON else "normal")
    ax.text(XL[1], base_y - 0.1, "θ", ha="right", va="top", fontsize=13, color=GRAY,
            style="italic")
    ax.text(0.176, 0.32, "balances at\n(50×0.10 + 200×0.15)/250", ha="left", va="center",
            fontsize=11, color=GRAY)
    ax.text(0.0, 0.32, "weight = trials:\nprior 50, data 200", ha="left", va="center",
            fontsize=11, color=GRAY)

    # ---- inset: the same 15% rate in more trials pulls the balance toward the data
    ax = fig.add_subplot(gs[:, 1])
    n = np.logspace(0, np.log10(20000), 200)
    ax.plot(n, (A0 + 0.15 * n) / (A0 + B0 + n), color=CRIMSON, lw=2.2)
    ax.axhline(prior_mean, color=BLUE, lw=1.2, ls=(0, (2, 2)))
    ax.axhline(data_rate, color=AMBER, lw=1.2, ls=(0, (5, 2.5)))
    for nn, kk, ha, dx, dy in [(20, 3, "left", 1.4, -0.0042), (200, 30, "left", 1.4, -0.0042),
                               (2000, 300, "right", 0.72, 0.0056)]:
        m = posterior_mean(kk, nn)
        ax.plot([nn], [m], "o", ms=6.5, color=CRIMSON, zorder=4)
        ax.text(nn * dx, m + dy, f"{kk:,}/{nn:,}: {m:.3f}", ha=ha, va="center", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlim(1, 20000); ax.set_ylim(0.090, 0.161)
    ax.set_xticks([1, 10, 100, 1000, 10000]); ax.set_xticklabels(["1", "10", "100", "1k", "10k"])
    ax.set_yticks([0.10, 0.15]); ax.set_yticklabels(["0.10", "0.15"])
    for t, c in zip(ax.get_yticklabels(), (BLUE, AMBER)):
        t.set_color(c)
    ax.tick_params(labelsize=11)
    ax.set_xlabel("trials, all at a 15% rate", fontsize=11)
    ax.set_title("more data, nearer the data", fontsize=12, pad=6)

    save(fig, "bayes_update")
    print(f"    prior mean {prior_mean:.3f} (peak {prior_peak:.3f}, not drawn), data {data_rate:.3f}, "
          f"posterior mean {post_mean:.4f}; 3/20 -> {small:.4f}, 300/2000 -> {large:.4f}")


# ================================================================== MCMC: four random walks
# The deck's PyMC call is pm.sample(2000, tune=1000, chains=4). The figures use the same budget, but
# with the textbook sampler the deck describes ("take a step; if higher, move; if lower, usually step
# back"): random-walk Metropolis. PyMC's NUTS takes smarter steps; the picture is the same.
SEED, STEP = 42, 0.02                  # numpy default_rng seed; proposal sd on theta
TUNE, DRAWS = 1000, 2000
STARTS = (0.01, 0.30, 0.55, 0.80)      # scattered on purpose, far from the posterior
THRESH = 0.15                          # the event counted on "The Samples Are the Answer"


def log_height(t):
    """log of likelihood x prior, theta^(30+5-1) (1-theta)^(170+45-1), WITHOUT p(D): the
    unnormalized posterior. Zero height (log = -inf) outside (0, 1)."""
    t = np.asarray(t, dtype=float)
    inside = (t > 0) & (t < 1)
    tt = np.where(inside, t, 0.5)
    return np.where(inside, (A0 + K - 1) * np.log(tt) + (B0 + N - K - 1) * np.log1p(-tt), -np.inf)


def measure() -> dict:
    """Run the four chains, summarize them, and cache both in measured.json."""
    import arviz as az
    rng = np.random.default_rng(SEED)
    chains = np.empty((len(STARTS), TUNE + DRAWS))
    accepted = np.zeros(len(STARTS))
    for c, x in enumerate(STARTS):
        lx = log_height(x)
        for t in range(TUNE + DRAWS):
            y = x + rng.normal(0.0, STEP)
            ly = log_height(y)
            if np.log(rng.uniform()) < ly - lx:      # accept with probability min(1, h(y)/h(x))
                x, lx = y, ly
                accepted[c] += 1
            chains[c, t] = x
    kept = chains[:, TUNE:]
    s = az.summary(kept, ci_kind="hdi", ci_prob=0.94)
    m = {"seed": SEED, "step": STEP, "tune": TUNE, "draws": DRAWS, "starts": list(STARTS),
         "acceptance": (accepted / (TUNE + DRAWS)).round(3).tolist(),
         # first step at which each chain is within 0.05 of the posterior mean
         "steps_to_band": [int(np.argmax(np.abs(ch - 0.14) < 0.05)) for ch in chains],
         "mean_counted": float(kept.mean()),
         "p_above_counted": float((kept > THRESH).mean()),
         "hdi94_counted": [float(v) for v in az.hdi(kept.ravel(), prob=0.94)],
         "ess_bulk": float(s["ess_bulk"].iloc[0]), "r_hat": float(s["r_hat"].iloc[0]),
         "chains": np.round(chains, 5).tolist()}
    OUT.mkdir(parents=True, exist_ok=True)
    MEASURED.write_text(json.dumps(m) + "\n")
    print(f"  measured.json written: acceptance {m['acceptance']}, ESS {m['ess_bulk']:.0f}, "
          f"R-hat {m['r_hat']:.3f}")
    return m


@lru_cache(maxsize=None)
def measured() -> dict:
    if "--retrain" in sys.argv or not MEASURED.exists():
        return measure()
    return json.loads(MEASURED.read_text())


CHAIN_COLORS = (BLUE, AMBER, GREEN, GRAY)


def mcmc_walk() -> None:
    """Left: one Metropolis step on the unnormalized posterior, as two heights. Right: four chains
    from scattered starts, warm-up shaded."""
    m = measured()
    chains = np.array(m["chains"])
    fig = plt.figure(figsize=sz(13.6, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.25], wspace=0.16,
                          left=0.02, right=0.99, top=0.88, bottom=0.17)

    # ---- one step, two heights
    ax = fig.add_subplot(gs[0, 0])
    x = np.linspace(0.04, 0.26, 600)
    peak = log_height((A0 + K - 1) / (A0 + B0 + N - 2))
    h = np.exp(log_height(x) - peak)                           # rescaled to peak 1: shape only
    ax.fill_between(x, h, color=LIGHT_CRIMSON, zorder=1)
    ax.plot(x, h, color=CRIMSON, lw=2.2, zorder=3)
    now, prop = 0.150, 0.180
    h_now, h_prop = (float(np.exp(log_height(v) - peak)) for v in (now, prop))
    ratio = h_prop / h_now
    for v, hv, c in [(now, h_now, BLACK), (prop, h_prop, BLUE)]:
        ax.plot([v, v], [0, hv], color=c, lw=2.4, solid_capstyle="butt", zorder=4)
        ax.plot([v], [hv], "o", ms=5.5, color=c, zorder=5)
    ax.text(now - 0.005, 0.22, "now", ha="right", va="center", fontsize=11, color=BLACK)
    ax.text(prop + 0.004, h_prop + 0.04, "proposed", ha="left", va="bottom", fontsize=11, color=BLUE)
    ax.text(0.04, 1.6, f"lower: move with\nprobability {ratio:.2f},\nthe ratio of heights",
            ha="left", va="top", fontsize=11, color=GRAY)
    ax.set_xlim(0.04, 0.26); ax.set_ylim(0, 1.62)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xticks([0.10, 0.15, 0.20]); ax.set_xticklabels(["0.10", "0.15", "0.20"])
    ax.tick_params(labelsize=11)
    ax.set_xlabel("θ", fontsize=12, style="italic", labelpad=1)
    ax.set_title("height = likelihood × prior", fontsize=12, pad=4)

    # ---- four chains
    ax = fig.add_subplot(gs[0, 1])
    T = chains.shape[1]
    ax.axvspan(0, m["tune"], color=LIGHT_GRAY, zorder=0)
    steps = np.arange(T)
    for c, col in zip(chains, CHAIN_COLORS):
        ax.plot(steps, c, color=col, lw=0.55, alpha=0.85, zorder=2)
    for st, col in zip(m["starts"], CHAIN_COLORS):
        ax.plot([0], [st], "o", ms=6, color=col, zorder=5, clip_on=False)
    ax.text(m["tune"] / 2, 0.83, "warm-up: discarded", ha="center", va="top", fontsize=11,
            color=GRAY)
    ax.text(m["tune"] + m["draws"] / 2, 0.83, f"kept: 4 × {m['draws']:,} draws", ha="center",
            va="top", fontsize=11, color=GRAY)
    ax.text(T * 0.995, 0.40, "every chain ends\nup in one band", ha="right", va="bottom",
            fontsize=11, color=BLACK)
    ax.set_xlim(0, T); ax.set_ylim(0, 0.86)
    ax.set_xticks([0, 1000, 2000, 3000]); ax.set_xticklabels(["0", "1,000", "2,000", "3,000"])
    ax.set_yticks([0, 0.14, 0.4, 0.8]); ax.set_yticklabels(["0", "0.14", "0.4", "0.8"])
    ax.tick_params(labelsize=11)
    ax.set_xlabel("step", fontsize=12, labelpad=1)
    ax.set_ylabel("θ", fontsize=12, style="italic", rotation=0, labelpad=8)
    ax.set_title("four walks from scattered starts", fontsize=12, pad=4)

    save(fig, "mcmc_walk")
    print(f"    one step: h(0.18)/h(0.15) = {ratio:.3f}; steps to the band {m['steps_to_band']}; "
          f"acceptance {m['acceptance']}")


def mcmc_histogram() -> None:
    """The kept draws as a histogram against the exact Beta(35, 215), with one event counted."""
    m = measured()
    kept = np.array(m["chains"])[:, m["tune"]:].ravel()
    a, b = A0 + K, B0 + N - K
    exact_mean, exact_p = a / (a + b), float(beta_dist.sf(THRESH, a, b))
    # what the figure prints, checked against the exact answer
    assert round(m["mean_counted"], 2) == round(exact_mean, 2) == 0.14
    assert abs(m["p_above_counted"] - exact_p) < 0.03

    fig, ax = plt.subplots(figsize=sz(9.4, 3.4))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.97, bottom=0.2)
    edges = np.arange(0.06, 0.2401, 0.005)
    dens, _ = np.histogram(kept, bins=edges, density=True)
    for lo, d in zip(edges[:-1], dens):
        above = lo >= THRESH - 1e-9
        ax.add_patch(Rectangle((lo, 0), 0.005, d, facecolor=LIGHT_CRIMSON if above else LIGHT_GRAY,
                               edgecolor=CRIMSON if above else "#A1A1AA", lw=0.6, zorder=2))
    x = np.linspace(0.06, 0.24, 600)
    ax.plot(x, beta_dist.pdf(x, a, b), color=BLACK, lw=2.0, zorder=4)
    ax.axvline(THRESH, color=CRIMSON, lw=1.2, ls=(0, (3, 2)), zorder=3)
    ax.text(0.063, 19.0, "8,000 kept draws", ha="left", va="center", fontsize=11, color=GRAY)
    ax.text(0.063, 16.5, "exact Beta(35, 215)", ha="left", va="center", fontsize=11, color=BLACK)
    ax.text(0.181, 14.6, f"share of draws above 0.15:\n{m['p_above_counted']:.2f} counted,"
            f" {exact_p:.2f} exact", ha="left", va="center", fontsize=11, color=CRIMSON)
    ax.set_xlim(0.06, 0.24); ax.set_ylim(0, 20.5)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xticks([0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22])
    ax.set_xticklabels(["0.08", "0.10", "0.12", "0.14", "0.16", "0.18", "0.20", "0.22"])
    ax.tick_params(labelsize=11)
    ax.set_xlabel("θ", fontsize=12, style="italic", labelpad=1)
    save(fig, "mcmc_histogram")
    print(f"    counted mean {m['mean_counted']:.4f} vs exact {exact_mean:.4f}; P(theta > 0.15) "
          f"counted {m['p_above_counted']:.3f} vs exact {exact_p:.3f}; 94% HDI counted "
          f"{m['hdi94_counted'][0]:.3f}-{m['hdi94_counted'][1]:.3f}; Metropolis ESS "
          f"{m['ess_bulk']:.0f}, R-hat {m['r_hat']:.3f}")


# ================================================================== priors by support
def priors_support() -> None:
    """Each prior over the values its parameter can take; then N(0, 1) on a sigma, with its
    impossible half shaded, folded onto sigma >= 0 as the HalfNormal."""
    fig = plt.figure(figsize=sz(13.6, 3.7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.75], wspace=0.13,
                          left=0.01, right=0.99, top=0.78, bottom=0.2)

    def panel(i, title, sub):
        ax = fig.add_subplot(gs[0, i])
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.tick_params(labelsize=11)
        ax.set_title(f"{title}\n", fontsize=12, pad=2, loc="center")
        ax.text(0.5, 1.03, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=11,
                color=GRAY)
        return ax

    ax = panel(0, "Beta(5, 45)", "rates: [0, 1]")
    x = np.linspace(0, 1, 800)
    ax.fill_between(x, beta_dist.pdf(x, A0, B0), color=LIGHT_BLUE)
    ax.plot(x, beta_dist.pdf(x, A0, B0), color=BLUE, lw=2.0)
    ax.set_xlim(0, 1); ax.set_ylim(0, None)
    ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])

    ax = panel(1, "Normal(0, 1)", "coefficients: all reals")
    x = np.linspace(-3.5, 3.5, 800)
    ax.fill_between(x, norm.pdf(x), color=LIGHT_BLUE)
    ax.plot(x, norm.pdf(x), color=BLUE, lw=2.0)
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(0, 0.5)
    ax.set_xticks([-3, 0, 3])

    ax = panel(2, "Exponential(1)", "scales: [0, ∞)")
    x = np.linspace(0, 4.5, 800)
    ax.fill_between(x, expon.pdf(x), color=LIGHT_BLUE)
    ax.plot(x, expon.pdf(x), color=BLUE, lw=2.0)
    ax.set_xlim(-0.6, 4.5); ax.set_ylim(0, 1.15)
    ax.set_xticks([0, 2, 4])

    # ---- a sigma: the wrong prior, and the fold that fixes it
    ax = panel(3, "a standard deviation σ", "N(0, 1) vs HalfNormal(1)")
    xn, xp = np.linspace(-3.2, 0, 400), np.linspace(0, 3.2, 400)
    neg_mass = float(norm.cdf(0))
    assert neg_mass == 0.5 and np.allclose(halfnorm.pdf(xp), 2 * norm.pdf(xp))   # the fold, exactly
    ax.fill_between(xn, norm.pdf(xn), color=LIGHT_CRIMSON, zorder=1)
    ax.plot(np.r_[xn, xp], norm.pdf(np.r_[xn, xp]), color=GRAY, lw=1.6, ls=(0, (4, 2)), zorder=3)
    ax.plot(xp, halfnorm.pdf(xp), color=BLUE, lw=2.2, zorder=4)
    ax.axvline(0, color=BLACK, lw=0.8, zorder=2)
    ax.text(-3.2, 0.99, f"σ < 0:\nimpossible, yet\n{neg_mass:.0%} of N(0, 1)",
            ha="left", va="top", fontsize=11, color=CRIMSON)
    ax.text(0.95, 0.99, "HalfNormal:\nfolded onto\nσ ≥ 0", ha="left", va="top",
            fontsize=11, color=BLUE)
    ax.annotate("", xy=(1.05, 0.05), xytext=(-1.05, 0.05),     # the fold, under both curves
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.1,
                                connectionstyle="arc3,rad=-0.3", mutation_scale=11))
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(0, 1.0)
    ax.set_xticks([-3, 0, 3])
    save(fig, "priors_support")
    print(f"    N(0,1) mass below 0: {neg_mass:.2f}; HalfNormal(1) = 2 x N(0,1) density on sigma >= 0")


# ================================================================== prior predictive counts
PRIORS_PP = [((1, 1), "flat Beta(1, 1)"), ((5, 45), "Beta(5, 45)"), ((500, 4500), "Beta(500, 4500)")]


def prior_predictive() -> None:
    """What each prior predicts for successes out of 200 before any data: the exact Beta-Binomial
    probabilities that pm.sample_prior_predictive approximates by simulation."""
    k = np.arange(N + 1)
    fig, axes = plt.subplots(3, 1, figsize=sz(12.4, 4.6), sharex=True)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.98, bottom=0.17, hspace=0.18)
    YMAX = 0.1
    notes = []
    for ax, ((a, b), name), col in zip(axes, PRIORS_PP, (GRAY, BLUE, CRIMSON)):
        d = betabinom(N, a, b)
        lo, hi = int(d.ppf(0.05)), int(d.ppf(0.95))
        notes.append((name, lo, hi, float(d.pmf(k).max())))
        ax.bar(k, d.pmf(k), width=1.0, color=col, edgecolor="none", alpha=0.9)
        ax.set_ylim(0, YMAX); ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.text(1.0, 0.92, name, transform=ax.transAxes, ha="right", va="top", fontsize=12,
                color=col)
        ax.tick_params(labelsize=11)
    assert notes[0][3] == 1 / 201                                    # flat: every count 1 in 201
    (_, lo1, hi1, _), (_, lo2, hi2, _) = notes[1], notes[2]
    assert (lo1, hi1, lo2, hi2) == (7, 37, 13, 27)
    axes[0].text(1.0, 0.45, "every count from 0 to 200 equally likely", transform=axes[0].transAxes,
                 ha="right", va="top", fontsize=11, color=GRAY)
    axes[1].text(1.0, 0.45, f"90% between {lo1} and {hi1}", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=11, color=BLUE)
    axes[2].text(1.0, 0.45, f"90% between {lo2} and {hi2}: θ is all but fixed at 0.10",
                 transform=axes[2].transAxes, ha="right", va="top", fontsize=11, color=CRIMSON)
    axes[-1].set_xlim(-1, N + 1)
    axes[-1].set_xticks([0, 50, 100, 150, 200])
    axes[-1].set_xlabel("successes out of 200, predicted before any data", fontsize=12, labelpad=2)
    save(fig, "prior_predictive")
    for name, lo, hi, mx in notes:
        print(f"    {name}: 90% {lo}-{hi}, tallest bar {mx:.4f}")


# ================================================================== Bayes' theorem as a mosaic
# Slides/Lecture10_BayesianFoundations.tex, "Bayes' Theorem, Drawn". CA_1 FOODS weekly, as Lab 10 loads
# it. theta: a high-SNAP week, snap_days >= 5 (Lab 10's definition). D: a "strong" week, one whose
# units are above its own calendar year's median (the owner's definition, 2026-10-18). 2011 and 2016
# are partial years (from 31 Jan; to mid-May), so their medians are over fewer weeks.
def snap_weeks():
    import pandas as pd
    w = pd.read_csv(ROOT / "data" / "processed" / "m5_weekly.csv", parse_dates=["week_start"])
    c = w[(w.store_id == "CA_1") & (w.cat_id == "FOODS")].sort_values("week_start").copy()
    c["snap"] = c.snap_days >= 5
    c["year"] = c.week_start.dt.year
    c["strong"] = c.units > c.groupby("year").units.transform("median")
    return c


def bayes_mosaic() -> None:
    """A unit square split left/right by P(SNAP), then each column by P(strong | column): areas
    are joint probabilities. Below it, the strong area laid end to end, read the other way."""
    c = snap_weeks()
    n = len(c)
    n_snap = int(c.snap.sum())
    n_ss = int((c.snap & c.strong).sum())              # SNAP and strong
    n_os = int((~c.snap & c.strong).sum())             # other week, strong
    n_strong = n_ss + n_os
    p_snap, lik, lik_other = n_snap / n, n_ss / n_snap, n_os / (n - n_snap)
    evidence, post = n_strong / n, n_ss / n_strong
    # The numbers the slide prints, checked before anything is drawn.
    assert (n, n_snap, n_ss, n_os) == (277, 72, 55, 82)
    assert (round(p_snap, 2), round(lik, 2), round(lik_other, 2)) == (0.26, 0.76, 0.40)
    assert (round(evidence, 2), round(post, 2)) == (0.49, 0.40)
    assert abs(post - lik * p_snap / evidence) < 1e-12                       # Bayes, exactly
    assert abs(evidence - (lik * p_snap + lik_other * (1 - p_snap))) < 1e-12  # the two strong areas
    assert round(0.76 * 0.26 / 0.49, 2) == 0.40        # the slide's rounded arithmetic still lands
    # every year, partial or not, SNAP weeks were strong more often than other weeks
    by_year = c.groupby("year").apply(lambda g: (g[g.snap].strong.mean(), g[~g.snap].strong.mean()))
    assert all(s > o for s, o in by_year)
    # the narration's aside: "beats the same week last year" makes SNAP nearly uninformative,
    # because the same week last year was a SNAP week too
    c["snap_l52"] = c.snap.shift(52)
    d = c[c.units_lag52.notna()]
    beat = d.units > d.units_lag52
    beat_snap, beat_other = beat[d.snap].mean(), beat[~d.snap].mean()
    l52_snap, l52_other = d[d.snap].snap_l52.mean(), d[~d.snap].snap_l52.mean()
    assert (len(d), round(beat_snap, 2), round(beat_other, 2)) == (225, 0.58, 0.57)

    DARK, MID, PALE = CRIMSON, "#F4B4B6", "#E4E4E7"
    fig, ax = plt.subplots(figsize=sz(4.95, 5.35))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.27, 1.20); ax.set_ylim(-0.35, 1.24)
    gap = dict(edgecolor="white", lw=2.0)
    ax.add_patch(Rectangle((0, 0), p_snap, lik, facecolor=DARK, **gap))
    ax.add_patch(Rectangle((0, lik), p_snap, 1 - lik, facecolor=PALE, **gap))
    ax.add_patch(Rectangle((p_snap, 0), 1 - p_snap, lik_other, facecolor=MID, **gap))
    ax.add_patch(Rectangle((p_snap, lik_other), 1 - p_snap, 1 - lik_other, facecolor=PALE, **gap))
    # counts of weeks, the areas' numerators (every area is count / 277)
    ax.text(p_snap / 2, lik / 2, f"{n_ss}", ha="center", va="center", fontsize=13, color="white",
            fontweight="bold")
    ax.text(p_snap / 2, (1 + lik) / 2, f"{n_snap - n_ss}", ha="center", va="center", fontsize=12,
            color=BLACK)
    xo = (1 + p_snap) / 2
    ax.text(xo, lik_other / 2, f"{n_os} strong", ha="center", va="center", fontsize=12,
            color=BLACK)
    ax.text(xo, (1 + lik_other) / 2, f"{n - n_snap - n_os} not strong", ha="center", va="center",
            fontsize=12, color=GRAY)
    # prior: the SNAP column's width (a bracket above it); the rest: the other weeks
    yb = 1.035
    b0, b1 = p_snap - 0.012, p_snap + 0.012             # a small gap between the two brackets
    ax.plot([0, 0, b0, b0], [yb, yb + 0.03, yb + 0.03, yb], color=BLUE, lw=1.4,
            solid_capstyle="butt")
    ax.text(p_snap / 2, yb + 0.06, f"SNAP\n{p_snap:.2f}", ha="center", va="bottom", fontsize=12,
            color=BLUE, linespacing=1.1)
    ax.plot([b1, b1, 1, 1], [yb, yb + 0.03, yb + 0.03, yb], color=GRAY, lw=1.4,
            solid_capstyle="butt")
    ax.text(xo, yb + 0.06, f"other weeks\n{1 - p_snap:.2f}", ha="center", va="bottom",
            fontsize=12, color=GRAY, linespacing=1.1)
    # likelihoods: the strong height in each column
    xl = -0.045
    ax.annotate("", xy=(xl, 0), xytext=(xl, lik),
                arrowprops=dict(arrowstyle="<|-|>", color=AMBER, lw=1.3, mutation_scale=9,
                                shrinkA=0, shrinkB=0))
    ax.text(xl - 0.025, lik / 2, f"{lik:.2f}", ha="right", va="center", fontsize=12, color=AMBER)
    xr = 1.045
    ax.annotate("", xy=(xr, 0), xytext=(xr, lik_other),
                arrowprops=dict(arrowstyle="<|-|>", color=GRAY, lw=1.3, mutation_scale=9,
                                shrinkA=0, shrinkB=0))
    ax.text(xr + 0.025, lik_other / 2, f"{lik_other:.2f}", ha="left", va="center", fontsize=12,
            color=GRAY)
    # the other way: the two strong areas end to end (length = area, so the shares are exact)
    ys, hs = -0.30, 0.13
    ax.add_patch(Rectangle((0, ys), n_ss / n, hs, facecolor=DARK, **gap))
    ax.add_patch(Rectangle((n_ss / n, ys), n_os / n, hs, facecolor=MID, **gap))
    ax.text(n_ss / n / 2, ys + hs / 2, f"{n_ss}", ha="center", va="center", fontsize=12,
            color="white", fontweight="bold")
    ax.text(n_ss / n + n_os / n / 2, ys + hs / 2, f"{n_os}", ha="center", va="center", fontsize=12,
            color=BLACK)
    ax.text(0, ys + hs + 0.035, "all strong weeks, end to end", ha="left", va="bottom",
            fontsize=11, color=GRAY)
    ax.text(evidence + 0.04, ys + hs / 2, f"{n_ss} of {n_strong}:\n{post:.2f} SNAP", ha="left",
            va="center", fontsize=12, color=CRIMSON, linespacing=1.1)
    save(fig, "bayes_mosaic")
    print(f"    {n} weeks, {n_snap} SNAP: P(SNAP) {p_snap:.4f}; P(strong|SNAP) {lik:.4f}, "
          f"P(strong|other) {lik_other:.4f}; P(strong) {evidence:.4f}; P(SNAP|strong) {post:.4f}")
    print("    by year, P(strong|SNAP) vs P(strong|other): "
          + ", ".join(f"{y} {s:.2f}/{o:.2f}" for y, (s, o) in by_year.items()))
    print(f"    aside, 'beats the same week last year' ({len(d)} weeks): P(beat|SNAP) {beat_snap:.3f} "
          f"vs P(beat|other) {beat_other:.3f}; last year's week was SNAP {l52_snap:.2f} vs "
          f"{l52_other:.2f}")


FIGURES = {"bayes_update": bayes_update, "bayes_mosaic": bayes_mosaic, "mcmc_walk": mcmc_walk,
           "mcmc_histogram": mcmc_histogram, "priors_support": priors_support,
           "prior_predictive": prior_predictive}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if not a.startswith("--")] or list(FIGURES)
    print(f"Lecture 10 figures -> {OUT.relative_to(ROOT)}   (font: {FONT})")
    if "--retrain" in sys.argv:
        measured()
    for name in pick:
        FIGURES[name]()
