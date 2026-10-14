#!/usr/bin/env python3
"""Figures for Lecture 10 (Bayesian Statistics I: Foundations) — one script, every figure, both formats.

    .venv/bin/python scripts/figures/lecture10_figures.py                 # all
    .venv/bin/python scripts/figures/lecture10_figures.py bayes_update    # one, by name

Writes Figures/Lecture10_BayesianFoundations/<name>.pdf (Beamer) and <name>.svg (Quarto).

Nothing here is sampled. The deck's conversion-rate example is conjugate, so every curve and every
number is exact: a Beta(5, 45) prior and 30 successes in 200 trials give a Beta(35, 215)
posterior. The script asserts the deck's printed numbers (0.10, 0.15, 0.14) before drawing them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon, Rectangle
import numpy as np
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Figures" / "Lecture10_BayesianFoundations"

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


FIGURES = {"bayes_update": bayes_update}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if not a.startswith("--")] or list(FIGURES)
    print(f"Lecture 10 figures -> {OUT.relative_to(ROOT)}   (font: {FONT})")
    for name in pick:
        FIGURES[name]()
