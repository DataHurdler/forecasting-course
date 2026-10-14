#!/usr/bin/env python3
"""Figures for Lecture 9 Part 2 (Attention and Transformers) — one script, every figure, both formats.

    .venv/bin/python scripts/figures/lecture09_part2_figures.py                     # all
    .venv/bin/python scripts/figures/lecture09_part2_figures.py attention_heatmap   # one, by name
    .venv/bin/python scripts/figures/lecture09_part2_figures.py --retrain           # re-measure

Writes Figures/Lecture09_Part2/<name>.pdf (Beamer) and <name>.svg (Quarto), both vector.

The attention heatmap is measured on the trained model. Training uses the lab's own data pipeline,
model class and loop (Labs/Lecture09_Part2_lab.qmd, Setup and Step 2), so the figure reads the
lab's model, not a second one. Training takes about two and a half minutes on a laptop CPU; the
results go to Figures/Lecture09_Part2/measured.json, and later runs redraw from that file. The
other three figures are exact calculations or diagrams and need no training.
"""
from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Figures" / "Lecture09_Part2"
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
# Labs/Lecture09_Part2_lab.qmd, Setup + Steps 1-2. Kept identical on purpose: the seeds, the
# initialisation order inside TransformerForecaster and the unshuffled loader decide every number.
CHANNELS, WINDOW = ["units", "snap_days", "event_days", "avg_price"], 52
CUTOFF = np.datetime64("2015-05-17")
SEEDS = (1, 2)
D_MODEL = 64


def sinusoidal(seq_len, d_model):
    import torch
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(seq_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


@lru_cache(maxsize=None)
def data():
    import torch
    weekly = (pd.read_csv(ROOT / "data" / "processed" / "m5_weekly.csv", parse_dates=["week_start"])
                .sort_values(["store_id", "cat_id", "week_start"]))
    X, Y, D, S = [], [], [], []
    for key, g in weekly.groupby(["store_id", "cat_id"]):
        arr, dates = g[CHANNELS].values.astype("float32"), g.week_start.values
        for i in range(len(g) - WINDOW):
            X.append(arr[i:i+WINDOW]); Y.append(arr[i+WINDOW, 0]); D.append(dates[i+WINDOW])
            S.append("_".join(key))
    X = np.stack(X); Y = np.array(Y, dtype="float32"); D = np.array(D); S = np.array(S)
    tr, te = D < CUTOFF, D >= CUTOFF
    mu, sd = X[tr].mean((0, 1), keepdims=True), X[tr].std((0, 1), keepdims=True) + 1e-8
    ym, ys = Y[tr].mean(), Y[tr].std()
    Xn, Yn = (X - mu)/sd, (Y - ym)/ys
    return dict(X_tr=torch.tensor(Xn[tr]), y_tr=torch.tensor(Yn[tr]), X_te=torch.tensor(Xn[te]),
                Y_te=Y[te], D_te=D[te], S_te=S[te], ym=ym, ys=ys, n_train=int(tr.sum()),
                n_windows=int(len(X)))


def _model(use_pe=True):
    import torch.nn as nn

    class TransformerForecaster(nn.Module):
        def __init__(self, n_channels=4, d_model=64, nhead=4, layers=2, use_pe=True):
            super().__init__()
            self.use_pe = use_pe
            self.inp = nn.Linear(n_channels, d_model)
            self.register_buffer("pe", sinusoidal(WINDOW, d_model))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=128,
                dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.fc = nn.Linear(d_model, 1)

        def forward(self, x):
            h = self.inp(x)
            if self.use_pe:
                h = h + self.pe
            return self.fc(self.enc(h)[:, -1, :]).squeeze(-1)

    return TransformerForecaster(use_pe=use_pe)


def fit(seed: int, epochs: int = 30):
    """The lab's train(use_pe=True, seed): returns the trained model and its test RMSE."""
    import torch
    import torch.nn as nn
    d = data()
    torch.manual_seed(seed)
    model = _model(use_pe=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossfn = nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(d["X_tr"], d["y_tr"]), batch_size=128, shuffle=False)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            lossfn(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(d["X_te"]).numpy() * d["ys"] + d["ym"]
    return model, float(np.sqrt(np.mean((d["Y_te"] - pred) ** 2)))


# ================================================================== measurement
def attention_from_last(model, X):
    """Each layer's attention weights from the LAST position — the one the forecast is read from.

    nn.TransformerEncoderLayer calls its attention with need_weights=False, so the forward pass is
    rebuilt from the layer's own modules (post-norm, the PyTorch default) with the weights kept.
    The rebuilt forecast is checked against model(X) before anything is read from it.
    Returns an array (layers, windows, heads, 52); column j is the input week j of the window,
    so column 0 is 52 weeks before the forecast week and column 51 is 1 week before.
    """
    import torch
    with torch.no_grad():
        h = model.inp(X) + model.pe
        maps = []
        for layer in model.enc.layers:
            a, w = layer.self_attn(h, h, h, need_weights=True, average_attn_weights=False)
            h = layer.norm1(h + layer.dropout1(a))
            h = layer.norm2(h + layer._ff_block(h))
            maps.append(w[:, :, -1, :].numpy())
        out = model.fc(h[:, -1, :]).squeeze(-1)
        assert torch.allclose(out, model(X), atol=1e-3), "rebuilt encoder disagrees with model(x)"
    return np.stack(maps)


def measure() -> dict:
    d = data()
    m: dict = {"note": "Written by scripts/figures/lecture09_part2_figures.py; every number the "
                       "Lecture 9 Part 2 attention figure prints. Seeds 1 and 2, as in the lab.",
               "windows": d["n_windows"], "train_windows": d["n_train"]}
    dec = (pd.DatetimeIndex(d["D_te"]).month == 12) & (pd.DatetimeIndex(d["D_te"]).year == 2015)
    rmse, attn_all, attn_dec = [], [], []
    for s in SEEDS:
        model, r = fit(s)
        rmse.append(r)
        A = attention_from_last(model, d["X_te"])          # (layers, windows, heads, 52)
        attn_all.append(A.mean(axis=1))                    # (layers, heads, 52)
        attn_dec.append(A[:, dec].mean(axis=(1, 2)))       # (layers, 52), heads pooled
        print(f"  seed {s}: test RMSE {r:,.0f}")
    m["test_rmse"] = rmse
    m["attention_all"] = np.array(attn_all).tolist()       # (seeds, layers, heads, 52)
    m["attention_december_2015"] = np.array(attn_dec).tolist()
    m["december_windows"] = int(dec.sum())
    OUT.mkdir(parents=True, exist_ok=True)
    MEASURED.write_text(json.dumps(m) + "\n")
    print(f"  measured.json written: RMSE {np.mean(rmse):,.0f} (seeds {[round(x) for x in rmse]})")
    return m


@lru_cache(maxsize=None)
def measured() -> dict:
    if "--retrain" in sys.argv or not MEASURED.exists():
        return measure()
    return json.loads(MEASURED.read_text())


# ================================================================== attention, geometrically
def attention_geometry() -> None:
    """One query, five keys: score = |q| x shadow; softmax weights; output inside the hull.

    A two-dimensional cartoon of what each 16-dimensional head does. The numbers are invented for
    the picture, but every relation drawn is exact for them: the query lies along the x-axis, so a
    key's shadow on it is just the key's x-coordinate; the bars are the softmax of the scores; and
    the star is the weighted average of the five value points.
    """
    names = ["t−1", "t−2", "t−4", "t−26", "t−52"]
    q = np.array([1.8, 0.0])
    K = np.array([(1.5, 0.75), (1.05, 1.3), (0.75, -0.85), (-0.15, 1.35), (-1.0, -0.6)])
    V = np.array([(2.5, 2.25), (1.0, 2.95), (3.25, 0.85), (0.25, 1.35), (1.45, 0.2)])
    score = K @ q
    w = np.exp(0.9 * score); w /= w.sum()          # 0.9 stands in for 1/sqrt(d_k)
    out = w @ V
    pos = [BLUE if sc > 0 else AMBER for sc in score]

    fig = plt.figure(figsize=sz(13, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.85, 1.0], wspace=0.42, top=0.84, bottom=0.08)

    # (a) keys, and where each one's shadow lands on the query line
    ax = fig.add_subplot(gs[0]); ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.45, 2.25); ax.set_ylim(-1.15, 1.75)
    ax.plot([-1.35, 2.2], [0, 0], color=GRAY, lw=0.8, zorder=1)
    for n, k, c in zip(names, K, pos):
        ax.plot([k[0], k[0]], [k[1], 0], color=GRAY, lw=0.9, ls=(0, (3, 2)), zorder=1)
        ax.annotate("", xy=k, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.4, mutation_scale=11))
        ax.plot([k[0]], [0], "o", ms=6, color=c, zorder=4)
        dy = 0.17 if k[1] > 0 else -0.17
        ax.text(k[0], k[1] + dy, n, ha="center", va="bottom" if k[1] > 0 else "top", fontsize=11,
                color=GRAY)
    ax.annotate("", xy=q, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=CRIMSON, lw=2.6, mutation_scale=15),
                zorder=3)
    ax.text(q[0], -0.2, "query", ha="center", va="top", fontsize=12, color=CRIMSON,
            fontweight="bold")
    axa = ax

    # (b) the softmax weights
    ax = fig.add_subplot(gs[1])
    y = np.arange(len(names))[::-1]
    ax.barh(y, w, color=pos, height=0.62)
    for yy, ww in zip(y, w):
        ax.text(ww + 0.015, yy, f"{ww:.2f}", ha="left", va="center", fontsize=11)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=11)
    ax.set_xlim(0, max(w) * 1.45); ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    axb = ax

    # (c) the output: a weighted average of the values, so inside their hull
    ax = fig.add_subplot(gs[2]); ax.set_aspect("equal"); ax.axis("off")
    from scipy.spatial import ConvexHull
    hull = V[ConvexHull(V).vertices]
    ax.add_patch(Polygon(hull, closed=True, facecolor=LIGHT_BLUE, edgecolor=BLUE, lw=1.2))
    for n, v in zip(names, V):
        ax.plot(*v, "o", ms=5, color=GRAY, zorder=3)
    lab = {"t−1": (0.2, 0.18, "left", "bottom"), "t−2": (0.0, 0.17, "center", "bottom"),
           "t−4": (0.15, -0.05, "left", "top"), "t−26": (-0.15, 0.0, "right", "center"),
           "t−52": (0.15, -0.05, "left", "top")}
    for n, v in zip(names, V):
        dx, dy, ha, va = lab[n]
        ax.text(v[0] + dx, v[1] + dy, n, ha=ha, va=va, fontsize=11, color=GRAY)
    ax.plot(*out, marker="*", ms=16, color=CRIMSON, zorder=4)
    ax.text(out[0] - 0.12, out[1] - 0.2, "output", ha="right", va="top", fontsize=12,
            color=CRIMSON, fontweight="bold")
    ax.set_xlim(-0.55, 3.9); ax.set_ylim(-0.25, 3.45)
    axc = ax
    # one baseline for all three titles: equal-aspect panels sit at different heights
    for a_, title in [(axa, "score: each key's shadow\non the query"),
                      (axb, "softmax weights:\npositive, sum to 1"),
                      (axc, "output: a blend,\ninside the hull")]:
        bb = a_.get_position()
        fig.text((bb.x0 + bb.x1) / 2, 0.86, title, ha="center", va="bottom", fontsize=12)
    fig.text(0.5, 0.04, "a two-dimensional cartoon of one 16-dimensional head", ha="center",
             va="top", fontsize=10, color=GRAY)
    save(fig, "attention_geometry")


# ================================================================== where attention lands
def attention_heatmap() -> None:
    """Both training runs, both layers, all four heads: weight from the forecasting position."""
    from matplotlib.colors import PowerNorm
    m = measured()
    A = np.array(m["attention_all"])                     # (seeds, layers, heads, 52)
    fig = plt.figure(figsize=sz(13, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045], wspace=0.08)
    norm = PowerNorm(0.6, vmin=0, vmax=0.2)
    for si, seed in enumerate(SEEDS):
        ax = fig.add_subplot(gs[si])
        M = A[si].reshape(8, 52)                         # rows: layer 1 heads 1-4, layer 2 heads 1-4
        im = ax.imshow(M, aspect="auto", cmap="Reds", norm=norm, interpolation="nearest",
                       extent=(52.5, 0.5, 8.5, 0.5))
        ax.axhline(4.5, color="white", lw=2.5)
        ax.set_xticks([52, 26, 4, 1]); ax.set_xticklabels(["52", "26", "4", "1"], fontsize=11)
        ax.set_xlabel("weeks before the forecast week", fontsize=11)
        if si == 0:
            ax.set_yticks([2.5, 6.5]); ax.set_yticklabels(["layer 1\n4 heads", "layer 2\n4 heads"],
                                                          fontsize=11)
        else:
            ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.add_patch(plt.Rectangle((51.5, 0.5), 1, 8, fill=False, edgecolor=BLUE, lw=1.6))
        ax.set_title(f"training run {si + 1}  (test RMSE {m['test_rmse'][si]:,.0f})", fontsize=12,
                     pad=6)
    # call out the one head that found last year
    s2 = A[1].reshape(8, 52)
    r, c = np.unravel_index(np.argmax(s2[:, 0]), (8,)), 0
    ax.annotate(f"{s2[r[0], 0]:.2f} on the same week\nlast year: {s2[r[0], 0] * 52:.0f}× an even share",
                xy=(52, r[0] + 1), xytext=(40, r[0] + 3.4), fontsize=11, color=BLUE, ha="left",
                va="center", arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.3),
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5))
    cax = fig.add_subplot(gs[2])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 1 / 52, 0.1, 0.2])
    cb.ax.set_yticklabels(["0", "1/52", "0.1", "0.2"], fontsize=10)
    cb.outline.set_visible(False)
    save(fig, "attention_heatmap")


# ================================================================== the encoder block
def encoder_block() -> None:
    """Attention, skip, LayerNorm, feed-forward, skip, LayerNorm: only the first box is new."""
    fig, ax = plt.subplots(figsize=sz(15.5, 3.9))
    ax.set_xlim(-0.2, 21.9); ax.set_ylim(-2.15, 2.35); ax.axis("off")
    BH, Y = 1.15, 0.0

    def box(xc, w, label, face, edge, bold=False):
        ax.add_patch(FancyBboxPatch((xc - w/2, Y - BH/2), w, BH,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=face, edgecolor=edge, lw=1.8))
        ax.text(xc, Y, label, ha="center", va="center", fontsize=12,
                fontweight="bold" if bold else "normal")
        return xc - w/2, xc + w/2

    def arrow(x0, x1, color=BLACK):
        ax.annotate("", xy=(x1, Y), xytext=(x0, Y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, mutation_scale=14))

    def plus(x):
        ax.add_patch(Circle((x, Y), 0.32, facecolor="white", edgecolor=BLACK, lw=1.8, zorder=3))
        ax.text(x, Y + 0.02, "+", ha="center", va="center", fontsize=15, fontweight="bold", zorder=4)

    def skip(x_from, x_to, label):
        SY = 1.35
        ax.plot([x_from, x_from, x_to, x_to], [Y + 0.05, SY, SY, Y + 0.34], color=GRAY, lw=2.0,
                solid_joinstyle="round")
        ax.annotate("", xy=(x_to, Y + 0.34), xytext=(x_to, Y + 0.8),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2.0, mutation_scale=14))
        ax.plot([x_from], [Y], "o", ms=6, color=GRAY, zorder=5)
        ax.text((x_from + x_to) / 2, SY + 0.15, label, ha="center", va="bottom", fontsize=11,
                color=GRAY)

    ax.text(0.0, Y, "week\n+ position", ha="left", va="center", fontsize=11)
    a0, a1 = box(4.5, 3.0, "multi-head\nattention", LIGHT_CRIMSON, CRIMSON, bold=True)
    P1 = 6.9
    n0, n1 = box(9.0, 2.7, "LayerNorm", LIGHT_GRAY, GRAY)
    f0, f1 = box(12.75, 3.3, "feed-forward", LIGHT_GRAY, GRAY)
    P2 = 15.3
    m0, m1 = box(17.4, 2.7, "LayerNorm", LIGHT_GRAY, GRAY)
    arrow(2.35, a0 - 0.04); arrow(a1 + 0.04, P1 - 0.34); arrow(P1 + 0.34, n0 - 0.04)
    arrow(n1 + 0.04, f0 - 0.04); arrow(f1 + 0.04, P2 - 0.34); arrow(P2 + 0.34, m0 - 0.04)
    arrow(m1 + 0.04, 19.85)
    ax.text(20.0, Y, "next\nlayer", ha="left", va="center", fontsize=11)
    plus(P1); plus(P2)
    skip(2.6, P1, "skip connection (Lecture 8)")
    skip((n1 + f0) / 2, P2, "skip connection")

    ax.text(4.5, Y - BH/2 - 0.15, "the only new part", ha="center", va="top", fontsize=11,
            color=CRIMSON, fontweight="bold")
    ax.text(12.75, Y - BH/2 - 0.15, "Lecture 7's network,\nat every week", ha="center", va="top",
            fontsize=11, color=GRAY)
    bx0, bx1, by = 2.45, 18.85, -1.9
    ax.plot([bx0, bx0, bx1, bx1], [by + 0.15, by, by, by + 0.15], color=GRAY, lw=1.2)
    ax.text(bx1 + 0.2, by + 0.02, "× 2 in our model", ha="left", va="center", fontsize=11,
            color=GRAY)
    save(fig, "encoder_block")


# ================================================================== positional encoding
def positional_clocks() -> None:
    """Each sin/cos pair is a clock hand; together they make week-to-week similarity a function
    of distance only."""
    T, d = WINDOW, D_MODEL
    t = np.arange(T)
    omega = 10000.0 ** (-np.arange(0, d, 2) / d)
    pe = np.zeros((T, d)); pe[:, 0::2] = np.sin(t[:, None] * omega); pe[:, 1::2] = np.cos(t[:, None] * omega)
    sim = pe @ pe.T / (d // 2)
    lag = np.abs(np.subtract.outer(t, t))
    assert np.allclose(sim, sim[0][lag]), "similarity should depend on |t - t'| only"

    fig = plt.figure(figsize=sz(13, 4.3))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 0.3, 1.45], wspace=0.22)
    cmap = plt.get_cmap("Blues")
    for col, i in enumerate((0, 4, 8, 16)):
        ax = fig.add_subplot(gs[col]); ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
        ax.add_patch(Circle((0, 0), 1, fill=False, edgecolor=LIGHT_GRAY, lw=4))
        ang = omega[i] * t
        ax.scatter(np.cos(ang), np.sin(ang), c=t, cmap=cmap, vmin=-12, vmax=T - 1, s=16, zorder=3)
        ax.plot([0, np.cos(ang[-1])], [0, np.sin(ang[-1])], color=CRIMSON, lw=2.2, zorder=4)
        ax.plot([0], [0], "o", ms=4, color=CRIMSON, zorder=5)
        period = 2 * np.pi / omega[i]
        ax.text(0, -1.5, f"1 turn per\n{period:,.0f} weeks", ha="center", va="top", fontsize=11)
        ax.set_title(f"pair {i + 1}", fontsize=12, pad=2)
    ax = fig.add_subplot(gs[5])
    im = ax.imshow(sim, cmap="Blues", vmin=0.4, vmax=1.0, origin="upper", extent=(0.5, T + 0.5, T + 0.5, 0.5))
    ax.set_xticks([1, 26, 52]); ax.set_yticks([1, 26, 52])
    ax.tick_params(labelsize=10)
    ax.set_xlabel("week t′ in the window", fontsize=11)
    ax.set_ylabel("week t", fontsize=11, labelpad=6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("similarity of two weeks", fontsize=12, pad=4)
    ax.text(T + 3, T / 2, "same along\nevery diagonal:\nonly the distance\nmatters", ha="left",
            va="center", fontsize=11, color=GRAY)
    save(fig, "positional_clocks")


FIGURES = {"attention_geometry": attention_geometry, "attention_heatmap": attention_heatmap,
           "encoder_block": encoder_block, "positional_clocks": positional_clocks}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if not a.startswith("--")] or list(FIGURES)
    print(f"Lecture 9 Part 2 figures -> {OUT.relative_to(ROOT)}   (font: {FONT})")
    if "--retrain" in sys.argv:
        measured()
    for name in pick:
        FIGURES[name]()
