#!/usr/bin/env python3
"""Figures for Lecture 9 Part 1 (RNNs and LSTMs) — one script, every figure, both formats.

    .venv/bin/python scripts/figures/lecture09_part1_figures.py                  # all
    .venv/bin/python scripts/figures/lecture09_part1_figures.py gradient_decay   # one, by name
    .venv/bin/python scripts/figures/lecture09_part1_figures.py --retrain        # re-measure

Writes Figures/Lecture09_Part1/<name>.pdf (Beamer) and <name>.svg (Quarto), both vector.

Three of the four figures print numbers measured on trained networks. Training uses the lab's
own data pipeline, model class and training loop (Labs/Lecture09_Part1_lab.qmd, Setup and
Steps 1-4), so the figures print the lab's numbers rather than a second set. Training takes about
three minutes on a laptop CPU. The results go to Figures/Lecture09_Part1/measured.json, which
records every number the figures print. Later runs redraw from that file; pass --retrain to
measure again.
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
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Figures" / "Lecture09_Part1"
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


# SVG text is drawn as outlines ("path"), not left as live text. Live text renders in whatever
# fonts the viewer has: on a machine without Helvetica Neue it falls back to Times, and the math
# glyphs (DejaVu, cmsy10, STIX private-use code points) turn into a "0" for a prime or a blank box.
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
# Labs/Lecture09_Part1_lab.qmd, Setup + Step 1 + Step 4. Kept identical on purpose: the seeds,
# the initialisation order inside Recurrent, and the unshuffled loader decide every number.
CHANNELS = ["units", "snap_days", "event_days", "avg_price"]
CUTOFF = np.datetime64("2015-05-17")
CKPT = [10, 20, 30, 45, 60, 80, 100]
SEEDS = (1, 2)


@lru_cache(maxsize=None)
def weekly() -> pd.DataFrame:
    return (pd.read_csv(ROOT / "data" / "processed" / "m5_weekly.csv", parse_dates=["week_start"])
              .sort_values(["store_id", "cat_id", "week_start"]))


@lru_cache(maxsize=None)
def build(W: int):
    import torch
    X, Y, D = [], [], []
    for _, g in weekly().groupby(["store_id", "cat_id"]):
        arr, dates = g[CHANNELS].values.astype("float32"), g.week_start.values
        for i in range(len(g) - W):
            X.append(arr[i:i+W]); Y.append(arr[i+W, 0]); D.append(dates[i+W])
    X = np.stack(X); Y = np.array(Y, dtype="float32"); D = np.array(D)
    tr, te = D < CUTOFF, D >= CUTOFF
    mu, sd = X[tr].mean((0, 1), keepdims=True), X[tr].std((0, 1), keepdims=True) + 1e-8
    ym, ys = Y[tr].mean(), Y[tr].std()
    Xn, Yn = (X - mu)/sd, (Y - ym)/ys
    return (torch.tensor(Xn[tr]), torch.tensor(Yn[tr]), torch.tensor(Xn[te]), Y[te], ym, ys)


def _recurrent(kind: str):
    import torch.nn as nn

    class Recurrent(nn.Module):
        def __init__(self, kind, n_channels=4, hidden=64):
            super().__init__()
            layer = nn.RNN if kind == "rnn" else nn.LSTM
            self.rnn = layer(n_channels, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    return Recurrent(kind)


def fit(kind: str, W: int, seed: int, epochs: int, ckpt=()):
    """The lab's train()/curve(): returns the model and {epoch: test RMSE} at each checkpoint."""
    import torch
    import torch.nn as nn
    Xtr, Ytr, Xte, Yte, ym, ys = build(W)
    torch.manual_seed(seed)
    model = _recurrent(kind)
    untrained = {k: v.clone() for k, v in model.state_dict().items()}
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossfn = nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr, Ytr), batch_size=128, shuffle=False)
    scores = {}
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); lossfn(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if ep in ckpt or ep == epochs:
            model.eval()
            with torch.no_grad():
                pred = model(Xte).numpy() * ys + ym
            scores[ep] = float(np.sqrt(np.mean((Yte - pred) ** 2)))
    model.eval()
    return model, untrained, scores


# ================================================================== measurements
def _hidden_gradients(model, X):
    """|d yhat / d h_t| for every step t of every window, by unrolling nn.RNN by hand.

    nn.RNN does not expose its intermediate states to autograd, so the recurrence is rebuilt from
    the layer's own weights. The rebuilt forecast is checked against model(X) before use.
    """
    import torch
    r = model.rnn
    h = torch.zeros(X.shape[0], r.hidden_size)
    states = []
    for t in range(X.shape[1]):
        h = torch.tanh(X[:, t] @ r.weight_ih_l0.T + r.bias_ih_l0 + h @ r.weight_hh_l0.T + r.bias_hh_l0)
        h.retain_grad(); states.append(h)
    yhat = model.fc(h).squeeze(-1)
    with torch.no_grad():
        assert torch.allclose(yhat, model(X), atol=1e-4), "hand-unrolled RNN disagrees with nn.RNN"
    yhat.sum().backward()
    g = torch.stack([s.grad.norm(dim=1) for s in states], dim=1).numpy()   # (windows, T)
    rel = g / g[:, -1:]                      # relative to the most recent step
    return rel[:, ::-1]                       # column k = k steps back


def _survival(model, X) -> list[float]:
    """Typical (geometric-mean) share of the gradient left k steps back, k = 0..T-1."""
    rel = _hidden_gradients(model, X)
    return np.exp(np.log(rel).mean(axis=0)).tolist()


def _forget_gates(model, X) -> np.ndarray:
    """Every forget-gate value the trained LSTM produces on the test windows."""
    import torch
    r, H = model.rnn, model.rnn.hidden_size
    sl = slice(H, 2 * H)                      # PyTorch packs the gates as input, forget, cell, output
    h = torch.zeros(X.shape[0], H); c = torch.zeros(X.shape[0], H)
    fs = []
    with torch.no_grad():
        for t in range(X.shape[1]):
            z = X[:, t] @ r.weight_ih_l0.T + r.bias_ih_l0 + h @ r.weight_hh_l0.T + r.bias_hh_l0
            i, f = torch.sigmoid(z[:, :H]), torch.sigmoid(z[:, sl])
            g, o = torch.tanh(z[:, 2*H:3*H]), torch.sigmoid(z[:, 3*H:])
            c = f * c + i * g; h = o * torch.tanh(c)
            fs.append(f)
        assert torch.allclose(model.fc(h).squeeze(-1), model(X), atol=1e-4)
    return torch.stack(fs, dim=1).numpy()     # (windows, T, H)


def measure() -> dict:
    import torch
    torch.set_num_threads(max(1, torch.get_num_threads()))
    m: dict = {"note": "Written by scripts/figures/lecture09_part1_figures.py; every number the "
                       "Lecture 9 Part 1 figures print. Seeds 1 and 2, as in the lab."}

    # Check 1: both models, 26-week window, scored at every checkpoint (lab Step 4)
    curves = {}
    for kind in ("rnn", "lstm"):
        runs = [fit(kind, 26, s, max(CKPT), CKPT)[2] for s in SEEDS]
        curves[kind] = [float(np.mean([r[e] for r in runs])) for e in CKPT]
        print(f"  {kind:4s} W=26  " + "  ".join(f"{e}:{v:,.0f}" for e, v in zip(CKPT, curves[kind])))
    m["budget"] = {"epochs": CKPT, **curves}

    # Gradient survival and forget gates: 52-week window (the course's canonical window),
    # 30 epochs — the setting behind the scoreboard's 805 and 998
    Xte52 = build(52)[2]
    surv, surv0, test52 = [], [], {}
    for s in SEEDS:
        model, init, sc = fit("rnn", 52, s, 30)
        test52.setdefault("rnn", []).append(sc[30])
        surv.append(_survival(model, Xte52))
        model.load_state_dict(init)
        surv0.append(_survival(model, Xte52))
    gm = lambda runs: np.exp(np.log(np.array(runs)).mean(axis=0))
    trained, untrained = gm(surv), gm(surv0)
    k = np.arange(len(trained))
    fit_range = (k >= 5)                      # past the first few steps, where the slope settles
    per_step = lambda s: float(np.exp(np.polyfit(k[fit_range], np.log(s[fit_range]), 1)[0]))
    m["gradient"] = {"steps_back": k.tolist(), "trained": trained.tolist(),
                     "untrained": untrained.tolist(),
                     "per_step_trained": per_step(trained), "per_step_untrained": per_step(untrained)}

    fs = []
    for s in SEEDS:
        model, _, sc = fit("lstm", 52, s, 30)
        test52.setdefault("lstm", []).append(sc[30])
        fs.append(_forget_gates(model, Xte52))
    f = np.concatenate([x.ravel() for x in fs])
    four = np.concatenate([np.prod(x[:, -4:, :], axis=1).ravel() for x in fs])   # last 4 weeks
    m["forget_gate"] = {"median": float(np.median(f)), "p90": float(np.quantile(f, 0.9)),
                        "median_product_4_weeks": float(np.median(four))}
    m["test_rmse_52"] = {kk: float(np.mean(v)) for kk, v in test52.items()}

    OUT.mkdir(parents=True, exist_ok=True)
    MEASURED.write_text(json.dumps(m, indent=1) + "\n")
    print(f"  measured.json written: RNN/LSTM at 52 weeks {m['test_rmse_52']}")
    return m


@lru_cache(maxsize=None)
def measured() -> dict:
    if "--retrain" in sys.argv or not MEASURED.exists():
        return measure()
    return json.loads(MEASURED.read_text())


# ================================================================== unrolled RNN
def unrolled_rnn() -> None:
    """A real 26-week window read one week at a time by 26 copies of the same cell."""
    w = weekly()
    y = (w[(w.store_id == "CA_1") & (w.cat_id == "FOODS")]
         .set_index("week_start")["units"].sort_index())
    # The window Lecture 8's receptive-field figure used: the 26 weeks before Christmas week 2015.
    target = y.index[(y.index <= "2015-12-25") & (y.index > "2015-12-18")][0]
    t = y.index.get_loc(target)
    seg = y.iloc[t - 26:t].values / 1000

    fig, ax = plt.subplots(figsize=sz(13, 3.7))
    ax.set_xlim(-2.4, 32.6); ax.set_ylim(-2.05, 3.55); ax.axis("off")
    xs = np.arange(26) * 1.0
    CW, CY = 0.62, 0.0                                   # cell width, centre line

    # the data, one dot per week, directly above the cell that reads it
    lo, hi = seg.min(), seg.max()
    ys = 1.75 + 1.25 * (seg - lo) / (hi - lo)
    ax.plot(xs, ys, color=BLUE, lw=1.8, zorder=2)
    ax.scatter(xs, ys, s=16, color=BLUE, zorder=3)
    for x_, y_ in zip(xs, ys):
        ax.annotate("", xy=(x_, CY + CW/2 + 0.08), xytext=(x_, y_ - 0.12),
                    arrowprops=dict(arrowstyle="-|>", color="#9DB9D8", lw=0.8,
                                    mutation_scale=7))
    ax.text(-0.9, (ys.min() + ys.max()) / 2, "26 weeks\nof sales", ha="right", va="center",
            color=BLUE, fontsize=12)

    # 26 identical cells, the state passed left to right
    for x_ in xs:
        ax.add_patch(FancyBboxPatch((x_ - CW/2, CY - CW/2), CW, CW,
                                    boxstyle="round,pad=0.02,rounding_size=0.1",
                                    facecolor=LIGHT_CRIMSON, edgecolor=CRIMSON, lw=1.4, zorder=3))
    for a_, b_ in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(b_ - CW/2 - 0.02, CY), xytext=(a_ + CW/2 + 0.02, CY),
                    arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.0, mutation_scale=7))
    ax.text(xs[0], CY - CW/2 - 0.2, r"$\mathbf{h}_1$", ha="center", va="top", fontsize=12)
    ax.text(xs[-1], CY - CW/2 - 0.2, r"$\mathbf{h}_{26}$", ha="center", va="top", fontsize=12)

    # one linear layer on the last state, then the forecast
    LX, LW = 28.7, 2.5
    ax.add_patch(FancyBboxPatch((LX - LW/2, CY - 0.45), LW, 0.9,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=LIGHT_GRAY, edgecolor=GRAY, lw=1.4))
    ax.text(LX, CY, "linear", ha="center", va="center", fontsize=12)
    ax.annotate("", xy=(LX - LW/2 - 0.05, CY), xytext=(xs[-1] + CW/2 + 0.02, CY),
                arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.4, mutation_scale=10))
    ax.annotate("", xy=(31.55, CY), xytext=(LX + LW/2 + 0.05, CY),
                arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.4, mutation_scale=10))
    ax.plot([31.95], [CY], marker="o", ms=9, color=CRIMSON)
    ax.text(31.95, CY - 0.55, "next\nweek", ha="center", va="top", color=CRIMSON, fontsize=12)

    # what is shared
    bx0, bx1, by = xs[0] - CW/2, xs[-1] + CW/2, CY - CW/2 - 0.95
    ax.plot([bx0, bx0, bx1, bx1], [by + 0.15, by, by, by + 0.15], color=CRIMSON, lw=1.3)
    ax.text((bx0 + bx1) / 2, by - 0.18,
            r"the same cell 26 times: one $\mathbf{W}_h$, one $\mathbf{W}_x$, shared by every week",
            ha="center", va="top", fontsize=12, color=CRIMSON)
    save(fig, "unrolled_rnn")


# ================================================================== vanishing gradient
def gradient_decay() -> None:
    """How much gradient reaches k steps back: the slide's 0.7 example, and our own RNN.

    A constant factor per step is a straight line on a log axis; its slope is the factor.
    (The untrained network's curve is in measured.json: x0.55 per step. Training stretched the
    memory, but the line stays straight.)
    """
    g = measured()["gradient"]
    k, tr = np.array(g["steps_back"]), np.array(g["trained"])
    kk = np.arange(0, 53)

    fig, ax = plt.subplots(figsize=sz(11.5, 4.8))
    ax.set_yscale("log")
    for s in (26, 51):
        ax.axvline(s, color=GRAY, lw=0.8, ls=":", zorder=1)
    ax.plot(kk, 0.7 ** kk, color=GRAY, lw=1.6, ls=(0, (5, 3)), zorder=2)
    ax.plot(k, tr, color=CRIMSON, lw=2.8, zorder=3)

    # the old table's rows, as points on the example line
    for s, lab in [(10, "0.028"), (26, "0.000 094")]:
        ax.plot([s], [0.7 ** s], marker="o", ms=6, color=GRAY, zorder=4)
        ax.text(s - 0.8, 0.7 ** s * 0.3, lab, ha="right", va="top", fontsize=12, color=GRAY)
    ax.plot([52], [0.7 ** 52], marker="o", ms=6, color=GRAY, zorder=4)
    # and the measured line at the same two places
    for s in (26, 51):
        ax.plot([s], [tr[s]], marker="o", ms=7, color=CRIMSON, zorder=5)
    ax.text(26.9, tr[26] * 1.9, f"{tr[26]:.4f}", ha="left", va="bottom", fontsize=12, color=CRIMSON)

    ax.text(25.5, 2.4, "26-week window", ha="right", va="center", fontsize=12, color=GRAY)
    ax.text(50.5, 2.4, "same week last year", ha="right", va="center", fontsize=12, color=GRAY)
    pt = g["per_step_trained"]
    ax.text(52.5, tr[51], f"our RNN, trained\non 52-week windows:\n×{pt:.2f} per step", ha="left", va="center",
            fontsize=12, color=CRIMSON, fontweight="bold")
    ax.text(30, 0.7 ** 30 * 0.04, "the example: ×0.70 per step", ha="right", va="top",
            fontsize=12, color=GRAY)

    ax.set_xlim(0, 69); ax.set_ylim(3e-9, 6)
    ax.set_xticks([0, 10, 26, 40, 51])
    ax.set_xlabel("steps back from the most recent week")
    ax.set_ylabel("share of the gradient left")
    ax.set_yticks([1, 1e-2, 1e-4, 1e-6, 1e-8])
    ax.set_yticklabels(["1", "0.01", r"$10^{-4}$", r"$10^{-6}$", r"$10^{-8}$"])
    ax.spines["bottom"].set_bounds(0, 52)
    save(fig, "gradient_decay")


# ================================================================== cell-state lane
def cell_lanes() -> None:
    """Backward through three steps: the RNN multiplies at every hop, the cell only scales by f."""
    fig, ax = plt.subplots(figsize=sz(9.5, 4.4))
    ax.set_xlim(-5.5, 12.8); ax.set_ylim(-2.5, 5.3); ax.axis("off"); ax.set_aspect("equal")
    xs = np.arange(4) * 4.0
    YR, YC, R = 3.55, 0.0, 0.65

    def node(x, y, label, edge):
        ax.add_patch(Circle((x, y), R, facecolor="white", edgecolor=edge, lw=1.8, zorder=3))
        ax.text(x, y, label, ha="center", va="center", fontsize=14, zorder=4)

    def back(x0, x1, y, lw, color):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    mutation_scale=8 + 2.2 * lw, shrinkA=0, shrinkB=0))

    def lane_title(y, title, note, color):
        ax.text(-1.25, y + 0.25, title, ha="right", va="bottom", fontsize=13, fontweight="bold")
        ax.text(-1.25, y + 0.05, note, ha="right", va="top", fontsize=12, color=color)

    # vanilla RNN: every hop passes a weight matrix and a tanh slope
    lane_title(YR, "vanilla RNN", "almost nothing\narrives", CRIMSON)
    for x in xs:
        node(x, YR, r"$\mathbf{h}$", GRAY)
    widths = [4.6, 2.4, 1.2]                             # what is left after each hop
    for j_, (a, b) in enumerate(zip(xs[:0:-1], xs[-2::-1])):
        back(a - R - 0.05, b + R + 0.05, YR, widths[j_], CRIMSON)
        ax.text((a + b) / 2, YR + R + 0.05, r"$\times\mathbf{W}_h\tanh'$", ha="center",
                va="bottom", fontsize=11, color=GRAY)

    # LSTM cell: every hop is a forget-gate valve, and the new information is ADDED
    lane_title(YC, "LSTM cell", r"arrives intact" + "\n" + r"if $\mathbf{f}\approx 1$", BLUE)
    for x in xs:
        node(x, YC, r"$\mathbf{c}$", BLUE)
    for a, b in zip(xs[:0:-1], xs[-2::-1]):
        back(a - R - 0.05, b + R + 0.05, YC, 3.4, BLUE)
        ax.text((a + b) / 2, YC + 0.3, r"$\odot\,\mathbf{f}$", ha="center", va="bottom",
                fontsize=13, color=BLUE)
    for x in xs[1:]:                                    # the + where i*c~ comes in, below each node
        ax.annotate("", xy=(x, YC - R - 0.04), xytext=(x, YC - 1.45),
                    arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.3, mutation_scale=9))
        ax.text(x, YC - 1.55, r"$+\,\mathbf{i}\odot\tilde{\mathbf{c}}$", ha="center", va="top",
                fontsize=12, color=GREEN)
    save(fig, "cell_lanes")


# ================================================================== training budget
def budget_curve() -> None:
    """Check 1: the RNN/LSTM gap at every checkpoint, not only at the 30 epochs we used."""
    b = measured()["budget"]
    ep, rnn, lstm = np.array(b["epochs"]), np.array(b["rnn"]), np.array(b["lstm"])

    fig, ax = plt.subplots(figsize=sz(11, 4.5))
    ax.axvline(30, color=GRAY, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.plot(ep, rnn, color=CRIMSON, lw=2.4, marker="o", ms=6, zorder=3)
    ax.plot(ep, lstm, color=BLUE, lw=2.4, marker="o", ms=6, zorder=3)

    # each label placed by hand where the two lines leave room (positions checked on render)
    at = {e: list(ep).index(e) for e in (10, 30, 60)}
    lab = dict(fontsize=12)
    ax.text(12, rnn[at[10]], f"{rnn[at[10]]:,.0f}", ha="left", va="center", color=CRIMSON, **lab)
    ax.text(10, lstm[at[10]] - 22, f"{lstm[at[10]]:,.0f}", ha="center", va="top", color=BLUE, **lab)
    ax.text(28.8, rnn[at[30]] - 22, f"{rnn[at[30]]:,.0f}", ha="right", va="top", color=CRIMSON, **lab)
    ax.text(31.2, lstm[at[30]] + 18, f"{lstm[at[30]]:,.0f}", ha="left", va="bottom", color=BLUE, **lab)
    ax.text(60, rnn[at[60]] - 22, f"{rnn[at[60]]:,.0f}", ha="center", va="top", color=CRIMSON, **lab)
    ax.text(60, lstm[at[60]] + 22, f"{lstm[at[60]]:,.0f}", ha="center", va="bottom", color=BLUE, **lab)

    i30 = list(ep).index(30)
    gap = lstm[i30] - rnn[i30]
    bx = 33.5
    ax.plot([bx - 0.8, bx, bx, bx - 0.8], [rnn[i30], rnn[i30], lstm[i30], lstm[i30]],
            color=BLACK, lw=1.2)
    ax.text(bx + 1.0, (rnn[i30] + lstm[i30]) / 2, f"gap +{gap:,.0f}", ha="left", va="center",
            fontsize=12, fontweight="bold")
    ax.text(31.5, 1140, "the 30 epochs we used", ha="left", va="top", fontsize=12, color=GRAY)
    ax.text(ep[-1] + 2.5, rnn[-1] + 16, f"vanilla RNN {rnn[-1]:,.0f}", ha="left", va="center",
            fontsize=12, color=CRIMSON, fontweight="bold")
    ax.text(ep[-1] + 2.5, lstm[-1] - 16, f"LSTM {lstm[-1]:,.0f}", ha="left", va="center",
            fontsize=12, color=BLUE, fontweight="bold")

    ax.set_xlim(0, 133); ax.set_ylim(760, 1145)
    ax.set_xticks(ep); ax.spines["bottom"].set_bounds(10, 100)
    ax.set_xlabel("training epochs (26-week window, mean of two seeds)", fontsize=12)
    ax.set_ylabel("test RMSE")
    save(fig, "budget_curve")


FIGURES = {"unrolled_rnn": unrolled_rnn, "gradient_decay": gradient_decay,
           "cell_lanes": cell_lanes, "budget_curve": budget_curve}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if not a.startswith("--")] or list(FIGURES)
    print(f"Lecture 9 Part 1 figures -> {OUT.relative_to(ROOT)}   (font: {FONT})")
    if "--retrain" in sys.argv:
        measured()
    for name in pick:
        FIGURES[name]()
