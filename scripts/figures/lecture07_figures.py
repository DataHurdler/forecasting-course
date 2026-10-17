#!/usr/bin/env python3
"""Figures for Lecture 7 (Neural networks) — one script, every figure, both formats.

    .venv/bin/python scripts/figures/lecture07_figures.py                       # all
    .venv/bin/python scripts/figures/lecture07_figures.py relu_hinges           # one, by name

Writes Figures/Lecture07_NeuralNets/<name>.pdf (Beamer) and <name>.svg (Quarto), both vector,
and Figures/Lecture07_NeuralNets/measured.json with every number the figure prints.

relu_hinges: weekday electricity load (data/processed/electricity_daily.csv, PJM East 2002-2018,
weekends dropped) against day of year, a W-shaped year. Left, the best straight line: what any
stack of linear layers reduces to. Right, four ReLU units with one input, y = b + sum_k v_k
max(0, x - knot_k), at knots fixed in advance; only the output weights v_k and b are fitted, by
least squares on the four hinge features. That fit is exact, so nothing is trained or sampled
and the figure needs no seed. The knots are the best four on a 10-day grid (searched below, in
about ten seconds), so the right panel is the best this family can do up to that grid, not a
lucky draw. A trained network would place its own knots.
"""
from __future__ import annotations

import itertools
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
OUT = ROOT / "Figures" / "Lecture07_NeuralNets"
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


def record(section: str | None, numbers: dict) -> None:
    """Merge one figure's numbers into measured.json (relu_hinges' keys stay at the top level)."""
    m = json.loads(MEASURED.read_text()) if MEASURED.exists() else {}
    m["note"] = ("Written by scripts/figures/lecture07_figures.py: every number the Lecture 7 "
                 "figures print. Top-level keys: relu_hinges (weekday days only, x = day of year). "
                 "Sections: gd_bowl, epochs (trained; --retrain re-measures), weight_blocks.")
    if section is None:
        m.update(numbers)
    else:
        m[section] = numbers
    MEASURED.write_text(json.dumps(m, indent=1) + "\n")


# ================================================================== the data and the two fits
@lru_cache(maxsize=None)
def weekdays() -> tuple[np.ndarray, np.ndarray]:
    e = pd.read_csv(ROOT / "data" / "processed" / "electricity_daily.csv")
    wd = e[e.is_weekend == 0]
    return wd.doy.values.astype(float), wd.mw_mean.values.astype(float)


def hinge_design(x: np.ndarray, knots) -> np.ndarray:
    """Columns: 1, then max(0, x - knot) for each knot (one ReLU unit each, input weight 1)."""
    return np.column_stack([np.ones_like(x)] + [np.maximum(0.0, x - k) for k in knots])


def lstsq_rmse(H: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    w, *_ = np.linalg.lstsq(H, y, rcond=None)
    return float(np.sqrt(np.mean((y - H @ w) ** 2))), w


@lru_cache(maxsize=None)
def fits() -> dict:
    x, y = weekdays()
    line_rmse, line_w = lstsq_rmse(np.column_stack([np.ones_like(x), x]), y)
    grid = np.arange(10, 361, 10)
    best = min(itertools.combinations(grid, 4),
               key=lambda kn: lstsq_rmse(hinge_design(x, kn), y)[0])
    knots = [int(k) for k in best]
    hinge_rmse, w = lstsq_rmse(hinge_design(x, knots), y)
    # the best any function of day-of-year can do here: the mean load on each day of the year
    doy_mean = pd.Series(y).groupby(x).transform("mean").values
    ceiling = float(np.sqrt(np.mean((y - doy_mean) ** 2)))
    return dict(n=int(len(y)), line_rmse=line_rmse, line_w=line_w.tolist(), knots=knots,
                hinge_rmse=hinge_rmse, hinge_w=w.tolist(), ceiling_rmse=ceiling)


# ================================================================== the figure
MONTHS = [(1, "Jan"), (91, "Apr"), (182, "Jul"), (274, "Oct")]


def relu_hinges() -> None:
    x, y = weekdays()
    f = fits()
    assert round(f["line_rmse"]) == 4513, f"line RMSE {f['line_rmse']:.0f}, audit had 4,513"
    xs = np.linspace(1, 366, 800)
    b, v = f["hinge_w"][0], f["hinge_w"][1:]
    H = hinge_design(xs, f["knots"])
    total = H @ np.array(f["hinge_w"])
    line = f["line_w"][0] + f["line_w"][1] * xs

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=sz(11.4, 4.3), sharey=True)
    fig.subplots_adjust(wspace=0.06, left=0.09, right=0.99, top=0.88, bottom=0.12)
    YL = (19, 53)
    for ax in (ax1, ax2):
        ax.scatter(x, y / 1000, s=1.6, color="#B8B8BC", lw=0, rasterized=True, zorder=1)
        ax.set_ylim(*YL); ax.set_xlim(-4, 370)
        ax.set_yticks([20, 30, 40, 50])
        ax.set_xticks([m for m, _ in MONTHS]); ax.set_xticklabels([n for _, n in MONTHS])
        ax.tick_params(labelsize=12)

    ax1.plot(xs, line / 1000, color=BLUE, lw=2.4, zorder=3)
    ax1.set_ylabel("weekday load\n(thousand MW)", fontsize=12)
    ax1.set_title("any stack of linear layers", fontsize=13, pad=6, color=BLUE)
    ax1.text(362, 51, f"RMSE {f['line_rmse']:,.0f} MW", ha="right", va="center", fontsize=12,
             color=BLUE, bbox=dict(facecolor="white", edgecolor="none", pad=0.8), zorder=5)

    # four ReLU units: each switches on at its knot and changes the slope by its weight v_k.
    # Dashed: the line as it was before that unit switched on, i.e. the sum without unit k.
    knots = f["knots"]
    w = np.array(f["hinge_w"])
    for i, k in enumerate(knots):
        before = hinge_design(xs, knots[:i]) @ w[:i + 1]          # b + units 1..i-1
        m_ = (xs >= k) & (xs <= k + 34)
        ax2.plot(xs[m_], before[m_] / 1000, color=CRIMSON, lw=1.5, ls=(0, (3.5, 2.5)), zorder=2)
    ax2.plot(xs, total / 1000, color=CRIMSON, lw=2.4, zorder=3)
    kv = hinge_design(np.array(knots, float), knots) @ w
    ax2.plot(knots, kv / 1000, "o", ms=7, color=CRIMSON, mec="white", mew=1.2, zorder=4)
    ax2.set_title("what 4 ReLU units can represent", fontsize=13, pad=6, color=CRIMSON)
    ax2.text(362, 51, f"RMSE {f['hinge_rmse']:,.0f} MW", ha="right", va="center", fontsize=12,
             color=CRIMSON, bbox=dict(facecolor="white", edgecolor="none", pad=0.8), zorder=5)
    ax2.tick_params(labelleft=False)
    fig.text(0.54, -0.03, "knots fixed at days " + ", ".join(str(k) for k in knots) +
             ", weights by least squares\ndashed: the line before each unit switches on",
             ha="center", va="top", fontsize=12, color=GRAY, linespacing=1.2)
    save(fig, "relu_hinges")

    record(None, f)
    print(f"  line RMSE {f['line_rmse']:,.0f}; 4 hinges at {f['knots']} RMSE {f['hinge_rmse']:,.0f};"
          f" day-of-year ceiling {f['ceiling_rmse']:,.0f}; n = {f['n']:,}")


# ================================================================== gradient descent on a real bowl
# Lab 6's feature pipeline, verbatim (as in lecture06_figures.py, diamond_and_disc): CA_1 FOODS
# daily units, every lag and rolling column built so dropna keeps the same rows, last 365 days held
# out. Only lag7 and lag14 are used, standardized, with the target standardized too.
@lru_cache(maxsize=None)
def lag_bowl() -> dict:
    daily = (pd.read_csv(ROOT / "data" / "processed" / "m5_daily.csv", parse_dates=["date"])
               .sort_values("date"))
    u = daily.set_index("date")["units"]
    for L in list(range(1, 29)) + [35, 42, 49, 56, 364, 365, 366]:
        daily[f"lag{L}"] = u.shift(L).values
    for W in (7, 14, 28, 91, 364):
        daily[f"roll{W}_mean"] = u.shift(1).rolling(W).mean().values
        daily[f"roll{W}_std"] = u.shift(1).rolling(W).std().values
    df = daily.dropna(subset=[c for c in daily.columns if c.startswith(("lag", "roll"))])
    tr = df.iloc[:-365]
    X = tr[["lag7", "lag14"]].values.astype(float)
    X = (X - X.mean(0)) / X.std(0)
    y = tr["units"].values.astype(float)
    y = (y - y.mean()) / y.std()
    n = len(y)
    R, c = X.T @ X / n, X.T @ y / n
    b_ols = np.linalg.solve(R, c)
    # MSE loss L(b) = mean((y - Xb)^2): gradient 2(Rb - c), Hessian 2R, exactly, everywhere.
    lam = np.linalg.eigvalsh(2 * R)
    return dict(n=n, R=R, c=c, rho=float(R[0, 1]), b_ols=b_ols, lam_min=float(lam[0]),
                lam_max=float(lam[1]), eta_zigzag=float(1 / lam[1]), eta_diverge=float(2 / lam[1]),
                mse_min=float(np.mean((y - X @ b_ols) ** 2)))


GD_START = np.array([-0.35, 0.77])        # off the valley floor, so both directions show
GD_ETAS = (0.15, 0.50, 0.65)              # one per regime: < 1/lam_max, between, > 2/lam_max


def gd_path(eta: float, steps: int) -> np.ndarray:
    d = lag_bowl()
    b = GD_START.copy()
    path = [b.copy()]
    for _ in range(steps):
        b = b - eta * 2 * (d["R"] @ b - d["c"])
        path.append(b.copy())
    return np.array(path)


def gd_bowl() -> None:
    d = lag_bowl()
    assert abs(d["eta_zigzag"] - 0.28) < 0.01 and abs(d["eta_diverge"] - 0.56) < 0.015, \
        f"thresholds {d['eta_zigzag']:.3f} / {d['eta_diverge']:.3f}, audit had 0.28 / 0.56"
    assert GD_ETAS[0] < d["eta_zigzag"] < GD_ETAS[1] < d["eta_diverge"] < GD_ETAS[2]
    R, bo = d["R"], d["b_ols"]
    XL, YL = (-0.6, 1.3), (-0.45, 1.45)
    g1, g2 = np.meshgrid(np.linspace(*XL, 300), np.linspace(*YL, 300))
    D1, D2 = g1 - bo[0], g2 - bo[1]
    excess = R[0, 0] * D1 ** 2 + 2 * R[0, 1] * D1 * D2 + R[1, 1] * D2 ** 2
    levels = [0.005, 0.02, 0.05, 0.1, 0.2, 0.35, 0.55, 0.8]

    fig, axes = plt.subplots(1, 3, figsize=sz(11.6, 4.6), sharey=True)
    fig.subplots_adjust(wspace=0.08, left=0.07, right=0.99, top=0.88, bottom=0.17)
    titles = [f"η = {GD_ETAS[0]}: smooth", f"η = {GD_ETAS[1]}: zig-zag",
              f"η = {GD_ETAS[2]}: diverges"]
    colors = [GREEN, AMBER, CRIMSON]
    out = {}
    for ax, eta, title, col in zip(axes, GD_ETAS, titles, colors):
        ax.contour(g1, g2, excess, levels=levels, colors="#9A9AA0", linewidths=0.8)
        ax.plot(*bo, marker="x", ms=9, mew=2.2, color=BLACK, zorder=5)
        p = gd_path(eta, 25)
        inside = (p[:, 0] > XL[0]) & (p[:, 0] < XL[1]) & (p[:, 1] > YL[0]) & (p[:, 1] < YL[1])
        last = int(np.argmin(inside)) if not inside.all() else len(p)
        q = p[:last + 1] if last < len(p) else p
        ax.plot(q[:, 0], q[:, 1], "-", color=col, lw=1.6, zorder=3)
        ax.plot(q[:last, 0], q[:last, 1], "o", ms=4.2, color=col, mec="white", mew=0.6, zorder=4)
        if last < len(p):                          # the step that leaves the panel
            ax.annotate("", xy=q[-1], xytext=q[-2], zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6))
        ax.plot(*GD_START, "o", ms=7, color=col, mec="white", mew=1.0, zorder=4)
        ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_aspect("equal")
        ax.set_title(title, fontsize=13, color=col, pad=5)
        ax.set_xticks([-0.5, 0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=11)
        ax.set_xlabel("weight on lag 7", fontsize=12)
        out[str(eta)] = dict(dist_after_10=float(np.linalg.norm(p[10] - bo)))
    axes[0].set_ylabel("weight on lag 14", fontsize=12)
    axes[0].annotate("start", xy=GD_START, xytext=(GD_START[0] + 0.05, GD_START[1] + 0.45),
                     fontsize=12, color=GRAY, ha="center",
                     arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8, shrinkB=5))
    axes[0].annotate("minimum", xy=bo, xytext=(bo[0] + 0.3, bo[1] - 0.5), fontsize=12,
                     color=GRAY, ha="center", va="top",
                     bbox=dict(facecolor="white", edgecolor="none", pad=0.6),
                     arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8, shrinkB=6))
    fig.text(0.53, -0.06, f"zig-zag above 1/λ$_{{max}}$ = {d['eta_zigzag']:.2f}; diverges above "
             f"2/λ$_{{max}}$ = {d['eta_diverge']:.2f}   (λ$_{{max}}$ = {d['lam_max']:.2f}, "
             "the bowl's steepest curvature)", ha="center", va="top", fontsize=12, color=GRAY)
    save(fig, "gd_bowl")
    record("gd_bowl", dict(n_train=d["n"], rho=d["rho"], b_ols=d["b_ols"].tolist(),
                           hessian_eigenvalues=[d["lam_min"], d["lam_max"]],
                           eta_zigzag=d["eta_zigzag"], eta_diverge=d["eta_diverge"],
                           start=GD_START.tolist(), etas=list(GD_ETAS), paths=out))
    print(f"  Hessian 2R eigenvalues {d['lam_min']:.3f}, {d['lam_max']:.3f}; rho {d['rho']:.3f}; "
          f"zig-zag above 1/lam_max = {d['eta_zigzag']:.3f}; diverges above 2/lam_max = "
          f"{d['eta_diverge']:.3f}")


# ================================================================== train vs held-out by epoch
# Labs/Lecture07_lab.qmd, Steps 1-2, verbatim: 30 M5 weekly series, 26-week windows of four
# channels flattened to 104 features, split by the target's date at 2015-05-17, standardized on
# training statistics; make_model (104 -> 128 -> 64 -> 1, dropout 0.2), Adam lr 5e-4, batch 64,
# shuffle=False, seeds 1-3. The only change: train for EPOCHS instead of 60 and record, after
# every epoch, RMSE in units on the training windows and on the lab's test weeks (eval mode).
EPOCHS = 200


def train_curves() -> dict:
    import torch
    import torch.nn as nn
    weekly = (pd.read_csv(ROOT / "data" / "processed" / "m5_weekly.csv", parse_dates=["week_start"])
                .sort_values(["store_id", "cat_id", "week_start"]))
    CH, W, CUT = ["units", "snap_days", "event_days", "avg_price"], 26, np.datetime64("2015-05-17")
    X, Y, D = [], [], []
    for _, g in weekly.groupby(["store_id", "cat_id"]):
        a, dates = g[CH].values.astype("float32"), g.week_start.values
        for i in range(len(g) - W):
            X.append(a[i:i + W]); Y.append(a[i + W, 0]); D.append(dates[i + W])
    X, Y, D = np.stack(X), np.array(Y, dtype="float32"), np.array(D)
    tr, te = D < CUT, D >= CUT
    mu = X[tr].mean(axis=(0, 1), keepdims=True)
    sd = X[tr].std(axis=(0, 1), keepdims=True) + 1e-8
    y_mu, y_sd = Y[tr].mean(), Y[tr].std()
    Xn, Yn = (X - mu) / sd, (Y - y_mu) / y_sd
    X_tr = torch.tensor(Xn[tr].reshape(tr.sum(), -1)); y_tr = torch.tensor(Yn[tr])
    X_te = torch.tensor(Xn[te].reshape(te.sum(), -1))
    rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

    curves = {}
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(X_tr.shape[1], 128), nn.ReLU(), nn.Dropout(0.2),
                              nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
        opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0.0)
        lossfn = nn.MSELoss()
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_tr, y_tr),
                                             batch_size=64, shuffle=False)
        trc, tec = [], []
        for _ in range(EPOCHS):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                lossfn(model(xb).squeeze(-1), yb).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                trc.append(rmse(Y[tr], model(X_tr).squeeze(-1).numpy() * y_sd + y_mu))
                tec.append(rmse(Y[te], model(X_te).squeeze(-1).numpy() * y_sd + y_mu))
        curves[str(seed)] = dict(train=trc, test=tec)
    n_params = sum(p.numel() for p in model.parameters())
    m = dict(epochs=EPOCHS, n_train=int(tr.sum()), n_test=int(te.sum()), n_params=int(n_params),
             torch=torch.__version__, curves=curves)
    for s, c in curves.items():
        m[f"seed{s}"] = dict(test_at_60=c["test"][59], train_at_60=c["train"][59],
                             best_epoch=int(np.argmin(c["test"])) + 1, best_test=min(c["test"]),
                             train_at_end=c["train"][-1], test_at_end=c["test"][-1])
    record("epochs", m)
    return m


def epochs_measured() -> dict:
    m = json.loads(MEASURED.read_text()) if MEASURED.exists() else {}
    if "--retrain" in sys.argv or "epochs" not in m:
        return train_curves()
    return m["epochs"]


def overfit_epochs() -> None:
    m = epochs_measured()
    lab60 = [round(m[f"seed{s}"]["test_at_60"]) for s in (1, 2, 3)]
    assert lab60 == [1109, 1176, 946], f"epoch-60 test RMSEs {lab60}; Lab 7 Step 3 prints 1,109 / 1,176 / 946"
    ep = np.arange(1, m["epochs"] + 1)
    fig, ax = plt.subplots(figsize=sz(11.0, 4.6))
    fig.subplots_adjust(left=0.10, right=0.82, top=0.97, bottom=0.15)
    for s, c in m["curves"].items():
        ax.plot(ep, c["train"], color="#A1A1A6", lw=0.8, alpha=0.45, zorder=2)
        ax.plot(ep, c["test"], color=CRIMSON, lw=0.8, alpha=0.35, zorder=2)
    mean_tr = np.mean([c["train"] for c in m["curves"].values()], axis=0)
    mean_te = np.mean([c["test"] for c in m["curves"].values()], axis=0)
    ax.plot(ep, mean_tr, color=GRAY, lw=2.2, zorder=3)
    ax.plot(ep, mean_te, color=CRIMSON, lw=2.2, zorder=3)
    for s, c in m["curves"].items():
        b = m[f"seed{s}"]["best_epoch"]
        ax.plot(b, c["test"][b - 1], "o", ms=7, color=CRIMSON, mec="white", mew=1.2, zorder=5)
    ax.axvline(60, color=BLACK, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(63, 1820, "Lab 7 stops at 60", fontsize=12, color=BLACK, va="center", ha="left")
    ax.set_xlim(0, m["epochs"]); ax.set_ylim(350, 1900)
    ax.set_yticks([400, 800, 1200, 1600]); ax.set_yticklabels(["400", "800", "1,200", "1,600"])
    ax.tick_params(labelsize=12)
    ax.set_xlabel("epoch", fontsize=12); ax.set_ylabel("RMSE (units)", fontsize=12)
    # direct labels at the right edge, at each family's level over the last 20 epochs
    last_te = np.mean([np.mean(c["test"][-20:]) for c in m["curves"].values()])
    last_tr = np.mean([np.mean(c["train"][-20:]) for c in m["curves"].values()])
    ax.text(m["epochs"] + 4, last_te + 90, "held-out\ntest weeks", color=CRIMSON, fontsize=12,
            va="center", ha="left", linespacing=1.1, clip_on=False)
    ax.text(m["epochs"] + 4, last_tr - 60, "training\nwindows", color=GRAY, fontsize=12,
            va="center", ha="left", linespacing=1.1, clip_on=False)
    bests = sorted(m[f"seed{s}"]["best_epoch"] for s in (1, 2, 3))
    fig.text(0.45, -0.08, f"thin: three seeds · bold: their mean · dots: each seed's best "
             f"held-out epoch ({bests[0]}, {bests[1]}, {bests[2]})", ha="center", va="top", fontsize=12, color=GRAY)
    save(fig, "overfit_epochs")
    print(f"  epoch-60 held-out {lab60} (Lab 7: 1,109 / 1,176 / 946); best epochs {bests}; "
          f"best RMSE {[round(m[f'seed{s}']['best_test']) for s in (1, 2, 3)]}")


# ================================================================== the weight matrices, to scale
LAYERS = [(46, 128), (128, 128), (128, 1)]       # the deck's network: 46 features, 128, 128, 1


def weight_blocks() -> None:
    counts = [i * o + o for i, o in LAYERS]
    total = sum(counts)
    assert total == 22657 and counts == [6016, 16512, 129]
    share = [c / total for c in counts]
    fig, ax = plt.subplots(figsize=sz(11.4, 3.9))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    gap, x = 62, 34
    face = [LIGHT_BLUE, LIGHT_CRIMSON, LIGHT_BLUE]
    edge = [BLUE, CRIMSON, BLUE]
    centres = []
    ax.text(x - 8, 64, "x\n46 features", ha="right", va="center", fontsize=12, color=GRAY,
            linespacing=1.1)
    for k, ((i, o), c, s) in enumerate(zip(LAYERS, counts, share)):
        y0 = 64 - o / 2
        ax.add_patch(plt.Rectangle((x, y0), i, o, facecolor=face[k], edgecolor=edge[k], lw=1.4))
        ax.add_patch(plt.Rectangle((x + i + 1.5, y0), 1.6, o, facecolor=edge[k], lw=0))  # biases
        centres.append(x + i / 2)
        pct = f"{s:.0%}" if s >= 0.01 else f"{s:.1%}"
        ax.text(x + i / 2, -10, f"{i} × {o}", ha="center", va="top", fontsize=12, color=BLACK)
        ax.text(x + i / 2, -27, f"{c:,} ({pct})", ha="center", va="top", fontsize=12,
                color=edge[k], fontweight="bold" if k == 1 else "normal")
        right = x + i + 3.1
        if k < 2:
            ax.annotate("", xy=(right + gap - 6, 64), xytext=(right + 4, 64),
                        arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.0))
            ax.text(right + gap / 2 - 1, 70, "ReLU\ndropout", ha="center", va="bottom",
                    fontsize=11, color=GRAY, linespacing=1.0)
        x = right + gap
    ax.text(x - gap + 10, 64, "ŷ", ha="left", va="center", fontsize=13, color=GRAY)
    ax.text((x - gap) / 2, -46, "drawn to scale: area = parameters; the thin strips are the biases."
            f"  Total {total:,}", ha="center", va="top", fontsize=12, color=GRAY)
    ax.set_xlim(-30, x - gap + 30); ax.set_ylim(-62, 132)
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "weight_blocks")
    record("weight_blocks", dict(layers=LAYERS, params=counts, total=total, share=share))
    print(f"  params {counts} = {total:,}; 128x128 share {share[1]:.1%}")


# ================================================================== activations and their slopes
def activation_slopes() -> None:
    z = np.linspace(-4, 4, 400)
    sig = 1 / (1 + np.exp(-z))
    fig, axes = plt.subplots(1, 3, figsize=sz(11.4, 3.5))
    fig.subplots_adjust(wspace=0.28, left=0.04, right=0.99, top=0.84, bottom=0.12)
    spec = [("ReLU", np.maximum(0, z), (-0.6, 4.2), "hidden layers", BLUE),
            ("sigmoid", sig, (-0.15, 1.15), "gates (Lecture 9)", AMBER),
            ("tanh", np.tanh(z), (-1.25, 1.25), "sequence states", GREEN)]
    for ax, (name, f, yl, use, col) in zip(axes, spec):
        ax.axhline(0, color="#C8C8CC", lw=0.8, zorder=0); ax.axvline(0, color="#C8C8CC", lw=0.8, zorder=0)
        ax.plot(z, f, color=col, lw=2.6, zorder=3)
        ax.set_xlim(-4, 4); ax.set_ylim(*yl)
        ax.set_xticks([-4, 0, 4]); ax.tick_params(labelsize=11)
        ax.set_title(f"{name}: {use}", fontsize=12.5, color=col, pad=5)
    # sigmoid: tangent of slope 1/4 through (0, 1/2); tanh: slope 1 through the origin
    t = np.linspace(-1.9, 1.9, 2)
    axes[1].plot(t, 0.5 + 0.25 * t, color=BLACK, lw=1.3, ls=(0, (3.5, 2.5)), zorder=4)
    axes[1].text(0.25, 0.22, "max slope ¼", fontsize=12, color=BLACK, ha="left", va="center")
    t = np.linspace(-1.1, 1.1, 2)
    axes[2].plot(t, t, color=BLACK, lw=1.3, ls=(0, (3.5, 2.5)), zorder=4)
    axes[2].text(0.25, -0.55, "max slope 1", fontsize=12, color=BLACK, ha="left", va="center")
    axes[0].text(-2.0, 0.55, "slope 0", fontsize=12, color=BLACK, ha="center", va="bottom")
    axes[0].text(1.3, 2.75, "slope 1", fontsize=12, color=BLACK, ha="right", va="center")
    axes[0].set_yticks([0, 2, 4]); axes[1].set_yticks([0, 0.5, 1]); axes[2].set_yticks([-1, 0, 1])
    for ax in axes:
        ax.set_xlabel("z", fontsize=12, labelpad=1)
    save(fig, "activation_slopes")


FIGURES = {"relu_hinges": relu_hinges, "activation_slopes": activation_slopes,
           "weight_blocks": weight_blocks, "gd_bowl": gd_bowl, "overfit_epochs": overfit_epochs}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if not a.startswith("--")] or list(FIGURES)
    print(f"Lecture 7 figures -> {OUT.relative_to(ROOT)}   (font: {FONT})")
    for name in pick:
        FIGURES[name]()
