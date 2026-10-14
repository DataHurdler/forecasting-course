#!/usr/bin/env python3
"""Figures for Lecture 13 (Synthesis) — one script, every figure, both formats.

    .venv/bin/python scripts/figures/lecture13_figures.py                  # all
    .venv/bin/python scripts/figures/lecture13_figures.py scoreboard       # one, by name
    .venv/bin/python scripts/figures/lecture13_figures.py --retrain        # re-measure

Writes Figures/Lecture13_Synthesis/<name>.pdf (Beamer) and <name>.svg (Quarto), both vector.

Every number either figure prints is recomputed here, from the code that produced it:

* The scoreboard's tabular rows (LASSO, XGBoost, random forest, seasonal naive) rerun the
  assignments' own setup code (Homework 3 and Homework 4) with the settings their questions fix.
* The CNN reruns Lab 8's model and loop, three seeds, 26-week windows.
* The RNN, LSTM and Transformer are read from Lecture 9's caches,
  Figures/Lecture09_Part1/measured.json and Figures/Lecture09_Part2/measured.json, which those
  decks' generators wrote by retraining the labs' own models (52-week windows).
* The Christmas figure reads data/processed/m5_daily.csv directly: CA_1 FOODS, seasonal naive =
  the same weekday one year (364 days) earlier.

The CNN takes a couple of minutes on a laptop CPU; everything else takes seconds. Results go to
Figures/Lecture13_Synthesis/measured.json, and later runs redraw from that file.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Figures" / "Lecture13_Synthesis"
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


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


CUTOFF = pd.Timestamp("2015-05-17")          # the held-out block every weekly model shares

# The scoreboard as the deck prints it (owner decision 2026-10-13: 52-week windows for the
# sequence models; the CNN's 1,032 is Lab 8's 26-week fit and is labelled as such).
# (key, label in the figure, course part, number on the slide)
BOARD = [
    ("lasso", "LASSO, 46 features", "trees", 744),
    ("xgboost", "XGBoost", "trees", 781),
    ("rnn", "Vanilla RNN", "deep", 805),
    ("rf", "Random forest", "trees", 899),
    ("transformer", "Transformer", "deep", 990),
    ("lstm", "LSTM", "deep", 998),
    ("cnn", "1D CNN (26-week window)", "deep", 1032),
]
NAIVE = 2152


# ================================================================== the assignments, verbatim
def _hw4_frame():
    """Homework 4's setup cell: 46 engineered features, train < 2015-05-17."""
    weekly = (pd.read_csv(ROOT / "data" / "processed" / "m5_weekly.csv", parse_dates=["week_start"])
                .sort_values(["store_id", "cat_id", "week_start"]))
    g = weekly.groupby(["store_id", "cat_id"])["units"]
    for L in (1, 2, 3, 4, 5, 6, 8, 13, 26, 52):
        weekly[f"lag{L}"] = g.shift(L)
    for W in (4, 13, 26, 52):
        weekly[f"roll{W}_mean"] = g.shift(1).rolling(W).mean().values
        weekly[f"roll{W}_std"] = g.shift(1).rolling(W).std().values
    weekly["month"] = weekly["week_start"].dt.month
    weekly["week_of_year"] = weekly["week_start"].dt.isocalendar().week.astype(int)
    weekly["year"] = weekly["week_start"].dt.year
    for k in (1, 2, 3):
        weekly[f"sin{k}"] = np.sin(2 * np.pi * k * weekly.week_of_year / 52)
        weekly[f"cos{k}"] = np.cos(2 * np.pi * k * weekly.week_of_year / 52)
    df_enc = pd.get_dummies(weekly, columns=["store_id", "cat_id", "state_id"],
                            prefix=["store", "cat", "state"])
    feats = ([c for c in df_enc.columns
              if c.startswith(("lag", "roll", "sin", "cos", "store_", "cat_", "state_"))]
             + ["avg_price", "snap_days", "event_days", "month", "week_of_year", "year"])
    lagged = [c for c in feats if c.startswith(("lag", "roll"))]
    df = df_enc.dropna(subset=lagged).copy()
    return df[df.week_start < CUTOFF], df[df.week_start >= CUTOFF], feats


def _hw3_frame():
    """Homework 3's setup cell: five lags, calendar, one-hot ids (27 features)."""
    weekly = pd.read_csv(ROOT / "data" / "processed" / "m5_weekly.csv", parse_dates=["week_start"])
    weekly["month"] = weekly["week_start"].dt.month
    weekly["week_of_year"] = weekly["week_start"].dt.isocalendar().week.astype(int)
    weekly["year"] = weekly["week_start"].dt.year
    df_enc = pd.get_dummies(weekly, columns=["store_id", "cat_id", "state_id"],
                            prefix=["store", "cat", "state"])
    lags = ["units_lag1", "units_lag2", "units_lag3", "units_lag4", "units_lag52"]
    feats = (lags + ["avg_price", "snap_days", "event_days", "month", "week_of_year", "year"]
             + [c for c in df_enc.columns if c.startswith(("store_", "cat_", "state_"))])
    df = df_enc.dropna(subset=lags).copy()
    return df[df.week_start < CUTOFF], df[df.week_start >= CUTOFF], feats


def tabular_board() -> dict:
    """Seasonal naive, LASSO and XGBoost (Homework 4) and the random forest (Homework 3)."""
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNetCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr, te, F = _hw4_frame()
    Xtr, ytr = tr[F].values.astype(float), tr["units"].values
    Xte, yte = te[F].values.astype(float), te["units"].values
    out = {"hw4_features": len(F), "train_rows": len(tr), "test_rows": len(te),
           "naive": rmse(yte, te["lag52"])}
    # Q4: ElasticNetCV at l1_ratio = 1.0 (pure LASSO) inside a scaling pipeline
    las = make_pipeline(StandardScaler(),
                        ElasticNetCV(l1_ratio=1.0, alphas=np.logspace(-2, 4, 50),
                                     cv=TimeSeriesSplit(n_splits=5), max_iter=50000))
    las.fit(Xtr, ytr)
    out["lasso"] = rmse(yte, las.predict(Xte))
    out["lasso_nonzero"] = int((las[-1].coef_ != 0).sum())
    # Q1: the baseline XGBoost the assignment fixes. The key's 781 came from an older xgboost;
    # under 3.x the default tree method is "hist". Both are recorded.
    kw = dict(n_estimators=200, learning_rate=0.1, max_depth=5, subsample=0.8,
              colsample_bytree=0.8, random_state=42)
    out["xgboost"] = rmse(yte, xgb.XGBRegressor(**kw).fit(Xtr, ytr).predict(Xte))
    out["xgboost_exact"] = rmse(yte, xgb.XGBRegressor(tree_method="exact", **kw)
                                .fit(Xtr, ytr).predict(Xte))
    out["xgboost_version"] = xgb.__version__
    # Homework 3 Q3: the forest
    tr3, te3, F3 = _hw3_frame()
    rf = RandomForestRegressor(n_estimators=200, max_features="sqrt", oob_score=True,
                               random_state=42, n_jobs=-1)
    rf.fit(tr3[F3].values.astype(float), tr3["units"].values)
    out["rf"] = rmse(te3["units"], rf.predict(te3[F3].values.astype(float)))
    return out


# ================================================================== Lab 8, verbatim
def cnn_runs() -> list[float]:
    """Lab 8 Step 3: the flatten-head CNN, 26-week windows, seeds 1-3, 40 epochs."""
    import torch
    import torch.nn as nn
    weekly = (pd.read_csv(ROOT / "data" / "processed" / "m5_weekly.csv", parse_dates=["week_start"])
                .sort_values(["store_id", "cat_id", "week_start"]))
    channels, W = ["units", "snap_days", "event_days", "avg_price"], 26
    X, Y, D = [], [], []
    for _, g in weekly.groupby(["store_id", "cat_id"]):
        arr, dates = g[channels].values.astype("float32"), g.week_start.values
        for i in range(len(g) - W):
            X.append(arr[i:i+W]); Y.append(arr[i+W, 0]); D.append(dates[i+W])
    X = np.stack(X); Y = np.array(Y, dtype="float32"); D = np.array(D)
    tr, te = D < np.datetime64("2015-05-17"), D >= np.datetime64("2015-05-17")
    mu, sd = X[tr].mean((0, 1), keepdims=True), X[tr].std((0, 1), keepdims=True) + 1e-8
    ym, ys = Y[tr].mean(), Y[tr].std()
    Xn, Yn = (X - mu) / sd, (Y - ym) / ys
    Xtr, Ytr = torch.tensor(Xn[tr].transpose(0, 2, 1)), torch.tensor(Yn[tr])
    Xte, Yte = torch.tensor(Xn[te].transpose(0, 2, 1)), Y[te]

    runs = []
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Conv1d(4, 32, 5), nn.ReLU(), nn.Conv1d(32, 64, 3), nn.ReLU(),
                              nn.Flatten(), nn.Linear(64 * (W - 4 - 2), 1))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        lossfn = nn.MSELoss()
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr, Ytr),
                                             batch_size=128, shuffle=False)
        for _ in range(40):
            model.train()
            for xb, yb in loader:
                opt.zero_grad(); lossfn(model(xb).squeeze(-1), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            runs.append(rmse(Yte, model(Xte).squeeze(-1).numpy() * ys + ym))
        print(f"  CNN seed {seed}: {runs[-1]:,.0f}", file=sys.stderr)
    return runs


# ================================================================== Christmas
def christmas() -> dict:
    """CA_1 FOODS daily, against the seasonal naive (the same weekday 364 days earlier)."""
    d = (pd.read_csv(ROOT / "data" / "processed" / "m5_daily.csv", parse_dates=["date"])
           .sort_values("date").set_index("date"))
    y = d["units"].astype(float)
    f = y.shift(364)
    e = (y - f).dropna()
    se = e ** 2
    xmas = se.index.strftime("%m-%d") == "12-25"
    echo = [t for t in f.index[f == 0] if t in se.index]          # last year's zero, replayed
    windows = {}
    for yr in range(2012, 2016):
        w = pd.DataFrame({"actual": y, "naive": f}).loc[f"{yr}-12-18":f"{yr + 1}-01-01"]
        windows[str(yr)] = {"dates": [t.strftime("%Y-%m-%d") for t in w.index],
                            "actual": w["actual"].tolist(), "naive": w["naive"].tolist()}
    return {
        "days_evaluated": int(len(e)),
        "first_day": e.index.min().strftime("%Y-%m-%d"),
        "daily_mean_units": float(y.mean()),
        "closures_in_sample": int((y == 0).sum()),
        "is_event_days": int(d["is_event"].sum()),
        "is_event_on_closures": int(d.loc[y == 0, "is_event"].sum()),
        "rmse": float(np.sqrt(se.mean())),
        "rmse_without_25dec": float(np.sqrt(se[~xmas].mean())),
        "closure_errors": {t.strftime("%Y"): float(-e[t]) for t in se.index[xmas]},
        "closure_share_of_days": float(xmas.mean()),
        "closure_share_of_sq_error": float(se[xmas].sum() / se.sum()),
        "closure_square_vs_average": float(se[xmas].mean() / se.mean()),
        "echo_days": [t.strftime("%Y-%m-%d") for t in echo],
        "echo_errors": [float(e[t]) for t in echo],
        "echo_share_of_sq_error": float(se[echo].sum() / se.sum()),
        "windows": windows,
    }


# ================================================================== measurement
def measure() -> dict:
    m: dict = {"note": "Written by scripts/figures/lecture13_figures.py; every number the "
                       "Lecture 13 figures print, recomputed from the code that produced it."}
    # torch and xgboost each bring an OpenMP runtime; loaded into one process they deadlock on
    # macOS. The CNN therefore trains in a child process of this same script.
    import subprocess
    child = subprocess.run([sys.executable, __file__, "--cnn-only"], capture_output=True,
                           text=True, check=True)
    print(child.stderr, end="")
    runs = json.loads(child.stdout.strip().splitlines()[-1])
    tab = tabular_board()
    l9a = json.loads((ROOT / "Figures" / "Lecture09_Part1" / "measured.json").read_text())
    l9b = json.loads((ROOT / "Figures" / "Lecture09_Part2" / "measured.json").read_text())
    m["board"] = {
        "lasso": tab["lasso"], "xgboost": tab["xgboost"], "rf": tab["rf"], "naive": tab["naive"],
        "rnn": l9a["test_rmse_52"]["rnn"], "lstm": l9a["test_rmse_52"]["lstm"],
        "transformer": float(np.mean(l9b["test_rmse"])), "cnn": float(np.mean(runs)),
    }
    m["board_detail"] = {k: v for k, v in tab.items() if k not in m["board"]}
    m["board_detail"]["cnn_runs"] = runs
    m["board_detail"]["transformer_runs"] = l9b["test_rmse"]
    m["christmas"] = christmas()
    OUT.mkdir(parents=True, exist_ok=True)
    MEASURED.write_text(json.dumps(m, indent=1) + "\n")
    print("  measured.json written")
    return m


@lru_cache(maxsize=None)
def measured() -> dict:
    if "--retrain" in sys.argv or not MEASURED.exists():
        return measure()
    return json.loads(MEASURED.read_text())


def check_board(m: dict) -> None:
    """Every slide number must reproduce before it is drawn — except XGBoost, which is reported."""
    b = m["board"]
    for key, _, _, slide in BOARD:
        got = round(b[key])
        if key == "xgboost":
            # The Homework 4 key's 781 does not reproduce exactly under xgboost 3.x: the default
            # tree method changed to "hist" (792 here; "exact" gives 779). The seed spread at this
            # config is 20-90, so the slide keeps the key's number and the gap is reported.
            assert abs(got - slide) < 20, f"xgboost {got} vs slide {slide}"
            if got != slide:
                print(f"  NOTE xgboost reproduces {got} (exact method "
                      f"{m['board_detail']['xgboost_exact']:,.0f}); slide keeps {slide}")
            continue
        assert got == slide, f"{key}: recomputed {got}, slide says {slide}"
    assert round(b["naive"]) == NAIVE, f"naive {b['naive']}"


# ================================================================== the scoreboard
def scoreboard() -> None:
    """Test RMSE on one axis from zero: the whole course in a band, the benchmark far away."""
    m = measured()
    check_board(m)
    colour = {"trees": BLUE, "deep": CRIMSON}

    fig, ax = plt.subplots(figsize=sz(8.4, 4.7))
    n = len(BOARD)
    ys = np.arange(n)[::-1]
    lo, hi = BOARD[0][3], BOARD[-1][3]
    ax.axvspan(lo, hi, color=LIGHT_GRAY, zorder=0, lw=0)
    for y in ys:
        ax.plot([0, hi + 120], [y, y], color="#E4E4E7", lw=0.8, zorder=1)
    for y, (_, label, part, val) in zip(ys, BOARD):
        ax.plot([val], [y], "o", ms=8.5, color=colour[part], zorder=4)
    ax.axvline(NAIVE, color=GRAY, lw=1.6, ls=(0, (4, 2.5)), zorder=2)
    ax.text(NAIVE - 45, n - 0.05, f"seasonal\nnaive\n{NAIVE:,}", ha="right", va="top",
            fontsize=12, color=GRAY, linespacing=1.1)
    ax.text((lo + hi) / 2, n - 0.05, f"{lo}–{hi:,}", ha="center", va="top", fontsize=12,
            color=BLACK, fontweight="bold")

    ax.set_yticks(ys)
    ax.set_yticklabels([b[1] for b in BOARD], fontsize=12)
    for t, b in zip(ax.get_yticklabels(), BOARD):
        t.set_color(colour[b[2]])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, n + 0.1)
    ax.set_xlim(0, 2350)
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_xticklabels(["0", "500", "1,000", "1,500", "2,000"], fontsize=11)
    ax.set_xlabel("test RMSE, weekly units", fontsize=11)
    ax.spines["left"].set_visible(False)

    handles = [Line2D([], [], ls="", marker="o", ms=8, color=BLUE, label="Lectures 4–6"),
               Line2D([], [], ls="", marker="o", ms=8, color=CRIMSON, label="Lectures 7–9")]
    ax.legend(handles=handles, loc="center", bbox_to_anchor=(1560 / 2350, 0.36),
              frameon=False, fontsize=11, handletextpad=0.3, borderaxespad=0)
    save(fig, "scoreboard")


# ================================================================== Christmas as squares
def christmas_squares() -> None:
    """Left: 18 Dec-1 Jan, 2012-15, actual vs seasonal naive. Right: squared error as area."""
    c = measured()["christmas"]
    side = c["rmse"]
    errs = c["closure_errors"]                                      # year -> forecast on 25 Dec
    assert [round(errs[y]) for y in ("2012", "2013", "2014", "2015")] == [2230, 2045, 2485, 2365]
    assert round(100 * c["closure_share_of_sq_error"], 1) == 4.6
    assert round(100 * c["closure_share_of_days"], 2) == 0.25
    assert round(c["closure_square_vs_average"]) == 18

    fig = plt.figure(figsize=sz(14.5, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.10)

    # ---- left: the window, four Christmases overlaid
    ax = fig.add_subplot(gs[0])
    x = np.arange(-7, 8)
    for yr, w in c["windows"].items():
        ax.plot(x, w["naive"], color=AMBER, lw=1.2, ls=(0, (3, 1.6)), zorder=2)
        ax.plot(x, w["actual"], color=BLACK, lw=1.2, zorder=3)
    jit = {"2012": -0.3, "2013": -0.1, "2014": 0.1, "2015": 0.3}
    for yr, v in errs.items():
        ax.plot([jit[yr], jit[yr]], [0, v], color=CRIMSON, lw=2.0, zorder=4,
                solid_capstyle="butt")
    ax.set_xlim(-7.3, 7.3)
    ax.set_ylim(0, 4100)
    ax.set_xticks([-7, 0, 7])
    ax.set_xticklabels(["18 Dec", "25 Dec", "1 Jan"], fontsize=11)
    ax.set_xticks(x, minor=True)
    ax.set_yticks([0, 1000, 2000, 3000, 4000])
    ax.set_yticklabels(["0", "1,000", "2,000", "3,000", "4,000"], fontsize=11)
    ax.set_ylabel("units a day", fontsize=11)
    for yr, w in c["windows"].items():                  # where each series touches zero
        for xi, (act, nv) in enumerate(zip(w["actual"], w["naive"])):
            if nv == 0:
                ax.plot([xi - 7], [0], "o", ms=5.5, mfc="white", mec=AMBER, mew=1.4, zorder=5,
                        clip_on=False)
    ax.plot([0], [0], "o", ms=5.5, color=BLACK, zorder=5, clip_on=False)
    ax.annotate("error on 25 Dec:\nthe side of a square", xy=(0.36, 300), xytext=(1.3, 750),
                fontsize=11, color=CRIMSON, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=CRIMSON, lw=1.0, mutation_scale=9,
                                shrinkA=1, shrinkB=1))
    ax.annotate("last year's\nzero, replayed", xy=(-2.12, 70), xytext=(-7.0, 900),
                fontsize=11, color=AMBER, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.0, mutation_scale=9,
                                shrinkA=3, shrinkB=1))
    handles = [Line2D([], [], color=BLACK, lw=1.4, label="actual, 2012–15"),
               Line2D([], [], color=AMBER, lw=1.4, ls=(0, (3, 1.6)),
                      label="seasonal naive (same weekday, a year earlier)")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2,
              frameon=False, fontsize=11, handlelength=1.8, borderaxespad=0.1,
              columnspacing=1.2)

    # ---- right: one average square against the four closure squares, to scale
    ax = fig.add_subplot(gs[1])
    ax.set_aspect("equal"); ax.axis("off")
    gap = 260
    col = 1300                                   # the column the average square sits in
    x0 = col + gap
    spots = {"2014": (x0, 0), "2015": (x0 + errs["2014"] + gap, 0)}
    top = errs["2014"] + gap
    spots["2012"] = (x0, top)
    spots["2013"] = (x0 + errs["2012"] + gap, top)
    for yr, (sx, sy) in spots.items():
        s = errs[yr]
        ax.add_patch(Rectangle((sx, sy), s, s, facecolor=LIGHT_CRIMSON, edgecolor=CRIMSON,
                               lw=1.2, zorder=2))
        for k in np.arange(side, s, side):                    # the average square, tiled
            ax.plot([sx + k, sx + k], [sy, sy + s], color="#F3B6B3", lw=0.6, zorder=3)
            ax.plot([sx, sx + s], [sy + k, sy + k], color="#F3B6B3", lw=0.6, zorder=3)
        ax.text(sx + s / 2, sy + s / 2, f"{yr}\n{s ** 2 / side ** 2:.0f}×", ha="center",
                va="center", fontsize=12, color=CRIMSON, zorder=4, linespacing=1.15)
    W = spots["2015"][0] + errs["2015"]
    H = top + errs["2012"]
    gx, gy = (col - side) / 2, H / 2 - side / 2
    ax.add_patch(Rectangle((gx, gy), side, side, facecolor="#A1A1AA", edgecolor=GRAY, lw=1.0,
                           zorder=2))
    ax.text(col / 2, gy + side + 140, "average\nday", ha="center", va="bottom", fontsize=11,
            color=GRAY, linespacing=1.1)
    ax.text(col / 2, gy - 140, f"RMSE\n{side:,.0f}", ha="center", va="top", fontsize=11,
            color=GRAY, linespacing=1.1)
    ax.text((x0 + W) / 2, -200, "area = squared error", ha="center", va="top", fontsize=11,
            color=GRAY)
    ax.set_xlim(-60, W + 40)
    ax.set_ylim(-60, H + 40)
    save(fig, "christmas_squares")


FIGURES = {"scoreboard": scoreboard, "christmas_squares": christmas_squares}


if __name__ == "__main__":
    if "--cnn-only" in sys.argv:              # the child process measure() starts
        print(json.dumps(cnn_runs()))
        sys.exit(0)
    pick = [a for a in sys.argv[1:] if not a.startswith("--")] or list(FIGURES)
    print(f"Lecture 13 figures -> {OUT.relative_to(ROOT)}   (font: {FONT})")
    if "--retrain" in sys.argv:
        measured()
    for name in pick:
        FIGURES[name]()
