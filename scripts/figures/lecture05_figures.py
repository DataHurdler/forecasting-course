#!/usr/bin/env python3
"""Figures for Lecture 5 (Tree Ensembles) — one script, every figure, both formats.

    .venv/bin/python scripts/figures/lecture05_figures.py                  # all
    .venv/bin/python scripts/figures/lecture05_figures.py root_splits      # one, by name
    .venv/bin/python scripts/figures/lecture05_figures.py --retrain        # re-measure

Writes Figures/Lecture05_TreeEnsembles/<name>.pdf (Beamer) and <name>.svg (Quarto), both vector.

What is measured, and on whose pipeline:

* root_splits — Lab 5's own data and split (Labs/Lecture05_lab.qmd, Setup): daily CA_1 FOODS, ten
  features, the final 365 days held out. Two 300-tree forests, Step 3's max_features=None (plain
  bagging, which is also scikit-learn's regressor default) and "sqrt", random_state=42. The test
  RMSEs must equal the lab's printed 248 and 254 before anything is drawn.
* extrapolation — RSXFS (data/processed/fred_monthly.csv), a forest with the deck's own template
  settings, trained through December 2019 on lags 1, 2, 3, 12 and the month, then asked for each
  later month one step ahead, with the true lags supplied.
* The weekly table on the "Random Forest in Python" slide and the importance table before it —
  weekly CA_1 FOODS, the eight numeric columns of m5_weekly.csv, the final 52 weeks held out (Lab
  1's test year, whose seasonal naive is the course's 1,660). The forest is the slide's own code
  template; the single deep tree is DecisionTreeRegressor(random_state=42).

Everything printed goes to measured.json; later runs redraw from it.
"""
from __future__ import annotations

import collections
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
OUT = ROOT / "Figures" / "Lecture05_TreeEnsembles"
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


SEED = 42
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ================================================================== Lab 5, verbatim
# Labs/Lecture05_lab.qmd, Setup. Kept identical on purpose: the row order decides the split.
LAB_FEATURES = ["units_lag1", "units_lag7", "units_lag14", "units_lag28", "units_lag365",
                "snap", "is_event", "dow", "doy", "avg_price"]
LAB_RMSE = {"None": 248, "sqrt": 254}            # Lab 5, Step 3, as rendered


def lab5_data():
    daily = pd.read_csv(ROOT / "data" / "processed" / "m5_daily.csv", parse_dates=["date"])
    daily["dow"] = daily["date"].dt.dayofweek
    daily["doy"] = daily["date"].dt.dayofyear
    df = daily.dropna(subset=LAB_FEATURES).copy()
    train, test = df.iloc[:-365], df.iloc[-365:]
    return (train[LAB_FEATURES].values, train["units"].values,
            test[LAB_FEATURES].values, test["units"].values)


# ================================================================== the weekly tables
WEEKLY_FEATURES = ["units_lag1", "units_lag2", "units_lag3", "units_lag4", "units_lag52",
                   "snap_days", "event_days", "avg_price"]


def weekly_data():
    w = pd.read_csv(ROOT / "data" / "processed" / "m5_weekly.csv", parse_dates=["week_start"])
    g = (w[(w.store_id == "CA_1") & (w.cat_id == "FOODS")]
         .sort_values("week_start").dropna(subset=WEEKLY_FEATURES).reset_index(drop=True))
    return g.iloc[:-52], g.iloc[-52:]


# ================================================================== RSXFS
RSXFS_FEATURES = ["lag1", "lag2", "lag3", "lag12", "month"]
RSXFS_TRAIN_END = "2019-12-01"


def rsxfs_frame() -> pd.DataFrame:
    f = pd.read_csv(ROOT / "data" / "processed" / "fred_monthly.csv", parse_dates=["date"])
    d = f.set_index("date")[["RSXFS"]].rename(columns={"RSXFS": "y"})
    for lag in (1, 2, 3, 12):
        d[f"lag{lag}"] = d["y"].shift(lag)
    d["month"] = d.index.month
    return d


def measure() -> dict:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    m: dict = {"note": "Written by scripts/figures/lecture05_figures.py: every number Lecture 5 "
                       "prints from a fitted ensemble. random_state=42 throughout."}

    # --- root splits: Lab 5, Step 3's two extremes
    Xtr, ytr, Xte, yte = lab5_data()
    m["roots"] = {}
    for mf in (None, "sqrt"):
        rf = RandomForestRegressor(n_estimators=300, max_features=mf, random_state=SEED,
                                   n_jobs=-1).fit(Xtr, ytr)
        r = rmse(yte, rf.predict(Xte))
        assert round(r) == LAB_RMSE[str(mf)], f"max_features={mf}: {r:.1f}, lab prints {LAB_RMSE[str(mf)]}"
        roots = collections.Counter(LAB_FEATURES[e.tree_.feature[0]] for e in rf.estimators_)
        m["roots"][str(mf)] = {"test_rmse": r, "counts": {f: roots.get(f, 0) for f in LAB_FEATURES},
                               "mean_single_tree_rmse": float(np.mean(
                                   [rmse(yte, e.predict(Xte)) for e in rf.estimators_]))}
        print(f"  max_features={mf}: RMSE {r:.1f}; roots {roots.most_common(3)}")

    # --- the weekly tables (Random Forest in Python; RF feature importance)
    tr, te = weekly_data()
    X, y, Xt, yt = tr[WEEKLY_FEATURES], tr.units, te[WEEKLY_FEATURES], te.units
    tree = DecisionTreeRegressor(random_state=SEED).fit(X, y)
    rf = RandomForestRegressor(n_estimators=500, max_features="sqrt", min_samples_leaf=5,
                               oob_score=True, random_state=SEED, n_jobs=-1).fit(X, y)
    m["weekly"] = {"train_weeks": len(tr), "test_first": str(te.week_start.iloc[0].date()),
                   "test_last": str(te.week_start.iloc[-1].date()),
                   "seasonal_naive": rmse(yt, te.units_lag52),
                   "deep_tree": rmse(yt, tree.predict(Xt)),
                   "random_forest": rmse(yt, rf.predict(Xt)),
                   "rf_importance": dict(sorted(zip(WEEKLY_FEATURES, map(float, rf.feature_importances_)),
                                                key=lambda kv: -kv[1])),
                   "oob_share_expected": (1 - 1 / len(tr)) ** len(tr)}
    print(f"  weekly: seasonal naive {m['weekly']['seasonal_naive']:,.0f}, deep tree "
          f"{m['weekly']['deep_tree']:,.0f}, forest {m['weekly']['random_forest']:,.0f}")

    # --- RSXFS: a forest cannot leave its training range
    d = rsxfs_frame().dropna()
    trn, tst = d[d.index <= RSXFS_TRAIN_END], d[d.index > RSXFS_TRAIN_END]
    rfx = RandomForestRegressor(n_estimators=500, max_features="sqrt", min_samples_leaf=5,
                                random_state=SEED, n_jobs=-1).fit(trn[RSXFS_FEATURES], trn.y)
    pred = rfx.predict(tst[RSXFS_FEATURES])
    lo, hi = float(trn.y.min()), float(trn.y.max())
    assert lo <= pred.min() and pred.max() <= hi, "a forest prediction left the training range"
    m["rsxfs"] = {"train_first": str(trn.index[0].date()), "train_last": str(trn.index[-1].date()),
                  "train_min": lo, "train_max": hi,
                  "dates": [str(t.date()) for t in tst.index], "actual": tst.y.tolist(),
                  "pred": pred.tolist(), "share_above_max": float((tst.y > hi).mean())}
    print(f"  RSXFS: training max {hi:,.0f}; forest max {pred.max():,.0f}; last actual "
          f"{tst.y.iloc[-1]:,.0f}; {m['rsxfs']['share_above_max']:.0%} of months above the max")

    OUT.mkdir(parents=True, exist_ok=True)
    MEASURED.write_text(json.dumps(m, indent=1) + "\n")
    print("  measured.json written")
    return m


@lru_cache(maxsize=None)
def measured() -> dict:
    if "--retrain" in sys.argv or not MEASURED.exists():
        return measure()
    return json.loads(MEASURED.read_text())


# ================================================================== root splits
def root_splits() -> None:
    """Which feature each of 300 trees splits on first: bagging vs a random forest."""
    m = measured()["roots"]
    bag, rf = m["None"], m["sqrt"]
    # features no tree chose first in either forest are left off (is_event and avg_price here)
    order = sorted([f for f in LAB_FEATURES if rf["counts"][f] + bag["counts"][f]],
                   key=lambda f: (-rf["counts"][f], -bag["counts"][f], f))
    y = np.arange(len(order))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=sz(11.2, 3.75), sharey=True,
                             gridspec_kw=dict(wspace=0.08))
    for ax, res, title in [(axes[0], bag, "bagging: all 10 per split"),
                           (axes[1], rf, "random forest: 3 of 10")]:
        counts = np.array([res["counts"][f] for f in order])
        cols = [CRIMSON if f == "dow" else GRAY for f in order]
        ax.barh(y, counts, color=cols, height=0.66)
        for yy, c in zip(y, counts):
            if c:
                ax.text(c + 5, yy, f"{c}", ha="left", va="center", fontsize=12)
        ax.set_xlim(0, 345); ax.set_xticks([0, 100, 200, 300])
        ax.tick_params(labelsize=11.5)
        ax.set_title(title, fontsize=12.5, pad=4)
        ax.text(340, y[-1] - 0.3, f"test RMSE {res['test_rmse']:.0f}\none tree, on average: "
                                  f"{res['mean_single_tree_rmse']:.0f}",
                ha="right", va="bottom", fontsize=12, color=BLACK,
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order, fontsize=11.5, family="monospace")
    axes[0].set_ylabel("feature at the root", fontsize=12)
    for ax in axes:
        ax.set_xlabel("trees, of 300", fontsize=12)
    save(fig, "root_splits")


# ================================================================== extrapolation
def extrapolation() -> None:
    """RSXFS after 2019: the forest stays under its training maximum while sales climb."""
    m = measured()["rsxfs"]
    d = rsxfs_frame()
    hist = d.loc["2012-01-01":RSXFS_TRAIN_END, "y"] / 1000
    dates = pd.to_datetime(m["dates"])
    act = np.array(m["actual"]) / 1000
    pred = np.array(m["pred"]) / 1000
    hi = m["train_max"] / 1000

    fig, ax = plt.subplots(figsize=sz(10.2, 4.1))
    ax.axvspan(pd.Timestamp("2012-01-01"), pd.Timestamp(RSXFS_TRAIN_END), color=LIGHT_GRAY,
               zorder=0, lw=0)
    ax.plot(hist.index, hist.values, color=BLACK, lw=1.6, zorder=2)
    ax.plot(dates, act, color=BLACK, lw=1.6, zorder=2)
    ax.plot(dates, pred, color=CRIMSON, lw=2.0, zorder=3)
    ax.axhline(hi, color=CRIMSON, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(pd.Timestamp("2012-04-01"), hi + 9, f"highest month in training: ${hi:,.0f}bn",
            ha="left", va="bottom", fontsize=11, color=CRIMSON)
    ax.text(pd.Timestamp("2019-09-01"), 335, "trained on\n1993–2019", ha="right", va="bottom",
            fontsize=11, color=GRAY)
    ax.text(dates[-1], act[-1] + 12, f"actual ${act[-1]:,.0f}bn", ha="right", va="bottom",
            fontsize=11)
    ax.text(pd.Timestamp("2026-06-01"), pred[-12:].mean() - 14, "forest forecast",
            ha="right", va="top", fontsize=11, color=CRIMSON, fontweight="bold")
    ax.set_xlim(pd.Timestamp("2012-01-01"), pd.Timestamp("2026-09-01"))
    ax.set_ylim(320, 700)
    ax.set_ylabel("retail sales\n($bn a month)", fontsize=11)
    ax.tick_params(labelsize=10.5)
    save(fig, "extrapolation")


FIGURES = {"root_splits": root_splits, "extrapolation": extrapolation}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if not a.startswith("--")] or list(FIGURES)
    print(f"Lecture 5 figures -> {OUT.relative_to(ROOT)}   (font: {FONT})")
    if "--retrain" in sys.argv:
        measured()
    for name in pick:
        FIGURES[name]()
