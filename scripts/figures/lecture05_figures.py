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
* oob_grid — that same weekly forest's own bootstrap draws (estimators_samples_), training weeks
  in date order: which trees score a week out-of-bag, and how many later weeks those trees saw.
* boosting_stages — the gradient boosting algorithm exactly as the slide writes it (F_0 = mean,
  fit a shallow tree to the residuals, add eta times it), on one feature so the fit can be drawn:
  weekday electricity load against day of year (data/processed/electricity_daily.csv, the same
  W-shaped year Lecture 7 uses). Depth-3 trees, eta = 0.3, random_state=42.
* bagging_floor — a schematic of Var/sigma^2 = rho + (1 - rho)/B. Nothing is measured: rho is a
  property of the sampling process, and one forest cannot estimate it.

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

    for k, fn in EXTRA.items():
        m[k] = fn()

    OUT.mkdir(parents=True, exist_ok=True)
    MEASURED.write_text(json.dumps(m, indent=1) + "\n")
    print("  measured.json written")
    return m


# ================================================================== OOB, in date order
def measure_oob() -> dict:
    """The weekly forest's bootstrap draws: who scores each week, and what those trees saw."""
    from sklearn.ensemble import RandomForestRegressor
    tr, _ = weekly_data()
    n = len(tr)
    rf = RandomForestRegressor(n_estimators=500, max_features="sqrt", min_samples_leaf=5,
                               oob_score=True, random_state=SEED, n_jobs=-1)
    rf.fit(tr[WEEKLY_FEATURES], tr.units)
    inbag = np.zeros((len(rf.estimators_), n), dtype=bool)
    for t, idx in enumerate(rf.estimators_samples_):
        inbag[t, idx] = True
    # For each week i: the trees that score it (i out of bag), and how many weeks AFTER i each
    # of those trees trained on. Every week but the last is scored by trees that saw its future.
    later_seen = []
    for i in range(n - 1):
        scorers = ~inbag[:, i]
        later_seen.append(float(inbag[scorers, i + 1:].sum(1).mean()) / (n - 1 - i))
    share_scored_by_future = float(np.mean([
        np.all(inbag[~inbag[:, i], i + 1:].any(1)) for i in range(n - 1)]))
    print(f"  OOB: {n} weeks; mean OOB share {1 - inbag.mean():.3f}; scorers saw "
          f"{np.mean(later_seen):.0%} of later weeks on average")
    return {"n_weeks": n, "n_trees": len(rf.estimators_),
            "first_week": str(tr.week_start.iloc[0].date()),
            "last_week": str(tr.week_start.iloc[-1].date()),
            "inbag_first_24_trees": ["".join("1" if v else "0" for v in row) for row in inbag[:24]],
            "oob_share": float(1 - inbag.mean()),
            "scorers_share_of_later_weeks_seen": float(np.mean(later_seen)),
            "share_weeks_all_scorers_saw_a_later_week": share_scored_by_future}


# ================================================================== boosting, stage by stage
BOOST_ETA, BOOST_DEPTH, BOOST_STAGES = 0.3, 3, (1, 2, 5, 50)


def electricity_weekdays() -> tuple[np.ndarray, np.ndarray]:
    e = pd.read_csv(ROOT / "data" / "processed" / "electricity_daily.csv")
    wd = e[e.is_weekend == 0]
    return wd.doy.values.astype(float), wd.mw_mean.values.astype(float) / 1000   # GW


def boost_fits() -> dict:
    """F_b on a day-of-year grid, and the training RMSE, at each drawn stage."""
    from sklearn.tree import DecisionTreeRegressor
    x, y = electricity_weekdays()
    X, grid = x.reshape(-1, 1), np.arange(1, 367, dtype=float).reshape(-1, 1)
    F, Fg = np.full_like(y, y.mean()), np.full(len(grid), y.mean())
    out = {"stages": {}}
    for b in range(1, max(BOOST_STAGES) + 1):
        tree = DecisionTreeRegressor(max_depth=BOOST_DEPTH, random_state=SEED).fit(X, y - F)
        F, Fg = F + BOOST_ETA * tree.predict(X), Fg + BOOST_ETA * tree.predict(grid)
        if b in BOOST_STAGES:
            # the curve on days 1..366, one string so measured.json stays readable
            out["stages"][str(b)] = {"rmse": rmse(y, F),
                                     "curve_days_1_366": " ".join(f"{v:.4f}" for v in Fg)}
    doy_mean = pd.Series(y).groupby(x).transform("mean").values
    out.update(n=int(len(y)), mean=float(y.mean()), rmse_mean_only=rmse(y, np.full_like(y, y.mean())),
               rmse_best_any_doy_function=rmse(y, doy_mean), eta=BOOST_ETA, depth=BOOST_DEPTH)
    print("  boosting: RMSE " + ", ".join(f"{b} trees {v['rmse']:.2f}"
                                         for b, v in out["stages"].items()) + " GW")
    return out


EXTRA = {"oob": measure_oob, "boosting": boost_fits}


@lru_cache(maxsize=None)
def measured() -> dict:
    if "--retrain" in sys.argv or not MEASURED.exists():
        return measure()
    m = json.loads(MEASURED.read_text())
    missing = [k for k in EXTRA if k not in m]
    for k in missing:                      # figures added later measure only what they need
        m[k] = EXTRA[k]()
    if missing:
        MEASURED.write_text(json.dumps(m, indent=1) + "\n")
        print(f"  measured.json: added {', '.join(missing)}")
    return m


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


# ================================================================== bagging's floor
def bagging_floor() -> None:
    """Schematic: Var/sigma^2 = rho + (1 - rho)/B for three correlations, floors dashed."""
    B = np.logspace(0, np.log10(500), 300)
    fig, ax = plt.subplots(figsize=sz(9.6, 4.4))
    cols = {0.9: GRAY, 0.5: AMBER, 0.1: BLUE}
    for rho, col in cols.items():
        ax.plot(B, rho + (1 - rho) / B, color=col, lw=2.2, zorder=3)
        ax.axhline(rho, color=col, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.text(560, rho, f"ρ = {rho}", ha="left", va="center", fontsize=12, color=col,
                fontweight="bold")
    # move 1: more trees slides along a curve (drawn just above the rho = 0.5 curve)
    bb = np.logspace(np.log10(2.2), np.log10(70), 60)
    ax.plot(bb, 0.5 + 0.5 / bb + 0.045, color=BLACK, lw=1.3, zorder=4)
    ax.annotate("", xy=(bb[-1] * 1.25, 0.5 + 0.5 / (bb[-1] * 1.25) + 0.045),
                xytext=(bb[-1], 0.5 + 0.5 / bb[-1] + 0.045),
                arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.3, shrinkA=0, shrinkB=0))
    ax.text(8, 0.70, "more trees: along the curve", ha="left", va="bottom", fontsize=12)
    # move 2: less alike trees drop to a lower curve
    ax.annotate("", xy=(260, 0.1 + 0.9 / 260 + 0.025), xytext=(260, 0.9 + 0.1 / 260 - 0.025),
                arrowprops=dict(arrowstyle="-|>", color=CRIMSON, lw=1.6))
    ax.text(230, 0.32, "less alike trees:\na lower curve", ha="right", va="center",
            fontsize=12, color=CRIMSON)
    ax.set_xscale("log")
    ax.set_xlim(1, 500); ax.set_ylim(0, 1.04)
    ax.set_xticks([1, 10, 100, 500]); ax.set_xticklabels(["1", "10", "100", "500"])
    ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "½", "1"])
    ax.tick_params(labelsize=11.5)
    ax.set_xlabel("trees averaged, B (log scale)", fontsize=12)
    ax.set_ylabel("variance of the average\n(one tree = 1)", fontsize=12)
    ax.set_title("schematic: ρ is assumed, not measured", loc="left", fontsize=11,
                 color=GRAY, style="italic", pad=6)
    save(fig, "bagging_floor")


# ================================================================== the OOB grid
def oob_grid() -> None:
    """Training weeks in date order x 24 trees: the trees that score one week saw its future."""
    m = measured()["oob"]
    inbag = np.array([[c == "1" for c in row] for row in m["inbag_first_24_trees"]])
    T, n = inbag.shape
    target = 100                                    # week 101 of 173 (0-based index 100)
    scorers = ~inbag[:, target]
    # 0 out of bag, 1 in bag, 2 in bag and later than the target, for a tree that scores it
    code = inbag.astype(int)
    later = np.zeros_like(inbag)
    later[:, target + 1:] = True
    code[scorers[:, None] & inbag & later] = 2
    fills = {1: "#C9C9CE", 2: CRIMSON}               # 0 (out of bag) stays blank
    fig, ax = plt.subplots(figsize=sz(10.2, 3.3))
    for t in range(T):                                # vector runs, not a raster: crisp at any size
        row, j = code[t], 0
        while j < n:
            k = j
            while k < n and row[k] == row[j]:
                k += 1
            if row[j]:
                ax.add_patch(plt.Rectangle((j, t + 0.08), k - j, 0.84, facecolor=fills[row[j]],
                                           edgecolor="none", zorder=2))
            j = k
    ax.add_patch(plt.Rectangle((target, 0), 1, T, fill=False, ec=BLACK, lw=1.4, zorder=4))
    for t in np.where(scorers)[0]:
        ax.plot([-1.2], [t + 0.5], marker=">", ms=5, color=CRIMSON, clip_on=False)
    ax.set_xlim(0, n); ax.set_ylim(T, 0)
    years = pd.date_range(m["first_week"], periods=n, freq="7D")
    ticks = [i for i in range(n) if years[i].month == 1 and years[i].day <= 7]
    ax.set_xticks(ticks); ax.set_xticklabels([str(years[i].year) for i in ticks])
    ax.set_yticks([])
    ax.tick_params(labelsize=11)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.set_ylabel(f"{T} of the {m['n_trees']} trees", fontsize=12, labelpad=12)
    ax.set_xlabel(f"the {n} training weeks, in date order", fontsize=12)
    k = int(scorers.sum())
    wk = years[target]
    ax.text(0, -5.3, f"week of {wk.day} {wk:%b %Y} (boxed): out of bag for the {k} marked trees",
            ha="left", va="bottom", fontsize=11.5)
    ax.text(0, -2.6, "red: the later weeks those trees trained on", ha="left", va="bottom",
            fontsize=11.5, color=CRIMSON)
    save(fig, "oob_grid")


# ================================================================== boosting stages
def boosting_stages() -> None:
    """The fitted curve after 1, 2, 5 and 50 trees, against the data it is chasing."""
    m = measured()["boosting"]
    x, y = electricity_weekdays()
    grid = np.arange(1, 367)
    fig, axes = plt.subplots(1, 4, figsize=sz(12.4, 3.6), sharey=True,
                             gridspec_kw=dict(wspace=0.08))
    for ax, b in zip(axes, BOOST_STAGES):
        st = m["stages"][str(b)]
        ax.scatter(x, y, s=1.2, color="#BDBDC2", lw=0, rasterized=True, zorder=1)
        ax.axhline(m["mean"], color=GRAY, lw=0.9, ls=(0, (3, 3)), zorder=2)
        ax.step(grid, [float(v) for v in st["curve_days_1_366"].split()], where="mid", color=CRIMSON, lw=2.0, zorder=3)
        ax.set_title(f"{b} tree{'s' if b > 1 else ''}", fontsize=12.5, pad=3)
        ax.text(360, 21.5, f"RMSE {st['rmse']:.2f}", ha="right", va="bottom", fontsize=11.5)
        ax.set_xlim(1, 366); ax.set_ylim(21, 47)
        ax.set_xticks([1, 182]); ax.set_xticklabels(["Jan", "Jul"])
        ax.tick_params(labelsize=11)
    axes[0].set_ylabel("load (GW)", fontsize=12)
    axes[0].set_yticks([25, 35, 45])
    save(fig, "boosting_stages")


FIGURES = {"root_splits": root_splits, "extrapolation": extrapolation,
           "bagging_floor": bagging_floor, "oob_grid": oob_grid,
           "boosting_stages": boosting_stages}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if not a.startswith("--")] or list(FIGURES)
    print(f"Lecture 5 figures -> {OUT.relative_to(ROOT)}   (font: {FONT})")
    if "--retrain" in sys.argv:
        measured()
    for name in pick:
        FIGURES[name]()
