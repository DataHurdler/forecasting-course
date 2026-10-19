#!/usr/bin/env python3
"""Figures for Lecture 2 (ARIMA, VAR & Multivariate Models) — one script, every figure, both formats.

    .venv/bin/python scripts/figures/lecture02_figures.py                  # all
    .venv/bin/python scripts/figures/lecture02_figures.py acf_levels_vs_diff         # one, by name

Writes Figures/Lecture02_ARIMA_VAR/<name>.pdf (Beamer) and <name>.svg (Quarto), both vector.

Nothing here is trained or sampled. The theory panels are exact (statsmodels ArmaProcess); the
real-data panels read m5_weekly.csv with the labs' own get_series(), m5_daily.csv (the SARIMA
frame) and fred_monthly.csv with Lab 2 Step 6's own Granger calls. Every number a figure prints
is also written to Figures/Lecture02_ARIMA_VAR/measured.json, so the deck and narration can be
checked against it; measured() asserts the deck's ADF and Lab 2's F statistics reproduce first.

The real series is TX_1 HOUSEHOLD (Lab 1's series), not CA_1 HOUSEHOLD: Lab 2 asks students to
read CA_1's ACF themselves and propose an order, so the deck must not show it.
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Figures" / "Lecture02_ARIMA_VAR"
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


# ================================================================== measurement
PHI, THETA = 0.7, 0.6            # the theory panels: AR(1) phi = 0.7, MA(1) theta = 0.6
LAGS = 10                        # lags shown in the theory panels
REAL_LAGS = 26                   # lags shown for the real series (half a year)
SHOCK_PHIS = (0.5, 0.9, 1.0)     # unit-root frame
SHOCK_T = 20
ROLL = 52                        # stationarity frame: centred rolling window, weeks
DAILY_LAGS = 35                  # SARIMA frame: five weeks of daily lags
COVID = ("2020-02-01", "2021-06-01")   # Lab 2, Step 6: the window it drops


@lru_cache(maxsize=None)
def series(store: str = "TX_1", cat: str = "HOUSEHOLD") -> pd.Series:
    """The labs' get_series(), verbatim in effect (Labs/Lecture01_lab.qmd, Lecture02_lab.qmd)."""
    weekly = pd.read_csv(ROOT / "data" / "processed" / "m5_weekly.csv", parse_dates=["week_start"])
    return (weekly[(weekly.store_id == store) & (weekly.cat_id == cat)]
            .set_index("week_start")["units"].sort_index().astype(float))


@lru_cache(maxsize=None)
def measured() -> dict:
    from statsmodels.tsa.arima_process import ArmaProcess
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    from statsmodels.tsa.stattools import acf, adfuller

    ar = ArmaProcess(ar=[1, -PHI], ma=[1])
    ma = ArmaProcess(ar=[1], ma=[1, THETA])
    y = series()
    dy = y.diff().dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Lab 1, Step 3: SES fitted on everything but the last 52 weeks.
        alpha = float(SimpleExpSmoothing(y.iloc[:-52]).fit().params["smoothing_level"])
    theta_ses = alpha - 1                       # ARIMA(0,1,1) theta that SES implies
    m = {
        "note": "Written by scripts/figures/lecture02_figures.py; every number the Lecture 2 "
                "figures print. Theory panels are exact (ArmaProcess); the real series is "
                "TX_1 HOUSEHOLD weekly units from m5_weekly.csv.",
        "phi": PHI, "theta": THETA,
        "ar1_acf": ar.acf(LAGS + 1)[1:].tolist(),
        "ar1_pacf": ar.pacf(LAGS + 1)[1:].tolist(),
        "ma1_acf": ma.acf(LAGS + 1)[1:].tolist(),
        "ma1_pacf": ma.pacf(LAGS + 1)[1:].tolist(),
        "ma1_rho1_formula": THETA / (1 + THETA ** 2),
        "real_series": "TX_1 HOUSEHOLD", "real_weeks": int(len(y)),
        "levels_acf": acf(y, nlags=REAL_LAGS)[1:].tolist(),
        "diff_acf": acf(dy, nlags=REAL_LAGS)[1:].tolist(),
        "levels_band": 1.96 / math.sqrt(len(y)),
        "diff_band": 1.96 / math.sqrt(len(dy)),
        "adf_levels": float(adfuller(y, autolag="AIC")[0]),
        "adf_diff": float(adfuller(dy, autolag="AIC")[0]),
        "ses_alpha_lab1": alpha,
        "ses_implied_rho1": theta_ses / (1 + theta_ses ** 2),
        "half_lives": {str(p): (math.log(0.5) / math.log(p) if p < 1 else None) for p in SHOCK_PHIS},
    }
    m["diff_max_after_lag1"] = max(abs(v) for v in m["diff_acf"][1:])

    # ---- stationarity frame: rolling mean and +/- 2 sd over a centred 52-week window
    for key, s_ in (("lev", y), ("dif", dy)):
        roll = s_.rolling(ROLL, center=True)
        m[f"band_{key}"] = {"dates": [d.strftime("%Y-%m-%d") for d in s_.index],
                            "values": s_.values.tolist(),
                            "mean": roll.mean().tolist(), "sd": roll.std().tolist()}
        m[f"band_{key}_mean_range"] = [float(roll.mean().min()), float(roll.mean().max())]
        m[f"band_{key}_sd_range"] = [float(roll.std().min()), float(roll.std().max())]

    # ---- SARIMA frame: CA_1 FOODS daily (the only daily series), before and after a lag-7 difference
    daily = (pd.read_csv(ROOT / "data" / "processed" / "m5_daily.csv", parse_dates=["date"])
             .set_index("date")["units"].astype(float))
    d7 = daily.diff(7).dropna()
    m["daily_acf"] = acf(daily, nlags=DAILY_LAGS)[1:].tolist()
    m["daily_d7_acf"] = acf(d7, nlags=DAILY_LAGS)[1:].tolist()
    m["daily_band"] = 1.96 / math.sqrt(len(daily))
    m["daily_d7_band"] = 1.96 / math.sqrt(len(d7))
    m["daily_n"] = int(len(daily))
    # the bump near lag 30 after the difference: the SNAP calendar (CA's SNAP days are the 1st-10th
    # of every month). Remove a SNAP dummy first and it goes away.
    import statsmodels.api as sm
    dd = pd.read_csv(ROOT / "data" / "processed" / "m5_daily.csv", parse_dates=["date"]).set_index("date")
    X = sm.add_constant(dd[["snap"]].astype(float))
    resid = dd["units"].astype(float) - sm.OLS(dd["units"].astype(float), X).fit().fittedvalues
    m["daily_d7_acf_snap_removed"] = acf(resid.diff(7).dropna(), nlags=DAILY_LAGS)[1:].tolist()
    m["snap_days_of_month"] = sorted(int(v) for v in dd[dd.snap == 1].index.day.unique())

    # ---- Granger frame: Lab 2, Step 6, verbatim in effect (lag 4, ssr F-test)
    import io
    from contextlib import redirect_stdout
    from statsmodels.tsa.stattools import grangercausalitytests
    fred = pd.read_csv(ROOT / "data" / "processed" / "fred_monthly.csv", parse_dates=["date"]).set_index("date")
    macro = fred[["retail_growth", "unrate_diff"]].dropna()
    ex = macro[(macro.index < COVID[0]) | (macro.index > COVID[1])]

    def granger(effect, cause, data, lag=4):
        with redirect_stdout(io.StringIO()):
            res = grangercausalitytests(data[[effect, cause]], maxlag=lag)
        F, p, df_den, df_num = res[lag][0]["ssr_ftest"]
        return {"F": float(F), "p": float(p), "df": [int(df_num), float(df_den)]}

    def granger_excluding(effect, cause, data, start, end, lag=4):
        # Lags are built on the unbroken series, then every row whose target or any lag falls in
        # [start, end] is dropped. Dropping the months first would splice Jan 2020 onto Jul 2021
        # (full-course review m8, 2026-10-19). On the full sample this equals grangercausalitytests.
        import statsmodels.api as sm
        from scipy.stats import f as f_dist
        d = pd.DataFrame({"y": data[effect]})
        for k in range(1, lag + 1):
            d[f"y{k}"], d[f"x{k}"] = data[effect].shift(k), data[cause].shift(k)
        d = d.dropna()
        d = d[(d.index < start) | (d.index > pd.Timestamp(end) + pd.DateOffset(months=lag))]
        own = sm.OLS(d["y"], sm.add_constant(d[[f"y{k}" for k in range(1, lag + 1)]])).fit()
        both = sm.OLS(d["y"], sm.add_constant(d.drop(columns="y"))).fit()
        F = ((own.ssr - both.ssr) / lag) / (both.ssr / both.df_resid)
        return {"F": float(F), "p": float(f_dist.sf(F, lag, both.df_resid)), "df": [lag, float(both.df_resid)]}

    m["granger"] = {
        "full": {"n": int(len(macro)), "u_to_r": granger("retail_growth", "unrate_diff", macro),
                 "r_to_u": granger("unrate_diff", "retail_growth", macro)},
        "ex_covid": {"n": int(len(ex)),
                     "u_to_r": granger_excluding("retail_growth", "unrate_diff", macro, *COVID),
                     "r_to_u": granger_excluding("unrate_diff", "retail_growth", macro, *COVID)}}
    from scipy.stats import f as f_dist
    df_num, df_den = m["granger"]["full"]["u_to_r"]["df"]
    m["granger_f_crit_5pct"] = float(f_dist.ppf(0.95, df_num, df_den))
    pre = macro[macro.index < COVID[0]]
    win = macro.loc[COVID[0]:COVID[1]]
    m["covid_multiple_of_pre_sd"] = {c: float(win[c].abs().max() / pre[c].std()) for c in macro}
    m["macro"] = {"dates": [d.strftime("%Y-%m-%d") for d in macro.index],
                  "retail_growth": macro["retail_growth"].tolist(),
                  "unrate_diff": macro["unrate_diff"].tolist()}
    # The shapes the slides describe must hold before anything is drawn.
    assert abs(m["ma1_acf"][0] - m["ma1_rho1_formula"]) < 1e-12 and abs(m["ma1_acf"][0]) <= 0.5
    assert all(abs(v) < 1e-12 for v in m["ma1_acf"][1:]), "MA(1) ACF must cut off after lag 1"
    assert all(abs(v) < 1e-12 for v in m["ar1_pacf"][1:]), "AR(1) PACF must cut off after lag 1"
    assert np.allclose(m["ar1_acf"], PHI ** np.arange(1, LAGS + 1)), "AR(1) ACF is phi^k"
    # ---- reproduce what the deck and Lab 2 already print, before anything is drawn
    g = m["granger"]
    for got, want in [(round(m["adf_levels"], 2), -1.38), (round(m["adf_diff"], 2), -8.51),
                      (round(g["full"]["u_to_r"]["F"], 2), 18.22), (round(g["full"]["r_to_u"]["F"], 2), 12.80),
                      (round(g["ex_covid"]["u_to_r"]["F"], 2), 2.42), (round(g["ex_covid"]["r_to_u"]["F"], 2), 9.96),
                      (g["full"]["n"], 412), (g["ex_covid"]["n"], 395)]:
        assert got == want, f"does not reproduce: got {got}, expected {want}"
    # the slide's claim: the seasonal peaks sit at 7, 14, 21, 28 before the difference
    dacf = np.array(m["daily_acf"])
    for k in (7, 14, 21, 28):
        assert dacf[k - 1] == dacf[k - 4:k + 3].max(), f"lag {k} is not a local peak"
    OUT.mkdir(parents=True, exist_ok=True)
    # the raw series behind the band and Granger figures are re-read every run; keep them out of
    # the file, which records the numbers the figures print
    bulky = ("band_lev", "band_dif", "macro")
    MEASURED.write_text(json.dumps({k: v for k, v in m.items() if k not in bulky}, indent=1) + "\n")
    return m


# ================================================================== helpers
def stems(ax, lags, vals, color=CRIMSON, lw=2.4, ms=5):
    ax.vlines(lags, 0, vals, color=color, lw=lw, zorder=3)
    ax.plot(lags, vals, "o", color=color, ms=ms, zorder=4)


def corr_axis(ax, xmax, yticks=(-1, -0.5, 0, 0.5, 1), labels=True):
    ax.axhline(0, color=BLACK, lw=0.9, zorder=2)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlim(0.3, xmax + 0.6)
    ax.set_yticks(yticks)
    if labels:
        ax.set_yticklabels([f"{t:g}".replace("-", "−") for t in yticks], fontsize=11)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="x", labelsize=11, length=3)
    ax.tick_params(axis="y", length=3)
    for t in yticks:
        if t not in (0,):
            ax.axhline(t, color=LIGHT_GRAY, lw=0.8, zorder=1)


# ================================================================== the 2x2, exact
def acf_theory() -> None:
    """AR(1) and MA(1), ACF and PACF, from ArmaProcess: exact values, no sampling noise."""
    m = measured()
    lags = np.arange(1, LAGS + 1)
    fig, axes = plt.subplots(2, 2, figsize=sz(12.5, 6.0), sharex=True)
    fig.subplots_adjust(hspace=0.3, wspace=0.12, left=0.13, right=0.99, top=0.92, bottom=0.12)
    panels = [(0, 0, m["ar1_acf"]), (0, 1, m["ar1_pacf"]), (1, 0, m["ma1_acf"]), (1, 1, m["ma1_pacf"])]
    for r, c, v in panels:
        ax = axes[r, c]
        corr_axis(ax, LAGS, labels=(c == 0))
        stems(ax, lags, v)
        ax.set_xticks([1, 2, 4, 6, 8, 10])
    # column and row headings, as in the table on the previous slide
    for c, head in enumerate(("ACF", "PACF")):
        bb = axes[0, c].get_position()
        fig.text((bb.x0 + bb.x1) / 2, 0.94, head, ha="center", va="bottom", fontsize=13,
                 fontweight="bold")
    for r, head in enumerate((f"AR(1)\nφ = {PHI}", f"MA(1)\nθ = {THETA}")):
        bb = axes[r, 0].get_position()
        fig.text(0.005, (bb.y0 + bb.y1) / 2, head, ha="left", va="center", fontsize=12,
                 fontweight="bold")
    for c in (0, 1):
        axes[1, c].set_xlabel("lag", fontsize=11, labelpad=1)

    # what each panel says, in the table's words
    def note(ax, text, color=BLACK, x=0.97, y=0.93):
        ax.text(x, y, text, transform=ax.transAxes, ha="right", va="top", fontsize=11, color=color)

    a = axes[0, 0]                                   # AR ACF: the staircase phi^k
    xx = np.linspace(1, LAGS, 200)
    a.plot(xx, PHI ** xx, color=GRAY, lw=1.0, ls=(0, (2, 2)), zorder=2)
    note(a, f"tails off: {PHI}$^k$,\n×{PHI} every lag", GRAY)
    note(axes[0, 1], f"cuts off after lag 1\n(spike = φ = {PHI})", GREEN)
    r1 = m["ma1_acf"][0]
    a = axes[1, 0]                                   # no MA(1) has |rho_1| > 0.5
    a.plot([0.5, LAGS + 0.4], [0.5, 0.5], color=GREEN, lw=1.0, ls=(0, (3, 2)), zorder=2)
    note(a, f"cuts off after lag 1", GREEN)
    a.text(0.97, 0.08, f"spike {r1:.2f}: no MA(1)\ngoes above the dashed 0.5", transform=a.transAxes,
           ha="right", va="bottom", fontsize=11, color=GREEN)
    note(axes[1, 1], "tails off,\nflipping sign", GRAY, y=0.93)
    save(fig, "acf_theory")


# ================================================================== a real series
def acf_levels_vs_diff() -> None:
    """TX_1 HOUSEHOLD: the ACF of the levels crawls; the ACF of the change has one spike."""
    m = measured()
    lags = np.arange(1, REAL_LAGS + 1)
    fig, axes = plt.subplots(1, 2, figsize=sz(13, 4.4))
    fig.subplots_adjust(wspace=0.1, left=0.09, right=0.99, top=0.78, bottom=0.17)
    for ax, key, band, adf, title in [
            (axes[0], "levels_acf", m["levels_band"], m["adf_levels"], "weekly units (levels)"),
            (axes[1], "diff_acf", m["diff_band"], m["adf_diff"], "week-to-week change")]:
        corr_axis(ax, REAL_LAGS, labels=(ax is axes[0]))
        ax.axhspan(-band, band, color=LIGHT_BLUE, zorder=1, lw=0)
        stems(ax, lags, m[key], lw=1.9, ms=3.6)
        ax.set_xticks([1, 4, 8, 13, 26])
        ax.set_xlabel("lag (weeks)", fontsize=11, labelpad=1)
        verdict = "unit root" if adf > -2.87 else "stationary"   # 5% critical value ≈ −2.87
        ax.set_title(f"{title}\nADF {adf:.2f}: {verdict}".replace("-", "−"), fontsize=12, pad=5)
    lv, dv = m["levels_acf"], m["diff_acf"]
    rest = m["diff_max_after_lag1"]
    axes[0].text(REAL_LAGS + 0.4, -0.22, f"{lv[0]:.2f} at lag 1, still {lv[-1]:.2f} at lag 26:\n"
                 "crawls down, says nothing\nabout p or q", ha="right", va="top", fontsize=11,
                 color=GRAY)
    axes[1].annotate(f"{dv[0]:.2f} at lag 1: the one big spike".replace("-", "−"),
                     xy=(1.3, dv[0]), xytext=(3.6, -0.5), fontsize=11, color=CRIMSON, ha="left",
                     va="center", arrowprops=dict(arrowstyle="-|>", color=CRIMSON, lw=1.2))
    bound = math.ceil(rest * 100) / 100          # round the bound up, so "within" stays true
    axes[1].text(3.6, -0.8, f"every other lag within ±{bound:.2f}", fontsize=11, color=GRAY,
                 ha="left", va="center")
    axes[1].text(REAL_LAGS + 0.4, 0.97, "shaded: ±1.96/√n,\nwhat pure noise shows", ha="right",
                 va="top", fontsize=11, color=BLUE)
    save(fig, "acf_levels_vs_diff")


# ================================================================== one shock, three phis
def shock_response() -> None:
    """y_t = phi y_{t-1} + e_t, one unit shock at t = 0: what is left of it t periods later."""
    m = measured()
    t = np.arange(0, SHOCK_T + 1)
    fig, ax = plt.subplots(figsize=sz(11.5, 4.6))
    fig.subplots_adjust(left=0.1, right=0.83, top=0.95, bottom=0.2)
    colors = {0.5: BLUE, 0.9: AMBER, 1.0: CRIMSON}
    ax.axhline(0.5, color=GRAY, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.text(SHOCK_T + 0.3, 0.5, "half the\nshock", ha="left", va="center", fontsize=11, color=GRAY)
    for p in SHOCK_PHIS:
        resp = p ** t
        ax.plot(t, resp, "-o", color=colors[p], lw=1.8, ms=4, zorder=3)
    hl5, hl9 = m["half_lives"]["0.5"], m["half_lives"]["0.9"]
    for hl, p, dy in [(hl5, 0.5, -0.0), (hl9, 0.9, 0.0)]:
        ax.plot([hl, hl], [0, 0.5], color=colors[p], lw=1.2, ls=(0, (1, 1.5)), zorder=2)
        ax.plot([hl], [0.5], "o", mfc="white", mec=colors[p], mew=1.6, ms=7, zorder=4)
    ax.annotate(f"half-life\n{hl5:.0f} period", xy=(hl5 + 0.15, 0.47), xytext=(2.7, 0.335),
                ha="left", va="center", fontsize=11, color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.9, shrinkA=1, shrinkB=3))
    ax.text(hl9 + 0.35, 0.06, f"half-life\n{hl9:.1f} periods", ha="left", va="bottom", fontsize=11,
            color=AMBER)
    # labels at the right-hand end of each line
    ax.text(SHOCK_T + 0.3, 1.0, "φ = 1: never halves.\nPermanent.", ha="left", va="center",
            fontsize=11, color=CRIMSON, fontweight="bold")
    ax.text(SHOCK_T + 0.3, 0.9 ** SHOCK_T + 0.02, "φ = 0.9", ha="left", va="center", fontsize=11,
            color=AMBER)
    ax.text(SHOCK_T + 0.3, 0.06 - 0.06, "φ = 0.5", ha="left", va="bottom", fontsize=11, color=BLUE)
    ax.set_xlim(-0.6, SHOCK_T + 0.2)
    ax.set_ylim(-0.04, 1.12)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=11)
    ax.set_xlabel("periods after the shock", fontsize=11)
    ax.set_ylabel("share of the shock left", fontsize=11)
    save(fig, "shock_response")


# ================================================================== stationarity: a band of fixed width
def stationarity_band() -> None:
    """TX_1 HOUSEHOLD, levels above the week-to-week change, each with a centred 52-week rolling
    mean and a +/- 2 rolling-sd band. Conditions 1-2 of weak stationarity (constant mean, constant
    variance) say the band is horizontal and keeps one width; the levels' band climbs and widens,
    the change's does not. ADF statistics as in Lab 2 (autolag AIC)."""
    import matplotlib.dates as mdates
    m = measured()
    fig, axes = plt.subplots(2, 1, figsize=sz(8.2, 6.0), sharex=True,
                             gridspec_kw={"hspace": 0.5})
    for ax, key, adf, title, c in [
            (axes[0], "band_lev", m["adf_levels"], "weekly units (levels)", CRIMSON),
            (axes[1], "band_dif", m["adf_diff"], "week-to-week change", BLUE)]:
        b = m[key]
        d = pd.to_datetime(b["dates"]); v = np.array(b["values"])
        mu = np.array(b["mean"], dtype=float); sd = np.array(b["sd"], dtype=float)
        ax.fill_between(d, mu - 2 * sd, mu + 2 * sd, color=c, alpha=0.13, lw=0, zorder=1)
        ax.plot(d, v, color="#A3A3A3", lw=0.8, zorder=2)
        ax.plot(d, mu, color=c, lw=2.0, zorder=3)
        ax.plot(d, mu + 2 * sd, color=c, lw=0.8, zorder=3)
        ax.plot(d, mu - 2 * sd, color=c, lw=0.8, zorder=3)
        verdict = "unit root" if adf > -2.87 else "stationary"   # 5% critical value ≈ −2.87
        ax.set_title(f"{title}: ADF {adf:.2f}, {verdict}".replace(" -", " −"), fontsize=12,
                     loc="left", pad=4, color=c)
        ax.tick_params(labelsize=11)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}".replace("-", "−")))
    axes[0].set_yticks([2000, 4000, 6000, 8000])
    axes[0].set_ylim(1500, 9200)
    axes[1].set_ylim(-1400, 1950)
    axes[1].set_yticks([-1000, 0, 1000])
    axes[1].axhline(0, color=BLACK, lw=0.6, zorder=1)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    lo, hi = m["band_lev_mean_range"]
    axes[0].text(0.02, 0.96, f"the band climbs: mean {lo:,.0f} to {hi:,.0f}", transform=axes[0].transAxes,
                 ha="left", va="top", fontsize=11, color=CRIMSON)
    axes[1].text(0.02, 0.96, "a horizontal band of fixed width", transform=axes[1].transAxes,
                 ha="left", va="top", fontsize=11, color=BLUE)
    save(fig, "stationarity_band")


# ================================================================== Granger with and without COVID
def granger_covid() -> None:
    """Lab 2, Step 6: monthly retail growth and unemployment changes, 1992-2026, with the window
    the lab drops (Feb 2020 - Jun 2021) shaded; right, the lag-4 F statistics both ways, with and
    without that window, against the 5% critical value. Both directions stay significant in both
    samples; which one looks stronger flips. No pre-2020 numbers: that cut is a lab exercise."""
    import matplotlib.dates as mdates
    m = measured()
    mc = m["macro"]; d = pd.to_datetime(mc["dates"])
    g = m["granger"]; crit = m["granger_f_crit_5pct"]; mult = m["covid_multiple_of_pre_sd"]
    fig = plt.figure(figsize=sz(14.5, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.15, 1], wspace=0.36, hspace=0.55)
    a0, a1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    for ax, key, unit, title, x in [(a0, "retail_growth", "%", "retail sales growth", mult["retail_growth"]),
                                    (a1, "unrate_diff", "pp", "change in unemployment rate", mult["unrate_diff"])]:
        ax.axvspan(pd.Timestamp(COVID[0]), pd.Timestamp(COVID[1]), color=LIGHT_CRIMSON, lw=0, zorder=0)
        ax.plot(d, mc[key], color=BLACK, lw=0.7, zorder=2)
        ax.axhline(0, color="#A3A3A3", lw=0.6, zorder=1)
        ax.set_title(f"{title} ({unit}, monthly)", fontsize=12, loc="left", pad=3)
        ax.text(pd.Timestamp("2019-06-01"), 0.93, f"biggest COVID month:\n{x:.0f}× the pre-2020 sd",
                transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=11, color=CRIMSON)
        ax.tick_params(labelsize=11)
        ax.xaxis.set_major_locator(mdates.YearLocator(5, month=1, day=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−")))
    a0.set_yticks([-10, 0, 10]); a1.set_yticks([0, 5, 10])
    a0.tick_params(labelbottom=False)
    # slope chart: F both ways, with and without the window
    ax = fig.add_subplot(gs[:, 1])
    xs = [0, 1]
    ur = [g["full"]["u_to_r"]["F"], g["ex_covid"]["u_to_r"]["F"]]
    ru = [g["full"]["r_to_u"]["F"], g["ex_covid"]["r_to_u"]["F"]]
    ax.axhline(crit, color=GRAY, lw=1.0, ls=(0, (3, 2)), zorder=1)
    ax.text(0.5, crit - 0.5, f"5% critical value, {crit:.2f}", ha="center", va="top", fontsize=10.5, color=GRAY)
    for vals, c in [(ur, AMBER), (ru, GREEN)]:
        ax.plot(xs, vals, "-o", color=c, lw=2.0, ms=6, zorder=3)
        for x, v in zip(xs, vals):
            near = abs(v - crit) < 1.5          # a point on the critical line: two decimals, label above it
            ax.text(x + (-0.1 if x == 0 else 0.1), v + (1.1 if near else 0), f"{v:.2f}" if near else f"{v:.1f}",
                    ha="right" if x == 0 else "left", va="bottom" if near else "center",
                    fontsize=11, color=c, fontweight="bold")
    # the legend sits above the panel, where nothing is drawn
    ax.text(-0.55, 1.13, "unemployment $\\rightarrow$ retail", transform=ax.get_xaxis_transform(),
            ha="left", va="bottom", fontsize=11, color=AMBER)
    ax.text(-0.55, 1.02, "retail $\\rightarrow$ unemployment", transform=ax.get_xaxis_transform(),
            ha="left", va="bottom", fontsize=11, color=GREEN)
    ax.set_xlim(-0.55, 1.55); ax.set_ylim(0, 20.5)
    ax.set_xticks(xs); ax.set_xticklabels([f"all\n{g['full']['n']} mo", f"COVID out\n{g['ex_covid']['n']} mo"], fontsize=11)
    ax.set_yticks([0, 5, 10, 15, 20]); ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel("Granger F, 4 lags", fontsize=11, labelpad=2)
    save(fig, "granger_covid")


# ================================================================== SARIMA: seasonal lags in a daily ACF
def sarima_acf() -> None:
    """CA_1 FOODS daily units: the ACF peaks at 7, 14, 21, 28 (the weekly season, m = 7); after the
    seasonal difference y_t - y_{t-7} the peaks are gone and a negative spike is left at lag 7, the
    seasonal-MA signature. The bump near lag 30 is the SNAP calendar (days 1-10 of every month):
    removing a SNAP dummy first takes it from 0.29 to about 0.09."""
    m = measured()
    lags = np.arange(1, DAILY_LAGS + 1)
    seas = lags % 7 == 0
    fig, axes = plt.subplots(1, 2, figsize=sz(13.4, 4.3), gridspec_kw={"wspace": 0.08})
    for ax, key, band, title in [(axes[0], "daily_acf", m["daily_band"], "daily units"),
                                 (axes[1], "daily_d7_acf", m["daily_d7_band"],
                                  "after the seasonal difference, $y_t - y_{t-7}$")]:
        v = np.array(m[key])
        corr_axis(ax, DAILY_LAGS, labels=(ax is axes[0]))
        ax.axhspan(-band, band, color=LIGHT_BLUE, zorder=1, lw=0)
        stems(ax, lags[~seas], v[~seas], color="#A3A3A3", lw=1.6, ms=2.8)
        stems(ax, lags[seas], v[seas], color=CRIMSON, lw=2.2, ms=4.2)
        ax.set_xticks([1, 7, 14, 21, 28, 35])
        ax.set_xlabel("lag (days)", fontsize=11, labelpad=1)
        ax.set_title(title, fontsize=12, loc="left", pad=4)
    v0, v1 = np.array(m["daily_acf"]), np.array(m["daily_d7_acf"])
    axes[0].text(DAILY_LAGS + 0.4, -0.62, "peaks at 7, 14, 21, 28:\nthe weekly season, m = 7",
                 ha="right", va="center", fontsize=11, color=CRIMSON)
    axes[1].annotate(f"{v1[6]:.2f} at lag 7: seasonal MA".replace("-", "−"),
                     xy=(7.3, v1[6]), xytext=(9.5, -0.72), fontsize=11, color=CRIMSON, ha="left",
                     va="center", arrowprops=dict(arrowstyle="-|>", color=CRIMSON, lw=1.1))
    k = int(np.argmax(v1[24:34])) + 25
    axes[1].annotate("SNAP days,\n1st–10th of the month", xy=(k, v1[k - 1] + 0.03), xytext=(k - 1, 0.78),
                     fontsize=11, color=GRAY, ha="center", va="center",
                     arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.0))
    save(fig, "sarima_acf")


FIGURES = {"acf_theory": acf_theory, "acf_levels_vs_diff": acf_levels_vs_diff, "shock_response": shock_response,
           "stationarity_band": stationarity_band, "granger_covid": granger_covid, "sarima_acf": sarima_acf}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if not a.startswith("--")] or list(FIGURES)
    print(f"Lecture 2 figures -> {OUT.relative_to(ROOT)}   (font: {FONT})")
    for name in pick:
        FIGURES[name]()
    m = measured()
    print(f"  TX_1 HOUSEHOLD: levels ACF {m['levels_acf'][0]:.2f} -> {m['levels_acf'][-1]:.2f}, "
          f"diff lag-1 {m['diff_acf'][0]:.3f}, ADF {m['adf_levels']:.2f} / {m['adf_diff']:.2f}, "
          f"SES alpha {m['ses_alpha_lab1']:.3f} -> rho1 {m['ses_implied_rho1']:.3f}")
