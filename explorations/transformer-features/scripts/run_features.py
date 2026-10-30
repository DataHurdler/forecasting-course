"""Lab 9 Part 2's Transformer, given the features Homework 4's LASSO had.

The lab's Transformer sees 52 past weeks of four channels. The LASSO that tops the scoreboard
(744) also knew which series it was forecasting, the calendar, and the forecast week's own SNAP
days, event days and price. This adds those to the Transformer, one group at a time, with the
lab's data, split, model and training loop otherwise unchanged, and three seeds per variant.

    .venv/bin/python explorations/transformer-features/scripts/run_features.py
"""
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "output"
SEEDS = (1, 2, 3)
# SHUFFLE=1 shuffles the training windows between batches (all of them precede the cutoff).
# The lab does not: its loader walks the training windows series by series.
SHUFFLE = os.environ.get("SHUFFLE") == "1"
# EPOCHS=100 trains longer and records the test RMSE at 30, 60 and 100 epochs of the same run.
EPOCHS = int(os.environ.get("EPOCHS", "30"))
CHECKPOINTS = [e for e in (30, 60, 100) if e <= EPOCHS] or [EPOCHS]
# MODEL=lstm or rnn swaps in Lab 9 Part 1's recurrent model (hidden 64) for the Transformer;
# MODEL=ffn is Lab 7's network (128, 64, 1, dropout 0.2) with Lab 7's lr 5e-4, no weight decay,
# batches of 64 and no gradient clipping.
MODEL = os.environ.get("MODEL", "transformer")
# VALID=1 chooses the epoch on a validation block (the last 52 weeks of the training period),
# trains on the weeks before it, and scores the test set once, at the chosen epoch.
VALID = os.environ.get("VALID") == "1"
torch.set_num_threads(int(os.environ.get("THREADS", torch.get_num_threads())))

weekly = (pd.read_csv(ROOT / "data/processed/m5_weekly.csv", parse_dates=["week_start"])
          .sort_values(["store_id", "cat_id", "week_start"]))
# WINDOW=26 with MODEL=ffn is Lab 7's feedforward network on its own 26-week window.
CHANNELS, WINDOW = ["units", "snap_days", "event_days", "avg_price"], int(os.environ.get("WINDOW", "52"))
CUTOFF = np.datetime64("2015-05-17")
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

woy = weekly.week_start.dt.isocalendar().week.astype(float).values
weekly["woy_sin"], weekly["woy_cos"] = np.sin(2 * np.pi * woy / 52.18), np.cos(2 * np.pi * woy / 52.18)
stores, cats = sorted(weekly.store_id.unique()), sorted(weekly.cat_id.unique())

# Windows exactly as the lab builds them, plus what each new feature group needs.
X, Y, D, IDS, CAL, FUT = [], [], [], [], [], []
for (s, c), g in weekly.groupby(["store_id", "cat_id"]):
    arr, dates = g[CHANNELS].values.astype("float32"), g.week_start.values
    cal = g[["woy_sin", "woy_cos"]].values.astype("float32")
    onehot = np.r_[np.eye(len(stores))[stores.index(s)], np.eye(len(cats))[cats.index(c)]].astype("float32")
    for i in range(len(g) - WINDOW):
        X.append(arr[i:i + WINDOW]); Y.append(arr[i + WINDOW, 0]); D.append(dates[i + WINDOW])
        IDS.append(np.tile(onehot, (WINDOW, 1))); CAL.append(cal[i:i + WINDOW])
        # the forecast week's own SNAP days, event days, price and calendar
        FUT.append(np.r_[arr[i + WINDOW, 1:], cal[i + WINDOW]])
X, Y, D = np.stack(X), np.array(Y, dtype="float32"), np.array(D)
IDS, CAL, FUT = np.stack(IDS), np.stack(CAL), np.stack(FUT).astype("float32")
tr, te = D < CUTOFF, D >= CUTOFF
mu, sd = X[tr].mean((0, 1), keepdims=True), X[tr].std((0, 1), keepdims=True) + 1e-8
fmu, fsd = FUT[tr].mean(0, keepdims=True), FUT[tr].std(0, keepdims=True) + 1e-8
ym, ys = Y[tr].mean(), Y[tr].std()
Xn, FUTn, Yn = (X - mu) / sd, (FUT - fmu) / fsd, (Y - ym) / ys     # one-hots and sin/cos left as they are
print(f"{len(X):,} windows — train {tr.sum():,}, test {te.sum():,}", flush=True)


def sinusoidal(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(seq_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


class TransformerForecaster(nn.Module):
    """The lab's model; n_future > 0 concatenates the forecast week's features before the head."""

    def __init__(self, n_channels, n_future=0, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.inp = nn.Linear(n_channels, d_model)
        self.register_buffer("pe", sinusoidal(WINDOW, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128,
                                               dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.fc = nn.Linear(d_model + n_future, 1)

    def forward(self, x, fut):
        h = self.enc(self.inp(x) + self.pe)[:, -1, :]
        return self.fc(torch.cat([h, fut], 1) if fut.shape[1] else h).squeeze(-1)


class Recurrent(nn.Module):
    """Lab 9 Part 1's model; the forecast week's features join the last hidden state, as above."""

    def __init__(self, kind, n_channels, n_future=0, hidden=64):
        super().__init__()
        self.rnn = (nn.RNN if kind == "rnn" else nn.LSTM)(n_channels, hidden, batch_first=True)
        self.fc = nn.Linear(hidden + n_future, 1)

    def forward(self, x, fut):
        h = self.rnn(x)[0][:, -1, :]
        return self.fc(torch.cat([h, fut], 1) if fut.shape[1] else h).squeeze(-1)


class FFN(nn.Module):
    """Lab 7's network on the flattened window; the forecast week's features join the inputs."""

    def __init__(self, n_channels, n_future=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_channels * WINDOW + n_future, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1))

    def forward(self, x, fut):
        return self.net(torch.cat([x.flatten(1), fut], 1)).squeeze(-1)


def make(n_channels, n_future):
    if MODEL == "transformer":
        return TransformerForecaster(n_channels, n_future)
    if MODEL == "ffn":
        return FFN(n_channels, n_future)
    return Recurrent(MODEL, n_channels, n_future)


LR, WD, BATCH, CLIP = (5e-4, 0.0, 64, False) if MODEL == "ffn" else (1e-3, 1e-4, 128, True)


def design(series_id, week_calendar, future, future_price=True):
    # The feedforward network sees the whole flattened window at once, so the series identity goes
    # in once, beside the window, as it did for the LASSO; repeating it on every week would hand
    # it 26 copies of the same thirteen columns.
    per_week_id = series_id and MODEL != "ffn"
    parts = [Xn] + ([IDS] if per_week_id else []) + ([CAL] if week_calendar else [])
    x = np.concatenate(parts, axis=2).astype("float32")
    f = FUTn if future else FUTn[:, :0]
    if future and not future_price:
        f = np.delete(FUTn, 2, axis=1)                      # columns: snap, event, price, sin, cos
    if series_id and not per_week_id:
        f = np.concatenate([f, IDS[:, 0, :]], axis=1)
    return x, f.astype("float32")


def train(x, f, seed, epochs=None):
    epochs = epochs or EPOCHS
    if VALID:
        return train_valid(x, f, seed, epochs)
    scores = {}
    torch.manual_seed(seed)
    model = make(x.shape[2], f.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lossfn = nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(x[tr]), torch.tensor(f[tr]), torch.tensor(Yn[tr])),
        batch_size=BATCH, shuffle=SHUFFLE)
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, fb, yb in loader:
            opt.zero_grad()
            lossfn(model(xb, fb), yb).backward()
            if CLIP:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if epoch in CHECKPOINTS:
            model.eval()
            with torch.no_grad():
                pred = model(torch.tensor(x[te]), torch.tensor(f[te])).numpy() * ys + ym
            scores[epoch] = rmse(Y[te], pred)
    return scores, sum(p.numel() for p in model.parameters())


VAL_START = CUTOFF - np.timedelta64(52 * 7, "D")
fit_ix, val_ix = tr & (D < VAL_START), tr & (D >= VAL_START)


def train_valid(x, f, seed, epochs):
    """Choose the number of epochs on the validation block (training on the weeks before it),
    then retrain on the whole training period for that many epochs and score the test set once."""
    torch.manual_seed(seed)
    model = make(x.shape[2], f.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lossfn = nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(x[fit_ix]), torch.tensor(f[fit_ix]), torch.tensor(Yn[fit_ix])),
        batch_size=BATCH, shuffle=SHUFFLE)
    xv, fv = torch.tensor(x[val_ix]), torch.tensor(f[val_ix])
    best = (np.inf, 0)
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, fb, yb in loader:
            opt.zero_grad()
            lossfn(model(xb, fb), yb).backward()
            if CLIP:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = rmse(Y[val_ix], model(xv, fv).numpy() * ys + ym)
        if v < best[0]:
            best = (v, epoch)
    # then retrain on the whole training period for that many epochs, and score the test set once
    torch.manual_seed(seed)
    model = make(x.shape[2], f.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(x[tr]), torch.tensor(f[tr]), torch.tensor(Yn[tr])),
        batch_size=BATCH, shuffle=SHUFFLE)
    for _ in range(best[1]):
        model.train()
        for xb, fb, yb in loader:
            opt.zero_grad()
            lossfn(model(xb, fb), yb).backward()
            if CLIP:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x[te]), torch.tensor(f[te])).numpy() * ys + ym
    return {"best_epoch": best[1], "val": best[0], "test": rmse(Y[te], pred)}, sum(p.numel() for p in model.parameters())


VARIANTS = [
    ("A. Lab 9 Part 2 as is (4 channels)",               dict(series_id=False, week_calendar=False, future=False)),
    ("B. + which series (store and category)",            dict(series_id=True,  week_calendar=False, future=False)),
    ("C. + forecast week's SNAP, events, price, calendar", dict(series_id=False, week_calendar=False, future=True)),
    ("D. B + C + each week's calendar (all)",            dict(series_id=True,  week_calendar=True,  future=True)),
    ("E. D without the forecast week's price",           dict(series_id=True,  week_calendar=True,  future=True, future_price=False)),
]
only = sys.argv[1:]
rows = []
for name, spec in VARIANTS:
    if only and name[0] not in only:
        continue
    x, f = design(**spec)
    runs = []
    for s in SEEDS:
        t0 = time.time()
        scores, n_par = train(x, f, s)
        runs.append(scores)
        print(f"  {name:52s} seed {s}: " + "  ".join(f"{e} {v:7.1f}" if not isinstance(e, int) else f"ep{e} {v:7.1f}"
                                                  for e, v in scores.items())
              + f"  ({n_par:,} params, {time.time() - t0:.0f}s)", flush=True)
    if VALID:
        rows.append({"model": MODEL, "variant": name, "epochs": f"best of {EPOCHS} on validation", "shuffled": SHUFFLE,
                     "params": n_par, "mean": round(np.mean([r["test"] for r in runs])),
                     "sd": round(np.std([r["test"] for r in runs])),
                     "runs": " / ".join(f"{r['test']:.0f} (ep {r['best_epoch']})" for r in runs)})
        pd.DataFrame(rows).to_csv(OUT / f"valid_{MODEL}{'_shuffled' if SHUFFLE else ''}_{EPOCHS}ep_{''.join(only)}.csv", index=False)
        continue
    for e in CHECKPOINTS:
        r = [sc[e] for sc in runs]
        rows.append({"model": MODEL, "variant": name, "epochs": e, "shuffled": SHUFFLE, "inputs per week": x.shape[2],
                     "forecast-week inputs": f.shape[1], "params": n_par, "mean": round(np.mean(r)),
                     "sd": round(np.std(r)), "runs": " / ".join(f"{v:.0f}" for v in r)})
    tag = ("" if MODEL == "transformer" else f"_{MODEL}") + (f"_w{WINDOW}" if WINDOW != 52 else "") + ("_shuffled" if SHUFFLE else "") + (f"_{EPOCHS}ep" if EPOCHS != 30 else "") + ("_" + "".join(only) if only else "")
    pd.DataFrame(rows).to_csv(OUT / f"results{tag}.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
