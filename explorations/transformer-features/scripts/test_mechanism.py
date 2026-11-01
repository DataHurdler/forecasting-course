"""Are unshuffled batches really pulling the model toward the series it just saw?

report.qmd explained the labs' training order two ways without testing either: the model ends a
pass tuned toward the last few series, and each series' identity weights fall out of step with the
rest of the model. This trains as the lab does, then watches the final pass batch by batch.

  tuned toward recent series  -> a series' test error is lower right after its own stretch of the
                                 pass than it is at other points in the pass
  out of step                 -> a series' level offset is small right after its own stretch and
                                 larger by the end of the pass

SHUFFLE=1 is the control: there is no stretch, so the first measure should sit near half.

    MODEL=rnn .venv/bin/python explorations/transformer-features/scripts/test_mechanism.py A
    MODEL=transformer .venv/bin/python explorations/transformer-features/scripts/test_mechanism.py A B
    MODEL=ffn WINDOW=26 EPOCHS=60 .venv/bin/python explorations/transformer-features/scripts/test_mechanism.py A
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

wanted = sys.argv[1:] or ["A"]
sys.argv = [sys.argv[0], "none"]                       # load data and classes; run no variants
sys.path.insert(0, str(Path(__file__).parent))
import run_features as rf                              # noqa: E402

assert len(rf.D) % 30 == 0
series = np.repeat(np.arange(30), len(rf.D) // 30)
SPECS = {"A": dict(series_id=False, week_calendar=False, future=False),
         "B": dict(series_id=True, week_calendar=False, future=False)}


def per_series(model, x, f):
    """Per-series test RMSE, and each series' mean error on its train and test windows."""
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(x), torch.tensor(f)).numpy() * rf.ys + rf.ym
    model.train()
    err = p - rf.Y
    return (np.array([np.sqrt(np.mean(err[rf.te & (series == s)] ** 2)) for s in range(30)]),
            np.array([err[rf.tr & (series == s)].mean() for s in range(30)]),
            np.array([err[rf.te & (series == s)].mean() for s in range(30)]),
            rf.rmse(rf.Y[rf.te], p[rf.te]))


def run(x, f, seed):
    torch.manual_seed(seed)
    model = rf.make(x.shape[2], f.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=rf.LR, weight_decay=rf.WD)
    lossfn = torch.nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(x[rf.tr]), torch.tensor(f[rf.tr]), torch.tensor(rf.Yn[rf.tr])),
        batch_size=rf.BATCH, shuffle=rf.SHUFFLE)
    snaps = []
    for epoch in range(1, rf.EPOCHS + 1):
        for xb, fb, yb in loader:
            opt.zero_grad()
            lossfn(model(xb, fb), yb).backward()
            if rf.CLIP:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if epoch == rf.EPOCHS:
                snaps.append(per_series(model, x, f))
    return snaps


per_stretch = int(rf.tr.sum()) // 30                   # training windows per series
rows = []
for name in wanted:
    x, f = rf.design(**SPECS[name])
    for seed in rf.SEEDS:
        snaps = run(x, f, seed)
        rm = np.stack([s[0] for s in snaps])           # batches x series
        tr_off, te_off = np.stack([s[1] for s in snaps]), np.stack([s[2] for s in snaps])
        overall = np.array([s[3] for s in snaps])
        # the batch that finishes each series' stretch
        own = np.minimum((np.arange(1, 31) * per_stretch - 1) // rf.BATCH, len(snaps) - 1)
        ix = np.arange(30)
        rows.append({
            "model": rf.MODEL, "variant": name, "shuffled": rf.SHUFFLE, "seed": seed,
            "test_end": round(overall[-1], 1), "test_min_in_pass": round(overall.min(), 1),
            "test_max_in_pass": round(overall.max(), 1),
            "series_lower_after_own_stretch": int((rm[own, ix] < np.median(rm, axis=0)).sum()),
            "rmse_after_own": round(float(np.mean(rm[own, ix])), 1), "rmse_at_end": round(float(np.mean(rm[-1])), 1),
            "train_offset_rms_after_own": round(float(np.sqrt(np.mean(tr_off[own, ix] ** 2))), 1),
            "train_offset_rms_at_end": round(float(np.sqrt(np.mean(tr_off[-1] ** 2))), 1),
            "test_offset_rms_after_own": round(float(np.sqrt(np.mean(te_off[own, ix] ** 2))), 1),
            "test_offset_rms_at_end": round(float(np.sqrt(np.mean(te_off[-1] ** 2))), 1),
        })
        print(rows[-1], flush=True)
out = rf.OUT / "mechanism.csv"
new = pd.DataFrame(rows)
if out.exists():
    new = pd.concat([pd.read_csv(out), new], ignore_index=True)
new.drop_duplicates(["model", "variant", "shuffled", "seed"], keep="last").to_csv(out, index=False)
