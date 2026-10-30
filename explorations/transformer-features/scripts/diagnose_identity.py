"""Why does telling the Transformer which series it is make the forecast worse?

Compares, for seed 1: the lab model (A); identity on every week (B); identity given to the head
only, beside the last week's 64 numbers (B2). For each: training RMSE, test RMSE, and how much of
the test error is a constant offset per series (bias) rather than week-to-week miss.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.argv = [sys.argv[0], "none"]                       # load data and classes; run no variants
sys.path.insert(0, str(Path(__file__).parent))
import run_features as rf                              # noqa: E402

# series index of every window, in build order (30 series of 225 windows each)
assert len(rf.D) % 30 == 0
series = np.repeat(np.arange(30), len(rf.D) // 30)
ID = rf.IDS[:, 0, :]                                   # one row of the one-hot per window


def fit(x, f, seed=1):
    torch.manual_seed(seed)
    m = rf.TransformerForecaster(x.shape[2], f.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.tensor(x[rf.tr]), torch.tensor(f[rf.tr]), torch.tensor(rf.Yn[rf.tr])), batch_size=128, shuffle=False)
    for _ in range(30):
        m.train()
        for xb, fb, yb in loader:
            opt.zero_grad(); torch.nn.MSELoss()(m(xb, fb), yb).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval()
    with torch.no_grad():
        p = m(torch.tensor(x), torch.tensor(f)).numpy() * rf.ys + rf.ym
    return p


x_a, f_a = rf.design(series_id=False, week_calendar=False, future=False)
x_b, f_b = rf.design(series_id=True, week_calendar=False, future=False)
f_b2 = ID.astype("float32")                            # identity beside the last week, at the head
for name, x, f in (("A lab", x_a, f_a), ("B identity every week", x_b, f_b), ("B2 identity at head", x_a, f_b2)):
    p = fit(x, f)
    err = p - rf.Y
    bias = np.array([err[rf.te & (series == s)].mean() for s in range(30)])
    within = np.sqrt(np.mean([(err[rf.te & (series == s)] - bias[s]) ** 2 for s in range(30)]))
    print(f"{name:24s} train {rf.rmse(rf.Y[rf.tr], p[rf.tr]):6.0f} | test {rf.rmse(rf.Y[rf.te], p[rf.te]):6.0f} "
          f"| per-series offset: RMS {np.sqrt(np.mean(bias ** 2)):5.0f}, worst {bias[np.argmax(abs(bias))]:+6.0f} "
          f"| week-to-week {within:5.0f}", flush=True)
