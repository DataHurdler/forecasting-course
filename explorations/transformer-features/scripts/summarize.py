"""One table from every results file: model x feature set x setting, test RMSE (mean ± sd, 3 seeds)."""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "output"
frames = []
for f in sorted(OUT.glob("results*.csv")) + sorted(OUT.glob("valid*.csv")):
    d = pd.read_csv(f)
    if "model" not in d:
        d["model"] = "transformer"
    if "epochs" not in d:
        d["epochs"] = 30
    if "shuffled" not in d:
        d["shuffled"] = "shuffled" in f.name
    frames.append(d)
df = pd.concat(frames, ignore_index=True)


def setting(r):
    order = "shuffled" if str(r.shuffled) == "True" else "lab order"
    ep = str(r.epochs)
    return f"{order}, epochs chosen on validation" if "validation" in ep else f"{order}, {int(float(ep))} ep"


df["setting"] = df.apply(setting, axis=1)
df["cell"] = df["mean"].astype(int).astype(str) + " ± " + df["sd"].astype(int).astype(str)
df["features"] = df["variant"].str[0]
order = ["lab order, 30 ep", "lab order, 60 ep", "lab order, 100 ep", "shuffled, 30 ep", "shuffled, 60 ep",
         "shuffled, 100 ep", "shuffled, epochs chosen on validation"]
models = ["ffn", "rnn", "lstm", "transformer"]
table = (df.drop_duplicates(["model", "features", "setting"], keep="last")
           .pivot(index=["model", "features"], columns="setting", values="cell")
           .reindex(columns=[c for c in order if c in set(df.setting)]))
table = table.reindex([(m, v) for m in models for v in "ABCDE" if (m, v) in table.index])


def markdown(t):
    """A Markdown table without the optional tabulate package."""
    cols = list(t.columns)
    lines = ["| model | features | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for (model, variant), row in t.iterrows():
        lines.append(f"| {model} | {variant} | " + " | ".join("" if pd.isna(v) else v for v in row) + " |")
    return "\n".join(lines) + "\n"


print(markdown(table))
(OUT / "summary.md").write_text(markdown(table))
