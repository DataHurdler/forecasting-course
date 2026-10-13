#!/usr/bin/env python3
"""Figures for Lecture 8 (CNNs) — one script, every figure, both formats.

    .venv/bin/python scripts/figures/lecture08_figures.py          # all
    .venv/bin/python scripts/figures/lecture08_figures.py conv1d   # one, by name

Writes Figures/Lecture08/<name>.pdf (Beamer) and <name>.svg (Quarto), both vector; sync_to_docs.sh
publishes Figures/ to docs/Figures/, so the decks reference ../Figures/Lecture08/<name>.{pdf,svg}.

Sources, all in the repository or fetched reproducibly:
  * the convolution example — Luo, Econ-ML-Book ch. 3: ts = [1,4,5,3,3,4,2,3,5,3], kernel [1,5,1]
  * the photograph — Figures/Lecture08/source/aisle-march-2020.jpg (CC0; see SOURCE.md there)
  * learned filters — torchvision's ImageNet-trained ResNet-18 (downloaded once, ~45 MB)
  * the forecasting window — data/processed/m5_weekly.csv, the course spine

Deterministic: no training, no sampling. random_state is fixed anyway, per the repo convention.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Figures" / "Lecture08"
PHOTO = OUT / "source" / "aisle-march-2020.jpg"
RANDOM_STATE = 42

# The deck's palette — Preambles/header.tex and Quarto/uno-clean.scss.
CRIMSON, BLACK, GRAY = "#D71920", "#1A1A1A", "#525252"
GREEN, AMBER, BLUE = "#15803D", "#B45309", "#005CA9"
LIGHT_CRIMSON, LIGHT_GRAY, LIGHT_BLUE = "#FDECEA", "#F4F4F5", "#E8F0FA"

_fonts = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ("Source Sans Pro", "Source Sans 3", "Helvetica Neue", "Arial")
             if f in _fonts), "DejaVu Sans")
# Figures are drawn near the size they are shown at. Beamer's text block is about 5.5 in wide,
# so a figure drawn 13 in wide arrives with its 15 pt labels at about 5 pt. SCALE shrinks every
# figure; text sizes are left alone, so they land at 8-10 pt on the page. SVG then scales up
# cleanly in the HTML deck.
SCALE = 0.55


def sz(w: float, h: float) -> tuple[float, float]:
    return (w * SCALE, h * SCALE)


plt.rcParams.update({"font.family": FONT, "font.size": 13, "svg.fonttype": "none",
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.05,
                    transparent=True, dpi=200)
    plt.close(fig)
    print(f"  {name}.pdf / .svg")


# ------------------------------------------------------------------ conv1d
TS = np.array([1, 4, 5, 3, 3, 4, 2, 3, 5, 3])
KERNEL = np.array([1, 5, 1])


def _cell(ax, x, y, text, w=1.0, *, edge=GRAY, face="white", color=BLACK, dashed=False,
          bold=False, size=13):
    ax.add_patch(Rectangle((x, y), w, 1, facecolor=face, edgecolor=edge, lw=1.6,
                           ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + 0.5, text, ha="center", va="center", color=color, fontsize=size,
            fontweight="bold" if bold else "normal")


def conv1d() -> None:
    """The book's example, worked: valid, same, then max pooling."""
    valid = np.convolve(TS, KERNEL, "valid")          # 8 values
    same = np.convolve(TS, KERNEL, "same")            # 10 values, one zero each end
    pooled = same.reshape(-1, 2).max(axis=1)          # 5 values
    assert valid.tolist() == [26, 32, 23, 22, 25, 17, 22, 31]       # the book's own numbers
    assert pooled.tolist() == [26, 32, 25, 22, 31]

    fig, ax = plt.subplots(figsize=sz(13, 6.6))
    ax.set_xlim(-4.6, 11.4); ax.set_ylim(-1.7, 7.3); ax.axis("off")
    LX = -1.55                     # one right edge for every row label, clear of the padding cell

    # kernel, drawn over the first window
    for i, k in enumerate(KERNEL):
        _cell(ax, i, 6.0, str(k), face=LIGHT_CRIMSON, edge=CRIMSON, color=CRIMSON, bold=True)
    ax.text(LX, 6.5, "kernel", ha="right", va="center", color=CRIMSON, fontweight="bold")

    # input, with the zero padding that "same" adds
    _cell(ax, -1, 4.2, "0", dashed=True, color=GRAY, edge=GRAY)
    _cell(ax, 10, 4.2, "0", dashed=True, color=GRAY, edge=GRAY)
    for i, v in enumerate(TS):
        hot = i < 3
        _cell(ax, i, 4.2, str(v), edge=CRIMSON if hot else GRAY, bold=hot)
    ax.text(LX, 4.7, "input", ha="right", va="center")

    # the first dot product, written out
    ax.text(3.35, 6.5, "1×1 + 4×5 + 5×1 = 26", ha="left", va="center", color=CRIMSON,
            fontsize=13)

    # valid: 8 outputs, each centred under its window
    for i, v in enumerate(valid):
        _cell(ax, i + 1, 2.4, str(v), face=LIGHT_CRIMSON if i == 0 else "white",
              edge=CRIMSON if i == 0 else GRAY, bold=i == 0)
    ax.text(LX, 2.98, "valid", ha="right", va="center", fontweight="bold")
    ax.text(LX, 2.36, "8 outputs", ha="right", va="center", color=GRAY, fontsize=10)

    # same: 10 outputs
    for i, v in enumerate(same):
        _cell(ax, i, 0.9, str(v))
    ax.text(LX, 1.48, "same", ha="right", va="center", fontweight="bold")
    ax.text(LX, 0.86, "10 outputs", ha="right", va="center", color=GRAY, fontsize=10)

    # max pooling over pairs of the same output
    for j, v in enumerate(pooled):
        _cell(ax, 2 * j, -1.2, str(v), w=2.0, face=LIGHT_BLUE, edge=BLUE, color=BLUE, bold=True)
        ax.plot([2 * j + 0.08, 2 * j + 1.92], [0.72, 0.72], color=BLUE, lw=1.4)
    ax.text(LX, -0.62, "max pool", ha="right", va="center", color=BLUE, fontweight="bold")
    ax.text(LX, -1.24, "size 2: 10 to 5", ha="right", va="center", color=GRAY, fontsize=10)
    save(fig, "conv1d_step_by_step")



# ------------------------------------------------------------------ the photograph
SOBEL_V = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])     # responds to VERTICAL edges
SOBEL_H = SOBEL_V.T                                           # responds to HORIZONTAL edges


def _gray_photo(width: int = 800) -> np.ndarray:
    from PIL import Image
    im = Image.open(PHOTO).convert("L")
    im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    return np.asarray(im, dtype=float) / 255.0


def _edge_map(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    from scipy.ndimage import convolve
    return np.abs(convolve(gray, kernel.astype(float), mode="nearest"))


def _show_kernel(ax, k, title, color):
    ax.set_xlim(0, 3); ax.set_ylim(0, 3); ax.set_aspect("equal"); ax.axis("off")
    for r in range(3):
        for c in range(3):
            v = int(k[r, c])
            face = LIGHT_CRIMSON if v > 0 else (LIGHT_BLUE if v < 0 else "white")
            ax.add_patch(Rectangle((c, 2 - r), 1, 1, facecolor=face, edgecolor=GRAY, lw=1.2))
            ax.text(c + 0.5, 2.5 - r, f"{v:+d}" if v else "0", ha="center", va="center",
                    fontsize=10, color=BLACK)
    ax.set_title(title, fontsize=11, color=color, pad=4)


def filter_sees() -> None:
    """One photo, two hand-set 3x3 filters, the two feature maps they produce."""
    gray = _gray_photo()
    v, h = _edge_map(gray, SOBEL_V), _edge_map(gray, SOBEL_H)
    cap = max(np.percentile(v, 97), np.percentile(h, 97))     # a projector washes out faint lines

    fig = plt.figure(figsize=sz(13, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[3.1, 1.25], hspace=0.12, wspace=0.06)
    panels = [(gray, "the photo", "gray", 1.0, BLACK),
              (v, "vertical edges", "Greys", cap, CRIMSON),
              (h, "horizontal edges", "Greys", cap, CRIMSON)]
    for col, (img, title, cmap, vmax, color) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax, interpolation="lanczos")
        ax.set_title(title, fontsize=13, color=color, pad=6); ax.axis("off")
    _show_kernel(fig.add_subplot(gs[1, 1]), SOBEL_V, "left vs right", BLACK)
    _show_kernel(fig.add_subplot(gs[1, 2]), SOBEL_H, "above vs below", BLACK)
    note = fig.add_subplot(gs[1, 0]); note.axis("off")
    note.text(0.0, 0.62, "Photo: Kurt Kaiser,\n18 March 2020 (CC0)", fontsize=10, color=GRAY,
              transform=note.transAxes, va="center")
    save(fig, "filter_sees")


def pooling() -> None:
    """A real 8x8 patch of the horizontal-edge map, max-pooled 2x2 to 4x4."""
    gray = _gray_photo()
    h = _edge_map(gray, SOBEL_H)
    # Shrink 8x by block mean so one patch spans a visible stretch of shelf, then take the
    # 8x8 window around the strongest response — a shelf edge, chosen by the data.
    f = 8
    H, W = (h.shape[0] // f) * f, (h.shape[1] // f) * f
    small = h[:H, :W].reshape(H // f, f, W // f, f).mean(axis=(1, 3))
    r0 = int(np.clip(np.argmax(small) // small.shape[1] - 3, 0, small.shape[0] - 8))
    c0 = int(np.clip(np.argmax(small) % small.shape[1] - 3, 0, small.shape[1] - 8))
    r0 -= r0 % 2; c0 -= c0 % 2
    patch = small[r0:r0 + 8, c0:c0 + 8]
    q = np.round(9 * patch / patch.max()).astype(int)            # one digit per cell
    pooled = q.reshape(4, 2, 4, 2).max(axis=(1, 3))

    fig = plt.figure(figsize=sz(13, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 0.62], wspace=0.28)
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(small, cmap="Greys", vmin=0, vmax=np.percentile(small, 99.5)); ax0.axis("off")
    ax0.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), 8, 8, fill=False, edgecolor=CRIMSON, lw=2.4))
    ax0.set_title("horizontal-edge map", fontsize=12, pad=6)

    def grid(ax, m, title, block=None, hot=None):
        n = m.shape[0]
        ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_aspect("equal"); ax.axis("off")
        for r in range(n):
            for c in range(n):
                is_hot = hot is not None and hot[r, c]
                ax.add_patch(Rectangle((c, n - 1 - r), 1, 1, lw=0.8, edgecolor="#BBBBBB",
                                       facecolor=LIGHT_BLUE if is_hot else "white"))
                ax.text(c + 0.5, n - 0.5 - r, str(m[r, c]), ha="center", va="center",
                        fontsize=10 if n == 8 else 13, color=BLUE if is_hot else BLACK,
                        fontweight="bold" if is_hot else "normal")
        if block:
            for k in range(0, n + 1, block):
                ax.plot([k, k], [0, n], color=BLUE, lw=2); ax.plot([0, n], [k, k], color=BLUE, lw=2)
        ax.set_title(title, fontsize=12, pad=6)

    hot = np.zeros_like(q, dtype=bool)
    for i in range(4):
        for j in range(4):
            blk = q[2*i:2*i+2, 2*j:2*j+2]
            rr, cc = np.unravel_index(np.argmax(blk), blk.shape)
            hot[2*i + rr, 2*j + cc] = True
    grid(fig.add_subplot(gs[1]), q, "the 8 × 8 patch", block=2, hot=hot)
    grid(fig.add_subplot(gs[2]), pooled, "2 × 2 max pool", block=1,
         hot=np.ones_like(pooled, dtype=bool))
    save(fig, "pooling")


def learned_filters() -> None:
    """ResNet-18's first layer: 64 filters nobody designed, and what four of them see."""
    import torch
    from PIL import Image
    from torchvision.models import resnet18, ResNet18_Weights
    torch.manual_seed(RANDOM_STATE)
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    w = net.conv1.weight.detach().numpy()                          # (64, 3, 7, 7)

    im = Image.open(PHOTO).convert("RGB").resize((640, 480), Image.LANCZOS)
    x = torch.tensor(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    with torch.no_grad():
        a = net.relu(net.bn1(net.conv1(((x - mean) / std)[None])))[0].numpy()   # (64, 240, 320)

    # Four filters, chosen by the data but forced to DIFFER: walk down the filters in order of
    # how much their response varies across this photo, and keep one only if its map is not
    # strongly correlated with any map already kept. Taking the top four by variance alone
    # returned four near-identical horizontal-edge detectors, which undersells the layer.
    flat = a.reshape(64, -1)
    pick: list[int] = []
    for k in np.argsort([-flat[k].std() for k in range(64)]):
        if flat[k].std() == 0:
            continue
        if all(abs(np.corrcoef(flat[k], flat[j])[0, 1]) < 0.35 for j in pick):
            pick.append(int(k))
        if len(pick) == 4:
            break

    fig = plt.figure(figsize=sz(13, 5.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1], wspace=0.08, hspace=0.32)
    axg = fig.add_subplot(gs[:, 0]); axg.axis("off")
    tile, gap = 7, 1
    canvas = np.ones((8 * (tile + gap) - gap, 8 * (tile + gap) - gap, 3))
    for k in range(64):
        f = w[k].transpose(1, 2, 0); f = (f - f.min()) / (f.max() - f.min() + 1e-9)
        r, c = divmod(k, 8)
        canvas[r*(tile+gap):r*(tile+gap)+tile, c*(tile+gap):c*(tile+gap)+tile] = f
    axg.imshow(canvas, interpolation="nearest")
    for k in pick:
        r, c = divmod(k, 8)
        axg.add_patch(Rectangle((c*(tile+gap) - 0.9, r*(tile+gap) - 0.9), tile + 0.8, tile + 0.8,
                                fill=False, edgecolor=CRIMSON, lw=2.6))
    axg.set_title("all 64 first-layer filters", fontsize=12, pad=6)
    for i, k in enumerate(pick):
        ax = fig.add_subplot(gs[i // 2, 1 + i % 2])
        ax.imshow(a[k], cmap="Greys", vmin=0, vmax=np.percentile(a[k], 99.5),
                  interpolation="lanczos"); ax.axis("off")
        ax.set_title(f"filter {k + 1}", fontsize=11, color=CRIMSON, pad=3)
    save(fig, "learned_filters")


# ------------------------------------------------------------------ residual block
def residual_block() -> None:
    """x -> conv -> ReLU -> conv -> (+) <- x, laid out on a grid so nothing collides."""
    fig, ax = plt.subplots(figsize=sz(13, 3.1))
    ax.set_xlim(-0.2, 16.2); ax.set_ylim(-1.35, 2.45); ax.axis("off")
    BW, BH, Y = 2.2, 1.0, 0.0                         # box width, height, centre line

    def box(xc, label, face, edge):
        ax.add_patch(FancyBboxPatch((xc - BW/2, Y - BH/2), BW, BH,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=face, edgecolor=edge, lw=1.8))
        ax.text(xc, Y, label, ha="center", va="center", fontsize=13)

    def arrow(x0, x1, y=Y, color=BLACK):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=16))

    ax.text(0.35, Y, r"$\mathbf{x}$", ha="center", va="center", fontsize=17)
    centres = [2.9, 5.75, 8.6]
    for xc, lab in zip(centres, ["conv 3×3", "ReLU", "conv 3×3"]):
        box(xc, lab, LIGHT_GRAY, GRAY)
    PX = 11.0                                          # the sum node
    ax.add_patch(plt.Circle((PX, Y), 0.36, facecolor="white", edgecolor=CRIMSON, lw=2.2))
    ax.text(PX, Y + 0.02, "+", ha="center", va="center", fontsize=17, color=CRIMSON, fontweight="bold")
    arrow(0.7, centres[0] - BW/2 - 0.05)
    for a_, b_ in zip(centres[:-1], centres[1:]):
        arrow(a_ + BW/2 + 0.05, b_ - BW/2 - 0.05)
    arrow(centres[-1] + BW/2 + 0.05, PX - 0.4)
    arrow(PX + 0.4, 12.45)
    ax.text(12.7, Y, r"$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$", ha="left",
            va="center", fontsize=16)
    ax.plot([1.0], [Y], marker="o", ms=7, color=CRIMSON, zorder=5)    # where the input forks

    # the skip path, routed square: up from x, across, down into the sum
    SY = 1.45
    ax.plot([1.0, 1.0, PX, PX], [Y + 0.05, SY, SY, Y + 0.38], color=CRIMSON, lw=2.4,
            solid_joinstyle="round")
    ax.annotate("", xy=(PX, Y + 0.38), xytext=(PX, Y + 0.95),
                arrowprops=dict(arrowstyle="-|>", color=CRIMSON, lw=2.4, mutation_scale=18))
    ax.text((1.0 + PX) / 2, SY + 0.25, "skip connection: the input, passed straight through",
            ha="center", va="bottom", fontsize=13, color=CRIMSON)

    # what the block learns, bracketed under the main path
    bx0, bx1, by = centres[0] - BW/2, centres[-1] + BW/2, Y - BH/2 - 0.3
    ax.plot([bx0, bx0, bx1, bx1], [by + 0.12, by, by, by + 0.12], color=GRAY, lw=1.4)
    ax.text((bx0 + bx1) / 2, by - 0.18,
            r"$\mathcal{F}(\mathbf{x})$: only the difference from $\mathbf{x}$ is learned",
            ha="center", va="top", fontsize=13, color=GRAY)
    save(fig, "residual_block")


# ------------------------------------------------------------------ receptive field on M5
def receptive_field() -> None:
    """A real 26-week window, the 7 weeks one filter output sees, and the 52-week lag."""
    import pandas as pd
    w = pd.read_csv(ROOT / "data" / "processed" / "m5_weekly.csv", parse_dates=["week_start"])
    y = (w[(w.store_id == "CA_1") & (w.cat_id == "FOODS")]
         .set_index("week_start")["units"].sort_index())
    # Forecast the Christmas week of 2015: what it needs to know happened 52 weeks earlier.
    target = y.index[(y.index <= "2015-12-25") & (y.index > "2015-12-18")][0]
    t = y.index.get_loc(target)
    lo = t - 60
    seg = y.iloc[lo:t + 1] / 1000
    xs = np.arange(lo, t + 1) - t                     # weeks relative to the target

    fig, ax = plt.subplots(figsize=sz(13, 4.6))
    win0, rf0 = -26, -7
    ax.axvspan(win0 - 0.5, -0.5, color=LIGHT_GRAY, zorder=0)
    ax.axvspan(rf0 - 0.5, -0.5, color=LIGHT_CRIMSON, zorder=1)
    ax.plot(xs[:-1], seg.values[:-1], color=BLUE, lw=2.2, marker="o", ms=3.5, zorder=3)
    ax.plot([0], [seg.values[-1]], marker="o", ms=11, color=CRIMSON, zorder=4)
    ax.plot([-52], [seg.values[xs.tolist().index(-52)]], marker="o", ms=11, mfc="white",
            mec=BLUE, mew=2.4, zorder=4)

    top = seg.max() * 1.18
    ax.set_ylim(seg.min() * 0.80, top * 1.08)
    lab = dict(ha="center", va="bottom", fontsize=11)
    ax.text((win0 + rf0) / 2 - 0.5, top * 0.99, "26-week window", color=GRAY, **lab)   # over the grey only
    ax.text(rf0 / 2 - 0.5, top * 0.99, "7 weeks", color=CRIMSON, **lab)
    ax.annotate("same week last year:\noutside the window", xy=(-52, seg.values[xs.tolist().index(-52)]),
                xytext=(-44, seg.min() * 0.86), color=BLUE, fontsize=11, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.5))
    ax.annotate("to forecast:\nChristmas\nweek", xy=(0, seg.values[-1]),
                xytext=(3.2, seg.values[-1] * 0.97), color=CRIMSON, fontsize=11, ha="left",
                va="center", arrowprops=dict(arrowstyle="-|>", color=CRIMSON, lw=1.5))
    ax.set_xlim(-61, 14)
    ax.set_xticks([-52, -26, -7, 0]); ax.set_xticklabels(["−52", "−26", "−7", "0"])
    ax.set_xlabel("weeks before the forecast week")
    ax.set_ylabel("units (thousands)")
    save(fig, "receptive_field")


FIGURES = {"conv1d": conv1d, "filter_sees": filter_sees, "pooling": pooling,
           "learned_filters": learned_filters, "residual_block": residual_block,
           "receptive_field": receptive_field}

if __name__ == "__main__":
    pick = sys.argv[1:] or list(FIGURES)
    print(f"Lecture 8 figures -> {OUT.relative_to(ROOT)}   (font: {FONT})")
    for name in pick:
        FIGURES[name]()
