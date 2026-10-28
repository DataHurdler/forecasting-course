"""make_lecture_video.py — a narrated lecture video from a deck's PDF and its recording script.

Each PDF page is shown while its `# ▶ SLIDE n` section of Narration/LectureNN_script.md is read
aloud by Kokoro, an open-source neural voice that runs offline (the script text never leaves the
machine). Stage directions are dropped and **[pause]** becomes a short silence.

    python3.12 -m venv .venv-video && .venv-video/bin/pip install -r scripts/video_requirements.txt
    .venv-video/bin/python scripts/make_lecture_video.py Lecture09_Part2           # whole deck
    .venv-video/bin/python scripts/make_lecture_video.py Lecture09_Part2 --only 1 2 3   # a preview

Output goes to ~/Movies/ECON8310/<Deck>.mp4 (videos are not kept in the repo). Compile the deck
first: the PDF must have exactly one page per narration section, and the script checks that.

Owner's choices (2026-10-27): the voice is Michael (am_michael); about one second of silence
before a slide changes and one more before the next slide's narration starts.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pymupdf
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
RATE = 24000                        # Kokoro's sample rate
LEAD_MS, TAIL_MS = 1100, 1000       # silence before and after each slide's narration
PARAGRAPH_MS, PAUSE_MS = 450, 900   # a breath between paragraphs; a **[pause]**

# How the voice should say what a lecturer would read off the page.
SAY_AS = [
    (r"`nn\.TransformerEncoderLayer`", "n-n dot Transformer Encoder Layer"),
    (r"`nn\.TransformerEncoder`", "n-n dot Transformer Encoder"),
    (r"`batch_first=True`", "batch first equals true"),
    (r"`self\.pe`", "self dot p-e"),
    (r"`Conv1d`", "Conv one-D"),
    (r"`KFold`", "K-fold"),
    (r"\bRMSE\b", "R-M-S-E"),
    (r"\bLASSO\b", "lasso"),
    (r"\bXGBoost\b", "X-G-Boost"),
    (r"\b1D\b", "one-D"),
    (r"\bM5\b", "M-five"),
    (r"\bLayerNorms?\b", lambda m: "layer norms" if m.group(0).endswith("s") else "layer norm"),
    (r"\bd-k\b", "d-sub-k"),
    (r"\bGPUs\b", "G-P-Us"),
]

# Kokoro's own respellings, as [word](/phonemes/), for words its American voices get wrong:
# it says EP-uhk (owner, 2026-10-27), spells out BERT and ARIMA, and says ray-el-you, luh-NET
# and RAYS-net.
PHONEMES = [
    (r"\bepochs\b", "[epochs](/ˈipɑks/)"),
    (r"\bepoch\b", "[epoch](/ˈipɑk/)"),
    (r"\bBERT\b", "[BERT](/bˈɜɹt/)"),
    (r"\bSARIMA\b", "[SARIMA](/səɹˈimə/)"),
    (r"\bARIMA\b", "[ARIMA](/əɹˈimə/)"),
    (r"\bReLU\b", "[ReLU](/ɹˈAlu/)"),
    (r"\bLeNet\b", "[LeNet](/lˈinɛt/)"),
    (r"\bResNet\b", "[ResNet](/ɹˈɛznɛt/)"),
]

SILENCE = re.compile(r"\[\[slnc (\d+)\]\]")


def sections(script: str) -> dict[int, str]:
    parts = re.split(r"(?m)^# ▶ SLIDE (\d+) — .*$", script)
    return {int(n): body for n, body in zip(parts[1::2], parts[2::2])}


def speakable(body: str) -> str:
    """The words to say, with [[slnc ms]] marking each silence."""
    body = re.sub(r"(?s)\*\([^)]*\)\*", " ", body)                    # stage directions
    body = body.replace("**[pause]**", f" [[slnc {PAUSE_MS}]] ")
    body = re.sub(r"(?m)^---\s*$", " ", body)
    for pat, rep in SAY_AS:
        body = re.sub(pat, rep, body)
    body = body.replace("**", "").replace("`", "").strip()
    body = re.sub(r"\n\s*\n\s*", f" [[slnc {PARAGRAPH_MS}]] ", body)
    # a pause next to a paragraph break, or a dropped stage direction, is one pause, not three
    body = re.sub(r"(?:\s*\[\[slnc \d+\]\])+",
                  lambda m: f" [[slnc {max(map(int, re.findall(r'\d+', m.group(0))))}]] ", body)
    body = re.sub(r"^\s*\[\[slnc \d+\]\]|\[\[slnc \d+\]\]\s*$", "", body.strip())
    return re.sub(r"\s+", " ", body).strip()


def pipeline(voice: str):
    import espeakng_loader
    import misaki.espeak  # noqa: F401  (points espeak at its bundled data on import)
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    from kokoro import KPipeline
    # espeak-ng truncates its data path at 160 characters, and a venv's site-packages can be
    # longer, so keep a copy of the data at a short path.
    data = Path.home() / ".cache/econ8310-tts/espeak-ng-data"
    if not (data / "phontab").exists():
        shutil.copytree(espeakng_loader.get_data_path(), data, dirs_exist_ok=True)
    EspeakWrapper.set_data_path(str(data))
    return KPipeline(lang_code=voice[0], repo_id="hexgrad/Kokoro-82M")


def trim(audio: np.ndarray, keep_ms: int = 120) -> np.ndarray:
    """Cut the voice's own padding to keep_ms at each end; the script's silences are explicit."""
    loud = np.flatnonzero(np.abs(audio) > 0.01)
    if not loud.size:
        return audio[:0]
    keep = int(RATE * keep_ms / 1000)
    return audio[max(loud[0] - keep, 0):loud[-1] + 1 + keep]


def silence(ms: int) -> np.ndarray:
    return np.zeros(int(RATE * ms / 1000), dtype=np.float32)


def narrate(words: str, pipe, voice: str, speed: float) -> np.ndarray:
    chunks = []
    for token in SILENCE.split(f" {words} "):
        if token.isdigit():                                   # the ms of a [[slnc ms]]
            chunks.append(silence(int(token)))
        elif token.strip():
            if voice[0] == "a":
                for pat, rep in PHONEMES:
                    token = re.sub(pat, rep, token)
            chunks += [trim(a.numpy()) for _, _, a in pipe(token.strip(), voice=voice, speed=speed)]
    audio = trim(np.concatenate(chunks), keep_ms=0)
    return np.concatenate([silence(LEAD_MS), audio, silence(TAIL_MS)])


def run(*cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("lecture", help="e.g. Lecture09_Part2 or Lecture10")
    ap.add_argument("--voice", default="am_michael", help="a Kokoro voice, e.g. am_michael, af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--only", type=int, nargs="*", help="build just these slide numbers")
    ap.add_argument("--out", type=Path, help="default: ~/Movies/ECON8310/<Deck>.mp4")
    a = ap.parse_args()

    pdfs = sorted((ROOT / "Slides").glob(f"{a.lecture}_*.pdf"))
    script = ROOT / "Narration" / f"{a.lecture}_script.md"
    if len(pdfs) != 1 or not script.exists():
        raise SystemExit(f"need exactly one Slides/{a.lecture}_*.pdf and {script.relative_to(ROOT)}")
    out = a.out or Path.home() / "Movies/ECON8310" / f"{pdfs[0].stem}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdfs[0])
    text = sections(script.read_text())
    if len(text) != doc.page_count:
        raise SystemExit(f"{len(text)} narration sections but {doc.page_count} PDF pages; recompile?")
    pipe = pipeline(a.voice)

    with tempfile.TemporaryDirectory() as tmp:
        tmp, segments = Path(tmp), []
        for n in a.only or range(1, doc.page_count + 1):
            png, wav, mp4 = (tmp / f"{n:02d}.{ext}" for ext in ("png", "wav", "mp4"))
            page = doc[n - 1]
            zoom = 1920 / page.rect.width
            page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).save(png)
            words = speakable(text[n])
            if SILENCE.sub("", words).strip():
                sf.write(wav, narrate(words, pipe, a.voice, a.speed), RATE)
            else:                                             # a slide with nothing to say
                sf.write(wav, silence(3000), RATE)
            # -t, not -shortest: -shortest lets a looped still image overrun its audio by the
            # encoder's buffer, tens of seconds of dead air at 2 frames a second.
            run(FFMPEG, "-y", "-loop", "1", "-framerate", "2", "-i", str(png), "-i", str(wav),
                "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
                "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,"
                       "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:white,setsar=1",
                "-c:a", "aac", "-b:a", "96k", "-ar", "44100",
                "-t", f"{sf.info(str(wav)).duration:.3f}", str(mp4))
            segments.append(mp4)
            print(f"  slide {n:2d}: {len(SILENCE.sub('', words).split()):4d} words")
        (tmp / "list.txt").write_text("".join(f"file '{s}'\n" for s in segments))
        run(FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"), "-c", "copy",
            "-movflags", "+faststart", str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
