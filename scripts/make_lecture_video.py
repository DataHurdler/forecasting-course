"""make_lecture_video.py — a narrated lecture video from a deck's PDF and its recording script.

Each PDF page is shown while its `# ▶ SLIDE n` section of Narration/LectureNN_script.md is read
aloud by Kokoro, an open-source neural voice that runs offline (the script text never leaves the
machine). Stage directions are dropped and **[pause]** becomes a short silence.

    python3.12 -m venv .venv-video && .venv-video/bin/pip install -r scripts/video_requirements.txt
    .venv-video/bin/python scripts/make_lecture_video.py Lecture09_Part2           # whole deck
    .venv-video/bin/python scripts/make_lecture_video.py Lecture09_Part2 --only 1 2 3   # a preview

Output goes to ~/Movies/ECON8310/<Deck>.mp4 (videos are not kept in the repo). Compile the deck
first: the PDF must have exactly one page per narration section, and the script checks that.

Closed captions come with every video: a .vtt and an .srt beside the .mp4 (for Canvas, YouTube
and other players that take an uploaded caption file) and the same captions as a switchable
English track inside the .mp4. They are the script's own words, not speech recognition, so
"RMSE" is captioned as written even where the voice says "R-M-S-E". Each caption is timed from
the voice model's word timings.

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
CAPTION_LINE = 42                   # characters per caption line, two lines at most
HOLD_S = 0.6                        # how long a caption stays up after its last word

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
    # the last slide's section stops at the next heading (a timing guide, an appendix)
    return {int(n): re.split(r"(?m)^#{1,6} ", body)[0] for n, body in zip(parts[1::2], parts[2::2])}


def speakable(body: str) -> str:
    """The script's words, still marked up, with [[slnc ms]] marking each silence."""
    body = re.sub(r"(?s)\*\([^)]*\)\*", " ", body)                    # stage directions
    body = body.replace("**[pause]**", f" [[slnc {PAUSE_MS}]] ")
    body = re.sub(r"(?m)^---\s*$", " ", body).strip()
    body = re.sub(r"\n\s*\n\s*", f" [[slnc {PARAGRAPH_MS}]] ", body)
    # a pause next to a paragraph break, or a dropped stage direction, is one pause, not three
    body = re.sub(r"(?:\s*\[\[slnc \d+\]\])+",
                  lambda m: f" [[slnc {max(map(int, re.findall(r'\d+', m.group(0))))}]] ", body)
    body = re.sub(r"^\s*\[\[slnc \d+\]\]|\[\[slnc \d+\]\]\s*$", "", body.strip())
    return re.sub(r"\s+", " ", body).strip()


def spoken(text: str) -> str:
    """What the voice is given for a stretch of the script."""
    for pat, rep in SAY_AS:
        text = re.sub(pat, rep, text)
    return text.replace("**", "").replace("`", "")


def caption(text: str) -> str:
    """What the viewer reads for the same stretch: the script's words without the markup."""
    return text.replace("*", "").replace("`", "")                      # **bold**, *emphasis*, `code`


def split_long(words: list[str]) -> list[list[str]]:
    """Halve a sentence too long for two caption lines, preferring a comma or a dash near the
    middle, until every part fits."""
    if len(words) < 2 or len(caption(" ".join(words))) <= 2 * CAPTION_LINE:
        return [words]
    def cost(k: int) -> float:
        longer = max(len(caption(" ".join(words[:k]))), len(caption(" ".join(words[k:]))))
        at_comma = re.search(r"[,;:]\W*$", caption(words[k - 1])) or words[k] == "—"
        return longer - (25 if at_comma else 0)
    k = min(range(1, len(words)), key=cost)
    return split_long(words[:k]) + split_long(words[k:])


def pieces(text: str) -> list[str]:
    """Split one stretch of speech into captions: a sentence each, split further where a sentence
    would run past two lines."""
    words = re.findall(r"(?:`[^`]*`|\S)+", text)       # a backticked span is one word
    sentences, cur = [], []
    for w in words:
        cur.append(w)
        if re.search(r"[.!?…][\"”’)\]*`]*$", w):
            sentences.append(cur)
            cur = []
    return [" ".join(p) for s in sentences + [cur] if s for p in split_long(s)]


def two_lines(text: str) -> str:
    if len(text) <= CAPTION_LINE:
        return text
    spaces = [i for i, ch in enumerate(text) if ch == " "]
    i = min(spaces, key=lambda i: abs(i - len(text) / 2))
    return text[:i] + "\n" + text[i + 1:]


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


def trim(audio: np.ndarray, keep_ms: int = 120) -> tuple[np.ndarray, int]:
    """Cut the voice's own padding to keep_ms at each end; the script's silences are explicit.
    Also returns how many samples came off the front, which the caption timing needs."""
    loud = np.flatnonzero(np.abs(audio) > 0.01)
    if not loud.size:
        return audio[:0], 0
    keep = int(RATE * keep_ms / 1000)
    start = max(loud[0] - keep, 0)
    return audio[start:loud[-1] + 1 + keep], start


def silence(ms: int) -> np.ndarray:
    return np.zeros(int(RATE * ms / 1000), dtype=np.float32)


def alnum(text: str) -> int:
    return sum(ch.isalnum() for ch in text)


def timed(texts: list[str], said: list[str], words: list[tuple[str, float, float]]):
    """(start, end, caption) for each piece of one stretch of speech, from the voice's word
    timings. Pieces and words are matched by counting letters and digits, which both sides
    share; if the counts disagree, the stretch's time is shared out by length instead."""
    ends = np.cumsum([alnum(s) for s in said])
    seen = np.cumsum([alnum(w) for w, _, _ in words])
    t0, t1 = words[0][1], words[-1][2]
    if not len(seen) or ends[-1] != seen[-1]:
        print(f"    captions: timing shared by length for {texts[0][:40]!r}...")
        edges = t0 + (t1 - t0) * np.concatenate([[0], ends]) / max(ends[-1], 1)
        return [(edges[i], edges[i + 1], t) for i, t in enumerate(texts)]
    out, first = [], 0
    for end, text in zip(ends, texts):
        last = int(np.searchsorted(seen, end))         # the word holding this piece's last letter
        out.append((words[min(first, last)][1], words[last][2], text))
        first = last + 1
    return out


def held(cues: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Keep each caption up a little after its last word, but never over the next one."""
    out = []
    for i, (s, e, t) in enumerate(cues):
        nxt = cues[i + 1][0] if i + 1 < len(cues) else e + HOLD_S
        out.append((s, max(e, min(e + HOLD_S, nxt)), t))
    return out


def write_captions(cues: list[tuple[float, float, str]], vtt: Path, srt: Path) -> None:
    def stamp(t: float, sep: str) -> str:
        ms = round(t * 1000)
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}{sep}{ms % 1000:03d}"
    vtt.write_text("WEBVTT\n\n" + "".join(
        f"{stamp(s, '.')} --> {stamp(e, '.')}\n{two_lines(t)}\n\n" for s, e, t in cues))
    srt.write_text("".join(
        f"{i}\n{stamp(s, ',')} --> {stamp(e, ',')}\n{two_lines(t)}\n\n"
        for i, (s, e, t) in enumerate(cues, 1)))


def narrate(words: str, pipe, voice: str, speed: float):
    """The slide's audio, and its captions as (start, end, text) in seconds from the slide's start."""
    chunks, cues, at = [], [], 0                          # at: samples written so far
    for token in SILENCE.split(f" {words} "):
        if token.isdigit():                                   # the ms of a [[slnc ms]]
            chunks.append(silence(int(token)))
            at += len(chunks[-1])
        elif token.strip():
            parts = pieces(token.strip())
            said = [spoken(p) for p in parts]
            text = " ".join(said)
            if voice[0] == "a":
                for pat, rep in PHONEMES:
                    text = re.sub(pat, rep, text)
            timing = []
            for r in pipe(text, voice=voice, speed=speed):
                audio, cut = trim(r.audio.numpy())
                shift = (at - cut) / RATE                     # chunk time -> slide time
                stamps = [(t.text, t.start_ts, t.end_ts) for t in r.tokens]
                timing += [(w, s + shift, e + shift) for w, s, e in stamps
                           if s is not None and e is not None]
                chunks.append(audio)
                at += len(audio)
            if timing:
                cues += timed([caption(p) for p in parts], said, timing)
    audio, cut = trim(np.concatenate(chunks), keep_ms=0)
    shift = LEAD_MS / 1000 - cut / RATE
    end = (len(audio) + cut) / RATE
    cues = [(max(s, 0) + shift, min(e, end) + shift, t) for s, e, t in cues]
    return np.concatenate([silence(LEAD_MS), audio, silence(TAIL_MS)]), cues


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

    # One continuous soundtrack, so every slide change and caption sits at an exact sample
    # count. (Joining per-slide clips instead let each clip's AAC padding nudge the timeline:
    # the captions drifted 0.4 s behind the voice over a 30-minute lecture.)
    with tempfile.TemporaryDirectory() as tmp:
        tmp, sound, slides, cues, start = Path(tmp), [], [], [], 0.0
        for n in a.only or range(1, doc.page_count + 1):
            png = tmp / f"{n:02d}.png"
            page = doc[n - 1]
            zoom = 1920 / page.rect.width
            page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).save(png)
            words = speakable(text[n])
            if SILENCE.sub("", words).strip():
                audio, slide_cues = narrate(words, pipe, a.voice, a.speed)
                cues += [(s + start, e + start, t) for s, e, t in slide_cues]
            else:                                             # a slide with nothing to say
                audio = silence(3000)
            sound.append(audio)
            slides.append(f"file '{png}'\nduration {len(audio) / RATE:.4f}\n")
            start += len(audio) / RATE
            print(f"  slide {n:2d}: {len(SILENCE.sub('', words).split()):4d} words")
        sf.write(tmp / "sound.wav", np.concatenate(sound), RATE)
        # the concat demuxer ignores the last image's duration unless the image is listed again
        (tmp / "slides.txt").write_text("".join(slides) + f"file '{png}'\n")
        vtt, srt = out.with_suffix(".vtt"), out.with_suffix(".srt")
        write_captions(held(cues), vtt, srt)
        # -t, not -shortest: -shortest lets a still image overrun the audio by the encoder's buffer
        run(FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(tmp / "slides.txt"),
            "-i", str(tmp / "sound.wav"), "-i", str(srt), "-map", "0:v", "-map", "1:a", "-map", "2",
            "-vf", "fps=5,scale=1920:1080:force_original_aspect_ratio=decrease,"
                   "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:white,setsar=1",
            "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "96k", "-ar", "44100",
            "-c:s", "mov_text", "-metadata:s:s:0", "language=eng", "-metadata:s:s:0", "title=English",
            "-t", f"{start:.3f}", "-movflags", "+faststart", str(out))
    print(f"wrote {out}\n      {vtt}\n      {srt}  ({len(cues)} captions)")


if __name__ == "__main__":
    main()
