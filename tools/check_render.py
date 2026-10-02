"""Render every voice of a score and check each sung note in the audio:
- silent: a note that should sound but is (almost) silent in its middle
- late: how long after the written beat the note's sound starts

usage: .venv/bin/python tools/check_render.py score.pdf [--bars 30-40]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from sing import pick_voice_kind, pick_voice_model  # noqa: E402
from singer import dsinger
from singer.pdfscore import read_pdf
from singer.perform import build, measure_order
from singer.synth import SR, sing


def envelope(y: np.ndarray, win: float = 0.01) -> np.ndarray:
    n = int(win * SR)
    k = len(y) // n
    return np.sqrt((y[: k * n].reshape(k, n) ** 2).mean(axis=1))


def pitch_errors(y: np.ndarray, notes) -> list[float]:
    """Cents between the sung pitch (WORLD harvest, middle half of the note) and the
    written note, for notes of at least 0.25 s."""
    import pyworld as pw
    from scipy.signal import resample_poly

    x = resample_poly(y.astype(np.float64), 1, 3)  # 14.7 kHz is plenty for f0
    sr = SR / 3
    f0, t = pw.dio(x, sr, f0_floor=60, f0_ceil=1100, frame_period=10)
    f0 = pw.stonemask(x, f0, t, sr)
    out = []
    for n in notes:
        if n.dur < 0.25:
            continue
        a0, a1 = int((n.t + n.dur * 0.3) / 0.01), int((n.t + n.dur * 0.7) / 0.01)
        v = f0[a0:a1]
        v = v[v > 0]
        if len(v) >= 3:
            out.append(float(1200 * np.log2(np.median(v) / (440 * 2 ** ((n.midi - 69) / 12)))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--bars", help="only report bars in this range, e.g. 30-40")
    ap.add_argument("--late-ms", type=float, default=70)
    ap.add_argument("--tempo", help="as in sing.py, e.g. 8=116")
    ap.add_argument("--engine", choices=["diffsinger", "mbrola"], default="diffsinger")
    ap.add_argument("--parts", help="only these parts, comma-separated (e.g. T.,B.)")
    a = ap.parse_args()
    score = read_pdf(a.pdf)
    if a.tempo:
        from sing import parse_tempo

        f = parse_tempo(a.tempo)[0] / score.tempos[0][1]
        score.tempos = [(t, b * f) for t, b in score.tempos]
    perf = build(score)
    order = measure_order(score)
    # performance time -> printed bar number
    spans = []
    t = 0.0
    for mi, _ in order:
        m = score.measures[mi]
        d = m.length * 60 / score.tempos[0][1]
        spans.append((t, t + d, m.number))
        t += d

    def bar_at(tt):
        for s, e, n in spans:
            if s <= tt < e:
                return n
        return spans[-1][2]

    lo, hi = (int(x) for x in a.bars.split("-")) if a.bars else (-1, 10 ** 6)
    total_late = []
    total_cents = []
    for part in score.parts:
        if a.parts and part.name not in a.parts.split(","):
            continue
        mids = [n.midi for v in part.voices for n in v.notes]
        for vi, notes in enumerate(perf.voices[part.name]):
            if a.engine == "diffsinger":
                y = dsinger.sing(notes, perf.duration + 1, pick_voice_kind(part.name, mids))
            else:
                y = sing(notes, perf.duration + 1, pick_voice_model(part.name, mids))
            cents = pitch_errors(y, notes)
            total_cents += cents
            env = envelope(y)
            level = np.median(env[env > env.max() * 0.05]) if (env > 0).any() else 1
            silent, late = [], []
            for i, n in enumerate(notes):
                bar = bar_at(n.t + 1e-3)
                if not (lo <= bar <= hi) or n.dur < 0.15:
                    continue
                a0, a1 = int((n.t + n.dur * 0.3) / 0.01), int((n.t + n.dur * 0.8) / 0.01)
                mid = env[a0:a1].mean() if a1 > a0 else 0
                if mid < level * 0.15:
                    silent.append((bar, round(n.t, 2), n.midi))
                    continue
                if n.syllable is None:
                    continue  # melisma: no new onset
                # sound onset: first 10 ms frame after (beat - 150 ms) that reaches half the note level
                b0 = int((n.t - 0.15) / 0.01)
                seg = env[b0:a1]
                hit = np.nonzero(seg >= mid * 0.5)[0]
                prev_sounding = i > 0 and notes[i - 1].t + notes[i - 1].dur > n.t - 0.03
                if len(hit) and not prev_sounding:
                    lag = (b0 + hit[0]) * 0.01 - n.t
                    total_late.append(lag)
                    if lag * 1000 > a.late_ms:
                        onset = "".join(n.syllable.onset) + "-" + n.syllable.nucleus
                        late.append((bar, round(lag * 1000), onset))
            name = f"{part.name} line {vi + 1}"
            off = sum(abs(c) > 50 for c in cents)
            print(f"{name:14s} notes {len(notes):4d}  silent: {len(silent):3d}  late(>{a.late_ms:.0f}ms): {len(late):3d}"
                  f"  pitch: median {np.median(np.abs(cents)) if cents else 0:.0f} cents off, {off} notes >50 cents")
            for s in silent[:8]:
                print(f"    silent  bar {s[0]}  t={s[1]}s  midi {s[2]}")
            for l in late[:8]:
                print(f"    late    bar {l[0]}  +{l[1]} ms  ({l[2]})")
    if total_cents:
        print(f"pitch (middle of notes >=0.25s): median {np.median(np.abs(total_cents)):.0f} cents off")
    if total_late:
        print(f"\nonset after a rest: median {np.median(total_late) * 1000:.0f} ms, 90% {np.percentile(total_late, 90) * 1000:.0f} ms")


if __name__ == "__main__":
    main()
