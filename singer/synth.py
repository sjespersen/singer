"""Sing a voice line with MBROLA: sung notes -> .pho file (phonemes with durations and
pitch curves) -> mono audio."""
from __future__ import annotations

import functools
import math
import os
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .perform import SungNote
from .phonemes import VOWELS, SungSyllable

ROOT = Path(__file__).resolve().parent.parent
MBROLA = os.environ.get("MBROLA_BIN", str(ROOT / "vendor" / "MBROLA" / "Bin" / "mbrola"))
VOICE_DIR = Path(os.environ.get("MBROLA_VOICES", str(ROOT / "vendor" / "voices")))
SR = 44100
MBROLA_SR = 16000

CONS_MS = {
    "p": 60, "b": 50, "t": 55, "d": 45, "k": 65, "g": 50, "4": 25,
    "f": 80, "v": 60, "s": 90, "z": 70, "S": 90, "Z": 70, "T": 80, "D": 45, "h": 60,
    "tS": 100, "dZ": 90, "m": 60, "n": 60, "N": 60, "l": 55, "r": 55, "w": 55, "j": 50,
}
NASALS = {"m", "n", "N"}


def hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


class Timeline:
    """Phoneme segments on an absolute time axis, written out as MBROLA .pho lines."""

    def __init__(self):
        self.segs: list[tuple[str, float, float, list[tuple[float, float]]]] = []  # ph, start, end, [(t, hz)]
        self.cursor = 0.0

    def add(self, ph: str, start: float, end: float, pitch: list[tuple[float, float]]):
        if start > self.cursor + 0.002:
            self.segs.append(("_", self.cursor, start, []))
        start = max(start, self.cursor)
        if end - start < 0.012:
            end = start + 0.012
        # two identical vowels in a row have no diphone: separate them with a tiny break
        if self.segs and self.segs[-1][0] == ph and ph in VOWELS and abs(self.segs[-1][2] - start) < 0.002:
            ph0, s0, e0, p0 = self.segs[-1]
            self.segs[-1] = (ph0, s0, e0 - 0.025, p0)
            self.segs.append(("_", e0 - 0.025, start, []))
        self.segs.append((ph, start, end, pitch))
        self.cursor = end

    def pho(self, total: float, self_joins: frozenset = frozenset()) -> str:
        if total > self.cursor:
            self.segs.append(("_", self.cursor, total + 0.2, []))
        lines = ["_ 20"]
        pos_ms = 0
        segs = []
        for ph, s, e, pitch in self.segs:
            # MBROLA stretches a whole phoneme, including its transitions to the neighbours.
            # On a long note that turns the quiet consonant->vowel transition into a late
            # onset, so a long sound keeps natural-length edges and only its middle is held
            # (a phoneme joined to itself). Long pieces also keep lines under the size limit.
            cuts = [s, e]
            if ph in self_joins and e - s > 0.2:
                edge = 0.07
                k = int((e - s - 2 * edge) // 1.2) + 1
                cuts = [s] + [s + edge + (e - s - 2 * edge) * j / k for j in range(k + 1)] + [e]
            elif e - s > 1.2 and (ph == "_" or ph in self_joins):
                k = int((e - s) // 1.2) + 1
                cuts = [s + (e - s) * j / k for j in range(k + 1)]
            for a, b in zip(cuts, cuts[1:]):
                p = pitch
                if len(cuts) > 2 and pitch:
                    ts, fs = zip(*sorted(pitch))
                    p = [(a, float(np.interp(a, ts, fs)))] + [q for q in pitch if a < q[0] < b] + [(b, float(np.interp(b, ts, fs)))]
                segs.append((ph, a, b, p))
        for ph, s, e, pitch in segs:
            end_ms = int(round(e * 1000))
            start_ms = int(round(s * 1000))
            if start_ms > pos_ms and ph != "_":
                lines.append(f"_ {start_ms - pos_ms}")
                pos_ms = start_ms
            dur = end_ms - pos_ms
            if dur <= 0:
                continue
            pts = []
            last = -1.0
            for t, f in sorted(pitch):
                pct = round((t - pos_ms / 1000) / (dur / 1000) * 100, 1)
                # points must be strictly increasing, or MBROLA panics on pitch jumps
                if -0.5 <= pct <= 100.5 and pct > last:
                    pts.append(f"{min(100.0, max(0.0, pct)):.1f} {f:.1f}")
                    last = pct
            # MBROLA reads lines into a fixed buffer: keep them short
            if len(pts) > 60:
                step = len(pts) / 60
                pts = [pts[int(i * step)] for i in range(60)]
            lines.append(f"{ph} {dur} " + " ".join(pts))
            pos_ms = end_ms
        return "\n".join(lines) + "\n"


def _pitch_curve(notes: list[SungNote], start: float, end: float, vibrato: bool, seed: int,
                 scoop: bool = True):
    """Pitch points (t, hz) across a vowel that may span several notes (a melisma)."""
    rng = np.random.default_rng(seed)
    pts = []
    for i, n in enumerate(notes):
        a = max(start, n.t)
        b = min(end, n.t + n.dur)
        if b <= a:
            continue
        f = hz(n.midi)
        glide = min(0.045, (b - a) / 3)
        if i > 0:
            pts.append((a, hz(notes[i - 1].midi)))
            pts.append((a + glide, f))
        else:
            pts.append((a, f * 2 ** ((-0.35 if scoop else 0) / 12)))  # slight scoop into the first note
            pts.append((a + glide, f))
        span = b - a
        # a human never holds a pitch perfectly: slow wander of a few cents
        drift = 0.0
        t = a + glide + 0.12
        while t < b - 0.05 and not (vibrato and span > 0.5):
            drift = 0.7 * drift + rng.normal(0, 0.035)
            pts.append((t, f * 2 ** (drift / 12)))
            t += 0.12
        if vibrato and span > 0.5:
            # smooth sine vibrato that fades in after the note has settled, with
            # slightly varying rate and depth so it doesn't sound mechanical
            rate = 5.0 + rng.uniform(-0.35, 0.35)
            depth_max = 0.16 + rng.uniform(0, 0.06)  # semitones
            onset = a + min(0.3, span * 0.3)
            t, phase = onset, 0.0
            step = 1 / rate / 8
            while t < b - 0.04:
                depth = min(1.0, (t - onset) / 0.35) * depth_max
                drift = 0.9 * drift + rng.normal(0, 0.01)
                pts.append((t, f * 2 ** ((depth * math.sin(phase) + drift) / 12)))
                phase += 2 * math.pi * rate * step * (1 + rng.normal(0, 0.03))
                t += step
        pts.append((b, f))
    return pts


# Vowel modification: closed vowels lose almost all their sound once the pitch rises above
# their first formant, so (like real sopranos) they are opened up on high notes.
# Thresholds from measuring the loudness of each vowel against pitch.
VOWEL_MOD = {
    "i": [(500, "I"), (640, "E"), (800, "{")],
    "I": [(640, "E"), (800, "{")],
    "u": [(500, "U"), (800, "O")],
    "U": [(800, "O")],
    "E": [(800, "{")],
    "EI": [(800, "E"), (850, "{")],
    "@U": [(800, "O")],
}


def modify_vowel(v: str, f: float) -> str:
    out = v
    for limit, repl in VOWEL_MOD.get(v, []):
        if f >= limit:
            out = repl
    return out


def _dur(ph: str) -> float:
    return CONS_MS.get(ph, 60) / 1000


@functools.lru_cache(maxsize=None)
def self_joins(voice: str) -> frozenset:
    """Sounds this MBROLA voice can join to themselves (not every voice has all of them)."""
    ok = set()
    with tempfile.TemporaryDirectory() as tmp:
        pf, wf = Path(tmp) / "p.pho", Path(tmp) / "p.wav"
        for ph in sorted(VOWELS | {"m", "n", "N", "l"}):
            pf.write_text(f"_ 50\nd 50 0 200\n{ph} 70 0 200\n{ph} 300 0 200\n{ph} 70 0 200\n_ 50\n")
            r = subprocess.run([MBROLA, str(VOICE_DIR / voice), str(pf), str(wf)], capture_output=True, text=True)
            if r.returncode == 0 and not r.stderr.strip():
                ok.add(ph)
    return frozenset(ok)


def render_pho(notes: list[SungNote], total: float, seed: int = 0, detune_cents: float = 0.0,
               voice: str | None = None) -> str:
    tl = Timeline()
    # groups: a syllable note plus the melisma notes that follow it
    groups: list[tuple[SungSyllable, list[SungNote]]] = []
    for n in notes:
        if n.syllable is None and groups and abs(groups[-1][1][-1].t + groups[-1][1][-1].dur - n.t) < 0.02:
            groups[-1][1].append(n)
        else:
            groups.append((n.syllable or SungSyllable(nucleus="A"), [n]))

    def lead_of(g):
        syl = g[0]
        return min(sum(_dur(p) for p in syl.onset) * 0.55, 0.07)

    for gi, (syl, gnotes) in enumerate(groups):
        if detune_cents:
            gnotes = [SungNote(n.t, n.dur, n.midi + detune_cents / 100, n.syllable, n.spoken) for n in gnotes]
        start = gnotes[0].t
        end = gnotes[-1].t + gnotes[-1].dur
        nxt = groups[gi + 1] if gi + 1 < len(groups) else None
        if nxt and nxt[1][0].t - end < 0.02:
            end = nxt[1][0].t - lead_of(nxt)  # make room for the next consonant before the beat
        else:
            end -= min(0.04, (end - start) * 0.1)  # breathe before rests
        subs = [syl] + list(syl.extra)
        if len(subs) > 1:
            # several words on one note share it evenly
            span = (end - start) / len(subs)
            for k, s in enumerate(subs):
                _place(tl, s, start + k * span, start + (k + 1) * span, gnotes, k == 0, seed + gi)
        else:
            _place(tl, syl, start, end, gnotes, True, seed + gi)
    return tl.pho(total, self_joins(voice) if voice else frozenset())


def _place(tl: Timeline, syl: SungSyllable, start: float, end: float, notes: list[SungNote], lead: bool, seed: int):
    length = end - start
    onset = [(p, _dur(p)) for p in syl.onset]
    coda = [(p, _dur(p)) for p in syl.coda]
    if coda and coda[-1][0] in NASALS and len(coda) == 1:
        coda[-1] = (coda[-1][0], min(0.16, max(0.07, length * 0.28)))  # hummed ending ("dam")
    on_total = sum(d for _, d in onset)
    co_total = sum(d for _, d in coda)
    budget = length * 0.55
    if on_total + co_total > budget:
        k = budget / (on_total + co_total)
        onset = [(p, d * k) for p, d in onset]
        coda = [(p, d * k) for p, d in coda]
        on_total *= k
        co_total *= k
    t = start - (min(on_total * 0.55, 0.07) if lead else 0)
    f_first = hz(notes[0].midi)
    f_last = hz(notes[-1].midi)
    for p, d in onset:
        tl.add(p, t, t + d, [(t, f_first), (t + d, f_first)])
        t = tl.cursor
    v_end = end - co_total
    pitch = _pitch_curve(notes, t, v_end, vibrato=True, seed=seed)
    # a held schwa is sung as a full "uh" (as singers do on "Christ-mas___")
    nucleus = "V" if syl.nucleus == "@" and v_end - t > 0.4 else syl.nucleus
    nucleus = modify_vowel(nucleus, max(hz(n.midi) for n in notes))
    tl.add(nucleus, t, v_end, pitch)
    t = tl.cursor
    for p, d in coda:
        tl.add(p, t, t + d, [(t, f_last), (t + d, f_last)])
        t = tl.cursor


def _biquad_shelf(x: np.ndarray, sr: int, f0: float, gain_db: float, high: bool) -> np.ndarray:
    """RBJ-cookbook shelving filter."""
    from scipy.signal import lfilter

    A = 10 ** (gain_db / 40)
    w = 2 * math.pi * f0 / sr
    cw, sw = math.cos(w), math.sin(w)
    alpha = sw / 2 * math.sqrt(2)
    sa = 2 * math.sqrt(A) * alpha
    if high:
        b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sa)]
        a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    else:
        b = [A * ((A + 1) - (A - 1) * cw + sa), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sa)]
        a = [(A + 1) + (A - 1) * cw + sa, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sa]
    return lfilter(np.array(b) / a[0], np.array(a) / a[0], x).astype(np.float32)


UNVOICED = {"_", "p", "t", "k", "f", "s", "S", "T", "h", "tS"}


def _f0_from_pho(pho: str, n_frames: int, frame: float) -> np.ndarray:
    """The pitch track we asked MBROLA for, rebuilt from the .pho file (frame times in s)."""
    t = 0.0
    pts, voiced = [], []
    for line in pho.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        ph, dur = parts[0], int(parts[1]) / 1000
        nums = [float(v) for v in parts[2:]]
        for i in range(0, len(nums) - 1, 2):
            pts.append((t + dur * nums[i] / 100, nums[i + 1]))
        if ph not in UNVOICED:
            voiced.append((t, t + dur))
        t += dur
    times = np.arange(n_frames) * frame
    if not pts:
        return np.zeros(n_frames)
    pts.sort()
    f0 = np.interp(times, [p[0] for p in pts], [p[1] for p in pts])
    mask = np.zeros(n_frames, bool)
    for a, b in voiced:
        mask[int(a / frame):int(np.ceil(b / frame)) + 1] = True
    return np.where(mask, f0, 0.0)


def _world_smooth(x: np.ndarray, sr: int, pho: str) -> np.ndarray:
    """Re-synthesise with the WORLD vocoder: a clean excitation and a time-smoothed
    spectral envelope replace MBROLA's buzzy overlap-add at high pitches. The pitch is the
    one we asked for, not an estimate, so low notes and hums don't crackle."""
    import pyworld as pw

    xd = x.astype(np.float64)
    frame = 0.005
    n = int(len(xd) / sr / frame) + 1
    f0 = _f0_from_pho(pho, n, frame)
    t = np.arange(n) * frame
    sp = pw.cheaptrick(xd, f0, t, sr)
    ap = pw.d4c(xd, f0, t, sr)
    # smooth the envelope over ~15 ms so frame-to-frame jitter (the "metal") goes away
    k = np.array([0.25, 0.5, 0.25])
    sp = np.exp(np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, np.log(sp + 1e-12)))
    y = pw.synthesize(f0, np.ascontiguousarray(sp), ap, sr, frame * 1000)
    y = y[: len(x)]
    # WORLD's pulses are very peaky: tame the crest factor without touching the level
    return (np.tanh(y * 2.5) / 2.5).astype(np.float32)


def _dynamics(y: np.ndarray, notes: list[SungNote]) -> np.ndarray:
    """Phrase-like loudness: long notes swell a little then relax, phrase ends taper."""
    env = np.ones(len(y), np.float32)
    for i, n in enumerate(notes):
        a, b = int(n.t * SR), int((n.t + n.dur) * SR)
        if b <= a or a >= len(y):
            continue
        b = min(b, len(y))
        L = b - a
        if n.dur > 0.6:
            x = np.linspace(0, 1, L, dtype=np.float32)
            swell = np.clip(np.sin(np.pi * np.minimum(1, x * 1.6)), 0, 1) ** 0.8
            env[a:b] *= 0.86 + 0.14 * swell - 0.1 * x ** 2
        nxt = notes[i + 1] if i + 1 < len(notes) else None
        if nxt is None or nxt.t - (n.t + n.dur) > 0.05:  # last note of a phrase
            r = min(L, int(0.12 * SR))
            env[b - r:b] *= np.linspace(1, 0.55, r, dtype=np.float32)
    return y * env


def synthesize(pho: str, voice: str, smooth: bool = True) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        pf = Path(tmp) / "v.pho"
        wf = Path(tmp) / "v.wav"
        pf.write_text(pho)
        r = subprocess.run([MBROLA, "-e", str(VOICE_DIR / voice), str(pf), str(wf)], capture_output=True, text=True)
        if r.returncode != 0 or not wf.exists():
            raise RuntimeError(f"mbrola failed: {r.stderr.strip()[:500]}")
        with wave.open(str(wf)) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
            sr = w.getframerate()
    if smooth:
        x = _world_smooth(x, sr, pho)
    g = math.gcd(SR, sr)
    y = resample_poly(x, SR // g, sr // g).astype(np.float32)
    # warmer, less buzzy tone: take the edge off the upper harmonics, add a little body
    y = _biquad_shelf(y, SR, 3000, -5.0, high=True)
    y = _biquad_shelf(y, SR, 250, 2.0, high=False)
    # MBROLA's leading "_ 20" silence
    return y[int(0.02 * SR):]


def sing(notes: list[SungNote], total: float, voice: str, singers: int = 1, seed: int = 0,
         smooth: bool = True) -> np.ndarray:
    """Render a voice line. Extra singers are clearly offset in time and pitch (a few ms
    apart would comb-filter into a metallic flanger sound)."""
    n_out = int((total + 0.5) * SR)
    out = np.zeros(n_out, np.float32)
    rng = np.random.default_rng(seed + 99)
    for k in range(singers):
        dc = 0.0 if k == 0 else rng.uniform(6, 12) * (1 if k % 2 else -1)
        y = synthesize(render_pho(notes, total, seed=seed * 31 + k, detune_cents=dc, voice=voice), voice, smooth)
        delay = 0 if k == 0 else int(rng.uniform(0.025, 0.045) * SR)
        m = min(len(y), n_out - delay)
        out[delay:delay + m] += y[:m] * (1.0 if k == 0 else 0.7)
    return _dynamics(out / math.sqrt(1 + 0.49 * (singers - 1)), notes)
