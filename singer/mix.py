"""Mix sung parts into a stereo rehearsal track, add metronome clicks, write MP3.

Mix layout follows the hand-made rehearsal tracks: in a focus track the chosen part is
panned hard left and the rest of the choir sits quieter on the right; a tutti track
spreads all parts evenly across the stereo field.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from .synth import SR


def click(freq: float, length: float = 0.035) -> np.ndarray:
    t = np.arange(int(length * SR)) / SR
    env = np.exp(-t * 90)
    return (np.sin(2 * np.pi * freq * t) * env + 0.3 * np.sin(2 * np.pi * freq * 2.01 * t) * env).astype(np.float32)


def metronome(beats: list[tuple[float, int]], n: int) -> np.ndarray:
    """beats: (seconds, level) with level 2 = bar start, 1 = beat, 0 = subdivision."""
    out = np.zeros(n, np.float32)
    sounds = {2: click(1760) * 1.0, 1: click(1320) * 0.8, 0: click(1100, 0.025) * 0.45}
    for t, level in beats:
        i = int(t * SR)
        c = sounds[int(level)]
        m = min(len(c), n - i)
        if 0 <= i < n:
            out[i:i + m] += c[:m]
    return out


def reverb_ir(seconds: float = 1.1, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal((2, n)).astype(np.float32) * np.exp(-t * 6.9 / seconds)
    ir[:, : int(0.02 * SR)] = 0  # pre-delay keeps the words clear
    # darker tail
    k = np.ones(16, np.float32) / 16
    ir = np.stack([np.convolve(ch, k, mode="same") for ch in ir])
    return ir / np.sqrt((ir ** 2).sum(axis=1, keepdims=True))


def _level(x: np.ndarray) -> float:
    active = np.abs(x) > 1e-3
    return float(np.sqrt(np.mean(x[active] ** 2))) if active.any() else 1.0


def _pan(x: np.ndarray, pos: float) -> np.ndarray:
    """pos -1 (left) .. +1 (right), equal power."""
    a = (pos + 1) * np.pi / 4
    return np.stack([x * np.cos(a), x * np.sin(a)])


def mix(parts: dict[str, np.ndarray], focus: str | None, clicks: np.ndarray | None,
        others_gain: float = 0.1, wet: float = 0.07) -> np.ndarray:
    n = max(len(x) for x in parts.values())
    if clicks is not None:
        n = max(n, len(clicks))
    names = list(parts)
    stereo = np.zeros((2, n), np.float32)
    for i, name in enumerate(names):
        x = np.zeros(n, np.float32)
        x[: len(parts[name])] = parts[name]
        x *= 0.2 / _level(x)
        if focus is None:
            pos = -0.7 + 1.4 * i / max(1, len(names) - 1)
            stereo += _pan(x, pos)
        elif name == focus:
            stereo[0] += x * 1.2
        else:
            stereo[1] += x * others_gain
    ir = reverb_ir()
    stereo = stereo + wet * np.stack([fftconvolve(stereo[c], ir[c])[:n] for c in range(2)])
    if clicks is not None:
        c = clicks[:n] * 0.35
        if focus is None:
            stereo += c
        else:
            stereo[1] += c * 1.3
    return loudness(stereo)


def loudness(stereo: np.ndarray, target_rms: float = 0.2, ceiling: float = 0.93) -> np.ndarray:
    """Bring the track to a consistent, fairly loud level, then catch peaks with a limiter."""
    mono = np.abs(stereo).max(axis=0)
    active = mono > 0.01 * (mono.max() or 1)
    # level by the louder channel: a focus track has almost everything on the left, and
    # averaging both channels would push that side into heavy limiting (audible distortion)
    rms = max(float(np.sqrt(np.mean(stereo[c, active] ** 2))) for c in range(2)) if active.any() else 1.0
    stereo = stereo * (target_rms / max(rms, 1e-6))
    # look-ahead limiter: gain follows the peak envelope, attacks in 5 ms, releases over 80 ms
    from scipy.ndimage import maximum_filter1d, uniform_filter1d

    env = maximum_filter1d(np.abs(stereo).max(axis=0), size=int(0.005 * SR) * 2 + 1)
    gain = np.minimum(1.0, ceiling / np.maximum(env, 1e-9))
    gain = -maximum_filter1d(-gain, size=int(0.08 * SR))  # hold the lowest gain for 80 ms
    gain = uniform_filter1d(gain, size=int(0.01 * SR))
    out = stereo * gain
    return np.clip(np.nan_to_num(out), -ceiling, ceiling)


def write_mp3(stereo: np.ndarray, path: str, kbps: int = 192):
    import lameenc

    enc = lameenc.Encoder()
    enc.set_bit_rate(kbps)
    enc.set_in_sample_rate(SR)
    enc.set_channels(2)
    enc.set_quality(2)
    pcm = (np.clip(stereo.T, -1, 1) * 32767).astype("<i2").tobytes()
    data = enc.encode(pcm) + enc.flush()
    with open(path, "wb") as f:
        f.write(data)
