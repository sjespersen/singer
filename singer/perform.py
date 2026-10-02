"""Turn a Score into a performance: repeats unrolled, quarters converted to seconds, and
every note paired with the syllable it is sung on."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .model import Score, Syllable, Voice
from .phonemes import SungSyllable, clean_word, phonemize_words, regroup_words


@dataclass
class SungNote:
    t: float  # seconds
    dur: float
    midi: int
    syllable: SungSyllable | None  # None: melisma, keeps the previous vowel
    spoken: bool = False


@dataclass
class Performance:
    duration: float
    beats: list[tuple[float, int]]  # (seconds, level: 2 bar, 1 beat, 0 subdivision)
    beat_len_first: float
    beats_per_bar_first: int
    voices: dict[str, list[list[SungNote]]] = field(default_factory=dict)


def measure_order(score: Score) -> list[tuple[int, int]]:
    """(measure index, pass number starting at 1) in performance order."""
    order = []
    i, start = 0, 0
    passes: dict[int, int] = {}
    while i < len(score.measures):
        m = score.measures[i]
        if m.repeat_start:
            start = i
        p = passes.get(start, 1)
        order.append((i, p))
        if m.repeat_end and p < m.repeat_times:
            passes[start] = p + 1
            i = start
            continue
        if m.repeat_end:
            start = i + 1
        i += 1
    return order


def _tempo_at(score: Score, q: float) -> float:
    bpm = score.tempos[0][1]
    for t, b in score.tempos:
        if t <= q + 1e-6:
            bpm = b
    return bpm


def _beat_unit(ts: tuple[int, int]) -> tuple[float, int]:
    num, den = ts
    if den == 8 and num % 3 == 0 and num > 3:  # compound time: dotted-quarter beats
        return 1.5, num // 3
    return 4 / den, num


def build(score: Score, lang: str = "en-us", click_unit: float | None = None) -> Performance:
    """click_unit: metronome click spacing in quarters (default: the bar's beat, e.g. a dotted
    quarter in 6/8). Clicks are (seconds, level): 2 bar start, 1 beat, 0 subdivision."""
    order = measure_order(score)
    clock = []  # performance clock for each (measure, pass)
    t = 0.0
    beats = []
    for mi, p in order:
        m = score.measures[mi]
        spq = 60.0 / _tempo_at(score, m.start)
        clock.append((mi, p, t, spq))
        beat, _ = _beat_unit(m.time_sig)
        unit = click_unit or beat
        full = m.time_sig[0] * 4 / m.time_sig[1]
        phase = full - m.length if m.length < full - 1e-6 else 0.0  # pickup: count from the bar's end
        q = 0.0
        while q < m.length - 1e-6:
            pos = q + phase
            level = 2 if abs(pos) < 1e-6 else (1 if abs(pos / beat - round(pos / beat)) < 1e-6 else 0)
            beats.append((t + q * spq, level))
            q += unit
        t += m.length * spq
    ts = score.measures[1 if len(score.measures) > 1 else 0].time_sig
    beat, per_bar = _beat_unit(ts)
    unit = click_unit or beat
    clicks_per_bar = round(ts[0] * 4 / ts[1] / unit)
    perf = Performance(t, beats, unit * 60.0 / _tempo_at(score, 0), clicks_per_bar)
    perf.beat_unit = beat

    for part in score.parts:
        perf.voices[part.name] = [_sing_voice(score, v, clock, lang) for v in part.voices]
    return perf


def _sing_voice(score: Score, voice: Voice, clock, lang: str) -> list[SungNote]:
    by_measure: dict[int, list] = {}
    for n in voice.notes:
        by_measure.setdefault(n.measure, []).append(n)
    events = []  # (seconds, dur seconds, note, syllable text or None)
    for mi, p, t0, spq in clock:
        m = score.measures[mi]
        for n in by_measure.get(mi, []):
            syl = n.lyrics.get(p) or n.lyrics.get(1)
            if p > 1 and p not in n.lyrics and n.lyrics:
                # later passes reuse the last verse printed
                syl = n.lyrics[max(n.lyrics)]
            events.append((t0 + (n.start - m.start) * spq, n.dur * spq, n, syl))

    # words: consecutive hyphenated syllables
    words: list[list[str]] = []
    word_of: list[tuple[int, int] | None] = []
    cur: list[str] = []
    last_text = None
    prev_end = -1.0
    prev_midi = None
    for idx, (t, d, n, syl) in enumerate(events):
        repeated, prev_midi = n.midi == prev_midi, n.midi
        if syl is None:
            gap = t - prev_end > 0.05
            if (gap or repeated) and last_text:
                # a note after a rest, or a repeated note (ties are already joined), without
                # text is sung again on the last syllable; only a slur to a new pitch is legato
                syl = Syllable(last_text)
                events[idx] = (t, d, n, syl)
            else:
                word_of.append(None)
                prev_end = t + d
                continue
        # words glued by punctuation under one note (say;"throw) are two words on that note
        text = re.sub(r"(?<=[^\W\d_])[;:,.!?\"“”]+(?=[^\W\d_])", " ", syl.text)
        text = text if clean_word(text) else "ah"
        cur.append(text)
        word_of.append((len(words), len(cur) - 1))
        if not syl.hyphen_after:
            words.append(cur)
            cur = []
        last_text = syl.text if not syl.hyphen_after else last_text
        prev_end = t + d
    if cur:
        words.append(cur)
    groups = regroup_words(words)
    new_of = {k: (gi, si) for gi, g in enumerate(groups) for si, k in enumerate(g)}
    words = [[words[wi][si] for wi, si in g] for g in groups]
    word_of = [new_of[w] if w is not None else None for w in word_of]
    # hyphen chains that were cut short by the end still form a word
    sung = phonemize_words(words, lang)
    out = []
    for (t, d, n, syl), w in zip(events, word_of):
        s = None
        if w is not None:
            wi, si = w
            if wi < len(sung) and si < len(sung[wi]):
                s = sung[wi][si]
        out.append(SungNote(t, d, n.midi, s, n.x_notehead))
    return out
