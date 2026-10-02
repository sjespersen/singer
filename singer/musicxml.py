"""Write a Score as MusicXML 4.0 (partwise), so the recognised notation can be checked and
corrected in MuseScore, Sibelius, Dorico or Finale."""
from __future__ import annotations

from xml.sax.saxutils import escape

from .model import Note, Score

DIV = 48  # divisions per quarter: covers 64ths (3) and triplets (e.g. 16 = triplet eighth)
TYPES = [(192, "whole"), (96, "half"), (48, "quarter"), (24, "eighth"), (12, "16th"), (6, "32nd"), (3, "64th")]
STEPS = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]
ALTERS = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
FLAT_STEPS = ["C", "D", "D", "E", "E", "F", "G", "G", "A", "A", "B", "B"]
FLAT_ALTERS = [0, -1, 0, -1, 0, 0, -1, 0, -1, 0, -1, 0]


def _shapes():
    """duration in divisions -> (type, dots, triplet)"""
    s = {}
    for d, t in TYPES:
        s.setdefault(d, (t, 0, False))
        s.setdefault(d * 3 // 2, (t, 1, False)) if d % 2 == 0 else None
        s.setdefault(d * 7 // 4, (t, 2, False)) if d % 4 == 0 else None
        if d * 2 % 3 == 0:
            s.setdefault(d * 2 // 3, (t, 0, True))
    return s


SHAPES = _shapes()


def _split(d: int) -> list[int]:
    """Split a duration into notatable pieces (tied together)."""
    out = []
    while d > 0:
        if d in SHAPES:
            out.append(d)
            break
        piece = max(k for k in SHAPES if k <= d) if any(k <= d for k in SHAPES) else d
        out.append(piece)
        d -= piece
    return out


def _pitch(midi: int, fifths: int) -> str:
    pc, octave = midi % 12, midi // 12 - 1
    if fifths < 0:
        step, alter = FLAT_STEPS[pc], FLAT_ALTERS[pc]
    else:
        step, alter = STEPS[pc], ALTERS[pc]
    alt = f"<alter>{alter}</alter>" if alter else ""
    return f"<pitch><step>{step}</step>{alt}<octave>{octave}</octave></pitch>"


def _clef(part) -> str:
    lo, hi = part.lowest(), part.highest()
    mid = (lo + hi) / 2
    if mid < 55:
        return "<clef><sign>F</sign><line>4</line></clef>"
    name = part.name.lower().lstrip()
    if name.startswith("t") or (mid < 62 and not name.startswith(("s", "a", "m"))):
        return "<clef><sign>G</sign><line>2</line><clef-octave-change>-1</clef-octave-change></clef>"
    return "<clef><sign>G</sign><line>2</line></clef>"


def _syllabic(prev_hyphen: bool, hyphen: bool) -> str:
    if prev_hyphen and hyphen:
        return "middle"
    if prev_hyphen:
        return "end"
    if hyphen:
        return "begin"
    return "single"


def write_musicxml(score: Score, path: str):
    names = getattr(score, "display_names", {}) or {}
    out = ['<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
           '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
           '"http://www.musicxml.org/dtds/partwise.dtd">',
           '<score-partwise version="4.0">',
           f"<work><work-title>{escape(score.title)}</work-title></work>",
           "<identification><encoding><software>singer (PDF vector reader)</software></encoding></identification>",
           "<part-list>"]
    for i, p in enumerate(score.parts, 1):
        nm = names.get(p.name, p.name)
        out.append(f'<score-part id="P{i}"><part-name>{escape(nm)}</part-name>'
                   f"<part-abbreviation>{escape(p.name)}</part-abbreviation></score-part>")
    out.append("</part-list>")

    for i, p in enumerate(score.parts, 1):
        out.append(f'<part id="P{i}">')
        # notes of each voice line by measure
        per_measure = [[[] for _ in score.measures] for _ in p.voices]
        for vi, v in enumerate(p.voices):
            for n in v.notes:
                # notes merged across a tie are split again at barlines
                start, left, mi_, first = n.start, n.dur, n.measure, True
                while left > 1e-6 and mi_ < len(score.measures):
                    mm = score.measures[mi_]
                    piece = min(left, mm.start + mm.length - start)
                    seg = Note(start, piece, n.midi, n.lyrics if first else {}, mi_, x_notehead=n.x_notehead)
                    seg.tie_prev, seg.tie_next = not first, left - piece > 1e-6
                    per_measure[vi][mi_].append(seg)
                    start, left, mi_, first = start + piece, left - piece, mi_ + 1, False
        prev_hyph = [dict() for _ in p.voices]
        prev_m = None
        for mi, m in enumerate(score.measures):
            attrs = []
            if mi == 0:
                attrs.append(f"<divisions>{DIV}</divisions>")
            if prev_m is None or m.fifths != prev_m.fifths:
                attrs.append(f"<key><fifths>{m.fifths}</fifths></key>")
            full = m.time_sig[0] * 4 / m.time_sig[1]
            if prev_m is None or m.time_sig != prev_m.time_sig:
                attrs.append(f"<time><beats>{m.time_sig[0]}</beats><beat-type>{m.time_sig[1]}</beat-type></time>")
            if mi == 0:
                attrs.append(_clef(p))
            implicit = ' implicit="yes"' if mi == 0 and m.length < full - 1e-6 else ""
            out.append(f'<measure number="{m.number}"{implicit}>')
            if m.repeat_start:
                out.append('<barline location="left"><bar-style>heavy-light</bar-style><repeat direction="forward"/></barline>')
            if attrs:
                out.append("<attributes>" + "".join(attrs) + "</attributes>")
            if i == 1 and mi == 0:
                bpm = score.tempos[0][1]
                out.append('<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit>'
                           f"<per-minute>{bpm:g}</per-minute></metronome></direction-type><sound tempo=\"{bpm:g}\"/></direction>")
            if m.text:
                out.append(f'<direction placement="above"><direction-type><words>{escape(m.text)}</words></direction-type></direction>')
            mlen = round(m.length * DIV)
            for vi in range(len(p.voices)):
                if vi > 0:
                    out.append(f"<backup><duration>{mlen}</duration></backup>")
                out += _voice_measure(per_measure[vi][mi], m, mlen, vi + 1, prev_hyph[vi])
            if m.repeat_end:
                out.append('<barline location="right"><bar-style>light-heavy</bar-style>'
                           f'<repeat direction="backward" times="{m.repeat_times}"/></barline>')
            elif mi == len(score.measures) - 1:
                out.append('<barline location="right"><bar-style>light-heavy</bar-style></barline>')
            out.append("</measure>")
            prev_m = m
        out.append("</part>")
    out.append("</score-partwise>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def _voice_measure(notes: list[Note], m, mlen: int, voice: int, prev_hyph: dict) -> list[str]:
    res = []
    pos = 0
    notes = sorted(notes, key=lambda n: n.start)

    def rest(d):
        for piece in _split(d):
            t, dots, trip = SHAPES.get(piece, ("quarter", 0, False))
            res.append(f"<note><rest/><duration>{piece}</duration><voice>{voice}</voice><type>{t}</type>"
                       + "<dot/>" * dots + ("<time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification>" if trip else "")
                       + "</note>")

    for n in notes:
        at = round((n.start - m.start) * DIV)
        if at > pos:
            rest(at - pos)
            pos = at
        if at < pos:
            continue  # overlapping note in the same voice line: skip rather than write invalid XML
        d = min(round(n.dur * DIV), mlen - pos)
        pieces = _split(d)
        for k, piece in enumerate(pieces):
            t, dots, trip = SHAPES.get(piece, ("quarter", 0, False))
            ties = ""
            notations = ""
            if k > 0 or getattr(n, "tie_prev", False):
                ties += '<tie type="stop"/>'
                notations += '<tied type="stop"/>'
            if k < len(pieces) - 1 or getattr(n, "tie_next", False):
                if True:
                    ties += '<tie type="start"/>'
                    notations += '<tied type="start"/>'
            lyr = ""
            if k == 0:
                for verse, syl in sorted(n.lyrics.items()):
                    sb = _syllabic(prev_hyph.get(verse, False), syl.hyphen_after)
                    prev_hyph[verse] = syl.hyphen_after
                    lyr += f'<lyric number="{verse}"><syllabic>{sb}</syllabic><text>{escape(syl.text)}</text></lyric>'
            head = "<notehead>x</notehead>" if n.x_notehead else ""
            res.append(f"<note>{_pitch(n.midi, m.fifths)}<duration>{piece}</duration>{ties}<voice>{voice}</voice>"
                       f"<type>{t}</type>" + "<dot/>" * dots
                       + ("<time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification>" if trip else "")
                       + head + (f"<notations>{notations}</notations>" if notations else "") + lyr + "</note>")
        pos += d
    if pos < mlen:
        rest(mlen - pos)
    return res
