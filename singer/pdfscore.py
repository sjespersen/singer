"""Read a vector PDF exported from Sibelius (Opus / Helsinki music fonts) into a Score.

The PDF is not rasterised: staff lines, stems, beams and barlines are vector lines,
and noteheads, clefs, rests, accidentals and flags are glyphs of the Opus font, so the
notation can be reconstructed exactly from the drawing instructions.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import pymupdf

from .model import Measure, Note, Part, Score, Syllable, Voice

MUSIC_FONT = "OpusStd"
SPECIAL_FONT = "OpusSpecialStd"
TEXT_FONT = "OpusTextStd"

NOTEHEADS = {"œ": 1.0, "˙": 2.0, "w": 4.0, "¿": 1.0, "Ï": 1.0}
X_HEADS = {"¿"}
RESTS = {"Œ": 1.0, "‰": 0.5, "≈": 0.25, "Ó": 2.0, "®": 0.125, "∑": None}
FLAGS = {"j": 1, "J": 1, "r": 2, "R": 2, "Â": 3, "Ê": 3}
ACCIDENTALS = {"#": 1, "b": -1, "n": 0, "Ü": 2, "º": -2}
CLEFS = {"&": ("G", 30), "?": ("F", 18), "B": ("C", 24)}  # diatonic index of bottom staff line
STEP_SEMITONES = [0, 2, 4, 5, 7, 9, 11]
LETTERS = "CDEFGAB"
TUPLET_FACTORS = {"3": 2 / 3, "5": 4 / 5, "6": 4 / 6, "7": 4 / 7, "2": 3 / 2, "4": 3 / 4}


# --------------------------------------------------------------------------- raw objects


@dataclass
class Glyph:
    font: str
    ch: str
    x: float  # origin
    y: float
    x0: float
    x1: float
    size: float


@dataclass
class Word:
    text: str
    x0: float
    x1: float
    y: float
    size: float
    font: str

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass
class Line:
    x0: float
    y0: float
    x1: float
    y1: float
    width: float


@dataclass
class Beam:
    pts: list[tuple[float, float]]
    x0: float
    x1: float

    def yrange(self, x: float):
        ys = []
        n = len(self.pts)
        for i in range(n):
            (ax, ay), (bx, by) = self.pts[i], self.pts[(i + 1) % n]
            if abs(bx - ax) < 1e-6:
                if abs(ax - x) < 0.8:
                    ys += [ay, by]
                continue
            t = (x - ax) / (bx - ax)
            if -0.05 <= t <= 1.05:
                ys.append(ay + t * (by - ay))
        return (min(ys), max(ys)) if ys else None


@dataclass
class Curve:
    x0: float
    y0: float
    x1: float
    y1: float


# --------------------------------------------------------------------------- page model


@dataclass
class Staff:
    page: int
    x0: float
    x1: float
    lines: list[float]
    name: str = ""
    glyphs: list[Glyph] = field(default_factory=list)
    stems: list[Line] = field(default_factory=list)

    @property
    def top(self):
        return self.lines[0]

    @property
    def bottom(self):
        return self.lines[-1]

    @property
    def space(self):
        return (self.bottom - self.top) / 4

    @property
    def mid(self):
        return self.lines[2]


@dataclass
class Barline:
    x: float
    thick: bool = False
    repeat_start: bool = False
    repeat_end: bool = False


@dataclass
class System:
    page: int
    staves: list[Staff]
    barlines: list[Barline] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)


@dataclass
class Chord:
    x: float  # notehead left edge
    heads: list[Glyph]
    stem: Line | None
    up: bool
    base: float
    beams: int = 0
    dots: int = 0
    tuplet: float = 1.0
    is_rest: bool = False
    full_measure_rest: bool = False
    y: float = 0.0
    midis: list[int] = field(default_factory=list)
    lyrics: dict[int, Syllable] = field(default_factory=dict)
    lyrics_above: dict[int, Syllable] = field(default_factory=dict)
    tie_start: bool = False

    @property
    def dur(self) -> float:
        d = self.base / (2 ** self.beams)
        d *= 2 - 0.5 ** self.dots
        return d * self.tuplet


# --------------------------------------------------------------------------- extraction


def _extract_page(page):
    glyphs, words = [], []
    raw = page.get_text("rawdict")
    for b in raw["blocks"]:
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                font = s["font"]
                if font.startswith("Opus"):
                    for c in s["chars"]:
                        if c["c"].strip():
                            glyphs.append(Glyph(font, c["c"], c["origin"][0], c["origin"][1], c["bbox"][0], c["bbox"][2], s["size"]))
                    continue
                cur = []
                for c in s["chars"] + [None]:
                    if c is None or not c["c"].strip():
                        if cur:
                            words.append(Word("".join(ch["c"] for ch in cur), cur[0]["bbox"][0], cur[-1]["bbox"][2], cur[0]["origin"][1], s["size"], font))
                        cur = []
                    else:
                        cur.append(c)
    lines, beams, curves = [], [], []
    for d in page.get_drawings():
        kinds = "".join(i[0] for i in d["items"])
        if d["type"] == "s" and kinds == "l":
            (_, p1, p2) = d["items"][0]
            a, b = (p1, p2) if (p1.y, p1.x) <= (p2.y, p2.x) else (p2, p1)
            lines.append(Line(a.x, a.y, b.x, b.y, d.get("width") or 0))
        elif d["type"] == "f" and set(kinds) == {"l"}:
            pts = [(i[1].x, i[1].y) for i in d["items"]]
            beams.append(Beam(pts, min(p[0] for p in pts), max(p[0] for p in pts)))
        elif "c" in kinds and d["type"] in ("f", "fs"):
            r = d["rect"]
            curves.append(Curve(r.x0, r.y0, r.x1, r.y1))
    return glyphs, words, lines, beams, curves


def _find_staves(lines, page_no):
    horiz = [l for l in lines if abs(l.y0 - l.y1) < 0.05 and l.x1 - l.x0 > 60]
    if not horiz:
        return []
    width = Counter(round(l.width, 2) for l in horiz).most_common(1)[0][0]
    horiz = sorted((l for l in horiz if abs(l.width - width) < 0.05), key=lambda l: (round(l.x0), l.y0))
    groups = defaultdict(list)
    for l in horiz:
        groups[(round(l.x0, 0), round(l.x1, 0))].append(l.y0)
    staves = []
    for (x0, x1), ys in groups.items():
        ys = sorted(set(round(y, 2) for y in ys))
        i = 0
        while i + 4 < len(ys):
            cand = ys[i:i + 5]
            gaps = [b - a for a, b in zip(cand, cand[1:])]
            if max(gaps) - min(gaps) < 0.3 and min(gaps) > 2:
                staves.append(Staff(page_no, x0, x1, cand))
                i += 5
            else:
                i += 1
    return sorted(staves, key=lambda s: s.top)


def _find_systems(staves, lines, page_no):
    """Staves joined by a vertical line at their left edge belong to one system."""
    systems: list[list[Staff]] = []
    for st in staves:
        joined = False
        if systems:
            prev = systems[-1][-1]
            for l in lines:
                if abs(l.x0 - l.x1) < 0.05 and abs(l.x0 - st.x0) < 1.5 and l.y0 <= prev.top + 0.5 and l.y1 >= st.bottom - 0.5 and abs(prev.x0 - st.x0) < 1.5:
                    joined = True
                    break
        if joined:
            systems[-1].append(st)
        else:
            systems.append([st])
    return [System(page_no, s) for s in systems]


def _abbrev_matches(abbr: str, full: str) -> bool:
    ap = [p for p in abbr.split("/")]
    fp = [p for p in full.split("/")]
    if len(ap) != len(fp):
        return False
    for a, f in zip(ap, fp):
        a = a.strip().rstrip(".")
        m = re.fullmatch(r"([A-Za-zÄÖÜäöü]+)\.?(\d*)", a)
        if not m:
            return False
        letters, digits = m.groups()
        if not re.fullmatch(re.escape(letters) + r"[A-Za-zäöüÄÖÜ\-]*\s*" + re.escape(digits), f.strip(), flags=re.I):
            return False
    return True


def _up_side(stem, h, sp):
    # glyph boxes include side bearing: an up-stem sits ~0.35 spaces inside the right edge
    return h.x1 - sp * 0.6 < stem.x0 < h.x1 + sp * 0.15


def _down_side(stem, h, sp):
    return h.x0 - sp * 0.15 < stem.x0 < h.x0 + sp * 0.3


# --------------------------------------------------------------------------- the reader


class PdfScoreReader:
    def __init__(self, path: str):
        self.path = path
        self.doc = pymupdf.open(path)
        self.warnings: list[str] = []
        self.systems: list[System] = []
        self.title = ""
        self.tempo_marks: list[tuple[int, float, float, float]] = []  # (system idx, x, y, bpm)
        self.repeat_texts: list[tuple[int, float, float, int]] = []

    def warn(self, msg):
        self.warnings.append(msg)

    # ---- page level -----------------------------------------------------------------

    def _load(self):
        sys_idx = 0
        for page_no, page in enumerate(self.doc):
            glyphs, words, lines, beams, curves = _extract_page(page)
            staves = _find_staves(lines, page_no)
            if not staves:
                continue
            if page_no == 0:
                big = max(words, key=lambda w: w.size, default=None)
                if big:
                    self.title = " ".join(w.text for w in words if abs(w.size - big.size) < 0.1 and abs(w.y - big.y) < 2)
            systems = _find_systems(staves, lines, page_no)
            space = staves[0].space
            for sy in systems:
                self._assign_names(sy, words)
                self._assign_objects(sy, glyphs, lines)
                self._find_barlines(sy, lines, glyphs)
                sy.beams = beams
                sy.curves = curves
                top, bot = sy.staves[0].top - 12 * space, sy.staves[-1].bottom + 8 * space
                sy.words = [w for w in words if top < w.y < bot]
                self._find_tempo(sy, sys_idx, glyphs, words)
                for w in words:
                    m = re.fullmatch(r"(\d+)x", w.text)
                    if m and top < w.y < bot:
                        self.repeat_texts.append((sys_idx, w.x0, w.y, int(m.group(1))))
                self.systems.append(sy)
                sys_idx += 1

    def _assign_names(self, sy: System, words):
        for st in sy.staves:
            cands = [w for w in words if w.x1 < st.x0 - 2 and abs(w.y - st.mid - st.space * 0.8) < st.space * 2.5]
            cands = [w for w in cands if not w.text.isdigit()]
            cands.sort(key=lambda w: w.x0)
            st.name = " ".join(w.text for w in cands).strip() or f"Staff{sy.staves.index(st) + 1}"

    def _assign_objects(self, sy: System, glyphs, lines):
        for g in glyphs:
            if g.font == TEXT_FONT or g.font == "OpusChordsStd":
                continue
            best, bd = None, 1e9
            for st in sy.staves:
                if g.x < st.x0 - 30 or g.x > st.x1 + 5:
                    continue
                d = 0 if st.top - st.space <= g.y <= st.bottom + st.space else min(abs(g.y - st.top), abs(g.y - st.bottom))
                if d < bd:
                    best, bd = st, d
            if best is not None and bd < best.space * 7:
                best.glyphs.append(g)
        for l in lines:
            if abs(l.x0 - l.x1) > 0.05 or l.width > 0.55 or l.y1 - l.y0 < 5:
                continue
            best, bd = None, 1e9
            for st in sy.staves:
                if not (st.x0 < l.x0 < st.x1):
                    continue
                if l.y1 >= st.top - 1 and l.y0 <= st.bottom + 1:
                    d = 0
                else:
                    d = min(abs(l.y1 - st.top), abs(l.y0 - st.bottom))
                if d < bd:
                    best, bd = st, d
            if best is not None and bd < best.space * 7:
                best.stems.append(l)

    def _find_barlines(self, sy: System, lines, glyphs):
        st0 = sy.staves[0]
        xs = []
        for l in lines:
            if abs(l.x0 - l.x1) > 0.05 or l.width < 0.55:
                continue
            if l.x0 < st0.x0 + 2 or l.x0 > st0.x1 + 1:
                continue
            for st in sy.staves:
                if l.y0 <= st.top + 0.5 and l.y1 >= st.bottom - 0.5:
                    xs.append((l.x0, l.width))
                    break
        xs.sort()
        clusters: list[list[tuple[float, float]]] = []
        for x, w in xs:
            if clusters and x - clusters[-1][-1][0] < 5:
                clusters[-1].append((x, w))
            else:
                clusters.append([(x, w)])
        dots = [g for st in sy.staves for g in st.glyphs if g.font == SPECIAL_FONT and g.ch == "™"
                and abs(g.y - st.mid) < st.space * 0.8]
        for cl in clusters:
            b = Barline(x=max(x for x, _ in cl), thick=any(w > 1.5 for _, w in cl))
            left, right = min(x for x, _ in cl), max(x for x, _ in cl)
            if b.thick:
                if any(right < d.x < right + 6 for d in dots):
                    b.repeat_start = True
                if any(left - 7 < d.x < left for d in dots):
                    b.repeat_end = True
            sy.barlines.append(b)
        # a (repeat) barline in front of the first note of the system is not the end of a bar:
        # it only marks where the first bar starts
        first_event = min((g.x for st in sy.staves for g in st.glyphs
                           if g.font == MUSIC_FONT and (g.ch in NOTEHEADS or g.ch in RESTS)), default=st0.x1)
        sy.start_repeat = False
        while sy.barlines and sy.barlines[0].x < first_event:
            sy.start_repeat = sy.start_repeat or sy.barlines[0].repeat_start
            sy.barlines.pop(0)
        if not sy.barlines or sy.barlines[-1].x < st0.x1 - 3:
            sy.barlines.append(Barline(x=st0.x1))

    def _find_tempo(self, sy, sys_idx, glyphs, words):
        st0 = sy.staves[0]
        for g in glyphs:
            if g.font != TEXT_FONT or g.ch not in "qhe" or not (st0.top - 14 * st0.space < g.y < st0.top):
                continue
            unit = {"q": 1.0, "h": 2.0, "e": 0.5}[g.ch]
            if any(d.font == TEXT_FONT and d.ch == "." and 0 < d.x - g.x < 10 and abs(d.y - g.y) < 3 for d in glyphs):
                unit *= 1.5
            for w in words:
                if 0 < w.x0 - g.x < 30 and abs(w.y - g.y) < 3:
                    m = re.search(r"(\d+(?:[.,]\d+)?)", w.text + " " + " ".join(
                        o.text for o in words if o is not w and 0 < o.x0 - w.x1 < 8 and abs(o.y - w.y) < 2))
                    if m:
                        self.tempo_marks.append((sys_idx, g.x, g.y, float(m.group(1).replace(",", ".")) * unit))
                        break

    # ---- staff level: symbols -> chords --------------------------------------------

    def _staff_events(self, st: Staff, sy: System):
        sp = st.space
        heads = [g for g in st.glyphs if g.font == MUSIC_FONT and g.ch in NOTEHEADS and g.size > 13]
        unknown = Counter(g.ch for g in st.glyphs if g.font == MUSIC_FONT and g.ch not in NOTEHEADS and g.ch not in RESTS
                          and g.ch not in FLAGS and g.ch not in ACCIDENTALS and g.ch not in CLEFS and g.ch not in "0123456789.->,^_UuŸ<!'\"")
        for ch, n in unknown.items():
            self.warn(f"page {st.page + 1}: unknown music glyph {ch!r} x{n}")
        stems = st.stems
        chords: list[Chord] = []
        used = set()
        for stem in stems:
            attached = []
            for h in heads:
                if id(h) in used:
                    continue
                if (_up_side(stem, h, sp) or _down_side(stem, h, sp)) and stem.y0 - sp * 0.7 <= h.y <= stem.y1 + sp * 0.7:
                    attached.append(h)
            if not attached:
                continue
            ys = [h.y for h in attached]
            up = (stem.y1 - max(ys)) < (min(ys) - stem.y0)
            # the notehead on the stem end must be the right one for the direction
            if up and not any(_up_side(stem, h, sp) for h in attached):
                continue
            if not up and not any(_down_side(stem, h, sp) for h in attached):
                continue
            for h in attached:
                used.add(id(h))
            base = max(NOTEHEADS[h.ch] for h in attached)
            c = Chord(x=min(h.x0 for h in attached), heads=attached, stem=stem, up=up, base=base)
            tip = stem.y0 if up else stem.y1
            if base <= 1:
                n_beams = 0
                for bm in sy.beams:
                    if bm.x0 - 0.8 <= stem.x0 <= bm.x1 + 0.8:
                        r = bm.yrange(stem.x0)
                        if r and r[1] >= stem.y0 - 0.8 and r[0] <= stem.y1 + 0.8:
                            n_beams += 1
                flag = 0
                for g in st.glyphs:
                    if g.font == MUSIC_FONT and g.ch in FLAGS and abs(g.x - stem.x0) < 1.5 and abs(g.y - tip) < sp * 1.5:
                        flag = max(flag, FLAGS[g.ch])
                c.beams = max(n_beams, flag)
            chords.append(c)
        for h in heads:
            if id(h) not in used:
                if h.ch == "w":
                    chords.append(Chord(x=h.x0, heads=[h], stem=None, up=h.y < st.mid, base=4.0))
                else:
                    self.warn(f"page {st.page + 1} staff {st.name}: notehead without stem at x={h.x:.0f}")
        # rests
        for g in st.glyphs:
            if g.font == MUSIC_FONT and g.ch in RESTS and g.size > 13:
                base = RESTS[g.ch]
                # the whole-bar rest hangs inside the staff; the same glyph above it is an articulation
                margin = sp * 0.3 if base is None else sp * 2.5
                if not (st.top - margin <= g.y <= st.bottom + margin):
                    continue
                chords.append(Chord(x=g.x0, heads=[g], stem=None, up=g.y < st.mid, base=base or 0, is_rest=True,
                                    full_measure_rest=base is None, y=g.y))
        # augmentation dots
        # augmentation dots are "™" in the special font (the same glyph forms repeat signs)
        thick = [b.x for b in sy.barlines if b.thick]
        dots = [g for g in st.glyphs if g.font == SPECIAL_FONT and g.ch == "™"
                and not any(abs(g.x - bx) < 7 for bx in thick)]
        for c in chords:
            if c.full_measure_rest:
                continue
            right = max(h.x1 for h in c.heads)
            mine = sorted({round(d.x, 1) for d in dots for h in c.heads
                           if right - 0.5 < d.x < right + sp * 2.6 and abs(d.y - h.y) <= sp * (1.6 if c.is_rest else 0.62)})
            n, last = 0, right
            for dx in mine:
                if dx - last < sp * 1.3:
                    n += 1
                    last = dx
            c.dots = n
        self._tuplets(st, sy, chords)
        # ties (curves starting just right of a notehead)
        for c in chords:
            if c.is_rest:
                continue
            for cv in sy.curves:
                for h in c.heads:
                    if h.x0 - 1 < cv.x0 < h.x1 + sp * 1.5 and (cv.y0 - sp * 1.2 < h.y < cv.y1 + sp * 1.2) and cv.x1 - cv.x0 > sp:
                        c.tie_start = True
        c_sorted = sorted(chords, key=lambda c: c.x)
        return c_sorted

    def _tuplets(self, st, sy, chords):
        sp = st.space
        for w in sy.words:
            if "Italic" not in w.font or w.text not in TUPLET_FACTORS:
                continue
            if not (st.top - 5 * sp < w.y < st.bottom + 5 * sp) or w.x0 < st.x0 + 25:
                continue
            # measure numbers sit at the very start of a system; tuplet numbers are over a group
            nearest = min(chords, key=lambda c: abs(c.x - w.xc), default=None)
            if nearest is None or abs(nearest.x - w.xc) > 6 * sp:
                continue
            group = None
            for bm in sy.beams:
                if bm.x0 - 2 <= w.xc <= bm.x1 + 2:
                    r = bm.yrange(w.xc)
                    if r and abs((r[0] + r[1]) / 2 - w.y) < 3 * sp:
                        group = [c for c in chords if c.stem is not None and bm.x0 - 1 <= c.stem.x0 <= bm.x1 + 1]
                        break
            if not group:
                n = int(w.text)
                group = sorted(chords, key=lambda c: abs(c.x - w.xc))[:n]
            for c in group:
                c.tuplet = TUPLET_FACTORS[w.text]

    # ---- pitch --------------------------------------------------------------------

    def _staff_context(self, st: Staff):
        """Clef and key-signature changes along the staff, as sorted (x, kind, value) events."""
        sp = st.space
        ev = []
        heads = [g for g in st.glyphs if g.font == MUSIC_FONT and g.ch in NOTEHEADS]
        for g in st.glyphs:
            if g.font == MUSIC_FONT and g.ch in CLEFS and st.top - sp < g.y < st.bottom + sp:
                kind, bottom = CLEFS[g.ch]
                if kind == "C":
                    line = round((st.bottom - g.y) / sp)  # line the C clef is centred on
                    bottom = 28 - 2 * line  # C4 is diatonic 28
                octave = 0
                for o in st.glyphs:
                    if o.font == SPECIAL_FONT and o.ch == "‹" and abs(o.x - g.x) < 12 and o.y > g.y:
                        octave = -7
                ev.append((g.x, "clef", bottom + octave))
        clef_ev = sorted(e for e in ev)

        def clef_at(x):
            b = 30
            for cx, _, v in clef_ev:
                if cx <= x + 1:
                    b = v
            return b

        key_accs = []
        for g in st.glyphs:
            if g.font != MUSIC_FONT or g.ch not in ACCIDENTALS:
                continue
            attached = any(0 < h.x0 - g.x1 + 1 < sp * 3.2 and abs(h.y - g.y) < sp * 0.3 for h in heads)
            if not attached:
                key_accs.append(g)
        key_accs.sort(key=lambda g: g.x)
        key_events = []
        cur, last_x = {}, -1e9
        for g in key_accs:
            if g.x - last_x > sp * 2.2:
                if cur or last_x > -1e9:
                    key_events.append((last_x, dict(cur)))
                cur = {}
            step = round((st.bottom - g.y) / (sp / 2))
            letter = (clef_at(g.x) + step) % 7
            alt = ACCIDENTALS[g.ch]
            if alt == 0:
                cur.pop(letter, None)
            else:
                cur[letter] = alt
            last_x = g.x
        if key_accs:
            key_events.append((last_x, dict(cur)))
        return clef_at, key_events

    # ---- system -> measures --------------------------------------------------------

    def read(self) -> Score:
        self._load()
        if not self.systems:
            raise ValueError("No staves found - is this a vector PDF exported from a notation program?")
        part_order = self._part_order()
        voices_by_part: dict[str, list[list[Chord]]] = defaultdict(lambda: [[], []])
        measures: list[Measure] = []
        time_sig = (4, 4)
        key_fifths = 0
        tempos = []
        m_no = 0
        pending_repeat_times = {}
        chord_meas = {}  # id(chord) -> (measure idx, voice idx)

        for si, sy in enumerate(self.systems):
            bars = sy.barlines
            starts = [sy.staves[0].x0] + [b.x for b in bars[:-1]]
            ends = [b.x for b in bars]
            staff_data = {}
            sys_keys = []
            for st in sy.staves:
                events = self._staff_events(st, sy)
                clef_at, key_events = self._staff_context(st)
                self._pitch(st, events, clef_at, key_events, starts)
                staff_data[st.name] = (st, events)
                sys_keys = sys_keys if st is not sy.staves[0] else key_events
            self._lyrics(sy, staff_data)
            ts_changes = self._time_sigs(sy)
            for mi, (x0, x1) in enumerate(zip(starts, ends)):
                for tx, ts in ts_changes:
                    if tx < x1 - 2 and (mi == 0 or tx > x0 - 2):
                        time_sig = ts
                ts_len = time_sig[0] * 4 / time_sig[1]
                per_voice = {}
                for name, (st, events) in staff_data.items():
                    evs = [e for e in events if x0 - 1 < e.x < x1 - 1]
                    per_voice[name] = self._split_voices(st, evs)
                sums = [sum(e.dur for e in v) for vs in per_voice.values() for v in vs
                        if v and not any(e.full_measure_rest for e in v)]
                length = max(sums) if sums else ts_len
                if sums and length > ts_len + 1e-6:
                    self.warn(f"measure {m_no + 1}: notes add up to {length:g} quarters, time signature says {ts_len:g}")
                if sums and abs(length - ts_len) > 1e-6 and m_no != 0:
                    common = Counter(round(s, 4) for s in sums).most_common(1)[0][0]
                    if abs(common - ts_len) < 1e-6:
                        length = ts_len
                start = measures[-1].start + measures[-1].length if measures else 0.0
                meas = Measure(m_no + 1, start, length, time_sig)
                for kx, kmap in sys_keys:
                    if kx < x1 - 2:
                        key_fifths = sum(kmap.values())
                meas.fifths = key_fifths
                bl_start = bars[mi - 1] if mi > 0 else None
                if bl_start is not None and bl_start.repeat_start:
                    meas.repeat_start = True
                if mi == 0 and (sy.start_repeat or (si > 0 and self.systems[si - 1].barlines[-1].repeat_start)):
                    meas.repeat_start = True
                if bars[mi].repeat_end:
                    meas.repeat_end = True
                if mi == 0 and bars and len(bars) and sy.barlines and False:
                    pass
                measures.append(meas)
                for sx_i, sx, sy_, times in self.repeat_texts:
                    if sx_i == si and x0 - 5 < sx < x1:
                        pending_repeat_times[m_no] = times
                        meas.text = f"sing {times}x"
                for tsys, tx, ty, bpm in self.tempo_marks:
                    if tsys == si and (x0 - 30 < tx < x1 - 5) and not any(abs(t[0] - start) < 1e-6 for t in tempos):
                        tempos.append((start, bpm))
                # x position -> time, from the voices that fill the bar (the system is aligned vertically)
                xmap = []
                for vs in per_voice.values():
                    for v in vs:
                        if v and not any(e.full_measure_rest for e in v) and abs(sum(e.dur for e in v) - length) < 1e-6:
                            t = start
                            for e in v:
                                xmap.append((e.x, t))
                                t += e.dur
                for name, vs in per_voice.items():
                    for vi, v in enumerate(vs):
                        if not v:
                            continue
                        total = sum(e.dur for e in v if not e.full_measure_rest)
                        partial = not any(e.full_measure_rest for e in v) and total < length - 1e-6
                        if not any(e.full_measure_rest for e in v) and abs(total - length) > 1e-6 and not (partial and vi > 0 and xmap):
                            self.warn(f"measure {m_no + 1} {name} voice {vi + 1}: {total:g} of {length:g} quarters")
                        t = start
                        for e in v:
                            d = length if e.full_measure_rest else e.dur
                            if partial and xmap:
                                # a voice with hidden rests: take the time of the aligned note elsewhere
                                near = min(xmap, key=lambda p: abs(p[0] - e.x))
                                if abs(near[0] - e.x) < 3 and near[1] >= t - 1e-6:
                                    t = near[1]
                            # rests are kept (no pitches): they tell divisi splitting who is silent
                            voices_by_part[name][vi].append((t, d, e, m_no))
                            t += d
                m_no += 1
        # a pickup bar is bar 0 in printed bar numbers
        if len(measures) > 1 and measures[0].length < measures[0].time_sig[0] * 4 / measures[0].time_sig[1] - 1e-6:
            for m in measures:
                m.number -= 1
        # resolve repeat counts: a count found anywhere inside a repeated section applies to its end
        open_start = 0
        for i, m in enumerate(measures):
            if m.repeat_start:
                open_start = i
            if m.repeat_end:
                for j in range(open_start, i + 1):
                    if j in pending_repeat_times:
                        m.repeat_times = pending_repeat_times[j]
                open_start = i + 1
        if not tempos:
            self.warn("no tempo marking found, using 90 bpm")
            tempos = [(0.0, 90.0)]
        elif tempos[0][0] > 0:
            tempos.insert(0, (0.0, tempos[0][1]))

        parts = []
        self.display_names = dict(getattr(self, "display_names", {}))
        split_order = []
        for name in part_order:
            vlist = voices_by_part.get(name)
            if not vlist or not any(e.midis for raw in vlist for *_, e, _m in raw):
                continue
            halves = self._split_shared_staff(name, vlist)
            if halves:
                split_order += [(n, v) for n, v in halves]
            else:
                split_order.append((name, vlist))
        for name, vlist in split_order:
            part = Part(name=name)
            for raw in vlist:
                if raw:
                    part.voices.extend(self._build_voices(raw))
            if part.voices:
                # divisi lines without their own text sing the words of the main line
                main = {round(n.start, 4): n for n in part.voices[0].notes}
                for v in part.voices[1:]:
                    for n in v.notes:
                        src = main.get(round(n.start, 4))
                        if not n.lyrics and src is not None and src.lyrics:
                            n.lyrics = dict(src.lyrics)
                parts.append(part)
        return Score(self.title or "Score", parts, measures, sorted(tempos), self.warnings)

    def _part_order(self):
        # canonical names: abbreviations used on later systems map to full names on the first
        names_seen = []
        for sy in self.systems:
            for st in sy.staves:
                names_seen.append(st.name)
        full = []
        for sy in self.systems:
            for st in sy.staves:
                if "." not in st.name and st.name not in full:
                    full.append(st.name)
        canon = {}
        abbrs = [n for n in dict.fromkeys(names_seen) if "." in n]
        for f in full:
            for a in abbrs:
                if _abbrev_matches(a, f):
                    canon[f] = a
        for sy in self.systems:
            for st in sy.staves:
                st.name = canon.get(st.name, st.name)
        display = {v: k for k, v in canon.items()}
        self.display_names = display
        order: list[str] = []
        for sy in self.systems:
            names = [st.name for st in sy.staves]
            for i, n in enumerate(names):
                if n in order:
                    continue
                after = [m for m in names[i + 1:] if m in order]
                if after:
                    order.insert(order.index(after[0]), n)
                else:
                    before = [m for m in names[:i] if m in order]
                    order.insert(order.index(before[-1]) + 1 if before else len(order), n)
        return order

    def _time_sigs(self, sy):
        res = []
        st = sy.staves[0]
        digs = [g for g in st.glyphs if g.font == MUSIC_FONT and g.ch.isdigit() and g.size > 13]
        cols = defaultdict(list)
        for g in digs:
            cols[round(g.x0 / 3)].append(g)
        for col in sorted(cols.values(), key=lambda c: c[0].x0):
            top = "".join(g.ch for g in sorted(col, key=lambda g: g.x) if g.y < st.mid)
            bot = "".join(g.ch for g in sorted(col, key=lambda g: g.x) if g.y > st.mid)
            if top and bot:
                res.append((col[0].x0, (int(top), int(bot))))
        for g in st.glyphs:
            if g.font == MUSIC_FONT and g.ch == "c" and abs(g.y - st.mid) < st.space:
                res.append((g.x0, (4, 4)))
            if g.font == MUSIC_FONT and g.ch == "C" and abs(g.y - st.mid) < st.space:
                res.append((g.x0, (2, 2)))
        return sorted(res)

    def _pitch(self, st, events, clef_at, key_events, starts):
        sp = st.space
        accs = [g for g in st.glyphs if g.font == MUSIC_FONT and g.ch in ACCIDENTALS]
        bar_xs = starts[1:]
        measure_alts: dict[tuple[int, int], int] = {}
        cur_bar = 0
        for c in events:
            if c.is_rest:
                continue
            while cur_bar < len(bar_xs) and c.x > bar_xs[cur_bar]:
                cur_bar += 1
                measure_alts = {}
            key = {}
            for kx, kmap in key_events:
                if kx < c.x:
                    key = kmap
            c.midis = []
            for h in sorted(c.heads, key=lambda h: -h.y):
                step = round((st.bottom - h.y) / (sp / 2))
                dia = clef_at(h.x) + step
                octave, letter = divmod(dia, 7)
                acc = [a for a in accs if 0 < h.x0 - a.x1 + 1 < sp * 3.2 and abs(h.y - a.y) < sp * 0.3]
                if acc:
                    alt = ACCIDENTALS[max(acc, key=lambda a: a.x).ch]
                    measure_alts[(octave, letter)] = alt
                elif (octave, letter) in measure_alts:
                    alt = measure_alts[(octave, letter)]
                else:
                    alt = key.get(letter, 0)
                c.midis.append(12 * (octave + 1) + STEP_SEMITONES[letter] + alt)

    def _split_voices(self, st, evs: list[Chord]) -> list[list[Chord]]:
        notes = [e for e in evs if not e.is_rest]
        if not evs:
            return [[], []]
        two = False
        for a in notes:
            for b in notes:
                if a is not b and a.up and not b.up and abs(a.x - b.x) < st.space * 1.6:
                    two = True
        if not two:
            total = sum(e.dur for e in evs if not e.full_measure_rest)
            ups = [e for e in notes if e.up]
            downs = [e for e in notes if not e.up]
            rests_hi = [e for e in evs if e.is_rest and e.y < st.mid - st.space * 1.2]
            rests_lo = [e for e in evs if e.is_rest and e.y > st.mid + st.space * 1.2]
            if ups and downs and (rests_hi or rests_lo) and total > max(sum(e.dur for e in ups), sum(e.dur for e in downs)) * 1.9:
                two = True
        if not two:
            return [evs, []]
        v1 = [e for e in evs if (not e.is_rest and e.up) or (e.is_rest and e.y <= st.mid)]
        v2 = [e for e in evs if (not e.is_rest and not e.up) or (e.is_rest and e.y > st.mid)]
        return [v1, v2]

    # ---- lyrics ---------------------------------------------------------------------

    def _lyrics(self, sy: System, staff_data):
        staves = sy.staves
        sp = staves[0].space
        texts = [w for w in sy.words if "Opus" not in w.font and "Italic" not in w.font and "Bold" not in w.font]
        # the lyric font size is the most common size among words right below staves
        below = [w for w in texts for st in staves if st.bottom + sp < w.y < st.bottom + 7 * sp and w.x0 > st.x0]
        if not below:
            return
        lyr_size = Counter(round(w.size, 1) for w in below).most_common(1)[0][0]
        lyr = []
        for w in texts:
            if abs(w.size - lyr_size) < 0.15 and w.x0 > staves[0].x0 - 5:
                # two voices with the same text are drawn on top of each other: keep one
                if not any(o.text == w.text and abs(o.x0 - w.x0) < 1 and abs(o.y - w.y) < 1 for o in lyr):
                    lyr.append(w)
        lines_by_staff = defaultdict(list)
        for w in lyr:
            if w.text in ("-", "–"):
                continue
            above_st = [st for st in staves if st.bottom < w.y]
            below_st = [st for st in staves if st.top > w.y]
            cand = []
            if above_st:
                cand.append((above_st[-1], "below"))
            if below_st:
                cand.append((below_st[0], "above"))
            best = None
            for st, where in cand:
                events = staff_data[st.name][1]
                notes = [e for e in events if not e.is_rest]
                if not notes:
                    continue
                n = min(notes, key=lambda e: abs(self._head_xc(e) - self._lyric_anchor(w, e)))
                dx = abs(self._head_xc(n) - self._lyric_anchor(w, n))
                gap = (w.y - st.bottom) if where == "below" else (st.top - w.y)
                if where == "above":
                    # text above a staff only belongs to its upper voice, which happens on
                    # staves shared by two parts ("B./Bar.") or with two voices
                    two_voices = n.up and any(not e.up for e in notes if abs(e.x - n.x) < sp * 1.6 and e is not n)
                    if not ("/" in st.name or two_voices):
                        continue
                    score = dx + 0.8 * gap + 2
                else:
                    score = dx + 0.8 * min(gap, 5 * sp) + max(0, gap - 6 * sp) * 0.5
                if dx < 6 * sp and (best is None or score < best[0]):
                    best = (score, st, where)
            if best:
                lines_by_staff[(best[1].name, best[2])].append(w)
        hyphens = [w for w in lyr if w.text in ("-", "–")]
        for (name, where), words in lines_by_staff.items():
            st, events = staff_data[name]
            # verse lines by baseline
            ys = sorted({round(w.y, 0) for w in words})
            rows = []
            for y in ys:
                if rows and y - rows[-1][-1] < sp * 1.2:
                    rows[-1].append(y)
                else:
                    rows.append([y])
            for w in words:
                verse = next(i for i, r in enumerate(rows) if round(w.y, 0) in r) + 1
                notes = [e for e in events if not e.is_rest]
                target = min(notes, key=lambda e: abs(self._head_xc(e) - self._lyric_anchor(w, e)))
                if where == "above":
                    # prefer the upper (stem-up) voice when two voices share this spot
                    ups = [e for e in notes if e.up and abs(e.x - target.x) < sp * 1.6]
                    if ups:
                        target = ups[0]
                text = w.text
                # the hyphen must sit before the next word on this line, or it belongs to that word
                nxt = min((o.x0 for o in lyr if abs(o.y - w.y) < 2 and o.x0 > w.x1 and o.text not in ("-", "–")),
                          default=w.x1 + 40)
                hyph = any(0 < h.x0 - w.x1 < 40 and h.x0 < nxt and abs(h.y - w.y) < 2 for h in hyphens)
                if text.endswith("-"):
                    text, hyph = text[:-1], True
                store = target.lyrics_above if where == "above" else target.lyrics
                if verse in store:
                    # two words on one note (e.g. "an' sing"): join them
                    store[verse] = Syllable(store[verse].text + " " + text, hyph)
                else:
                    store[verse] = Syllable(text, hyph)
        for _, events in staff_data.values():
            for e in events:
                if e.lyrics_above and not e.lyrics:
                    e.lyrics = e.lyrics_above

    @staticmethod
    def _head_xc(e: Chord) -> float:
        return sum((h.x0 + h.x1) / 2 for h in e.heads) / len(e.heads)

    @staticmethod
    def _lyric_anchor(w: Word, e: Chord) -> float:
        # Sibelius centres syllables under the notehead, but left-aligns a syllable that
        # starts a melisma; take whichever reading fits this note better
        head_x0 = min(h.x0 for h in e.heads)
        return w.xc if abs(w.xc - PdfScoreReader._head_xc(e)) <= abs(w.x0 - head_x0) else w.x0 + (PdfScoreReader._head_xc(e) - head_x0)

    # ---- voices -------------------------------------------------------------------

    VOICE_RANK = [("sopran", 6), ("mezzo", 5), ("alt", 4), ("tenor", 3), ("bariton", 2), ("bass", 1),
                  ("s", 6), ("m", 5), ("a", 4), ("t", 3), ("bar", 2), ("b", 1)]

    def _rank(self, label: str) -> int:
        w = re.sub(r"[^a-zäöü]", "", label.lower())
        for key, r in self.VOICE_RANK:
            if w.startswith(key) and (len(key) > 1 or len(w) <= 2):
                return r
        return 0

    def _split_shared_staff(self, name: str, vlist):
        """A staff shared by two parts ("B./Bar.", "S1/S2") becomes two parts: the higher
        voice type takes the upper notes, the other the lower ones. Where the staff shows a
        single line, both parts sing it; a rest written in one voice silences only that part."""
        full = self.display_names.get(name, name)
        abbr, names = name.split("/"), full.split("/")
        if len(abbr) != 2:
            return None
        if len(names) != 2:
            names = abbr
        ranks = [self._rank(n) for n in names]
        upper_i = 0 if ranks[0] >= ranks[1] else 1
        v1 = sorted(vlist[0], key=lambda r: r[0])
        v2 = sorted(vlist[1], key=lambda r: r[0])

        def covered(raw, t, d):
            return any(s < t + d - 1e-6 and t < s + dd - 1e-6 for s, dd, *_ in raw)

        def pick(raw, fill, top):
            out = []
            for s, d, e, m in raw:
                if e.midis:
                    c = Chord(e.x, e.heads, e.stem, e.up, e.base, midis=[max(e.midis) if top else min(e.midis)],
                              lyrics=e.lyrics, tie_start=e.tie_start)
                    out.append((s, d, c, m))
            for s, d, e, m in fill:
                if e.midis and not covered(raw, s, d):
                    c = Chord(e.x, e.heads, e.stem, e.up, e.base, midis=[max(e.midis) if top else min(e.midis)],
                              lyrics=e.lyrics, tie_start=e.tie_start)
                    out.append((s, d, c, m))
            return sorted(out, key=lambda r: r[0])

        upper = pick(v1, v2, True)
        lower = pick(v2, v1, False)
        # where one part has no text of its own, it sings the words of the other at that moment
        for a, b in ((upper, lower), (lower, upper)):
            words = {round(s, 4): e.lyrics for s, d, e, m in a if e.lyrics}
            for s, d, e, m in b:
                if not e.lyrics and round(s, 4) in words:
                    e.lyrics = words[round(s, 4)]
        lower_i = 1 - upper_i
        res = []
        for i, raw in ((upper_i, upper), (lower_i, lower)):
            key = abbr[i].strip()
            self.display_names[key] = names[i].strip()
            res.append((key, [raw, []]))
        return res

    def _build_voices(self, raw) -> list[Voice]:
        """raw: list of (start, dur, chord, measure). The top note of every chord goes to the
        first voice; lower chord notes (divisi) go to extra voices that sing the same text."""
        lines: list[list[Note]] = []
        raw.sort(key=lambda r: r[0])
        for start, dur, c, m in raw:
            for k, midi in enumerate(sorted(c.midis, reverse=True)):
                while len(lines) <= k:
                    lines.append([])
                lines[k].append(Note(start, dur, midi, dict(c.lyrics), m, c.tie_start,
                                     any(h.ch in X_HEADS for h in c.heads)))
        return [Voice(self._merge_ties(notes)) for notes in lines]

    @staticmethod
    def _merge_ties(notes: list[Note]) -> list[Note]:
        """A tie curve into the directly following note of the same pitch without its own
        syllable joins the two notes."""
        merged: list[Note] = []
        for n in notes:
            prev = merged[-1] if merged else None
            if (prev is not None and prev.tie_start and prev.midi == n.midi
                    and abs(prev.start + prev.dur - n.start) < 1e-6 and not n.lyrics):
                prev.dur += n.dur
                prev.tie_start = n.tie_start
                continue
            merged.append(n)
        return merged


def read_pdf(path: str) -> Score:
    r = PdfScoreReader(path)
    score = r.read()
    score.display_names = r.display_names
    return score
