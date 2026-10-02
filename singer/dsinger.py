"""Sing a voice line with a DiffSinger voicebank (neural singing synthesis).

Same input as synth.py (SungNotes with syllables in MBROLA SAMPA), but the sound comes
from a model trained on real singing: the bank's acoustic model turns phonemes, their
durations and our pitch curve into a mel spectrogram, and its vocoder turns that into
audio. Timing and pitch stay ours; the bank's duration model only proposes how long each
consonant should be. Phrases (separated by rests) are rendered one by one.

Voicebanks live in vendor/voicebanks/<name>/ (OpenUtau DiffSinger format; see README).
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .perform import SungNote
from .phonemes import VOWELS, SungSyllable
from .synth import SR, _dynamics, _pitch_curve, hz

ROOT = Path(__file__).resolve().parent.parent
BANK_DIR = ROOT / "vendor" / "voicebanks"

# MBROLA SAMPA (phonemes.py) -> the banks' English ARPAbet
TO_ARPA = {
    "i": "iy", "I": "ih", "E": "eh", "{": "ae", "A": "aa", "V": "ah", "O": "ao", "U": "uh",
    "u": "uw", "@": "ax", "r=": "er", "EI": "ey", "AI": "ay", "OI": "oy", "@U": "ow", "aU": "aw",
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "g": "g", "4": "dx", "f": "f", "v": "v",
    "s": "s", "z": "z", "S": "sh", "Z": "zh", "T": "th", "D": "dh", "h": "hh", "m": "m", "n": "n",
    "N": "ng", "l": "l", "r": "r", "w": "w", "j": "y", "tS": "ch", "dZ": "jh",
}

# Sung consonant lengths (s): the duration model's proposal is kept inside these bounds.
# Voiced stops held long turn into glides (b -> w), so they stay short; nasals and
# fricatives carry the words and may run longer.
CLIP = {
    **{c: (0.03, 0.065) for c in ("b", "d", "g")}, "dx": (0.02, 0.05),
    **{c: (0.06, 0.15) for c in ("m", "n")}, "ng": (0.07, 0.2),
    **{c: (0.04, 0.13) for c in ("p", "t", "k")}, "ch": (0.08, 0.17), "jh": (0.08, 0.15),
    **{c: (0.055, 0.18) for c in ("f", "th", "s", "sh", "hh")},
    **{c: (0.05, 0.15) for c in ("v", "z", "zh", "dh")},
    **{c: (0.04, 0.13) for c in ("l", "r", "w", "y")},
}
# Between voiced sounds the model glides through a stop ("da da" -> "da za", "dam bam" ->
# "dam dam"): a short silence before it makes the closure audible (seconds).
CLOSURE = {**{c: 0.055 for c in ("b", "d", "g")}, "jh": 0.035, **{c: 0.02 for c in ("p", "t", "k", "ch")}}
CLOSURE_TOKEN = "SP"
# onset consonant -> the token actually sung: the banks' English stops are soft (Russian
# accent); their German ones keep a real closure and release
ONSET_SUBS = {c: f"de/{c}" for c in ("b", "d", "g", "p", "t", "k")}
VELOCITY = 1.0  # the model's consonant speed: above 1 is crisper
# a closing diphthong held on a long note freezes on its first vowel ("I" -> "ah"):
# hold the first vowel and sing the glide as its own short segment at the end
DIPH_SPLIT = {"ay": ("aa", "y"), "ey": ("eh", "y"), "oy": ("ao", "y"), "aw": ("aa", "w"), "ow": ("ao", "w")}

HEAD, TAIL = 0.12, 0.15  # silence rendered around each phrase (s)


@dataclass(frozen=True)
class Singer:
    bank: str
    speakers: tuple[tuple[str, float], ...]  # voice-mode embeds and their weights
    gender: float = 0.0  # formant shift: -1..1 = the bank's key-shift range, negative is darker
    tension: float = 0.0


# Which bank sings which part. Tenor, baritone and bass are always sung by men: Keiro
# Revenant (LUNAI's male singer) and Liam Thorne; soprano and alto by Katyusha.
MALE_BANKS = ("keiro_revenant", "liam_thorne")
FEMALE_BANKS = ("katyusha",)
SINGERS = {
    "soprano": Singer("katyusha", (("delicate", 0.6), ("standard", 0.4))),
    "alto": Singer("katyusha", (("lullaby", 0.7), ("standard", 0.3)), gender=-0.25),
    "tenor": Singer("keiro_revenant", (("standard", 0.7), ("mellow", 0.3))),
    "baritone": Singer("liam_thorne", (("standard", 0.6), ("gentle", 0.4))),
    "bass": Singer("liam_thorne", (("deep", 0.8), ("standard", 0.2)), gender=-0.15),
}
MALE_PARTS = ("tenor", "baritone", "bass")


def available_banks() -> list[str]:
    return sorted(p.name for p in BANK_DIR.glob("*") if (p / "configs" / "dsconfig.yaml").exists())


def singer_for(kind: str) -> Singer:
    s = SINGERS[kind]
    banks = available_banks()
    if s.bank in banks:
        return s
    # stand-in from the same sex, with the plain voice mode
    same = [b for b in (MALE_BANKS if kind in MALE_PARTS else FEMALE_BANKS) if b in banks]
    if not same:
        raise RuntimeError(f"No {'male' if kind in MALE_PARTS else 'female'} DiffSinger voicebank for the "
                           f"{kind} in {BANK_DIR} (run ./setup.sh)")
    return Singer(same[0], (("standard", 1.0),), gender=s.gender)


class Bank:
    def __init__(self, name: str):
        import onnxruntime as ort

        self.dir = BANK_DIR / name
        cfg_dir = self.dir / "configs"
        self.cfg = yaml.safe_load(open(cfg_dir / "dsconfig.yaml"))
        voc = yaml.safe_load(open(cfg_dir / self.cfg["vocoder"] / "vocoder.yaml"))
        self.sr = self.cfg["sample_rate"]
        assert self.sr == SR, f"{name}: sample rate {self.sr} != {SR}"
        self.frame = self.cfg["hop_size"] / self.sr
        self.tokens = json.load(open((cfg_dir / self.cfg["phonemes"]).resolve()))
        self.langs = json.load(open((cfg_dir / self.cfg["languages"]).resolve()))
        self.lang = self.langs.get("en", 0)
        self.embeds = {p.stem: np.frombuffer(p.read_bytes(), dtype="<f4") for p in (cfg_dir / "embeds").glob("*.emb")}
        self.max_depth = float(self.cfg.get("max_depth", 1.0))
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        load = lambda p: ort.InferenceSession(str(p), opts, providers=["CPUExecutionProvider"])
        self.acoustic = load((cfg_dir / self.cfg["acoustic"]).resolve())
        self.vocoder = load((cfg_dir / self.cfg["vocoder"] / voc["model"]).resolve())
        dur_cfg = cfg_dir / "dsdur" / "dsconfig.yaml"
        self.dur = self.ling = None
        if dur_cfg.exists():
            d = yaml.safe_load(open(dur_cfg))
            self.ling = load((dur_cfg.parent / d["linguistic"]).resolve())
            self.dur = load((dur_cfg.parent / d["dur"]).resolve())
        self.acoustic_inputs = {i.name for i in self.acoustic.get_inputs()}

    def token(self, ph: str) -> int:
        # "ru/d": borrow a phoneme from another of the bank's languages
        for key in (ph, f"en/{ph}") if "/" in ph else (f"en/{ph}", ph):
            if key in self.tokens:
                return self.tokens[key]
        raise KeyError(f"phoneme {ph!r} not in voicebank {self.dir.name}")

    def lang_of(self, ph: str) -> int:
        if "/" in ph:
            return self.langs.get(ph.split("/")[0], 0)
        return self.lang if f"en/{ph}" in self.tokens else 0

    def embed(self, speakers) -> np.ndarray:
        total = sum(w for _, w in speakers)
        emb = sum(self.embeds[n] * (w / total) for n, w in speakers if n in self.embeds)
        return emb if isinstance(emb, np.ndarray) else self.embeds["standard"]

    def predict(self, words: list[tuple[list[str], int, int]], emb: np.ndarray) -> list[list[float]]:
        """words: (phonemes, midi, frames), each starting at a vowel onset as DiffSinger
        expects (a note's leading consonants belong to the word before). -> seconds each."""
        if self.dur is None:
            return [[0.07] * len(p) for p, _, _ in words]
        phs = [p for w in words for p in w[0]]
        enc, mask = self.ling.run(None, {
            "tokens": np.array([[self.token(p) for p in phs]], np.int64),
            "languages": np.array([[self.lang_of(p) for p in phs]], np.int64),
            "word_div": np.array([[len(w[0]) for w in words]], np.int64),
            "word_dur": np.array([[max(1, w[2]) for w in words]], np.int64),
        })
        pred = self.dur.run(None, {
            "encoder_out": enc, "x_masks": mask,
            "ph_midi": np.array([[m for p, m, _ in words for _ in p]], np.int64),
            "spk_embed": np.tile(emb[None, None, :], (1, len(phs), 1)).astype(np.float32),
        })[0][0]
        out, k = [], 0
        for p, _, _ in words:
            out.append([float(v) * self.frame for v in pred[k:k + len(p)]])
            k += len(p)
        return out

    def render(self, segs: list[tuple[str, float, float]], f0: np.ndarray, emb: np.ndarray,
               gender: float, tension: float, steps: int) -> np.ndarray:
        """segs: (phoneme, start, end) in seconds from the phrase start, contiguous;
        f0: Hz per frame. -> audio at SR."""
        tokens, langs, durs = [], [], []
        pos = 0
        for ph, _, e in segs:
            end = int(round(e / self.frame))
            if end - pos < 1:
                continue
            tokens.append(self.token(ph))
            langs.append(self.lang_of(ph))
            durs.append(end - pos)
            pos = end
        n = pos
        f0 = np.pad(f0[:n], (0, max(0, n - len(f0))), mode="edge").astype(np.float32)[None]
        const = lambda v: np.full((1, n), v, np.float32)
        feed = {
            "tokens": np.array([tokens], np.int64), "languages": np.array([langs], np.int64),
            "durations": np.array([durs], np.int64), "f0": f0,
            "tension": const(tension), "gender": const(gender), "velocity": const(VELOCITY),
            "spk_embed": np.tile(emb[None, None, :], (1, n, 1)).astype(np.float32),
            "depth": np.array(min(0.6, self.max_depth), np.float32), "steps": np.array(steps, np.int64),
        }
        mel = self.acoustic.run(None, {k: v for k, v in feed.items() if k in self.acoustic_inputs})[0]
        return self.vocoder.run(None, {"mel": mel, "f0": f0})[0].squeeze().astype(np.float32)


@functools.lru_cache(maxsize=None)
def bank(name: str) -> Bank:
    return Bank(name)


def _arpa(sampa: list[str]) -> list[str]:
    return [TO_ARPA[p] for p in sampa if p in TO_ARPA]


def _groups(notes: list[SungNote]):
    """A syllable note plus the melisma notes that follow it, as in synth.render_pho."""
    groups: list[tuple[SungSyllable, list[SungNote]]] = []
    for n in notes:
        if n.syllable is None and groups and abs(groups[-1][1][-1].t + groups[-1][1][-1].dur - n.t) < 0.02:
            groups[-1][1].append(n)
        else:
            groups.append((n.syllable or SungSyllable(nucleus="A"), [n]))
    return groups


def _units(groups):
    """One sung unit per vowel: several words printed under one note share it evenly."""
    units = []
    for syl, gnotes in groups:
        subs = [syl] + list(syl.extra)
        start, end = gnotes[0].t, gnotes[-1].t + gnotes[-1].dur
        span = (end - start) / len(subs)
        for k, s in enumerate(subs):
            a, b = start + k * span, start + (k + 1) * span
            ns = [n for n in gnotes if n.t < b - 1e-6 and n.t + n.dur > a + 1e-6] or gnotes[-1:]
            vowel = TO_ARPA.get(s.nucleus, "ah")
            if s.nucleus not in VOWELS:  # hummed syllable ("mm"): the consonant carries the note
                vowel = TO_ARPA.get(s.nucleus, "m")
            units.append({"onset": _arpa(s.onset), "vowel": vowel, "coda": _arpa(s.coda),
                          "t": a, "end": b, "notes": ns, "midi": ns[0].midi})
    return units


def _phrases(units, gap: float = 0.3):
    out, cur = [], []
    for u in units:
        if cur and u["t"] - cur[-1]["end"] > gap:
            out.append(cur)
            cur = []
        cur.append(u)
    return out + ([cur] if cur else [])


def _plan(b: Bank, units, emb: np.ndarray):
    """Consonant durations for one phrase: the duration model's proposal within CLIP,
    squeezed so every vowel keeps at least half its note."""
    words = [(["SP"] + units[0]["onset"], units[0]["midi"], int(HEAD / b.frame) + 4)]
    for i, u in enumerate(units):
        nxt = units[i + 1]["onset"] if i + 1 < len(units) else []
        words.append(([u["vowel"]] + u["coda"] + nxt, u["midi"], int((u["end"] - u["t"]) / b.frame)))
    try:
        pred = b.predict(words, emb)
    except Exception as e:  # unknown phoneme etc.: fall back to fixed lengths
        print(f"    duration model failed ({e}); using fixed consonant lengths")
        pred = [[0.07] * len(w[0]) for w in words]
    clip = lambda ph, v: float(np.clip(v, *CLIP.get(ph, (0.04, 0.15))))
    for i, u in enumerate(units):
        u["on_d"] = [clip(p, v) for p, v in zip(u["onset"], pred[i][-len(u["onset"]):] if u["onset"] else [])]
        u["co_d"] = [clip(p, v) for p, v in zip(u["coda"], pred[i + 1][1:1 + len(u["coda"])])]
        # a hummed final m/n on a long note ("dam", "ding") needs weight to be heard
        if u["coda"] and u["coda"][-1] in ("m", "n", "ng"):
            u["co_d"][-1] = max(u["co_d"][-1], min(0.18, (u["end"] - u["t"]) * 0.25))
    for i, u in enumerate(units):
        length = u["end"] - u["t"]
        # the next onset is sung before the next beat, i.e. inside this note
        nxt_on = sum(units[i + 1]["on_d"]) if i + 1 < len(units) and units[i + 1]["t"] - u["end"] < 0.02 else 0.0
        # in a melisma the consonants may only take time from the last note
        last = u["notes"][-1]
        budget = min(length * 0.5, (u["end"] - max(u["t"], last.t)) * 0.6)
        used = sum(u["co_d"]) + nxt_on
        if used > budget:
            k = budget / used
            u["co_d"] = [d * k for d in u["co_d"]]
            if i + 1 < len(units) and nxt_on:
                units[i + 1]["on_d"] = [d * k for d in units[i + 1]["on_d"]]


def _segments(units, t0: float):
    """Phoneme segments (phoneme, start, end) relative to t0: onsets before the beat, the
    vowel on it, codas at the end of the note, and a breath before the phrase."""
    segs: list[list] = []

    def push(ph, s, e):
        if e - s < 0.005:
            return
        if segs and segs[-1][2] > s:  # overlaps: the earlier segment yields
            segs[-1][2] = s
            if segs[-1][2] - segs[-1][1] < 0.005:
                segs.pop()
        if segs and s > segs[-1][2] + 1e-6:
            segs.append(["SP", segs[-1][2], s])
        segs.append([ph, s, e])

    first_on = sum(units[0]["on_d"])
    start = units[0]["t"] - first_on
    push("SP", t0, start - 0.2 if start - t0 > 0.3 else start)
    if start - t0 > 0.3:
        push("AP", start - 0.2, start)  # an audible breath before the phrase
    for i, u in enumerate(units):
        nxt = units[i + 1] if i + 1 < len(units) else None
        t = u["t"] - sum(u["on_d"])
        for j, (p, d) in enumerate(zip(u["onset"], u["on_d"])):
            if j == 0 and p in CLOSURE and segs and segs[-1][0] not in ("SP", "AP"):
                w = min(CLOSURE[p], (segs[-1][2] - segs[-1][1]) * 0.4)
                push(CLOSURE_TOKEN, t - w, t)
            push(ONSET_SUBS.get(p, p), t, t + d)
            t += d
        if nxt is not None and nxt["t"] - u["end"] < 0.02:
            v_end = nxt["t"] - sum(nxt["on_d"]) - sum(u["co_d"])
        else:
            v_end = u["end"] - sum(u["co_d"]) - min(0.04, (u["end"] - u["t"]) * 0.1)  # breathe before rests
        v_end = max(v_end, u["t"] + 0.03)
        u["v"] = (u["t"], v_end)
        if u["vowel"] in DIPH_SPLIT and v_end - u["t"] > 0.5:
            nuc, glide = DIPH_SPLIT[u["vowel"]]
            g = min(0.11 if u["coda"] else 0.16, max(0.08, (v_end - u["t"]) * 0.15))
            push(nuc, u["t"], v_end - g)
            push(glide, v_end - g, v_end)
        else:
            push(u["vowel"], u["t"], v_end)
        t = v_end
        for p, d in zip(u["coda"], u["co_d"]):
            push(p, t, t + d)
            t += d
    return [(p, s - t0, e - t0) for p, s, e in segs]


def _f0(units, t0: float, n: int, frame: float, detune: float, seed: int) -> np.ndarray:
    """Frame-wise pitch: our pitch curves on the vowels, glides through the consonants."""
    pts = []
    for k, u in enumerate(units):
        notes = u["notes"]
        if detune:
            notes = [SungNote(x.t, x.dur, x.midi + detune / 100, x.syllable, x.spoken) for x in notes]
        a, b = u["v"]
        # scooping into every note sounds sluggish: only phrase openings do
        pts += _pitch_curve(notes, a, b, vibrato=True, seed=seed + k, scoop=k == 0)
        pts.append((b + 0.03, hz(notes[-1].midi)))  # hold into the coda
    pts.sort()
    ts, fs = zip(*pts)
    times = t0 + np.arange(n) * frame
    return np.interp(times, ts, fs)


def sing(notes: list[SungNote], total: float, kind: str, singers: int = 1, seed: int = 0,
         steps: int = 20) -> np.ndarray:
    """Render a voice line with the singer for this part kind (soprano/alto/tenor/baritone/bass).
    Extra singers use another blend of the bank's voice modes, a little detuned and late."""
    s = singer_for(kind)
    b = bank(s.bank)
    n_out = int((total + 0.5) * SR)
    out = np.zeros(n_out, np.float32)
    rng = np.random.default_rng(seed + 99)
    modes = [m for m in b.embeds if m not in ("growl", "whisper", "scream")]
    for k in range(singers):
        speakers = s.speakers if k == 0 else ((modes[k % len(modes)], 0.5),) + s.speakers
        emb = b.embed(speakers).astype(np.float32)
        detune = 0.0 if k == 0 else rng.uniform(5, 10) * (1 if k % 2 else -1)
        delay = 0 if k == 0 else int(rng.uniform(0.015, 0.03) * SR)
        units = _units(_groups(notes))
        for pi, phrase in enumerate(_phrases(units)):
            _plan(b, phrase, emb)
            t0 = phrase[0]["t"] - sum(phrase[0]["on_d"]) - HEAD - 0.2
            segs = _segments(phrase, t0)
            end = phrase[-1]["end"] - t0 + TAIL
            segs.append(("SP", segs[-1][2], end))
            n = int(round(end / b.frame))
            f0 = _f0(phrase, t0, n, b.frame, detune, seed * 1000 + k * 100 + pi)
            y = b.render(segs, f0, emb, s.gender, s.tension, steps)
            # fade the padding so phrases join cleanly
            f = int(0.01 * SR)
            y[:f] *= np.linspace(0, 1, f, dtype=np.float32)
            y[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
            i = int(round(t0 * SR)) + delay
            src = y[max(0, -i):]
            i = max(0, i)
            m = min(len(src), n_out - i)
            if m > 0:
                out[i:i + m] += src[:m] * (1.0 if k == 0 else 0.7)
    return _dynamics(out / np.sqrt(1 + 0.49 * (singers - 1)), notes)
