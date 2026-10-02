"""Lyrics -> MBROLA (us1/us2/us3 SAMPA) phonemes, split into sung syllables.

Words are phonemised whole with espeak-ng ("ri - ver" as "river") and then divided
over the syllables the score gives, so each note gets onset consonants, one vowel
nucleus and coda consonants.
"""
from __future__ import annotations

import functools
import re
import subprocess
import unicodedata
from dataclasses import dataclass, field

# scat / vocal-percussion syllables espeak would spell out or read as English words
SCAT = {
    "la": "l A", "da": "d A", "dah": "d A", "ja": "j A", "ba": "b A", "na": "n A", "ta": "t A",
    "dam": "d A m", "bam": "b A m", "dum": "d U m", "bum": "b U m", "dm": "d I m", "dn": "d I n",
    "doo": "d u", "du": "d u", "dun": "d V n", "doon": "d u n", "dap": "d A p", "bap": "b A p",
    "ah": "A", "aah": "A", "oh": "@U", "ooh": "u", "oo": "u", "uh": "V", "mm": "m", "hm": "h m",
    "hmm": "h m", "m": "m", "n": "n", "hah": "h A", "ha": "h A", "ho": "h @U", "dee": "d i",
    "di": "d i", "ding": "d I N", "dong": "d A N", "dang": "d { N", "ding-dong": "d I N d A N",
    "shoo": "S u", "wah": "w A", "wa": "w A", "bop": "b A p", "ba-da": "b A d A", "pa": "p A",
    "dom": "d O m", "bom": "b O m", "bim": "b I m", "dim": "d I m", "tu": "t u", "tm": "t m",
    "o": "@U", "a": "A", "ahh": "A", "nah": "n A", "yeah": "j E", "oh-oh": "@U @U",
}

VOWELS = {"i", "I", "E", "{", "A", "V", "O", "U", "u", "@", "r=", "EI", "AI", "OI", "@U", "aU"}
# consonants that can carry a hum when there is no vowel ("dm", "hmm")
SONORANTS = {"m", "n", "N", "l"}

# longest match first
IPA_MAP = [
    ("eɪ", "EI"), ("aɪ", "AI"), ("ɔɪ", "OI"), ("oʊ", "@U"), ("əʊ", "@U"), ("aʊ", "aU"),
    ("tʃ", "tS"), ("dʒ", "dZ"), ("ɑːɹ", "A r"), ("ɔːɹ", "O r"), ("ɛɹ", "E r"), ("ɪɹ", "I r"),
    ("ʊɹ", "U r"), ("iə", "I @"), ("eə", "E @"), ("ʊə", "U @"),
    ("iː", "i"), ("uː", "u"), ("ɑː", "A"), ("ɔː", "O"), ("ɜː", "r="), ("ɚ", "r="), ("ɝ", "r="),
    ("i", "i"), ("ɪ", "I"), ("ᵻ", "I"), ("ɛ", "E"), ("e", "E"), ("æ", "{"), ("ɑ", "A"), ("a", "A"),
    ("ʌ", "V"), ("ɔ", "O"), ("ɒ", "A"), ("o", "@U"), ("ʊ", "U"), ("u", "u"), ("ə", "@"), ("ɐ", "@"),
    ("θ", "T"), ("ð", "D"), ("ʃ", "S"), ("ʒ", "Z"), ("ŋ", "N"), ("ɹ", "r"), ("r", "r"), ("ɾ", "4"),
    ("ɫ", "l"), ("l", "l"), ("x", "h"), ("ç", "h"),
    ("p", "p"), ("b", "b"), ("t", "t"), ("d", "d"), ("k", "k"), ("g", "g"), ("ɡ", "g"), ("f", "f"),
    ("v", "v"), ("s", "s"), ("z", "z"), ("h", "h"), ("m", "m"), ("n", "n"), ("w", "w"), ("j", "j"),
]
IGNORE = set("ˈˌːˑ‍ʔ̩̃ -")


@dataclass
class SungSyllable:
    onset: list[str] = field(default_factory=list)
    nucleus: str = "A"
    coda: list[str] = field(default_factory=list)
    # more syllables printed under the same note ("an' sing"): the note is shared
    extra: list["SungSyllable"] = field(default_factory=list)


@functools.lru_cache(maxsize=None)
def _espeak_ipa(word: str, lang: str) -> str:
    out = subprocess.run(["espeak-ng", "-q", "--ipa=3", "-v", lang, word], capture_output=True, text=True)
    return out.stdout.strip()


def _ipa_to_sampa(ipa: str) -> list[str]:
    res, i = [], 0
    # espeak joins diphthong halves with a zero-width joiner ("e\u200dɪ")
    ipa = unicodedata.normalize("NFC", ipa).replace("\u200d", "").replace("_", " ")
    while i < len(ipa):
        if ipa[i] in IGNORE or unicodedata.combining(ipa[i]):
            i += 1
            continue
        for k, v in IPA_MAP:
            if ipa.startswith(k, i):
                res += v.split()
                i += len(k)
                break
        else:
            i += 1  # unknown symbol: drop it
    return res


def clean_word(text: str) -> str:
    t = text.lower().replace("’", "'").replace("ﬂ", "fl").replace("ﬁ", "fi")
    return re.sub(r"[^a-zäöüß'\- ]", "", t).strip("-' ")


def word_phonemes(word: str, lang: str = "en-us") -> list[str]:
    w = clean_word(word)
    if not w:
        return []
    if w in SCAT:
        return SCAT[w].split()
    if all(p in SCAT for p in w.split()):
        return " ".join(SCAT[p] for p in w.split()).split()
    return _ipa_to_sampa(_espeak_ipa(w, lang))


@functools.lru_cache(maxsize=1)
def _dictionary() -> frozenset:
    """An English word list (macOS ships one); empty if there is none."""
    for path in ("/usr/share/dict/web2", "/usr/share/dict/words"):
        try:
            with open(path) as f:
                return frozenset(w.strip().lower() for w in f)
        except OSError:
            pass
    return frozenset()


def _is_word(parts: list[str]) -> bool:
    return "".join(clean_word(p).replace("'", "") for p in parts) in _dictionary()


def regroup_words(words: list[list[str]]) -> list[list[tuple[int, int]]]:
    """Fix word boundaries the score's hyphens got wrong. Sibelius leaves out the hyphen
    between close syllables ("Christ mas", "joy ful"), and a stray hyphen can chain
    separate words ("a- way- Christ"). Returns the new words as (word, syllable) indices
    into the input. Scat syllables and several words under one note are left alone."""
    lex = _dictionary()
    flat = [(wi, si) for wi, w in enumerate(words) for si in range(len(w))]
    text = {(wi, si): words[wi][si] for wi, si in flat}
    if not lex:
        return [[(wi, si) for si in range(len(w))] for wi, w in enumerate(words)]
    fixed = lambda k: " " in text[k].strip() or not clean_word(text[k])
    scat = lambda keys: all(clean_word(text[k]) in SCAT for k in keys)
    # 1. split hyphen chains that aren't a word into the fewest words from the list
    pieces: list[list[tuple[int, int]]] = []
    for wi, w in enumerate(words):
        keys = [(wi, si) for si in range(len(w))]
        if len(keys) < 2 or _is_word(w) or any(fixed(k) for k in keys) or scat(keys):
            pieces.append(keys)
            continue
        best = {0: []}
        for j in range(1, len(keys) + 1):
            for i in range(j):
                if i in best and _is_word([text[k] for k in keys[i:j]]):
                    cand = best[i] + [keys[i:j]]
                    if j not in best or len(cand) < len(best[j]):
                        best[j] = cand
        pieces += best.get(len(keys), [keys])
    # 2. join neighbours that only make a word together (longest run first)
    out, i = [], 0
    while i < len(pieces):
        for k in range(min(4, len(pieces) - i), 1, -1):
            run = pieces[i:i + k]
            keys = [x for p in run for x in p]
            words_alone = all(_is_word([text[x] for x in p]) and len(clean_word(text[p[0]])) > 3 for p in run)
            if not any(fixed(x) for x in keys) and not scat(keys) and not words_alone and _is_word([text[x] for x in keys]):
                out.append(keys)
                i += k
                break
        else:
            out.append(pieces[i])
            i += 1
    return out


def split_syllables(phones: list[str], n: int, parts: list[str]) -> list[SungSyllable]:
    """Divide a word's phonemes over n sung syllables (one vowel nucleus each)."""
    nuclei = [i for i, p in enumerate(phones) if p in VOWELS]
    if not nuclei:
        # hum-only syllable such as "dm": the last sonorant carries the pitch
        son = [i for i, p in enumerate(phones) if p in SONORANTS]
        if son:
            nuclei = [son[-1]]
        elif phones:
            return [SungSyllable(onset=phones, nucleus="@")] + [SungSyllable(nucleus="@") for _ in range(n - 1)]
        else:
            return [SungSyllable(nucleus="@") for _ in range(n)]
    # merge surplus nuclei (espeak found more syllables than the score has)
    while len(nuclei) > n:
        # drop the weakest (schwa first, else the last)
        cand = [k for k in range(1, len(nuclei)) if phones[nuclei[k]] == "@"] or [len(nuclei) - 1]
        nuclei.pop(cand[0])
    sylls: list[SungSyllable] = []
    prev_end = 0
    for k, ni in enumerate(nuclei):
        nxt = nuclei[k + 1] if k + 1 < len(nuclei) else None
        onset = phones[prev_end:ni]
        if nxt is None:
            coda = phones[ni + 1:]
            prev_end = len(phones)
        else:
            between = phones[ni + 1:nxt]
            # vowel inside the gap belongs to a merged syllable: keep it in the coda
            next_part = parts[k + 1] if k + 1 < len(parts) else ""
            starts_consonant = bool(re.match(r"[^aeiouyäöü']", clean_word(next_part) or "a"))
            if not starts_consonant or not between:
                split = len(between)
            elif len(between) >= 2 and len(re.match(r"[^aeiouyäöü]*", clean_word(next_part)).group(0)) >= 2:
                split = len(between) - 2
            else:
                split = len(between) - 1
            coda = between[:split]
            prev_end = ni + 1 + split
        sylls.append(SungSyllable(onset=onset, nucleus=phones[ni], coda=coda))
    while len(sylls) < n:
        sylls.append(SungSyllable(nucleus=sylls[-1].nucleus))
    return sylls


def phonemize_words(words: list[list[str]], lang: str = "en-us") -> list[list[SungSyllable]]:
    """words: each word as its list of printed syllables, e.g. [["ri", "ver"], ["a"]]."""
    out = []
    for parts in words:
        # several words printed under one note ("an' sing") are sung on that note
        if len(parts) == 1 and " " in parts[0].strip():
            subs = []
            for w in parts[0].split():
                ph = word_phonemes(w, lang)
                subs += split_syllables(ph, max(1, sum(p in VOWELS for p in ph)), [w]) if ph else []
            first = subs[0] if subs else SungSyllable(nucleus="@")
            first.extra = subs[1:]
            out.append([first])
            continue
        whole = "".join(clean_word(p) for p in parts)
        phones = word_phonemes(whole, lang) if len(parts) > 1 else word_phonemes(parts[0], lang)
        if len(parts) > 1 and all(clean_word(p) in SCAT for p in parts):
            out.append([split_syllables(SCAT[clean_word(p)].split(), 1, [p])[0] for p in parts])
            continue
        out.append(split_syllables(phones, len(parts), parts))
    return out
