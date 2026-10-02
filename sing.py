#!/usr/bin/env python3
"""Choir rehearsal tracks from a score PDF: every part sung with its lyrics.

Interactive:      .venv/bin/python sing.py
Non-interactive:  .venv/bin/python sing.py input/score.pdf --voice T --count-in --click
                  .venv/bin/python sing.py input/score.pdf --voice all
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

from singer.mix import metronome, mix, write_mp3
from singer.pdfscore import read_pdf
from singer.perform import SungNote, build
from singer.phonemes import SungSyllable
from singer.dsinger import sing as dsing
from singer.synth import SR, sing

ROOT = Path(__file__).resolve().parent


def pick_voice_model(name: str, part_notes) -> str:
    n = re.sub(r"[^a-z]", "", name.lower())
    if n.startswith(("t", "ten", "bar")):  # tenor and baritone: the lighter male voice
        return "us3"
    if n.startswith(("b", "bar")):
        return "us2"
    if n.startswith(("s", "m", "a")):
        return "us1"
    mids = sorted(m for m in part_notes)
    return "us2" if mids and mids[len(mids) // 2] < 57 else "us1"


def pick_voice_kind(name: str, part_notes) -> str:
    """soprano / alto / tenor / bass, from the part name or else the median pitch."""
    n = re.sub(r"[^a-z]", "", name.lower())
    for prefixes, kind in ((("s", "m"), "soprano"), (("a", "c"), "alto"), (("t",), "tenor"), (("b",), "bass")):
        if n.startswith(prefixes):
            return "baritone" if n.startswith("bar") else kind
    mids = sorted(part_notes)
    med = mids[len(mids) // 2] if mids else 60
    return "bass" if med < 52 else "tenor" if med < 60 else "alto" if med < 67 else "soprano"


def parse_tempo(spec: str) -> tuple[float, float | None]:
    """'8=116' -> (58.0 quarters per minute, click every 0.5 quarter); '72' -> (72, None)."""
    m = re.fullmatch(r"\s*(1|2|4|8|16)(\.?)\s*=\s*(\d+(?:[.,]\d+)?)\s*", spec)
    if m:
        unit = 4 / int(m.group(1)) * (1.5 if m.group(2) else 1)
        return float(m.group(3).replace(",", ".")) * unit, unit
    try:
        return float(spec.replace(",", ".")), None
    except ValueError:
        sys.exit(f"Tempo not understood: {spec!r} (examples: 8=116, 4.=39, 4=58, 58)")


def describe_tempo(qpm: float) -> str:
    return f"4={qpm:g} (8={qpm * 2:g}, 4.={qpm / 1.5:.4g})"


def ask(prompt: str, options: list[str], default: int | None = None) -> int:
    for i, o in enumerate(options, 1):
        print(f"  {i}) {o}")
    while True:
        s = input(f"{prompt}{f' [{default + 1}]' if default is not None else ''}: ").strip()
        if not s and default is not None:
            return default
        if s.isdigit() and 1 <= int(s) <= len(options):
            return int(s) - 1
        print("  please enter a number from the list")


def yes(prompt: str, default: bool) -> bool:
    s = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    return default if not s else s.startswith(("y", "j"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", nargs="?", help="score PDF (vector PDF from Sibelius)")
    ap.add_argument("--voice", help="part to focus on (name as printed, e.g. 'T.' or 'T'), 'tutti' or 'all'")
    ap.add_argument("--count-in", dest="count_in", action=argparse.BooleanOptionalAction, default=None,
                    help="one bar of metronome clicks before the music")
    ap.add_argument("--click", action=argparse.BooleanOptionalAction, default=None,
                    help="metronome clicks throughout the piece")
    ap.add_argument("--pitch-cue", dest="pitch_cue", action=argparse.BooleanOptionalAction, default=True,
                    help="each part sings its first note before the count-in")
    ap.add_argument("--singers", type=int, default=1, help="singers per part (2+ = section sound, slower)")
    ap.add_argument("--engine", choices=["diffsinger", "mbrola"], default="diffsinger",
                    help="diffsinger: neural singing voices (vendor/voicebanks); mbrola: the old diphone voices")
    ap.add_argument("--steps", type=int, default=20, help="diffsinger quality/speed (more steps = cleaner, slower)")
    ap.add_argument("--smooth", action=argparse.BooleanOptionalAction, default=True,
                    help="re-synthesise with the WORLD vocoder for a less metallic tone")
    ap.add_argument("--lang", default="en-us", help="espeak-ng language for the lyrics (e.g. en-us, de)")
    ap.add_argument("--tempo", help="override tempo: '8=116' (eighths), '4.=39', '4=58' or a number "
                                    "(quarters per minute). With a note value, the metronome clicks in that value")
    ap.add_argument("--out", default=str(ROOT / "output"), help="output folder")
    ap.add_argument("--musicxml", action="store_true", help="also write the recognised score as MusicXML")
    a = ap.parse_args()
    interactive = a.pdf is None or a.voice is None

    # 1. input file
    pdf = a.pdf
    if pdf is None:
        found = sorted((ROOT / "input").glob("*.pdf")) + sorted(p for p in ROOT.glob("*.pdf"))
        if found:
            print("Input file:")
            i = ask("Choose a PDF", [p.name for p in found] + ["other path..."], 0)
            pdf = str(found[i]) if i < len(found) else input("Path to PDF: ").strip().strip("'\"")
        else:
            pdf = input("Path to the score PDF: ").strip().strip("'\"")
    if not Path(pdf).exists():
        sys.exit(f"File not found: {pdf}")

    print(f"Reading {Path(pdf).name} ...")
    score = read_pdf(pdf)
    click_unit = None
    tempo_spec = a.tempo
    if tempo_spec is None and interactive:
        tempo_spec = input(f"Tempo [{describe_tempo(score.tempos[0][1])} from the score; e.g. 8=116]: ").strip() or None
    if tempo_spec:
        qpm, click_unit = parse_tempo(tempo_spec)
        # scale every tempo marking by the same factor, so tempo changes keep their proportion
        f = qpm / score.tempos[0][1]
        score.tempos = [(t, b * f) for t, b in score.tempos]
    names = [p.name for p in score.parts]
    disp = {p.name: score.display_names.get(p.name, p.name) for p in score.parts}
    bars = score.measures[-1].number
    print(f"  '{score.title}': {bars} bars, parts: {', '.join(disp[n] for n in names)}, "
          f"tempo {describe_tempo(score.tempos[0][1])}")
    if score.warnings:
        print(f"  {len(score.warnings)} notation warning(s), e.g. {score.warnings[0]}")

    # 2. focus voice
    if a.voice is None:
        print("\nWhich voice should be in focus?")
        opts = [f"{disp[n]}" for n in names] + ["Tutti (all voices equal)", "All: one file per voice + tutti"]
        i = ask("Choose", opts, len(names))
        focus_list = [names[i]] if i < len(names) else ([None] if i == len(names) else names + [None])
    else:
        v = a.voice.strip().lower()
        if v == "tutti":
            focus_list = [None]
        elif v == "all":
            focus_list = names + [None]
        else:
            m = [n for n in names if n.lower().rstrip(".") == v.rstrip(".") or disp[n].lower() == v]
            if not m:
                sys.exit(f"No part called {a.voice!r}. Parts: {', '.join(names)}")
            focus_list = m[:1]

    # 3. metronome
    count_in = a.count_in if a.count_in is not None else (yes("Count-in (one bar of metronome)?", True) if interactive else True)
    click_all = a.click if a.click is not None else (yes("Metronome clicks throughout?", False) if interactive else False)

    # 4. render
    t0 = time.time()
    perf = build(score, a.lang, click_unit)
    beat = perf.beat_len_first
    # like the hand-made tracks: every part first gives its starting note, then the count-in
    main_beat = getattr(perf, "beat_unit", 1.0) * 60 / score.tempos[0][1]
    cue = min(2.5, max(1.2, 2 * main_beat)) if a.pitch_cue else 0.0
    count_start = cue + (beat if a.pitch_cue else 0.0)
    music_start = count_start + (beat * perf.beats_per_bar_first if count_in else 0.0)
    total = music_start + perf.duration + 2.5
    tracks = {}
    for pi, part in enumerate(score.parts):
        mids = [n.midi for v in part.voices for n in v.notes]
        model = pick_voice_kind(part.name, mids) if a.engine == "diffsinger" else pick_voice_model(part.name, mids)
        print(f"Singing {disp[part.name]} ({model}, {len(part.voices)} line{'s' if len(part.voices) > 1 else ''}) ...")
        acc = np.zeros(int((total + 0.5) * SR), np.float32)
        for vi, notes in enumerate(perf.voices[part.name]):
            notes = [replace(n, t=n.t + music_start) for n in notes]
            if cue and vi == 0 and notes:
                notes.insert(0, SungNote(0.0, cue, notes[0].midi, SungSyllable(nucleus="A")))
            if a.engine == "diffsinger":
                y = dsing(notes, total, model, singers=a.singers, seed=pi * 10 + vi, steps=a.steps)
            else:
                y = sing(notes, total, model, singers=a.singers, seed=pi * 10 + vi, smooth=a.smooth)
            m = min(len(y), len(acc))
            acc[:m] += y[:m] * (1.0 if vi == 0 else 0.8)
        tracks[part.name] = acc

    beats = []
    if count_in:
        per_beat = max(1, round(getattr(perf, "beat_unit", 1.0) * 60 / score.tempos[0][1] / beat))
        beats += [(count_start + k * beat, 2 if k == 0 else (1 if k % per_beat == 0 else 0))
                  for k in range(perf.beats_per_bar_first)]
    if click_all:
        beats += [(music_start + t, d) for t, d in perf.beats]
    n = max(len(x) for x in tracks.values())
    clicks = metronome(beats, n) if beats else None

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_title = re.sub(r"[^\w\- ]", "", score.title).strip() or Path(pdf).stem
    written = []
    for focus in focus_list:
        label = "Tutti" if focus is None else re.sub(r"[^\w]", "", disp[focus]) or focus
        stereo = mix(tracks, focus, clicks)
        path = out_dir / f"{safe_title} - {label}.mp3"
        write_mp3(stereo, str(path))
        written.append(path)
    if a.musicxml:
        from singer.musicxml import write_musicxml

        xml = out_dir / f"{safe_title}.musicxml"
        write_musicxml(score, str(xml))
        written.append(xml)
    print(f"\nDone in {time.time() - t0:.0f}s:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
