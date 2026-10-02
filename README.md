# singer

Choir rehearsal tracks from a score PDF: every part is sung with its lyrics.

```
./setup.sh                      # once
.venv/bin/python sing.py        # asks: input PDF, focus voice (or tutti / all), count-in, clicks
```

Non-interactive (repeatable):

```
.venv/bin/python sing.py input/score.pdf --voice all --tempo 8=116 --count-in --click --musicxml
.venv/bin/python sing.py score.pdf --voice T --no-click --singers 1
```

Output goes to `output/`: `<Title> - <Part>.mp3` for each focus track, `<Title> - Tutti.mp3`, and
with `--musicxml` the recognised score, which you can open in MuseScore to check the notes.

## How it works

1. **Read the PDF** (`singer/pdfscore.py`). Sibelius PDFs are vector files: staff lines, stems,
   beams and barlines are lines, and noteheads, rests, clefs, accidentals and flags are glyphs of
   the Opus music font. The reader rebuilds pitches, durations (beams, flags, dots, tuplets),
   ties, key and time signatures, repeats ("sing 4x"), tempo, two-voice staves, divisi chords and
   lyrics (with hyphenation and verses). It also handles staves that are hidden when a part rests.
   A staff shared by two parts ("B./Bar.", "S1/S2") becomes two parts. The higher voice type
   takes the upper notes and the other the lower ones. Where the staff shows one line, both sing it.
   Text printed above such a staff belongs to its upper voice.
   It only works on vector PDFs; scanned scores would need an OMR tool such as Audiveris first.
2. **Lyrics to phonemes** (`singer/phonemes.py`): espeak-ng phonemises whole words, which are then
   split over the notes. Scat syllables have fixed pronunciations (`SCAT` in `phonemes.py`):
   "dm" is sung "dim", "bam" as "bahm", and so on. Sibelius leaves out the hyphen between close
   syllables ("Christ mas"), so word boundaries are checked against the system word list
   (`/usr/share/dict/web2`): syllables that only make a word together are joined, and a hyphen
   chain that isn't a word is split into words.
3. **Singing** (`singer/dsinger.py`): DiffSinger, a neural singing synthesizer trained on real
   singers. For each phrase it gets the phonemes, how long each one lasts and the exact pitch
   curve we want (small scoop, legato glides, slow drift, delayed vibrato). The voicebank's duration
   model proposes the consonant lengths, which are kept within sung limits. Onsets come before the
   beat and the vowel lands on it. A long closing diphthong ("I", "day") holds its first vowel and
   glides at the end. Before each phrase the singer takes a breath. Stops (b d g p t k) get a
   short silence before them and use the banks' German stops, whose English ones are soft and
   blur into the vowels ("dam bam" would sound like "dam dam"). Only phrase openings scoop into
   the note. Voices, from the LUNAI
   Project voicebanks in `vendor/voicebanks/`: soprano and alto are sung by Katyusha, tenor by
   Keiro Revenant, baritone and bass by Liam Thorne (tenor, baritone and bass are always male
   voices). Each part blends the bank's voice modes differently (`SINGERS` in `dsinger.py`).
   `--steps` trades quality for speed (default 20). `--singers 2` adds a second singer per part
   with another blend of voice modes, slightly detuned and late.
   `--engine mbrola` uses the old MBROLA diphone voices (`singer/synth.py`) instead.
4. **Mix** (`singer/mix.py`): like the hand-made rehearsal tracks, a focus track puts the chosen
   part hard left and the rest of the choir quietly on the right, with the metronome.
   Tutti spreads all parts across the stereo field. Each track starts with every part singing its
   first note, then a one-bar count-in. A short, light room reverb is added, and a limiter brings
   the track to about -14 dBFS RMS.

Notation the reader doesn't handle yet: D.S./D.C./Coda jumps, first and second endings,
fermatas and ritardandos (it plays in strict tempo), and grace notes (skipped).
Any bar whose notes don't add up is printed as a warning.

## Licence

The code is MIT licensed (see `LICENSE`). The voicebanks and MBROLA are separate projects under
their own licences.

## Voicebank licence

The voicebanks are not part of this project; `setup.sh` downloads them. They are licensed
CC BY-NC-SA 4.0 with the LUNAI Project terms of use (in each bank folder): non-commercial use
only, credited as "Katyusha, Keiro Revenant and Liam Thorne from LUNAI Project". Commercial use
needs their written permission, and the audio may not be used to train other voice models.

## Checking a render

`.venv/bin/python tools/check_render.py score.pdf [--bars 30-40]` renders every voice and
reports notes that are silent in the audio, sung onsets that start more than 70 ms after the
beat, and how far the sung pitch is from the written note (`--parts T.,B.` for some parts only,
`--engine mbrola` for the old voices).
