"""Score data model shared by the PDF reader, MusicXML writer and synthesizer.

All times are in quarter notes (float) unless stated otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Syllable:
    text: str
    hyphen_after: bool = False  # syllable continues into the next one ("ri - ver")


@dataclass
class Note:
    start: float  # quarter notes from the beginning of the written score
    dur: float
    midi: int
    lyrics: dict[int, Syllable] = field(default_factory=dict)  # verse number -> syllable
    measure: int = 0
    tie_start: bool = False  # a tie/slur curve starts on this note
    x_notehead: bool = False  # spoken / "x" notehead


@dataclass
class Voice:
    notes: list[Note] = field(default_factory=list)


@dataclass
class Part:
    name: str
    clef_midi_center: int = 60  # rough centre of the staff, used to pick a singer
    voices: list[Voice] = field(default_factory=list)

    def lowest(self) -> int:
        return min((n.midi for v in self.voices for n in v.notes), default=60)

    def highest(self) -> int:
        return max((n.midi for v in self.voices for n in v.notes), default=60)


@dataclass
class Measure:
    number: int
    start: float
    length: float
    time_sig: tuple[int, int]
    repeat_start: bool = False
    repeat_end: bool = False
    repeat_times: int = 2  # total number of passes for a repeat ending at this bar
    fifths: int = 0  # key signature: sharps > 0, flats < 0
    text: str = ""  # instructions printed at this bar (e.g. "sing 4x")


@dataclass
class Score:
    title: str
    parts: list[Part]
    measures: list[Measure]
    tempos: list[tuple[float, float]]  # (start in quarters, quarter notes per minute)
    warnings: list[str] = field(default_factory=list)

    def part(self, name: str) -> Part:
        return next(p for p in self.parts if p.name == name)
