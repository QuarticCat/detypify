from typing import Literal

from msgspec import Struct

type Point = tuple[float, float]
type Stroke = list[Point]
type Strokes = list[Stroke]
type SplitName = Literal["train", "test", "val"]


class TypstSymInfo(Struct, kw_only=True, omit_defaults=True):
    char: str
    names: list[str]
    latex_name: str | None
    markup_shorthand: str | None
    math_shorthand: str | None
    accent: bool
    alternates: list[str]


class UnmappedSymbols(Struct, kw_only=True, omit_defaults=True):
    name: str
    unmapped: set[str] | None


class DetexifySymInfo(Struct, kw_only=True, omit_defaults=True):
    command: str
    id: str


class MathSymbolSample(Struct):
    label: str
    symbol: Strokes
