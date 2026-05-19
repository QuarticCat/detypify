from typing import Literal

from msgspec import Struct

type Point = tuple[float, float]
type Stroke = list[Point]
type Strokes = list[Stroke]
type SplitName = Literal["train", "test", "val"]


class TypstSymInfo(Struct, kw_only=True, omit_defaults=True):
    char: str
    names: list[str]
    latex_name: str | None = None
    markup_shorthand: str | None = None
    math_shorthand: str | None = None
    accent: bool = False
    alternates: list[str] | None = None


class UnmappedSymbols(Struct, kw_only=True, omit_defaults=True):
    name: str
    unmapped: set[str] | None = None


class DetexifySymInfo(Struct, kw_only=True, omit_defaults=True):
    command: str
    id: str


class MathSymbolSample(Struct):
    label: str
    symbol: Strokes
