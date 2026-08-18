from msgspec import Struct

type Point = tuple[float, float]
type Stroke = list[Point]
type Strokes = list[Stroke]


class TypstSymInfo(Struct, kw_only=True, omit_defaults=True):
    char: str
    names: list[str]
    markup_shorthand: str | None = None
    math_shorthand: str | None = None


class DetexifySymInfo(Struct, kw_only=True):
    command: str
    id: str
