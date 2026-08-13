from msgspec import Struct

type Point = tuple[float, float]
type Stroke = list[Point]
type Strokes = list[Stroke]


class TypstSymInfo(Struct, kw_only=True):
    char: str
    names: list[str]
    latex_name: str | None
    markup_shorthand: str | None
    math_shorthand: str | None
    accent: bool
    alternates: list[str]


class DetexifySymInfo(Struct, kw_only=True):
    command: str
    id: str
