from functools import cache
from hashlib import sha256
from importlib import resources
from json import dumps
from typing import cast

from detypify.types import TypstSymInfo


@cache
def is_invisible(c: str) -> bool:
    from unicodedata import category

    return category(c) in {"Zs", "Cc", "Cf"}


@cache
def get_typst_symbol_info() -> list[TypstSymInfo]:
    """Parse Typst symbol metadata from the online reference."""
    import logging
    import re
    from urllib.request import urlopen

    from bs4 import BeautifulSoup

    page_url = "https://typst.app/docs/reference/symbols/sym/"
    with urlopen(page_url) as resp:
        page_data = resp.read()

    sym_info = {}
    if page_data:
        soup = BeautifulSoup(page_data, "lxml")
        for li in soup.find_all("li", id=re.compile("^symbol-")):
            name = li["id"][len("symbol-") :]
            char = li["data-value"][0]
            if is_invisible(char) or li.get("data-deprecation"):
                continue
            if char in sym_info:
                sym_info[char].names.append(name)
                continue

            latex_name, markup_shorthand, math_shorthand, alternates = (
                li.get("data-latex-name"),
                li.get("data-markup-shorthand"),
                li.get("data-math-shorthand"),
                li.get("data-alternates", ""),
            )
            sym_info[char] = TypstSymInfo(
                char=char,
                names=[cast("str", name)],
                latex_name=cast("str | None", latex_name),
                markup_shorthand=cast("str | None", markup_shorthand),
                math_shorthand=cast("str | None", math_shorthand),
                accent=li.get("accent") == "true",
                alternates=cast("str", alternates).split(),
            )
    else:
        logging.getLogger(__name__).warning("Unable to retrieve page data.")

    return list(sym_info.values())


def get_tex_typ_map() -> dict[str, TypstSymInfo]:
    """Create a mapping from TeX command names to Typst symbol information."""
    typ_sym_info = get_typst_symbol_info()
    tex_to_typ = {s.latex_name: s for s in typ_sym_info if s.latex_name is not None}
    name_to_typ = {name: s for s in typ_sym_info for name in s.names}

    mapping_path = resources.files("detypify") / "assets" / "tex_to_typ_sup.yaml"
    with mapping_path.open("rb") as f:
        from msgspec.yaml import decode

        manual_mapping = decode(f.read(), type=dict[str, str])

    tex_to_typ |= {k: name_to_typ[v] for k, v in manual_mapping.items()}
    return tex_to_typ


@cache
def get_tex_typ_map_digest() -> str:
    """Return a stable digest for the effective LaTeX-to-Typst mapping."""
    records = [
        {
            "char": typ.char,
            "latex": latex,
            "names": sorted(typ.names),
        }
        for latex, typ in sorted(get_tex_typ_map().items())
    ]
    payload = dumps(records, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return sha256(payload).hexdigest()


def get_tex_to_char() -> dict[str, str]:
    return {k: v.char for k, v in get_tex_typ_map().items()}
