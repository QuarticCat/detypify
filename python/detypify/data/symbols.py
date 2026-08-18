from importlib import resources

from detypify.data.paths import DEFAULT_DATA_PATHS
from detypify.types import TypstSymInfo


def get_typst_symbol_info() -> list[TypstSymInfo]:
    """Load non-ASCII, non-deprecated Typst `sym` metadata."""
    from msgspec.json import decode

    with DEFAULT_DATA_PATHS.typst_symbols.open("rb") as f:
        return decode(f.read(), type=list[TypstSymInfo])


def get_tex_typ_map() -> dict[str, TypstSymInfo]:
    """Map LaTeX labels only to eligible characters from Typst's `sym` module."""
    from msgspec.yaml import decode
    from unicodeit.data import REPLACEMENTS

    typ_sym_info = get_typst_symbol_info()
    char_to_typ = {s.char: s for s in typ_sym_info}
    tex_to_typ = {latex: char_to_typ[char] for latex, char in REPLACEMENTS if len(char) == 1 and char in char_to_typ}

    # The small manual table fills dataset-specific aliases and corrects visually distinct variants.
    mapping_path = resources.files("detypify") / "assets" / "tex_to_typ_sup.yaml"
    with mapping_path.open("rb") as f:
        manual_mapping = decode(f.read(), type=dict[str, str])
    name_to_typ = {name: info for info in typ_sym_info for name in info.names}
    tex_to_typ.update({latex: name_to_typ[name] for latex, name in manual_mapping.items()})
    return tex_to_typ


def get_tex_to_char() -> dict[str, str]:
    return {k: v.char for k, v in get_tex_typ_map().items()}
