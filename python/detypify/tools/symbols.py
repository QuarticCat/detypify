"""Generate the local Typst symbol asset from pinned upstream sources."""

from collections import defaultdict
from unicodedata import category
from urllib.request import urlopen

from detypify.data.paths import DEFAULT_DATA_PATHS
from detypify.types import TypstSymInfo

CODEX_VERSION = "0.3.0"
TYPST_VERSION = "0.15.1"
CODEX_URL = f"https://raw.githubusercontent.com/typst/codex/v{CODEX_VERSION}/src/modules/sym.txt"
TYPST_AST_URL = f"https://raw.githubusercontent.com/typst/typst/v{TYPST_VERSION}/crates/typst-syntax/src/ast.rs"
INVISIBLE_CATEGORIES = {"Zs", "Cc", "Cf"}
SHORTHAND_LIST_MARKER = "pub const LIST: &'static [(&'static str, char)] = &["


def _download_text(url: str) -> str:
    with urlopen(url) as response:  # noqa: S310
        return response.read().decode()


def _decode_codex_value(value: str) -> str:
    """Decode Unicode and variation-selector escapes from codex."""
    decoded = ""
    while "\\" in value:
        literal, escape = value.split("\\", 1)
        token, value = escape.split("}", 1)
        kind, payload = token.split("{", 1)
        decoded += literal
        if kind == "u":
            decoded += chr(int(payload, 16))
        else:
            selector = int(payload) if payload.isdigit() else {"text": 15, "emoji": 16}[payload]
            decoded += chr(0xFDFF + selector)
    return decoded + value


def _parse_codex(source: str) -> list[tuple[str, str]]:
    """Flatten codex modules and variants, skipping deprecated entries."""
    entries = []
    modules: list[str] = []
    symbol_name = ""
    symbol_deprecated = False
    deprecated = False

    for raw_line in source.splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue
        if line.startswith("@deprecated:"):
            deprecated = True
            continue
        if line.endswith(" {"):
            modules.append(line.removesuffix(" {").strip())
            symbol_name = ""
            deprecated = False
            continue
        if line == "}":
            modules.pop()
            symbol_name = ""
            continue

        name, _, value = line.partition(" ")
        if name.startswith("."):
            if not symbol_deprecated and not deprecated:
                full_name = ".".join([*modules, symbol_name, name[1:]])
                entries.append((full_name, _decode_codex_value(value)))
        else:
            symbol_name = name
            symbol_deprecated = deprecated
            if value and not deprecated:
                entries.append((".".join([*modules, name]), _decode_codex_value(value)))
        deprecated = False

    return entries


def _decode_rust_char(value: str) -> str:
    if value.startswith(r"\u{"):
        return chr(int(value[3:-1], 16))
    return value


def _parse_shorthands(source: str, occurrence: int) -> dict[str, str]:
    block = source.split(SHORTHAND_LIST_MARKER)[occurrence].split("];", 1)[0]
    shorthands = {}
    for source_line in block.splitlines():
        line = source_line.split("//", 1)[0].strip()
        if not line.startswith('("'):
            continue
        key, value = line[2:].split('", ', 1)
        char = value.removeprefix("'").split("'", 1)[0]
        shorthands[_decode_rust_char(char)] = key
    return shorthands


def _build_typst_symbols(codex_source: str, typst_source: str) -> list[TypstSymInfo]:
    markup_shorthands = _parse_shorthands(typst_source, 1)
    math_shorthands = _parse_shorthands(typst_source, 2)
    names_by_char: dict[str, list[str]] = defaultdict(list)

    for name, value in _parse_codex(codex_source):
        char = value[0]
        if not char.isascii() and category(char) not in INVISIBLE_CATEGORIES:
            names_by_char[char].append(name)

    return [
        TypstSymInfo(
            char=char,
            names=sorted(names),
            markup_shorthand=markup_shorthands.get(char),
            math_shorthand=math_shorthands.get(char),
        )
        for char, names in sorted(names_by_char.items())
    ]


def gen_symbols() -> None:
    from msgspec import json

    records = _build_typst_symbols(_download_text(CODEX_URL), _download_text(TYPST_AST_URL))
    path = DEFAULT_DATA_PATHS.typst_symbols
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.format(json.encode(records)) + b"\n")
