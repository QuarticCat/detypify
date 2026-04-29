from detypify.config import DataSetName
from detypify.data.datasets import map_raw_dataset
from detypify.data.paths import DEFAULT_DATA_PATHS, DataPaths
from detypify.data.symbols import get_typst_symbol_info


def generate_data_info(classes: list[str], paths: DataPaths = DEFAULT_DATA_PATHS) -> None:
    """Generate frontend inference and contribution metadata."""
    import logging

    from msgspec import json

    paths.generated_dir.mkdir(exist_ok=True, parents=True)

    typ_sym_info = get_typst_symbol_info()
    infer = []
    contrib = {n: s.char for s in typ_sym_info for n in s.names}
    chr_to_sym = {s.char: s for s in typ_sym_info}
    for c in classes:
        if c not in chr_to_sym:
            continue
        sym = chr_to_sym[c]
        info = {"char": sym.char, "names": sym.names}
        if sym.markup_shorthand and sym.math_shorthand:
            info["shorthand"] = sym.markup_shorthand
        elif sym.markup_shorthand:
            info["markupShorthand"] = sym.markup_shorthand
        elif sym.math_shorthand:
            info["mathShorthand"] = sym.math_shorthand
        infer.append(info)

    logger = logging.getLogger(__name__)
    for path, info_data in [(paths.infer_json, infer), (paths.contrib_json, contrib)]:
        with path.open("wb") as f:
            f.write(json.encode(info_data))
        logger.info("Generated data at %s", path)

    _, unmapped = map_raw_dataset([DataSetName.mathwriting, DataSetName.detexify])
    with paths.unmapped_latex_symbols_json.open("wb") as f:
        f.write(json.format(json.encode(unmapped)))
