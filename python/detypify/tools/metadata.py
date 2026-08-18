from collections.abc import Sequence

from detypify.config import DataSetName
from detypify.data.datasets import map_raw_dataset
from detypify.data.paths import DEFAULT_DATA_PATHS, DataPaths
from detypify.data.symbols import get_typst_symbol_info


def gen_metadata(dataset_names: Sequence[DataSetName], paths: DataPaths = DEFAULT_DATA_PATHS) -> None:
    """Generate frontend inference and contribution metadata."""
    import logging

    from msgspec import json

    paths.metadata_dir.mkdir(exist_ok=True, parents=True)

    mapped = map_raw_dataset(dataset_names, paths=paths)
    classes = mapped.get_column("label").unique().sort().to_list()
    typ_sym_info = get_typst_symbol_info()

    # Inference metadata follows model class order, while contribution metadata accepts every Typst alias.
    infer = []
    contrib = {n: s.char for s in typ_sym_info for n in s.names}
    chr_to_sym = {s.char: s for s in typ_sym_info}
    for c in classes:
        sym = chr_to_sym[c]
        info = {"char": sym.char, "names": sym.names}
        if sym.markup_shorthand and sym.math_shorthand:
            info["shorthand"] = sym.markup_shorthand
        elif sym.markup_shorthand:
            info["markupShorthand"] = sym.markup_shorthand
        elif sym.math_shorthand:
            info["mathShorthand"] = sym.math_shorthand
        infer.append(info)

    # Keep generated files independent so consumers can load only the metadata they need.
    logger = logging.getLogger(__name__)
    for path, info_data in [
        (paths.metadata_dir / "infer.json", infer),
        (paths.metadata_dir / "contrib.json", contrib),
    ]:
        with path.open("wb") as f:
            f.write(json.encode(info_data))
        logger.info("Generated data at %s", path)
