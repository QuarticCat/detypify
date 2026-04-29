from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

from detypify.data.paths import DEFAULT_DATA_PATHS, DataPaths
from detypify.types import DetexifySymInfo, MathSymbolSample

if TYPE_CHECKING:
    from pathlib import Path

RAW_POINT_COORD_COUNT = 3


@cache
def get_xml_parser():
    """Cached XML parser for reuse."""
    from lxml import etree

    return etree.XMLParser()


def parse_mathwriting_symbol(filepath: Path) -> MathSymbolSample | None:
    """Parse a single InkML file into its raw LaTeX label and strokes."""
    from lxml import etree

    root = etree.parse(filepath, parser=get_xml_parser()).getroot()
    namespace = {"ink": "http://www.w3.org/2003/InkML"}
    tex_label = root.findtext(".//ink:annotation[@type='label']", namespaces=namespace)
    if not tex_label:
        return None

    return MathSymbolSample(
        tex_label,
        [
            [
                (float(x), float(y))
                for x, y, _ in (
                    point_str.split()
                    for point_str in trace.text.split(",")  # type: ignore[missing-attribute]
                    if len(point_str.split()) == RAW_POINT_COORD_COUNT
                )
            ]
            for trace in root.iterfind(".//ink:trace", namespaces=namespace)
            if trace.text
        ],
    )


def collect_mathwriting_raw(paths: DataPaths = DEFAULT_DATA_PATHS):
    """Collect raw MathWriting data with original LaTeX labels."""
    from concurrent.futures import ProcessPoolExecutor

    import polars as pl

    label_acc = []
    data_acc = []
    with ProcessPoolExecutor() as executor:
        results = executor.map(parse_mathwriting_symbol, paths.mathwriting_raw_dir.glob("*.inkml"), chunksize=500)
        for result in results:
            if result is None:
                continue
            label_acc.append(result.label)
            data_acc.append(result.symbol)

    pl_schema = {
        "latex_label": pl.String,
        "symbol": pl.List(pl.List(pl.Array(pl.Float32, 2))),
    }
    return pl.DataFrame({"latex_label": label_acc, "symbol": data_acc}, schema=pl_schema).lazy()


def collect_detexify_raw(paths: DataPaths = DEFAULT_DATA_PATHS):
    """Collect raw Detexify data with original command labels."""
    import polars as pl
    from msgspec import json

    pl.Config.set_engine_affinity("streaming")
    with (paths.detexify_raw_dir / "symbols.json").open("rb") as f:
        tex_sym_info = json.decode(f.read(), type=list[DetexifySymInfo])
    key_to_command = {x.id: x.command for x in tex_sym_info}

    with (paths.detexify_raw_dir / "detexify.json").open("rb") as f:
        data = json.decode(f.read(), type=list[tuple[str, list[list[tuple[float, float, float]]]]])

    raw_lf = pl.DataFrame(
        data,
        schema={"key": pl.String, "strokes": pl.List(pl.List(pl.Array(pl.Float32, 3)))},
        orient="row",
    ).lazy()
    mapping_lf = pl.DataFrame({"key": list(key_to_command), "latex_label": list(key_to_command.values())}).lazy()

    return (
        raw_lf.join(mapping_lf, on="key", how="left")
        .filter(pl.col("latex_label").is_not_null())
        .select(
            [
                pl.col("latex_label"),
                pl.col("strokes")
                .list.eval(pl.element().list.eval(pl.element().arr.head(2).list.to_array(2)))
                .alias("symbol"),
            ]
        )
        .filter(pl.col("symbol").list.len() > 0)
    )


def collect_contrib_raw(paths: DataPaths = DEFAULT_DATA_PATHS):
    """Collect raw contributed data with symbol names."""
    import polars as pl
    from msgspec import json

    if not paths.contrib_accepted_json.exists():
        msg = f"Reviewed contribution dataset not found: {paths.contrib_accepted_json}"
        raise FileNotFoundError(msg)

    with paths.contrib_accepted_json.open("rb") as f:
        data = json.decode(f.read(), type=list[dict[str, str]])

    return (
        pl.DataFrame(data)
        .lazy()
        .rename({"sym": "latex_label"})
        .with_columns(
            pl.col("strokes")
            .map_elements(json.decode, return_dtype=pl.List(pl.List(pl.Array(pl.Float32, 2))))
            .alias("symbol")
        )
        .drop("strokes")
        .filter(pl.col("latex_label").is_not_null())
        .select(["latex_label", "symbol"])
        .filter(pl.col("symbol").list.len() > 0)
    )
