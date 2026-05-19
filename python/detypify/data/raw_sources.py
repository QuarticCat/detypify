from __future__ import annotations

from typing import TYPE_CHECKING

from detypify.data.paths import DEFAULT_DATA_PATHS, DataPaths
from detypify.types import DetexifySymInfo, MathSymbolSample

if TYPE_CHECKING:
    from pathlib import Path

RAW_POINT_COORD_COUNT = 3


def parse_mathwriting_symbol(filepath: Path) -> MathSymbolSample | None:
    """Parse a single InkML file into its raw LaTeX label and strokes."""
    from lxml import etree

    root = etree.parse(filepath).getroot()
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
    return pl.LazyFrame({"latex_label": label_acc, "symbol": data_acc}, schema=pl_schema)


def collect_detexify_raw(paths: DataPaths = DEFAULT_DATA_PATHS):
    """Collect raw Detexify data with original command labels."""
    import polars as pl
    from msgspec import json

    with (paths.detexify_raw_dir / "symbols.json").open("rb") as f:
        tex_sym_info = json.decode(f.read(), type=list[DetexifySymInfo])
    key_to_command = {x.id: x.command for x in tex_sym_info}

    with (paths.detexify_raw_dir / "detexify.json").open("rb") as f:
        data = json.decode(f.read(), type=list[tuple[str, list[list[tuple[float, float, float]]]]])

    return (
        pl.LazyFrame(
            data,
            schema={"key": pl.String, "strokes": pl.List(pl.List(pl.Array(pl.Float32, 3)))},
            orient="row",
        )
        .select(
            pl.col("key").replace_strict(key_to_command, default=None).alias("latex_label"),
            pl.col("strokes")
            .list.eval(pl.element().list.eval(pl.element().arr.head(2).list.to_array(2)))
            .alias("symbol"),
        )
        .filter(pl.col("latex_label").is_not_null())
        .filter(pl.col("symbol").list.len() > 0)
    )


def collect_contrib_raw(paths: DataPaths = DEFAULT_DATA_PATHS):
    """Collect raw contributed data with symbol names."""
    import polars as pl
    from msgspec import json

    with paths.contrib_accepted_json.open("rb") as f:
        data = json.decode(f.read(), type=list[dict[str, str]])

    return (
        pl.LazyFrame(data)
        .select(
            pl.col("sym").alias("latex_label"),
            pl.col("strokes")
            .map_elements(json.decode, return_dtype=pl.List(pl.List(pl.Array(pl.Float32, 2))))
            .alias("symbol"),
        )
        .filter(pl.col("latex_label").is_not_null())
        .filter(pl.col("symbol").list.len() > 0)
    )
