from __future__ import annotations

import gzip
import logging
import tarfile
from time import perf_counter
from typing import TYPE_CHECKING, cast

from detypify.config import DataSetName
from detypify.data.paths import DEFAULT_DATA_PATHS, DataPaths
from detypify.types import DetexifySymInfo

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    import polars as pl

    from detypify.types import Strokes

RAW_POINT_COORD_COUNT = 3
DETEXIFY_COPY_HEADER = b"COPY samples (id, key, strokes) FROM stdin;\n"
MATHWRITING_SYMBOL_PREFIX = "mathwriting-2024/symbols/"
INKML_NAMESPACE = {"ink": "http://www.w3.org/2003/InkML"}


def _find_detexify_copy_data_start(filepath: Path) -> int:
    """Return the number of SQL dump lines before the Detexify COPY data."""
    with gzip.open(filepath, "rb") as f:
        for line_number, line in enumerate(f, start=1):
            if line == DETEXIFY_COPY_HEADER:
                return line_number

    msg = f"Could not find the Detexify samples COPY section in {filepath}"
    raise ValueError(msg)


def _parse_mathwriting_symbol(data: bytes) -> tuple[str, Strokes] | None:
    """Parse a single InkML file into its raw LaTeX label and strokes."""
    from lxml import etree

    root = etree.fromstring(data)
    tex_label = root.findtext(".//ink:annotation[@type='label']", namespaces=INKML_NAMESPACE)
    if not tex_label:
        return None

    strokes = []
    for trace in root.iterfind(".//ink:trace", namespaces=INKML_NAMESPACE):
        trace_text = cast("str | None", trace.text)
        if not trace_text:
            continue

        # InkML points contain x, y, and time; only geometry is needed for rasterization.
        stroke = []
        for raw_point in trace_text.split(","):
            point = raw_point.split()
            if len(point) == RAW_POINT_COORD_COUNT:
                x, y, _ = point
                stroke.append((float(x), float(y)))
        strokes.append(stroke)

    return tex_label, strokes


def _iter_mathwriting_symbol_data(filepath: Path) -> Iterator[bytes]:
    """Yield the official archive's contiguous section of symbol InkML files."""
    in_symbol_section = False
    # Stream the large archive and stop once its ordered symbol section ends.
    with tarfile.open(filepath, "r|gz") as archive:
        for member in archive:
            if not member.name.startswith(MATHWRITING_SYMBOL_PREFIX):
                if in_symbol_section:
                    break
                continue
            in_symbol_section = True

            if not member.isfile() or not member.name.endswith(".inkml"):
                continue

            extracted = archive.extractfile(member)
            if extracted is None:
                msg = f"Could not read {member.name} from {filepath}"
                raise ValueError(msg)
            yield extracted.read()


def _collect_mathwriting_raw(paths: DataPaths) -> pl.LazyFrame:
    """Collect raw MathWriting data with original LaTeX labels."""
    import polars as pl

    labels = []
    strokes = []
    for data in _iter_mathwriting_symbol_data(paths.raw_mathwriting_dir / "mathwriting-2024.tgz"):
        sample = _parse_mathwriting_symbol(data)
        if sample is None:
            continue
        label, symbol_strokes = sample
        labels.append(label)
        strokes.append(symbol_strokes)

    schema = {"latex_label": pl.String, "strokes": pl.List(pl.List(pl.Array(pl.Float32, 2)))}
    return pl.LazyFrame({"latex_label": labels, "strokes": strokes}, schema=schema)


def _collect_detexify_raw(paths: DataPaths) -> pl.LazyFrame:
    """Collect raw Detexify data with original command labels."""
    import polars as pl
    from msgspec import json

    with (paths.raw_detexify_dir / "symbols.json").open("rb") as f:
        tex_sym_info = json.decode(f.read(), type=list[DetexifySymInfo])
    key_to_command = {x.id: x.command for x in tex_sym_info}

    dump_path = paths.raw_detexify_dir / "detexify.sql.gz"
    data_start = _find_detexify_copy_data_start(dump_path)
    strokes_dtype = pl.List(pl.List(pl.List(pl.Float32)))

    # Decode the PostgreSQL COPY section lazily and normalize every point to its x/y coordinates.
    return (
        pl.scan_csv(
            dump_path,
            separator="\t",
            has_header=False,
            skip_rows=data_start,
            new_columns=["id", "key", "strokes_json"],
            schema_overrides={"id": pl.String, "key": pl.String, "strokes_json": pl.String},
            quote_char=None,
            truncate_ragged_lines=True,
        )
        .filter(
            pl.col("strokes_json").is_not_null()
            & ~pl.col("strokes_json").str.contains(r"^\s*\[\s*\]\s*$")
            & pl.col("key").is_in(list(key_to_command))
        )
        .select(
            pl.col("key").replace_strict(key_to_command, default=None).alias("latex_label"),
            pl.col("strokes_json")
            .str.json_decode(dtype=strokes_dtype)
            .list.eval(pl.element().list.eval(pl.element().list.head(2).list.to_array(2)))
            .alias("strokes"),
        )
    )


def convert_raw_dataset(dataset_names: Sequence[DataSetName], paths: DataPaths = DEFAULT_DATA_PATHS) -> pl.DataFrame:
    """Convert original source files into the local raw Parquet dataset."""
    import polars as pl

    logger = logging.getLogger(__name__)
    conversion_started = perf_counter()
    logger.info("Converting raw datasets: %s", ", ".join(dataset_names))

    lfs: list[pl.LazyFrame] = []
    for dataset_name in dataset_names:
        match dataset_name:
            case DataSetName.mathwriting:
                raw = _collect_mathwriting_raw(paths)
            case DataSetName.detexify:
                raw = _collect_detexify_raw(paths)
        source_col = pl.lit(dataset_name.value).alias("source")
        lfs.append(raw.sort("latex_label").with_columns(source_col))

    # Execute all source pipelines together so Polars can collect them with its streaming engine.
    collect_started = perf_counter()
    df = pl.concat(lfs).collect(engine="streaming")
    logger.info(
        "Collected %s samples with %s distinct LaTeX labels in %.2f s",
        f"{len(df):,}",
        f"{df.get_column('latex_label').n_unique():,}",
        perf_counter() - collect_started,
    )

    paths.raw_converted_parquet.parent.mkdir(parents=True, exist_ok=True)
    write_started = perf_counter()
    logger.info("Writing Zstd Parquet to %s", paths.raw_converted_parquet)
    df.write_parquet(paths.raw_converted_parquet, compression="zstd")
    file_size_mib = paths.raw_converted_parquet.stat().st_size / (1024 * 1024)
    logger.info(
        "Saved raw dataset (%.1f MiB) in %.2f s; total conversion time %.2f s",
        file_size_mib,
        perf_counter() - write_started,
        perf_counter() - conversion_started,
    )
    return df
