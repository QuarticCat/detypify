"""Data processing entry script."""

import logging
from dataclasses import dataclass, field
from typing import Annotated

import cappa

from detypify.config import DataSetName
from detypify.tools.metadata import generate_data_info
from detypify.tools.raw import convert_raw_dataset


@dataclass
class CommonArgs:
    datasets: Annotated[list[DataSetName], cappa.Arg(short="-d", help="Datasets to process.")] = field(
        default_factory=lambda: [DataSetName.detexify, DataSetName.mathwriting]
    )
    """Datasets to process."""


@cappa.command(name="convert-raw", default_long=True)
@dataclass
class ConvertRaw(CommonArgs):
    """Convert original source files into the local raw Parquet dataset."""


@cappa.command(name="gen-metadata", default_long=True)
@dataclass
class GenMetadata(CommonArgs):
    """Generate frontend metadata and unmapped-symbol review data."""


@cappa.command(name="preview", default_long=True)
@dataclass
class Preview(CommonArgs):
    """Serve a local browser for mapped dataset samples."""

    host: str = "127.0.0.1"
    """Server host."""

    port: int = 8000
    """Server port."""

    image_size: int = 224
    """Rendered sample size."""

    page_size: int = 120
    """Default samples per page."""


@cappa.command(name="data")
@dataclass
class Args:
    """Process and inspect datasets."""

    command: cappa.Subcommands[ConvertRaw | GenMetadata | Preview]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = cappa.parse(Args, completion=False)
    match args.command:
        case ConvertRaw(datasets=datasets):
            convert_raw_dataset(dataset_names=list(dict.fromkeys(datasets)))
        case GenMetadata(datasets=datasets):
            generate_data_info(dataset_names=list(dict.fromkeys(datasets)))
        case Preview() as preview:
            from detypify.tools.preview import serve_dataset_preview

            serve_dataset_preview(
                dataset_names=tuple(dict.fromkeys(preview.datasets)),
                host=preview.host,
                port=preview.port,
                image_size=preview.image_size,
                page_size=preview.page_size,
            )
