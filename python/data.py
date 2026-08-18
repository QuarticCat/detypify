"""Data processing entry script."""

import logging
from dataclasses import dataclass, field
from typing import Annotated

import cappa

from detypify.config import DataSetName
from detypify.tools.metadata import gen_metadata
from detypify.tools.raw import convert_raw_dataset
from detypify.tools.symbols import gen_symbols


@dataclass
class CommonArgs:
    datasets: Annotated[list[DataSetName], cappa.Arg(short="-d", help="Datasets to process.")] = field(
        default_factory=lambda: [DataSetName.detexify, DataSetName.mathwriting]
    )
    """Datasets to process."""


@cappa.command(name="prepare", default_long=True)
@dataclass
class Prepare(CommonArgs):
    """Convert raw data, generate symbols, and generate frontend metadata."""


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

    command: cappa.Subcommands[Prepare | Preview]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = cappa.parse(Args, completion=False)
    match args.command:
        case Prepare(datasets=datasets):
            dataset_names = list(dict.fromkeys(datasets))
            convert_raw_dataset(dataset_names=dataset_names)
            gen_symbols()
            gen_metadata(dataset_names=dataset_names)
        case Preview() as preview:
            from detypify.tools.preview import serve_dataset_preview

            serve_dataset_preview(
                dataset_names=tuple(dict.fromkeys(preview.datasets)),
                host=preview.host,
                port=preview.port,
                image_size=preview.image_size,
                page_size=preview.page_size,
            )
