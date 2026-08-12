"""Data processing entry script."""

from dataclasses import dataclass, field
from os import process_cpu_count
from typing import Annotated

import cappa
from detypify.config import DataSetName
from detypify.data.datasets import create_raw_dataset
from detypify.data.metadata import generate_data_info
from detypify.data.symbols import get_tex_typ_map_digest


@dataclass
class CommonArgs:
    datasets: Annotated[list[DataSetName], cappa.Arg(short="-d", help="Datasets to process.")] = field(
        default_factory=lambda: [DataSetName.detexify, DataSetName.mathwriting]
    )
    """Datasets to process."""


@cappa.command(name="digest")
@dataclass
class Digest:
    """Print the effective LaTeX-to-Typst mapping digest."""


@cappa.command(name="upload", default_long=True)
@dataclass
class Upload(CommonArgs):
    """Convert local raw files and upload the raw LaTeX-annotated dataset."""


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

    num_workers: int = process_cpu_count() or 1
    """Dataset mapping worker count."""


@cappa.command(name="proc-data")
@dataclass
class Args:
    """Process and inspect datasets."""

    command: cappa.Subcommands[Digest | Upload | GenMetadata | Preview]


if __name__ == "__main__":
    args = cappa.parse(Args, completion=False)
    match args.command:
        case Digest():
            print(get_tex_typ_map_digest())  # noqa: T201
        case Upload(datasets=datasets):
            create_raw_dataset(dataset_names=list(dict.fromkeys(datasets)))
        case GenMetadata(datasets=datasets):
            generate_data_info(dataset_names=list(dict.fromkeys(datasets)))
        case Preview() as preview:
            from detypify.tools.dataset_preview import serve_dataset_preview

            serve_dataset_preview(
                dataset_names=tuple(dict.fromkeys(preview.datasets)),
                host=preview.host,
                port=preview.port,
                image_size=preview.image_size,
                page_size=preview.page_size,
                num_proc=preview.num_workers,
            )
