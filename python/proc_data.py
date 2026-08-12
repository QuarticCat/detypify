"""Data processing entry script."""

from dataclasses import dataclass, field
from os import process_cpu_count
from typing import Annotated

import cappa
from detypify.config import DataSetName
from detypify.data.metadata import generate_data_info
from detypify.data.raw import convert_raw_dataset, upload_raw_dataset
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


@cappa.command(name="convert-raw", default_long=True)
@dataclass
class ConvertRaw(CommonArgs):
    """Convert original source files into the local raw Parquet dataset."""


@cappa.command(name="upload-raw")
@dataclass
class UploadRaw:
    """Upload the local raw Parquet dataset to Hugging Face."""


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

    command: cappa.Subcommands[Digest | ConvertRaw | UploadRaw | GenMetadata | Preview]


if __name__ == "__main__":
    args = cappa.parse(Args, completion=False)
    match args.command:
        case Digest():
            print(get_tex_typ_map_digest())  # noqa: T201
        case ConvertRaw(datasets=datasets):
            convert_raw_dataset(dataset_names=list(dict.fromkeys(datasets)))
        case UploadRaw():
            upload_raw_dataset()
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
