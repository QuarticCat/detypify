"""Data processing entry script."""

from dataclasses import dataclass, field
from os import process_cpu_count
from typing import Annotated

import cappa
from detypify.config import DataSetName
from detypify.data.datasets import create_raw_dataset, get_dataset_classes
from detypify.data.metadata import generate_data_info
from detypify.data.symbols import get_tex_typ_map_digest


@cappa.command(name="proc-data", default_long=True)
@dataclass
class Args:
    """Preprocess raw datasets and generate metadata."""

    datasets: Annotated[list[DataSetName], cappa.Arg(short="-d")] = field(
        default_factory=lambda: [DataSetName.detexify, DataSetName.mathwriting]
    )
    """Datasets to process."""

    upload_raw: bool = False
    """Convert local raw files and upload the raw LaTeX-annotated dataset."""

    gen_metadata: bool = False
    """Generate frontend metadata and unmapped-symbol review data."""

    print_tex_typ_map_digest: bool = False
    """Print the effective LaTeX-to-Typst mapping digest and exit."""

    preview_dataset: bool = False
    """Serve a local browser for mapped dataset samples."""

    preview_host: str = "127.0.0.1"
    """Host for --preview-dataset."""

    preview_port: int = 8000
    """Port for --preview-dataset."""

    preview_image_size: int = 224
    """Rendered sample size for --preview-dataset."""

    preview_page_size: int = 120
    """Default samples per page for --preview-dataset."""

    num_workers: int = process_cpu_count() or 1
    """Dataset mapping worker count."""


if __name__ == "__main__":
    args = cappa.parse(Args, completion=False)
    if args.print_tex_typ_map_digest:
        from json import dumps

        print(dumps({"tex_typ_map_digest": get_tex_typ_map_digest()}, separators=(",", ":")))  # noqa: T201
        raise SystemExit

    dataset_names = list(dict.fromkeys(args.datasets))

    if args.upload_raw:
        create_raw_dataset(dataset_names=dataset_names)

    if args.gen_metadata:
        generate_data_info(classes=get_dataset_classes(dataset_names))

    if args.preview_dataset:
        from detypify.tools.dataset_preview import serve_dataset_preview

        serve_dataset_preview(
            dataset_names=tuple(dataset_names),
            host=args.preview_host,
            port=args.preview_port,
            image_size=args.preview_image_size,
            page_size=args.preview_page_size,
            num_proc=args.num_workers,
        )
