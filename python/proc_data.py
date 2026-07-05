"""Data processing entry script."""

from typing import Annotated

import typer
from detypify.config import DataSetName
from detypify.data.datasets import create_raw_dataset, get_dataset_classes
from detypify.data.metadata import generate_data_info
from detypify.data.symbols import get_tex_typ_map_digest

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    *,
    datasets: Annotated[
        list[DataSetName] | None,
        typer.Option("--datasets", "-d", help="Datasets to process."),
    ] = None,
    upload_raw: Annotated[
        bool,
        typer.Option("--upload-raw", help="Convert local raw files and upload the raw LaTeX-annotated dataset."),
    ] = False,
    gen_metadata: Annotated[
        bool,
        typer.Option("--gen-metadata", help="Generate frontend metadata and unmapped-symbol review data."),
    ] = False,
    print_tex_typ_map_digest: Annotated[
        bool,
        typer.Option("--print-tex-typ-map-digest", help="Print the effective LaTeX-to-Typst mapping digest and exit."),
    ] = False,
    preview_dataset: Annotated[
        bool,
        typer.Option("--preview-dataset", help="Serve a local browser for mapped dataset samples."),
    ] = False,
    preview_host: Annotated[str, typer.Option("--preview-host", help="Host for --preview-dataset.")] = "127.0.0.1",
    preview_port: Annotated[int, typer.Option("--preview-port", help="Port for --preview-dataset.")] = 8000,
    preview_image_size: Annotated[
        int,
        typer.Option("--preview-image-size", help="Rendered sample size for --preview-dataset."),
    ] = 224,
    preview_page_size: Annotated[
        int,
        typer.Option("--preview-page-size", help="Default samples per page for --preview-dataset."),
    ] = 120,
    num_proc: Annotated[int | None, typer.Option("--num-proc", help="Dataset mapping worker count.")] = 1,
):
    """Preprocess raw datasets and generate metadata."""
    if print_tex_typ_map_digest:
        from json import dumps

        typer.echo(dumps({"tex_typ_map_digest": get_tex_typ_map_digest()}, separators=(",", ":")))
        raise typer.Exit

    dataset_names = list(dict.fromkeys(datasets or [DataSetName.detexify, DataSetName.mathwriting]))

    if upload_raw:
        create_raw_dataset(dataset_names=dataset_names)

    if gen_metadata:
        generate_data_info(classes=get_dataset_classes(dataset_names))

    if preview_dataset:
        from detypify.tools.dataset_preview import serve_dataset_preview

        serve_dataset_preview(
            dataset_names=tuple(dataset_names),
            host=preview_host,
            port=preview_port,
            image_size=preview_image_size,
            page_size=preview_page_size,
            num_proc=num_proc,
        )


if __name__ == "__main__":
    app()
