"""Data processing entry script."""

import typer
from detypify.config import DataSetName
from detypify.data.datasets import create_raw_dataset
from detypify.data.metadata import generate_data_info
from detypify.data.symbols import get_tex_typ_map, get_tex_typ_map_digest

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    datasets: list[DataSetName] = typer.Option(  # noqa: B008
        [DataSetName.detexify, DataSetName.mathwriting],
        "--datasets",
        "-d",
        help="Datasets to process when uploading raw data.",
    ),
    upload_raw: bool = typer.Option(
        False,
        "--upload-raw",
        help="Convert local raw files and upload the raw LaTeX-annotated dataset to Hugging Face.",
    ),
    gen_metadata: bool = typer.Option(
        False,
        "--gen-metadata",
        help="Generate frontend infer/contrib metadata and unmapped-symbol review data.",
    ),
    print_tex_typ_map_digest: bool = typer.Option(
        False,
        "--print-tex-typ-map-digest",
        help="Print the digest for the effective LaTeX-to-Typst mapping and exit.",
    ),
):
    """Preprocess raw datasets and generate metadata."""
    if print_tex_typ_map_digest:
        from json import dumps

        typer.echo(dumps({"tex_typ_map_digest": get_tex_typ_map_digest()}, separators=(",", ":")))
        raise typer.Exit

    dataset_names = list(dict.fromkeys(datasets))

    if upload_raw:
        create_raw_dataset(dataset_names=dataset_names)

    if gen_metadata:
        generate_data_info(classes=sorted({v.char for v in get_tex_typ_map().values()}))


if __name__ == "__main__":
    app()
