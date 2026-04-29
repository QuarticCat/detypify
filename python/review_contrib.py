"""Contribution review entry script."""

import typer
from detypify.tools.review_contrib import main as review_contrib

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main():
    """Review and collect contributed symbol samples."""
    review_contrib()


if __name__ == "__main__":
    app()
