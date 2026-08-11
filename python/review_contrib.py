"""Contribution review entry script."""

from dataclasses import dataclass

import cappa
from detypify.tools.review_contrib import main as review_contrib


@cappa.command(name="review-contrib")
@dataclass
class Args:
    """Review and collect contributed symbol samples."""


if __name__ == "__main__":
    args = cappa.parse(Args, completion=False)
    review_contrib()
