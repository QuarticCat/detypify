from pathlib import Path
from typing import cast

import numpy as np
import pytest
from datasets import ClassLabel, Dataset, Features, List, Sequence, Value
from detypify.data.paths import DataPaths
from detypify.data.rendering import rasterize_strokes
from detypify.data.synthetic import (
    SyntheticSettings,
    _generate_class_records,
    _stroke_fingerprint,
    build_or_load_synthetic_training_dataset,
    is_valid_synthetic_image,
    synthetic_dataset_fingerprint,
    validate_synthetic_font,
)
from PIL import ImageFont

SAMPLE_STROKES = [
    [[(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)]],
    [[(0.0, 1.0), (0.5, 0.0), (1.0, 1.0)]],
]


def _test_font_path() -> Path:
    from matplotlib import get_data_path

    return Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"


def test_fingerprint_is_stable_and_invalidates_inputs() -> None:
    settings = SyntheticSettings(image_size=32, samples_per_class=6, seed=42)
    fingerprint = synthetic_dataset_fingerprint(
        source_fingerprint="source-a",
        settings=settings,
        font_digest="font-a",
        mapping_digest="mapping-a",
    )

    assert fingerprint == synthetic_dataset_fingerprint(
        source_fingerprint="source-a",
        settings=settings,
        font_digest="font-a",
        mapping_digest="mapping-a",
    )
    assert fingerprint != synthetic_dataset_fingerprint(
        source_fingerprint="source-b",
        settings=settings,
        font_digest="font-a",
        mapping_digest="mapping-a",
    )
    assert fingerprint != synthetic_dataset_fingerprint(
        source_fingerprint="source-a",
        settings=settings,
        font_digest="font-b",
        mapping_digest="mapping-a",
    )
    assert fingerprint != synthetic_dataset_fingerprint(
        source_fingerprint="source-a",
        settings=SyntheticSettings(image_size=32, samples_per_class=7, seed=42),
        font_digest="font-a",
        mapping_digest="mapping-a",
    )


def test_stroke_generation_is_deterministic_and_uses_only_supplied_seeds() -> None:
    settings = SyntheticSettings(image_size=64, samples_per_class=8, seed=7)
    first = list(
        _generate_class_records(
            label="A",
            seed_strokes=SAMPLE_STROKES,
            settings=settings,
            font=None,
            glyph_supported=False,
        )
    )
    second = list(
        _generate_class_records(
            label="A",
            seed_strokes=SAMPLE_STROKES,
            settings=settings,
            font=None,
            glyph_supported=False,
        )
    )

    assert len(first) == settings.samples_per_class
    assert {record["synthetic_source"] for record in first} == {"stroke"}
    assert [record["seed_fingerprint"] for record in first] == [record["seed_fingerprint"] for record in second]
    assert all(np.array_equal(left["image"], right["image"]) for left, right in zip(first, second, strict=True))
    assert len({record["image"].tobytes() for record in first}) == settings.samples_per_class
    assert all(is_valid_synthetic_image(record["image"]) for record in first)


def test_supported_glyphs_receive_half_of_the_class_quota() -> None:
    settings = SyntheticSettings(image_size=64, samples_per_class=10, seed=11)
    font = ImageFont.truetype(str(_test_font_path()), size=72)
    records = list(
        _generate_class_records(
            label="A",
            seed_strokes=SAMPLE_STROKES,
            settings=settings,
            font=font,
            glyph_supported=True,
        )
    )

    assert len(records) == settings.samples_per_class
    assert sum(record["synthetic_source"] == "glyph" for record in records) == 5
    assert sum(record["synthetic_source"] == "stroke" for record in records) == 5
    assert all(not record["seed_fingerprint"] for record in records if record["synthetic_source"] == "glyph")


def test_image_validation_rejects_blank_degenerate_and_clipped_images() -> None:
    blank = np.zeros((32, 32), dtype=np.uint8)
    degenerate = blank.copy()
    degenerate[15:17, 15:17] = 255
    clipped = blank.copy()
    clipped[:, 0] = 255
    valid = blank.copy()
    valid[8:24, 15:17] = 255

    assert not is_valid_synthetic_image(blank)
    assert not is_valid_synthetic_image(degenerate)
    assert not is_valid_synthetic_image(clipped)
    assert is_valid_synthetic_image(valid)


def test_rasterizer_preserves_single_point_pen_taps() -> None:
    image = rasterize_strokes([[(0.0, 0.0)], [(0.0, 1.0)]], 64, padding_ratio=0.13)

    assert np.count_nonzero(image) > 0
    assert is_valid_synthetic_image(image)


def test_missing_font_error_is_actionable(tmp_path: Path) -> None:
    missing = tmp_path / "missing.otf"

    with pytest.raises(FileNotFoundError, match="--synthetic-font"):
        validate_synthetic_font(missing)


def test_font_coverage_reads_unicode_cmap() -> None:
    assert ord("A") in validate_synthetic_font(_test_font_path())


def test_materialized_cache_has_exact_quotas_and_training_seeds_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import detypify.data.synthetic as synthetic_module

    classes = ["A", "B"]
    train_strokes = [
        [[(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)]],
        [[(0.0, 1.0), (0.5, 0.0), (1.0, 1.0)]],
        [[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]],
        [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]],
    ]
    held_out_strokes = [[[(20.0, 20.0), (30.0, 40.0), (40.0, 20.0)]]]
    features = Features(
        {
            "label": ClassLabel(names=classes),
            "strokes": List(List(Sequence(Value("float32"), length=2))),
        }
    )
    vector_train = Dataset.from_dict(
        {"label": [0, 0, 1, 1], "strokes": train_strokes},
        features=features,
    )
    same_content_different_fingerprint = cast(
        "Dataset",
        vector_train.map(
            lambda row: row,
            new_fingerprint="worker-independent-fingerprint",
        ),
    )
    paths = DataPaths(build_dir=tmp_path / "build", external_dir=tmp_path / "external")
    settings = SyntheticSettings(image_size=32, samples_per_class=4, seed=19)
    monkeypatch.setattr(synthetic_module, "get_tex_typ_map_digest", lambda: "test-mapping")

    first, fingerprint = build_or_load_synthetic_training_dataset(
        vector_train,
        classes,
        settings,
        _test_font_path(),
        paths=paths,
    )
    second, second_fingerprint = build_or_load_synthetic_training_dataset(
        same_content_different_fingerprint,
        classes,
        settings,
        _test_font_path(),
        paths=paths,
    )

    assert fingerprint == second_fingerprint
    assert len(first) == len(second) == len(classes) * settings.samples_per_class
    assert first["label"].count(0) == first["label"].count(1) == settings.samples_per_class
    used_seeds = {value for value in first["seed_fingerprint"] if value}
    expected_seeds = {_stroke_fingerprint(strokes) for strokes in train_strokes}
    held_out_seeds = {_stroke_fingerprint(strokes) for strokes in held_out_strokes}
    assert used_seeds <= expected_seeds
    assert used_seeds.isdisjoint(held_out_seeds)
