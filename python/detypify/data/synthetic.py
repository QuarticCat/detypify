"""Deterministic hybrid synthetic data generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import blake2b
from json import dumps, loads
from math import ceil
from os import getpid
from shutil import rmtree
from typing import TYPE_CHECKING, Any, cast

from detypify.data.datasets import (
    DETERMINISTIC_SPLIT_SEED,
    get_mapped_dataset_splits,
    render_mapped_dataset,
)
from detypify.data.paths import DEFAULT_DATA_PATHS, DataPaths
from detypify.data.rendering import rasterize_strokes
from detypify.data.symbols import get_tex_typ_map_digest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import numpy as np
    from datasets import Dataset, DatasetDict
    from detypify.config import DataSetName
    from detypify.types import Strokes
    from PIL.ImageFont import FreeTypeFont

SYNTHETIC_GENERATOR_VERSION = "hybrid-v1"
MAX_GENERATION_ATTEMPTS = 64
IMAGE_DIMENSIONS = 2
INK_THRESHOLD = 8
_SOURCE_DIGEST_CACHE: dict[str, str] = {}


@dataclass(frozen=True)
class SyntheticSettings:
    image_size: int = 224
    samples_per_class: int = 500
    seed: int = DETERMINISTIC_SPLIT_SEED
    glyph_fraction: float = 0.5

    def validate(self) -> None:
        if self.image_size < 16:  # noqa: PLR2004
            msg = "Synthetic image size must be at least 16 pixels."
            raise ValueError(msg)
        if self.samples_per_class < 1:
            msg = "Synthetic samples per class must be positive."
            raise ValueError(msg)
        if not 0 <= self.glyph_fraction <= 1:
            msg = "Synthetic glyph fraction must be between 0 and 1."
            raise ValueError(msg)


def _file_digest(path: Path) -> str:
    digest = blake2b()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_synthetic_font(font_path: Path) -> set[int]:
    """Validate the configured font and return its Unicode character map."""
    if not font_path.is_file():
        msg = (
            f"Synthetic font does not exist: {font_path}. "
            "Install New Computer Modern Math at external/fonts/NewCMMath-Regular.otf "
            "or pass --synthetic-font."
        )
        raise FileNotFoundError(msg)

    from fontTools.ttLib import TTFont

    try:
        with TTFont(font_path, lazy=True) as font:
            cmap = font.getBestCmap()
    except Exception as error:
        msg = f"Unable to read synthetic font {font_path}: {error}"
        raise ValueError(msg) from error
    if not cmap:
        msg = f"Synthetic font has no Unicode character map: {font_path}"
        raise ValueError(msg)
    return set(cmap)


def synthetic_dataset_fingerprint(
    *,
    source_fingerprint: str,
    settings: SyntheticSettings,
    font_digest: str,
    mapping_digest: str,
) -> str:
    """Return a stable content fingerprint for a synthetic dataset."""
    payload = {
        "font_digest": font_digest,
        "generator_version": SYNTHETIC_GENERATOR_VERSION,
        "mapping_digest": mapping_digest,
        "settings": asdict(settings),
        "source_fingerprint": source_fingerprint,
        "split_seed": DETERMINISTIC_SPLIT_SEED,
    }
    return blake2b(dumps(payload, separators=(",", ":"), sort_keys=True).encode(), digest_size=20).hexdigest()


def _sample_seed(global_seed: int, label: str, branch: str, index: int, attempt: int) -> int:
    payload = f"{global_seed}\0{label}\0{branch}\0{index}\0{attempt}".encode()
    return int.from_bytes(blake2b(payload, digest_size=8).digest(), "little")


def _stroke_fingerprint(strokes: list) -> str:
    return blake2b(dumps(strokes, separators=(",", ":")).encode(), digest_size=12).hexdigest()


def _fit_to_canvas(image: np.ndarray, output_size: int, padding_ratio: float = 0.14) -> np.ndarray:
    import cv2
    import numpy as np

    mask = image > 0
    if not mask.any():
        return np.zeros((output_size, output_size), dtype=np.uint8)
    ys, xs = np.nonzero(mask)
    cropped = image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    target = max(1, round(output_size * (1 - 2 * padding_ratio)))
    scale = target / max(cropped.shape)
    width = max(1, round(cropped.shape[1] * scale))
    height = max(1, round(cropped.shape[0] * scale))
    resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((output_size, output_size), dtype=np.uint8)
    x = (output_size - width) // 2
    y = (output_size - height) // 2
    canvas[y : y + height, x : x + width] = resized
    return canvas


def _render_glyph_template(label: str, font: FreeTypeFont, output_size: int) -> np.ndarray:
    import numpy as np
    from PIL import Image, ImageDraw

    bbox = font.getbbox(label)
    if bbox is None:
        return np.zeros((output_size, output_size), dtype=np.uint8)
    width = max(1, ceil(bbox[2] - bbox[0]))
    height = max(1, ceil(bbox[3] - bbox[1]))
    margin = max(4, output_size // 8)
    image = Image.new("L", (width + 2 * margin, height + 2 * margin), color=0)
    draw = ImageDraw.Draw(image)
    draw.text((margin - bbox[0], margin - bbox[1]), label, fill=255, font=font)
    return _fit_to_canvas(np.asarray(image, dtype=np.uint8), output_size)


def _deform_strokes(strokes: Strokes, rng: np.random.Generator) -> Strokes:
    import numpy as np

    arrays = [np.asarray(stroke, dtype=np.float32) for stroke in strokes if stroke]
    if not arrays:
        return []
    points = np.vstack(arrays)
    center = points.mean(axis=0)
    extent = max(float(np.ptp(points, axis=0).max()), 1.0)
    angle = float(rng.uniform(-0.16, 0.16))
    shear = float(rng.uniform(-0.16, 0.16))
    scale_x, scale_y = rng.uniform(0.86, 1.14, size=2)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float32,
    )
    affine = rotation @ np.array([[scale_x, shear], [0.0, scale_y]], dtype=np.float32)

    deformed: Strokes = []
    for array in arrays:
        normalized = (array - center) @ affine.T
        jitter = rng.normal(0, extent * 0.012, size=normalized.shape)
        if len(array) > 2:  # noqa: PLR2004
            jitter[1:-1] = (jitter[:-2] + 2 * jitter[1:-1] + jitter[2:]) / 4
        result = normalized + jitter + center
        deformed.append([(float(x), float(y)) for x, y in result])
    return deformed


def _distort_raster(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    import cv2
    import numpy as np

    size = image.shape[0]
    angle = float(rng.uniform(-8, 8))
    scale = float(rng.uniform(0.9, 1.08))
    tx, ty = rng.uniform(-0.045 * size, 0.045 * size, size=2)
    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, scale)
    matrix[:, 2] += [tx, ty]
    distorted = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderValue=0)

    noise_x = rng.normal(size=(size, size)).astype(np.float32)
    noise_y = rng.normal(size=(size, size)).astype(np.float32)
    sigma = max(1.0, size * 0.035)
    displacement = size * 0.012
    dx = cv2.GaussianBlur(noise_x, (0, 0), sigmaX=sigma)
    dy = cv2.GaussianBlur(noise_y, (0, 0), sigmaX=sigma)
    dx *= displacement / max(float(np.max(np.abs(dx))), 1e-6)
    dy *= displacement / max(float(np.max(np.abs(dy))), 1e-6)
    grid_x, grid_y = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32))
    distorted = cv2.remap(distorted, grid_x + dx, grid_y + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    morphology = int(rng.integers(-1, 2))
    if morphology:
        kernel_size = 2 if size < 128 else 3  # noqa: PLR2004
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        operation = cv2.dilate if morphology > 0 else cv2.erode
        distorted = operation(distorted, kernel, iterations=1)
    if rng.random() < 0.35:  # noqa: PLR2004
        distorted = cv2.GaussianBlur(distorted, (3, 3), sigmaX=float(rng.uniform(0.25, 0.8)))
    return np.asarray(distorted, dtype=np.uint8)


def is_valid_synthetic_image(image: np.ndarray) -> bool:
    """Reject empty, degenerate, or visibly clipped generated images."""
    import numpy as np

    if image.ndim != IMAGE_DIMENSIONS or image.shape[0] != image.shape[1] or image.dtype != np.uint8:
        return False
    mask = image > INK_THRESHOLD
    if int(mask.sum()) < max(4, image.size // 10000):
        return False
    ys, xs = np.nonzero(mask)
    if max(int(xs.max() - xs.min()), int(ys.max() - ys.min())) < image.shape[0] * 0.15:
        return False
    return bool(xs.min() > 0 and ys.min() > 0 and xs.max() < image.shape[1] - 1 and ys.max() < image.shape[0] - 1)


def _generate_class_records(
    *,
    label: str,
    seed_strokes: list[Strokes],
    settings: SyntheticSettings,
    font: FreeTypeFont | None,
    glyph_supported: bool,
) -> Iterator[dict[str, Any]]:
    import numpy as np

    if not seed_strokes:
        msg = f"No real training strokes are available for class {label!r}."
        raise ValueError(msg)

    glyph_template = None
    if glyph_supported:
        if font is None:
            msg = "Glyph generation requested without a loaded font."
            raise RuntimeError(msg)
        candidate = _render_glyph_template(label, font, settings.image_size)
        if is_valid_synthetic_image(candidate):
            glyph_template = candidate
        else:
            glyph_supported = False

    glyph_count = round(settings.samples_per_class * settings.glyph_fraction) if glyph_supported else 0
    stroke_count = settings.samples_per_class - glyph_count
    branches = [("stroke", index) for index in range(stroke_count)]
    branches.extend(("glyph", index) for index in range(glyph_count))
    seen_images: set[bytes] = set()

    for branch, index in branches:
        for attempt in range(MAX_GENERATION_ATTEMPTS):
            sample_seed = _sample_seed(settings.seed, label, branch, index, attempt)
            rng = np.random.default_rng(sample_seed)
            seed_fingerprint = ""
            if branch == "glyph":
                if glyph_template is None:
                    msg = "Glyph generation requested without a valid glyph template."
                    raise RuntimeError(msg)
                image = glyph_template.copy()
            else:
                seed_stroke = seed_strokes[int(rng.integers(0, len(seed_strokes)))]
                seed_fingerprint = _stroke_fingerprint(seed_stroke)
                deformed = _deform_strokes(seed_stroke, rng)
                image = rasterize_strokes(
                    deformed,
                    settings.image_size,
                    padding_ratio=0.13,
                    thickness_scale=float(rng.uniform(0.72, 1.32)),
                )
            image = _distort_raster(image, rng)
            image_digest = blake2b(image.tobytes(), digest_size=16).digest()
            if is_valid_synthetic_image(image) and image_digest not in seen_images:
                seen_images.add(image_digest)
                yield {
                    "image": image,
                    "label": label,
                    "synthetic_source": branch,
                    "seed_fingerprint": seed_fingerprint,
                }
                break
        else:
            msg = (
                f"Unable to create a unique valid {branch} sample for {label!r} "
                f"after {MAX_GENERATION_ATTEMPTS} attempts."
            )
            raise RuntimeError(msg)


def _synthetic_records(
    *,
    classes: list[str],
    seed_strokes_by_class: list[list[Strokes]],
    settings_values: dict[str, Any],
    font_path: str,
    supported_codepoints: set[int],
) -> Iterator[dict[str, Any]]:
    from PIL import ImageFont

    settings = SyntheticSettings(**settings_values)
    font = ImageFont.truetype(font_path, size=max(16, round(settings.image_size * 1.2)))
    for label, seed_strokes in zip(classes, seed_strokes_by_class, strict=True):
        yield from _generate_class_records(
            label=label,
            seed_strokes=seed_strokes,
            settings=settings,
            font=font,
            glyph_supported=all(ord(character) in supported_codepoints for character in label),
        )


def _cache_is_complete(cache_path: Path, fingerprint: str) -> bool:
    metadata_path = cache_path / "synthetic_metadata.json"
    if not metadata_path.is_file() or not (cache_path / "dataset_info.json").is_file():
        return False
    try:
        metadata = loads(metadata_path.read_text())
    except (OSError, ValueError):
        return False
    return metadata.get("fingerprint") == fingerprint


def _source_records_and_digest(
    vector_train: Dataset,
    classes: list[str],
) -> tuple[str, list[dict[str, Any]] | None]:
    runtime_fingerprint = str(getattr(vector_train, "_fingerprint", ""))
    cache_key = f"{runtime_fingerprint}\0{dumps(classes, ensure_ascii=False, separators=(',', ':'))}"
    if cache_key in _SOURCE_DIGEST_CACHE:
        return _SOURCE_DIGEST_CACHE[cache_key], None

    seed_dataset = cast("Dataset", vector_train.select_columns(["label", "strokes"]))
    records = cast("list[dict[str, Any]]", seed_dataset.to_list())
    source_digest = blake2b()
    source_digest.update(dumps(classes, ensure_ascii=False, separators=(",", ":")).encode())
    for row in records:
        source_digest.update(int(row["label"]).to_bytes(4, "little"))
        source_digest.update(dumps(row["strokes"], separators=(",", ":")).encode())
    digest = source_digest.hexdigest()
    _SOURCE_DIGEST_CACHE[cache_key] = digest
    return digest, records


def build_or_load_synthetic_training_dataset(
    vector_train: Dataset,
    classes: list[str],
    settings: SyntheticSettings,
    font_path: Path,
    *,
    paths: DataPaths = DEFAULT_DATA_PATHS,
) -> tuple[Dataset, str]:
    """Materialize or load the fingerprinted synthetic training cache."""
    import logging

    from datasets import Array2D, ClassLabel, Dataset, Features, Value, load_from_disk

    settings.validate()
    supported_codepoints = validate_synthetic_font(font_path)
    source_fingerprint, source_records = _source_records_and_digest(vector_train, classes)
    fingerprint = synthetic_dataset_fingerprint(
        source_fingerprint=source_fingerprint,
        settings=settings,
        font_digest=_file_digest(font_path),
        mapping_digest=get_tex_typ_map_digest(),
    )
    cache_path = paths.synthetic_datasets_dir / fingerprint
    if _cache_is_complete(cache_path, fingerprint):
        return cast("Dataset", load_from_disk(cache_path)), fingerprint

    if source_records is None:
        seed_dataset = cast("Dataset", vector_train.select_columns(["label", "strokes"]))
        source_records = cast("list[dict[str, Any]]", seed_dataset.to_list())
    seed_strokes_by_class: list[list[Strokes]] = [[] for _ in classes]
    for row in source_records:
        seed_strokes_by_class[int(row["label"])].append(row["strokes"])

    features = Features(
        {
            "image": Array2D(shape=(settings.image_size, settings.image_size), dtype="uint8"),
            "label": ClassLabel(names=classes),
            "synthetic_source": Value("string"),
            "seed_fingerprint": Value("string"),
        }
    )
    logger = logging.getLogger(__name__)
    logger.info(
        "Generating %s synthetic samples (%s per class) into %s",
        len(classes) * settings.samples_per_class,
        settings.samples_per_class,
        cache_path,
    )
    generated = Dataset.from_generator(
        _synthetic_records,
        features=features,
        cache_dir=str(paths.datasets_cache_dir),
        gen_kwargs={
            "classes": classes,
            "seed_strokes_by_class": seed_strokes_by_class,
            "settings_values": asdict(settings),
            "font_path": str(font_path),
            "supported_codepoints": supported_codepoints,
        },
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.parent / f".{fingerprint}.tmp-{getpid()}"
    if temp_path.exists():
        rmtree(temp_path)
    generated.save_to_disk(temp_path)
    metadata = {
        "fingerprint": fingerprint,
        "font": str(font_path),
        "generator_version": SYNTHETIC_GENERATOR_VERSION,
        "num_classes": len(classes),
        "num_samples": len(generated),
        "settings": asdict(settings),
    }
    (temp_path / "synthetic_metadata.json").write_text(
        dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    if cache_path.exists():
        if _cache_is_complete(cache_path, fingerprint):
            rmtree(temp_path)
        else:
            rmtree(cache_path)
            temp_path.replace(cache_path)
    else:
        temp_path.replace(cache_path)
    return cast("Dataset", load_from_disk(cache_path)), fingerprint


def get_synthetic_dataset_splits(
    dataset_names: tuple[DataSetName, ...],
    settings: SyntheticSettings,
    font_path: Path,
    *,
    num_proc: int | None = None,
    paths: DataPaths = DEFAULT_DATA_PATHS,
    max_samples: int | None = None,
) -> tuple[DatasetDict, list[str], str]:
    """Return synthetic train and untouched real validation/test splits."""
    from datasets import DatasetDict

    validate_synthetic_font(font_path)
    vector_splits, classes = get_mapped_dataset_splits(
        dataset_names,
        num_proc=num_proc,
        paths=paths,
        max_samples=max_samples,
    )
    synthetic_train, fingerprint = build_or_load_synthetic_training_dataset(
        vector_splits["train"],
        classes,
        settings,
        font_path,
        paths=paths,
    )
    val = render_mapped_dataset(
        vector_splits["val"],
        classes,
        settings.image_size,
        num_proc=num_proc,
        description=f"Rasterizing real validation at {settings.image_size}px",
    )
    test = render_mapped_dataset(
        vector_splits["test"],
        classes,
        settings.image_size,
        num_proc=num_proc,
        description=f"Rasterizing real test at {settings.image_size}px",
    )
    return DatasetDict({"train": synthetic_train, "val": val, "test": test}).with_format("torch"), classes, fingerprint


def prepare_synthetic_training_dataset(
    dataset_names: tuple[DataSetName, ...],
    settings: SyntheticSettings,
    font_path: Path,
    *,
    num_proc: int | None = None,
    paths: DataPaths = DEFAULT_DATA_PATHS,
    max_samples: int | None = None,
) -> tuple[Dataset, list[str], str]:
    """Generate or reuse the synthetic training cache without rendering evaluation splits."""
    validate_synthetic_font(font_path)
    vector_splits, classes = get_mapped_dataset_splits(
        dataset_names,
        num_proc=num_proc,
        paths=paths,
        max_samples=max_samples,
    )
    dataset, fingerprint = build_or_load_synthetic_training_dataset(
        vector_splits["train"],
        classes,
        settings,
        font_path,
        paths=paths,
    )
    return dataset, classes, fingerprint
