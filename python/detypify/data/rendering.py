from detypify.types import Strokes


def rasterize_strokes(strokes: Strokes, output_size: int):
    """Normalize vector strokes and rasterize them into a uint8 image."""
    import cv2
    import numpy as np

    if not strokes:
        return np.zeros((output_size, output_size), dtype=np.uint8)

    stroke_arrays = [np.array(s, dtype=np.float32) for s in strokes if s]
    if not stroke_arrays:
        return np.zeros((output_size, output_size), dtype=np.uint8)

    all_points = np.vstack(stroke_arrays)
    min_x, min_y = all_points.min(axis=0)
    max_x, max_y = all_points.max(axis=0)
    padding = max(1, round(output_size * 0.04))
    target_size = max(1, output_size - (2 * padding))

    width = max(max_x - min_x, max_y - min_y)
    scale = target_size / width if width > 1e-6 else 1.0  # noqa: PLR2004
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    all_points = ((all_points - [center_x, center_y]) * scale) + [output_size / 2, output_size / 2]
    all_points = np.rint(all_points)

    lengths = [len(a) for a in stroke_arrays]
    split_indices = np.cumsum(lengths)[:-1]
    normalized_strokes = np.split(all_points.astype(np.int32), split_indices)
    normalized_strokes = [stroke for stroke in normalized_strokes if len(stroke) > 1]

    canvas = np.zeros((output_size, output_size), dtype=np.uint8)
    if normalized_strokes:
        thickness = max(1, output_size // 25)
        cv2.polylines(canvas, normalized_strokes, isClosed=False, color=255, thickness=thickness, lineType=cv2.LINE_AA)
    return canvas
