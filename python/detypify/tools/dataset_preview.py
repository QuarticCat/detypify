"""Local browser for inspecting mapped dataset samples."""

from __future__ import annotations

import base64
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, quote_plus, urlparse

import cv2
import numpy as np
from detypify.data.datasets import map_raw_dataset
from detypify.data.rendering import rasterize_strokes

if TYPE_CHECKING:
    from collections.abc import Sequence

    from detypify.config import DataSetName

MAX_PAGE_SIZE = 500
FILTER_CACHE_SIZE = 32


def _int_param(params: dict[str, list[str]], name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(params.get(name, [str(default)])[0])
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _image_data_url(strokes: list, image_size: int) -> str:
    image = np.asarray(rasterize_strokes(strokes, image_size), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        msg = "Failed to encode rendered sample as PNG"
        raise RuntimeError(msg)
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def _page_options(current: int) -> str:
    options = sorted({60, 120, 240, 500, current})
    return "".join(
        f'<option value="{size}" {"selected" if size == current else ""}>{size}</option>' for size in options
    )


class DatasetPreviewServer:
    def __init__(
        self,
        dataset_names: Sequence[DataSetName],
        image_size: int,
        default_page_size: int,
        num_proc: int,
    ) -> None:
        self.dataset, _ = map_raw_dataset(dataset_names, num_proc=num_proc)
        self.image_size = image_size
        self.default_page_size = default_page_size
        self.classes = sorted(self.dataset.unique("label"))
        self.class_to_index = {label: index for index, label in enumerate(self.classes)}
        self.search_labels = [str(label).lower() for label in self.dataset["label"]]
        self.search_sources = [str(source).lower() for source in self.dataset["source"]]
        self.filter_cache: dict[str, tuple[int, ...]] = {}

    def filtered_indices(self, query: str) -> tuple[int, ...]:
        query = query.strip().lower()
        if query in self.filter_cache:
            return self.filter_cache[query]

        if not query:
            matches = tuple(range(len(self.dataset)))
        elif query.isdigit():
            index = int(query)
            matches = (index,) if 0 <= index < len(self.dataset) else ()
        else:
            matches_list: list[int] = []
            for index, (label, source) in enumerate(zip(self.search_labels, self.search_sources, strict=True)):
                if query in label or query in source:
                    matches_list.append(index)
            matches = tuple(matches_list)

        if len(self.filter_cache) >= FILTER_CACHE_SIZE:
            self.filter_cache.pop(next(iter(self.filter_cache)))
        self.filter_cache[query] = matches
        return matches

    def page_html(self, page: int, page_size: int, query: str) -> str:
        query = query.strip()
        filtered_indices = self.filtered_indices(query)
        total = len(filtered_indices)
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = max(1, min(total_pages, page))
        start = (page - 1) * page_size
        end = min(total, start + page_size)
        page_indices = filtered_indices[start:end]
        rows = self.dataset.select(page_indices) if page_indices else []

        cards = []
        for sample_index, row in zip(page_indices, rows, strict=True):
            label = str(row["label"])
            source = str(row["source"])
            label_index = self.class_to_index[label]
            data_url = _image_data_url(row["strokes"], self.image_size)
            cards.append(
                f"""
                <article class="card">
                  <img src="{data_url}" alt="{html.escape(label)}" loading="lazy" />
                  <div class="meta">
                    <span>{html.escape(source)} #{sample_index}</span>
                    <strong>{html.escape(label)}</strong>
                    <code>class {label_index}</code>
                  </div>
                </article>
                """
            )

        prev_page = max(1, page - 1)
        next_page = min(total_pages, page + 1)
        query_arg = f"&q={quote_plus(query)}" if query else ""
        summary = f"{len(self.dataset):,} samples, {len(self.classes):,} classes"
        if query:
            summary += f", {total:,} matches"
        elif total:
            summary += f", showing {start + 1:,}-{end:,}"

        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Detypify Dataset Browser</title>
<style>
  body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; background: #f7f7f4; color: #181a1f; }}
  header {{ position: sticky; top: 0; background: rgba(247,247,244,.94); backdrop-filter: blur(10px); border-bottom: 1px solid #ddd8cc; padding: 14px 20px; z-index: 2; }}
  h1 {{ margin: 0 0 8px; font-size: 20px; }}
  .bar {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; color: #555b55; font-size: 14px; }}
  a, button {{ color: #111827; background: #ffffff; border: 1px solid #cfc9ba; border-radius: 6px; padding: 6px 10px; text-decoration: none; font: inherit; }}
  input, select {{ border: 1px solid #cfc9ba; border-radius: 6px; padding: 6px 8px; font: inherit; background: #ffffff; }}
  input[name="q"] {{ width: 180px; }}
  main {{ padding: 20px; display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 14px; }}
  .card {{ background: white; border: 1px solid #dedbd2; border-radius: 8px; overflow: hidden; }}
  img {{ display: block; width: 100%; aspect-ratio: 1; object-fit: contain; image-rendering: pixelated; background: #fff; }}
  .meta {{ border-top: 1px solid #ece8df; padding: 8px 10px; display: grid; gap: 3px; }}
  .meta span {{ color: #6f756f; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
  .meta strong {{ font-size: 18px; line-height: 1.1; min-height: 22px; }}
  .meta code {{ color: #5a5f68; font-size: 12px; }}
</style>
</head>
<body>
<header>
  <h1>Detypify Dataset Browser</h1>
  <form class="bar" method="get">
    <span>{summary}</span>
    <label>Search <input name="q" value="{html.escape(query)}" placeholder="label, source, or index" /></label>
    <a href="/?page={prev_page}&page_size={page_size}{query_arg}">Previous</a>
    <label>Page <input name="page" value="{page}" size="6" /> / {total_pages:,}</label>
    <label>Per page <select name="page_size">{_page_options(page_size)}</select></label>
    <button type="submit">Go</button>
    <a href="/?page={next_page}&page_size={page_size}{query_arg}">Next</a>
    <a href="/?page_size={page_size}">Clear</a>
  </form>
</header>
<main>
{"".join(cards)}
</main>
</body>
</html>
"""

    def handler_class(self) -> type[BaseHTTPRequestHandler]:
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path != "/":
                    self.send_error(404)
                    return

                params = parse_qs(parsed.query)
                page_size = _int_param(params, "page_size", preview.default_page_size, 1, MAX_PAGE_SIZE)
                page = _int_param(params, "page", 1, 1, 10**9)
                query = params.get("q", [""])[0]
                body = preview.page_html(page, page_size, query).encode("utf-8")

                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler


def serve_dataset_preview(
    dataset_names: Sequence[DataSetName],
    host: str,
    port: int,
    image_size: int,
    page_size: int,
    num_proc: int,
) -> None:
    preview = DatasetPreviewServer(
        dataset_names=dataset_names,
        image_size=image_size,
        default_page_size=page_size,
        num_proc=num_proc,
    )
    server = ThreadingHTTPServer((host, port), preview.handler_class())
    print(f"Serving dataset preview at http://{host}:{port}")  # noqa: T201
    server.serve_forever()
