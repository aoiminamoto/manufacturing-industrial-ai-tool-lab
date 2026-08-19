"""Resumable, page-chunked renderer for very large translated PDFs."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import pymupdf


LOCATION = re.compile(
    r"^pdf:p(?P<page>\d+):b\d+:l\d+:s\d+:"
    r"(?P<x0>-?[\d.]+),(?P<y0>-?[\d.]+),(?P<x1>-?[\d.]+),(?P<y1>-?[\d.]+):"
    r"(?P<size>[\d.]+)$"
)

ENGINEERING_FIELD_TERMS = {
    "寿命時間 [年]": "Service Life [years]",
    "寿命時間[年]": "Service Life [years]",
    "最短使命時間 [年]": "Minimum Mission Time [years]",
    "最短使命時間[年]": "Minimum Mission Time [years]",
    "参照記号": "Reference Designation",
    "技術的分類": "Technical Classification",
    "部品の製造者": "Component Manufacturer",
    "部品の識別子": "Component Identifier",
    "部品のグループ": "Component Group",
    "部品番号": "Part Number",
    "評価者": "Evaluator",
}


def clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def source_text_by_location(document: pymupdf.Document) -> dict[str, str]:
    result: dict[str, str] = {}
    for page_index, page in enumerate(document):
        page_dict = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)
        for block_index, block in enumerate(page_dict.get("blocks", [])):
            if block.get("type") != 0:
                continue
            for line_index, line in enumerate(block.get("lines", [])):
                spans = [span for span in line.get("spans", []) if clean_text(span.get("text"))]
                boxes = [span.get("bbox") for span in spans if len(span.get("bbox") or ()) == 4]
                if not spans or not boxes:
                    continue
                x0 = round(min(float(box[0]) for box in boxes), 3)
                y0 = round(min(float(box[1]) for box in boxes), 3)
                x1 = round(max(float(box[2]) for box in boxes), 3)
                y1 = round(max(float(box[3]) for box in boxes), 3)
                size = round(max(float(span.get("size", 8.0)) for span in spans), 3)
                location = (
                    f"pdf:p{page_index}:b{block_index}:l{line_index}:s0:"
                    f"{x0},{y0},{x1},{y1}:{size}"
                )
                result[location] = clean_text("".join(str(span.get("text", "")) for span in spans))
    return result


def canonical_translation(source: str, translated: str) -> str:
    source = clean_text(source)
    suffix = ":" if source.endswith(":") else ""
    core = source[:-1].strip() if suffix else source
    canonical = ENGINEERING_FIELD_TERMS.get(core)
    if canonical:
        return f"{canonical}{suffix}"
    for japanese, english in ENGINEERING_FIELD_TERMS.items():
        if source.startswith(japanese):
            return f"{english}{source[len(japanese):]}"
    return clean_text(translated)


def dedupe_replacements(replacements):
    accepted = []
    for rect, font_size, translated in sorted(
        replacements, key=lambda item: (item[0].y0, item[0].x0, -item[0].get_area())
    ):
        duplicate = False
        for prior_rect, _, _ in accepted[-12:]:
            intersection = rect & prior_rect
            smaller = min(rect.get_area(), prior_rect.get_area())
            if smaller > 0 and intersection.get_area() / smaller >= 0.82:
                duplicate = True
                break
            if (
                abs(rect.x0 - prior_rect.x0) <= 1.2
                and abs(rect.y0 - prior_rect.y0) <= 1.2
                and abs(rect.x1 - prior_rect.x1) <= 1.2
                and abs(rect.y1 - prior_rect.y1) <= 1.2
            ):
                duplicate = True
                break
        if not duplicate:
            accepted.append((rect, font_size, translated))
    return accepted


def consolidate_row_fragments(replacements):
    items = dedupe_replacements(replacements)
    rows = []
    for item in items:
        rect = item[0]
        for row in rows:
            row_rect = row[-1][0]
            if abs(rect.y0 - row_rect.y0) <= 1.8 and abs(rect.y1 - row_rect.y1) <= 1.8:
                row.append(item)
                break
        else:
            rows.append([item])
    consolidated = []
    for row in rows:
        row.sort(key=lambda item: item[0].x0)
        current = None
        for rect, font_size, translated in row:
            if current is None:
                current = [pymupdf.Rect(rect), font_size, translated]
                continue
            if rect.x0 - current[0].x1 <= 12:
                current[0] |= rect
                current[1] = max(current[1], font_size)
                if clean_text(translated) != clean_text(current[2]):
                    current[2] = f"{current[2]} {translated}".strip()
            else:
                consolidated.append(tuple(current))
                current = [pymupdf.Rect(rect), font_size, translated]
        if current is not None:
            consolidated.append(tuple(current))
    return consolidated


def fitted_size(text: str, rect: pymupdf.Rect, original_size: float) -> float:
    """Estimate a one-pass Helvetica size that fits the original visual box."""
    clean = " ".join(str(text).split())
    if not clean:
        return 1.5
    height_limit = max(1.5, rect.height * 0.72)
    # Helvetica's average Latin glyph is approximately 0.52 em wide.
    width_limit = max(1.5, rect.width / max(len(clean) * 0.52, 1.0))
    return max(4.5, min(float(original_size), height_limit, width_limit))


def available_text_rect(page, source_rect: pymupdf.Rect, page_words=None) -> pymupdf.Rect:
    right_limit = page.rect.width - 8
    for item in (page_words if page_words is not None else page.get_text("words")):
        other = pymupdf.Rect(item[:4])
        if other.x0 <= source_rect.x1 + 0.5:
            continue
        overlap = min(source_rect.y1, other.y1) - max(source_rect.y0, other.y0)
        if overlap >= min(source_rect.height, other.height) * 0.35:
            right_limit = min(right_limit, other.x0 - 2)
    return pymupdf.Rect(
        max(0, source_rect.x0 - 0.4), max(0, source_rect.y0 - 0.4),
        max(source_rect.x1 + 1, right_limit),
        source_rect.y1 + max(1.0, source_rect.height * 0.25),
    )


def render_chunk(
    source: pymupdf.Document,
    replacements: dict[int, list[tuple[pymupdf.Rect, float, str]]],
    start: int,
    end: int,
    destination: Path,
) -> None:
    chunk = pymupdf.open()
    chunk.insert_pdf(source, from_page=start, to_page=end - 1)
    for absolute_page in range(start, end):
        page = chunk[absolute_page - start]
        page_width = page.rect.width
        page_words = page.get_text("words")
        for rect, original_size, translated in consolidate_row_fragments(replacements.get(absolute_page, [])):
            cover = pymupdf.Rect(
                max(0, rect.x0 - 0.4),
                max(0, rect.y0 - 0.4),
                min(page_width, rect.x1 + 1.0),
                rect.y1 + 0.8,
            )
            target = available_text_rect(page, rect, page_words)
            page.draw_rect(cover, color=None, fill=(1, 1, 1), overlay=True)
            font_size = fitted_size(translated, target, original_size)
            remaining = page.insert_textbox(
                target,
                " ".join(str(translated).split()),
                fontname="helv",
                fontsize=font_size,
                color=(0, 0, 0),
                align=pymupdf.TEXT_ALIGN_LEFT,
                overlay=True,
            )
            if remaining < 0:
                page.insert_text(
                    (target.x0, max(target.y0 + 1.5, target.y1 - 0.5)),
                    " ".join(str(translated).split()),
                    fontname="helv",
                    fontsize=4.5,
                    color=(0, 0, 0),
                    overlay=True,
                )
    chunk.save(destination, garbage=1, deflate=True)
    chunk.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--chunk-pages", type=int, default=100)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = args.work_dir / "generation-progress.json"
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    translations = checkpoint.get("translations", checkpoint)

    with pymupdf.open(args.source) as source:
        source_text = source_text_by_location(source)
        replacements: dict[int, list[tuple[pymupdf.Rect, float, str]]] = defaultdict(list)
        for location, translated in translations.items():
            match = LOCATION.match(location)
            if not match:
                continue
            values = match.groupdict()
            replacements[int(values["page"])].append(
                (
                    pymupdf.Rect(
                        float(values["x0"]), float(values["y0"]),
                        float(values["x1"]), float(values["y1"]),
                    ),
                    float(values["size"]),
                    canonical_translation(source_text.get(location, ""), str(translated)),
                )
            )
        total_pages = source.page_count
        metadata = source.metadata or {}
        toc = source.get_toc(simple=False)
        chunks = math.ceil(total_pages / args.chunk_pages)
        started = time.time()
        for chunk_index in range(chunks):
            start = chunk_index * args.chunk_pages
            end = min(total_pages, start + args.chunk_pages)
            chunk_path = args.work_dir / f"chunk-{chunk_index:05d}.pdf"
            if not chunk_path.exists():
                render_chunk(source, replacements, start, end, chunk_path)
            progress = {
                "status": "generating",
                "generated_pages": end,
                "total_pages": total_pages,
                "completed_chunks": chunk_index + 1,
                "total_chunks": chunks,
                "elapsed_seconds": round(time.time() - started, 1),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            print(
                f"Generated pages {end:,}/{total_pages:,} "
                f"({100 * end / total_pages:.1f}%)",
                flush=True,
            )

    final = pymupdf.open()
    for chunk_index in range(chunks):
        chunk_path = args.work_dir / f"chunk-{chunk_index:05d}.pdf"
        with pymupdf.open(chunk_path) as chunk:
            final.insert_pdf(chunk)
    metadata["subject"] = (
        "Term1 glossary-first translation review copy; "
        "verify before controlled engineering use."
    )
    final.set_metadata(metadata)
    if toc:
        try:
            final.set_toc(toc)
        except Exception:
            pass
    final.save(args.output, garbage=1, deflate=True)
    final.close()
    progress_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "generated_pages": total_pages,
                "total_pages": total_pages,
                "completed_chunks": chunks,
                "total_chunks": chunks,
                "output": str(args.output),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Completed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
