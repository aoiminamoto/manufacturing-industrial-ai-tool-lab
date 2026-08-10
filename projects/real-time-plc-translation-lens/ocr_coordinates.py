import re


JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaffｦ-ﾟ]")


def build_regions(texts, scores, boxes, width: int, height: int, minimum_confidence: float = 0.35) -> list[dict]:
    """Convert OCR pixel boxes into validated, normalized Japanese regions."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    regions = []
    for text, score, box in zip(texts, scores, boxes):
        text = " ".join(str(text).split())
        confidence = float(score)
        if not text or confidence < minimum_confidence or not JAPANESE_RE.search(text):
            continue
        left, top, right, bottom = [int(value) for value in box]
        if right <= left or bottom <= top:
            continue
        regions.append(
            {
                "id": 0,
                "jp": text,
                "draft_en": "",
                "bbox": [
                    max(0, min(1000, round(left * 1000 / width))),
                    max(0, min(1000, round(top * 1000 / height))),
                    max(0, min(1000, round(right * 1000 / width))),
                    max(0, min(1000, round(bottom * 1000 / height))),
                ],
                "confidence": confidence,
            }
        )
    regions.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    for index, region in enumerate(regions, 1):
        region["id"] = index
    return regions
