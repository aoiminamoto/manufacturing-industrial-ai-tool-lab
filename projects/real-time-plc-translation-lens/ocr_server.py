import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

from ocr_coordinates import build_regions


BASE_DIR = Path(__file__).resolve().parent
MAX_FRAME_BYTES = 12 * 1024 * 1024
OCR_LOCK = threading.Lock()

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("OMP_NUM_THREADS", "4")

OCR = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="cpu",
)


def recognize(raw: bytes) -> dict:
    with Image.open(io.BytesIO(raw)) as image:
        width, height = image.size
        image_array = np.asarray(image.convert("RGB"))
    with OCR_LOCK:
        pages = list(OCR.predict(image_array))
    if not pages:
        return {"width": width, "height": height, "regions": []}

    payload = pages[0].json
    data = payload.get("res", payload)
    texts = data.get("rec_texts", [])
    scores = data.get("rec_scores", [])
    boxes = data.get("rec_boxes", [])
    regions = build_regions(texts, scores, boxes, width, height)
    return {"width": width, "height": height, "regions": regions}


class Handler(BaseHTTPRequestHandler):
    server_version = "PLC-Lens-OCR/1.0"

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ok", "engine": "PaddleOCR", "coordinates": "pixel"})
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/ocr":
            self._json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_FRAME_BYTES:
            self._json(400, {"error": "Invalid frame size"})
            return
        raw = self.rfile.read(length)
        try:
            self._json(200, recognize(raw))
        except Exception as exc:
            self._json(500, {"error": f"Local OCR failed: {type(exc).__name__}: {exc}"})

    def log_message(self, format: str, *args) -> None:
        # Do not log recognized text or request bodies.
        print(f"[Local OCR] {self.address_string()} {format % args}", flush=True)


if __name__ == "__main__":
    print("[Local OCR] Ready on http://127.0.0.1:8506", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8506), Handler).serve_forever()
