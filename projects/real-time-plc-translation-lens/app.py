import asyncio
import base64
import hashlib
import io
import json
import os
import re
import ssl
import threading
import unicodedata
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageOps
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

try:
    import truststore
except ImportError:
    truststore = None


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MAX_FRAME_BYTES = 12 * 1024 * 1024
MAX_IMAGE_EDGE = 1500
JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaffｦ-ﾟ]")
SCAN_SEMAPHORE = asyncio.Semaphore(2)
RESULT_CACHE: OrderedDict[str, dict] = OrderedDict()
RESULT_CACHE_LOCK = threading.Lock()


def clean_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def extract_json_payload(text: str) -> dict:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def glossary_path() -> Path:
    configured = clean_text(os.getenv("PLC_LENS_GLOSSARY_PATH", ""))
    candidates = [
        Path(configured) if configured else None,
        BASE_DIR / "glossary.xlsx",
        BASE_DIR / "glossary.csv",
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    raise FileNotFoundError("Set PLC_LENS_GLOSSARY_PATH or add glossary.xlsx/glossary.csv.")


@lru_cache(maxsize=4)
def load_glossary_cached(path_text: str, modified_ns: int) -> tuple[tuple[str, str], ...]:
    del modified_ns
    path = Path(path_text)
    frame = pd.read_csv(path, dtype=str) if path.suffix.lower() == ".csv" else pd.read_excel(path, dtype=str)
    normalized_columns = {clean_text(column).casefold(): column for column in frame.columns}
    jp_column = normalized_columns.get("jp") or normalized_columns.get("japanese")
    en_column = normalized_columns.get("en") or normalized_columns.get("english")
    if not jp_column or not en_column:
        raise ValueError("Glossary must contain JP and EN columns.")

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    usable = frame[[jp_column, en_column]].dropna(how="all")
    for raw_jp, raw_en in usable.itertuples(index=False, name=None):
        jp = clean_text(raw_jp)
        en = clean_text(raw_en)
        key = (jp, en)
        if not jp or not en or key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return tuple(sorted(pairs, key=lambda item: len(item[0]), reverse=True))


def glossary_pairs() -> tuple[tuple[str, str], ...]:
    path = glossary_path()
    return load_glossary_cached(str(path), path.stat().st_mtime_ns)


@lru_cache(maxsize=1)
def openai_client() -> OpenAI:
    api_key = clean_text(os.getenv("OPENAI_API_KEY", ""))
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Copy .env.example to .env and add your key.")
    base_url = clean_text(os.getenv("OPENAI_BASE_URL", "")) or None
    verify = os.getenv("OPENAI_SSL_VERIFY", "true").strip().lower() not in {"0", "false", "no"}
    verify_value = verify
    if verify and truststore is not None:
        verify_value = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    http_client = httpx.Client(
        verify=verify_value,
        timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "90")),
        trust_env=True,
    )
    return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)


def prepare_frame(raw: bytes) -> tuple[bytes, int, int]:
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    if max(image.size) > MAX_IMAGE_EDGE:
        scale = MAX_IMAGE_EDGE / max(image.size)
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90, optimize=True)
    return output.getvalue(), image.width, image.height


def extract_japanese_regions(raw: bytes, width: int, height: int) -> list[dict]:
    encoded = base64.b64encode(raw).decode("ascii")
    prompt = f"""
You are the OCR layer of a real-time PLC translation lens.

Read every visible Japanese PLC comment or Japanese engineering label in this single camera frame.
Scan systematically from top-left to bottom-right. Small PLC comment rows are important; do not silently skip readable rows.

Return only JSON:
{{
  "regions": [
    {{
      "jp": "Japanese exactly as visible",
      "draft_en": "concise manufacturing English",
      "bbox": [0, 0, 1000, 1000],
      "confidence": 0.95
    }}
  ]
}}

Rules:
1. bbox is [left, top, right, bottom] normalized to 0-1000 relative to the full frame ({width} x {height}).
2. Return one logical PLC comment row or label per region. Make every bbox very tight around the Japanese glyphs only; do not include nearby ladder lines, device addresses, or blank space.
3. Preserve Japanese exactly in jp. Also provide one concise PLC-comment English draft in draft_en.
4. Include Japanese mixed with PLC addresses, device codes, numbers, symbols, or English.
5. Do not return pure English, pure numbers, timestamps, addresses, or device codes without Japanese.
6. Do not invent text. Include uncertain readable Japanese with lower confidence.
7. Keep repeated comments when they appear on different visible rows.
8. The English draft must preserve PLC addresses, device codes, numbers, symbols, ON/OFF, and engineering meaning.
9. Return an empty regions array only when no Japanese is readable.
""".strip()
    response = openai_client().responses.create(
        model=MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{encoded}",
                    "detail": "high",
                },
            ],
        }],
        temperature=0,
    )
    payload = extract_json_payload(response.output_text)
    regions: list[dict] = []
    for item in payload.get("regions", []):
        jp = clean_text(item.get("jp", ""))
        if not jp or not JAPANESE_RE.search(jp):
            continue
        bbox = item.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            left, top, right, bottom = [max(0, min(1000, int(round(float(value))))) for value in bbox]
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
        except (TypeError, ValueError):
            continue
        if right <= left or bottom <= top:
            continue
        regions.append({
            "id": len(regions) + 1,
            "jp": jp,
            "draft_en": clean_text(item.get("draft_en", "")),
            "bbox": [left, top, right, bottom],
            "confidence": confidence,
        })
    return sorted(regions, key=lambda item: (item["bbox"][1], item["bbox"][0]))


def glossary_matches(text: str, pairs: tuple[tuple[str, str], ...]) -> list[dict]:
    occupied = [False] * len(text)
    matches: list[dict] = []
    for jp, en in pairs:
        for found in re.finditer(re.escape(jp), text):
            start, end = found.span()
            if any(occupied[start:end]):
                continue
            for index in range(start, end):
                occupied[index] = True
            matches.append({"start": start, "end": end, "jp": jp, "en": en})
    return sorted(matches, key=lambda item: item["start"])


def protect_glossary_terms(text: str, matches: list[dict]) -> tuple[str, dict[str, str]]:
    if not matches:
        return text, {}
    output: list[str] = []
    mapping: dict[str, str] = {}
    cursor = 0
    for index, match in enumerate(matches, start=1):
        output.append(text[cursor : match["start"]])
        marker = f"[[GLOSSARY_{index}]]"
        output.append(marker)
        mapping[marker] = match["en"]
        cursor = match["end"]
    output.append(text[cursor:])
    return "".join(output), mapping


def restore_glossary_terms(text: str, mapping: dict[str, str]) -> str:
    restored = clean_text(text)
    for marker, approved in mapping.items():
        restored = restored.replace(marker, approved)
    return clean_text(restored)


def translate_regions(regions: list[dict], pairs: tuple[tuple[str, str], ...]) -> list[dict]:
    items: list[dict] = []
    prepared: dict[int, dict] = {}
    translations: dict[int, str] = {}

    for region in regions:
        matches = glossary_matches(region["jp"], pairs)
        protected, mapping = protect_glossary_terms(region["jp"], matches)
        prepared[region["id"]] = {"matches": matches, "mapping": mapping, "protected": protected}
        if mapping and protected in mapping:
            translations[region["id"]] = mapping[protected]
            continue
        if not mapping and clean_text(region.get("draft_en", "")):
            # Fast path: Vision already produced the translation. A second model
            # call is unnecessary when no controlled glossary term was detected.
            translations[region["id"]] = clean_text(region["draft_en"])
            continue
        items.append({
            "id": region["id"],
            "source": protected,
            "required_glossary": list(mapping.values()),
        })

    if items:
        prompt = f"""
Translate short Japanese PLC comments into concise, accurate manufacturing English.

The input may contain protected markers such as [[GLOSSARY_1]]. Preserve every marker exactly and in the correct semantic position. Do not translate, remove, rename, or alter a marker. The application replaces each marker with an approved glossary term after translation.

Rules:
1. Preserve PLC addresses, device codes, equipment numbers, symbols, ON/OFF, and numbers exactly.
2. Do not invent a cause, status, action, or equipment name.
3. Use one concise PLC-comment translation, not an explanation.
4. Required glossary wording is mandatory and must not be replaced by a synonym.
5. Return every input id exactly once.

Return only JSON:
{{"translations":[{{"id":1,"en":"English PLC comment"}}]}}

Input:
{json.dumps(items, ensure_ascii=False)}
""".strip()
        response = openai_client().responses.create(
            model=MODEL,
            input=prompt,
            temperature=0,
        )
        payload = extract_json_payload(response.output_text)
        for item in payload.get("translations", []):
            try:
                item_id = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            translations[item_id] = clean_text(item.get("en", ""))

    results: list[dict] = []
    for region in regions:
        item_id = region["id"]
        info = prepared[item_id]
        translated = restore_glossary_terms(translations.get(item_id, ""), info["mapping"])
        required = list(dict.fromkeys(info["mapping"].values()))
        missing = [term for term in required if term.casefold() not in translated.casefold()]
        if missing:
            # Never hide a controlled-term failure. The red glossary badge remains visible
            # and the row is explicitly marked for review rather than silently accepting it.
            status = "review_required"
        else:
            status = "ok"
        results.append({
            **region,
            "en": translated,
            "glossary": [
                {"jp": match["jp"], "en": match["en"]}
                for match in info["matches"]
            ],
            "controlled_terms": required,
            "status": status,
        })
    return results


def process_frame(raw: bytes) -> dict:
    print(f"[PLC Lens] Preparing camera frame ({len(raw):,} bytes)...", flush=True)
    prepared, width, height = prepare_frame(raw)
    print(f"[PLC Lens] Running Japanese OCR at {width} x {height}...", flush=True)
    regions = extract_japanese_regions(prepared, width, height)
    print(f"[PLC Lens] OCR detected {len(regions)} Japanese region(s).", flush=True)
    if not regions:
        return {
            "regions": [],
            "message": "No readable Japanese PLC comments were detected. Move closer and scan again.",
            "glossary_terms": len(glossary_pairs()),
        }
    pairs = glossary_pairs()
    print(
        f"[PLC Lens] Matching {len(pairs)} controlled glossary term(s); "
        "using the one-pass fast path when possible...",
        flush=True,
    )
    results = translate_regions(regions, pairs)
    controlled_count = sum(len(item["glossary"]) for item in results)
    print(
        f"[PLC Lens] Complete: {len(results)} translation(s), "
        f"{controlled_count} controlled match(es).",
        flush=True,
    )
    return {
        "regions": results,
        "message": "Scan complete.",
        "glossary_terms": len(glossary_pairs()),
        "controlled_matches": controlled_count,
    }


async def homepage(_request: Request) -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


async def health(_request: Request) -> JSONResponse:
    try:
        terms = len(glossary_pairs())
        return JSONResponse({"status": "ok", "model": MODEL, "glossary_terms": terms})
    except Exception as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=500)


async def scan(request: Request) -> JSONResponse:
    raw = await request.body()
    if not raw:
        return JSONResponse({"error": "The camera frame was empty."}, status_code=400)
    if len(raw) > MAX_FRAME_BYTES:
        return JSONResponse({"error": "The camera frame is too large."}, status_code=413)

    digest = hashlib.sha256(raw).hexdigest()
    with RESULT_CACHE_LOCK:
        cached = RESULT_CACHE.get(digest)
        if cached is not None:
            RESULT_CACHE.move_to_end(digest)
            return JSONResponse(cached)

    try:
        client_host = request.client.host if request.client else "unknown"
        print(f"[PLC Lens] Scan request received from {client_host}.", flush=True)
        async with SCAN_SEMAPHORE:
            result = await asyncio.wait_for(
                asyncio.to_thread(process_frame, raw),
                timeout=120,
            )
    except asyncio.TimeoutError:
        print("[PLC Lens] Scan timed out after 120 seconds.", flush=True)
        return JSONResponse(
            {"error": "The scan timed out. Move closer to the PLC screen and try again."},
            status_code=504,
        )
    except Exception as exc:
        print(f"[PLC Lens] Scan failed: {type(exc).__name__}: {exc}", flush=True)
        message = str(exc)
        if len(message) > 500:
            message = message[:500]
        return JSONResponse({"error": f"Scan failed: {message}"}, status_code=500)

    with RESULT_CACHE_LOCK:
        RESULT_CACHE[digest] = result
        RESULT_CACHE.move_to_end(digest)
        while len(RESULT_CACHE) > 24:
            RESULT_CACHE.popitem(last=False)
    return JSONResponse(result)


routes = [
    Route("/", homepage, methods=["GET"]),
    Route("/health", health, methods=["GET"]),
    Route("/api/scan", scan, methods=["POST"]),
]
app = Starlette(debug=False, routes=routes)


INDEX_HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, user-scalable=no">
  <title>Real-Time PLC Translation Lens</title>
  <style>
    :root { color-scheme: dark; --red:#e51b23; --panel:rgba(10,12,16,.92); --line:#30343b; }
    * { box-sizing:border-box; }
    html,body { margin:0; min-height:100%; background:#050608; color:#fff; font-family:Arial,Helvetica,sans-serif; }
    body { padding-bottom:env(safe-area-inset-bottom); }
    header { padding:12px 16px 10px; background:#0d1014; border-bottom:1px solid #272b31; }
    header h1 { margin:0; font-size:18px; letter-spacing:.2px; }
    header p { margin:4px 0 0; color:#aeb4bd; font-size:12px; }
    #viewer { position:relative; width:100%; aspect-ratio:16/10; height:auto; background:#000; overflow:hidden; }
    #video,#freeze { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; background:#000; }
    #freeze { display:none; }
    #overlay { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
    #guide { position:absolute; inset:4%; border:2px solid rgba(255,255,255,.72); border-radius:10px; box-shadow:0 0 0 999px rgba(0,0,0,.10); pointer-events:none; }
    #status { position:absolute; left:12px; right:12px; top:12px; z-index:3; text-align:center; }
    #status span { display:inline-block; max-width:100%; padding:7px 11px; border-radius:999px; background:rgba(0,0,0,.75); font-size:12px; }
    #controls { display:grid; grid-template-columns:1fr 1.6fr 1fr; gap:10px; padding:12px; background:#0d1014; border-top:1px solid #272b31; }
    button { min-height:48px; border:1px solid #3a4048; border-radius:12px; background:#20242a; color:#fff; font-weight:700; font-size:14px; }
    button.primary { background:#fff; color:#08090b; border-color:#fff; font-size:16px; }
    button:disabled { opacity:.45; }
    #results { background:#f3f4f6; color:#15171a; min-height:180px; padding:12px; }
    .summary { color:#626872; font-size:12px; margin:0 0 10px; }
    .row { background:#fff; border:1px solid #d9dde3; border-radius:10px; padding:10px 12px; margin-bottom:9px; }
    .jp { color:#6c727b; font-size:12px; margin-bottom:5px; }
    .en { color:#101217; font-size:17px; line-height:1.28; font-weight:700; }
    .controlled { color:var(--red); font-weight:800; }
    .badge { display:inline-block; margin:7px 5px 0 0; padding:3px 7px; border-radius:999px; background:#ffe9ea; color:#c20f18; font-size:11px; font-weight:700; }
    .review { margin-top:6px; color:#b26a00; font-size:11px; font-weight:700; }
    .empty { color:#626872; text-align:center; padding:28px 10px; }
    @media (min-width:800px) { body { max-width:720px; margin:auto; border-left:1px solid #222; border-right:1px solid #222; } }
  </style>
</head>
<body>
  <header>
    <h1>Real-Time PLC Translation Lens</h1>
    <p>Live Japanese PLC comments → controlled manufacturing English</p>
  </header>
  <section id="viewer">
    <video id="video" autoplay playsinline muted></video>
    <img id="freeze" alt="Current scanned PLC screen">
    <canvas id="overlay"></canvas>
    <div id="guide"></div>
    <div id="status"><span>Tap START CAMERA</span></div>
  </section>
  <section id="controls">
    <button id="cameraBtn">START CAMERA</button>
    <button id="scanBtn" class="primary" disabled>SCAN</button>
    <button id="nextBtn" disabled>NEXT SCREEN</button>
  </section>
  <input id="cameraFallback" type="file" accept="image/*" capture="environment" hidden>
  <section id="results">
    <div class="empty">Point the live camera at one PLC comment screen, then tap SCAN.</div>
  </section>

<script>
const video = document.getElementById('video');
const freeze = document.getElementById('freeze');
const overlay = document.getElementById('overlay');
const viewer = document.getElementById('viewer');
const cameraBtn = document.getElementById('cameraBtn');
const scanBtn = document.getElementById('scanBtn');
const nextBtn = document.getElementById('nextBtn');
const cameraFallback = document.getElementById('cameraFallback');
const results = document.getElementById('results');
const statusEl = document.querySelector('#status span');
let stream = null;
let busy = false;
let frozenUrl = null;
let latestRegions = [];

function status(text) { statusEl.textContent = text; }

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    cameraFallback.value='';
    cameraFallback.click();
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode:{ideal:'environment'}, width:{ideal:1920}, height:{ideal:1080} },
      audio:false
    });
    video.srcObject = stream;
    await video.play();
    cameraBtn.textContent = 'CAMERA ON';
    cameraBtn.disabled = true;
    scanBtn.disabled = false;
    status('Align one PLC screen inside the frame');
  } catch (err) {
    status('Camera unavailable — HTTPS and camera permission are required');
    results.innerHTML = `<div class="empty">${escapeHtml(err.message || String(err))}</div>`;
  }
}

function escapeHtml(text) {
  return String(text ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}

function frameToBlob() {
  // Capture only the landscape PLC area visible in the viewer. This sends fewer
  // pixels to OCR and keeps returned coordinates aligned with the frozen frame.
  const vw=video.videoWidth, vh=video.videoHeight;
  const targetAspect=16/10;
  let sx=0, sy=0, sw=vw, sh=vh;
  if (vw/vh > targetAspect) { sw=vh*targetAspect; sx=(vw-sw)/2; }
  else { sh=vw/targetAspect; sy=(vh-sh)/2; }
  const canvas = document.createElement('canvas');
  canvas.width=1280; canvas.height=800;
  canvas.getContext('2d').drawImage(video,sx,sy,sw,sh,0,0,canvas.width,canvas.height);
  return new Promise(resolve => canvas.toBlob(blob => resolve({blob, canvas}), 'image/jpeg', .80));
}

function controlledHtml(text, terms) {
  let escaped = escapeHtml(text);
  [...new Set(terms || [])].sort((a,b)=>b.length-a.length).forEach(term => {
    const safe = escapeHtml(term);
    escaped = escaped.split(safe).join(`<span class="controlled">${safe}</span>`);
  });
  return escaped;
}

function renderResults(data) {
  const rows = data.regions || [];
  if (!rows.length) {
    results.innerHTML = `<div class="empty">${escapeHtml(data.message || 'No Japanese text detected.')}</div>`;
    return;
  }
  const controlled = data.controlled_matches || 0;
  results.innerHTML = `<p class="summary">${rows.length} PLC comment(s) detected · ${controlled} controlled glossary match(es) · controlled wording is red</p>` + rows.map(row => {
    const badges = (row.glossary || []).map(t => `<span class="badge">${escapeHtml(t.jp)} → ${escapeHtml(t.en)}</span>`).join('');
    const review = row.status === 'review_required' ? '<div class="review">REVIEW REQUIRED — controlled wording needs confirmation</div>' : '';
    return `<div class="row"><div class="jp">${escapeHtml(row.jp)}</div><div class="en">${controlledHtml(row.en, row.controlled_terms)}</div>${badges}${review}</div>`;
  }).join('');
}

function videoRect() {
  const cw = viewer.clientWidth, ch = viewer.clientHeight;
  const useFreeze = freeze.style.display !== 'none' && freeze.naturalWidth;
  const vw = useFreeze ? freeze.naturalWidth : (video.videoWidth || 1);
  const vh = useFreeze ? freeze.naturalHeight : (video.videoHeight || 1);
  const scale = Math.max(cw/vw, ch/vh);
  return {x:(cw-vw*scale)/2, y:(ch-vh*scale)/2, w:vw*scale, h:vh*scale};
}

function drawRichLine(ctx, text, terms, x, y, maxWidth, maxHeight) {
  const compact=String(text || '').replace(/\s+/g,' ').trim();
  const estimatedByWidth=maxWidth/Math.max(4,compact.length*.56);
  const fontSize=Math.max(7,Math.min(17,Math.floor(Math.min(maxHeight*.72,estimatedByWidth*1.45))));
  ctx.font = `700 ${fontSize}px Arial`;
  const controlled = (terms || []).map(t => t.toLowerCase());
  const words = String(text || '').split(/(\s+)/).filter(Boolean);
  let cx=x+3, cy=y+fontSize+2;
  const lineHeight=fontSize*1.08;
  words.forEach(word => {
    const width=ctx.measureText(word).width;
    if (cx+width > x+maxWidth-3 && cx>x+3) { cx=x+3; cy+=lineHeight; }
    if (cy > y+maxHeight-1) return;
    const lower=word.trim().toLowerCase().replace(/^[^a-z0-9]+|[^a-z0-9]+$/g,'');
    const isControlled=controlled.some(term => term.includes(lower) && lower.length>0);
    ctx.fillStyle=isControlled ? '#e51b23' : '#111318';
    ctx.fillText(word,cx,cy);
    cx+=width;
  });
}

function drawOverlay(rows) {
  const rect = viewer.getBoundingClientRect();
  overlay.width = Math.round(rect.width);
  overlay.height = Math.round(rect.height);
  const ctx=overlay.getContext('2d');
  ctx.clearRect(0,0,overlay.width,overlay.height);
  const vr=videoRect();
  (rows || []).forEach(row => {
    const [l,t,r,b]=row.bbox;
    const rawX=vr.x+vr.w*l/1000, rawY=vr.y+vr.h*t/1000;
    const rawW=vr.w*(r-l)/1000, rawH=vr.h*(b-t)/1000;
    const pad=2, x=Math.max(vr.x,rawX-pad), y=Math.max(vr.y,rawY-pad);
    const w=Math.min(vr.x+vr.w-x,Math.max(18,rawW+pad*2));
    const h=Math.min(vr.y+vr.h-y,Math.max(14,rawH+pad*2));
    ctx.fillStyle='rgba(255,255,255,.94)';
    ctx.strokeStyle=(row.controlled_terms || []).length ? '#e51b23' : '#4b5563';
    ctx.lineWidth=(row.controlled_terms || []).length ? 1.5 : 1;
    ctx.fillRect(x,y,w,h); ctx.strokeRect(x,y,w,h);
    drawRichLine(ctx,row.en,row.controlled_terms,x,y,w,h);
  });
}

async function processScanBlob(blob) {
  if (busy || !blob) return;
  busy=true; scanBtn.disabled=true; nextBtn.disabled=true;
  status('Scanning Japanese PLC comments…');
  try {
    if (frozenUrl) URL.revokeObjectURL(frozenUrl);
    frozenUrl=URL.createObjectURL(blob);
    freeze.src=frozenUrl;
    freeze.style.display='block';
    video.style.visibility='hidden';
    try { await freeze.decode(); } catch (_) {}
    const controller=new AbortController();
    const timeoutId=setTimeout(()=>controller.abort(),125000);
    let response;
    try {
      response=await fetch('/api/scan',{method:'POST',headers:{'Content-Type':'image/jpeg'},body:blob,signal:controller.signal});
    } finally {
      clearTimeout(timeoutId);
    }
    const data=await response.json();
    if (!response.ok) throw new Error(data.error || 'Scan failed.');
    latestRegions=data.regions || [];
    drawOverlay(latestRegions); renderResults(data);
    status(data.message || 'Scan complete');
    nextBtn.disabled=false;
  } catch(err) {
    status('Scan failed');
    const message=err.name==='AbortError' ? 'Scan timed out after 125 seconds. Move closer and try again.' : (err.message || String(err));
    results.innerHTML=`<div class="empty">${escapeHtml(message)}</div>`;
    nextBtn.disabled=false;
  } finally { busy=false; }
}

async function scanFrame() {
  if (busy || !stream) return;
  const {blob}=await frameToBlob();
  if (!blob) { status('Unable to scan the live camera frame'); return; }
  await processScanBlob(blob);
}

function nextScreen() {
  freeze.style.display='none'; video.style.visibility='visible';
  overlay.getContext('2d').clearRect(0,0,overlay.width,overlay.height);
  latestRegions=[]; scanBtn.disabled=false; nextBtn.disabled=true;
  results.innerHTML='<div class="empty">Move to the next PLC screen, align it, then tap SCAN.</div>';
  status('Align the next PLC screen');
  if (!stream) {
    cameraFallback.value='';
    cameraFallback.click();
  }
}

cameraBtn.addEventListener('click',startCamera);
scanBtn.addEventListener('click',scanFrame);
nextBtn.addEventListener('click',nextScreen);
cameraFallback.addEventListener('change',async event => {
  const file=event.target.files && event.target.files[0];
  if (file) await processScanBlob(file);
});
window.addEventListener('resize',()=>{ if(latestRegions.length) drawOverlay(latestRegions); });
if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
  cameraBtn.textContent='SCAN WITH CAMERA';
  status('Tap SCAN WITH CAMERA for the first PLC screen');
}
</script>
</body>
</html>'''
