import csv
import html
import mimetypes
import io
import hashlib
import json
import os
import re
import smtplib
import sqlite3
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from urllib.parse import quote
from zipfile import BadZipFile, ZipFile, ZIP_DEFLATED
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openpyxl import load_workbook
from openai import APIConnectionError, APIStatusError, AuthenticationError, OpenAI, RateLimitError

try:
    import truststore
except ImportError:
    truststore = None

try:
    from langsmith.wrappers import wrap_openai
except ImportError:
    wrap_openai = None


BASE_DIR = Path(__file__).parent
APP_STARTED_AT = datetime.now()
ENV_PATH = BASE_DIR / "app.env"
DEFAULT_GLOSSARY_PATHS = [
    BASE_DIR / "glossary.xlsx",
    BASE_DIR / "glossary.csv",
]
DEFAULT_PLC_RULE_PATHS = [
    BASE_DIR / "plc_abbreviation_rules.xlsx",
    BASE_DIR / "plc_abbreviation_rules.csv",
]
DEFAULT_MODEL = "gpt-4.1-mini"
DOCUMENT_BATCH_SIZE = 25
MAX_PARALLEL_BATCHES = 12
MAX_TRANSLATION_RETRIES = 2
OPENAI_TIMEOUT_SECONDS = 60
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_EMAIL_ATTACHMENT_BYTES = 20 * 1024 * 1024
PROGRESS_DIR = BASE_DIR / ".term1_progress"
USAGE_COUNT_PATH = BASE_DIR / ".term1_usage_count.json"
JOB_DB_PATH = BASE_DIR / ".term1_jobs.db"
TRANSLATION_MEMORY_DB_PATH = BASE_DIR / "translation_memory.sqlite"
JOB_STORAGE_DIR = BASE_DIR / ".term1_job_storage"
JOB_UPLOAD_DIR = JOB_STORAGE_DIR / "uploads"
JOB_RESULT_DIR = JOB_STORAGE_DIR / "results"
GENERAL_TRANSLATION_MODE = "General Plant Translation"
PLC_TRANSLATION_MODE = "PLC/SPLC Comment Standardization"
SUPPLIER_EMAIL_TRANSLATION_MODE = "Supplier Email Translation"
PRODUCT_CATALOG_TRANSLATION_MODE = "Product Catalog Translation"
ROBOT_PROGRAM_TRANSLATION_MODE = "Kawasaki Robot .as file"
TRANSLATION_MODES = [
    PLC_TRANSLATION_MODE,
    GENERAL_TRANSLATION_MODE,
    SUPPLIER_EMAIL_TRANSLATION_MODE,
    PRODUCT_CATALOG_TRANSLATION_MODE,
    ROBOT_PROGRAM_TRANSLATION_MODE,
]
PLC_DUPLICATE_STATUS_WORDS = [
    "ON",
    "OFF",
    "OK",
    "NG",
    "Mode",
    "Error",
    "Complete",
    "Confirm",
    "Request",
    "Command",
    "Present",
    "Absent",
]
PLC_SYNONYM_CLEANUPS = [
    (re.compile(r"\bpoor,\s*defective,\s*NG,\s*inoperative\b", re.IGNORECASE), "NG"),
    (re.compile(r"\bdefective,\s*NG\b", re.IGNORECASE), "NG"),
]
PROTECTED_PATTERN = re.compile(
    r"\b(?:[A-Z]{1,6}[-_]?\d{1,6}[A-Z]?|\d+[A-Z]{1,4}|[XYMDSZR][0-9]{1,5}|[A-Z]{2,}-[A-Z0-9-]+)\b"
)
ENCLOSED_ALNUM_PATTERN = re.compile(r"[\u2460-\u24ff\u3200-\u32ff]")
LEADING_CODE_PATTERN = re.compile(r"^([A-Z]{1,6}[-_]?\d{1,6}[A-Z]?)(.*)$")

WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
PPT_NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
EXCEL_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
EXCEL_SERIALIZE_NAMESPACES = {
    "": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "x14ac": "http://schemas.microsoft.com/office/spreadsheetml/2009/9/ac",
    "xr": "http://schemas.microsoft.com/office/spreadsheetml/2014/revision",
    "xr2": "http://schemas.microsoft.com/office/spreadsheetml/2015/revision2",
    "xr3": "http://schemas.microsoft.com/office/spreadsheetml/2016/revision3",
}


@dataclass(frozen=True)
class TermHit:
    jp: str
    en: str
    count: int


@dataclass(frozen=True)
class TextBlock:
    location: str
    text: str


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens

    def display(self) -> str:
        if self.total_tokens <= 0:
            return "Token usage unavailable."
        return (
            f"Tokens: input {self.input_tokens:,}, "
            f"output {self.output_tokens:,}, total {self.total_tokens:,}."
        )


def load_env() -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    else:
        load_dotenv()
    if truststore is not None:
        truststore.inject_into_ssl()


def clean_text(value: str) -> str:
    text = str(value)
    protected = {}

    def stash(match: re.Match) -> str:
        token = f"__TERM1_ENCLOSED_{len(protected)}__"
        protected[token] = match.group(0)
        return token

    normalized = unicodedata.normalize("NFKC", ENCLOSED_ALNUM_PATTERN.sub(stash, text)).strip()
    for token, original in protected.items():
        normalized = normalized.replace(token, original)
    return normalized


def has_japanese_text(value: str) -> bool:
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", value))


def should_translate(text: str) -> bool:
    return has_japanese_text(clean_text(text))


def decode_document_text_with_encoding(raw: bytes) -> tuple[str, str]:
    candidates = []
    has_utf16_bom = raw.startswith((b"\xff\xfe", b"\xfe\xff"))
    null_ratio = raw.count(b"\x00") / max(len(raw), 1)
    encodings = ["utf-8-sig", "utf-8", "cp932", "shift_jis", "euc_jp", "iso2022_jp"]
    if has_utf16_bom or null_ratio > 0.1:
        encodings = ["utf-16", "utf-16-le", "utf-16-be"] + encodings
    encodings.append("cp1252")

    for encoding in encodings:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue

        japanese_count = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", text))
        replacement_count = text.count("�")
        mojibake_count = sum(text.count(marker) for marker in ("ã", "Â", "Ã", "\x00"))
        control_count = sum(1 for char in text if ord(char) < 32 and char not in "\r\n\t")
        c1_count = sum(1 for char in text if 0x80 <= ord(char) <= 0x9F)
        ascii_count = sum(1 for char in text if ord(char) < 128)
        score = (
            japanese_count * 50
            + min(ascii_count, 200) * 0.01
            - replacement_count * 200
            - mojibake_count * 80
            - control_count * 60
            - c1_count * 40
        )
        candidates.append((score, encoding, text))

    if candidates:
        _score, encoding, text = max(candidates, key=lambda candidate: candidate[0])
        return text, encoding

    return raw.decode("utf-8", errors="replace"), "utf-8"


def decode_document_text(raw: bytes) -> str:
    return decode_document_text_with_encoding(raw)[0]


def robot_program_decode_score(text: str) -> float:
    comment_text = "\n".join(value for _start, _end, value in robot_comment_segments(text))
    japanese_comment_count = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", comment_text))
    total_japanese_count = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", text))
    replacement_count = text.count("\ufffd")
    mojibake_count = sum(text.count(marker) for marker in ("Ã£", "Ã‚", "Ãƒ", "鐃緒申", "鐃", "\x00"))
    control_count = sum(1 for char in text if ord(char) < 32 and char not in "\r\n\t")
    c1_count = sum(1 for char in text if 0x80 <= ord(char) <= 0x9F)
    return (
        japanese_comment_count * 200
        + total_japanese_count * 10
        - replacement_count * 500
        - mojibake_count * 100
        - control_count * 80
        - c1_count * 40
    )


def decode_robot_program_text_with_encoding(raw: bytes) -> tuple[str, str]:
    encodings = [
        "euc_jp",
        "cp932",
        "shift_jis",
        "utf-8-sig",
        "utf-8",
        "iso2022_jp",
    ]
    strict_candidates = []
    for encoding in encodings:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        strict_candidates.append((robot_program_decode_score(text), encoding, text))

    if strict_candidates:
        _score, encoding, text = max(strict_candidates, key=lambda candidate: candidate[0])
        return text, encoding

    replacement_candidates = []
    for encoding in encodings:
        text = raw.decode(encoding, errors="replace")
        replacement_candidates.append((robot_program_decode_score(text), encoding, text))
    _score, encoding, text = max(replacement_candidates, key=lambda candidate: candidate[0])
    return text, encoding


def looks_like_mojibake(value: str) -> bool:
    markers = ("�", "鐃", "鐃緒申", "Ã£", "Ã‚", "Ãƒ")
    return any(marker in value for marker in markers)


def has_robot_mojibake_marker(value: str) -> bool:
    markers = (
        "\ufffd",
        chr(0x9403),
        chr(0x9403) + chr(0x7DD2) + chr(0x7533),
        "Ã£",
        "Ã‚",
        "Ãƒ",
    )
    return any(marker in value for marker in markers) or looks_like_mojibake(value)


def compact_warning_line(line: str, limit: int = 180) -> str:
    compact = re.sub(r"\s+", " ", line).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def robot_block_ranges(raw: bytes) -> list[tuple[int, int]]:
    ranges = []
    try:
        blocks = extract_robot_program_blocks(raw)
    except Exception:
        return ranges
    for block in blocks:
        parts = block.location.split(":")
        if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
            continue
        ranges.append((int(parts[1]), int(parts[2])))
    return ranges


def robot_encoding_warning(raw: bytes, file_name: str) -> str:
    if not file_name.lower().endswith((".as", ".ad")):
        return ""

    replacement_bytes = raw.count(b"\xef\xbf\xbd")
    warning_parts = []
    if replacement_bytes:
        warning_parts.append(
            f"{replacement_bytes:,} replacement-character byte sequence(s) were found. "
            "This usually means Japanese text was already damaged by a wrong encoding conversion."
        )

    try:
        text, encoding = decode_robot_program_text_with_encoding(raw)
    except Exception:
        return (
            "Encoding warning: this robot program could not be decoded reliably. "
            "Please request the original exported .as/.ad file."
        )

    bad_lines = []
    suspicious_segments = 0
    unsupported_japanese_lines = []
    translated_ranges = robot_block_ranges(raw)
    line_start = 0
    for line_number, raw_line in enumerate(text.splitlines(keepends=True), start=1):
        line = raw_line.rstrip("\r\n")
        line_end = line_start + len(line)
        has_bad_replacement = "\ufffd" in line
        comment = line.split(";", 1)[1] if ";" in line else ""
        has_bad_comment = bool(comment and has_robot_mojibake_marker(comment))
        has_japanese_outside_supported_fields = (
            has_japanese_text(line)
            and not has_bad_replacement
            and not has_bad_comment
            and not any(start < line_end and end > line_start for start, end in translated_ranges)
        )
        if has_bad_comment:
            suspicious_segments += 1
        if has_japanese_outside_supported_fields:
            unsupported_japanese_lines.append(line_number)
        if has_bad_replacement or has_bad_comment or has_japanese_outside_supported_fields:
            bad_lines.append((line_number, compact_warning_line(line)))
        line_start += len(raw_line)

    if suspicious_segments:
        warning_parts.append(
            f"{suspicious_segments:,} robot comment line(s) look like mojibake/fake Japanese after decoding as {encoding}."
        )
    if unsupported_japanese_lines:
        warning_parts.append(
            f"{len(unsupported_japanese_lines):,} line(s) contain Japanese outside the supported Kawasaki comment/label fields, so they were not translated."
        )

    if not warning_parts:
        return ""

    line_details = ""
    if bad_lines:
        line_details = (
            "\n\nProblem line(s) to send to the robot/program team:\n"
            + "\n".join(f"Line {line_number}: {line}" for line_number, line in bad_lines[:20])
        )
        if len(bad_lines) > 20:
            line_details += f"\n...and {len(bad_lines) - 20:,} more suspicious line(s)."

    return (
        "AS File Warning: This Kawasaki AS/AD file has Japanese text that may not be safely translated or written back. "
        + " ".join(warning_parts)
        + " Review the listed line(s) before translation. If the lines contain corrupted Japanese, do not open and re-save the robot file with Notepad, Excel, or browser preview; ask for the original exported file or a file confirmed to display Japanese correctly."
        + line_details
    )


def document_fingerprint(file_name: str, raw: bytes) -> str:
    digest = hashlib.sha256(raw).hexdigest()[:16]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(file_name).stem).strip("_") or "document"
    return f"{safe_name}_{len(raw)}_{digest}"


def checkpoint_path_for(file_name: str, raw: bytes, translation_mode: str = GENERAL_TRANSLATION_MODE) -> Path:
    mode_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", translation_mode).strip("_").lower()
    return PROGRESS_DIR / f"{document_fingerprint(file_name, raw)}_{mode_key}.json"


def load_checkpoint(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    translations = data.get("translations", {})
    if not isinstance(translations, dict):
        return {}
    return {str(key): str(value) for key, value in translations.items()}


def save_checkpoint(path: Path, translations: dict[str, str]) -> None:
    PROGRESS_DIR.mkdir(exist_ok=True)
    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "translations": translations,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    minutes, second = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minute}m"
    if minute:
        return f"{minute}m {second}s"
    return f"{second}s"


def format_file_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.0f} KB"
    return f"{size_bytes} B"


def estimate_remaining_time(done: int, total: int, elapsed_seconds: float) -> str:
    if done <= 0 or total <= 0 or done >= total or elapsed_seconds <= 0:
        return ""
    progress_ratio = done / total
    if progress_ratio < 0.25:
        return ""
    remaining_seconds = (elapsed_seconds / done) * max(total - done, 0)
    return format_duration(remaining_seconds)


def progress_text(done: int, total: int, elapsed_seconds: float | None = None) -> str:
    if total <= 0:
        return "Preparing file"
    if done >= total:
        return "Complete"
    if done <= 0:
        return "Starting translation"
    return "Translating"


def progress_percent(done: int, total: int) -> str:
    if total <= 0:
        return "Preparing"
    return f"{min(max(done / total, 0), 1) * 100:.2f}%"


def elapsed_since_timestamp(timestamp_text: str) -> float | None:
    try:
        started_at = datetime.strptime(timestamp_text, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None
    return max((datetime.now() - started_at).total_seconds(), 0)


def parse_timestamp(timestamp_text: str) -> datetime | None:
    try:
        return datetime.strptime(timestamp_text, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def render_download_ready(data: bytes, file_name: str, mime: str, key: str = "download_ready") -> None:
    st.success("Complete | Download ready")
    st.download_button(
        "Download Translated File",
        data=data,
        file_name=file_name,
        mime=mime,
        type="primary",
        key=key,
    )


def translation_pairs_preview(
    blocks: list[TextBlock],
    translations: dict[str, str],
    limit: int = 200,
) -> pd.DataFrame:
    rows = []
    for block in blocks:
        if block.location not in translations or not should_translate(block.text):
            continue
        english = clean_text(translations[block.location])
        japanese = clean_text(block.text)
        if not english or english == japanese:
            continue
        rows.append(
            {
                "Location": block.location.split(":", 1)[0],
                "JP": japanese,
                "EN": english,
            }
        )
        if len(rows) >= limit:
            break
    return pd.DataFrame(rows)


def render_translation_pairs_preview(
    raw_document: bytes,
    file_name: str,
    translation_mode: str,
    max_rows: int = 200,
) -> None:
    try:
        blocks = extract_text_blocks(raw_document, file_name)
        translations = load_checkpoint(checkpoint_path_for(file_name, raw_document, translation_mode))
        preview = translation_pairs_preview(blocks, translations, max_rows)
    except Exception as exc:
        st.caption(f"JP/EN preview unavailable: {exc}")
        return

    if preview.empty:
        st.caption("JP/EN preview: no translated Japanese text was available for this file.")
        return

    st.subheader("JP/EN Preview")
    st.dataframe(preview, use_container_width=True, hide_index=True)
    total_pairs = sum(
        1
        for block in blocks
        if block.location in translations and should_translate(block.text)
    )
    if total_pairs > len(preview):
        st.caption(f"Showing first {len(preview):,} of {total_pairs:,} translated JP/EN pairs.")


def start_background_translation_job(
    raw_document: bytes,
    file_name: str,
    blocks: list[TextBlock],
    glossary: pd.DataFrame,
    translation_mode: str,
    keep_source_with_translation: bool,
    notify_email: str,
    batch_count: int,
    progress_path: Path,
) -> str:
    translatable_blocks = [block for block in blocks if should_translate(block.text)]
    job_id = create_translation_job(
        file_name,
        len(raw_document),
        len(blocks),
        len(translatable_blocks),
        batch_count,
        translation_mode,
        notify_email=notify_email,
        status="pending",
    )
    source_path = job_upload_path(job_id, file_name)
    source_path.write_bytes(raw_document)
    update_translation_job(job_id, source_file_path=str(source_path))
    background_job_executor().submit(
        run_document_translation_job,
        job_id,
        raw_document,
        file_name,
        blocks,
        glossary,
        translation_mode,
        keep_source_with_translation,
        progress_path,
        batch_count,
    )
    return job_id


def start_queued_document_translation_job(
    raw_document: bytes,
    file_name: str,
    glossary: pd.DataFrame,
    translation_mode: str,
    keep_source_with_translation: bool,
    notify_email: str,
) -> str:
    stop_active_translation_jobs_for_file(
        file_name,
        "Stopped automatically because a newer job was started for this file.",
    )
    job_id = create_translation_job(
        file_name,
        len(raw_document),
        0,
        0,
        0,
        translation_mode,
        notify_email=notify_email,
        status="pending",
    )
    source_path = job_upload_path(job_id, file_name)
    source_path.write_bytes(raw_document)
    update_translation_job(
        job_id,
        source_file_path=str(source_path),
        progress_message="Queued. Preparing file.",
    )
    background_job_executor().submit(
        prepare_and_run_document_translation_job,
        job_id,
        raw_document,
        file_name,
        glossary,
        translation_mode,
        keep_source_with_translation,
    )
    return job_id


def clean_office_xml_text(value: str) -> str:
    # Office XML files cannot contain most control characters.
    return "".join(
        char
        for char in str(value)
        if char in "\t\n\r" or ord(char) >= 32
    )


def rerun_app() -> None:
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun:
        rerun()


def read_usage_count() -> int:
    if not USAGE_COUNT_PATH.exists():
        return 0
    try:
        data = json.loads(USAGE_COUNT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    return int(data.get("count", 0))


def increment_usage_count_once() -> int:
    if st.session_state.get("usage_counted"):
        return read_usage_count()

    count = read_usage_count() + 1
    payload = {
        "count": count,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    USAGE_COUNT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    st.session_state["usage_counted"] = True
    return count


def init_job_store() -> None:
    with sqlite3.connect(JOB_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS translation_jobs (
                job_id TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                status TEXT NOT NULL,
                total_blocks INTEGER DEFAULT 0,
                translatable_blocks INTEGER DEFAULT 0,
                completed_blocks INTEGER DEFAULT 0,
                total_batches INTEGER DEFAULT 0,
                completed_batches INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                result_file_name TEXT DEFAULT '',
                error_message TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                finished_at TEXT DEFAULT ''
            )
            """
        )
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(translation_jobs)").fetchall()
        }
        if "translation_mode" not in columns:
            conn.execute(
                "ALTER TABLE translation_jobs ADD COLUMN translation_mode TEXT DEFAULT ''"
            )
        for column_name in (
            "source_file_path",
            "result_file_path",
            "result_mime",
            "progress_message",
            "notify_email",
            "notification_status",
        ):
            if column_name not in columns:
                conn.execute(
                    f"ALTER TABLE translation_jobs ADD COLUMN {column_name} TEXT DEFAULT ''"
                )


def create_translation_job(
    file_name: str,
    file_size_bytes: int,
    total_blocks: int,
    translatable_blocks: int,
    total_batches: int,
    translation_mode: str = GENERAL_TRANSLATION_MODE,
    source_file_path: str = "",
    result_file_path: str = "",
    result_mime: str = "",
    notify_email: str = "",
    status: str = "running",
) -> str:
    init_job_store()
    job_id = f"job_{int(time.time())}_{hashlib.sha256(f'{file_name}:{file_size_bytes}:{time.time()}'.encode()).hexdigest()[:8]}"
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(JOB_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO translation_jobs (
                job_id, file_name, file_size_bytes, translation_mode, status,
                source_file_path, result_file_path, result_mime, total_blocks,
                translatable_blocks, completed_blocks, total_batches,
                completed_batches, notify_email, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                file_name,
                file_size_bytes,
                translation_mode,
                status,
                source_file_path,
                result_file_path,
                result_mime,
                total_blocks,
                translatable_blocks,
                0,
                total_batches,
                0,
                notify_email,
                now,
                now,
            ),
        )
    return job_id


def update_translation_job(job_id: str, **fields) -> None:
    if not job_id or not fields:
        return
    init_job_store()
    allowed_fields = {
        "status",
        "total_blocks",
        "translatable_blocks",
        "completed_blocks",
        "total_batches",
        "completed_batches",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "result_file_name",
        "source_file_path",
        "result_file_path",
        "result_mime",
        "notify_email",
        "notification_status",
        "progress_message",
        "error_message",
        "finished_at",
    }
    updates = {key: value for key, value in fields.items() if key in allowed_fields}
    if not updates:
        return
    updates["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    assignments = ", ".join(f"{key} = ?" for key in updates)
    values = list(updates.values()) + [job_id]
    with sqlite3.connect(JOB_DB_PATH) as conn:
        conn.execute(f"UPDATE translation_jobs SET {assignments} WHERE job_id = ?", values)


def recent_translation_jobs(limit: int = 10) -> pd.DataFrame:
    init_job_store()
    with sqlite3.connect(JOB_DB_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT
                job_id AS "Job ID",
                file_name AS "File",
                translation_mode AS "Mode",
                status AS "Status",
                completed_blocks || '/' || translatable_blocks AS "Blocks",
                completed_batches || '/' || total_batches AS "Batches",
                result_file_name AS "Result",
                updated_at AS "Updated"
            FROM translation_jobs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )


def recent_translation_job_details(limit: int = 8) -> pd.DataFrame:
    init_job_store()
    with sqlite3.connect(JOB_DB_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT
                job_id,
                file_name,
                translation_mode,
                status,
                translatable_blocks,
                completed_blocks,
                total_batches,
                completed_batches,
                result_file_name,
                created_at,
                updated_at
            FROM translation_jobs
            ORDER BY updated_at DESC, created_at DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )


def translation_job_detail(job_id: str) -> pd.DataFrame:
    init_job_store()
    with sqlite3.connect(JOB_DB_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT
                job_id,
                file_name,
                file_size_bytes,
                translation_mode,
                source_file_path,
                result_file_path,
                result_mime,
                notify_email,
                notification_status,
                progress_message,
                status,
                total_blocks,
                translatable_blocks,
                completed_blocks,
                total_batches,
                completed_batches,
                input_tokens,
                output_tokens,
                total_tokens,
                result_file_name,
                error_message,
                created_at,
                updated_at,
                finished_at
            FROM translation_jobs
            WHERE job_id = ?
            """,
            conn,
            params=(job_id,),
        )


def latest_running_translation_job_id() -> str:
    init_job_store()
    with sqlite3.connect(JOB_DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT job_id
            FROM translation_jobs
            WHERE status IN ('pending', 'running')
            ORDER BY
                updated_at DESC,
                created_at DESC
            LIMIT 1
            """
        ).fetchone()
    return row[0] if row else ""


def active_translation_job_count() -> int:
    init_job_store()
    with sqlite3.connect(JOB_DB_PATH) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM translation_jobs WHERE status IN ('pending', 'running')"
        ).fetchone()
    return int(row[0] if row else 0)


def translation_job_is_active(job_id: str) -> bool:
    init_job_store()
    with sqlite3.connect(JOB_DB_PATH) as conn:
        row = conn.execute(
            "SELECT status FROM translation_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
    return bool(row and row[0] in {"pending", "running"})


def stop_translation_job(job_id: str, reason: str = "Stopped by user.") -> None:
    update_translation_job(
        job_id,
        status="failed",
        error_message=f"{reason} Saved progress is preserved and can be resumed later.",
        progress_message=reason,
        finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def stop_all_active_translation_jobs(reason: str = "Stopped by user.") -> int:
    init_job_store()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    message = f"{reason} Saved progress is preserved and can be resumed later."
    with sqlite3.connect(JOB_DB_PATH) as conn:
        cursor = conn.execute(
            """
            UPDATE translation_jobs
            SET status = 'failed',
                error_message = ?,
                progress_message = ?,
                finished_at = ?,
                updated_at = ?
            WHERE status IN ('pending', 'running')
            """,
            (message, reason, now, now),
        )
        return cursor.rowcount


def stop_active_translation_jobs_for_file(file_name: str, reason: str, keep_job_id: str = "") -> None:
    init_job_store()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(JOB_DB_PATH) as conn:
        if keep_job_id:
            conn.execute(
                """
                UPDATE translation_jobs
                SET status = 'failed',
                    error_message = ?,
                    progress_message = 'Stopped.',
                    finished_at = ?,
                    updated_at = ?
                WHERE file_name = ?
                  AND job_id != ?
                  AND status IN ('pending', 'running')
                """,
                (reason, now, now, file_name, keep_job_id),
            )
        else:
            conn.execute(
                """
                UPDATE translation_jobs
                SET status = 'failed',
                    error_message = ?,
                    progress_message = 'Stopped.',
                    finished_at = ?,
                    updated_at = ?
                WHERE file_name = ?
                  AND status IN ('pending', 'running')
                """,
                (reason, now, now, file_name),
            )


def init_translation_memory() -> None:
    with sqlite3.connect(TRANSLATION_MEMORY_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS translation_memory (
                source_key TEXT NOT NULL,
                source_text TEXT NOT NULL,
                translation_mode TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (source_key, translation_mode)
            )
            """
        )


def translation_memory_lookup(source_texts: list[str], translation_mode: str) -> dict[str, str]:
    if not source_texts:
        return {}
    init_translation_memory()
    keys = sorted({translation_memory_key(text) for text in source_texts if clean_text(text)})
    if not keys:
        return {}
    found = {}
    with sqlite3.connect(TRANSLATION_MEMORY_DB_PATH) as conn:
        for start in range(0, len(keys), 900):
            key_chunk = keys[start:start + 900]
            placeholders = ",".join("?" for _ in key_chunk)
            rows = conn.execute(
                f"""
                SELECT source_key, translated_text
                FROM translation_memory
                WHERE translation_mode = ? AND source_key IN ({placeholders})
                """,
                [translation_mode, *key_chunk],
            ).fetchall()
            found.update({str(key): str(value) for key, value in rows})
    return found


def save_translation_memory_pairs(pairs: list[tuple[str, str]], translation_mode: str) -> None:
    cleaned_pairs = [
        (translation_memory_key(source), clean_text(source), clean_text(translated))
        for source, translated in pairs
        if clean_text(source) and clean_text(translated)
    ]
    if not cleaned_pairs:
        return
    init_translation_memory()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(TRANSLATION_MEMORY_DB_PATH) as conn:
        conn.executemany(
            """
            INSERT INTO translation_memory (
                source_key, source_text, translation_mode, translated_text, updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_key, translation_mode) DO UPDATE SET
                translated_text = excluded.translated_text,
                updated_at = excluded.updated_at
            """,
            [
                (source_key, source_text, translation_mode, translated_text, now)
                for source_key, source_text, translated_text in cleaned_pairs
            ],
        )


def hydrate_translation_memory_from_checkpoint(
    blocks: list[TextBlock],
    checkpoint_translations: dict[str, str],
    translation_mode: str,
) -> None:
    pairs = [
        (block.text, checkpoint_translations[block.location])
        for block in blocks
        if block.location in checkpoint_translations
    ]
    save_translation_memory_pairs(pairs, translation_mode)


def is_safe_glossary_term(jp: str) -> bool:
    jp = clean_text(jp)
    if len(jp) < 2:
        return False
    if not has_japanese_text(jp):
        return False
    if PROTECTED_PATTERN.fullmatch(jp):
        return False
    return True


def empty_terms_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=["JP", "EN", "Note", "Category", "Owner"])


def xlsx_to_dataframe(raw: bytes, sheet_index: int = 0) -> pd.DataFrame:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    def column_index(cell_ref: str) -> int:
        letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
        index = 0
        for letter in letters:
            index = index * 26 + (ord(letter) - ord("A") + 1)
        return max(index - 1, 0)

    with ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        shared_strings = []

        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", ns):
                shared_strings.append("".join(text.text or "" for text in item.findall(".//a:t", ns)))

        sheet_names = sorted(
            name for name in names if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )
        if not sheet_names:
            raise ValueError("No worksheet found in Excel file.")

        sheet_name = sheet_names[min(sheet_index, len(sheet_names) - 1)]
        root = ET.fromstring(archive.read(sheet_name))
        rows = []

        for row in root.findall("a:sheetData/a:row", ns):
            indexed_values = {}
            for cell in row.findall("a:c", ns):
                if cell.attrib.get("t") == "inlineStr":
                    value = "".join(text.text or "" for text in cell.findall(".//a:t", ns))
                else:
                    value_node = cell.find("a:v", ns)
                    value = "" if value_node is None else value_node.text or ""
                    if cell.attrib.get("t") == "s" and value.isdigit():
                        value = shared_strings[int(value)]
                ref = cell.attrib.get("r", "")
                indexed_values[column_index(ref)] = value
            values = [indexed_values.get(index, "") for index in range(max(indexed_values.keys(), default=-1) + 1)]
            if any(str(value).strip() for value in values):
                rows.append(values)

    if not rows:
        return pd.DataFrame(columns=["JP", "EN"])

    header = [str(value).strip() for value in rows[0]]
    width = len(header)
    normalized_rows = [(row + [""] * width)[:width] for row in rows[1:]]
    return pd.DataFrame(normalized_rows, columns=header)


def read_glossary(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        glossary_path = next((path for path in DEFAULT_GLOSSARY_PATHS if path.exists()), None)
        if glossary_path is None:
            expected = ", ".join(path.name for path in DEFAULT_GLOSSARY_PATHS)
            raise FileNotFoundError(f"No glossary file found. Expected one of: {expected}")
        raw = glossary_path.read_bytes()
        name = glossary_path.name
    else:
        raw = uploaded_file.getvalue()
        name = uploaded_file.name

    is_excel = raw[:2] == b"PK" or name.lower().endswith((".xlsx", ".xlsm", ".xls"))
    if is_excel:
        try:
            return pd.read_excel(io.BytesIO(raw), sheet_name=0)
        except ImportError:
            return xlsx_to_dataframe(raw)
        except BadZipFile as exc:
            raise ValueError("The glossary Excel file could not be opened.") from exc

    for encoding in ("utf-8-sig", "utf-8", "cp932", "shift_jis", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not read glossary file. Please use CSV UTF-8 or Excel format.")


def read_rules_file(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    is_excel = raw[:2] == b"PK" or path.name.lower().endswith((".xlsx", ".xlsm", ".xls"))
    if is_excel:
        try:
            return pd.read_excel(io.BytesIO(raw), sheet_name=0)
        except ImportError:
            return xlsx_to_dataframe(raw)
        except BadZipFile as exc:
            raise ValueError(f"The rule Excel file could not be opened: {path.name}") from exc

    for encoding in ("utf-8-sig", "utf-8", "cp932", "shift_jis", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read rule file: {path.name}")


def read_plc_rules() -> pd.DataFrame:
    rule_path = next((path for path in DEFAULT_PLC_RULE_PATHS if path.exists()), None)
    if rule_path is None:
        return empty_terms_dataframe()
    return read_rules_file(rule_path)


def normalize_glossary(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "jp": "JP",
        "japanese": "JP",
        "japanese term": "JP",
        "日本語": "JP",
        "en": "EN",
        "english": "EN",
        "english term": "EN",
        "英語": "EN",
        "note": "Note",
        "notes": "Note",
        "comment": "Note",
    }
    aliases.update(
        {
            "source japanese": "JP",
            "source": "JP",
            "japanese source": "JP",
            "japanese comment": "JP",
            "source term": "JP",
            "source text": "JP",
            "preferred english": "EN",
            "preferred abbreviation": "EN",
            "preferred en": "EN",
            "approved english": "EN",
            "approved abbreviation": "EN",
            "abbreviation": "EN",
            "standard english": "EN",
            "standard wording": "EN",
            "target": "EN",
            "target english": "EN",
            "do not use": "Note",
        }
    )

    renamed = {}
    for column in df.columns:
        key = str(column).strip().lower()
        renamed[column] = aliases.get(key, str(column).strip())

    glossary = df.rename(columns=renamed).fillna("")
    if "JP" not in glossary.columns or "EN" not in glossary.columns:
        raise ValueError("Glossary must include JP/Japanese and EN/English columns.")

    glossary = glossary[[column for column in ["JP", "EN", "Note", "Category", "Owner"] if column in glossary.columns]]
    glossary = glossary.copy()
    glossary["JP"] = glossary["JP"].astype(str).map(clean_text)
    glossary["EN"] = glossary["EN"].astype(str).map(clean_text)
    glossary = glossary[(glossary["JP"] != "") & (glossary["EN"] != "")]
    glossary = glossary[glossary["JP"].map(is_safe_glossary_term)]
    glossary = glossary.drop_duplicates(subset=["JP"], keep="first")
    glossary["term_length"] = glossary["JP"].str.len()
    return glossary.sort_values("term_length", ascending=False).drop(columns=["term_length"]).reset_index(drop=True)


def normalize_plc_rules(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty and not set(df.columns).intersection({"JP", "EN", "Japanese", "English"}):
        return empty_terms_dataframe()
    try:
        rules = normalize_glossary(df)
    except Exception:
        return empty_terms_dataframe()
    if rules.empty:
        return empty_terms_dataframe()
    rules = rules.copy()
    rules["Category"] = "PLC/SPLC Rule"
    return rules


def glossary_for_mode(glossary: pd.DataFrame, plc_rules: pd.DataFrame, translation_mode: str) -> pd.DataFrame:
    if translation_mode != PLC_TRANSLATION_MODE or plc_rules.empty:
        return glossary

    combined = pd.concat([plc_rules, glossary], ignore_index=True, sort=False).fillna("")
    combined = combined.drop_duplicates(subset=["JP"], keep="first")
    combined["term_length"] = combined["JP"].str.len()
    return combined.sort_values("term_length", ascending=False).drop(columns=["term_length"]).reset_index(drop=True)


def apply_glossary_to_source(text: str, glossary: pd.DataFrame, replace_source: bool = True) -> tuple[str, list[TermHit]]:
    translated_source = clean_text(text)
    hits = []

    for _, row in glossary.iterrows():
        jp = clean_text(row["JP"])
        en = clean_text(row["EN"])
        if not jp or jp not in translated_source:
            continue

        count = len(re.findall(re.escape(jp), translated_source))
        if count:
            hits.append(TermHit(jp=jp, en=en, count=count))
            if replace_source:
                translated_source = re.sub(re.escape(jp), en, translated_source)

    return translated_source, hits


def exact_controlled_term_match(text: str, glossary: pd.DataFrame) -> tuple[str | None, list[TermHit]]:
    source = clean_text(text)
    if not source:
        return None, []

    for _, row in glossary.iterrows():
        jp = clean_text(row["JP"])
        en = clean_text(row["EN"])
        if source == jp:
            return en, [TermHit(jp=jp, en=en, count=1)]
    return None, []


def find_protected_codes(text: str) -> list[str]:
    return sorted(set(PROTECTED_PATTERN.findall(text)) | set(ENCLOSED_ALNUM_PATTERN.findall(text)))


def plc_mode_rules() -> str:
    return """
PLC/SPLC comment mode:
1. Treat the source as PLC/SPLC device comments, HMI labels, alarm labels, or control logic comments, not normal prose.
2. Output short engineering labels only. Avoid full sentences unless the source is clearly a sentence.
3. Do not string together multiple synonyms. Never output lists like "poor, defective, NG, inoperative".
4. Choose one stable plant-control term for each Japanese concept. Prefer concise PLC terms such as ON, OFF, OK, NG, Present, Absent, Complete, Confirm, Request, Command, Auto, Manual, Standby, Error.
5. Preserve PLC addresses, device IDs, robot names, station names, prefixes, symbols, arrows, brackets, and separators exactly.
6. Preserve enclosed/circled markers such as ⓐ, ⓑ, Ⓒ, and ① exactly. Do not change them to plain letters or numbers.
7. Keep repeated Japanese patterns translated with repeated English patterns.
8. If a company glossary term is provided, it overrides the default PLC wording.
""".strip()


def general_mode_rules() -> str:
    return """
General plant translation mode:
1. Translate into clear, natural plant-floor engineering English for manufacturing users.
2. Use concise plant-floor English suitable for controls, seibi, production, and engineering users.
3. Preserve line breaks and list structure when useful.
""".strip()


def supplier_email_mode_rules() -> str:
    return """
Supplier email translation mode:
1. Translate into natural, professional business English for a manufacturing technical email.
2. Keep names, company names, signal names, PLC addresses, HMI terms, ladder terms, and quoted alarm names accurate.
3. Use normal business email phrasing for standard Japanese email expressions:
   - inquiry-opening phrases should become "Regarding your inquiry"
   - relationship/polite opening phrases should become "Thank you for your continued support"
   - attachment-reference phrases should become "as shown in the attached document"
   - "also check/confirm together with" phrases should become "please also confirm" or "please confirm together with"
   - soft opinion phrases should become "it would likely be appropriate" or "I believe"
4. Do not over-translate polite Japanese. Keep the tone clear, respectful, and practical.
5. Prefer natural phrases such as "a specified period of time" for fixed-duration wording and "waiting for aging completion" for aging-completion-wait wording when context fits.
""".strip()


def product_catalog_mode_rules() -> str:
    return """
Product catalog translation mode:
1. Translate into concise, polished technical catalog English for product catalogs, specifications, case studies, and marketing-technical brochures.
2. Preserve product names, model names, belt types, pulley types, page references, catalog numbers, dimensions, symbols, registered marks, and part numbers exactly unless the source clearly requires translation.
3. Keep headings, table labels, captions, index entries, and callout labels short. Do not expand short catalog labels into long sentences.
4. Preserve line breaks and compact list/table structure where possible. Avoid merging unrelated headings, page references, and descriptions.
5. Use natural product English, but do not add marketing claims, features, applications, or recommendations that are not in the source.
6. If a glossary term has multiple English choices, choose one context-appropriate term. Never output multiple alternatives separated by commas or slashes.
7. Prefer concise phrases such as "Application", "Features", "Technical Data", "Dimension Tolerance", "Surface Roughness", and "Case Study" when context fits.
""".strip()


def robot_program_mode_rules() -> str:
    return """
Robot program comment translation mode:
1. Treat the source as Kawasaki/industrial robot program comments or operator messages, not normal prose.
2. Translate only the Japanese comment meaning into concise engineering English.
3. Preserve robot commands, variables, positions, labels, numbers, symbols, punctuation, and code-style wording exactly when they appear.
4. Keep the translation short enough to fit inside a robot program comment when possible.
5. Do not add explanations, troubleshooting advice, or programming changes.
6. Use stable plant-floor terms such as Home Position, Workpiece, Clamp, Unclamp, Pick, Place, Start, Stop, Complete, Error, and Check when context fits.
""".strip()


def mode_rules_for(translation_mode: str) -> str:
    if translation_mode == PLC_TRANSLATION_MODE:
        return plc_mode_rules()
    if translation_mode == SUPPLIER_EMAIL_TRANSLATION_MODE:
        return supplier_email_mode_rules()
    if translation_mode == PRODUCT_CATALOG_TRANSLATION_MODE:
        return product_catalog_mode_rules()
    if translation_mode == ROBOT_PROGRAM_TRANSLATION_MODE:
        return robot_program_mode_rules()
    return general_mode_rules()


def normalize_plc_translation_line(line: str) -> str:
    normalized = re.sub(r"\s+", " ", line).strip()
    for pattern, replacement in PLC_SYNONYM_CLEANUPS:
        normalized = pattern.sub(replacement, normalized)

    changed = True
    while changed:
        changed = False
        for word in PLC_DUPLICATE_STATUS_WORDS:
            pattern = re.compile(rf"\b({re.escape(word)})\s+\1\b", re.IGNORECASE)
            normalized, count = pattern.subn(r"\1", normalized)
            if count:
                changed = True

    return normalized


def post_process_translation(output_text: str, translation_mode: str) -> str:
    cleaned = output_text.strip()
    if translation_mode != PLC_TRANSLATION_MODE:
        return cleaned
    return "\n".join(
        normalize_plc_translation_line(line)
        for line in cleaned.splitlines()
    ).strip()


def restore_missing_enclosed_markers(source_text: str, translated_text: str) -> str:
    source_markers = list(dict.fromkeys(ENCLOSED_ALNUM_PATTERN.findall(source_text)))
    if not source_markers:
        return translated_text

    restored = translated_text
    for marker in source_markers:
        if marker in restored:
            continue
        if source_text.startswith(marker):
            restored = marker + restored
        elif source_text.rstrip().endswith(marker):
            restored = restored.rstrip() + marker
        else:
            restored = restored.rstrip() + marker
    return restored


INSTRUCTION_LINE_PATTERNS = [
    re.compile(r"^\s*工場で働く人(?:のため|向け)?に訳して[。｡.!！]?\s*$"),
    re.compile(r"^\s*現場(?:の人|作業者)?(?:のため|向け)?に訳して[。｡.!！]?\s*$"),
    re.compile(r"^\s*製造現場(?:の人|作業者)?(?:のため|向け)?に訳して[。｡.!！]?\s*$"),
]
INSTRUCTION_SUFFIX_PATTERNS = [
    re.compile(r"\s*工場で働く人(?:のため|向け)?に訳して[。｡.!！]?\s*$"),
    re.compile(r"\s*現場(?:の人|作業者)?(?:のため|向け)?に訳して[。｡.!！]?\s*$"),
    re.compile(r"\s*製造現場(?:の人|作業者)?(?:のため|向け)?に訳して[。｡.!！]?\s*$"),
]


def split_text_translation_input(text: str) -> tuple[str, str]:
    source_lines = []
    guidance = []
    for line in str(text).splitlines():
        stripped = line.strip()
        if any(pattern.match(stripped) for pattern in INSTRUCTION_LINE_PATTERNS):
            guidance.append("Translate for people working on the manufacturing floor.")
            continue
        for pattern in INSTRUCTION_SUFFIX_PATTERNS:
            line, count = pattern.subn("", line)
            if count:
                guidance.append("Translate for people working on the manufacturing floor.")
                break
        source_lines.append(line)

    source_text = "\n".join(source_lines).strip()
    return source_text, " ".join(dict.fromkeys(guidance))


def build_prompt(
    source_text: str,
    hits: list[TermHit],
    protected_codes: list[str],
    translation_mode: str,
    user_guidance: str = "",
) -> str:
    terms = "\n".join(f"{hit.jp} = {hit.en}" for hit in hits)
    codes = ", ".join(protected_codes) if protected_codes else "None detected"
    mode_rules = mode_rules_for(translation_mode)
    guidance = user_guidance.strip() or "None"

    return f"""
You are a professional Japanese-to-English translator for a battery manufacturing plant.

Translation mode: {translation_mode}

{mode_rules}

Mandatory rules:
1. Use approved company glossary terms when they are precise technical terms.
2. If a glossary English value contains multiple choices or synonyms, choose the single best English wording for the context. Never output a list such as "alignment, fitting, fit issue" or "indication, marking, display, illustration".
3. Do not force a glossary term when it would mistranslate a normal business phrase. For example, standard inquiry-opening phrases should become "Regarding your inquiry".
4. Preserve PLC addresses, device IDs, model names, station IDs, alarm codes, part numbers, and equipment codes exactly.
5. Do not invent missing information, causes, actions, measurements, or context that is not in the source.
6. If the source is ambiguous, translate only the meaning that is present and keep the wording neutral.
7. Output only the English translation. Do not add explanations, notes, or commentary.

Company terminology detected in the source:
{terms if terms else "None"}

Protected codes detected:
{codes}

Additional translation guidance:
{guidance}

Source Japanese text:
{source_text}
""".strip()


def openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY was not found. Add it to app.env for local development "
            "or configure it as a secret/environment variable in the deployment environment."
        )
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)

    tracing_enabled = os.getenv("LANGSMITH_TRACING", "").lower() == "force"
    if tracing_enabled and wrap_openai is not None:
        return wrap_openai(client)
    return client


def openai_model() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def openai_timeout_seconds() -> float:
    try:
        return float(os.getenv("OPENAI_TIMEOUT_SECONDS", OPENAI_TIMEOUT_SECONDS))
    except ValueError:
        return float(OPENAI_TIMEOUT_SECONDS)


def azure_translator_configured() -> bool:
    return bool(os.getenv("AZURE_TRANSLATOR_KEY"))


def azure_translate_texts(texts: list[str]) -> list[str]:
    key = os.getenv("AZURE_TRANSLATOR_KEY", "")
    endpoint = os.getenv("AZURE_TRANSLATOR_ENDPOINT", "https://api.cognitive.microsofttranslator.com")
    region = os.getenv("AZURE_TRANSLATOR_REGION", "")
    if not key:
        raise RuntimeError("Azure Translator is not configured.")

    route = "/translate?" + urllib.parse.urlencode({"api-version": "3.0", "from": "ja", "to": "en"})
    request = urllib.request.Request(
        endpoint.rstrip("/") + route,
        data=json.dumps([{"Text": text} for text in texts], ensure_ascii=False).encode("utf-8"),
        headers={
            "Ocp-Apim-Subscription-Key": key,
            "Ocp-Apim-Subscription-Region": region,
            "Content-Type": "application/json",
            "X-ClientTraceId": str(uuid.uuid4()),
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=openai_timeout_seconds()) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [item["translations"][0]["text"] for item in payload]


def machine_translate_texts(texts: list[str]) -> list[str] | None:
    if not azure_translator_configured():
        return None
    translated = []
    for start in range(0, len(texts), 100):
        translated.extend(azure_translate_texts(texts[start:start + 100]))
    return translated


def max_parallel_batches() -> int:
    try:
        value = int(os.getenv("MAX_PARALLEL_BATCHES", MAX_PARALLEL_BATCHES))
    except ValueError:
        value = MAX_PARALLEL_BATCHES
    return max(value, 1)


def response_token_usage(response) -> TokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()

    def usage_value(*names: str) -> int:
        for name in names:
            value = getattr(usage, name, None)
            if value is not None:
                return int(value)
        if isinstance(usage, dict):
            for name in names:
                value = usage.get(name)
                if value is not None:
                    return int(value)
        return 0

    input_tokens = usage_value("input_tokens", "prompt_tokens")
    output_tokens = usage_value("output_tokens", "completion_tokens")
    total_tokens = usage_value("total_tokens")
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def format_translation_error(exc: Exception) -> str:
    if isinstance(exc, APIConnectionError):
        detail = str(exc.__cause__ or exc)
        if "CERTIFICATE_VERIFY_FAILED" in detail or "certificate verify failed" in detail:
            return (
                "SSL certificate verification failed while calling the OpenAI API. "
                "This usually means the company proxy/VPN is inspecting HTTPS and Python does not trust the company root certificate. "
                "Ask IT for the corporate root CA and set SSL_CERT_FILE or REQUESTS_CA_BUNDLE to that certificate file."
            )
        return (
            "Connection error while calling the GPT API. Check whether OPENAI_BASE_URL is required "
            "for the company API, and confirm VPN/proxy/firewall access from this computer."
        )
    if isinstance(exc, AuthenticationError):
        return "Authentication failed. Check that OPENAI_API_KEY is correct and approved for this endpoint."
    if isinstance(exc, RateLimitError):
        return "Rate limit or quota exceeded. Check the API quota, rate limit, or billing/usage policy."
    if isinstance(exc, APIStatusError):
        return f"GPT API returned HTTP {exc.status_code}. {exc.message}"
    return str(exc)


def translate_text(
    source_text: str,
    hits: list[TermHit],
    protected_codes: list[str],
    translation_mode: str,
    user_guidance: str = "",
) -> tuple[str, TokenUsage]:
    client = openai_client()
    response = client.responses.create(
        model=openai_model(),
        input=build_prompt(source_text, hits, protected_codes, translation_mode, user_guidance),
        temperature=0.1,
        timeout=openai_timeout_seconds(),
    )
    return post_process_translation(response.output_text, translation_mode), response_token_usage(response)


def translate_block(
    text: str,
    glossary: pd.DataFrame,
    translation_mode: str,
    user_guidance: str = "",
) -> tuple[str, list[TermHit], TokenUsage]:
    exact_translation, exact_hits = exact_controlled_term_match(text, glossary)
    if exact_translation is not None:
        translation = post_process_translation(exact_translation, translation_mode)
        return restore_missing_enclosed_markers(text, translation), exact_hits, TokenUsage()

    glossary_applied_text, hits = apply_glossary_to_source(
        text,
        glossary,
        replace_source=translation_mode == PLC_TRANSLATION_MODE,
    )
    protected_codes = find_protected_codes(text)
    translation, token_usage = translate_text(
        glossary_applied_text,
        hits,
        protected_codes,
        translation_mode,
        user_guidance,
    )
    return restore_missing_enclosed_markers(text, translation), hits, token_usage


def build_batch_prompt(items: list[tuple[int, str, list[TermHit], list[str]]], translation_mode: str) -> str:
    item_text = "\n\n".join(
        "\n".join(
            [
                f"[BLOCK {item_id}]",
                source_text,
                f"[/BLOCK {item_id}]",
            ]
        )
        for item_id, source_text, _, _ in items
    )
    terms = []
    codes = []
    for _, _, hits, protected_codes in items:
        terms.extend(f"{hit.jp} = {hit.en}" for hit in hits)
        codes.extend(protected_codes)

    unique_terms = "\n".join(sorted(set(terms))) or "None"
    unique_codes = ", ".join(sorted(set(codes))) if codes else "None detected"
    mode_rules = mode_rules_for(translation_mode)

    return f"""
You are a professional Japanese-to-English translator for a battery manufacturing plant.

Translation mode: {translation_mode}

{mode_rules}

Mandatory rules:
1. Use approved company glossary terms when they are precise technical terms.
2. If a glossary English value contains multiple choices or synonyms, choose the single best English wording for the context. Never output a list such as "alignment, fitting, fit issue" or "indication, marking, display, illustration".
3. Do not force a glossary term when it would mistranslate a normal business phrase. For example, standard inquiry-opening phrases should become "Regarding your inquiry".
4. Preserve PLC addresses, device IDs, model names, station IDs, alarm codes, part numbers, and equipment codes exactly.
5. Do not invent missing information, causes, actions, measurements, or context that is not in the source.
6. If the source is ambiguous, translate only the meaning that is present and keep the wording neutral.
7. Return each translated block using the same markers and do not add explanations, notes, or commentary:
[BLOCK 1]
English translation
[/BLOCK 1]

Company terminology detected:
{unique_terms}

Protected codes detected:
{unique_codes}

Source blocks:
{item_text}
""".strip()


def parse_batch_translation(output_text: str, item_ids: list[int]) -> dict[int, str]:
    translations = {}
    for item_id in item_ids:
        pattern = re.compile(
            rf"\[BLOCK {item_id}\]\s*(.*?)\s*\[/BLOCK {item_id}\]",
            re.DOTALL,
        )
        match = pattern.search(output_text)
        if match:
            translations[item_id] = match.group(1).strip()
    return translations


def translate_batch_chunk(chunk: list[TextBlock], glossary: pd.DataFrame, translation_mode: str) -> tuple[dict[str, str], list[TermHit], TokenUsage]:
    items = []
    chunk_hits = []
    direct_translations = {}

    for offset, block in enumerate(chunk, start=1):
        exact_translation, exact_hits = exact_controlled_term_match(block.text, glossary)
        if exact_translation is not None:
            translation = post_process_translation(exact_translation, translation_mode)
            direct_translations[block.location] = restore_missing_enclosed_markers(block.text, translation)
            chunk_hits.extend(exact_hits)
            continue

        glossary_applied_text, hits = apply_glossary_to_source(
            block.text,
            glossary,
            replace_source=translation_mode == PLC_TRANSLATION_MODE,
        )
        protected_codes = find_protected_codes(block.text)
        items.append((offset, glossary_applied_text, hits, protected_codes))
        chunk_hits.extend(hits)

    parsed = {}
    token_usage = TokenUsage()
    last_error = None
    if items:
        machine_translations = None
        try:
            machine_translations = machine_translate_texts([item[1] for item in items])
        except Exception as exc:
            last_error = exc

        if machine_translations is not None:
            parsed = {
                item[0]: post_process_translation(translated, translation_mode)
                for item, translated in zip(items, machine_translations)
            }
        else:
            client = openai_client()
            for attempt in range(1, MAX_TRANSLATION_RETRIES + 1):
                try:
                    response = client.responses.create(
                        model=openai_model(),
                        input=build_batch_prompt(items, translation_mode),
                        temperature=0.1,
                        timeout=openai_timeout_seconds(),
                    )
                    token_usage.add(response_token_usage(response))
                    parsed = parse_batch_translation(response.output_text.strip(), [item[0] for item in items])
                    missing_ids = [item[0] for item in items if item[0] not in parsed]
                    if missing_ids:
                        raise ValueError(f"Translation response missed block marker(s): {missing_ids}")
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt == MAX_TRANSLATION_RETRIES:
                        raise
                    time.sleep(5 * attempt)

    if items and not parsed and last_error:
        raise last_error

    chunk_translations = dict(direct_translations)
    for offset, block in enumerate(chunk, start=1):
        if block.location in chunk_translations:
            continue
        translation = post_process_translation(parsed[offset], translation_mode)
        chunk_translations[block.location] = restore_missing_enclosed_markers(block.text, translation)

    return chunk_translations, chunk_hits, token_usage


def translate_batch_chunk_resilient(
    chunk: list[TextBlock],
    glossary: pd.DataFrame,
    translation_mode: str,
) -> tuple[dict[str, str], list[TermHit], TokenUsage]:
    try:
        return translate_batch_chunk(chunk, glossary, translation_mode)
    except Exception:
        if len(chunk) <= 1:
            raise

    midpoint = max(len(chunk) // 2, 1)
    combined_translations = {}
    combined_hits = []
    combined_usage = TokenUsage()
    for part in (chunk[:midpoint], chunk[midpoint:]):
        part_translations, part_hits, part_usage = translate_batch_chunk_resilient(
            part,
            glossary,
            translation_mode,
        )
        combined_translations.update(part_translations)
        combined_hits.extend(part_hits)
        combined_usage.add(part_usage)
    return combined_translations, combined_hits, combined_usage


def translation_memory_key(text: str) -> str:
    return clean_text(text)


def translate_blocks_batch(
    blocks: list[TextBlock],
    glossary: pd.DataFrame,
    translation_mode: str,
    checkpoint_path=None,
    progress_callback=None,
    should_continue=None,
) -> tuple[dict[str, str], list[TermHit], TokenUsage]:
    def ensure_continue() -> None:
        if should_continue is not None and not should_continue():
            raise RuntimeError("Translation job was stopped.")

    translations = load_checkpoint(checkpoint_path) if checkpoint_path else {}
    translatable_blocks = [block for block in blocks if should_translate(block.text)]
    hydrate_translation_memory_from_checkpoint(blocks, translations, translation_mode)
    source_by_key = {}
    for block in translatable_blocks:
        source_by_key.setdefault(translation_memory_key(block.text), block.text)
    memory_hits = translation_memory_lookup(list(source_by_key.values()), translation_mode)
    memory_applied = 0
    for block in translatable_blocks:
        if block.location in translations:
            continue
        memory_translation = memory_hits.get(translation_memory_key(block.text))
        if memory_translation:
            translations[block.location] = memory_translation
            memory_applied += 1
    if memory_applied and checkpoint_path:
        save_checkpoint(checkpoint_path, translations)
    source_memory = {}
    for block in translatable_blocks:
        if block.location in translations:
            source_memory.setdefault(translation_memory_key(block.text), translations[block.location])

    for block in translatable_blocks:
        key = translation_memory_key(block.text)
        if block.location not in translations and key in source_memory:
            translations[block.location] = source_memory[key]

    pending_by_key = {}
    duplicate_locations_by_key = {}
    for block in translatable_blocks:
        if block.location in translations:
            continue
        key = translation_memory_key(block.text)
        if key not in pending_by_key:
            pending_by_key[key] = block
            duplicate_locations_by_key[key] = []
        duplicate_locations_by_key[key].append(block.location)

    pending_blocks = list(pending_by_key.values())
    all_hits = []
    token_usage = TokenUsage()
    started_at = time.time()
    pending_location_count = sum(len(locations) for locations in duplicate_locations_by_key.values())
    completed_at_start = len(translatable_blocks) - pending_location_count
    total_batches = (len(pending_blocks) + DOCUMENT_BATCH_SIZE - 1) // DOCUMENT_BATCH_SIZE
    parallel_batches = min(max_parallel_batches(), max(total_batches, 1))

    ensure_continue()
    if progress_callback:
        progress_callback(
            completed_at_start,
            len(translatable_blocks),
            0,
            total_batches,
            0,
            "Resuming translation." if completed_at_start else "Starting translation.",
        )

    chunks = [
        pending_blocks[start:start + DOCUMENT_BATCH_SIZE]
        for start in range(0, len(pending_blocks), DOCUMENT_BATCH_SIZE)
    ]
    completed_batches = 0
    completed_blocks = completed_at_start

    if chunks:
        ensure_continue()
        if progress_callback:
            progress_callback(
                completed_blocks,
                len(translatable_blocks),
                0,
                total_batches,
                time.time() - started_at,
                f"Translating with {parallel_batches} parallel workers.",
            )
        with ThreadPoolExecutor(max_workers=parallel_batches) as executor:
            chunk_index = 0
            future_to_chunk = {}

            def submit_until_full() -> None:
                nonlocal chunk_index
                while len(future_to_chunk) < parallel_batches and chunk_index < len(chunks):
                    ensure_continue()
                    chunk = chunks[chunk_index]
                    chunk_index += 1
                    future_to_chunk[
                        executor.submit(translate_batch_chunk_resilient, chunk, glossary, translation_mode)
                    ] = chunk

            submit_until_full()
            while future_to_chunk:
                done_futures, _ = wait(future_to_chunk, return_when=FIRST_COMPLETED)
                for future in done_futures:
                    chunk = future_to_chunk.pop(future)
                    ensure_continue()
                    chunk_translations, chunk_hits, chunk_token_usage = future.result()
                    expanded_translations = {}
                    for block in chunk:
                        key = translation_memory_key(block.text)
                        translated_text = chunk_translations[block.location]
                        source_memory[key] = translated_text
                        for location in duplicate_locations_by_key.get(key, [block.location]):
                            expanded_translations[location] = translated_text

                    translations.update(expanded_translations)
                    save_translation_memory_pairs(
                        [
                            (block.text, chunk_translations[block.location])
                            for block in chunk
                            if block.location in chunk_translations
                        ],
                        translation_mode,
                    )
                    all_hits.extend(chunk_hits)
                    token_usage.add(chunk_token_usage)
                    completed_batches += 1
                    completed_blocks += len(expanded_translations)

                    if checkpoint_path:
                        save_checkpoint(checkpoint_path, translations)

                    if progress_callback:
                        progress_callback(
                            min(completed_blocks, len(translatable_blocks)),
                            len(translatable_blocks),
                            completed_batches,
                            total_batches,
                            time.time() - started_at,
                            "Translating",
                        )
                submit_until_full()

    return translations, all_hits, token_usage


def read_text_file(raw: bytes) -> str:
    return decode_document_text(raw)


def output_translation_for(
    location: str,
    source_text: str,
    translations: dict[str, str],
    keep_source_with_translation: bool = False,
) -> str:
    translated = translations.get(location, source_text)
    translated = restore_missing_enclosed_markers(source_text, translated)
    if not keep_source_with_translation or clean_text(translated) == clean_text(source_text):
        return translated
    return source_with_translation_lines(source_text, translated)


def source_with_translation_lines(source_text: str, translated_text: str) -> str:
    source_lines = str(source_text).splitlines()
    translated_lines = str(translated_text).splitlines()
    if len(source_lines) <= 1 or len(translated_lines) <= 1:
        return f"{source_text}\n{translated_text}"

    paired_lines = []
    max_lines = max(len(source_lines), len(translated_lines))
    for index in range(max_lines):
        if index < len(source_lines) and source_lines[index].strip():
            paired_lines.append(source_lines[index])
        if index < len(translated_lines) and translated_lines[index].strip():
            paired_lines.append(translated_lines[index])
    return "\n".join(paired_lines)


def write_text_file(
    blocks: list[TextBlock],
    translations: dict[str, str],
    keep_source_with_translation: bool = False,
) -> bytes:
    lines = []
    for block in blocks:
        lines.append(output_translation_for(block.location, block.text, translations, keep_source_with_translation))
    return "\n\n".join(lines).encode("utf-8-sig")


def robot_comment_segments(text: str) -> list[tuple[int, int, str]]:
    segments = []
    line_start = 0
    paired_comment_pattern = re.compile(r";([^;\r\n]*[\u3040-\u30ff\u3400-\u9fff][^;\r\n]*);")
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        paired_matches = list(paired_comment_pattern.finditer(content))
        if paired_matches:
            for match in paired_matches:
                value = match.group(1).strip()
                if looks_like_mojibake(value):
                    continue
                segments.append((
                    line_start + match.start(1),
                    line_start + match.end(1),
                    value,
                ))
        else:
            comment_start = content.find(";")
            if comment_start >= 0:
                comment_text = content[comment_start + 1:]
                if has_japanese_text(comment_text) and not looks_like_mojibake(comment_text):
                    segments.append((
                        line_start + comment_start + 1,
                        line_start + len(content),
                        comment_text.strip(),
                    ))
        line_start += len(line)
    return segments


def robot_quoted_string_segments(text: str) -> list[tuple[int, int, str]]:
    segments = []
    for match in re.finditer(r'"([^"\r\n]*[\u3040-\u30ff\u3400-\u9fff][^"\r\n]*)"', text):
        value = match.group(1).strip()
        if value and not looks_like_mojibake(value):
            segments.append((match.start(1), match.end(1), value))
    return segments


def extract_robot_program_blocks(raw: bytes) -> list[TextBlock]:
    text, _encoding = decode_robot_program_text_with_encoding(raw)
    blocks = []
    for start, end, value in robot_comment_segments(text):
        if value:
            blocks.append(TextBlock(location=f"robot_comment:{start}:{end}", text=value))
    for start, end, value in robot_quoted_string_segments(text):
        if value:
            blocks.append(TextBlock(location=f"robot_string:{start}:{end}", text=value))
    return blocks


def build_translated_robot_program(
    raw: bytes,
    translations: dict[str, str],
    source_by_location: dict[str, str],
    keep_source_with_translation: bool = False,
) -> bytes:
    text, encoding = decode_robot_program_text_with_encoding(raw)
    replacements = []
    for location, source_text in source_by_location.items():
        if not location.startswith(("robot_comment:", "robot_string:")) or location not in translations:
            continue
        _prefix, start_text, end_text = location.split(":", 2)
        replacement = output_translation_for(
            location,
            source_text,
            translations,
            keep_source_with_translation,
        )
        if keep_source_with_translation:
            replacement = replacement.replace("\r\n", " / ").replace("\n", " / ").replace("\r", " / ")
        replacements.append((int(start_text), int(end_text), replacement))

    for start, end, replacement in sorted(replacements, reverse=True):
        text = text[:start] + replacement + text[end:]

    try:
        return text.encode(encoding)
    except UnicodeEncodeError:
        return text.encode("utf-8-sig")


def sniff_csv_dialect(text: str):
    sample = "\n".join(text.splitlines()[:100])
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except csv.Error:
        return csv.excel


def split_leading_code_cell(value: str) -> list[str]:
    cleaned = clean_text(value)
    match = LEADING_CODE_PATTERN.match(cleaned)
    if not match:
        return [value]
    code, remainder = match.groups()
    remainder = clean_text(remainder)
    if not remainder:
        return [value]
    return [code, remainder]


def normalize_csv_structure(rows: list[list[str]]) -> list[list[str]]:
    normalized = []
    for row in rows:
        if len(row) == 1:
            normalized.append(split_leading_code_cell(row[0]))
        else:
            normalized.append(row)
    return normalized


def read_csv_rows(raw: bytes) -> list[list[str]]:
    text = decode_document_text(raw)
    return parse_csv_rows_lenient(text)


def parse_csv_rows_lenient(text: str) -> list[list[str]]:
    if "�" in text and not has_japanese_text(text):
        raise ValueError(
            "The CSV text could not be decoded into readable Japanese. "
            "Please save the CSV as UTF-8 CSV or Excel .xlsx, then upload again."
        )
    dialect = sniff_csv_dialect(text)
    try:
        return normalize_csv_structure([row for row in csv.reader(io.StringIO(text, newline=""), dialect)])
    except csv.Error:
        rows = []
        for line in text.splitlines():
            try:
                rows.append(next(csv.reader([line], dialect)))
            except csv.Error:
                rows.append([line])
        return normalize_csv_structure(rows)


def extract_csv_blocks(raw: bytes) -> list[TextBlock]:
    rows = read_csv_rows(raw)
    blocks = []
    for row_index, row in enumerate(rows):
        for column_index, cell in enumerate(row):
            value = clean_text(cell)
            if value:
                blocks.append(TextBlock(location=f"csv:{row_index}:{column_index}", text=value))
    return blocks


def build_translated_csv(
    raw: bytes,
    translations: dict[str, str],
    source_by_location: dict[str, str],
    keep_source_with_translation: bool = False,
) -> bytes:
    rows = read_csv_rows(raw)
    for row_index, row in enumerate(rows):
        for column_index, cell in enumerate(row):
            key = f"csv:{row_index}:{column_index}"
            if key in translations:
                row[column_index] = output_translation_for(
                    key,
                    source_by_location.get(key, clean_text(cell)),
                    translations,
                    keep_source_with_translation,
                )

    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def extract_text_blocks(raw: bytes, file_name: str) -> list[TextBlock]:
    lower_name = file_name.lower()
    if lower_name.endswith((".as", ".ad")):
        return extract_robot_program_blocks(raw)

    if lower_name.endswith(".txt"):
        text = read_text_file(raw)
        return [TextBlock(location=f"text:{index}", text=part.strip()) for index, part in enumerate(text.split("\n\n")) if part.strip()]

    if lower_name.endswith(".csv"):
        return extract_csv_blocks(raw)

    if lower_name.endswith(".docx"):
        return extract_docx_blocks(raw)

    if lower_name.endswith(".pptx"):
        return extract_pptx_blocks(raw)

    if lower_name.endswith((".xlsx", ".xlsm")):
        return extract_xlsx_blocks(raw)

    raise ValueError("Supported document types: CSV, TXT, AS, AD, DOCX, PPTX, XLSX, XLSM.")


def no_blocks_error_message(file_name: str) -> str:
    if file_name.lower().endswith((".as", ".ad")):
        return (
            "No Japanese text was found inside semicolon-delimited robot comments. "
            "For AS/AD files, this app translates only Japanese text between semicolons, such as ;搬送開始;."
        )
    return (
        "No translatable text was found in this document. "
        "Upload a CSV, TXT, AS, AD, DOCX, PPTX, XLSX, or XLSM file with selectable text."
    )


def extract_docx_blocks(raw: bytes) -> list[TextBlock]:
    blocks = []
    with ZipFile(io.BytesIO(raw)) as archive:
        xml_names = [
            name for name in archive.namelist()
            if name == "word/document.xml" or name.startswith("word/header") or name.startswith("word/footer")
        ]
        for xml_name in xml_names:
            root = ET.fromstring(archive.read(xml_name))
            for index, paragraph in enumerate(root.findall(".//w:p", WORD_NS)):
                text = "".join(node.text or "" for node in paragraph.findall(".//w:t", WORD_NS)).strip()
                if text:
                    blocks.append(TextBlock(location=f"{xml_name}#{index}", text=text))
    return blocks


def extract_pptx_blocks(raw: bytes) -> list[TextBlock]:
    blocks = []
    with ZipFile(io.BytesIO(raw)) as archive:
        slide_names = sorted(
            name for name in archive.namelist()
            if name.startswith("ppt/slides/slide") and name.endswith(".xml")
        )
        for slide_name in slide_names:
            root = ET.fromstring(archive.read(slide_name))
            for index, paragraph in enumerate(root.findall(".//a:p", PPT_NS)):
                text = "".join(node.text or "" for node in paragraph.findall(".//a:t", PPT_NS)).strip()
                if text:
                    blocks.append(TextBlock(location=f"{slide_name}#{index}", text=text))
    return blocks


def read_shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []

    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    strings = []
    for item in root.findall("a:si", EXCEL_NS):
        strings.append("".join(text.text or "" for text in item.findall(".//a:t", EXCEL_NS)))
    return strings


def cell_text(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//a:t", EXCEL_NS)).strip()

    value_node = cell.find("a:v", EXCEL_NS)
    if value_node is None or value_node.text is None:
        return ""

    value = value_node.text
    if cell_type == "s" and value.isdigit():
        index = int(value)
        if 0 <= index < len(shared_strings):
            return shared_strings[index].strip()

    if cell_type in {"str", "e"}:
        return value.strip()

    return ""


def extract_xlsx_blocks(raw: bytes) -> list[TextBlock]:
    blocks = []
    try:
        workbook = load_workbook(io.BytesIO(raw), read_only=True, data_only=False)
        for sheet_index, worksheet in enumerate(workbook.worksheets, start=1):
            sheet_name = f"xl/worksheets/sheet{sheet_index}.xml"
            for row in worksheet.iter_rows():
                for cell in row:
                    value = cell.value
                    if isinstance(value, str) and value.strip():
                        blocks.append(TextBlock(location=f"{sheet_name}#{cell.coordinate}", text=clean_text(value)))
        workbook.close()
        if blocks:
            return blocks
    except Exception:
        blocks = []

    with ZipFile(io.BytesIO(raw)) as archive:
        shared_strings = read_shared_strings(archive)
        sheet_names = sorted(
            name for name in archive.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )

        for sheet_name in sheet_names:
            root = ET.fromstring(archive.read(sheet_name))
            for cell in root.findall(".//a:c", EXCEL_NS):
                ref = cell.attrib.get("r", "")
                text = cell_text(cell, shared_strings)
                if text:
                    blocks.append(TextBlock(location=f"{sheet_name}#{ref}", text=text))
    return blocks


def first_run_with_text(paragraph: ET.Element) -> ET.Element | None:
    for run in paragraph.findall(".//w:r", WORD_NS):
        if run.findall(".//w:t", WORD_NS):
            return run
    return None


def replace_text_in_paragraph(paragraph: ET.Element, text_nodes: list[ET.Element], translated_text: str) -> None:
    if not text_nodes:
        return
    lines = str(translated_text).splitlines() or [""]
    text_nodes[0].text = clean_office_xml_text(lines[0])
    for node in text_nodes[1:]:
        node.text = ""
    if len(lines) <= 1:
        return

    first_run = first_run_with_text(paragraph)
    if first_run is None:
        return
    for line in lines[1:]:
        ET.SubElement(first_run, f"{{{WORD_NS['w']}}}br")
        text_node = ET.SubElement(first_run, f"{{{WORD_NS['w']}}}t")
        text_node.text = clean_office_xml_text(line)


def build_translated_document(
    raw: bytes,
    file_name: str,
    translations: dict[str, str],
    blocks: list[TextBlock],
    keep_source_with_translation: bool = False,
) -> bytes:
    lower_name = file_name.lower()
    source_by_location = {block.location: block.text for block in blocks}
    if lower_name.endswith((".as", ".ad")):
        return build_translated_robot_program(raw, translations, source_by_location, keep_source_with_translation)

    if lower_name.endswith(".txt"):
        return write_text_file(blocks, translations, keep_source_with_translation)

    if lower_name.endswith(".csv"):
        return build_translated_csv(raw, translations, source_by_location, keep_source_with_translation)

    if lower_name.endswith(".docx"):
        return build_translated_docx(raw, translations, source_by_location, keep_source_with_translation)

    if lower_name.endswith(".pptx"):
        return build_translated_pptx(raw, translations)

    if lower_name.endswith((".xlsx", ".xlsm")):
        return build_translated_xlsx(raw, translations, source_by_location, keep_source_with_translation)

    raise ValueError("Supported document types: CSV, TXT, AS, AD, DOCX, PPTX, XLSX, XLSM.")


def build_translated_docx(
    raw: bytes,
    translations: dict[str, str],
    source_by_location: dict[str, str],
    keep_source_with_translation: bool = False,
) -> bytes:
    source = io.BytesIO(raw)
    target = io.BytesIO()

    with ZipFile(source) as input_zip, ZipFile(target, "w", ZIP_DEFLATED) as output_zip:
        for item in input_zip.infolist():
            data = input_zip.read(item.filename)
            if item.filename == "word/document.xml" or item.filename.startswith("word/header") or item.filename.startswith("word/footer"):
                root = ET.fromstring(data)
                for index, paragraph in enumerate(root.findall(".//w:p", WORD_NS)):
                    key = f"{item.filename}#{index}"
                    if key in translations:
                        translated = output_translation_for(
                            key,
                            source_by_location.get(key, ""),
                            translations,
                            keep_source_with_translation,
                        )
                        replace_text_in_paragraph(paragraph, paragraph.findall(".//w:t", WORD_NS), translated)
                data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
            output_zip.writestr(item, data)

    return target.getvalue()


def build_translated_pptx(raw: bytes, translations: dict[str, str]) -> bytes:
    source = io.BytesIO(raw)
    target = io.BytesIO()

    with ZipFile(source) as input_zip, ZipFile(target, "w", ZIP_DEFLATED) as output_zip:
        for item in input_zip.infolist():
            data = input_zip.read(item.filename)
            if item.filename.startswith("ppt/slides/slide") and item.filename.endswith(".xml"):
                root = ET.fromstring(data)
                for index, paragraph in enumerate(root.findall(".//a:p", PPT_NS)):
                    key = f"{item.filename}#{index}"
                    if key in translations:
                        replace_text_in_paragraph(paragraph, paragraph.findall(".//a:t", PPT_NS), translations[key])
                data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
            output_zip.writestr(item, data)

    return target.getvalue()


def replace_excel_cell_text(cell: ET.Element, translated_text: str) -> None:
    for child in list(cell):
        cell.remove(child)

    cell.attrib["t"] = "inlineStr"
    inline_string = ET.SubElement(cell, f"{{{EXCEL_NS['a']}}}is")
    text_node = ET.SubElement(inline_string, f"{{{EXCEL_NS['a']}}}t")
    text_node.text = clean_office_xml_text(translated_text)


def serialize_excel_xml(root: ET.Element) -> bytes:
    used_uris = set()
    for element in root.iter():
        if element.tag.startswith("{"):
            used_uris.add(element.tag[1:].split("}", 1)[0])
        for attr_name in element.attrib:
            if attr_name.startswith("{"):
                used_uris.add(attr_name[1:].split("}", 1)[0])

    ignorable_attr = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Ignorable"
    if ignorable_attr in root.attrib:
        kept_prefixes = [
            prefix
            for prefix in root.attrib[ignorable_attr].split()
            if EXCEL_SERIALIZE_NAMESPACES.get(prefix) in used_uris
        ]
        if kept_prefixes:
            root.attrib[ignorable_attr] = " ".join(kept_prefixes)
        else:
            root.attrib.pop(ignorable_attr, None)

    for prefix, uri in EXCEL_SERIALIZE_NAMESPACES.items():
        ET.register_namespace(prefix, uri)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def build_translated_xlsx(
    raw: bytes,
    translations: dict[str, str],
    source_by_location: dict[str, str],
    keep_source_with_translation: bool = False,
) -> bytes:
    source = io.BytesIO(raw)
    target = io.BytesIO()

    with ZipFile(source) as input_zip, ZipFile(target, "w", ZIP_DEFLATED) as output_zip:
        for item in input_zip.infolist():
            data = input_zip.read(item.filename)
            if item.filename.startswith("xl/worksheets/sheet") and item.filename.endswith(".xml"):
                root = ET.fromstring(data)
                for cell in root.findall(".//a:c", EXCEL_NS):
                    key = f"{item.filename}#{cell.attrib.get('r', '')}"
                    if key in translations:
                        translated = output_translation_for(
                            key,
                            source_by_location.get(key, ""),
                            translations,
                            keep_source_with_translation,
                        )
                        replace_excel_cell_text(cell, translated)
                data = serialize_excel_xml(root)
            output_zip.writestr(item, data)

    return target.getvalue()


def output_file_name(file_name: str) -> str:
    path = Path(file_name)
    return f"Translated-{path.stem}{path.suffix or '.txt'}"


def mime_type(file_name: str) -> str:
    lower_name = file_name.lower()
    if lower_name.endswith(".csv"):
        return "text/csv"
    if lower_name.endswith((".as", ".ad")):
        return "text/plain"
    if lower_name.endswith(".docx"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if lower_name.endswith(".pptx"):
        return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    if lower_name.endswith(".xlsx"):
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if lower_name.endswith(".xlsm"):
        return "application/vnd.ms-excel.sheet.macroEnabled.12"
    return "text/plain"


def safe_storage_name(file_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(file_name).name).strip("_")
    return safe_name or "document.txt"


def ensure_job_storage_dirs() -> None:
    JOB_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    JOB_RESULT_DIR.mkdir(parents=True, exist_ok=True)


def job_upload_path(job_id: str, file_name: str) -> Path:
    ensure_job_storage_dirs()
    return JOB_UPLOAD_DIR / f"{job_id}_{safe_storage_name(file_name)}"


def job_result_path(job_id: str, file_name: str) -> Path:
    ensure_job_storage_dirs()
    translated_name = output_file_name(file_name)
    return JOB_RESULT_DIR / f"{job_id}_{safe_storage_name(translated_name)}"


def is_valid_email(value: str) -> bool:
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value.strip()))


def is_smtp_configured() -> bool:
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_from = os.getenv("SMTP_FROM", os.getenv("SMTP_USERNAME", "")).strip()
    return bool(smtp_host and smtp_from)


def manual_email_status() -> str:
    return "Manual email draft ready. Download the translated file and attach it yourself."


def translation_mailto_link(to_email: str, file_name: str, result_file_name: str) -> str:
    subject = f"Term1 translation completed: {result_file_name}"
    body = "\n".join(
        [
            "Your Term1 translation job is complete.",
            "",
            f"Source file: {file_name}",
            f"Translated file: {result_file_name}",
            "",
            "Please attach the downloaded translated file before sending this email.",
        ]
    )
    return f"mailto:{quote(to_email)}?subject={quote(subject)}&body={quote(body)}"


def send_completed_translation_email(to_email: str, file_name: str, result_path: Path, result_file_name: str) -> str:
    if not to_email:
        return ""
    if not is_valid_email(to_email):
        return "Invalid email address."

    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USERNAME", "").strip()
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    smtp_from = os.getenv("SMTP_FROM", smtp_user).strip()
    smtp_tls = os.getenv("SMTP_TLS", "true").lower() in {"1", "true", "yes"}

    if not smtp_host or not smtp_from:
        return manual_email_status()
    if not result_path.exists():
        return "Email not sent: translated file was not found."
    if result_path.stat().st_size > MAX_EMAIL_ATTACHMENT_BYTES:
        return "Email not sent: translated file is larger than the email attachment limit. Download it from the app."

    message = EmailMessage()
    message["From"] = smtp_from
    message["To"] = to_email
    message["Subject"] = f"Term1 translation completed: {result_file_name}"
    message.set_content(
        "\n".join(
            [
                "Your Term1 translation job is complete.",
                "",
                f"Source file: {file_name}",
                f"Translated file: {result_file_name}",
                "",
                "The translated file is attached.",
            ]
        )
    )

    content_type, _ = mimetypes.guess_type(result_file_name)
    maintype, subtype = (content_type or "application/octet-stream").split("/", 1)
    message.add_attachment(
        result_path.read_bytes(),
        maintype=maintype,
        subtype=subtype,
        filename=result_file_name,
    )

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        if smtp_tls:
            server.starttls()
        if smtp_user and smtp_password:
            server.login(smtp_user, smtp_password)
        server.send_message(message)

    return "Email sent."


@st.cache_resource
def background_job_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=4)


def prepare_and_run_document_translation_job(
    job_id: str,
    raw_document: bytes,
    file_name: str,
    glossary: pd.DataFrame,
    translation_mode: str,
    keep_source_with_translation: bool,
) -> None:
    try:
        update_translation_job(job_id, status="running", error_message="", progress_message="Preparing file.")
        blocks = extract_text_blocks(raw_document, file_name)
        translatable_blocks = [block for block in blocks if should_translate(block.text)]
        checkpoint_path = checkpoint_path_for(file_name, raw_document, translation_mode)
        saved_translations = load_checkpoint(checkpoint_path)
        hydrate_translation_memory_from_checkpoint(blocks, saved_translations, translation_mode)
        source_by_key = {}
        for block in translatable_blocks:
            source_by_key.setdefault(translation_memory_key(block.text), block.text)
        memory_hits = translation_memory_lookup(list(source_by_key.values()), translation_mode)
        memory_applied = 0
        for block in translatable_blocks:
            if block.location in saved_translations:
                continue
            memory_translation = memory_hits.get(translation_memory_key(block.text))
            if memory_translation:
                saved_translations[block.location] = memory_translation
                memory_applied += 1
        if memory_applied:
            save_checkpoint(checkpoint_path, saved_translations)
        saved_memory_keys = {
            translation_memory_key(block.text)
            for block in translatable_blocks
            if block.location in saved_translations
        }
        saved_count = sum(1 for block in translatable_blocks if block.location in saved_translations)
        pending_unique_count = len(
            {
                translation_memory_key(block.text)
                for block in translatable_blocks
                if block.location not in saved_translations and translation_memory_key(block.text) not in saved_memory_keys
            }
        )
        batch_count = (pending_unique_count + DOCUMENT_BATCH_SIZE - 1) // DOCUMENT_BATCH_SIZE
        update_translation_job(
            job_id,
            status="running",
            total_blocks=len(blocks),
            translatable_blocks=len(translatable_blocks),
            completed_blocks=saved_count,
            total_batches=batch_count,
            completed_batches=0,
            progress_message=(
                f"Preflight complete. Unique JP: {len(source_by_key):,}. "
                f"TM/checkpoint hits: {saved_count:,}. Remaining unique: {pending_unique_count:,}."
            ),
        )

        if not blocks:
            update_translation_job(
                job_id,
                status="failed",
                error_message=no_blocks_error_message(file_name),
                finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            return

        if not translatable_blocks:
            translated_document = build_translated_document(
                raw_document,
                file_name,
                {},
                blocks,
                keep_source_with_translation,
            )
            translated_name = output_file_name(file_name)
            result_path = job_result_path(job_id, file_name)
            result_path.write_bytes(translated_document)
            update_translation_job(
                job_id,
                status="completed",
                result_file_name=translated_name,
                result_file_path=str(result_path),
                result_mime=mime_type(translated_name),
                progress_message="No Japanese text found. Original document is ready.",
                finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            return

        run_document_translation_job(
            job_id,
            raw_document,
            file_name,
            blocks,
            glossary,
            translation_mode,
            keep_source_with_translation,
            checkpoint_path,
            batch_count,
        )
    except Exception as exc:
        if str(exc) == "Translation job was stopped.":
            return
        update_translation_job(
            job_id,
            status="failed",
            error_message=format_translation_error(exc),
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )


def run_document_translation_job(
    job_id: str,
    raw_document: bytes,
    file_name: str,
    blocks: list[TextBlock],
    glossary: pd.DataFrame,
    translation_mode: str,
    keep_source_with_translation: bool,
    checkpoint_path: Path,
    total_batches: int,
) -> None:
    started_at = time.time()

    def update_progress(done, total, done_batches, total_batches, elapsed, message):
        update_translation_job(
            job_id,
            status="running",
            completed_blocks=done,
            completed_batches=done_batches,
            progress_message=message,
        )

    try:
        update_translation_job(job_id, status="running", error_message="", progress_message="Starting translation.")
        translations, _, token_usage = translate_blocks_batch(
            blocks,
            glossary,
            translation_mode,
            checkpoint_path=checkpoint_path,
            progress_callback=update_progress,
            should_continue=lambda: translation_job_is_active(job_id),
        )
        translated_document = build_translated_document(
            raw_document,
            file_name,
            translations,
            blocks,
            keep_source_with_translation,
        )
        translated_name = output_file_name(file_name)
        result_path = job_result_path(job_id, file_name)
        result_path.write_bytes(translated_document)
        notify_email = ""
        with sqlite3.connect(JOB_DB_PATH) as conn:
            row = conn.execute(
                "SELECT notify_email FROM translation_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            notify_email = row[0] if row and row[0] else ""
        notification_status = ""
        if notify_email:
            try:
                notification_status = send_completed_translation_email(
                    notify_email,
                    file_name,
                    result_path,
                    translated_name,
                )
            except Exception as exc:
                notification_status = f"Email failed: {exc}"
        update_translation_job(
            job_id,
            status="completed",
            completed_blocks=sum(1 for block in blocks if should_translate(block.text)),
            completed_batches=total_batches,
            input_tokens=token_usage.input_tokens,
            output_tokens=token_usage.output_tokens,
            total_tokens=token_usage.total_tokens,
            result_file_name=translated_name,
            result_file_path=str(result_path),
            result_mime=mime_type(translated_name),
            notification_status=notification_status,
            progress_message=f"Completed in {format_duration(time.time() - started_at)}.",
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
    except Exception as exc:
        if str(exc) == "Translation job was stopped.":
            return
        update_translation_job(
            job_id,
            status="failed",
            error_message=format_translation_error(exc),
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )


def terminology_report(hits: list[TermHit]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Japanese": hit.jp, "Required English": hit.en, "Count": hit.count} for hit in hits]
    )


def ai_model_version_text() -> str:
    return "\n".join(
        [
            "OpenAI",
            openai_model(),
        ]
    )


def glossary_version_text() -> str:
    glossary_path = next((path for path in DEFAULT_GLOSSARY_PATHS if path.exists()), None)
    if glossary_path is None:
        return "Glossary file was not found."

    raw = glossary_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()[:12]
    modified = glossary_path.stat().st_mtime
    modified_text = datetime.fromtimestamp(modified, ZoneInfo("America/New_York")).strftime(
        "%Y-%m-%d %I:%M %p ET"
    )
    try:
        workbook = load_workbook(glossary_path, read_only=True, data_only=True)
        try:
            worksheet = workbook[workbook.sheetnames[0]]
            non_empty_a_cells = sum(
                1
                for (value,) in worksheet.iter_rows(min_col=1, max_col=1, values_only=True)
                if str(value or "").strip()
            )
            term_count = max(non_empty_a_cells - 1, 0)
        finally:
            workbook.close()
    except Exception:
        term_count = "Unavailable"
    term_count_text = f"{term_count:,}" if isinstance(term_count, int) else str(term_count)

    return "\n".join(
        [
            "Glossary Information",
            "",
            "Version: v1.0",
            f"Terms: {term_count_text}",
            f"Last Updated: {modified_text}",
            "Owner: Aoi Minamoto",
            "",
            "Source: glossary.xlsx",
        ]
    )


def plc_rules_version_text() -> str:
    rule_path = next((path for path in DEFAULT_PLC_RULE_PATHS if path.exists()), None)
    if rule_path is None:
        expected = ", ".join(path.name for path in DEFAULT_PLC_RULE_PATHS)
        return f"PLC rule file was not found. Expected one of: {expected}"

    raw = rule_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()[:12]
    modified = rule_path.stat().st_mtime
    modified_text = datetime.fromtimestamp(modified, ZoneInfo("America/New_York")).strftime(
        "%Y-%m-%d %I:%M %p ET"
    )
    try:
        rule_count = len(normalize_plc_rules(read_plc_rules()))
    except Exception:
        rule_count = "Unavailable"

    return "\n".join(
        [
            "PLC Rules Information",
            "",
            "Version: v1.0",
            f"Rules: {rule_count}",
            f"Last Updated: {modified_text}",
            "Owner: Aoi Minamoto",
            "",
            f"Source: {rule_path.name}",
        ]
    )


def apply_compact_style() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stCaptionContainer"],
        div[data-testid="stWidgetLabel"],
        div[data-testid="stFileUploader"],
        div[data-testid="stRadio"] label,
        div[data-testid="stCodeBlock"] pre {
            font-size: 1.5rem !important;
            line-height: 1.35;
        }

        button[data-baseweb="tab"],
        div[data-testid="stExpander"] summary {
            font-size: 1.5rem !important;
            font-weight: 600;
        }

        div.stButton > button,
        div.stDownloadButton > button {
            width: auto;
            min-width: 190px;
            border-radius: 4px;
            padding: 0.56rem 1.1rem;
            font-size: 1.25rem;
            font-weight: 500;
            box-shadow: none;
        }

        div.stButton > button[kind="primary"],
        div.stDownloadButton > button[kind="primary"] {
            background-color: #ff4b4b;
            border: 1px solid #d63b3b;
        }

        div.stButton > button:hover,
        div.stDownloadButton > button:hover {
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.16);
        }

        div[data-testid="stHorizontalBlock"] {
            align-items: center;
        }

        div[data-testid="stProgress"] > div > div > div {
            height: 14px;
        }

        div[data-testid="stFileUploaderDropzoneInstructions"] small,
        div[data-testid="stFileUploaderDropzoneInstructions"] > div:last-child {
            display: none;
        }

        div[data-testid="stFileUploaderFile"] {
            max-width: 100%;
        }

        div[data-testid="stFileUploaderFileName"],
        div[data-testid="stFileUploaderFile"] span {
            max-width: 100%;
            overflow: visible;
            text-overflow: clip;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: break-word;
        }

        .usage-card {
            border: 1px solid #d0d7de;
            border-radius: 6px;
            background: #ffffff;
            padding: 14px 16px 16px;
            margin-bottom: 18px;
        }

        .usage-card-label {
            color: #475569;
            font-size: 1rem;
            font-weight: 650;
            letter-spacing: 0;
            line-height: 1.2;
            margin-bottom: 6px;
        }

        .usage-card-value {
            color: #111827;
            font-size: 2.45rem;
            font-weight: 600;
            line-height: 1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_usage_card(usage_count: int) -> None:
    st.markdown(
        f"""
        <div class="usage-card">
          <div class="usage-card-label">App use times</div>
          <div class="usage-card-value">{usage_count:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_translation_result(translated_text: str, key: str = "english_translation_result") -> None:
    line_count = max(translated_text.count("\n") + 1, 4)
    height = min(max(210, line_count * 28 + 92), 520)
    st.text_area(
        "English Translation",
        value=translated_text,
        height=height,
        key=key,
    )


def render_text_translation(glossary: pd.DataFrame, plc_rules: pd.DataFrame) -> None:
    translation_mode = st.radio(
        "Translation mode",
        TRANSLATION_MODES,
        horizontal=True,
        help="Use Supplier Email for business emails, Product Catalog for catalogs/spec sheets, PLC/SPLC for short control comments, and General for normal plant text.",
        key="text_translation_mode",
    )
    jp_text = st.text_area(
        "Input or paste Japanese text",
        height=220,
        placeholder="Example: paste Japanese manufacturing text here.",
        key="jp_text_input",
    )
    source_text, user_guidance = split_text_translation_input(jp_text)
    current_text_key = f"{translation_mode}::{source_text}::{user_guidance}"

    if st.button("Translate Text", type="primary"):
        if not source_text:
            st.warning("Please paste Japanese text first.")
            return

        progress = st.progress(0)
        status = st.empty()
        active_glossary = glossary_for_mode(glossary, plc_rules, translation_mode)

        try:
            status.write("Preparing glossary terms and protected codes...")
            progress.progress(0.2)
            translated_text, hits, token_usage = translate_block(
                source_text,
                active_glossary,
                translation_mode,
                user_guidance,
            )
            progress.progress(1.0)
            status.success("Translation complete.")
            if token_usage.total_tokens == 0 and hits:
                st.success("Translated by controlled rule. OpenAI API was not called.")

            st.session_state["last_text_translation_key"] = current_text_key
            st.session_state["last_text_translation"] = translated_text
            st.session_state["last_text_translation_terms"] = terminology_report(hits) if hits else None
            result_key = f"english_translation_result_{abs(hash(current_text_key))}"
            render_translation_result(translated_text, key=result_key)

            st.download_button(
                "Download translation",
                data=translated_text.encode("utf-8-sig"),
                file_name="Translated-text.txt",
                mime="text/plain",
            )

            if hits:
                st.subheader("Detected Terminology")
                st.dataframe(terminology_report(hits), use_container_width=True, hide_index=True)
            else:
                st.info("No glossary terms were detected in this text.")
        except Exception as exc:
            status.error("Translation failed.")
            st.error(f"Translation failed: {format_translation_error(exc)}")
    elif (
        st.session_state.get("last_text_translation_key") == current_text_key
        and st.session_state.get("last_text_translation")
    ):
        result_key = f"english_translation_result_{abs(hash(current_text_key))}"
        render_translation_result(st.session_state["last_text_translation"], key=result_key)
        terms = st.session_state.get("last_text_translation_terms")
        if terms is not None:
            st.subheader("Detected Terminology")
            st.dataframe(terms, use_container_width=True, hide_index=True)
@st.fragment(run_every="15s")
def render_active_document_job(
    active_job_id: str,
    glossary: pd.DataFrame,
    plc_rules: pd.DataFrame,
    translation_mode: str,
) -> None:
    detail = translation_job_detail(active_job_id)
    if detail.empty:
        return

    active_job = detail.iloc[0].to_dict()
    result_path_text = str(active_job.get("result_file_path") or "")
    result_path = Path(result_path_text) if result_path_text else None
    if active_job["status"] == "completed" and result_path is not None and result_path.exists():
        st.caption(active_job["file_name"])
        result_file_name = active_job["result_file_name"] or output_file_name(active_job["file_name"])
        render_download_ready(
            data=result_path.read_bytes(),
            file_name=result_file_name,
            mime=active_job["result_mime"] or mime_type(active_job["file_name"]),
            key=f"active_download_{active_job_id}",
        )
        source_path_text = str(active_job.get("source_file_path") or "")
        source_path = Path(source_path_text) if source_path_text else None
        if source_path is not None and source_path.exists():
            render_translation_pairs_preview(
                source_path.read_bytes(),
                active_job["file_name"],
                active_job["translation_mode"] or translation_mode,
            )
        if active_job["notify_email"]:
            st.link_button(
                "Open Email Draft",
                translation_mailto_link(active_job["notify_email"], active_job["file_name"], result_file_name),
            )
        if st.button("Dismiss", key=f"dismiss_completed_{active_job_id}"):
            st.session_state.pop("active_document_job_id", None)
            rerun_app()
    elif active_job["status"] == "failed":
        st.caption(active_job["file_name"])
        error_message = active_job["error_message"] or "No error detail."
        if error_message.lower().startswith("stopped by user"):
            st.warning(f"Stopped | {error_message}")
        else:
            st.error(f"Failed | {error_message}")
        action_cols = st.columns([1, 1, 2])
        if action_cols[0].button("Dismiss", key=f"dismiss_failed_{active_job_id}"):
            st.session_state.pop("active_document_job_id", None)
            rerun_app()
        if action_cols[1].button("Retry", key=f"retry_{active_job_id}"):
            source_path_text = str(active_job.get("source_file_path") or "")
            source_path = Path(source_path_text) if source_path_text else None
            if source_path is None or not source_path.exists():
                st.warning("Source file not found.")
            else:
                retry_raw = source_path.read_bytes()
                retry_blocks = extract_text_blocks(retry_raw, active_job["file_name"])
                retry_mode = active_job["translation_mode"] or translation_mode
                retry_glossary = glossary_for_mode(glossary, plc_rules, retry_mode)
                retry_progress_path = checkpoint_path_for(active_job["file_name"], retry_raw, retry_mode)
                retry_batch_count = int(active_job["total_batches"] or 0)
                retry_job_id = start_background_translation_job(
                    retry_raw,
                    active_job["file_name"],
                    retry_blocks,
                    retry_glossary,
                    retry_mode,
                    bool(st.session_state.get("document_keep_source_with_translation", False)),
                    active_job["notify_email"] or "",
                    retry_batch_count,
                    retry_progress_path,
                )
                st.session_state["active_document_job_id"] = retry_job_id
                rerun_app()
    else:
        st.caption(active_job["file_name"])
        st.info("Document is still being translated. You do not need to upload the file again.")
        stop_cols = st.columns([1, 1, 2])
        if stop_cols[0].button("Stop Translation", key=f"stop_translation_{active_job_id}"):
            stop_translation_job(active_job_id)
            st.session_state.pop("active_document_job_id", None)
            rerun_app()
        if stop_cols[1].button("Stop All", key=f"stop_all_translation_{active_job_id}"):
            stop_all_active_translation_jobs()
            st.session_state.pop("active_document_job_id", None)
            rerun_app()
        active_done = int(active_job["completed_blocks"] or 0)
        active_total = int(active_job["translatable_blocks"] or 0)
        active_done_batches = int(active_job["completed_batches"] or 0)
        active_total_batches = int(active_job["total_batches"] or 0)
        active_elapsed = elapsed_since_timestamp(active_job.get("created_at", ""))
        updated_elapsed = elapsed_since_timestamp(active_job.get("updated_at", ""))
        updated_at = parse_timestamp(active_job.get("updated_at", ""))
        job_is_orphaned = updated_at is not None and updated_at < APP_STARTED_AT
        active_ratio = 0.0 if active_total == 0 else min(active_done / active_total, 1.0)
        visual_ratio = active_ratio
        if active_total > 0 and active_done == 0:
            visual_ratio = 0.04
        elif active_total > 0 and active_done > 0:
            visual_ratio = max(active_ratio, 0.04)
        st.progress(visual_ratio)
        st.write(f"{progress_text(active_done, active_total, active_elapsed)} | {progress_percent(active_done, active_total)}")
        progress_message = str(active_job.get("progress_message") or "")
        if progress_message:
            st.caption(progress_message)
        remaining_batches = max(active_total_batches - active_done_batches, 0)
        updated_label = "unknown"
        if updated_elapsed is not None:
            updated_label = f"{format_duration(updated_elapsed)} ago"
        metric_cols = st.columns(4)
        metric_cols[0].metric("Completed batches", f"{active_done_batches:,}")
        metric_cols[1].metric("Total batches", f"{active_total_batches:,}")
        metric_cols[2].metric("Remaining batches", f"{remaining_batches:,}")
        metric_cols[3].metric("Last update", updated_label)
        st.caption(f"Translated blocks: {active_done:,}/{active_total:,}")
        if active_total <= 0:
            st.caption(f"Reading file and calculating batches | Updated: {active_job.get('updated_at', '')}")
        else:
            st.caption(
                f"Blocks: {active_done:,}/{active_total:,} | "
                f"Batches: {active_done_batches:,}/{active_total_batches:,} | "
                f"Updated: {active_job.get('updated_at', '')}"
            )
        job_is_stalled = (updated_elapsed is not None and updated_elapsed > 300) or job_is_orphaned
        if job_is_orphaned:
            st.warning("This job was started by an older app process. Continue can attach a new background worker.")
        elif job_is_stalled:
            st.warning("No progress update for more than 5 minutes. Continue can restart from saved progress.")
        source_path_text = str(active_job.get("source_file_path") or "")
        source_path = Path(source_path_text) if source_path_text else None
        if job_is_stalled and source_path is not None and source_path.exists():
            if st.button("Continue Translation", type="primary", key=f"continue_translation_{active_job_id}"):
                source_raw = source_path.read_bytes()
                restart_mode = active_job["translation_mode"] or translation_mode
                restart_glossary = glossary_for_mode(glossary, plc_rules, restart_mode)
                update_translation_job(
                    active_job_id,
                    status="failed",
                    error_message="Stopped by user to continue with a new background worker.",
                    finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                restart_job_id = start_queued_document_translation_job(
                    source_raw,
                    active_job["file_name"],
                    restart_glossary,
                    restart_mode,
                    bool(st.session_state.get("document_keep_source_with_translation", False)),
                    active_job["notify_email"] or "",
                )
                st.session_state["active_document_job_id"] = restart_job_id
                rerun_app()


def render_current_document_job(glossary: pd.DataFrame, plc_rules: pd.DataFrame, translation_mode: str) -> bool:
    active_job_id = latest_running_translation_job_id() or st.session_state.get("active_document_job_id")
    if not active_job_id and render_batch_log_status():
        return True
    if not active_job_id:
        return False
    st.session_state["active_document_job_id"] = active_job_id
    st.caption("Current translation")
    render_active_document_job(active_job_id, glossary, plc_rules, translation_mode)
    return True


def parse_batch_log_status() -> dict[str, str | int | float] | None:
    log_path = BASE_DIR / "batch_outputs" / "COMMENT_batch_translate.log"
    if not log_path.exists():
        return None
    lines = [line.strip() for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not lines:
        return None
    latest = lines[-1]
    updated_text = latest.split("|", 1)[0].strip()
    updated_at = parse_timestamp(updated_text)
    status = {
        "updated_text": updated_text,
        "updated_age": elapsed_since_timestamp(updated_text),
        "latest": latest,
    }
    if updated_at is not None:
        status["updated_epoch"] = updated_at.timestamp()
    batch_match = re.search(r"batch=([\d,]+)/([\d,]+)", latest)
    saved_match = re.search(r"saved=([\d,]+)/([\d,]+)", latest)
    if batch_match:
        status["completed_batches"] = int(batch_match.group(1).replace(",", ""))
        status["total_batches"] = int(batch_match.group(2).replace(",", ""))
    if saved_match:
        status["saved_blocks"] = int(saved_match.group(1).replace(",", ""))
        status["total_blocks"] = int(saved_match.group(2).replace(",", ""))
    recent_points = []
    for line in lines[-60:]:
        line_time = parse_timestamp(line.split("|", 1)[0].strip())
        line_saved_match = re.search(r"saved=([\d,]+)/([\d,]+)", line)
        if line_time is not None and line_saved_match:
            recent_points.append((line_time, int(line_saved_match.group(1).replace(",", ""))))
    if len(recent_points) >= 2 and "saved_blocks" in status and "total_blocks" in status:
        first_time, first_saved = recent_points[0]
        last_time, last_saved = recent_points[-1]
        seconds = max((last_time - first_time).total_seconds(), 0)
        saved_delta = max(last_saved - first_saved, 0)
        if seconds > 0 and saved_delta > 0:
            remaining_blocks = max(int(status["total_blocks"]) - int(status["saved_blocks"]), 0)
            status["eta_seconds"] = remaining_blocks / (saved_delta / seconds)
    status["is_recent"] = updated_at is not None and (datetime.now() - updated_at).total_seconds() < 600
    return status


def render_batch_log_status() -> bool:
    if not latest_running_translation_job_id():
        return False
    status = parse_batch_log_status()
    if status is None:
        return False
    saved_blocks = int(status.get("saved_blocks", 0))
    total_blocks = int(status.get("total_blocks", 0))
    completed_batches = int(status.get("completed_batches", 0))
    total_batches = int(status.get("total_batches", 0))
    if total_blocks <= 0 and total_batches <= 0:
        return False

    ratio = min(saved_blocks / total_blocks, 1.0) if total_blocks else 0.0
    progress_label = "Complete" if ratio >= 1.0 else "Translating"
    updated_age = float(status.get("updated_age") or 0)
    partial_output_path = BASE_DIR / "batch_outputs" / "Translated-COMMENT.partial.csv"
    final_output_path = BASE_DIR / "batch_outputs" / "Translated-COMMENT.csv"
    latest_log_epoch = float(status.get("updated_epoch") or 0)

    st.subheader("Current translation")
    st.write("COMMENT.csv")
    if status.get("is_recent"):
        st.success("Translation is running. You can refresh this page or leave it open.")
    else:
        st.warning("Translation has not updated recently. The saved progress is still available.")
    st.progress(max(ratio, 0.02))
    st.write(f"{progress_label} | {ratio * 100:.2f}%")

    eta_seconds = status.get("eta_seconds")
    status_cols = st.columns(4)
    status_cols[0].metric("Live progress", f"{saved_blocks:,}/{total_blocks:,}")
    status_cols[1].metric("Last update", f"{format_duration(updated_age)} ago")
    status_cols[2].metric("Estimated finish", format_duration(float(eta_seconds)) if eta_seconds else "Calculating")

    partial_mtime = partial_output_path.stat().st_mtime if partial_output_path.exists() else 0
    partial_label = datetime.fromtimestamp(partial_mtime).strftime("%Y-%m-%d %H:%M:%S") if partial_mtime else "Not generated yet"
    status_cols[3].metric("Download file", partial_label)

    if latest_log_epoch and partial_mtime and partial_mtime + 300 < latest_log_epoch:
        st.info("The downloadable file is behind the live progress. Refresh it before downloading.")

    download_cols = st.columns([1, 1])
    if download_cols[0].button("Refresh Download File", key="update_comment_partial_output"):
        try:
            source_path = max(JOB_UPLOAD_DIR.glob("*COMMENT.csv"), key=lambda path: path.stat().st_mtime)
            raw = source_path.read_bytes()
            checkpoint_path = checkpoint_path_for("COMMENT.csv", raw, PLC_TRANSLATION_MODE)
            translations = load_checkpoint(checkpoint_path)
            blocks = extract_text_blocks(raw, "COMMENT.csv")
            partial_output_path.parent.mkdir(exist_ok=True)
            partial_output_path.write_bytes(build_translated_document(raw, "COMMENT.csv", translations, blocks))
            st.success("Download file refreshed from saved progress.")
        except Exception as exc:
            st.error(f"Could not refresh download file: {exc}")
    if partial_output_path.exists():
        download_cols[1].download_button(
            "Download Current File",
            data=partial_output_path.read_bytes(),
            file_name="Translated-COMMENT.partial.csv",
            mime="text/csv",
            key="download_batch_partial_output",
        )
    if ratio >= 1.0 and final_output_path.exists():
        final_updated = datetime.fromtimestamp(final_output_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        st.caption(f"Final output file updated: {final_updated}")
        st.download_button(
            "Download Final Output",
            data=final_output_path.read_bytes(),
            file_name="Translated-COMMENT.csv",
            mime="text/csv",
            key="download_batch_final_output",
        )
    with st.expander("Advanced details"):
        st.write(f"Completed batches: {completed_batches:,}/{total_batches:,}")
        st.write(f"Translated blocks: {saved_blocks:,}/{total_blocks:,}")
        st.write(f"Latest log: {status.get('latest', '')}")
    return True


def render_document_translation(glossary: pd.DataFrame, plc_rules: pd.DataFrame) -> None:
    st.caption("Mode")
    translation_mode = st.radio(
        "Translation mode",
        TRANSLATION_MODES,
        horizontal=False,
        key="document_translation_mode",
        label_visibility="collapsed",
    )
    keep_source_with_translation = st.checkbox(
        "Keep Japanese with English translation",
        value=False,
        help="When enabled, translated documents keep each original Japanese block and add the English translation on the next line.",
        key="document_keep_source_with_translation",
    )

    active_jobs = active_translation_job_count()
    control_cols = st.columns([1, 1, 3])
    control_cols[0].metric("Active jobs", f"{active_jobs:,}")
    if active_jobs and control_cols[1].button("Stop All Active Jobs", key="stop_all_active_document_jobs"):
        stopped_count = stop_all_active_translation_jobs()
        st.session_state.pop("active_document_job_id", None)
        st.success(f"Stopped {stopped_count:,} active job(s). Saved progress is preserved.")
        rerun_app()

    has_current_job = render_current_document_job(glossary, plc_rules, translation_mode)
    if has_current_job:
        return

    st.caption("Upload document (CSV, TXT, AS, AD, DOCX, PPTX, XLSX, XLSM | Max 100 MB)")
    uploaded_document = st.file_uploader(
        "Upload document (Max 100 MB)",
        type=["csv", "txt", "as", "ad", "docx", "pptx", "xlsx", "xlsm"],
        label_visibility="collapsed",
    )

    if uploaded_document is None:
        return

    raw_document = uploaded_document.getvalue()
    if len(raw_document) > MAX_UPLOAD_BYTES:
        st.warning("This document is larger than the 100 MB safety limit. Please split the file or test with a smaller copy.")
        return
    if uploaded_document.name.lower().endswith((".as", ".ad")) and translation_mode != ROBOT_PROGRAM_TRANSLATION_MODE:
        st.info("For AS/AD robot files, select Kawasaki Robot .as file. The app will translate readable Japanese robot comments/labels and write English back into those fields.")
    encoding_warning = robot_encoding_warning(raw_document, uploaded_document.name)
    if encoding_warning:
        warning_html = html.escape(encoding_warning).replace("\n", "<br>")
        st.markdown(
            f"""
            <div style="border: 3px solid #b91c1c; background: #fee2e2; color: #7f1d1d; padding: 18px 20px; margin: 14px 0; border-radius: 6px;">
              <div style="font-size: 2.1rem; font-weight: 800; line-height: 1.2; margin-bottom: 8px;">AS FILE WARNING</div>
              <div style="font-size: 1.45rem; font-weight: 600; line-height: 1.35;">{warning_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    document_key = (
        f"{document_fingerprint(uploaded_document.name, raw_document)}::"
        f"{translation_mode}::keep_source={keep_source_with_translation}"
    )
    progress_path = checkpoint_path_for(uploaded_document.name, raw_document, translation_mode)
    if st.session_state.get("translated_document_key") != document_key:
        st.session_state.pop("translated_document_bytes", None)
        st.session_state.pop("translated_document_name", None)
        st.session_state.pop("translated_document_mime", None)
        st.session_state.pop("translated_document_preview", None)
        st.session_state.pop("translated_document_terms", None)
        st.session_state["translated_document_key"] = document_key

    notify_email = ""
    if latest_running_translation_job_id():
        st.warning("Another translation job is active. Starting a second job may slow both jobs down.")

    st.caption("Ready")
    st.info(f"{uploaded_document.name} is ready to queue. Size: {len(raw_document) / (1024 * 1024):.1f} MB.")
    translate_clicked = st.button("Start Translation", type="primary")

    if translate_clicked:
        active_glossary = glossary_for_mode(glossary, plc_rules, translation_mode)
        st.caption("Progress")
        status = st.empty()
        job_id = start_queued_document_translation_job(
            raw_document,
            uploaded_document.name,
            active_glossary,
            translation_mode,
            keep_source_with_translation,
            notify_email,
        )
        st.session_state["active_document_job_id"] = job_id
        status.success("Queued. Preparing file in the background.")
        rerun_app()

    if st.session_state.get("translated_document_bytes"):
        render_download_ready(
            data=st.session_state["translated_document_bytes"],
            file_name=st.session_state["translated_document_name"],
            mime=st.session_state["translated_document_mime"],
        )
        render_translation_pairs_preview(raw_document, uploaded_document.name, translation_mode)

def main() -> None:
    load_env()

    st.set_page_config(page_title="Battery Manufacturing AI Platform", layout="wide")
    apply_compact_style()
    usage_count = increment_usage_count_once()
    st.title("Battery Manufacturing AI Platform")

    with st.sidebar:
        render_usage_card(usage_count)
        with st.expander("AI Model"):
            st.code(ai_model_version_text(), language="text")
        st.header("Knowledge Base")
        with st.expander("Glossary"):
            st.code(glossary_version_text(), language="text")
        with st.expander("PLC rules"):
            st.code(plc_rules_version_text(), language="text")

    try:
        glossary = normalize_glossary(read_glossary(None))
    except Exception as exc:
        st.error(f"Glossary error: {exc}")
        st.stop()

    plc_rules_error = ""
    try:
        plc_rules = normalize_plc_rules(read_plc_rules())
    except Exception as exc:
        plc_rules_error = str(exc)
        plc_rules = empty_terms_dataframe()

    if plc_rules_error:
        st.warning(f"PLC rule file could not be loaded, so the app will continue without PLC abbreviation rules: {plc_rules_error}")

    text_tab, document_tab = st.tabs(["Text Translation", "Document Translation"])

    with text_tab:
        render_text_translation(glossary, plc_rules)

    with document_tab:
        render_document_translation(glossary, plc_rules)


if __name__ == "__main__":
    main()
