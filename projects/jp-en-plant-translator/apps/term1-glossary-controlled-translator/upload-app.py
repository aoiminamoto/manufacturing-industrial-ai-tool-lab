import csv
import io
import hashlib
import json
import os
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zipfile import BadZipFile, ZipFile, ZIP_DEFLATED
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openpyxl import load_workbook
from openai import APIConnectionError, APIStatusError, AuthenticationError, OpenAI, RateLimitError

try:
    from langsmith.wrappers import wrap_openai
except ImportError:
    wrap_openai = None


BASE_DIR = Path(__file__).parent
ENV_PATH = BASE_DIR / "app.env"
DEFAULT_GLOSSARY_PATHS = [
    BASE_DIR / "glossary.xlsx",
    BASE_DIR / "glossary.csv",
]
DEFAULT_MODEL = "gpt-4.1-mini"
DOCUMENT_BATCH_SIZE = 75
MAX_TRANSLATION_RETRIES = 3
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
PROGRESS_DIR = BASE_DIR / ".term1_progress"
USAGE_COUNT_PATH = BASE_DIR / ".term1_usage_count.json"
PROTECTED_PATTERN = re.compile(
    r"\b(?:[A-Z]{1,6}[-_]?\d{1,6}[A-Z]?|\d+[A-Z]{1,4}|[XYMDSZR][0-9]{1,5}|[A-Z]{2,}-[A-Z0-9-]+)\b"
)

WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
PPT_NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
EXCEL_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


@dataclass(frozen=True)
class TermHit:
    jp: str
    en: str
    count: int


@dataclass(frozen=True)
class TextBlock:
    location: str
    text: str


def load_env() -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    else:
        load_dotenv()


def clean_text(value: str) -> str:
    return unicodedata.normalize("NFKC", str(value)).strip()


def has_japanese_text(value: str) -> bool:
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", value))


def should_translate(text: str) -> bool:
    return has_japanese_text(clean_text(text))


def document_fingerprint(file_name: str, raw: bytes) -> str:
    digest = hashlib.sha256(raw).hexdigest()[:16]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(file_name).stem).strip("_") or "document"
    return f"{safe_name}_{len(raw)}_{digest}"


def checkpoint_path_for(file_name: str, raw: bytes) -> Path:
    return PROGRESS_DIR / f"{document_fingerprint(file_name, raw)}.json"


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


def is_safe_glossary_term(jp: str) -> bool:
    jp = clean_text(jp)
    if len(jp) < 2:
        return False
    if not has_japanese_text(jp):
        return False
    if PROTECTED_PATTERN.fullmatch(jp):
        return False
    return True


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


def apply_glossary_to_source(text: str, glossary: pd.DataFrame) -> tuple[str, list[TermHit]]:
    translated_source = clean_text(text)
    hits = []

    for _, row in glossary.iterrows():
        jp = clean_text(row["JP"])
        en = clean_text(row["EN"])
        if not jp or jp not in translated_source:
            continue

        translated_source, count = re.subn(re.escape(jp), en, translated_source)
        if count:
            hits.append(TermHit(jp=jp, en=en, count=count))

    return translated_source, hits


def find_protected_codes(text: str) -> list[str]:
    return sorted(set(PROTECTED_PATTERN.findall(text)))


def build_prompt(source_text: str, hits: list[TermHit], protected_codes: list[str]) -> str:
    terms = "\n".join(f"{hit.jp} = {hit.en}" for hit in hits)
    codes = ", ".join(protected_codes) if protected_codes else "None detected"

    return f"""
You are a professional Japanese-to-English translator for a battery manufacturing plant.

Translate the source text into clear, natural plant-floor engineering English for manufacturing users.

Mandatory rules:
1. Apply approved company glossary terms exactly as written. Do not paraphrase approved terms.
2. Preserve PLC addresses, device IDs, model names, station IDs, alarm codes, part numbers, and equipment codes exactly.
3. Use concise plant-floor English suitable for controls, seibi, production, and engineering users.
4. Do not invent missing information, causes, actions, measurements, or context that is not in the source.
5. If the source is ambiguous, translate only the meaning that is present and keep the wording neutral.
6. Preserve line breaks and list structure when useful.
7. Output only the English translation. Do not add explanations, notes, or commentary.

Company terminology detected in the source:
{terms if terms else "None"}

Protected codes detected:
{codes}

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

    tracing_enabled = os.getenv("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"}
    if tracing_enabled and wrap_openai is not None:
        return wrap_openai(client)
    return client


def openai_model() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def format_translation_error(exc: Exception) -> str:
    if isinstance(exc, APIConnectionError):
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


def translate_text(source_text: str, hits: list[TermHit], protected_codes: list[str]) -> str:
    client = openai_client()
    response = client.responses.create(
        model=openai_model(),
        input=build_prompt(source_text, hits, protected_codes),
        temperature=0.1,
    )
    return response.output_text.strip()


def translate_block(text: str, glossary: pd.DataFrame) -> tuple[str, list[TermHit]]:
    glossary_applied_text, hits = apply_glossary_to_source(text, glossary)
    protected_codes = find_protected_codes(text)
    translation = translate_text(glossary_applied_text, hits, protected_codes)
    return translation, hits


def build_batch_prompt(items: list[tuple[int, str, list[TermHit], list[str]]]) -> str:
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

    return f"""
You are a professional Japanese-to-English translator for a battery manufacturing plant.

Translate each block into clear, natural plant-floor engineering English for manufacturing users.

Mandatory rules:
1. Apply approved company glossary terms exactly as written. Do not paraphrase approved terms.
2. Preserve PLC addresses, device IDs, model names, station IDs, alarm codes, part numbers, and equipment codes exactly.
3. Use concise plant-floor English suitable for controls, seibi, production, and engineering users.
4. Do not invent missing information, causes, actions, measurements, or context that is not in the source.
5. If the source is ambiguous, translate only the meaning that is present and keep the wording neutral.
6. Preserve line breaks inside each block when useful.
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


def translate_blocks_batch(
    blocks: list[TextBlock],
    glossary: pd.DataFrame,
    checkpoint_path=None,
    progress_callback=None,
) -> tuple[dict[str, str], list[TermHit]]:
    client = openai_client()
    translations = load_checkpoint(checkpoint_path) if checkpoint_path else {}
    translatable_blocks = [block for block in blocks if should_translate(block.text)]
    pending_blocks = [block for block in translatable_blocks if block.location not in translations]
    all_hits = []
    started_at = time.time()
    completed_at_start = len(translatable_blocks) - len(pending_blocks)
    total_batches = (len(pending_blocks) + DOCUMENT_BATCH_SIZE - 1) // DOCUMENT_BATCH_SIZE

    if progress_callback:
        progress_callback(
            completed_at_start,
            len(translatable_blocks),
            0,
            total_batches,
            0,
            "Resuming from saved progress." if completed_at_start else "Starting translation.",
        )

    for start in range(0, len(pending_blocks), DOCUMENT_BATCH_SIZE):
        chunk = pending_blocks[start:start + DOCUMENT_BATCH_SIZE]
        items = []

        for offset, block in enumerate(chunk, start=1):
            glossary_applied_text, hits = apply_glossary_to_source(block.text, glossary)
            protected_codes = find_protected_codes(block.text)
            items.append((offset, glossary_applied_text, hits, protected_codes))
            all_hits.extend(hits)

        parsed = {}
        last_error = None
        for attempt in range(1, MAX_TRANSLATION_RETRIES + 1):
            try:
                response = client.responses.create(
                    model=openai_model(),
                    input=build_batch_prompt(items),
                    temperature=0.1,
                )
                parsed = parse_batch_translation(response.output_text.strip(), [item[0] for item in items])
                missing_ids = [item[0] for item in items if item[0] not in parsed]
                if missing_ids:
                    raise ValueError(f"Translation response missed block marker(s): {missing_ids}")
                break
            except Exception as exc:
                last_error = exc
                if attempt == MAX_TRANSLATION_RETRIES:
                    raise
                if progress_callback:
                    progress_callback(
                        completed_at_start + start,
                        len(translatable_blocks),
                        start // DOCUMENT_BATCH_SIZE,
                        total_batches,
                        time.time() - started_at,
                        f"Batch retry {attempt}/{MAX_TRANSLATION_RETRIES} after: {format_translation_error(exc)}",
                    )
                time.sleep(5 * attempt)

        if not parsed and last_error:
            raise last_error

        for offset, block in enumerate(chunk, start=1):
            translations[block.location] = parsed[offset]

        if checkpoint_path:
            save_checkpoint(checkpoint_path, translations)

        if progress_callback:
            completed = min(completed_at_start + start + len(chunk), len(translatable_blocks))
            progress_callback(
                completed,
                len(translatable_blocks),
                (start // DOCUMENT_BATCH_SIZE) + 1,
                total_batches,
                time.time() - started_at,
                "Saved batch progress.",
            )

    return translations, all_hits


def read_text_file(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp932", "shift_jis"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def write_text_file(blocks: list[TextBlock], translations: dict[str, str]) -> bytes:
    lines = []
    for block in blocks:
        lines.append(translations.get(block.location, block.text))
    return "\n\n".join(lines).encode("utf-8-sig")


def read_csv_rows(raw: bytes) -> list[list[str]]:
    for encoding in ("utf-8-sig", "utf-8", "cp932", "shift_jis", "cp1252"):
        try:
            text = raw.decode(encoding)
            return [row for row in csv.reader(io.StringIO(text))]
        except UnicodeDecodeError:
            continue
    text = raw.decode("utf-8", errors="replace")
    return [row for row in csv.reader(io.StringIO(text))]


def extract_csv_blocks(raw: bytes) -> list[TextBlock]:
    rows = read_csv_rows(raw)
    blocks = []
    for row_index, row in enumerate(rows):
        for column_index, cell in enumerate(row):
            value = clean_text(cell)
            if value:
                blocks.append(TextBlock(location=f"csv:{row_index}:{column_index}", text=value))
    return blocks


def build_translated_csv(raw: bytes, translations: dict[str, str]) -> bytes:
    rows = read_csv_rows(raw)
    for row_index, row in enumerate(rows):
        for column_index, _ in enumerate(row):
            key = f"csv:{row_index}:{column_index}"
            if key in translations:
                row[column_index] = translations[key]

    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def extract_text_blocks(raw: bytes, file_name: str) -> list[TextBlock]:
    lower_name = file_name.lower()
    if lower_name.endswith(".txt"):
        text = read_text_file(raw)
        return [TextBlock(location=f"text:{index}", text=part.strip()) for index, part in enumerate(text.split("\n\n")) if part.strip()]

    if lower_name.endswith(".csv"):
        return extract_csv_blocks(raw)

    if lower_name.endswith(".docx"):
        return extract_docx_blocks(raw)

    if lower_name.endswith((".xlsx", ".xlsm")):
        return extract_xlsx_blocks(raw)

    raise ValueError("Supported document types: CSV, TXT, DOCX, XLSX, XLSM.")


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


def replace_text_in_paragraph(paragraph: ET.Element, text_nodes: list[ET.Element], translated_text: str) -> None:
    if not text_nodes:
        return
    text_nodes[0].text = translated_text
    for node in text_nodes[1:]:
        node.text = ""


def build_translated_document(raw: bytes, file_name: str, translations: dict[str, str], blocks: list[TextBlock]) -> bytes:
    lower_name = file_name.lower()
    if lower_name.endswith(".txt"):
        return write_text_file(blocks, translations)

    if lower_name.endswith(".csv"):
        return build_translated_csv(raw, translations)

    if lower_name.endswith(".docx"):
        return build_translated_docx(raw, translations)

    if lower_name.endswith((".xlsx", ".xlsm")):
        return build_translated_xlsx(raw, translations)

    raise ValueError("Supported document types: CSV, TXT, DOCX, XLSX, XLSM.")


def build_translated_docx(raw: bytes, translations: dict[str, str]) -> bytes:
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
                        replace_text_in_paragraph(paragraph, paragraph.findall(".//w:t", WORD_NS), translations[key])
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
    text_node.text = translated_text


def build_translated_xlsx(raw: bytes, translations: dict[str, str]) -> bytes:
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
                        replace_excel_cell_text(cell, translations[key])
                data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
            output_zip.writestr(item, data)

    return target.getvalue()


def output_file_name(file_name: str) -> str:
    path = Path(file_name)
    return f"Translated-{path.stem}{path.suffix or '.txt'}"


def mime_type(file_name: str) -> str:
    lower_name = file_name.lower()
    if lower_name.endswith(".csv"):
        return "text/csv"
    if lower_name.endswith(".docx"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if lower_name.endswith(".xlsx"):
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if lower_name.endswith(".xlsm"):
        return "application/vnd.ms-excel.sheet.macroEnabled.12"
    return "text/plain"


def terminology_report(hits: list[TermHit]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Japanese": hit.jp, "Required English": hit.en, "Count": hit.count} for hit in hits]
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
        term_count = max(len(read_glossary(None)) - 1, 0)
    except Exception:
        term_count = "Unavailable"

    return "\n".join(
        [
            "Glossary file: glossary.xlsx",
            f"Last updated: {modified_text}",
            "Updated by: Aoi Minamoto",
            f"Glossary terms: {term_count}",
            f"Version hash: {digest}",
        ]
    )


def apply_compact_style() -> None:
    st.markdown(
        """
        <style>
        div.stButton > button,
        div.stDownloadButton > button {
            width: auto;
            min-width: 190px;
            border-radius: 4px;
            padding: 0.42rem 0.9rem;
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_text_translation(glossary: pd.DataFrame) -> None:
    jp_text = st.text_area(
        "Paste Japanese text",
        height=220,
        placeholder="例: 稼働モニで設備異常を確認してください。",
    )

    if st.button("Translate Text", type="primary"):
        if not jp_text.strip():
            st.warning("Please paste Japanese text first.")
            return

        progress = st.progress(0)
        status = st.empty()

        try:
            status.write("Preparing glossary terms and protected codes...")
            progress.progress(0.2)
            translated_text, hits = translate_block(jp_text, glossary)
            progress.progress(1.0)
            status.success("Translation complete.")
            st.subheader("English Translation")
            st.write(translated_text)

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


def render_document_translation(glossary: pd.DataFrame) -> None:
    st.info("Accepted document types: CSV (.csv), Word (.docx), Excel (.xlsx/.xlsm), and Text (.txt). Large file safety limit: 50 MB.")
    uploaded_document = st.file_uploader(
        "Upload Japanese document",
        type=["csv", "txt", "docx", "xlsx", "xlsm"],
    )

    if uploaded_document is None:
        st.info("Upload a Word, Excel, or TXT file to start.")
        return

    raw_document = uploaded_document.getvalue()
    if len(raw_document) > MAX_UPLOAD_BYTES:
        st.warning("This document is larger than the 50 MB safety limit. Please split the file or test with a smaller copy.")
        return

    document_key = document_fingerprint(uploaded_document.name, raw_document)
    progress_path = checkpoint_path_for(uploaded_document.name, raw_document)
    if st.session_state.get("translated_document_key") != document_key:
        st.session_state.pop("translated_document_bytes", None)
        st.session_state.pop("translated_document_name", None)
        st.session_state.pop("translated_document_mime", None)
        st.session_state.pop("translated_document_preview", None)
        st.session_state.pop("translated_document_terms", None)
        st.session_state["translated_document_key"] = document_key

    try:
        blocks = extract_text_blocks(raw_document, uploaded_document.name)
    except Exception as exc:
        st.error(f"Document error: {exc}")
        return

    st.subheader("Document Text Blocks")
    translatable_blocks = [block for block in blocks if should_translate(block.text)]
    saved_translations = load_checkpoint(progress_path)
    saved_count = sum(1 for block in translatable_blocks if block.location in saved_translations)
    batch_count = (max(len(translatable_blocks) - saved_count, 0) + DOCUMENT_BATCH_SIZE - 1) // DOCUMENT_BATCH_SIZE

    st.write(
        f"Detected {len(blocks)} text block(s). "
        f"{len(translatable_blocks)} need Japanese translation. "
        f"{saved_count} already saved. "
        f"{batch_count} batch(es) remaining."
    )

    if len(blocks) >= 1000:
        st.info("Large File Mode is active: the app skips non-Japanese cells and saves progress after every batch.")

    preview_rows = [{"Location": block.location, "Japanese Text": block.text[:300]} for block in blocks[:20]]
    st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

    action_col, clear_col, _ = st.columns([0.28, 0.22, 0.5])
    with action_col:
        translate_clicked = st.button("Translate Document", type="primary")
    with clear_col:
        clear_clicked = st.button("Clear Saved Progress")

    if clear_clicked:
        if progress_path.exists():
            progress_path.unlink()
        st.session_state.pop("translated_document_bytes", None)
        st.session_state.pop("translated_document_name", None)
        st.session_state.pop("translated_document_mime", None)
        st.session_state.pop("translated_document_preview", None)
        st.session_state.pop("translated_document_terms", None)
        st.success("Saved progress cleared for this document.")
        rerun_app()

    if translate_clicked:
        if not blocks:
            st.warning(
                "No translatable text was found in this document. "
                "Please upload a CSV, TXT, DOCX, XLSX, or XLSM file that contains selectable text, not scanned images. "
                "For Excel, save old .xls files as .xlsx first."
            )
            return
        if not translatable_blocks:
            st.info("No Japanese text was found. Downloading the original document is enough.")
            translated_document = build_translated_document(raw_document, uploaded_document.name, {}, blocks)
            translated_name = output_file_name(uploaded_document.name)
            st.session_state["translated_document_bytes"] = translated_document
            st.session_state["translated_document_name"] = translated_name
            st.session_state["translated_document_mime"] = mime_type(translated_name)
            st.session_state["translated_document_preview"] = []
            st.session_state["translated_document_terms"] = []
            return

        progress = st.progress(0)
        status = st.empty()
        metrics = st.empty()

        def update_progress(done, total, done_batches, total_batches, elapsed, message):
            ratio = 1.0 if total == 0 else min(done / total, 1.0)
            progress.progress(ratio)
            remaining = 0
            if done > saved_count and ratio < 1.0:
                rate = elapsed / max(done - saved_count, 1)
                remaining = rate * (total - done)
            status.write(message)
            metrics.write(
                f"Translated {done}/{total} Japanese block(s). "
                f"Batch {done_batches}/{total_batches}. "
                f"Elapsed {format_duration(elapsed)}. "
                f"ETA {format_duration(remaining)}."
            )

        try:
            status.write(f"Translating {len(translatable_blocks)} Japanese block(s) in {batch_count} remaining batch(es)...")
            translations, all_hits = translate_blocks_batch(
                blocks,
                glossary,
                checkpoint_path=progress_path,
                progress_callback=update_progress,
            )
            progress.progress(1.0)

            translated_document = build_translated_document(raw_document, uploaded_document.name, translations, blocks)
            translated_name = output_file_name(uploaded_document.name)

            st.session_state["translated_document_bytes"] = translated_document
            st.session_state["translated_document_name"] = translated_name
            st.session_state["translated_document_mime"] = mime_type(translated_name)
            st.session_state["translated_document_preview"] = [
                {
                    "Original": block.text,
                    "Translation": translations.get(block.location, ""),
                }
                for block in blocks[:20]
            ]
            st.session_state["translated_document_terms"] = all_hits
            status.success("Translation complete.")
            metrics.write(
                f"Translated {len(translatable_blocks)}/{len(translatable_blocks)} Japanese block(s). "
                "Download is ready."
            )
        except Exception as exc:
            st.error(
                f"Translation failed: {format_translation_error(exc)} "
                "Saved progress remains available; click Translate Document again to resume."
            )

    if st.session_state.get("translated_document_bytes"):
        st.success("Download ready. Click the button below to save the translated file.")
        st.download_button(
            "Download Translated File",
            data=st.session_state["translated_document_bytes"],
            file_name=st.session_state["translated_document_name"],
            mime=st.session_state["translated_document_mime"],
            type="primary",
        )

        st.subheader("Translation Preview")
        st.dataframe(
            pd.DataFrame(st.session_state.get("translated_document_preview", [])),
            use_container_width=True,
            hide_index=True,
        )

        all_hits = st.session_state.get("translated_document_terms", [])
        if all_hits:
            st.subheader("Detected Terminology")
            st.dataframe(terminology_report(all_hits), use_container_width=True, hide_index=True)
        else:
            st.info("No glossary terms were detected in this document.")


load_env()

st.set_page_config(page_title="JP to EN Translator", layout="wide")
apply_compact_style()
usage_count = increment_usage_count_once()
st.title("JP to EN Plant Translator")
st.caption("Type Japanese text or upload Word, Excel, or TXT files. The app applies company terminology first, then translates with OpenAI API.")
st.warning(
    "Data notice: Text entered or uploaded in this app is sent to OpenAI API for translation. "
    "Do not upload confidential or restricted information unless approved by company policy.",
)

with st.sidebar:
    st.metric("App uses", usage_count)
    st.header("Knowledge Base")
    st.success("Internal glossary is already loaded.")
    st.code(glossary_version_text(), language="text")
    st.caption("This demo uses the built-in glossary.xlsx file. Users do not need to upload a glossary.")
    st.caption("For demo testing, please use small documents to avoid unnecessary OpenAI API cost.")

try:
    glossary = normalize_glossary(read_glossary(None))
except Exception as exc:
    st.error(f"Glossary error: {exc}")
    st.stop()

text_tab, document_tab = st.tabs(["Text Translation", "Document Translation"])

with text_tab:
    render_text_translation(glossary)

with document_tab:
    render_document_translation(glossary)
