import io
import os
import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = Path(__file__).parent
DEFAULT_GLOSSARY_PATHS = [
    BASE_DIR / "glossary.xlsx",
    BASE_DIR / "glossary.csv",
]
ENV_PATH = BASE_DIR / "app.env"

DEFAULT_MODEL = "gpt-4.1-mini"
PROTECTED_PATTERN = re.compile(
    r"\b(?:[A-Z]{1,6}[-_]?\d{1,6}[A-Z]?|\d+[A-Z]{1,4}|[XYMDSZR][0-9]{1,5}|[A-Z]{2,}-[A-Z0-9-]+)\b"
)


@dataclass(frozen=True)
class TermHit:
    jp: str
    en: str
    count: int


def load_env() -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    else:
        load_dotenv()


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
    body = rows[1:]
    width = len(header)
    normalized_rows = [(row + [""] * width)[:width] for row in body]
    return pd.DataFrame(normalized_rows, columns=header)


def read_uploaded_glossary(uploaded_file) -> pd.DataFrame:
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
            raise ValueError("The uploaded Excel file could not be opened.") from exc

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

    keep_columns = [column for column in ["JP", "EN", "Note", "Category", "Owner"] if column in glossary.columns]
    glossary = glossary[keep_columns].copy()
    glossary["JP"] = glossary["JP"].astype(str).map(clean_text)
    glossary["EN"] = glossary["EN"].astype(str).map(clean_text)
    glossary = glossary[(glossary["JP"] != "") & (glossary["EN"] != "")]
    glossary = glossary.drop_duplicates(subset=["JP"], keep="first")
    glossary["term_length"] = glossary["JP"].str.len()
    return glossary.sort_values("term_length", ascending=False).drop(columns=["term_length"]).reset_index(drop=True)


def clean_text(value: str) -> str:
    return unicodedata.normalize("NFKC", str(value)).strip()


def apply_glossary_to_source(text: str, glossary: pd.DataFrame) -> tuple[str, list[TermHit]]:
    translated_source = clean_text(text)
    hits = []

    for index, row in glossary.iterrows():
        jp = clean_text(row["JP"])
        en = clean_text(row["EN"])
        if not jp or jp not in translated_source:
            continue

        translated_source, count = re.subn(re.escape(jp), en, translated_source)
        if count:
            hits.append(TermHit(jp=jp, en=en, count=count))

    return translated_source, hits


def enforce_terms(text: str, hits: list[TermHit]) -> str:
    restored = text
    for hit in hits:
        restored = restored.replace(hit.jp, hit.en)
    return restored


def missing_required_terms(text: str, hits: list[TermHit]) -> list[TermHit]:
    normalized_output = clean_text(text).lower()
    return [hit for hit in hits if clean_text(hit.en).lower() not in normalized_output]


def find_protected_codes(text: str) -> list[str]:
    return sorted(set(PROTECTED_PATTERN.findall(text)))


def build_prompt(source_text: str, hits: list[TermHit], protected_codes: list[str]) -> str:
    terms = "\n".join(f"{hit.jp} = {hit.en}" for hit in hits)
    codes = ", ".join(protected_codes) if protected_codes else "None detected"

    return f"""
You are a professional Japanese-to-English translator for a battery manufacturing plant.

Translate the source text into clear, natural, engineering English.

Mandatory rules:
1. Some company terms have already been replaced with approved English terms in the source text.
2. Keep those approved English terms exactly as written.
3. Do not translate or modify PLC addresses, model names, station IDs, codes, or part numbers.
4. Use concise plant-floor English. Prefer direct maintenance/engineering wording over literary translation.
5. Preserve line breaks and list structure when useful.
6. Output only the English translation.

Company terminology detected in the source:
{terms if terms else "None"}

Protected codes detected:
{codes}

Source Japanese text:
{source_text}
""".strip()


def translate_text(source_text: str, hits: list[TermHit], protected_codes: list[str]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY was not found. Add it to app.env.")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        input=build_prompt(source_text, hits, protected_codes),
        temperature=0.1,
    )
    return response.output_text.strip()


def terminology_report(hits: list[TermHit]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Japanese": hit.jp, "Required English": hit.en, "Count": hit.count} for hit in hits]
    )


load_env()

st.set_page_config(page_title="JP to EN Plant Terminology Translator", layout="wide")
st.title("JP to EN Plant Terminology Translator")
st.caption("Accurate plant translation with company terminology locked before AI translation.")

with st.sidebar:
    st.header("Knowledge Base")
    uploaded = st.file_uploader("Upload glossary", type=["csv", "xlsx", "xlsm", "xls"])
    st.caption("Required columns: JP/EN or Japanese/English.")

try:
    raw_glossary = read_uploaded_glossary(uploaded)
    glossary = normalize_glossary(raw_glossary)
except Exception as exc:
    st.error(f"Glossary error: {exc}")
    st.stop()

jp_text = st.text_area(
    "Paste Japanese text",
    height=220,
    placeholder="例: 稼働モニで設備異常を確認してください。",
)

glossary_applied_text, hits = apply_glossary_to_source(jp_text, glossary)
protected_codes = find_protected_codes(jp_text)

left, right = st.columns([1.1, 0.9])

with left:
    if st.button("Translate", type="primary", use_container_width=True):
        if not jp_text.strip():
            st.warning("Please paste Japanese text first.")
        else:
            try:
                raw_translation = translate_text(glossary_applied_text, hits, protected_codes)
                final_translation = enforce_terms(raw_translation, hits)
                missing_terms = missing_required_terms(final_translation, hits)

                st.subheader("English Translation")
                st.write(final_translation)

                if missing_terms:
                    st.warning("Some required glossary terms were detected in Japanese but are missing from the English output.")
                    st.dataframe(
                        pd.DataFrame(
                            [{"Japanese": hit.jp, "Required English": hit.en} for hit in missing_terms]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.success("All detected glossary terms were applied.")

                st.download_button(
                    "Download translation",
                    data=final_translation.encode("utf-8-sig"),
                    file_name="translation.txt",
                    mime="text/plain",
                )
            except Exception as exc:
                st.error(f"Translation failed: {exc}")

with right:
    st.subheader("Detected Terminology")
    if hits:
        report = terminology_report(hits)
        st.dataframe(report, use_container_width=True, hide_index=True)
        st.download_button(
            "Download terminology report",
            data=report.to_csv(index=False, encoding="utf-8-sig"),
            file_name="terminology_report.csv",
            mime="text/csv",
        )
    else:
        st.info("No glossary terms detected yet.")

    with st.expander("Text sent to translation after glossary replacement"):
        st.code(glossary_applied_text or "No source text yet.", language="text")
