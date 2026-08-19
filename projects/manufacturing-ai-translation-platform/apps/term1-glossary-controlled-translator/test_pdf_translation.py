import importlib.util
from pathlib import Path

import fitz


APP_PATH = Path(__file__).with_name("upload-app.py")
SPEC = importlib.util.spec_from_file_location("term1_upload_app_pdf_test", APP_PATH)
APP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(APP)


def make_vector_drawing_pdf() -> bytes:
    document = fitz.open()
    for page_number, label in enumerate(("安全扉", "非常停止"), start=1):
        page = document.new_page(width=842, height=595)
        page.draw_rect(fitz.Rect(60, 80, 780, 520), color=(0, 0, 0), width=1)
        page.draw_line(fitz.Point(90, 200), fitz.Point(730, 200), color=(0, 0, 0), width=2)
        page.insert_text((100, 160), label, fontname="japan", fontsize=14)
        page.insert_text((100, 240), f"Page {page_number}", fontname="helv", fontsize=10)
    document.set_toc([[1, "Machine Drawing", 1], [2, "Safety", 1], [2, "Emergency", 2]])
    return document.tobytes()


def test_pdf_translation_preserves_pages_outline_and_vector_drawings():
    raw = make_vector_drawing_pdf()
    blocks = APP.extract_pdf_blocks(raw)
    japanese = [block for block in blocks if APP.has_japanese_text(block.text)]
    assert [block.text for block in japanese] == ["安全扉", "非常停止"]

    translations = {
        japanese[0].location: "Safety door",
        japanese[1].location: "Emergency stop",
    }
    output = APP.build_translated_pdf(raw, translations, blocks)

    with fitz.open(stream=raw, filetype="pdf") as source, fitz.open(stream=output, filetype="pdf") as translated:
        assert translated.page_count == source.page_count == 2
        assert translated.get_toc() == source.get_toc()
        assert all(len(page.get_drawings()) >= 2 for page in translated)
        assert "Safety door" in translated[0].get_text()
        assert "Emergency stop" in translated[1].get_text()
        # The fast engineering-drawing path preserves the original PDF text layer
        # under the visual English overlay to avoid redaction scans across large
        # vector drawings. The output is explicitly metadata-labeled as a review copy.
        assert "review copy" in translated.metadata["subject"]


def test_pdf_spans_on_one_visual_line_are_one_translation_unit():
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 100), "安全", fontname="japan", fontsize=11)
    page.insert_text((96, 100), "扉", fontname="japan", fontsize=11)
    blocks = APP.extract_pdf_blocks(document.tobytes())
    japanese = [block for block in blocks if APP.has_japanese_text(block.text)]
    assert len(japanese) == 1
    assert japanese[0].text == "安全扉"
    assert not japanese[0].context


def test_repeated_pdf_labels_share_one_translation_key():
    document = fitz.open()
    for _ in range(50):
        page = document.new_page()
        page.insert_text((72, 100), "非常停止", fontname="japan", fontsize=11)
    blocks = APP.extract_pdf_blocks(document.tobytes())
    japanese = [block for block in blocks if APP.has_japanese_text(block.text)]
    assert len(japanese) == 50
    assert len({APP.block_translation_key(block) for block in japanese}) == 1


def test_pdf_engineering_labels_use_canonical_terminology():
    assert APP.canonical_pdf_translation("寿命時間 [年]", "Mission Time [years]") == "Service Life [years]"
    assert APP.canonical_pdf_translation("寿命時間 [年]: 20", "Mission Time [years]: 20") == "Service Life [years]: 20"
    assert APP.canonical_pdf_translation("参照記号:", "Reference Code:") == "Reference Designation:"


def test_pdf_available_rect_expands_into_empty_row_space():
    document = fitz.open()
    page = document.new_page(width=500, height=300)
    page.insert_text((50, 100), "Label", fontsize=10)
    page.insert_text((300, 100), "Value", fontsize=10)
    target = APP.pdf_available_text_rect(page, fitz.Rect(50, 88, 90, 102))
    assert target.x1 > 250
    assert target.x1 < 300


def test_pdf_shadow_text_replacements_are_spatially_deduplicated():
    replacements = [
        (fitz.Rect(50, 50, 100, 62), 10, "Project Name"),
        (fitz.Rect(50.4, 50.3, 100.4, 62.3), 10, "Project Name"),
        (fitz.Rect(200, 50, 250, 62), 10, "Status"),
    ]
    assert len(APP.dedupe_pdf_replacements(replacements)) == 2


def test_adjacent_pdf_fragments_are_consolidated_before_layout():
    replacements = [
        (fitz.Rect(50, 50, 85, 62), 10, "Project"),
        (fitz.Rect(90, 50, 130, 62), 10, "File Name:"),
        (fitz.Rect(300, 50, 340, 62), 10, "Value"),
    ]
    result = APP.consolidate_pdf_row_fragments(replacements)
    assert len(result) == 2
    assert result[0][2] == "Project File Name:"
