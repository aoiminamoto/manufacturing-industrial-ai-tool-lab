import importlib.util
import io
import sys
import types
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


def install_dependency_stubs() -> None:
    if importlib.util.find_spec("streamlit") is None:
        streamlit = types.ModuleType("streamlit")

        def identity_decorator(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return lambda func: func

        streamlit.cache_resource = identity_decorator
        streamlit.fragment = identity_decorator
        sys.modules["streamlit"] = streamlit
    if importlib.util.find_spec("dotenv") is None:
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda *args, **kwargs: None
        sys.modules["dotenv"] = dotenv
    if importlib.util.find_spec("openai") is None:
        openai = types.ModuleType("openai")
        for name in ("APIConnectionError", "APIStatusError", "AuthenticationError", "RateLimitError"):
            setattr(openai, name, type(name, (Exception,), {}))
        openai.OpenAI = object
        sys.modules["openai"] = openai


install_dependency_stubs()
APP_PATH = Path(__file__).with_name("upload-app.py")
SPEC = importlib.util.spec_from_file_location("public_term1_app", APP_PATH)
APP = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = APP
SPEC.loader.exec_module(APP)


SLIDE_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree>
    <p:sp><p:txBody><a:bodyPr/><a:lstStyle/>
      <a:p><a:r><a:t>設備を起動する前に、作業者は</a:t></a:r></a:p>
      <a:p><a:r><a:t>安全扉が閉じていることを確認してください。</a:t></a:r></a:p>
    </p:txBody></p:sp>
    <p:sp><p:txBody><a:bodyPr/><a:lstStyle/>
      <a:p><a:r><a:t>・主電源を確認する</a:t></a:r></a:p>
      <a:p><a:r><a:t>・安全扉を閉じる</a:t></a:r></a:p>
    </p:txBody></p:sp>
  </p:spTree></p:cSld>
</p:sld>
"""


def pptx_bytes() -> bytes:
    target = io.BytesIO()
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        archive.writestr("ppt/slides/slide1.xml", SLIDE_XML)
    return target.getvalue()


def test_textbox_is_one_semantic_block():
    blocks = APP.extract_pptx_blocks(pptx_bytes())
    assert blocks[0].location == "ppt/slides/slide1.xml#textbox:0"
    assert blocks[0].text == "設備を起動する前に、作業者は\n安全扉が閉じていることを確認してください。"


def test_continuous_textbox_writes_one_complete_translation():
    blocks = APP.extract_pptx_blocks(pptx_bytes())
    translation = "Before starting the equipment, the operator must confirm that the safety door is closed."
    output = APP.build_translated_pptx(pptx_bytes(), {blocks[0].location: translation})
    with ZipFile(io.BytesIO(output)) as archive:
        root = APP.ET.fromstring(archive.read("ppt/slides/slide1.xml"))
    paragraphs = APP.ppt_text_bodies(root)[0].findall("a:p", APP.PPT_NS)
    assert len(paragraphs) == 1
    assert APP.ppt_paragraph_text(paragraphs[0]) == translation


def test_list_items_remain_separate():
    blocks = APP.extract_pptx_blocks(pptx_bytes())
    output = APP.build_translated_pptx(
        pptx_bytes(),
        {blocks[1].location: "- Check main power\n- Close the safety door"},
    )
    with ZipFile(io.BytesIO(output)) as archive:
        root = APP.ET.fromstring(archive.read("ppt/slides/slide1.xml"))
    paragraphs = APP.ppt_text_bodies(root)[1].findall("a:p", APP.PPT_NS)
    assert [APP.ppt_paragraph_text(paragraph) for paragraph in paragraphs] == [
        "- Check main power",
        "- Close the safety door",
    ]


def test_quality_gate_detects_missing_actor():
    source = "設備を起動する前に、作業者は安全扉が閉じていることを確認してください。"
    assert "Missing actor: 作業者" in APP.powerpoint_translation_quality_issues(
        source,
        "Confirm safety door closed before equipment start.",
    )
    assert APP.powerpoint_translation_quality_issues(
        source,
        "Before starting the equipment, the operator must confirm that the safety door is closed.",
    ) == []


def test_quality_gate_checks_negation_and_required_term():
    source = "異常が解除されていない状態で運転準備ボタンを押さないでください。"
    hit = APP.TermHit(jp="運転準備", en="Master ON", count=1)
    issues = APP.powerpoint_translation_quality_issues(source, "Press the preparation button.", [hit])
    assert "Missing negation or prohibition" in issues
    assert "Missing required glossary term: 運転準備 → Master ON" in issues


def test_preview_exposes_all_available_glossary_columns():
    block = APP.TextBlock(location="ppt/slides/slide1.xml#textbox:0", text="安全扉を閉じる")
    glossary = pd.DataFrame(
        [{
            "JP": "安全扉",
            "EN": "safety door",
            "Validated By": "Synthetic Reviewer",
            "Validated Date": "2026-08-10",
            "Approved By": "Synthetic Approver",
            "Category": "Synthetic Safety",
        }]
    )
    preview = APP.translation_pairs_preview(
        [block],
        {block.location: "Close the safety door"},
        glossary=glossary,
    )
    assert preview.loc[0, "Glossary Match"] == "安全扉 → safety door"
    assert preview.loc[0, "Glossary Validated By"] == "Synthetic Reviewer"
    assert preview.loc[0, "Glossary Approved By"] == "Synthetic Approver"
    assert preview.loc[0, "Glossary Category"] == "Synthetic Safety"


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} public-safe PowerPoint semantic-quality tests passed")
