import importlib.util
from pathlib import Path
from types import SimpleNamespace


APP_PATH = Path(__file__).with_name("upload-app.py")
SPEC = importlib.util.spec_from_file_location("term1_upload_app_image_test", APP_PATH)
APP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(APP)


def test_ocr_evidence_match_accepts_only_same_visible_text():
    assert APP.verified_ocr_text_matches("安全 扉", "安全扉")
    assert APP.verified_ocr_text_matches("ｷｬﾝｾﾙ", "キャンセル")
    assert not APP.verified_ocr_text_matches("安全扉", "非常停止")
    assert not APP.verified_ocr_text_matches("登録", "")


def test_production_ocr_thresholds_are_conservative():
    assert APP.HMI_MIN_OCR_CONFIDENCE >= 0.80
    assert APP.HMI_MIN_VERIFICATION_CONFIDENCE >= 0.90


class RecordingResponses:
    def __init__(self, output_text):
        self.output_text = output_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text, usage=None)


def test_full_image_ocr_sends_one_untouched_high_detail_image(monkeypatch):
    responses = RecordingResponses(
        '{"regions":[{"text":"安全扉","x":10,"y":20,"width":80,"height":30,'
        '"confidence":0.98,"kind":"parameter_label","visible_context":"door setting"}]}'
    )
    monkeypatch.setattr(APP, "openai_client", lambda: SimpleNamespace(responses=responses))

    regions, _usage = APP.extract_full_image_text_regions_with_vision(
        b"original-image-bytes",
        "screen.png",
        640,
        480,
    )

    assert [region.jp for region in regions] == ["安全扉"]
    content = responses.calls[0]["input"][0]["content"]
    images = [item for item in content if item["type"] == "input_image"]
    assert len(images) == 1
    assert images[0]["detail"] == "high"
    assert images[0]["image_url"].startswith("data:image/png;base64,")


def test_translation_receives_complete_original_image_context(monkeypatch):
    responses = RecordingResponses("[BLOCK 1]\nSafety door\n[/BLOCK 1]")
    monkeypatch.setattr(APP, "openai_client", lambda: SimpleNamespace(responses=responses))
    region = APP.HmiTextRegion("hmi:1", "安全扉", 10, 20, 80, 30, 0.98)

    translations, _hits, _usage = APP.translate_hmi_regions(
        [region],
        APP.empty_terms_dataframe(),
        raw_image=b"original-image-bytes",
        file_name="screen.png",
    )

    assert translations["hmi:1"] == "Safety door"
    content = responses.calls[0]["input"][0]["content"]
    assert any(item["type"] == "input_image" for item in content)
