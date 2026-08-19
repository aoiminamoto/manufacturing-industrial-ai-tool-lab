import importlib.util
from pathlib import Path


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
