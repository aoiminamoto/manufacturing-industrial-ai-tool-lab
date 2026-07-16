from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quality_engine import (
    QualityEngine,
    OutputPlacement,
    RequirementProfile,
    TerminologyController,
    TerminologyEntry,
    output_contract_for,
    reconstruct_tabular_fields,
    translate_robot_program,
)


TRANSLATIONS = {
    "確認": "Check",
    "してください": "required",
    "温度": "Temperature",
}


def synthetic_translator(fragment: str) -> str:
    return TRANSLATIONS.get(fragment, fragment)


class QualityEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        terms = TerminologyController(
            [
                TerminologyEntry("非常停止", "Emergency Stop"),
                TerminologyEntry("運転準備", "Ready to Run"),
                TerminologyEntry("保守モード", "Maintenance Mode", status="draft"),
            ]
        )
        self.engine = QualityEngine(synthetic_translator, terms)

    def test_preserves_existing_english_and_identifier(self) -> None:
        result = self.engine.translate_text(
            "Ready signal M500 (非常停止)", RequirementProfile.PLC
        )
        self.assertEqual(result.output, "Ready signal M500 (Emergency Stop)")
        self.assertTrue(result.passed)

    def test_reports_unknown_japanese_instead_of_silent_approval(self) -> None:
        result = self.engine.translate_text("未登録", RequirementProfile.PLC)
        self.assertFalse(result.passed)
        self.assertIn("Untranslated Japanese remains", result.warnings[0])

    def test_draft_terminology_is_not_injected(self) -> None:
        result = self.engine.translate_text("保守モード", RequirementProfile.PLC)
        self.assertEqual(result.glossary_hits, ())
        self.assertFalse(result.passed)

    def test_hmi_profile_checks_display_length(self) -> None:
        engine = QualityEngine(synthetic_translator, hmi_character_limit=8)
        result = engine.translate_text("温度 Temperature", RequirementProfile.HMI)
        self.assertFalse(result.passed)
        self.assertTrue(any("display-length" in warning for warning in result.warnings))

    def test_robot_translation_preserves_code_and_encoding(self) -> None:
        source = ".PROGRAM demo()\n  SIGNAL X120 ;非常停止\n.END\n"
        raw = source.encode("euc_jp")
        output, encoding, results = translate_robot_program(
            raw, self.engine, preferred_encoding="euc_jp"
        )
        decoded = output.decode("euc_jp")
        contract = output_contract_for(RequirementProfile.ROBOT)
        self.assertEqual(encoding, "euc_jp")
        self.assertEqual(contract.placement, OutputPlacement.IN_PLACE)
        self.assertFalse(contract.preserve_source)
        self.assertIn("  SIGNAL X120 ;Emergency Stop", decoded)
        self.assertIn(".PROGRAM demo()", decoded)
        self.assertEqual(len(results), 1)

    def test_plc_contract_preserves_source_in_adjacent_column(self) -> None:
        result = self.engine.translate_text("非常停止", RequirementProfile.PLC)
        contract = output_contract_for(RequirementProfile.PLC)
        fields = reconstruct_tabular_fields("非常停止", result)
        self.assertEqual(contract.placement, OutputPlacement.ADJACENT_COLUMN)
        self.assertTrue(contract.preserve_source)
        self.assertEqual(fields, ("非常停止", "Emergency Stop"))

    def test_safety_plc_contract_also_supports_side_by_side_review(self) -> None:
        result = self.engine.translate_text("運転準備", RequirementProfile.SAFETY_PLC)
        fields = reconstruct_tabular_fields("運転準備", result)
        self.assertEqual(fields, ("運転準備", "Ready to Run"))

    def test_hmi_contract_replaces_text_in_place(self) -> None:
        result = self.engine.translate_text("非常停止", RequirementProfile.HMI)
        contract = output_contract_for(RequirementProfile.HMI)
        fields = reconstruct_tabular_fields("非常停止", result)
        self.assertEqual(contract.placement, OutputPlacement.IN_PLACE)
        self.assertFalse(contract.preserve_source)
        self.assertEqual(fields, ("Emergency Stop",))


if __name__ == "__main__":
    unittest.main()
