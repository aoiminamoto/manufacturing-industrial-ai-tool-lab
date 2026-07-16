"""Run a synthetic demonstration without calling an external AI service."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from quality_engine import QualityEngine, RequirementProfile, TerminologyController


SYNTHETIC_TRANSLATIONS = {
    "確認": "Check",
    "してください": "required",
    "を確認してください": " - Check required",
}


def demo_translator(fragment: str) -> str:
    return SYNTHETIC_TRANSLATIONS.get(fragment, fragment)


def main() -> None:
    terminology = TerminologyController.from_csv(ROOT / "synthetic-data" / "glossary.csv")
    engine = QualityEngine(demo_translator, terminology)
    result = engine.translate_text(
        "Ready signal M500: 非常停止を確認してください", RequirementProfile.PLC
    )

    print(f"Output: {result.output}")
    print(f"Passed: {result.passed}")
    print("Glossary hits:")
    for hit in result.glossary_hits:
        print(f"  - {hit}")
    print("Quality checks:")
    for check in result.checks:
        print(f"  - {check.name}: {'PASS' if check.passed else 'REVIEW'}")


if __name__ == "__main__":
    main()
