"""Deterministic controls around an interchangeable translation adapter."""

from collections.abc import Callable

from .fragments import (
    contains_japanese,
    identifiers,
    protect_identifiers,
    restore_tokens,
    translate_japanese_fragments,
)
from .models import QualityCheck, RequirementProfile, TranslationResult
from .terminology import TerminologyController


Translator = Callable[[str], str]


class QualityEngine:
    """Apply terminology, protection, translation, and validation in sequence."""

    def __init__(
        self,
        translator: Translator,
        terminology: TerminologyController | None = None,
        hmi_character_limit: int = 24,
    ):
        self.translator = translator
        self.terminology = terminology or TerminologyController([])
        self.hmi_character_limit = hmi_character_limit

    def translate_text(
        self, source: str, profile: RequirementProfile
    ) -> TranslationResult:
        controlled, term_tokens, hits = self.terminology.inject(source)
        protected, id_tokens = protect_identifiers(controlled)
        translated = translate_japanese_fragments(protected, self.translator)
        output = restore_tokens(restore_tokens(translated, id_tokens), term_tokens)

        source_ids = identifiers(source)
        output_ids = identifiers(output)
        checks = [
            QualityCheck(
                "technical-identifiers-preserved",
                source_ids == output_ids,
                f"source={source_ids}; output={output_ids}",
            ),
            QualityCheck(
                "no-untranslated-japanese",
                not contains_japanese(output),
                "No Japanese fragments remain."
                if not contains_japanese(output)
                else "Japanese fragments remain and require engineering review.",
            ),
        ]

        warnings: list[str] = []
        if contains_japanese(output):
            warnings.append("Untranslated Japanese remains; do not auto-approve this output.")

        if profile == RequirementProfile.HMI:
            within_limit = len(output) <= self.hmi_character_limit
            checks.append(
                QualityCheck(
                    "hmi-length-within-profile",
                    within_limit,
                    f"length={len(output)}; limit={self.hmi_character_limit}",
                )
            )
            if not within_limit:
                warnings.append("HMI output exceeds the configured display-length limit.")

        return TranslationResult(
            source=source,
            output=output,
            profile=profile,
            glossary_hits=hits,
            checks=tuple(checks),
            warnings=tuple(warnings),
        )
