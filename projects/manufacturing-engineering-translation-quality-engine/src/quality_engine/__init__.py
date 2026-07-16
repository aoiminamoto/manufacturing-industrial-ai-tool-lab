"""Public API for the clean-room engineering translation quality engine."""

from .encoding import decode_engineering_file, translate_robot_program
from .engine import QualityEngine
from .models import (
    OutputContract,
    OutputPlacement,
    QualityCheck,
    RequirementProfile,
    TranslationResult,
)
from .output_contracts import output_contract_for, reconstruct_tabular_fields
from .terminology import TerminologyController, TerminologyEntry

__all__ = [
    "OutputContract",
    "OutputPlacement",
    "QualityCheck",
    "QualityEngine",
    "RequirementProfile",
    "TerminologyController",
    "TerminologyEntry",
    "TranslationResult",
    "decode_engineering_file",
    "output_contract_for",
    "reconstruct_tabular_fields",
    "translate_robot_program",
]
