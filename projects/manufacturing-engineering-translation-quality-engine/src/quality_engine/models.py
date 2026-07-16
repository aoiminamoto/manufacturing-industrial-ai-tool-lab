"""Domain models shared by the quality-engine pipeline."""

from dataclasses import dataclass
from enum import Enum


class RequirementProfile(str, Enum):
    """Output constraints for representative engineering artifacts."""

    PLC = "plc"
    SAFETY_PLC = "safety-plc"
    ROBOT = "robot"
    HMI = "hmi"
    STRUCTURED_FILE = "structured-file"


class OutputPlacement(str, Enum):
    """Where translated content is reconstructed for the floor-level user."""

    ADJACENT_COLUMN = "adjacent-column"
    IN_PLACE = "in-place"


@dataclass(frozen=True)
class OutputContract:
    """A user-visible promise for a requirement-specific output document."""

    profile: RequirementProfile
    placement: OutputPlacement
    preserve_source: bool
    floor_workflow: str


@dataclass(frozen=True)
class QualityCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class TranslationResult:
    source: str
    output: str
    profile: RequirementProfile
    glossary_hits: tuple[str, ...]
    checks: tuple[QualityCheck, ...]
    warnings: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)
