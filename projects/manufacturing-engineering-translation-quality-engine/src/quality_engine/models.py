"""Domain models shared by the quality-engine pipeline."""

from dataclasses import dataclass
from enum import Enum


class RequirementProfile(str, Enum):
    """Output constraints for representative engineering artifacts."""

    PLC = "plc"
    ROBOT = "robot"
    HMI = "hmi"
    STRUCTURED_FILE = "structured-file"


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
