"""Approved terminology control independent of any model provider."""

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TerminologyEntry:
    japanese: str
    english: str
    status: str = "approved"
    source: str = "synthetic-demo"

    @property
    def approved(self) -> bool:
        return self.status.casefold() == "approved"


class TerminologyController:
    def __init__(self, entries: list[TerminologyEntry] | tuple[TerminologyEntry, ...]):
        self.entries = tuple(
            sorted(
                (entry for entry in entries if entry.approved),
                key=lambda entry: len(entry.japanese),
                reverse=True,
            )
        )

    @classmethod
    def from_csv(cls, path: str | Path) -> "TerminologyController":
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(handle)
            return cls(
                [
                    TerminologyEntry(
                        japanese=row["japanese"].strip(),
                        english=row["english"].strip(),
                        status=row.get("status", "approved").strip(),
                        source=row.get("source", "synthetic-demo").strip(),
                    )
                    for row in rows
                    if row.get("japanese") and row.get("english")
                ]
            )

    def inject(self, text: str) -> tuple[str, dict[str, str], tuple[str, ...]]:
        replacements: dict[str, str] = {}
        hits: list[str] = []
        controlled = text

        for entry in self.entries:
            while entry.japanese in controlled:
                token = f"__TERM_{len(replacements):03d}__"
                controlled = controlled.replace(entry.japanese, token, 1)
                replacements[token] = entry.english
                hits.append(f"{entry.japanese} -> {entry.english}")

        return controlled, replacements, tuple(hits)
