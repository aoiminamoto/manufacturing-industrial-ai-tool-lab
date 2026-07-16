"""Encoding-aware reconstruction for synthetic robot-program examples."""

from .engine import QualityEngine
from .models import RequirementProfile, TranslationResult


SUPPORTED_ENCODINGS = ("utf-8", "cp932", "shift_jis", "euc_jp", "iso2022_jp")


def decode_engineering_file(
    data: bytes, preferred: str | None = None
) -> tuple[str, str]:
    candidates = ([preferred] if preferred else []) + list(SUPPORTED_ENCODINGS)
    attempted: set[str] = set()
    for encoding in candidates:
        if encoding is None or encoding in attempted:
            continue
        attempted.add(encoding)
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("engineering-file", data, 0, len(data), "unsupported encoding")


def translate_robot_program(
    data: bytes,
    engine: QualityEngine,
    preferred_encoding: str | None = None,
) -> tuple[bytes, str, tuple[TranslationResult, ...]]:
    """Translate semicolon comments while preserving instructions and encoding."""

    source, encoding = decode_engineering_file(data, preferred_encoding)
    rebuilt: list[str] = []
    results: list[TranslationResult] = []

    for line in source.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        newline = line[len(body) :]
        instruction, marker, comment = body.partition(";")
        if marker and comment:
            result = engine.translate_text(comment, RequirementProfile.ROBOT)
            results.append(result)
            rebuilt.append(f"{instruction};{result.output}{newline}")
        else:
            rebuilt.append(line)

    return "".join(rebuilt).encode(encoding), encoding, tuple(results)
