"""Japanese-fragment detection and technical-identifier protection."""

import re
from collections.abc import Callable


JAPANESE_FRAGMENT = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uff66-\uff9f]+")

# Deliberately generic examples of identifiers commonly found in engineering text.
PROTECTED_IDENTIFIER = re.compile(
    r"\b(?:[XYZMDRW]\d{1,6}|[A-Z]{2,8}[-_]\d{1,6}|[A-Z]{2,8}\d{2,8})\b"
)


def contains_japanese(text: str) -> bool:
    return bool(JAPANESE_FRAGMENT.search(text))


def protect_identifiers(text: str) -> tuple[str, dict[str, str]]:
    protected: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        token = f"__ID_{len(protected):03d}__"
        protected[token] = match.group(0)
        return token

    return PROTECTED_IDENTIFIER.sub(replace, text), protected


def translate_japanese_fragments(
    text: str, translate: Callable[[str], str]
) -> str:
    """Translate only Japanese spans, preserving surrounding English and syntax."""

    return JAPANESE_FRAGMENT.sub(lambda match: translate(match.group(0)), text)


def restore_tokens(text: str, protected: dict[str, str]) -> str:
    for token, value in protected.items():
        text = text.replace(token, value)
    return text


def identifiers(text: str) -> tuple[str, ...]:
    return tuple(PROTECTED_IDENTIFIER.findall(text))
