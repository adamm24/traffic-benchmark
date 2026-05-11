from __future__ import annotations

import re
from dataclasses import dataclass


LETTERS = ("A", "B", "C", "D", "E")


@dataclass(frozen=True)
class ParsedAnswer:
    answer: str | None
    parse_failed: bool


def parse_answer(raw_output: str | None) -> ParsedAnswer:
    if raw_output is None:
        return ParsedAnswer(None, True)

    text = raw_output.strip().upper()
    if not text:
        return ParsedAnswer(None, True)

    patterns = (
        r"\b(?:ANSWER|CHOICE|OPTION)\b\s*(?:IS\s*)?(?:[:=]\s*)?[\(\[]?\s*([A-E])\s*[\)\]]?\b",
        r"\b(?:CHOOSE|SELECT|PICK)\s*(?:OPTION|CHOICE)?\s*[\(\[]?\s*([A-E])\s*[\)\]]?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return ParsedAnswer(match.group(1), False)

    if re.match(r"^\s*[\(\[]?\s*[A-E]\s+(?:OR|AND)\s+[A-E]\b", text):
        return ParsedAnswer(None, True)

    leading = re.match(r"^\s*[\(\[]?\s*([A-E])\s*(?:[\)\]]|[.:])?(?:\s|$)", text)
    if leading:
        return ParsedAnswer(leading.group(1), False)

    tokens = re.findall(r"(?<![A-Z])([A-E])(?![A-Z])", text)
    unique = sorted(set(tokens))
    if len(unique) == 1:
        return ParsedAnswer(unique[0], False)

    return ParsedAnswer(None, True)
