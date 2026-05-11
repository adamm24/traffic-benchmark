from __future__ import annotations

import hashlib
from typing import Protocol


class ModelBackend(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class DummyBackend:
    formats = ("{letter}", "{letter}.", "({letter})", "The answer is {letter}", "{letter}) selected")

    def generate(self, prompt: str) -> str:
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        letter = "ABCDE"[digest[0] % 5]
        template = self.formats[digest[1] % len(self.formats)]
        return template.format(letter=letter)

