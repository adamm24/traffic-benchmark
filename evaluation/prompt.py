from __future__ import annotations

from evaluation.loader import Example, LETTERS


INSTRUCTION = "Answer with only the letter of your choice (A, B, C, D, or E)."


def build_prompt(example: Example) -> str:
    choice_lines = [f"{letter}) {example.choices[letter]}" for letter in LETTERS]
    return "\n\n".join(
        [
            example.prompt.rstrip(),
            "\n".join(choice_lines),
            INSTRUCTION,
        ]
    )

