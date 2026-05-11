from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.loader import Example
from evaluation.metrics import compute_metrics
from evaluation.parser import parse_answer
from evaluation.prompt import INSTRUCTION, build_prompt


class ParserTests(unittest.TestCase):
    def test_common_answer_formats(self):
        cases = {
            "B": "B",
            "B.": "B",
            "(B)": "B",
            "B)": "B",
            "B:": "B",
            "The answer is B": "B",
            "Answer: B": "B",
            "The correct answer is:\n\nD) Vehicle C": "D",
            "B) because it yields": "B",
            "C) Vehicle A": "C",
            "D) Vehicle A": "D",
            "E) Vehicle C": "E",
            "E) Vehicle B is directly behind Vehicle A on the road.": "E",
        }
        for raw, expected in cases.items():
            parsed = parse_answer(raw)
            self.assertFalse(parsed.parse_failed)
            self.assertEqual(parsed.answer, expected)

    def test_ambiguous_output_fails(self):
        parsed = parse_answer("A or B")
        self.assertTrue(parsed.parse_failed)
        self.assertIsNone(parsed.answer)


class PromptTests(unittest.TestCase):
    def test_prompt_contains_choices_and_instruction(self):
        example = Example(
            id="x",
            task="position_tracking",
            prompt="Question text",
            choices={"A": "one", "B": "two", "C": "three", "D": "four", "E": "five"},
            answer="C",
            source_file="task",
        )
        prompt = build_prompt(example)
        self.assertIn("Question text", prompt)
        self.assertIn("A) one", prompt)
        self.assertIn("E) five", prompt)
        self.assertTrue(prompt.endswith(INSTRUCTION))


class MetricsTests(unittest.TestCase):
    def test_metrics_count_accuracy_distribution_and_parse_failures(self):
        rows = [
            {
                "id": "1",
                "task": "t1",
                "correct_answer": "A",
                "raw_response": "A",
                "parsed_answer": "A",
                "is_correct": True,
                "parse_failed": False,
            },
            {
                "id": "2",
                "task": "t1",
                "correct_answer": "B",
                "raw_response": "A",
                "parsed_answer": "A",
                "is_correct": False,
                "parse_failed": False,
            },
            {
                "id": "3",
                "task": "t2",
                "correct_answer": "C",
                "raw_response": "",
                "parsed_answer": None,
                "is_correct": False,
                "parse_failed": True,
            },
        ]
        metrics = compute_metrics(rows)
        self.assertEqual(metrics["overall_accuracy"], 0.3333)
        self.assertEqual(metrics["answer_distribution"]["A"]["count"], 2)
        self.assertEqual(metrics["format_errors"]["count"], 1)
        self.assertEqual(metrics["per_task_accuracy"]["t1"]["correct"], 1)


if __name__ == "__main__":
    unittest.main()
