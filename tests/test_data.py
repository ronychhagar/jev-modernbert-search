import csv
import tempfile
import unittest
from pathlib import Path

from jev_modernbert_search.data import iter_pair_rows, reservoir_sample, write_sample


class KaggleDataTests(unittest.TestCase):
    def test_iter_pair_rows_filters_empty_questions_and_invalid_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "pairs.csv"
            source.write_text(
                "id,question1,question2,is_duplicate\n"
                "1,How are you?,How are you doing?,1\n"
                "2,,Missing question,0\n"
                "3,Question,Other,not-a-label\n",
                encoding="utf-8",
            )

            self.assertEqual(
                list(iter_pair_rows(source)),
                [
                    {
                        "question1": "How are you?",
                        "question2": "How are you doing?",
                        "is_duplicate": "1",
                    }
                ],
            )

    def test_reservoir_sample_is_deterministic_and_bounded(self):
        rows = [{"value": str(index)} for index in range(100)]

        self.assertEqual(reservoir_sample(rows, 10, seed=9), reservoir_sample(rows, 10, seed=9))
        self.assertEqual(len(reservoir_sample(rows, 10, seed=9)), 10)
        self.assertEqual(len(reservoir_sample(rows, 200, seed=9)), 100)

    def test_write_sample_creates_parent_and_csv_header(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "sample.csv"
            rows = [
                {
                    "question1": "Q1",
                    "question2": "Q2",
                    "is_duplicate": "1",
                }
            ]

            self.assertEqual(write_sample(rows, output), 1)
            with output.open(encoding="utf-8", newline="") as source:
                self.assertEqual(list(csv.DictReader(source)), rows)

    def test_iter_pair_rows_reports_missing_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "invalid.csv"
            source.write_text("question1,question2\none,two\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                list(iter_pair_rows(source))

if __name__ == "__main__":
    unittest.main()
