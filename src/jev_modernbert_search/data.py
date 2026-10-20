from __future__ import annotations

import csv
import io
import random
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TextIO

COMPETITION = "quora-question-pairs"
SAMPLE_FIELDS = ("question1", "question2", "is_duplicate")


def _open_csv(path: Path) -> TextIO:
    return path.open("r", encoding="utf-8-sig", newline="")


def iter_pair_rows(path: Path) -> Iterator[dict[str, str]]:
    if path.is_dir():
        matches = list(path.rglob("train.csv"))
        if not matches:
            raise FileNotFoundError(f"No train.csv found under {path}")
        path = matches[0]

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith("train.csv")]
            if not names:
                raise FileNotFoundError(f"No train.csv found in {path}")
            with archive.open(names[0]) as source:
                text = io.TextIOWrapper(source, encoding="utf-8-sig", newline="")
                yield from _read_rows(text)
        return

    with _open_csv(path) as source:
        yield from _read_rows(source)


def _read_rows(source: TextIO) -> Iterator[dict[str, str]]:
    reader = csv.DictReader(source)
    missing = set(SAMPLE_FIELDS).difference(reader.fieldnames or ())
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    for row in reader:
        question1 = (row.get("question1") or "").strip()
        question2 = (row.get("question2") or "").strip()
        label = (row.get("is_duplicate") or "").strip()
        if question1 and question2 and label in {"0", "1"}:
            yield {
                "question1": question1,
                "question2": question2,
                "is_duplicate": label,
            }


def reservoir_sample(
    rows: Iterable[dict[str, str]], sample_size: int, seed: int = 42
) -> list[dict[str, str]]:
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")

    rng = random.Random(seed)
    sample: list[dict[str, str]] = []
    for count, row in enumerate(rows):
        if count < sample_size:
            sample.append(row)
        else:
            replacement = rng.randrange(count + 1)
            if replacement < sample_size:
                sample[replacement] = row
    return sample


def download_quora_competition() -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Install project dependencies before downloading Kaggle data.") from exc

    return Path(kagglehub.competition_download(COMPETITION))


def write_sample(rows: Iterable[dict[str, str]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in SAMPLE_FIELDS})
            count += 1
    return count
