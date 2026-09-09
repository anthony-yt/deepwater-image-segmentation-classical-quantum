"""Validate the expected MMSegmentation directory structure."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_DIRS = [
    "images/training",
    "images/validation",
    "annotations/training",
    "annotations/validation",
]


def count_files(path: Path) -> int:
    return sum(1 for item in path.iterdir() if item.is_file()) if path.exists() else 0


def validate_dataset(dataset_dir: Path) -> dict[str, int]:
    missing = [name for name in REQUIRED_DIRS if not (dataset_dir / name).is_dir()]
    if missing:
        raise FileNotFoundError("Missing required directories: " + ", ".join(missing))

    counts = {name: count_files(dataset_dir / name) for name in REQUIRED_DIRS}
    if counts["images/training"] != counts["annotations/training"]:
        raise ValueError("Training image and annotation counts do not match.")
    if counts["images/validation"] != counts["annotations/validation"]:
        raise ValueError("Validation image and annotation counts do not match.")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an MMSegmentation dataset folder.")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    counts = validate_dataset(parse_args().dataset_dir)
    print("MMSegmentation dataset structure is valid:")
    for name, count in counts.items():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
