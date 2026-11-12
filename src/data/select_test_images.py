"""Select a small deterministic set of test images for qualitative inference."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def select_test_images(source_dir: Path, output_dir: Path, count: int) -> list[Path]:
    images = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and "mask" not in path.name.lower()
    )
    if len(images) < count:
        raise ValueError(f"Found only {len(images)} images in {source_dir}; need {count}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_paths: list[Path] = []
    for image_path in images[:count]:
        target_path = output_dir / image_path.name
        shutil.copy2(image_path, target_path)
        selected_paths.append(target_path)

    return selected_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select test images for qualitative inference.")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", default=5, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_paths = select_test_images(args.source_dir, args.output_dir, args.count)
    print("Selected test images:")
    for path in selected_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
