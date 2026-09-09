"""Prepare underwater segmentation datasets for MMSegmentation.

This utility copies image/mask pairs into:

    images/training
    images/validation
    annotations/training
    annotations/validation

It intentionally does not download datasets. Keep large data in Drive and run
this script in Colab or locally against mounted Drive folders.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}


def find_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    """Match images and masks by filename stem."""
    masks_by_stem: dict[str, Path] = {}
    for mask_path in masks_dir.rglob("*"):
        if mask_path.is_file() and mask_path.suffix.lower() in MASK_EXTENSIONS:
            masks_by_stem.setdefault(mask_path.stem, mask_path)

    pairs: list[tuple[Path, Path]] = []
    for image_path in images_dir.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = masks_by_stem.get(image_path.stem)
        if mask_path is not None:
            pairs.append((image_path, mask_path))

    return sorted(pairs, key=lambda pair: pair[0].name)


def prepare_mmseg_dataset(
    images_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    validation_ratio: float,
    seed: int,
) -> dict[str, int]:
    pairs = find_pairs(images_dir, masks_dir)
    if not pairs:
        raise ValueError(f"No image/mask pairs found in {images_dir} and {masks_dir}")

    rng = random.Random(seed)
    shuffled_pairs = pairs[:]
    rng.shuffle(shuffled_pairs)

    validation_count = max(1, int(len(shuffled_pairs) * validation_ratio))
    validation_pairs = shuffled_pairs[:validation_count]
    training_pairs = shuffled_pairs[validation_count:]

    split_map = {
        "training": training_pairs,
        "validation": validation_pairs,
    }

    for split_name, split_pairs in split_map.items():
        image_out = output_dir / "images" / split_name
        mask_out = output_dir / "annotations" / split_name
        image_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        for image_path, mask_path in split_pairs:
            shutil.copy2(image_path, image_out / image_path.name)
            shutil.copy2(mask_path, mask_out / f"{image_path.stem}{mask_path.suffix.lower()}")

    return {
        "pairs": len(pairs),
        "training": len(training_pairs),
        "validation": len(validation_pairs),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a dataset for MMSegmentation.")
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--masks-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation-ratio", default=0.2, type=float)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_mmseg_dataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    print("MMSegmentation dataset prepared:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
