"""Package local Task 4.8 sources for Colab without a commit or push."""
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    'src/data/build_final_retraining_manifest.py',
    'src/evaluation/segmentation_metrics.py',
    'src/training/final_retraining.py',
    'src/training/mmseg_final_backend.py',
    'requirements-final-retraining.txt',
    'docs/final_retraining_segformers.md',
    'configs/final_retraining_segformers.example.json',
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('/tmp/task4_8_sources.zip'))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.out, 'w', ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(ROOT / name, name)
    print(args.out.resolve())
