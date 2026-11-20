"""Genera el manifiesto de folds leave-one-video-out para la task 4.8.

El protocolo de evaluacion (docs/protocolo_evaluacion.md) exige que los tres modelos entrenen
sobre exactamente el mismo reparto. Decir "las dos hacemos leave-one-video-out" no lo garantiza:
lo que lo garantiza es un unico archivo con su sha256, que cada corrida registra.

Uso:

    python -m src.data.build_fold_manifest \\
        --masks-dir "/content/drive/MyDrive/Under_Water_Dataset/Dataset_Processed/Test_Set_Curado" \\
        --out       "/content/drive/MyDrive/Under_Water_Dataset/Dataset_Processed/Seleccion/folds_4_8.csv"

Produce dos archivos:

- `folds_4_8.csv` con columnas `fold, frame_id, video, split`. Una fila por (fold, frame).
- `folds_4_8.meta.json` con el sha256 del CSV, los conteos por video y la fraccion de estructura
  de cada video, que es la linea base "todo estructura" contra la que se lee cada columna de la
  tabla final.

El sha256 se calcula sobre los bytes del CSV ya escrito, de modo que cualquiera puede verificarlo
con `sha256sum` sin volver a ejecutar este script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

MASK_SUFFIX = "_mask"


def frame_id_of(path: Path) -> str:
    return path.stem[: -len(MASK_SUFFIX)] if path.stem.endswith(MASK_SUFFIX) else path.stem


def video_of(name: str) -> int:
    match = re.search(r"Video(\d+)", name)
    if match is None:
        raise ValueError(f"no puedo deducir el video de {name!r}")
    return int(match.group(1))


def read_binary_mask(path: Path) -> np.ndarray:
    """Mascara booleana. convert("L") aplica la paleta.

    Necesario: en Test_Set_Curado hay archivos en modo P (algunos con la paleta invertida), en L
    y en RGB. Leerlos como escala de grises con cv2 devuelve el indice de paleta, no el color.
    """
    grey = np.asarray(Image.open(path).convert("L"))
    return grey > 0 if grey.max() <= 1 else grey > 127


def collect_masks(masks_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(masks_dir.glob("*.png")):
        mask = read_binary_mask(path)
        rows.append({"frame_id": frame_id_of(path), "video": video_of(path.name),
                     "fraccion_estructura": float(mask.mean()),
                     "height": int(mask.shape[0]), "width": int(mask.shape[1])})
    if not rows:
        raise SystemExit(f"no hay mascaras .png en {masks_dir}")
    return pd.DataFrame(rows).sort_values(["video", "frame_id"]).reset_index(drop=True)


def build_folds(masks: pd.DataFrame, min_masks: int) -> pd.DataFrame:
    """Un fold por video: ese video es el test y los demas el entrenamiento."""
    counts = masks.video.value_counts()
    elegibles = sorted(v for v in counts.index if counts[v] >= min_masks)
    excluidos = sorted(v for v in counts.index if counts[v] < min_masks)
    if excluidos:
        print(f"videos sin fold propio por tener menos de {min_masks} mascaras: {excluidos} "
              f"(siguen usandose como entrenamiento)")

    rows = []
    for held_out in elegibles:
        for row in masks.itertuples():
            rows.append({"fold": f"Video{held_out}", "frame_id": row.frame_id,
                         "video": f"Video{row.video}",
                         "split": "test" if row.video == held_out else "train"})
    return pd.DataFrame(rows)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--masks-dir", required=True, type=Path,
                        help="carpeta de mascaras curadas, normalmente Test_Set_Curado")
    parser.add_argument("--out", required=True, type=Path, help="CSV de salida")
    parser.add_argument("--min-masks", default=4, type=int,
                        help="minimo de mascaras para que un video tenga fold propio (def. 4)")
    args = parser.parse_args()

    masks = collect_masks(args.masks_dir)
    folds = build_folds(masks, args.min_masks)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # line_terminator fijo: el sha256 debe ser el mismo en Windows y en Colab.
    folds.to_csv(args.out, index=False, lineterminator="\n")
    checksum = sha256_of(args.out)

    por_video = masks.groupby("video").agg(
        n_mascaras=("frame_id", "size"),
        fraccion_estructura=("fraccion_estructura", "mean"),
        tamano=("width", lambda s: f"{s.iloc[0]}x{masks.loc[s.index[0], 'height']}"))
    por_video["peso_en_el_conjunto"] = por_video.n_mascaras / por_video.n_mascaras.sum()
    # Linea base "todo estructura": IoU = fraccion de estructura, por imagen y promediado.
    por_video["base_iou"] = por_video.fraccion_estructura

    meta = {
        "generado": str(date.today()),
        "masks_dir": str(args.masks_dir),
        "csv": args.out.name,
        "sha256": checksum,
        "n_mascaras": int(len(masks)),
        "n_folds": int(folds.fold.nunique()),
        "min_masks_por_fold": args.min_masks,
        "protocolo": "docs/protocolo_evaluacion.md",
        "por_video": {f"Video{v}": {
            "n_mascaras": int(r.n_mascaras),
            "peso_en_el_conjunto": round(float(r.peso_en_el_conjunto), 4),
            "base_iou_todo_estructura": round(float(r.base_iou), 4),
            "tamano": r.tamano,
        } for v, r in por_video.iterrows()},
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(masks)} mascaras | {folds.fold.nunique()} folds | {len(folds)} filas")
    print(f"\n{por_video.round(4).to_string()}")
    print(f"\n--- tamano de cada fold ---")
    print(folds.groupby(["fold", "split"]).size().unstack(fill_value=0).to_string())
    print(f"\nCSV  : {args.out}")
    print(f"meta : {meta_path}")
    print(f"sha256: {checksum}")
    print("\nCada corrida de la task 4.8 y de la 5.4 debe registrar este sha256.")


if __name__ == "__main__":
    main()
