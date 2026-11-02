"""Dataset de pares imagen/mascara para los modelos clasicos, con soporte de ignore_index.

Corrige dos cosas del dataset original de `3_1_unet_baseline_training.ipynb`:

1. **Mascaras de paleta.** El original usaba `cv2.imread(..., IMREAD_GRAYSCALE)`, que en un PNG
   en modo P devuelve el indice de paleta y no el color. En `Test_Set_Curado` hay 19 archivos en
   modo P (6 con la paleta invertida), 8 en L y 1 en RGB: con cv2, parte de las mascaras se leen
   al reves. Aqui se usa `PIL.Image.convert("L")`, que aplica la paleta.

2. **Padding de la augmentacion de escala.** `zoom_in_out_transforms` (task 1.2) rellena con
   `mask_fill=255` para marcar los pixeles que deben ignorarse. El original hacia
   `mask = (mask > 0)` *antes* de transformar y no declaraba `ignore_index`, asi que el relleno
   terminaba contando como estructura y la variante con escala salia peor por un bug y no por la
   augmentacion. Aqui la mascara se binariza a {0, 1} antes de transformar y el valor 255 que
   introduce el padding se conserva como ignore.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = 255


def list_pairs(root, split: str, image_ext: str = "*.png") -> list:
    """Pares (imagen, mascara) de un dataset en formato MMSegmentation.

    Espera `root/images/<split>` y `root/annotations/<split>` con el mismo nombre de archivo.
    """
    root = Path(root)
    img_dir = root / "images" / split
    ann_dir = root / "annotations" / split
    pairs = []
    for img in sorted(glob.glob(str(img_dir / image_ext))):
        mask = ann_dir / os.path.basename(img)
        if mask.exists():
            pairs.append((img, str(mask)))
    return pairs


def read_image(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def read_binary_mask(path) -> np.ndarray:
    """Mascara en {0, 1} uint8. Acepta mascaras indice (0/1) y mascaras 0/255, en modo P, L o RGB."""
    grey = np.asarray(Image.open(path).convert("L"))
    binary = grey > 0 if grey.max() <= 1 else grey > 127
    return binary.astype(np.uint8)


class PipeDataset(Dataset):
    """Pares imagen/mascara con una pipeline de albumentations.

    La mascara devuelta puede contener el valor `IGNORE_INDEX` si la pipeline hace padding; la
    perdida y las metricas deben excluir esos pixeles.
    """

    def __init__(self, pairs: list, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        image = read_image(img_path)
        mask = read_binary_mask(mask_path)
        out = self.transform(image=image, mask=mask)
        image, mask = out["image"], out["mask"]
        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        return image, torch.from_numpy(np.ascontiguousarray(mask)).long()


def frame_id_of(path) -> str:
    """Identificador de frame a partir del nombre de archivo, para agrupar por video."""
    return Path(path).stem.replace("_mask", "")
