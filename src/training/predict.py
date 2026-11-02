"""Inferencia que devuelve las máscaras a la resolución original del frame.

La decisión 7 del protocolo (`docs/protocolo_evaluacion.md`): cada modelo predice a su resolución
de trabajo y la máscara se devuelve al tamaño del frame con vecino más cercano, y las métricas se
calculan ahí.

Hace falta porque los frames miden 640x360, 480x360 o 640x352 según el video, de modo que
reescalar a cuadrado distorsiona la relación de aspecto de forma distinta en cada uno. Medir a
512x512 obligaría a reescalar también la máscara de referencia, lo que inventa bordes y hace que
el F1 de borde deje de significar lo mismo entre videos.

Vecino más cercano y no bilineal: una máscara binaria interpolada suavemente deja de ser binaria.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEFAULT_IMG_SIZE = 512


def read_image(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def read_binary_mask(path) -> np.ndarray:
    """Máscara booleana. `convert("L")` aplica la paleta, al contrario que cv2.IMREAD_GRAYSCALE."""
    grey = np.asarray(Image.open(path).convert("L"))
    return grey > 0 if grey.max() <= 1 else grey > 127


def frame_id_of(path) -> str:
    return Path(path).stem.replace("_mask", "")


@torch.no_grad()
def predict_masks(model, pairs, device: str, img_size: int = DEFAULT_IMG_SIZE):
    """Devuelve (preds, gts) como {frame_id: máscara booleana} al tamaño original del frame.

    `pairs`: iterable de (ruta_imagen, ruta_máscara), tal como lo devuelven `list_pairs` o
    `pairs_from_manifest`. Las claves son el `frame_id`, que es lo que el arnés de la task 5.7
    usa para agrupar por video.
    """
    model.eval()
    preds, gts = {}, {}
    for image_path, mask_path in pairs:
        fid = frame_id_of(image_path)
        gt = read_binary_mask(mask_path)

        image = read_image(image_path)
        x = np.asarray(Image.fromarray(image).resize((img_size, img_size), Image.BILINEAR))
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float().div(255)
        logits = model(x.unsqueeze(0).to(device))
        small = logits.argmax(1)[0].to(torch.uint8).cpu().numpy()

        pred = np.asarray(Image.fromarray(small).resize((gt.shape[1], gt.shape[0]),
                                                        Image.NEAREST)) > 0
        preds[fid], gts[fid] = pred, gt
    return preds, gts
