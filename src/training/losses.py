"""Perdidas de segmentacion binaria, todas respetando `ignore_index`.

La task 4.2(c) pide variar la perdida y la task 5.11(c) compara CE+Dice contra Focal o Tversky.
Las cuatro se construyen con `make_criterion`, para que la ablacion cambie una sola variable.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = 255
EPS = 1e-6


def _valid_and_target(target: torch.Tensor, ignore_index: int):
    """Mascara de pixeles validos y target binario float, ya con los ignorados puestos a 0."""
    valid = (target != ignore_index)
    t = ((target == 1) & valid).float()
    return valid, t


def dice_loss(logits: torch.Tensor, target: torch.Tensor,
              ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """1 - Dice soft de la clase 1, por imagen y promediado."""
    valid, t = _valid_and_target(target, ignore_index)
    probs = F.softmax(logits, dim=1)[:, 1] * valid.float()
    inter = (probs * t).sum(dim=(1, 2))
    denom = probs.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
    return 1.0 - ((2 * inter + EPS) / (denom + EPS)).mean()


def tversky_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.7,
                 beta: float = 0.3, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """1 - Tversky. alpha pesa los falsos negativos y beta los falsos positivos.

    alpha = beta = 0.5 equivale a Dice. alpha > beta castiga mas omitir estructura; es la
    direccion util cuando el modelo tiende a predecir fondo.
    """
    valid, t = _valid_and_target(target, ignore_index)
    probs = F.softmax(logits, dim=1)[:, 1] * valid.float()
    tp = (probs * t).sum(dim=(1, 2))
    fn = ((1 - probs) * t).sum(dim=(1, 2))
    fp = (probs * (1 - t) * valid.float()).sum(dim=(1, 2))
    tversky = (tp + EPS) / (tp + alpha * fn + beta * fp + EPS)
    return 1.0 - tversky.mean()


def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0,
               alpha: float = 0.25, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Focal loss de Lin et al. (2017) sobre las dos clases.

    `alpha` es el peso de la clase 1 (estructura); 1 - alpha el de la clase 0.
    """
    valid = (target != ignore_index)
    safe = torch.where(valid, target, torch.zeros_like(target))
    log_probs = F.log_softmax(logits, dim=1)
    log_pt = log_probs.gather(1, safe.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    weight = torch.where(safe == 1, torch.full_like(pt, alpha), torch.full_like(pt, 1 - alpha))
    loss = -weight * (1 - pt).pow(gamma) * log_pt
    loss = loss * valid.float()
    denom = valid.float().sum().clamp(min=1.0)
    return loss.sum() / denom


def make_criterion(name: str = "ce_dice", ignore_index: int = IGNORE_INDEX):
    """Devuelve una funcion (logits, target) -> escalar.

    Nombres: `ce_dice` (la del baseline), `ce`, `dice`, `focal`, `tversky`.
    """
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
    name = name.lower()

    if name == "ce_dice":
        return lambda logits, target: ce(logits, target) + dice_loss(logits, target, ignore_index)
    if name == "ce":
        return lambda logits, target: ce(logits, target)
    if name == "dice":
        return lambda logits, target: dice_loss(logits, target, ignore_index)
    if name == "focal":
        return lambda logits, target: focal_loss(logits, target, ignore_index=ignore_index)
    if name == "tversky":
        return lambda logits, target: tversky_loss(logits, target, ignore_index=ignore_index)
    raise ValueError(f"perdida desconocida: {name}")
