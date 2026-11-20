"""Bucle de entrenamiento instrumentado para los modelos clasicos.

Diferencias con el bucle del baseline (`3_1_unet_baseline_training.ipynb`):

- Registra **perdida de entrenamiento y de validacion**, y las metricas sobre **ambos** splits.
  El original solo guardaba `train_loss`, `val_miou` y `val_dice`, con lo que no se podia
  distinguir sobreajuste de colapso a la clase mayoritaria.
- Registra en cada epoca la **linea base "todo estructura"** del split de validacion, para que
  cualquier curva se lea contra ella y no en el aire.
- Excluye los pixeles `ignore_index` de la perdida y de las metricas.
- Devuelve el historial completo, serializable, para no tener que reentrenar para volver a
  graficar.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

IGNORE_INDEX = 255


@torch.no_grad()
def _accumulate(pred: torch.Tensor, target: torch.Tensor, ignore_index: int):
    """IoU y Dice de la clase 1 por imagen. Devuelve dos listas."""
    valid = (target != ignore_index)
    p = (pred == 1) & valid
    t = (target == 1) & valid
    inter = (p & t).sum(dim=(1, 2)).float()
    union = (p | t).sum(dim=(1, 2)).float()
    total = p.sum(dim=(1, 2)).float() + t.sum(dim=(1, 2)).float()
    # Imagen sin estructura ni prediccion: IoU y Dice valen 1 por convencion, igual que en
    # src/evaluation/segmentation_metrics.py (task 5.7).
    iou = torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))
    dice = torch.where(total > 0, 2 * inter / total.clamp(min=1), torch.ones_like(total))
    return iou.tolist(), dice.tolist()


@torch.no_grad()
def evaluate_loader(model, loader, device: str, criterion=None,
                    ignore_index: int = IGNORE_INDEX) -> dict:
    """Perdida media, IoU y Dice medios por imagen sobre un loader."""
    model.eval()
    ious, dices, losses = [], [], []
    for image, mask in loader:
        image, mask = image.to(device), mask.to(device)
        logits = model(image)
        if criterion is not None:
            losses.append(float(criterion(logits, mask)))
        i, d = _accumulate(logits.argmax(1), mask, ignore_index)
        ious += i
        dices += d
    out = {"iou": float(np.mean(ious)), "dice": float(np.mean(dices)), "n": len(ious)}
    if criterion is not None:
        out["loss"] = float(np.mean(losses))
    return out


@torch.no_grad()
def baseline_metrics(loader, ignore_index: int = IGNORE_INDEX) -> dict:
    """Linea base trivial: predecir que todo es estructura, y que todo es fondo.

    Es la referencia que pide la task 4.2(b) y la que usa el arnes compartido de la task 5.7.
    """
    all_struct_iou, all_struct_dice, all_bg_iou, fg = [], [], [], []
    for _, mask in loader:
        ones = torch.ones_like(mask)
        zeros = torch.zeros_like(mask)
        i, d = _accumulate(ones, mask, ignore_index)
        all_struct_iou += i
        all_struct_dice += d
        i0, _ = _accumulate(zeros, mask, ignore_index)
        all_bg_iou += i0
        valid = (mask != ignore_index)
        fg += (((mask == 1) & valid).sum(dim=(1, 2)).float()
               / valid.sum(dim=(1, 2)).float().clamp(min=1)).tolist()
    return {"todo_estructura_iou": float(np.mean(all_struct_iou)),
            "todo_estructura_dice": float(np.mean(all_struct_dice)),
            "todo_fondo_iou": float(np.mean(all_bg_iou)),
            "fraccion_estructura": float(np.mean(fg)),
            "n": len(fg)}


def train_model(model, train_loader, val_loader, criterion, device: str, epochs: int = 40,
                lr: float = 1e-4, scheduler: str = "cosine", ignore_index: int = IGNORE_INDEX,
                eval_train: bool = True, log_every: int = 5, select_on: str = "dice",
                quiet: bool = False) -> dict:
    """Entrena y devuelve el historial completo.

    `scheduler`: "cosine" (como el baseline) o "constant".
    `select_on`: metrica de validacion con la que se elige el mejor checkpoint.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
             if scheduler == "cosine" else None)

    base = baseline_metrics(val_loader, ignore_index)
    history = {k: [] for k in ["epoch", "train_loss", "val_loss", "train_iou", "train_dice",
                               "val_iou", "val_dice", "lr"]}
    best = {"score": -1.0, "epoch": -1, "state": None}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for image, mask in train_loader:
            image, mask = image.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = criterion(model(image), mask)
            loss.backward()
            optimizer.step()
            running += float(loss)
        current_lr = optimizer.param_groups[0]["lr"]
        if sched is not None:
            sched.step()

        val = evaluate_loader(model, val_loader, device, criterion, ignore_index)
        train_eval = (evaluate_loader(model, train_loader, device, None, ignore_index)
                      if eval_train else {"iou": float("nan"), "dice": float("nan")})

        history["epoch"].append(epoch)
        history["train_loss"].append(running / max(len(train_loader), 1))
        history["val_loss"].append(val["loss"])
        history["train_iou"].append(train_eval["iou"])
        history["train_dice"].append(train_eval["dice"])
        history["val_iou"].append(val["iou"])
        history["val_dice"].append(val["dice"])
        history["lr"].append(current_lr)

        score = val[select_on]
        if score > best["score"]:
            best = {"score": score, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

        if not quiet and (epoch % log_every == 0 or epoch == 1):
            print(f"epoca {epoch:3d} | train loss {history['train_loss'][-1]:.4f} "
                  f"| val loss {val['loss']:.4f} | val IoU {val['iou']:.4f} "
                  f"| val Dice {val['dice']:.4f}")

    return {"history": history, "baseline": base, "best_epoch": best["epoch"],
            f"best_val_{select_on}": best["score"], "best_state": best["state"],
            "seconds": time.time() - t0,
            "config": {"epochs": epochs, "lr": lr, "scheduler": scheduler,
                       "select_on": select_on}}


def save_history(result: dict, path) -> None:
    """Guarda historial, linea base y configuracion en JSON. No guarda los pesos."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in result.items() if k != "best_state"}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("historial guardado en", path)
