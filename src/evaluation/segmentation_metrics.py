"""Shared evaluation harness: same metrics, same baseline, per video."""
import re

import cv2
import numpy as np
import pandas as pd

METRICS = ["iou", "dice", "precision", "recall", "boundary_f1"]


def binarize(mask, threshold: int = 127) -> np.ndarray:
    """Foreground = grey level > threshold. Accepts 0/255 masks and 0/1 masks."""
    mask = np.asarray(mask)
    if mask.dtype == bool:
        return mask
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    return mask > (0 if mask.max() <= 1 else threshold)


def boundary(mask: np.ndarray) -> np.ndarray:
    """One-pixel inner boundary. The image border does not count as a boundary."""
    m = mask.astype(np.uint8)
    inner = cv2.erode(m, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=1)
    return (m - inner).astype(bool)


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tol: int = 2) -> float:
    """F1 between the two boundaries, a pixel matching if it lies within `tol` px of the other one."""
    bp, bg = boundary(pred), boundary(gt)
    if not bp.any() and not bg.any():
        return 1.0
    if not bp.any() or not bg.any():
        return 0.0
    dist_to_gt = cv2.distanceTransform((~bg).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_pred = cv2.distanceTransform((~bp).astype(np.uint8), cv2.DIST_L2, 3)
    precision = (dist_to_gt[bp] <= tol).mean()
    recall = (dist_to_pred[bg] <= tol).mean()
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


def frame_metrics(pred, gt, tol: int = 2) -> dict:
    """IoU, Dice, precision, recall and boundary F1 for one frame.

    Empty frame (no structure in gt and pred): IoU = Dice = 1. Undefined precision or recall is NaN.
    """
    pred, gt = binarize(pred), binarize(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    union = tp + fp + fn
    return {
        "iou": 1.0 if union == 0 else tp / union,
        "dice": 1.0 if union == 0 else 2 * tp / (2 * tp + fp + fn),
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "boundary_f1": boundary_f1(pred, gt, tol),
    }


def video_of(frame_id: str) -> str:
    match = re.search(r"Video(\d+)", frame_id)
    return f"Video{match.group(1)}" if match else "unknown"


def evaluate(preds: dict, gts: dict, tol: int = 2, n_boot: int = 2000, seed: int = 0):
    """preds, gts: {frame_id: mask}. Returns (per_frame, per_video).

    per_frame: the 5 metrics for the model and for the 'everything is structure' baseline.
    per_video: mean per metric, baseline IoU, model - baseline IoU, and 95 % bootstrap intervals over
    frames (of the model IoU and of the paired IoU difference). Frames of one video are correlated,
    so the intervals are a guide, not a test.
    """
    ids = sorted(set(preds) & set(gts))
    if not ids:
        raise ValueError("preds and gts share no frame id")
    rows = []
    for fid in ids:
        row = {"frame_id": fid, "video": video_of(fid)}
        row.update(frame_metrics(preds[fid], gts[fid], tol))
        base = frame_metrics(np.ones_like(binarize(gts[fid])), gts[fid], tol)
        row.update({f"base_{k}": v for k, v in base.items()})
        rows.append(row)
    per_frame = pd.DataFrame(rows)

    rng = np.random.default_rng(seed)

    def boot(values):
        values = np.asarray(values, dtype=float)
        means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
        return np.percentile(means, [2.5, 97.5])

    out = []
    for video, g in per_frame.groupby("video"):
        lo, hi = boot(g["iou"])
        dlo, dhi = boot(g["iou"] - g["base_iou"])
        out.append({
            "video": video, "n_frames": len(g),
            **{m: g[m].mean() for m in METRICS},
            "base_iou": g["base_iou"].mean(),
            "iou_minus_base": (g["iou"] - g["base_iou"]).mean(),
            "iou_ci_low": lo, "iou_ci_high": hi,
            "diff_ci_low": dlo, "diff_ci_high": dhi,
        })
    return per_frame, pd.DataFrame(out)


def across_seeds(per_frame_by_seed: dict) -> pd.DataFrame:
    """per_frame_by_seed: {seed: per_frame DataFrame}. Mean and sd over seeds of each video's mean."""
    per_seed = pd.concat(
        [df.groupby("video")[METRICS + ["base_iou"]].mean().assign(seed=s) for s, df in per_frame_by_seed.items()])
    grouped = per_seed.groupby("video")[METRICS + ["base_iou"]]
    table = grouped.mean().join(grouped.std(ddof=1), rsuffix="_sd")
    table["n_seeds"] = per_seed.groupby("video")["seed"].nunique()
    return table
