"""Accuracy and latency metrics. Qt-free and model-free, so fully unit-testable."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def iou(a, b) -> float:
    """IoU of two (x, y, w, h) boxes."""
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def evaluate(pred, truth, iou_thr: float = 0.5) -> dict:
    """Greedy one-to-one matching (predictions in the given order, best IoU first)."""
    free = list(range(len(truth)))
    tp = 0
    for p in pred:
        best = max(free, key=lambda j: iou(p, truth[j]), default=None)
        if best is not None and iou(p, truth[best]) >= iou_thr:
            free.remove(best)
            tp += 1
    fp, fn = len(pred) - tp, len(truth) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


def load_ground_truth(path) -> list[tuple[float, float, float, float]]:
    """Read ground truth: JSON list of [x, y, w, h] pixel boxes, or {"boxes": [...]}."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read ground-truth file: {exc}") from exc
    boxes = data.get("boxes") if isinstance(data, dict) else data
    try:
        out = [tuple(float(v) for v in b) for b in boxes]
    except (TypeError, ValueError) as exc:
        raise ValueError("Ground truth must be a list of [x, y, w, h] boxes") from exc
    if any(len(b) != 4 or b[2] <= 0 or b[3] <= 0 for b in out):
        raise ValueError("Each box needs 4 numbers [x, y, w, h] with positive w and h")
    return out


def latency_summary(times_ms) -> dict:
    a = np.asarray(times_ms, dtype=float)
    if a.size == 0:
        raise ValueError("No timing samples")
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "min": float(a.min()),
            "max": float(a.max()), "fps": 1000.0 / float(a.mean()), "n": int(a.size)}
