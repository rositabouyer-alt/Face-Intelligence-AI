"""Face detection + landmarks (MediaPipe, CPU) and annotation drawing. Qt-free."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

CYAN = (238, 211, 34)     # BGR of #22D3EE
VIOLET = (246, 92, 139)   # BGR of #8B5CF6


class DetectorError(RuntimeError):
    """Raised when the MediaPipe models cannot be loaded."""


@dataclass
class Face:
    box: tuple[int, int, int, int]            # x, y, w, h in pixels
    score: float
    landmarks: np.ndarray | None = None       # (468, 2) pixel coords, if computed


def _iou(a, b) -> float:
    iw = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class FaceDetector:
    """MediaPipe BlazeFace detector with optional 468-point FaceMesh landmarks.

    Runs at a low internal threshold; `detect(min_conf=...)` filters, so the confidence
    slider needs no model rebuild. Not thread-safe: create one instance per thread.
    """

    def __init__(self, model_selection: int = 1, static: bool = False, landmarks: bool = False):
        try:
            import mediapipe as mp
            self._sol = mp.solutions
            self._det = self._sol.face_detection.FaceDetection(
                model_selection=model_selection, min_detection_confidence=0.2)
        except Exception as exc:  # missing package, DLL error, broken model
            raise DetectorError(f"Could not load MediaPipe face detector: {exc}") from exc
        self.static, self.landmarks, self._mesh = static, landmarks, None

    def _get_mesh(self):
        if self._mesh is None:
            try:
                self._mesh = self._sol.face_mesh.FaceMesh(
                    static_image_mode=self.static, max_num_faces=10,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
            except Exception as exc:
                raise DetectorError(f"Could not load landmark model: {exc}") from exc
        return self._mesh

    def detect(self, bgr: np.ndarray, min_conf: float = 0.5) -> list[Face]:
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("Expected a BGR image of shape (H, W, 3)")
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        faces: list[Face] = []
        for d in self._det.process(rgb).detections or []:
            score = float(d.score[0])
            bb = d.location_data.relative_bounding_box
            x0, y0 = max(0, int(bb.xmin * w)), max(0, int(bb.ymin * h))
            x1, y1 = min(w, int((bb.xmin + bb.width) * w)), min(h, int((bb.ymin + bb.height) * h))
            if score >= min_conf and x1 > x0 and y1 > y0:
                faces.append(Face((x0, y0, x1 - x0, y1 - y0), score))
        faces.sort(key=lambda f: -f.score)
        if self.landmarks and faces:
            for lm in self._get_mesh().process(rgb).multi_face_landmarks or []:
                pts = np.array([(p.x * w, p.y * h) for p in lm.landmark], dtype=np.float32)
                (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
                box = (x0, y0, x1 - x0, y1 - y0)
                best = max(faces, key=lambda f: _iou(f.box, box))
                if best.landmarks is None and _iou(best.box, box) > 0.3:
                    best.landmarks = pts
        return faces

    def close(self) -> None:
        self._det.close()
        if self._mesh is not None:
            self._mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def annotate(img: np.ndarray, faces: list[Face], show_landmarks: bool = True) -> np.ndarray:
    """Draw boxes, confidence labels and landmarks onto `img` in place and return it."""
    t = max(2, round(max(img.shape[:2]) / 500))
    scale = 0.3 + 0.35 * t
    for i, f in enumerate(faces, 1):
        x, y, w, h = f.box
        if show_landmarks and f.landmarks is not None:
            for px, py in f.landmarks.astype(int):
                cv2.circle(img, (int(px), int(py)), max(1, t // 2), VIOLET, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + w, y + h), CYAN, t, cv2.LINE_AA)
        label = f"#{i} {f.score * 100:.0f}%"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, t - 1))
        top = max(0, y - th - base - 6)
        cv2.rectangle(img, (x, top), (x + tw + 10, top + th + base + 6), CYAN, -1)
        cv2.putText(img, label, (x + 5, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (28, 16, 4), max(1, t - 1), cv2.LINE_AA)
    return img
