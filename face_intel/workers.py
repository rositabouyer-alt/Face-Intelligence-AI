"""Qt worker threads: camera/video streaming and benchmarking."""
from __future__ import annotations

import sys
import time
from collections import deque

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from .detector import FaceDetector, annotate
from .metrics import latency_summary

EMIT_INTERVAL = 1 / 30   # cap GUI updates at ~30/s so the event queue never floods


def to_qimage(bgr) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def open_capture(source):
    if isinstance(source, int) and sys.platform == "win32":
        return cv2.VideoCapture(source, cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)


class StreamWorker(QThread):
    """Reads a camera (int) or video file (str) frame by frame; never buffers the video."""
    frame = Signal(QImage, dict)
    failed = Signal(str)
    done = Signal(dict)

    def __init__(self, source, cfg: dict, parent=None):
        super().__init__(parent)
        self.source, self.cfg = source, cfg

    def run(self):
        cam = isinstance(self.source, int)
        cap = det = None
        try:
            det = FaceDetector(self.cfg["model"], landmarks=self.cfg["landmarks"])
            cap = open_capture(self.source)
            if not cap.isOpened():
                self.failed.emit(
                    f"Cannot open camera {self.source}. It may be missing, in use by another app, "
                    "or blocked in Windows Settings > Privacy > Camera." if cam
                    else f"Cannot open video file: {self.source}")
                return
            if cam:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            total = 0 if cam else int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            step = 1 if cam else max(1, int(self.cfg.get("step", 1)))
            stamps: deque = deque(maxlen=30)
            n = faces_sum = idx = 0
            lat_sum, last_emit, start, completed = 0.0, 0.0, time.perf_counter(), False
            while not self.isInterruptionRequested():
                ok, bgr = cap.read()
                if not ok:
                    if n == 0:
                        self.failed.emit("No frames could be read from the camera." if cam
                                         else "The file contains no readable video frames.")
                        return
                    if cam:
                        self.failed.emit("Camera stream was lost.")
                        return
                    completed = True
                    break
                idx += 1
                det.landmarks = self.cfg["landmarks"]
                t0 = time.perf_counter()
                faces = det.detect(bgr, self.cfg["min_conf"])
                lat = (time.perf_counter() - t0) * 1000
                n, faces_sum, lat_sum = n + 1, faces_sum + len(faces), lat_sum + lat
                now = time.perf_counter()
                stamps.append(now)
                if now - last_emit >= EMIT_INTERVAL:
                    last_emit = now
                    fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
                    scores = [f.score for f in faces]
                    st = {"faces": len(faces), "fps": fps, "latency": lat, "frame": idx, "total": total,
                          "conf": sum(scores) / len(scores) if scores else None}
                    self.frame.emit(to_qimage(annotate(bgr, faces, self.cfg["landmarks"])), st)
                for _ in range(step - 1):          # cheap skip: grab() does not fully decode
                    if not cap.grab():
                        break
                    idx += 1
            elapsed = time.perf_counter() - start
            self.done.emit({"frames": n, "faces": faces_sum, "mean_latency": lat_sum / n if n else 0.0,
                            "elapsed": elapsed, "fps": n / elapsed if elapsed else 0.0,
                            "completed": completed, "total": total})
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if cap is not None:
                cap.release()
            if det is not None:
                det.close()


class BenchWorker(QThread):
    """Times detection on a real image across model configurations."""
    progress = Signal(int)
    result = Signal(list)
    failed = Signal(str)
    CONFIGS = [("Short-range detection", 0, False), ("Full-range detection", 1, False),
               ("Full-range + landmarks", 1, True)]
    WARMUP = 5

    def __init__(self, img, iterations: int = 50, parent=None):
        super().__init__(parent)
        self.img, self.iterations = img, iterations

    def run(self):
        try:
            total, done, rows = len(self.CONFIGS) * (self.WARMUP + self.iterations), 0, []
            for name, model, lm in self.CONFIGS:
                times = []
                with FaceDetector(model, landmarks=lm) as det:
                    for i in range(self.WARMUP + self.iterations):
                        if self.isInterruptionRequested():
                            return
                        t0 = time.perf_counter()
                        det.detect(self.img, 0.5)
                        if i >= self.WARMUP:
                            times.append((time.perf_counter() - t0) * 1000)
                        done += 1
                        self.progress.emit(int(100 * done / total))
                rows.append({"name": name, **latency_summary(times)})
            self.result.emit(rows)
        except Exception as exc:
            self.failed.emit(str(exc))
