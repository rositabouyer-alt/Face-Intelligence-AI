"""Main dashboard window (PySide6)."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFrame,
                               QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox,
                               QProgressBar, QPushButton, QSlider, QSpinBox, QStackedWidget,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .detector import DetectorError, FaceDetector, annotate
from .metrics import evaluate, load_ground_truth
from .workers import BenchWorker, StreamWorker, to_qimage

QSS = """
* { font-family: 'Segoe UI', 'Inter', sans-serif; color: #E5ECF6; font-size: 13px; }
QMainWindow, QWidget#root { background: #0B1220; }
QFrame#sidebar { background: #0E1628; border-right: 1px solid #1B2740; }
QLabel#brand { font-size: 19px; font-weight: 700; }
QLabel#muted { color: #7C8DB0; }
QLabel#title { font-size: 22px; font-weight: 700; }
QPushButton#nav { text-align: left; padding: 12px 16px; border: none; border-radius: 10px;
  background: transparent; color: #9FB0CF; font-size: 14px; }
QPushButton#nav:hover { background: #131E36; }
QPushButton#nav:checked { color: #FFFFFF; border-left: 3px solid #22D3EE;
  background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 rgba(34,211,238,60), stop:1 rgba(139,92,246,60)); }
QFrame#card, QFrame#panel { background: #111A2E; border: 1px solid #1E2A44; border-radius: 14px; }
QFrame#card QLabel, QFrame#panel QLabel { border: none; background: transparent; }
QLabel#cardTitle { color: #7C8DB0; font-size: 11px; font-weight: 600; }
QLabel#cardValue { font-size: 26px; font-weight: 700; }
QLabel#preview { background: #070C18; border: 1px solid #1E2A44; border-radius: 14px; color: #52638A; font-size: 14px; }
QPushButton { background: #16223B; border: 1px solid #243457; border-radius: 8px; padding: 8px 16px; }
QPushButton:hover { border-color: #22D3EE; }
QPushButton:disabled { color: #4A5A7C; border-color: #1B2740; background: #101930; }
QPushButton#primary { border: none; color: #04101C; font-weight: 700;
  background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #22D3EE, stop:1 #8B5CF6); }
QPushButton#primary:disabled { background: #1B2740; color: #4A5A7C; }
QSlider::groove:horizontal { height: 4px; background: #1E2A44; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #22D3EE; border-radius: 2px; }
QSlider::handle:horizontal { background: #8B5CF6; width: 14px; margin: -6px 0; border-radius: 7px; }
QSpinBox, QComboBox { background: #0E1628; border: 1px solid #243457; border-radius: 8px; padding: 5px 8px; }
QComboBox QAbstractItemView { background: #0E1628; selection-background-color: #243457; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px; border: 1px solid #243457; background: #0E1628; }
QCheckBox::indicator:checked { background: #22D3EE; }
QProgressBar { background: #0E1628; border: 1px solid #1E2A44; border-radius: 6px; max-height: 10px; text-align: center; color: transparent; }
QProgressBar::chunk { border-radius: 5px; background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #22D3EE, stop:1 #8B5CF6); }
QTableWidget { background: #0E1628; border: 1px solid #1E2A44; border-radius: 10px; gridline-color: #1B2740; }
QHeaderView::section { background: #111A2E; color: #7C8DB0; border: none; padding: 6px; font-weight: 600; }
QStatusBar { background: #0E1628; color: #7C8DB0; }
"""

PAGES = [("◉  Live Detection", "Live Detection"), ("▣  Image Analysis", "Image Analysis"),
         ("▶  Video Analysis", "Video Analysis"), ("◈  Performance Analytics", "Performance Analytics")]


class Card(QFrame):
    def __init__(self, title: str, accent: str):
        super().__init__()
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(2)
        t = QLabel(title.upper())
        t.setObjectName("cardTitle")
        self.value = QLabel("—")
        self.value.setObjectName("cardValue")
        self.value.setStyleSheet(f"color: {accent};")
        lay.addWidget(t)
        lay.addWidget(self.value)

    def set(self, text: str):
        self.value.setText(text)


class Preview(QLabel):
    def __init__(self, hint: str):
        super().__init__(hint)
        self.hint, self._img = hint, None
        self.setObjectName("preview")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 220)
        self.setSizePolicy(self.sizePolicy().Policy.Ignored, self.sizePolicy().Policy.Ignored)

    def show_image(self, img: QImage):
        self._img = img
        self._render()

    def reset(self):
        self._img = None
        self.clear()
        self.setText(self.hint)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render()

    def _render(self):
        if self._img is not None:
            self.setPixmap(QPixmap.fromImage(self._img).scaled(
                self.size() - QSize(4, 4), Qt.KeepAspectRatio, Qt.SmoothTransformation))


def _btn(text, primary=False, enabled=True):
    b = QPushButton(text)
    b.setEnabled(enabled)
    b.setCursor(Qt.PointingHandCursor)
    if primary:
        b.setObjectName("primary")
    return b


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Intelligence AI")
        self.resize(1280, 820)
        self.setMinimumSize(980, 660)
        self.cfg = {"min_conf": 0.5, "landmarks": True, "model": 1, "step": 1}
        self.worker = self.bench = None
        self.active_preview = None
        self.img_bgr = self.img_out = None
        self.img_faces: list = []
        self.video_path: str | None = None
        self.redetect = QTimer(singleShot=True, interval=250)
        self.redetect.timeout.connect(self._analyze_image)

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._sidebar())
        outer.addLayout(self._content(), 1)
        self.statusBar().showMessage("Ready · all processing is local (CPU)")

    # ---------- layout ----------
    def _sidebar(self):
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(240)
        lay = QVBoxLayout(side)
        lay.setContentsMargins(16, 24, 16, 20)
        lay.setSpacing(6)
        brand = QLabel("Face Intelligence <span style='color:#22D3EE'>AI</span>")
        brand.setObjectName("brand")
        brand.setTextFormat(Qt.RichText)
        sub = QLabel("Computer vision studio")
        sub.setObjectName("muted")
        lay.addWidget(brand)
        lay.addWidget(sub)
        lay.addSpacing(22)
        self.nav = QButtonGroup(self)
        for i, (label, _) in enumerate(PAGES):
            b = QPushButton(label)
            b.setObjectName("nav")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            self.nav.addButton(b, i)
            lay.addWidget(b)
        self.nav.button(0).setChecked(True)
        self.nav.idClicked.connect(self._goto)
        lay.addStretch(1)
        foot = QLabel("MediaPipe · OpenCV · PySide6\n100% local · CPU inference")
        foot.setObjectName("muted")
        lay.addWidget(foot)
        return side

    def _content(self):
        col = QVBoxLayout()
        col.setContentsMargins(24, 20, 24, 12)
        col.setSpacing(14)
        self.title = QLabel(PAGES[0][1])
        self.title.setObjectName("title")
        col.addWidget(self.title)

        cards = QHBoxLayout()
        cards.setSpacing(14)
        self.c_faces, self.c_fps = Card("Faces", "#22D3EE"), Card("FPS", "#8B5CF6")
        self.c_lat, self.c_conf = Card("Latency", "#22D3EE"), Card("Avg confidence", "#8B5CF6")
        for c in (self.c_faces, self.c_fps, self.c_lat, self.c_conf):
            cards.addWidget(c)
        col.addLayout(cards)

        col.addWidget(self._settings_bar())
        self.stack = QStackedWidget()
        self.stack.addWidget(self._live_page())
        self.stack.addWidget(self._image_page())
        self.stack.addWidget(self._video_page())
        self.stack.addWidget(self._analytics_page())
        col.addWidget(self.stack, 1)
        return col

    def _settings_bar(self):
        f = QFrame()
        f.setObjectName("panel")
        r = QHBoxLayout(f)
        r.setContentsMargins(16, 10, 16, 10)
        r.setSpacing(12)
        r.addWidget(QLabel("Min confidence"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(10, 95)
        self.slider.setValue(50)
        self.slider.setFixedWidth(180)
        self.conf_lbl = QLabel("50%")
        self.slider.valueChanged.connect(self._settings_changed)
        r.addWidget(self.slider)
        r.addWidget(self.conf_lbl)
        r.addSpacing(16)
        self.chk_lm = QCheckBox("Facial landmarks")
        self.chk_lm.setChecked(True)
        self.chk_lm.toggled.connect(self._settings_changed)
        r.addWidget(self.chk_lm)
        r.addSpacing(16)
        r.addWidget(QLabel("Model"))
        self.combo = QComboBox()
        self.combo.addItem("Full-range (up to ~5 m)", 1)
        self.combo.addItem("Short-range (≤2 m, fastest)", 0)
        self.combo.setToolTip("Applies to the next camera/video session and to image analysis.")
        self.combo.currentIndexChanged.connect(self._settings_changed)
        r.addWidget(self.combo)
        r.addStretch(1)
        return f

    def _row(self, *widgets):
        r = QHBoxLayout()
        r.setSpacing(10)
        for w in widgets:
            r.addWidget(w) if w is not None else r.addStretch(1)
        return r

    def _live_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self.cam_idx = QSpinBox()
        self.cam_idx.setRange(0, 9)
        self.btn_cam_start, self.btn_cam_stop = _btn("Start camera", True), _btn("Stop", enabled=False)
        self.btn_cam_start.clicked.connect(lambda: self._stream(self.cam_idx.value(), self.live_view))
        self.btn_cam_stop.clicked.connect(self._stop_worker)
        lay.addLayout(self._row(QLabel("Camera index"), self.cam_idx, self.btn_cam_start, self.btn_cam_stop, None))
        self.live_view = Preview("Press “Start camera” to begin live detection")
        lay.addWidget(self.live_view, 1)
        return w

    def _image_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self.btn_img_open = _btn("Open image…", True)
        self.btn_img_export = _btn("Export annotated image…", enabled=False)
        self.btn_img_open.clicked.connect(self._open_image)
        self.btn_img_export.clicked.connect(self._export_image)
        self.img_info = QLabel("")
        self.img_info.setObjectName("muted")
        lay.addLayout(self._row(self.btn_img_open, self.btn_img_export, self.img_info, None))
        self.img_view = Preview("Open an image (JPG, PNG, BMP, WEBP) to analyze it")
        lay.addWidget(self.img_view, 1)
        return w

    def _video_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self.btn_vid_open = _btn("Open video…")
        self.vid_name = QLabel("No file selected")
        self.vid_name.setObjectName("muted")
        self.step = QSpinBox()
        self.step.setRange(1, 30)
        self.step.setToolTip("1 = analyze every frame; N = analyze every Nth frame (faster)")
        self.btn_vid_start, self.btn_vid_stop = _btn("Analyze", True, False), _btn("Stop", enabled=False)
        self.btn_vid_open.clicked.connect(self._open_video)
        self.btn_vid_start.clicked.connect(lambda: self._stream(self.video_path, self.vid_view))
        self.btn_vid_stop.clicked.connect(self._stop_worker)
        lay.addLayout(self._row(self.btn_vid_open, self.vid_name, None, QLabel("Every Nth frame"),
                                self.step, self.btn_vid_start, self.btn_vid_stop))
        self.progress = QProgressBar()
        lay.addWidget(self.progress)
        self.vid_view = Preview("Open a local video file, then press “Analyze”")
        lay.addWidget(self.vid_view, 1)
        self.vid_summary = QLabel("")
        self.vid_summary.setObjectName("muted")
        lay.addWidget(self.vid_summary)
        return w

    def _analytics_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self.btn_bench = _btn("Run benchmark on loaded image", True)
        self.btn_bench.clicked.connect(self._run_bench)
        self.bench_bar = QProgressBar()
        lay.addLayout(self._row(self.btn_bench, self.bench_bar))
        note = QLabel("Benchmarks time detection on the image opened in Image Analysis "
                      "(5 warm-up + 50 timed runs per configuration, this machine's CPU).")
        note.setObjectName("muted")
        lay.addWidget(note)
        cols = ["Configuration", "Mean ms", "Median ms", "P95 ms", "Min ms", "Max ms", "FPS"]
        self.table = QTableWidget(0, len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self.table, 1)
        panel = QFrame()
        panel.setObjectName("panel")
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(16, 12, 16, 12)
        head = QLabel("Accuracy vs. ground truth")
        head.setStyleSheet("font-weight:700; font-size:15px;")
        self.btn_gt = _btn("Load ground-truth JSON…")
        self.btn_gt.clicked.connect(self._load_gt)
        self.gt_result = QLabel("Precision, recall and F1 are computed only when you supply real annotations "
                                "([x, y, w, h] boxes) for the image loaded in Image Analysis.")
        self.gt_result.setObjectName("muted")
        self.gt_result.setWordWrap(True)
        pl.addWidget(head)
        pl.addWidget(self.gt_result)
        pl.addLayout(self._row(self.btn_gt, None))
        lay.addWidget(panel)
        return w

    # ---------- shared state ----------
    def _goto(self, i: int):
        self._stop_worker()
        self.stack.setCurrentIndex(i)
        self.title.setText(PAGES[i][1])

    def _settings_changed(self, *_):
        self.cfg.update(min_conf=self.slider.value() / 100, landmarks=self.chk_lm.isChecked(),
                        model=self.combo.currentData())
        self.conf_lbl.setText(f"{self.slider.value()}%")
        if self.img_bgr is not None:
            self.redetect.start()

    def _cards(self, faces, fps, lat, conf):
        self.c_faces.set(str(faces))
        self.c_fps.set("—" if fps is None else f"{fps:.1f}")
        self.c_lat.set("—" if lat is None else f"{lat:.1f} ms")
        self.c_conf.set("—" if conf is None else f"{conf * 100:.0f}%")

    def _error(self, text: str):
        self.statusBar().showMessage(text)
        QMessageBox.warning(self, "Face Intelligence AI", text)

    # ---------- streaming (camera + video) ----------
    def _stream(self, source, view: Preview):
        if source is None:
            return
        self._stop_worker()
        self.cfg["step"] = self.step.value()
        self.active_preview = view
        self.progress.setValue(0)
        self.vid_summary.setText("")
        w = StreamWorker(source, self.cfg, self)
        w.frame.connect(self._on_frame)
        w.failed.connect(self._on_failed)
        w.done.connect(self._on_done)
        w.finished.connect(self._on_finished)
        self.worker = w
        self._running(True)
        self.statusBar().showMessage("Running…")
        w.start()

    def _running(self, on: bool):
        self.btn_cam_start.setEnabled(not on)
        self.btn_cam_stop.setEnabled(on)
        self.btn_vid_start.setEnabled(not on and self.video_path is not None)
        self.btn_vid_stop.setEnabled(on)
        self.btn_vid_open.setEnabled(not on)

    def _stop_worker(self):
        w = self.worker
        if w is not None and w.isRunning():
            w.requestInterruption()
            w.wait(5000)
        QApplication.processEvents()   # flush queued signals so the UI settles

    def _on_frame(self, img: QImage, st: dict):
        if self.sender() is not self.worker:
            return
        self.active_preview.show_image(img)
        self._cards(st["faces"], st["fps"], st["latency"], st["conf"])
        if st["total"]:
            self.progress.setValue(int(100 * st["frame"] / st["total"]))

    def _on_failed(self, msg: str):
        self._error(msg)

    def _on_done(self, s: dict):
        if self.active_preview is self.vid_view:
            if s["completed"]:
                self.progress.setValue(100)
            avg = s["faces"] / s["frames"] if s["frames"] else 0
            self.vid_summary.setText(
                f"{'Completed' if s['completed'] else 'Stopped'}: {s['frames']} frames analyzed in "
                f"{s['elapsed']:.1f} s ({s['fps']:.1f} fps) · mean latency {s['mean_latency']:.1f} ms · "
                f"{s['faces']} detections ({avg:.2f} per analyzed frame)")
        self.statusBar().showMessage("Finished")

    def _on_finished(self):
        if self.sender() is self.worker:
            self._running(False)
            self.worker = None
        self.sender().deleteLater()

    # ---------- image ----------
    def _pick(self, title, filt, save=False):
        fn = QFileDialog.getSaveFileName if save else QFileDialog.getOpenFileName
        return fn(self, title, "", filt)[0]

    def _open_image(self):
        path = self._pick("Open image", "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)")
        if not path:
            return
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)   # unicode-safe
        except OSError as exc:
            return self._error(f"Cannot read file: {exc}")
        if img is None:
            return self._error("This file is not a valid or supported image.")
        self.img_bgr = img
        self.img_info.setText(f"{Path(path).name} · {img.shape[1]}×{img.shape[0]}")
        self._analyze_image()

    def _analyze_image(self):
        if self.img_bgr is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            with FaceDetector(self.cfg["model"], static=True, landmarks=self.cfg["landmarks"]) as det:
                t0 = time.perf_counter()
                faces = det.detect(self.img_bgr, self.cfg["min_conf"])
                ms = (time.perf_counter() - t0) * 1000
        except (DetectorError, ValueError) as exc:
            return self._error(str(exc))
        finally:
            QApplication.restoreOverrideCursor()
        self.img_faces = faces
        self.img_out = annotate(self.img_bgr.copy(), faces, self.cfg["landmarks"])
        self.img_view.show_image(to_qimage(self.img_out))
        self.btn_img_export.setEnabled(True)
        conf = sum(f.score for f in faces) / len(faces) if faces else None
        self._cards(len(faces), None, ms, conf)
        self.statusBar().showMessage(f"{len(faces)} face(s) detected in {ms:.1f} ms (single-shot, includes warm-up)")

    def _export_image(self):
        if self.img_out is None:
            return
        path = self._pick("Export annotated image", "PNG (*.png);;JPEG (*.jpg)", save=True)
        if not path:
            return
        ext = Path(path).suffix.lower() or ".png"
        ok, buf = cv2.imencode(ext, self.img_out)
        if not ok:
            return self._error(f"Unsupported export format “{ext}”. Use .png or .jpg.")
        try:
            buf.tofile(path if Path(path).suffix else path + ext)
        except OSError as exc:
            return self._error(f"Could not write file: {exc}")
        self.statusBar().showMessage(f"Exported {Path(path).name}")

    # ---------- video ----------
    def _open_video(self):
        path = self._pick("Open video", "Video (*.mp4 *.avi *.mov *.mkv *.wmv *.webm)")
        if path:
            self.video_path = path
            self.vid_name.setText(Path(path).name)
            self.btn_vid_start.setEnabled(True)
            self.vid_view.reset()
            self.progress.setValue(0)

    # ---------- analytics ----------
    def _run_bench(self):
        if self.img_bgr is None:
            return QMessageBox.information(self, "Face Intelligence AI",
                                           "Open an image in Image Analysis first; the benchmark runs on it.")
        self.btn_bench.setEnabled(False)
        self.bench_bar.setValue(0)
        b = BenchWorker(self.img_bgr, parent=self)
        b.progress.connect(self.bench_bar.setValue)
        b.result.connect(self._bench_done)
        b.failed.connect(self._error)
        b.finished.connect(self._bench_finished)
        self.bench = b
        b.start()

    def _bench_finished(self):
        self.btn_bench.setEnabled(True)
        self.bench = None
        self.sender().deleteLater()

    def _bench_done(self, rows: list):
        self.table.setRowCount(len(rows))
        for r, d in enumerate(rows):
            vals = [d["name"], d["mean"], d["median"], d["p95"], d["min"], d["max"], d["fps"]]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v if c == 0 else f"{v:.1f}")
                item.setTextAlignment(Qt.AlignVCenter | (Qt.AlignLeft if c == 0 else Qt.AlignRight))
                self.table.setItem(r, c, item)
        self.statusBar().showMessage("Benchmark complete")

    def _load_gt(self):
        if self.img_bgr is None:
            return QMessageBox.information(self, "Face Intelligence AI",
                                           "Open and analyze an image first; ground truth refers to it.")
        path = self._pick("Ground-truth boxes", "JSON (*.json)")
        if not path:
            return
        try:
            gt = load_ground_truth(path)
        except ValueError as exc:
            return self._error(str(exc))
        m = evaluate([f.box for f in self.img_faces], gt, 0.5)
        self.gt_result.setText(
            f"TP {m['tp']} · FP {m['fp']} · FN {m['fn']}     Precision {m['precision']:.3f} · "
            f"Recall {m['recall']:.3f} · F1 {m['f1']:.3f}\n"
            f"IoU ≥ 0.5, min confidence {self.slider.value()}%, {len(gt)} annotated face(s), single image.")

    def closeEvent(self, e):
        self._stop_worker()
        if self.bench is not None and self.bench.isRunning():
            self.bench.requestInterruption()
            self.bench.wait(5000)
        super().closeEvent(e)
