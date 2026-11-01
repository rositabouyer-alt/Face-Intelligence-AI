# Face Intelligence AI

Local desktop app for face detection and analysis. Python 3.12 · OpenCV · MediaPipe · PySide6.
No login, no cloud, no API keys, no Docker. Nothing is saved unless you press **Export**.

## Install and run (Windows 11, PowerShell, VS Code terminal)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

Tests: `pip install -r requirements-dev.txt; python -m pytest -q`

## Features

| Page | What it does |
|---|---|
| Live Detection | Webcam, multiple faces, boxes, confidence, face count, FPS, latency (Qt worker thread) |
| Image Analysis | Open JPG/PNG/BMP/WEBP, detect, optional annotated export (PNG/JPG) |
| Video Analysis | Streams frame by frame (never loads the whole video); "every Nth frame" option; run summary |
| Landmarks | 468-point MediaPipe FaceMesh overlay (toggle) |
| Performance Analytics | Benchmark (mean/median/P95/min/max ms, FPS) on your loaded image; precision/recall/F1 only when you load ground truth |

The confidence slider, landmark toggle and model choice (full-range / short-range) are shared across pages.

### Ground-truth format
A JSON file of pixel boxes for the image currently loaded in Image Analysis:
```json
[[209, 204, 179, 179], [400, 80, 120, 120]]
```
`[x, y, w, h]`; matching is greedy one-to-one at IoU ≥ 0.5.

## Limitations (honest notes)
- **CPU only.** MediaPipe's Python package on Windows has no GPU delegate, so GPU acceleration is not offered.
- Landmarks come from MediaPipe FaceMesh (`mediapipe==0.10.14` legacy `solutions` API, models bundled in the wheel; no downloads).
- Precision/recall/F1 reflect a single image with your annotations; this is not a dataset benchmark.
- Camera index and DirectShow behaviour depend on your driver; try indexes 0, 1, 2 if the first fails.
- Video decoding uses OpenCV's bundled codecs; unusual codecs may fail to open (an error is shown).
- Do not install `opencv-python` alongside `opencv-contrib-python` (mediapipe requires the latter).

## Layout
`main.py` entry · `face_intel/detector.py` detection + drawing · `metrics.py` IoU/PRF1/latency · `workers.py` Qt threads · `ui.py` dashboard · `tests/`
