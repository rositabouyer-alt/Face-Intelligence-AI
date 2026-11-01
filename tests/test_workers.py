import cv2
import numpy as np

from face_intel.workers import StreamWorker

CFG = {"min_conf": 0.5, "landmarks": False, "model": 0, "step": 1}


def _collect(source, cfg=CFG):
    w, out = StreamWorker(source, dict(cfg)), {"frames": 0, "failed": None, "done": None}
    w.frame.connect(lambda *_: out.__setitem__("frames", out["frames"] + 1))
    w.failed.connect(lambda m: out.__setitem__("failed", m))
    w.done.connect(lambda s: out.__setitem__("done", s))
    w.run()          # synchronous: same code path as the thread body
    return out


def test_video_streams_and_respects_step(tmp_path):
    p = str(tmp_path / "v.avi")
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    for _ in range(12):
        vw.write(np.zeros((120, 160, 3), np.uint8))
    vw.release()
    assert _collect(p)["done"]["frames"] == 12
    assert _collect(p, {**CFG, "step": 3})["done"]["frames"] == 4


def test_invalid_video_reports_error(tmp_path):
    bad = tmp_path / "x.mp4"
    bad.write_bytes(b"not a video")
    out = _collect(str(bad))
    assert out["failed"] and out["done"] is None
