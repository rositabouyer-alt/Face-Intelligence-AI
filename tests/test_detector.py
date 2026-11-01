import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from face_intel.detector import Face, FaceDetector, annotate

FACE = Path(__file__).parent / "data" / "face.jpg"   # optional, git-ignored; supply any portrait photo


def test_blank_image_has_no_faces():
    with FaceDetector(static=True) as d:
        assert d.detect(np.zeros((240, 320, 3), np.uint8)) == []


def test_rejects_bad_input():
    with FaceDetector() as d, pytest.raises(ValueError):
        d.detect(np.zeros((10, 10), np.uint8))


def test_annotate_draws_inside_image():
    img = np.zeros((200, 200, 3), np.uint8)
    out = annotate(img, [Face((50, 50, 80, 80), 0.9)])
    assert out.shape == (200, 200, 3) and out.any()


@pytest.mark.skipif(not FACE.exists(), reason="put a portrait at tests/data/face.jpg to run")
def test_real_face_detected_with_landmarks():
    img = cv2.imread(str(FACE))
    with FaceDetector(1, static=True, landmarks=True) as d:
        faces = d.detect(img, 0.5)
        assert len(faces) >= 1 and faces[0].landmarks.shape == (468, 2)
        assert d.detect(img, 0.999) == [] or d.detect(img, 0.999)[0].score >= 0.999   # threshold filters
