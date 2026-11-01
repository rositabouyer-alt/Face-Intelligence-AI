"""Face Intelligence AI - entry point. Opens straight into the dashboard (no login)."""
import sys


def main() -> int:
    try:
        import cv2, mediapipe, numpy  # noqa: F401
        from PySide6.QtWidgets import QApplication, QMessageBox
    except ImportError as exc:
        print(f"Missing dependency: {exc}\nRun: pip install -r requirements.txt", file=sys.stderr)
        return 1
    from face_intel.detector import DetectorError, FaceDetector
    from face_intel.ui import QSS, MainWindow

    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    win = MainWindow()
    try:
        FaceDetector().close()            # fail early and clearly if models cannot load
    except DetectorError as exc:
        win.stack.setEnabled(False)
        win.show()
        QMessageBox.critical(win, "Face Intelligence AI", f"Face engine unavailable.\n\n{exc}")
        return app.exec()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
