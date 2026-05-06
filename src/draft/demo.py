"""
ISE Human Recognizer — Demo Application

A graphical interface for real-time human detection, distance estimation,
and gesture classification. Start capture_windows.py on Windows first,
then click Connect & Start.

Controls (in the GUI):
  Connect & Start     — open the camera stream and begin processing
  Calibrate Distance  — capture the current frame to set the focal length reference
  Toggle Model        — switch between MediaPipe and custom gesture classifier
  Threshold spinbox   — adjust the OFF-signal distance threshold live
"""

import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QDoubleSpinBox, QGroupBox, QVBoxLayout, QHBoxLayout, QGridLayout,
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap

sys.path.insert(0, str(Path(__file__).parent))

from draft.capture import receive_frames
from draft.pose import PoseDetector
from draft.distance import DistanceEstimator
from draft.hands import HandDetector
from draft.gesture import build_classifier, GESTURE_COMMANDS, MODEL_PATH
from draft.overlay import draw_landmarks
from draft.logger import append_event

LOG_PATH = Path(__file__).parent.parent / "logs" / "events.csv"

_DARK_STYLE = """
QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", Ubuntu, sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #313244;
    border-radius: 8px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    font-weight: bold;
    color: #89b4fa;
    font-size: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    left: 10px;
}
QPushButton {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 8px 12px;
    color: #cdd6f4;
    min-height: 32px;
}
QPushButton:hover {
    background-color: #45475a;
    border-color: #89b4fa;
}
QPushButton:pressed {
    background-color: #585b70;
}
QPushButton:disabled {
    color: #45475a;
    border-color: #1e1e2e;
    background-color: #181825;
}
QPushButton#start_btn {
    background-color: #1e66f5;
    border-color: #1e66f5;
    color: #ffffff;
    font-weight: bold;
}
QPushButton#start_btn:hover {
    background-color: #4889f5;
    border-color: #4889f5;
}
QPushButton#start_btn:disabled {
    background-color: #313244;
    border-color: #45475a;
    color: #45475a;
    font-weight: normal;
}
QDoubleSpinBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 8px;
    color: #cdd6f4;
    min-height: 28px;
    selection-background-color: #89b4fa;
}
QDoubleSpinBox:focus {
    border-color: #89b4fa;
}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background-color: #45475a;
    border: none;
    width: 16px;
}
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #585b70;
}
"""


class _PipelineThread(QThread):
    """Runs the full detection pipeline in a background thread.

    Emits frame_ready with an annotated frame and status dict each iteration.
    Emits status_msg with informational strings for user-facing feedback.
    """

    frame_ready = Signal(object, object)  # (annotated_frame: np.ndarray, status: dict)
    status_msg = Signal(str)

    def __init__(self, height_m: float, ref_distance_m: float, threshold_m: float) -> None:
        super().__init__()
        self.height_m = height_m
        self.ref_distance_m = ref_distance_m
        self._threshold_m = threshold_m
        self._lock = threading.Lock()
        self._calibrate_event = threading.Event()
        self._toggle_event = threading.Event()

    def set_threshold(self, value: float) -> None:
        with self._lock:
            self._threshold_m = value

    def request_calibrate(self) -> None:
        self._calibrate_event.set()

    def request_toggle(self) -> None:
        self._toggle_event.set()

    def run(self) -> None:
        pose_detector = PoseDetector()
        hand_detector = HandDetector()

        with self._lock:
            threshold_m = self._threshold_m

        estimator = DistanceEstimator(user_height_m=self.height_m, threshold_m=threshold_m)
        custom_available = MODEL_PATH.exists()
        use_custom = False
        classifier = build_classifier(use_custom)
        frame_num = 0

        try:
            for frame in receive_frames():
                if self.isInterruptionRequested():
                    break

                timestamp_ms = int(time.time() * 1000)
                pose_result = pose_detector.detect(frame, timestamp_ms)

                if self._calibrate_event.is_set():
                    self._calibrate_event.clear()
                    ok = estimator.calibrate(pose_result, frame.shape[0], self.ref_distance_m)
                    self.status_msg.emit("Calibrated!" if ok else "No person detected — try again.")

                if self._toggle_event.is_set():
                    self._toggle_event.clear()
                    if not custom_available and not use_custom:
                        self.status_msg.emit("Custom model not found. Train the model first.")
                    else:
                        use_custom = not use_custom
                        classifier.close()
                        classifier = build_classifier(use_custom)
                        self.status_msg.emit(f"Switched to {classifier.name} classifier.")

                with self._lock:
                    estimator.threshold_m = self._threshold_m

                distance_result = estimator.estimate(pose_result, frame.shape[0])
                all_landmarks = []
                gesture = "no_signal"
                confidence = 1.0

                if pose_result.person_detected:
                    hand_result = hand_detector.detect(frame, timestamp_ms)
                    all_landmarks = hand_result.all_landmarks
                    if hand_result.hand_detected:
                        gesture, confidence = classifier.predict(frame, all_landmarks, timestamp_ms)

                command = GESTURE_COMMANDS.get(gesture, "")
                annotated = frame.copy()
                if all_landmarks:
                    draw_landmarks(annotated, all_landmarks)

                self.frame_ready.emit(annotated, {
                    "person_detected": pose_result.person_detected,
                    "distance_m": distance_result.distance_m,
                    "off_signal": distance_result.off_signal,
                    "gesture": gesture,
                    "confidence": confidence,
                    "command": command,
                    "classifier_name": classifier.name,
                    "calibrated": estimator.is_calibrated,
                })

                append_event(LOG_PATH, {
                    "frame_num": frame_num,
                    "person_detected": pose_result.person_detected,
                    "distance_m": round(distance_result.distance_m, 3) if distance_result.distance_m else "",
                    "off_signal": distance_result.off_signal,
                    "hand_detected": bool(all_landmarks),
                    "gesture": gesture,
                    "confidence": round(confidence, 4),
                    "command": command,
                    "classifier": classifier.name,
                })
                frame_num += 1

        except ConnectionRefusedError:
            self.status_msg.emit("Connection refused — is capture_windows.py running on Windows?")
        except ConnectionError as exc:
            self.status_msg.emit(f"Connection lost: {exc}")
        finally:
            pose_detector.close()
            hand_detector.close()
            classifier.close()


class _MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ISE Human Recognizer")
        self._thread: _PipelineThread | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ── Left panel ──────────────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(264)
        left_vbox = QVBoxLayout(left)
        left_vbox.setContentsMargins(0, 0, 0, 0)
        left_vbox.setSpacing(8)

        # Configuration inputs
        config_group = QGroupBox("Configuration")
        config_vbox = QVBoxLayout(config_group)
        config_vbox.setSpacing(4)

        self._height_spin = self._spinbox(0.5, 2.5, 1.75, 0.01, "m")
        self._ref_spin = self._spinbox(0.5, 10.0, 2.0, 0.1, "m")
        self._threshold_spin = self._spinbox(0.1, 10.0, 1.5, 0.1, "m")
        self._threshold_spin.valueChanged.connect(self._on_threshold_changed)

        for label_text, widget in [
            ("User Height", self._height_spin),
            ("Calibration Distance", self._ref_spin),
            ("OFF-Signal Threshold", self._threshold_spin),
        ]:
            lbl = QLabel(label_text)
            lbl.setStyleSheet("color: #a6adc8; font-size: 12px;")
            config_vbox.addWidget(lbl)
            config_vbox.addWidget(widget)

        # Action buttons
        self._start_btn = QPushButton("Connect & Start")
        self._start_btn.setObjectName("start_btn")
        self._start_btn.clicked.connect(self._on_start)

        self._calibrate_btn = QPushButton("Calibrate Distance")
        self._calibrate_btn.setEnabled(False)
        self._calibrate_btn.clicked.connect(self._on_calibrate)

        self._toggle_btn = QPushButton("Toggle Model")
        self._toggle_btn.setEnabled(False)
        self._toggle_btn.clicked.connect(self._on_toggle)

        # Status readouts
        status_group = QGroupBox("Status")
        status_grid = QGridLayout(status_group)
        status_grid.setVerticalSpacing(6)
        status_grid.setColumnStretch(1, 1)

        self._status: dict[str, QLabel] = {}
        rows = [
            ("Machine",    "machine"),
            ("Signal",     "signal"),
            ("Gesture",    "gesture"),
            ("Distance",   "distance"),
            ("Confidence", "confidence"),
            ("Model",      "model"),
        ]
        for i, (display, key) in enumerate(rows):
            key_lbl = QLabel(display + ":")
            key_lbl.setStyleSheet("color: #6c7086;")
            val_lbl = QLabel("—")
            status_grid.addWidget(key_lbl, i, 0)
            status_grid.addWidget(val_lbl, i, 1)
            self._status[key] = val_lbl

        self._msg_lbl = QLabel("")
        self._msg_lbl.setWordWrap(True)
        self._msg_lbl.setStyleSheet("color: #f9e2af; font-size: 12px;")

        left_vbox.addWidget(config_group)
        left_vbox.addSpacing(4)
        left_vbox.addWidget(self._start_btn)
        left_vbox.addWidget(self._calibrate_btn)
        left_vbox.addWidget(self._toggle_btn)
        left_vbox.addSpacing(4)
        left_vbox.addWidget(status_group)
        left_vbox.addWidget(self._msg_lbl)
        left_vbox.addStretch()

        # ── Right panel — video feed ─────────────────────────────────────────
        self._video_lbl = QLabel(
            "No signal\n\nStart capture_windows.py on Windows,\nthen click Connect & Start."
        )
        self._video_lbl.setAlignment(Qt.AlignCenter)
        self._video_lbl.setMinimumSize(640, 480)
        self._video_lbl.setStyleSheet(
            "background-color: #11111b; color: #585b70;"
            "border: 1px solid #313244; border-radius: 6px;"
        )

        root.addWidget(left)
        root.addWidget(self._video_lbl, stretch=1)

    @staticmethod
    def _spinbox(lo: float, hi: float, default: float, step: float, suffix: str) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setValue(default)
        sb.setSingleStep(step)
        sb.setDecimals(2)
        sb.setSuffix(f" {suffix}")
        return sb

    @Slot()
    def _on_start(self) -> None:
        self._thread = _PipelineThread(
            self._height_spin.value(),
            self._ref_spin.value(),
            self._threshold_spin.value(),
        )
        self._thread.frame_ready.connect(self._update_frame)
        self._thread.status_msg.connect(self._msg_lbl.setText)
        self._thread.start()

        self._start_btn.setEnabled(False)
        self._calibrate_btn.setEnabled(True)
        self._toggle_btn.setEnabled(True)
        self._msg_lbl.setText("Connecting to camera stream…")

    @Slot()
    def _on_calibrate(self) -> None:
        if self._thread:
            self._thread.request_calibrate()

    @Slot()
    def _on_toggle(self) -> None:
        if self._thread:
            self._thread.request_toggle()

    @Slot(float)
    def _on_threshold_changed(self, value: float) -> None:
        if self._thread:
            self._thread.set_threshold(value)

    @Slot(object, object)
    def _update_frame(self, frame: np.ndarray, status: dict) -> None:
        # Render video frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self._video_lbl.width(),
            self._video_lbl.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._video_lbl.setPixmap(pixmap)

        # Machine status
        off = status["off_signal"]
        calibrated = status["calibrated"]
        if not calibrated:
            machine_text, machine_color = "—", "#cdd6f4"
        elif off:
            machine_text, machine_color = "OFF", "#f38ba8"
        else:
            machine_text, machine_color = "ON", "#a6e3a1"
        self._status["machine"].setText(machine_text)
        self._status["machine"].setStyleSheet(f"color: {machine_color}; font-weight: bold;")

        # Signal
        cmd = status["command"]
        cmd_color = "#a6e3a1" if cmd == "GO" else ("#f38ba8" if cmd == "STOP" else "#cdd6f4")
        self._status["signal"].setText(cmd if cmd else "—")
        self._status["signal"].setStyleSheet(f"color: {cmd_color};")

        # Gesture, distance, confidence, model
        self._status["gesture"].setText(status["gesture"])
        dist = status["distance_m"]
        self._status["distance"].setText(f"{dist:.2f} m" if dist is not None else "—")
        conf = status["confidence"]
        gesture = status["gesture"]
        self._status["confidence"].setText(f"{conf:.1%}" if gesture != "no_signal" else "—")
        self._status["model"].setText(status["classifier_name"])

    def closeEvent(self, event) -> None:
        if self._thread and self._thread.isRunning():
            self._thread.requestInterruption()
            self._thread.wait(3000)
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(_DARK_STYLE)
    window = _MainWindow()
    window.resize(960, 560)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
