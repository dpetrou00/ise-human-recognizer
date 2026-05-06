from pathlib import Path
from dataclasses import dataclass
import threading

import cv2
import numpy as np
import mediapipe as mp

# MediaPipe PoseLandmarker model and options
_BaseOptions = mp.tasks.BaseOptions
_PoseLandmarker = mp.tasks.vision.PoseLandmarker
_PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
_RunningMode = mp.tasks.vision.RunningMode

# Path to the MediaPipe Pose Landmarker Lite model
MODEL_PATH = Path(__file__).parent.parent / "model" / "mediapipe" / "pose_landmarker_lite.task"


@dataclass
class PoseResult:
    """A result used when a pose detection is requested.

    Used to return pose detection results in a structured manner.

    Attributes:
        person_detected (bool): Whether at least one person was found in the frame.
        landmarks (list): Normalized MediaPipe pose landmarks for the first person, or [] if none.
    """
    person_detected: bool
    landmarks: list  # normalized landmarks for the first detected person, or []


class PoseDetector:
    """Detects people in frames using MediaPipe Pose Landmarker Lite.

    Loads the model once on construction and reuses it across frames.
    Supports use as a context manager.

    Uses LIVE_STREAM mode: detect() submits each frame asynchronously and returns
    the result from the previous frame's inference. The first call always returns
    PoseResult(person_detected=False, landmarks=[]).

    Attributes:
        _landmarker: The MediaPipe PoseLandmarker instance.
        _latest_result (PoseResult): The most recent result delivered by the callback.
        _lock (threading.Lock): Guards _latest_result against concurrent access.
    """
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self._latest_result = PoseResult(person_detected=False, landmarks=[])
        self._lock = threading.Lock()

        # Configure the PoseLandmarker for live-stream, single-person detection
        options = _PoseLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=str(model_path)),
            running_mode=_RunningMode.LIVE_STREAM,
            num_poses=1,
            result_callback=self._on_result,
        )

        # Load the model
        self._landmarker = _PoseLandmarker.create_from_options(options)

    def _on_result(self, result, output_image, timestamp_ms) -> None:
        if not result.pose_landmarks:
            new_result = PoseResult(person_detected=False, landmarks=[])
        else:
            new_result = PoseResult(person_detected=True, landmarks=result.pose_landmarks[0])
        with self._lock:
            self._latest_result = new_result

    def detect(self, frame: np.ndarray, timestamp_ms: int) -> PoseResult:
        """Submit a BGR frame for async pose detection and return the latest completed result.

        The returned PoseResult reflects inference from the previous frame. The first call
        always returns PoseResult(person_detected=False, landmarks=[]).

        Args:
            frame (np.ndarray): A BGR image frame from OpenCV.
            timestamp_ms (int): Strictly increasing millisecond timestamp for this frame.

        Returns:
            PoseResult: The most recently completed detection result.
        """

        # MediaPipe requires RGB input, so convert from BGR
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap the numpy array in a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Submit frame for async inference; result arrives via _on_result callback
        self._landmarker.detect_async(mp_image, timestamp_ms)

        # Return the latest completed result (from the previous frame's inference)
        with self._lock:
            return self._latest_result

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
