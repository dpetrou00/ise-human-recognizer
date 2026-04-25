from pathlib import Path
from dataclasses import dataclass
import threading

import cv2
import numpy as np
import mediapipe as mp

# MediaPipe HandLandmarker model and options
_BaseOptions = mp.tasks.BaseOptions
_HandLandmarker = mp.tasks.vision.HandLandmarker
_HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
_RunningMode = mp.tasks.vision.RunningMode

# Path to the MediaPipe Hand Landmarker model
MODEL_PATH = Path(__file__).parent.parent / "model" / "mediapipe" / "hand_landmarker.task"


@dataclass
class HandResult:
    """A result used when a hand detection is requested.

    Used to return hand detection results in a structured manner.

    Attributes:
        hand_detected (bool): Whether at least one hand was found in the frame.
        all_landmarks (list[list]): List of 21-landmark lists, one per detected hand (up to 2), or [].
    """
    hand_detected: bool
    all_landmarks: list  # list of per-hand landmark lists, one entry per detected hand


class HandDetector:
    """Detects hands in frames using MediaPipe Hand Landmarker.

    Loads the model once on construction and reuses it across frames.
    Supports use as a context manager.

    Uses LIVE_STREAM mode: detect() submits each frame asynchronously and returns
    the result from the previous frame's inference. The first call always returns
    HandResult(hand_detected=False, all_landmarks=[]).

    Attributes:
        _landmarker: The MediaPipe HandLandmarker instance.
        _latest_result (HandResult): The most recent result delivered by the callback.
        _lock (threading.Lock): Guards _latest_result against concurrent access.
    """
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self._latest_result = HandResult(hand_detected=False, all_landmarks=[])
        self._lock = threading.Lock()

        # Configure the HandLandmarker for live-stream, two-hand detection
        options = _HandLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=str(model_path)),
            running_mode=_RunningMode.LIVE_STREAM,
            num_hands=2,
            result_callback=self._on_result,
        )

        # Load the model
        self._landmarker = _HandLandmarker.create_from_options(options)

    def _on_result(self, result, output_image, timestamp_ms) -> None:
        if not result.hand_landmarks:
            new_result = HandResult(hand_detected=False, all_landmarks=[])
        else:
            new_result = HandResult(hand_detected=True, all_landmarks=result.hand_landmarks)
        with self._lock:
            self._latest_result = new_result

    def detect(self, frame: np.ndarray, timestamp_ms: int) -> HandResult:
        """Submit a BGR frame for async hand detection and return the latest completed result.

        The returned HandResult reflects inference from the previous frame. The first call
        always returns HandResult(hand_detected=False, all_landmarks=[]).

        Args:
            frame (np.ndarray): A BGR image frame from OpenCV.
            timestamp_ms (int): Strictly increasing millisecond timestamp for this frame.

        Returns:
            HandResult: The most recently completed detection result with landmark lists
                for each detected hand (0, 1, or 2 entries).
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
