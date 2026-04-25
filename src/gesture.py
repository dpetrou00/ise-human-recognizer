from pathlib import Path
from typing import Protocol
import threading

import cv2
import numpy as np
import mediapipe as mp

# MediaPipe GestureRecognizer model and options
_BaseOptions = mp.tasks.BaseOptions
_GestureRecognizer = mp.tasks.vision.GestureRecognizer
_GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
_RunningMode = mp.tasks.vision.RunningMode

# Path to the custom model and the MediaPipe GestureRecognizer
MODEL_PATH = Path(__file__).parent.parent / "model" / "gesture_classifier.pkl"
MEDIAPIPE_MODEL_PATH = Path(__file__).parent.parent / "model" / "mediapipe" / "gesture_recognizer.task"

# Dict of gesture commands for reference use in the demo
GESTURE_COMMANDS: dict[str, str] = {
    "thumbs_up": "GO",
    "palm_open": "STOP",
    "no_signal": "",
}

# Model label references for use in this file
_MEDIAPIPE_LABEL_MAP: dict[str, str] = {
    "Open_Palm": "palm_open",
    "Thumb_Up": "thumbs_up",
}

# Priority order for resolving conflicting gestures across two hands (lower index = higher priority)
_PRIORITY = ["palm_open", "thumbs_up", "no_signal"]


def _priority_winner(pairs: list[tuple[str, float]]) -> tuple[str, float]:
    """Return the highest-priority gesture from a list of per-hand predictions.

    Priority order is STOP (palm_open) > GO (thumbs_up) > no_signal, reflecting
    that a STOP signal from either hand must always take precedence for safety.
    Confidence is used as a tiebreaker when two hands produce the same gesture.

    Args:
        pairs (list[tuple[str, float]]): List of (gesture_label, confidence) pairs,
            one per detected hand.

    Returns:
        tuple[str, float]: The winning gesture label and its confidence score.
    """

    # If no hands were detected, default to no_signal
    if not pairs:
        return "no_signal", 1.0

    # Sort by priority tier first, then by descending confidence within the same tier
    return min(pairs, key=lambda p: (_PRIORITY.index(p[0]), -p[1]))


class GestureClassifier(Protocol):
    """A unified interface for CustomClassifier or MediaPipeClassifier backends.

    Any class implementing name and predict() satisfies this interface.

    Attributes:
        name (str): An identifier for the type of classifier.
    """
    name: str

    def predict(self, frame: np.ndarray, all_landmarks: list[list], timestamp_ms: int) -> tuple[str, float]: ...
    def close(self) -> None: ...


class CustomClassifier:
    """Contains the model manually trained for classifying hand gestures.

    This class loads the trained model and exposes it for inference.

    Attributes:
        name (str): Set to 'custom.'
    """
    name = "custom"

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        import joblib

        # Raise immediately if the model file has not been trained and saved yet
        if not model_path.exists():
            raise FileNotFoundError(
                f"Custom model not found at {model_path}. Train the model first (Phase 3)."
            )

        # Load the trained scikit-learn MLPClassifier from disk
        self._model = joblib.load(model_path)

    def predict(self, frame: np.ndarray, all_landmarks: list[list], timestamp_ms: int) -> tuple[str, float]:
        """Classify hand gestures using the custom trained MLPClassifier.

        Predicts a gesture for each detected hand and returns the highest-priority result.

        Args:
            frame (np.ndarray): Unused. Included for interface consistency with MediaPipeClassifier.
            all_landmarks (list[list]): List of 21-landmark lists, one per detected hand.
            timestamp_ms (int): Unused. Included for interface consistency with MediaPipeClassifier.

        Returns:
            tuple[str, float]: The winning gesture label and its confidence score.
        """

        # Return no_signal immediately if no hands were detected
        if not all_landmarks:
            return "no_signal", 1.0

        pairs = []

        # Predict a gesture for each detected hand independently
        for hand_landmarks in all_landmarks:

            # Flatten the 21 landmarks (x, y, z each) into a 63-element feature vector
            features = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks]).flatten().reshape(1, -1)

            # Get the predicted class label
            label = self._model.predict(features)[0]

            # Get the highest class probability as the confidence score
            confidence = float(self._model.predict_proba(features)[0].max())

            pairs.append((label, confidence))

        # Return the highest-priority gesture across all detected hands
        return _priority_winner(pairs)

    # Included in CustomClassifier as well to not break the close() call in demo.py
    def close(self) -> None:
        pass


class MediaPipeClassifier:
    """Contains the pre-trained MediaPipe GestureRecognizer model for classifying hand gestures.

    This class loads the model and exposes it for inference.

    Uses LIVE_STREAM mode: predict() submits each frame asynchronously and returns
    the result from the previous frame's inference. The first call always returns
    ("no_signal", 1.0).

    Attributes:
        name (str): Set to 'mediapipe.'
        _latest_result (tuple[str, float]): The most recent result delivered by the callback.
        _lock (threading.Lock): Guards _latest_result against concurrent access.
    """
    name = "mediapipe"

    def __init__(self, model_path: Path = MEDIAPIPE_MODEL_PATH) -> None:
        self._latest_result: tuple[str, float] = ("no_signal", 1.0)
        self._lock = threading.Lock()

        # Configure the GestureRecognizer for live-stream inference
        options = _GestureRecognizerOptions(
            base_options=_BaseOptions(model_asset_path=str(model_path)),
            running_mode=_RunningMode.LIVE_STREAM,
            result_callback=self._on_result,
        )
        self._recognizer = _GestureRecognizer.create_from_options(options)

    def _on_result(self, result, output_image, timestamp_ms) -> None:
        if not result.gestures:
            new_result: tuple[str, float] = ("no_signal", 1.0)
        else:
            pairs = []

            # Collect the top prediction for each detected hand
            for hand_gestures in result.gestures:
                top = hand_gestures[0]
                label = _MEDIAPIPE_LABEL_MAP.get(top.category_name, "no_signal")
                pairs.append((label, float(top.score)))

            new_result = _priority_winner(pairs)

        with self._lock:
            self._latest_result = new_result

    def predict(self, frame: np.ndarray, all_landmarks: list[list], timestamp_ms: int) -> tuple[str, float]:
        """Submit a BGR frame for async gesture recognition and return the latest completed result.

        Runs recognition on the full frame. The returned result reflects inference from the
        previous frame. The first call always returns ("no_signal", 1.0).

        Args:
            frame (np.ndarray): The BGR image frame to run recognition on.
            all_landmarks (list[list]): Unused. Accepted for interface consistency with CustomClassifier.
            timestamp_ms (int): Strictly increasing millisecond timestamp for this frame.

        Returns:
            tuple[str, float]: The winning gesture label and its confidence score.
        """

        # MediaPipe requires RGB input, so convert from BGR
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap the numpy array in a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Submit frame for async inference; result arrives via _on_result callback
        self._recognizer.recognize_async(mp_image, timestamp_ms)

        # Return the latest completed result (from the previous frame's inference)
        with self._lock:
            return self._latest_result

    def close(self) -> None:
        self._recognizer.close()


def build_classifier(use_custom: bool) -> GestureClassifier:
    """Returns the requested type of model based on use_custom flag."""
    if use_custom:
        return CustomClassifier()
    return MediaPipeClassifier()
