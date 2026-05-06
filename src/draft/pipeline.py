from dataclasses import dataclass
from typing import Iterator
import time

import numpy as np

from draft.pose import PoseDetector
from draft.distance import DistanceEstimator
from draft.hands import HandDetector
from draft.gesture import GestureClassifier, GESTURE_COMMANDS


@dataclass
class PipelineResult:
    """The result of processing a single frame through the full pipeline.

    Attributes:
        frame (np.ndarray): The original BGR frame.
        person_detected (bool): Whether a person was found by the pose detector.
        distance_m (float | None): Estimated distance to the person in meters, or None if unavailable.
        off_signal (bool): Whether the person is closer than the configured distance threshold.
        hand_detected (bool): Whether at least one hand was found in the frame.
        all_landmarks (list[list]): List of per-hand landmark lists for all detected hands, or [].
        gesture (str): Winning gesture label after priority resolution (thumbs_up, palm_open, or no_signal).
        confidence (float): Classifier confidence score for the winning gesture.
        command (str): Mapped command string (GO, STOP, or empty).
        classifier_name (str): Name of the active classifier backend.
    """
    frame: np.ndarray
    person_detected: bool
    distance_m: float | None
    off_signal: bool
    hand_detected: bool
    all_landmarks: list
    gesture: str
    confidence: float
    command: str
    classifier_name: str


def run_pipeline(
    frames: Iterator[np.ndarray],
    pose_detector: PoseDetector,
    hand_detector: HandDetector,
    estimator: DistanceEstimator,
    classifier: GestureClassifier,
) -> Iterator[PipelineResult]:
    """Process frames through the full detection and classification pipeline.

    Accepts any frame iterator, making it usable with both live camera input
    and pre-recorded frame sequences for testing or evaluation.

    Pipeline order per frame:
        1. Pose detection — determines if a person is present.
        2. Distance estimation — estimates distance and checks the OFF threshold.
        3. Hand detection — only runs if a person was detected; returns up to 2 hands.
        4. Gesture classification — only runs if at least one hand was detected;
           the classifier resolves all detected hands into one winning gesture.

    Args:
        frames (Iterator[np.ndarray]): Source of BGR frames to process.
        pose_detector (PoseDetector): Loaded pose detection model.
        hand_detector (HandDetector): Loaded hand detection model.
        estimator (DistanceEstimator): Calibrated distance estimator.
        classifier (GestureClassifier): Active gesture classifier backend.

    Yields:
        PipelineResult: The full result for each processed frame.
    """
    for frame in frames:

        # Generate a strictly increasing timestamp for async MediaPipe calls
        timestamp_ms = int(time.time() * 1000)

        # Run pose detection to check for a person and get landmarks
        pose_result = pose_detector.detect(frame, timestamp_ms)

        # Estimate distance using pose landmarks
        distance_result = estimator.estimate(pose_result, frame.shape[0])

        # Default to no hands or gesture detected
        all_landmarks = []
        gesture = "no_signal"
        confidence = 1.0

        # Only run hand detection and gesture classification if a person is present
        if pose_result.person_detected:
            hand_result = hand_detector.detect(frame, timestamp_ms)
            all_landmarks = hand_result.all_landmarks

            # Classify across all detected hands if at least one was found
            if hand_result.hand_detected:
                gesture, confidence = classifier.predict(frame, all_landmarks, timestamp_ms)

        yield PipelineResult(
            frame=frame,
            person_detected=pose_result.person_detected,
            distance_m=distance_result.distance_m,
            off_signal=distance_result.off_signal,
            hand_detected=bool(all_landmarks),
            all_landmarks=all_landmarks,
            gesture=gesture,
            confidence=confidence,
            command=GESTURE_COMMANDS.get(gesture, ""),
            classifier_name=classifier.name,
        )
