import cv2
import numpy as np

from distance import DistanceResult

# Pairs of landmark indices defining the hand skeleton connections.
# Each tuple (start, end) draws a line between those two landmark points.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring finger
    (0, 17), (13, 17), (17, 18), (18, 19), (19, 20),  # pinky and palm
]


def draw_landmarks(frame: np.ndarray, all_landmarks: list[list]) -> None:
    """Draw hand landmark points and skeleton connections onto a frame in-place.

    Draws a skeleton for each detected hand in all_landmarks.

    Args:
        frame (np.ndarray): The BGR image frame to draw on.
        all_landmarks (list[list]): List of per-hand landmark lists, one entry per detected hand.
    """

    # Get the frame dimensions to convert normalized coordinates to pixels
    h, w = frame.shape[:2]

    # Draw landmarks and skeleton for each detected hand
    for hand_landmarks in all_landmarks:

        # Draw a green dot at each landmark position
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # Draw green lines between connected landmark pairs
        for start, end in HAND_CONNECTIONS:
            if start < len(hand_landmarks) and end < len(hand_landmarks):
                x1 = int(hand_landmarks[start].x * w)
                y1 = int(hand_landmarks[start].y * h)
                x2 = int(hand_landmarks[end].x * w)
                y2 = int(hand_landmarks[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)


def draw_distance(frame: np.ndarray, result: DistanceResult) -> None:
    """Draw the current distance estimate onto a frame in-place.

    Shows grey dashes when distance cannot be estimated, green when safe,
    and red when the OFF signal is active.

    Args:
        frame (np.ndarray): The BGR image frame to draw on.
        result (DistanceResult): The distance estimate result to display.
    """

    # Use dashes and grey color when no estimate is available
    if result.distance_m is None:
        text, color = "Distance: --", (200, 200, 200)
    else:
        text = f"Distance: {result.distance_m:.2f}m"

        # Red if the person is too close, green if safe
        color = (0, 0, 255) if result.off_signal else (0, 255, 0)

    cv2.putText(frame, text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_status_panel(
    frame: np.ndarray,
    gesture: str,
    command: str,
    off_signal: bool,
    classifier_name: str,
) -> None:
    """Draw a four-line status panel onto a frame in-place.

    Displays machine status, current signal, detected gesture, and active classifier.

    Args:
        frame (np.ndarray): The BGR image frame to draw on.
        gesture (str): The winning gesture label after priority resolution.
        command (str): The mapped command string (GO, STOP, or empty).
        off_signal (bool): Whether the machine OFF signal is active.
        classifier_name (str): The name of the active classifier backend.
    """

    # Machine status text and color depend on whether the OFF signal is active
    machine_status = "OFF" if off_signal else "ON"
    machine_color = (0, 0, 255) if off_signal else (0, 255, 0)

    # Each line is a tuple of (text, color)
    lines = [
        (f"Machine: {machine_status}", machine_color),
        (f"Signal:  {command if command else '—'}", (255, 255, 255)),
        (f"Gesture: {gesture}", (200, 200, 200)),
        (f"Model:   {classifier_name}", (150, 150, 150)),
    ]

    # Draw each line spaced 30 pixels apart starting at y=40
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (20, 40 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
