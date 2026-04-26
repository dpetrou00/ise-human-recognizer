"""
ISE Human Recognizer — Landmark Data Collection

Connects to the Windows camera stream and saves hand landmark rows to a CSV
for training the custom gesture classifier.

Controls:
  1  — save current hand(s) as no_signal
  2  — save current hand(s) as thumbs_up
  3  — save current hand(s) as palm_open
  q  — quit
"""

import csv
import sys
import time
from io import TextIOWrapper
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))

from capture import receive_frames
from hands import HandDetector
from overlay import draw_landmarks

DATA_PATH = Path(__file__).parent.parent / "model" / "data" / "landmarks.csv"

# Maps keypress ASCII codes to their corresponding class labels
_LABEL_KEYS = {
    ord("1"): "no_signal",
    ord("2"): "thumbs_up",
    ord("3"): "palm_open",
}

# CSV column order: label followed by x/y/z for each of the 21 landmarks
_FIELDNAMES = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


def _open_csv() -> tuple[csv.DictWriter, TextIOWrapper]:
    """Open the landmark CSV for appending, writing a header only on first creation.

    Creates the parent directory if it does not already exist.

    Returns:
        tuple[csv.DictWriter, TextIOWrapper]: A configured DictWriter and the
            underlying file handle (needed for flushing and closing).
    """
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Only write the header row when creating a new file; appending skips it
    write_header = not DATA_PATH.exists()
    f = open(DATA_PATH, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
    if write_header:
        writer.writeheader()
    return writer, f


def _save_hands(writer: csv.DictWriter, label: str, all_landmarks: list) -> int:
    """Write one CSV row per detected hand with the given class label.

    Each row contains the label and the (x, y, z) coordinates of all 21
    MediaPipe landmarks, rounded to six decimal places.

    Args:
        writer (csv.DictWriter): The open CSV writer to append rows to.
        label (str): The class label to assign to each hand in this frame.
        all_landmarks (list): List of 21-landmark lists, one per detected hand.

    Returns:
        int: Number of rows written (equal to the number of detected hands).
    """
    for hand_landmarks in all_landmarks:
        row: dict = {"label": label}
        for i, lm in enumerate(hand_landmarks):
            row[f"x{i}"] = round(lm.x, 6)
            row[f"y{i}"] = round(lm.y, 6)
            row[f"z{i}"] = round(lm.z, 6)
        writer.writerow(row)
    return len(all_landmarks)


def _draw_hud(frame, counts: dict, hand_count: int) -> None:
    """Overlay collection status information onto the video frame in-place.

    Draws running sample counts per class, key-binding hints, and a hand
    detection indicator that turns green when at least one hand is visible.

    Args:
        frame: The BGR image frame to annotate.
        counts (dict): Running sample counts keyed by class label.
        hand_count (int): Number of hands currently detected in the frame.
    """
    # Top bar: running totals for each class
    labels = ["no_signal", "thumbs_up", "palm_open"]
    count_text = "  |  ".join(f"{k}: {counts[k]}" for k in labels)
    cv2.putText(frame, count_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Key-binding legend below the count bar
    key_hints = [
        ("1", "no_signal"),
        ("2", "thumbs_up"),
        ("3", "palm_open"),
        ("q", "quit"),
    ]
    for i, (key, hint) in enumerate(key_hints):
        cv2.putText(frame, f"[{key}] {hint}", (20, 65 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

    # Hand presence indicator: green when a hand is detected, red when not
    hand_text = f"Hands in frame: {hand_count}"
    hand_color = (0, 255, 0) if hand_count > 0 else (0, 0, 255)
    cv2.putText(frame, hand_text, (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)


def main() -> None:
    """Run the interactive data collection loop.

    Streams frames from the camera, runs hand detection on each, and saves
    landmark rows to the CSV when the operator presses a label key. Prints
    running totals to the console after each save. Cleans up all resources
    on exit regardless of how the loop terminates.
    """
    # Open the CSV and initialize the hand detector and per-class counters
    writer, csv_file = _open_csv()
    detector = HandDetector()
    counts = {"no_signal": 0, "thumbs_up": 0, "palm_open": 0}

    print(f"Saving to {DATA_PATH}")
    print("Press 1/2/3 to label and save, q to quit.")

    try:
        for frame in receive_frames():
            ts = int(time.time() * 1000)

            # Run hand detection on the current frame
            result = detector.detect(frame, ts)

            # Draw landmark skeleton if at least one hand is detected
            annotated = frame.copy()
            if result.hand_detected:
                draw_landmarks(annotated, result.all_landmarks)

            # Overlay HUD and display the annotated frame
            _draw_hud(annotated, counts, len(result.all_landmarks))
            cv2.imshow("Data Collection", annotated)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key in _LABEL_KEYS:
                label = _LABEL_KEYS[key]
                if not result.hand_detected:
                    # Ignore the keypress if no hand is visible in this frame
                    print("No hand detected — nothing saved.")
                else:
                    # Write one row per detected hand and flush to disk immediately
                    saved = _save_hands(writer, label, result.all_landmarks)
                    csv_file.flush()
                    counts[label] += saved
                    print(f"Saved {saved} row(s) as '{label}' — totals: {counts}")

    finally:
        # Always release the detector, file handle, and window on exit
        detector.close()
        csv_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
