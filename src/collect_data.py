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

_LABEL_KEYS = {
    ord("1"): "no_signal",
    ord("2"): "thumbs_up",
    ord("3"): "palm_open",
}

_FIELDNAMES = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


def _open_csv() -> tuple[csv.DictWriter, TextIOWrapper]:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not DATA_PATH.exists()
    f = open(DATA_PATH, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
    if write_header:
        writer.writeheader()
    return writer, f


def _save_hands(writer: csv.DictWriter, label: str, all_landmarks: list) -> int:
    for hand_landmarks in all_landmarks:
        row: dict = {"label": label}
        for i, lm in enumerate(hand_landmarks):
            row[f"x{i}"] = round(lm.x, 6)
            row[f"y{i}"] = round(lm.y, 6)
            row[f"z{i}"] = round(lm.z, 6)
        writer.writerow(row)
    return len(all_landmarks)


def _draw_hud(frame, counts: dict, hand_count: int) -> None:
    labels = ["no_signal", "thumbs_up", "palm_open"]
    count_text = "  |  ".join(f"{k}: {counts[k]}" for k in labels)
    cv2.putText(frame, count_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    key_hints = [
        ("1", "no_signal"),
        ("2", "thumbs_up"),
        ("3", "palm_open"),
        ("q", "quit"),
    ]
    for i, (key, hint) in enumerate(key_hints):
        cv2.putText(frame, f"[{key}] {hint}", (20, 65 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

    hand_text = f"Hands in frame: {hand_count}"
    hand_color = (0, 255, 0) if hand_count > 0 else (0, 0, 255)
    cv2.putText(frame, hand_text, (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)


def main() -> None:
    writer, csv_file = _open_csv()
    detector = HandDetector()
    counts = {"no_signal": 0, "thumbs_up": 0, "palm_open": 0}

    print(f"Saving to {DATA_PATH}")
    print("Press 1/2/3 to label and save, q to quit.")

    try:
        for frame in receive_frames():
            ts = int(time.time() * 1000)
            result = detector.detect(frame, ts)

            annotated = frame.copy()
            if result.hand_detected:
                draw_landmarks(annotated, result.all_landmarks)

            _draw_hud(annotated, counts, len(result.all_landmarks))
            cv2.imshow("Data Collection", annotated)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key in _LABEL_KEYS:
                label = _LABEL_KEYS[key]
                if not result.hand_detected:
                    print("No hand detected — nothing saved.")
                else:
                    saved = _save_hands(writer, label, result.all_landmarks)
                    csv_file.flush()
                    counts[label] += saved
                    print(f"Saved {saved} row(s) as '{label}' — totals: {counts}")

    finally:
        detector.close()
        csv_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
