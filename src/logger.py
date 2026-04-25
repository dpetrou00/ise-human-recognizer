import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Ordered list of CSV column names matching the data logging schema
FIELDNAMES = [
    "timestamp", "frame_num", "person_detected", "distance_m", "off_signal",
    "hand_detected", "gesture", "confidence", "command", "classifier",
]


def append_event(log_path: Path, event: dict) -> None:
    """Append one event row to the CSV log file.

    Creates the log file and writes the header automatically on the first call.
    Missing fields default to an empty string.

    Args:
        log_path (Path): Path to the CSV log file.
        event (dict): Dictionary of field names to values for this event.
    """

    # Create the logs directory if it does not exist
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if the file exists before opening so we know whether to write the header
    write_header = not log_path.exists()

    # Add a timestamp if the caller did not provide one
    event.setdefault("timestamp", datetime.now().isoformat())

    # Append the event row to the CSV
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

        # Write the header row on first write
        if write_header:
            writer.writeheader()

        # Write only the fields defined in FIELDNAMES, defaulting missing ones to ""
        writer.writerow({k: event.get(k, "") for k in FIELDNAMES})
