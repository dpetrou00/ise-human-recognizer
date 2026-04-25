"""
ISE 572 Human/Gesture Safety Demo

Controls:
    Touch/mouse: Calibrate (c), Snapshot (s), Pose (p)
    Keyboard:    q quit, c calibrate, s snapshot, p pose, +/- threshold
                 r ROI boxes, f full-frame fallback, l logging, k clear cal
"""

from __future__ import annotations

import csv
import os
import socket
import struct
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
SNAPSHOT_DIR = PROJECT_ROOT / "results" / "snapshots"

POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"
GESTURE_MODEL_PATH = MODELS_DIR / "gesture_recognizer.task"
LOG_PATH = LOGS_DIR / "step11_events_touch.csv"
WINDOW_NAME = "Step 11 - Human/Gesture Safety Demo"

# Camera and performance
LOCAL_CAMERA_INDICES = [0, 1, 2]
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30
STREAM_PORT = int(os.environ.get("CAM_STREAM_PORT", "5000"))
STREAM_CONNECT_TIMEOUT_SEC = 2.0

MAX_POSES = 2
POSE_EVERY_N_FRAMES = 3
POSE_VISIBILITY = 0.5
ROI_BOX_SIZE_PX = 220
MIN_COMMAND_CONFIDENCE = 0.55
FULL_FRAME_FALLBACK_EVERY_N_FRAMES = 6

# Defaults for touchscreen use
DEFAULT_USER_HEIGHT_FT = 5 + 10 / 12
DEFAULT_CALIBRATION_DISTANCE_FT = 10.0
DEFAULT_STOP_THRESHOLD_FT = 6.0
THRESHOLD_STEP_FT = 0.25
MIN_STOP_THRESHOLD_FT = 0.5

# Logging and snapshots
LOG_EVERY_N_FRAMES = 10
SNAPSHOT_CONFIDENCE_THRESHOLD = 0.60
SNAPSHOT_COOLDOWN_SEC = 2.0
JPEG_QUALITY = 85

# MediaPipe pose landmarks
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16

FULL_POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

STATE_COLORS = {
    "CLEAR": (0, 220, 0),
    "HUMAN_NEARBY": (0, 255, 255),
    "GO": (255, 180, 0),
    "STOP": (0, 0, 255),
    "TOO_CLOSE_STOP": (0, 0, 255),
}


@dataclass
class SetupConfig:
    user_height_ft: float = DEFAULT_USER_HEIGHT_FT
    calibration_distance_ft: float = DEFAULT_CALIBRATION_DISTANCE_FT
    stop_threshold_ft: float = DEFAULT_STOP_THRESHOLD_FT


@dataclass
class DistanceResult:
    distance_ft: float | None
    stop_distance_signal: bool


@dataclass
class GestureDetection:
    label: str
    score: float
    command: str
    person_id: int | None
    side: str
    roi: tuple[int, int, int, int]
    handedness: str = ""


@dataclass
class TouchButton:
    label: str
    action: str
    rect: tuple[int, int, int, int]


@dataclass
class RuntimeUI:
    buttons: list[TouchButton] = field(default_factory=list)
    pending_action: str | None = None
    message: str = "Tap Calibrate when ready."
    message_until: float = 0.0

    def set_message(self, message: str, seconds: float = 3.0) -> None:
        self.message = message
        self.message_until = time.time() + seconds

    def active_message(self) -> str:
        if self.message_until == 0 or time.time() <= self.message_until:
            return self.message
        return ""


@dataclass
class RuntimeFlags:
    draw_pose: bool = False
    draw_rois: bool = False
    logging: bool = True
    full_frame_fallback: bool = False
    calibration_mode: bool = False
    force_snapshot: bool = False


def calibration_prompt(config: SetupConfig) -> str:
    distance = f"{config.calibration_distance_ft:g}ft"
    return f"Stand {distance} away and give a thumbs up"


def ft_text(value: float | None) -> str:
    return "-- ft" if value is None else f"{value:.2f} ft"


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def visibility(lm) -> float:
    return float(getattr(lm, "visibility", 1.0))


def pixel_xy(lm, frame_shape) -> tuple[int, int]:
    h, w = frame_shape[:2]
    return int(lm.x * w), int(lm.y * h)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    if frame.shape[1] == FRAME_WIDTH and frame.shape[0] == FRAME_HEIGHT:
        return frame
    return cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)


def command_from_gesture(label: str, score: float) -> str:
    if score < MIN_COMMAND_CONFIDENCE:
        return ""
    if label == "Open_Palm":
        return "STOP"
    if label == "Thumb_Up":
        return "GO"
    return ""


def check_models() -> None:
    missing = [str(p) for p in (POSE_MODEL_PATH, GESTURE_MODEL_PATH) if not p.exists()]
    if not missing:
        return

    print("ERROR: Missing MediaPipe model file(s):")
    for path in missing:
        print(f"  - {path}")
    raise SystemExit("Place the .task files in the models folder and try again.")


class TimestampTracker:
    def __init__(self) -> None:
        self._last: dict[str, int] = {}

    def next(self, key: str) -> int:
        now_ms = int(time.time() * 1000)
        ts = max(now_ms, self._last.get(key, 0) + 1)
        self._last[key] = ts
        return ts


class DistanceEstimator:
    def __init__(self, user_height_ft: float, threshold_ft: float) -> None:
        self.user_height_ft = user_height_ft
        self.threshold_ft = threshold_ft
        self._focal_length_px: float | None = None

    @property
    def is_calibrated(self) -> bool:
        return self._focal_length_px is not None

    def reset(self) -> None:
        self._focal_length_px = None

    def calibrate(self, pose_landmarks: list, frame_height: int, reference_distance_ft: float) -> bool:
        pixel_height = self._landmark_pixel_height(pose_landmarks, frame_height)
        if pixel_height is None:
            return False
        self._focal_length_px = (pixel_height * reference_distance_ft) / self.user_height_ft
        return True

    def estimate_for_person(self, pose_landmarks: list, frame_height: int) -> DistanceResult:
        if not self.is_calibrated or self._focal_length_px is None:
            return DistanceResult(None, False)

        pixel_height = self._landmark_pixel_height(pose_landmarks, frame_height)
        if pixel_height is None:
            return DistanceResult(None, False)

        distance_ft = (self.user_height_ft * self._focal_length_px) / pixel_height
        return DistanceResult(distance_ft, distance_ft < self.threshold_ft)

    def estimate_min_distance(self, pose_lists: list, frame_height: int) -> DistanceResult:
        distances = [
            r.distance_ft
            for pose in pose_lists
            if (r := self.estimate_for_person(pose, frame_height)).distance_ft is not None
        ]
        if not distances:
            return DistanceResult(None, False)

        nearest = min(distances)
        return DistanceResult(nearest, nearest < self.threshold_ft)

    @staticmethod
    def _landmark_pixel_height(pose_landmarks: list, frame_height: int) -> float | None:
        ys = [lm.y * frame_height for lm in pose_landmarks if visibility(lm) >= POSE_VISIBILITY]
        if len(ys) < 4:
            return None
        span = max(ys) - min(ys)
        return span if span > 0 else None


def wrist_roi(frame, wrist_landmark, box_size_px: int) -> tuple[int, int, int, int] | None:
    if visibility(wrist_landmark) < POSE_VISIBILITY:
        return None

    h, w = frame.shape[:2]
    cx, cy = int(wrist_landmark.x * w), int(wrist_landmark.y * h)
    half = box_size_px // 2
    x1 = clamp(cx - half, 0, w - 1)
    y1 = clamp(cy - half, 0, h - 1)
    x2 = clamp(cx + half, 0, w - 1)
    y2 = clamp(cy + half, 0, h - 1)
    return None if x2 <= x1 or y2 <= y1 else (x1, y1, x2, y2)


def candidate_wrist(pose_landmarks) -> tuple[str, object] | None:
    left_elbow = pose_landmarks[LEFT_ELBOW]
    right_elbow = pose_landmarks[RIGHT_ELBOW]
    left_wrist = pose_landmarks[LEFT_WRIST]
    right_wrist = pose_landmarks[RIGHT_WRIST]

    left_ok = (
        visibility(left_wrist) >= POSE_VISIBILITY
        and visibility(left_elbow) >= POSE_VISIBILITY
        and left_wrist.y < left_elbow.y
    )
    right_ok = (
        visibility(right_wrist) >= POSE_VISIBILITY
        and visibility(right_elbow) >= POSE_VISIBILITY
        and right_wrist.y < right_elbow.y
    )

    if not left_ok and not right_ok:
        return None
    if left_ok and not right_ok:
        return "left", left_wrist
    if right_ok and not left_ok:
        return "right", right_wrist
    return ("left", left_wrist) if left_wrist.y <= right_wrist.y else ("right", right_wrist)


def winning_detection(detections: list[GestureDetection]) -> GestureDetection | None:
    if not detections:
        return None
    for command in ("STOP", "GO"):
        matches = [d for d in detections if d.command == command]
        if matches:
            return max(matches, key=lambda d: d.score)
    return max(detections, key=lambda d: d.score)


def recognize_roi(
    gesture_recognizer,
    frame,
    roi: tuple[int, int, int, int],
    timestamps: TimestampTracker,
    person_id: int | None,
    side: str,
):
    x1, y1, x2, y2 = roi
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return [], None

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = gesture_recognizer.recognize_for_video(mp_image, timestamps.next("gesture"))

    detections: list[GestureDetection] = []
    if not result.hand_landmarks:
        return detections, result

    for i, _ in enumerate(result.hand_landmarks):
        label, score, handedness = "None", 0.0, ""

        if result.gestures and i < len(result.gestures) and result.gestures[i]:
            top = result.gestures[i][0]
            label, score = top.category_name, float(top.score)

        if result.handedness and i < len(result.handedness) and result.handedness[i]:
            handedness = result.handedness[i][0].category_name

        detections.append(
            GestureDetection(
                label=label,
                score=score,
                command=command_from_gesture(label, score),
                person_id=person_id,
                side=side,
                roi=roi,
                handedness=handedness,
            )
        )

    return detections, result


def draw_panel(image, rows: list[tuple[str, tuple[int, int, int]]], origin=(10, 10), width=360) -> None:
    x, y = origin
    line_h = 24
    height = 14 + line_h * len(rows)

    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, image, 0.30, 0, image)

    for i, (text, color) in enumerate(rows):
        cv2.putText(image, text, (x + 10, y + 24 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1, cv2.LINE_AA)


def draw_banner(image, message: str) -> None:
    if not message:
        return

    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 56), (w, 96), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)
    cv2.putText(image, message, (18, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (0, 255, 255), 2, cv2.LINE_AA)


def draw_pose(image, landmarks, person_id: int, distance_ft: float | None) -> None:
    h, w = image.shape[:2]
    for start, end in FULL_POSE_CONNECTIONS:
        if start >= len(landmarks) or end >= len(landmarks):
            continue
        a, b = landmarks[start], landmarks[end]
        if visibility(a) < POSE_VISIBILITY or visibility(b) < POSE_VISIBILITY:
            continue
        cv2.line(image, (int(a.x * w), int(a.y * h)), (int(b.x * w), int(b.y * h)), (0, 220, 0), 2)

    for lm in landmarks:
        if visibility(lm) >= POSE_VISIBILITY:
            cv2.circle(image, pixel_xy(lm, image.shape), 3, (0, 255, 255), -1)

    draw_person_label(image, landmarks, person_id, distance_ft, compact=True)


def draw_person_label(image, landmarks, person_id: int, distance_ft: float | None, compact: bool = False) -> None:
    nose = landmarks[NOSE]
    if visibility(nose) < POSE_VISIBILITY:
        return

    nx, ny = pixel_xy(nose, image.shape)
    if compact:
        label = f"P{person_id}" + (f" {distance_ft:.1f}ft" if distance_ft is not None else "")
    else:
        label = f"Person {person_id}" + (f"  {distance_ft:.2f} ft" if distance_ft is not None else "")

    cv2.putText(image, label, (nx + 10, max(ny - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (255, 255, 0), 2, cv2.LINE_AA)


def roi_points(hand_landmarks, roi: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    x1, y1, x2, y2 = roi
    roi_w, roi_h = x2 - x1, y2 - y1
    return [(int(x1 + lm.x * roi_w), int(y1 + lm.y * roi_h)) for lm in hand_landmarks]


def draw_hand(image, hand_landmarks, roi, label: str) -> None:
    points = roi_points(hand_landmarks, roi)
    if not points:
        return

    xs, ys = [p[0] for p in points], [p[1] for p in points]
    bx1 = clamp(min(xs) - 10, 0, image.shape[1] - 1)
    by1 = clamp(min(ys) - 10, 0, image.shape[0] - 1)
    bx2 = clamp(max(xs) + 10, 0, image.shape[1] - 1)
    by2 = clamp(max(ys) + 10, 0, image.shape[0] - 1)

    cv2.rectangle(image, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
    if label:
        cv2.putText(image, label, (bx1, max(by1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (255, 0, 0), 2, cv2.LINE_AA)

    for p in points:
        cv2.circle(image, p, 4, (0, 255, 0), -1)
    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(image, points[start], points[end], (0, 180, 0), 2)


def draw_hud(
    image,
    people_count: int,
    robot_state: str,
    winning: GestureDetection | None,
    distance: DistanceResult,
    estimator: DistanceEstimator,
    fps: float,
    flags: RuntimeFlags,
    source_name: str,
) -> None:
    color = STATE_COLORS.get(robot_state, (255, 255, 255))
    command = winning.command if winning else ""
    gesture = f"{winning.label} {winning.score:.2f}" if winning else "-"
    dist = ft_text(distance.distance_ft) if estimator.is_calibrated else "not calibrated"
    cal = "CAL MODE" if flags.calibration_mode else ("CAL" if estimator.is_calibrated else "NO CAL")

    rows = [
        (robot_state, color),
        (f"People {people_count} | Dist {dist}", (230, 230, 230)),
        (f"Cmd {command or '-'} | Gesture {gesture}", (230, 230, 230)),
        (f"{cal} | FPS {fps:.1f}", (180, 180, 180)),
    ]
    draw_panel(image, rows, origin=(10, 8), width=370)
    cv2.putText(image, source_name, (image.shape[1] - 210, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (170, 170, 170), 1, cv2.LINE_AA)


def draw_buttons(image, ui: RuntimeUI, flags: RuntimeFlags) -> None:
    h, w = image.shape[:2]
    margin, gap, button_h = 10, 10, 52
    y1, y2 = h - button_h - margin, h - margin
    items = [
        ("Calibrating (c)" if flags.calibration_mode else "Calibrate (c)", "calibrate"),
        ("Snapshot (s)", "snapshot"),
        ("Pose (p): ON" if flags.draw_pose else "Pose (p): OFF", "toggle_pose"),
    ]

    button_w = int((w - 2 * margin - gap * (len(items) - 1)) / len(items))
    ui.buttons.clear()

    overlay = image.copy()
    cv2.rectangle(overlay, (0, y1 - 10), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.70, image, 0.30, 0, image)

    for i, (label, action) in enumerate(items):
        x1 = margin + i * (button_w + gap)
        x2 = x1 + button_w
        ui.buttons.append(TouchButton(label, action, (x1, y1, x2, y2)))

        fill, border = (45, 45, 45), (180, 180, 180)
        if action == "calibrate" and flags.calibration_mode:
            fill, border = (0, 95, 130), (0, 255, 255)
        if action == "toggle_pose" and flags.draw_pose:
            fill, border = (40, 90, 40), (0, 220, 0)

        cv2.rectangle(image, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), border, 2)

        font_scale, thickness = 0.62, 2
        text_w, text_h = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        if text_w > button_w - 16:
            font_scale = 0.52
            text_w, text_h = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

        tx = x1 + max(8, (button_w - text_w) // 2)
        ty = y1 + (button_h + text_h) // 2 - 2
        cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def mouse_callback(event, x, y, _flags, ui: RuntimeUI) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    for button in ui.buttons:
        x1, y1, x2, y2 = button.rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            ui.pending_action = button.action
            return


CSV_FIELDS = [
    "timestamp", "frame_num", "people_count", "candidate_count", "hand_count",
    "wrist_roi_found", "distance_min_ft", "stop_distance_signal", "stop_threshold_ft",
    "calibrated", "robot_state", "gesture", "confidence", "command", "person_id",
    "side", "fps", "snapshot_path", "frame_source",
]


def append_log(row: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def save_snapshot(image, frame_num: int, state: str, winning: GestureDetection | None) -> str:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gesture = (winning.label if winning else "none").lower().replace(" ", "_")
    score = f"{winning.score:.2f}" if winning else "0.00"
    filename = f"{stamp}_f{frame_num:06d}_{state.lower()}_{gesture}_{score}.jpg"
    path = SNAPSHOT_DIR / filename
    cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return str(path)


def should_snapshot(now: float, last_time: float, state_changed: bool, winning, force: bool) -> bool:
    if force:
        return True
    if now - last_time < SNAPSHOT_COOLDOWN_SEC:
        return False
    if state_changed and winning and winning.command:
        return True
    return bool(winning and winning.command and winning.score >= SNAPSHOT_CONFIDENCE_THRESHOLD)


class LocalCameraSource:
    def __init__(self, index: int) -> None:
        self.index = index
        self.name = f"local /dev/video{index}"
        self.kind = "local"
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open {self.name}")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        return self.cap.read()

    def release(self) -> None:
        self.cap.release()


class TcpStreamSource:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.name = f"stream {host}:{port}"
        self.kind = "stream"
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(STREAM_CONNECT_TIMEOUT_SEC)
        self.sock.connect((host, port))
        self.sock.settimeout(None)

    def _recv_exactly(self, n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("stream server disconnected")
            data += chunk
        return data

    def read(self):
        try:
            frame_len = struct.unpack(">I", self._recv_exactly(4))[0]
            payload = self._recv_exactly(frame_len)
            frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            return (frame is not None), frame
        except (ConnectionError, OSError, struct.error) as exc:
            print(f"Stream read failed from {self.name}: {exc}")
            return False, None

    def release(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def default_gateway_ip() -> str | None:
    try:
        output = subprocess.check_output(
            ["ip", "route"], text=True, stderr=subprocess.DEVNULL, timeout=1.0
        )
    except Exception:
        return None

    for line in output.splitlines():
        parts = line.split()
        if parts and parts[0] == "default" and "via" in parts:
            via = parts.index("via")
            if via + 1 < len(parts):
                return parts[via + 1]
    return None


def stream_hosts() -> list[str]:
    candidates = [os.environ.get("CAM_STREAM_HOST"), default_gateway_ip(), "127.0.0.1", "172.27.48.1"]
    seen: set[str] = set()
    hosts: list[str] = []
    for host in candidates:
        if host and host not in seen:
            hosts.append(host)
            seen.add(host)
    return hosts


def test_source(source, reads: int = 8) -> bool:
    for _ in range(reads):
        ok, frame = source.read()
        if ok and frame is not None:
            return True
    return False


def open_local_source():
    for index in LOCAL_CAMERA_INDICES:
        try:
            source = LocalCameraSource(index)
            if test_source(source):
                print(f"Using camera: {source.name}")
                return source
            source.release()
        except Exception as exc:
            print(f"No local camera /dev/video{index}: {exc}")
    return None


def open_stream_source():
    for host in stream_hosts():
        try:
            source = TcpStreamSource(host, STREAM_PORT)
            if test_source(source, reads=2):
                print(f"Using camera stream: {source.name}")
                return source
            source.release()
        except Exception as exc:
            print(f"No stream {host}:{STREAM_PORT}: {exc}")
    return None


def open_frame_source():
    print("Trying local camera first...")
    source = open_local_source()
    if source:
        return source

    print("Trying TCP/JPEG stream fallback...")
    source = open_stream_source()
    if source:
        return source

    raise SystemExit("No usable camera source found.")


def person_distances(estimator: DistanceEstimator, pose_lists: list, frame_height: int) -> dict[int, float | None]:
    return {
        i: estimator.estimate_for_person(pose, frame_height).distance_ft
        for i, pose in enumerate(pose_lists, start=1)
    }


def run_pose(pose_landmarker, frame, timestamps: TimestampTracker):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return pose_landmarker.detect_for_video(mp_image, timestamps.next("pose"))


def process_pose_guided_gestures(
    frame,
    display,
    pose_lists: list,
    gesture_recognizer,
    timestamps: TimestampTracker,
    flags: RuntimeFlags,
) -> tuple[list[GestureDetection], int, int, bool]:
    detections: list[GestureDetection] = []
    candidate_count = 0
    hand_count = 0
    roi_found = False

    for person_id, pose_landmarks in enumerate(pose_lists, start=1):
        candidate = candidate_wrist(pose_landmarks)
        if not candidate:
            continue

        candidate_count += 1
        side, wrist = candidate
        roi = wrist_roi(frame, wrist, ROI_BOX_SIZE_PX)
        if not roi:
            continue

        roi_found = True
        x1, y1, x2, y2 = roi
        if flags.draw_rois:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.putText(display, f"P{person_id} {side} ROI", (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        roi_detections, result = recognize_roi(gesture_recognizer, frame, roi, timestamps, person_id, side)
        detections.extend(roi_detections)

        if result and result.hand_landmarks:
            hand_count += len(result.hand_landmarks)
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                label = ""
                if i < len(roi_detections):
                    d = roi_detections[i]
                    parts = [p for p in (d.handedness, f"{d.label} {d.score:.2f}", d.command) if p]
                    label = " | ".join(parts)
                draw_hand(display, hand_landmarks, roi, label)

    return detections, candidate_count, hand_count, roi_found


def process_full_frame_fallback(frame, display, gesture_recognizer, timestamps: TimestampTracker):
    roi = (0, 0, frame.shape[1], frame.shape[0])
    detections, result = recognize_roi(gesture_recognizer, frame, roi, timestamps, None, "full-frame")
    hand_count = 0

    if result and result.hand_landmarks:
        hand_count = len(result.hand_landmarks)
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            label = "fallback"
            if i < len(detections):
                d = detections[i]
                label = f"fallback | {d.label} {d.score:.2f}" + (f" | {d.command}" if d.command else "")
            draw_hand(display, hand_landmarks, roi, label)

    return detections, hand_count


def robot_state_for(people_count: int, distance: DistanceResult, winning: GestureDetection | None) -> str:
    if people_count == 0:
        return "CLEAR"
    if distance.stop_distance_signal:
        return "TOO_CLOSE_STOP"
    if winning and winning.command == "STOP":
        return "STOP"
    if winning and winning.command == "GO":
        return "GO"
    return "HUMAN_NEARBY"


def action_from_key(key: int) -> str | None:
    return {
        ord("c"): "calibrate",
        ord("s"): "snapshot",
        ord("p"): "toggle_pose",
    }.get(key)


def apply_action(action: str | None, key: int, flags: RuntimeFlags, ui: RuntimeUI,
                 estimator: DistanceEstimator, config: SetupConfig) -> bool:
    if key == ord("q"):
        return False

    if action == "calibrate":
        flags.calibration_mode = True
        ui.message = calibration_prompt(config)
        print(f"Calibration mode: {ui.message}")
    elif action == "snapshot":
        flags.force_snapshot = True
        ui.set_message("Snapshot queued.", 2.0)
        print("Snapshot queued.")
    elif action == "toggle_pose":
        flags.draw_pose = not flags.draw_pose
        ui.set_message(f"Pose overlay {'on' if flags.draw_pose else 'off'}.", 2.0)
        print(f"Pose overlay: {'on' if flags.draw_pose else 'off'}")
    elif key == ord("k"):
        estimator.reset()
        flags.calibration_mode = False
        ui.set_message("Distance calibration cleared.", 3.0)
        print("Distance calibration cleared.")
    elif key in (ord("+"), ord("=")):
        estimator.threshold_ft += THRESHOLD_STEP_FT
        config.stop_threshold_ft = estimator.threshold_ft
        ui.set_message(f"STOP threshold {estimator.threshold_ft:.2f} ft", 2.0)
        print(f"STOP threshold: {estimator.threshold_ft:.2f} ft")
    elif key == ord("-"):
        estimator.threshold_ft = max(MIN_STOP_THRESHOLD_FT, estimator.threshold_ft - THRESHOLD_STEP_FT)
        config.stop_threshold_ft = estimator.threshold_ft
        ui.set_message(f"STOP threshold {estimator.threshold_ft:.2f} ft", 2.0)
        print(f"STOP threshold: {estimator.threshold_ft:.2f} ft")
    elif key == ord("r"):
        flags.draw_rois = not flags.draw_rois
        ui.set_message(f"ROI boxes {'on' if flags.draw_rois else 'off'}.", 2.0)
        print(f"ROI boxes: {'on' if flags.draw_rois else 'off'}")
    elif key == ord("f"):
        flags.full_frame_fallback = not flags.full_frame_fallback
        ui.set_message(f"Fallback {'on' if flags.full_frame_fallback else 'off'}.", 2.0)
        print(f"Full-frame fallback: {'on' if flags.full_frame_fallback else 'off'}")
    elif key == ord("l"):
        flags.logging = not flags.logging
        ui.set_message(f"Logging {'on' if flags.logging else 'off'}.", 2.0)
        print(f"Logging: {'on' if flags.logging else 'off'}")

    return True


def log_row(frame_num, people_count, candidate_count, hand_count, roi_found, distance,
            estimator, state, winning, fps, snapshot_path, source_name) -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "frame_num": frame_num,
        "people_count": people_count,
        "candidate_count": candidate_count,
        "hand_count": hand_count,
        "wrist_roi_found": roi_found,
        "distance_min_ft": round(distance.distance_ft, 3) if distance.distance_ft is not None else "",
        "stop_distance_signal": distance.stop_distance_signal,
        "stop_threshold_ft": round(estimator.threshold_ft, 3),
        "calibrated": estimator.is_calibrated,
        "robot_state": state,
        "gesture": winning.label if winning else "",
        "confidence": round(winning.score, 4) if winning else "",
        "command": winning.command if winning else "",
        "person_id": winning.person_id if winning else "",
        "side": winning.side if winning else "",
        "fps": round(fps, 2),
        "snapshot_path": snapshot_path,
        "frame_source": source_name,
    }


def main() -> None:
    check_models()

    config = SetupConfig()
    estimator = DistanceEstimator(config.user_height_ft, config.stop_threshold_ft)
    timestamps = TimestampTracker()
    flags = RuntimeFlags()
    ui = RuntimeUI()
    frame_source = open_frame_source()

    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=MAX_POSES,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    gesture_options = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_path=str(GESTURE_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("\nDefaults:")
    print(f"  Height: {config.user_height_ft:.2f} ft")
    print(f"  Calibration distance: {config.calibration_distance_ft:.2f} ft")
    print(f"  STOP threshold: {config.stop_threshold_ft:.2f} ft")
    print("\nControls: touch/mouse buttons or keyboard c/s/p/q. Extras: +/- r f l k.\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, ui)

    last_pose_result = None
    last_snapshot_time = 0.0
    last_state = ""
    frame_num = 0
    prev_time = time.time()

    with vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
         vision.GestureRecognizer.create_from_options(gesture_options) as gesture_recognizer:

        while True:
            ok, frame = frame_source.read()
            if not ok or frame is None:
                print(f"Could not read frame from {frame_source.name}.")
                if getattr(frame_source, "kind", "") == "local":
                    frame_source.release()
                    fallback = open_stream_source()
                    if fallback:
                        frame_source = fallback
                        continue
                break

            frame = normalize_frame(frame)
            display = frame.copy()
            frame_num += 1

            if frame_num % POSE_EVERY_N_FRAMES == 0 or last_pose_result is None:
                last_pose_result = run_pose(pose_landmarker, frame, timestamps)

            pose_lists = last_pose_result.pose_landmarks if last_pose_result and last_pose_result.pose_landmarks else []
            people_count = len(pose_lists)
            distance = estimator.estimate_min_distance(pose_lists, frame.shape[0])
            distances = person_distances(estimator, pose_lists, frame.shape[0])

            for person_id, pose_landmarks in enumerate(pose_lists, start=1):
                if flags.draw_pose:
                    draw_pose(display, pose_landmarks, person_id, distances.get(person_id))
                else:
                    draw_person_label(display, pose_landmarks, person_id, distances.get(person_id))

            detections, candidate_count, hand_count, roi_found = process_pose_guided_gestures(
                frame, display, pose_lists, gesture_recognizer, timestamps, flags
            )

            fallback_due = frame_num % (3 if flags.calibration_mode else FULL_FRAME_FALLBACK_EVERY_N_FRAMES) == 0
            use_fallback = (
                people_count > 0
                and fallback_due
                and (flags.calibration_mode or (flags.full_frame_fallback and candidate_count == 0))
            )
            if use_fallback:
                ff_detections, ff_hands = process_full_frame_fallback(frame, display, gesture_recognizer, timestamps)
                detections.extend(ff_detections)
                hand_count += ff_hands

            winning = winning_detection(detections)

            if flags.calibration_mode:
                if people_count == 0:
                    ui.message = calibration_prompt(config)
                elif winning and winning.command == "GO":
                    ok = estimator.calibrate(pose_lists[0], frame.shape[0], config.calibration_distance_ft)
                    if ok:
                        flags.calibration_mode = False
                        ui.set_message("Calibrated. Distance STOP is active.", 4.0)
                        print("Distance calibration: OK")
                    else:
                        ui.message = "Full body must be visible to calibrate."
                else:
                    ui.message = calibration_prompt(config)

            state = robot_state_for(people_count, distance, winning)
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            state_changed = state != last_state
            last_state = state

            draw_hud(display, people_count, state, winning, distance, estimator, fps, flags, frame_source.name)

            message = ui.active_message()
            if flags.calibration_mode:
                message = calibration_prompt(config)
            elif not estimator.is_calibrated:
                message = message or "Tap Calibrate to enable distance STOP"
            draw_banner(display, message)
            draw_buttons(display, ui, flags)

            snapshot_path = ""
            if should_snapshot(now, last_snapshot_time, state_changed, winning, flags.force_snapshot):
                snapshot_path = save_snapshot(display, frame_num, state, winning)
                last_snapshot_time = now
                flags.force_snapshot = False

            if flags.logging and (frame_num % LOG_EVERY_N_FRAMES == 0 or state_changed or snapshot_path):
                append_log(log_row(frame_num, people_count, candidate_count, hand_count, roi_found,
                                   distance, estimator, state, winning, fps, snapshot_path, frame_source.name))

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF

            action = ui.pending_action or action_from_key(key)
            ui.pending_action = None
            if not apply_action(action, key, flags, ui, estimator, config):
                break

    frame_source.release()
    cv2.destroyAllWindows()
    print(f"\nLog file: {LOG_PATH}")
    print(f"Snapshots: {SNAPSHOT_DIR}")


if __name__ == "__main__":
    main()
