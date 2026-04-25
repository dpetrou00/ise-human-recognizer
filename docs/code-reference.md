# Code Reference

---

## capture.py

The source of camera frames. On the NUC, replace `receive_frames()` with a direct V4L2 implementation — the generator interface stays the same.

`capture_windows.py` is the **server** (it owns the webcam and binds the port). `capture.py` in WSL2 is the **client** (it connects to the Windows server).

### **Constants**

| Name | Value | Description |
|---|---|---|
| `HOST` | `"127.0.0.1"` | Address of the Windows camera server |
| `PORT` | `5000` | Port the Windows server listens on |

### **Functions**

#### `receive_frames() -> Iterator[np.ndarray]`
Generator. Connects to the Windows camera server and yields decoded BGR frames indefinitely. Each frame arrives as a `numpy` array in OpenCV BGR format.

Frames are length-prefixed: the server sends a 4-byte big-endian integer (frame byte count) followed by the JPEG-encoded frame. The client reads exactly that many bytes before decoding.

Raises `ConnectionError` if the server disconnects mid-frame.

---

## pose.py

Person detection using MediaPipe Pose Landmarker Lite.

### **Constants**

| Name | Description |
|---|---|
| `MODEL_PATH` | Path to `model/mediapipe/pose_landmarker_lite.task` |

### **Classes**

#### `PoseResult`
Dataclass returned by `PoseDetector.detect()`.

| Field | Type | Description |
|---|---|---|
| `person_detected` | `bool` | Whether at least one person was found in the frame |
| `landmarks` | `list` | Normalized MediaPipe pose landmarks for the first person, or `[]` if none |

#### `PoseDetector`
Wraps MediaPipe Pose Landmarker Lite. Loads the model once on construction and reuses it across frames.

Supports use as a context manager (`with PoseDetector() as d:`).

##### `__init__(model_path: Path = MODEL_PATH)`
Loads the Pose Landmarker Lite model in `IMAGE` running mode, configured to detect at most one person.

##### `detect(frame: np.ndarray) -> PoseResult`
Runs pose detection on a single BGR frame. Converts to RGB internally before passing to MediaPipe. Returns a `PoseResult` with `person_detected=False` and empty landmarks if no person is found.

##### `close() -> None`
Releases the underlying MediaPipe landmarker. Called automatically when used as a context manager.

---

## distance.py

Distance estimation using the pinhole camera model. Requires a one-time calibration before use.

**How it works:** The calibrated focal length `f` relates pixel height to real-world distance via `distance = (known_height × f) / pixel_height`, where `pixel_height` is the vertical span of visible pose landmarks in the frame.

### **Classes**

#### `DistanceResult`
Dataclass returned by `DistanceEstimator.estimate()`.

| Field | Type | Description |
|---|---|---|
| `distance_m` | `float \| None` | Estimated distance in meters, or `None` if not enough landmarks are visible |
| `off_signal` | `bool` | `True` when distance is below the configured threshold |

#### `DistanceEstimator`
Owns the calibrated focal length and computes per-frame distance estimates.

##### `__init__(user_height_m: float, threshold_m: float)`
- `user_height_m` — known height of the person being tracked, in meters
- `threshold_m` — distance below which `off_signal` is set to `True`; adjustable at runtime via `estimator.threshold_m`

##### `is_calibrated -> bool` *(property)*
`True` once `calibrate()` has been called successfully.

##### `calibrate(pose_result: PoseResult, frame_height: int, reference_distance_m: float) -> bool`
Computes and stores the focal length from a single calibration frame. The person must be standing at exactly `reference_distance_m` from the camera.

Returns `True` on success, `False` if not enough visible landmarks were found.

##### `estimate(pose_result: PoseResult, frame_height: int) -> DistanceResult`
Estimates the person's distance from the camera using the stored focal length. Returns `DistanceResult(distance_m=None, off_signal=False)` if not yet calibrated or no person is detected.

### **Functions**

#### `_landmark_pixel_height(pose_result: PoseResult, frame_height: int) -> float | None`
Internal. Computes the vertical pixel span of all pose landmarks with `visibility > 0.5`. Returns `None` if fewer than 4 landmarks meet the threshold.

---

## hands.py

Hand landmark detection using MediaPipe Hand Landmarker.

### **Constants**

| Name | Description |
|---|---|
| `MODEL_PATH` | Path to `model/mediapipe/hand_landmarker.task` |

### **Classes**

#### `HandResult`
Dataclass returned by `HandDetector.detect()`.

| Field | Type | Description |
|---|---|---|
| `hand_detected` | `bool` | Whether a hand was found in the frame |
| `landmarks` | `list` | 21 normalized MediaPipe hand landmarks for the first detected hand, or `[]` |

#### `HandDetector`
Wraps MediaPipe Hand Landmarker. Loads the model once on construction and reuses it across frames.

Supports use as a context manager.

##### `__init__(model_path: Path = MODEL_PATH)`
Loads the Hand Landmarker model in `IMAGE` running mode, configured to detect at most one hand.

##### `detect(frame: np.ndarray) -> HandResult`
Runs hand detection on a single BGR frame. Returns `HandResult(hand_detected=False, landmarks=[])` if no hand is found.

##### `close() -> None`
Releases the underlying MediaPipe landmarker.

---

## gesture.py

Gesture classification with two swappable backends and a shared interface.

### **Constants**

| Name | Description |
|---|---|
| `MODEL_PATH` | Path to `model/gesture_classifier.pkl` (custom trained model) |
| `MEDIAPIPE_MODEL_PATH` | Path to `model/mediapipe/gesture_recognizer.task` |
| `GESTURE_COMMANDS` | Maps gesture labels to command strings: `{"thumbs_up": "GO", "palm_open": "STOP", "no_signal": ""}` |

### **Gesture Labels**

| Label | Source |
|---|---|
| `"thumbs_up"` | Both backends |
| `"palm_open"` | Both backends |
| `"no_signal"` | Both backends (default when no recognized gesture) |

### **Classes**

#### `GestureClassifier` *(Protocol)*
Structural interface that both classifier backends must satisfy.

| Member | Type | Description |
|---|---|---|
| `name` | `str` | Identifier string for the backend (`"custom"` or `"mediapipe"`) |
| `predict(frame, landmarks)` | `-> tuple[str, float]` | Returns `(gesture_label, confidence)` |

#### `CustomClassifier`
Loads and runs a scikit-learn `MLPClassifier` saved at `model/gesture_classifier.pkl`.

Input features: 21 hand landmarks × 3 coordinates (x, y, z) = 63 floats, flattened into a 1D array.

Raises `FileNotFoundError` on construction if the model file does not exist.

##### `__init__(model_path: Path = MODEL_PATH)`
Loads the model using `joblib`.

##### `predict(frame: np.ndarray, landmarks: list) -> tuple[str, float]`
Converts landmarks to a feature vector, runs `model.predict()` and `model.predict_proba()`, returns the top label and its probability. `frame` is unused.

##### `close() -> None`
No-op. Exists for interface consistency.

#### `MediaPipeClassifier`
Runs MediaPipe Gesture Recognizer directly on the full frame.

MediaPipe gesture labels are mapped to the project's labels: `"Open_Palm"` → `"palm_open"`, `"Thumb_Up"` → `"thumbs_up"`, all others → `"no_signal"`.

##### `__init__(model_path: Path = MEDIAPIPE_MODEL_PATH)`
Loads the MediaPipe Gesture Recognizer in `IMAGE` running mode.

##### `predict(frame: np.ndarray, landmarks: list) -> tuple[str, float]`
Runs recognition on the full BGR frame. `landmarks` is unused. Returns `("no_signal", 1.0)` if no gesture is detected.

##### `close() -> None`
Releases the underlying MediaPipe recognizer.

### **Functions**

#### `build_classifier(use_custom: bool) -> GestureClassifier`
Factory. Returns a new `CustomClassifier` if `use_custom=True`, otherwise a new `MediaPipeClassifier`. Called by `demo.py` whenever the classifier toggle is activated.

---

## pipeline.py

Wires all components into a single frame-by-frame processing loop.

### **Classes**

#### `PipelineResult`
Dataclass yielded once per frame by `run_pipeline()`.

| Field | Type | Description |
|---|---|---|
| `frame` | `np.ndarray` | Original BGR frame |
| `person_detected` | `bool` | From `PoseDetector` |
| `distance_m` | `float \| None` | Estimated distance in meters |
| `off_signal` | `bool` | Whether distance is below threshold |
| `hand_detected` | `bool` | Whether a hand was found |
| `hand_landmarks` | `list` | Raw MediaPipe hand landmarks |
| `gesture` | `str` | Gesture label (`"thumbs_up"`, `"palm_open"`, or `"no_signal"`) |
| `confidence` | `float` | Classifier confidence score |
| `command` | `str` | Mapped command string (`"GO"`, `"STOP"`, or `""`) |
| `classifier_name` | `str` | Name of the active classifier backend |

### **Functions**

#### `run_pipeline(frames, pose_detector, hand_detector, estimator, classifier) -> Iterator[PipelineResult]`
Generator. Accepts any frame iterator as its first argument, making it usable with both `receive_frames()` (live camera) and pre-recorded frame sequences (testing/evaluation).

Pipeline order per frame:
1. Pose detection → person presence
2. Distance estimation → distance and off-signal
3. Hand detection (only if person detected)
4. Gesture classification (only if hand detected)

---

## overlay.py

Pure functions for compositing information onto frames. All functions modify the frame in-place.

### **Constants**

| Name | Description |
|---|---|
| `HAND_CONNECTIONS` | List of `(start, end)` landmark index pairs defining the hand skeleton topology (21 connections) |

### **Functions**

#### `draw_landmarks(frame: np.ndarray, landmarks: list) -> None`
Draws green dots at each of the 21 hand landmark positions and green lines along the hand skeleton connections.

#### `draw_distance(frame: np.ndarray, result: DistanceResult) -> None`
Draws the current distance estimate at the top-left of the frame. Color is green when safe and red when the OFF signal is active. Shows `"Distance: --"` in grey when distance cannot be estimated.

#### `draw_status_panel(frame, gesture, command, off_signal, classifier_name) -> None`
Draws a four-line status panel at the top-left of the frame showing:
- Machine status (`ON` in green / `OFF` in red)
- Current signal (`GO`, `STOP`, or `—`)
- Detected gesture label
- Active classifier name

---

## logger.py

Plain functions for persisting events.

### **Constants**

| Name | Description |
|---|---|
| `FIELDNAMES` | Ordered list of CSV column names |

### **CSV Schema**

| Field | Description |
|---|---|
| `timestamp` | ISO 8601 datetime of the event |
| `frame_num` | Sequential frame index since app start |
| `person_detected` | Boolean |
| `distance_m` | Estimated distance in meters, or empty |
| `off_signal` | Boolean |
| `hand_detected` | Boolean |
| `gesture` | Gesture label |
| `confidence` | Classifier confidence score |
| `command` | Mapped command string |
| `classifier` | Active classifier name |

### **Functions**

#### `append_event(log_path: Path, event: dict) -> None`
Appends one row to the CSV at `log_path`. Creates the file and writes the header automatically on first call. Missing fields default to `""`. Adds `timestamp` automatically if not provided.

---

## demo.py

Entry point for the live demo application.

### **Startup prompts** (collected once before models load):
- User height in meters
- Calibration reference distance in meters
- OFF-signal distance threshold in meters

### **Runtime keyboard controls**

| Key | Action |
|---|---|
| `c` | Calibrate (or recalibrate) the distance estimator using the current frame |
| `g` | Toggle between MediaPipe and custom gesture classifier |
| `+` / `=` | Increase the OFF-signal distance threshold by 0.1m |
| `-` | Decrease the OFF-signal distance threshold by 0.1m (minimum 0.1m) |
| `q` | Quit and release all resources |

### **Startup flow:**
1. Collects user inputs
2. Loads `PoseDetector`, `HandDetector`, and `MediaPipeClassifier` (default)
3. Starts TCP server and waits for `capture_windows.py` to connect
4. Shows calibration prompt until `c` is pressed with a person in frame
5. Enters the main processing loop

### **Calibration state:** 
Until calibration succeeds, only the calibration prompt is shown. Pressing `c` attempts calibration on the current frame; if no person is detected, it prints a message and waits for the next attempt.

### **Classifier toggle:** 
Pressing `g` calls `build_classifier()` and reassigns the active classifier reference. The switch takes effect on the next frame. If the custom model file (`model/gesture_classifier.pkl`) does not exist, switching to it is blocked with a message.

### **Functions**

#### `_prompt_startup() -> tuple[float, float, float]`
Prompts the user for height, calibration distance, and OFF-signal threshold. Returns all three as floats.

#### `main() -> None`
Orchestrates the full application lifecycle. Ensures models are closed and the display window is destroyed on exit, even if an exception occurs.
