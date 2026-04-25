# Project Plan: Human Recognizer for Industrial Robots

ISE 572 Final Project — Matt Carrell and Demetrios Petrou

---

## Milestones

### Phase 1 — Environment & Baseline Detection
- [x] Set up GitHub and project environment
  - Created GitHub repository at `github.com/dpetrou00/ise-human-recognizer` and configured local remote
  - Initialized Python 3.12 virtual environment (`.venv`)
  - Installed core packages: `mediapipe`, `opencv-python`, `numpy`, `pandas`; pinned all dependencies in `requirements.txt`
  - Established project folder structure: `src/` for source code, `logs/` for CSV event logs, `snapshots/` for gesture images, `model/` for any custom-trained classifier, `docs/` for project documents
- [x] Connect and verify USB webcam input
  - Discovered that WSL2 USB passthrough (usbipd-win) does not support isochronous USB transfers, preventing direct V4L2 camera access from WSL2
  - Implemented a TCP socket streaming workaround for WSL2 development: `capture_windows.py` captures frames via DirectShow on Windows and streams JPEG-encoded frames over localhost port 5000
  - `capture.py` exposes a `receive_frames()` generator that yields OpenCV BGR frames — on the NUC this should be replaced with direct V4L2 capture, but the generator interface remains the same for the rest of the pipeline
  - Verified live frame delivery end-to-end from Windows webcam into WSL2 pipeline
- [x] Integrate MediaPipe Pose Landmarker Lite
  - `PoseDetector` in `src/pose.py` loads `pose_landmarker_lite.task` from `model/mediapipe/`
  - Runs in `LIVE_STREAM` mode with async inference via a thread-safe callback
- [x] Confirm reliable person detection in live frames
  - `PoseResult.person_detected` is `True` when at least one pose is found; landmarks filtered to `visibility > 0.5`
- [x] Display skeleton overlay in a local window
  - `src/overlay.py` draws hand landmarks; live video renders in the PySide6 demo window

---

### Phase 2 — Distance Estimation
- [x] Implement focal length calibration routine that runs once at startup using a person of known height at a known reference distance
  - `DistanceEstimator.calibrate()` computes focal length from a person standing at a known reference distance using their pixel-height span
- [x] Compute per-frame distance estimate from Pose Landmarker landmark span, calibrated focal length, and known user height
  - `DistanceEstimator.estimate()` applies the pinhole formula each frame using calibrated focal length and user-supplied height
- [x] Trigger OFF signal when estimated distance falls below configurable threshold
  - `DistanceResult.off_signal` is set `True` when `distance < threshold_m`; threshold is configurable live in the demo UI

---

### Phase 3 — Gesture Data Collection & Custom Classifier
- [ ] Use MediaPipe Hand Landmarker to collect landmark data for three classes: No Signal, Thumbs Up, and Palm Open
- [ ] Apply 70 / 15 / 15 train / val / test split and train custom gesture classifier on collected landmarks
- [ ] Evaluate custom classifier: accuracy, precision, recall, and confusion matrix per class

---

### Phase 4 — Gesture Classification Interface
- [x] Implement inference function for custom gesture classifier
  - `CustomClassifier.predict()` in `src/gesture.py` flattens 21 hand landmarks into a 63-element feature vector and runs a saved scikit-learn MLPClassifier
- [x] Implement inference function for MediaPipe Gesture Recognizer
  - `MediaPipeClassifier.predict()` submits frames asynchronously to MediaPipe's `GestureRecognizer` and returns the latest result
- [x] Expose a unified interface that routes to either backend based on a toggle flag
  - `GestureClassifier` Protocol defines the shared `predict()` / `close()` interface; `build_classifier(use_custom)` returns the right backend; priority resolver always ranks STOP > GO > No Signal

---

### Phase 5 — Demo Application
- [x] Build demo app with: user height input and focal length calibration trigger
  - PySide6 `_MainWindow` in `src/demo.py` exposes height and calibration-distance spinboxes; "Calibrate Distance" button fires `DistanceEstimator.calibrate()` on the live frame
- [x] Add configurable distance threshold for OFF signal
  - Threshold spinbox updates `DistanceEstimator.threshold_m` in real time via `set_threshold()`
- [x] Add toggle to switch between custom classifier and MediaPipe Gesture Recognizer
  - "Toggle Model" button swaps classifiers mid-stream; gracefully falls back with a status message if the custom model file is absent
- [x] Display live video feed with Hand Landmarker markers overlaid
  - `_PipelineThread` emits annotated frames via `frame_ready` signal; `draw_landmarks()` overlays hand markers before emission
- [x] Display current machine status and active signal (OFF / Thumbs Up / Palm Open / No Signal)
  - Status panel shows Machine (ON/OFF/—), Signal, Gesture, Distance, Confidence, and Model in a live-updating grid

---

### Phase 6 — Evaluation & Comparison
- [ ] Test both classifiers across varied lighting conditions, distances, and backgrounds
- [ ] Record accuracy, precision, recall, confusion matrix, frame rate, and end-to-end latency for each
- [ ] Document failure cases and compare the two approaches

---

### Phase 7 — Submission Materials
- [ ] Comment all code
- [ ] Write README.md for project usage
- [ ] Record demo video showing full system running with both classifier modes
- [ ] Write final report sections tied to each phase outcome
- [ ] Prepare presentation slides

---

## Data Logging Schema

Each pipeline event is appended to a local CSV on the NUC:

| Field | Description |
|---|---|
| `timestamp` | ISO datetime of the frame |
| `frame_num` | Sequential frame index |
| `person_detected` | Boolean |
| `wrist_roi_found` | Boolean |
| `gesture` | Raw MediaPipe gesture label or empty |
| `confidence` | Confidence score if available |
| `command` | Mapped command: STOP, GO, or empty |

---

## Gesture Reference

| Gesture | Mapped Command |
|---|---|
| Open palm (`Open_Palm`) | STOP |
| Thumbs up (`Thumb_Up`) | GO |
| Anything else | No command |

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| NUC too slow for real-time inference | Use MediaPipe Lite models; profile and drop frame rate if needed |
| Lighting / background variability hurts accuracy | Collect samples across conditions; augment with brightness, rotation, zoom |
| Too many false commands erode operator trust | Tune confidence threshold; keep system advisory only |
| Missed STOP signal near live equipment | Track recall explicitly; treat missed STOP as high-priority failure case |
| Gesture stage degrades over time | Fall back to person-detection only; revert to physical machine constraints |

---

## Evaluation Targets

- Person detection: low false-positive and false-negative rate in the target workspace
- Gesture classification: precision and recall both acceptable for STOP and GO classes
- Latency: fast enough that a hand signal gets a visible system response in under ~1 second
- Frame rate: sufficient for smooth live display on NUC hardware

---

## Out of Scope

- Direct machine control (system is advisory only; does not replace e-stops or lockout procedures)
- Cloud connectivity or remote logging
- Identity tracking or biometric data
- More than two command gestures in the initial prototype
