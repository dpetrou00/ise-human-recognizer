# ISE 572 MediaPipe Human, Hand, and Gesture Tracking Lab Notes

## 1. System and Project Goal

**Hardware target:** Intel NUC5i3MYHE  
**CPU:** Intel Core i3-5010U @ 2.10 GHz  
**GPU:** Intel HD Graphics 5500  
**OS:** Ubuntu 22.04.5 LTS 64-bit  
**Primary camera mode:** local USB webcam through V4L2  
**Fallback camera mode:** TCP/JPEG stream from partner setup

The goal of this project is to build a lightweight edge-vision prototype for Industry 4.0 human-machine interaction. The system detects when a person is near a machine, watches for simple hand gestures, and maps those gestures into advisory robot states such as `STOP`, `GO`, and `HUMAN_NEARBY`.

The final prototype is designed to run locally on the Intel NUC without depending on cloud services. It is not meant to replace emergency stops, lockout/tagout, or industrial safety controls. It is an added awareness layer for a classroom prototype.

## 2. Why We Moved Away From HOG

The first person-detection direction used OpenCV HOG. It was useful as a baseline, but it was not a good long-term fit.

Main reasons:

- HOG only gives rough person detection.
- It does not naturally provide body landmarks.
- It does not help locate wrists or hands.
- It performed inconsistently in the webcam setup.
- It did not give us a clean path toward gesture recognition.

We moved to MediaPipe Tasks because MediaPipe Pose Landmarker gives body landmarks, can act as a person detector, and provides wrist locations that can guide hand and gesture processing.

## 3. Environment Setup

Project folder:

~/ise572/mediapipe_pose_baseline

Main folders:

models/
scripts/
logs/
results/
results/snapshots/

Virtual environment setup:

sudo apt update
sudo apt install -y python3.10-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install mediapipe opencv-python numpy pandas scikit-learn joblib

The current model folder is expected to contain:

models/
  pose_landmarker_lite.task
  hand_landmarker.task
  gesture_classifier.pkl
  gesture_recognizer.task

## 4. Camera Setup

The NUC uses direct V4L2 camera capture when possible.

Known-good local webcam settings:

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

The current script tries local cameras first:

/dev/video0
/dev/video1
/dev/video2


If no local camera works, it tries a stream format (for those running WSL):

4-byte big-endian JPEG frame length
JPEG frame bytes
OpenCV decodes JPEG into a BGR frame

Environment variables for the stream:

CAM_STREAM_HOST=172.27.48.1
CAM_STREAM_PORT=5000

All incoming frames are normalized to **640x480** so a high-resolution stream does not accidentally overload the NUC.

## 5. MediaPipe API Issue and Fix

An early issue came from older MediaPipe examples using:

mediapipe.solutions

That path did not work cleanly with the newer MediaPipe install. We kept the newer MediaPipe Tasks API and removed dependency on old drawing utilities.

Drawing is now handled manually with OpenCV:

cv2.line
cv2.circle
cv2.rectangle
cv2.putText

This made the code easier to control and easier to explain in the final project.

## 6. Pose Baseline

The first working MediaPipe version used:

- MediaPipe Pose Landmarker Lite
- VIDEO mode
- 640x480 frames
- manual OpenCV skeleton drawing
- pose detection as person detection

The system treats a returned pose as evidence that a person is present. This avoids needing a separate person detector.

The script was then expanded from one person to two people by setting:

num_poses = 2

This allowed the system to label up to two people and made it more useful for a shared work area.

## 7. Wrist-Guided Hand and Gesture Processing

The next design step was to avoid running expensive full-frame hand/gesture detection all the time.

Instead, the system uses pose landmarks to find likely signaling hands:

1. Run Pose Landmarker.
2. Find visible wrist and elbow landmarks.
3. Check whether the wrist is above the elbow.
4. Build a small wrist-centered region of interest (ROI).
5. Run hand/gesture processing only on that crop.

Current ROI size:

220 x 220 pixels

This keeps the NUC from constantly searching the entire frame for hands.

## 8. MediaPipe Timestamp Issue

When multiple wrist ROIs were processed in one frame, MediaPipe produced this error:

ValueError: Input timestamp must be monotonically increasing

Cause:

- The same MediaPipe landmarker was called multiple times using the same timestamp.
- VIDEO mode requires strictly increasing timestamps.

Fix:

- A `TimestampTracker` class was added.
- Each MediaPipe model gets its own increasing timestamp sequence.
- Pose, hand, and gesture calls no longer reuse the same timestamp.

## 9. Gesture Recognition Baseline

The first gesture version used MediaPipe Gesture Recognizer:

Model: gesture_recognizer.task
Mode: VIDEO
Gestures used: Thumb_Up and Open_Palm

Command mapping:

Thumb_Up   -> GO
Open_Palm  -> STOP
Other      -> no command

Safety rule:

STOP wins over GO

This means that if conflicting gestures are detected, STOP has priority.

## 10. Project Merge

The group project had two efforts that overlapped heavily. We both used the general goal of detecting a human near a machine and reading simple hand commands.

What was merged:

- distance estimation
- calibration
- CSV logging
- snapshot saving
- streamed camera input
- custom `.pkl` gesture classifier support
- UI improvements

We did not replace the existing Step 9 code. The existing code was already more NUC-friendly because it used local V4L2 camera capture, pose throttling, and wrist-guided ROIs. These other features were merged into that base.

## 11. Distance Calibration in Feet

The system now estimates distance in feet using a simple pinhole-style approximation.

Defaults:

User height: 5 ft 10 in
Calibration distance: 10 ft
Too-close STOP threshold: 6 ft

Calibration process:

1. Place the camera in its final position.
2. Mark a floor spot exactly 10 ft from the camera.
3. Tap `Calibrate (c)` or press `c`.
4. Stand on the 10 ft mark with the full body visible.
5. Give a thumbs-up.
6. The system calibrates when it sees both a person and a GO gesture.

Important limitation:

This is not true depth sensing. It is an estimated distance based on visible pose landmark height. If the camera moves, calibration should be repeated.

## 12. Robot State Logic

Current robot states:

CLEAR
HUMAN_NEARBY
GO
STOP
TOO_CLOSE_STOP

State priority:

1. If no person is visible, state is `CLEAR`.
2. If calibrated distance is below the threshold, state is `TOO_CLOSE_STOP`.
3. If the winning gesture is STOP, state is `STOP`.
4. If the winning gesture is GO, state is `GO`.
5. Otherwise, state is `HUMAN_NEARBY`.

Keeps distance safety above gesture commands.

## 13. Logging and Snapshots

The system logs structured CSV rows so results can be reviewed later.

Current log file:

logs/step13_2_events_clean.csv

Snapshots are saved to:

results/snapshots/

Logged fields include:

- timestamp
- frame number
- people count
- hand candidate count
- detected hand count
- whether a wrist ROI was found
- estimated distance in feet
- STOP threshold
- calibration status
- robot state
- gesture label
- confidence
- command
- active classifier
- hand candidate limit
- FPS
- system CPU percentage
- process CPU percentage
- snapshot path
- frame source

Logging is throttled so disk writes do not happen every frame.

Snapshots are saved for important events, high-confidence commands, state changes, or manual snapshot requests.

## 14. Touchscreen and Clean View UI

A lightweight OpenCV button layer was added instead of a heavier PySide interface.

Touch/mouse buttons:

Calibrate (c)
Snapshot (s)
Pose (p)
Model (m)
Hide UI (u) / Show UI (u)

Keyboard controls:

q     quit
c     calibrate
s     snapshot
p     toggle pose overlay
m     toggle gesture model
u     hide/show UI
h     cycle hand candidate load
[     decrease hand candidate load
]     increase hand candidate load
+/-   adjust STOP threshold
r     toggle ROI boxes
f     toggle full-frame fallback
l     toggle logging
k     clear calibration

Clean view behavior:

- Hides HUD, banners, and bottom buttons.
- Keeps the top-right Show UI button.
- Keeps person, pose, hand, and gesture overlays visible.

Useful for demos because the screen is less cluttered but still shows tracking results on the person.

## 15. Custom `.pkl` Gesture Classifier

We created a custom gesture model saved as:

gesture_classifier.pkl

This model is different from the MediaPipe `.task` gesture model.

The MediaPipe `gesture_recognizer.task` does both:

hand detection + gesture classification

The custom `.pkl` model only does:

gesture classification from hand landmarks

So the custom path requires an extra hand-landmark step:

Pose Landmarker
-> wrist ROI
-> Hand Landmarker
-> 21 hand landmarks
-> flatten to 63 features
-> custom gesture_classifier.pkl
-> STOP / GO / no signal

Feature format:

21 landmarks x 3 values = 63 values

The current script supports both classifiers and can toggle between them live:

custom-pkl
MediaPipe gesture-task

## 16. Hand Pressure and CPU Monitoring

The latest feature is a hand-load test. Earlier versions effectively processed one candidate hand at a time. The current version can process more wrist/hand candidates so we can see how much load the NUC can handle.

Hand candidate controls:

h   cycle 1 -> 2 -> 3 -> 4 -> 1
]   increase hand load
[   decrease hand load

With two detected people, the maximum hand candidate count is four:

2 people x 2 raised wrists = 4 candidate hands

The HUD now shows:

CPU sys: overall system CPU usage
CPU app: this Python process CPU usage
FPS: approximate loop frame rate

This lets us compare performance as more hands are processed.  During basic tests the CPU never increased above 60%.

## 18. Current Pipeline Summary

Current runtime flow:

1. Check required model files.
2. Open local camera if available.
3. Fall back to streamed camera if local camera fails.
4. Normalize frames to 640x480.
5. Run Pose Landmarker Lite every few frames.
6. Treat pose detection as person detection.
7. Use raised wrists to create candidate hand ROIs.
8. Process up to the selected hand candidate limit.
9. Use either the custom `.pkl` model path or the MediaPipe `.task` gesture path.
10. Pick the winning command using STOP-over-GO priority.
11. Apply distance-based STOP if calibrated and too close.
12. Draw tracking overlays and optional UI.
13. Log events and save snapshots.
14. Track FPS and CPU load.

## 19. Current Strengths

- Runs locally on low-cost edge hardware.
- Avoids cloud dependency.
- Uses pose landmarks instead of rough person boxes.
- Uses wrist-guided ROIs to reduce processing load.
- Supports local webcam and partner stream input.
- Supports touchscreen, mouse, and keyboard controls.
- Supports both MediaPipe and custom gesture models.
- Includes event logging and snapshots for report evidence.
- Includes a hand-load test to measure CPU pressure.

## 20. Current Limitations

- Distance is approximate and depends on camera placement and calibration quality.
- Gesture recognition is sensitive to lighting, hand angle, motion blur, and occlusion.
- Full-frame fallback is slower and should normally stay off.
- The custom `.pkl` classifier depends on how well the training data matches the live camera setup.
- This remains an advisory demo and should not be presented as a replacement for certified industrial safety equipment.

## 21. Final Tests

### Test 1: Local camera startup - Good
- Confirm `/dev/video0` opens correctly.
- Confirm stable video at 640x480.
- Record approximate FPS and CPU.

### Test 2: Stream fallback - Good
- Disable or unplug the local webcam.
- Start the partner stream.
- Confirm fallback to TCP/JPEG stream.

### Test 3: Distance calibration - Good
- Mark 10 ft from the camera.
- Tap `Calibrate (c)`.
- Give thumbs-up at the floor mark.
- Confirm calibration completes.

### Test 4: Distance STOP - Good
- Walk from beyond 10 ft toward the camera.
- Confirm `TOO_CLOSE_STOP` activates near 6 ft.

### Test 5: Gesture model comparison - Good
- Test `Thumb_Up` and `Open_Palm` with `custom-pkl`.
- Toggle to `gesture-task` and repeat.
- Compare confidence, false positives, missed commands, FPS, and CPU.

### Test 6: Hand pressure - Good
- Start with one raised hand.
- Press `h` to increase candidate load.
- Test one person with two hands raised.
- Test two people with both hands raised if possible.
- Record FPS and CPU changes.

### Test 7: UI modes - Good
- Test all touchscreen buttons.
- Hide UI with `u`.
- Confirm pose, hand, and gesture overlays remain visible.
- Show UI again with the top-right button.

### Test 8: Logs and snapshots 
- Confirm CSV log is created.
- Confirm snapshots save to `results/snapshots/`.
- Confirm logs include state, gesture, classifier, hand load, FPS, CPU, and frame source.

## 22. Final Status

The project has moved from a basic pose demo to a practical edge-vision prototype with:

- multi-person pose detection
- wrist-guided hand/gesture processing
- distance calibration in feet
- STOP/GO state logic
- local and streamed camera support
- touchscreen controls
- custom and MediaPipe gesture model toggle
- clean demo view
- hand pressure testing
- CPU/FPS monitoring
- logs and snapshots for evaluation

The current best file is:

scripts/ISE572_Final_Code.py
