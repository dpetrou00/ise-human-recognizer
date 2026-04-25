#!/bin/bash
set -e

echo "=== Installing system dependencies ==="
sudo apt-get install -y libgles2 libxcb-xinerama0 libxcb-cursor0

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Downloading MediaPipe models ==="
mkdir -p model/mediapipe model/data model/checkpoints

MEDIAPIPE_BASE="https://storage.googleapis.com/mediapipe-models"

wget -q --show-progress -O model/mediapipe/pose_landmarker_lite.task \
    "$MEDIAPIPE_BASE/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

wget -q --show-progress -O model/mediapipe/hand_landmarker.task \
    "$MEDIAPIPE_BASE/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

wget -q --show-progress -O model/mediapipe/gesture_recognizer.task \
    "$MEDIAPIPE_BASE/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"

echo "=== Setup complete ==="