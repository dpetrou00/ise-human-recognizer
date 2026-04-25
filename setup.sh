#!/bin/bash
set -e

echo "=== Installing system dependencies ==="
PKGS="libgles2 libxcb-xinerama0 libxcb-cursor0"
MISSING=""
for pkg in $PKGS; do
    dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed" || MISSING="$MISSING $pkg"
done
if [ -n "$MISSING" ]; then
    sudo apt-get install -y $MISSING
else
    echo "System packages already installed, skipping."
fi

echo "=== Installing Python dependencies ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -r requirements.txt

echo "=== Downloading MediaPipe models ==="
mkdir -p model/mediapipe model/data model/checkpoints logs/

MEDIAPIPE_BASE="https://storage.googleapis.com/mediapipe-models"

wget -q --show-progress -O model/mediapipe/pose_landmarker_lite.task \
    "$MEDIAPIPE_BASE/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

wget -q --show-progress -O model/mediapipe/hand_landmarker.task \
    "$MEDIAPIPE_BASE/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

wget -q --show-progress -O model/mediapipe/gesture_recognizer.task \
    "$MEDIAPIPE_BASE/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"

echo "=== Setup complete ==="