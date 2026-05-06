# HALT: Human-Aware Local Triggering

A lightweight machine vision system for safety during human-robot interaction.

HALT uses a standard webcam and edge computer (Intel NUC) to watch for nearby human operators, estimate whether they are too close to dangerous equipment, recognize simple hand signals, and convert those observations into machine states: `CLEAR`, `HUMAN_NEARBY`, `GO`, `STOP`, or `TOO_CLOSE_STOP`. The system runs fully locally without cloud dependence, using MediaPipe for person and hand detection and a custom-trained gesture classifier that achieved 97.8% test accuracy and a macro-F1 of 0.978 across three gesture classes (no signal, thumbs up, palm open).

## Setup

Run the setup script to install all required dependencies:

```bash
bash setup.sh
```

## Repository Structure

```
docs/       Project documents and analysis
model/      Final trained gesture classifier and training data
src/        All source code
```

### docs/

| File | Description |
|------|-------------|
| `HALT Final Report.pdf` | Final project report with full system design, data pipeline, and evaluation |
| `evaluation.ipynb` | Raw analysis notebook — not a deliverable; refined analysis is in the final report |
| `ISE572_LabNotes_Final_Clean.md` | Lab notes (supplemental) |
| `ISE572_Final_Presentation.avi` | Final presentation recording (supplemental) |
| `Proposal.pdf` | Original project proposal (past deliverable) |
| `System Design.pdf` | System design document (past deliverable) |

### model/

Contains the final trained gesture classifier (`gesture_classifier.pkl`) and the landmark dataset used to train it (`data/landmarks.csv` — 1,515 labeled samples across three gesture classes).

### src/

| Path | Description |
|------|-------------|
| `src/final/ISE572_Final_Code.py` | Final deliverable — complete HALT application |
| `src/final/collect_data.py` | Script to collect and label hand landmark training data |
| `src/final/train.py` | Script to train the custom gesture classifier from collected landmarks |
| `src/draft/` | Earlier component scripts that were combined into the final code |
