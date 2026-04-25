"""
ISE Human Recognizer — Model Training

Loads landmark data from model/data/landmarks.csv, trains a 3-class MLP,
evaluates on train / val / test splits, and saves the model + metrics to
a timestamped checkpoint in model/checkpoints/.

Also copies the trained model to model/gesture_classifier.pkl so the demo
can load it immediately via the custom classifier toggle.

Usage:
    python src/train.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = Path(__file__).parent.parent / "model" / "data" / "landmarks.csv"
CHECKPOINTS_DIR = Path(__file__).parent.parent / "model" / "checkpoints"
PROMOTED_PATH = Path(__file__).parent.parent / "model" / "gesture_classifier.pkl"

CLASSES = ["no_signal", "thumbs_up", "palm_open"]


def _split(X: np.ndarray, y: np.ndarray):
    """Stratified 70 / 15 / 15 train / val / test split."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _evaluate(model: MLPClassifier, X: np.ndarray, y: np.ndarray) -> dict:
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    return {
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "precision_macro": round(report["macro avg"]["precision"], 4),
        "recall_macro": round(report["macro avg"]["recall"], 4),
        "f1_macro": round(report["macro avg"]["f1-score"], 4),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"], 4),
                "f1":        round(report[cls]["f1-score"], 4),
                "support":   int(report[cls]["support"]),
            }
            for cls in CLASSES
        },
    }


def main() -> None:
    if not DATA_PATH.exists():
        print(f"No data found at {DATA_PATH}. Run collect_data.py first.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples.")
    print(df["label"].value_counts().to_string())
    print()

    X = df.drop("label", axis=1).values.astype(np.float32)
    y = df["label"].values

    X_train, X_val, X_test, y_train, y_val, y_test = _split(X, y)
    print(f"Split — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}\n")

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=500,
        random_state=42,
        verbose=True,
    )

    print("Training...\n")
    model.fit(X_train, y_train)
    print(f"\nConverged in {model.n_iter_} iterations.\n")

    splits_metrics: dict = {}
    for name, X_s, y_s in [
        ("train", X_train, y_train),
        ("val",   X_val,   y_val),
        ("test",  X_test,  y_test),
    ]:
        result = _evaluate(model, X_s, y_s)
        splits_metrics[name] = result
        print(f"{name:5s} — accuracy: {result['accuracy']:.4f}  "
              f"precision: {result['precision_macro']:.4f}  "
              f"recall: {result['recall_macro']:.4f}  "
              f"F1: {result['f1_macro']:.4f}")

    print()

    metrics = {
        "model_config": {
            "hidden_layer_sizes": list(model.hidden_layer_sizes),
            "activation": model.activation,
            "max_iter": model.max_iter,
        },
        "n_iter": model.n_iter_,
        "loss_curve": [round(v, 6) for v in model.loss_curve_],
        "splits": splits_metrics,
    }

    run_id = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = CHECKPOINTS_DIR / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, checkpoint_dir / "gesture_classifier.pkl")
    (checkpoint_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Promote to the path CustomClassifier loads from
    joblib.dump(model, PROMOTED_PATH)

    print(f"Checkpoint saved: {checkpoint_dir}")
    print(f"Model promoted:   {PROMOTED_PATH}")


if __name__ == "__main__":
    main()
