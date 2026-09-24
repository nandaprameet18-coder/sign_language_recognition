"""
train_model.py
──────────────
Trains a KNN classifier on the synthetic ASL dataset.
Saves the model and label encoder to model/.
"""

import os
import sys
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))
from utils.preprocessing import preprocess

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_split(split="train"):
    X, y = [], []
    split_dir = os.path.join(DATASET_DIR, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Dataset split not found: {split_dir}")
    for label in sorted(os.listdir(split_dir)):
        label_dir = os.path.join(split_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if not fname.lower().endswith(".png"):
                continue
            path = os.path.join(label_dir, fname)
            try:
                X.append(preprocess(path))
                y.append(label)
            except Exception as e:
                print(f"  [skip] {path}: {e}")
    return np.array(X), np.array(y)


def train():
    print("Loading training data …")
    X_train, y_train = load_split("train")
    print(f"  {len(X_train)} train samples, {len(set(y_train))} classes")

    print("Loading test data …")
    X_test, y_test = load_split("test")
    print(f"  {len(X_test)} test samples")

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    print("\nTraining KNN classifier (k=5) …")
    clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1)
    clf.fit(X_train, y_train_enc)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test_enc, y_pred)
    print(f"\n  Test Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

    # Save
    with open(os.path.join(MODEL_DIR, "knn_model.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    with open(os.path.join(MODEL_DIR, "accuracy.txt"), "w") as f:
        f.write(f"{acc*100:.2f}")

    print(f"\n[✓] Model saved to '{MODEL_DIR}/'")
    return acc


if __name__ == "__main__":
    train()
