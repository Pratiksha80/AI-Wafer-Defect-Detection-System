"""
Train wafer defect classifier and save model/wafer_model.h5.
Generates a small synthetic dataset if dataset/ folders are empty.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "wafer_model.h5"
IMG_SIZE = 128
LABELS = ["Crack", "Scratch", "Spot", "Contamination", "Normal wafer"]
SAMPLES_PER_CLASS = 40


def _draw_wafer(canvas: np.ndarray) -> None:
    h, w = canvas.shape[:2]
    cx, cy = w // 2, h // 2
    radius = min(h, w) // 2 - 8
    cv2.circle(canvas, (cx, cy), radius, (200, 200, 200), -1)
    cv2.circle(canvas, (cx, cy), radius, (120, 120, 120), 2)


def _synthetic_image(label: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((IMG_SIZE, IMG_SIZE, 3), 30, dtype=np.uint8)
    _draw_wafer(img)
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    if label == "Crack":
        for _ in range(rng.integers(2, 5)):
            x1, y1 = int(rng.integers(20, w - 20)), int(rng.integers(20, h - 20))
            x2, y2 = int(rng.integers(20, w - 20)), int(rng.integers(20, h - 20))
            cv2.line(img, (x1, y1), (x2, y2), (40, 40, 40), rng.integers(1, 3))
    elif label == "Scratch":
        angle = float(rng.uniform(0, 180))
        length = int(rng.integers(40, 90))
        x2 = int(cx + length * np.cos(np.deg2rad(angle)))
        y2 = int(cy + length * np.sin(np.deg2rad(angle)))
        cv2.line(img, (cx, cy), (x2, y2), (70, 70, 70), 2)
    elif label == "Spot":
        for _ in range(rng.integers(3, 8)):
            px, py = int(rng.integers(cx - 40, cx + 40)), int(rng.integers(cy - 40, cy + 40))
            cv2.circle(img, (px, py), int(rng.integers(2, 6)), (50, 50, 50), -1)
    elif label == "Contamination":
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            mask,
            (int(rng.integers(cx - 30, cx + 30)), int(rng.integers(cy - 30, cy + 30))),
            (int(rng.integers(15, 35)), int(rng.integers(10, 25))),
            int(rng.integers(0, 180)),
            0,
            360,
            255,
            -1,
        )
        img[mask > 0] = (img[mask > 0] * 0.5).astype(np.uint8)
    noise = rng.integers(0, 12, img.shape, dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def ensure_dataset() -> int:
    total = 0
    for label in LABELS:
        folder = DATASET_DIR / label.replace(" ", "_")
        folder.mkdir(parents=True, exist_ok=True)
        existing = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
        if len(existing) >= 10:
            total += len(existing)
            continue
        for i in range(SAMPLES_PER_CLASS):
            path = folder / f"synth_{i:03d}.png"
            if path.exists():
                continue
            image = _synthetic_image(label, seed=abs(hash(label)) % (2**31) + i)
            cv2.imwrite(str(path), image)
            total += 1
    return total


def load_dataset():
    import tensorflow as tf

    images, labels = [], []
    name_to_idx = {name: idx for idx, name in enumerate(LABELS)}
    folder_map = {
        "Crack": ["Crack", "crack"],
        "Scratch": ["Scratch", "scratch"],
        "Spot": ["Spot", "spot"],
        "Contamination": ["Contamination", "contamination"],
        "Normal wafer": ["Normal_wafer", "Normal", "normal", "Normal wafer"],
    }
    seen_paths: set[Path] = set()
    for label, folder_names in folder_map.items():
        idx = name_to_idx[label]
        for folder_name in folder_names:
            folder = DATASET_DIR / folder_name
            if not folder.exists():
                continue
            for path in list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")):
                resolved = path.resolve()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                img = cv2.imread(str(path))
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(idx)
    if not images:
        raise RuntimeError("No training images found under dataset/")
    x = np.array(images, dtype=np.float32) / 255.0
    y = tf.keras.utils.to_categorical(labels, num_classes=len(LABELS))
    return x, y


def build_model():
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(LABELS), activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("Preparing dataset...")
    count = ensure_dataset()
    print(f"Dataset images available: {count}")
    x, y = load_dataset()
    print(f"Training on {len(x)} images...")
    model = build_model()
    model.fit(
        x,
        y,
        epochs=12,
        batch_size=16,
        validation_split=0.2,
        shuffle=True,
        verbose=1,
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
