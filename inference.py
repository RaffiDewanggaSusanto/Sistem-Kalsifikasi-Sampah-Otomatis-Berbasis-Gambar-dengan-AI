from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from config import CLASS_NAMES_PATH, IMG_SIZE, KERAS_MODEL_PATH
from data import load_class_names
from model import ChannelAttention, FocalLoss


def load_model(model_path=KERAS_MODEL_PATH):
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"ChannelAttention": ChannelAttention, "FocalLoss": FocalLoss},
    )


def preprocess_image(image_path: str | Path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def predict_image(image_path, model=None, top_k=3):
    if model is None:
        model = load_model()

    class_names = load_class_names(CLASS_NAMES_PATH)
    inputs = preprocess_image(image_path)
    probabilities = model.predict(inputs, verbose=0)[0]
    top_indices = probabilities.argsort()[-top_k:][::-1]

    predictions = [
        {"label": class_names[index], "confidence": float(probabilities[index])}
        for index in top_indices
    ]
    return predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference model klasifikasi sampah.")
    parser.add_argument("image", help="Path gambar yang ingin diprediksi")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    for item in predict_image(args.image, top_k=args.top_k):
        print(f"{item['label']}: {item['confidence']:.4f}")
