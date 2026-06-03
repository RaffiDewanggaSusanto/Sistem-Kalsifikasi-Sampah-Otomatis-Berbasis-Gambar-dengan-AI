from pathlib import Path

import tensorflow as tf

from config import BATCH_SIZE, IMG_SIZE, SEED


def load_split(directory: Path, shuffle: bool):
    if not directory.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {directory}")

    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )


def prepare_dataset(dataset, cache=False):
    if cache:
        dataset = dataset.cache()
    return dataset.prefetch(tf.data.AUTOTUNE)


def save_class_names(class_names, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(class_names), encoding="utf-8")


def load_class_names(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
