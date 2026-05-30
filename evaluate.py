import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from config import CLASS_NAMES_PATH, KERAS_MODEL_PATH, TEST_DIR
from data import load_class_names, load_split, prepare_dataset
from model import ChannelAttention, FocalLoss


def main():
    class_names = load_class_names(CLASS_NAMES_PATH)
    test_ds = prepare_dataset(load_split(TEST_DIR, shuffle=False))
    model = tf.keras.models.load_model(
        KERAS_MODEL_PATH,
        custom_objects={"ChannelAttention": ChannelAttention, "FocalLoss": FocalLoss},
    )

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        probabilities = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(probabilities, axis=1))

    print("Classification Report")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
