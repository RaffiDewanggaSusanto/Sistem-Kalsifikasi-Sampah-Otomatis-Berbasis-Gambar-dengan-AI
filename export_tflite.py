import tensorflow as tf

from config import KERAS_MODEL_PATH, TFLITE_DIR, TFLITE_MODEL_PATH
from model import ChannelAttention, FocalLoss


def main():
    TFLITE_DIR.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(
        KERAS_MODEL_PATH,
        custom_objects={"ChannelAttention": ChannelAttention, "FocalLoss": FocalLoss},
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    TFLITE_MODEL_PATH.write_bytes(tflite_model)
    print(f"TFLite model tersimpan di: {TFLITE_MODEL_PATH}")


if __name__ == "__main__":
    main()
