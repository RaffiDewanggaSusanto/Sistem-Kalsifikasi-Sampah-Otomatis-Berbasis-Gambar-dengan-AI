from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "dataset"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"

ARTIFACT_DIR = ROOT_DIR / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "model"
TFLITE_DIR = ARTIFACT_DIR / "tflite"
LOG_DIR = ARTIFACT_DIR / "logs"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
SEED = 42

CLASS_NAMES_PATH = ARTIFACT_DIR / "class_names.txt"
KERAS_MODEL_PATH = MODEL_DIR / "waste_classifier.keras"
SAVED_MODEL_PATH = MODEL_DIR / "saved_model"
TFLITE_MODEL_PATH = TFLITE_DIR / "waste_classifier.tflite"
