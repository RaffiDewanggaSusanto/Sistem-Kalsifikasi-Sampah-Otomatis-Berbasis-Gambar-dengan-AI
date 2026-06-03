import datetime as dt

import tensorflow as tf

from config import (
    CLASS_NAMES_PATH,
    EPOCHS,
    KERAS_MODEL_PATH,
    LEARNING_RATE,
    LOG_DIR,
    MODEL_DIR,
    SAVED_MODEL_PATH,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
)
from data import load_split, prepare_dataset, save_class_names
from model import FocalLoss, build_waste_classifier


def train_step(model, images, labels, loss_fn, optimizer, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_accuracy.update_state(labels, predictions)
    return loss


def validation_step(model, images, labels, loss_fn, val_accuracy):
    predictions = model(images, training=False)
    loss = loss_fn(labels, predictions)
    val_accuracy.update_state(labels, predictions)
    return loss


def run_epoch(model, dataset, loss_fn, optimizer, metric, training):
    losses = []
    for images, labels in dataset:
        if training:
            loss = train_step(model, images, labels, loss_fn, optimizer, metric)
        else:
            loss = validation_step(model, images, labels, loss_fn, metric)
        losses.append(loss)

    return tf.reduce_mean(losses), metric.result()


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = load_split(TRAIN_DIR, shuffle=True)
    val_ds = load_split(VAL_DIR, shuffle=False)
    test_ds = load_split(TEST_DIR, shuffle=False) if TEST_DIR.exists() else None

    class_names = train_ds.class_names
    save_class_names(class_names, CLASS_NAMES_PATH)

    train_ds = prepare_dataset(train_ds)
    val_ds = prepare_dataset(val_ds)
    if test_ds is not None:
        test_ds = prepare_dataset(test_ds)

    model = build_waste_classifier(num_classes=len(class_names))
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = FocalLoss()

    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(str(LOG_DIR / run_id))
    best_val_accuracy = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
        val_accuracy = tf.keras.metrics.CategoricalAccuracy(name="val_accuracy")

        train_loss, train_acc = run_epoch(
            model, train_ds, loss_fn, optimizer, train_accuracy, training=True
        )
        val_loss, val_acc = run_epoch(
            model, val_ds, loss_fn, optimizer, val_accuracy, training=False
        )

        with summary_writer.as_default():
            tf.summary.scalar("loss/train", train_loss, step=epoch)
            tf.summary.scalar("loss/validation", val_loss, step=epoch)
            tf.summary.scalar("accuracy/train", train_acc, step=epoch)
            tf.summary.scalar("accuracy/validation", val_acc, step=epoch)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if float(val_acc) > best_val_accuracy:
            best_val_accuracy = float(val_acc)
            model.save(KERAS_MODEL_PATH)

    print(f"Model terbaik disimpan ke: {KERAS_MODEL_PATH}")
    model.export(SAVED_MODEL_PATH)
    print(f"SavedModel diekspor ke: {SAVED_MODEL_PATH}")

    if test_ds is not None:
        test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")
        test_loss, test_acc = run_epoch(
            model, test_ds, loss_fn, optimizer, test_accuracy, training=False
        )
        print(f"Test loss={test_loss:.4f} test_accuracy={test_acc:.4f}")


if __name__ == "__main__":
    main()
