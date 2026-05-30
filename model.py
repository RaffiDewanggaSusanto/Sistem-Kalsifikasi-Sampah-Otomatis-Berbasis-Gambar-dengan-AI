import tensorflow as tf


class ChannelAttention(tf.keras.layers.Layer):
    """Lightweight custom attention layer for image feature maps."""

    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = int(input_shape[-1])
        hidden_units = max(channels // self.ratio, 1)
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.max_pool = tf.keras.layers.GlobalMaxPooling2D()
        self.shared_dense_1 = tf.keras.layers.Dense(hidden_units, activation="relu")
        self.shared_dense_2 = tf.keras.layers.Dense(channels)
        super().build(input_shape)

    def call(self, inputs):
        avg = self.shared_dense_2(self.shared_dense_1(self.avg_pool(inputs)))
        max_value = self.shared_dense_2(self.shared_dense_1(self.max_pool(inputs)))
        attention = tf.nn.sigmoid(avg + max_value)
        attention = tf.reshape(attention, [-1, 1, 1, tf.shape(inputs)[-1]])
        return inputs * attention

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config


class FocalLoss(tf.keras.losses.Loss):
    """Custom loss to help when waste classes are imbalanced."""

    def __init__(self, gamma=2.0, alpha=0.25, name="focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * tf.pow(1.0 - y_pred, self.gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=-1))

    def get_config(self):
        return {"gamma": self.gamma, "alpha": self.alpha, "name": self.name}


def build_waste_classifier(num_classes, image_size=(224, 224), train_base=False):
    inputs = tf.keras.Input(shape=(*image_size, 3), name="image")

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    x = data_augmentation(inputs)
    x = tf.keras.applications.efficientnet_v2.preprocess_input(x)

    base_model = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(*image_size, 3),
    )
    base_model.trainable = train_base

    x = base_model(x, training=False)
    x = ChannelAttention(name="channel_attention")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.Dropout(0.35, name="dropout")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(0.25, name="classifier_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_output")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="waste_classifier")
