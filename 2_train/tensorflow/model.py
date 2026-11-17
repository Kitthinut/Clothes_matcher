"""Multi-head garment classifier - TensorFlow/Keras (2_train/tensorflow/model.py).

Theory:
- Transfer learning with Keras Applications (MobileNetV2, ImageNet weights).
- Functional API: one shared backbone tensor feeds several Dense outputs —
  Keras trains them jointly with one .fit() call.
- Single-label heads use softmax; multi-label heads (e.g. pattern) use
  sigmoid + binary cross-entropy so an item can be "dotted AND striped".
- MobileNetV2 is used instead of ResNet18 because Keras has no ResNet18;
  both are ImageNet-pretrained, MobileNetV2 is the lighter twin.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(spec, pretrained=True, input_size=(288, 384)):
    """spec -> Keras model with one output per attribute.
    input_size: (W, H) — must match shared/preprocessing.INPUT_W/H."""
    weights = "imagenet" if pretrained else None
    base = tf.keras.applications.MobileNetV2(
        input_shape=(input_size[1], input_size[0], 3),  # Keras wants (H, W)
        include_top=False, pooling="avg", weights=weights)
    base.trainable = False  # first epochs: train heads only (frozen backbone)

    x = layers.Dropout(0.3)(base.output)          # regularisation vs overfitting
    outputs = {}
    for k, cat in spec.items():
        if cat.get("mode") == "multi":
            outputs[k] = layers.Dense(len(cat["options"]), activation="sigmoid",
                                      name=k)(x)
        else:
            outputs[k] = layers.Dense(len(cat["options"]), activation="softmax",
                                      name=k)(x)
    return models.Model(base.input, outputs, name="garment_multihead")


def unfreeze_backbone(model):
    """Phase 2 of fine-tuning: unfreeze and continue with a smaller LR.
    (Keras 3 flattens the MobileNetV2 layers, so unfreeze by name pattern.)"""
    for layer in model.layers:
        if layer.name != "input_layer":  # leave the input op alone
            layer.trainable = True


def compile_model(model, spec, lr=1e-3):
    """Per-head loss: sparse CE (single) / BCE (multi); Keras averages them."""
    losses, metrics = {}, {}
    for k, cat in spec.items():
        if cat.get("mode") == "multi":
            losses[k] = tf.keras.losses.BinaryCrossentropy()
            metrics[k] = tf.keras.metrics.BinaryAccuracy(name=f"{k}_acc")
        else:
            losses[k] = tf.keras.losses.SparseCategoricalCrossentropy()
            metrics[k] = tf.keras.metrics.SparseCategoricalAccuracy(name=f"{k}_acc")
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4),
        loss=losses, weighted_metrics=metrics,
    )


def predict_dict(model, img_batch):
    """model.predict -> {key: probs_array} with probabilities per option."""
    outs = model.predict(img_batch, verbose=0)
    return {name: outs[i] if isinstance(outs, list) else outs[name]
            for i, name in enumerate(model.output_names)}
