"""Grad-CAM - TensorFlow/Keras (2_train/tensorflow/gradcam.py).

Theory (Grad-CAM, Selvaraju et al. 2017):
For each attribute head, take the gradients of the predicted class score with
respect to the last conv feature map. Average them per channel to get channel
importance weights; the ReLU'd weighted sum of the feature channels is a
heatmap of WHERE the model looked for that attribute. shared/visualize.py
turns the heatmap into a box, so no box labels are needed for training.
"""

import cv2
import numpy as np
import tensorflow as tf

from shared.visualize import bbox_from_heat


def _grad_model(model):
    """Model that also returns the spatial conv feature map (7x7x1280).
    MobileNetV2's last conv layer is named 'out_relu' even when pooled."""
    conv_out = model.get_layer("out_relu").output
    out_layers = [model.get_layer(n).output for n in model.output_names]
    return tf.keras.Model(model.inputs, [conv_out, *out_layers])


def gradcam_boxes(model, x):
    """(1,H,W,3) input -> {head key: (x1,y1,x2,y2) px box or None} in
    INPUT-PIXEL coordinates of x."""
    grad_model = _grad_model(model)
    h, w = int(x.shape[1]), int(x.shape[2])
    # persistent tape: one forward, one gradient per head
    with tf.GradientTape(persistent=True) as tape:
        res = grad_model(x, training=False)
        fmap, outs = res[0], res[1:]  # fmap (1,7,7,1280), one tensor per head
        # score = probability of the predicted class, per head
        scores = [outs[i][0, tf.argmax(outs[i][0])] for i in range(len(outs))]
    fmap_arr = fmap.numpy()[0]  # (7,7,1280)
    result = {}
    for i, name in enumerate(model.output_names):
        grads = tape.gradient(scores[i], fmap).numpy()[0]  # (7,7,1280)
        wch = grads.mean(axis=(0, 1))                      # channel importance
        cam = np.maximum((fmap_arr * wch[None, None, :]).sum(-1), 0.0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        result[name] = bbox_from_heat(cv2.resize(cam, (w, h)))
    del tape
    return result
