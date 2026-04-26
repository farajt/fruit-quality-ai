# ============================================================
# predictor.py — Inference pipeline
# ============================================================

import os, json, time
import numpy as np
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

# ── Register custom CBAM layers ───────────────────────────
@keras.saving.register_keras_serializable(package="FruitAI")
class ChannelAvgPool(keras.layers.Layer):
    def call(self, x):
        return keras.ops.mean(x, axis=-1, keepdims=True)
    def get_config(self):
        return super().get_config()

@keras.saving.register_keras_serializable(package="FruitAI")
class ChannelMaxPool(keras.layers.Layer):
    def call(self, x):
        return keras.ops.max(x, axis=-1, keepdims=True)
    def get_config(self):
        return super().get_config()

# ── Paths ─────────────────────────────────────────────────
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "..", "model")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "model_weights.weights.h5")
CONFIG_PATH  = os.path.join(MODEL_DIR, "project_config.json")

# ── Load config ───────────────────────────────────────────
with open(CONFIG_PATH) as f:
    _cfg = json.load(f)

CLASSES      = _cfg["CLASSES"]
CLASS_TO_IDX = _cfg["CLASS_TO_IDX"]
IDX_TO_CLASS = {int(k): v for k, v in _cfg["IDX_TO_CLASS"].items()}
CONFIG       = _cfg["CONFIG"]
GRADE_CONFIG = _cfg["GRADE_CONFIG"]
SHELF_LIFE   = _cfg["SHELF_LIFE"]
IMG_SIZE     = CONFIG["image_size"]
NUM_CLASSES  = len(CLASSES)

CONFIDENCE_GATES = {"apple": 0.70, "banana": 0.70, "orange": 0.80}

# ── CBAM ─────────────────────────────────────────────────
def channel_attention(x, ratio=16, prefix="ca"):
    c   = x.shape[-1]
    r   = max(1, c // ratio)
    d1  = keras.layers.Dense(r, activation="relu",
                              use_bias=False, name=f"{prefix}_d1")
    d2  = keras.layers.Dense(c, activation=None,
                              use_bias=False, name=f"{prefix}_d2")
    avg     = keras.layers.GlobalAveragePooling2D(
                  name=f"{prefix}_avg")(x)
    mx      = keras.layers.GlobalMaxPooling2D(
                  name=f"{prefix}_max")(x)
    scale   = keras.layers.Activation(
                  "sigmoid", name=f"{prefix}_sig")(
              keras.layers.Add(name=f"{prefix}_add")(
                  [d2(d1(avg)), d2(d1(mx))]))
    scale   = keras.layers.Reshape(
                  (1, 1, c), name=f"{prefix}_reshape")(scale)
    return keras.layers.Multiply(
               name=f"{prefix}_mul")([x, scale])

def spatial_attention(x, kernel_size=7, prefix="sa"):
    avg   = ChannelAvgPool(name=f"{prefix}_avg")(x)
    mx    = ChannelMaxPool(name=f"{prefix}_max")(x)
    cat   = keras.layers.Concatenate(
                axis=-1, name=f"{prefix}_cat")([avg, mx])
    scale = keras.layers.Conv2D(
                1, kernel_size, padding="same",
                activation="sigmoid", use_bias=False,
                name=f"{prefix}_conv")(cat)
    return keras.layers.Multiply(
               name=f"{prefix}_mul")([x, scale])

def cbam_block(x, ratio=16, kernel_size=7, prefix="cbam"):
    x = channel_attention(x, ratio,       prefix=f"{prefix}_ch")
    x = spatial_attention(x, kernel_size, prefix=f"{prefix}_sp")
    return x

# ── Build model ───────────────────────────────────────────
def build_model():
    inputs   = keras.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
    backbone = EfficientNetB0(
        include_top=False, weights=None,
        input_tensor=inputs)
    backbone.trainable = True
    x = backbone.output
    x = cbam_block(x, prefix="cbam")
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dense(256, activation="relu",
                            name="head_dense")(x)
    x = keras.layers.BatchNormalization(name="head_bn")(x)
    x = keras.layers.Dropout(CONFIG["dropout_rate"],
                              name="head_dropout")(x)
    outputs = keras.layers.Dense(
        NUM_CLASSES, activation="softmax",
        name="predictions")(x)
    m = keras.Model(inputs=inputs, outputs=outputs,
                    name="FruitQualityModel")
    m.compile(optimizer=Adam(learning_rate=1e-5),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
    return m

print("Building model architecture...")
_model = build_model()
print("Loading trained weights...")
_model.load_weights(WEIGHTS_PATH)
print("✓ Model ready")

def get_model():
    return _model

# ── Preprocessing ─────────────────────────────────────────
def preprocess_pil(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB").resize(
        (IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return np.expand_dims(
        np.array(img, dtype=np.float32), axis=0)

# ── Grad-CAM — uses PIL only, no cv2 ─────────────────────
def make_gradcam(img_array: np.ndarray) -> np.ndarray:
    """Returns RGB uint8 numpy array with heatmap overlaid."""
    model = get_model()
    last_conv = next(
        (l.name for l in reversed(model.layers)
         if isinstance(l, keras.layers.Conv2D)), None)
    if not last_conv:
        return np.clip(img_array[0], 0, 255).astype(np.uint8)

    try:
        grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv).output,
                     model.output])
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(
                img_array, training=False)
            pred_idx = tf.argmax(preds[0])
            loss     = preds[:, pred_idx]
        grads   = tape.gradient(loss, conv_out)
        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(tf.maximum(heatmap, 0)).numpy()
        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val

        # Original image as uint8
        orig = np.clip(img_array[0], 0, 255).astype(np.uint8)
        h, w = orig.shape[:2]

        # Resize heatmap using PIL (no cv2 dependency)
        hmap_pil    = Image.fromarray(
            (heatmap * 255).astype(np.uint8)).resize(
            (w, h), Image.BILINEAR)
        hmap_arr    = np.array(hmap_pil, dtype=np.float32) / 255.0

        # Apply colormap manually (jet: blue→green→red)
        r = np.clip(1.5 - np.abs(hmap_arr * 4 - 3), 0, 1)
        g = np.clip(1.5 - np.abs(hmap_arr * 4 - 2), 0, 1)
        b = np.clip(1.5 - np.abs(hmap_arr * 4 - 1), 0, 1)
        colored = np.stack([r, g, b], axis=-1)
        colored = (colored * 255).astype(np.uint8)

        # Blend
        alpha   = 0.4
        blended = (orig * (1 - alpha) +
                   colored * alpha).astype(np.uint8)
        return blended

    except Exception:
        return np.clip(img_array[0], 0, 255).astype(np.uint8)

# ── Main predict ──────────────────────────────────────────
def predict(pil_image: Image.Image) -> dict:
    model      = get_model()
    img_array  = preprocess_pil(pil_image)
    probs      = model(img_array, training=False).numpy()[0]
    pred_idx   = int(np.argmax(probs))
    pred_cls   = IDX_TO_CLASS[pred_idx]
    confidence = float(probs[pred_idx])
    fruit_type = pred_cls.split("_")[0]
    gate       = CONFIDENCE_GATES.get(fruit_type, 0.70)

    if confidence < gate:
        return {
            "status":     "uncertain",
            "confidence": round(confidence, 4),
            "message":    (f"Low confidence ({confidence:.1%}). "
                           "Use a clearer image with plain "
                           "background and good lighting."),
            "top3": [{"class": IDX_TO_CLASS[i],
                       "prob":  round(float(probs[i]), 4)}
                      for i in np.argsort(probs)[::-1][:3]],
        }

    fruit, condition = pred_cls.split("_")
    fresh_idx        = CLASS_TO_IDX[f"{fruit}_fresh"]
    freshness_score  = float(probs[fresh_idx])

    grade = "F"
    for g, info in GRADE_CONFIG.items():
        if freshness_score >= info["min"]:
            grade = g
            break

    sorted_p = sorted(probs)
    top2_gap = float(sorted_p[-1] - sorted_p[-2])
    gradcam  = make_gradcam(img_array)

    return {
        "status":           "success",
        "fruit":            fruit,
        "condition":        condition,
        "predicted_class":  pred_cls,
        "confidence":       round(confidence, 4),
        "freshness_score":  round(freshness_score, 4),
        "grade":            grade,
        "grade_label":      GRADE_CONFIG[grade]["label"],
        "risk_level":       GRADE_CONFIG[grade]["risk"],
        "recommendation":   GRADE_CONFIG[grade]["recommendation"],
        "shelf_life":       SHELF_LIFE[fruit][grade],
        "warning":          (f"Low margin ({top2_gap:.0%} gap). "
                             "Consider retaking photo."
                             if top2_gap < 0.15 else None),
        "top3": [{"class": IDX_TO_CLASS[i],
                   "prob":  round(float(probs[i]), 4)}
                  for i in np.argsort(probs)[::-1][:3]],
        "gradcam": gradcam,
    }