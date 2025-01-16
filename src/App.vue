<template>
  <div class="container">
    <h2>QuickDraw Classifier (Single Output)</h2>
    <p>Draw a doodle in the canvas and see the CNN's prediction.</p>

    <div class="main-layout">
      <!-- LEFT COLUMN: Canvas + Clear Button -->
      <div class="canvas-column">
        <canvas
          ref="drawingCanvas"
          :width="canvasWidth"
          :height="canvasHeight"
          @mousedown="startDrawing"
          @mousemove="draw"
          @mouseup="stopDrawing"
          @mouseleave="stopDrawing"
          @touchstart="handleTouchStart"
          @touchmove="handleTouchMove"
          @touchend="stopDrawing"
          style="border: 2px solid #444; cursor: crosshair"
        ></canvas>

        <div class="buttons">
          <button @click="clearCanvas">Clear</button>
        </div>
      </div>

      <!-- RIGHT COLUMN: Probability Gauges -->
      <div class="pipeline-column">
        <!-- Probability Bars for top 5 predictions -->
        <div class="prob-gauges">
          <h3>Top Predictions</h3>
          <div
            v-for="(item, index) in topK(probabilities, 5)"
            :key="index"
            class="prob-item"
          >
            <span class="label">{{
              labels[item.index] ? labels[item.index] : "Unknown"
            }}</span>
            <span class="bar-outer"
              ><span
                class="bar-fill"
                :style="{ width: (item.prob * 100).toFixed(1) + '%' }"
              ></span
            ></span>
            <span class="prob-val">{{
              isNaN(item.prob * 100)
                ? "0.0%"
                : (item.prob * 100).toFixed(1) + "%"
            }}</span>
          </div>
        </div>
      </div>
    </div>
    <!-- end .main-layout -->

    <!-- BLOG / DOCUMENTATION SECTION -->
    <div class="documentation-section">
      <h2>How This QuickDraw Classifier Was Built</h2>

      <p>
        This project showcases a custom deep learning pipeline for recognizing
        doodles drawn on a canvas. Below is an overview of how the model was
        trained, followed by a closer look at the code structure.
      </p>

      <h3>1. Data Acquisition &amp; Preprocessing</h3>
      <p>
        I used the <strong>QuickDraw dataset</strong> from Google, which
        contains tens of millions of drawn sketches across numerous categories.
        Each category (e.g., <em>circle</em>, <em>chair</em>, <em>cat</em>) is
        stored in separate <code>.npy</code> files under
        <code>./quickdraw_data/numpy_bitmap</code>.
      </p>
      <p>
        Because this dataset can be very large, I employ a
        <strong>custom chunked generator</strong> to:
      </p>
      <ul>
        <li>Load only a random subset of each class per epoch</li>
        <li>
          Maintain class balance (each class yields the same number of samples)
        </li>
        <li>Yield batches to the model without exceeding memory limits</li>
      </ul>

      <h3>2. Model Architecture</h3>
      <p>
        The model is a custom <strong>CNN for 28x28 grayscale images</strong>.
        It includes several convolutional blocks to extract features, followed
        by <code>GlobalAveragePooling2D</code> and a final dense layer of size
        <code>num_classes</code>. Some key highlights:
      </p>
      <ul>
        <li>
          <strong>Convolution + BatchNorm + ReLU</strong> layers to process
          sketches effectively
        </li>
        <li>
          <strong>Pooling layers</strong> to downsample while preserving
          essential features
        </li>
        <li><strong>Dropout</strong> to reduce overfitting</li>
      </ul>

      <h3>3. Training Workflow</h3>
      <p>
        Here is the high-level training code you wrote (showing your advanced
        mastery of custom data loading in Keras and TFJS export):
      </p>
      <pre>
<code>{{ trainingCodeSnippet }}</code>
      </pre>
      <p>
        After building the model with <code>build_model()</code>, I compile it
        using <code>Adam</code> and <code>SparseCategoricalCrossentropy</code>.
        I then fit the model on the <strong>training dataset</strong> for a
        specified number of epochs and evaluate on a
        <strong>test dataset</strong>
        to measure accuracy. Finally, I export the model in
        <strong>TensorFlow.js format</strong>, so it can be used directly in a
        browser environment (as this Vue component demonstrates).
      </p>

      <h3>4. Inference in the Browser</h3>
      <p>
        Once the model is saved in <code>./frontend/public/tfjs_model</code>,
        this Vue app loads <code>model.json</code> and uses it to make
        predictions in real-time. When you draw on the canvas:
      </p>
      <ul>
        <li>
          The drawing is downscaled from <code>280x280</code> to
          <code>28x28</code>
        </li>
        <li>Extract pixel intensities and invert them</li>
        <li>
          The <code>tfjs</code> model outputs logits, which is converted to
          probabilities
        </li>
        <li>The top predictions (by probability) are displayed to the user</li>
      </ul>
    </div>
  </div>
</template>

<script>
import * as tf from "@tensorflow/tfjs";
import { ref, onMounted, computed } from "vue";

export default {
  setup() {
    // --------------------------------------------------
    // STATE
    // --------------------------------------------------
    const canvasWidth = 280;
    const canvasHeight = 280;
    const drawingCanvas = ref(null);
    let ctx = null;
    let drawing = false;
    const canvasIsTouched = ref(false);

    // The model (single-output or multi-output, but we'll only use the first output)
    let model = null;

    // For storing final probabilities, predicted label
    const labels = [
      "circle",
      "square",
      "triangle",
      "star",
      "line",
      "cup",
      "clock",
      "chair",
      "book",
      "laptop",
      "cell phone",
      "key",
      "umbrella",
      "car",
      "cat",
      "dog",
      "bird",
      "fish",
      "tree",
      "flower",
      "sun",
      "cloud",
      "eye",
      "hand",
      "face",
      "smiley face",
      "scissors",
      "pencil",
      "hammer",
      "guitar",
      "bicycle",
      "airplane",
      "sailboat",
      "apple",
      "banana",
      "pizza",
    ];

    const probabilities = ref([]);
    const predictedLabel = ref("");
    const trainingCodeSnippet = computed(() => {
      return `
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import tensorflowjs as tfjs

########################################
# MODEL DEFINITION
########################################
def build_model(num_classes):
    inputs = layers.Input(shape=(28, 28, 1))

    # Block 1
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 28 -> 14

    # Block 2
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 14 -> 7

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes)(x)
    return Model(inputs, outputs)

########################################
# CHUNKED GENERATOR
########################################
def chunked_generator(
    categories,
    data_folder,
    samples_per_class_per_chunk=1000,
    batch_size=256,
    is_training=True,
    train_split=0.8,
    samples_per_class=None,
):
    # ...
    # see the generator logic for balanced, chunked data
    # ...

########################################
# GET TF.DATA DATASET
########################################
def get_chunked_dataset(...):
    # ...

########################################
# TRAINING ENTRY POINT
########################################
def main():
    DATA_FOLDER = "./quickdraw_data/numpy_bitmap"
    CATEGORIES = [ ... ]  # 37 classes

    BATCH_SIZE = 64
    EPOCHS = 20
    SAMPLES_PER_CLASS_PER_CHUNK = 1000
    STEPS_PER_EPOCH = (SAMPLES_PER_CLASS_PER_CHUNK * len(CATEGORIES)) // BATCH_SIZE

    # Build datasets
    train_ds = get_chunked_dataset(...)
    test_ds = get_chunked_dataset(...)

    # Build model
    model = build_model(num_classes=len(CATEGORIES))
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
    )

    test_loss, test_acc = model.evaluate(test_ds, steps=STEPS_PER_EPOCH)
    print(f"Test accuracy: {test_acc:.2%}")

    # Save model for TFJS
    tfjs.converters.save_keras_model(model, "./frontend/public/tfjs_model")

if __name__ == "__main__":
    main()
      `;
    });

    // --------------------------------------------------
    // onMounted: Load the model & init canvas
    // --------------------------------------------------
    onMounted(async () => {
      // Prepare canvas for drawing
      if (drawingCanvas.value) {
        ctx = drawingCanvas.value.getContext("2d");
        // Fill background white
        ctx.fillStyle = "#fff";
        ctx.fillRect(0, 0, canvasWidth, canvasHeight);
      }

      try {
        // Load the model (check your path)
        model = await tf.loadLayersModel("tfjs_model/model.json");
        console.log("Model loaded:", model);
      } catch (error) {
        console.error("Error loading model:", error);
      }
    });

    // --------------------------------------------------
    // Drawing Logic (Mouse & Touch)
    // --------------------------------------------------
    function startDrawing(e) {
      drawing = true;
      canvasIsTouched.value = true;
      if (!ctx) return;
      ctx.beginPath();
      ctx.moveTo(e.offsetX, e.offsetY);
    }

    function draw(e) {
      if (!drawing || !ctx) return;
      ctx.lineWidth = 10;
      ctx.lineCap = "round";
      ctx.strokeStyle = "#000";
      ctx.lineTo(e.offsetX, e.offsetY);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(e.offsetX, e.offsetY);
      runInference();
    }

    function stopDrawing() {
      drawing = false;
      if (ctx) {
        ctx.beginPath();
      }
      runInference();
    }

    function handleTouchStart(e) {
      e.preventDefault();
      const rect = drawingCanvas.value.getBoundingClientRect();
      const x = e.touches[0].clientX - rect.left;
      const y = e.touches[0].clientY - rect.top;
      drawing = true;
      canvasIsTouched.value = true;
      ctx.beginPath();
      ctx.moveTo(x, y);
      runInference();
    }

    function handleTouchMove(e) {
      if (!drawing || !ctx) return;
      e.preventDefault();
      const rect = drawingCanvas.value.getBoundingClientRect();
      const x = e.touches[0].clientX - rect.left;
      const y = e.touches[0].clientY - rect.top;
      ctx.lineWidth = 10;
      ctx.lineCap = "round";
      ctx.strokeStyle = "#000";
      ctx.lineTo(x, y);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x, y);
      runInference();
    }

    // --------------------------------------------------
    // Clear Canvas & Reset Predictions
    // --------------------------------------------------
    function clearCanvas() {
      if (!ctx) return;
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, canvasWidth, canvasHeight);
      probabilities.value = [];
      predictedLabel.value = "";
      canvasIsTouched.value = false;
    }

    // --------------------------------------------------
    // runInference: handle single (final) output
    // --------------------------------------------------
    async function runInference() {
      // If model or canvas isn't ready, skip
      if (!model || !drawingCanvas.value || !canvasIsTouched.value) return;

      // 1) Create a temp <canvas> to downscale to 28x28
      const smallCanvas = document.createElement("canvas");
      smallCanvas.width = 28;
      smallCanvas.height = 28;
      const smallCtx = smallCanvas.getContext("2d");
      // Draw the big canvas onto the small canvas
      smallCtx.drawImage(drawingCanvas.value, 0, 0, 28, 28);

      // 2) Extract pixel data => create Float32Array
      const imgData = smallCtx.getImageData(0, 0, 28, 28);
      const inputBuffer = new Float32Array(28 * 28);

      for (let i = 0; i < 28 * 28; i++) {
        const idx = i * 4;
        const grayscaleVal = imgData.data[idx]; // R channel
        // invert so that black lines become white lines
        inputBuffer[i] = (255 - grayscaleVal) / 255.0;
      }

      // 3) Construct a 4D tensor => [1, 28, 28, 1]
      const inputTensor = tf.tensor4d(inputBuffer, [1, 28, 28, 1]);

      // 4) model.predict => take the first or named 'logits'
      let logits;
      let results;
      try {
        results = model.predict(inputTensor);
      } catch (err) {
        console.error("Prediction error:", err);
        tf.dispose(inputTensor);
        return;
      }

      if (Array.isArray(results)) {
        // If multi-output, the first is typically the 'logits'
        logits = results[0];
      } else if (
        typeof results === "object" &&
        (results["Identity"] || results["output_0"])
      ) {
        logits = results["Identity"] || results["output_0"];
      } else {
        // Possibly a single-tensor result
        logits = results;
      }

      if (!logits) {
        console.warn("No logits found in model output.");
        tf.dispose([inputTensor, results]);
        return;
      }

      // 5) Convert logits -> probabilities
      const logitsData = await logits.data();
      const exps = logitsData.map((x) => Math.exp(x));
      let sumExps = exps.reduce((a, b) => a + b, 0);
      if (!isFinite(sumExps) || sumExps === 0) sumExps = 1;
      const probs = exps.map((x) => x / sumExps);

      // 6) Update reactive state
      probabilities.value = probs;
      const maxProb = Math.max(...probs);
      const maxIndex = probs.indexOf(maxProb);
      predictedLabel.value = labels[maxIndex] || `Class ${maxIndex}`;

      // 7) Dispose used tensors
      tf.dispose([inputTensor, logits, results]);
    }

    // --------------------------------------------------
    // Utility: top K from probabilities
    // --------------------------------------------------
    function topK(probArray, k = 3) {
      if (!probArray || !probArray.length) return [];
      const arr = Array.from(probArray).map((prob, i) => ({ prob, index: i }));
      arr.sort((a, b) => b.prob - a.prob);
      return arr.slice(0, k);
    }

    // Return references / methods for template
    return {
      canvasWidth,
      canvasHeight,
      drawingCanvas,

      probabilities,
      predictedLabel,
      labels,
      trainingCodeSnippet,

      startDrawing,
      draw,
      stopDrawing,
      handleTouchStart,
      handleTouchMove,
      clearCanvas,
      topK,
    };
  },
};
</script>

<style scoped>
.container {
  width: 100%;
  margin: 1rem auto;
  text-align: center;
  font-family: sans-serif;
}

.main-layout {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  justify-content: center;
  margin-top: 1rem;
}

.canvas-column {
  min-width: 300px;
  text-align: center;
}

.pipeline-column {
  min-width: 200px;
  text-align: left;
}

.buttons {
  margin-top: 1rem;
}

/* Probability bar styling */
.prob-gauges {
  margin-top: 1rem;
}
.prob-item {
  display: flex;
  align-items: center;
  margin-bottom: 4px;
}
.label {
  width: 80px;
  text-align: right;
  margin-right: 8px;
}
.bar-outer {
  flex: 1;
  height: 10px;
  background: #eee;
  margin-right: 8px;
  border-radius: 3px;
  overflow: hidden;
}
.bar-fill {
  display: block;
  height: 100%;
  background: #2196f3;
  transition: width 0.2s ease;
}
.prob-val {
  width: 50px;
  text-align: right;
}

/* Documentation / Blog Section */
.documentation-section {
  margin-top: 2rem;
  text-align: left;
  max-width: 800px;
  margin-left: auto;
  margin-right: auto;
}
.documentation-section pre {
  background: #f5f5f5;
  padding: 10px;
  white-space: pre-wrap;
  border-radius: 4px;
  overflow-x: auto;
}
</style>
