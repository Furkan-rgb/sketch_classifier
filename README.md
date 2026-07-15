# QuickDraw Sketch Classifier

A browser-based doodle classifier powered by a custom 36-class convolutional neural network trained on Google’s [Quick, Draw! dataset](https://quickdraw.withgoogle.com/data). Draw with a mouse or touchscreen and see the model’s top predictions update in real time.

Inference runs entirely in the browser with TensorFlow.js—no backend or image upload is required.

**[Try the live demo →](https://furkan-rgb.github.io/sketch_classifier/)**

## Highlights

- Custom CNN for `28 × 28` grayscale sketches
- Real-time, client-side inference with TensorFlow.js
- Top-five predictions with confidence scores
- Mouse and touchscreen drawing support
- A pretrained model bundled with the application
- Static deployment through GitHub Pages

## How it works

1. The user draws on a `280 × 280` HTML canvas.
2. The drawing is resized to `28 × 28` pixels.
3. Pixel values are converted to grayscale, inverted, and normalized to `[0, 1]`.
4. The resulting tensor is passed to the TensorFlow.js model.
5. The model’s logits are converted to probabilities and the five most likely classes are displayed.

## Model architecture

```text
28 × 28 × 1 input
  → Conv2D(64) × 2 → BatchNorm → MaxPool
  → Conv2D(128) × 2 → BatchNorm → MaxPool
  → Global Average Pooling
  → Dropout(0.3)
  → Dense(36 logits)
```

## Supported classes

<details>
<summary>View all 36 categories</summary>

`circle`, `square`, `triangle`, `star`, `line`, `cup`, `clock`, `chair`, `book`, `laptop`, `cell phone`, `key`, `umbrella`, `car`, `cat`, `dog`, `bird`, `fish`, `tree`, `flower`, `sun`, `cloud`, `eye`, `hand`, `face`, `smiley face`, `scissors`, `pencil`, `hammer`, `guitar`, `bicycle`, `airplane`, `sailboat`, `apple`, `banana`, and `pizza`.

</details>

## Model and data

The model was trained on `28 × 28` grayscale bitmap samples from Quick, Draw!. The documented training approach uses balanced, chunked sampling across categories so the large per-class NumPy files do not need to be loaded into memory at once.

This repository contains the exported TensorFlow.js model and the browser inference application. The source dataset and a standalone, executable Python training pipeline are not included, and the repository does not publish an evaluation score.

## Tech stack

- **Vue 3** for the interface
- **TensorFlow.js** for client-side inference
- **TensorFlow/Keras** for the exported model
- **HTML Canvas** for drawing input
- **Vite** for development and builds
- **GitHub Pages** for deployment

## Project structure

```text
.
├── public/
│   └── tfjs_model/
│       ├── model.json
│       └── group1-shard1of1.bin
├── src/
│   ├── App.vue          # Drawing UI, preprocessing, and inference
│   ├── main.js          # Vue entry point
│   └── style.css        # Global styles
├── index.html
├── package.json
└── vite.config.js       # Vite and GitHub Pages configuration
```

## Run locally

Requirements: Node.js and npm.

```bash
git clone https://github.com/Furkan-rgb/sketch_classifier.git
cd sketch_classifier
npm ci
npm run dev
```

Open the local URL printed by Vite and start drawing. No API key, backend, or additional model download is required.

Create and preview a production build with:

```bash
npm run build
npm run preview
```

## Deployment

The Vite base path is configured for `/sketch_classifier/`. With GitHub credentials configured, build the app and publish `dist/` to the `gh-pages` branch with:

```bash
npm run deploy
```
