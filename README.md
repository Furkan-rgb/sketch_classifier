# QuickDraw Sketch Classifier

A browser-based doodle classifier powered by a custom 36-class convolutional
neural network trained on Google’s [Quick, Draw! dataset](https://quickdraw.withgoogle.com/data).
Draw with a mouse or touchscreen and watch the model rank its predictions in
real time.

Inference runs entirely in the browser with TensorFlow.js—no backend, account,
or image upload is required.

**[Try the live demo →](https://furkan-rgb.github.io/sketch_classifier/)**

## Highlights

- Custom CNN for `28 × 28` grayscale sketches
- Real-time, client-side inference with TensorFlow.js
- Top-five predictions with confidence scores
- Mouse, stylus, and touchscreen drawing support
- Undo, clear, and randomized drawing prompts
- All 36 supported categories documented in the interface
- Static deployment through GitHub Pages

## How it works

1. Pointer strokes are recorded on an `800 × 800` backing canvas.
2. The complete drawing area is resized to `28 × 28` pixels.
3. Pixels are inverted and normalized so ink is `1` and the background is `0`.
4. The resulting tensor is passed to the TensorFlow.js model.
5. Logits are converted to probabilities and the five most likely classes are
   displayed.

The model and its ordered `class_names.json` file are loaded together. Runtime
and build-time checks ensure that model outputs cannot silently drift from the
labels shown in the interface.

## Model architecture

The currently bundled model uses:

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

`circle`, `square`, `triangle`, `star`, `line`, `cup`, `clock`, `chair`,
`book`, `laptop`, `cell phone`, `key`, `umbrella`, `car`, `cat`, `dog`,
`bird`, `fish`, `tree`, `flower`, `sun`, `cloud`, `eye`, `hand`, `face`,
`smiley face`, `scissors`, `pencil`, `hammer`, `guitar`, `bicycle`,
`airplane`, `sailboat`, `apple`, `banana`, and `pizza`.

</details>

## Model and data

The model was trained on `28 × 28` grayscale bitmap samples from Quick, Draw!.
The development workspace keeps the source dataset and Python training pipeline
outside this frontend repository; this repository contains the browser app and
the exported TensorFlow.js artifacts.

The classifier only knows its 36 training categories. An unfamiliar drawing is
still assigned to the closest known class, so confidence should not be treated
as proof that the drawing belongs to the model’s vocabulary.

## Tech stack

- **Vue 3** for the interface
- **TensorFlow.js** for client-side inference
- **TensorFlow/Keras** for model training and export
- **HTML Canvas** for drawing input
- **Vite** for development and builds
- **GitHub Pages** for deployment

## Project structure

```text
.
├── public/
│   └── tfjs_model/
│       ├── class_names.json
│       ├── model.json
│       └── group1-shard1of1.bin
├── scripts/
│   └── validate-model-contract.mjs
├── src/
│   ├── components/
│   │   ├── CaseStudy.vue
│   │   ├── DrawingCanvas.vue
│   │   ├── KnownCategories.vue
│   │   └── PredictionPanel.vue
│   ├── composables/
│   │   └── useSketchClassifier.js
│   ├── data/
│   │   └── categories.js
│   ├── App.vue
│   ├── main.js
│   └── style.css
├── index.html
├── package.json
└── vite.config.js
```

## Run locally

Requirements: Node.js and npm.

```sh
git clone https://github.com/Furkan-rgb/sketch_classifier.git
cd sketch_classifier
npm ci
npm run dev
```

Open the local URL printed by Vite and start drawing. No API key or backend is
required.

Create and preview a production build with:

```sh
npm run build
npm run preview
```

`npm run build` validates the model input shape, output count, ordered label
contract, and UI category coverage before building the application.

## Deployment

The Vite base path is configured for `/sketch_classifier/`. With GitHub
credentials configured, publish `dist/` to the `gh-pages` branch with:

```sh
npm run deploy
```
