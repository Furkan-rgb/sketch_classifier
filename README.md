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
- Deterministic, memory-efficient Python training pipeline
- Evaluation reports with top-3 accuracy, per-class recall, and confusion matrices
- Static deployment through GitHub Pages

## How inference works

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

The included training pipeline defines the next iteration as three feature
blocks (`48 → 96 → 192`), a 128-unit embedding, 35% dropout, and the same
36-class logits output. Training-only augmentation is not exported to the
browser model.

## Training pipeline

[`model/train.py`](model/train.py) is the single entry point for splitting,
loading, augmenting, training, evaluation, and TensorFlow.js export. It includes:

- Seeded, non-overlapping 70/15/15 train, validation, and test splits
- Balanced loading from memory-mapped NumPy files
- Restrained translation, rotation, zoom, dilation, and erosion
- Up to 30 epochs with early stopping and learning-rate reduction
- Top-1 and top-3 accuracy, macro/per-class recall, and confusion matrices
- Safe model export with an ordered `class_names.json` contract

The raw Quick, Draw! dataset is intentionally excluded from Git because the
complete local copy is approximately 59 GB. Recreate the complete data folder—
all 345 categories as both 28 × 28 NumPy bitmaps and simplified stroke NDJSON—
with:

```sh
python quickdraw_data/fetch.py
```

The downloader reads Google’s official category list, fetches files in parallel,
validates completed downloads, and skips valid files that already exist. Data is
stored in `quickdraw_data/numpy_bitmap/` and `quickdraw_data/strokes/` and
remains ignored by Git. Use `--format bitmaps` or `--format strokes` when only
one representation is needed.

Use Python 3.11 from the repository root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r model/requirements.txt
python model/train.py
```

Training reports are written to `model/training_artifacts/`. A successful run
exports the browser model directly to `public/tfjs_model/`.

## Supported classes

<details>
<summary>View all 36 categories</summary>

`circle`, `square`, `triangle`, `star`, `line`, `cup`, `clock`, `chair`,
`book`, `laptop`, `cell phone`, `key`, `umbrella`, `car`, `cat`, `dog`,
`bird`, `fish`, `tree`, `flower`, `sun`, `cloud`, `eye`, `hand`, `face`,
`smiley face`, `scissors`, `pencil`, `hammer`, `guitar`, `bicycle`,
`airplane`, `sailboat`, `apple`, `banana`, and `pizza`.

</details>

The classifier only knows these categories. An unfamiliar drawing is still
assigned to the closest known class, so confidence should not be treated as
proof that the drawing belongs to the model’s vocabulary.

## Tech stack

- **Vue 3** for the interface
- **TensorFlow.js** for client-side inference
- **TensorFlow/Keras** for model training and export
- **NumPy** for memory-mapped bitmap loading
- **HTML Canvas** for drawing input
- **Vite** for development and builds
- **GitHub Pages** for deployment

## Project structure

```text
.
├── public/tfjs_model/           # Browser model and ordered labels
├── scripts/                     # Model contract validation
├── src/components/              # Drawing and portfolio UI
├── src/composables/             # TensorFlow.js model lifecycle
├── src/data/                    # Display category groups
├── model/
│   ├── requirements.txt
│   └── train.py                 # Complete training pipeline
├── quickdraw_data/
│   └── fetch.py                 # Full dataset downloader
├── package.json
└── vite.config.js
```

## Run the frontend locally

Requirements: Node.js and npm.

```sh
git clone https://github.com/Furkan-rgb/sketch_classifier.git
cd sketch_classifier
npm ci
npm run dev
```

Create and preview a production build with:

```sh
npm run build
npm run preview
```

`npm run build` validates the model input shape, output count, ordered label
contract, and UI category coverage before building the application.

## Deployment

The Vite base path is configured for `/sketch_classifier/`. Publish `dist/` to
the `gh-pages` branch with:

```sh
npm run deploy
```
