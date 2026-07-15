# QuickDraw Sketch Classifier

A browser-based doodle classifier powered by a custom 100-class convolutional
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

```text
28 × 28 × 1 input
  → [Conv2D(48, 3×3) → BatchNorm → ReLU] × 2 → MaxPool
  → [Conv2D(96, 3×3) → BatchNorm → ReLU] × 2 → MaxPool
  → [Conv2D(192, 3×3) → BatchNorm → ReLU] × 2
  → Global Average Pooling
  → Dense(128, ReLU) → Dropout(0.35)
  → Dense(100 logits)
```

Design rationale, given the constraint that the model must download and run in
the browser:

- **Stacked 3 × 3 convolutions** — two per block give the receptive field of a
  5 × 5 kernel with fewer parameters and an extra non-linearity.
- **Widening blocks (48 → 96 → 192) with pooling** — spatial resolution is
  traded for channel depth as features grow from strokes to shapes to objects.
- **Batch normalization, no conv biases** — stabilizes training at a learning
  rate of `1e-3`; biases are redundant directly before normalization.
- **Global average pooling instead of flattening** — collapses each feature map
  to one value, cutting the usual dense-layer parameter bulk and much of the
  overfitting risk.
- **128-unit embedding with 35% dropout** — a small mixing layer between pooled
  features and the classifier, regularized because it is the layer most prone
  to memorization.
- **Raw logits output** — softmax is applied in the browser, so exported
  weights stay loss-function agnostic.

Geometric and stroke-width augmentation wrap the classifier only during
training; the exported browser model contains none of it. The result is
roughly 684k parameters — about 2.6 MB of float32 weights.

## Training pipeline

[`model/train.py`](model/train.py) is the single entry point for splitting,
loading, augmenting, training, evaluation, and TensorFlow.js export. It includes:

- Seeded, non-overlapping 70/15/15 train, validation, and test splits
- Balanced loading from memory-mapped NumPy files
- Restrained translation, rotation, zoom, dilation, and erosion
- Up to 50 epochs with early stopping and learning-rate reduction
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
<summary>View all 100 categories</summary>

- **Shapes:** `circle`, `square`, `triangle`, `star`, `line`, `hexagon`,
  `diamond`, `zigzag`
- **Everyday objects:** `cup`, `clock`, `chair`, `book`, `laptop`,
  `cell phone`, `key`, `umbrella`, `table`, `bed`, `door`, `light bulb`,
  `ladder`, `envelope`
- **Clothing:** `hat`, `shoe`, `sock`, `t-shirt`, `pants`, `eyeglasses`
- **Animals:** `cat`, `dog`, `bird`, `fish`, `elephant`, `giraffe`, `horse`,
  `cow`, `pig`, `rabbit`, `monkey`, `lion`, `duck`, `owl`, `penguin`, `frog`,
  `snake`, `spider`, `butterfly`, `whale`
- **Nature:** `tree`, `flower`, `sun`, `cloud`, `moon`, `mountain`, `rainbow`,
  `lightning`, `snowflake`, `cactus`, `mushroom`, `leaf`
- **People & features:** `eye`, `hand`, `face`, `smiley face`, `ear`, `nose`,
  `mouth`, `tooth`
- **Tools & music:** `scissors`, `pencil`, `hammer`, `guitar`, `axe`, `saw`,
  `screwdriver`, `piano`
- **Transport:** `car`, `bicycle`, `airplane`, `sailboat`, `bus`, `truck`,
  `train`, `helicopter`, `hot air balloon`, `motorbike`
- **Food:** `apple`, `banana`, `pizza`, `grapes`, `strawberry`, `watermelon`,
  `carrot`, `ice cream`, `donut`, `hamburger`
- **Buildings & landmarks:** `house`, `castle`, `bridge`, `lighthouse`

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

Every push to `main` is deployed automatically: the
[GitHub Actions workflow](.github/workflows/deploy.yml) validates the model
contract, builds the site, and publishes it to GitHub Pages. The Vite base
path is configured for `/sketch_classifier/`.
