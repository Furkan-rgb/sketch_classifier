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

We use a custom, scaled-down architecture based on **VGG-Net's design pattern**
rather than a standard VGG model. It borrows VGG's pattern of stacking small
`3 × 3` convolutions and widening the channel count as spatial resolution falls
([Simonyan & Zisserman, 2014](https://arxiv.org/abs/1409.1556)). It combines
that pattern with [batch normalization](https://arxiv.org/abs/1502.03167) and
[global average pooling](https://arxiv.org/abs/1312.4400), both established
ideas from other CNN research.

VGG was originally developed and evaluated on ImageNet, but this classifier is
not pretrained on ImageNet and uses no VGG weights. Its layers are initialized
from scratch and trained only on Quick, Draw! bitmaps.

### Design rationale

The easiest way to understand the network is as a gradual conversion from
pixels into evidence:

```text
pixels → strokes and corners → larger shapes and parts
       → a compact description of the drawing → 100 class scores
```

Early layers preserve location because the model still needs to learn how
strokes connect. Deeper layers care more about which patterns are present than
their exact pixel positions. Every choice balances this recognition task
against the need to download quickly and run in real time in a browser.

VGG-style feature extraction was chosen as a simple, well-understood baseline
for building local strokes into higher-level shapes. Once the scaled-down model
produced a working classifier within the project's download-size and real-time
browser constraints, we did not continue into a wider architecture search. It
should therefore not be read as a claim that VGG is optimal for this task.

A scaled-down [MobileNet](https://arxiv.org/abs/1704.04861), or another
lightweight CNN, could also be a good fit. MobileNet primarily uses
depthwise-separable convolutions in place of standard convolutions, which can
reduce parameter count and computation. Whether that would improve accuracy,
model size, or actual TensorFlow.js latency here would need to be measured. We
have not implemented or benchmarked that alternative.

- **Convolution blocks learn increasingly complex features.** Each `3 × 3`
  convolution examines a small neighborhood. Stacking two lets a block combine
  simple features such as edges into larger patterns such as corners, curves,
  and object parts. The second convolution also adds another non-linear
  transformation, allowing the block to represent more complex patterns.
- **Pooling trades spatial detail for higher-level features.** The feature maps
  shrink from `28 × 28` to `14 × 14` and then `7 × 7`. At the same time, the
  number of channels grows from 48 to 96 to 192. The network therefore keeps
  less precise location information while learning a larger vocabulary of
  detectable patterns.
- **Batch normalization keeps training stable.** It keeps intermediate values
  on a manageable scale as they pass through the network. Convolution biases
  are omitted because the following normalization layer already learns an
  equivalent offset.
- **Global average pooling creates a feature inventory.** At the end of the
  convolution blocks, the model has 192 feature maps of size `7 × 7`. Averaging
  each map produces 192 values that roughly answer, "How strongly was this
  feature detected anywhere in the drawing?" This is much smaller than
  flattening all `7 × 7 × 192 = 9,408` values and makes the prediction less
  dependent on an object appearing at one exact position.
- **The 128-unit dense layer combines those features.** It turns the feature
  inventory into a compact signature of the whole drawing. During training,
  dropout hides 35% of these values at random, discouraging the classifier from
  relying too heavily on any one feature.
- **The final layer produces 100 logits.** Each logit is an unnormalized score
  for one class. The browser applies softmax to turn these scores into
  probabilities and displays the five highest ones.

Geometric and stroke-width augmentation wrap the classifier only during
training; the exported browser model contains none of it. The result is a
task-specific arrangement of established CNN building blocks with roughly 684k
parameters — about 2.6 MB of float32 weights.

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
