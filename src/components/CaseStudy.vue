<template>
  <section id="case-study" class="case-study section-shell" aria-labelledby="case-study-title">
    <div class="section-intro">
      <span class="eyebrow">Under the hood</span>
      <h2 id="case-study-title">From a rough line to a ranked prediction.</h2>
      <p>
        A compact convolutional neural network turns each sketch into one of 100
        familiar objects—all without sending the drawing to a server.
      </p>
    </div>

    <div class="pipeline" aria-label="Model inference pipeline">
      <article class="pipeline-step">
        <span class="step-number">01</span>
        <div class="step-icon draw-icon" aria-hidden="true">
          <svg viewBox="0 0 56 56"><path d="M10 39c10-25 18 10 27-14 5-13 9 7 10 15" /></svg>
        </div>
        <h3>Capture</h3>
        <p>Pointer strokes are recorded on an 800 × 800 canvas.</p>
      </article>
      <span class="pipeline-arrow" aria-hidden="true">→</span>
      <article class="pipeline-step">
        <span class="step-number">02</span>
        <div class="step-icon pixels-icon" aria-hidden="true">
          <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
        </div>
        <h3>Normalize</h3>
        <p>The full canvas is reduced to the same 28 × 28 ink map used for training.</p>
      </article>
      <span class="pipeline-arrow" aria-hidden="true">→</span>
      <article class="pipeline-step">
        <span class="step-number">03</span>
        <div class="step-icon network-icon" aria-hidden="true">
          <span></span><span></span><span></span><span></span><span></span>
        </div>
        <h3>Classify</h3>
        <p>Convolution blocks extract shapes before a 100-class output layer.</p>
      </article>
      <span class="pipeline-arrow" aria-hidden="true">→</span>
      <article class="pipeline-step">
        <span class="step-number">04</span>
        <div class="step-icon bars-icon" aria-hidden="true">
          <i></i><i></i><i></i>
        </div>
        <h3>Rank</h3>
        <p>TensorFlow.js converts logits into five live confidence scores.</p>
      </article>
    </div>

    <div class="story-grid">
      <article class="story-card story-card-large">
        <span class="eyebrow">The approach</span>
        <h3>A training pipeline designed around limited memory.</h3>
        <p>
          Google’s Quick, Draw! bitmap dataset is much larger than this proof of
          concept needs to hold in memory at once. A deterministic pipeline
          memory-maps each class, keeps its splits separate, and streams balanced
          mini-batches to Keras.
        </p>
        <ol class="training-flow">
          <li>
            <span>01</span>
            <p><strong>Load</strong> — memory-map one NumPy bitmap file per class</p>
          </li>
          <li>
            <span>02</span>
            <p><strong>Split</strong> — seeded, disjoint 70/15/15 train/validation/test</p>
          </li>
          <li>
            <span>03</span>
            <p><strong>Stream</strong> — balanced batches; a rolling per-epoch window per class</p>
          </li>
          <li>
            <span>04</span>
            <p><strong>Augment</strong> — shift, rotate, zoom and stroke width, training only</p>
          </li>
          <li>
            <span>05</span>
            <p><strong>Train</strong> — early stopping, LR halving, best weights restored</p>
          </li>
          <li>
            <span>06</span>
            <p><strong>Evaluate</strong> — top-1/top-3, per-class recall, confusion matrices</p>
          </li>
          <li>
            <span>07</span>
            <p><strong>Export</strong> — TensorFlow.js model + ordered class_names.json</p>
          </li>
        </ol>
      </article>

      <article class="story-card architecture-card">
        <span class="eyebrow">Model shape</span>
        <h3>Small enough for the browser.</h3>
        <div class="architecture-list">
          <div><span>Input</span><strong>28 × 28 × 1</strong></div>
          <div><span>Feature blocks</span><strong>48 → 96 → 192</strong></div>
          <div><span>Pooling</span><strong>Global average</strong></div>
          <div><span>Regularization</span><strong>35% dropout</strong></div>
          <div><span>Output</span><strong>100 logits</strong></div>
        </div>
      </article>

      <article class="story-card learning-card">
        <span class="eyebrow">What I learned</span>
        <h3>The interface is part of the ML system.</h3>
        <p>
          Training and browser preprocessing must agree exactly. Showing the
          cropped, centered 28 × 28 input makes that model constraint visible
          instead of hiding it from the user.
        </p>
      </article>

      <article class="story-card limits-card">
        <span class="eyebrow">Honest limitations</span>
        <h3>A proof of concept, with a clear next step.</h3>
        <p>
          The model only recognizes its 100 training categories and can sound
          confident about unfamiliar drawings. On 500,000 held-out sketches,
          the exported model reaches 87.10% top-1 accuracy and 95.35% top-3
          accuracy. The training run also records per-class recall and confusion
          matrices so failure cases can be measured instead of guessed at.
        </p>
      </article>
    </div>
  </section>
</template>
