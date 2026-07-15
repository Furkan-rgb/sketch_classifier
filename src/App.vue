<template>
  <div class="site-shell">
    <header class="site-header section-shell">
      <a
        class="profile-brand"
        href="https://github.com/Furkan-rgb"
        target="_blank"
        rel="noreferrer"
        aria-label="Furkan-rgb on GitHub"
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path
            d="M12 .7a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2.24c-3.22.7-3.9-1.37-3.9-1.37-.53-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.7.08-.7 1.16.08 1.78 1.2 1.78 1.2 1.04 1.77 2.72 1.26 3.38.96.1-.75.4-1.26.74-1.55-2.57-.3-5.28-1.29-5.28-5.69 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.16 1.18a10.98 10.98 0 0 1 5.76 0c2.2-1.49 3.16-1.18 3.16-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.42-2.71 5.39-5.29 5.68.42.36.79 1.06.79 2.14v3.17c0 .31.21.67.79.56A11.5 11.5 0 0 0 12 .7Z"
          />
        </svg>
        <span>Furkan-rgb</span>
      </a>

      <nav aria-label="Primary navigation">
        <a href="#playground">Playground</a>
        <a href="#case-study">Case study</a>
        <a class="github-link" href="https://github.com/Furkan-rgb/sketch_classifier" target="_blank" rel="noreferrer">
          View source
          <svg viewBox="0 0 16 16" aria-hidden="true">
            <path d="M5 11 11 5M6 5h5v5" />
          </svg>
        </a>
      </nav>
    </header>

    <main id="top">
      <section class="hero section-shell" aria-labelledby="hero-title">
        <div class="hero-copy">
          <div class="hero-kicker">
            <span class="spark" aria-hidden="true">✦</span>
            Interactive machine learning experiment
          </div>
          <h1 id="hero-title">
            Draw something.<br />
            <span>The model will guess.</span>
          </h1>
          <p>
            A convolutional neural network that recognizes freehand doodles in real time. Everything runs locally in
            your browser.
          </p>
          <div class="hero-actions">
            <a class="primary-button" href="#playground">
              Start drawing
              <svg viewBox="0 0 20 20" aria-hidden="true">
                <path d="m6 10 8 0M11 6l4 4-4 4" />
              </svg>
            </a>
            <details class="category-disclosure">
              <summary>{{ categories.length }} things it knows</summary>
              <div class="category-popover">
                <p>Try any of these</p>
                <div>
                  <span v-for="category in categories" :key="category">{{ category }}</span>
                </div>
              </div>
            </details>
          </div>
        </div>

        <div class="hero-art" aria-hidden="true">
          <span class="doodle-label label-cat">cat?</span>
          <span class="doodle-label label-tree">tree?</span>
          <svg class="hero-squiggle" viewBox="0 0 480 320">
            <path d="M67 224c31-111 91 57 133-73 27-84 69 92 119-29 28-67 65-8 94 68" />
            <path class="accent-path" d="m394 180 21 17-25 9" />
          </svg>
          <div class="floating-card confidence-card">
            <span>Top prediction</span>
            <strong>umbrella</strong>
            <i><b></b></i>
            <small>87.4% confidence</small>
          </div>
          <div class="floating-card pixel-card">
            <span>28 × 28</span>
            <div class="pixel-grid">
              <i v-for="index in 25" :key="index"></i>
            </div>
          </div>
          <span class="hero-star star-one">✦</span>
          <span class="hero-star star-two">✦</span>
        </div>
      </section>

      <section id="playground" class="playground-section">
        <div class="section-shell">
          <div class="playground-intro">
            <div>
              <span class="eyebrow">Live playground</span>
              <h2>Put the model to the test.</h2>
            </div>
            <p>No uploads. No API calls. Your drawing never leaves this page.</p>
          </div>

          <div class="playground-grid">
            <DrawingCanvas
              :prompt="currentPrompt"
              :model-ready="isReady"
              @predict="handlePrediction"
              @cleared="handleClear"
              @shuffle="choosePrompt"
            />
            <PredictionPanel
              :status="status"
              :predictions="predictions"
              :inference-time="inferenceTime"
              :preview-url="previewUrl"
              :error-message="errorMessage"
              @retry="loadModel"
            />
          </div>

          <div class="model-facts" aria-label="Model facts">
            <div>
              <span class="fact-icon">⌁</span>
              <p><strong>Local inference</strong><small>TensorFlow.js</small></p>
            </div>
            <div>
              <span class="fact-icon">100</span>
              <p><strong>Categories</strong><small>Balanced training</small></p>
            </div>
            <div>
              <span class="fact-icon pixel-fact">▦</span>
              <p><strong>28 × 28 input</strong><small>Grayscale bitmap</small></p>
            </div>
            <div>
              <span class="fact-icon">≈3</span>
              <p><strong>MB model</strong><small>Browser-ready</small></p>
            </div>
          </div>

          <NetworkVisualizer :activations="activations" :predictions="predictions" />

          <KnownCategories :selected-category="currentPrompt" @select="selectPrompt" />
        </div>
      </section>

      <CaseStudy />

      <section class="project-cta section-shell">
        <div>
          <span class="eyebrow">Explore the implementation</span>
          <h2>Built as a small experiment.<br />Refined as a complete product.</h2>
        </div>
        <a
          class="primary-button light-button"
          href="https://github.com/Furkan-rgb/sketch_classifier"
          target="_blank"
          rel="noreferrer"
        >
          Browse the source
          <svg viewBox="0 0 20 20" aria-hidden="true">
            <path d="m6 10 8 0M11 6l4 4-4 4" />
          </svg>
        </a>
      </section>
    </main>

    <footer class="site-footer section-shell">
      <a
        class="profile-brand"
        href="https://github.com/Furkan-rgb"
        target="_blank"
        rel="noreferrer"
        aria-label="Furkan-rgb on GitHub"
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path
            d="M12 .7a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2.24c-3.22.7-3.9-1.37-3.9-1.37-.53-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.7.08-.7 1.16.08 1.78 1.2 1.78 1.2 1.04 1.77 2.72 1.26 3.38.96.1-.75.4-1.26.74-1.55-2.57-.3-5.28-1.29-5.28-5.69 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.16 1.18a10.98 10.98 0 0 1 5.76 0c2.2-1.49 3.16-1.18 3.16-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.42-2.71 5.39-5.29 5.68.42.36.79 1.06.79 2.14v3.17c0 .31.21.67.79.56A11.5 11.5 0 0 0 12 .7Z"
          />
        </svg>
        <span>Furkan-rgb</span>
      </a>
      <a href="#top">Back to top ↑</a>
    </footer>
  </div>
</template>

<script setup>
import { onMounted, ref } from "vue";
import CaseStudy from "./components/CaseStudy.vue";
import DrawingCanvas from "./components/DrawingCanvas.vue";
import KnownCategories from "./components/KnownCategories.vue";
import NetworkVisualizer from "./components/NetworkVisualizer.vue";
import PredictionPanel from "./components/PredictionPanel.vue";
import { useSketchClassifier } from "./composables/useSketchClassifier";
import { categories, suggestedCategories } from "./data/categories";

const currentPrompt = ref(suggestedCategories[0]);
const previewUrl = ref("");
let lastPromptIndex = 0;

const {
  status,
  predictions,
  inferenceTime,
  errorMessage,
  activations,
  isReady,
  loadModel,
  classify,
  resetPrediction,
} = useSketchClassifier();

onMounted(loadModel);

function handlePrediction({ input, previewUrl: nextPreview }) {
  previewUrl.value = nextPreview;
  classify(input);
}

function handleClear() {
  previewUrl.value = "";
  resetPrediction();
}

function choosePrompt() {
  let nextIndex = lastPromptIndex;
  while (nextIndex === lastPromptIndex) {
    nextIndex = Math.floor(Math.random() * suggestedCategories.length);
  }
  lastPromptIndex = nextIndex;
  currentPrompt.value = suggestedCategories[nextIndex];
}

function selectPrompt(category) {
  currentPrompt.value = category;
}
</script>
