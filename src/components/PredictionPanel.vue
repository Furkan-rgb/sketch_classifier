<template>
  <section class="prediction-card" aria-labelledby="prediction-title">
    <div class="card-heading prediction-heading">
      <div>
        <span class="eyebrow">02 · Model output</span>
        <h2 id="prediction-title">The model says</h2>
      </div>
      <span v-if="inferenceTime" class="latency-pill">{{ inferenceTime }} ms</span>
    </div>

    <div v-if="status === 'loading' || status === 'idle'" class="model-state">
      <div class="loader-orbit" aria-hidden="true"><span></span></div>
      <h3>Warming up the model</h3>
      <p>Loading the neural network into your browser…</p>
    </div>

    <div v-else-if="status === 'error'" class="model-state error-state" role="alert">
      <span class="state-icon">!</span>
      <h3>Model unavailable</h3>
      <p>{{ errorMessage }}</p>
      <button class="primary-button small" type="button" @click="$emit('retry')">
        Try again
      </button>
    </div>

    <div v-else-if="!topPrediction" class="model-state empty-state">
      <div class="empty-doodle" aria-hidden="true">
        <span>?</span>
        <svg viewBox="0 0 130 80">
          <path d="M8 57c16-45 31 15 49-26 12-28 28 40 42-7 4-13 15-13 23 0" />
        </svg>
      </div>
      <h3>Your result will appear here</h3>
      <p>Draw on the canvas and I’ll rank the five closest matches.</p>
    </div>

    <div v-else class="prediction-results">
      <div class="primary-result" aria-live="polite" aria-atomic="true">
        <p>{{ confidenceCopy }}</p>
        <div class="result-label-row">
          <h3>{{ topPrediction.label }}</h3>
          <span>{{ formatPercent(topPrediction.probability) }}</span>
        </div>
        <div
          class="confidence-track"
          role="progressbar"
          aria-label="Top prediction confidence"
          aria-valuemin="0"
          aria-valuemax="100"
          :aria-valuenow="Math.round(topPrediction.probability * 100)"
        >
          <span :style="{ width: `${topPrediction.probability * 100}%` }"></span>
        </div>
      </div>

      <div class="alternatives">
        <div class="section-label-row">
          <span>Other possibilities</span>
          <span>Confidence</span>
        </div>
        <div
          v-for="prediction in alternativePredictions"
          :key="prediction.index"
          class="prediction-row"
        >
          <span class="prediction-name">{{ prediction.label }}</span>
          <span class="mini-track" aria-hidden="true">
            <span :style="{ width: `${prediction.probability * 100}%` }"></span>
          </span>
          <span class="prediction-value">{{ formatPercent(prediction.probability) }}</span>
        </div>
      </div>
    </div>

    <div class="model-vision">
      <div class="vision-preview" :class="{ empty: !previewUrl }">
        <img v-if="previewUrl" :src="previewUrl" alt="The processed 28 by 28 pixel model input" />
        <span v-else>28</span>
      </div>
      <div>
        <span class="vision-label">What the model sees</span>
        <p>Centered, inverted and resized to 28 × 28 pixels.</p>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({
  status: { type: String, required: true },
  predictions: { type: Array, default: () => [] },
  inferenceTime: { type: Number, default: null },
  previewUrl: { type: String, default: "" },
  errorMessage: { type: String, default: "" },
});

defineEmits(["retry"]);

const topPrediction = computed(() => props.predictions[0] || null);
const alternativePredictions = computed(() => props.predictions.slice(1));
const confidenceCopy = computed(() => {
  const probability = topPrediction.value?.probability || 0;
  if (probability >= 0.75) return "I’m pretty sure it’s a…";
  if (probability >= 0.45) return "My best guess is…";
  return "This is a tricky one. Maybe a…";
});

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}
</script>
