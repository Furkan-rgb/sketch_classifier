<template>
  <section class="network-viz" aria-labelledby="network-viz-title">
    <div class="network-viz-heading">
      <div>
        <span class="eyebrow">03 · Inside the network</span>
        <h2 id="network-viz-title">Watch it think</h2>
      </div>
      <p>
        Live activations from the network in your browser. Each block shows its
        six most active feature detectors for your current drawing—brighter
        means a stronger response.
      </p>
    </div>

    <div v-if="!stages" class="network-viz-empty">
      <p>Draw on the canvas above and the layers will light up here.</p>
    </div>

    <div v-else class="network-viz-flow" aria-label="Layer-by-layer activations">
      <article class="viz-stage">
        <header>
          <h3>Input</h3>
          <span>28 × 28 × 1</span>
        </header>
        <div class="viz-single">
          <img :src="stages.input" alt="The 28 by 28 bitmap the model receives" />
        </div>
        <p>Your drawing, as the model receives it.</p>
      </article>

      <span class="viz-arrow" aria-hidden="true">→</span>

      <template v-for="(block, index) in stages.blocks" :key="block.caption">
        <article class="viz-stage">
          <header>
            <h3>Block {{ index + 1 }}</h3>
            <span>{{ block.caption }}</span>
          </header>
          <div class="viz-map-grid">
            <img
              v-for="map in block.images"
              :key="map.channel"
              :src="map.src"
              :alt="`Block ${index + 1} feature map ${map.channel}`"
              :title="`Channel ${map.channel}`"
            />
          </div>
          <p>{{ block.note }}</p>
        </article>
        <span class="viz-arrow" aria-hidden="true">→</span>
      </template>

      <article class="viz-stage">
        <header>
          <h3>Pooled</h3>
          <span>{{ pooledCaption }}</span>
        </header>
        <div class="viz-single viz-strip">
          <img :src="stages.pooled" alt="Globally pooled feature vector" />
        </div>
        <p>Each map collapses to one number: “how strongly did I fire anywhere?”</p>
      </article>

      <span class="viz-arrow" aria-hidden="true">→</span>

      <article class="viz-stage">
        <header>
          <h3>Embedding</h3>
          <span>{{ embeddingCaption }}</span>
        </header>
        <div class="viz-single viz-strip">
          <img :src="stages.embedding" alt="Embedding layer activations" />
        </div>
        <p>A compact mix of every feature—the drawing’s fingerprint.</p>
      </article>

      <span class="viz-arrow" aria-hidden="true">→</span>

      <article class="viz-stage viz-verdict">
        <header>
          <h3>Verdict</h3>
          <span>{{ categories.length }} logits → softmax</span>
        </header>
        <div class="viz-pred-rows">
          <div v-for="prediction in topPredictions" :key="prediction.index" class="viz-pred-row">
            <span class="viz-pred-name">{{ prediction.label }}</span>
            <span class="viz-pred-track" aria-hidden="true">
              <span :style="{ width: `${prediction.probability * 100}%` }"></span>
            </span>
            <span class="viz-pred-value">{{ (prediction.probability * 100).toFixed(1) }}%</span>
          </div>
        </div>
        <p>The highest score wins.</p>
      </article>
    </div>
  </section>
</template>

<script setup>
import { computed } from "vue";
import { categories } from "../data/categories";

const props = defineProps({
  activations: { type: Object, default: null },
  predictions: { type: Array, default: () => [] },
});

const BLOCK_NOTES = [
  "Edge and stroke detectors on the raw ink.",
  "Corners and curves built from those strokes.",
  "Object parts—coarse, but semantically rich.",
];

const PAPER = [255, 253, 247];
const INK = [25, 24, 20];
const ACCENT = [255, 92, 53];
const PURPLE = [117, 103, 255];
const STRIP_COLUMNS = 16;

const mix = (from, to, t) => from.map((value, i) => Math.round(value + (to[i] - value) * t));
const inkPalette = (v) => mix(PAPER, INK, v);
const heatPalette = (v) =>
  v < 0.6 ? mix(PAPER, ACCENT, v / 0.6) : mix(ACCENT, INK, (v - 0.6) / 0.4);
const vectorPalette = (v) => mix(PAPER, PURPLE, v);

function renderBitmap(values, width, height, palette, max = null) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  const image = context.createImageData(width, height);
  const peak = max ?? values.reduce((top, value) => (value > top ? value : top), 0);

  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const value = values[pixel] ?? 0;
    const [r, g, b] = palette(peak > 0 ? Math.min(value / peak, 1) : 0);
    image.data[pixel * 4] = r;
    image.data[pixel * 4 + 1] = g;
    image.data[pixel * 4 + 2] = b;
    image.data[pixel * 4 + 3] = 255;
  }

  context.putImageData(image, 0, 0);
  return canvas.toDataURL();
}

function renderVectorStrip(values) {
  const rows = Math.ceil(values.length / STRIP_COLUMNS);
  return renderBitmap(values, STRIP_COLUMNS, rows, vectorPalette);
}

const stages = computed(() => {
  const current = props.activations;
  if (!current) return null;

  return {
    input: renderBitmap(current.input, 28, 28, inkPalette, 1),
    blocks: current.blocks.map((block, index) => ({
      caption: `${block.channels} maps · ${block.width} × ${block.height}`,
      note: BLOCK_NOTES[index] || "",
      images: block.maps.map((map) => ({
        channel: map.channel,
        src: renderBitmap(map.values, block.width, block.height, heatPalette, map.max),
      })),
    })),
    pooled: renderVectorStrip(current.pooled),
    embedding: renderVectorStrip(current.embedding),
  };
});

const pooledCaption = computed(() =>
  props.activations ? `${props.activations.pooled.length} values` : "",
);
const embeddingCaption = computed(() =>
  props.activations ? `${props.activations.embedding.length} values` : "",
);
const topPredictions = computed(() => props.predictions.slice(0, 5));
</script>
