<template>
  <section class="known-categories" aria-labelledby="known-categories-title">
    <div class="known-categories-heading">
      <div>
        <span class="eyebrow">The model’s vocabulary</span>
        <h2 id="known-categories-title">What can I draw?</h2>
      </div>
      <p>
        The classifier only knows these {{ categories.length }} doodles. Pick
        one for a prompt, or draw anything and see which known shape it
        resembles.
      </p>
    </div>

    <div class="category-groups">
      <article v-for="group in categoryGroups" :key="group.name" class="category-group">
        <div class="category-group-title">
          <span aria-hidden="true">{{ group.icon }}</span>
          <h3>{{ group.name }}</h3>
        </div>
        <div class="category-chips">
          <button
            v-for="category in group.categories"
            :key="category"
            type="button"
            :class="{ selected: category === selectedCategory }"
            :aria-pressed="category === selectedCategory"
            @click="$emit('select', category)"
          >
            {{ category }}
          </button>
        </div>
      </article>
    </div>

    <p class="category-note">
      <span aria-hidden="true">i</span>
      If you draw something outside this list, the model will still choose its
      closest known category—it cannot return “unknown.”
    </p>
  </section>
</template>

<script setup>
import { categories, categoryGroups } from "../data/categories";

defineProps({
  selectedCategory: { type: String, default: "" },
});

defineEmits(["select"]);
</script>
