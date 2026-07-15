export const categoryGroups = [
  {
    name: "Shapes",
    icon: "🔺",
    categories: ["circle", "square", "triangle", "star", "line"],
  },
  {
    name: "Everyday objects",
    icon: "🏠",
    categories: ["cup", "clock", "chair", "book", "laptop", "cell phone", "key", "umbrella"],
  },
  {
    name: "Animals",
    icon: "🐾",
    categories: ["cat", "dog", "bird", "fish"],
  },
  {
    name: "Nature",
    icon: "🌿",
    categories: ["tree", "flower", "sun", "cloud"],
  },
  {
    name: "People & features",
    icon: "🙂",
    categories: ["eye", "hand", "face", "smiley face"],
  },
  {
    name: "Tools & music",
    icon: "🎸",
    categories: ["scissors", "pencil", "hammer", "guitar"],
  },
  {
    name: "Transport",
    icon: "🚲",
    categories: ["car", "bicycle", "airplane", "sailboat"],
  },
  {
    name: "Food",
    icon: "🍕",
    categories: ["apple", "banana", "pizza"],
  },
];

// Display order is intentionally independent from the model's output order.
// Inference labels come from tfjs_model/class_names.json.
export const categories = categoryGroups.flatMap((group) => group.categories);

export const suggestedCategories = [
  "cat",
  "umbrella",
  "bicycle",
  "guitar",
  "sailboat",
  "flower",
  "pizza",
  "clock",
];
