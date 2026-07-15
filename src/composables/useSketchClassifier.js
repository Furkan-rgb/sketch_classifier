import { computed, ref } from "vue";
import { categories as displayedCategories } from "../data/categories";

export function useSketchClassifier() {
  const status = ref("idle");
  const predictions = ref([]);
  const inferenceTime = ref(null);
  const errorMessage = ref("");

  let tf = null;
  let model = null;
  let classNames = [];
  let latestRequest = 0;

  const isReady = computed(() => status.value === "ready");

  async function loadModel() {
    status.value = "loading";
    errorMessage.value = "";

    try {
      tf = await import("@tensorflow/tfjs");
      await tf.ready();
      const modelBaseUrl = `${import.meta.env.BASE_URL}tfjs_model`;
      const [loadedModel, classResponse] = await Promise.all([
        tf.loadLayersModel(`${modelBaseUrl}/model.json`),
        fetch(`${modelBaseUrl}/class_names.json`),
      ]);

      if (!classResponse.ok) {
        throw new Error(`Unable to load model classes (${classResponse.status}).`);
      }

      model = loadedModel;
      const contract = await classResponse.json();
      classNames = validateModelContract(model, contract);

      const warmupInput = tf.zeros([1, 28, 28, 1]);
      const warmupOutput = model.predict(warmupInput);
      const warmupTensor = Array.isArray(warmupOutput)
        ? warmupOutput[0]
        : warmupOutput;
      await warmupTensor.data();
      tf.dispose([warmupInput, warmupOutput]);

      status.value = "ready";
    } catch (error) {
      console.error("Unable to load the sketch model", error);
      status.value = "error";
      errorMessage.value =
        "The model files are incomplete or incompatible. Refresh and try again.";
    }
  }

  function validateModelContract(loadedModel, contract) {
    const names = contract?.classNames;
    if (
      !Array.isArray(names) ||
      names.length === 0 ||
      names.some((name) => typeof name !== "string" || !name.trim()) ||
      new Set(names).size !== names.length
    ) {
      throw new Error("The model class list is missing, empty, or contains duplicates.");
    }

    const outputShape = loadedModel.outputs?.[0]?.shape;
    const outputUnits = outputShape?.[outputShape.length - 1];
    if (Number.isInteger(outputUnits) && outputUnits !== names.length) {
      throw new Error(
        `The model has ${outputUnits} outputs but ${names.length} class names.`,
      );
    }

    const inputShape = loadedModel.inputs?.[0]?.shape?.slice(1);
    const expectedInputShape = [
      contract?.input?.height,
      contract?.input?.width,
      contract?.input?.channels,
    ];
    if (
      inputShape?.length !== expectedInputShape.length ||
      inputShape.some((dimension, index) => dimension !== expectedInputShape[index]) ||
      contract?.input?.source !== "full-canvas" ||
      contract?.input?.foreground !== 1 ||
      contract?.input?.background !== 0
    ) {
      throw new Error("The model input preprocessing contract is incompatible.");
    }

    const trainedSet = new Set(names);
    const displayedSet = new Set(displayedCategories);
    const missingFromUi = names.filter((name) => !displayedSet.has(name));
    const notTrained = displayedCategories.filter((name) => !trainedSet.has(name));
    if (
      displayedSet.size !== displayedCategories.length ||
      missingFromUi.length ||
      notTrained.length
    ) {
      throw new Error(
        `Model/UI classes disagree. Missing from UI: ${missingFromUi.join(", ") || "none"}; not trained: ${notTrained.join(", ") || "none"}.`,
      );
    }

    return names;
  }

  function getLogits(output) {
    if (Array.isArray(output)) return output[0];
    if (output && typeof output === "object" && !output.data) {
      return output.Identity || output.output_0 || Object.values(output)[0];
    }
    return output;
  }

  async function classify(inputBuffer) {
    if (!model || !tf || status.value !== "ready" || !inputBuffer) return;

    const requestId = ++latestRequest;
    const startedAt = performance.now();
    let inputTensor = null;
    let output = null;
    let probabilitiesTensor = null;

    try {
      inputTensor = tf.tensor4d(inputBuffer, [1, 28, 28, 1]);
      output = model.predict(inputTensor);
      const logits = getLogits(output);

      if (!logits) throw new Error("The model returned no prediction tensor.");

      probabilitiesTensor = tf.softmax(logits);
      const values = Array.from(await probabilitiesTensor.data());

      if (requestId !== latestRequest) return;

      predictions.value = values
        .map((probability, index) => ({
          index,
          label: classNames[index] || `Class ${index + 1}`,
          probability,
        }))
        .sort((a, b) => b.probability - a.probability)
        .slice(0, 5);
      inferenceTime.value = Math.max(1, Math.round(performance.now() - startedAt));
    } catch (error) {
      console.error("Unable to classify the drawing", error);
      if (requestId === latestRequest) {
        errorMessage.value = "That prediction did not complete. Try another stroke.";
      }
    } finally {
      if (inputTensor) inputTensor.dispose();
      if (probabilitiesTensor) probabilitiesTensor.dispose();
      if (output) tf.dispose(output);
    }
  }

  function resetPrediction() {
    latestRequest += 1;
    predictions.value = [];
    inferenceTime.value = null;
    errorMessage.value = "";
  }

  return {
    status,
    predictions,
    inferenceTime,
    errorMessage,
    isReady,
    loadModel,
    classify,
    resetPrediction,
  };
}
