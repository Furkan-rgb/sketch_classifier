import { computed, ref } from "vue";
import { categories as displayedCategories } from "../data/categories";

// Layers tapped for the live network visualization, in forward order. The
// last entry must be the logits layer; the rest feed the activation display.
const ACTIVATION_LAYERS = [
  "block_1_relu_2",
  "block_2_relu_2",
  "block_3_relu_2",
  "global_average_pooling",
  "embedding",
  "class_logits",
];
const FEATURE_MAPS_PER_BLOCK = 6;

export function useSketchClassifier() {
  const status = ref("idle");
  const predictions = ref([]);
  const inferenceTime = ref(null);
  const errorMessage = ref("");
  const activations = ref(null);

  let tf = null;
  let model = null;
  let activationModel = null;
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
      // The model ships as three files that must come from the same deploy:
      // model.json, the weight shard(s) it lists, and class_names.json. The
      // build tag busts stale caches; weightUrlConverter applies it to the
      // shard requests, which TensorFlow.js resolves internally.
      const withBuildTag = (url) => `${url}?v=${__BUILD_ID__}`;
      const [loadedModel, classResponse] = await Promise.all([
        tf.loadLayersModel(
          tf.io.http(withBuildTag(`${modelBaseUrl}/model.json`), {
            weightUrlConverter: async (weightFileName) =>
              withBuildTag(`${modelBaseUrl}/${weightFileName}`),
          }),
        ),
        fetch(withBuildTag(`${modelBaseUrl}/class_names.json`)),
      ]);

      if (!classResponse.ok) {
        throw new Error(`Unable to load model classes (${classResponse.status}).`);
      }

      model = loadedModel;
      const contract = await classResponse.json();
      classNames = validateModelContract(model, contract);
      activationModel = buildActivationModel(model);

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

  function buildActivationModel(loadedModel) {
    try {
      return tf.model({
        inputs: loadedModel.inputs,
        outputs: ACTIVATION_LAYERS.map((name) => loadedModel.getLayer(name).output),
      });
    } catch (error) {
      // Older or differently named exports simply skip the visualization.
      console.warn("Layer activations are unavailable for this model.", error);
      return null;
    }
  }

  function extractTopFeatureMaps(data, height, width, channels) {
    const area = height * width;
    const channelMeans = new Float32Array(channels);
    for (let index = 0; index < data.length; index += 1) {
      channelMeans[index % channels] += data[index];
    }

    const rankedChannels = Array.from(channelMeans.keys()).sort(
      (a, b) => channelMeans[b] - channelMeans[a],
    );

    const maps = rankedChannels.slice(0, FEATURE_MAPS_PER_BLOCK).map((channel) => {
      const values = new Array(area);
      let max = 0;
      for (let pixel = 0; pixel < area; pixel += 1) {
        const value = data[pixel * channels + channel];
        values[pixel] = value;
        if (value > max) max = value;
      }
      return { channel, values, max };
    });

    return { height, width, channels, maps };
  }

  async function publishActivations(inputBuffer, outputs, requestId) {
    const blockTensors = outputs.slice(0, 3);
    const [pooledTensor, embeddingTensor] = outputs.slice(3, 5);
    const [blockData, pooledData, embeddingData] = await Promise.all([
      Promise.all(blockTensors.map((tensor) => tensor.data())),
      pooledTensor.data(),
      embeddingTensor.data(),
    ]);

    if (requestId !== latestRequest) return;

    activations.value = {
      input: Array.from(inputBuffer),
      blocks: blockTensors.map((tensor, index) => {
        const [, height, width, channels] = tensor.shape;
        return extractTopFeatureMaps(blockData[index], height, width, channels);
      }),
      pooled: Array.from(pooledData),
      embedding: Array.from(embeddingData),
    };
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
      output = activationModel
        ? activationModel.predict(inputTensor)
        : model.predict(inputTensor);
      const logits = activationModel ? output[output.length - 1] : getLogits(output);

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

      if (activationModel) {
        await publishActivations(inputBuffer, output, requestId);
      }
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
    activations.value = null;
  }

  return {
    status,
    predictions,
    inferenceTime,
    errorMessage,
    activations,
    isReady,
    loadModel,
    classify,
    resetPrediction,
  };
}
