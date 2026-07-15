import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { categories } from "../src/data/categories.js";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const modelDirectory = path.join(projectRoot, "public", "tfjs_model");
const model = JSON.parse(
  fs.readFileSync(path.join(modelDirectory, "model.json"), "utf8"),
);
const contract = JSON.parse(
  fs.readFileSync(path.join(modelDirectory, "class_names.json"), "utf8"),
);
const evaluation = JSON.parse(
  fs.readFileSync(path.join(modelDirectory, "metrics.json"), "utf8"),
);

function fail(message) {
  throw new Error(`Model contract validation failed: ${message}`);
}

const classNames = contract.classNames;
if (!Array.isArray(classNames) || classNames.length === 0) {
  fail("class_names.json must contain a non-empty classNames array.");
}
if (classNames.some((name) => typeof name !== "string" || !name.trim())) {
  fail("every class name must be a non-empty string.");
}
if (new Set(classNames).size !== classNames.length) {
  fail("classNames contains duplicate labels.");
}

const modelConfig = model.modelTopology?.model_config?.config;
const outputLayerName = modelConfig?.output_layers?.[0]?.[0];
const outputLayer = modelConfig?.layers?.find(
  (layer) => layer.name === outputLayerName,
);
const outputUnits = outputLayer?.config?.units;
if (!Number.isInteger(outputUnits)) {
  fail("could not determine the model's output size.");
}
if (outputUnits !== classNames.length) {
  fail(`the model has ${outputUnits} outputs but ${classNames.length} labels.`);
}

const inputLayerName = modelConfig?.input_layers?.[0]?.[0];
const inputLayer = modelConfig?.layers?.find((layer) => layer.name === inputLayerName);
const modelInputShape = inputLayer?.config?.batch_input_shape?.slice(1);
const contractInputShape = [
  contract.input?.height,
  contract.input?.width,
  contract.input?.channels,
];
if (
  modelInputShape?.length !== contractInputShape.length ||
  modelInputShape.some(
    (dimension, index) => dimension !== contractInputShape[index],
  )
) {
  fail(
    `model input ${JSON.stringify(modelInputShape)} does not match contract ${JSON.stringify(contractInputShape)}.`,
  );
}
if (
  contract.input?.source !== "bounding-box-centered" ||
  contract.input?.foreground !== 1 ||
  contract.input?.background !== 0
) {
  fail("the frontend requires a centered-bounds, ink-as-one input contract.");
}

const displayedSet = new Set(categories);
const trainedSet = new Set(classNames);
if (displayedSet.size !== categories.length) {
  fail("the UI category groups contain duplicate labels.");
}

const missingFromUi = classNames.filter((name) => !displayedSet.has(name));
const notTrained = categories.filter((name) => !trainedSet.has(name));
if (missingFromUi.length || notTrained.length) {
  fail(
    `model/UI classes disagree (missing from UI: ${missingFromUi.join(", ") || "none"}; not trained: ${notTrained.join(", ") || "none"}).`,
  );
}

const evaluationMetrics = evaluation.metrics;
const evaluationSplit = evaluation.split;
if (
  evaluationSplit?.classCount !== classNames.length ||
  !Number.isInteger(evaluationSplit?.samplesPerClass) ||
  evaluationSplit.samplesPerClass <= 0 ||
  evaluationSplit?.sampleCount !==
    classNames.length * evaluationSplit.samplesPerClass
) {
  fail("metrics.json has an incompatible evaluation split.");
}
for (const name of ["top1Accuracy", "top3Accuracy", "macroRecall"]) {
  const value = evaluationMetrics?.[name];
  if (typeof value !== "number" || value < 0 || value > 1) {
    fail(`metrics.json has an invalid ${name}.`);
  }
}
if (
  typeof evaluationMetrics?.loss !== "number" ||
  evaluationMetrics.loss < 0 ||
  evaluationMetrics.top3Accuracy < evaluationMetrics.top1Accuracy
) {
  fail("metrics.json has invalid loss or accuracy ordering.");
}

console.log(
  `Model contract valid: ${outputUnits} outputs, ${classNames.length} ordered labels, complete UI coverage, and published evaluation metrics.`,
);
