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
  contract.input?.source !== "full-canvas" ||
  contract.input?.foreground !== 1 ||
  contract.input?.background !== 0
) {
  fail("the frontend requires a full-canvas, ink-as-one input contract.");
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

console.log(
  `Model contract valid: ${outputUnits} outputs, ${classNames.length} ordered labels, and complete UI coverage.`,
);
