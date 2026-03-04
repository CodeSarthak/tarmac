#!/usr/bin/env npx tsx
/**
 * Train the conformal prediction model and export weights.
 *
 * This runs OFFLINE during development. It produces:
 *   src/data/model-weights.ts — regression coefficients + conformal quantiles
 *
 * The trained model is then used at runtime by the cost calculator.
 * No ML libraries needed at runtime — just multiply features × weights.
 */

import { readFileSync, writeFileSync, existsSync } from "fs";
import { join } from "path";

// ==========================================
// Data Loading (same as signal-analysis.ts)
// ==========================================

interface Instance {
  instanceId: string;
  repo: string;
  model: string;
  cost: number;
  apiCalls: number;
  resolved: boolean;
  statement: string;
}

function loadData(): Instance[] {
  const raw = JSON.parse(readFileSync(join(process.cwd(), "data-swebench.json"), "utf-8"));
  const bashOnly = raw.leaderboards.find((b: { name: string }) => b.name === "bash-only");
  const statements = existsSync(join(process.cwd(), "data-swebench-statements.json"))
    ? JSON.parse(readFileSync(join(process.cwd(), "data-swebench-statements.json"), "utf-8"))
    : {};

  const modelMap: Record<string, string> = {
    "Claude 4.5 Opus": "opus",
    "Claude Opus 4.6": "opus",
    "Claude 4.5 Sonnet": "sonnet",
    "Claude 4.5 Haiku": "haiku",
  };

  const instances: Instance[] = [];
  for (const result of bashOnly.results) {
    let model = "";
    for (const [pattern, m] of Object.entries(modelMap)) {
      if ((result.name || "").includes(pattern)) { model = m; break; }
    }
    if (!model) continue;
    const details = result.per_instance_details || {};
    for (const [instId, d] of Object.entries(details)) {
      const dd = d as any;
      if (dd.cost <= 0) continue;
      instances.push({
        instanceId: instId, repo: instId.split("__")[0], model,
        cost: dd.cost, apiCalls: dd.api_calls, resolved: dd.resolved,
        statement: statements[instId] || "",
      });
    }
  }
  return instances;
}

// ==========================================
// Feature Extraction
// ==========================================

// These are the features we use at runtime. Must match exactly what
// the production code extracts from a user's prompt.
const FEATURE_NAMES = [
  "logCharCount",
  "wordCount",
  "lineCount",
  "sentenceCount",
  "codeBlockCount",
  "filePathCount",
  "functionNameCount",
  "classNameCount",
  "hasStackTrace",
  "hasErrorMsg",
  "vocabRichness",
  "technicalDensity",
  "avgLineLength",
  "maxLineLength",
  "mentionsFix",
  "mentionsAdd",
  "mentionsRefactor",
  "mentionsTest",
  "mentionsDeprecation",
  "mentionsRegression",
  "mentionsPerformance",
  "questionCount",
  "urlCount",
  "codeRefCount",
] as const;

type FeatureName = typeof FEATURE_NAMES[number];

function extractFeatures(text: string): Record<FeatureName, number> {
  const lower = text.toLowerCase();
  const words = text.split(/\s+/).filter(w => w.length > 0);
  const uniqueWords = new Set(words.map(w => w.toLowerCase()));
  const lines = text.split("\n");

  return {
    logCharCount: text.length > 0 ? Math.log10(text.length) : 0,
    wordCount: words.length,
    lineCount: lines.length,
    sentenceCount: (text.match(/[.!?]+/g) || []).length,
    codeBlockCount: (text.match(/```/g) || []).length / 2,
    filePathCount: (text.match(/[\w\-./]+\.(py|js|ts|java|go|rb|rs|cpp|c|h|md|json|yaml|yml|toml|cfg)/g) || []).length,
    functionNameCount: (text.match(/\b[a-z_]\w*\s*\(/g) || []).length,
    classNameCount: (text.match(/\bclass\s+[A-Z]\w+/g) || []).length,
    hasStackTrace: /traceback|exception|stack trace/i.test(text) ? 1 : 0,
    hasErrorMsg: /error|bug|fail|crash|broke|wrong|unexpected/i.test(text) ? 1 : 0,
    vocabRichness: words.length > 0 ? uniqueWords.size / words.length : 0,
    technicalDensity: text.length > 0 ? (text.match(/[{}\[\]()<>:;=+\-*\/|&!@#$%^~`]/g) || []).length / text.length : 0,
    avgLineLength: lines.length > 0 ? lines.reduce((a, l) => a + l.length, 0) / lines.length : 0,
    maxLineLength: Math.max(...lines.map(l => l.length), 0),
    mentionsFix: /\bfix(es|ed|ing)?\b|\bbug\b|\bpatch\b/i.test(lower) ? 1 : 0,
    mentionsAdd: /\badd\b|\bimplement\b|\bcreate\b|\bintroduce\b|\bnew\b/i.test(lower) ? 1 : 0,
    mentionsRefactor: /\brefactor\b|\brestructure\b|\bclean\b|\bsimplif/i.test(lower) ? 1 : 0,
    mentionsTest: /\btest\b|\bspec\b|\bassert\b|\bverify\b/i.test(lower) ? 1 : 0,
    mentionsDeprecation: /\bdeprecate\b|\bremove\b|\bdrop\b|\bobsolete\b/i.test(lower) ? 1 : 0,
    mentionsRegression: /\bregression\b|\bused to\b|\bno longer\b|\bsince\b/i.test(lower) ? 1 : 0,
    mentionsPerformance: /\bperformance\b|\bslow\b|\boptimize\b|\bmemory\b/i.test(lower) ? 1 : 0,
    questionCount: (text.match(/\?/g) || []).length,
    urlCount: (text.match(/https?:\/\//g) || []).length,
    codeRefCount: (text.match(/`[^`]+`/g) || []).length,
  };
}

// ==========================================
// Linear Algebra
// ==========================================

function solveSystem(A: number[][], b: number[]): number[] {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let i = 0; i < n; i++) {
    let maxVal = Math.abs(M[i][i]), maxRow = i;
    for (let k = i + 1; k < n; k++)
      if (Math.abs(M[k][i]) > maxVal) { maxVal = Math.abs(M[k][i]); maxRow = k; }
    [M[i], M[maxRow]] = [M[maxRow], M[i]];
    if (Math.abs(M[i][i]) < 1e-12) continue;
    for (let k = i + 1; k < n; k++) {
      const f = M[k][i] / M[i][i];
      for (let j = i; j <= n; j++) M[k][j] -= f * M[i][j];
    }
  }
  const x = Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = M[i][n];
    for (let j = i + 1; j < n; j++) x[i] -= M[i][j] * x[j];
    x[i] /= M[i][i] || 1;
  }
  return x;
}

function ridgeRegression(X: number[][], y: number[], lambda: number = 0.01): number[] {
  const n = X.length, p = X[0].length;
  const Xa = X.map(row => [1, ...row]); // add intercept
  const pa = p + 1;
  const XtX: number[][] = Array.from({ length: pa }, () => Array(pa).fill(0));
  for (let i = 0; i < n; i++)
    for (let j = 0; j < pa; j++)
      for (let k = 0; k < pa; k++)
        XtX[j][k] += Xa[i][j] * Xa[i][k];
  const Xty: number[] = Array(pa).fill(0);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < pa; j++)
      Xty[j] += Xa[i][j] * y[i];
  for (let j = 1; j < pa; j++) XtX[j][j] += lambda; // don't regularize intercept
  return solveSystem(XtX, Xty);
}

function predict(beta: number[], features: number[]): number {
  return [1, ...features].reduce((sum, x, j) => sum + x * beta[j], 0);
}

// ==========================================
// Training
// ==========================================

interface ModelWeights {
  featureNames: string[];
  // Per-model regression weights (intercept + features)
  models: Record<string, {
    beta: number[];           // [intercept, feat1, feat2, ...]
    rSquared: number;
    conformalQuantiles: Record<string, number>;  // {"70": qhat, "80": qhat, ...}
    trainMeanLogCost: number;
    trainStdLogCost: number;
  }>;
  // Global regression (single model for all)
  global: {
    beta: number[];
    rSquared: number;
    conformalQuantiles: Record<string, number>;
  };
  trainingInfo: {
    nTrain: number;
    nCalibration: number;
    dataSource: string;
    timestamp: string;
  };
}

function train(): ModelWeights {
  console.log("Loading data...");
  const instances = loadData();
  console.log(`  ${instances.length} instances\n`);

  // Shuffle deterministically and split: 60% train, 20% calibration, 20% test
  const shuffled = [...instances].sort((a, b) =>
    (a.instanceId + a.model).localeCompare(b.instanceId + b.model)
  );
  const trainEnd = Math.floor(shuffled.length * 0.6);
  const calEnd = Math.floor(shuffled.length * 0.8);
  const trainSet = shuffled.slice(0, trainEnd);
  const calSet = shuffled.slice(trainEnd, calEnd);
  const testSet = shuffled.slice(calEnd);

  console.log(`  Train: ${trainSet.length}, Calibration: ${calSet.length}, Test: ${testSet.length}\n`);

  // Feature names for the models (text features + model dummies for global)
  const textFeatures = [...FEATURE_NAMES];
  const globalFeatures = [...textFeatures, "isOpus", "isSonnet", "isHaiku"];

  // ===== GLOBAL MODEL =====
  console.log("Training global model...");
  const globalTrainX = trainSet.map(inst => {
    const f = extractFeatures(inst.statement);
    return [...textFeatures.map(fn => f[fn]), inst.model === "opus" ? 1 : 0, inst.model === "sonnet" ? 1 : 0, inst.model === "haiku" ? 1 : 0];
  });
  const globalTrainY = trainSet.map(inst => Math.log10(inst.cost));
  const globalBeta = ridgeRegression(globalTrainX, globalTrainY);

  // R² on train
  const globalPreds = globalTrainX.map(row => predict(globalBeta, row));
  const yMean = globalTrainY.reduce((a, b) => a + b) / globalTrainY.length;
  const ssTot = globalTrainY.reduce((s, y) => s + (y - yMean) ** 2, 0);
  const ssRes = globalTrainY.reduce((s, y, i) => s + (y - globalPreds[i]) ** 2, 0);
  const globalR2 = 1 - ssRes / ssTot;
  console.log(`  R² = ${globalR2.toFixed(4)}`);

  // Calibration: compute residuals on cal set
  const globalCalResiduals = calSet.map(inst => {
    const f = extractFeatures(inst.statement);
    const x = [...textFeatures.map(fn => f[fn]), inst.model === "opus" ? 1 : 0, inst.model === "sonnet" ? 1 : 0, inst.model === "haiku" ? 1 : 0];
    const pred = predict(globalBeta, x);
    return Math.abs(pred - Math.log10(inst.cost));
  }).sort((a, b) => a - b);

  const globalQuantiles: Record<string, number> = {};
  for (const pct of [50, 60, 70, 75, 80, 85, 90, 95]) {
    const idx = Math.min(Math.ceil((globalCalResiduals.length + 1) * pct / 100) - 1, globalCalResiduals.length - 1);
    globalQuantiles[String(pct)] = globalCalResiduals[idx];
  }

  // Test set evaluation
  console.log("\n  Global model — test set evaluation:");
  for (const pct of [70, 80, 85, 90, 95]) {
    const qhat = globalQuantiles[String(pct)];
    let covered = 0;
    const widths: number[] = [];
    for (const inst of testSet) {
      const f = extractFeatures(inst.statement);
      const x = [...textFeatures.map(fn => f[fn]), inst.model === "opus" ? 1 : 0, inst.model === "sonnet" ? 1 : 0, inst.model === "haiku" ? 1 : 0];
      const pred = predict(globalBeta, x);
      const logCost = Math.log10(inst.cost);
      if (logCost >= pred - qhat && logCost <= pred + qhat) covered++;
      widths.push(10 ** (pred + qhat) - 10 ** (pred - qhat));
    }
    widths.sort((a, b) => a - b);
    const medWidth = widths[Math.floor(widths.length / 2)];
    console.log(`    ${pct}% target: ${(covered / testSet.length * 100).toFixed(1)}% actual, median width $${medWidth.toFixed(2)}, ×${(10 ** qhat).toFixed(2)} multiplier`);
  }

  // ===== PER-MODEL MODELS =====
  const modelWeights: Record<string, ModelWeights["models"][string]> = {};

  for (const model of ["opus", "sonnet", "haiku"]) {
    console.log(`\nTraining ${model} model...`);
    const mTrain = trainSet.filter(i => i.model === model);
    const mCal = calSet.filter(i => i.model === model);
    const mTest = testSet.filter(i => i.model === model);

    const mTrainX = mTrain.map(inst => {
      const f = extractFeatures(inst.statement);
      return textFeatures.map(fn => f[fn]);
    });
    const mTrainY = mTrain.map(inst => Math.log10(inst.cost));

    const mBeta = ridgeRegression(mTrainX, mTrainY);

    // R²
    const mPreds = mTrainX.map(row => predict(mBeta, row));
    const mYMean = mTrainY.reduce((a, b) => a + b) / mTrainY.length;
    const mSsTot = mTrainY.reduce((s, y) => s + (y - mYMean) ** 2, 0);
    const mSsRes = mTrainY.reduce((s, y, i) => s + (y - mPreds[i]) ** 2, 0);
    const mR2 = 1 - mSsRes / mSsTot;
    console.log(`  R² = ${mR2.toFixed(4)} (n_train=${mTrain.length})`);

    // Calibration
    const mCalResiduals = mCal.map(inst => {
      const f = extractFeatures(inst.statement);
      const x = textFeatures.map(fn => f[fn]);
      return Math.abs(predict(mBeta, x) - Math.log10(inst.cost));
    }).sort((a, b) => a - b);

    const mQuantiles: Record<string, number> = {};
    for (const pct of [50, 60, 70, 75, 80, 85, 90, 95]) {
      const idx = Math.min(Math.ceil((mCalResiduals.length + 1) * pct / 100) - 1, mCalResiduals.length - 1);
      mQuantiles[String(pct)] = mCalResiduals[idx];
    }

    // Mean/std for fallback
    const mLogCostMean = mTrainY.reduce((a, b) => a + b) / mTrainY.length;
    const mLogCostStd = Math.sqrt(mTrainY.reduce((s, y) => s + (y - mLogCostMean) ** 2, 0) / mTrainY.length);

    // Test evaluation
    console.log(`  Test set (n=${mTest.length}):`);
    for (const pct of [80, 90]) {
      const qhat = mQuantiles[String(pct)];
      let covered = 0;
      for (const inst of mTest) {
        const f = extractFeatures(inst.statement);
        const x = textFeatures.map(fn => f[fn]);
        const pred = predict(mBeta, x);
        const logCost = Math.log10(inst.cost);
        if (logCost >= pred - qhat && logCost <= pred + qhat) covered++;
      }
      console.log(`    ${pct}% target: ${(covered / mTest.length * 100).toFixed(1)}% actual, ×${(10 ** qhat).toFixed(2)} multiplier`);
    }

    modelWeights[model] = {
      beta: mBeta.map(b => Math.round(b * 100000) / 100000), // 5 decimal places
      rSquared: Math.round(mR2 * 10000) / 10000,
      conformalQuantiles: Object.fromEntries(
        Object.entries(mQuantiles).map(([k, v]) => [k, Math.round(v * 10000) / 10000])
      ),
      trainMeanLogCost: Math.round(mLogCostMean * 10000) / 10000,
      trainStdLogCost: Math.round(mLogCostStd * 10000) / 10000,
    };
  }

  const weights: ModelWeights = {
    featureNames: textFeatures,
    models: modelWeights,
    global: {
      beta: globalBeta.map(b => Math.round(b * 100000) / 100000),
      rSquared: Math.round(globalR2 * 10000) / 10000,
      conformalQuantiles: Object.fromEntries(
        Object.entries(globalQuantiles).map(([k, v]) => [k, Math.round(v * 10000) / 10000])
      ),
    },
    trainingInfo: {
      nTrain: trainSet.length,
      nCalibration: calSet.length,
      dataSource: "SWE-bench Verified (Claude models, bash-only track)",
      timestamp: new Date().toISOString(),
    },
  };

  return weights;
}

// ==========================================
// Export
// ==========================================

function exportWeights(weights: ModelWeights) {
  const code = `// AUTO-GENERATED by train-model.ts — do not edit manually
// Training data: ${weights.trainingInfo.dataSource}
// Date: ${weights.trainingInfo.timestamp}
// N_train: ${weights.trainingInfo.nTrain}, N_calibration: ${weights.trainingInfo.nCalibration}

export interface TrainedModelWeights {
  featureNames: string[];
  models: Record<string, {
    beta: number[];
    rSquared: number;
    conformalQuantiles: Record<string, number>;
    trainMeanLogCost: number;
    trainStdLogCost: number;
  }>;
  global: {
    beta: number[];
    rSquared: number;
    conformalQuantiles: Record<string, number>;
  };
}

export const MODEL_WEIGHTS: TrainedModelWeights = ${JSON.stringify(weights, (_, v) => {
    if (v === null || v === undefined) return v;
    // Keep trainingInfo out of the runtime export
    if (typeof v === 'object' && 'trainingInfo' in v) {
      const { trainingInfo, ...rest } = v;
      return rest;
    }
    return v;
  }, 2)};
`;

  const outPath = join(process.cwd(), "src", "data", "model-weights.ts");
  writeFileSync(outPath, code);
  console.log(`\nExported weights to ${outPath}`);
}

// ==========================================
// Main
// ==========================================

const weights = train();
exportWeights(weights);

console.log("\nDone. Now implement the conformal predictor in src/core/conformal-predictor.ts");
