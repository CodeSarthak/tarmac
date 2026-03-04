/**
 * Conformal Cost Predictor
 *
 * Replaces the heuristic-based cost estimation with a trained regression model
 * + conformal prediction intervals. This gives us:
 *   - Calibrated coverage guarantees (80% interval actually captures ~80%)
 *   - Per-model regression for tighter intervals
 *   - Data-driven ranges instead of hand-tuned multipliers
 *
 * The model was trained on 3,000 SWE-bench instances (see train-model.ts).
 * At runtime, no ML libraries are needed — just feature extraction + dot product.
 */

import { MODEL_WEIGHTS } from "../data/model-weights.js";
import { MODEL_PRICING } from "../data/pricing.js";

export interface ConformalEstimate {
  model: string;
  modelId: string;
  /** Point prediction (median) in dollars */
  predicted: number;
  /** Low end of the confidence interval */
  costLow: number;
  /** Midpoint of the confidence interval */
  costMid: number;
  /** High end of the confidence interval */
  costHigh: number;
  /** Target coverage level used (e.g., 0.80) */
  coverageTarget: number;
  /** The multiplier used (e.g., 2.3x means range is predicted/2.3 to predicted*2.3) */
  rangeMultiplier: number;
}

// Map from modelId to our training model key
const MODEL_KEY_MAP: Record<string, string> = {
  "claude-opus-4-6": "opus",
  "claude-sonnet-4-6": "sonnet",
  "claude-haiku-4-5-20251001": "haiku",
};

/**
 * Extract features from prompt text.
 * MUST match train-model.ts feature extraction exactly.
 */
export function extractPromptFeatures(text: string): number[] {
  const lower = text.toLowerCase();
  const words = text.split(/\s+/).filter(w => w.length > 0);
  const uniqueWords = new Set(words.map(w => w.toLowerCase()));
  const lines = text.split("\n");

  const features: Record<string, number> = {
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

  // Return in the exact order expected by the model
  return MODEL_WEIGHTS.featureNames.map(name => features[name] ?? 0);
}

/**
 * Predict log10(cost) using the regression model.
 */
function predictLogCost(beta: number[], features: number[]): number {
  // beta[0] is intercept, beta[1..n] are feature weights
  let pred = beta[0];
  for (let i = 0; i < features.length; i++) {
    pred += beta[i + 1] * features[i];
  }
  return pred;
}

/**
 * Get conformal cost estimate for a prompt across all models.
 *
 * @param promptText - The user's prompt text
 * @param coverageTarget - Desired coverage level (default 0.80 = 80%)
 * @returns Array of estimates, one per model
 */
export function getConformalEstimates(
  promptText: string,
  coverageTarget: number = 0.80
): ConformalEstimate[] {
  const features = extractPromptFeatures(promptText);
  const coverageKey = String(Math.round(coverageTarget * 100));

  return MODEL_PRICING.map((pricing) => {
    const modelKey = MODEL_KEY_MAP[pricing.modelId];

    // Try per-model weights first, fall back to global
    const weights = modelKey && MODEL_WEIGHTS.models[modelKey]
      ? MODEL_WEIGHTS.models[modelKey]
      : null;

    let logCostPred: number;
    let qhat: number;

    if (weights) {
      // Per-model prediction (uses text features only, no model dummies)
      logCostPred = predictLogCost(weights.beta, features);
      qhat = weights.conformalQuantiles[coverageKey]
        ?? weights.conformalQuantiles["80"]
        ?? 0.4; // fallback
    } else {
      // Global prediction (uses text features + model dummies)
      const globalFeatures = [
        ...features,
        pricing.modelId === "claude-opus-4-6" ? 1 : 0,
        pricing.modelId === "claude-sonnet-4-6" ? 1 : 0,
        pricing.modelId === "claude-haiku-4-5-20251001" ? 1 : 0,
      ];
      logCostPred = predictLogCost(MODEL_WEIGHTS.global.beta, globalFeatures);
      qhat = MODEL_WEIGHTS.global.conformalQuantiles[coverageKey]
        ?? MODEL_WEIGHTS.global.conformalQuantiles["80"]
        ?? 0.4;
    }

    // Convert from log10-space to dollar-space
    const predicted = 10 ** logCostPred;
    const costLow = 10 ** (logCostPred - qhat);
    const costHigh = 10 ** (logCostPred + qhat);
    const costMid = predicted; // point prediction IS the midpoint in log-space
    const rangeMultiplier = 10 ** qhat;

    return {
      model: pricing.model,
      modelId: pricing.modelId,
      predicted: roundCost(predicted),
      costLow: roundCost(costLow),
      costMid: roundCost(costMid),
      costHigh: roundCost(costHigh),
      coverageTarget,
      rangeMultiplier: Math.round(rangeMultiplier * 100) / 100,
    };
  });
}

function roundCost(cost: number): number {
  if (cost < 0.01) return Math.round(cost * 1000) / 1000;
  return Math.round(cost * 100) / 100;
}
