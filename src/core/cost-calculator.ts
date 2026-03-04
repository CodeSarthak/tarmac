import type {
  ModelCostEstimate,
  ContextEstimate,
  OutputEstimate,
} from "../types.js";
import { MODEL_PRICING } from "../data/pricing.js";
import { getTaskProfile } from "./prompt-classifier.js";
import type { TaskType } from "../types.js";

/**
 * Model-specific loop multipliers calibrated from SWE-bench data.
 *
 * SWE-bench shows:
 *   Opus: ~30 API calls per task (avg)
 *   Sonnet: ~50 API calls per task (avg)
 *   Haiku: ~70+ API calls per task (estimated)
 *
 * Sonnet/Haiku need more iterations because they're less capable per-call.
 */
const MODEL_LOOP_MULTIPLIER: Record<string, number> = {
  "claude-sonnet-4-6": 3.0,
  "claude-opus-4-6": 1.0,
  "claude-haiku-4-5-20251001": 6.0,
};

/**
 * After the first API call, most of the context is served from cache.
 * Claude Code uses prompt caching aggressively.
 *
 * Empirically, cache hit rate is 80-90% after the first call.
 * Cache reads are 10x cheaper than full input.
 */
const CACHE_HIT_RATE_AFTER_FIRST = 0.85;

export function calculateCosts(
  context: ContextEstimate,
  output: OutputEstimate,
  taskType: TaskType
): ModelCostEstimate[] {
  const profile = getTaskProfile(taskType);

  return MODEL_PRICING.map((pricing) => {
    const loopMultiplier = MODEL_LOOP_MULTIPLIER[pricing.modelId] || 1.0;

    const loopLow = Math.max(1, Math.round(output.estimatedLoops[0] * loopMultiplier));
    const loopMid = Math.max(1, Math.round(
      ((output.estimatedLoops[0] + output.estimatedLoops[1]) / 2) * loopMultiplier
    ));
    const loopHigh = Math.max(1, Math.round(output.estimatedLoops[1] * loopMultiplier));

    const contextGrowthLow = profile.contextGrowthPerLoop[0];
    const contextGrowthMid = Math.round(
      (profile.contextGrowthPerLoop[0] + profile.contextGrowthPerLoop[1]) / 2
    );
    const contextGrowthHigh = profile.contextGrowthPerLoop[1];

    const costLow = calculateMultiLoopCost(
      context.estimatedNextInput,
      output.p25,
      loopLow,
      contextGrowthLow,
      pricing.inputPerMillion,
      pricing.outputPerMillion,
      pricing.cacheReadPerMillion
    );

    const costMid = calculateMultiLoopCost(
      context.estimatedNextInput,
      output.p50,
      loopMid,
      contextGrowthMid,
      pricing.inputPerMillion,
      pricing.outputPerMillion,
      pricing.cacheReadPerMillion
    );

    const costHigh = calculateMultiLoopCost(
      context.estimatedNextInput,
      output.p75,
      loopHigh,
      contextGrowthHigh,
      pricing.inputPerMillion,
      pricing.outputPerMillion,
      pricing.cacheReadPerMillion
    );

    return {
      model: pricing.model,
      modelId: pricing.modelId,
      costLow: roundCost(costLow),
      costMid: roundCost(costMid),
      costHigh: roundCost(costHigh),
    };
  });
}

function calculateMultiLoopCost(
  initialContext: number,
  totalOutputTokens: number,
  loops: number,
  contextGrowthPerLoop: number,
  inputPricePerMillion: number,
  outputPricePerMillion: number,
  cacheReadPricePerMillion: number
): number {
  if (loops <= 0) loops = 1;

  const outputPerLoop = Math.round(totalOutputTokens / loops);
  let totalCost = 0;
  let currentContext = initialContext;

  for (let i = 0; i < loops; i++) {
    let inputCost: number;

    if (i === 0) {
      // First call: full input price (no cache yet)
      inputCost = (currentContext * inputPricePerMillion) / 1_000_000;
    } else {
      // Subsequent calls: most context is cached
      const cachedTokens = currentContext * CACHE_HIT_RATE_AFTER_FIRST;
      const uncachedTokens = currentContext * (1 - CACHE_HIT_RATE_AFTER_FIRST);
      inputCost =
        (cachedTokens * cacheReadPricePerMillion) / 1_000_000 +
        (uncachedTokens * inputPricePerMillion) / 1_000_000;
    }

    const outputCost = (outputPerLoop * outputPricePerMillion) / 1_000_000;
    totalCost += inputCost + outputCost;

    // Context grows each loop
    currentContext += outputPerLoop + contextGrowthPerLoop;
  }

  return totalCost;
}

function roundCost(cost: number): number {
  if (cost < 0.01) return Math.round(cost * 1000) / 1000;
  return Math.round(cost * 100) / 100;
}
