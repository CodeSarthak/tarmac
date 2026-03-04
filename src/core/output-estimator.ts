import type {
  ClassificationResult,
  OutputEstimate,
  ContextEstimate,
  TarmacConfig,
} from "../types.js";
import { getTaskProfile, TASK_PROFILES } from "./prompt-classifier.js";
import { getHistoryAdjustment } from "./history-analyzer.js";
import { runHaikuPreflight, shouldRunHaiku } from "./haiku-preflight.js";
import { MODEL_PRICING } from "../data/pricing.js";

interface Tier1Estimate {
  p25: number;
  p50: number;
  p75: number;
  p95: number;
  loops: [number, number];
}

export async function estimateOutput(
  classification: ClassificationResult,
  context: ContextEstimate,
  promptText: string,
  config: TarmacConfig
): Promise<OutputEstimate> {
  const tiersUsed: string[] = [];

  // Tier 1: Feature-based classification (always runs)
  const tier1 = getTier1Estimate(classification, context);
  tiersUsed.push("feature-classification");
  let confidence = classification.confidence;

  let finalEstimate: Tier1Estimate = { ...tier1 };

  // Tier 2: Historical pattern matching
  const historyAdj = getHistoryAdjustment(classification, promptText);

  if (historyAdj.adjustedLoops && historyAdj.adjustedOutputPerLoop) {
    tiersUsed.push(`history(n=${historyAdj.sampleCount})`);

    // Blend Tier 1 + Tier 2 based on sample count
    const historyWeight = Math.min(historyAdj.sampleCount / 20, 0.6);
    const tier1Weight = 1 - historyWeight;

    const adjustedTotal = calculateTotalOutput(
      historyAdj.adjustedLoops,
      historyAdj.adjustedOutputPerLoop,
      context
    );

    finalEstimate = {
      p25: Math.round(tier1.p25 * tier1Weight + adjustedTotal.p25 * historyWeight),
      p50: Math.round(tier1.p50 * tier1Weight + adjustedTotal.p50 * historyWeight),
      p75: Math.round(tier1.p75 * tier1Weight + adjustedTotal.p75 * historyWeight),
      p95: Math.round(tier1.p95 * tier1Weight + adjustedTotal.p95 * historyWeight),
      loops: [
        Math.round(
          tier1.loops[0] * tier1Weight +
            historyAdj.adjustedLoops[0] * historyWeight
        ),
        Math.round(
          tier1.loops[1] * tier1Weight +
            historyAdj.adjustedLoops[1] * historyWeight
        ),
      ],
    };

    if (historyAdj.confidence === "high") confidence = "high";
    else if (confidence === "low") confidence = "medium";
  }

  // Tier 3: Haiku pre-flight (conditional)
  // Quick cost estimate for gating
  const midCostEstimate = quickCostEstimate(finalEstimate.p50, context);

  if (
    shouldRunHaiku(
      confidence,
      midCostEstimate,
      config.haikuPreflightEnabled,
      config.haikuCostThreshold
    )
  ) {
    const haikuResult = await runHaikuPreflight(promptText);

    if (haikuResult) {
      tiersUsed.push("haiku-preflight");

      // Determine blend weights based on history depth
      let haikuWeight: number;
      let histWeight: number;
      let featWeight: number;

      if (historyAdj.sampleCount >= 100) {
        haikuWeight = 0.2;
        histWeight = 0.5;
        featWeight = 0.3;
      } else if (historyAdj.sampleCount >= 20) {
        haikuWeight = 0.3;
        histWeight = 0.3;
        featWeight = 0.4;
      } else {
        haikuWeight = 0.6;
        histWeight = 0;
        featWeight = 0.4;
      }

      const haikuTotal = {
        p25: haikuResult.estimatedOutputTokens[0],
        p50: Math.round(
          (haikuResult.estimatedOutputTokens[0] +
            haikuResult.estimatedOutputTokens[1]) /
            2
        ),
        p75: haikuResult.estimatedOutputTokens[1],
        p95: Math.round(haikuResult.estimatedOutputTokens[1] * 1.3),
      };

      // Re-blend with all three tiers
      const blendedEstimate = {
        p25: Math.round(
          finalEstimate.p25 * (featWeight + histWeight) +
            haikuTotal.p25 * haikuWeight
        ),
        p50: Math.round(
          finalEstimate.p50 * (featWeight + histWeight) +
            haikuTotal.p50 * haikuWeight
        ),
        p75: Math.round(
          finalEstimate.p75 * (featWeight + histWeight) +
            haikuTotal.p75 * haikuWeight
        ),
        p95: Math.round(
          finalEstimate.p95 * (featWeight + histWeight) +
            haikuTotal.p95 * haikuWeight
        ),
      };

      finalEstimate = {
        ...blendedEstimate,
        loops: [
          Math.round(
            finalEstimate.loops[0] * (1 - haikuWeight) +
              haikuResult.estimatedLoops[0] * haikuWeight
          ),
          Math.round(
            finalEstimate.loops[1] * (1 - haikuWeight) +
              haikuResult.estimatedLoops[1] * haikuWeight
          ),
        ],
      };

      if (confidence === "low") confidence = "medium";
    }
  }

  return {
    ...finalEstimate,
    estimatedLoops: finalEstimate.loops,
    confidence,
    tiersUsed,
  };
}

function getTier1Estimate(
  classification: ClassificationResult,
  context: ContextEstimate
): Tier1Estimate {
  const profile = getTaskProfile(classification.taskType);

  const [loopMin, loopMax] = profile.loopRange;
  const [outMin, outMax] = profile.outputPerLoop;

  // Adjust for features — calibrated against 2,500 SWE-bench instances
  let loopMultiplier = 1.0;
  let outAdjust = 1.0;

  const features = classification.features;

  // Prompt length scales complexity (r=0.127 from SWE-bench data)
  // Longer prompts → more context → more iterations
  const promptLen = features.promptLength || 0;
  if (promptLen > 2000) {
    loopMultiplier *= 1.15; // Long detailed tasks
  } else if (promptLen > 500) {
    loopMultiplier *= 1.05; // Medium tasks
  } else if (promptLen < 100) {
    loopMultiplier *= 0.7; // Very short → probably simple
  }

  // Scope keywords increase loops
  if (features.scopeScore > 0) {
    loopMultiplier *= 1.1;
  }

  // Many files mentioned
  if (features.fileCount > 3) {
    loopMultiplier *= 1.1;
  }

  // Test signal means verify + fix cycles
  if (features.testSignal) {
    loopMultiplier *= 1.15;
  }

  // Breadth adjusts output slightly
  if (features.breadthScore > 0) {
    outAdjust *= 1.0 + features.breadthScore * 0.1;
  } else if (features.breadthScore < 0) {
    outAdjust *= 0.8;
  }

  // Questions get shorter answers
  if (features.isQuestion) {
    outAdjust *= 0.7;
  }

  // Constraints increase output slightly
  if (features.constraintCount > 0) {
    outAdjust *= 1.0 + features.constraintCount * 0.05;
  }

  const adjLoopMin = Math.max(1, Math.round(loopMin * loopMultiplier));
  const adjLoopMax = Math.max(adjLoopMin, Math.round(loopMax * loopMultiplier));
  const adjOutMin = Math.round(outMin * outAdjust);
  const adjOutMax = Math.round(outMax * outAdjust);

  const total = calculateTotalOutput(
    [adjLoopMin, adjLoopMax],
    [adjOutMin, adjOutMax],
    context
  );

  return {
    ...total,
    loops: [adjLoopMin, adjLoopMax],
  };
}

function calculateTotalOutput(
  loops: [number, number],
  outputPerLoop: [number, number],
  _context: ContextEstimate
): { p25: number; p50: number; p75: number; p95: number } {
  const [loopMin, loopMax] = loops;
  const [outMin, outMax] = outputPerLoop;

  // p25: conservative (min loops * min output)
  const p25 = loopMin * outMin;
  // p50: midpoint
  const midLoops = Math.round((loopMin + loopMax) / 2);
  const midOut = Math.round((outMin + outMax) / 2);
  const p50 = midLoops * midOut;
  // p75: high estimate (max loops * mid-high output)
  const p75 = loopMax * Math.round(outMax * 0.8);
  // p95: worst case (max loops * max output * 1.3)
  const p95 = Math.round(loopMax * outMax * 1.3);

  return { p25, p50, p75, p95 };
}

function quickCostEstimate(outputTokens: number, context: ContextEstimate): number {
  // Quick estimate using Sonnet pricing
  const sonnet = MODEL_PRICING[0];
  const inputCost =
    (context.estimatedNextInput * sonnet.inputPerMillion) / 1_000_000;
  const outputCost = (outputTokens * sonnet.outputPerMillion) / 1_000_000;
  return inputCost + outputCost;
}
