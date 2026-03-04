import type { ModelPricing } from "../types.js";

export const MODEL_PRICING: ModelPricing[] = [
  {
    model: "Sonnet 4.6",
    modelId: "claude-sonnet-4-6",
    inputPerMillion: 3.0,
    outputPerMillion: 15.0,
    cacheReadPerMillion: 0.3,
    cacheWritePerMillion: 3.75,
  },
  {
    model: "Opus 4.6",
    modelId: "claude-opus-4-6",
    inputPerMillion: 15.0,
    outputPerMillion: 75.0,
    cacheReadPerMillion: 1.5,
    cacheWritePerMillion: 18.75,
  },
  {
    model: "Haiku 4.5",
    modelId: "claude-haiku-4-5-20251001",
    inputPerMillion: 0.8,
    outputPerMillion: 4.0,
    cacheReadPerMillion: 0.08,
    cacheWritePerMillion: 1.0,
  },
];

export function getPricing(modelId: string): ModelPricing | undefined {
  return MODEL_PRICING.find((m) => m.modelId === modelId);
}
