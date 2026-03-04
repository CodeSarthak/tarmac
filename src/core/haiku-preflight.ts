import type { HaikuPreflightResult } from "../types.js";

const HAIKU_TIMEOUT = 5000; // 5s timeout

const META_PROMPT = `You are a cost estimation assistant for Claude Code (an AI coding agent).
Given the following task, predict WITHOUT executing it:

1. COMPLEXITY: simple | medium | complex | very_complex
2. ESTIMATED_LOOPS: Agent iterations needed (file reads, edits, test runs). Give [min, max].
3. FILES_TOUCHED: Files likely read or modified (count).
4. WILL_REQUIRE_TESTS: yes | no | maybe
5. ITERATION_RISK: low | medium | high (will first attempt work, or will debugging be needed?)
6. ESTIMATED_OUTPUT_TOKENS: Total across all loops. Give [min, max].

Respond ONLY in JSON with these exact keys: complexity, estimatedLoops, filesTouched, willRequireTests, iterationRisk, estimatedOutputTokens`;

export async function runHaikuPreflight(
  prompt: string
): Promise<HaikuPreflightResult | null> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return null;

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), HAIKU_TIMEOUT);

    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: "claude-haiku-4-5-20251001",
        max_tokens: 256,
        messages: [
          {
            role: "user",
            content: `${META_PROMPT}\n\nTask: "${prompt.slice(0, 1000)}"`,
          },
        ],
      }),
      signal: controller.signal,
    });

    clearTimeout(timeout);

    if (!response.ok) return null;

    const data = (await response.json()) as {
      content: Array<{ type: string; text: string }>;
    };

    const text = data.content?.[0]?.text;
    if (!text) return null;

    // Extract JSON from response (handle markdown code blocks)
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) return null;

    const parsed = JSON.parse(jsonMatch[0]);

    return {
      complexity: parsed.complexity || "medium",
      estimatedLoops: Array.isArray(parsed.estimatedLoops)
        ? parsed.estimatedLoops
        : [2, 5],
      filesTouched: parsed.filesTouched || 3,
      willRequireTests: parsed.willRequireTests || "maybe",
      iterationRisk: parsed.iterationRisk || "medium",
      estimatedOutputTokens: Array.isArray(parsed.estimatedOutputTokens)
        ? parsed.estimatedOutputTokens
        : [2000, 8000],
    };
  } catch {
    return null;
  }
}

export function shouldRunHaiku(
  confidence: "low" | "medium" | "high",
  estimatedCostMid: number,
  haikuEnabled: boolean,
  costThreshold: number
): boolean {
  if (!haikuEnabled) return false;
  if (confidence === "low") return true;
  if (estimatedCostMid > costThreshold) return true;
  return false;
}
