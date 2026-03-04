import { readFileSync } from "fs";
import {
  loadConfig,
  loadLastEstimate,
  buildTelemetryPayload,
  postTelemetry,
} from "../core/telemetry.js";
import { appendHistory } from "../core/history-analyzer.js";
import type { HistoryEntry, HookInput, LastEstimate } from "../types.js";
import { MODEL_PRICING } from "../data/pricing.js";

export async function runReport(): Promise<void> {
  try {
    // Read hook input from stdin
    const input = await readStdin();
    const hookInput = parseHookInput(input);

    const lastEstimate = loadLastEstimate();
    if (!lastEstimate) {
      // No estimate to compare against — nothing to do
      return;
    }

    // Parse transcript for actual outcome
    const outcome = extractOutcome(hookInput.transcript_path);
    if (!outcome) return;

    // Build history entry for Tier 2
    const historyEntry: HistoryEntry = {
      timestamp: new Date().toISOString(),
      taskType: lastEstimate.classification.taskType,
      promptSnippet: "", // We don't store prompts in report, just classification
      actualLoops: outcome.loops,
      actualInputTokens: outcome.totalInputTokens,
      actualOutputTokens: outcome.totalOutputTokens,
      actualCost: outcome.estimatedCost,
      model: outcome.model,
      initialContext: outcome.initialContext,
      finalContext: outcome.finalContext,
    };

    // Always save locally for Tier 2
    appendHistory(historyEntry);

    // Build and output the comparison
    const comparison = formatComparison(lastEstimate, outcome);
    if (comparison) {
      const output = {
        hookSpecificOutput: {
          hookEventName: "Stop",
          additionalContext: comparison,
        },
      };
      process.stdout.write(JSON.stringify(output));
    }

    // Optionally post telemetry
    const config = loadConfig();
    if (config.telemetryOptIn) {
      const payload = buildTelemetryPayload(
        lastEstimate,
        outcome.loops,
        outcome.totalOutputTokens,
        outcome.initialContext,
        outcome.finalContext,
        outcome.durationSeconds,
        outcome.model
      );
      await postTelemetry(payload);
    }
  } catch (err) {
    // Report should never fail loudly
    process.stderr.write(
      `[tarmac] report error: ${err instanceof Error ? err.message : String(err)}\n`
    );
  }
}

interface SessionOutcome {
  loops: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  estimatedCost: number;
  model: string;
  initialContext: number;
  finalContext: number;
  durationSeconds: number;
}

function extractOutcome(
  transcriptPath: string | undefined
): SessionOutcome | null {
  if (!transcriptPath) return null;

  try {
    const content = readFileSync(transcriptPath, "utf-8");
    const lines = content.trim().split("\n").filter(Boolean);

    let loops = 0;
    let totalInputTokens = 0;
    let totalOutputTokens = 0;
    let model = "unknown";
    let initialContext = 0;
    let finalContext = 0;
    let firstTimestamp: number | null = null;
    let lastTimestamp: number | null = null;

    for (const line of lines) {
      try {
        const entry = JSON.parse(line);

        // Track timestamps for duration
        if (entry.timestamp) {
          const ts = new Date(entry.timestamp).getTime();
          if (!firstTimestamp) firstTimestamp = ts;
          lastTimestamp = ts;
        }

        if (entry.type === "assistant" && entry.message?.usage) {
          const usage = entry.message.usage;
          totalInputTokens += usage.input_tokens || 0;
          totalOutputTokens += usage.output_tokens || 0;
          loops++;

          if (loops === 1) {
            initialContext = usage.input_tokens || 0;
          }
          finalContext = usage.input_tokens || 0;

          if (entry.message.model) {
            model = entry.message.model;
          }
        }
      } catch {
        continue;
      }
    }

    if (loops === 0) return null;

    // Calculate approximate cost
    const pricing = MODEL_PRICING.find((p) => model.includes(p.modelId)) ||
      MODEL_PRICING[0];
    const estimatedCost =
      (totalInputTokens * pricing.inputPerMillion) / 1_000_000 +
      (totalOutputTokens * pricing.outputPerMillion) / 1_000_000;

    const durationSeconds =
      firstTimestamp && lastTimestamp
        ? Math.round((lastTimestamp - firstTimestamp) / 1000)
        : 0;

    return {
      loops,
      totalInputTokens,
      totalOutputTokens,
      estimatedCost,
      model,
      initialContext,
      finalContext,
      durationSeconds,
    };
  } catch {
    return null;
  }
}

function readStdin(): Promise<string> {
  return new Promise((resolve) => {
    let data = "";
    const timeout = setTimeout(() => resolve(data || "{}"), 5000);
    process.stdin.setEncoding("utf-8");
    process.stdin.on("data", (chunk) => {
      data += chunk;
    });
    process.stdin.on("end", () => {
      clearTimeout(timeout);
      resolve(data);
    });
    process.stdin.on("error", () => {
      clearTimeout(timeout);
      resolve("{}");
    });
  });
}

function formatComparison(
  estimate: LastEstimate,
  outcome: SessionOutcome
): string | null {
  const actualCost = outcome.estimatedCost;
  if (actualCost <= 0) return null;

  // Find the estimate for the model that was actually used
  const modelEstimate = estimate.models.find((m) =>
    outcome.model.includes(m.modelId)
  ) || estimate.models[0];

  const inRange =
    actualCost >= modelEstimate.costLow && actualCost <= modelEstimate.costHigh;

  const lines: string[] = [];
  lines.push("📊 TARMAC SESSION REPORT");
  lines.push("━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  lines.push("");
  lines.push(`  Model:      ${modelEstimate.model}`);
  lines.push(`  Estimated:  ${fmtDollars(modelEstimate.costLow)} - ${fmtDollars(modelEstimate.costHigh)}`);
  lines.push(`  Actual:     ${fmtDollars(actualCost)}`);
  lines.push(`  Result:     ${inRange ? "✅ Within estimate" : "❌ Outside estimate"}`);
  lines.push("");
  lines.push(`  API calls:  ${outcome.loops}`);
  lines.push(`  Duration:   ${outcome.durationSeconds}s`);
  lines.push("");
  lines.push("━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  lines.push("");
  lines.push(
    "Show this session cost report to the user so they can see how the estimate compared to actual spend."
  );

  return lines.join("\n");
}

function fmtDollars(amount: number): string {
  if (amount < 0.01) return `$${amount.toFixed(3)}`;
  return `$${amount.toFixed(2)}`;
}

function parseHookInput(raw: string): HookInput {
  try {
    const parsed = JSON.parse(raw);
    return {
      hook_event_name: parsed.hook_event_name || "Stop",
      transcript_path: parsed.transcript_path || undefined,
      session_id: parsed.session_id || undefined,
    };
  } catch {
    return { hook_event_name: "Stop" };
  }
}
