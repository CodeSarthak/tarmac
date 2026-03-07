import { estimateContext } from "../core/context-estimator.js";
import { countTokens } from "../core/token-counter.js";
import { classifyPrompt } from "../core/prompt-classifier.js";
import { getConformalEstimates } from "../core/conformal-predictor.js";
import { formatEstimate } from "../core/formatter.js";
import { loadConfig, saveLastEstimate } from "../core/telemetry.js";
import type { HookInput, HookOutput, CostEstimate } from "../types.js";

export async function runEstimate(): Promise<void> {
  try {
    // Read hook input from stdin
    const input = await readStdin();
    const hookInput = parseHookInput(input);

    if (!hookInput.prompt) {
      // No prompt — pass through silently
      writeHookOutput(null);
      return;
    }

    const prompt = hookInput.prompt;

    // Skip estimate for very short prompts (< 5 words) — likely confirmations/follow-ups
    const wordCount = prompt.trim().split(/\s+/).filter(w => w.length > 0).length;
    if (wordCount < 5) {
      writeHookOutput(null);
      return;
    }

    const config = loadConfig();

    // Step 1: Estimate context from transcript
    const contextEstimate = estimateContext(
      hookInput.transcript_path,
      prompt
    );

    // Step 2: Count prompt tokens (exact via API, fallback to heuristic)
    const inputTokens = await countTokens(prompt);

    // Step 3: Classify the prompt (for task type label)
    const classification = classifyPrompt(prompt);

    // Step 4: Conformal prediction — trained regression + calibrated intervals
    // Uses 80% coverage target (captures ~80% of actual costs)
    const conformalEstimates = getConformalEstimates(prompt, 0.80);

    // Step 5: Build the full estimate
    const estimate: CostEstimate = {
      models: conformalEstimates,
      classification,
      contextEstimate,
      outputEstimate: {
        p25: 0, p50: 0, p75: 0, p95: 0,
        estimatedLoops: [0, 0],
        confidence: classification.confidence,
        tiersUsed: ["conformal-regression"],
      },
      inputTokens,
    };

    // Step 7: Save for later comparison (report command)
    saveLastEstimate(estimate, hookInput.session_id || "unknown");

    // Step 8: Format and output
    const formatted = formatEstimate(estimate);
    writeHookOutput(formatted);
  } catch (err) {
    // Never block the user — if anything fails, pass through silently
    process.stderr.write(
      `[tarmac] estimate error: ${err instanceof Error ? err.message : String(err)}\n`
    );
    writeHookOutput(null);
  }
}

function readStdin(): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = "";

    // Set a timeout — don't hang forever
    const timeout = setTimeout(() => {
      resolve(data || "{}");
    }, 5000);

    process.stdin.setEncoding("utf-8");
    process.stdin.on("data", (chunk) => {
      data += chunk;
    });
    process.stdin.on("end", () => {
      clearTimeout(timeout);
      resolve(data);
    });
    process.stdin.on("error", (err) => {
      clearTimeout(timeout);
      reject(err);
    });
  });
}

function parseHookInput(raw: string): HookInput {
  try {
    const parsed = JSON.parse(raw);
    return {
      hook_event_name: parsed.hook_event_name || "UserPromptSubmit",
      prompt: parsed.prompt || parsed.message || undefined,
      transcript_path: parsed.transcript_path || undefined,
      session_id: parsed.session_id || undefined,
      cwd: parsed.cwd || undefined,
    };
  } catch {
    return {
      hook_event_name: "UserPromptSubmit",
    };
  }
}

function writeHookOutput(additionalContext: string | null): void {
  const output: HookOutput = {};

  if (additionalContext) {
    output.hookSpecificOutput = {
      hookEventName: "UserPromptSubmit",
      additionalContext,
    };
  }

  process.stdout.write(JSON.stringify(output));
}
