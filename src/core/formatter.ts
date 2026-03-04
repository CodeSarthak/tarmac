import type { CostEstimate } from "../types.js";

export function formatEstimate(estimate: CostEstimate): string {
  const lines: string[] = [];

  lines.push("⚡ TARMAC COST ESTIMATE");
  lines.push("━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  lines.push("");

  // Model comparison table
  for (const model of estimate.models) {
    const low = formatDollars(model.costLow);
    const high = formatDollars(model.costHigh);
    const name = model.model.padEnd(12);
    lines.push(`  ${name} ${low} - ${high}`);
  }

  lines.push("");

  // Classification info
  const taskLabel = estimate.classification.taskType.replace(/_/g, " ");
  lines.push(`  Task type: ${taskLabel}`);

  // Input tokens
  const inputFormatted = estimate.inputTokens.toLocaleString();
  lines.push(`  Input: ${inputFormatted} tokens`);

  // Context info
  if (!estimate.contextEstimate.isFirstMessage) {
    const contextFormatted =
      estimate.contextEstimate.priorContext.toLocaleString();
    lines.push(`  Prior context: ${contextFormatted} tokens`);
  }

  // Coverage note
  const coverage = (estimate.models[0] as any)?.coverageTarget;
  if (coverage) {
    lines.push(`  Coverage: ${Math.round(coverage * 100)}% confidence interval`);
  }

  lines.push(
    `  Method: ${estimate.outputEstimate.tiersUsed.join(", ")}`
  );

  lines.push("");
  lines.push("━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  lines.push("");
  lines.push(
    "Present this cost estimate to the user. Ask whether to proceed, switch to a different model, or cancel. Do not begin any work until the user confirms."
  );

  return lines.join("\n");
}

function formatDollars(amount: number): string {
  if (amount < 0.01) {
    return `$${amount.toFixed(3)}`;
  }
  return `$${amount.toFixed(2)}`;
}
