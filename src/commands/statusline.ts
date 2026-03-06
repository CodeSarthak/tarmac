import { loadLastEstimate } from "../core/telemetry.js";

/**
 * Statusline command — called by Claude Code's statusLine feature.
 * Reads session JSON from stdin, reads last estimate from file,
 * and outputs a persistent cost display.
 */
export async function runStatusline(): Promise<void> {
  try {
    const input = await readStdin();
    const session = parseSessionInput(input);
    const lastEstimate = loadLastEstimate();

    const parts: string[] = [];

    // Show last estimate if we have one
    if (lastEstimate && lastEstimate.models.length > 0) {
      // Find the estimate for the current model, or show the first one
      const currentModelId = session.modelId || "";
      const matched = lastEstimate.models.find(m =>
        currentModelId.includes(m.modelId) || m.modelId.includes(currentModelId)
      );
      const primary = matched || lastEstimate.models[0];

      parts.push(
        `\x1b[33m✈ Est: ${fmtDollars(primary.costLow)}-${fmtDollars(primary.costHigh)}\x1b[0m`
      );
    }

    // Show actual session cost if available
    if (session.totalCost > 0) {
      parts.push(`\x1b[32mSpent: ${fmtDollars(session.totalCost)}\x1b[0m`);
    }

    // Show context usage if available
    if (session.contextPct > 0) {
      const color = session.contextPct >= 80 ? "\x1b[31m" : session.contextPct >= 50 ? "\x1b[33m" : "\x1b[36m";
      parts.push(`${color}Ctx: ${Math.round(session.contextPct)}%\x1b[0m`);
    }

    if (parts.length > 0) {
      process.stdout.write(parts.join(" | "));
    }
  } catch {
    // Never error — statusline should be silent on failure
  }
}

interface SessionInput {
  modelId: string;
  modelName: string;
  totalCost: number;
  contextPct: number;
}

function parseSessionInput(raw: string): SessionInput {
  try {
    const parsed = JSON.parse(raw);
    return {
      modelId: parsed.model?.id || "",
      modelName: parsed.model?.display_name || "",
      totalCost: parsed.cost?.total_cost_usd || 0,
      contextPct: parsed.context_window?.used_percentage || 0,
    };
  } catch {
    return { modelId: "", modelName: "", totalCost: 0, contextPct: 0 };
  }
}

function readStdin(): Promise<string> {
  return new Promise((resolve) => {
    let data = "";
    const timeout = setTimeout(() => resolve(data || "{}"), 2000);
    process.stdin.setEncoding("utf-8");
    process.stdin.on("data", (chunk) => { data += chunk; });
    process.stdin.on("end", () => { clearTimeout(timeout); resolve(data || "{}"); });
    process.stdin.on("error", () => { clearTimeout(timeout); resolve("{}"); });
  });
}

function fmtDollars(amount: number): string {
  if (amount < 0.01) return `$${amount.toFixed(3)}`;
  return `$${amount.toFixed(2)}`;
}
