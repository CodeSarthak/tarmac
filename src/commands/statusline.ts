import { readFileSync, writeFileSync, existsSync } from "fs";
import { join } from "path";
import { homedir } from "os";
import { loadLastEstimate } from "../core/telemetry.js";

const TARMAC_DIR = join(homedir(), ".tarmac");
const COST_BASELINE_PATH = join(TARMAC_DIR, "cost-baseline.json");

/**
 * Statusline command — called by Claude Code's statusLine feature.
 *
 * Shows:
 *   Est: $X-$Y      — Tarmac's predicted range for the last prompt
 *   Last: $Z         — actual cost of the last completed response
 *   Session: $W      — total session spend
 *   Ctx: N%          — context window usage
 */
export async function runStatusline(): Promise<void> {
  try {
    const input = await readStdin();
    const session = parseSessionInput(input);
    const lastEstimate = loadLastEstimate();
    const baseline = loadBaseline();

    const parts: string[] = [];

    // Show last estimate
    if (lastEstimate && lastEstimate.models.length > 0) {
      const currentModelId = session.modelId || "";
      const matched = lastEstimate.models.find(m =>
        currentModelId.includes(m.modelId) || m.modelId.includes(currentModelId)
      );
      const primary = matched || lastEstimate.models[0];

      parts.push(
        `\x1b[33m✈ Est: ${fmtDollars(primary.costLow)}-${fmtDollars(primary.costHigh)}\x1b[0m`
      );
    }

    // Compute "Last" — actual cost of the last response
    if (lastEstimate && session.totalCost > 0) {
      const estimateTs = lastEstimate.timestamp;
      const isNewEstimate = !baseline || baseline.estimateTimestamp !== estimateTs;

      if (isNewEstimate) {
        // First refresh after a new prompt. Lock in the costs:
        //   costBefore = what the session total was before this response
        //                (= previous baseline's costAfter, or 0 if first prompt)
        //   costAfter  = current session total (includes this response's cost)
        const costBefore = baseline?.costAfter ?? 0;
        const costAfter = session.totalCost;

        saveBaseline({
          sessionId: session.sessionId,
          estimateTimestamp: estimateTs,
          costBefore,
          costAfter,
        });

        // Show cost of this response
        const lastMsgCost = costAfter - costBefore;
        if (costBefore > 0) {
          // Only show after first prompt (costBefore=0 means no previous reference)
          parts.push(`\x1b[32mLast: ${fmtDollars(lastMsgCost)}\x1b[0m`);
        }
      } else {
        // Subsequent refresh, same estimate — use locked-in values
        // But update costAfter in case Claude is still streaming (cost grows mid-response)
        const lastMsgCost = session.totalCost - baseline.costBefore;
        if (baseline.costBefore > 0) {
          parts.push(`\x1b[32mLast: ${fmtDollars(lastMsgCost)}\x1b[0m`);
        }

        // Update costAfter for next prompt's baseline
        if (session.totalCost > baseline.costAfter) {
          saveBaseline({ ...baseline, costAfter: session.totalCost });
        }
      }
    }

    // Show total session cost
    if (session.totalCost > 0) {
      parts.push(`\x1b[36mSession: ${fmtDollars(session.totalCost)}\x1b[0m`);
    }

    // Show context usage
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
  sessionId: string;
}

interface CostBaseline {
  sessionId: string;
  estimateTimestamp: string;
  costBefore: number;   // session total before this response started
  costAfter: number;     // session total after this response finished
}

function parseSessionInput(raw: string): SessionInput {
  try {
    const parsed = JSON.parse(raw);
    return {
      modelId: parsed.model?.id || "",
      modelName: parsed.model?.display_name || "",
      totalCost: parsed.cost?.total_cost_usd ?? 0,
      contextPct: parsed.context_window?.used_percentage ?? 0,
      sessionId: parsed.session_id || "unknown",
    };
  } catch {
    return { modelId: "", modelName: "", totalCost: 0, contextPct: 0, sessionId: "unknown" };
  }
}

function loadBaseline(): CostBaseline | null {
  try {
    if (existsSync(COST_BASELINE_PATH)) {
      return JSON.parse(readFileSync(COST_BASELINE_PATH, "utf-8"));
    }
  } catch { /* ignore */ }
  return null;
}

function saveBaseline(baseline: CostBaseline): void {
  try {
    writeFileSync(COST_BASELINE_PATH, JSON.stringify(baseline));
  } catch { /* ignore */ }
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
