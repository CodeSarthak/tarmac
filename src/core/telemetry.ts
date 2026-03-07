import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { join } from "path";
import { homedir } from "os";
import type {
  TelemetryPayload,
  TarmacConfig,
  LastEstimate,
  CostEstimate,
} from "../types.js";

const TARMAC_DIR = join(homedir(), ".tarmac");
const CONFIG_PATH = join(TARMAC_DIR, "config.json");
const LAST_ESTIMATE_PATH = join(TARMAC_DIR, "last-estimate.json");

export function loadConfig(): TarmacConfig {
  try {
    if (existsSync(CONFIG_PATH)) {
      const data = readFileSync(CONFIG_PATH, "utf-8");
      return JSON.parse(data) as TarmacConfig;
    }
  } catch {
    // Corrupted config — use defaults
  }
  return getDefaultConfig();
}

export function saveConfig(config: TarmacConfig): void {
  ensureTarmacDir();
  writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2));
}

export function getDefaultConfig(): TarmacConfig {
  return {
    telemetryOptIn: false,
    haikuPreflightEnabled: true,
    haikuCostThreshold: 0.5,
    version: "0.1.0",
  };
}

export function saveLastEstimate(
  estimate: CostEstimate,
  sessionId: string
): void {
  ensureTarmacDir();
  const lastEstimate: LastEstimate = {
    timestamp: new Date().toISOString(),
    sessionId,
    classification: estimate.classification,
    contextEstimate: estimate.contextEstimate,
    outputEstimate: estimate.outputEstimate,
    models: estimate.models,
  };
  writeFileSync(LAST_ESTIMATE_PATH, JSON.stringify(lastEstimate, null, 2));
}

export function loadLastEstimate(): LastEstimate | null {
  try {
    if (existsSync(LAST_ESTIMATE_PATH)) {
      const data = readFileSync(LAST_ESTIMATE_PATH, "utf-8");
      return JSON.parse(data) as LastEstimate;
    }
  } catch {
    // Corrupted — ignore
  }
  return null;
}

export function buildTelemetryPayload(
  lastEstimate: LastEstimate,
  actualLoops: number,
  actualOutputTokens: number,
  initialContext: number,
  finalContext: number,
  durationSeconds: number,
  model: string
): TelemetryPayload {
  return {
    task_type: lastEstimate.classification.taskType,
    model,
    estimated_loops: lastEstimate.outputEstimate.estimatedLoops,
    actual_loops: actualLoops,
    estimated_output_tokens: [
      lastEstimate.outputEstimate.p25,
      lastEstimate.outputEstimate.p75,
    ],
    actual_output_tokens: actualOutputTokens,
    initial_context_tokens: initialContext,
    final_context_tokens: finalContext,
    session_duration_seconds: durationSeconds,
    tarmac_version: lastEstimate.classification.taskType ? "0.1.0" : "0.1.0",
    timestamp: new Date().toISOString(),
  };
}

/**
 * POST anonymized telemetry to the Tarmac API.
 * Stubbed for v0.1 — will activate when api.tarmac.dev is ready.
 */
export async function postTelemetry(
  _payload: TelemetryPayload
): Promise<boolean> {
  // v0.1: stubbed. Local-only collection.
  // When api.tarmac.dev is ready:
  // const response = await fetch('https://api.tarmac.dev/v1/telemetry', {
  //   method: 'POST',
  //   headers: { 'Content-Type': 'application/json' },
  //   body: JSON.stringify(payload),
  // });
  // return response.ok;
  return false;
}

function ensureTarmacDir(): void {
  if (!existsSync(TARMAC_DIR)) {
    mkdirSync(TARMAC_DIR, { recursive: true });
  }
}
