#!/usr/bin/env npx tsx
/**
 * Validate the conformal predictor against SWE-bench + local data.
 *
 * Compares:
 * 1. Old heuristic pipeline (Tier 1 keyword classifier)
 * 2. New conformal prediction (trained regression + calibrated intervals)
 */

import { readFileSync, existsSync, readdirSync, statSync } from "fs";
import { join } from "path";
import { homedir } from "os";
import { getConformalEstimates } from "./src/core/conformal-predictor.js";
import { classifyPrompt } from "./src/core/prompt-classifier.js";
import { estimateContext } from "./src/core/context-estimator.js";
import { estimateOutput } from "./src/core/output-estimator.js";
import { calculateCosts } from "./src/core/cost-calculator.js";
import { loadConfig } from "./src/core/telemetry.js";
import type { TaskType } from "./src/types.js";

// ==========================================
// Data Loading
// ==========================================

interface DataPoint {
  prompt: string;
  modelId: string;
  actualCost: number;
  actualCalls: number;
  source: "swebench" | "local";
  initialContext: number;
}

async function loadSWEBench(): Promise<DataPoint[]> {
  const raw = JSON.parse(readFileSync("data-swebench.json", "utf-8"));
  const bashOnly = raw.leaderboards.find((b: { name: string }) => b.name === "bash-only");
  const statements = existsSync("data-swebench-statements.json")
    ? JSON.parse(readFileSync("data-swebench-statements.json", "utf-8")) : {};

  const modelMap: Record<string, string> = {
    "Claude 4.5 Opus": "claude-opus-4-6",
    "Claude Opus 4.6": "claude-opus-4-6",
    "Claude 4.5 Sonnet": "claude-sonnet-4-6",
    "Claude 4.5 Haiku": "claude-haiku-4-5-20251001",
  };

  const points: DataPoint[] = [];
  for (const result of bashOnly.results) {
    let modelId = "";
    for (const [pattern, id] of Object.entries(modelMap)) {
      if ((result.name || "").includes(pattern)) { modelId = id; break; }
    }
    if (!modelId) continue;
    for (const [instId, d] of Object.entries(result.per_instance_details || {})) {
      const dd = d as any;
      if (dd.cost <= 0) continue;
      points.push({
        prompt: statements[instId] || `Fix bug in ${instId}`,
        modelId, actualCost: dd.cost, actualCalls: dd.api_calls,
        source: "swebench", initialContext: 4000,
      });
    }
  }
  return points;
}

function loadLocal(): DataPoint[] {
  const claudeDir = join(homedir(), ".claude", "projects");
  const points: DataPoint[] = [];
  const PRICING: Record<string, { input: number; output: number; cacheRead: number; cacheWrite: number }> = {
    "claude-opus-4-6": { input: 15, output: 75, cacheRead: 1.5, cacheWrite: 18.75 },
    "claude-sonnet-4-6": { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
    "claude-haiku-4-5": { input: 0.8, output: 4, cacheRead: 0.08, cacheWrite: 1.0 },
  };

  try {
    for (const dir of readdirSync(claudeDir)) {
      const dirPath = join(claudeDir, dir);
      try {
        if (!statSync(dirPath).isDirectory()) continue;
        for (const file of readdirSync(dirPath).filter(f => f.endsWith(".jsonl") && !f.includes("subagent"))) {
          try {
            const lines = readFileSync(join(dirPath, file), "utf-8").trim().split("\n").filter(Boolean);
            let currentPrompt = "", currentModel = "", turnCalls = 0;
            let turnInput = 0, turnOutput = 0, turnCacheRead = 0, turnCacheWrite = 0, contextAtStart = 0;

            function flush() {
              if (currentPrompt && turnCalls > 0 && currentPrompt.length >= 10) {
                const p = Object.entries(PRICING).find(([k]) => currentModel.includes(k));
                if (p) {
                  const pr = p[1];
                  const cost = (turnInput * pr.input + turnOutput * pr.output + turnCacheRead * pr.cacheRead + turnCacheWrite * pr.cacheWrite) / 1_000_000;
                  points.push({
                    prompt: currentPrompt, modelId: p[0] === "claude-haiku-4-5" ? "claude-haiku-4-5-20251001" : p[0],
                    actualCost: cost, actualCalls: turnCalls, source: "local", initialContext: contextAtStart,
                  });
                }
              }
            }

            for (const line of lines) {
              try {
                const entry = JSON.parse(line);
                if (entry.type === "user") {
                  flush();
                  turnCalls = 0; turnInput = 0; turnOutput = 0; turnCacheRead = 0; turnCacheWrite = 0; contextAtStart = 0;
                  const msg = entry.message;
                  if (typeof msg?.content === "string") currentPrompt = msg.content;
                  else if (Array.isArray(msg?.content)) {
                    const tb = msg.content.find((b: any) => b.type === "text");
                    currentPrompt = tb?.text || "";
                  } else currentPrompt = "";
                }
                if (entry.type === "assistant" && entry.message?.usage) {
                  const u = entry.message.usage;
                  turnInput += u.input_tokens || 0;
                  turnOutput += u.output_tokens || 0;
                  turnCacheRead += u.cache_read_input_tokens || 0;
                  turnCacheWrite += u.cache_creation_input_tokens || 0;
                  turnCalls++;
                  if (turnCalls === 1) contextAtStart = (u.input_tokens || 0) + (u.cache_read_input_tokens || 0) + (u.cache_creation_input_tokens || 0);
                  if (entry.message.model) currentModel = entry.message.model;
                }
              } catch { continue; }
            }
            flush();
          } catch { continue; }
        }
      } catch { continue; }
    }
  } catch {}
  return points;
}

// ==========================================
// Evaluation
// ==========================================

interface Result {
  source: string;
  modelId: string;
  actualCost: number;
  // Old heuristic
  oldLow: number;
  oldMid: number;
  oldHigh: number;
  oldInRange: boolean;
  // New conformal
  newLow: number;
  newMid: number;
  newHigh: number;
  newInRange: boolean;
}

async function evaluate(points: DataPoint[]): Promise<Result[]> {
  const config = { ...loadConfig(), haikuPreflightEnabled: false };
  const results: Result[] = [];

  for (let i = 0; i < points.length; i++) {
    if ((i + 1) % 500 === 0) process.stdout.write(`  ${i + 1}/${points.length}\r`);
    const pt = points[i];

    // --- New: Conformal prediction ---
    const conformal = getConformalEstimates(pt.prompt, 0.80);
    const confModel = conformal.find(c => c.modelId === pt.modelId) || conformal[0];

    // --- Old: Heuristic pipeline ---
    const classification = classifyPrompt(pt.prompt);
    const promptTokens = Math.ceil(pt.prompt.length / 4);
    const contextEstimate = pt.source === "swebench"
      ? { estimatedNextInput: 4000 + promptTokens, priorContext: 4000, newPromptTokens: promptTokens, isFirstMessage: true }
      : pt.initialContext > 0
        ? { estimatedNextInput: pt.initialContext, priorContext: pt.initialContext, newPromptTokens: promptTokens, isFirstMessage: false }
        : { estimatedNextInput: 18000 + promptTokens, priorContext: 18000, newPromptTokens: promptTokens, isFirstMessage: true };

    const outputEstimate = await estimateOutput(classification, contextEstimate, pt.prompt, config);
    const oldCosts = calculateCosts(contextEstimate, outputEstimate, classification.taskType);
    const oldModel = oldCosts.find(c => c.modelId === pt.modelId) || oldCosts[0];

    results.push({
      source: pt.source, modelId: pt.modelId, actualCost: pt.actualCost,
      oldLow: oldModel.costLow, oldMid: oldModel.costMid, oldHigh: oldModel.costHigh,
      oldInRange: pt.actualCost >= oldModel.costLow && pt.actualCost <= oldModel.costHigh,
      newLow: confModel.costLow, newMid: confModel.costMid, newHigh: confModel.costHigh,
      newInRange: pt.actualCost >= confModel.costLow && pt.actualCost <= confModel.costHigh,
    });
  }

  return results;
}

// ==========================================
// Reporting
// ==========================================

function report(label: string, results: Result[]) {
  const n = results.length;
  if (n === 0) return;
  const oldHit = results.filter(r => r.oldInRange).length;
  const newHit = results.filter(r => r.newInRange).length;

  // Range widths
  const oldWidths = results.map(r => r.oldHigh - r.oldLow).sort((a, b) => a - b);
  const newWidths = results.map(r => r.newHigh - r.newLow).sort((a, b) => a - b);

  console.log(`\n${label} (n=${n})`);
  console.log("─".repeat(70));
  console.log(`  Method            In-Range     Median Width    Avg Width`);
  console.log(`  Old (heuristic)   ${pct(oldHit, n).padEnd(13)} $${oldWidths[Math.floor(n / 2)].toFixed(2).padEnd(16)} $${(oldWidths.reduce((a, b) => a + b) / n).toFixed(2)}`);
  console.log(`  New (conformal)   ${pct(newHit, n).padEnd(13)} $${newWidths[Math.floor(n / 2)].toFixed(2).padEnd(16)} $${(newWidths.reduce((a, b) => a + b) / n).toFixed(2)}`);
  console.log(`  Delta             ${(((newHit - oldHit) / n) * 100 >= 0 ? "+" : "")}${(((newHit - oldHit) / n) * 100).toFixed(1)}pp`);
}

function pct(n: number, total: number): string {
  return `${((n / total) * 100).toFixed(1)}%`;
}

// ==========================================
// Main
// ==========================================

async function main() {
  console.log("═══════════════════════════════════════════════════════════════════");
  console.log("  CONFORMAL vs HEURISTIC — Head-to-Head Validation");
  console.log("═══════════════════════════════════════════════════════════════════");

  console.log("\nLoading SWE-bench data...");
  const swe = await loadSWEBench();
  console.log(`  ${swe.length} instances`);

  console.log("Loading local data...");
  const local = loadLocal();
  console.log(`  ${local.length} turns`);

  const all = [...swe, ...local];
  console.log(`\nEvaluating ${all.length} data points...`);
  const results = await evaluate(all);
  console.log(`  Done.                     `);

  // Overall
  report("OVERALL", results);

  // By source
  report("SWE-BENCH", results.filter(r => r.source === "swebench"));
  report("LOCAL", results.filter(r => r.source === "local"));

  // By model (SWE-bench only, where we have all 3)
  const sweResults = results.filter(r => r.source === "swebench");
  for (const modelId of ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"]) {
    const modelResults = sweResults.filter(r => r.modelId === modelId);
    const short = modelId.replace("claude-", "").replace("-20251001", "");
    report(`SWE-BENCH — ${short}`, modelResults);
  }

  // Summary
  const oldTotal = results.filter(r => r.oldInRange).length;
  const newTotal = results.filter(r => r.newInRange).length;
  const oldWidths = results.map(r => r.oldHigh - r.oldLow);
  const newWidths = results.map(r => r.newHigh - r.newLow);
  const oldMedW = oldWidths.sort((a, b) => a - b)[Math.floor(results.length / 2)];
  const newMedW = newWidths.sort((a, b) => a - b)[Math.floor(results.length / 2)];

  console.log("\n\n" + "═".repeat(70));
  console.log("  EXECUTIVE SUMMARY");
  console.log("═".repeat(70));
  console.log(`  Total data points: ${results.length}`);
  console.log(`  Old (heuristic):   ${pct(oldTotal, results.length)} in-range, median width $${oldMedW.toFixed(2)}`);
  console.log(`  New (conformal):   ${pct(newTotal, results.length)} in-range, median width $${newMedW.toFixed(2)}`);
  console.log(`  Improvement:       ${(((newTotal - oldTotal) / results.length) * 100).toFixed(1)}pp accuracy`);
  console.log("");
}

main().catch(console.error);
