import { readFileSync, writeFileSync, existsSync, mkdirSync, readdirSync } from "fs";
import { join } from "path";
import { homedir } from "os";
import type { HistoryEntry, TaskType, ClassificationResult } from "../types.js";
import { classifyPrompt } from "./prompt-classifier.js";
import { MODEL_PRICING } from "../data/pricing.js";

const TARMAC_DIR = join(homedir(), ".tarmac");
const HISTORY_PATH = join(TARMAC_DIR, "history.json");
const CLAUDE_PROJECTS_DIR = join(homedir(), ".claude", "projects");

export interface HistoryAdjustment {
  adjustedLoops: [number, number] | null;
  adjustedOutputPerLoop: [number, number] | null;
  sampleCount: number;
  confidence: "low" | "medium" | "high";
}

export function loadHistory(): HistoryEntry[] {
  try {
    if (existsSync(HISTORY_PATH)) {
      const data = readFileSync(HISTORY_PATH, "utf-8");
      return JSON.parse(data) as HistoryEntry[];
    }
  } catch {
    // Corrupted history — start fresh
  }
  return [];
}

export function saveHistory(entries: HistoryEntry[]): void {
  if (!existsSync(TARMAC_DIR)) {
    mkdirSync(TARMAC_DIR, { recursive: true });
  }
  writeFileSync(HISTORY_PATH, JSON.stringify(entries, null, 2));
}

export function appendHistory(entry: HistoryEntry): void {
  const history = loadHistory();
  history.push(entry);
  // Keep last 500 entries to avoid unbounded growth
  const trimmed = history.slice(-500);
  saveHistory(trimmed);
}

export function findSimilarTasks(
  classification: ClassificationResult,
  promptText: string,
  k: number = 5
): HistoryEntry[] {
  const history = loadHistory();
  if (history.length === 0) return [];

  // Score each history entry by similarity
  const scored = history.map((entry) => {
    let score = 0;

    // Task type match is the strongest signal
    if (entry.taskType === classification.taskType) {
      score += 10;
    }

    // Simple text overlap (poor man's TF-IDF)
    const promptWords = new Set(
      promptText.toLowerCase().split(/\s+/).filter((w) => w.length > 3)
    );
    const entryWords = new Set(
      entry.promptSnippet.toLowerCase().split(/\s+/).filter((w) => w.length > 3)
    );
    const overlap = [...promptWords].filter((w) => entryWords.has(w)).length;
    score += overlap * 2;

    return { entry, score };
  });

  // Sort by score descending, take top k
  scored.sort((a, b) => b.score - a.score);
  return scored
    .slice(0, k)
    .filter((s) => s.score > 0)
    .map((s) => s.entry);
}

export function getHistoryAdjustment(
  classification: ClassificationResult,
  promptText: string
): HistoryAdjustment {
  const similar = findSimilarTasks(classification, promptText);

  if (similar.length < 3) {
    return {
      adjustedLoops: null,
      adjustedOutputPerLoop: null,
      sampleCount: similar.length,
      confidence: "low",
    };
  }

  // Calculate percentiles from similar tasks
  const loops = similar.map((s) => s.actualLoops).sort((a, b) => a - b);
  const outputPerLoop = similar
    .map((s) =>
      s.actualLoops > 0 ? Math.round(s.actualOutputTokens / s.actualLoops) : 0
    )
    .sort((a, b) => a - b);

  const p25Idx = Math.floor(loops.length * 0.25);
  const p75Idx = Math.min(Math.floor(loops.length * 0.75), loops.length - 1);

  const confidence = similar.length >= 10 ? "high" : "medium";

  return {
    adjustedLoops: [loops[p25Idx], loops[p75Idx]],
    adjustedOutputPerLoop: [outputPerLoop[p25Idx], outputPerLoop[p75Idx]],
    sampleCount: similar.length,
    confidence,
  };
}

/**
 * Scan past Claude Code sessions to bootstrap history.
 * Called once during setup or when history is empty.
 */
export function bootstrapFromClaudeSessions(): number {
  if (!existsSync(CLAUDE_PROJECTS_DIR)) return 0;

  const history = loadHistory();
  if (history.length > 0) return history.length; // Already bootstrapped

  let count = 0;

  try {
    const projectDirs = readdirSync(CLAUDE_PROJECTS_DIR);

    for (const projectDir of projectDirs) {
      const projectPath = join(CLAUDE_PROJECTS_DIR, projectDir);
      try {
        const files = readdirSync(projectPath).filter((f) =>
          f.endsWith(".jsonl")
        );

        for (const file of files) {
          try {
            const entry = parseSessionFile(join(projectPath, file));
            if (entry) {
              history.push(entry);
              count++;
            }
          } catch {
            continue;
          }
        }
      } catch {
        continue;
      }
    }

    if (count > 0) {
      saveHistory(history.slice(-500));
    }
  } catch {
    // Can't read projects dir
  }

  return count;
}

function parseSessionFile(filePath: string): HistoryEntry | null {
  try {
    const content = readFileSync(filePath, "utf-8");
    const lines = content.trim().split("\n").filter(Boolean);

    if (lines.length < 2) return null;

    // Find first user prompt
    let firstPrompt = "";
    let totalInputTokens = 0;
    let totalOutputTokens = 0;
    let loopCount = 0;
    let model = "unknown";
    let initialContext = 0;
    let finalContext = 0;

    for (const line of lines) {
      try {
        const entry = JSON.parse(line);

        if (entry.type === "summary") {
          // Use summary if available
          if (entry.cost_usd !== undefined && entry.total_input_tokens !== undefined) {
            // Found summary — use it
          }
          continue;
        }

        if (entry.type === "user" && !firstPrompt) {
          if (typeof entry.message?.content === "string") {
            firstPrompt = entry.message.content.slice(0, 200);
          } else if (Array.isArray(entry.message?.content)) {
            const textBlock = entry.message.content.find(
              (b: { type: string }) => b.type === "text"
            );
            if (textBlock) {
              firstPrompt = textBlock.text?.slice(0, 200) || "";
            }
          }
        }

        if (entry.type === "assistant" && entry.message?.usage) {
          const usage = entry.message.usage;
          totalInputTokens += usage.input_tokens || 0;
          totalOutputTokens += usage.output_tokens || 0;
          loopCount++;

          if (loopCount === 1) {
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

    if (!firstPrompt || loopCount === 0) return null;

    // Classify the original prompt
    const classification = classifyPrompt(firstPrompt);

    // Calculate approximate cost
    const pricing = MODEL_PRICING.find((p) => model.includes(p.modelId)) ||
      MODEL_PRICING[0]; // Default to Sonnet
    const cost =
      (totalInputTokens * pricing.inputPerMillion) / 1_000_000 +
      (totalOutputTokens * pricing.outputPerMillion) / 1_000_000;

    return {
      timestamp: new Date().toISOString(),
      taskType: classification.taskType,
      promptSnippet: firstPrompt,
      actualLoops: loopCount,
      actualInputTokens: totalInputTokens,
      actualOutputTokens: totalOutputTokens,
      actualCost: cost,
      model,
      initialContext,
      finalContext,
    };
  } catch {
    return null;
  }
}
