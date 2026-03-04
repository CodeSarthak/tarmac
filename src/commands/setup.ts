import {
  readFileSync,
  writeFileSync,
  existsSync,
  mkdirSync,
} from "fs";
import { join } from "path";
import { homedir } from "os";
import { createInterface } from "readline";
import { saveConfig, getDefaultConfig } from "../core/telemetry.js";
import { bootstrapFromClaudeSessions } from "../core/history-analyzer.js";

const CLAUDE_SETTINGS_PATH = join(homedir(), ".claude", "settings.json");
const CLAUDE_DIR = join(homedir(), ".claude");

interface ClaudeSettings {
  hooks?: {
    UserPromptSubmit?: HookEntry[];
    Stop?: HookEntry[];
    [key: string]: HookEntry[] | undefined;
  };
  [key: string]: unknown;
}

interface HookEntry {
  matcher?: string;
  hooks: Array<{
    type: string;
    command: string;
    timeout?: number;
  }>;
}

export async function runSetup(): Promise<void> {
  console.log("");
  console.log("  ✈  Tarmac v0.1 — Pre-flight cost estimation for Claude Code");
  console.log("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("");

  // Check for ANTHROPIC_API_KEY
  if (!process.env.ANTHROPIC_API_KEY) {
    console.log(
      "  ⚠  ANTHROPIC_API_KEY not found in environment."
    );
    console.log(
      "     Tarmac will use heuristic token counting (less accurate)."
    );
    console.log(
      "     Set ANTHROPIC_API_KEY for exact token counts via the Anthropic API."
    );
    console.log("");
  } else {
    console.log("  ✓  ANTHROPIC_API_KEY detected");
  }

  // Ask about telemetry
  const telemetryOptIn = await askYesNo(
    "  Enable anonymous telemetry to improve estimates for everyone?\n" +
      "  (No prompts, code, or file names — only token counts and task types) [Y/n] "
  );

  // Save config
  const config = getDefaultConfig();
  config.telemetryOptIn = telemetryOptIn;
  saveConfig(config);
  console.log(`  ✓  Config saved (~/.tarmac/config.json)`);

  // Install hooks
  installHooks();
  console.log("  ✓  Hooks installed (~/.claude/settings.json)");

  // Bootstrap history from past sessions
  console.log("  ⏳ Scanning past Claude Code sessions for calibration...");
  const sessionCount = bootstrapFromClaudeSessions();
  if (sessionCount > 0) {
    console.log(
      `  ✓  Found ${sessionCount} past sessions → stored in ~/.tarmac/history.json`
    );
  } else {
    console.log(
      "  ℹ  No past sessions found. Estimates will improve as you use Claude Code."
    );
  }

  console.log("");
  console.log("  ✈  Setup complete! Tarmac is now active.");
  console.log("");
  console.log("  How it works:");
  console.log("    1. Open Claude Code normally: claude");
  console.log("    2. Type any prompt");
  console.log(
    "    3. Tarmac intercepts → estimates cost → Claude presents it"
  );
  console.log("    4. You decide: proceed, switch model, or cancel");
  console.log("");
  console.log("  To uninstall: remove the Tarmac entries from ~/.claude/settings.json");
  console.log("");
}

function installHooks(): void {
  // Ensure .claude directory exists
  if (!existsSync(CLAUDE_DIR)) {
    mkdirSync(CLAUDE_DIR, { recursive: true });
  }

  // Read existing settings
  let settings: ClaudeSettings = {};
  if (existsSync(CLAUDE_SETTINGS_PATH)) {
    try {
      const content = readFileSync(CLAUDE_SETTINGS_PATH, "utf-8");
      settings = JSON.parse(content);
    } catch {
      // Malformed settings — start fresh but preserve the file
      console.log("  ⚠  Existing settings.json was malformed, creating new one");
    }
  }

  // Initialize hooks object if missing
  if (!settings.hooks) {
    settings.hooks = {};
  }

  // Define our hook entries
  const estimateHook: HookEntry = {
    matcher: "",
    hooks: [
      {
        type: "command",
        command: "tarmac-cost estimate",
        timeout: 10000,
      },
    ],
  };

  const reportHook: HookEntry = {
    matcher: "",
    hooks: [
      {
        type: "command",
        command: "tarmac-cost report",
        timeout: 5000,
      },
    ],
  };

  // Add UserPromptSubmit hook (preserve existing)
  const existingPromptHooks = settings.hooks.UserPromptSubmit || [];
  // Check if Tarmac hook already exists
  const hasTarmacEstimate = existingPromptHooks.some((h) =>
    h.hooks?.some((hh) => hh.command?.includes("tarmac-cost estimate"))
  );
  if (!hasTarmacEstimate) {
    existingPromptHooks.push(estimateHook);
  }
  settings.hooks.UserPromptSubmit = existingPromptHooks;

  // Add Stop hook (preserve existing)
  const existingStopHooks = settings.hooks.Stop || [];
  const hasTarmacReport = existingStopHooks.some((h) =>
    h.hooks?.some((hh) => hh.command?.includes("tarmac-cost report"))
  );
  if (!hasTarmacReport) {
    existingStopHooks.push(reportHook);
  }
  settings.hooks.Stop = existingStopHooks;

  // Write back
  writeFileSync(CLAUDE_SETTINGS_PATH, JSON.stringify(settings, null, 2));
}

function askYesNo(question: string): Promise<boolean> {
  return new Promise((resolve) => {
    // If not interactive (piped), default to yes
    if (!process.stdin.isTTY) {
      resolve(true);
      return;
    }

    const rl = createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    rl.question(question, (answer) => {
      rl.close();
      const lower = answer.trim().toLowerCase();
      resolve(lower === "" || lower === "y" || lower === "yes");
    });
  });
}
