import { readFileSync } from "fs";
import type { ContextEstimate } from "../types.js";

const SYSTEM_PROMPT_OVERHEAD = 18000; // ~18K tokens for Claude Code system prompt + tool definitions
const CHARS_PER_TOKEN = 4;

interface TranscriptMessage {
  type: string;
  message?: {
    role: string;
    usage?: {
      input_tokens: number;
      output_tokens: number;
      cache_creation_input_tokens?: number;
      cache_read_input_tokens?: number;
    };
  };
}

export function estimateContext(
  transcriptPath: string | undefined,
  promptText: string
): ContextEstimate {
  const newPromptTokens = Math.ceil(promptText.length / CHARS_PER_TOKEN);

  if (!transcriptPath) {
    return {
      estimatedNextInput: SYSTEM_PROMPT_OVERHEAD + newPromptTokens,
      priorContext: SYSTEM_PROMPT_OVERHEAD,
      newPromptTokens,
      isFirstMessage: true,
    };
  }

  try {
    const content = readFileSync(transcriptPath, "utf-8");
    const lines = content.trim().split("\n").filter(Boolean);

    // Find the most recent assistant message with usage info
    let lastInputTokens = 0;
    let lastOutputTokens = 0;
    let foundUsage = false;

    for (let i = lines.length - 1; i >= 0; i--) {
      try {
        const entry = JSON.parse(lines[i]) as TranscriptMessage;
        if (
          entry.type === "assistant" &&
          entry.message?.role === "assistant" &&
          entry.message?.usage
        ) {
          lastInputTokens = entry.message.usage.input_tokens;
          lastOutputTokens = entry.message.usage.output_tokens;
          foundUsage = true;
          break;
        }
      } catch {
        // Skip malformed lines
        continue;
      }
    }

    if (!foundUsage) {
      return {
        estimatedNextInput: SYSTEM_PROMPT_OVERHEAD + newPromptTokens,
        priorContext: SYSTEM_PROMPT_OVERHEAD,
        newPromptTokens,
        isFirstMessage: true,
      };
    }

    // Next input = previous context + previous output (now in history) + new prompt
    const priorContext = lastInputTokens + lastOutputTokens;
    const estimatedNextInput = priorContext + newPromptTokens;

    return {
      estimatedNextInput,
      priorContext,
      newPromptTokens,
      isFirstMessage: false,
    };
  } catch {
    // Transcript unreadable — fall back to first-message estimate
    return {
      estimatedNextInput: SYSTEM_PROMPT_OVERHEAD + newPromptTokens,
      priorContext: SYSTEM_PROMPT_OVERHEAD,
      newPromptTokens,
      isFirstMessage: true,
    };
  }
}
