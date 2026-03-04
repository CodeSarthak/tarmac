import type { ClassificationResult, TaskType, TaskProfile } from "../types.js";

/**
 * Task profiles calibrated against 2,500 SWE-bench instances + 363 local turns.
 *
 * Key calibration insights:
 * - SWE-bench coding tasks: Opus avg 30 calls, Sonnet avg 50 calls
 * - Output per call: ~800 tokens median
 * - Cache hit rate: ~85% after first call (dramatically reduces per-call input cost)
 * - Context grows ~2-5K per loop (tool results, file reads, test output)
 *
 * Loop counts here are for Opus (base model). The cost calculator applies
 * model-specific multipliers (Sonnet 1.6x, Haiku 2.0x).
 *
 * For non-coding prompts (followup, simple_question), loops stay low (1-2).
 * For coding prompts (generation, modification, debugging), loops reflect
 * real agent behavior: read files → edit → test → fix → iterate.
 */
export const TASK_PROFILES: Record<TaskType, TaskProfile> = {
  followup: {
    type: "followup",
    loopRange: [1, 2],
    outputPerLoop: [200, 600],
    contextGrowthPerLoop: [0, 500],
    costMultiplier: 1,
  },
  simple_question: {
    type: "simple_question",
    loopRange: [1, 3],
    outputPerLoop: [300, 1000],
    contextGrowthPerLoop: [0, 1000],
    costMultiplier: 1,
  },
  explanation: {
    type: "explanation",
    loopRange: [1, 5],
    outputPerLoop: [500, 1500],
    contextGrowthPerLoop: [500, 2000],
    costMultiplier: 1.5,
  },
  code_generation: {
    type: "code_generation",
    loopRange: [3, 10],
    outputPerLoop: [300, 800],
    contextGrowthPerLoop: [500, 2000],
    costMultiplier: 3,
  },
  code_modification: {
    type: "code_modification",
    loopRange: [3, 8],
    outputPerLoop: [300, 700],
    contextGrowthPerLoop: [500, 1500],
    costMultiplier: 2.5,
  },
  code_review: {
    type: "code_review",
    loopRange: [2, 6],
    outputPerLoop: [300, 800],
    contextGrowthPerLoop: [500, 1500],
    costMultiplier: 2,
  },
  debugging: {
    type: "debugging",
    loopRange: [3, 12],
    outputPerLoop: [300, 700],
    contextGrowthPerLoop: [500, 2000],
    costMultiplier: 3,
  },
  refactoring: {
    type: "refactoring",
    loopRange: [4, 12],
    outputPerLoop: [400, 1000],
    contextGrowthPerLoop: [800, 2500],
    costMultiplier: 3.5,
  },
  architecture: {
    type: "architecture",
    loopRange: [2, 6],
    outputPerLoop: [400, 1200],
    contextGrowthPerLoop: [500, 1500],
    costMultiplier: 2,
  },
};

interface FeatureScores {
  taskTypeKeywords: TaskType;
  scopeScore: number;
  fileCount: number;
  testSignal: boolean;
  breadthScore: number;
  isQuestion: boolean;
  hasCodeBlocks: boolean;
  lengthCue: "brief" | "detailed" | "neutral";
  constraintCount: number;
  isShortFollowup: boolean;
  promptLength: number;
}

const TASK_KEYWORDS: Record<TaskType, string[]> = {
  followup: [], // Detected by structure, not keywords
  simple_question: [
    "what is",
    "what are",
    "how does",
    "where is",
    "which",
    "can you tell me",
    "what's the",
    "what does",
    "is there",
    "does it",
    "how many",
    "how much",
    "tell me",
  ],
  explanation: [
    "explain",
    "describe",
    "walk me through",
    "what happens when",
    "why does",
    "tell me about",
    "break down",
    "overview",
    "summarize",
  ],
  code_generation: [
    "create",
    "build",
    "implement",
    "write",
    "generate",
    "make a",
    "set up",
    "scaffold",
    "bootstrap",
    "new file",
    "add a new",
    "from scratch",
  ],
  code_modification: [
    "change",
    "modify",
    "update",
    "edit",
    "add",
    "remove",
    "rename",
    "move",
    "replace",
    "adjust",
    "tweak",
    "convert",
    "switch",
    "migrate",
  ],
  code_review: [
    "review",
    "check",
    "look at",
    "audit",
    "inspect",
    "evaluate",
    "assess",
    "analyze this",
    "what do you think",
    "feedback",
  ],
  debugging: [
    "fix",
    "bug",
    "error",
    "broken",
    "not working",
    "failing",
    "crash",
    "issue",
    "wrong",
    "debug",
    "troubleshoot",
    "investigate",
    "figure out why",
    "doesn't work",
  ],
  refactoring: [
    "refactor",
    "restructure",
    "reorganize",
    "clean up",
    "simplify",
    "optimize",
    "extract",
    "split",
    "consolidate",
    "decompose",
    "decouple",
  ],
  architecture: [
    "design",
    "architect",
    "plan",
    "structure",
    "layout",
    "pattern",
    "approach",
    "strategy",
    "how should",
    "best way to",
    "system design",
    "high level",
  ],
};

const SCOPE_KEYWORDS = [
  "all files",
  "every",
  "entire",
  "whole",
  "across the",
  "throughout",
  "everywhere",
  "the whole",
  "all of",
  "each file",
  "codebase",
  "project-wide",
];

const TEST_KEYWORDS = [
  "test",
  "spec",
  "verify",
  "make sure",
  "ensure",
  "validate",
  "assert",
  "coverage",
  "passing",
  "unit test",
  "integration test",
];

const BROAD_KEYWORDS = [
  "implement",
  "build",
  "create",
  "develop",
  "full",
  "comprehensive",
  "end-to-end",
  "e2e",
];

const NARROW_KEYWORDS = [
  "rename",
  "delete",
  "remove",
  "typo",
  "one line",
  "single",
  "just",
  "only",
  "quick",
  "simple",
  "small",
];

/**
 * Patterns that indicate a short conversational follow-up rather than a new task.
 * These prompts typically get 1 API call with short output.
 */
const FOLLOWUP_PATTERNS = [
  /^(yes|no|yeah|nah|yep|nope|ok|okay|sure|thanks|thank you|perfect|great|good|nice|cool|done|got it|sounds good)\b/i,
  /^(try again|do it|go ahead|proceed|continue|go for it|let's go|ship it|lgtm)\b/i,
  /^(hmm|hm|ah|oh|interesting|i see)\b/i,
  /^(can you|could you|please)\s.{0,40}$/i, // Short requests
  /^(make|use|do|run|show|try)\s.{0,40}$/i, // Short imperatives
  /^<(ide_|task-|system)/i, // System-injected prompts
];

export function classifyPrompt(prompt: string): ClassificationResult {
  const lower = prompt.toLowerCase().trim();
  const features = extractFeatures(lower, prompt);
  let taskType = features.taskTypeKeywords;

  // Override: short prompts without strong task signals → followup
  if (features.isShortFollowup) {
    taskType = "followup";
  }

  // Determine confidence
  let confidence: "low" | "medium" | "high" = "medium";

  const matchingTypes = Object.entries(TASK_KEYWORDS).filter(
    ([key, keywords]) => key !== "followup" && keywords.some((kw) => lower.includes(kw))
  );

  if (features.isShortFollowup) {
    confidence = "high"; // We're confident it's a followup
  } else if (matchingTypes.length === 0) {
    confidence = "low";
  } else if (matchingTypes.length === 1) {
    confidence = "high";
  } else if (matchingTypes.length > 2) {
    confidence = "low";
  }

  if (prompt.length < 20 && !features.isShortFollowup) {
    confidence = "low";
  }

  return {
    taskType,
    confidence,
    features: {
      scopeScore: features.scopeScore,
      fileCount: features.fileCount,
      testSignal: features.testSignal ? 1 : 0,
      breadthScore: features.breadthScore,
      isQuestion: features.isQuestion ? 1 : 0,
      hasCodeBlocks: features.hasCodeBlocks ? 1 : 0,
      constraintCount: features.constraintCount,
      promptLength: features.promptLength,
      isShortFollowup: features.isShortFollowup ? 1 : 0,
    },
  };
}

function extractFeatures(lower: string, original: string): FeatureScores {
  const promptLength = original.length;

  // Check for short conversational follow-ups FIRST
  const isShortFollowup = detectFollowup(lower, original);

  // Task type keywords — find best match
  let bestType: TaskType = "simple_question"; // Changed default from code_modification
  let bestScore = 0;

  for (const [type, keywords] of Object.entries(TASK_KEYWORDS)) {
    if (type === "followup") continue; // Followup detected structurally
    const score = keywords.filter((kw) => lower.includes(kw)).length;
    if (score > bestScore) {
      bestScore = score;
      bestType = type as TaskType;
    }
  }

  // Scope keywords
  const scopeScore = SCOPE_KEYWORDS.filter((kw) => lower.includes(kw)).length;

  // File count mentioned
  const fileMatches = lower.match(/(\d+)\s*files?/g);
  const fileCount = fileMatches
    ? Math.max(...fileMatches.map((m) => parseInt(m.match(/(\d+)/)![1], 10)))
    : 0;

  // Test signal
  const testSignal = TEST_KEYWORDS.some((kw) => lower.includes(kw));

  // Breadth score
  const broadHits = BROAD_KEYWORDS.filter((kw) => lower.includes(kw)).length;
  const narrowHits = NARROW_KEYWORDS.filter((kw) => lower.includes(kw)).length;
  const breadthScore = broadHits - narrowHits;

  // Question marks
  const isQuestion = original.includes("?");

  // Code blocks
  const hasCodeBlocks = original.includes("```");

  // Length cues
  let lengthCue: "brief" | "detailed" | "neutral" = "neutral";
  if (/\b(brief|short|concise|quick)\b/.test(lower)) {
    lengthCue = "brief";
  } else if (/\b(detailed|comprehensive|thorough|complete)\b/.test(lower)) {
    lengthCue = "detailed";
  }

  // Constraint count
  const constraintCount = (
    lower.match(/\b(make sure|ensure|also|must|should|need to|don't forget)\b/g) || []
  ).length;

  // Adjust: questions with no keyword match → simple_question
  if (isQuestion && bestScore === 0) {
    bestType = "simple_question";
  }

  // Adjust: very short prompts with no strong signals → default to simple_question
  if (bestScore === 0 && promptLength < 200) {
    bestType = "simple_question";
  }

  return {
    taskTypeKeywords: bestType,
    scopeScore,
    fileCount,
    testSignal,
    breadthScore,
    isQuestion,
    hasCodeBlocks,
    lengthCue,
    constraintCount,
    isShortFollowup: isShortFollowup,
    promptLength,
  };
}

function detectFollowup(lower: string, original: string): boolean {
  // Short prompts that look like conversational responses
  if (original.length < 100) {
    // Matches known followup patterns
    if (FOLLOWUP_PATTERNS.some((p) => p.test(lower))) return true;

    // Very short with no task keywords → likely followup
    if (original.length < 50) {
      const hasTaskKeyword = Object.entries(TASK_KEYWORDS).some(
        ([key, keywords]) =>
          key !== "followup" && keywords.some((kw) => lower.includes(kw))
      );
      if (!hasTaskKeyword) return true;
    }
  }

  // System-injected prompts (IDE events, task notifications)
  if (original.startsWith("<ide_") || original.startsWith("<task-")) return true;

  return false;
}

export function getTaskProfile(taskType: TaskType): TaskProfile {
  return TASK_PROFILES[taskType];
}
