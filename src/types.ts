export type TaskType =
  | "followup"
  | "simple_question"
  | "explanation"
  | "code_generation"
  | "code_modification"
  | "code_review"
  | "debugging"
  | "refactoring"
  | "architecture";

export interface TaskProfile {
  type: TaskType;
  loopRange: [number, number];
  outputPerLoop: [number, number];
  contextGrowthPerLoop: [number, number];
  costMultiplier: number;
}

export interface ClassificationResult {
  taskType: TaskType;
  confidence: "low" | "medium" | "high";
  features: Record<string, number>;
}

export interface ContextEstimate {
  estimatedNextInput: number;
  priorContext: number;
  newPromptTokens: number;
  isFirstMessage: boolean;
}

export interface OutputEstimate {
  p25: number;
  p50: number;
  p75: number;
  p95: number;
  estimatedLoops: [number, number];
  confidence: "low" | "medium" | "high";
  tiersUsed: string[];
}

export interface ModelCostEstimate {
  model: string;
  modelId: string;
  costLow: number;
  costMid: number;
  costHigh: number;
  /** Point prediction (if using conformal) */
  predicted?: number;
  /** Target coverage (e.g., 0.80) */
  coverageTarget?: number;
  /** Range multiplier (e.g., 2.3x) */
  rangeMultiplier?: number;
}

export interface CostEstimate {
  models: ModelCostEstimate[];
  classification: ClassificationResult;
  contextEstimate: ContextEstimate;
  outputEstimate: OutputEstimate;
  inputTokens: number;
}

export interface HookInput {
  hook_event_name: string;
  prompt?: string;
  transcript_path?: string;
  session_id?: string;
  cwd?: string;
  sessionCost?: number;
}

export interface HookOutput {
  hookSpecificOutput?: {
    hookEventName: string;
    additionalContext: string;
  };
}

export interface HaikuPreflightResult {
  complexity: "simple" | "medium" | "complex" | "very_complex";
  estimatedLoops: [number, number];
  filesTouched: number;
  willRequireTests: "yes" | "no" | "maybe";
  iterationRisk: "low" | "medium" | "high";
  estimatedOutputTokens: [number, number];
}

export interface HistoryEntry {
  timestamp: string;
  taskType: TaskType;
  promptSnippet: string;
  actualLoops: number;
  actualInputTokens: number;
  actualOutputTokens: number;
  actualCost: number;
  model: string;
  initialContext: number;
  finalContext: number;
}

export interface TarmacConfig {
  telemetryOptIn: boolean;
  haikuPreflightEnabled: boolean;
  haikuCostThreshold: number;
  version: string;
}

export interface TelemetryPayload {
  task_type: TaskType;
  model: string;
  estimated_loops: [number, number];
  actual_loops: number;
  estimated_output_tokens: [number, number];
  actual_output_tokens: number;
  initial_context_tokens: number;
  final_context_tokens: number;
  session_duration_seconds: number;
  tarmac_version: string;
  timestamp: string;
}

export interface LastEstimate {
  timestamp: string;
  sessionId: string;
  classification: ClassificationResult;
  contextEstimate: ContextEstimate;
  outputEstimate: OutputEstimate;
  models: ModelCostEstimate[];
}

export interface ModelPricing {
  model: string;
  modelId: string;
  inputPerMillion: number;
  outputPerMillion: number;
  cacheReadPerMillion: number;
  cacheWritePerMillion: number;
}
