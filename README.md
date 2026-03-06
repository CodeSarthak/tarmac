# Tarmac

**Pre-flight cost estimation for Claude Code.**

Know what your AI coding task will cost *before* it runs. Tarmac hooks into Claude Code, intercepts your prompt, and shows a calibrated cost range — so you can proceed, switch models, or cancel before spending a cent.

![Tarmac Demo](demo.gif)

## The Problem

Claude Code has zero cost visibility. You type a prompt, it runs for 2 minutes or 20 minutes, and you find out the cost after it's done. For complex tasks on Opus, that can be $5-20+ per prompt — and there's no way to know in advance.

## The Solution

Tarmac installs as a [Claude Code hook](https://docs.anthropic.com/en/docs/claude-code/hooks). Every time you submit a prompt, Tarmac intercepts it, extracts features, runs a trained regression model with conformal prediction intervals, and injects a cost estimate into Claude's context. Claude then presents the estimate and asks whether to proceed.

```
⚡ TARMAC COST ESTIMATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Sonnet 4.6    $0.12 - $0.89
  Opus 4.6      $0.58 - $4.34
  Haiku 4.5     $0.03 - $0.22

  Task type: code modification
  Input: 847 tokens
  Coverage: 80% confidence interval
  Method: conformal-regression
━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

No API key required. No external calls. Everything runs locally in ~5ms.

## Quick Start

```bash
npm install -g tarmac-cost
tarmac-cost setup
```

That's it. Open Claude Code and every prompt (5+ words) will now include a cost estimate.

After a session, run `tarmac-cost report` to compare the estimate to what actually happened:

```
$ tarmac-cost report

  📊 TARMAC SESSION REPORT
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Model:      Opus 4.6
    Estimated:  $0.58 - $4.34
    Actual:     $2.17
    Result:     ✅ Within estimate

    API calls:  12
    Duration:   94s

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

The report compares the last estimate against the actual cost from that session's transcript. Run it after exiting a Claude Code session to see how the prediction held up.

To uninstall, remove the Tarmac hook entries from `~/.claude/settings.json`.

## Benchmarks

Validated on 3,381 real tasks (3,000 SWE-bench + 381 local Claude Code sessions):

| Dataset | Coverage (80% target) | Median Interval Width | vs Heuristic Baseline |
|---|---|---|---|
| **Overall** (n=3,381) | 81.1% | $0.78 | +19.3pp |
| SWE-bench (n=3,000) | 83.6% | $0.85 | +14.3pp |
| Opus 4.6 | 84.5% | $1.13 | +17.3pp |
| Sonnet 4.6 | 81.7% | $0.67 | +13.0pp |
| Haiku 4.5 | 84.6% | $0.47 | +7.8pp |

"Coverage" = percentage of actual costs that fell within the predicted range. An 80% target means you should expect ~4 out of 5 estimates to contain the true cost. We hit 81.1% overall.

## How It Works

```
┌────────────┐     ┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│  You type   │────▶│  Claude Code  │────▶│    Tarmac       │────▶│   Claude      │
│  a prompt   │     │  hook fires   │     │  estimates cost │     │  presents it  │
└────────────┘     └──────────────┘     └────────────────┘     └──────────────┘
```

1. **Hook intercept** — Claude Code's `UserPromptSubmit` hook pipes your prompt to `tarmac-cost estimate` via stdin
2. **Feature extraction** — 24 features extracted from prompt text (length, code blocks, file paths, task keywords, vocabulary richness, etc.)
3. **Per-model regression** — Separate ridge regression models for Opus, Sonnet, and Haiku predict log₁₀(cost)
4. **Conformal calibration** — Residuals from a held-out calibration set determine the interval width needed for 80% coverage
5. **Output** — The estimate is injected as `additionalContext` into Claude's system prompt, which Claude then presents to the user

### Why Conformal Prediction?

Traditional approaches (heuristic multipliers, percentile-based ranges) can't provide coverage guarantees. Conformal prediction is a distribution-free method that gives calibrated prediction intervals: if you ask for 80% coverage, you get ~80% coverage, regardless of the underlying distribution. No assumptions about normality or homoscedasticity needed.

### Features Used (24 total)

| Category | Features |
|---|---|
| **Size** | log char count, word count, line count, sentence count |
| **Code signals** | code blocks, file paths, function names, class names |
| **Error signals** | stack traces, error messages |
| **Text properties** | vocabulary richness, technical density, avg/max line length |
| **Task indicators** | mentions fix, add, refactor, test, deprecation, regression, performance |
| **Structure** | question count, URL count, inline code references |

## Reproducing the Results

The model was trained on [SWE-bench](https://www.swebench.com/) data (3,000 instances across Opus 4.6, Sonnet 4.6, and Haiku 4.5).

```bash
# Install dependencies
npm install

# Train the model (outputs src/data/model-weights.ts)
npx tsx train-model.ts

# Run head-to-head validation against heuristic baseline
npx tsx validate-conformal.ts

# Feature importance analysis
npx tsx signal-analysis.ts
```

Training data files:
- `data-swebench.json` — SWE-bench leaderboard data with per-instance costs
- `data-swebench-statements.json` — Problem statements for each SWE-bench instance

## Architecture

```
tarmac/
├── src/
│   ├── cli.ts                          # CLI entry point
│   ├── types.ts                        # TypeScript interfaces
│   ├── commands/
│   │   ├── estimate.ts                 # Cost estimation (UserPromptSubmit hook)
│   │   ├── report.ts                   # Outcome recording (Stop hook)
│   │   └── setup.ts                    # Hook installation + config
│   ├── core/
│   │   ├── conformal-predictor.ts      # Regression model + conformal intervals
│   │   ├── prompt-classifier.ts        # Task type classification
│   │   ├── context-estimator.ts        # Prior context estimation
│   │   ├── token-counter.ts            # Token counting (API or heuristic)
│   │   ├── output-estimator.ts         # Output token estimation
│   │   ├── cost-calculator.ts          # Token → dollar conversion
│   │   ├── formatter.ts               # Estimate → formatted output
│   │   ├── history-analyzer.ts         # Past session analysis
│   │   ├── haiku-preflight.ts          # Optional Haiku pre-analysis
│   │   └── telemetry.ts               # Config + last-estimate persistence
│   └── data/
│       ├── model-weights.ts            # Trained model weights (auto-generated)
│       └── pricing.ts                  # Claude model pricing
├── train-model.ts                      # Training script
├── validate-conformal.ts               # Validation script
├── signal-analysis.ts                  # Feature analysis
├── data-swebench.json                  # Training data
├── data-swebench-statements.json       # Problem statements
├── package.json
├── tsconfig.json
└── LICENSE
```

## Limitations

What tarmac can't do (yet):

- **Short/vague prompts** — A 4-word prompt like "fix the login bug" gives the model very little signal. Estimates will be wide.
- **No context awareness** — The model sees only the current prompt text, not the conversation history or codebase. A follow-up "do it differently" has no features to work with.
- **Local data gap** — Trained primarily on SWE-bench (3,000 instances). Only 381 local sessions in the validation set. Real-world usage patterns may differ.
- **Irreducible variance** — Even with perfect features, ~38% of cost variance is irreducible (same prompt can cost 2x or 0.5x depending on codebase state, model behavior, etc). This is a fundamental limit, not a model problem.
- **Claude Code only** — Currently only supports Claude Code's hook system. No support for other AI coding tools yet.

## Contributing

Contributions welcome. The biggest impact areas:

1. **More training data** — Run `validate-conformal.ts` on your local sessions and share anonymized results
2. **Better features** — The model uses 24 text features. Codebase-aware features (repo size, language, recent changes) could help
3. **Context awareness** — Using conversation history to improve follow-up estimates
4. **Multi-provider** — Extending beyond Claude to support other LLM providers

## License

MIT
