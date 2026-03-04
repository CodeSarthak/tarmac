#!/usr/bin/env npx tsx
/**
 * Deep Signal Analysis — What features predict agent cost?
 *
 * Treats this as a proper ML problem:
 * 1. Feature engineering — extract everything possible from pre-execution data
 * 2. Univariate analysis — which features correlate with cost?
 * 3. Multivariate regression — what's the best R² from combinations?
 * 4. Decision tree — find the best splits
 * 5. Conformal prediction — guaranteed coverage with calibrated intervals
 * 6. Variance decomposition — what's theoretically possible?
 */

import { readFileSync, existsSync } from "fs";
import { join } from "path";

// ==========================================
// Data Loading
// ==========================================

interface Instance {
  instanceId: string;
  repo: string;
  model: string;
  cost: number;
  apiCalls: number;
  resolved: boolean;
  statement: string;
}

function loadData(): Instance[] {
  const raw = JSON.parse(readFileSync(join(process.cwd(), "data-swebench.json"), "utf-8"));
  const bashOnly = raw.leaderboards.find((b: { name: string }) => b.name === "bash-only");
  const statements = existsSync(join(process.cwd(), "data-swebench-statements.json"))
    ? JSON.parse(readFileSync(join(process.cwd(), "data-swebench-statements.json"), "utf-8"))
    : {};

  const modelMap: Record<string, string> = {
    "Claude 4.5 Opus": "opus",
    "Claude Opus 4.6": "opus",
    "Claude 4.5 Sonnet": "sonnet",
    "Claude 4.5 Haiku": "haiku",
  };

  const instances: Instance[] = [];
  for (const result of bashOnly.results) {
    let model = "";
    for (const [pattern, m] of Object.entries(modelMap)) {
      if ((result.name || "").includes(pattern)) { model = m; break; }
    }
    if (!model) continue;
    const details = result.per_instance_details || {};
    for (const [instId, d] of Object.entries(details)) {
      const dd = d as any;
      if (dd.cost <= 0) continue;
      instances.push({
        instanceId: instId, repo: instId.split("__")[0], model,
        cost: dd.cost, apiCalls: dd.api_calls, resolved: dd.resolved,
        statement: statements[instId] || "",
      });
    }
  }
  return instances;
}

// ==========================================
// Feature Engineering
// ==========================================

interface FeatureRow {
  features: Record<string, number>;
  cost: number;
  logCost: number;
  model: string;
  instanceId: string;
}

function extractFeatures(inst: Instance, repoStats: Record<string, { mean: number; std: number; resolveRate: number; medianCalls: number }>): Record<string, number> {
  const s = inst.statement;
  const lower = s.toLowerCase();
  const words = s.split(/\s+/).filter(w => w.length > 0);
  const uniqueWords = new Set(words.map(w => w.toLowerCase()));
  const lines = s.split("\n");

  return {
    // Text length features
    logCharCount: s.length > 0 ? Math.log10(s.length) : 0,
    wordCount: words.length,
    lineCount: lines.length,
    sentenceCount: (s.match(/[.!?]+/g) || []).length,

    // Text structure
    codeBlockCount: (s.match(/```/g) || []).length / 2,
    filePathCount: (s.match(/[\w\-./]+\.(py|js|ts|java|go|rb|rs|cpp|c|h|md|json|yaml|yml|toml|cfg)/g) || []).length,
    functionNameCount: (s.match(/\b[a-z_]\w*\s*\(/g) || []).length,
    classNameCount: (s.match(/\bclass\s+[A-Z]\w+/g) || []).length,
    hasStackTrace: /traceback|exception|stack trace/i.test(s) ? 1 : 0,
    hasErrorMsg: /error|bug|fail|crash|broke|wrong|unexpected/i.test(s) ? 1 : 0,
    vocabRichness: words.length > 0 ? uniqueWords.size / words.length : 0,
    technicalDensity: s.length > 0 ? (s.match(/[{}\[\]()<>:;=+\-*\/|&!@#$%^~`]/g) || []).length / s.length : 0,
    avgLineLength: lines.length > 0 ? lines.reduce((a, l) => a + l.length, 0) / lines.length : 0,
    maxLineLength: Math.max(...lines.map(l => l.length), 0),

    // Semantic signals
    mentionsFix: /\bfix(es|ed|ing)?\b|\bbug\b|\bpatch\b/i.test(lower) ? 1 : 0,
    mentionsAdd: /\badd\b|\bimplement\b|\bcreate\b|\bintroduce\b|\bnew\b/i.test(lower) ? 1 : 0,
    mentionsRefactor: /\brefactor\b|\brestructure\b|\bclean\b|\bsimplif/i.test(lower) ? 1 : 0,
    mentionsTest: /\btest\b|\bspec\b|\bassert\b|\bverify\b/i.test(lower) ? 1 : 0,
    mentionsDeprecation: /\bdeprecate\b|\bremove\b|\bdrop\b|\bobsolete\b/i.test(lower) ? 1 : 0,
    mentionsRegression: /\bregression\b|\bused to\b|\bno longer\b|\bsince\b/i.test(lower) ? 1 : 0,
    mentionsPerformance: /\bperformance\b|\bslow\b|\boptimize\b|\bmemory\b/i.test(lower) ? 1 : 0,
    questionCount: (s.match(/\?/g) || []).length,
    urlCount: (s.match(/https?:\/\//g) || []).length,
    codeRefCount: (s.match(/`[^`]+`/g) || []).length,

    // Repo features (from training data)
    repoMeanCost: repoStats[inst.repo]?.mean ?? 0.5,
    repoStdCost: repoStats[inst.repo]?.std ?? 0.3,
    repoResolveRate: repoStats[inst.repo]?.resolveRate ?? 0.7,
    repoMedianCalls: repoStats[inst.repo]?.medianCalls ?? 40,

    // Model features
    isOpus: inst.model === "opus" ? 1 : 0,
    isSonnet: inst.model === "sonnet" ? 1 : 0,
    isHaiku: inst.model === "haiku" ? 1 : 0,

    // Outcome (known in training, unknown in production — but useful for understanding ceiling)
    resolved: inst.resolved ? 1 : 0,
  };
}

function computeRepoStats(instances: Instance[]) {
  const byRepo: Record<string, Instance[]> = {};
  for (const inst of instances) {
    if (!byRepo[inst.repo]) byRepo[inst.repo] = [];
    byRepo[inst.repo].push(inst);
  }
  const stats: Record<string, { mean: number; std: number; resolveRate: number; medianCalls: number }> = {};
  for (const [repo, insts] of Object.entries(byRepo)) {
    const costs = insts.map(i => i.cost);
    const calls = insts.map(i => i.apiCalls).sort((a, b) => a - b);
    const mean = costs.reduce((a, b) => a + b) / costs.length;
    const std = Math.sqrt(costs.map(c => (c - mean) ** 2).reduce((a, b) => a + b) / costs.length);
    stats[repo] = { mean, std, resolveRate: insts.filter(i => i.resolved).length / insts.length, medianCalls: calls[Math.floor(calls.length / 2)] };
  }
  return stats;
}

// ==========================================
// Stats
// ==========================================

function pearsonR(x: number[], y: number[]): number {
  const n = x.length;
  const mx = x.reduce((a, b) => a + b) / n;
  const my = y.reduce((a, b) => a + b) / n;
  let num = 0, dx2 = 0, dy2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx, dy = y[i] - my;
    num += dx * dy; dx2 += dx * dx; dy2 += dy * dy;
  }
  return dx2 === 0 || dy2 === 0 ? 0 : num / Math.sqrt(dx2 * dy2);
}

function spearmanRho(x: number[], y: number[]): number {
  return pearsonR(rankArray(x), rankArray(y));
}

function rankArray(arr: number[]): number[] {
  const indexed = arr.map((v, i) => ({ v, i }));
  indexed.sort((a, b) => a.v - b.v);
  const ranks = new Array(arr.length);
  let i = 0;
  while (i < indexed.length) {
    let j = i;
    while (j < indexed.length && indexed[j].v === indexed[i].v) j++;
    const avgRank = (i + j + 1) / 2;
    for (let k = i; k < j; k++) ranks[indexed[k].i] = avgRank;
    i = j;
  }
  return ranks;
}

function linearRegression(X: number[][], y: number[]): { beta: number[]; rSquared: number } {
  const n = X.length, p = X[0].length;
  const Xa = X.map(row => [1, ...row]);
  const pa = p + 1;
  const XtX: number[][] = Array.from({ length: pa }, () => Array(pa).fill(0));
  for (let i = 0; i < n; i++)
    for (let j = 0; j < pa; j++)
      for (let k = 0; k < pa; k++)
        XtX[j][k] += Xa[i][j] * Xa[i][k];
  const Xty: number[] = Array(pa).fill(0);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < pa; j++)
      Xty[j] += Xa[i][j] * y[i];
  // Ridge
  for (let j = 0; j < pa; j++) XtX[j][j] += 0.01;
  const beta = solveSystem(XtX, Xty);
  const preds = Xa.map(row => row.reduce((s, x, j) => s + x * beta[j], 0));
  const yMean = y.reduce((a, b) => a + b) / n;
  const ssTot = y.reduce((s, yi) => s + (yi - yMean) ** 2, 0);
  const ssRes = y.reduce((s, yi, i) => s + (yi - preds[i]) ** 2, 0);
  return { beta, rSquared: 1 - ssRes / ssTot };
}

function predict(beta: number[], row: number[]): number {
  return [1, ...row].reduce((s, x, j) => s + x * beta[j], 0);
}

function solveSystem(A: number[][], b: number[]): number[] {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let i = 0; i < n; i++) {
    let maxVal = Math.abs(M[i][i]), maxRow = i;
    for (let k = i + 1; k < n; k++)
      if (Math.abs(M[k][i]) > maxVal) { maxVal = Math.abs(M[k][i]); maxRow = k; }
    [M[i], M[maxRow]] = [M[maxRow], M[i]];
    if (Math.abs(M[i][i]) < 1e-12) continue;
    for (let k = i + 1; k < n; k++) {
      const f = M[k][i] / M[i][i];
      for (let j = i; j <= n; j++) M[k][j] -= f * M[i][j];
    }
  }
  const x = Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = M[i][n];
    for (let j = i + 1; j < n; j++) x[i] -= M[i][j] * x[j];
    x[i] /= M[i][i] || 1;
  }
  return x;
}

// ==========================================
// Decision Tree
// ==========================================

interface TreeNode {
  feature?: string; threshold?: number; left?: TreeNode; right?: TreeNode;
  prediction?: { median: number; p25: number; p75: number; count: number };
}

function buildTree(data: { f: Record<string, number>; y: number }[], feats: string[], depth: number, minN: number = 30): TreeNode {
  if (data.length < minN * 2 || depth === 0) {
    const vals = data.map(d => d.y).sort((a, b) => a - b);
    return { prediction: { median: vals[Math.floor(vals.length / 2)], p25: vals[Math.floor(vals.length * 0.25)], p75: vals[Math.min(Math.floor(vals.length * 0.75), vals.length - 1)], count: vals.length } };
  }
  let bestFeat = "", bestThresh = 0, bestRed = 0;
  const totVar = variance(data.map(d => d.y));
  for (const feat of feats) {
    const vals = [...new Set(data.map(d => d.f[feat]))].sort((a, b) => a - b);
    const step = Math.max(1, Math.floor(vals.length / 10));
    for (let i = step; i < vals.length; i += step) {
      const thresh = (vals[i - 1] + vals[i]) / 2;
      const left = data.filter(d => d.f[feat] <= thresh);
      const right = data.filter(d => d.f[feat] > thresh);
      if (left.length < minN || right.length < minN) continue;
      const wVar = (left.length * variance(left.map(d => d.y)) + right.length * variance(right.map(d => d.y))) / data.length;
      const red = totVar - wVar;
      if (red > bestRed) { bestRed = red; bestFeat = feat; bestThresh = thresh; }
    }
  }
  if (!bestFeat || bestRed < 0.001) {
    const vals = data.map(d => d.y).sort((a, b) => a - b);
    return { prediction: { median: vals[Math.floor(vals.length / 2)], p25: vals[Math.floor(vals.length * 0.25)], p75: vals[Math.min(Math.floor(vals.length * 0.75), vals.length - 1)], count: vals.length } };
  }
  return {
    feature: bestFeat, threshold: bestThresh,
    left: buildTree(data.filter(d => d.f[bestFeat] <= bestThresh), feats, depth - 1, minN),
    right: buildTree(data.filter(d => d.f[bestFeat] > bestThresh), feats, depth - 1, minN),
  };
}

function predictTree(node: TreeNode, f: Record<string, number>): { median: number; p25: number; p75: number } {
  if (node.prediction) return node.prediction;
  return (f[node.feature!] ?? 0) <= node.threshold! ? predictTree(node.left!, f) : predictTree(node.right!, f);
}

function printTree(node: TreeNode, indent: string = ""): void {
  if (node.prediction) {
    const p = node.prediction;
    console.log(`${indent}→ n=${p.count} median=$${(10 ** p.median).toFixed(2)} [$${(10 ** p.p25).toFixed(2)}, $${(10 ** p.p75).toFixed(2)}]`);
    return;
  }
  console.log(`${indent}IF ${node.feature} <= ${node.threshold!.toFixed(3)}:`);
  printTree(node.left!, indent + "  ");
  console.log(`${indent}ELSE:`);
  printTree(node.right!, indent + "  ");
}

function variance(arr: number[]): number {
  if (arr.length === 0) return 0;
  const m = arr.reduce((a, b) => a + b) / arr.length;
  return arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length;
}

// ==========================================
// Main
// ==========================================

async function main() {
  console.log("═══════════════════════════════════════════════════════════════════");
  console.log("  DEEP SIGNAL ANALYSIS — What Predicts Agent Cost?");
  console.log("═══════════════════════════════════════════════════════════════════\n");

  const instances = loadData();
  console.log(`Loaded ${instances.length} instances\n`);

  // 80/20 split for honest evaluation
  const shuffled = [...instances].sort((a, b) => (a.instanceId + a.model).localeCompare(b.instanceId + b.model));
  const splitIdx = Math.floor(shuffled.length * 0.8);
  const trainInstances = shuffled.slice(0, splitIdx);
  const testInstances = shuffled.slice(splitIdx);

  // Compute repo stats from TRAINING data only (no leakage)
  const repoStats = computeRepoStats(trainInstances);

  // Extract features
  const trainData: FeatureRow[] = trainInstances.map(inst => ({
    features: extractFeatures(inst, repoStats),
    cost: inst.cost, logCost: Math.log10(inst.cost), model: inst.model, instanceId: inst.instanceId,
  }));
  const testData: FeatureRow[] = testInstances.map(inst => ({
    features: extractFeatures(inst, repoStats),
    cost: inst.cost, logCost: Math.log10(inst.cost), model: inst.model, instanceId: inst.instanceId,
  }));

  // Feature names (excluding 'resolved' which isn't known pre-execution)
  const allFeats = Object.keys(trainData[0].features);
  const preExecFeats = allFeats.filter(f => f !== "resolved");

  console.log(`Train: ${trainData.length}, Test: ${testData.length}`);
  console.log(`Features: ${allFeats.length} total, ${preExecFeats.length} pre-execution\n`);

  // ===== SECTION 1: Univariate Correlations =====
  console.log("1. UNIVARIATE CORRELATIONS (feature vs log10(cost)) — TRAIN SET");
  console.log("─".repeat(80));

  const correlations: { name: string; r: number; rho: number }[] = [];
  for (const feat of allFeats) {
    const x = trainData.map(d => d.features[feat]);
    const y = trainData.map(d => d.logCost);
    correlations.push({ name: feat, r: pearsonR(x, y), rho: spearmanRho(x, y) });
  }
  correlations.sort((a, b) => Math.abs(b.rho) - Math.abs(a.rho));

  console.log("  Feature".padEnd(30) + "Pearson r".padEnd(14) + "Spearman rho".padEnd(14) + "Pre-exec?");
  console.log("  " + "─".repeat(65));
  for (const c of correlations) {
    const preExec = c.name !== "resolved" ? "YES" : "no (oracle)";
    console.log(`  ${c.name.padEnd(28)} ${((c.r >= 0 ? "+" : "") + c.r.toFixed(4)).padEnd(14)} ${((c.rho >= 0 ? "+" : "") + c.rho.toFixed(4)).padEnd(14)} ${preExec}`);
  }

  // ===== SECTION 2: Feature Group R² =====
  console.log("\n\n2. FEATURE GROUP R² (ridge regression on log-cost, TRAIN SET)");
  console.log("─".repeat(80));

  const trainY = trainData.map(d => d.logCost);
  const groups: [string, string[]][] = [
    ["Model only", ["isOpus", "isSonnet", "isHaiku"]],
    ["Text length only", ["logCharCount", "wordCount", "lineCount"]],
    ["Text structure", ["codeBlockCount", "filePathCount", "functionNameCount", "classNameCount", "hasStackTrace", "hasErrorMsg", "technicalDensity", "vocabRichness"]],
    ["Semantic keywords", ["mentionsFix", "mentionsAdd", "mentionsRefactor", "mentionsTest", "mentionsDeprecation", "mentionsRegression", "mentionsPerformance"]],
    ["Repo stats", ["repoMeanCost", "repoStdCost", "repoResolveRate", "repoMedianCalls"]],
    ["Model + text length", ["isOpus", "isSonnet", "isHaiku", "logCharCount", "wordCount"]],
    ["Model + repo stats", ["isOpus", "isSonnet", "isHaiku", "repoMeanCost", "repoStdCost"]],
    ["All pre-execution", preExecFeats],
    ["All + resolved (oracle)", allFeats],
  ];

  for (const [name, feats] of groups) {
    const X = trainData.map(d => feats.map(f => d.features[f]));
    const reg = linearRegression(X, trainY);
    console.log(`  ${name.padEnd(35)} R² = ${reg.rSquared.toFixed(4)}`);
  }

  // ===== SECTION 3: Decision Tree =====
  console.log("\n\n3. DECISION TREE (depth=4, pre-execution features)");
  console.log("─".repeat(80));

  const treeTrainData = trainData.map(d => ({ f: d.features, y: d.logCost }));
  const tree = buildTree(treeTrainData, preExecFeats, 4, 40);
  printTree(tree, "  ");

  // Tree evaluation on test set
  let treeTrainHit = 0, treeTestHit = 0;
  for (const d of trainData) {
    const p = predictTree(tree, d.features);
    if (d.logCost >= p.p25 && d.logCost <= p.p75) treeTrainHit++;
  }
  for (const d of testData) {
    const p = predictTree(tree, d.features);
    if (d.logCost >= p.p25 && d.logCost <= p.p75) treeTestHit++;
  }
  console.log(`\n  Tree [p25,p75] capture rate: train=${(treeTrainHit / trainData.length * 100).toFixed(1)}%, test=${(treeTestHit / testData.length * 100).toFixed(1)}%`);

  // ===== SECTION 4: Conformal Prediction =====
  console.log("\n\n4. CONFORMAL PREDICTION — Calibrated Intervals");
  console.log("─".repeat(80));
  console.log("  (Train ridge regression, calibrate interval width on residuals, test on held-out)\n");

  // Split train into proper-train and calibration
  const calSplit = Math.floor(trainData.length * 0.6);
  const properTrain = trainData.slice(0, calSplit);
  const calSet = trainData.slice(calSplit);

  const ptX = properTrain.map(d => preExecFeats.map(f => d.features[f]));
  const ptY = properTrain.map(d => d.logCost);
  const reg = linearRegression(ptX, ptY);

  console.log(`  Regression R² (proper train): ${reg.rSquared.toFixed(4)}`);

  // Calibration residuals
  const calResiduals = calSet.map(d => {
    const pred = predict(reg.beta, preExecFeats.map(f => d.features[f]));
    return Math.abs(pred - d.logCost);
  }).sort((a, b) => a - b);

  // Test predictions
  const testPreds = testData.map(d => predict(reg.beta, preExecFeats.map(f => d.features[f])));
  const testY = testData.map(d => d.logCost);

  console.log("\n  Target%    Actual%    qhat(log10)    Multiplier    Example range ($0.50 predicted)");
  console.log("  " + "─".repeat(85));
  for (const target of [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]) {
    const idx = Math.min(Math.ceil((calResiduals.length + 1) * target) - 1, calResiduals.length - 1);
    const qhat = calResiduals[idx];

    let covered = 0;
    const widths: number[] = [];
    for (let i = 0; i < testPreds.length; i++) {
      const lo = testPreds[i] - qhat;
      const hi = testPreds[i] + qhat;
      if (testY[i] >= lo && testY[i] <= hi) covered++;
      widths.push(10 ** hi - 10 ** lo);
    }
    const medWidth = widths.sort((a, b) => a - b)[Math.floor(widths.length / 2)];
    const mult = 10 ** qhat;

    console.log(`  ${(target * 100).toFixed(0)}%`.padEnd(13) +
      `${(covered / testPreds.length * 100).toFixed(1)}%`.padEnd(11) +
      `±${qhat.toFixed(3)}`.padEnd(15) +
      `×${mult.toFixed(2)} / ÷${mult.toFixed(2)}`.padEnd(14) +
      `[$${(0.50 / mult).toFixed(2)}, $${(0.50 * mult).toFixed(2)}]`);
  }

  // ===== SECTION 5: Per-Model Conformal =====
  console.log("\n\n5. PER-MODEL CONFORMAL (separate regression per model)");
  console.log("─".repeat(80));

  for (const model of ["opus", "sonnet", "haiku"]) {
    const mTrain = trainData.filter(d => d.model === model);
    const mTest = testData.filter(d => d.model === model);
    const mCalSplit = Math.floor(mTrain.length * 0.6);
    const mPT = mTrain.slice(0, mCalSplit);
    const mCal = mTrain.slice(mCalSplit);
    const mFeats = preExecFeats.filter(f => !f.startsWith("is")); // no model dummies

    const mReg = linearRegression(mPT.map(d => mFeats.map(f => d.features[f])), mPT.map(d => d.logCost));
    const mCalRes = mCal.map(d => Math.abs(predict(mReg.beta, mFeats.map(f => d.features[f])) - d.logCost)).sort((a, b) => a - b);
    const mTestPreds = mTest.map(d => predict(mReg.beta, mFeats.map(f => d.features[f])));
    const mTestY = mTest.map(d => d.logCost);

    console.log(`\n  ${model.toUpperCase()} (train=${mPT.length}, cal=${mCal.length}, test=${mTest.length}, R²=${mReg.rSquared.toFixed(3)}):`);
    for (const target of [0.80, 0.90]) {
      const idx = Math.min(Math.ceil((mCalRes.length + 1) * target) - 1, mCalRes.length - 1);
      const qhat = mCalRes[idx];
      let covered = 0;
      for (let i = 0; i < mTestPreds.length; i++) {
        if (mTestY[i] >= mTestPreds[i] - qhat && mTestY[i] <= mTestPreds[i] + qhat) covered++;
      }
      const mult = 10 ** qhat;
      console.log(`    ${(target * 100).toFixed(0)}% target: ${(covered / mTestPreds.length * 100).toFixed(1)}% actual, range ×${mult.toFixed(2)}`);
    }
  }

  // ===== SECTION 6: Variance Decomposition =====
  console.log("\n\n6. VARIANCE DECOMPOSITION — What's Theoretically Possible?");
  console.log("─".repeat(80));

  // Same instance across models → between-instance vs within-instance variance
  const instanceCosts: Record<string, number[]> = {};
  for (const d of [...trainData, ...testData]) {
    if (!instanceCosts[d.instanceId]) instanceCosts[d.instanceId] = [];
    instanceCosts[d.instanceId].push(d.cost);
  }

  const allCosts = [...trainData, ...testData].map(d => d.cost);
  const overallMean = allCosts.reduce((a, b) => a + b) / allCosts.length;
  const totalVar = allCosts.reduce((s, c) => s + (c - overallMean) ** 2, 0) / allCosts.length;

  let withinSum = 0, withinN = 0;
  for (const costs of Object.values(instanceCosts)) {
    if (costs.length < 2) continue;
    const m = costs.reduce((a, b) => a + b) / costs.length;
    for (const c of costs) { withinSum += (c - m) ** 2; withinN++; }
  }
  const withinVar = withinN > 0 ? withinSum / withinN : 0;
  const betweenVar = totalVar - withinVar;

  console.log(`  Total variance:          ${totalVar.toFixed(4)}`);
  console.log(`  Between-instance:        ${betweenVar.toFixed(4)} (${(betweenVar / totalVar * 100).toFixed(1)}%) — what the task IS matters`);
  console.log(`  Within-instance:         ${withinVar.toFixed(4)} (${(withinVar / totalVar * 100).toFixed(1)}%) — irreducible (same task, different run)`);

  // Model choice as source of variance
  const modelMeans: Record<string, number> = {};
  const byModel: Record<string, number[]> = {};
  for (const d of [...trainData, ...testData]) {
    if (!byModel[d.model]) byModel[d.model] = [];
    byModel[d.model].push(d.cost);
  }
  for (const [m, costs] of Object.entries(byModel)) {
    modelMeans[m] = costs.reduce((a, b) => a + b) / costs.length;
  }
  let modelExplainedVar = 0;
  for (const d of [...trainData, ...testData]) {
    modelExplainedVar += (modelMeans[d.model] - overallMean) ** 2;
  }
  modelExplainedVar /= allCosts.length;

  console.log(`\n  Model choice explains:   ${(modelExplainedVar / totalVar * 100).toFixed(1)}% of total variance`);
  for (const [m, costs] of Object.entries(byModel)) {
    const mean = costs.reduce((a, b) => a + b) / costs.length;
    console.log(`    ${m.padEnd(10)} mean=$${mean.toFixed(2)}`);
  }

  // What R² could embeddings achieve? (upper bound = between-instance / total)
  const embeddingCeiling = betweenVar / totalVar;
  const currentR2 = reg.rSquared;

  console.log(`\n  Upper bound R² (perfect task identification): ${embeddingCeiling.toFixed(3)}`);
  console.log(`  Current R² (hand-crafted features):           ${currentR2.toFixed(3)}`);
  console.log(`  Room for improvement (embeddings etc.):       ${((embeddingCeiling - currentR2) * 100).toFixed(1)}pp`);

  // ===== SECTION 7: What would 80-90% look like? =====
  console.log("\n\n7. PATH TO 80-90% IN-RANGE ACCURACY");
  console.log("─".repeat(80));

  // Current Tarmac performance on test set
  // (reimpllement quick version of Tier 1 for comparison)
  console.log("\n  Current state (conformal with hand-crafted features):");
  for (const targetPct of [80, 85, 90]) {
    const target = targetPct / 100;
    const idx = Math.min(Math.ceil((calResiduals.length + 1) * target) - 1, calResiduals.length - 1);
    const qhat = calResiduals[idx];
    let covered = 0;
    const widths: number[] = [];
    for (let i = 0; i < testPreds.length; i++) {
      const lo = 10 ** (testPreds[i] - qhat);
      const hi = 10 ** (testPreds[i] + qhat);
      if (testData[i].cost >= lo && testData[i].cost <= hi) covered++;
      widths.push(hi - lo);
    }
    const medWidth = widths.sort((a, b) => a - b)[Math.floor(widths.length / 2)];
    const avgWidth = widths.reduce((a, b) => a + b) / widths.length;
    console.log(`    ${targetPct}% target: ${(covered / testPreds.length * 100).toFixed(1)}% actual, median range width $${medWidth.toFixed(2)}, avg $${avgWidth.toFixed(2)}`);
  }

  // What if we had embeddings that give us R² = 0.40?
  console.log("\n  Simulated: if R² improved to 0.40 (e.g., with embeddings):");
  // Simulate by reducing residuals proportionally
  const currentMSE = calResiduals.reduce((s, r) => s + r ** 2, 0) / calResiduals.length;
  const targetR2s = [0.30, 0.40, 0.50];
  for (const tR2 of targetR2s) {
    const reduction = Math.sqrt((1 - tR2) / (1 - currentR2));
    const simResiduals = calResiduals.map(r => r * reduction).sort((a, b) => a - b);
    console.log(`\n    Simulated R² = ${tR2.toFixed(2)} (residuals × ${reduction.toFixed(2)}):`);
    for (const targetPct of [80, 90]) {
      const target = targetPct / 100;
      const idx = Math.min(Math.ceil((simResiduals.length + 1) * target) - 1, simResiduals.length - 1);
      const qhat = simResiduals[idx];
      const mult = 10 ** qhat;
      // Apply simulated qhat to test set (assume proportional reduction in test residuals too)
      const simTestPreds = testPreds; // same point predictions
      let covered = 0;
      const widths: number[] = [];
      for (let i = 0; i < simTestPreds.length; i++) {
        const lo = simTestPreds[i] - qhat;
        const hi = simTestPreds[i] + qhat;
        const simTestResidual = Math.abs(simTestPreds[i] - testY[i]) * reduction;
        if (simTestResidual <= qhat) covered++;
        widths.push(10 ** hi - 10 ** lo);
      }
      const medWidth = widths.sort((a, b) => a - b)[Math.floor(widths.length / 2)];
      console.log(`      ${targetPct}%: range ×${mult.toFixed(2)}, width ~$${medWidth.toFixed(2)}`);
    }
  }

  // ===== SECTION 8: Summary =====
  console.log("\n\n" + "═".repeat(70));
  console.log("  SUMMARY");
  console.log("═".repeat(70));
  console.log(`
  Problem structure:
  • ${(betweenVar / totalVar * 100).toFixed(0)}% of cost variance is between tasks (predictable in theory)
  • ${(withinVar / totalVar * 100).toFixed(0)}% is within-task (same bug, different run = different cost)
  • Hand-crafted features explain R² = ${currentR2.toFixed(3)} (${(currentR2 * 100).toFixed(0)}%)
  • Gap to theoretical max: ${((embeddingCeiling - currentR2) * 100).toFixed(0)}pp

  Key insight: even with perfect features, ~${(withinVar / totalVar * 100).toFixed(0)}% variance is irreducible.
  This means 100% in-range accuracy is IMPOSSIBLE — some tasks will always surprise.

  To get to 80% in-range:
  • Need to either IMPROVE prediction (reduce residuals) or WIDEN ranges (lower precision)
  • Current: ~80% coverage needs ×${(10 ** calResiduals[Math.floor(calResiduals.length * 0.8)]).toFixed(1)} range multiplier
  • Better features (R²=0.4) would tighten this to ×${(10 ** (calResiduals[Math.floor(calResiduals.length * 0.8)] * Math.sqrt((1 - 0.4) / (1 - currentR2)))).toFixed(1)}

  Most promising paths to improve R²:
  1. Text embeddings (CodeBERT/code2vec) — capture semantic similarity better than keywords
  2. Codebase features — repo size, test suite, file complexity (requires access)
  3. Adaptive conformal — per-model, per-repo calibration (already helps)
  4. Online learning — update model after each task completes
  `);
}

main().catch(console.error);
