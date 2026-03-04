#!/usr/bin/env node

import { Command } from "commander";
import { runEstimate } from "./commands/estimate.js";
import { runReport } from "./commands/report.js";
import { runSetup } from "./commands/setup.js";

const program = new Command();

program
  .name("tarmac-cost")
  .description(
    "Pre-flight cost estimation for Claude Code. Know what your AI task will cost before it runs."
  )
  .version("0.1.0");

program
  .command("setup")
  .description(
    "Install Tarmac hooks into Claude Code settings for automatic cost estimation"
  )
  .action(async () => {
    await runSetup();
  });

program
  .command("estimate")
  .description(
    "Estimate cost for a prompt (called by UserPromptSubmit hook — reads stdin)"
  )
  .action(async () => {
    await runEstimate();
  });

program
  .command("report")
  .description(
    "Record actual outcome after task completion (called by Stop hook — reads stdin)"
  )
  .action(async () => {
    await runReport();
  });

program.parse();
