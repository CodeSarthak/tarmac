Output demo.gif

Set Shell "bash"
Set FontSize 15
Set Width 1000
Set Height 550
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 35ms

# Silently alias commands to no-ops and set clean prompt
Hide
Type "alias npm=':' && alias tarmac-cost=':' && alias claude=':' && export PS1='$ '"
Enter
Sleep 500ms
Show
Ctrl+L
Sleep 200ms

# --- Scene 1: Install ---
Type "npm install -g tarmac-cost"
Sleep 500ms
Enter
Sleep 200ms
Type@0 "added 3 packages in 1.2s"
Enter
Sleep 1.2s

# --- Scene 2: Setup ---
Type "tarmac-cost setup"
Sleep 500ms
Enter
Sleep 200ms
Type@0 ""
Enter
Type@0 "  ✈  Tarmac v0.1 — Pre-flight cost estimation for Claude Code"
Enter
Type@0 "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Enter
Type@0 ""
Enter
Type@0 "  ✓  Hooks installed (~/.claude/settings.json)"
Enter
Type@0 "  ✓  Config saved (~/.tarmac/config.json)"
Enter
Type@0 ""
Enter
Type@0 "  ✈  Setup complete! Tarmac is now active."
Enter
Sleep 2s

# --- Scene 3: Open Claude Code ---
Type "claude"
Sleep 500ms
Enter
Sleep 200ms
Type@0 ""
Enter
Type@0 "╭─────────────────────────────────────────────╮"
Enter
Type@0 "│  Claude Code                  ◉ Opus 4.6    │"
Enter
Type@0 "╰─────────────────────────────────────────────╯"
Enter
Type@0 ""
Enter
Type@0 "> "
Sleep 1s

# --- Scene 4: User types a prompt ---
Type "Refactor the auth module to use JWT tokens instead of sessions"
Sleep 800ms
Enter
Sleep 1.5s

# --- Scene 5: Tarmac cost estimate ---
Type@0 ""
Enter
Type@0 "  ⚡ TARMAC COST ESTIMATE"
Enter
Type@0 "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Enter
Type@0 ""
Enter
Type@0 "    Sonnet 4.6    $0.45 - $1.62"
Enter
Type@0 "    Opus 4.6      $2.18 - $7.84"
Enter
Type@0 "    Haiku 4.5     $0.11 - $0.39"
Enter
Type@0 ""
Enter
Type@0 "    Task type: refactoring"
Enter
Type@0 "    Input: 847 tokens"
Enter
Type@0 "    Coverage: 80% confidence interval"
Enter
Type@0 ""
Enter
Type@0 "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Enter
Type@0 ""
Enter
Sleep 1.2s

# --- Scene 6: Claude presents it ---
Type@0 "  This task is estimated at $2.18 - $7.84 on Opus 4.6."
Enter
Type@0 "  Want to proceed, or would you prefer a cheaper model?"
Enter
Type@0 "  (Run: claude --model sonnet for ~$0.45 - $1.62)"
Enter

Sleep 4s
