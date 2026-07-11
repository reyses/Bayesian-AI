# Bayesian-AI: Ways of Working (WoW)
*Note: This is a high-level summary. Always refer to the specialized skills for deeper instructions.*

## 1. Research Discipline & Rigor
- **Structure First**: Always conduct research *before* writing production code. Use the DMAIC cycle for projects and PDCA for individual cycles. *(See `research-discipline` skill)*
- **Organization**: Maintain exactly ONE organized `research/<topic>/` folder (using `pipeline/`, `builders/`, `tools/`, and `reports/` subfolders). Never mix files into the shared top-level reports directory. *(See `research-discipline` skill)*
- **Statistical Firewall**: All findings must pass strict causal rigor checks (base-rate analysis, null-controls, lookahead firewalls, and CI/significance testing) before being claimed as valid. *(See `causal-rigor` and `research-paper-formatting` skills)*

## 2. Coding & Architectural Standards
- **State Tracking**: Enforce JSON parameter state tracking across the architecture. *(See `bayesian-ai-coding-standards` skill)*
- **Resource Management**: Implement dynamic CPU/GPU RAM optimization guardrails. *(See `bayesian-ai-coding-standards` skill)*
- **Telemetry**: Use the `core_v2.telemetry` JSON IPC standard for safely integrating multi-process progress bars into scripts. *(See `bayesian-ai-telemetry` skill)*

## 3. Communication & Artifacts
- **Evidence Presentation**: Do not use "walls of text". Enrich walkthroughs and reports with Markdown data tables, distributions, and embedded plots. *(See `walkthrough-enrichment` skill)*
- **Visual Validation**: All evidence plots must be tightly zoomed and visually validated for legibility before presentation. Actively use animated GIFs where dynamic context helps. *(See `legible-evidence-plotting`, `animate-market-regime`, and `inline-gif-trade-lifecycle` skills)*
- **Collaboration**: When collaborating with Claude via the file-based comms channel, act as a full partner. Execute tasks efficiently, but proactively push back if Claude's technical approach is wrong. *(See `comms-collaboration` skill)*
- **Mobile Review**: Proactively use the TelegramNotifier to beam critical files (MD, plots, CSVs) directly to the user's phone for immediate review. *(See `telegram-file-delivery` skill)*

## 4. Operational Habits
- **Tool Reusability**: Always check the tools index before building something new. Save reusable tools when finished. *(See `research-discipline` skill)*
- **Persistence**: Write every intermediate and final result to a file immediately. *(See `research-discipline` skill)*
- **Read-Only Auditing**: When requested to "audit", "review", or "evaluate" a codebase or component, limit actions strictly to evaluation and review. Do not modify or edit the codebase being reviewed unless explicitly instructed otherwise.
- **Session Wrap-Up**: At the end of every coding session, automatically write an exit report to `docs/daily/YYYY-MM-DD.md` using the strict 4-point structure. *(See `update-daily-journal` skill)*
