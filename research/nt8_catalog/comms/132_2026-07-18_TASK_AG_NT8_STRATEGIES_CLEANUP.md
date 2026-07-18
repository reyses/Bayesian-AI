# TASK 132 — NT8 Custom Strategies folder cleanup (pre-port)
**Doc:** 132 · **Date:** 2026-07-18 · **Author:** Claude Fable · **Executor: AG**
**Status:** TASK — Moises-directed: "clean the NT8 folder of custom strategies
so only the new one lives." The new strategy arrives via P2/P4 deploy gates;
this task EMPTIES the folder safely first.

The Strategies folder is the LIVE-DEPLOY BOUNDARY (production). Order is
NON-NEGOTIABLE: inventory -> archive -> VERIFY archive -> only then remove.
1. INVENTORY: list every file in
   C:\Users\reyse\OneDrive\Documents\NinjaTrader 8\bin\Custom\Strategies\
   (name, size, mtime; check the actual Documents path — OneDrive redirection
   possible). Note whether NinjaTrader is RUNNING (process check); if running,
   note it in the report — Moises decides timing of a recompile.
2. ARCHIVE: copy EVERY file to docs\archive\NT8\pre_port_cleanup_2026-07-18\
   in the repo. git add + commit the archive ON MAIN with the inventory table
   in the commit message.
3. VERIFY: file count + per-file SHA256 match between source and archive.
   Paste the verification table in your report. NO removal before this passes.
4. REMOVE: delete the originals from the Strategies folder. Re-list to show
   it empty.
5. REPORT as the next comms number: inventory table, hash verification,
   NT8-running status, final empty listing. Claims = pasted evidence
   (protocol: your artifact lines, not process names).
Do NOT touch bin\Custom\Indicators or any other NT8 folder.
