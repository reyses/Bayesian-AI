# RESUME AFTER REBOOT — 2026-07-26

Reboot reason: NVIDIA driver mismatch (595.71 loaded vs 595.84 installed) broke
CUDA. Reboot loads the matching 595.84 module.

## First 3 steps when the box is back
1. **Verify CUDA:** `nvidia-smi` (should show the GPU, no NVML error) and
   `cat /sys/module/nvidia/version` (should read 595.84).
2. **Pin the driver:** `bash ~/pin_nvidia_driver.sh` (stops unattended-upgrades
   silently breaking CUDA again — 2nd incident).
3. **Confirm bridge:** send `/health` from the phone; tg-ingress + timers are
   systemd-enabled so they auto-restart. Re-arm the session inbox consumer
   (Monitor on tools/telegram_bridge/inbox_stream.sh) when the session reopens.

## The two GPU jobs that were waiting on CUDA
A. **Dojo live-verify** (confirm the retrieval fix in the real pipeline):
   `python research/dojo_forge/pipeline/eval_native_memo.py --days
   2025_04_08,2025_04_09 --use-memory on --write-memos on --guard reflection
   --gauge on --memo-system-file research/dojo_forge/genome/sprints/overlay_hyp.txt
   --knowledge v2 --curation 4 --db .../teacher_memory_verify.db --ledger
   .../memory_ledger_verify.jsonl --limit-per-day 2 --arm-tag dojo_verify
   --num-ctx 16384` — then check day-2 episodes show retrievals>0.
B. **Rich-feature probe** (the REAL blackboard-target test): materialize 185D
   F-space (core_v2/build_dataset) then run a flexible model predicting
   TRADE-OUTCOME (not raw return — that's efficient, confirmed) walk-forward
   OOS. Reuse the harness in research/edge_probe/tools/probe_forward_return.py.

## Program state (2026-07-26)
- EXIT: closed. Ride + 50pt catastrophic floor. Exhaustively proven; never-bail
  optimal (binary/trail/scale/gauge/layered all lose). LegExitEngine committed.
- RATIFICATIONS: all 3 retired (exit-premised, exit is edgeless).
- DOJO: complete. Meaning-loop proven, retrieval bug fixed, repointed as
  blackboard substrate. (DOJO_COMPLETE_2026-07-26.md)
- NEW NORTH STAR: blackboard — mamba(finder) <-> qwen(interpreter) <->
  bank(bus), time-sliced. (docs/northstar/BLACKBOARD_ARCH.md) Prereqs: probe
  (raw-return done=efficient; rich-feature version pending GPU), fix retrieval
  (done), extraction protocol (to build).
- EDGE LOCATION: entry/trade-outcome (wrong-direction score 73%, passed
  tune/holdout), NOT exit, NOT raw price.
- OPEN owner items: Fable-dojo scope; lockbox (one-shot); whether to pivot the
  program formally to entry/discovery.
