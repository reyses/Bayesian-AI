#!/bin/bash
# Monitor the gen-0 qwen teacher acceptance run: every INTERVAL seconds parse the
# run log for per-episode timing and push progress (done/total, s/episode, s/frame,
# taint count, ETA) to the owner's Telegram. Exits with a final summary when the
# run prints "[done]" or the process dies. (Owner directive 2026-07-22: "monitor
# strongly what's the time it takes to do iterations".)
LOG="${1:-/home/moi/gen0_run.log}"
INTERVAL="${2:-300}"
REPO=/media/moi/WindowsCode/Bayesian-AI
set -a; . "$REPO/.env"; set +a

send() {
  curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d chat_id="${TELEGRAM_CHAT_ID}" --data-urlencode text="$1" >/dev/null 2>&1
}

stats() {
  python3 - "$LOG" <<'PY'
import re, sys
log = open(sys.argv[1], errors="replace").read()
eps = re.findall(r"\[(\d+)/(\d+)\]\s+(\S+):\s+(\d+)\s+frames\s+([\d.]+)s", log)
taints = log.count("TAINTED")
if not eps:
    print("starting up — no episodes finished yet (model loading)"); sys.exit()
done, total = int(eps[-1][0]), int(eps[-1][1])
times = [float(e[4]) for e in eps]; frames = sum(int(e[3]) for e in eps)
avg_ep = sum(times)/len(times); avg_fr = sum(times)/max(frames,1)
eta_min = (total-done)*avg_ep/60
print(f"{done}/{total} episodes | {avg_ep:.1f}s/episode, {avg_fr:.1f}s/frame "
      f"| taints so far: {taints} | ETA ~{eta_min:.0f} min")
PY
}

send "🏃 gen-0 qwen run monitor armed (updates every $((INTERVAL/60)) min)."
while true; do
  sleep "$INTERVAL"
  if grep -q "\[done\]" "$LOG" 2>/dev/null; then
    send "🏁 gen-0 qwen run COMPLETE: $(stats). Full log: gen0_run.log; csv: research/dojo_forge/reports/acceptance_native_gen0.csv"
    exit 0
  fi
  if ! pgrep -f "eval_native_ckpt.py" >/dev/null; then
    send "⚠️ gen-0 run process not found and no [done] marker — it may have crashed. Last state: $(stats)"
    exit 1
  fi
  send "⏱ gen-0 progress: $(stats)"
done
