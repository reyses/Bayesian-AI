#!/bin/bash
# AG SENTINEL — the nuke precaution (owner 2026-07-24): every 2h an
# INDEPENDENT-VENDOR AI (Antigravity) verifies the daemon + Telegram leg.
# Design: bash = hands (probes + repairs, deterministic, auditable);
# agy = brain (judges the evidence, no shell access — nothing to hijack).
# Silent when healthy (log line only); loud on failure (repair + phone +
# desktop). If agy itself is down, that too is a finding -> mechanical alert.
HERE="$(cd "$(dirname "$0")" && pwd)"
STATE="$HERE/state"
LOG="$STATE/ag_sentinel.log"
AGY="$HOME/.local/bin/agy"
set -a; . "$HERE/../../.env" 2>/dev/null; set +a

ts() { date '+%F %T'; }

# ---- gather evidence (mechanical) -----------------------------------------
ING=$(systemctl --user is-active tg-ingress.service 2>&1)
VER=$(systemctl --user is-active tg-verify.timer 2>&1)
WDG=$(systemctl --user is-active tg-watchdog.timer 2>&1)
HB_AGE=9999
[ -f "$STATE/heartbeat.txt" ] && HB_AGE=$(( $(date +%s) - $(stat -c %Y "$STATE/heartbeat.txt") ))
GETME=$(curl -sm 20 "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getMe" | head -c 60)
N=$(wc -l < "$STATE/inbox.jsonl" 2>/dev/null || echo 0)
C=$(head -c 16 "$STATE/consumed.txt" 2>/dev/null || echo 0)
EVIDENCE="ingress=$ING verify_timer=$VER watchdog_timer=$WDG heartbeat_age_s=$HB_AGE telegram_getMe=$GETME queue=$N/$C"

# ---- independent judgment (Antigravity, no shell) -------------------------
VERDICT=$(timeout 150 "$AGY" --prompt "You are a monitoring judge. Evidence from a Telegram-bridge health probe:
$EVIDENCE
Healthy means: ingress=active, both timers=active, heartbeat_age_s < 180, telegram_getMe contains '\"ok\":true', queue counts equal or nearly equal.
Reply EXACTLY one line: HEALTHY or UNHEALTHY: <short reason>. No other text." 2>/dev/null | tail -1)

echo "$(ts) verdict=${VERDICT:-AGY_SILENT} | $EVIDENCE" >> "$LOG"

# ---- act (mechanical) ------------------------------------------------------
alert() {
  curl -sm 20 "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d chat_id="${TELEGRAM_CHAT_ID}" --data-urlencode text="$1" >/dev/null 2>&1
  DISPLAY=:0 DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/bus" \
    notify-send -u critical "ag-sentinel" "$1" 2>/dev/null
}

case "$VERDICT" in
  HEALTHY*) exit 0 ;;
  UNHEALTHY*)
    systemctl --user restart tg-ingress.service tg-verify.timer tg-watchdog.timer
    alert "🛰️ AG-sentinel (independent vendor): $VERDICT — evidence: $EVIDENCE. Restarted all bridge legs; next probe in 2h. /health to re-check now."
    ;;
  *)
    # agy itself failed — the diverse-vendor leg is down; that is reportable
    # on its own, and the mechanical health facts still get judged crudely.
    if [ "$ING" != "active" ] || [ "$HB_AGE" -gt 180 ]; then
      systemctl --user restart tg-ingress.service
      alert "🛰️ AG-sentinel: agy gave no verdict AND bridge looks unhealthy ($EVIDENCE) — restarted ingress."
    else
      alert "🛰️ AG-sentinel: bridge looks fine mechanically, but the Antigravity judge did not answer — the independent-vendor leg needs attention. ($EVIDENCE)"
    fi
    ;;
esac
