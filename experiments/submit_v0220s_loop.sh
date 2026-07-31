#!/bin/bash
set -u
cd /home/sti/Desktop/projects/whest-l31-side
export HTTP_PROXY=http://127.0.0.1:10809
export HTTPS_PROXY=http://127.0.0.1:10809
export ALL_PROXY=socks5h://127.0.0.1:10808
LOG=/tmp/whest-v0220s-submit-loop.log
echo "loop start $(date -Is)" | tee -a "$LOG"
for i in $(seq 1 80); do
  echo "=== $(date -Is) try $i ===" | tee -a "$LOG"
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 35 -x "$HTTP_PROXY" https://www.aicrowd.com/ || echo 000)
  echo "curl_http=$code" | tee -a "$LOG"
  if [ "$code" != "200" ]; then sleep 45; continue; fi
  if ! .venv/bin/python - <<'PY' >>"$LOG" 2>&1
import httpx, os, sys
try:
    with httpx.Client(timeout=35, proxy=os.environ["HTTPS_PROXY"]) as c:
        r = c.get("https://www.aicrowd.com/")
        print("httpx", r.status_code)
        sys.exit(0 if r.status_code == 200 else 1)
except Exception as e:
    print("httpx_fail", type(e).__name__, e)
    sys.exit(1)
PY
  then sleep 45; continue; fi
  uv run whest submit submission.tar.gz \
    --description "v0.2.20s pooled dual multi-CV + pooled-retrained corrector λ=0.85" \
    --watch --watch-timeout 900 --format plain \
    2>&1 | tee -a "$LOG" | tee /tmp/whest-v0220s-submit.txt
  if grep -qE 'Graded|submission id [0-9]+' /tmp/whest-v0220s-submit.txt; then
    echo SUCCESS | tee -a "$LOG"
    exit 0
  fi
  sleep 45
done
echo FAILED_ALL | tee -a "$LOG"
