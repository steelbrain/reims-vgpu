#!/usr/bin/env bash
# Compare Chrome's default ANGLE/Metal path with its ANGLE/SwiftShader fallback.
# Uses fresh profiles and never stops or modifies the user's existing browser.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUEST="${GUEST:-macos-vm}"
PORT="${CDP_PORT:-19321}"
CHROME='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1024 || PORT > 65535 )); then
  echo "CDP_PORT must be an unprivileged TCP port" >&2
  exit 64
fi

if ! command -v python3 >/dev/null || ! command -v ssh >/dev/null; then
  echo "This probe requires host python3 and ssh" >&2
  exit 69
fi

OUT="$(mktemp -d "${TMPDIR:-/tmp}/reims-angle-backend.XXXXXX")"
SSH_CONTROL="$OUT/ssh-control"
TUNNEL_PID=""
REMOTE_RUN=""
ssh_guest() {
  ssh -S "$SSH_CONTROL" "$GUEST" "$@"
}
cleanup() {
  if [[ -n "$TUNNEL_PID" ]]; then
    kill "$TUNNEL_PID" 2>/dev/null || true
    wait "$TUNNEL_PID" 2>/dev/null || true
  fi
  if [[ -n "$REMOTE_RUN" && -n "${REMOTE_PID:-}" ]]; then
    ssh_guest "kill -TERM '$REMOTE_PID' >/dev/null 2>&1 || true" >/dev/null 2>&1 || true
  fi
  ssh -S "$SSH_CONTROL" -O exit "$GUEST" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

wait_for_cdp() {
  local i
  for i in {1..30}; do
    if PYTHONPATH="$SCRIPT_DIR" CDP_ENDPOINT="127.0.0.1:$PORT" python3 -c \
      'import cdp; cdp.http_json(cdp.DEFAULT_ENDPOINT, "/json/version")' \
      2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  return 1
}

run_case() {
  local backend="$1" remote_dir pid report_status=0
  remote_dir="$(ssh_guest 'mktemp -d /tmp/reims-angle-backend.XXXXXX')"
  if ! [[ "$remote_dir" =~ ^/tmp/reims-angle-backend\.[A-Za-z0-9]+$ ]]; then
    echo "Unexpected guest temp directory: $remote_dir" >&2
    return 1
  fi

  # Each case gets its own profile and Chrome process. Only its ANGLE backend
  # changes; neither the existing profile nor persistent guest settings change.
  ssh_guest "
    CHROME='$CHROME'
    PROFILE='$remote_dir/profile'
    LOG='$remote_dir/chrome.log'
    mkdir -p \"\$PROFILE\"
    if [[ '$backend' == default ]]; then
      nohup \"\$CHROME\" --user-data-dir=\"\$PROFILE\" --no-first-run \\
        --no-default-browser-check --disable-background-networking \\
        --remote-debugging-port='$PORT' --enable-logging=stderr \\
        chrome://gpu >\"\$LOG\" 2>&1 </dev/null &
    else
      nohup \"\$CHROME\" --user-data-dir=\"\$PROFILE\" --no-first-run \\
        --no-default-browser-check --disable-background-networking \\
        --remote-debugging-port='$PORT' --enable-logging=stderr \\
        --use-gl=angle --use-angle='$backend' chrome://gpu \\
        >\"\$LOG\" 2>&1 </dev/null &
    fi
    echo \$! > '$remote_dir/chrome.pid'
  "
  pid="$(ssh_guest "cat '$remote_dir/chrome.pid'")"
  if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
    echo "Could not validate diagnostic Chrome PID" >&2
    return 1
  fi
  REMOTE_RUN="$remote_dir"
  REMOTE_PID="$pid"

  # Forward the guest's loopback-only DevTools endpoint without exposing it to
  # the LAN. Same port locally so Chrome's advertised WebSocket URL stays valid.
  ssh -o ExitOnForwardFailure=yes -S "$SSH_CONTROL" -N \
    -L "$PORT:127.0.0.1:$PORT" "$GUEST" >"$OUT/$backend-tunnel.log" 2>&1 &
  TUNNEL_PID=$!
  if ! wait_for_cdp; then
    ssh_guest "grep -iE 'ANGLE|EGL|Metal|GPU process|GLDisplay|error' '$remote_dir/chrome.log' | tail -80" \
      >"$OUT/$backend-chrome.log" 2>&1 || true
    echo "Chrome DevTools did not start for $backend; see $OUT/$backend-chrome.log" >&2
    return 1
  fi

  {
    echo "=== ANGLE backend: $backend ==="
    PYTHONPATH="$SCRIPT_DIR" CDP_ENDPOINT="127.0.0.1:$PORT" python3 "$SCRIPT_DIR/chrome_gpu_report.py"
    echo
    echo "--- WebGL context probe ---"
    PYTHONPATH="$SCRIPT_DIR" CDP_ENDPOINT="127.0.0.1:$PORT" python3 "$SCRIPT_DIR/cdp.py" eval \
      'data:text/html,ANGLE%20probe' \
      '(() => { const c=document.createElement("canvas"); const g=c.getContext("webgl2")||c.getContext("webgl"); if(!g)return {context:false}; const e=g.getExtension("WEBGL_debug_renderer_info"); return {context:true,version:g.getParameter(g.VERSION),vendor:g.getParameter(g.VENDOR),renderer:e?g.getParameter(e.UNMASKED_RENDERER_WEBGL):g.getParameter(g.RENDERER)}; })()'
  } >"$OUT/$backend.txt" 2>&1 || report_status=$?
  cat "$OUT/$backend.txt"
  if (( report_status != 0 )); then
    echo "Probe collection failed for $backend (status $report_status); preserving logs" \
      | tee -a "$OUT/$backend.txt"
  fi

  ssh_guest "grep -iE 'ANGLE|EGL|Metal|GPU process|GLDisplay|error' '$remote_dir/chrome.log' | tail -80" \
    >"$OUT/$backend-chrome.log" 2>&1 || true
  echo "--- relevant Chrome log lines ---"
  cat "$OUT/$backend-chrome.log"

  ssh_guest "kill -TERM '$pid' >/dev/null 2>&1 || true"
  REMOTE_PID=""
  REMOTE_RUN=""
  kill "$TUNNEL_PID" 2>/dev/null || true
  wait "$TUNNEL_PID" 2>/dev/null || true
  TUNNEL_PID=""
  echo "Guest diagnostic files preserved at $remote_dir" | tee -a "$OUT/$backend.txt"
  return "$report_status"
}

# Sequential runs avoid competing for the guest's virtual GPU. Default is the
# key control: it tests Chromium's own ANGLE selection/fallback policy.
ssh -M -S "$SSH_CONTROL" -o ControlPersist=10m "$GUEST" true
run_case default
run_case metal
run_case swiftshader || true
echo "Results preserved at $OUT"
