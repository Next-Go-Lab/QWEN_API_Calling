#!/usr/bin/env bash
set -euo pipefail

SKIP_LABEL=0
PYTHON_EXE="${PYTHON_EXE:-}"

usage() {
  echo "Usage: ./re.sh [--skip-label]"
}

if [[ $# -gt 1 ]]; then
  usage
  exit 1
fi

if [[ $# -eq 1 ]]; then
  if [[ "$1" == "--skip-label" ]]; then
    SKIP_LABEL=1
  else
    usage
    exit 1
  fi
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ -z "$PYTHON_EXE" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_EXE="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_EXE="python3"
  else
    echo "Python not found in PATH. Please install python3."
    exit 1
  fi
fi

echo "[1/4] Stop existing board_viewer.py process..."
PIDS="$(pgrep -f "python.*board_viewer\.py" || true)"
if [[ -n "$PIDS" ]]; then
  # shellcheck disable=SC2086
  kill -9 $PIDS
  echo "  Stopped PID(s): $PIDS"
else
  echo "  No existing process found."
fi

if [[ "$SKIP_LABEL" -eq 0 ]]; then
  echo "[2/4] Run label.py..."
  "$PYTHON_EXE" "label.py"
else
  echo "[2/4] Skip label.py (because --skip-label was set)."
fi

echo "[3/4] Start board_viewer.py..."
nohup "$PYTHON_EXE" "board_viewer.py" >"/tmp/board_viewer.log" 2>&1 &
echo "  Started PID $!"

sleep 1

echo "[4/4] Refresh browser page..."
URL="http://127.0.0.1:8000/"
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL" >/dev/null 2>&1 || true
elif command -v gio >/dev/null 2>&1; then
  gio open "$URL" >/dev/null 2>&1 || true
else
  echo "  Open manually: $URL"
fi

echo "Done."
