#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  watch_jsonl_and_report.sh [--log LOGFILE] <jsonl> <out_md> [target_lines] [sleep_seconds]

Examples:
  ./watch_jsonl_and_report.sh outputs/run.jsonl outputs/run_report.md
  ./watch_jsonl_and_report.sh --log outputs/watch.log outputs/run.jsonl outputs/run_report.md 200 30
EOF
}

LOGFILE=""
if [[ ${1:-} == "--log" ]]; then
  LOGFILE=${2:-""}
  if [[ -z "$LOGFILE" ]]; then
    usage
    exit 2
  fi
  shift 2
fi

JSONL=${1:-""}
OUT_MD=${2:-""}
TARGET_LINES=${3:-200}
SLEEP_S=${4:-60}

if [[ -z "$JSONL" || -z "$OUT_MD" ]]; then
  usage
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: python venv not found at $PY" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUT_MD")"

if [[ -n "$LOGFILE" ]]; then
  mkdir -p "$(dirname "$LOGFILE")"
  exec >>"$LOGFILE" 2>&1
fi

trap 'echo "[watch] interrupted"; exit 130' INT
trap 'code=$?; if [[ $code -ne 0 ]]; then echo "[watch] exit code=$code"; fi' EXIT

echo "[watch] jsonl=$JSONL target_lines=$TARGET_LINES out=$OUT_MD"

while true; do
  if [[ -f "$JSONL" ]]; then
    # wc output: '<lines> <file>'
    LINES=$(wc -l < "$JSONL" | tr -d ' ')
  else
    LINES=0
  fi

  TS=$(date '+%F %T')
  echo "[watch] $TS lines=$LINES/$TARGET_LINES file=$JSONL"

  if [[ "$LINES" -ge "$TARGET_LINES" ]]; then
    echo "[watch] reached $LINES lines; generating report -> $OUT_MD"
    "$PY" "$ROOT_DIR/analyze_cf_jsonl.py" "$JSONL" --out "$OUT_MD" > /dev/null
    echo "[watch] done"
    exit 0
  fi

  sleep "$SLEEP_S"
done
