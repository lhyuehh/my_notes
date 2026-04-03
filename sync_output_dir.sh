#!/bin/bash
# Continuous sync script: monitor-copy-sleep loop
# Usage: sync_output_dir.sh <source_dir> <dest_dir> [interval_seconds]
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <source_dir> <dest_dir> [interval_seconds]"
  exit 1
fi

SRC_DIR="$1"
DEST_DIR="$2"
INTERVAL="${3:-600}"

log() {
  local ts
  ts="$(date +'%Y-%m-%d %H:%M:%S')"
  local prefix="log of sync_output_dir "
  local line="[${ts}] ${prefix}$*"
  # Always write to stdout; fallback to stderr if stdout fails
  if ! printf '%s\n' "$line"; then
    printf '%s\n' "$line" >&2 || true
  fi
}

mkdir -p "$SRC_DIR" "$DEST_DIR"
log "Ensured directories exist: src=$SRC_DIR dest=$DEST_DIR"
if command -v rsync >/dev/null 2>&1; then
  log "Detected rsync; incremental sync enabled"
else
  log "rsync not found; using cp -a fallback"
fi

copy_once() {
  if command -v rsync >/dev/null 2>&1; then
    log "Running rsync from $SRC_DIR to $DEST_DIR"
    rsync -a --info=progress2 "$SRC_DIR"/ "$DEST_DIR"/ 2>&1 | while IFS= read -r line; do log "$line"; done || true
    log "rsync completed"
  else
    log "Running cp -a fallback from $SRC_DIR to $DEST_DIR"
    cp -a "$SRC_DIR"/. "$DEST_DIR"/ || true
    log "cp -a completed"
  fi
}

log "Start syncing: src=$SRC_DIR dest=$DEST_DIR interval=${INTERVAL}s"
trap 'log "Received signal, exiting."; exit 0' INT TERM
CYCLE=0
LAST_SYNC_MARKER="$DEST_DIR/.sync_output_dir_last_sync_marker"

while true; do
  CYCLE=$((CYCLE+1))
  log "Starting sync cycle #${CYCLE}"

  if [[ -f "$LAST_SYNC_MARKER" ]]; then
    changed_count=$(find "$SRC_DIR" -type f -newer "$LAST_SYNC_MARKER" 2>/dev/null | wc -l || echo "?")
    log "Changed since last cycle: ${changed_count}"
  else
    log "Changed since last cycle: N/A (first run)"
  fi

  # Pre-stats
  log "Pre-stats: src_files=$(find \"$SRC_DIR\" -type f 2>/dev/null | wc -l || echo '?') dest_files=$(find \"$DEST_DIR\" -type f 2>/dev/null | wc -l || echo '?') src_size=$(du -sh \"$SRC_DIR\" 2>/dev/null | awk '{print $1}' || echo '?') dest_size=$(du -sh \"$DEST_DIR\" 2>/dev/null | awk '{print $1}' || echo '?')"
  log "System: loadavg=$(cat /proc/loadavg 2>/dev/null | awk '{print $1,$2,$3}' || echo 'N/A') dest_disk($(df -h \"$DEST_DIR\" 2>/dev/null | tail -1 | awk '{print \"fs=\"$1,\"size=\"$2,\"used=\"$3,\"avail=\"$4,\"use%=\"$5}' || echo 'N/A'))"

  copy_once

  # Mark sync time and post-stats
  touch "$LAST_SYNC_MARKER" 2>/dev/null || true
  log "Post-stats: src_files=$(find \"$SRC_DIR\" -type f 2>/dev/null | wc -l || echo '?') dest_files=$(find \"$DEST_DIR\" -type f 2>/dev/null | wc -l || echo '?') src_size=$(du -sh \"$SRC_DIR\" 2>/dev/null | awk '{print $1}' || echo '?') dest_size=$(du -sh \"$DEST_DIR\" 2>/dev/null | awk '{print $1}' || echo '?')"
  log "Completed sync cycle #${CYCLE}"
  log "Heartbeat: still running (pid=$$). Sleeping ${INTERVAL}s..."
  sleep "$INTERVAL"
done