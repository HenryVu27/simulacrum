#!/usr/bin/env bash
# Runs one job on the GPU PC, inside WSL. remote.sh starts it; do not call it by hand.
#
# Waits until the GPU is free, takes the lock shared with other projects on this PC (the folder
# C:\gpu-lock, created atomically), runs results/remote/<name>.cmd, then releases the lock.
# "Free" means no lock and less than BUSY_MB of GPU memory in use (the Windows desktop alone uses
# about 3 GB).
set -u
name=$1
cd ~/umwelt || exit 1
mkdir -p results/remote
log=results/remote/$name.log
lock=/mnt/c/gpu-lock
smi=/usr/lib/wsl/lib/nvidia-smi
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # take GPU memory as needed, not 75% up front: the card is shared

while true; do
  used=$($smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$used" -lt "${BUSY_MB:-6000}" ] && mkdir "$lock" 2>/dev/null; then
    break
  fi
  echo "[$(date -Is)] waiting: GPU memory in use ${used} MiB; lock: $(cat "$lock/owner" 2>/dev/null || echo none)" >> "$log"
  sleep 60
done
echo "project=umwelt job=$name started=$(date -Is) wsl_pid=$$" > "$lock/owner"
trap 'rm -rf "$lock"' EXIT

echo "[$(date -Is)] start: $(cat "results/remote/$name.cmd")" >> "$log"
bash "results/remote/$name.cmd" >> "$log" 2>&1
code=$?
echo "[$(date -Is)] exit $code" >> "$log"
echo "$code" > "results/remote/$name.exit"
