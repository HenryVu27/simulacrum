#!/usr/bin/env bash
# Runs one job on the GPU PC, inside WSL. remote.sh starts it; do not call it by hand.
#
# 1. Waits until the GPU is free: the lock folder C:\gpu-lock (shared with the other project) does
#    not exist and less than BUSY_MB of GPU memory is in use (the Windows desktop alone uses about 3 GB).
# 2. Takes the lock with mkdir (atomic) and writes its own owner line. It only ever removes a lock it
#    created and whose owner line is still its own.
# 3. Refuses to run if the code in ~/umwelt is no longer the code pushed for this job, or if JAX does
#    not see the GPU (remote.sh run --allow-cpu writes results/remote/<name>.allow_cpu to allow a CPU
#    job). Records both checks in the job's log.
# 4. Runs results/remote/<name>.cmd in its own process group. SIGTERM, SIGINT or SIGHUP to this script
#    stops that whole group first; only then is the lock released. A SIGKILL leaves the lock in place
#    (held too long, never released early); remote.sh unlock clears it.
#
# UMWELT_ROOT, UMWELT_LOCK and UMWELT_SMI exist so test_job.sh can run this on the Mac with fakes.
set -u
set -m  # background commands get their own process group, so a stop signal reaches all of the job
name=$1
root=${UMWELT_ROOT:-$HOME/umwelt}
lock=${UMWELT_LOCK:-/mnt/c/gpu-lock}
smi=${UMWELT_SMI:-/usr/lib/wsl/lib/nvidia-smi}
cd "$root" || exit 1
dir=results/remote
log=$dir/$name.log
now() { date '+%Y-%m-%dT%H:%M:%S%z'; }
token="project=umwelt job=$name host=$(hostname) pid=$$ started=$(now)"
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # take GPU memory as needed, not 75% up front: the card is shared
have_lock=0
child=

say() { echo "[$(now)] $*" >> "$log"; }
finish() {
  if [ "$have_lock" = 1 ]; then
    owner=$(head -n 1 "$lock/owner" 2>/dev/null || true)
    if [ -z "$owner" ] || [ "$owner" = "$token" ]; then
      rm -rf "$lock"
    else
      say "lock not released: its owner is now '$owner'"
    fi
  fi
  rm -f "$dir/$name.pid"
}
stop() {
  say "stop signal: stopping the command"
  if [ -n "$child" ]; then
    kill -TERM -- "-$child" 2>/dev/null || kill -TERM "$child" 2>/dev/null
    wait "$child" 2>/dev/null
  fi
  echo 143 > "$dir/$name.exit"
  exit 143
}
end() { say "$2"; echo "$1" > "$dir/$name.exit"; exit "$1"; }
trap finish EXIT
trap stop TERM INT HUP
echo $$ > "$dir/$name.pid"

tries=0
while true; do
  used=$("$smi" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' \r')
  if ! [[ "$used" =~ ^[0-9]+$ ]]; then
    tries=$((tries + 1))
    say "nvidia-smi gave no memory reading ('$used'), attempt $tries of 10"
    [ "$tries" -ge 10 ] && end 1 "gave up: cannot read GPU memory"
  elif [ "$used" -lt "${BUSY_MB:-6000}" ] && mkdir "$lock" 2>/dev/null; then
    have_lock=1
    echo "$token" > "$lock/owner"
    break
  else
    say "waiting: GPU memory in use ${used} MiB; lock: $(head -n 1 "$lock/owner" 2>/dev/null || echo none)"
  fi
  sleep "${POLL_S:-60}"
done
say "lock taken: $token"

pushed=$(grep '^code ' "$dir/$name.push" 2>/dev/null)
current=$(grep '^code ' .umwelt_push 2>/dev/null)
[ -n "$pushed" ] && [ "$pushed" = "$current" ] || end 3 "refused: code changed since this job was queued (job: '$pushed', now: '$current')"
say "code: $(tr '\n' ' ' < "$dir/$name.push")"
uv sync --frozen --extra gpu --quiet >> "$log" 2>&1 || end 4 "refused: uv sync failed"
export UV_NO_SYNC=1  # from here on uv run never changes the environment (a plain sync would drop the CUDA plugin)
backend=$(uv run python -c "import jax; print(jax.default_backend(), jax.devices()[0].device_kind)" 2>> "$log")
say "JAX backend: $backend"
case "$backend" in
  gpu*) ;;
  *) [ -f "$dir/$name.allow_cpu" ] || end 5 "refused: JAX does not see the GPU (remote.sh run --allow-cpu for a CPU job)" ;;
esac

say "start: $(cat "$dir/$name.cmd")"
bash "$dir/$name.cmd" >> "$log" 2>&1 &
child=$!
wait "$child"
code=$?
child=
end "$code" "exit $code"
