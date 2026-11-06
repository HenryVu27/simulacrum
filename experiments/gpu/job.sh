#!/usr/bin/env bash
# Runs one job on the GPU PC, inside WSL. remote.sh starts it; do not call it by hand.
#
# 1. Waits until the GPU is free: the lock folder C:\gpu-lock (shared with the other project) does
#    not exist and less than BUSY_MB of GPU memory is in use (the Windows desktop alone uses about 3 GB).
# 2. Takes the lock with mkdir (atomic) and writes its own owner line. It only ever removes a lock whose
#    owner line is exactly its own (an empty one may be another project's, just created).
# 3. Refuses to run if the code in ~/umwelt is no longer the code pushed for this job (the files are
#    rehashed, not only the manifest), or if JAX does not see the GPU (remote.sh run --allow-cpu writes
#    results/remote/<name>.allow_cpu to allow a CPU job; otherwise JAX_PLATFORMS=cuda,cpu makes a
#    missing GPU an error, never a quiet CPU fallback). Records both checks in the job's log.
# 4. Runs results/remote/<name>.cmd in its own process group. SIGTERM, SIGINT or SIGHUP to this script
#    stops that whole group first; anything the command leaves running is stopped too; only then is the
#    lock released. A SIGKILL leaves the lock in place (held too long, never released early); remote.sh
#    unlock clears it.
# 5. Results that will be pooled with the Mac's must share recordings (docs/gpu.md, E7b), so a job may not
#    record a world (UMWELT_NO_RECORD=1) unless started with remote.sh run --record. It caps JAX at half the
#    card (8 GB) so a job cannot crowd Henry or the other project, and passes the pushed code version on to the
#    results' provenance (UMWELT_CODE).
# 6. While waiting, it removes a lock left by an umwelt job that is no longer running (a restart or a kill),
#    so a dead umwelt job cannot block the other project's jobs for long. Never another project's lock.
# 7. Pause switch, for when Henry wants the PC: while C:\gpu-pause exists (a file or a folder), a job
#    does not start, and a running job's processes are stopped (SIGSTOP) until it is deleted, then
#    continued (SIGCONT). A paused job keeps the lock and the GPU memory it holds, but uses no GPU time.
#    remote.sh pause and remote.sh resume do the same from the Mac.
#
# UMWELT_ROOT, UMWELT_LOCK, UMWELT_SMI and UMWELT_PAUSE exist so test_job.sh can run this on the Mac
# with fakes.
set -u
set -m  # background commands get their own process group, so a stop signal reaches all of the job
name=$1
root=${UMWELT_ROOT:-$HOME/umwelt}
lock=${UMWELT_LOCK:-/mnt/c/gpu-lock}
smi=${UMWELT_SMI:-/usr/lib/wsl/lib/nvidia-smi}
pause=${UMWELT_PAUSE:-/mnt/c/gpu-pause}
cd "$root" || exit 1
dir=results/remote
log=$dir/$name.log
now() { date '+%Y-%m-%dT%H:%M:%S%z'; }
token="project=umwelt job=$name host=$(hostname) pid=$$ started=$(now)"
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # take GPU memory as needed, not 75% up front: the card is shared
# A persistent compile cache: compiling was nearly half of a mind's time on the GPU, and the cache changed
# no result (docs/gpu.md, E6 follow-up). Keys include the program, precision and XLA flags.
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$root/results/jax_cache}
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_COMPILATION_CACHE_MAX_SIZE=${JAX_COMPILATION_CACHE_MAX_SIZE:-2000000000}  # oldest entries go past 2 GB
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}  # at most half the card (cuda_async ignores it)
have_lock=0
child=

say() { echo "[$(now)] $*" >> "$log"; }
is_job() { [ -n "$1" ] && ps -p "$1" -o args= 2>/dev/null | grep -q "job.sh $2\$"; }  # a process id can be reused
dead_umwelt_lock() {  # the lock is an umwelt job's, and that job's job.sh is not running
  local o p j
  o=$(head -n 1 "$lock/owner" 2>/dev/null) || return 1
  case "$o" in project=umwelt*) ;; *) return 1 ;; esac
  p=$(echo "$o" | sed -n 's/.* pid=\([0-9]*\) .*/\1/p')
  j=$(echo "$o" | sed -n 's/.* job=\([^ ]*\) .*/\1/p')
  ! is_job "$p" "$j"
}
finish() {
  if [ "$have_lock" = 1 ]; then
    owner=$(head -n 1 "$lock/owner" 2>/dev/null || true)
    if [ "$owner" = "$token" ]; then
      rm -rf "$lock"
    else
      say "lock not released: its owner is now '$owner'"
    fi
  fi
  rm -f "$dir/$name.pid"
}
reap() {  # stop whatever the command left running in its process group, so nothing outlives the lock
  kill -TERM -- "-$1" 2>/dev/null || return 0
  kill -CONT -- "-$1" 2>/dev/null
  sleep 2
  kill -KILL -- "-$1" 2>/dev/null
  return 0
}
stop() {
  say "stop signal: stopping the command"
  if [ -n "$child" ]; then
    kill -TERM -- "-$child" 2>/dev/null || kill -TERM "$child" 2>/dev/null
    kill -CONT -- "-$child" 2>/dev/null  # a paused command only acts on the TERM once continued
    wait "$child" 2>/dev/null
    reap "$child"
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
  if [ -d "$lock" ] && dead_umwelt_lock; then
    say "removed a lock whose umwelt job is gone: $(head -n 1 "$lock/owner")"
    rm -rf "$lock"
  fi
  if [ -e "$pause" ]; then
    say "waiting: paused ($pause exists)"
  elif ! [[ "$used" =~ ^[0-9]+$ ]]; then
    tries=$((tries + 1))
    say "nvidia-smi gave no memory reading ('$used'), attempt $tries of 10"
    [ "$tries" -ge 10 ] && end 1 "gave up: cannot read GPU memory"
  elif [ "$used" -lt "${BUSY_MB:-6000}" ] && mkdir "$lock" 2>/dev/null; then
    have_lock=1
    echo "$token" > "$lock/owner"
    break
  else
    owner=$(head -n 1 "$lock/owner" 2>/dev/null || echo none)
    case "$owner" in project=umwelt*|none) ;; *) owner="held by another project" ;; esac  # logs are published: not its details
    say "waiting: GPU memory in use ${used} MiB; lock: $owner"
  fi
  sleep "${POLL_S:-60}"
done
say "lock taken: $token"

pushed=$(grep '^code ' "$dir/$name.push" 2>/dev/null)
current=$(grep '^code ' .umwelt_push 2>/dev/null)
files="code $(tr '\n' '\0' < .umwelt_files 2>/dev/null | xargs -0 sha256sum 2>/dev/null | sha256sum | cut -c1-16)"
[ -n "$pushed" ] && [ "$pushed" = "$current" ] && [ "$pushed" = "$files" ] \
  || end 3 "refused: code changed since this job was queued (job: '$pushed', manifest now: '$current', files now: '$files')"
say "code: $(tr '\n' ' ' < "$dir/$name.push")"
UMWELT_CODE=$(tr '\n' ' ' < "$dir/$name.push")
export UMWELT_CODE
[ -f "$dir/$name.allow_record" ] || export UMWELT_NO_RECORD=1  # give it its recordings, or run --record
uv sync --frozen --extra gpu --quiet >> "$log" 2>&1 || end 4 "refused: uv sync failed"
export UV_NO_SYNC=1  # from here on uv run never changes the environment (a plain sync would drop the CUDA plugin)
[ -f "$dir/$name.allow_cpu" ] || export JAX_PLATFORMS=cuda,cpu  # no GPU is then an error, not a CPU fallback
backend=$(uv run python -c "import jax; print(jax.default_backend(), jax.devices()[0].device_kind)" 2>> "$log")
say "JAX backend: $backend"
case "$backend" in
  gpu*) ;;
  *) [ -f "$dir/$name.allow_cpu" ] || end 5 "refused: JAX does not see the GPU (remote.sh run --allow-cpu for a CPU job)" ;;
esac
# Without filelock JAX cannot keep the size cap, and then every cache read and write fails with a single
# warning: every job from c0e0cd4 on compiled from scratch (docs/gpu.md). The cap goes before the cache does.
if ! uv run python -c "import filelock" 2> /dev/null; then
  export JAX_COMPILATION_CACHE_MAX_SIZE=-1
  say "compile cache: filelock missing, size cap dropped"
fi
cached() { find "$JAX_COMPILATION_CACHE_DIR" -type f -name '*-cache' 2> /dev/null | wc -l | tr -d ' '; }
cached0=$(cached)

say "start: $(cat "$dir/$name.cmd")"
bash "$dir/$name.cmd" >> "$log" 2>&1 &
child=$!
paused=0
while kill -0 "$child" 2>/dev/null; do
  if [ -e "$pause" ] && [ "$paused" = 0 ]; then
    kill -STOP -- "-$child" 2>/dev/null && paused=1 && say "paused ($pause exists): command stopped, lock kept"
  elif [ ! -e "$pause" ] && [ "$paused" = 1 ]; then
    kill -CONT -- "-$child" 2>/dev/null; paused=0; say "resumed"
  fi
  sleep "${PAUSE_POLL_S:-5}"
done
wait "$child"
code=$?
reap "$child"
child=
say "compile cache: $cached0 entries before, $(cached) after; $(grep -c 'persistent compilation cache' "$log") cache errors"
end "$code" "exit $code"
