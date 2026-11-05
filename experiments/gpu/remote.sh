#!/usr/bin/env bash
# Run umwelt on the GPU PC (RTX 5080, inside WSL2) from the Mac. See docs/gpu.md.
#
#   remote.sh push [PATH...]                 copy the working tree (not results/), plus PATHs, to ~/umwelt
#   remote.sh setup                          one-time install (runs setup_wsl.sh)
#   remote.sh sh 'CMD'                       run a command in ~/umwelt now and wait for it (no lock: keep
#                                            it off the GPU, or small)
#   remote.sh run [--allow-cpu] [--record] NAME CMD...
#                                            start CMD as a background job; it waits for a free GPU. A job
#                                            may not record a world unless --record (pooled results share
#                                            recordings: push them); --allow-cpu lets it run without a GPU
#   remote.sh status [NAME]                  GPU use, lock holder, pause switch, last lines of NAME's log,
#                                            and whether NAME finished, is queued or running, or died
#   remote.sh stop NAME                      stop a queued or running job (its lock, if any, is released)
#   remote.sh pause | resume [--force]       create or delete C:\gpu-pause: while it exists no job starts
#                                            and running jobs are frozen. resume removes only a pause that
#                                            remote.sh made; Henry's own (mkdir C:\gpu-pause) needs --force
#   remote.sh pull [--overwrite] PATH...     copy files under ~/umwelt back here (a local file that
#                                            differs is kept, unless --overwrite)
#   remote.sh unlock [--force]               remove C:\gpu-lock left by a killed umwelt job
#
# A job survives the SSH connection closing: it is started through Windows' Win32_Process.Create,
# outside the SSH session. Each push writes .umwelt_push (code hash, git commit, uncommitted changes),
# removes files the previous push sent that are gone here, and has the PC rehash what arrived (the code,
# and any PATHs given, file by file: a stray or altered data file fails the push);
# a job keeps a copy, records it in its log and refuses to start on different code. A push that would
# change the code while an umwelt job is queued or running is refused. Job names are used once.
set -euo pipefail
self=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
PC=${UMWELT_PC:-<pc-user>@<pc-ip>}  # known_hosts has the PC under its Tailscale IP
DISTRO=${UMWELT_DISTRO:-Ubuntu-26.04}
# umwelt runs as the distribution's root user, in /root/umwelt: no Linux account had to be created, and
# naming the user keeps this working if Henry later makes his own (root in WSL has no extra Windows rights).
WSL="wsl -d $DISTRO -u ${UMWELT_WSL_USER:-root} --cd ~ --"
cd "$(dirname "$self")/../.."
export COPYFILE_DISABLE=1  # with --no-xattrs --no-mac-metadata: no macOS ._ files or extended attributes

ssh_pc() { ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 "$PC" "$@"; }
wsl_sh() { ssh_pc "$WSL bash -s"; }  # runs the bash script on stdin inside WSL
# For WSL-side scripts: is PID still job NAME's job.sh? A bare kill -0 is fooled by a reused process id.
IS_JOB='is_job() { [ -n "$1" ] && ps -p "$1" -o args= 2>/dev/null | grep -q "job.sh $2\$"; }'
usage() { sed -n '2,26p' "$self"; exit 2; }
valid_name() { [[ "$1" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "job name: letters, digits, . _ - only" >&2; exit 2; }; }
code_files() {  # tracked and new files outside results/, skipping deleted ones
  git ls-files -co --exclude-standard | grep -v '^results/' | while IFS= read -r f; do
    if [ -f "$f" ]; then printf '%s\n' "$f"; fi
  done
}
manifest() {  # the code line hashes the files listed in .umwelt_files, which push writes first
  echo "code $(tr '\n' '\0' < .umwelt_files | xargs -0 shasum -a 256 | shasum -a 256 | cut -c1-16)"
  echo "git $(git rev-parse --short HEAD)"
  echo "uncommitted $(git status --porcelain -- . ':!results' | wc -l | tr -d ' ') files, diff $(git diff HEAD -- . ':!results' | shasum -a 256 | cut -c1-16)"
  echo "pushed $(date '+%Y-%m-%dT%H:%M:%S%z')"
}

cmd=${1:-}
[ -n "$cmd" ] && shift
case "$cmd" in
  push)
    code_files > .umwelt_files
    m=$(manifest)
    state=$({ echo "$IS_JOB"; cat <<'EOF'
cd umwelt 2>/dev/null || exit 0
grep '^code ' .umwelt_push 2>/dev/null || echo "code none"
for p in results/remote/*.pid; do
  n=$(basename "$p" .pid)
  if [ -f "$p" ] && is_job "$(cat "$p")" "$n"; then echo "active $n"; fi
done
EOF
    } | wsl_sh | tr -d '\r')
    active=$(echo "$state" | sed -n 's/^active //p' | tr '\n' ' ')
    if [ -n "$active" ] && [ "$(echo "$state" | grep '^code ')" != "$(echo "$m" | grep '^code ')" ]; then
      echo "refused: the code differs from what queued or running jobs were given ($active); wait for them" >&2
      exit 1
    fi
    printf '%s\n' "$m" > .umwelt_push
    ssh_pc "$WSL mkdir -p umwelt"
    echo 'cd umwelt && if [ -f .umwelt_files ]; then cp .umwelt_files .umwelt_files.prev; else : > .umwelt_files.prev; fi' | wsl_sh
    { cat .umwelt_files; echo .umwelt_files; echo .umwelt_push; if [ $# -gt 0 ]; then printf '%s\n' "$@"; fi; } \
      | tar --no-xattrs --no-mac-metadata -cf - -T - | ssh_pc "$WSL tar -xf - -C umwelt"
    # files the previous push sent that are gone here: remove them there, or the PC keeps running them
    gone=$(echo 'cd umwelt && sort .umwelt_files.prev > .p1 && sort .umwelt_files > .p2 && comm -23 .p1 .p2; rm -f .p1 .p2 .umwelt_files.prev' | wsl_sh | tr -d '\r')
    if [ -n "$gone" ]; then
      { echo 'cd umwelt'; printf '%s\n' "$gone" | while IFS= read -r f; do printf 'rm -f -- %q\n' "$f"; done; } | wsl_sh
      echo "removed on the PC (deleted here): $(echo "$gone" | tr '\n' ' ')"
    fi
    # the PC hashes what arrived the same way: catches anything the SSH, cmd and wsl.exe pipe altered
    there=$(echo 'cd umwelt && echo "code $(tr "\n" "\0" < .umwelt_files | xargs -0 sha256sum | sha256sum | cut -c1-16)"' | wsl_sh | tr -d '\r')
    if [ "$there" != "$(echo "$m" | grep '^code ')" ]; then
      echo "push failed verification: the PC has '$there', the Mac sent '$(echo "$m" | grep '^code ')'" >&2
      exit 1
    fi
    if [ $# -gt 0 ]; then  # the data pushed with it, file by file: a stray, missing or altered file fails
      sums='LC_ALL=C sort | tr "\n" "\0" | xargs -0 SUM | SUM | cut -c1-16'
      here=$(find "$@" -type f | eval "${sums//SUM/shasum -a 256}")
      there=$({ echo 'cd umwelt'; printf 'find '; printf '%q ' "$@"; echo "-type f | ${sums//SUM/sha256sum}"; } | wsl_sh | tr -d '\r')
      if [ "$here" != "$there" ]; then
        echo "push failed verification for $*: the PC's files differ (an extra, missing or altered file). Compare:" >&2
        echo "  find $* -type f | LC_ALL=C sort | xargs shasum -a 256   (here and, with sha256sum, there)" >&2
        exit 1
      fi
    fi
    echo "pushed the working tree${*:+ and $*} to $PC:~/umwelt ($DISTRO): $(echo "$m" | head -3 | tr '\n' ' ')"
    ;;
  setup)
    "$self" push
    echo 'cd umwelt && bash experiments/gpu/setup_wsl.sh' | wsl_sh
    ;;
  sh)
    [ $# -gt 0 ] || usage
    printf 'cd umwelt && export PATH="$HOME/.local/bin:$PATH" XLA_PYTHON_CLIENT_PREALLOCATE=false && %s\n' "$*" | wsl_sh
    ;;
  run)
    allow=0
    record=0
    while [ "${1:-}" = --allow-cpu ] || [ "${1:-}" = --record ]; do
      if [ "$1" = --allow-cpu ]; then allow=1; else record=1; fi
      shift
    done
    [ $# -ge 2 ] || usage
    name=$1; shift
    valid_name "$name"
    "$self" push
    made=$({ echo "name=$name allow=$allow record=$record"
             cat <<'EOF'
cd umwelt && mkdir -p results/remote
if [ -e "results/remote/$name.cmd" ]; then echo exists; exit 0; fi
cp .umwelt_push "results/remote/$name.push"
if [ "$allow" = 1 ]; then touch "results/remote/$name.allow_cpu"; fi
if [ "$record" = 1 ]; then touch "results/remote/$name.allow_record"; fi
cat > "results/remote/$name.cmd" <<'UMWELT_CMD_END'
EOF
             printf '%q ' "$@"; echo
             echo 'UMWELT_CMD_END'
             echo 'echo created'; } | wsl_sh | tr -d '\r')
    [ "$made" = created ] || { echo "a job named $name already exists on the PC; pick a new name" >&2; exit 1; }
    ssh_pc "powershell -NoProfile -Command \"(Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='$WSL bash umwelt/experiments/gpu/job.sh $name'}).ReturnValue\"" \
      | tr -d '\r' | grep -qx 0 || { echo "could not start the job" >&2; exit 1; }
    echo "started $name; follow it with: experiments/gpu/remote.sh status $name"
    ;;
  status)
    if [ -n "${1:-}" ]; then valid_name "$1"; fi
    ssh_pc "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader" | tr -d '\r' \
      | sed 's/^/GPU (used, total, busy): /'
    { echo "name=${1:-}"; echo "$IS_JOB"
      cat <<'EOF'
lock=${UMWELT_LOCK:-/mnt/c/gpu-lock}
if [ -f "$lock/owner" ]; then sed -n '1s/^/lock: /p' "$lock/owner"; else echo "lock: free"; fi
if [ -e "${UMWELT_PAUSE:-/mnt/c/gpu-pause}" ]; then echo "pause: on (remote.sh resume, or delete C:\\gpu-pause)"; fi
if [ -n "$name" ]; then
  f=umwelt/results/remote/$name
  tail -n 15 "$f.log" 2>/dev/null
  if [ -f "$f.exit" ]; then echo "finished, exit code $(cat "$f.exit")"
  elif [ -f "$f.pid" ] && is_job "$(cat "$f.pid")" "$name"; then echo "not finished: queued or running"
  elif [ ! -e "$f.cmd" ]; then echo "no such job"
  elif [ -n "$(find "$f.cmd" -mmin -2)" ]; then echo "not finished: starting"
  else echo "dead: no exit code and no job.sh (a restart or a kill); its lock, if any, goes when the next job waits"
  fi
fi
EOF
    } | wsl_sh
    ;;
  pull)
    overwrite=0
    if [ "${1:-}" = --overwrite ]; then overwrite=1; shift; fi
    [ $# -gt 0 ] || usage
    stage=$(mktemp -d "${TMPDIR:-/tmp}/umwelt-pull.XXXXXX")
    { echo 'cd umwelt'; printf 'tar -cf - '; printf '%q ' "$@"; echo; } | wsl_sh | tar -xf - -C "$stage"
    moved=0
    kept=0
    while IFS= read -r f; do
      f=${f#./}
      if [ -e "$f" ] && [ "$overwrite" = 0 ] && ! cmp -s "$stage/$f" "$f"; then
        echo "kept the local $f (the PC's version differs: $stage/$f)"
        kept=$((kept + 1))
      else
        mkdir -p "$(dirname "$f")"
        mv "$stage/$f" "$f"
        moved=$((moved + 1))
      fi
    done < <(cd "$stage" && find . -type f)
    if [ "$kept" = 0 ]; then
      rm -rf "$stage"
      echo "pulled $moved files"
    else
      echo "pulled $moved files; kept $kept local files that differ (the PC's versions are in $stage)"
    fi
    ;;
  stop)
    [ $# -eq 1 ] || usage
    valid_name "$1"
    { echo "name=$1"; echo "$IS_JOB"
      cat <<'EOF'
p=umwelt/results/remote/$name.pid
if [ -f "$p" ] && is_job "$(cat "$p")" "$name"; then
  kill -TERM "$(cat "$p")" && echo "stop sent to $name; it ends within a minute (exit code 143)"
else
  echo "$name is not queued or running"
fi
EOF
    } | wsl_sh
    ;;
  pause)
    echo 'p=${UMWELT_PAUSE:-/mnt/c/gpu-pause}; [ -e "$p" ] || echo "made by remote.sh pause $(date)" > "$p"; echo "paused: no umwelt job starts, running ones freeze within 5 s"' | wsl_sh
    ;;
  resume)
    { echo "force=$([ "${1:-}" = --force ] && echo 1 || echo 0)"
      cat <<'EOF'
p=${UMWELT_PAUSE:-/mnt/c/gpu-pause}
if [ ! -e "$p" ]; then echo "not paused"
elif [ "$force" = 1 ] || { [ -f "$p" ] && grep -q "^made by remote.sh pause" "$p"; }; then rm -rf "$p" && echo "resumed"
else echo "refused: someone else made C:\gpu-pause (Henry?); remote.sh resume --force removes it anyway"; exit 1
fi
EOF
    } | wsl_sh
    ;;
  unlock)
    force=0
    if [ "${1:-}" = --force ]; then force=1; fi
    { echo "force=$force"; echo "$IS_JOB"
      cat <<'EOF'
lock=${UMWELT_LOCK:-/mnt/c/gpu-lock}
if [ ! -d "$lock" ]; then echo "no lock"; exit 0; fi
owner=$(head -n 1 "$lock/owner" 2>/dev/null)
pid=$(echo "$owner" | sed -n 's/.* pid=\([0-9]*\) .*/\1/p')
job=$(echo "$owner" | sed -n 's/.* job=\([^ ]*\) .*/\1/p')
if [ "$force" != 1 ]; then
  case "$owner" in
    project=umwelt*) ;;
    *) echo "refused: the lock is not umwelt's ('$owner'); use --force only if that job is surely gone"; exit 1 ;;
  esac
  if is_job "$pid" "$job"; then echo "refused: its job is still running (pid $pid)"; exit 1; fi
fi
rm -rf "$lock" && echo "removed lock: $owner"
EOF
    } | wsl_sh
    ;;
  *)
    usage
    ;;
esac
