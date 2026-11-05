#!/usr/bin/env bash
# Run umwelt on the GPU PC (RTX 5080, inside WSL2) from the Mac. See docs/gpu.md.
#
#   remote.sh push [PATH...]                 copy the working tree (not results/), plus PATHs, to ~/umwelt
#   remote.sh setup                          one-time install (runs setup_wsl.sh)
#   remote.sh sh 'CMD'                       run a command in ~/umwelt now and wait for it
#   remote.sh run [--allow-cpu] NAME CMD...  start CMD as a background job; it waits for a free GPU
#   remote.sh status [NAME]                  GPU use, lock holder, last lines of NAME's log
#   remote.sh pull [--overwrite] PATH...     copy files under ~/umwelt back here (a local file that
#                                            differs is kept, unless --overwrite)
#   remote.sh unlock [--force]               remove C:\gpu-lock left by a killed umwelt job
#
# A job survives the SSH connection closing: it is started through Windows' Win32_Process.Create,
# outside the SSH session. Each push writes .umwelt_push (code hash, git commit, uncommitted changes);
# a job keeps a copy, records it in its log and refuses to start on different code. A push that would
# change the code while an umwelt job is queued or running is refused. Job names are used once.
set -euo pipefail
self=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
PC=${UMWELT_PC:-<pc-user>@<pc-ip>}  # known_hosts has the PC under its Tailscale IP
DISTRO=${UMWELT_DISTRO:-Ubuntu-24.04}
cd "$(dirname "$self")/../.."
export COPYFILE_DISABLE=1  # with --no-xattrs --no-mac-metadata: no macOS ._ files or extended attributes

ssh_pc() { ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 "$PC" "$@"; }
wsl_sh() { ssh_pc "wsl -d $DISTRO --cd ~ -- bash -s"; }  # runs the bash script on stdin inside WSL
usage() { sed -n '2,16p' "$self"; exit 2; }
valid_name() { [[ "$1" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "job name: letters, digits, . _ - only" >&2; exit 2; }; }
code_files() {  # tracked and new files outside results/, skipping deleted ones
  git ls-files -co --exclude-standard | grep -v '^results/' | while IFS= read -r f; do
    if [ -f "$f" ]; then printf '%s\n' "$f"; fi
  done
}
manifest() {
  echo "code $(code_files | tr '\n' '\0' | xargs -0 shasum -a 256 | shasum -a 256 | cut -c1-16)"
  echo "git $(git rev-parse --short HEAD)"
  echo "uncommitted $(git status --porcelain -- . ':!results' | wc -l | tr -d ' ') files, diff $(git diff HEAD -- . ':!results' | shasum -a 256 | cut -c1-16)"
  echo "pushed $(date '+%Y-%m-%dT%H:%M:%S%z')"
}

cmd=${1:-}
[ -n "$cmd" ] && shift
case "$cmd" in
  push)
    m=$(manifest)
    state=$({ cat <<'EOF'
cd umwelt 2>/dev/null || exit 0
grep '^code ' .umwelt_push 2>/dev/null || echo "code none"
for p in results/remote/*.pid; do
  if [ -f "$p" ] && kill -0 "$(cat "$p")" 2>/dev/null; then echo "active $(basename "$p" .pid)"; fi
done
EOF
    } | wsl_sh | tr -d '\r')
    active=$(echo "$state" | sed -n 's/^active //p' | tr '\n' ' ')
    if [ -n "$active" ] && [ "$(echo "$state" | grep '^code ')" != "$(echo "$m" | grep '^code ')" ]; then
      echo "refused: the code differs from what queued or running jobs were given ($active); wait for them" >&2
      exit 1
    fi
    printf '%s\n' "$m" > .umwelt_push
    ssh_pc "wsl -d $DISTRO --cd ~ -- mkdir -p umwelt"
    { code_files; echo .umwelt_push; if [ $# -gt 0 ]; then printf '%s\n' "$@"; fi; } \
      | tar --no-xattrs --no-mac-metadata -cf - -T - | ssh_pc "wsl -d $DISTRO --cd ~ -- tar -xf - -C umwelt"
    echo "pushed the working tree${*:+ and $*} to $PC:~/umwelt ($DISTRO): $(echo "$m" | head -3 | tr '\n' ' ')"
    ;;
  setup)
    "$self" push
    echo 'cd umwelt && bash experiments/gpu/setup_wsl.sh' | wsl_sh
    ;;
  sh)
    [ $# -gt 0 ] || usage
    printf 'cd umwelt && export PATH="$HOME/.local/bin:$PATH" && %s\n' "$*" | wsl_sh
    ;;
  run)
    allow=0
    if [ "${1:-}" = --allow-cpu ]; then allow=1; shift; fi
    [ $# -ge 2 ] || usage
    name=$1; shift
    valid_name "$name"
    "$self" push
    made=$({ echo "name=$name allow=$allow"
             cat <<'EOF'
cd umwelt && mkdir -p results/remote
if [ -e "results/remote/$name.cmd" ]; then echo exists; exit 0; fi
cp .umwelt_push "results/remote/$name.push"
if [ "$allow" = 1 ]; then touch "results/remote/$name.allow_cpu"; fi
cat > "results/remote/$name.cmd" <<'UMWELT_CMD_END'
EOF
             printf '%q ' "$@"; echo
             echo 'UMWELT_CMD_END'
             echo 'echo created'; } | wsl_sh | tr -d '\r')
    [ "$made" = created ] || { echo "a job named $name already exists on the PC; pick a new name" >&2; exit 1; }
    ssh_pc "powershell -NoProfile -Command \"(Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='wsl.exe -d $DISTRO --cd ~ -- bash umwelt/experiments/gpu/job.sh $name'}).ReturnValue\"" \
      | tr -d '\r' | grep -qx 0 || { echo "could not start the job" >&2; exit 1; }
    echo "started $name; follow it with: experiments/gpu/remote.sh status $name"
    ;;
  status)
    if [ -n "${1:-}" ]; then valid_name "$1"; fi
    ssh_pc "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader" | tr -d '\r' \
      | sed 's/^/GPU (used, total, busy): /'
    { echo "name=${1:-}"
      cat <<'EOF'
lock=${UMWELT_LOCK:-/mnt/c/gpu-lock}
if [ -f "$lock/owner" ]; then sed -n '1s/^/lock: /p' "$lock/owner"; else echo "lock: free"; fi
if [ -n "$name" ]; then
  f=umwelt/results/remote/$name
  tail -n 15 "$f.log" 2>/dev/null
  if [ -f "$f.exit" ]; then echo "finished, exit code $(cat "$f.exit")"; else echo "not finished"; fi
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
  unlock)
    force=0
    if [ "${1:-}" = --force ]; then force=1; fi
    { echo "force=$force"
      cat <<'EOF'
lock=${UMWELT_LOCK:-/mnt/c/gpu-lock}
if [ ! -d "$lock" ]; then echo "no lock"; exit 0; fi
owner=$(head -n 1 "$lock/owner" 2>/dev/null)
pid=$(echo "$owner" | sed -n 's/.* pid=\([0-9]*\) .*/\1/p')
if [ "$force" != 1 ]; then
  case "$owner" in
    project=umwelt*) ;;
    *) echo "refused: the lock is not umwelt's ('$owner'); use --force only if that job is surely gone"; exit 1 ;;
  esac
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "refused: its job is still running (pid $pid)"; exit 1; fi
fi
rm -rf "$lock" && echo "removed lock: $owner"
EOF
    } | wsl_sh
    ;;
  *)
    usage
    ;;
esac
