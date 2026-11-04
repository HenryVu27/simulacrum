#!/usr/bin/env bash
# Run umwelt on the GPU PC (RTX 5080, inside WSL2) from the Mac. See docs/gpu.md.
#
#   experiments/gpu/remote.sh push [PATH...]        copy the working tree (not results/), plus PATHs, to ~/umwelt
#   experiments/gpu/remote.sh setup                 one-time install (runs setup_wsl.sh)
#   experiments/gpu/remote.sh sh 'uv run pytest -q' run a command now and wait for it
#   experiments/gpu/remote.sh run NAME CMD...       start CMD in the background; it waits for the GPU
#   experiments/gpu/remote.sh status [NAME]         GPU use, lock holder, last lines of NAME's log
#   experiments/gpu/remote.sh pull PATH...          copy files or folders under ~/umwelt back here
#   experiments/gpu/remote.sh unlock                remove a stale C:\gpu-lock (a job killed outright)
#
# A background job survives the SSH connection closing: it is started through Windows'
# Win32_Process.Create, outside the SSH session. Its log is results/remote/NAME.log on the PC, and
# results/remote/NAME.exit holds its exit code once it ends.
set -euo pipefail
PC=${UMWELT_PC:-<pc-user>@<pc-ip>}  # known_hosts has the PC under its Tailscale IP
DISTRO=${UMWELT_DISTRO:-Ubuntu-24.04}
cd "$(dirname "$0")/../.."

ssh_pc() { ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 "$PC" "$@"; }
# Run a bash script (read from stdin) inside WSL, in ~/umwelt.
wsl_sh() { ssh_pc "wsl -d $DISTRO --cd ~ -- bash -s"; }

cmd=${1:-}
[ -n "$cmd" ] && shift
case "$cmd" in
  push)
    ssh_pc "wsl -d $DISTRO --cd ~ -- mkdir -p umwelt"
    { git ls-files -co --exclude-standard | grep -v '^results/'; if [ $# -gt 0 ]; then printf '%s\n' "$@"; fi; } | tar -cf - -T - \
      | ssh_pc "wsl -d $DISTRO --cd ~ -- tar -xf - -C umwelt"
    echo "pushed the working tree${*:+ and $*} to $PC:~/umwelt ($DISTRO)"
    ;;
  setup)
    "$0" push
    echo 'cd umwelt && bash experiments/gpu/setup_wsl.sh' | wsl_sh
    ;;
  sh)
    printf 'cd umwelt && export PATH="$HOME/.local/bin:$PATH" && %s\n' "$*" | wsl_sh
    ;;
  run)
    name=$1; shift
    [[ "$name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "job name: letters, digits, . _ - only" >&2; exit 2; }
    [ $# -gt 0 ] || { echo "usage: remote.sh run NAME CMD..." >&2; exit 2; }
    "$0" push
    { echo 'mkdir -p umwelt/results/remote'
      echo "rm -f umwelt/results/remote/$name.exit"
      echo "cat > umwelt/results/remote/$name.cmd <<'UMWELT_CMD_END'"
      printf '%q ' "$@"; echo
      echo 'UMWELT_CMD_END'; } | wsl_sh
    ssh_pc "powershell -NoProfile -Command \"(Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='wsl.exe -d $DISTRO --cd ~ -- bash umwelt/experiments/gpu/job.sh $name'}).ReturnValue\"" \
      | tr -d '\r' | grep -qx 0 || { echo "could not start the job" >&2; exit 1; }
    echo "started $name; follow it with: experiments/gpu/remote.sh status $name"
    ;;
  status)
    ssh_pc "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader" | tr -d '\r' \
      | sed 's/^/GPU (used, total, busy): /'
    { echo 'if [ -f /mnt/c/gpu-lock/owner ]; then sed "s/^/lock: /" /mnt/c/gpu-lock/owner; else echo "lock: free"; fi'
      if [ -n "${1:-}" ]; then
        echo "f=umwelt/results/remote/$1"
        echo 'tail -n 15 $f.log 2>/dev/null; [ -f $f.exit ] && echo "finished, exit code $(cat $f.exit)" || echo "not finished"'
      fi; } | wsl_sh
    ;;
  pull)
    [ $# -gt 0 ] || { echo "usage: remote.sh pull PATH..." >&2; exit 2; }
    ssh_pc "wsl -d $DISTRO --cd ~ -- tar -cf - -C umwelt $*" | tar -xf -
    echo "pulled $*"
    ;;
  unlock)
    echo 'if [ -f /mnt/c/gpu-lock/owner ]; then sed "s/^/removing lock: /" /mnt/c/gpu-lock/owner; fi; rm -rf /mnt/c/gpu-lock' | wsl_sh
    ;;
  *)
    sed -n '2,14p' "$0"
    exit 2
    ;;
esac
