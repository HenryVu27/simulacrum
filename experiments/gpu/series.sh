#!/usr/bin/env bash
# Runs GPU jobs one after another from the Mac. The next job is queued only after the last has finished,
# and after a gap (GAP_S, default 90 s, longer than the 60 s the other project's jobs wait between
# looks at the lock), so a job of theirs already waiting gets the GPU first: umwelt never takes the lock
# twice in a row while someone else is waiting. Each job is an ordinary `remote.sh run`. Stops at the first
# job that does not exit 0. docs/gpu.md, "How to run on the GPU".
#
#   experiments/gpu/series.sh [--record] [--allow-cpu] NAME 'CMD' [NAME 'CMD' ...]   (options apply to every job)
set -euo pipefail
main() {  # all in a function, so bash has read the whole file before it runs: editing it mid-run is then safe
  here=$(cd "$(dirname "$0")" && pwd)
  opts=()
  while [ "${1:-}" = --record ] || [ "${1:-}" = --allow-cpu ]; do opts+=("$1"); shift; done
  [ $# -ge 2 ] && [ $(($# % 2)) = 0 ] || { echo "usage: series.sh [--record] [--allow-cpu] NAME 'CMD' [NAME 'CMD' ...]" >&2; exit 2; }
  while [ $# -ge 2 ]; do
    name=$1
    cmd=$2
    shift 2
    "$here/remote.sh" run ${opts[@]+"${opts[@]}"} "$name" bash -c "$cmd"
    until state=$("$here/remote.sh" status "$name" 2>/dev/null | tr -d '\r' | grep -E '^(finished|dead)'); do sleep "${POLL_S:-60}"; done
    if [[ "$state" == dead* ]]; then echo "[$(date '+%Y-%m-%d %H:%M:%S')] $name $state" >&2; exit 1; fi
    code=${state#finished, exit code }
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $name finished, exit code $code"
    if [ "$code" != 0 ]; then exit 1; fi
    if [ $# -ge 2 ]; then sleep "${GAP_S:-90}"; fi
  done
}
main "$@"
