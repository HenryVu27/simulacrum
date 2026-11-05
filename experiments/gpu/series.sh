#!/usr/bin/env bash
# Runs GPU jobs one after another from the Mac. The next job is queued only after the last has finished,
# and after a gap (GAP_S, default 90 s, longer than the 60 s the other project's jobs wait between
# looks at the lock), so a job of theirs already waiting gets the GPU first: umwelt never takes the lock
# twice in a row while someone else is waiting. Each job is an ordinary `remote.sh run`. Stops at the first
# job that does not exit 0. docs/gpu.md, "How to run on the GPU".
#
#   experiments/gpu/series.sh NAME 'CMD' [NAME 'CMD' ...]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 2 ] && [ $(($# % 2)) = 0 ] || { echo "usage: series.sh NAME 'CMD' [NAME 'CMD' ...]" >&2; exit 2; }
while [ $# -ge 2 ]; do
  name=$1
  cmd=$2
  shift 2
  "$here/remote.sh" run "$name" bash -c "$cmd"
  until "$here/remote.sh" status "$name" 2>/dev/null | grep -q '^finished'; do sleep "${POLL_S:-60}"; done
  code=$("$here/remote.sh" status "$name" 2>/dev/null | tr -d '\r' | sed -n 's/^finished, exit code //p')
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $name finished, exit code $code"
  if [ "$code" != 0 ]; then exit 1; fi
  if [ $# -ge 2 ]; then sleep "${GAP_S:-90}"; fi
done
