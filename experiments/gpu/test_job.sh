#!/usr/bin/env bash
# shellcheck disable=SC2034  # n1 and n2 are read inside the strings check evaluates
# Tests job.sh on the Mac with stand-ins for nvidia-smi, uv and the C:\gpu-lock and C:\gpu-pause paths.
#
#   bash experiments/gpu/test_job.sh
set -u
job=${1:-$(cd "$(dirname "$0")" && pwd)/job.sh}  # or a path to another version, to check the tests can fail
T=$(mktemp -d)
trap 'pkill -f "$T" 2>/dev/null; rm -rf "$T"' EXIT
export HOME=$T/home UMWELT_ROOT=$T/root UMWELT_LOCK=$T/gpu-lock UMWELT_SMI=$T/smi POLL_S=0.2
export UMWELT_PAUSE=$T/gpu-pause PAUSE_POLL_S=0.2
mkdir -p "$HOME/.local/bin" "$UMWELT_ROOT/results/remote"
cat > "$UMWELT_SMI" <<EOF
#!/usr/bin/env bash
cat "$T/mem"
EOF
cat > "$HOME/.local/bin/uv" <<'EOF'
#!/usr/bin/env bash
case "$1" in
  sync) exit 0 ;;
  run) shift
       if [[ "$*" == *default_backend* ]]; then echo "${FAKE_BACKEND:-gpu} NVIDIA GeForce RTX 5080"; else exec "$@"; fi ;;
esac
EOF
chmod +x "$UMWELT_SMI" "$HOME/.local/bin/uv"
R=$UMWELT_ROOT/results/remote
fails=0
check() { if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi; }
new() {  # new job: name, command
  echo "code abc" > "$UMWELT_ROOT/.umwelt_push"
  printf 'code abc\ngit 1234567\n' > "$R/$1.push"
  echo "$2" > "$R/$1.cmd"
}
wait_for() { for _ in $(seq 100); do eval "$1" && return 0; sleep 0.1; done; return 1; }
limit_wait() {  # wait for a background job.sh, at most 20 s; a hang counts as a failure, not a stuck test run
  for _ in $(seq 200); do kill -0 "$1" 2>/dev/null || { wait "$1"; return; }; sleep 0.1; done
  echo "TIMEOUT: still running after 20 s, killed"
  pkill -KILL -P "$1" 2>/dev/null; kill -KILL "$1" 2>/dev/null; wait "$1" 2>/dev/null
}
limit() { "$@" & limit_wait $!; }

echo 3000 > "$T/mem"
new ok 'echo hello'
limit bash "$job" ok
check "a free GPU runs the job" '[ "$(cat $R/ok.exit)" = 0 ] && grep -q hello $R/ok.log'
check "the lock is released after a job" '[ ! -e $UMWELT_LOCK ]'
check "the log records code and backend" 'grep -q "code: code abc git 1234567" $R/ok.log && grep -q "JAX backend: gpu" $R/ok.log'
check "the pid file is removed" '[ ! -e $R/ok.pid ]'

new fail 'exit 7'
limit bash "$job" fail
check "a failing command's exit code is recorded" '[ "$(cat $R/fail.exit)" = 7 ] && [ ! -e $UMWELT_LOCK ]'

mkdir "$UMWELT_LOCK" && echo "other-project job" > "$UMWELT_LOCK/owner"
new waits 'echo ran'
bash "$job" waits & pid=$!
sleep 1
check "waits while another project holds the lock" 'grep -q "waiting:.*other-project" $R/waits.log && ! grep -q "^ran" $R/waits.log'
check "never touches another project's lock while waiting" '[ "$(cat $UMWELT_LOCK/owner)" = "other-project job" ]'
rm -rf "$UMWELT_LOCK"
limit_wait $pid
check "runs once the other lock is gone" '[ "$(cat $R/waits.exit)" = 0 ] && grep -q "^ran" $R/waits.log && [ ! -e $UMWELT_LOCK ]'

echo 9000 > "$T/mem"
new busy 'echo ran'
bash "$job" busy & pid=$!
sleep 1
check "waits while GPU memory is in use without a lock" 'grep -q "waiting: GPU memory in use 9000" $R/busy.log && [ ! -e $R/busy.exit ]'
echo 3000 > "$T/mem"
limit_wait $pid
check "runs once the memory frees" '[ "$(cat $R/busy.exit)" = 0 ]'

new owner 'sleep 1.5'
bash "$job" owner & pid=$!
wait_for '[ -f $UMWELT_LOCK/owner ]'
echo "other-project took it" > "$UMWELT_LOCK/owner"
limit_wait $pid
check "does not remove a lock whose owner changed" '[ "$(cat $UMWELT_LOCK/owner)" = "other-project took it" ] && grep -q "lock not released" $R/owner.log'
rm -rf "$UMWELT_LOCK"

new term "sleep 31 & echo \$! > $T/g1; sleep 32 & echo \$! > $T/g2; wait"
bash "$job" term & pid=$!
wait_for '[ -s $T/g1 ] && [ -s $T/g2 ]'
kill -TERM $pid
limit_wait $pid
check "SIGTERM stops the whole command" '! kill -0 $(cat $T/g1) 2>/dev/null && ! kill -0 $(cat $T/g2) 2>/dev/null'
check "SIGTERM records 143 and releases the lock" '[ "$(cat $R/term.exit)" = 143 ] && [ ! -e $UMWELT_LOCK ]'

new stale 'echo ran'
echo "code xyz" > "$UMWELT_ROOT/.umwelt_push"
limit bash "$job" stale
check "refuses code pushed after the job was queued" '[ "$(cat $R/stale.exit)" = 3 ] && ! grep -q "^ran" $R/stale.log && [ ! -e $UMWELT_LOCK ]'

new cpu 'echo ran'
limit env FAKE_BACKEND=cpu bash "$job" cpu
check "refuses when JAX does not see the GPU" '[ "$(cat $R/cpu.exit)" = 5 ] && ! grep -q "^ran" $R/cpu.log && [ ! -e $UMWELT_LOCK ]'
new cpu2 'echo ran'
touch "$R/cpu2.allow_cpu"
limit env FAKE_BACKEND=cpu bash "$job" cpu2
check "runs a CPU job when allowed" '[ "$(cat $R/cpu2.exit)" = 0 ]'

touch "$UMWELT_PAUSE"
new held 'echo ran'
bash "$job" held & pid=$!
sleep 1
check "a paused queue does not start a job" 'grep -q "waiting: paused" $R/held.log && ! grep -q "^ran" $R/held.log && [ ! -e $UMWELT_LOCK ]'
rm "$UMWELT_PAUSE"
limit_wait $pid
check "the job starts once the pause is lifted" '[ "$(cat $R/held.exit)" = 0 ] && grep -q "^ran" $R/held.log'

new pz "echo \$\$ > $T/pz.pid; for i in \$(seq 40); do echo t >> $T/ticks; sleep 0.1; done"
bash "$job" pz & pid=$!
wait_for '[ -s $T/ticks ]'
mkdir "$UMWELT_PAUSE"  # a folder works as well as a file
wait_for 'grep -q "^.*paused (" $R/pz.log'
sleep 0.5; n1=$(wc -l < "$T/ticks"); sleep 1; n2=$(wc -l < "$T/ticks")
check "pause stops a running command and keeps the lock" '[ "$n1" = "$n2" ] && ps -o stat= -p "$(cat $T/pz.pid)" | grep -q T && [ -d $UMWELT_LOCK ]'
rm -r "$UMWELT_PAUSE"
limit_wait $pid
check "resume continues it to the end" '[ "$(cat $R/pz.exit)" = 0 ] && [ "$(wc -l < $T/ticks)" -eq 40 ] && grep -q "resumed" $R/pz.log && [ ! -e $UMWELT_LOCK ]'

new pt "echo \$\$ > $T/pt.pid; sleep 33"
bash "$job" pt & pid=$!
wait_for '[ -s $T/pt.pid ]'
touch "$UMWELT_PAUSE"
wait_for 'grep -q "paused (" $R/pt.log'
kill -TERM $pid
limit_wait $pid
rm -f "$UMWELT_PAUSE"
check "SIGTERM stops a paused command too" '! kill -0 $(cat $T/pt.pid) 2>/dev/null && [ "$(cat $R/pt.exit)" = 143 ] && [ ! -e $UMWELT_LOCK ]'

echo "N/A" > "$T/mem"
new nosmi 'echo ran'
limit env POLL_S=0 bash "$job" nosmi
check "gives up when nvidia-smi gives no number" '[ "$(cat $R/nosmi.exit)" = 1 ] && grep -q "attempt 10 of 10" $R/nosmi.log && [ ! -e $UMWELT_LOCK ]'

echo "$fails failed"
[ "$fails" = 0 ]
