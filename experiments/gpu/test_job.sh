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
echo "print('code')" > "$UMWELT_ROOT/code.py"  # the pushed code, hashed the way remote.sh push does
echo code.py > "$UMWELT_ROOT/.umwelt_files"
codehash() { (cd "$UMWELT_ROOT" && tr '\n' '\0' < .umwelt_files | xargs -0 shasum -a 256 | shasum -a 256 | cut -c1-16); }
new() {  # new job: name, command
  h=$(codehash)
  echo "code $h" > "$UMWELT_ROOT/.umwelt_push"
  printf 'code %s\ngit 1234567\n' "$h" > "$R/$1.push"
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
check "the log records code and backend" 'grep -q "code: code $(codehash) git 1234567" $R/ok.log && grep -q "JAX backend: gpu" $R/ok.log'
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

new edited 'echo ran'
echo "print('changed on the PC')" >> "$UMWELT_ROOT/code.py"
limit bash "$job" edited
check "refuses code edited on the PC after the job was queued" '[ "$(cat $R/edited.exit)" = 3 ] && ! grep -q "^ran" $R/edited.log'
echo "print('code')" > "$UMWELT_ROOT/code.py"

new empty 'sleep 1.5'
bash "$job" empty & pid=$!
wait_for '[ -f $UMWELT_LOCK/owner ]'
: > "$UMWELT_LOCK/owner"  # as if another project had just created the lock and not yet written its owner
limit_wait $pid
check "never removes a lock whose owner line is empty" '[ -d $UMWELT_LOCK ] && grep -q "lock not released" $R/empty.log'
rm -rf "$UMWELT_LOCK"

new stray "sleep 34 & echo \$! > $T/stray; echo started"
limit bash "$job" stray
check "stops what the command left running before releasing the lock" '[ "$(cat $R/stray.exit)" = 0 ] && ! kill -0 $(cat $T/stray) 2>/dev/null && [ ! -e $UMWELT_LOCK ]'

new plat 'echo "platforms=$JAX_PLATFORMS cache=$JAX_COMPILATION_CACHE_DIR"; echo "norecord=${UMWELT_NO_RECORD:-} fraction=$XLA_PYTHON_CLIENT_MEM_FRACTION code=$UMWELT_CODE"'
limit bash "$job" plat
check "a GPU job makes a missing GPU an error, not a CPU fallback" 'grep -q "^platforms=cuda,cpu " $R/plat.log'
check "a job uses the persistent compile cache" 'grep -q "cache=$UMWELT_ROOT/results/jax_cache$" $R/plat.log'
check "a job may not record a world, takes at most half the GPU, and knows its code" 'grep -q "^norecord=1 fraction=0.5 code=code $(codehash) git 1234567" $R/plat.log'
new rec 'echo "norecord=${UMWELT_NO_RECORD:-}"'
touch "$R/rec.allow_record"
limit bash "$job" rec
check "a job started with --record may record" 'grep -q "^norecord=$" $R/rec.log'

mkdir "$UMWELT_LOCK" && echo "project=umwelt job=gone host=h pid=999999 started=x" > "$UMWELT_LOCK/owner"
new heir 'echo ran'
limit bash "$job" heir
check "removes a lock whose umwelt job is gone, then runs" '[ "$(cat $R/heir.exit)" = 0 ] && grep -q "removed a lock whose umwelt job is gone" $R/heir.log'
mkdir "$UMWELT_LOCK"
bash -c 'sleep 30; :' umwelt/experiments/gpu/job.sh alive & live=$!  # looks like job alive's job.sh to ps
echo "project=umwelt job=alive host=h pid=$live started=x" > "$UMWELT_LOCK/owner"
new heir2 'echo ran'
bash "$job" heir2 & pid=$!
sleep 1
check "never removes a running umwelt job's lock" 'grep -q "waiting:.*job=alive" $R/heir2.log && ! grep -q "^ran" $R/heir2.log && [ -d $UMWELT_LOCK ]'
kill $live
limit_wait $pid
check "takes over once that job is gone" '[ "$(cat $R/heir2.exit)" = 0 ] && grep -q "removed a lock whose umwelt job is gone" $R/heir2.log'

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
