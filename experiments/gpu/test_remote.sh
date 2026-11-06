#!/usr/bin/env bash
# shellcheck disable=SC2034  # out is read inside the strings check evaluates
# Tests remote.sh on the Mac against a stand-in PC: a fake ssh runs the WSL side of each command in a
# temporary folder, and fakes stand in for nvidia-smi, uv and C:\gpu-lock. What it cannot test: Windows
# OpenSSH, cmd quoting, wsl.exe and Win32_Process.Create themselves.
#
#   bash experiments/gpu/test_remote.sh
set -u
remote=${1:-$(cd "$(dirname "$0")" && pwd)/remote.sh}  # or a path to another version, to check the tests can fail
repo=$(cd "$(dirname "$0")/../.." && pwd)
T=$(mktemp -d)
tmpcode=$repo/experiments/gpu/_test_remote_tmp.txt  # a new file, to change the code hash
tag=trtest$$
cleanup() {
  pkill -f "$T" 2>/dev/null
  rm -rf "$T" "$tmpcode" "$repo/results/remote/$tag"* "$repo/results/data/${tag}_rec"
  rmdir "$repo/results/remote" 2>/dev/null
}
trap cleanup EXIT
export UMWELT_PC=test@fake-pc FAKE_PC=$T/pc UMWELT_LOCK=$T/gpu-lock UMWELT_SMI=$T/smi UMWELT_PAUSE=$T/gpu-pause POLL_S=0.2 PAUSE_POLL_S=0.2 PATH=$T/bin:$PATH
mkdir -p "$T/bin" "$FAKE_PC/.local/bin"
echo 3000 > "$T/mem"
cat > "$T/bin/ssh" <<'EOF'
#!/usr/bin/env bash
# Stand-in for ssh to the PC: drops the options and host, then runs the command here, in $FAKE_PC.
while [[ "$1" == -* ]]; do if [ "$1" = -o ]; then shift; fi; shift; done
shift
c="$*"
cd "$FAKE_PC" || exit 1
case "$c" in
  "wsl -d "*" --cd ~ -- "*)
    HOME=$FAKE_PC bash -c "${c#* -- }"; r=$?
    # FAKE_CORRUPT: alter a file after it arrives, as a faulty pipe would
    if [ -n "${FAKE_CORRUPT:-}" ] && [[ "$c" == *"tar -xf"* ]]; then echo "# altered" >> umwelt/pyproject.toml; fi
    if [ -n "${FAKE_CORRUPT_DATA:-}" ] && [[ "$c" == *"tar -xf"* ]]; then echo "altered" >> "umwelt/$FAKE_CORRUPT_DATA"; fi
    exit $r ;;
  nvidia-smi*) echo "$(cat "$UMWELT_SMI.mem") MiB, 16303 MiB, 5 %" ;;
  powershell*Win32_Process*)
    cl=$(echo "$c" | sed -n "s/.*CommandLine='\([^']*\)'.*/\1/p")
    (HOME=$FAKE_PC nohup bash -c "${cl#* -- }" >/dev/null 2>&1 &)
    echo 0 ;;
  *) echo "fake ssh: unknown command: $c" >&2; exit 1 ;;
esac
EOF
printf '#!/usr/bin/env bash\ncat "%s"\n' "$T/mem" > "$UMWELT_SMI"
ln -s "$T/mem" "$UMWELT_SMI.mem"
cat > "$FAKE_PC/.local/bin/uv" <<'EOF'
#!/usr/bin/env bash
case "$1" in
  sync) exit 0 ;;
  run) shift; if [[ "$*" == *default_backend* ]]; then echo "gpu NVIDIA GeForce RTX 5080"; else exec "$@"; fi ;;
esac
EOF
chmod +x "$T/bin/ssh" "$UMWELT_SMI" "$FAKE_PC/.local/bin/uv"
P=$FAKE_PC/umwelt
fails=0
check() { if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi; }
wait_for() { for _ in $(seq 150); do eval "$1" && return 0; sleep 0.1; done; return 1; }

out=$(bash "$remote" push 2>&1)
check "push copies the code" '[ -f $P/experiments/gpu/job.sh ] && [ -f $P/src/umwelt/data.py ] && [ -f $P/pyproject.toml ]'
check "push leaves results/ behind" '[ ! -d $P/results/hardware ]'
check "push sends no macOS ._ files" '[ -z "$(find $P -name "._*")" ]'
check "push writes a manifest" 'grep -q "^code [0-9a-f]\{16\}$" $P/.umwelt_push && grep -q "^git " $P/.umwelt_push'
out=$(FAKE_CORRUPT=1 bash "$remote" push 2>&1)
check "push catches a file altered on the way" 'echo "$out" | grep -q "push failed verification"'
out=$(bash "$remote" push 2>&1)
check "push verifies a clean copy" 'echo "$out" | grep -q "^pushed" && cmp -s $repo/pyproject.toml $P/pyproject.toml'

out=$(bash "$remote" run "${tag}a" echo "hello world" 2>&1)
check "run starts a job" 'echo "$out" | grep -q "started ${tag}a"'
wait_for '[ -f $P/results/remote/${tag}a.exit ]'
check "the job runs its command with arguments intact" 'grep -q "^hello world$" $P/results/remote/${tag}a.log && [ "$(cat $P/results/remote/${tag}a.exit)" = 0 ]'
check "the job log records the pushed code" 'grep -q "code: code " $P/results/remote/${tag}a.log'
out=$(bash "$remote" status "${tag}a" 2>&1)
check "status shows the GPU, lock and result" 'echo "$out" | grep -q "GPU (used, total, busy): 3000 MiB" && echo "$out" | grep -q "lock: free" && echo "$out" | grep -q "finished, exit code 0"'
out=$(bash "$remote" status 'x;rm' 2>&1)
check "status rejects a bad job name" 'echo "$out" | grep -q "job name:"'

out=$(bash "$remote" run "${tag}a" echo again 2>&1)
check "a job name cannot be reused" 'echo "$out" | grep -q "already exists"'

bash "$remote" run "${tag}b" sleep 3 >/dev/null 2>&1
wait_for '[ -f $P/results/remote/${tag}b.pid ]'
echo "changed $$" > "$tmpcode"
out=$(bash "$remote" push 2>&1)
check "a push with changed code is refused while a job runs" 'echo "$out" | grep -q "refused: the code differs" && [ ! -f $P/experiments/gpu/_test_remote_tmp.txt ]'
rm -f "$tmpcode"
out=$(bash "$remote" push 2>&1)
check "a push with the same code is allowed while a job runs" 'echo "$out" | grep -q "^pushed"'
wait_for '[ -f $P/results/remote/${tag}b.exit ]'
echo "changed $$" > "$tmpcode"
out=$(bash "$remote" push 2>&1)
check "a push with changed code is allowed once jobs end" 'echo "$out" | grep -q "^pushed" && [ -f $P/experiments/gpu/_test_remote_tmp.txt ]'
rm -f "$tmpcode"

out=$(bash "$remote" pull "results/remote/${tag}a.log" 2>&1)
check "pull copies a file back" 'cmp -s "$repo/results/remote/${tag}a.log" "$P/results/remote/${tag}a.log"'
echo "local version" > "$repo/results/remote/${tag}a.log"
out=$(bash "$remote" pull "results/remote/${tag}a.log" 2>&1)
check "pull keeps a differing local file" '[ "$(cat $repo/results/remote/${tag}a.log)" = "local version" ] && echo "$out" | grep -q "kept 1"'
out=$(bash "$remote" pull --overwrite "results/remote/${tag}a.log" 2>&1)
check "pull --overwrite replaces it" 'cmp -s "$repo/results/remote/${tag}a.log" "$P/results/remote/${tag}a.log"'
touch "$P/results/remote/${tag} spaced name.txt"
out=$(bash "$remote" pull "results/remote/${tag} spaced name.txt" 2>&1)
check "pull handles a path with spaces" '[ -f "$repo/results/remote/${tag} spaced name.txt" ]'

mkdir "$UMWELT_LOCK" && echo "other-project job" > "$UMWELT_LOCK/owner"
out=$(bash "$remote" unlock 2>&1)
check "unlock refuses another project's lock" 'echo "$out" | grep -q "refused: the lock is not umwelt" && [ -d $UMWELT_LOCK ]'
bash -c 'sleep 30; :' umwelt/experiments/gpu/job.sh x & live=$!  # looks like job x's job.sh to ps
echo "project=umwelt job=x host=h pid=$live started=now" > "$UMWELT_LOCK/owner"
out=$(bash "$remote" unlock 2>&1)
check "unlock refuses a running umwelt job's lock" 'echo "$out" | grep -q "still running" && [ -d $UMWELT_LOCK ]'
kill $live
sleep 30 & other=$!
echo "project=umwelt job=x host=h pid=$other started=now" > "$UMWELT_LOCK/owner"
out=$(bash "$remote" unlock 2>&1)
check "unlock is not fooled by a reused process id" 'echo "$out" | grep -q "removed lock" && [ ! -d $UMWELT_LOCK ]'
mkdir "$UMWELT_LOCK"
echo "project=umwelt job=x host=h pid=999999 started=now" > "$UMWELT_LOCK/owner"
out=$(bash "$remote" unlock 2>&1)
check "unlock removes a dead umwelt job's lock" 'echo "$out" | grep -q "removed lock" && [ ! -d $UMWELT_LOCK ]'

out=$(bash "$remote" pause 2>&1)
check "pause creates the switch" '[ -e $UMWELT_PAUSE ] && echo "$out" | grep -q "^paused"'
out=$(bash "$remote" status 2>&1)
check "status shows the pause" 'echo "$out" | grep -q "^pause: on"'
bash "$remote" run "${tag}c" echo held >/dev/null 2>&1
sleep 1
check "a job started while paused waits" '[ ! -e $P/results/remote/${tag}c.exit ] && grep -q "waiting: paused" $P/results/remote/${tag}c.log'
out=$(bash "$remote" resume 2>&1)
wait_for '[ -f $P/results/remote/${tag}c.exit ]'
check "resume removes the switch and the job runs" '[ ! -e $UMWELT_PAUSE ] && [ "$(cat $P/results/remote/${tag}c.exit)" = 0 ]'
out=$(bash "$remote" status 2>&1)
check "status shows no pause after resume" '! echo "$out" | grep -q "^pause:"'

bash "$remote" run "${tag}d" sleep 30 >/dev/null 2>&1
wait_for '[ -f $P/results/remote/${tag}d.pid ] && grep -q "start:" $P/results/remote/${tag}d.log'
out=$(bash "$remote" stop "${tag}d" 2>&1)
wait_for '[ -f $P/results/remote/${tag}d.exit ]'
check "stop ends a running job and frees the lock" 'echo "$out" | grep -q "stop sent" && [ "$(cat $P/results/remote/${tag}d.exit)" = 143 ] && [ ! -e $UMWELT_LOCK ]'
out=$(bash "$remote" stop "${tag}d" 2>&1)
check "stop reports a job that is not running" 'echo "$out" | grep -q "not queued or running"'
sleep 30 & other=$!
echo "$other" > "$P/results/remote/${tag}e.pid"  # a stale pid file whose number another program now has
out=$(bash "$remote" stop "${tag}e" 2>&1)
check "stop never signals a process that is not that job" 'echo "$out" | grep -q "not queued or running" && kill -0 $other'
kill $other
mkdir "$UMWELT_PAUSE"  # Henry pausing by hand on the PC
out=$(bash "$remote" resume 2>&1)
check "resume leaves a pause Henry made" 'echo "$out" | grep -q "refused" && [ -d $UMWELT_PAUSE ]'
out=$(bash "$remote" resume --force 2>&1)
check "resume --force removes it" '[ ! -e $UMWELT_PAUSE ]'

out=$(bash "$remote" sh 'echo "prealloc=$XLA_PYTHON_CLIENT_PREALLOCATE"' 2>&1)
check "sh never lets JAX reserve most of the GPU" 'echo "$out" | grep -q "^prealloc=false$"'

bash "$remote" run "${tag}n" bash -c 'echo "norecord=${UMWELT_NO_RECORD:-}"' >/dev/null 2>&1
bash "$remote" run --record "${tag}r" bash -c 'echo "norecord=${UMWELT_NO_RECORD:-}"' >/dev/null 2>&1
wait_for '[ -f $P/results/remote/${tag}n.exit ] && [ -f $P/results/remote/${tag}r.exit ]'
check "a job may not record a world unless run --record" 'grep -q "^norecord=1$" $P/results/remote/${tag}n.log && grep -q "^norecord=$" $P/results/remote/${tag}r.log'

echo "sleep 30" > "$P/results/remote/${tag}x.cmd"
touch -t 202601010000 "$P/results/remote/${tag}x.cmd"
out=$(bash "$remote" status "${tag}x" 2>&1)
check "status reports a job that died without an exit code" 'echo "$out" | grep -q "^dead:"'
touch "$P/results/remote/${tag}x.cmd"
out=$(bash "$remote" status "${tag}x" 2>&1)
check "status does not call a job just started dead" 'echo "$out" | grep -q "^not finished: starting"'

sleep 30 & other=$!
echo "$other" > "$P/results/remote/${tag}g.pid"  # a stale pid file whose number another program now has
echo "changed again $$ $RANDOM" > "$tmpcode"  # content no earlier push sent, so the code really differs
out=$(bash "$remote" push 2>&1)
check "a reused process id does not block a push" 'echo "$out" | grep -q "^pushed"'
kill $other
rm -f "$tmpcode" "$P/results/remote/${tag}g.pid"

mkdir -p "$repo/results/data/${tag}_rec"
echo one > "$repo/results/data/${tag}_rec/a.npy"; echo two > "$repo/results/data/${tag}_rec/b.npy"
out=$(bash "$remote" push "results/data/${tag}_rec" 2>&1)
check "push verifies the data it sends" 'echo "$out" | grep -q "^pushed" && cmp -s "$repo/results/data/${tag}_rec/a.npy" "$P/results/data/${tag}_rec/a.npy"'
out=$(FAKE_CORRUPT_DATA="results/data/${tag}_rec/b.npy" bash "$remote" push "results/data/${tag}_rec" 2>&1)
check "push catches a data file altered on the way" 'echo "$out" | grep -q "push failed verification for results/data/${tag}_rec"'
bash "$remote" push "results/data/${tag}_rec" >/dev/null 2>&1
echo stray > "$P/results/data/${tag}_rec/c.npy"
out=$(bash "$remote" push "results/data/${tag}_rec" 2>&1)
check "push catches an extra file already on the PC in a pushed folder" 'echo "$out" | grep -q "push failed verification"'
rm -f "$P/results/data/${tag}_rec/c.npy"

series=$(dirname "$remote")/series.sh
out=$(GAP_S=0 POLL_S=0.2 bash "$series" "${tag}s1" 'echo one' "${tag}s2" 'echo two' 2>&1)
check "series runs jobs one after another" 'echo "$out" | grep -q "${tag}s1 finished, exit code 0" && echo "$out" | grep -q "${tag}s2 finished, exit code 0"'
GAP_S=0 POLL_S=0.2 bash "$series" "${tag}s3" 'exit 3' "${tag}s4" 'echo four' >/dev/null 2>&1; code=$?
check "series stops at a failed job" '[ "$code" = 1 ] && [ ! -e $P/results/remote/${tag}s4.cmd ]'
out=$(GAP_S=0 POLL_S=0.2 bash "$series" --record "${tag}s5" 'echo "norecord=${UMWELT_NO_RECORD:-}"' 2>&1)
check "series passes --record to its jobs" 'grep -q "^norecord=$" $P/results/remote/${tag}s5.log'

git clone -q "$repo" "$T/clone"
cp "$remote" "$T/clone/experiments/gpu/remote.sh"  # the version under test, which may be uncommitted
rm "$T/clone/README.md"
out=$(bash "$T/clone/experiments/gpu/remote.sh" push 2>&1)
check "push skips a tracked file deleted but not committed" 'echo "$out" | grep -q "^pushed"'
check "push removes on the PC a file deleted here" '[ ! -e $P/README.md ] && echo "$out" | grep -q "removed on the PC.*README.md"'

cd /tmp && out=$(bash "$remote" 2>&1)
check "works when called from another folder" 'echo "$out" | grep -q "remote.sh push"'

echo "$fails failed"
[ "$fails" = 0 ]
