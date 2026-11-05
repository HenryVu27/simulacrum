# GPU machine: setup log and hardware checks

A running log of putting umwelt on Henry's Windows PC (RTX 5080) and of the checks that decide whether
results from it can stand next to results from the Mac. Newest entries at the bottom of each section.

## Machines

| | Mac (reference) | PC (<pc-name>) |
|---|---|---|
| CPU | Apple M2 Pro, 10 cores | Intel Core Ultra 9 285K, 24 cores |
| Memory | 16 GB unified | 32 GB |
| GPU | none used (JAX on CPU) | NVIDIA GeForce RTX 5080, 16 GB, driver 616.56 |
| OS | macOS (Darwin 27) | Windows 11 (build 26200) |
| Reached by | local | Tailscale, `ssh <pc-user>@<pc-ip>` (OpenSSH, key from the agent, admin account) |

The other project (another Claude session) also uses this GPU, from `the other project's folder` with its own
PyTorch environment. umwelt lives in `C:\umwelt` and never touches `the other project's folder`. Use the host's IP:
`known_hosts` holds the key under the IP, so `ssh <pc-name>` fails host-key verification.

## Why WSL2

JAX publishes CUDA builds for Linux only. On 2026-11-04 PyPI had Windows wheels for `jaxlib 0.12.0.1`
(CPU) but none for `jax-cuda12-plugin` or `jax-cuda13-plugin`. So on this PC JAX can use the GPU only
inside WSL2 (Windows Subsystem for Linux), where the Windows NVIDIA driver is passed through.

On 2026-11-04 WSL was not installed: the Windows features `VirtualMachinePlatform` and
`Microsoft-Windows-Subsystem-Linux` were both disabled. Turning them on changes Windows system
settings and needs a reboot, which would also stop any other-project job, so Henry does this step:

    wsl --install -d Ubuntu-24.04      (admin PowerShell on the PC, then reboot, then open Ubuntu once)

(Ubuntu 26.04 was installed in the end; see the setup log.)

## Setup log

- 2026-11-04. Found the PC on Tailscale (`<pc-name>`, <pc-ip>). GPU idle: 3 GB of 16 GB used by
  the desktop, no Python processes.
- 2026-11-04. Native Windows install, CPU only: copied the committed code (`git archive HEAD`, no
  `results/`) to `C:\umwelt` and ran `uv sync --frozen`. Every locked package installed, including
  `python-sat` and `mujoco`. Test suite: 9 of 9 pass (31 s; the Mac takes 15 s).
  Note for cmd: write `set "PYTHONUTF8=1"&& ...`; `set PYTHONUTF8=1 && ...` keeps the trailing space
  and Python refuses to start.
- 2026-11-04. Bug found on Windows: `data.save` renamed the finished `.partial` recording folder while
  one memory map inside it was still open (the loop variable `f` kept the last one alive). macOS allows
  that; Windows refuses (`PermissionError: [WinError 5]`). Fixed by popping and flushing each map so
  the last reference goes before the rename. The bytes written are unchanged. A 256-episode recording
  then saved and loaded on Windows.
- 2026-11-04 17:15. The other project's session started a 13B generation job on the GPU (14.8 GB of 16 GB in
  use, GPU 23 to 31 percent busy), 44 seconds before umwelt's E4 run started on the PC's CPU. To check
  that the CPU job was not starving the GPU job's feeding thread, umwelt's processes were set to idle
  priority: GPU utilization stayed at 23 to 31 percent, so the CPU job was not the bottleneck. The
  E4 run stayed at idle priority anyway, as a courtesy.
- 2026-11-04. Independent check (outside-judge agent, given the artifacts and claims, not the
  reasoning) of commit c4e8320. It recomputed every number from the raw arrays, on both machines for E4,
  and reran stage 3 on the Mac a third time (byte-identical again). All headline numbers held. It found
  two wrong statements, corrected below (the cause of the E2 differences; "everything else 9 s" in E6),
  and script bugs, fixed: the lock could be removed by its non-owner (`unlock`, and `job.sh`'s exit
  trap); a stop signal released the lock while the command kept running; a queued job could run code
  pushed after it was queued; a plain `uv sync` would remove the CUDA plugin and JAX would fall back to
  the CPU silently; `push` sent macOS `._` files and failed on a deleted tracked file; `pull` could
  overwrite Mac files; `nvidia-smi` without a number made a job wait forever; the full-float32 E4 pass
  reused the default-precision recording. The old `job.sh` fails 7 of the 17 new tests; the new one
  passes all 17, and `remote.sh` passes 22 of 22.
- 2026-11-04 19:22 to 21:13. A second other-project generation job held the GPU (15.7 GB, 97 percent busy).
  The restart waited until it released the lock.
- 2026-11-04 20:02. Henry ran `wsl --install`. It turned on Virtual Machine Platform and then waited for
  a restart. Before it, `wsl --status` said virtualization was not enabled, and Windows reported
  `VirtualizationFirmwareEnabled: False`. That reading is misleading on this PC: Windows' own hypervisor
  was already running (virtualization-based security, status 2 = running), which needs firmware
  virtualization, and it hides the CPU's virtualization flags from Windows. No firmware change was needed.
- 2026-11-04 22:35. The PC restarted. WSL 3.0.3.0, kernel 6.18.54.1. The restart did not install a
  distribution on its own, so it was installed over SSH with `wsl --install -d Ubuntu-26.04 --no-launch`.
  Ubuntu 26.04 LTS is the current long-term release (it was offered alongside 24.04); JAX's CUDA wheels
  do not depend on it, since the driver comes from Windows and uv brings its own Python. `remote.sh`
  now defaults to `Ubuntu-26.04`.
- 2026-11-04. `C:\Users\Vuduc\.wslconfig` written with `memory=12GB` (the default is half of the 32 GB):
  Henry uses this PC daily, so a job must leave room for him. Inside WSL `free` then shows 11 GB.
- 2026-11-04. Pause switch, for the same reason: while `C:\gpu-pause` exists, no umwelt job starts and a
  running job is frozen (SIGSTOP to its process group) until the file is deleted (SIGCONT). A frozen
  job keeps the lock and the GPU memory it holds but uses no GPU time. Tests: `job.sh` 22 of 22 (the
  previous version fails 3 of the 5 new pause cases), `remote.sh` 27 of 27. The other project's
  jobs do not look at this switch.
- 2026-11-04 22:45. No Linux account was created. umwelt runs as the distribution's root user, in
  `/root/umwelt`; every WSL command names the user (`wsl -d Ubuntu-26.04 -u root`), so it keeps working
  if Henry later creates his own with Ubuntu's first-launch prompt. A non-interactive command does not
  start that prompt. Root inside WSL has no extra rights on Windows. Ubuntu 26.04.1, 24 cores, the GPU
  visible through `/usr/lib/wsl/lib/nvidia-smi`.
- 2026-11-04 22:47. `remote.sh setup`, 1 minute 51 seconds: uv 0.13.7, CPython 3.13.18, the locked
  packages plus `jax-cuda13-plugin 0.12.0.1`. JAX reports backend `gpu` (`CudaDevice(id=0)`), a 4096 by
  4096 matrix multiply runs, and the test suite passes 9 of 9 (42 s). Log: `results/hardware/setup_wsl.log`.
- 2026-11-04. Finding: inside WSL one GPU allocation of 4 GiB or more fails (`CUDA_ERROR_OUT_OF_MEMORY`)
  with 14.7 GB free. With the plain allocator (`XLA_PYTHON_CLIENT_ALLOCATOR=platform`) 3.99 GiB works and
  4.00 GiB fails. (This entry first said the same of JAX's default allocator; its saved probe on
  2026-11-05 shows the default fails already at 3.99 GiB and works at 3.5.) Total memory is not capped there: ten 1 GiB arrays held
  together worked (10.8 GB), stopping only at JAX's own ceiling of 75 percent of the card (11.9 GiB).
  This is why JAX's default up-front reservation (11.9 GiB) failed and retried smaller; jobs do not
  reserve up front anyway. `experiments/gpu/memprobe.py`. (Corrected below: the limit binds on a
  compiled step's whole working memory, not only on single arrays.)
- 2026-11-04. Every push is now checked on the PC: it rehashes the files that arrived and must match
  the Mac's code hash (a test alters a file in transit and the push fails; `remote.sh` 29 of 29). The
  first real push matched. The Mac's reference episodes (`data_dark.npz`) also arrived byte-identical.
- 2026-11-04 22:52. First real job (`pause-a`): started through `Win32_Process.Create`, took the lock,
  passed the code and GPU checks, ran 3,000 training updates of the current learner in 37 s with
  deterministic GPU operations, released the lock. A second run (`pause-b`) was paused 40 s partway:
  its processes showed state T (stopped) for the whole pause, it took 40 s longer, and its trained
  weights were byte-identical to the first run's (hash `69122ab38190b194`). `experiments/gpu/pause_check.py`.
- 2026-11-04 22:55. The Linux VM had stopped by itself within 20 s of that job ending (`wsl -l -v`:
  Stopped, no VM process), so between jobs WSL holds no memory.
- 2026-11-04. Other GPU load: the PC's GPU showed 18 to 31 percent busy with no umwelt job. Per-process
  counters put it on Henry's desktop: an animated wallpaper (`wallpaper64`, 12 percent), the window
  compositor (`dwm`, 9 percent) and Chrome. Speed numbers below are measured with that load present;
  `checks.sh` records it just before E6.
- 2026-11-04 23:03. First checks job (`gpu-checks`, log `results/hardware/jobs/gpu-checks-1.log`): E1 to
  E3 and E5 finished; E6 stopped at 128 minds at once, `RESOURCE_EXHAUSTED ... allocate 8.12GiB`. XLA
  packs all of a compiled step's temporary buffers into one allocation, so the 4 GiB limit applies to a
  step's whole working memory: 64 minds at once (just over 4 GiB) fails as well. Correction to the
  memory entry above: the binding limit is per compiled step, not per array.
- 2026-11-04 23:05. JAX's `cuda_async` allocator (`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`) gets blocks
  of 4, 6 and 8.2 GiB (seen over SSH; the output was not saved until `speed_check.sh` recorded it). Compared with the default in one job (`experiments/gpu/alloc_check.sh`): training
  with deterministic operations gives byte-identical weights (hash `69122ab38190b194`); on-device
  updates for 1 to 32 minds take about the same time (within 4 percent, three repeats each, except the
  current learner alone: 2.57 ms under `cuda_async` against 2.87 ms, 11 percent faster); 64 minds run
  (37 ms per update) where the default fails; but the host-batch training loop that experiments use is
  25 percent slower (19.2 against 15.3 ms per update; the pause runs agree, 41 s against 37 s). Decision:
  jobs keep the default allocator; a job that needs a step of 4 GiB or more sets `cuda_async` itself:
  `remote.sh run NAME env XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async CMD...`. (Provisional: those
  training-loop numbers include compiling, see 2026-11-05 below; `speed_check.sh` times the loop again.)
- 2026-11-04. 128 minds at once under `cuda_async` run but take 457 ms per update (three repeats, 423 to
  466 ms), 12 times the 64-mind step for twice the work. Not explained; one guess, untested, is that
  Windows (WDDM) starts paging GPU memory when one process holds that much. In practice: at most 64
  minds per compiled step. (2026-11-05: under `cuda_async` one process held 16 GiB on this 15.9 GiB card,
  so spilling past the card is possible; whether the 128-mind step spilled was not measured.)
- 2026-11-04. Finding: on the GPU the matmul precision changed the recorded world, not only learning.
  The ray caster tests which object each ray hits with `u @ f.T` (`src/umwelt/senses.py:52`); at TF32
  its inputs keep about three significant digits, so hits at an object's edge flip. Against the Mac,
  object visibility differed at 130 of 98,304 entries (47 of 64 episodes) at the default precision, 4
  at full float32, 2 on the PC's CPU; vision values differed in 3.0, 0.6 and 0.3 percent of entries.
  Physics, actions, hearing, touch, smell, body sense, compass and oracle were byte-identical between
  the two GPU precisions; vision, ray texture and the visibility labels were not, so every mind reading
  the ray caster (eyes, or echolocation's depth and texture) saw different observations: bat 2.8,
  chimera 4.2, vision_only 2.0, human 1.6 percent of entries (counted by the second independent check;
  this entry first said "every other sense", which was wrong). First fix:
  `data.chunks` rolls the world out under `jax.default_matmul_precision("highest")`, so a recording
  no longer depends on the device's default (superseded 2026-11-05: it missed the surprise-test movies). On the Mac the output is byte-identical with the fix (world
  19 of 19 arrays, senses 14 of 14, training identical; tests 9 of 9). The check on the GPU is
  `checks.sh precfix` (a default-precision recording must equal the full-precision one).
- 2026-11-04 23:19. The other project started a 13B training run (12.8 GB, 99 percent
  busy) and took the lock between two umwelt jobs; the next umwelt job waited, as designed. It was
  stopped with the new `remote.sh stop` (the other project's lock untouched) so the precision fix could go
  in first, and `gpu-checks-3` (`checks.sh precfix bench stage3`) was queued behind the other project run.
- 2026-11-04 23:30. Overnight coordination with the other project's session (Henry asked both to share the GPU
  while he sleeps). Agreed: one job per lock acquisition, released between jobs; any job planned ahead,
  or over an hour, gets a line in `C:\gpu-schedule.txt` (project, job, start, end, GPU memory) before it
  takes the lock; neither side touches the other's processes or lock; the other project splits long
  generation into chunks of at most two hours. umwelt's line: `gpu-checks-4`, about 23:55 to 00:50, at
  most 5 GB. The PC CPU benchmark (`pc-wsl-cpu`) left the default checks: it uses no GPU but would hold
  the lock for 15 to 20 minutes (`checks.sh cpu` runs it). `gpu-checks-3` was stopped for that change
  and requeued as `gpu-checks-4`.
- 2026-11-05 00:05. `gpu-checks-4` took the lock at 23:40, when the other project's training run ended early,
  15 minutes before its announced slot (23:55), and finished at 00:05:03 (log
  `results/hardware/jobs/gpu-checks-4.log`). The GPU's other load at the start was 1 to 2 percent (the
  desktop was idle). The other project's session was told the GPU was free.
- 2026-11-05. The precision fix holds on the GPU: a default-precision recording (`pc-gpu-precfix`) now
  equals the full-float32 one array for array (19 of 19, 14 of 14 presets), and against the Mac only 4
  visibility entries differ, as at full float32. At full scale too: the two stage 3 runs (default and
  full float32) recorded byte-identical 8,448-episode worlds (3.4 GB, hash `f8379b8ed49a1423`).
- 2026-11-05. Bug in the comparison tooling: `compare_stage3.py` wrote `compare_stage3.json` into the
  compared run's folder, so two comparisons against one run overwrote each other (it happened twice).
  It now names the file after the reference (`compare_stage3_vs_<machine>_<folder>.json`); every
  comparison was redone.
- 2026-11-05. Finding about the benchmark: `learner.train` defines its compiled update inside the function,
  so every call compiles again, and E6's "training loop" number (one 500-update call) included that
  compile. The benchmark now times two run lengths (200 and 1,200 updates, three times each): the slope
  is the cost of an update, the intercept the fixed cost of a call. On the Mac the fixed cost is 1.0 s.
  The Mac rerun (39.3 ms per update) is not clean: another project's text-generation job (MLX) was running on
  the Mac at the time (load average 3.5 to 4). The idle measurement by the old method was 33.7 ms
  including about 1 s of compiling, so about 31.7 ms per update. To repeat when the Mac is idle.
- 2026-11-05 00:15. The other project took the GPU at 00:07 for 13B generation (two chunks, then a training
  run, to about 03:30, per `C:\gpu-schedule.txt`). `speed_check.sh` (the training loop by the new method,
  the cost of deterministic operations, and a compile-cache test) was queued as `speed-check-2` for the
  first gap, with a line in the schedule.
- 2026-11-05 about 00:45. Second independent check (outside-judge agent, given the artifacts and the
  claims, not the reasoning; it recomputed E1 to E5 and E4 from the raw arrays and read the stage 3
  recordings' hashes on the PC). Every number held. What it found, and what was done:
  - The precision fix missed a path: `voe._render` calls `sense_all` outside `data.chunks`, so on the GPU
    the surprise-test movies were still ray-cast at TF32. Proof: surprise from the stimuli alone
    (`S_copy`, the same for every learner seed) came out 0.005146 / -0.001158 / -0.001371 on the GPU against
    0.005597 / -0.001628 / -0.001204 on the Mac, and rounding the ray caster's inputs to TF32 on the Mac
    reproduced the GPU values within 4e-6. Fixed at the source (next entry).
  - E4's PASS was stated too broadly (it covers M, not probes or surprise tests); the "isolate one cause"
    lines left out run-to-run variation and a last-bit NumPy difference; "68 to 204 times", not "70 to
    220"; two timing-agreement claims were wrong; `gpu-checks-4` started at 23:40, not 23:55; a sense
    list and the test counts were wrong; STATUS was stale; `compare_gpu_stage3.log` predated the
    file-naming fix (regenerated). All corrected in place, each correction marked.
  - Not backed by a saved file: the memory-probe numbers, the `cuda_async` block sizes, the VM stopping
    within 20 s, the per-process GPU load (wallpaper, compositor) and the stopped (T) state of paused
    processes. They were seen over SSH; their output was not saved. The memory probe now saves its
    output (`results/hardware/memprobe_<allocator>.txt`, in `speed_check.sh`); the rest stand as
    observations.
  - Four tooling risks to the shared PC, and four smaller ones, fixed in the entry after next.
- 2026-11-05. Precision fixed at the source: every matrix product in `src/umwelt/senses.py` (ray caster,
  hearing, touch, body sense) names `precision=HIGHEST`, so every caller is covered: recordings, the
  surprise-test movies, anything added later. The `data.chunks` context is removed. A new test
  (`tests/test_precision.py`) traces a rollout in every world, and the senses, and fails if any
  `dot_general` is not at full precision; it fails on the previous code and passes now. On the Mac the
  output is byte-identical: recordings 19 of 19 arrays, senses 14 of 14, training identical, and the
  surprise-test movies hash the same before and after (`a7c905e4e8ab2e20`; that hash describes the movies
  before the 2026-11-05 wall correction). Tests 11 of 11.
- 2026-11-05. Tooling fixes from the check, each with a test that the previous version fails:
  - `job.sh` removed the lock when its owner line was empty; another project may have just created it
    and not yet written its line. It now removes only a lock whose owner line is exactly its own.
  - `job.sh` compared manifest lines, so a file edited on the PC after queueing ran unnoticed. It now
    rehashes the files at start, as `push` verification does.
  - `push` never removed files deleted on the Mac, so the PC could keep importing a module the Mac no
    longer has. It now removes the files the previous push sent that are gone.
  - A `.pid` file outlives a VM restart, and the process id can be reused: `stop` could signal an
    unrelated process (shown in isolation with the previous version), and `unlock` and the push guard
    could be fooled. They now check that the process is that job's `job.sh`.
  - Smaller: `job.sh` stops whatever the command left running before releasing the lock; it sets
    `JAX_PLATFORMS=cuda,cpu` so a missing GPU is an error, not a CPU fallback (unless `--allow-cpu`);
    `resume` removes only a pause `remote.sh pause` made, so Henry's own `mkdir C:\gpu-pause` stays
    (`--force` overrides); `remote.sh sh` no longer lets JAX reserve 75 percent of the GPU; `setup` runs
    outside the lock, so its GPU check is now a small product run only when the card is nearly idle, and
    the tests run on the CPU.
  - Tests: `job.sh` 26 of 26 (the previous version fails the 4 new ones); `remote.sh` 36 of 36. (Corrected
    by the third check: the push-guard and `sh` fixes had no test that failed on the previous version, and
    `series.sh` had none at all. Added 2026-11-05, entry below.)
  - Not changed: jobs take the lock whenever it is free and do not read the schedule's times; the
    schedule is a courtesy between sessions, the lock is the rule.
- 2026-11-05 01:00. `series.sh` started E7 and the speed check, one job at a time (`e7-f32det`,
  `e7-default`, `speed-check-3`, `e7-f32det-rerun`), behind the other project's second generation chunk and its
  queued training run; the schedule line was updated. The first push with the new deletion step went
  through on the real PC (verified, nothing to delete). Log: `results/hardware/jobs/series-2026-11-05.log`.
- 2026-11-05 03:08. Series done: all four jobs exit 0 (`e7-f32det` 01:49, `e7-default` 02:31,
  `speed-check-3` 02:52, `e7-f32det-rerun` 03:08). The first ran at commit 7d70dde, the others at 3447602;
  the two differ only in `docs/gpu.md`. Results under E7 and E6 below.
- 2026-11-05. Saved memory probes (`results/hardware/memprobe_default.txt`, `memprobe_cuda_async.txt`):
  - Default allocator: one array of 3.5 GiB works, 3.99 GiB fails. Ten 1 GiB arrays held at once work,
    then the ceiling (JAX's 75 percent, 11.9 GiB).
  - `cuda_async`: every single size up to 12 GiB works, and sixteen 1 GiB arrays were held at once
    (sixteen is the probe's own loop bound, not a limit it found): 16 GiB on a card with 15.9 GiB, past
    the 11.9 GiB ceiling the default allocator keeps. With the desktop's share on top, the card was
    oversubscribed, which is consistent with Windows placing GPU memory in system memory; residency was
    not measured.
  - Hazard: a `cuda_async` job can grow past the card unnoticed, slowing itself, and anything else on the
    GPU, without an error. It stays opt-in, for steps that need more than 3.5 GiB, and such a job should
    keep its working set well under the card's free memory.
- 2026-11-05 03:11. Pushing the Mac's lawful recording for E7b found the PC's copy of that folder holding
  a 19th array (`gt_light.npy`) left from an earlier GPU recording: `push` adds and replaces files but
  does not remove extras inside a pushed data folder. The stray array was all lit (no effect in the lawful
  world) and was removed; the 18 arrays matched the Mac's file by file. Recording hashes are now taken in
  C-locale file order (`e7.sh`), since the two shells list underscored names in different orders. Rule:
  push a recording only into a folder that does not exist on the PC yet, or compare file lists after.
- 2026-11-05 03:27. E7b passed (results below). `job.sh` now gives every job a persistent compile cache
  (`results/jax_cache`); 27 of 27 job tests.
- 2026-11-05 about 03:40. Third independent check (outside-judge agent; it recomputed E7, E7b and E6 from
  the raw files, decomposed one `cause` score by swapping recordings, and read hashes on the PC). Every
  number held. It corrected wording and margins (marked in place), found the dog's wall tie in the
  surprise-test movies and the `cause` probe's hypersensitivity (both under E7 and E7b), showed that
  nothing enforced the pooling rule or recorded how a result was made, and listed tooling risks. What
  was done:
  - `data.load` refuses to record when `UMWELT_NO_RECORD` is set, which `job.sh` sets unless the job was
    started with `run --record`; `data.recording_hash` gives the same 16 hex digits as the shell.
  - Every result file of `umwelt run` gets a `provenance` block, and a tag refuses minds made with other
    settings or another recording (`tests/test_provenance.py`, which fails on the previous code).
  - `push PATH` verifies the pushed data file by file on the PC (extra, missing or altered fails).
  - `job.sh`: removes a dead umwelt job's lock while waiting (never another project's, never a running
    job's); caps JAX at half the card; caps the compile cache at 2 GB; passes the code version on.
  - `status` reports a dead job; `series.sh` stops on one, passes `--record`, and runs from a function, so
    editing it mid-run is safe (editing it mid-run broke its last line once, after the cache check's job
    had finished).
  - Tests: `job.sh` 34 of 34 [correction: 32 of 32, see 2026-11-05 13:05], `remote.sh` 47 of 47 (now
    including `series.sh`), project 14 of 14. The previous `job.sh` fails the new record and dead-lock
    tests; the previous `remote.sh` fails the record, dead-job and data-verification tests; the version
    before the second check fails the `sh` test. The push guard's process check was shown in isolation (the
    old guard refuses a push because of an unrelated process holding a reused id; the new one does not).
  - The dog's wall tie is not fixed: moving the hidden wall changes the dog's stored results on the Mac,
    so it is Henry's call.
- 2026-11-05 04:21. The new tooling on the real PC (`smoke-final`, log in `results/hardware/jobs/`): the
  Mac's lawful recording pushed and verified file by file; the job ran with the recording guard on, half
  the card, the 2 GB cache cap (correction: that cap switched the cache off, see 12:45 below) and its code
  version (ecf3527); `experiment.provenance` on the GPU gave the
  recording hash `e9b74db4218d7e27`, the same as the Mac's; asking for a recording that is not there was
  refused and nothing was recorded.
- 2026-11-05 about 04:45 to 10:05. Henry chose to fix the surprise-test geometry. `voe.HIDDEN_WALL` now lies
  one unit outside the arena (a test checks that moving it further changes only the oracle's input). Only
  the dog's permanence movies and the oracle's change; their scores were recomputed from the stored networks
  in all 8 Mac runs (`experiments/recompute_voe.py`, which first reproduced every stored value exactly), and
  both reports regenerated with no verdict changed. Full record: `docs/preregistration.md`, last section.
  E7b was queued again (`e7b-wallfix`) behind the other project's server, which held the GPU from 09:10.
  How to read it when it comes back: not by `compare_e7.py`'s PASS line. The independent check showed that
  the old E7b output, compared with the corrected Mac values, still prints PASS, because the per-family rule
  absorbs one oracle score and `S_copy` cannot see a static wall. The test is the dog's and the oracle's
  permanence surprise directly: GPU against corrected Mac, expected within about 1e-6.
- 2026-11-05 12:28 to 12:42. `e7b-wallfix` ran (the other project's server released the lock at 12:28). Result:
  the dog's mismatch is gone, everything else repeats bit for bit (section "E7b after the stimulus fix").
  Fourth independent check (outside-judge agent, its own code on the raw files, the code fingerprint
  rebuilt from `git archive cf27e28`): confirmed, with three corrections, all applied.
  - My comparison skipped the interval ends: the largest difference among the dog's and oracle's 48
    surprise numbers is 5.0e-7 (an oracle solidity interval end, 5.1e-7 apart before the fix too), not the
    2.5e-7 I first wrote. `compare_wallfix.py` now compares all 48.
  - `compare_wallfix.py` failed when rerun, on `compare_e7.py`'s summary file in the same folder. It now
    reads only the minds' files.
  - Nothing could tell a result scored on the old surprise-test movies from one scored on the new:
    `compare_e7.py` passes the pre-fix GPU run against the corrected Mac, `S_copy` cannot see a static
    wall, and provenance had no stimulus field. Now `voe.suite_hash` names the rendered movies, every
    result's provenance records it as `stimuli`, and a tag refuses to reuse a mind scored on other or
    unknown movies (c63b23e; two new tests, which the previous code fails). The same tiny run on the
    commits before and after gave the same 59 numbers. `e7.sh` writes the code version into `env.txt`,
    since the two E7b runs' `env.txt` files were byte-identical.
- 2026-11-05 about 12:45 to 13:05. Finding, from a warning in `e7b-wallfix`'s log: the compile cache had
  not worked since ecf3527 (04:20). JAX keeps the cache's size cap (`JAX_COMPILATION_CACHE_MAX_SIZE`, set
  by `job.sh` since ecf3527) only with the `filelock` package, which the environment lacked. Without it
  the first cache read raises, JAX prints one warning ("Please install the `filelock` package") and
  compiles everything itself, and nothing is written: on the PC no cache entry was newer than 02:49.
  Speed only: no result depends on the cache (`cache-check` above), and both E7b runs compiled from
  scratch. Fix (a30fb0b): `filelock` joins the `gpu` extra; `job.sh` drops the cap rather than the cache
  if it is ever missing, and ends each log with the cache's entry count before and after and its number
  of cache errors (three new job tests, which the previous `job.sh` fails). On the PC's CPU, two
  processes building the surprise-test movies with the cap set: the first wrote 1,351 entries in 65 s,
  the second read them in 24 s, no errors, no new entries.
- 2026-11-05 13:00 to 13:25. Finding: the surprise-test movies are bit-identical on the Mac's CPU and the
  PC's CPU (`voe.suite_hash` `e8b112a5e2a11879` on both, different processor families), but not on the GPU.
  `stimuli-check` (log in `results/hardware/jobs/`) hashed them on the GPU: `ceb701ed0ddd1c3a` at full
  float32 with deterministic operations, `3776a5c3ffed017d` at the GPU's defaults. `stimuli-diff`
  (`experiments/gpu/stimuli_diff.py`, both devices in one process) located the difference: 8 of the solidity
  movies' 2,211,840 vision values, by at most 0.000244, in 8 of 72 movies, frames 51 to 59;
  every other array identical, at both settings. Plausibly the same kind of last-bit tie as the hidden wall,
  here from the solidity wall seen edge-on (not shown). Too small to move a score (E7b after the fix agrees
  within 2.5e-7 with these movies), but it made the movies, and so the `stimuli` hash, machine-dependent.
  Fix (7651cea): `voe.build_suite` renders on the CPU on every machine, the way a recording is shared rather
  than regenerated; nothing changes on the Mac, which renders on its CPU already (a new test, which the
  previous code fails). The cache check in the same job: the first process wrote 1,351 entries in 109 s, the
  second read them in 42 s, no errors. Confirmation of the fix on the GPU: `stimuli-check-2`, below.
- 2026-11-05 13:20 to 13:21. `stimuli-check-2` (code 845d1e4, CPU rendering): all three processes on the PC,
  at full float32 with deterministic operations twice and at the GPU's defaults, give `e8b112a5e2a11879`,
  the Mac's hash. So a GPU job now scores on exactly the Mac's movies, and `stimuli` in provenance can be
  compared across machines. 20 s each, from the compile cache (5,480 entries before and after, no errors;
  the CPU compiles were cached by `stimuli-diff`).
- 2026-11-05 13:05. Correction to the test counts: `test_job.sh` had 32 cases, not 34 (the 03:40 entry and
  "How to run"; counted from the file and from a run of it against the `job.sh` of that time). It now has
  35; a30fb0b's message says 37, which is wrong.

## How to run on the GPU

Status: working against the real PC since 2026-11-04 (setup, push with verification, jobs, pause,
pull; see the setup log). Both scripts also have tests on the Mac against stand-ins
(`experiments/gpu/test_job.sh`, 35 cases; `experiments/gpu/test_remote.sh`, 47 cases, with a fake ssh
that plays the PC).

    experiments/gpu/remote.sh setup                       # once: copy code, install uv and CUDA JAX, run tests
    experiments/gpu/remote.sh run NAME CMD...             # background job; waits for a free GPU
    experiments/gpu/remote.sh status NAME                 # GPU use, lock, last log lines, exit code
    experiments/gpu/remote.sh pull results/runs/v3        # copy results back to the Mac
    experiments/gpu/remote.sh pause                       # freeze umwelt GPU work; resume undoes it
    experiments/gpu/remote.sh stop NAME                   # end a queued or running job
    experiments/gpu/series.sh NAME 'CMD' NAME 'CMD' ...   # several jobs, one at a time, a gap after each

On the PC itself, `mkdir C:\gpu-pause` pauses and `rmdir C:\gpu-pause` resumes (a file works too).

For example, one learner seed of the dark-world sweep, on the Mac's own recording of data seed 1:

    experiments/gpu/remote.sh push results/data/dark_s1_e8448_t256      # verified file by file on the PC
    experiments/gpu/remote.sh run dark-s2 env JAX_DEFAULT_MATMUL_PRECISION=highest \
        XLA_FLAGS=--xla_gpu_deterministic_ops=true \
        uv run umwelt run --world dark --seed 2 --tag v3-gpu --data-seed 1 --n-train 8192

What a job guarantees (`experiments/gpu/job.sh`):

- It starts only when the folder `C:\gpu-lock` does not exist and less than 6 GB of GPU memory is in
  use. The other project uses the same lock (`the other project's job script` waits on it and writes a
  `other-project ...` owner line). A job creates the folder with its own owner line and only ever removes a
  lock that is still its own.
- It refuses to start (exit 3) if the code on the PC changed after it was queued, and refuses (exit 5)
  if JAX does not see the GPU, unless started with `run --allow-cpu`. The log records the code hash, the
  git commit, the number of uncommitted files with a hash of their diff, and the JAX backend.
- It syncs the environment once (`uv sync --frozen --extra gpu`) and then sets `UV_NO_SYNC=1`, so no
  `uv run` inside the job can drop the CUDA plugin and fall back to the CPU silently.
- A stop signal (SIGTERM, SIGINT, SIGHUP) stops the whole command first, then releases the lock. A
  SIGKILL leaves the lock in place; `remote.sh unlock` removes it only if it is umwelt's and its job is
  dead (`--force` overrides).
- Job names are used once, so a log never mixes two runs. JAX takes GPU memory as needed
  (`XLA_PYTHON_CLIENT_PREALLOCATE=false`).
- Every job uses a persistent compile cache (`results/jax_cache` on the PC), which halved a mind's time
  and changed no result. (Correction: from ecf3527 to a30fb0b the size cap below switched it off; see the
  setup log, 2026-11-05 12:45.) The last lines of a job's log give the cache's entries before and after
  and its errors, so a job that did not use it shows.

- A job may not record a world: it must be given its recordings (`UMWELT_NO_RECORD=1`; `data.load` then
  refuses to record). `run --record` allows it, for checks that record on purpose. JAX may use at most
  half the card (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`, 8 GB); the compile cache is capped at 2 GB (the cap
  needs the `filelock` package, in the `gpu` extra).
- Every result file of `umwelt run` carries a `provenance` block: machine, backend, device, JAX version,
  matmul precision, XLA flags, code version, the recording's content hash and, since c63b23e, the
  surprise-test movies' hash (`stimuli`). Rerunning a tag with other settings, another recording or other
  movies stops with an error instead of keeping the earlier minds. Before pooling a GPU result with a Mac
  result, compare `recording_hash` and `stimuli` (the movies are rendered on the CPU everywhere since
  7651cea, so the hash is the same on every machine). The Mac's stored runs predate provenance, but all their
  surprise scores are those of the movies with hash `e8b112a5e2a11879`: every stored dog and oracle file
  (17) carries the 2026-11-05 correction, and every other mind's inputs are the same under the old and
  new movies.
- While waiting, a job removes a lock left by an umwelt job that is no longer running (a restart or a
  kill), so a dead umwelt job cannot block the other project for long. `status NAME` says whether a job
  finished, is queued or running, or died without an exit code; `series.sh` stops on a dead job.

For numbers that will be reported or pooled with the Mac's (E7b):
1. Use a recording made once and copied, never one regenerated from the same seed on the other machine.
2. Run at full float32 with deterministic operations. A whole run then repeats bit for bit, and on the
   same recording matches the Mac's to the sixth decimal for nearly every score. At the GPU's default
   precision one cause score moved by a seed's worth on identical episodes (E7, blind, linear probe).
   Full float32 costs 10 to 50 percent per update on the update alone and was faster in the training
   loop; deterministic operations cost 3 to 16 percent, within the noise between jobs; the two together
   were not timed separately (a full 14-mind run took about 15 minutes either way).
3. The surprise-test movies are rendered on each machine, not copied. They contained a geometric tie that
   made the dog's permanence score device-dependent (see E7b); it was fixed on 2026-11-05, and E7b was rerun
   to confirm.

    experiments/gpu/remote.sh run NAME env JAX_DEFAULT_MATMUL_PRECISION=highest \
        XLA_FLAGS=--xla_gpu_deterministic_ops=true uv run umwelt run ...

Several jobs go through `series.sh`, not several `run`s at once: it queues the next job only after the
last has finished and 90 s have passed, so another project's job already waiting on the lock gets the GPU first.

Limits inside WSL: with the default allocator one block of 3.99 GiB already fails (3.5 works), so a
compiled step whose working memory is near 4 GiB fails (for the current learner, 64 minds at once; 32
work). `env XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` lifts that, slows the ordinary training loop by 9
percent, and has no memory ceiling at all (it held 16 GiB on the 15.9 GiB card): watch such a job's memory.
Keep to 64 minds per compiled step.

Code goes to the PC with `push`: every tracked or new file outside `results/` (deleted files skipped,
no macOS metadata), plus `.umwelt_push`, the manifest a job checks against. The PC rehashes what
arrived; a mismatch fails the push. A push that would change
the code while an umwelt job is queued or running is refused. Results stay on the PC until pulled;
`pull` never overwrites a local file that differs unless given `--overwrite` (it reports the file and
leaves the PC's version in a temporary folder). Recordings go to the PC only when pushed:
`remote.sh push results/data/<recording>`. The PC rehashes them file by file, and an extra, missing or
altered file fails the push. A job that lacks a recording stops instead of regenerating it, since a
regenerated recording is not the same data (E1, E7).

All hardware checks in one job, to rerun after a driver, CUDA or JAX upgrade: `experiments/gpu/checks.sh`
(its header has the commands to start it and to compare its results).

## Experiment protocol (written before running)

The question: can a number computed on the PC (CPU or GPU) be compared with, or pooled with, a number
computed on the Mac? Floating-point results differ across chips (different instruction sets, fused
multiply-add, transcendental functions, and on the GPU, TF32 matrix multiplies and parallel reductions
in a different order). Bit-identity across machines is not expected. The standard is: differences that
hardware causes must be much smaller than differences between learner seeds, which every claim in this
project already has to survive.

- E1 World data. Generate the same episodes (same world, same seed) on each machine and compare every
  recorded array: identical, or how many elements differ and by how much. Discrete outcomes (contacts,
  visibility, light) show whether small float differences grow into different episodes.
- E2 Senses. Feed one machine's recorded episodes to every preset's sense model on each machine and
  compare observations. This separates the sense code from the world simulation.
- E3 Training trajectory. Same inputs (one machine's episodes, same standardization), same seed: train
  the human mind for 300 updates on each machine and compare the loss at every update and the final
  weights.
- E4 Result level. Rerun calibration stage 3 end to end on each machine (`calibrate.py dark
  --episodes 8192`: human and memoryless, data seed 0, 4,000 updates). Compare test error, memory value
  M overall, and M at object returns with the Mac's numbers (M 0.022, M at returns 0.106). The yardstick
  is the learner-seed spread in the confirmatory run: M at returns 0.093 (seed 0) against 0.057 (seed 1).
  Pass: every hardware difference is under a quarter of that spread (0.009 in M at returns).
  Added before the GPU run: a third GPU pass on the very episodes the PC's CPU run recorded under
  Windows, compared with that CPU run. Same bytes in, so the difference is GPU arithmetic alone; the
  end-to-end passes also carry the small difference between recordings made on different chips (E1).
- E5 GPU only. Matrix multiply precision (JAX's default on this GPU class, TF32, against
  `jax_default_matmul_precision=highest`) and run-to-run determinism (the same GPU run twice, with and
  without `--xla_gpu_deterministic_ops=true`).
- E6 Speed. Time one training update for the current learner (hidden 128, batch 32), larger batches,
  larger cores (hidden 512, 1024), and many minds trained at once (`vmap` over 1, 8, 32 minds). Time
  the world simulator. Time one full mind end to end (training, probes, surprise tests).
  Expectation, to be tested: the current learner is too small to fill a GPU, so one mind may run no
  faster than on the Mac; the gain should come from many minds at once and from larger learners.

Added 2026-11-05, after the second independent check and before running it:

- E7 Full results. E4 checks prediction error only. E7 checks every score a run reports. The run:
  `umwelt run --world lawful --seed 0`, all 14 minds with controls, surprise tests and translation. On
  the GPU it records its own world, in three configurations, one job each (`experiments/gpu/e7.sh`):
  full float32 with deterministic operations (`f32det`, the candidate standard for real GPU work), the
  same again (`f32det-rerun`), and the GPU's defaults (`default`: TF32 products, nondeterministic
  operations). The reference is the Mac's stored run (`results/runs/main/lawful_s0`). Today's code
  reproduces it exactly: every non-timing number of the human mind is identical
  (`results/hardware/cache_and_reproduction_check.txt`). The yardstick is
  each score's own learner-seed difference on the Mac (`results/runs/main/lawful_s1` against `lawful_s0`).
  Comparison: `experiments/gpu/compare_e7.py`. Families: test error (14 scores), MLP probes (168),
  linear probes (168), surprise S (42), translation CKA (182), translation R squared (182). A family passes
  when the median and the 95th percentile of its hardware differences are both at most a quarter of the
  same statistics of its seed differences. E7 passes when every family passes and no score is missing.
  Reported but not judged:
  - the controls (untrained network, raw senses; the seed-1 run skipped them, so there is no yardstick);
  - `S_copy`, surprise from the stimuli alone, which must match the Mac's to float precision now that the
    movies render at full precision;
  - for `f32det` against `f32det-rerun`, whether every number is identical.
  The comparison was tried on the Mac first: the Mac against itself passes with every difference zero; seed
  1 posing as another machine fails every family; a run with only one mind fails as incomplete.
- E7b, added 2026-11-05 03:10 (commit e9966c7) after E7 failed and before running it: E7's `f32det`
  configuration on the Mac's own recording (`results/data/lawful_s0_e1280_t256`, pushed and copied in;
  hash checked on both machines), so the only difference from the Mac's run is GPU arithmetic (not quite:
  the surprise-test movies are rendered on each machine, see the result). Same comparison, same rule.
  Prediction, from the exploratory diagnostics under E7's result: it passes, which would make recording
  differences, not GPU arithmetic, the cause of E7's failure.
- Standing rule, from E4, E7 and E7b together (see "What E4, E7 and E7b establish together"): pooled
  results share recordings, and a result is validated for the learner it was run with. Before a
  new learner family (larger cores, attention) is pooled across machines, rerun E4 for it, or run its
  confirmatory work at full float32 with deterministic operations.

## Results

### E1 to E3: Mac CPU against PC CPU (2026-11-04)

`experiments/gpu/repro.py`, 64 episodes per world, data seed 0. Outputs in `results/hardware/`.

A machine reproduces itself exactly: two runs on the Mac gave identical world data (19 of 19 arrays),
identical observations (14 of 14 presets) and identical losses at all 300 updates.

Across machines (Mac M2 Pro against PC Core Ultra 9, both JAX 0.12.0.1 on CPU, same locked packages):

- E1 World data is not bit-identical. Of 19 arrays, only the three driven by random draws alone are
  identical: light timing (`gt_light`), which objects are active, and their kinds. Actions differ in 61
  of 64 episodes, by at most 0.01 (in one episode from the first step; the median episode first differs
  at step 50). Agent heading differs by at most 0.005 radians and
  agent position by at most 0.007 (the arena is 10 units across). Most of these differences stay
  small. Small differences grow into a different event in only a few episodes: object positions differ
  by more than 0.0001 in 3 of 64 episodes, by more than 0.1 in 1 of 64 (from step 200 on), and
  visibility of an object differs in 1 episode. The likely source is the CPU math libraries (sine,
  cosine, square root, fused multiply-add) on ARM against x86, fed into a world with collisions.
  The lawful and dark recordings share their physics (darkness acts only at the eye), so they are
  identical in 17 of 19 arrays and E1 is one test, not two.
- E2 Senses, on identical episodes: 5 of 14 presets identical. The other 9 differ by at most 1.2e-7
  (one or two float32 rounding steps, relative size 1e-9). They are exactly the 9 presets with a smell
  sense, and every differing value is in a smell column. The smell channel is the only sense passed
  through a transcendental function (`np.log1p`, `src/umwelt/presets.py:88`), whose last bit differs
  between the two platforms' math libraries. Sensor noise is not the cause: four presets with noise and
  no smell (vision only, hearing only, touch only, bat) are identical. (The first version of this log
  blamed NumPy's normal sampler; the independent check showed that was wrong.)
- E3 Training, on the Mac's episodes: the loss is identical at updates 0 and 1 and differs by at most
  1.8e-6 (relative) over 300 updates. Final weights differ by at most 3e-6 (relative L2 norm). Held-out
  error: 0.394354 on both, equal to six decimals.

What this means: a data seed names the same world and nearly the same episodes on any machine, not the
same bytes. A run that must reuse exact episodes (for example a new learner seed on the data of an
earlier run) should copy the recorded data, not regenerate it. E4 tests whether the episode
differences matter for the project's numbers.

### The Mac reproduces its stored reference (2026-11-04)

Calibration stage 3 rerun on the idle Mac with today's code: the per-step error arrays of both minds are
byte-identical to the stored ones (`results/calibration/errors/dark_n8192_*.npy`). So the Mac's numbers
are a fixed reference. Only the recorded training times changed (human 134 s and memoryless 58 s on the
idle Mac, against 496 s and 363 s when the reference ran beside the other project fine-tune). The committed
files were restored; the rerun's timing files are in `results/hardware/mac-cpu/stage3_rerun/`.

### E4: calibration stage 3 on the PC's CPU (2026-11-04)

`calibrate.py dark --episodes 8192` on the PC, end to end: it recorded its own 8,448 episodes (which
differ slightly from the Mac's, E1), then trained and scored both minds. Compared with
`experiments/gpu/compare_stage3.py`:

| | Mac | PC CPU | difference |
|---|---|---|---|
| M, all steps | 0.0218 | 0.0218 | +0.00001 |
| M, dark steps | -0.0278 | -0.0278 | -0.00000 |
| M, first lit step | 0.1362 | 0.1363 | +0.00003 |
| M, object returns | 0.1060 | 0.1060 | +0.00002 |
| test error, human | 0.21917 | 0.21917 | -0.00% |
| test error, memoryless | 0.22405 | 0.22405 | -0.00% |

The largest difference in M is 0.00003, 300 times under the bar (0.009, a quarter of the learner-seed
spread). Per step, errors differ by a median of 2 to 3 parts in 10,000; 8 or 9 of the 256 test episodes
have a step that differs by more than 0.01, none by more than 0.1. The episodes differ a little too: 46
of the PC's 8,448 episodes see an object at a different moment somewhere, and the PC's test set has one
object-return step more (2,431 against 2,430). The bar comes from the learner-seed spread of M at object
returns; for overall M that spread is smaller (0.0029, a quarter is 0.0007), and the PC's difference
(0.00001) is still 50 times under it. PASS: results from the PC's CPU can be compared with and pooled
with the Mac's.

The PC's CPU was slower than the Mac here: human 331 s and memoryless 254 s, against 134 s and 58 s.
That run was at idle priority beside the other project GPU job, so this is not a clean timing; the 300-update
E3 run, alone on the PC, was also slower (28 s against 11 s, compile included).

### E1 to E3 and E5 on the GPU (2026-11-04)

`experiments/gpu/checks.sh` section `repro`, on the RTX 5080 inside WSL (code 2a972b5, before the
world precision fix). Comparisons in `results/hardware/compare_gpu.log` and `compare_*_vs_pc-gpu*.json`.

- E1 World data, GPU against Mac. The world simulator is JAX code, so a GPU job also simulates on the
  GPU, whose math functions (sine, cosine, square root, exponential) round differently from either CPU.
  Episodes therefore part from the Mac's sooner than the PC CPU's did: actions differ in all 64
  episodes (14 of them from the first step, median first difference at step 2; PC CPU: 61 episodes,
  median step 50), by at most 0.031 (PC CPU 0.010). Agent position differs by at most 0.035 and heading
  by 0.014 radians; object positions by more than 0.0001 in 9 episodes and by more than 0.1 in 2 (PC
  CPU: 3 and 1). Physics and every sense but the ray caster's (vision, ray texture, and through them
  echolocation) were byte-identical between the two GPU precisions. Those, and the object-visibility
  labels, were not: at the default precision (TF32) the ray caster flipped visibility at 130 of 98,304
  entries, against 4 at full float32 and 2 on the PC's CPU. That was fixed (setup log, two steps);
  `checks.sh precfix` checks the fix on the GPU.
- E2 Senses, on the Mac's episodes: the sense presets are NumPy code and run on the CPU in a GPU job
  too. Same pattern as the PC CPU: the same 5 presets are identical, the 9 with smell differ, by at most
  7.5e-9 here (Linux's math library against the Mac's; Windows' against the Mac's gave 1.2e-7).
- E3 Training, on the Mac's episodes, 300 updates:

| GPU setting | loss, largest relative difference from the Mac | weights, largest relative difference | held-out error (Mac 0.394354) |
|---|---|---|---|
| full float32 (`highest`) | 8.1e-6 | 5.8e-6 | 0.394354 |
| default (TF32) | 1.2e-2 | 1.4e-2 | 0.397407 |
| default, deterministic operations | 1.3e-2 | 1.5e-2 | 0.398745 |

  At full float32 the GPU reproduces the Mac's training to about five significant digits, as closely as
  the PC's CPU did. TF32 matrix multiplies (inputs rounded to about three digits) move the trajectory
  by about one percent within 300 updates.
- E5 Run-to-run, on the GPU. At the default settings two identical runs record identical worlds and
  senses (19 of 19 arrays, 14 of 14 presets) but train differently: the loss differs by up to 0.3
  percent and held-out error is 0.397407 against 0.398281. With `--xla_gpu_deterministic_ops=true` two
  runs are identical to the last bit (every loss, every weight, held-out error 0.398745 both times). A
  run paused for 40 s mid-training is also bit-identical to an unpaused one under that flag (setup log).
  Timing note: the 300-update runs took 16.6 s at the default and 7.3 s with deterministic operations;
  both include compiling, and E6 times the update itself.

What this means so far: the GPU is not a source of error in itself. At full float32 it trains like the
CPUs; the open question is whether TF32's one-percent drift, and run-to-run variation, matter at the
level of the project's numbers. E4 answers that, with the GPU at both precisions.

### E4 on the GPU (2026-11-05)

`checks.sh stage3` on the RTX 5080, with the world precision fix in. Three passes, each recording its own
8,448 episodes except the last: the GPU's default precision (TF32 learner), full float32, and the default
precision on the very episodes the PC's CPU run recorded under Windows. Compared with
`experiments/gpu/compare_stage3.py`; full output `results/hardware/compare_gpu_stage3.log`.

| M (memory value) | Mac | GPU default | GPU full float32 | GPU on the PC CPU run's episodes |
|---|---|---|---|---|
| all steps | 0.0218 | 0.0218 | 0.0218 | 0.0217 |
| dark steps | -0.0278 | -0.0277 | -0.0278 | -0.0278 |
| first lit step | 0.1362 | 0.1363 | 0.1361 | 0.1362 |
| object returns | 0.1060 | 0.1060 | 0.1059 | 0.1060 |
| other steps | 0.0098 | 0.0099 | 0.0099 | 0.0098 |
| largest difference from the Mac | | 0.00004 | 0.00011 | 0.00006 |
| test error, human (Mac 0.21917) | | 0.21917 | 0.21917 | 0.21917 |
| test error, memoryless (Mac 0.22405) | | 0.22406 | 0.22405 | 0.22404 |

Two comparisons isolate one cause each:
- Same episodes in, GPU against the PC's CPU: largest difference in M 0.00008. This is GPU arithmetic,
  plus a last-bit difference in the smell channel between Windows' and Linux's math libraries (E2),
  which is negligible.
- GPU default against GPU full float32, on byte-identical recorded worlds: 0.00013. This is TF32 in the
  learner plus the run-to-run variation default GPU runs have (E5); stage 3 at the default precision ran
  once, so the two are not separated. The one-percent drift in the loss at 300 updates (E3) does not reach the result.

Every difference is 68 to 204 times under the bar (0.009, a quarter of the learner-seed spread of M at
object returns), and the largest difference in overall M (0.00005) is 14 times under a quarter of overall
M's own seed spread (0.0007). Per-step errors correlate at 1.0000 with the Mac's in every pass (0.999996 at least).

PASS, for what E4 measures: prediction error and memory value M, for two minds, one world, one seed and
the current learner. It does not cover probe scores or surprise tests. The one full mind run on both
machines (human, lawful world, `bench.py` e2e) differs from the Mac in some probe scores by more than a
quarter of their learner-seed difference, and its surprise-test movies were still rendered at TF32 (fixed
2026-11-05). E7 tests that level before any GPU result is pooled. (The first version of this section
said "GPU results can be pooled with the Mac's, at either precision"; the second independent check showed
that went beyond the evidence.)

Speed on the way: recording the 8,448 episodes took 14 s on the GPU (41 s on the PC's CPU under Windows,
at idle priority beside another project's GPU job, so not a clean comparison);
training took 68 s for human and 22 s for memoryless (idle Mac 134 s and 58 s).

### E7 on the GPU (2026-11-05): FAIL by the preregistered rule

Three runs (`series.sh`, 01:00 to 03:08; logs in `results/hardware/jobs/`), compared with
`experiments/gpu/compare_e7.py`; full output `results/hardware/compare_e7.log`. Hardware difference is
|GPU - Mac seed 0|; seed difference is |Mac seed 1 - Mac seed 0|; a family passes when both its median and
its 95th percentile are at most a quarter of the seed difference's.

| family | n | GPU f32det: median / p95 | GPU default: median / p95 | seed difference: median / p95 | verdict |
|---|---|---|---|---|---|
| test error | 14 | 0.00002 / 0.00063 | 0.00001 / 0.00060 | 0.00066 / 0.00396 | pass |
| MLP probes | 168 | 0.00019 / 0.00442 | 0.00020 / 0.00446 | 0.00463 / 0.02059 | pass |
| linear probes | 168 | 0.00017 / 0.00647 | 0.00016 / 0.00660 | 0.00333 / 0.01975 | **fail** (p95 bar 0.00494) |
| surprise S | 42 | 0.00003 / 0.00033 | 0.00003 / 0.00030 | 0.00200 / 0.01123 | pass |
| translation CKA | 182 | 0.00020 / 0.00142 | 0.00021 / 0.00153 | 0.00811 / 0.03951 | pass |
| translation R squared | 182 | 0.00011 / 0.00066 | 0.00011 / 0.00062 | 0.00384 / 0.02516 | pass |
| controls (reported only) | 336 | 0.00015 / 0.00839 | 0.00015 / 0.00811 | none | |

Both configurations fail on the tail of the linear probes. Every other family passes, but not all by much.
Against the bar, the 95th percentiles pass by 1.17 times for MLP probes (0.00442 against 0.00515; 10 of
their 12 largest differences are also `cause`) and 1.57 times for test error, and by 7 to 10 times for
the rest; the medians pass by 5 to 19 times. (This line first said "most by a factor of 5 to 40", which
compared with the seed statistic, not the bar.) The repeat (`f32det-rerun`) is identical to `f32det` in every non-timing number: a whole run
on the GPU (14 minds, probes, controls, surprise tests, translation) repeats bit for bit with full float32
and deterministic operations. `S_copy`: the compare script reports a largest difference of 2.88, but that
is one value, the oracle's permanence `S_copy` (5027.84 on the Mac, 5030.72 on the GPU, 0.06 percent);
every other `S_copy` is within 0.00004 of the Mac's. Before the source-level precision fix the human
mind's differed by 8 percent, so the surprise-test movies now match.

Exploratory diagnostics, after the verdict (not part of the preregistered test):
- The three GPU runs recorded byte-identical worlds (hash `306a35611bc61e58`, checked by the third
  independent check), so GPU default against GPU full float32 isolates the learner's arithmetic. It is
  far inside the seed noise for every family (95th percentiles 0.00003 to 0.00044), so precision is not
  what fails the family. Single scores are another matter: blind's linear `cause` probe moved 0.0127,
  more than its own learner-seed difference (0.0088), from TF32 alone.
- The failing tail is one concept. Of the 12 largest linear-probe differences, 9 are `cause` (the 95th
  percentile by concept: cause 0.0195, relations 0.0077, every other concept 0.0032 or less). Without
  `cause` the linear probes pass (0.0016 against a bar of 0.0035).
- The raw-senses control, which involves no learner, shows the same pattern (95th percentile for cause
  0.0289, every other concept 0.0035 or less). So the cause is not the arithmetic of training and probing
  but the data: each GPU run simulated its own world on the GPU, its episodes part from the Mac's within
  the first steps (E1), and `cause` labels rare collision events.
- The third independent check decomposed one case (touch_only, raw senses, cause: 0.1405 on the Mac,
  0.1793 on the GPU) by swapping observations and labels between the two recordings: about 97 percent of
  the change comes from the probe's training labels, 6 fewer positives among 7,873, and the observations
  contribute about 0.001. Across the 1,280 episodes the cause label differs in 124 entries (9 training
  episodes, no test episode).
- Finding for the project's science, not only the hardware: the `cause` probe is hypersensitive to a
  handful of labels. Any change of data, on any machine, moves cause scores by about a learner seed's
  worth. Cause results so far should be read with that in mind.
- E7b tests the recording explanation directly.

### E7b: the same run on the Mac's recording (2026-11-05): PASS

Job `e7b-macdata` (03:12 to 03:26), configuration `f32det` (full float32, deterministic operations) on
the Mac's own lawful recording, pushed and copied in (hash `e9b74db4218d7e27` on both machines, file by
file). Full output `results/hardware/compare_e7b.log`.

| family | n | median difference | 95th percentile | identical | largest (score: GPU, seed) |
|---|---|---|---|---|---|
| test error | 14 | 0.0000000 | 0.0000000 | 5 | hearing_only: 6e-8, seed 0.0072 |
| MLP probes | 168 | 0.0000000 | 0.0000003 | 27 | hearing_only/number: 0.00008, seed 0.0062 |
| linear probes | 168 | 0.0000000 | 0.0000001 | 28 | 2e-7 |
| controls (reported) | 336 | 0.0000000 | 0.0000000 | 56 | 1.5e-7 |
| surprise S | 42 | 0.0000001 | 0.0000005 | 6 | dog/permanence: 0.00087, seed 0.0039 |
| translation CKA | 182 | 0.0000001 | 0.0000004 | 12 | 7e-7 |
| translation R squared | 182 | 0.0000001 | 0.0000029 | 0 | 4.5e-6 |

`S_copy` matches to 2e-7. Given the same recorded episodes, the GPU at full float32 with deterministic
operations reproduces the Mac's run to about the sixth decimal for nearly every score: medians at most
1e-7, 95th percentiles at most 3e-6. Three exceptions:
- hearing_only's MLP `number` probe, 8.3e-5: exactly one flipped prediction among 15,872 test rows.
- The dog's permanence surprise, 8.7e-4 (a fifth of its seed difference), and the untrained dog's, 2.4e-4.
  The comparison script leaves the untrained surprise scores out of the judged family (seed 1 has none),
  so that one is not in the table. The independent check found the cause: the surprise-test movies are
  not part of the recording but rendered on each machine, and in them a hidden wall
  (`voe.HIDDEN_WALL`) lies exactly on the boundary wall at y = 0. 3 to 5 of the dog's 240-degree rays,
  97 to 108 degrees off its axis, meet both walls at the same distance (0 or 1 unit in the last place),
  so whether a ray shows the boundary's colour or internal grey depends on last-bit rounding, which
  differs by device. Human eyes stop at 90 degrees, and echolocation sees the same depth for both walls,
  so only the dog moves. On the Mac, forcing either colour brackets the GPU's value (trained 0.00509 and
  0.00586 around the GPU's 0.00560). Strongly supported, not proven: proof needs a GPU render. That spread
  (0.0011) is 29 percent of the dog's permanence seed difference, above the quarter bar for that score.
  The fix is to move the hidden wall off the boundary, which changes the dog's stored results on the Mac
  too; Henry chose to fix it (2026-11-05, see "E7b after the stimulus fix" below).
So E7b isolates GPU arithmetic for everything but the surprise-test movies. The prediction held: E7's
failure came from the recordings.

### E7b after the stimulus fix (2026-11-05): the dog's mismatch is gone

Job `e7b-wallfix` (queued 10:05, waited behind the other project's servers, held the GPU 12:28 to
12:42), configuration `f32det-macdata` again: the Mac's lawful recording (hash `e9b74db4218d7e27`), full
float32, deterministic operations, code cf27e28 with no uncommitted files. cf27e28 is E7b's code plus the
moved hidden wall (the job log's code fingerprint, rebuilt independently from `git archive cf27e28`;
settings and recording hash from `env.txt` and every file's provenance). Later commits changed no
computation (the provenance field, the cache fix and new scripts; the same tiny run before and after gave
identical numbers). Judged as the setup log entry of 10:05 says: score by score, not by `compare_e7.py`'s
PASS line. `experiments/gpu/compare_wallfix.py`, output `results/hardware/compare_e7b_wallfix.log`:

- The dog's and the oracle's surprise numbers, GPU against the Mac's corrected values: all 48 (`S`, both
  interval ends and `S_copy`, for the trained and untrained network, three tests each) agree within
  5.0e-7, the largest an oracle solidity interval end that was 5.1e-7 apart before the fix too; the 12
  `S` values within 2.5e-7. The dog's permanence surprise is 0.0050933 on both (difference 1.2e-8); before the fix the Mac had 0.0047262 and
  the GPU 0.0055979. The untrained dog's is -0.0001687 on both (before: -0.0002376 and +0.0000046). The
  oracle's permanence surprise is 0.38558 on both (0.48394 before the fix, on both).
- Against the GPU's own E7b run from before the fix: every number of the other 12 minds and of the
  translation file is bit-identical. Of the dog's 123 numbers exactly 6 differ, the ones the new movies
  reach (permanence `S` and its interval ends, trained and untrained); of the oracle's, exactly 18 (`S`
  and interval ends of all three tests, trained and untrained). No `S_copy` changed. So the fix changed
  what it was meant to and nothing else, and a GPU run at these settings still repeats bit for bit.
- Against the Mac's stored run, every number: largest difference 8.3e-5, hearing_only's MLP `number` probe
  (the one flipped prediction above); every other mind's file at most 2.2e-6, translation 4.5e-6.
- `compare_e7.py` (appended to `results/hardware/compare_e7b.log`): PASS, every family's median and 95th
  percentile 0.00000 to five decimals.

The cause given in E7b above (strongly supported, not proven) now has its GPU test: with the hidden wall
off the boundary, the dog's difference fell from 8.7e-4 to 1.2e-8. Given the same recording, at full
float32 with deterministic operations, the GPU now reproduces the Mac's full run within 2.5e-7 on every
surprise score, 5e-7 on their intervals and 5e-6 on everything else but that one probe prediction. An
independent check reproduced every number above with its own code (setup log, 12:28 entry). The GPU job
also compiled everything from scratch (the cache was off, setup log 12:45), so it says nothing about the
cache either way. It rendered the surprise-test movies on the GPU, where 8 solidity vision values differ
from the Mac's by at most 0.000244 (setup log 13:00); runs from 7651cea on render them on the CPU.

### What E4, E7 and E7b establish together

- At full float32 with deterministic operations, GPU arithmetic is not a source of error for this
  project. Same episodes in: stage 3 M within 0.00008 of the CPU (E4), and nearly every score of a full
  run within about 1e-6 of the Mac (E7b), the exceptions being the dog's surprise scores (a geometric tie
  in the surprise-test movies, fixed: after the fix they agree within 1.2e-8) and one flipped probe
  prediction. At the GPU's default precision, single
  `cause` scores can move by a seed's worth.
- A data seed is not portable across machines; a recording is. The world simulator rounds slightly
  differently on each chip, so episodes regenerated from the same seed part within a few steps (E1).
  M barely notices (E4), but rare-event concepts do: `cause` probe scores moved by up to 0.02 (E7), as
  much as a learner seed moves them.
- Rule for pooling GPU and Mac results: record each world once, on one machine, and copy that recording
  to the other (`remote.sh push results/data/<recording>`, verified file by file); run at full float32
  with deterministic operations. (The hidden-wall tie that made the dog's surprise scores device-dependent
  was fixed on 2026-11-05.) Do not regenerate a data seed on another machine and call it the same data. Since 2026-11-05 the
  tooling enforces the first part (a GPU job refuses to record) and every result says how it was made
  (`provenance`).
- Repeatability: with full float32 and deterministic operations a whole GPU run repeats bit for bit
  (E7, `f32det-rerun`). At the GPU's defaults it does not (E5), though the differences are far inside the
  seed noise (E7, default against f32det).
- Scope: the current learner (hidden 128). E4 passed while E7 failed, so E4 alone is not enough: a new
  learner family needs an E7b-style full run on a shared recording before its GPU results are pooled.

### E6 on the GPU (2026-11-05)

`checks.sh bench` in `gpu-checks-4` (GPU load from other programs 1 to 2 percent), plus the allocator
comparison (`alloc_check.sh`). Each on-device setting is the median of three timings; table from
`experiments/gpu/bench_table.py`, saved in `results/hardware/bench_table.txt`.

| minds at once | hidden | batch | Mac (ms) | GPU default (ms) | GPU full float32 (ms) |
|---|---|---|---|---|---|
| 1 | 128 | 32 | 31.8 | 2.7 (12x) | 3.0 (11x) |
| 1 | 128 | 256 | 185 | 5.7 (33x) | 7.0 (26x) |
| 1 | 512 | 32 | 98 | 4.0 (24x) | 5.0 (20x) |
| 1 | 512 | 256 | 570 | 12.8 (45x) | 16.9 (34x) |
| 1 | 1024 | 256 | 1,542 | 25.1 (61x) | 38.2 (40x) |
| 8 | 128 | 32 | 171 | 28.1 (6x) | 14.3 (12x) |
| 16 | 128 | 32 | not run | 16.5 | 18.1 |
| 32 | 128 | 32 | 783 | 22.8 (34x) | 25.9 (30x) |
| 64 | 128 | 32 | not run | fails; 37.1 with `cuda_async` | fails |
| 128 | 128 | 32 | 9,852 | fails; 457 with `cuda_async` | fails |

- Larger learners gain most: the biggest core tested runs 61 times faster than on the Mac. The current
  learner (hidden 128, batch 32) is too small to fill the GPU and gains 12 times.
- Many minds at once: 32 minds run 34 times faster than the Mac, about 1,400 mind-updates per second; 64 (with
  `cuda_async`) reach 1,725. At 128 the step collapses (setup log).
- 8 minds at once is slower than 16 at the default precision, in every repeat and in two separate jobs,
  but not at full float32 (14.3 ms). The compiler picks a poor plan for that shape under TF32.
- Full float32 costs 10 percent at the current size and 50 percent at the largest.
- The same setting timed in separate jobs differs by up to 7 percent for one mind and 10 to 11 percent
  for 8 minds (three jobs), more than within a job. Treat differences under about 10 percent between jobs
  as noise. (This line first said "within 5 percent", which was wrong.)
- The default-precision column compares TF32 on the GPU with float32 on the Mac. The like-for-like
  comparison is the full-float32 column: 11 to 40 times. Mac cells are single timings, GPU cells medians
  of three.
- One full mind, end to end (human, lawful world): 183 s against 295 s on the Mac. Training 55 s (Mac
  131), nonlinear probes 12 s (39), the two controls 1 and 9 s (38 and 41). But the surprise tests took 16 s
  (Mac 0.7) and time outside the mind's stages doubled (77 s against 38 s; the GPU run recorded its
  1,280-episode world, 5 s, where the Mac's loaded an existing one). The guess was compile time on many
  small functions; the follow-up below confirms it.

### E6 follow-up: training loop, deterministic operations, compile cache (2026-11-05)

`speed_check.sh` as `speed-check-3` (02:33 to 02:51). Training loop as experiments run it
(`learner.train`, host batches), timed at 200 and 1,200 updates three times each, so compiling is
separated from the cost of an update:

| setting | ms per update | fixed cost per call (mostly compiling) |
|---|---|---|
| Mac, CPU (rerun under another job's load, see setup log) | 39.3 | 1.0 s |
| Mac, CPU (idle, old method, about 1 s of compiling subtracted) | about 31.7 | |
| GPU, full float32 | 10.6 | 0.9 s |
| GPU, default (TF32) | 13.4 | 0.8 s |
| GPU, default with deterministic operations | 11.5 | 0.8 s |
| GPU, default with `cuda_async` | 14.6 | 1.5 s |

- Like for like (float32), the training loop is about 3 times faster than on the idle Mac. The update
  itself takes about 3 ms of the 10.6 (E6), so building each batch on the CPU and copying it over is most
  of the loop. This is now measured, not assumed. Gathering batches on the GPU is the obvious next speedup.
- Full float32 is faster than TF32 in this loop, in two separate measurements (10.6 against 13.4 ms; 12.6
  against 15.1 by the old method). Not explained.
- `cuda_async` costs 9 percent per update and 0.6 s more compiling per call. The earlier "25 percent"
  included compiling. Keeping the default allocator still holds, on this smaller margin.

What deterministic operations cost (`pc-gpu-det`, update alone, median of three): 1 mind 2.91 against 2.67
ms (9 percent), largest learner 25.9 against 25.1 (3 percent), 32 minds 26.5 against 22.8 (16 percent),
and 8 minds 14.2 against 28.1: the deterministic plan avoids the slow one. These compare two jobs, and
differences between jobs run up to 7 to 11 percent, so the 3 and 9 percent are within noise; within one job
the training loop was faster with them (11.5 against 13.4 ms). Cheap enough for any reported work.

One full mind three times, the same job (human, lawful world, default precision):

| | total | training | surprise tests | MLP probes | outside the stages | test error |
|---|---|---|---|---|---|---|
| no compile cache | 186 s | 65 s | 15.7 s | 11.6 s | 73 s | 0.227238 |
| persistent cache, empty | 171 s | 64 s | 15.9 s | 11.6 s | 58 s | 0.227238 |
| persistent cache, full | 98 s | 54 s | 0.2 s | 1.4 s | 38 s | 0.227238 |
| Mac | 295 s | 131 s | 0.7 s | 38.7 s | 39 s | 0.227244 |

Compiling was nearly half of a mind's time on the GPU. With a full cache a mind takes 98 s, 3 times faster
than the Mac (at default precision). The cache took 20 MB (2,940 entries). Training time also fell, from
65 to 54 s, more than its 0.8 s of compiling explains; one run per setting, so not explained.

Does the cache change results? (`cache_and_reproduction_check.txt`, `cache_check.sh`.) Test error was
identical above, but 36 of the mind's 74 other numbers (probes, up to 0.003) differed between the no-cache
run and the cached ones, while the empty-cache and full-cache runs were identical to each other. A second
check, `cache-check` (04:00 to 04:10), ran the same mind at full float32 with deterministic operations:
no cache, cache empty and cache full gave identical results in all 74 numbers, and 150, 122 and 91 s. So
for reported work the cache is safe, and `job.sh` uses it for every job. At the default settings the
question is not settled: the second no-cache run there used a different recording (the PC's copy had been
replaced by the Mac's at 03:12 for E7b), a design mistake in that check, so its 61 differing numbers mix
two causes.

### E6 on the Mac (2026-11-04, Mac otherwise idle)

`experiments/gpu/bench.py mac-cpu`; table by `experiments/gpu/bench_table.py`.

| minds at once | core size (hidden) | batch | ms per update | minutes for 4,000 updates |
|---|---|---|---|---|
| 1 | 128 | 32 | 31.8 | 2.1 |
| 1 | 128 | 256 | 185 | 12 |
| 1 | 512 | 32 | 98 | 6.5 |
| 1 | 512 | 256 | 570 | 38 |
| 1 | 1024 | 256 | 1,542 | 103 |
| 8 | 128 | 32 | 171 | 11 (all 8) |
| 32 | 128 | 32 | 783 | 52 (all 32) |
| 128 | 128 | 32 | 9,852 | 657 (all 128) |

`learner.train` as experiments run it (host batches): 33.7 ms per update, so gathering batches on the
host costs about 6 percent. Training several minds at once helps the Mac only up to about 8 (47 mind
updates per second at 8, 41 at 32, 13 at 128, where memory runs short).

One full mind end to end (human, lawful world, v1 sizes): 295 s, of which training 131 s, nonlinear
probes 39 s, untrained-network control 38 s, raw-senses control 41 s, and 46 s for everything else:
7 s in the small timed stages (latents, linear probes, surprise tests) and about 39 s outside the stage
timers (loading data, probe targets, building the surprise-test suite, translation). Training is about
45 percent of a mind's time and probe fits (the probes and both controls) are 72 percent of the rest,
so a GPU can speed up both.
