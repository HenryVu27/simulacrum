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
  with 14.7 GB free. 3.99 GiB works, 4.00 GiB fails, with JAX's default allocator and with the plain
  one (`XLA_PYTHON_CLIENT_ALLOCATOR=platform`). Total memory is not capped there: ten 1 GiB arrays held
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
  of 4, 6 and 8.2 GiB. Compared with the default in one job (`experiments/gpu/alloc_check.sh`): training
  with deterministic operations gives byte-identical weights (hash `69122ab38190b194`); on-device
  updates for 1 to 32 minds take the same time (within 3 percent, three repeats each); 64 minds run
  (37 ms per update) where the default fails; but the host-batch training loop that experiments use is
  25 percent slower (19.2 against 15.3 ms per update; the pause runs agree, 41 s against 37 s). Decision:
  jobs keep the default allocator; a job that needs a step of 4 GiB or more sets `cuda_async` itself:
  `remote.sh run NAME env XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async CMD...`.
- 2026-11-04. 128 minds at once under `cuda_async` run but take 457 ms per update (three repeats, 423 to
  466 ms), 12 times the 64-mind step for twice the work. Not explained; one guess, untested, is that
  Windows (WDDM) starts paging GPU memory when one process holds that much. In practice: at most 64
  minds per compiled step.
- 2026-11-04. Finding: on the GPU the matmul precision changed the recorded world, not only learning.
  The ray caster tests which object each ray hits with `u @ f.T` (`src/umwelt/senses.py:52`); at TF32
  its inputs keep about three significant digits, so hits at an object's edge flip. Against the Mac,
  object visibility differed at 130 of 98,304 entries (47 of 64 episodes) at the default precision, 4
  at full float32, 2 on the PC's CPU; vision values differed in 3.0, 0.6 and 0.3 percent of entries.
  Physics, actions and every other sense were byte-identical between the two GPU precisions. Fix:
  `data.chunks` now rolls the world out under `jax.default_matmul_precision("highest")`, so a recording
  no longer depends on the device's default. On the Mac the output is byte-identical with the fix (world
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
- 2026-11-05 00:05. `gpu-checks-4` ran from about 23:55 and finished at 00:05, earlier than planned (log
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

## How to run on the GPU

Status: working against the real PC since 2026-11-04 (setup, push with verification, jobs, pause,
pull; see the setup log). Both scripts also have tests on the Mac against stand-ins
(`experiments/gpu/test_job.sh`, 22 cases; `experiments/gpu/test_remote.sh`, 29 cases, with a fake ssh
that plays the PC).

    experiments/gpu/remote.sh setup                       # once: copy code, install uv and CUDA JAX, run tests
    experiments/gpu/remote.sh run NAME CMD...             # background job; waits for a free GPU
    experiments/gpu/remote.sh status NAME                 # GPU use, lock, last log lines, exit code
    experiments/gpu/remote.sh pull results/runs/v3        # copy results back to the Mac
    experiments/gpu/remote.sh pause                       # freeze umwelt GPU work; resume undoes it
    experiments/gpu/remote.sh stop NAME                   # end a queued or running job

On the PC itself, `mkdir C:\gpu-pause` pauses and `rmdir C:\gpu-pause` resumes (a file works too).

For example, one learner seed of the dark-world sweep:

    experiments/gpu/remote.sh run dark-s2 uv run umwelt run --world dark --seed 2 --tag v3 --data-seed 1 --n-train 8192

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

Limits inside WSL: a compiled step whose working memory reaches 4 GiB fails with the default allocator
(for the current learner, 64 minds at once); add `env XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` to such a
job, which slows the ordinary training loop by a quarter. Keep to 64 minds per compiled step.

Code goes to the PC with `push`: every tracked or new file outside `results/` (deleted files skipped,
no macOS metadata), plus `.umwelt_push`, the manifest a job checks against. The PC rehashes what
arrived; a mismatch fails the push. A push that would change
the code while an umwelt job is queued or running is refused. Results stay on the PC until pulled;
`pull` never overwrites a local file that differs unless given `--overwrite` (it reports the file and
leaves the PC's version in a temporary folder). Recorded data is not pushed: a run regenerates it
there, which gives nearly but not exactly the same episodes (E1). To reuse the Mac's exact episodes,
push the recording too: `remote.sh push results/data/<recording>`.

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
  CPU: 3 and 1). Physics and every sense but vision were byte-identical between the two GPU
  precisions. Vision and the object-visibility labels were not: at the default precision (TF32) the
  ray caster flipped visibility at 130 of 98,304 entries, against 4 at full float32 and 2 on the PC's
  CPU. That was fixed (setup log); `checks.sh precfix` checks the fix on the GPU.
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
| largest difference from the Mac | | 0.00004 | 0.00011 | 0.00006 |
| test error, human (Mac 0.21917) | | 0.21917 | 0.21917 | 0.21917 |
| test error, memoryless (Mac 0.22405) | | 0.22406 | 0.22405 | 0.22404 |

Two comparisons isolate one cause each:
- Same episodes in, GPU against the PC's CPU: largest difference in M 0.00008. This is GPU arithmetic
  alone.
- GPU default against GPU full float32, on byte-identical recorded worlds: 0.00013. This is TF32 in the
  learner alone. The one-percent drift in the loss at 300 updates (E3) does not reach the result.

Every difference is 70 to 220 times under the bar (0.009, a quarter of the learner-seed spread of M at
object returns), and the largest difference in overall M (0.00005) is 14 times under a quarter of overall
M's own seed spread (0.0007). Per-step errors correlate at 1.0000 with the Mac's in every pass. PASS: GPU
results can be compared with and pooled with the Mac's, at either precision.

Speed on the way: recording the 8,448 episodes took 14 s on the GPU (41 s on the PC's CPU under Windows);
training took 68 s for human and 22 s for memoryless (idle Mac 134 s and 58 s).

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
- The same settings timed in two jobs an hour apart agree within 5 percent.
- One full mind, end to end (human, lawful world): 183 s against 295 s on the Mac. Training 55 s (Mac
  131), nonlinear probes 12 s (39), the two controls 1 and 9 s (38 and 41). But the surprise tests took 16 s
  (Mac 0.7) and time outside the mind's stages doubled (77 s against 38 s, including 5 s to record the
  world). The working guess is compile time on many small functions; `speed-check-2` tests it with a
  persistent compile cache, and also times the training loop properly and the cost of deterministic
  operations.

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
