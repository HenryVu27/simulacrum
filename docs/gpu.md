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

## How to run on the GPU

Status: the scripts below are written and pass shellcheck. They have not run against WSL yet,
because WSL is not installed. The detach mechanism was tested on its own: a process started through
`Win32_Process.Create` over SSH kept running after the connection closed.

    experiments/gpu/remote.sh setup                       # once: copy code, install uv and CUDA JAX, run tests
    experiments/gpu/remote.sh run NAME CMD...             # background job; waits for a free GPU
    experiments/gpu/remote.sh status NAME                 # GPU use, lock, last log lines, exit code
    experiments/gpu/remote.sh pull results/runs/v3        # copy results back to the Mac

For example, one learner seed of the dark-world sweep:

    experiments/gpu/remote.sh run dark-s2 uv run --frozen umwelt run --world dark --seed 2 --tag v3 --data-seed 1 --n-train 8192

Sharing the GPU with the other project: a job starts only when the folder `C:\gpu-lock` does not
exist and less than 6 GB of GPU memory is in use. It then creates the folder (with an `owner` file)
and removes it when it ends. `remote.sh unlock` clears a lock left by a job that was killed outright.
JAX jobs run with `XLA_PYTHON_CLIENT_PREALLOCATE=false`, so they take GPU memory as they need it.

Code goes to the PC with `push` (every tracked or new file outside `results/`). Results stay on the
PC until pulled. Recorded data is not pushed: a run regenerates it there, which gives nearly but not
exactly the same episodes (E1). To reuse the Mac's exact episodes, push the recording too:
`remote.sh push results/data/<recording>`.

All hardware checks in one job, to rerun after a driver, CUDA or JAX upgrade: `experiments/gpu/checks.sh`
(its header has the commands).

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
  identical: light timing (`gt_light`), which objects are active, and their kinds. Actions differ from
  the first step (61 of 64 episodes), by at most 0.01. Agent heading differs by at most 0.005 radians and
  agent position by at most 0.007 (the arena is 10 units across). Most of these differences stay
  small. Small differences grow into a different event in only a few episodes: object positions differ
  by more than 0.0001 in 3 of 64 episodes, by more than 0.1 in 1 of 64 (from step 200 on), and
  visibility of an object differs in 1 episode. The likely source is the CPU math libraries (sine,
  cosine, square root, fused multiply-add) on ARM against x86, fed into a world with collisions.
- E2 Senses, on identical episodes: 5 of 14 presets identical. The other 9 differ by at most 1.2e-7
  (one or two float32 rounding steps, relative size 1e-9). These are the presets with sensor noise;
  NumPy's normal sampler goes through the platform's `log` and `exp`.
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
spread). Per step, errors differ by a median of 2 parts in 10,000; 8 or 9 of the 256 test episodes have
a step that differs by more than 0.01, none by more than 0.1. PASS: results from the PC's CPU can be
compared with and pooled with the Mac's.

The PC's CPU was slower than the Mac here: human 331 s and memoryless 254 s, against 134 s and 58 s.
That run was at idle priority beside the other project GPU job, so this is not a clean timing; the 300-update
E3 run, alone on the PC, was also slower (28 s against 11 s, compile included).

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
probes 39 s, untrained-network control 38 s, raw-senses control 41 s, everything else 9 s. Training is
about 45 percent of a mind's time and the JAX probe fits most of the rest, so a GPU can speed up both.
