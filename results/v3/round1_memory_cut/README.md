# Memory-cut control, both dark-world seeds (round 1 close, 2026-11-06)

The Stage 0 rerun's independent check (`results/v3/stage0_rerun/independent_check/c8_rescore_sample.py`) compared each
trained network with the same network with its hidden state reset before every step, which keeps its learned encoding
of the present and removes the past. Only a summary of one seed was kept. `experiments/round1_memory_cut.sh` reran that
script, unchanged except for the repository path, on the stored v2 dark-world human, memoryless, passive and passive_novest minds
(learner seeds 0 and 1; the job `round1-memory-cut-2` added the passive minds and reproduced the first job's lines
exactly), on the GPU PC at full float32 with deterministic operations; the job logs, which record the code
fingerprint, are `results/remote/round1-memory-cut.log` and `round1-memory-cut-2.log` (job logs are not in git).

Each file prints the fixed probe's scores (ridge, 1,024 fitting episodes, 256 held-out episodes) for trained, raw-sense
and memory-cut features, with stored values from `results/v3/stage0_rescore` in parentheses (all match), then paired
bootstrap differences (1,000 resamples of held-out episodes).

| human, trained minus memory cut | seed 0 | seed 1 |
|---|---|---|
| 1 to 5 steps, dark | +0.086 [0.067, 0.105] | +0.112 [0.088, 0.136] |
| 1 to 5 steps, lit | +0.001 [-0.008, 0.010] | -0.006 [-0.015, 0.002] |
| 6 to 20 steps, dark | +0.008 [-0.005, 0.022] | +0.016 [0.002, 0.029] |
| 6 to 20 steps, lit | -0.002 [-0.008, 0.005] | -0.007 [-0.013, -0.000] |

Minds that never know their own actions, trained minus memory cut, 1 to 5 steps in the dark:

| mind | seed 0 | seed 1 |
|---|---|---|
| passive (human senses, no efference copy) | +0.098 [0.079, 0.117] | +0.107 [0.084, 0.129] |
| passive_novest (no body sense either) | +0.053 [0.037, 0.069] | +0.077 [0.058, 0.095] |

Lit, and at 6 to 20 steps in the dark, both are within 0.015 of zero. The memoryless mind's trained minus memory cut
is exactly 0 everywhere, as it must be. Seed 0 reproduces the
independent check's summary (+0.086 [0.067, 0.105] dark, +0.001 lit).
