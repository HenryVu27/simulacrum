# Independent check of the Stage 0 rerun and rescoring (2026-11-06)

Scripts of the outside-judge agent that checked Amendment 6's Stage 0 rerun (`experiments/stage0_rerun.py`) and the
rescoring (`experiments/stage0_rescore.py`), copied unchanged. They ran on the Mac at low priority from a scratch
folder; their saved predictions (39 MB of .npz) are left out, and their printed outputs are summarized here.

- c1: the bootstrap-from-per-episode-sums code equals probes._r2_cols (differences below 3e-15).
- c2 to c5: ridge at 1,024 episodes recomputed on the Mac matches the GPU within 1.2e-8; criterion a's bootstrap,
  criteria b to d, and the 1,024-episode MLP (which fails criterion a) reproduce exactly.
- c6, c6b: the lawful mind's gate (0.87066 against 0.87808, ratio 0.9915) reproduces; about 82 percent of the error
  at object-return steps comes from first-lit steps, which no object channel can predict, so the gate was close to
  unreachable; the lawful mind ignores its channel (another episode's channel moves its error from 0.8707 to 0.8733)
  and is 6 percent worse than its twin at lit re-entries.
- c7: the lawful channel in the movies is as specified (I1 gets P1's, I2 gets P2's; never the impossible state);
  permanence S +0.000844 [-0.000732, +0.002402] reproduces.
- c8, c11: the rescore reproduces for human, memoryless, hearing_only and the oracle (lawful seed 0, dark seed 0),
  and H1 and P1 by their preregistered rules (both FAIL). A memory-cut control (the same network with its hidden
  state reset every step) separates memory from encoding: for the dark-world human at 1 to 5 steps, memory alone
  adds +0.086 [0.067, 0.105] in the dark and +0.001 when lit; about 78 percent of its object_hidden gain over raw
  senses is encoding.
- c9, c12: the ridge probe reads memory only when it is stored as linear egocentric coordinates. A fixture with
  "last seen, updated"'s information stored in room coordinates scores -0.001 under ridge and 0.941 under the MLP;
  the stored oracle scores -0.034 at 1 to 5 steps under ridge and 0.951 under the MLP.
- c10: the rescore gives held-out episodes the fitting episodes' sensor-noise draw (effect at most 0.0004).
