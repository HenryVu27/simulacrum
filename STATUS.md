# STATUS

Last updated: 2026-11-04, night (GPU machine). Read this first. v2 details: `results/report_v2/REPORT.md`.
v1 details: `results/report_main/REPORT.md`. Preregistration and Amendments 1 to 4: `docs/preregistration.md`.
GPU machine (setup log, protocol, every check and result): `docs/gpu.md`.

## v2 in one paragraph

v1 found that no mind remembers hidden objects. v2 asked whether that changes when the world makes memory
pay. It does not, at this learner's size. Four memory worlds were built (`lookback`, `dark`, `occluders`,
`occlusion`). Calibration then showed why the learner had no reason to remember: memory can only pay at rare
moments (an object coming back into view, the lights returning), about 4 percent of steps. A remembering
mind also memorized its 1,024 training episodes. With 8,192 training episodes, the dark world (the v1 lawful
world plus random lights-out spells) narrowly passed the replacement manipulation check. On fresh data it
then fell short (memory value at object returns 0.093 and 0.057 against a bar of 0.10). All four
preregistered v2 hypotheses fail: no mind knows more about a hidden object than its raw senses give, and
none is surprised by a vanishing object. The world lever is spent. The next lever is the learner.

## v2 results

1. **Making memory pay is harder than it looks.** Memory value M is the share of a memoryless mind's
   held-out prediction error that a mind with memory removes. Lawful world: 0.019. Calibration ran in three
   stages, each preregistered before it ran (Amendments 2 to 4):
   - At 4,000 updates no world reached the bar of 0.10. Long walls (0.013) and look-away-and-back with fast
     turns (0.003) added nothing. The dark worlds went negative (-0.045, -0.068), yet at the moments they
     were built for, memory helped a lot: 22 to 30 percent less error on the first lit step, 19 to 23
     percent when an object returns.
   - At 16,000 updates M fell further (dark -0.279, occlusion -0.402).
   - Diagnosis, from prediction error only. (a) Memorization: the remembering mind's error rises 9 percent
     from training to held-out episodes (memoryless 4 percent). It can recognize an episode by its room and
     recall when its lights come back. (b) Hedging: on training episodes, its error late in a dark spell
     matches what an ideal predictor pays. (c) The bar was set too high: memory moments are about 4 percent of
     steps but carry 18 to 23 percent of the memoryless mind's error, so even perfect memory with perfect
     timing cuts overall error by about a fifth, and a bar of 0.10 asked for half of that. (Amendment 4 said
     "at most about 15 percent"; an independent recomputation corrected it, see the erratum.)
   - With 8,192 training episodes: dark M 0.022 (0.106 at object returns), occlusion 0.017 (0.094). Dark was
     chosen.
2. **The manipulation check failed on fresh data.** Confirmatory run: dark world, data seed 1, learner seeds 0
   and 1, all 14 minds. Overall M 0.016 and 0.013 (memory no longer hurts). M at object returns 0.093 and
   0.057, below the bar of 0.10 (bootstrap intervals over episodes: seed 0 0.081 to 0.105, a near miss; seed 1
   0.047 to 0.067, a clear miss). Dark was chosen because it scored 0.106, just over the bar, so some drop on
   fresh data was expected (a winner's curse). Every hypothesis below is labeled as tested under a weaker
   manipulation than preregistered.
3. **P1 to P4 fail.**
   - P1, permanence: for objects out of view 1 to 5 steps, the human mind scores 0.02, its raw senses 0.00,
     memoryless -0.03. At 6 to 20 steps: -0.02, -0.02, -0.04.
   - P2, passive_novest is not below human (6 to 20 steps: -0.01 versus -0.02).
   - P3, counting: human -0.06, memoryless -0.07.
   - P4, surprise: human permanence S 0.004 in both seeds, intervals include 0, the same as in v1.
4. **Exploratory: a brief trace in the first moments of darkness, probably not memory updated by
   self-motion.** For objects hidden 1 to 5 steps and in a dark spell, the human mind scores 0.014 and 0.088
   (seeds 0 and 1) against its raw senses at -0.032, a gap of 0.05 and 0.12. It is gone by 6 to 20 steps. The
   minds with no efference copy do as well or better (passive 0.046 and 0.063, passive_novest 0.059 and
   0.084), so the likeliest reading is that the last lit scene persists briefly in the hidden state, not that
   the mind tracks objects through its own motion. (Every recurrent mind with eyes also beats memoryless here,
   but memoryless sits below its own raw senses, so that comparison flatters them.)
5. **Exploratory: what counts as hidden depends on the body.** Bat and chimera locate objects the eyes cannot
   see (about 0.17 to 0.19 at 1 to 5 steps, in v1 and v2 alike), because echolocation covers 360 degrees and
   works in the dark. That is sensing, not memory.
6. **Exploratory: darkness takes self-location from eyes, not from a compass.** Self-location drops from 0.49
   to 0.29 for human and from 0.54 to 0.35 for vision_only. Chimera (smell, echolocation, compass) holds 0.68
   in both worlds. Without an efference copy or body sense, self-motion knowledge falls from 0.53 in v1 to
   0.30 in the dark (human: 0.84), because the dark also removes optic flow.
7. **H5 replicates in a second world.** Translation runs from richer minds to poorer ones: human to blind 0.74
   versus 0.15, human to vision_only 0.96 versus 0.85, dog to smell_only 0.84 versus 0.18.

## v2 engineering

- New world rules (`src/umwelt/world.py`): look-away-and-back policy, turn gain, lights-out spells (applied at
  the eye in `presets.py`, so echolocation, hearing, smell and touch are unaffected), wall length, friction.
  Defaults reproduce v1 bit for bit (95 recorded arrays and the surprise-test movies checked).
- Memory-light pipeline: recordings stream to disk, observation streams are memory-mapped, training batches
  are gathered on the host. Bit-identical to the old learner after 60 updates. 8,448-episode worlds now fit
  beside other jobs on this Mac.
- `experiments/calibrate.py` (memory value per world), `experiments/overfit_check.py` (training versus
  held-out error), `experiments/world_stats.py` (how often objects leave and return), `experiments/sweep_v2.sh`,
  `umwelt report-v2`. Tests: 9 passing.

## GPU machine (2026-11-04, night)

umwelt now runs on Henry's Windows PC with an RTX 5080, from this Mac, for the learner work in next step 1.
JAX publishes GPU builds for Linux only, so it runs inside WSL2 (Ubuntu 26.04, as the distribution's root
user, no Linux account needed). `experiments/gpu/remote.sh` pushes the code (verified on the PC), queues
jobs that wait for a shared GPU lock (`C:\gpu-lock`, also used by the other project), and pulls results
back. Henry can pause umwelt jobs while he uses the PC (`mkdir C:\gpu-pause`). Everything is in
`docs/gpu.md`: the setup log, a protocol written before any run (E1 to E6), and the results.

What is known so far (all in `docs/gpu.md`):
- The PC's CPU gives the same project numbers as the Mac (stage 3 memory value within 0.00003).
- On the GPU at full float32, training matches the Mac to about five digits. At the GPU's default TF32
  precision it drifts about 1 percent in 300 updates, and default GPU runs are not repeatable (with
  deterministic operations they are, bit for bit).
- Bug found and fixed: at TF32 the ray caster flipped object visibility; the world now always simulates at
  full precision (`src/umwelt/data.py`; Mac output byte-identical).
- Speed: one update of the current learner 2.6 ms against 31.8 ms on the Mac; the largest learner tested
  25 ms against 1,542 ms. The training loop as experiments run it gains only 2.2 times, because batches
  are built on the CPU: building them on the GPU is the obvious next speedup. WSL limits one compiled step to
  under 4 GiB of GPU memory with the default allocator (32 minds at once; 64 with `cuda_async`).
- Pending (job `gpu-checks-4`, queued overnight): stage 3 on the GPU at both precisions, which decides
  whether GPU results can be pooled with the Mac's, and the full speed table.

## Next steps, in order

1. Change the learner, not the world. The next-frame objective gives memory little gradient, because memory
   moments are a few percent of the loss. Candidates, cheapest first: weight the loss toward moments when
   the senses change in ways the present cannot predict; predict further ahead across dark spells; a larger
   recurrent core; a learner that attends over past frames. The last two want a GPU: the RTX 5080 machine is
   set up for that (see the GPU section above and `docs/gpu.md`).
2. Test the smell-only mind's unnamed features against scent-field variables (from v1, still open).
3. Nonlinear translation maps, to separate "different frame" from "different content" for the oracle.
4. Aporia Atlas: join aporias that share claims and tag umwelt results to premises.

---

# v1 (2026-11-04, overnight)

## What exists

- A 2D physics world in JAX with four rule sets: lawful, impermanent (objects jump), ghost (nothing is
  solid) and nonlocal (pushing one object moves another).
- An independent MuJoCo build of the same world, used to audit every lawful-world result.
- 14 minds defined by senses, efference copy (whether a mind knows its own actions) and memory. All
  share one recurrent learner that predicts its next sensations.
- A probe battery with untrained, raw-sense and oracle controls, matched violation-of-expectation (VoE)
  quadruplets, translation maps between minds, a report generator, a cloud path, concept import with
  sparse autoencoders, and a minimal Aporia Atlas.
- The preregistration (`docs/preregistration.md`) was committed before the main sweep, and so was
  Amendment 1. Every deviation since is listed below.

## Main results

1. **The test battery works.**
   - The oracle scores 0.949 to 0.969 on object location in every world and engine.
   - Seeds reproduce: across 168 lawful-world scores, the mean difference is 0.006 and the largest
     is 0.04.
   - Flatland and MuJoCo agree: Pearson r 0.997 across 168 mind-concept scores in the lawful world,
     mean absolute difference 0.017, largest 0.12. The pooled r is helped by large differences between
     concepts. Within single concepts it is lowest for cause (0.91).
   - An independent verifier recomputed these numbers from the raw result files.
2. **At this scale every mind is reactive.** No mind remembers objects beyond what its senses give at
   the moment:
   - A/B test: prediction horizons of 5, 15 and 30 steps and 30 percent observational dropout all
     failed to add memory.
   - A JEPA-style latent-prediction learner and four times the training (16,000 updates) also failed.
   - Permanence decay curves for all 14 minds: trained minds track their raw senses within about
     0.03 at every delay after an object leaves view.

   The preregistered memory hypotheses (H1 permanence, H4 counting, H6 surprise) therefore fail, for
   that reason. This is the main finding to build on: object permanence did not emerge from
   self-supervised prediction in this world. The world or the objective has to make remembering pay.
3. **Translation runs from richer minds to poorer ones (H5, passed).** Human to blind 0.74 versus blind
   to human 0.36. Human to vision-only 0.89 versus 0.77. Dog to smell-only 0.82 versus 0.20. Minds
   cluster by their main sense. Smell-only is the most alien mind: smell-rich minds can read it, but
   it can read almost nobody.
4. **The view from nowhere does not translate.** The oracle's world-centered state has essentially
   zero linear translation to or from any embodied mind (R^2 about 0, CKA about 0.04). This is
   exploratory, and linear maps cannot rotate between frames.
5. **Held and Hein, half confirmed (H3, failed as written).** Without an efference copy and body
   sense, self-motion knowledge collapses (0.88 to 0.53) while object knowledge is unchanged.
   Self-location does not drop, because vision reads location off the room's landmarks.
6. **Concept import, first pass.** Vision-based minds have no unnamed features. The smell-only mind
   has 33 slow-changing features that neither its current senses nor any named world variable
   explains. The likely source is the diffusing scent field, which our vocabulary does not name. Not
   yet tested.
7. **Mind versus world (H7, failed as written, mostly right).** Two-way ANOVA (analysis of variance)
   over 13 minds and 4 worlds. The design is unbalanced (lawful has two seeds, the others one), and the
   report uses sample-size-weighted (Type I/II) sums of squares:
   - Causal attribution is the one concept the world decides: world share 0.74, mind share 0.23.
     Other ways of handling the imbalance give world 0.74 to 0.79 and mind 0.18 to 0.23. The world
     share comes mostly from the nonlocal world, where pushes act at a distance.
   - Seeing, counting, self-motion, self-location and spatial relations are set by the mind
     (mind share 0.86 to 1.00).
   - H7 fails only on hidden objects (mind share 0.95). That follows from result 2: with no memory,
     what a mind knows about hidden things is just what its senses give.
   - Elapsed time splits evenly (0.47 mind, 0.48 world).
   - The allocentric hidden-object split (0.50 and 0.46) turns into mind 0.88 and world 0.05 once
     scored as skill above baseline, so its world share was the scale artifact.

## Deviations from the preregistration

- Amendment 1 (before the sweep): seeds trimmed, controls limited to the lawful worlds (in practice
  seed 0 only), allocentric probes added. The same commit changed learner, probe and runner code;
  the amendment text says what and why.
- Skipped for time: impermanent_mj and ghost_mj audit worlds. The lawful_mj audit ran in full.
- Added after seeing data, labeled as such in the report:
  - skill scores above a no-information baseline, which fix a scale artifact in the seen and hidden
    object subsets;
  - permanence decay curves;
  - concept import;
  - memory diagnostics.
- A guard was added for concepts that do not exist in a world (nothing is ever pushed in the ghost
  world, so cause is "not applicable" there).

## Machine notes

Two jobs from another project ran on this Mac overnight (`b1_probe.py generate --base olmo`, about
10 GB, and `another project's job`). They were left alone. They slowed the sweep through memory
pressure. `caffeinate` held the Mac awake until the sweep scripts exited.

## v1 next steps (superseded above)

1. Make memory pay. Add moving objects that pass behind occluders and reappear, a narrower field of
   view, and look-away-and-back behavior. Then rerun the human and memoryless contrast first. This
   unblocks H1, H4 and H6.
2. Test the smell-only mind's unnamed features against scent-field variables (total scent per kind,
   gradients, trail age).
3. Cloud: more seeds, longer training, the full factorial sweep over sense settings (`cloud/README.md`).
   This needs an account only Henry can create.
4. Nonlinear translation maps, to separate "different frame" from "different content" for the oracle.
5. Aporia Atlas: join aporias that share claims so unoccupied joint positions can appear, and tag
   each umwelt result to the premise it bears on.
