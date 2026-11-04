# STATUS

Last updated: 2026-11-04, morning after the overnight build. Read this first, then
`results/report_main/REPORT.md`.

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

1. **The test battery works.** The oracle scores 0.95 to 0.98 on object location in every world and
   engine. Seeds reproduce to within about 0.01. Flatland and MuJoCo agree: Pearson r above 0.99
   across 144 mind-concept scores in the lawful world, mean absolute difference about 0.02.
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
   over 13 minds and 4 worlds:
   - Causal attribution is the one concept the world decides: world share 0.74, mind share 0.23.
   - Seeing, counting, self-motion, self-location and spatial relations are set by the mind
     (mind share 0.86 to 1.00).
   - H7 fails only on hidden objects (mind share 0.95). That follows from result 2: with no memory,
     what a mind knows about hidden things is just what its senses give.
   - Elapsed time splits evenly (0.47 mind, 0.48 world).
   - The allocentric hidden-object split (0.50 and 0.46) turns into mind 0.88 and world 0.05 once
     scored as skill above baseline, so its world share was the scale artifact.

## Deviations from the preregistration

- Amendment 1 (before the sweep): seeds trimmed, controls limited to lawful worlds, allocentric probes added.
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

## Next steps, in order

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
