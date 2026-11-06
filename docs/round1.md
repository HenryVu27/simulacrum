# Round 1 (2026-11-04 to 2026-11-06): what it found, and what round 2 should do

Round 1 is closed. It ran on one Mac and one RTX 5080 and should be read as an exploratory first round.

**Bottom line.** No mind in round 1 developed object permanence from predicting its next sensations, and the one
memory effect that survives every check is a short trace in the first few steps of darkness. But round 1's larger
findings are about its instruments. It found that the probes lost information as their
inputs widened and, once fixed, still read only one format of memory; that several headline results came from the
inputs rather than from learning; and that the surprise test has never been checked on a mind known to remember. Round 2 should build and validate the whole instrument panel
before training any new learner, and only then scale up.

## The question

Which concepts does the world force on any mind, and which does a body add? Fourteen minds that differ in their
senses and body (human, dog, blind, smell-only, bat, a passive observer, a memoryless mind, an oracle that sees the
true state, and others) share one learner, a recurrent network (a GRU, gated recurrent unit, with 128 hidden units;
the memoryless mind has a feedforward core in its place) trained to predict its next sensations in a 2D world. Probes read concepts out of each mind's hidden state, and
violation-of-expectation (VoE) movies measure surprise at impossible events. Worlds with broken physics (objects
that jump, pass through walls, or push at a distance) were meant to split each concept into what the mind adds and
what the world forces.

Three stages, each preregistered before it ran (`docs/preregistration.md`):
- v1: 14 minds in four worlds, with a MuJoCo build of the lawful world as an audit (it shares the sense formulas, policy,
  learner and probes with the main build).
- v2: four worlds built to make memory pay (look away and back, darkness, occluders), with a confirmatory run in
  the dark world.
- v3: fix the instrument, then change the learner. Only its Stage 0 (the instrument) ran, twice: the first run
  stopped by its preregistered rule, and Amendment 6 restated the probe criterion after seeing that data, then
  reran on a fresh recording.

## What holds

Each item survived the independent audit and the later checks; the evidence is in the file or folder named.

1. **The record reproduces.** The audit regenerated both reports from a scratch copy of the results and recomputed
   a sample of probe scores, test errors and surprise scores exactly from the stored networks. On shared recordings
   the GPU matches the Mac to about the sixth decimal for nearly every score and repeats a whole run bit for bit
   (`results/checks/2026-11-05_audit`, `results/hardware/compare_e7b_noise200.log`, `docs/gpu.md`). The git history
   shows every amendment committed before the run it governs, and the result files keep the values each correction
   replaced.
2. **No mind with eyes expects a hidden object to stay.** Across the stored v1 and v2 runs, scored over 200
   sensor-noise draws, every vision mind's permanence surprise is within 0.0007 of zero. For the human and
   vision_only, a mind expecting the object back would score about 0.01 from the first mismatched frame alone (the
   dog about 0.002), against intervals of about plus or minus 0.0003 (0.0007 for the dog;
   `results/checks/2026-11-05_audit`, `04_voe_power`, computed for four minds). The oracle is surprised in every run because it sees the hidden
   event, not because it remembers. A few small intervals exclude zero and are unexplained: touch_only in 4 of 8
   runs (about 0.002) and passive in 3 (under 0.0004). The power figure is a calculation, not a positive control:
   no mind known to remember has been run through the test (see below).
3. **No mind shows a learned solidity expectation.** Some solidity scores differ from zero, but minds trained in
   the ghost world, where walls stop nothing, score within 0.008 of the same minds trained in the lawful world (`results/checks/2026-11-05_battery`).
4. **Memory is hard to make pay with this learner.** Moments where memory can help (an object coming back into
   view, the lights returning) are about 4 percent of steps (`results/checks/2026-11-05_audit`, `11_error_shares`).
   With 1,024 training episodes and 4,000 updates, memory cut prediction error at those moments by 19 to 30
   percent in the dark and occlusion worlds, but the
   remembering mind memorized its episodes and its overall error rose. With 8,192 episodes the cut was 9 to 14
   percent, and on fresh data the dark world's memory value at object returns fell short of its bar (0.093 and 0.057
   against 0.10; `results/report_v2/REPORT.md`).
5. **A short memory trace in darkness.** In the dark world, for objects out of view 1 to 5 steps during a dark
   spell, the human mind's network knows more than the same network with its memory cut (hidden state reset every
   step): +0.086 (95 percent interval 0.067 to 0.105) and +0.112 (0.088 to 0.136) for learner seeds 0 and 1. When the
   lights are on, memory adds nothing (+0.001 and -0.006). By 6 to 20 steps the trace is barely there (+0.008 and
   +0.016, the first interval including zero). The memoryless mind, cut the same way, changes by exactly 0, as it
   should (`results/v3/round1_memory_cut`). The same control on minds that never know their own actions: passive
   (human senses but no efference copy, that is, no copy of its own motor commands) carries the same trace (+0.098 and +0.107),
   and passive_novest (no body sense either) a smaller one (+0.053 and +0.077). So most of the trace needs no knowledge
   of one's own movements, which fits the last lit scene lingering in the hidden state. Whether the smaller trace
   without body sense means some updating by self-motion was not tested (the difference between minds has no
   interval).
6. **What counts as hidden depends on the body.** The bat and chimera locate objects the eyes cannot see, because
   echolocation covers 360 degrees and works in the dark. That is sensing, not memory. Darkness takes self-location
   from minds that rely on their eyes (human 0.49 to 0.29) but not from one that has a compass (chimera 0.68 in
   both; `results/report_main/REPORT.md`, `results/report_v2/REPORT.md`).

## Claims withdrawn or narrowed

These were reported earlier and do not stand as first stated. Most are marked "(audit)" in `STATUS.md`; the
surprise-score corrections are in `docs/preregistration.md`.

- **"The test battery works."** Overstated: nothing in v1 was a positive control for memory, and the MuJoCo audit
  shares everything with the main build except the physics engine.
- **"Every mind is reactive", probe side.** The v1 and v2 probes read memory held in a compact form (an engineered
  "last seen, updated" state scores 0.96 at 1 to 5 steps, the oracle 0.95), but they lost information as their
  inputs widened: given exact copies of the frames 2 and 4 steps back, the pooled hidden-object score went down and
  the 1 to 5 step score rose only from -0.006 to 0.056 (0.125 with four times the probe data), so memory kept as
  copies of past input registered weakly. The decay curves' "within about 0.03 of raw senses" also rested on raw-sense
  curves for only 5 of the 14 minds (smell_only departs from its own by 0.10). Rescored with the probe Stage 0 fixed (ridge
  regression, 1,024 fitting episodes), H1 and P1 still fail (`results/v3/stage0_rescore/verdicts.json`). But ridge
  reads memory only when it is stored as linear egocentric coordinates: for objects out of view 1 to 5 steps it scores
  the oracle, which knows every position, at -0.034, where a small neural-network probe scores 0.95. The failures are about linear egocentric
  memory only; a mind that stored memory any other way would have been scored as having none.
- **Translation asymmetry (H5).** Richer minds translate into poorer ones better than the reverse, but untrained
  networks show the same asymmetry. It reflects one mind's inputs containing another's, not learned concepts.
- **Mind versus world shares (H7).** The variance shares depend on which minds and worlds were sampled. With only
  the five minds that have human eyes, the world decides hidden objects almost entirely. They cannot say what the
  world forces.
- **Causal attribution.** The cause probe mostly reads how near an object is (distance alone scores 0.588 in the
  lawful world, the human 0.52), and changing 6 labels among 7,873 moved a raw-senses control's score by 0.038, about a learner
  seed's worth.
- **Held and Hein.** The drop in self-motion knowledge without an efference copy follows the inputs: a passive mind with body sense but no efference copy scores 0.91 against the
  human's 0.88.
- **Learning in general.** The better of the raw-senses and untrained-network controls explains 98 percent of the
  variance in trained scores. Most of what the battery measured was the input, not what was learned.
- **Earlier surprise scores.** Three corrections to the surprise movies and their scoring withdrew smell_only's
  apparent permanence surprise (up to +0.045, an artifact of one shared noise draw), the oracle's scores under the
  old movies (they depended on how a missing wall was encoded), and P4's 0.004. No preregistered verdict changed.
- **Process.** Two deviations went unlogged until the audit: probes were fit on 1,024 training episodes, not the 384
  preregistered, and V2 counted more concepts than the nine stated. Amendment 4's bound on what perfect memory could
  save ("at most about 15 percent") was corrected to about a fifth. Amendment 6 restated Stage 0's probe criterion
  after seeing the first run's data; the rerun used a fresh recording.

## What round 1 never tested

- v3 Stages 1 to 3: changing the learner (residual-weighted loss, far horizons, a larger GRU, causal attention), the
  confirmatory hypotheses Q0 to Q5, and the world question R1 and R2. No learner was trained under v3.
- The surprise test on a mind with memory. Stage 0's two attempts failed for reasons unrelated to the test (a
  tracking mind that did not track; a gate dominated by the first lit step after darkness).
- Smell-only's 33 unnamed slow features against scent-field variables.
- Nonlinear translation maps between frames (needed for the oracle).
- Aporia Atlas: joining aporias that share claims.

## Round 2 brief, for when larger GPU access is available

Round 1's main lesson is that instruments were validated one gap at a time, after the fact, and each fix exposed the
next gap. Round 2 should reverse the order: design and validate every instrument at once, have it reviewed from
outside, then spend the compute.

**1. Outside review before any run.** Four reviewers working in parallel, each doing their field's actual checks:
- an ML evaluation expert (probe theory, control tasks, selectivity);
- a developmental psychologist (how infant VoE studies establish permanence, and what our movies would need);
- a statistician (power, seeds, multiple comparisons);
- an adversarial reviewer (can this design answer the question at all?).

Add a literature scan: IntPhys, PLATO (Piloto et al. 2022), and current video world models. Their findings go into
one amendment.

**2. The full measurement suite, written as one amendment before any learner runs.**
- Positive controls for every memory format: linear egocentric (a frame buffer), room coordinates, a nonlinear
  embedding of the last-seen state, the oracle, and a learner with an auxiliary tracking loss that is checked to
  actually track. The probe is chosen as the one that reads all of them without losing information on the present
  moment (round 1: ridge reads the first, a small neural-network probe reads the rest but loses information as
  inputs widen; more probe data or a probe family may be needed).
- Negative controls: raw senses, an untrained network, shuffled-label probes, and a memory-cut copy of every trained
  network. "Memory adds" is measured against the memory-cut copy, not against raw senses.
- A surprise-test control whose permanence is guaranteed: a hand-built physics predictor that keeps hidden objects,
  and its memoryless version, with a gate that counts only lit re-entries.
- Cause rescored within distance strata, or replaced by a target that does not reduce to proximity.

**3. Then the learner, at scale.** Amendment 5's screen (L0 to L4) plus current sequence models, with at least five
learner seeds (round 1's two dark-world seeds gave memory values of 0.093 and 0.057 at object returns), larger recordings (32,768 episodes or more,
since the remembering mind memorized 1,024), and longer training. Then the confirmatory run on a fresh recording,
and, only if memory appears, the world question with within-mind contrasts instead of variance shares.

**4. Engineering for scale.**
- Gather training batches on the GPU. Today an update takes about 10.6 ms, of which the update itself is about 3; most
  of the rest is building batches on the CPU, so a faster GPU alone buys little.
- Run an E7b-style check on any new hardware before pooling results (`docs/gpu.md`). Share recordings between
  machines, never regenerate them. Compare `recording_hash` and `stimuli` in each result's provenance block.
- `cloud/modal_app.py` runs the same sweep on cloud GPUs; `experiments/gpu/remote.sh` is specific to the current PC.

**5. A richer world only after the instruments pass in 2D.** Every instrument problem in round 1 would carry over
to a 3D world, and would cost more there.

Cost, for scale: Amendment 5 estimated Stages 1 to 3 at about 19 hours on one RTX 5080. Five seeds instead of three and four times
the training multiply that about sevenfold, to about 130 GPU hours at today's speed, before the batching fix.

## How to resume

Read this file, then `STATUS.md` (current state and every correction), then `docs/preregistration.md`
(Amendments 5 and 6 and the round 1 close at its end). `uv sync && uv run pytest -q` checks the code.
`experiments/stage0_rerun.py` and `experiments/stage0_rescore.py` regenerate the round 1 instrument results on a
GPU machine.
