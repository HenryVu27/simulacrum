# Preregistration: umwelt v1

Written 2026-11-04 before the main sweep. Pilot runs (lawful world, at most four minds, short training)
were used only to debug the code and validate the test battery. No threshold below was tuned on them.
The commit that adds this file is the timestamp.

## Design

- Worlds (4): lawful; impermanent (objects jump to random places, 2 percent chance per step); ghost
  (objects pass through walls, each other and the agent); nonlocal (pushing an object moves a different
  one, consistently).
- Minds (14): oracle, human, dog, blind, deafblind, vision_only, smell_only, hearing_only, touch_only, bat,
  chimera, passive, passive_novest, memoryless. Definitions live in `src/umwelt/presets.py`.
- One learner for all: a gated recurrent unit (GRU) world model with 128 hidden units, trained to predict
  the next observation and the observation five steps ahead. 4,000 updates, batch 32, window 96.
- Data: 1,024 training and 256 test episodes of 256 steps per world. Every mind sees the same episodes.
- Seeds: 2 learner seeds per (mind, world) cell. 112 trained minds in total.
- Probes: one masked multi-output probe per concept family, fit on 384 training episodes, scored on the
  256 test episodes. Primary: MLP (multilayer perceptron) probe with 256 hidden units. Secondary: linear.
- Controls per cell: the same mind untrained (floor), raw current observations (what the senses give with
  no memory), and the oracle mind (ceiling).
- Surprise tests: matched quadruplets for permanence, solidity and causality (`src/umwelt/voe.py`).

## Validation criteria (the battery must pass these before any hypothesis is read)

- V1. In the lawful world the oracle scores at least as high as every other trained mind on object_seen,
  object_hidden, number and self_location, within 0.05.
- V2. Every trained mind beats its untrained copy on at least five of the nine concepts.
- V3. Untrained minds show no surprise: the 95 percent interval of S includes 0, or |S| < 0.01.

If a criterion fails, the report says so and the affected concepts are reported as unvalidated.

## Hypotheses (lawful world unless stated; scores averaged over seeds)

- H1. Permanence needs memory. object_hidden(human) exceeds object_hidden(memoryless) by more than 0.05,
  and memoryless is within 0.05 of the raw-observation baseline for human senses.
- H2. Smell lives in the past. Define past bias B = object_past minus object_now. B(smell_only) > 0 and
  B(smell_only) > B(vision_only).
- H3. Held and Hein. Without an efference copy and without a body sense, self-knowledge suffers more than
  object knowledge: self_location and self_motion of passive_novest are each more than 0.05 below human,
  while object_seen of passive is within 0.05 of human.
- H4. Counting needs a sense that separates objects. number for vision_only and for bat each exceed number
  for smell_only and for hearing_only.
- H5. Translation is asymmetric toward the richer mind. Translation R^2 from human to blind exceeds blind to
  human; human to vision_only exceeds vision_only to human; dog to smell_only exceeds smell_only to dog.
- H6. Surprise. In the lawful world, human and vision_only show permanence surprise with the 95 percent
  interval above 0. memoryless does not (interval includes 0). Minds trained in the impermanent world show
  lower permanence surprise than the same minds trained in the lawful world.
- H7. What the mind adds versus what the world forces. In a two-way analysis of variance (ANOVA) over
  minds (excluding the oracle) and worlds, with seeds as replicates: the world share exceeds the mind share
  for object_hidden and cause; the mind share exceeds the world share for object_seen, number and
  self_motion.

## Exploratory (no prediction)

- Whether representational similarity (CKA, centered kernel alignment) between minds is higher in the
  lawful world than in the others: a structural-realism style convergence signal.
- Which minds are mutually translatable at all, and which concepts each mind has that the human lacks.

## Known limits, stated in advance

- Shared trajectories mean no mind chooses its own actions. The passive contrast tests access to an action
  signal (efference copy), not agency.
- The simulator's ontology (discrete objects, collisions) is code we wrote. Findings are about how this kind
  of body in this kind of world shapes concepts, and need the second-engine audit before any broader claim.
- Probes show that information is decodable, not that the mind uses it. Causal-use tests come later.

## Amendment 1 (2026-11-04, before the main sweep)

Made after the pilot and an A/B test, before any main-sweep result existed.

1. Learner unchanged from the preregistration (one-step and five-step heads). An A/B test on the human mind
   at 4,000 updates compared far horizons of 5, 5 and 15, and 5, 15 and 30 steps, plus observational dropout
   of 30 percent (Freeman, Metz and Ha 2019). All four scored the same on object_hidden (0.18 to 0.19) as raw
   senses (0.17). At this budget the learner is mostly reactive. H1 and H6 are kept as written and may fail
   for that reason. Logs: results/logs/ab (not committed).
2. Seeds reduced to fit one night on a laptop: lawful has seeds 0 and 1; impermanent, ghost and nonlocal have
   seed 0. Seed noise in the ANOVA is estimated from the lawful cells.
3. Untrained and raw-sense controls run in the lawful world and the lawful MuJoCo world only. V2, V3 and H1
   only use those cells.
4. Exploratory probes added: object_allo_seen and object_allo_hidden, where each object is in the room
   rather than relative to the agent. They separate object memory from self-localization. No prediction.
5. MuJoCo audit worlds: lawful_mj, impermanent_mj and ghost_mj, seed 0. The surprise-test movies are rendered
   by the Flatland sense functions for every mind, including those trained on MuJoCo data.

## Amendment 2 (2026-11-04, v2 memory world, before any mind is trained in it)

Written after the v1 sweep, and before any learner was trained on data from the worlds below. The commit
that adds this section is the timestamp.

### Why

In v1 no mind remembered hidden objects. The reason is in the prediction errors: in the lawful world, memory
cut the human mind's held-out one-step error by 2 percent (human 0.227, memoryless 0.232). The learner had
almost no reason to remember. Objects did leave and return to view often (about 13 returns per episode), but
the agent turned slowly, so a returning object entered at the edge of view one ray at a time and only that
first ray depended on memory. v2 changes the world, not the learner, so that remembering pays.

### Worlds

Each v2 world is the lawful world with some rule parameters switched on (`src/umwelt/world.py`):

- lookback: half of all new goals are a look-away-and-back routine (stand still, face an object, turn 120 to
  180 degrees away, hold 0 to 24 steps, turn back), and the body turns three times faster (turn gain 6
  instead of 2), so turning around takes about five steps.
- dark: the lights go out at random (2 percent chance per lit step) for 8 to 30 steps. Eyes see black and no
  depth. Echolocation, hearing, smell, touch and body sense are unaffected. This is the lights-out paradigm in
  which infants reach in the dark for objects they saw before the lights went out (Hood and Willatts 1986;
  Clifton and colleagues 1991).
- occluders: internal walls twice as long (half-length 1.5 to 2.5 instead of 0.75 to 1.5) and objects roll
  farther (friction 0.2 instead of 0.6), so moving objects pass behind walls and come back out.
- occlusion: all three together.

The learner, the probes, the surprise tests and the minds are unchanged from v1.

### Calibration (exploratory, picks the world)

- Train human and memoryless in each v2 world: data seed 0, learner seed 0, 4,000 updates.
- Measure memory value M = 1 - err(human) / err(memoryless), where err is the mean one-step prediction error
  on 256 held-out episodes after a 16-step burn-in (the `test_error` of v1).
- Only prediction error is looked at. No probe, surprise test or hidden-state analysis is run on calibration
  models, and they are not kept.
- Selection rule: choose the world with the largest M among those with M of at least 0.10. If none reaches
  0.10, the report says so, the world with the largest M is chosen anyway, and every hypothesis below is
  labeled as tested under a weak manipulation.

### Confirmatory run

- The chosen world, fresh data (data seed 1: 1,024 training and 256 test episodes), learner seeds 0 and 1,
  all 14 minds, the full v1 battery. Untrained and raw-sense controls on learner seed 0.
- Permanence decay curves (`experiments/posthoc_permanence.py`, post hoc in v1, preregistered here) for every
  trained mind: skill above a no-information baseline at locating each object relative to the agent, by time
  since the object was last in view. The no-information baseline is the training-set mean location of each
  kind. Raw-sense curves for human, vision_only, dog, smell_only and memoryless.

### Hypotheses (chosen world; averaged over the two learner seeds)

- M1. Manipulation check. M is at least 0.10 on the fresh data, for each learner seed.
- P1. Permanence needs memory. The human's permanence skill for objects out of view 1 to 5 steps, and 6 to
  20 steps, each exceeds (a) the human raw-senses curve by more than 0.05 and (b) the memoryless mind by
  more than 0.05.
- P2. Updating a hidden object's place needs knowledge of your own motion (Held and Hein, applied to objects).
  passive_novest (no efference copy, no body sense) has permanence skill at 6 to 20 steps more than 0.05
  below the human's. The passive mind (no efference copy, but body sense) has no prediction.
- P3. Counting needs memory when objects hide. number(human) exceeds number(memoryless) by more than 0.05.
- P4. Surprise. The human's permanence surprise S has a 95 percent interval above 0, the memoryless mind's
  interval includes 0, and the human's S exceeds its S in the v1 lawful world (mean of v1 seeds 0 and 1).

### Exploratory (no prediction)

- Which other minds remember, and through which sense: bat and chimera keep echolocation in the dark, dog and
  smell_only have a scent trail, blind and hearing_only lose nothing to darkness.
- Hidden moments split into dark spells versus looked away or behind a wall.
- Egocentric versus allocentric memory (object_hidden versus object_allo_hidden).
- Translation between minds and concept import, compared with v1.

### Known limits, stated in advance

- Making the world reward memory changes the world. A positive result says a recurrent predictor of this
  size develops object permanence when the world makes it pay, not that it develops it in any world.
- Turn gain changes the body for every mind in lookback and occlusion, including minds whose senses are not
  affected by turning (smell_only).
- The surprise-test movies are unchanged from v1 (lit, still agent, short wall as the screen), so they test
  transfer from the v2 training world to a standard test.

## Amendment 3 (2026-11-04, after calibration stage 1, before any confirmatory run)

Written after the Amendment 2 calibration and before any confirmatory run, probe, surprise test or
hidden-state analysis in a v2 world.

### What calibration found (4,000 updates)

| world | M | M, dark steps | M, first lit step | M, object returns | M, other steps |
|---|---|---|---|---|---|
| lawful (reference) | 0.019 | | | 0.014 | 0.020 |
| occluders | 0.013 | | | 0.012 | 0.013 |
| lookback | 0.003 | | | -0.019 | 0.006 |
| dark | -0.045 | -0.216 | 0.217 | 0.185 | -0.045 |
| occlusion | -0.068 | -0.374 | 0.305 | 0.230 | -0.042 |

No world reached M of 0.10. In the two worlds with dark spells, memory removes 22 to 30 percent of the error on
the first lit step and 19 to 23 percent when an object returns to view: the moments these worlds were built to
reward. But the recurrent mind is worse than the memoryless one during dark spells and on ordinary lit steps,
where in the lawful world it was 2 percent better. Lookback and occluders alone add nothing.

### Reading

At 4,000 updates, M measures how fast each learner fits as well as what memory is worth. v1's long run showed
the human mind still improving at 16,000 updates (test error 0.227 at 4,000, 0.190 at 16,000). A memoryless mind
has less to learn and fits sooner. Amendment 2's fallback would choose occluders (M 0.013), a world that rewards
memory less than lawful does; a confirmatory run there would repeat v1 and test nothing.

### Change

- Calibration stage 2: human and memoryless at 16,000 updates in dark, occlusion and lawful (reference). Data
  seed 0, learner seed 0, otherwise the v1 learner. Prediction error only, as before.
- Selection rule: the world with the larger M among dark and occlusion, if it reaches 0.10.
- If one passes, the confirmatory run is Amendment 2's design with 16,000 updates for every mind. Hypotheses
  M1 and P1 to P4 are unchanged; M1 is checked at 16,000 updates.
- If neither passes, the v2 confirmatory run is not done. The report states that making memory pay at specific
  moments was not enough for this learner at this budget, and the next amendment changes the learner, not the
  world.

## Amendment 4 (2026-11-04, after calibration stage 2, before any confirmatory run)

Still no probe, surprise test or hidden-state analysis has been run on any mind trained in a v2 world.

### What stage 2 found (16,000 updates)

| world | M | M, dark steps | M, first lit step | M, object returns | M, other steps |
|---|---|---|---|---|---|
| lawful (reference) | 0.021 | | | 0.012 | 0.021 |
| dark | -0.279 | -0.919 | 0.302 | 0.255 | -0.104 |
| occlusion | -0.402 | -1.041 | 0.334 | 0.228 | -0.328 |

The bar of 0.10 failed again, by more. Diagnosis, prediction error only:

1. Not sequence length. Training uses 96-step windows and testing whole 256-step episodes, but the recurrent
   mind's error is as high in steps 16 to 96 as later.
2. Memorization. Retrained at 4,000 updates and scored on 256 training episodes and 256 held-out episodes
   (`experiments/overfit_check.py`): the human mind's error rises 9 percent from training to held-out episodes
   (0.212 to 0.232), the memoryless mind's 4 percent (0.213 to 0.222). With 1,024 training episodes, each step
   is seen about 47 times at 4,000 updates; a mind with memory can recognize an episode by its room and recall
   when its lights come back. Late in a dark spell, its held-out error is about twice its training error.
3. Hedging. On training episodes, the human mind's error 28 steps into the dark (0.32) matches what an ideal
   predictor pays there (dark spells last 8 to 30 steps, so the lights come back next step with probability
   one in three; blending the remembered scene with black at that weight costs about 0.33).
4. The bar was nearly unreachable. Memory can only pay where the senses change in a way the present does not
   predict: the first lit step after a dark spell (1 percent of steps) and objects coming back into view (3 to
   4 percent). Even a perfect memory with perfect timing would cut overall error by at most about 15 percent
   (dark world: object-return steps 3 percent of steps, memoryless error 1.06 there against 0.207 overall), and
   any real mind pays hedging costs in the dark. The overall bar measured the rarity of those moments more than
   whether the world rewards memory.

### Change

- Training (all minds, calibration and confirmatory): 8,192 training episodes instead of 1,024, 256 held-out
  episodes, 4,000 updates. Each step is then seen about 6 times, so memorizing episodes stops paying. The
  learner itself is unchanged. This is the training change Amendment 3 required when stage 2 failed.
- Manipulation check, replacing M1: on held-out episodes, memory value at object-return steps (an object coming
  back into view after at least 3 steps out of it, which includes the first lit step) is at least 0.10, and
  overall M is at least 0 (memory does not hurt overall). The original bar of 0.10 on overall M was failed at
  both budgets, and every hypothesis below is reported with that label.
- Calibration stage 3: human and memoryless in dark and occlusion with 8,192 training episodes, data seed 0,
  learner seed 0. The world that passes the new check with the larger event-level M is chosen. If neither
  passes, there is no confirmatory run in v2.
- Confirmatory run: as in Amendment 2, with 8,192 training episodes and 4,000 updates, on data seed 1. P1 to P4
  are unchanged. Probes are still fit on 1,024 training episodes.

### Erratum to Amendment 4 (added after the confirmatory run)

Point 4 of the diagnosis says perfect memory would cut overall error by at most about 15 percent. An
independent recomputation from the error arrays puts the ceiling higher: memoryless error at object-return steps
is 18 to 19 percent of its total error after the burn-in (dark world, stages 2 and 3), and 22 to 23 percent with
the first lit steps added. The 15 percent came from rounding the share of return steps (3.7 percent) down to 3
and ignoring the burn-in. The conclusion is softened, not reversed: a bar of 0.10 on overall M asked a learner to
capture about half of what a perfect memory with perfect timing could, and no learner here came close.

## Correction to the surprise-test stimuli (2026-11-05, after all v1 and v2 runs)

Not an amendment: nothing was decided before data here. This corrects a defect in the stimuli, found by an
independent check of the GPU work (`docs/gpu.md`, E7b), and records exactly what changed.

**The defect.** In the surprise-test movies, the marker for "no wall here" (`voe.HIDDEN_WALL`, used in the
wall-absent variants and to fill unused wall slots) lay on the boundary wall at y = 0. Some rays met both walls
at the same distance, to the last bit or within one unit in the last place, and the colour they returned
depended on rounding. That differs between devices, so a mind's surprise scores could differ between the Mac
and the GPU for a reason that has nothing to do with the mind. Only the dog's eyes (240 degrees) reach those
rays; echolocation sees the same depth and texture for both walls. Recorded worlds are not affected: their
internal walls never coincide with the boundary (wall centres lie 2.5 to 7.5 units from the edges; endpoints come
within about 1 unit of an edge in the worlds with surprise runs, and within 0.03 in the occlusion calibration
recording, where no ray can meet two walls at the same distance either).

**The fix.** The marker now lies one unit outside the arena, where no ray, sound path or touch reaches it.
`tests/test_core.py::test_the_absent_wall_is_really_absent` checks that moving it further away changes no
observation except the oracle's (it fails on the old geometry). Every preset's observations of the whole suite
were compared under the old and new geometry: only the dog's permanence movies (0.65 percent of entries) and all
three of the oracle's (6.9 to 10.3 percent, because the oracle reads wall coordinates directly) change; the other
12 minds' inputs are byte-identical.

**What was redone.** No mind was retrained. `experiments/recompute_voe.py` recomputed the dog's and the oracle's
surprise scores (trained and untrained networks) from their stored trained networks in all 8 stored runs (v1:
lawful seeds 0 and 1, ghost, impermanent, nonlocal, lawful in MuJoCo; v2: dark seeds 0 and 1), and the oracle in
the v1 pilot run (whose files are not in git; its committed report, `results/report_pilot`, was made by an older
version of the report code and is left as it was, so it still shows the oracle's pre-correction surprise: 0.202
for permanence, now 0.157, listed in `results/wall_fix_recompute.json`). It first
required the old movies to reproduce every stored value exactly (they did, all 16 files) and an unaffected mind
(human) to be unchanged by the new movies (it was). The old values are kept in each file under
`voe_before_wall_fix` and `voe_untrained_before_wall_fix`; every change is listed in
`results/wall_fix_recompute.json`. Both reports were regenerated (they first regenerated byte-identically from
the unchanged files, so the differences are the fix's alone).

**What changed.**
- Dog: permanence surprise moved by at most 0.0005 (11 scores); no other dog score changed.
- Oracle, trained network, permanence: down in every run, by 0.03 to 0.11, or 18 to 30 percent (lawful seed 0:
  0.484 to 0.386; dark seed 0: 0.499 to 0.411). Its other surprise scores moved by at most 0.0054. In the ghost
  world its solidity interval now excludes zero (0.040, [0.005, 0.080]; before [-0.004, 0.084]); see the caveat,
  this is an artefact of the encoding. (The message of commit cf27e28 says "down 0.07 to 0.11"; the ghost world
  fell 0.028, so the range is 0.03 to 0.11.)
- Verdicts: none changed. Every check line of both reports (V1 to V3, H1 to H7, the MuJoCo audit, M1, P1 to P4)
  is identical before and after, confirmed by the independent check, which also regenerated both reports. H6 and
  P4 use the permanence surprise of minds other than the oracle. The dog's shift (at most 0.0005) is about a tenth
  of the permanence surprise those checks look at (around 0.004) and as large as the dog's own H6 difference
  (impermanent minus lawful, +0.0002 before and +0.0003 after), so the dog alone does not decide them: H6's share
  of minds less surprised after impermanent training stays 8 of 13. (The first version of this section said "three
  orders of magnitude below"; that was wrong.)

**What this exposes about the oracle.** The oracle reads wall coordinates, so for it an absent wall has to be
encoded as some coordinates, and no encoding is like anything it saw in training. The independent check put ten
off-arena encodings into the oracle's input, every one invisible to the other senses
(`results/checks/2026-11-05_stimulus_fix/encoding_out.txt`). The trained oracle's permanence surprise ranges
from about 0 (marker far away, at (-50, -50)) to 0.48 (lawful seed 0; dark seed 0: 0 to 0.57); at y = -2 it is
0.18 to 0.33 across runs and at y = -5 0.02 to 0.15. Its solidity surprise in the ghost world changes sign
(+0.040 [0.005, 0.080] with the marker below the arena, -0.035 [-0.044, -0.024] above it). So the oracle's surprise
scores are set largely by an arbitrary encoding choice, before and after this correction, and should be treated
as uninterpretable, not as a control. One preregistered verdict depends on them: V3 ("untrained minds show no
surprise") fails in v2 only because of the untrained oracle's permanence surprise (-0.0149, against a threshold of
0.01); with the marker at y = -5 that score is -0.0090 and v2's V3 passes. In v1, V3 fails with or without the
oracle (the untrained dog's causality surprise also fails it). The preregistered rule is not changed here; the
verdict is reported with this dependence stated. (The second correction, below, removes the encoding. V3 in v2
still fails on the untrained oracle's permanence surprise, for a reason that no longer depends on a choice.)

**A second defect, found by the same check and not corrected here.** In the solidity movies the agent stands at
x = 6.0 and the wall lies on the line x = 6.0, so it sees the wall edge-on and no ray ever hits it. For every mind
except the oracle, the "wall and bounce" movie is byte-identical to "no wall and bounce", and "no wall and pass" to
"wall and pass", in all 18 instances. The solidity surprise of the 13 non-oracle minds has therefore measured noise
from the stimuli's other differences (none, for the senses), in every run; for the oracle it contrasts only wall
coordinates. No preregistered verdict rests on solidity surprise alone, but every solidity number in the reports
is uninformative. Fixing it means redesigning those movies (moving the agent off the wall's line) and recomputing
every mind's solidity surprise from the stored networks; that is left for a decision. (Decided and done: second
correction, below.)

## Second correction to the surprise-test stimuli (2026-11-05, after all v1 and v2 runs)

Not an amendment either: it corrects the two defects the previous section records, at Henry's go-ahead ("Fix
both"). As before, nothing was retrained and every stored surprise score was recomputed from the stored networks.
Unlike before, every mind's movies change, so every surprise number in both reports moves. The test that guarded
the first correction (`test_the_absent_wall_is_really_absent`) went with the marker it tested; three new tests
replace it (below).

**The oracle's input.** Every training scene has exactly three internal walls (`world.reset`), so no encoding of a
missing wall is like anything the oracle has seen. An input flag for "wall present" would not help: it would be
constant in training, its "absent" value as unfamiliar as any marker, and the oracle would have to be retrained in
every run. So the movies no longer have missing walls. Every movie has three internal walls in the training range
(centre in [2.5, 7.5] squared, half-length in [0.75, 1.5]): the event's own walls, and "bystander" walls drawn the
way `world.reset` draws walls and kept only where they touch no object's path and block no line of sight or sound
between the agent and an object or the event's walls. A quadruplet's four movies share them (`voe._bystanders`,
`voe.LAYOUT_SEED`).

**The solidity movies.** The agent now stands below the path and to the left of the wall's line, at (5, y - 2)
facing the path, so it sees the wall's face. Checking that view turned up a problem with seeing the wall at all. A
visible wall hides an object that passes through it, and no possible movie has a matching moment. So the first
version of the redesign broke the balance the design rests on. A predictor that copies the last frame found the
impossible movies easier: S = -0.11 for the human's senses, -0.22 for the chimera's. Untrained networks of the
vision minds scored -0.011 to -0.021 (the dog +0.017; lawful seed 0), beyond V3's threshold of 0.01. Moving the
agent toward the wall's line shrinks the imbalance only as the wall disappears
(`results/checks/2026-11-05_battery/solidity_designs.txt`, which also lists two occluder designs that missed). The
adopted design adds a screen below the path, in all four movies. It covers exactly the sight lines from the agent
to the path beyond the wall's line, so in both passing movies the object vanishes behind the same screen, and the
wall shows above the screen from the path up. The screen's edge is the sight line to the point where the wall
crosses the path, set by geometry, not tuned. With the wall in its first form (1 unit above the path, not 1.4),
moving that edge 2 degrees kept the balance (within 0.0025) and 4 or 6 degrees did not (-0.009 and -0.031 for the
human). "No wall" is now the same wall slid along its line until it clears the path (by 0.25 for the largest
object), because the oracle cannot be shown a missing wall.

**Checks** (`experiments/battery_stimuli.py`, output `results/checks/2026-11-05_battery/stimuli.txt`, unless
stated):
- All 54 quadruplets: walls in the training range, bystanders shared, and bystanders hide nothing. Rendered with
  and without them, only rays that met a boundary wall change (to a nearer wall), and no other sense changes. Two
  walls placed on purpose to block something are caught.
- Solidity: before the contact, 2 to 6 of the human's rays per frame tell wall-in-the-path from wall-slid-away
  (0 before). The passing object is seen in the same frames with the wall in its path and without (9.3 of 24 on
  average).
- Balance, copy predictor on noise-free inputs: solidity within 0.0025 of zero for human, vision_only, dog, bat and
  chimera, and every one of the 18 instances within 0.0062. Permanence and causality are unchanged at 0. Found in
  passing and unchanged: permanence is not balanced for smell (-0.082 for blind and smell_only), because scent
  lingers after its source has gone.
- `tests/test_core.py`: walls in the training range and bystanders that hide nothing, both on all 54
  quadruplets; solidity shows the wall and stays balanced (this test fails without the screen); permanence is
  balanced for the copy predictor (noise-free now, within 0.01; the old version, with sensor noise and a bound of
  0.1, let an object vanish in plain view). 19 tests pass. As first committed (c35cc6e) the wall tests rendered 6
  quadruplets, and a bystander allowed to cross the agent's line of sight to an object got through; the
  independent check found it, and the 54-quadruplet tests catch it.
- Where the bystanders stand (`experiments/battery_layouts.py`, `layouts.txt`): every mind of lawful seed 0 and
  dark seed 0 was scored on five bystander layouts. Across layouts a score moves by at most 0.68 of its own 95
  percent interval width (median 0.09 to 0.21 by test). The trained oracle's permanence surprise is 0.92 to 0.97
  (lawful seed 0) and 0.90 to 0.92 (dark seed 0). Under the off-arena marker it ranged from 0 to 0.57 with the
  marker's place.

**What was redone.** `experiments/recompute_battery.py` recomputed all 498 surprise scores of the 121 stored minds,
trained and untrained networks: the 8 runs of 14 minds, the pilot's 3, and the 6 single-mind follow-up runs (long,
jepa, ab_*). It first required the previous movies to reproduce every stored score exactly; they did, for all 121
minds (the first correction checked 3 minds per run). The previous scores are kept under `voe_before_battery_fix`
and `voe_untrained_before_battery_fix`. Each file's provenance now names the new movies (`stimuli`
016fc3d72807e39d), and every change is listed in `results/battery_fix_recompute.json`. Both reports first
regenerated byte-identically from the unchanged files, then from the new ones.

**What changed.**
- Oracle: trained permanence surprise 0.386 to 0.953 (lawful seed 0) and 0.411 to 0.919 (dark seed 0); solidity
  -0.062 to +0.007 (lawful seed 0). Untrained permanence -0.015 to -0.025, the same on every layout (within
  0.001).
- Causality, trained vision minds: most of the large negative scores moved toward zero (lawful seed 0: human -0.073
  to -0.020, memoryless -0.076 to -0.022, passive -0.077 to -0.023; dark seed 0: human -0.037 to +0.002). Copy
  balance did not change, so the networks, which only ever saw scenes with three walls, behave differently in an
  empty arena. bat and chimera stay clearly negative (-0.074 and -0.058). The largest change for a mind other than
  the oracle is 0.064.
- Solidity: no mind shows a learned expectation that walls stop objects. Some minds' solidity intervals do exclude
  zero (of 114 trained minds in the 8 full runs and the jepa and long runs: 16 above, 5 below; 9 also have |S| of
  at least 0.01, V3's other condition). The bat accounts for most (above zero in all 8 v1 and v2 runs, up to +0.020
  in the MuJoCo world), with the chimera and the jepa human (+0.020). But minds trained in the ghost world, where
  objects pass through walls, score as high: the bat +0.012 and the chimera +0.011 there. Lawful minus ghost is
  within 0.006 for every mind that can sense the wall (`experiments/solidity_split.py`, `solidity_split.txt`; post
  hoc). So these scores come from how some learned predictors respond to the movies, not from solidity. (Commit
  c35cc6e's message says "no trained mind shows solidity surprise"; that is wrong as stated, and the independent
  check caught it.) Splitting S into "passing through the wall" (I1 - P2) and "bouncing off nothing" (I2 - P1) does
  not help: for every vision mind both carry an equal and opposite effect of where the wall stands (about 0.1 of
  the possible movies' error), which cancels only in S. (These counts and contrasts used one noise draw, as every
  score did then; over 20 draws, next section, the answer is the same.)
- Permanence, minds other than the oracle: changed by at most 0.0035.
- Verdicts. V3 fails as before, now only on the untrained oracle's permanence surprise (main report, both blocks;
  v2). The untrained dog's causality surprise (+0.0118 to +0.0052) no longer fails it, so the non-preregistered "V3
  excluding the oracle" goes from FAIL to PASS in v1 (it already passed in v2). The oracle's failure no longer
  depends on a choice: its input contains the hidden event itself (whether an object is present), so the permanence
  quadruplet is not balanced for it by construction. H6 and P4 fail as before, with parts that flip: H6's
  "memoryless shows no surprise" goes from false to true in v1 and from true to false in v2. P4 now fails all three
  of its parts: the memoryless mind's interval in dark seed 0 excludes zero (0.0045 [0.0012, 0.0076]), and the
  human's mean S is no longer above its v1 value (0.0015 against 0.0023; before, 0.0039 against 0.0037). No other
  check line changes: V1, V2, H1 to H5, H7, M1 and P1 to P3 use no surprise score.

**Independent check** (outside-judge agent, on snapshots of c35cc6e and d6c6763; scripts and outputs in
`results/checks/2026-11-05_battery/independent`). It confirmed every claim above with its own code: walls and
bystanders on all 54 quadruplets with its own ray caster; balance, with how sensitive it is (moving the screen by
up to 0.05 or turning it by up to 2 degrees keeps it; shortening it by 0.1 does not, -0.009 for the human; the
echo minds are the most sensitive, up to +0.004); all 996 stored scores of the 121 minds, previous and new, in its
own loop, exactly; 17 networks by its own float64 forward pass, within 3.2e-7; the oracle's permanence surprise
across 20 layouts (0.89 to 0.98, lawful seed 0); and both reports, byte-identical. It found the wrong solidity
statement, the 6-quadruplet tests and the limits below; each is corrected or recorded here.

**Limits.**
- Where the bystanders stand is a draw. The layout check above measures what it does, for two runs.
- In the solidity movies the passing object is hidden from the wall's line on (it reappears, at x = 8.35 to 9.25,
  in the last 4 or 5 frames), in both passing movies. So a mind can show solidity surprise only as "it should have
  come back" or "it should not have come back".
- Only vision and echolocation can locate the wall. For blind, deafblind, smell_only, hearing_only and touch_only,
  solidity scores carry no information about solidity, as they should not.
- The walls stay within v1's range of lengths; the occluder worlds (half-length 1.5 to 2.5) were never
  surprise-tested.
- Every score saw one draw of sensor noise: `presets.observe` seeds it with the preset's name, so every run of a
  preset sees the same draw. Across 20 other draws the bat's solidity S (lawful seed 0) spans -0.0071 to +0.0123
  (independent check, `noise_draws.py`). This predates the redesign and applies to every surprise score. (As
  first written this item said the intervals did not cover the noise. Within a run they did, since the noise
  acts on each quadruplet independently; the trouble was the noise in the score itself, shared by every run.
  Corrected and dealt with in the last section.)
- Found by the independent check, older than the redesign, not changed: the movies' physics is simpler than the
  world's. A bounce's sound is weighted by the object's mass (the world's is not), objects approach at constant
  speed (the world slows them by 0.94 a step), for three radii the last step before a bounce is shortened, and in
  the possible causality movie the struck object always leaves at the striker's speed while the striker stops dead,
  whatever their masses. Each is shared by a possible and an impossible movie, so balance is unaffected, but a mind
  that learned the world's physics exactly may find the possible movies slightly odd too.
- `results/runs/bench_mac-cpu` (a timing run, not in git) keeps a network but was not recomputed; its human file
  still holds the previous movies' scores. No report uses it.

## Surprise scores over sensor-noise draws (2026-11-05, after all v1 and v2 runs)

A change to how every surprise score is computed, at Henry's go-ahead, prompted by the independent check of the
second correction. Not an amendment either: no movie, mind or rule changes, only how much sensor noise a score
averages over and how its interval is resampled. A first version (73f46c9) was wrong; its own independent check
found why, and the version below replaces it.

**The problem.** Each mind senses the movies through Gaussian sensor noise (`presets.observe`). Every surprise score
so far saw one draw of it, seeded by the preset's name, so every run of a mind saw the same draw. On 156 scores
(every mind with sensor noise in lawful seed 0 and dark seed 0, trained and untrained, three tests;
`experiments/noise_draws_check.py`, `results/checks/2026-11-05_noise_draws/check.txt`):
- One draw moved S with a standard deviation of 0.0024 (median; largest 0.031), about half the interval's
  half-width, and whether the interval excludes zero differed from draw 0's in 17 percent of draw pairs. The
  permanence effects H6 and P4 look for are that size.
- smell_only's permanence surprise was +0.019 to +0.045 in all 8 runs. That was the shared draw: averaged over
  many draws it is -0.0025 to -0.0039.
- Within a run the interval did cover the noise (a one-draw interval contains the 200-draw score 94.6 percent of
  the time): the noise acts on each quadruplet independently, so the bootstrap over quadruplets saw it. The trouble
  was the noise carried by the score itself, the same in every run of a mind.

**The first version and what was wrong with it.** 73f46c9 averaged 20 draws and made the bootstrap resample draws
as well as quadruplets. Its independent check (outside-judge agent; `results/checks/2026-11-05_noise_draws/
independent`) reproduced every number exactly and found: resampling draws counted the remaining noise a second
time (permanence intervals about 1.7 times too wide); 20 draws left noise as large as the permanence effects, so
on fresh draws one of P4's parts came out the other way; at 500 resamples the bootstrap's own randomness changed
2.2 percent of decisions; three of the new tests passed with broken code; and the commit's account of the problem
("the interval did not cover the noise") was wrong.

**The change.** Each score is taken on the mean summed errors of 200 draws (`voe.NOISE_DRAWS`; draw 0 is the old
one, so training inputs are unchanged), with a bootstrap over the quadruplets of that mean and 2,000 resamples
(`voe.surprise`, `voe.N_BOOT`). Minds without sensor noise (the oracle) keep one draw, since every draw would be
the same. Provenance records `voe_draws` and `voe_boot`, and a tag refuses to mix results with other values.

**Checks.**
- The independent check of the first version found the statistic itself sound: the noise acts independently on
  each quadruplet (single-draw spread against the per-quadruplet noise: ratio 0.98, median), and the ratio's bias
  is negligible. In its coverage simulation, quadruplets-only intervals on the multi-draw mean covered 92 percent
  (the percentile bootstrap with 18 quadruplets runs a little short of 95; the one-draw interval did the same).
- Two disjoint sets of 200 draws (0 to 199, 200 to 399), same networks: S differs by a median 0.0002 (0.22 of the
  interval's half-width; largest 0.0039), and whether the interval excludes zero agrees for 94 percent of the 156
  scores; the 9 that disagree are intervals touching zero. One set's interval contains the other set's score 86
  percent of the time, as expected where the remaining noise is a large part of the interval, since the
  difference of two estimates carries it twice.
- Every part of H6 and P4 is the same on both sets of draws (H6: no interval above zero, memoryless null holds,
  share 0.54; P4: the human's interval not above zero, memoryless null holds, the human not above v1).
- Tests: draw 0 matches the historical seeding rule; one draw gives exactly the original statistic (kept in the
  test as code); several draws give it on the draws' mean; provenance names every setting. Four mutants (draw 0
  reseeded, resampling order changed, draws resampled too, `voe_draws` dropped) all fail. 22 tests pass.
- The GPU reproduces the rescored values within 6.6e-7 (`docs/gpu.md`).

**What was redone.** `experiments/recompute_noise.py` rescored all 498 scores of the 121 stored minds, after
requiring one draw with 500 resamples to reproduce every original score exactly (it did). The original values are
kept under `voe_before_noise_draws` and `voe_untrained_before_noise_draws`, and every change is listed in
`results/noise_draws_recompute.json`. The reports were regenerated.

**What changed** (against the original one-draw scores).
- Scores of minds with sensor noise moved by a median 0.0022 (largest 0.049, smell_only's permanence in the
  nonlocal world, +0.045 to -0.003). Intervals narrowed to a median 0.09 of their width (permanence 0.08, solidity
  0.40, causality 0.90): most of the spread between quadruplets was noise. 90 of the 459 intervals now exclude zero
  (71 before).
- Permanence, lawful seed 0: human +0.0020 to -0.0001 [-0.0004, 0.0001], vision_only +0.0017 to +0.0000 [-0.0002,
  0.0002], memoryless +0.0030 to +0.0001 [-0.0002, 0.0004]. The vision minds' permanence surprise is within a
  few ten-thousandths of zero.
- Solidity: 19 of 114 trained minds' intervals lie above zero and 12 below, the bat's above in all 8 runs. The
  world contrast gives the same answer as before: the ghost-trained bat +0.0088 and chimera +0.0155, and lawful
  minus ghost within 0.0076 for every mind (`solidity_split.txt`).
- Verdicts: every check line passes or fails as before. H6 (v1): the share of minds less surprised after
  impermanent training goes from 0.62 to 0.54. P4 now fails two of its three parts: the human's interval is not
  above zero, and its dark-world permanence S (-0.0003 [-0.0005, 0.0000] and -0.0002) is below its v1 value
  (-0.0001); the memoryless mind shows no surprise. V3 is unchanged; it fails only on the oracle, which has no
  sensor noise.

**Limits.**
- The 200 draws are the same in every run of a mind (seeded by its name), so the noise left in a score, which each
  interval includes, is shared across runs: agreement between runs of one mind says nothing about it. (It also
  makes comparisons between runs paired, which helps H6 and P4.)
- Percentile intervals over 18 quadruplets cover a little under 95 percent (about 92 in the simulation).
