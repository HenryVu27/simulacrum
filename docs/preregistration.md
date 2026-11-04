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
