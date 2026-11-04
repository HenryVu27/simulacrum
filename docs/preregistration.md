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
