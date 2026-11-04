# Design

## The question

Kant claimed that some structure in experience (objects, causes, space, time) comes from the mind rather
than the world. For almost 250 years that was untestable, because there was one kind of mind to study.
umwelt builds many kinds of mind that share a single learning rule and a single world, and differ only
in body and senses. It then asks which concepts every mind ends up with (candidates for what the world
forces) and which depend on the body (what each mind contributes). A second axis breaks the world's
physics on purpose, so the variance in each concept can be split into a mind share, a world share and an
interaction. That split is the empirical version of Kant's question; the agreement across minds is a
structural-realism style test of what survives every lens.

## Why these parts

| Part | Choice | Why |
|---|---|---|
| World | 2D continuous physics with walls that hide things | Occlusion makes permanence a real question; collisions make causes; animate objects make "moved by itself" a category |
| Rules | lawful, impermanent, ghost, nonlocal | Each removes one thing a lawful world forces: persistence, solidity, contact causation |
| Senses | functions of world state, chosen by information profile | Vision is fast, far and occludable; smell is slow, lingers and goes around walls; hearing reports events; touch is local; body sense reports self-motion |
| Minds | 14 presets on dials (senses, efference copy, memory) | Each preset isolates one philosophical contrast: Condillac, Strawson, Held and Hein, Molyneux-like deprivation, the oracle ceiling |
| Shared data | every mind sees the same recorded episodes | Differences in what minds learn come only from the body, never from what happened to them |
| Learner | one recurrent world model predicting the next observation and the one five steps ahead | Self-supervised, no task reward, so nothing is told what matters |
| Probes | masked multi-output probes per concept, MLP and linear, validation by held-out episode | A probe that memorizes episodes is caught; the oracle and untrained minds bound every score |
| Surprise | matched quadruplets (IntPhys design) | Every frame transition appears equally often in possible and impossible sets, so only expectation produces surprise |
| Translation | ridge maps and CKA between hidden states on the same moments | Measures which minds can be read in terms of which others, and asymmetrically |
| Audit | the same world rebuilt in MuJoCo | A finding that survives both engines is not an artifact of one physics implementation |

## Choices made during validation

The pilot (lawful world, three or four minds, short training) changed three things before the sweep:

1. Linear probes on raw oracle input scored below zero on object location, because rotating world coordinates
   into the agent's frame is not linear. Probes are now one-hidden-layer MLPs (multilayer perceptrons),
   with linear probes reported alongside.
2. Probes memorized episodes: objects mostly sit still within an episode, and the validation rows came from the
   same episodes as the fitting rows. Validation now holds out whole episodes, and probes fit on all 1,024
   training episodes.
3. The oracle received the true state in slot order with one-hot kinds, which forced the probe to route by
   kind. It now receives the state indexed by kind, which is the honest ceiling.

## What comes next

- Causal-use tests: ablate a probe direction and measure whether prediction of the relevant future degrades.
- Concept import: sparse autoencoders on non-human minds, looking for features no ground-truth variable explains.
- Self-driven minds (each chooses its own actions) to separate agency from efference copy.
- The full factorial sweep over sense dials on cloud GPUs, more seeds, and frontier systems through the same battery.
- The Aporia Atlas: map each finding onto the premise of the classic inconsistent set it bears on.
