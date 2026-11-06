# umwelt

Which concepts does the world force on any mind, and which does a body add?

umwelt trains many artificial minds that share one learning rule and one world but differ in their
senses and body: a human-like mind, a dog-like mind, a blind one, Condillac's smell-only statue, an
echolocating bat, a yoked passive observer in the style of Held and Hein's kittens, a memoryless mind
and an oracle that sees the true state. It then measures what each mind knows (object permanence,
number, causes, self-location, time, spatial relations), how surprised it is by impossible events, and
whether one mind's inner world can be translated into another's. A second axis breaks the world's
physics on purpose (objects that jump, objects that pass through walls, causes that act at a distance),
so each concept's variance can be split into a share the mind contributes and a share the world forces.

Start with `STATUS.md`. The design and its predictions are in `docs/preregistration.md`; the related
work and the honest novelty claim are in `docs/related-work.md`.

## Layout

| Path | Holds |
|---|---|
| `src/umwelt/world.py` | Flatland: 2D physics in JAX with breakable rules (lawful, impermanent, ghost, nonlocal) and v2 memory worlds (lookback, dark, occluders, occlusion) |
| `src/umwelt/senses.py` | Every sense as a function of world state: vision rays, ears, nostrils on a scent field, skin, body sense, compass, echolocation |
| `src/umwelt/presets.py` | The 14 minds, as settings on sense, action and memory dials |
| `src/umwelt/data.py` | Records each world once, so every mind sees the same history |
| `src/umwelt/learner.py` | The shared learner: a recurrent world model predicting its next sensation |
| `src/umwelt/probes.py` | The concept battery, with untrained, raw-sense and oracle controls |
| `src/umwelt/voe.py` | Violation-of-expectation quadruplets: permanence, solidity, causality |
| `src/umwelt/translate.py` | Translation maps and representational similarity between minds |
| `src/umwelt/report.py` | Preregistered checks, mind-versus-world variance split, figures |
| `src/umwelt/report_v2.py` | v2 (memory world) checks for Amendments 2 to 4, decay curves, calibration figures |
| `experiments/` | Calibration, overfitting check, world statistics, decay curves, sweep scripts |
| `cloud/modal_app.py` | The same sweep on cloud GPUs |
| `experiments/gpu/` | Running on the RTX 5080 PC from this Mac (`remote.sh`), and the hardware checks; see `docs/gpu.md` |

## Run

```bash
uv sync
uv run pytest -q
uv run umwelt data                       # record the four worlds
uv run umwelt run --world lawful --seed 0 # train all 14 minds and run every test
uv run umwelt report                     # results/report_main/REPORT.md
experiments/sweep_v2.sh dark             # v2 confirmatory run (8,192 training episodes)
uv run umwelt report-v2 --world dark     # results/report_v2/REPORT.md
experiments/gpu/remote.sh run NAME CMD   # a job on the RTX 5080 PC (docs/gpu.md)
```

`--presets` picks a subset of minds, `--steps` sets training length, `--tag` keeps runs apart.

## Publishing

The repository is public. Details of the machines it runs on and of other projects stay out of it:

- `experiments/gpu/remote.sh` reads the GPU PC's address from `UMWELT_PC` or from `experiments/gpu/.pc`, which
  git ignores.
- Job logs record that another project holds the shared GPU lock, not what it is running.
- `.githooks/pre-push` refuses a push whose commits contain any string listed in `.private-patterns` (ignored by
  git; one extended regular expression per line). Turn it on once per clone: `git config core.hooksPath .githooks`.

The history was rewritten once, on 2026-11-05, before the first push, to take those details out. Commit names
cited in the docs and in result files were updated to match; `docs/commit-map.txt` lists the old names. Code
fingerprints in job logs from before then were computed on the files as they were, so they do not match one
computed from this history.
