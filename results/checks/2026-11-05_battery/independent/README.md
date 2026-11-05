# Independent check of the 2026-11-05 surprise-battery redesign (commit c35cc6e)

Scripts and outputs of the outside-judge agent that checked the second correction (`docs/preregistration.md`),
copied here unchanged so the record's numbers have a source. They ran on the Mac from a scratch folder, against
`git archive` snapshots of c35cc6e and d6c6763 (not copied: they are those commits), and read the stored runs in
`results/runs`; paths inside them point at that scratch folder. Pickled intermediate files are left out.

- `check_movies.py`: walls in the training range, standing still and shared; bystanders hide nothing, with an
  own float64 ray caster (all 54 quadruplets).
- `check_solidity.py`, `who_sees_wall.py`, `pass_visibility.py`: the old movies equal for every sense but the
  oracle's; the new ones tell wall from no wall; the passing object seen in the same frames with and without.
- `balance.py`, `standardised_copy.py`: copy-predictor balance and how sensitive it is to the screen and the eye.
- `full_loop.py`, `my_forward.py`: every stored score recomputed in an own loop (121 minds, 996 scores, exact) and
  by an own float64 forward pass (17 networks, within 3.2e-7).
- `oracle_layouts.py`, `oracle_slots.py`: the oracle's permanence surprise across 20 layouts and slot orders.
- `solidity_ci.py`, `noise_draws.py`: solidity intervals that exclude 0, and how much the sensor-noise draw moves S.
- `mutate.py`, `mutate2.py`, `m2_hide.py`: mutants of voe.py against the tests (M2 escaped the tests of c35cc6e).
- `world_bounce.py`, `world_launch.py`, `contact_timing.py`, `phys_report.py`: the movies' physics against the
  world's (findings that predate the redesign).
- `screen_determinism.py`, `edge_centres.py`: determinism of the screen; bystanders on the range's edge.
