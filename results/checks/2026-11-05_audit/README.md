# Independent audit of umwelt (2026-11-05, at commit bdf9cf5)

Scripts and outputs of a skeptical outside audit (outside-judge agent) of the project's methodology, process and
results, copied unchanged so the record's numbers have a source. Each `NN_name.py` has its output in
`NN_name.out`; they ran from the repository with `uv run --no-sync python`, some reading recordings and stored
networks that are not in git. `regen.log` is the regeneration of both reports from a scratch copy of the results.

Findings, most severe first (STATUS.md, "Independent audit", and the "(audit)" notes beside each claim):
- The probe battery cannot register memory: adding exact copies of earlier frames to the human's raw senses
  lowers its hidden-object score (12_h1_power, 05_probe_sensitivity), so the probe side of every memory result
  was close to untestable.
- The mind-versus-world variance shares depend on which minds and worlds were sampled (01_json_checks), and the
  cause probe mostly reads proximity (02_cause_confound).
- Learning adds little the battery can see (10_learning_vs_input); H5's asymmetry appears between untrained
  networks too (06_translation).
- v2's dark world shows a brief trace of memory (+0.10 over raw senses at 1 to 5 steps in the dark) that the
  pooled test hid (08_probe_bootstrap).
- What holds: the record reproduces exactly (09, 04, regen.log); the process followed the preregistration; the
  permanence surprise test is well powered (04_voe_power); probe contamination is ruled out (03).
Amendment 5 (`docs/preregistration.md`) is the plan that follows from it.
