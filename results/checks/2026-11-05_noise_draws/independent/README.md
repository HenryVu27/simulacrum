# Independent check of the 20-draw surprise scores (commit 73f46c9)

Scripts and text outputs of the outside-judge agent that checked commit 73f46c9 against e1007a5, copied here
unchanged so the record's numbers have a source. They ran on the Mac from a scratch folder against `git archive`
snapshots; paths inside them point at that folder. Large intermediate files (per-draw totals, JSON rows, pickles)
are left out.

What it established, and what changed because of it (`docs/preregistration.md`, surprise scores over sensor-noise
draws): the code reproduced every number exactly (c1, c2); the noise acts independently on each quadruplet, so the
old one-draw interval did cover it (c3_old_cover.txt), and resampling draws as well counted it twice (c3_sim.txt,
c3_summary.txt); 20 draws left noise as large as the permanence effects, shared by every run of a mind
(c3_draw_dist.txt, c3_cross_run.txt, c4_p4_h6_fresh.txt); the bootstrap's own Monte Carlo error flipped 2.2% of
decisions at 500 resamples (c3_seed_sensitivity.txt); three tests passed with broken code (mut1 to mut3, their
outputs in mut1_fullsuite.txt). The final statistic uses 200 draws, a bootstrap over quadruplets of the draws'
mean and 2,000 resamples, and the tests catch those mutants.
