# Independent check of the 2026-11-05 surprise-test stimulus correction

Scripts and outputs of the outside-judge agent that checked the correction (`docs/preregistration.md`, last
section), copied here unchanged so the record's numbers have a source. They ran on the Mac from a scratch
folder and read the stored runs in `results/runs`; paths inside them point at that scratch folder.

- `geometry.py`: float64 ray geometry for every movie pose; ties with the boundary under the old marker, none under the new.
- `far_marker.py`: a suite with the marker far away equals the committed one in every non-oracle sense.
- `build_compare.py`: both suites built and every preset's observations compared (only dog permanence and the oracle change).
- `indep_scores_lib.py`, `indep_scores.py`, `indep_all.txt`: an independent float64 NumPy forward pass of the
  stored networks and surprise statistic; all 132 recomputed scores agree with the stored ones within 3.1e-7.
- `encoding.py`, `encoding_out.txt`: the oracle's surprise scores under ten off-arena encodings of the absent wall.
