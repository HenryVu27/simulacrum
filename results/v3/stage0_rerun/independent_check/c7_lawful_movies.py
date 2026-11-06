"""The lawful channel inside the surprise movies: I1 must carry P1's channel and I2 P2's, P1 and P2 their own truth, and
an impossible movie's own state must never appear where it differs from the possible one. Then the lawful mind's
permanence S over 200 noise draws, recomputed from the stored network."""
import importlib.util, pickle, json, time
from pathlib import Path
import numpy as np
REPO = Path("/Users/henryvu/personal/simulacrum")
spec = importlib.util.spec_from_file_location("rr", REPO / "experiments/stage0_rerun.py"); rr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rr)
from umwelt import learner, presets, voe
t0 = time.time(); suite = voe.build_suite(); print(f"suite built in {time.time()-t0:.0f}s, hash {voe.suite_hash(suite)}")
for test, (dm, crit) in suite.items():
    own = rr.channel_from_oracle(dm["obs_oracle"]); n = len(own) // 4; own = own.reshape(n, 4, *own.shape[1:])
    given = rr.lawful_channel_for_movies(dm).reshape(n, 4, *own.shape[2:])
    p_ok = np.array_equal(given[:, 0], own[:, 0]) and np.array_equal(given[:, 1], own[:, 1])
    i_ok = np.array_equal(given[:, 2], own[:, 0]) and np.array_equal(given[:, 3], own[:, 1])
    leak = 0; diffsteps = 0
    for i in range(n):
        for v_imp, v_pos in ((2, 0), (3, 1)):
            differs = np.abs(own[i, v_imp] - own[i, v_pos]).max(-1) > 0  # steps where the impossible movie's own state differs
            diffsteps += differs.sum()
            leak += (np.abs(given[i, v_imp][differs] - own[i, v_imp][differs]).max(-1) == 0).sum()
    # sanity: before the critical moment the impossible and possible movies agree (they should, by design)
    pre = max(np.abs(own[i, 2, : crit[i] - 1] - own[i, 0, : crit[i] - 1]).max() for i in range(n))
    print(f"{test:10} P1,P2 get their own: {p_ok}; I1<-P1, I2<-P2: {i_ok}; steps where an impossible movie's own state differs: {diffsteps}, "
          f"of which the given channel equals the impossible state: {leak}; max |I1-P1| own channel before critical moment: {pre:.4f}")
stored = json.loads((REPO / "results/v3/stage0_rerun/lawful.json").read_text())
p = presets.PRESETS["human"]
for name in ("lawful", "twin"):
    with open(REPO / f"results/v3/stage0_rerun/{name}_params.pkl", "rb") as f: sv = pickle.load(f)
    dm, crit = suite["permanence"]
    extra = rr.lawful_channel_for_movies(dm) if name == "lawful" else None
    am = presets.actions(dm, p); errs = []
    t0 = time.time()
    for k in range(voe.NOISE_DRAWS):
        x = presets.observe(dm, p, draw=k)
        if extra is not None: x = np.concatenate([x, extra], -1)
        errs.append(learner.batched(rr.sense_errors, sv["params"], (x - sv["mu"]) / sv["sd"], am))
    s, ci = voe.surprise(np.stack(errs), crit)
    st = stored[name]["voe"]["permanence"]
    print(f"{name:6} permanence S {s:+.6f} [{ci[0]:+.6f}, {ci[1]:+.6f}]  stored {st['S']:+.6f} [{st['ci'][0]:+.6f}, {st['ci'][1]:+.6f}]  ({time.time()-t0:.0f}s)")
