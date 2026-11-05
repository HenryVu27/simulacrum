"""How much do the oracle's surprise scores depend on how an absent wall is encoded? Substitute other encodings
into the oracle's wall dimensions (other minds' inputs are unaffected by any off-arena marker) and recompute."""
import json, pickle, sys
import numpy as np
import jax
sys.path.insert(0, sys.argv[2])
from indep_scores_lib import my_errors, my_S
from umwelt import learner, presets

suites = pickle.load(open(sys.argv[1], "rb"))
ROOT = "/Users/henryvu/personal/umwelt/results/runs/"
NEW = np.array([0.5, -1.0, 1.5, -1.0], np.float32)
OLD = np.array([0.5, 0.0, 1.5, 0.0], np.float32)

def with_encoding(d_new, enc):
    o = d_new["obs_oracle"].copy()
    W = o[..., -12:].reshape(o.shape[:-1] + (3, 4))
    hidden = np.all(W == (NEW / 10).astype(np.float16), -1)
    W[hidden] = (np.asarray(enc, np.float32) / 10).astype(np.float16)
    o[..., -12:] = W.reshape(o.shape[:-1] + (12,))
    return o, hidden

# sanity: substituting the old encoding into the new suite reproduces the old suite's oracle input exactly
for sc in suites["new"]:
    o, hidden = with_encoding(suites["new"][sc][0], OLD)
    assert np.array_equal(o, suites["old"][sc][0]["obs_oracle"]), sc
    print(sc, "hidden slots per frame (mean)", hidden.sum(-1).mean())
print("substitution reproduces the old oracle input exactly")

ENC = {"old y=0 (on boundary)": OLD, "new y=-1": NEW, "y=-0.5": [0.5, -0.5, 1.5, -0.5], "y=-2": [0.5, -2, 1.5, -2],
       "y=-5": [0.5, -5, 1.5, -5], "below centre y=-1": [4.5, -1, 5.5, -1], "right x=11": [11, 0.5, 11, 1.5],
       "top y=11": [0.5, 11, 1.5, 11], "point at origin": [0, 0, 0, 0], "far (-50,-50)": [-50, -50, -49, -50]}
runs = sys.argv[3].split(",")
for run in runs:
    rec = json.load(open(ROOT + run + "/oracle.json"))
    saved = pickle.load(open(ROOT + run + "/oracle_params.pkl", "rb"))
    nets = {"trained": saved["params"]}
    if "voe_untrained" in rec:
        nets["untrained"] = learner.init(jax.random.PRNGKey(10_000 + rec["seed"]), rec["obs_dim"], learner.LearnerConfig(), True)
    for nname, net in nets.items():
        res = {}
        for ename, enc in ENC.items():
            row = {}
            for sc, (d, crit) in suites["new"].items():
                o, _ = with_encoding(d, enc)
                x = (o.astype(np.float32) - saved["mu"]) / saved["sd"]
                s, ci = my_S(my_errors(net, x.astype(np.float64), np.zeros(x.shape[:2] + (2,))), crit)
                row[sc] = (s, ci)
            res[ename] = row
        stored = rec["voe" if nname == "trained" else "voe_untrained"]
        assert abs(res["new y=-1"]["permanence"][0] - stored["permanence"]["S"]) < 1e-5
        P = [r["permanence"][0] for r in res.values()]
        print(f"\n{run} oracle {nname}: permanence S across encodings min {min(P):+.4f} max {max(P):+.4f} (stored new {stored['permanence']['S']:+.4f})")
        for ename, row in res.items():
            print(f"   {ename:24} " + "  ".join(f"{sc[:4]} {s:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]" for sc, (s, ci) in row.items()))
