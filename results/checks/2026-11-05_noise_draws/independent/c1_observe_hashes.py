"""Hash every preset's observations of the saved inputs, using whichever umwelt is on sys.path."""
import sys, json, hashlib, pickle, tempfile, os, inspect
import numpy as np
import umwelt
from umwelt import presets
inp, out = sys.argv[1], sys.argv[2]
with open(inp, "rb") as f:
    D = pickle.load(f)
has_draw = "draw" in inspect.signature(presets.observe).parameters
res = {"umwelt": umwelt.__file__, "has_draw": has_draw, "h": {}}
def h(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:20] + f"/{x.dtype}/{x.shape}"
for name, p in presets.PRESETS.items():
    for test, (d, crit) in D["suite"].items():
        res["h"][f"suite/{test}/{name}"] = h(presets.observe(d, p))
        if has_draw:
            res["h"][f"suite/{test}/{name}/draw0"] = h(presets.observe(d, p, draw=0))
    for w, d in D["rec"].items():
        res["h"][f"rec/{w}/{name}/f32"] = h(presets.observe(d, p))
        with tempfile.TemporaryDirectory() as t:
            m = presets.observe(d, p, dtype=np.float16, path=os.path.join(t, "c.npy"))
            res["h"][f"rec/{w}/{name}/f16memmap"] = h(np.asarray(m))
            del m
        if has_draw:
            res["h"][f"rec/{w}/{name}/f32/draw0"] = h(presets.observe(d, p, draw=0))
json.dump(res, open(out, "w"), indent=1)
print(res["umwelt"], has_draw, len(res["h"]))
