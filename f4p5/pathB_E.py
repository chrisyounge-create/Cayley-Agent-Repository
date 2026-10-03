"""Path B on the J_E side: Aut(J_E)-orbits of P1-points of J_E/5J_E (je_setup switches the pipeline to J_E coordinates)."""
import je_setup
import numpy as np, json, sys, time
import f4_p5 as F, pathB as P
rng = np.random.default_rng(int(sys.argv[1])); budget = float(sys.argv[2]); out = sys.argv[3]; t0 = time.time(); n = 0
with open(out, 'a') as fh:
    while time.time() - t0 < budget:
        v = F.random_rank_one(rng); S, _, _ = F.p1_point_through(v, rng)
        if S is None: continue
        try:
            r = P.analyse(S)
        except Exception as ex:
            r = dict(error=repr(ex)[:300])
        r['n'] = n; r['S'] = S.tolist(); fh.write(json.dumps(r) + "\n"); fh.flush(); n += 1
print(f"{n} J_E six-spaces in {time.time()-t0:.0f}s")
