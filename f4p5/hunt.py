"""Hunt for rare large-symmetry orbits: sample six-spaces, keep only those whose intersection lattice has |O(M)| > 2e5."""
import numpy as np, json, sys, time, pathB as P, f4_p5 as F
from f4_nbrs_head import hnf_rows
rng = np.random.default_rng(int(sys.argv[1])); budget = float(sys.argv[2]); out = sys.argv[3]; t0 = time.time(); n = kept = 0
with open(out, 'a') as fh:
    while time.time() - t0 < budget:
        v = F.random_rank_one(rng); S, _, _ = F.p1_point_through(v, rng)
        if S is None: continue
        n += 1
        try:
            G0 = P.G0_of(S); B = hnf_rows([list(map(int, g)) for g in G0] + [list(map(int, 5*e)) for e in F.E]); Gram = B @ F.GRAM @ B.T
            order = int(P.pari.qfauto(P.gp_mat(Gram))[0])
        except Exception as ex:
            fh.write(json.dumps({'error': repr(ex)[:200], 'S': S.tolist()}) + "\n"); fh.flush(); continue
        if order > 200000:
            kept += 1; fh.write(json.dumps({'order_OM': order, 'S': S.tolist()}) + "\n"); fh.flush()
    fh.write(json.dumps({'summary': True, 'sampled': n, 'kept': kept}) + "\n")
print(f"{n} sampled, {kept} kept in {time.time()-t0:.0f}s")
