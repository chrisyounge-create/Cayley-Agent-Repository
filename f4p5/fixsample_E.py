"""Fixed-point sampling on the J_E side: g-invariant six-spaces for conjugacy-class representatives g of Aut(J_E) (perm rep on 1,640 vectors)."""
import je_setup
import numpy as np, pickle, re, time, sys, collections, json, f4_p5 as F, pathB as P
from neighbors_struct import nullspace_modp
p = 5
V = pickle.load(open('autJE_perm.pkl', 'rb'))['V']
NV = V.shape[0]
reps = [r[:NV] for r in eval(re.sub(r'\s+', '', open('ccE_reps.g').read().replace('\\\n', '').split(':=', 1)[1]).rstrip(';'))]
src = open('fixsample.py').read().split("def mat_of(perm):")[1].split('if __name__ == "__main__":')[0]
Bidx = []
for i in range(len(V)):
    if np.linalg.matrix_rank(V[Bidx + [i]].astype(float)) > len(Bidx): Bidx.append(i)
    if len(Bidx) == 27: break
Binv = np.linalg.inv(V[Bidx].astype(float))
exec("def mat_of(perm):" + src)
if __name__ == "__main__":
    which = [int(x) for x in sys.argv[1].split(',')]; budget = float(sys.argv[2]); rng = np.random.default_rng(int(sys.argv[3]))
    mats = [mat_of(reps[k]) for k in which]; t0 = time.time(); stats = collections.Counter(); out = []
    while time.time() - t0 < budget:
        k = rng.integers(len(mats)); S = sample_invariant(mats[k], rng)
        if S is None: continue
        r = RM_size(S); stats[(which[k], r)] += 1
        if r >= 10: out.append({'cls': which[k], 'RM': r, 'S': S.tolist()})
    print("(class index, |R_M|) counts:", dict(sorted(stats.items())))
    json.dump(out, open(sys.argv[4], 'w')); print(len(out), "large-symmetry candidates saved")
