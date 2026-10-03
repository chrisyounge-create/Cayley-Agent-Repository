import numpy as np, pickle, re, time, sys, collections, json, f4_p5 as F, pathB as P
from neighbors_struct import nullspace_modp
p = 5
V = pickle.load(open('/home/claude/autJZ_perm.pkl', 'rb'))['V']
reps = [r[:738] for r in eval(re.sub(r'\s+', '', open('/home/claude/cc_reps.g').read().replace('\\\n', '').split(':=', 1)[1]).rstrip(';'))]
Bidx = []; 
for i in range(len(V)):
    if np.linalg.matrix_rank(V[Bidx + [i]].astype(float)) > len(Bidx): Bidx.append(i)
    if len(Bidx) == 27: break
Binv = np.linalg.inv(V[Bidx].astype(float))
def mat_of(perm):
    A = np.rint(Binv @ V[[perm[i] - 1 for i in Bidx]]).astype(np.int64)
    assert np.array_equal(V @ A, V[[x - 1 for x in perm]]); return A
def eig(A, lam):
    return np.array(nullspace_modp(((A - lam * np.eye(27, dtype=np.int64)) % p).T % p, p), dtype=np.int64)
def sample_invariant(A, rng, tries=200):
    Ep, Em = eig(A, 1), eig(A, p - 1)
    spaces = [E for E in (Ep, Em) if len(E)]
    for _ in range(tries):
        E = spaces[rng.integers(len(spaces))]
        x = (rng.integers(0, p, len(E)) @ E) % p
        if int(F.albdet(x)) % p: continue
        v = F.sharp(x) % p
        if not v.any() or int(F.trace(v)) % p: continue
        C = F.cross_matrix(v); F0 = np.array(nullspace_modp(C.T % p, p), dtype=np.int64)
        K2 = nullspace_modp((C @ C).T % p, p); F1 = np.array([(np.array(k) @ C) % p for k in K2], dtype=np.int64)
        F1, _ = F.rref_p(F1) if len(F1) else (np.zeros((0, 27), dtype=np.int64), [])
        for Ed in spaces:
            W = np.array(nullspace_modp(np.vstack([nullspace_modp(F0 % p, p), nullspace_modp(Ed % p, p)]) % p, p), dtype=np.int64) if True else None
            if len(W) == 0: continue
            U = (rng.integers(0, p, (20000, len(W))) @ W) % p
            good = ~(F.sharp(U) % p).any(axis=1) & U.any(axis=1) & ((F.trace(U) % p) == 0)
            for u in U[good][:50]:
                if F.in_span(F1, u): continue
                Cu = F.cross_matrix(u); ns = nullspace_modp(((F1 @ Cu) % p).T % p, p)
                L = [(np.array(c) @ F1) % p for c in ns]; S, _ = F.rref_p(np.array(L + [u], dtype=np.int64))
                if len(S) != 6 or any((F.sharp(S[i]) % p).any() for i in range(6)) or any((F.cross(S[i], S[j]) % p).any() for i in range(6) for j in range(i+1, 6)): continue
                if not np.array_equal(F.rref_p((S @ A) % p)[0], S): continue          # must be g-invariant
                if len(P.G0_of(S)) != 21: continue
                return S
    return None
def RM_size(S):
    G0 = P.G0_of(S); H = np.array(nullspace_modp(G0 % p, p), dtype=np.int64)
    return int((~((V @ H.T) % p).any(axis=1)).sum())
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
