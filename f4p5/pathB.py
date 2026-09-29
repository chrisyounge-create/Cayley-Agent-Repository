"""Path B: exact T(omega_1)(5) at level one on J_Z. For sampled P1-points S: M = G0 + 5J, O(M) (qfauto), Stab(S) = integral Jordan
automorphisms of J in O(M), lift stabilisers and types, and isometry invariants of M for orbit bucketing."""
import numpy as np, json, sys, time, sympy
from fractions import Fraction as Fr
import f4_p5 as F
from f4_nbrs_head import hnf_rows
pari = F.pari; p = F.p; E = F.E; GRAM = F.GRAM
I27 = np.array(F.I27, dtype=np.int64).reshape(-1)[:27] if np.ndim(F.I27) > 1 else np.array(F.I27, dtype=np.int64)
def G0_of(S):
    Sr, piv = F.rref_p(S)
    def red(vv):
        vv = vv % p
        for r, pc in zip(Sr, piv):
            if vv[pc]: vv = (vv - vv[pc]*r) % p
        return vv
    cols = [np.concatenate([red(F.cross(E[j], s)) for s in S]) for j in range(27)]
    return np.array(F.nullspace_modp(np.array(cols).T % p, p), dtype=np.int64)
def gp_mat(A): return pari.matrix(A.shape[0], A.shape[1], [int(x) for x in A.flatten()])
def np_mat(g): return np.array([[int(g[i, j]) for j in range(len(g))] for i in range(len(g))], dtype=object)
def closure(gens):
    gens = [np.array(g, dtype=np.int64) for g in gens]; Id = np.eye(len(gens[0]), dtype=np.int64)
    seen = {Id.tobytes(): Id}; frontier = [Id]
    while frontier and len(seen) < 200000:
        nxt = []
        for X in frontier:
            for g in gens:
                Y = X @ g
                k = Y.tobytes()
                if k not in seen: seen[k] = Y; nxt.append(Y)
        frontier = nxt
    return list(seen.values())
def jordan_stab(B, OM):
    """elements of O(M) (M-coords, columns) that are integral Jordan automorphisms of J (as row-maps on J-coords)."""
    Bs = sympy.Matrix(B.tolist()); Binv = Bs.inv(); D = int(sympy.Matrix(B.tolist()).det()); Badj = np.array((Binv * D).tolist(), dtype=object)
    out = []
    rng = np.random.default_rng(0); X = rng.integers(-2, 3, (3, 27))
    for g in OM:
        A5 = Badj.dot(np.array(g.T, dtype=object)).dot(np.array(B, dtype=object))
        if any(int(x) % D for x in A5.flatten()): continue
        A = (A5 // D).astype(np.int64)
        if not np.array_equal(I27 @ A, I27): continue
        if not all(np.array_equal(F.sharp(x @ A), F.sharp(x) @ A) for x in X): continue
        out.append(A)
    return out
def invariants(Gram):
    v = pari.qfrep(gp_mat(Gram), 4, 1)                  # counts of vectors of norm 1..4 (up to sign): cheap fingerprint
    return tuple(int(x) for x in v)
def analyse(S):
    G0 = G0_of(S)
    B = hnf_rows([list(map(int, g)) for g in G0] + [list(map(int, p*e)) for e in E])
    Gram = B @ GRAM @ B.T
    res = pari.qfauto(gp_mat(Gram)); order = int(res[0])
    OM = closure([np_mat(g) for g in res[1]]) if order <= 200000 else None
    stab = jordan_stab(B, OM) if OM is not None else None
    Ls = F.neighbours_of_S(S); types = [F.typ(L) if F.is_albert(L) else -1 for L in Ls]
    keys = [L.tobytes() for L in Ls]
    lift_stab = []
    for c, L in enumerate(Ls):
        cnt = 0
        for A in stab:
            img = hnf_rows([list(map(int, r)) for r in (L @ A)])
            if img.tobytes() == keys[c]: cnt += 1
        lift_stab.append(cnt)
    return dict(order_OM=order, stab=len(stab), lift_stab=lift_stab, types=types, inv=invariants(Gram))
if __name__ == "__main__":
    rng = np.random.default_rng(int(sys.argv[1])); budget = float(sys.argv[2]); out = sys.argv[3]; t0 = time.time(); n = 0
    with open(out, 'a') as fh:
        while time.time() - t0 < budget:
            v = F.random_rank_one(rng); S, _, _ = F.p1_point_through(v, rng)
            if S is None: continue
            r = analyse(S); r['seed'] = int(sys.argv[1]); r['n'] = n; r['S'] = S.tolist(); fh.write(json.dumps(r) + "\n"); fh.flush(); n += 1
    print(f"{n} six-spaces in {time.time()-t0:.0f}s")
