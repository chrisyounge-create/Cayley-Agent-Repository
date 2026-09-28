"""F4 level one at p = 5: sample 6-dim sharp-null subspaces S of J/5J (P1-points), build the omega_1 neighbour
J' = G0 + 5J + S~/5, test it is an Albert lattice, identify J_Z vs J_E."""
import numpy as np, pickle, time, sys
from albert import sharp, GRAM, I27, det as albdet, trace
from neighbors_struct import nullspace_modp
from sympy import Matrix
from f4_nbrs_head import hnf_rows
import cypari2; pari = cypari2.Pari(); pari.allocatemem(600*10**6)
p = 5; E = np.eye(27, dtype=np.int64)
def cross(a, b): return sharp(a + b) - sharp(a) - sharp(b)
def cross_matrix(v): Sv = sharp(v); return np.array([sharp(E[i] + v) - sharp(E[i]) - Sv for i in range(27)]) % p
def rref_p(rows):
    A = np.array(rows, dtype=np.int64) % p; m, n = A.shape; r = 0; piv = []
    for c in range(n):
        rr = [i for i in range(r, m) if A[i, c]]
        if not rr: continue
        A[[r, rr[0]]] = A[[rr[0], r]]; A[r] = (A[r] * pow(int(A[r, c]), -1, p)) % p
        for i in range(m):
            if i != r and A[i, c]: A[i] = (A[i] - A[i, c] * A[r]) % p
        piv.append(c); r += 1
        if r == m: break
    return A[:r], piv
def in_span(B, v):                                    # is v in the row span of B mod p?
    R, _ = rref_p(B); R2, _ = rref_p(np.vstack([B, v[None, :]])); return len(R2) == len(R)
def random_rank_one(rng):
    while True:
        x = rng.integers(0, p, 27).astype(np.int64)
        if int(albdet(x)) % p == 0:
            v = sharp(x) % p
            if v.any() and int(trace(v)) % p == 0: return v
def p1_point_through(v, rng, batch=400000):
    """A P1-point through the rank-one trace-zero v: S = <v,u> + L_u with u rank-one, TRACE ZERO, in F0(v) \\ F1(v)."""
    C = cross_matrix(v); F0 = np.array(nullspace_modp(C.T % p, p), dtype=np.int64)
    K2 = nullspace_modp((C @ C).T % p, p); F1 = np.array([(np.array(k) @ C) % p for k in K2], dtype=np.int64)
    F1, _ = rref_p(F1) if len(F1) else (np.zeros((0, 27), dtype=np.int64), [])
    for _ in range(5):
        U = (rng.integers(0, p, (batch, len(F0))) @ F0) % p
        good = ~(sharp(U) % p).any(axis=1) & U.any(axis=1) & ((trace(U) % p) == 0)
        for u in U[good]:
            if in_span(F1, u): continue
            Cu = cross_matrix(u); ns = nullspace_modp(((F1 @ Cu) % p).T % p, p)
            L = [(np.array(c) @ F1) % p for c in ns]
            S, _ = rref_p(np.array(L + [u], dtype=np.int64))
            if len(S) != 6: continue
            if any((sharp(S[i]) % p).any() for i in range(6)): continue
            if any((cross(S[i], S[j]) % p).any() for i in range(6) for j in range(i+1, 6)): continue
            return S, len(F0), len(F1)
    return None, len(F0), len(F1)
def solve_modp(A, b):
    """x with A x ≡ b (mod p), A: m x n. Returns (particular solution, nullspace basis) or (None, None)."""
    A = A % p; b = b % p; m, n = A.shape
    M = np.concatenate([A, b[:, None]], axis=1).astype(np.int64); r = 0; piv = []
    for c in range(n):
        rows = np.nonzero(M[r:, c])[0]
        if len(rows) == 0: continue
        k = r + rows[0]; M[[r, k]] = M[[k, r]]; M[r] = (M[r] * pow(int(M[r, c]), -1, p)) % p
        for i in range(m):
            if i != r and M[i, c]: M[i] = (M[i] - M[i, c] * M[r]) % p
        piv.append(c); r += 1
        if r == n: break
    if (M[r:, n] % p).any(): return None, None
    x = np.zeros(n, dtype=np.int64)
    for i, c in enumerate(piv): x[c] = M[i, n]
    return x, nullspace_modp(A, p)
def neighbour_from_S(S, rng, tries=1):
    # G0 = {x : x cross s_i in span(S) mod p for all i}
    Sr, pivots = rref_p(S)
    def reduce_mod_S(vv):
        vv = vv % p
        for r, pc in zip(Sr, pivots):
            if vv[pc]: vv = (vv - vv[pc] * r) % p
        return vv
    cols = [np.concatenate([reduce_mod_S(cross(E[j], s)) for s in S]) for j in range(27)]
    A = np.array(cols, dtype=np.int64)
    G0 = np.array(nullspace_modp(A.T % p, p), dtype=np.int64)
    # lift: S~_i = S_i + p T_i with sharp(Σ c_i S~_i) ≡ 0 mod p^2 for all c:
    #   sharp(s_i)/p + s_i x t_i ≡ 0,   (s_i x s_j)/p + s_i x t_j + s_j x t_i ≡ 0   (mod p)
    Cse = np.array([[cross(S[i], E[k]) for k in range(27)] for i in range(6)])   # s_i x e_k
    unknown = lambda j, k: 27*j + k                                             # t_j = Σ_k T[j,k] e_k
    rows, rhs = [], []
    for i in range(6):
        for coord in range(27):
            row = np.zeros(162, dtype=np.int64)
            for k in range(27): row[unknown(i, k)] += Cse[i, k][coord]
            rows.append(row % p); rhs.append((-(sharp(S[i])[coord] // p)) % p)
    for i in range(6):
        for j in range(i+1, 6):
            cij = cross(S[i], S[j])
            for coord in range(27):
                row = np.zeros(162, dtype=np.int64)
                for k in range(27): row[unknown(j, k)] += Cse[i, k][coord]; row[unknown(i, k)] += Cse[j, k][coord]
                rows.append(row % p); rhs.append((-(cij[coord] // p)) % p)
    x0, ns = solve_modp(np.array(rows), np.array(rhs))
    if x0 is None: return None, "no lift mod 25"
    out = []
    for tr in range(tries):
        x = x0.copy()
        if ns and tr: x = (x + (rng.integers(0, p, len(ns)) @ np.array(ns))) % p
        T = x.reshape(6, 27); St = S + p * T
        gens = [list(map(int, p*g)) for g in G0] + [list(map(int, p*p*e)) for e in E] + [list(map(int, s)) for s in St]   # basis of p J'
        Lam = hnf_rows(gens)
        out.append(Lam)
    return out, (len(ns) if ns else 0)
def is_albert(Lam, s=p):
    """Lam is a basis of s*J'; J' is an Albert lattice iff det = s^54, I ∈ J', sharp(J') ⊂ J'."""
    if Matrix((Lam @ GRAM @ Lam.T).tolist()).det() != s**54: return False
    Ainv = Matrix(Lam.T.tolist()); Sh = sharp(Lam)
    if not all(q.q == 1 for q in Ainv.solve(Matrix((I27*s).tolist()))): return False
    return all(all(x.q == 1 for x in Ainv.solve(Matrix(Sh[i].tolist())/s)) for i in range(27))
def typ(Lam, s=p): return int(pari.qfminim(pari.matrix(27, 27, (Lam @ GRAM @ Lam.T).flatten().tolist()), s*s, 0, 2)[0]) // 2
if __name__ == "__main__" and not (len(sys.argv) > 3 and sys.argv[3] == "five"):
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0); t0 = time.time()
    for trial in range(int(sys.argv[2]) if len(sys.argv) > 2 else 3):
        v = random_rank_one(rng); S, f0, f1 = p1_point_through(v, rng)
        if S is None: print(f"trial {trial}: no P1-point found (dim F0={f0}, F1={f1})", flush=True); continue
        res, nsdim = neighbour_from_S(S, rng, tries=int(sys.argv[3]) if len(sys.argv) > 3 else 40)
        if res is None: print(f"trial {trial}: {nsdim}", flush=True); continue
        lifts = {}
        for Lam in res:
            key = Lam.tobytes()
            if key in lifts: continue
            lifts[key] = typ(Lam) if is_albert(Lam) else None
        alb = [t for t in lifts.values() if t is not None]
        print(f"trial {trial}: F0={f0}, F1={f1}; lift-space dim {nsdim}; distinct lattices {len(lifts)}, Albert among them {len(alb)} (expect 5), types {alb}  [{time.time()-t0:.0f}s]", flush=True)

def neighbours_of_S(S):
    """The p lattices p*J' attached to the P1-point S (all lifts modulo the directions that do not change J')."""
    Sr, pivots = rref_p(S)
    def reduce_mod_S(vv):
        vv = vv % p
        for r, pc in zip(Sr, pivots):
            if vv[pc]: vv = (vv - vv[pc] * r) % p
        return vv
    cols = [np.concatenate([reduce_mod_S(cross(E[j], s)) for s in S]) for j in range(27)]
    G0 = np.array(nullspace_modp(np.array(cols).T % p, p), dtype=np.int64)
    Cse = np.array([[cross(S[i], E[k]) for k in range(27)] for i in range(6)])
    unknown = lambda j, k: 27*j + k
    rows, rhs = [], []
    for i in range(6):
        for coord in range(27):
            row = np.zeros(162, dtype=np.int64)
            for k in range(27): row[unknown(i, k)] += Cse[i, k][coord]
            rows.append(row % p); rhs.append((-(sharp(S[i])[coord] // p)) % p)
    for i in range(6):
        for j in range(i+1, 6):
            cij = cross(S[i], S[j])
            for coord in range(27):
                row = np.zeros(162, dtype=np.int64)
                for k in range(27): row[unknown(j, k)] += Cse[i, k][coord]; row[unknown(i, k)] += Cse[j, k][coord]
                rows.append(row % p); rhs.append((-(cij[coord] // p)) % p)
    x0, ns = solve_modp(np.array(rows), np.array(rhs))
    if x0 is None: return None
    ns = np.array(ns, dtype=np.int64)
    trivial = np.array([np.concatenate([G0[a] if b == i else np.zeros(27, dtype=np.int64) for b in range(6)]) for i in range(6) for a in range(len(G0))])
    r_tr = len(rref_p(trivial)[0]); w = None
    for cand in ns:
        if len(rref_p(np.vstack([trivial, cand[None, :]]))[0]) > r_tr: w = cand; break
    if w is None: return None
    out = []
    for c in range(p):
        T = ((x0 + c * w) % p).reshape(6, 27); St = S + p * T
        gens = [list(map(int, p*g)) for g in G0] + [list(map(int, p*p*e)) for e in E] + [list(map(int, s)) for s in St]
        out.append(hnf_rows(gens))
    return out
if __name__ == "__main__" and len(sys.argv) > 3 and sys.argv[3] == "five":
    rng = np.random.default_rng(int(sys.argv[1])); t0 = time.time()
    for trial in range(int(sys.argv[2])):
        v = random_rank_one(rng); S, f0, f1 = p1_point_through(v, rng)
        if S is None: print(f"trial {trial}: no P1-point"); continue
        Ls = neighbours_of_S(S)
        if Ls is None: print(f"trial {trial}: lift failed"); continue
        keys = {L.tobytes() for L in Ls}
        types = [typ(L) if is_albert(L) else None for L in Ls]
        print(f"trial {trial}: {len(keys)} distinct neighbours; Albert: {sum(t is not None for t in types)}/5; types {types}  [{time.time()-t0:.0f}s]", flush=True)

if __name__ == "__main__" and len(sys.argv) > 3 and sys.argv[3] == "sample":
    # sampling run: argv[1] = seed, argv[2] = seconds budget; appends one JSON line per point to argv[4]
    import json
    rng = np.random.default_rng(int(sys.argv[1])); budget = float(sys.argv[2]); out = sys.argv[4]; t0 = time.time(); n = 0
    with open(out, 'a') as fh:
        while time.time() - t0 < budget:
            v = random_rank_one(rng); S, f0, f1 = p1_point_through(v, rng)
            if S is None: continue
            Ls = neighbours_of_S(S)
            if Ls is None: continue
            types = [typ(L) if is_albert(L) else -1 for L in Ls]
            fh.write(json.dumps({'seed': int(sys.argv[1]), 'n': n, 'types': types}) + "\n"); fh.flush(); n += 1
    print(f"seed {sys.argv[1]}: {n} points in {time.time()-t0:.0f}s")
