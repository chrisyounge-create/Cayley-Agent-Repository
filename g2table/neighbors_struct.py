"""Structured enumeration of the lambda_1-neighbours of Coxeter's order at p:
null lines u (trace 0, norm 0 mod p) -> null planes <u,v> (uv = vu = 0 mod p, v isotropic)
-> exactly p neighbours per plane, from the p lift classes (U, V) with
n(U) = n(V) = (U,V) = 0 mod p^2 modulo R∩R'.  Count must be p(p^6-1)/(p-1)."""
import numpy as np, itertools, time, pickle, sys
from coxeter_aut import pair, ONE, hnf_basis
from exc_alg import E8, STRUCT
S_ = STRUCT.astype(np.int64)

def nullspace_modp(A, p):
    """basis (rows) of {x : A x = 0 mod p}, A integer matrix (m x n)."""
    A = np.array(A, dtype=np.int64) % p; m, n = A.shape
    piv = []; r = 0
    for c in range(n):
        rows = [i for i in range(r, m) if A[i, c] % p]
        if not rows: continue
        A[[r, rows[0]]] = A[[rows[0], r]]
        inv = pow(int(A[r, c]), -1, p); A[r] = (A[r] * inv) % p
        for i in range(m):
            if i != r and A[i, c]: A[i] = (A[i] - A[i, c] * A[r]) % p
        piv.append(c); r += 1
        if r == m: break
    free = [c for c in range(n) if c not in piv]
    basis = []
    for f in free:
        x = np.zeros(n, dtype=np.int64); x[f] = 1
        for i, c in enumerate(piv): x[c] = (-A[i, f]) % p
        basis.append(x)
    return basis

def rref_key(vs, p):
    A = np.array(vs, dtype=np.int64) % p; m, n = A.shape; r = 0
    for c in range(n):
        rows = [i for i in range(r, m) if A[i, c] % p]
        if not rows: continue
        A[[r, rows[0]]] = A[[rows[0], r]]
        A[r] = (A[r] * pow(int(A[r, c]), -1, p)) % p
        for i in range(m):
            if i != r and A[i, c]: A[i] = (A[i] - A[i, c] * A[r]) % p
        r += 1
        if r == m: break
    return tuple(map(tuple, A[:r]))

def closed_over(Bp, p):
    B = np.array(Bp, dtype=np.int64); Binv = np.linalg.inv(B.astype(float))
    c = (p * ONE.astype(float)) @ Binv
    if not np.allclose(c, np.round(c), atol=1e-7): return False
    P = np.einsum('ia,abc,jb->ijc', B, S_, B).reshape(64, 8).astype(float) / p
    c = P @ Binv
    return np.allclose(c, np.round(c), atol=1e-7)

def neighbours(p):
    t0 = time.time()
    I8 = np.eye(8, dtype=np.int64)
    X = np.array(list(itertools.product(range(p), repeat=8)), dtype=np.int64)[1:]
    tr = (X @ E8 @ ONE) % p
    nrm = (np.einsum('ki,ij,kj->k', X, E8, X) // 2) % p
    iso = X[(tr == 0) & (nrm == 0)]
    lines = {}
    for u in iso:
        k = rref_key([u], p)
        if k not in lines: lines[k] = u
    L = list(lines.values())
    print(f"p={p}: null lines {len(L)} (expected {(p**6-1)//(p-1)})  [{time.time()-t0:.0f}s]", flush=True)
    planes = {}
    for u in L:
        Lu = np.einsum('a,abc->bc', u, S_)     # v -> u v : (uv)_c = sum_b v_b Lu[b,c]
        Ru = np.einsum('b,abc->ac', u, S_)     # v -> v u
        A = np.concatenate([Lu.T, Ru.T], axis=0)   # rows: conditions on v
        ann = nullspace_modp(A, p)
        # isotropic, trace-0 vectors in ann not proportional to u
        for coeffs in itertools.product(range(p), repeat=len(ann)):
            v = sum(c * b for c, b in zip(coeffs, ann)) % p
            if not v.any(): continue
            if (int(v @ E8 @ ONE)) % p: continue
            if (int(v @ E8 @ v) // 2) % p: continue
            k = rref_key([u, v], p)
            if len(k) < 2: continue
            if k not in planes: planes[k] = (u, v)
    print(f"p={p}: null planes {len(planes)} (expected {(p**6-1)//(p-1)})  [{time.time()-t0:.0f}s]", flush=True)
    found = {}
    for (u, v) in planes.values():
        fu, fv = (E8 @ u) % p, (E8 @ v) % p
        # w_u, w_v with (w_u,u)=1,(w_u,v)=0 ; (w_v,u)=0,(w_v,v)=1  mod p
        Sb = nullspace_modp(np.stack([fu, fv]), p)                 # 6 vectors
        # find w's by solving the 2x8 system
        Mx = np.stack([fu, fv])
        def solve_w(target):
            Aug = np.concatenate([Mx, (-target.reshape(2, 1)) % p], axis=1)
            for x in nullspace_modp(Aug, p):
                if x[8] % p:
                    inv = pow(int(x[8]), -1, p)
                    return (x[:8] * inv) % p
            raise RuntimeError
        wu = solve_w(np.array([1, 0])); wv = solve_w(np.array([0, 1]))
        nu = (int(u @ E8 @ u) // 2); tu = ((-nu // p) if nu % p == 0 else None)
        U0 = u + p * (((-(nu // p)) % p) * wu)       # n(U0) = 0 mod p^2
        nv = (int(v @ E8 @ v) // 2)
        V0 = v + p * (((-(nv // p)) % p) * wv)
        assert (int(U0 @ E8 @ U0) // 2) % (p*p) == 0 and (int(V0 @ E8 @ V0) // 2) % (p*p) == 0
        c = (-(int(U0 @ E8 @ V0) // p)) % p
        Sint = [[int(t) for t in r] for r in Sb] + [[p*int(t) for t in r] for r in I8]
        for b in range(p):
            U = U0 + p * b * wv
            V = V0 + p * ((c - b) % p) * wu
            assert (int(U @ E8 @ V)) % (p*p) == 0
            Bp = hnf_basis([[p*t for t in r] for r in Sint] + [list(map(int, U)), list(map(int, V))])
            if len(Bp) != 8 or abs(round(np.linalg.det(np.array(Bp, dtype=float)))) != p**8: continue
            if closed_over(Bp, p): found[Bp] = (U, V)
    print(f"p={p}: neighbours {len(found)} (expected {p*(p**6-1)//(p-1)})  [{time.time()-t0:.0f}s]", flush=True)
    return found

if __name__ == "__main__":
    p = int(sys.argv[1])
    nb = neighbours(p)
    pickle.dump(nb, open(f'nbrs{p}.pkl', 'wb'))
