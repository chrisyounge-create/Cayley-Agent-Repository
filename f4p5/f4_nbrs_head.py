import numpy as np, pickle, time, sys
from albert import sharp, trace, GRAM, I27
from sympy import Matrix
from sympy.matrices.normalforms import hermite_normal_form
def hnf_rows(vecs):
    return np.array(hermite_normal_form(Matrix([list(map(int, r)) for r in vecs]).T).T.tolist(), dtype=np.int64)
p = 2; K = 4; q = p**K
rng = np.random.default_rng(0)
def cross_matrix(v):
    E = np.eye(27, dtype=np.int64); Sv = sharp(v); return np.array([sharp(E[i] + v) - sharp(E[i]) - Sv for i in range(27)])
def solve_mod2(A, b):
    """random solution x of x A ≡ b (mod 2); A: (n, m), b: (m,). None if inconsistent."""
    A = A % 2; b = b % 2; n, m = A.shape
    M = np.concatenate([A.T % 2, b[:, None] % 2], axis=1).astype(np.int64)   # m equations in n unknowns
    piv = []; r = 0
    for c in range(n):
        rows = np.nonzero(M[r:, c])[0]
        if len(rows) == 0: continue
        k = r + rows[0]; M[[r, k]] = M[[k, r]]
        mask = M[:, c].copy(); mask[r] = 0
        M[mask == 1] ^= M[r]
        piv.append(c); r += 1
        if r == m: break
    if r < m and M[r:, n].any(): return None
    x = np.zeros(n, dtype=np.int64); free = [c for c in range(n) if c not in piv]
    x[free] = rng.integers(0, 2, size=len(free))
    for i, c in enumerate(piv):
        x[c] = (M[i, n] - int(M[i, free] @ x[free])) % 2
    return x
def hensel(v0):
    """lift a rank-one trace-zero point mod 2 to v with sharp(v) ≡ 0, trace ≡ 0 mod 16"""
    v = v0.copy() % 2
    for j in range(1, K):
        m = p**j; s = sharp(v); t = trace(v)
        if (s % m).any() or t % m: return None
        A = np.concatenate([cross_matrix(v), np.ones((27, 1), dtype=np.int64)], axis=1)   # u -> (u x v, trace u)
        b = np.concatenate([(-(s // m)) % 2, [(-(t // m)) % 2]])
        u = solve_mod2(A, b)
        if u is None: return None
        v = (v + m * u) % (2*m)
    return v
from neighbors_struct import nullspace_modp
def kernel_mod(M, m):
    """exact Z-lattice {y in Z^n : y M ≡ 0 mod m}, m = 2^j, by iterated lifting from the F_2 kernel"""
    n = M.shape[0]; j = int(round(np.log2(m)))
    B = np.eye(n, dtype=np.int64)                       # basis of L_0 = Z^n
    for t in range(1, j + 1):
        W = (B @ M) // (2**(t-1))                        # rows: (b M)/2^{t-1}, integral for b in L_{t-1}
        ns = nullspace_modp(W.T % 2, 2)                  # c with c W ≡ 0 mod 2
        vecs = [np.array(c, dtype=np.int64) @ B for c in ns] + [2*b for b in B]
        B = hnf_rows(vecs)
    return B
def neighbour(vb):
    Cb = cross_matrix(vb)
    G_m2 = hnf_rows([vb] + [q*e for e in np.eye(27, dtype=np.int64)])
    K2 = kernel_mod(Cb @ Cb, p**3); G_m1 = hnf_rows(list(K2 @ Cb) + [p**3*e for e in np.eye(27, dtype=np.int64)])
    G_0 = kernel_mod(GRAM @ G_m1.T, p**2); G_1 = kernel_mod(GRAM @ np.array([vb]).T, p)
    Lam = hnf_rows(list(G_m2) + list(p*G_m1) + list(p**2*G_0) + list(p**3*G_1) + [q*e for e in np.eye(27, dtype=np.int64)])
    return Lam
def checks(Lam):
    den = p**2; d = Matrix((Lam @ GRAM @ Lam.T).tolist()).det() / den**(2*27)
    Ainv = Matrix(Lam.T.tolist()); okI = all(s.q == 1 for s in Ainv.solve(Matrix((I27*den).tolist())))
    S = sharp(Lam); okS = all(all(x.q == 1 for x in Ainv.solve(Matrix(S[i].tolist())/den)) for i in range(0, 27, 5))
    return d, okI, okS
def classify(Lam):
    """min norm of J' = Lam/4 via LLL on Lam: 1 -> J_Z type, 2 -> J_E type"""
    from sympy.polys.matrices import DomainMatrix
    from sympy import ZZ
    DM = DomainMatrix(Lam.tolist(), Lam.shape, ZZ); Rd = np.array(DM.lll().to_Matrix().tolist(), dtype=np.int64)
    norms = np.einsum('ij,jk,ik->i', Rd, GRAM, Rd); mn = norms.min()
    # also pairwise sums/differences of the shortest few
    idx = np.argsort(norms)[:8]
    for i in idx:
        for j in idx:
            if i < j:
                for s in (1, -1):
                    w = Rd[i] + s*Rd[j]; mn = min(mn, int(w @ GRAM @ w))
    return mn / 16
