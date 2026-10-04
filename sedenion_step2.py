"""Part IV step 2: finite Dirac operators on H = A (+) A with gamma = diag(1,-1), A = sedenions or octonions represented on themselves.
D = [[0, M^T],[M, 0]] (real, symmetric, anticommutes with gamma). Flexible-class order-one condition (Boyle-Farnsworth associator reading):
[R_{db}, L_a] + [R_a, L_{db}] = 0 with L_{db} = [D, L_b], R_{db} = J L_{db}^* J^*, J = conjugation on each summand (KO-dim 0)."""
import numpy as np, sys
from sedenion import smul
from exc_alg import STRUCT, CONJ
def setup(alg):
    if alg == 'S':
        n = 16; E = np.eye(n); T = np.array([[smul(E[i], E[j]) for j in range(n)] for i in range(n)])
        from sedenion import ONE; C = np.zeros((16, 16)); 
        for i in range(16):                       # sedenion conjugation (a,b) -> (conj a, -b)
            v = E[i]; C[:, i] = np.concatenate([v[:8] @ CONJ, -v[8:]])
    else:
        n = 8; E = np.eye(n); T = np.array([[np.einsum('i,j,ijk->k', E[i], E[j], STRUCT) for j in range(n)] for i in range(n)]); C = CONJ.T.astype(float)
    L = lambda a: np.einsum('i,ijk->jk', a, T).T; R = lambda a: np.einsum('j,ijk->ik', a, T).T
    return n, E, T, C, L, R
def constraint_jacobian(alg):
    n, E, T, C, L, R = setup(alg); N = 2 * n
    J = np.block([[C, np.zeros((n, n))], [np.zeros((n, n)), C]])
    LL = [np.block([[L(E[i]), np.zeros((n, n))], [np.zeros((n, n)), L(E[i])]]) for i in range(n)]
    RR = [np.block([[R(E[i]), np.zeros((n, n))], [np.zeros((n, n)), R(E[i])]]) for i in range(n)]
    def Dof(M): return np.block([[np.zeros((n, n)), M.T], [M, np.zeros((n, n))]])
    basis = [np.eye(n)[:, k:k+1] @ np.eye(n)[l:l+1, :] for k in range(n) for l in range(n)]
    Gram = np.zeros((n*n, n*n)); rows_total = 0
    # precompute, for each basis M_k and each b, L_{db} and R_{db}
    Ldb = np.zeros((n*n, n, N, N)); Rdb = np.zeros((n*n, n, N, N))
    for k, Mk in enumerate(basis):
        D = Dof(Mk)
        for b in range(n):
            X = D @ LL[b] - LL[b] @ D; Ldb[k, b] = X; Rdb[k, b] = J @ X.T @ J
    for a in range(n):
        for b in range(n):
            # constraint matrix for each basis element: [R_db, L_a] + [R_a, L_db]
            Cm = np.einsum('kij,jl->kil', Rdb[:, b], LL[a]) - np.einsum('ij,kjl->kil', LL[a], Rdb[:, b]) + np.einsum('ij,kjl->kil', RR[a], Ldb[:, b]) - np.einsum('kij,jl->kil', Ldb[:, b], RR[a])
            Jm = Cm.reshape(n*n, N*N)          # rows = basis index k, cols = matrix entries
            Gram += Jm @ Jm.T; rows_total += N*N
    return Gram, setup(alg)
if __name__ == "__main__":
    for alg, name in (('O', 'octonions'), ('S', 'sedenions')):
        Gram, (n, E, T, C, L, R) = constraint_jacobian(alg)
        w, V = np.linalg.eigh(Gram); null = V[:, w < 1e-8 * w.max()]
        print(f"{name}: allowed M (flexible order-one condition alone): dimension {null.shape[1]} of {n*n}")
        mats = [null[:, k].reshape(n, n) for k in range(null.shape[1])]
        def inspace(M):
            v = M.flatten(); return np.linalg.norm(v - null @ (null.T @ v)) < 1e-8 * (np.linalg.norm(v) + 1e-300)
        print(f"   identity (bare common mass) allowed: {inspace(np.eye(n))} | L_x allowed for random x: {inspace(L(np.random.default_rng(1).normal(size=n)))} | R_x: {inspace(R(np.random.default_rng(1).normal(size=n)))}")
        # J-compatibility: M must commute with conjugation C (JD = DJ, KO-dim 0)
        Kc = np.array([ (M @ C - C @ M).flatten() for M in mats]).T       # columns: image of each nullspace basis vector
        if null.shape[1]:
            u, s, vt = np.linalg.svd(Kc, full_matrices=False); kerdim = int((s < 1e-8 * (s.max() if s.size else 1)).sum()) + (null.shape[1] - len(s))
            print(f"   ...and commuting with J: dimension {kerdim}")
        np.save(f'step2_null_{alg}.npy', null)
