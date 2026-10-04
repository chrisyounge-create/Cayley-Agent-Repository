"""Part IV step 3: iterative kernel solver for the flexible order-one condition on general bimodule structures."""
import numpy as np
from step2 import setup
def solve(n, LL, RR, J, Dof, nparams, param_to_M, tol=1e-8):
    """LL[a], RR[a]: N x N actions for basis a; J: N x N; Dof(M) builds D from the parameter matrix M.
    Returns a basis of parameter vectors satisfying [R_db, L_a] + [R_a, L_db] = 0 for all basis a, b."""
    N = LL[0].shape[0]; nb = len(LL)
    basis = np.eye(nparams)                                 # columns = parameter vectors
    rng = np.random.default_rng(7)
    pairs = [(sum(c*M for c, M in zip(rng.normal(size=nb), LL)), sum(c*M for c, M in zip(rng.normal(size=nb), RR)), None) for _ in range(6)]
    # generic pre-pass: (La, Ra) random combinations with matching coefficients
    gen = []
    for _ in range(6):
        ca, cb = rng.normal(size=nb), rng.normal(size=nb)
        gen.append((sum(x*M for x, M in zip(ca, LL)), sum(x*M for x, M in zip(ca, RR)), sum(x*M for x, M in zip(cb, LL)), sum(x*M for x, M in zip(cb, RR))))
    sched = [(La, Ra, Lb) for (La, Ra, Lb, Rb) in gen] + [(LL[a], RR[a], LL[b]) for b in range(nb) for a in range(nb)]
    for (La, Ra, Lb) in sched:
        if True:
            if basis.shape[1] == 0: return basis
            imgs = []
            for k in range(basis.shape[1]):
                D = Dof(param_to_M(basis[:, k])); X = D @ Lb - Lb @ D; Rdb = J @ X.T @ J
                imgs.append((Rdb @ La - La @ Rdb + Ra @ X - X @ Ra).flatten())
            A = np.array(imgs).T                            # columns: image of each basis vector
            u, s, vt = np.linalg.svd(A, full_matrices=False)
            r = int((s > 1e-7).sum())                      # absolute tolerance: images are O(1) for non-solutions
            kern = vt[r:].T                                 # combinations of basis vectors mapped to zero
            basis = basis @ kern
            if basis.shape[1] == 0: return basis
    return basis
if __name__ == "__main__":
    import sys
    for alg, name in (('O', 'octonions'), ('S', 'sedenions')):
        n, E, T, C, L, R = setup(alg); N = 2*n; Z = np.zeros((n, n))
        Dof = lambda M: np.block([[Z, M.T], [M, Z]]); p2M = lambda v: v.reshape(n, n)
        # Route 1: copy 1 = regular bimodule (L_a, R_a); copy 2 = opposite bimodule (left action R_a, right action L_a)
        LL = [np.block([[L(E[i]), Z], [Z, R(E[i])]]) for i in range(n)]
        RR = [np.block([[R(E[i]), Z], [Z, L(E[i])]]) for i in range(n)]
        # J consistent with these right actions on each copy: conjugation on both (checked below)
        J = np.block([[C, Z], [Z, C]])
        ok = all(np.allclose(RR[i], J @ np.block([[L(C @ E[i]), Z], [Z, R(C @ E[i])]]) @ J, atol=1e-9) for i in range(n))
        ok0 = all(np.allclose(RR[b] @ LL[a] - LL[a] @ RR[b], LL[b] @ RR[a] - RR[a] @ LL[b], atol=1e-9) for a in range(n) for b in range(n))
        basis = solve(n, LL, RR, J, Dof, n*n, p2M)
        print(f"{name}, copy 2 with the opposite bimodule: R = J L_* J consistent: {ok}; flexible order-zero: {ok0}; allowed M: {basis.shape[1]}")
        if basis.shape[1]:
            P = basis
            def inspace(M): v = M.flatten(); return np.linalg.norm(v - P @ (P.T @ v)) < 1e-7 * (np.linalg.norm(v) + 1e-300)
            rng = np.random.default_rng(0); x = rng.normal(size=n)
            print(f"   identity: {inspace(np.eye(n))} | conjugation: {inspace(C)} | L_x: {inspace(L(x))} | R_x: {inspace(R(x))}")
            if alg == 'S':
                import pickle; Zd = np.rint(pickle.load(open('sedenion_zd2.pkl', 'rb'))['Z']).astype(float); Zd = Zd[np.abs(Zd).sum(1) > 0]
                print(f"   L_x, zero divisor x: {inspace(L(Zd[0]))} | R_x: {inspace(R(Zd[0]))}")
            # what is the space? print a basis element's structure
            M0 = p2M(basis[:, 0]); print("   first basis element: symmetric part norm", round(np.linalg.norm(M0 + M0.T) / 2, 3), "| antisymmetric part norm", round(np.linalg.norm(M0 - M0.T) / 2, 3), "| rank", np.linalg.matrix_rank(M0, tol=1e-8))
