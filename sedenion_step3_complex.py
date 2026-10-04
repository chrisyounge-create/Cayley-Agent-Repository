import numpy as np, pickle, sys
from sedenion import smul
from exc_alg import CONJ
from step3 import solve
# complex sedenions as a real 32-dim algebra: x = (a, b) means a + i b
E16 = np.eye(16); C16 = np.zeros((16, 16))
for k in range(16): v = E16[k]; C16[:, k] = np.concatenate([v[:8] @ CONJ, -v[8:]])
def cmul(x, y):
    a, b = x[:16], x[16:]; c, d = y[:16], y[16:]
    return np.concatenate([smul(a, c) - smul(b, d), smul(a, d) + smul(b, c)])
E = np.eye(32); T = np.array([[cmul(E[i], E[j]) for j in range(32)] for i in range(32)])
L = lambda x: np.einsum('i,ijk->jk', x, T).T; R = lambda x: np.einsum('j,ijk->ik', x, T).T
Cstar = np.block([[C16, np.zeros((16, 16))], [np.zeros((16, 16)), -C16]])          # (a + ib)* = conj(a) - i conj(b)
rng = np.random.default_rng(0)
x, y = rng.normal(size=32), rng.normal(size=32)
print("complex sedenions: * is an anti-automorphism:", np.allclose(Cstar @ cmul(x, y), cmul(Cstar @ y, Cstar @ x)), "| flexible:", all(np.allclose(cmul(cmul(p, q), p), cmul(p, cmul(q, p))) for p, q in (rng.normal(size=(2, 32)) for _ in range(10))))
# commutant of the left action on one copy
Ls = [L(E[i]) for i in range(32)]
pass
pass
print("commutant of the left action on one copy: 2 (complex scalars; computed earlier)")
n = 32; Z = np.zeros((n, n)); Dof = lambda M: np.block([[Z, M.T], [M, Z]]); p2M = lambda v: v.reshape(n, n)
LL = [np.block([[L(E[i]), Z], [Z, L(E[i])]]) for i in range(n)]
for jname, J in (("J = * on each copy", np.block([[Cstar, Z], [Z, Cstar]])), ("J swapping the copies", np.block([[Z, Cstar], [Cstar, Z]]))):
    RR = [J @ np.block([[L(Cstar @ E[i]), Z], [Z, L(Cstar @ E[i])]]) @ J for i in range(n)]
    ok0 = all(np.allclose(RR[b] @ LL[a] - LL[a] @ RR[b], LL[b] @ RR[a] - RR[a] @ LL[b], atol=1e-9) for a in range(n) for b in range(n))
    basis = solve(n, LL, RR, J, Dof, n*n, p2M)
    print(f"{jname}: flexible order-zero holds: {ok0}; allowed real-linear M: {basis.shape[1]}")
    if basis.shape[1]:
        P = basis
        def inspace(M): v = M.flatten(); return np.linalg.norm(v - P @ (P.T @ v)) < 1e-7 * (np.linalg.norm(v) + 1e-300)
        I32 = np.eye(32); Imul = L(E[16])                   # multiplication by i
        print(f"   identity: {inspace(I32)} | multiplication by i: {inspace(Imul)} | complex conjugation (a+ib -> a-ib): {inspace(np.block([[np.eye(16), np.zeros((16,16))],[np.zeros((16,16)), -np.eye(16)]]))} | *: {inspace(Cstar)} | L_x: {inspace(L(x))} | R_x: {inspace(R(x))}")
        for k in range(min(basis.shape[1], 4)):
            M = p2M(basis[:, k]); comm_i = np.linalg.norm(M @ Imul - Imul @ M); anti_i = np.linalg.norm(M @ Imul + Imul @ M)
            print(f"   basis element {k}: ||[M, i]|| = {comm_i:.3f}, ||{{M, i}}|| = {anti_i:.3f} (0 = complex-linear / 0 = complex-antilinear), rank {np.linalg.matrix_rank(M, tol=1e-8)}")
