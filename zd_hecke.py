"""Build the Hecke algebra of the sedenion zero-divisor manifold G2/K: K = stabiliser of a zero divisor (a,b), its action on the
4-dim annihilator, and the multiplicities m_pi = dim pi^K for the G2 irreps 1, 7, 14, 27, 64, 77, 77'."""
import numpy as np, pickle, itertools
from exc_alg import STRUCT, E8
from sedenion import smul
omul = lambda x, y: np.einsum('i,j,ijk->k', x, y, STRUCT); one = np.eye(8)[5]
ip = lambda x, y: float(x @ E8 @ y) / 2
# derivations of the octonions (14), as 8x8 matrices in column convention
rows = []
E = np.eye(8); T = np.array([[omul(E[i], E[j]) for j in range(8)] for i in range(8)])
for i in range(8):
    for j in range(8):
        M = np.zeros((8, 8, 8)); prod = T[i, j]
        for c in range(8): M[:, c, :] += prod[c] * np.eye(8)
        for b in range(8): M[:, i, b] -= T[b, j]; M[:, j, b] -= T[i, b]
        rows.append(M.reshape(8, 64))
u, s, vt = np.linalg.svd(np.vstack(rows), full_matrices=True); ders = [vt[k].reshape(8, 8).T for k in range(np.sum(s > 1e-9), 64)]
assert len(ders) == 14
# an orthonormal imaginary basis (units of the integral octonions), from the earlier search
d = pickle.load(open('sedenion_zd2.pkl', 'rb')); octs = np.rint(d['octs']).astype(np.int64); octs = octs[np.abs(octs).sum(1) > 0]
imag = [u for u in octs if abs(ip(u, one)) < 1e-9]
def find_std():
    for a in imag:
        for b in imag:
            if abs(ip(a, b)) > 1e-9: continue
            c = omul(a, b)
            for dd in imag:
                if abs(ip(dd, a)) > 1e-9 or abs(ip(dd, b)) > 1e-9 or abs(ip(dd, c)) > 1e-9: continue
                cand = [a, b, c, dd, omul(a, dd), omul(b, dd), omul(c, dd)]
                if all(abs(ip(cand[i], cand[j]) - (1.0 if i == j else 0.0)) < 1e-9 for i in range(7) for j in range(7)): return cand
std = find_std(); B = np.array([one] + std, dtype=float)          # rows: orthonormal basis 1, i1..i7 (w.r.t. the octonion norm)
a, b = std[0], std[1]
# change of basis: coordinates x (Coxeter) -> y = B E8 x / 2 ... use the inner product to get orthonormal coordinates
def to_std(x): return np.array([ip(x, Bi) for Bi in B])          # orthonormal coordinates
P = np.array([to_std(E[i]) for i in range(8)]).T                # 8x8: Coxeter coords -> std coords (column convention)
Pinv = np.linalg.inv(P)
ders_std = [P @ D @ Pinv for D in ders]                          # derivations in the orthonormal basis
seven = [D[1:, 1:] for D in ders_std]                            # the 7-dim representation (derivations kill 1)
print("derivations antisymmetric in the orthonormal basis:", all(np.allclose(X + X.T, 0, atol=1e-8) for X in seven))
# K = stabiliser of the zero divisor (a, b): derivations killing a and b
A = np.array([np.concatenate([D @ a, D @ b]) for D in ders]).T  # 16 x 14
u, s, vt = np.linalg.svd(A, full_matrices=True); K = [sum(c * D for c, D in zip(vt[k], ders)) for k in range(np.sum(s > 1e-9), 14)]
print("dim of the stabiliser K of a zero divisor:", len(K), " (expected 3 = su(2))")
Kstd = [P @ X @ Pinv for X in K]
H = np.array([to_std(one), to_std(a), to_std(b), to_std(omul(a, b))])      # the quaternion subalgebra <1,a,b,ab>
print("K fixes the quaternion subalgebra pointwise:", all(np.allclose(X @ h, 0, atol=1e-8) for X in Kstd for h in H))
# K on the annihilator of the zero divisor x = (a, b) in the sedenions
x = np.concatenate([a, b]); E16 = np.eye(16); T16 = np.array([[smul(E16[i], E16[j]) for j in range(16)] for i in range(16)])
Lx = np.einsum('i,ijk->jk', x, T16).T; u, s, vt = np.linalg.svd(Lx); ker = vt[np.sum(s > 1e-9):].T
print("annihilator dimension:", ker.shape[1])
K16 = [np.block([[X, np.zeros((8, 8))], [np.zeros((8, 8)), X]]) for X in K]
inv_ker = all(np.linalg.norm(ker @ np.linalg.lstsq(ker, X @ ker, rcond=None)[0] - X @ ker) < 1e-8 for X in K16)
# commutant of K on the kernel: 1 = irreducible real, 4 = quaternionic (H), 2 = complex
Kk = [np.linalg.lstsq(ker, X @ ker, rcond=None)[0] for X in K16]
C = np.vstack([np.kron(Y.T, np.eye(4)) - np.kron(np.eye(4), Y) for Y in Kk]); sv = np.linalg.svd(C, compute_uv=False)
print("K preserves the annihilator:", inv_ker, "| commutant of K on it has dimension", int((sv < 1e-9).sum()), "(4 = quaternionic: the annihilator is a copy of H with K = Sp(1))")
pickle.dump({'seven': seven, 'K7': [X[1:, 1:] for X in Kstd]}, open('zd_hecke_reps.pkl', 'wb'))
