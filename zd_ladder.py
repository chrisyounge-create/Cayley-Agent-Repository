"""Unit-half zero divisors of the integral 2^k-ions (Cayley-Dickson ladder over the Coxeter octonions), any rung.
usage: zd_ladder.py K j njobs    (K = log2 dimension, e.g. 7 for the 128-ions; anchors sliced j::njobs). Writes zd_ladder_K_j.json."""
import numpy as np, cypari2, time, json, sys, collections
from exc_alg import STRUCT, CONJ, E8
K, j, nj = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
G8 = np.array(E8, dtype=np.int64); S = np.array(STRUCT, dtype=np.int64); C = np.array(CONJ, dtype=np.int64)
pari = cypari2.Pari(); pm = lambda A: pari.matrix(A.shape[0], A.shape[1], [int(x) for x in A.flatten()])
M = pari.qfminim(pm(G8), 2)[2]; U = np.array([[int(M[i, jj]) for i in range(8)] for jj in range(int(pari.matsize(M)[1]))], dtype=np.int64); U = np.vstack([U, -U])
Lo = lambda x: np.einsum('i,ijk->jk', x, S); Ro = lambda y: np.einsum('j,ijk->ik', y, S)
def LR(n, X):
    """left and right multiplication matrices (row action: q @ L = X q, p @ R = p X) of an element X of the 2^n-dimensional algebra, and its conjugation J."""
    if n == 3: return Lo(X), Ro(X), C
    h = 2 ** (n - 1); x, y = X[:h], X[h:]; Lx, Rx, Jh = LR(n - 1, x); Ly, Ry, _ = LR(n - 1, y)
    L = np.block([[Lx, Jh @ Ly], [-(Jh @ Ry), Rx]]); R = np.block([[Rx, Lx if False else None, ], [None, None]]) if False else None
    # right multiplication: p @ R(Q) = p Q with Q = (z, w): (x,y)(z,w) = (xz - conj(w) y, w x + y conj(z))
    Lz, Rz, _ = LR(n - 1, x); 
    return L, None, np.block([[Jh, np.zeros((h, h), dtype=np.int64)], [np.zeros((h, h), dtype=np.int64), -np.eye(h, dtype=np.int64)]])
def Lmat(n, X):
    if n == 3: return Lo(X)
    h = 2 ** (n - 1); x, y = X[:h], X[h:]; Jh = Jmat(n - 1)
    return np.block([[Lmat(n - 1, x), Jh @ Lmat(n - 1, y)], [-(Jh @ Rmat(n - 1, y)), Rmat(n - 1, x)]])
def Rmat(n, Q):
    if n == 3: return Ro(Q)
    h = 2 ** (n - 1); z, w = Q[:h], Q[h:]; Jh = Jmat(n - 1)
    return np.block([[Rmat(n - 1, z), Lmat(n - 1, w)], [-Lmat(n - 1, w @ Jh), Rmat(n - 1, z @ Jh)]])
def Jmat(n):
    if n == 3: return C
    h = 2 ** (n - 1); return np.block([[Jmat(n - 1), np.zeros((h, h), dtype=np.int64)], [np.zeros((h, h), dtype=np.int64), -np.eye(h, dtype=np.int64)]])
n = K - 1; h = 2 ** n; m = h // 8                                  # units of the previous rung: m octonion slots
units = np.zeros((m * 240, h), dtype=np.int64)
for k in range(m): units[k*240:(k+1)*240, 8*k:8*k+8] = U
one8 = np.zeros(8, dtype=np.int64); one8[5] = 1
imag = np.array([(i // 240 > 0) or int(U[i % 240] @ G8 @ one8) == 0 for i in range(len(units))])
Gh = np.kron(np.eye(m, dtype=np.int64), G8); IP = units @ Gh @ units.T
t0 = time.time(); Lm = np.array([Lmat(n, v) for v in units], dtype=np.float32); Rm = np.array([Rmat(n, v) for v in units], dtype=np.float32); Jh = Jmat(n).astype(np.float32)
JL = np.einsum('ij,njk->nik', Jh, Lm); JR = np.einsum('ij,njk->nik', Jh, Rm); print(f"matrices ready: {len(units)} units [{time.time()-t0:.0f}s]", flush=True)
total = 0; law_ok = True; census = collections.Counter(); anchors = list(range(j, len(units), nj)); done = 0
for i in anchors:
    B = np.zeros((len(units), 2 * h, 2 * h), dtype=np.float32); B[:, :h, :h] = Lm[i]; B[:, h:, h:] = Rm[i]; B[:, :h, h:] = JL; B[:, h:, :h] = -JR
    sv = np.linalg.svd(B, compute_uv=False); nul = (sv < 1e-3 * sv[:, 0:1]).sum(1); zd = nul > 0
    law = imag[i] & imag & (IP[i] == 0)
    if not np.array_equal(zd, law): law_ok = False
    total += int(zd.sum()); slot_i = i // 240
    for jj in np.nonzero(zd)[0]: census[f"{slot_i},{jj // 240},{int(nul[jj])}"] += 1
    done += 1
    if done % 20 == 0: print(f"anchor {done}/{len(anchors)}: running total {total} [{time.time()-t0:.0f}s]", flush=True); json.dump({'K': K, 'j': j, 'nj': nj, 'anchors_done': done, 'total': total, 'law_ok': law_ok, 'census': census}, open(f'zd_ladder_{K}_{j}.json', 'w'))
json.dump({'K': K, 'j': j, 'nj': nj, 'anchors_done': done, 'total': total, 'law_ok': law_ok, 'census': census}, open(f'zd_ladder_{K}_{j}.json', 'w'))
print(f"done: K={K} slice {j}/{nj}: {total} zero divisors over {done} anchors; law exact: {law_ok}  [{time.time()-t0:.0f}s]")
