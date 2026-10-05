import numpy as np, itertools, collections, cypari2, time, sys
exec(open('g2_weight77.py').read().split('print("rho77 anti-multiplicative')[0])
t0 = time.time(); G8 = np.array(E8, dtype=np.int64); one = np.array(ONE, dtype=np.int64); omul = lambda x, y: np.einsum('i,j,ijk->k', x, y, S)
pari = cypari2.Pari(); qm = pari.qfminim(pari.matrix(8, 8, [int(x) for x in G8.flatten()]), 2); qM = qm[2]; nc = int(pari.matsize(qM)[1])
units = [np.array([int(qM[i, j]) for i in range(8)]) for j in range(nc)]; units = units + [-u for u in units]; imag = [u for u in units if u @ G8 @ one == 0]
def units_in_span(vs):
    B = np.array(vs, dtype=float); P = B.T @ np.linalg.pinv(B.T); return sum(1 for u in units if np.allclose(P @ u, u))
zds = {}
for a in imag:
    for b in imag:
        if a @ G8 @ b != 0: continue
        n = units_in_span([one, a, b, omul(a, b)])
        if n not in zds: zds[n] = (a, b)
    if len(zds) == 2: break
w8 = (G8 @ one).astype(float); P7 = np.linalg.svd(w8[None, :])[2][1:]
rho7 = lambda g: P7 @ np.array(g, dtype=float).T @ P7.T
# characters of all irreps up to 77 from eigenvalues of g on the 7
def chars(g):
    e = np.linalg.eigvals(rho7(g)); p = lambda k: np.sum(e**k)                       # power sums = chi_7(g^k)
    c7 = p(1); c7_2 = p(2); c7_3 = p(3)
    lam2 = (c7**2 - c7_2) / 2; c14 = lam2 - c7                                         # Λ²7 = 7 + 14
    c27 = c7**2 - 1 - c7 - c14                                                         # 7⊗7 = 1+7+14+27
    c64 = c7 * c14 - c7 - c27                                                          # 7⊗14 = 7+27+64
    sym3 = (c7**3 + 3*c7*c7_2 + 2*c7_3) / 6; c77 = sym3 - c7                           # Sym³7 = 7 + 77
    e14 = np.linalg.eigvals(ad(np.array(g, dtype=float))); c14_2 = np.sum(e14**2); c77p = (c14**2 + c14_2) / 2 - 1 - c27   # Sym²14 = 1+27+77'
    return {'1': 1, '7': c7, '14': c14, '27': c27, '64': c64, '77': c77, "77'": c77p}
print("dim V_pi^{Stab(H_z)} (conic sums factor through this space; 0 => conic sums vanish identically):")
for n, (a, b) in sorted(zds.items()):
    ab = omul(a, b); Hm = np.array([one, a, b, ab], dtype=float); Pr = Hm.T @ np.linalg.pinv(Hm.T); inH = lambda v: np.allclose(Pr @ v, v)
    Sz = [g for g in Gf if inH(a @ g) and inH(b @ g) and inH(ab @ g)]
    tot = collections.defaultdict(complex)
    for g in Sz:
        for k, v in chars(g).items(): tot[k] += v
    dims = {k: round((v / len(Sz)).real, 6) for k, v in tot.items()}
    print(f"   type {n} (|Stab| = {len(Sz)}): {dims}")
