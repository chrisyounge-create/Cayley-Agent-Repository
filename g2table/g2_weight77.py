exec(open('g2_weight27.py').read().split("triv = lambda g: np.eye(1)")[0])
# adjoint rep on derivations, right action D -> g^{-1} D g ; basis Ders (14)
Dstack = np.array([D.ravel() for D in Ders]).T; Dpinv = np.linalg.pinv(Dstack)
def ad(g): gi = np.linalg.inv(g); return Dpinv @ np.array([(gi @ D @ g).ravel() for D in Ders]).T
def ad_der(X): return Dpinv @ np.array([(D @ X - X @ D).ravel() for D in Ders]).T
pairs14 = [(i, j) for i in range(14) for j in range(i, 14)]; pidx = {t: k for k, t in enumerate(pairs14)}
full = np.array([pidx[tuple(sorted(x))] for x in itertools.product(range(14), repeat=2)])
def sym2m(A, der=False):
    Sm = np.zeros((105, 105)); I = np.eye(14)
    for k, (i, j) in enumerate(pairs14):
        T = (np.outer(A[:, i], I[:, j]) + np.outer(I[:, i], A[:, j])).ravel() if der else np.outer(A[:, i], A[:, j]).ravel()
        Sm[:, k] = np.bincount(full, weights=T, minlength=105)
    return Sm
DM2 = [sym2m(ad_der(D), der=True) for D in Ders]
Cas2 = sum(Bki[a, b] * DM2[a] @ DM2[b] for a in range(14) for b in range(14))
ev2 = np.linalg.eigvals(Cas2).real; print("Casimir on Sym²(adjoint):", {v: int(np.sum(np.abs(ev2 - v) < 1e-4)) for v in sorted(set(np.round(ev2, 4)))})
top = max(ev2); _, s2, v2 = np.linalg.svd(Cas2 - top*np.eye(105)); Q7 = v2[s2 < 1e-6*s2.max()].T; Q7, _ = np.linalg.qr(Q7); Q7p = np.linalg.pinv(Q7)
print("top component dimension:", Q7.shape[1], "(expect 77 for 2ω₂)")
def rho77(g): return Q7p @ sym2m(ad(g)) @ Q7
print("rho77 anti-multiplicative:", np.allclose(rho77(G[5].astype(float) @ G[77].astype(float)), rho77(G[77].astype(float)) @ rho77(G[5].astype(float)), atol=1e-8))
for level in ('line', 'plane', 'flag'):
    d, T77, res = build(level, rho77, 77, False)
    print(f"weight 77, {level}: dims {d}, residual {res:.1e}, T2 eigenvalues {np.round(np.sort(np.linalg.eigvals(T77).real), 4) if T77.size else '[]'}")
print("PREDICTIONS (via G2 x SO(3), SO(3) = Arthur [3]):  3087 -> T2 = 9 ;  567 -> T2 = -13.5")
