import numpy as np, itertools, collections, cypari2, time, sys, glob
exec(open('g2_weight77.py').read().split('print("rho77 anti-multiplicative')[0])     # rho (27), ad (14), ad_der, der_mat, Q, Qp, Ders, Gf, maps, S, E8, ONE
t0 = time.time(); p = 7
G8 = np.array(E8, dtype=np.int64); one = np.array(ONE, dtype=np.int64); omul = lambda x, y: np.einsum('i,j,ijk->k', x, y, S)
pari = cypari2.Pari()
qm = pari.qfminim(pari.matrix(8, 8, [int(x) for x in G8.flatten()]), 2); qM = qm[2]; nc = int(pari.matsize(qM)[1])
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
# isotropic lines mod 7 (row vectors), classes, transporters
def normalise(W):
    W = W % p; f = (W != 0).argmax(1); l = W[np.arange(len(W)), f]; iv = np.array([pow(int(x), -1, p) for x in l]); return (W * iv[:, None]) % p
coeffs = np.array(list(itertools.product(range(p), repeat=8)), dtype=np.int64)
tr = (coeffs @ (G8 @ one)) % p; Vv = coeffs[(tr == 0) & coeffs.any(1)]
norms = np.einsum('ij,jk,ik->i', Vv, G8, Vv) % p; iso = np.unique(normalise(Vv[norms == 0]), axis=0); lidx = {tuple(v): i for i, v in enumerate(iso)}
Gm = np.array(Gf, dtype=np.int64)
cls = -np.ones(len(iso), dtype=np.int64); trans = -np.ones(len(iso), dtype=np.int64); reps = []; stabs = []
for i in range(len(iso)):
    if cls[i] >= 0: continue
    imgs = normalise(np.einsum('j,gjk->gk', iso[i], Gm)); c = len(reps); st = []
    for gi, im in enumerate(imgs):
        j = lidx[tuple(im)]
        if cls[j] < 0: cls[j] = c; trans[j] = gi
        if j == i: st.append(gi)
    reps.append(i); stabs.append(st)
print(f"level {p} lines: {len(iso)}; classes {len(reps)}; stabiliser orders {[len(s) for s in stabs]}  [{time.time()-t0:.0f}s]")
# conic points per zero-divisor type
conic = {}
for n, (a, b) in zds.items():
    Hs = np.array([a % p, b % p, omul(a, b) % p]); pts = set()
    for c in itertools.product(range(p), repeat=3):
        if not any(c): continue
        v = (np.array(c) @ Hs) % p
        if v.any() and (v @ G8 @ v) % p == 0: pts.add(lidx[tuple(normalise(v[None, :])[0])])
    conic[n] = sorted(pts); print(f"type {n}: conic points {len(pts)} in class(es) {sorted({int(cls[t]) for t in pts})}")
# representations: 7 (trace-zero part of the 8), 14 (ad), 27 (rho)
w8 = (G8 @ one).astype(float); P7 = np.linalg.svd(w8[None, :])[2][1:]              # rows: orthonormal basis of the trace-zero coordinate subspace
R7 = lambda g: P7 @ np.array(g, dtype=float).T @ P7.T; R7der = lambda X: P7 @ np.array(X, dtype=float).T @ P7.T
weights = {'27': (rho, lambda X: Qp @ der_mat(X) @ Q, 27)} if len(sys.argv) > 1 and sys.argv[1] == '27' else {'7': (R7, R7der, 7), '14': (ad, ad_der, 14), '27': (rho, lambda X: Qp @ der_mat(X) @ Q, 27)}
def kz_fixed(Rder, z):
    a, b = z; A = np.array([np.concatenate([a @ D, b @ D]) for D in Ders]).T; u, s, vt = np.linalg.svd(A, full_matrices=True)
    kz = [sum(c * D for c, D in zip(vt[k], Ders)) for k in range(np.sum(s > 1e-9), 14)]
    Mz = np.vstack([Rder(X) for X in kz]); u, s, vt = np.linalg.svd(Mz, full_matrices=True); return vt[np.sum(s > 1e-8):].T
def hecke(q, R_, dimV, bases, off):
    maps_q = np.load(glob.glob(f'nbr_maps{q}.npy')[0] if glob.glob(f'nbr_maps{q}.npy') else f'/home/claude/nbr_maps{q}.npy')
    dims = [b.shape[1] for b in bases]; nW = off[-1]; T = np.zeros((nW, nW))
    for o2 in range(len(reps)):
        for k2 in range(dims[o2]):
            v = bases[o2][:, k2]
            for o, r in enumerate(reps):
                val = np.zeros(dimV)
                for m in maps_q:
                    Tm = np.rint(np.linalg.inv(m) * q * q).astype(np.int64) * pow(q * q, -1, p) % p
                    y = lidx[tuple(normalise(((iso[r] @ Tm) % p)[None, :])[0])]
                    if cls[y] != o2: continue
                    val += R_(m) @ R_(Gf[trans[y]].astype(float)) @ v
                if dims[o]:
                    c = np.linalg.lstsq(bases[o], val, rcond=None)[0]; T[off[o]:off[o+1], off[o2]+k2] = c
    return T
for wname, (R_, Rder, dimV) in weights.items():
    bases = []
    for st in stabs:
        Pm = sum(R_(Gf[k].astype(float)) for k in st) / len(st); u, s, _ = np.linalg.svd(Pm); bases.append(u[:, s > 0.5])
    dims = [b.shape[1] for b in bases]; off = np.cumsum([0] + dims)
    if off[-1] == 0: print(f"\nweight {wname}: no forms at level {p} line level"); continue
    T2 = hecke(2, R_, dimV, bases, off); T3 = hecke(3, R_, dimV, bases, off)
    print(f"\nweight {wname}: dims per class {dims} (total {off[-1]}); [T2,T3] = {np.linalg.norm(T2 @ T3 - T3 @ T2):.1e}  [{time.time()-t0:.0f}s]")
    Mx = T2 + 0.3719 * T3; w, V = np.linalg.eig(Mx); V = V.real
    lam2 = [(V[:, i] @ T2 @ V[:, i]) / (V[:, i] @ V[:, i]) for i in range(off[-1])]; lam3 = [(V[:, i] @ T3 @ V[:, i]) / (V[:, i] @ V[:, i]) for i in range(off[-1])]
    fixed = {n: kz_fixed(Rder, z) for n, z in zds.items()}
    for n, (a, b) in zds.items():
        ab = omul(a, b); Hm = np.array([one, a, b, ab], dtype=float); Pr = Hm.T @ np.linalg.pinv(Hm.T); inH = lambda v: np.allclose(Pr @ v, v)
        Sz = [k for k, g in enumerate(Gf) if inH(a @ g) and inH(b @ g) and inH(ab @ g)]
        Pst = sum(R_(Gf[k].astype(float)) for k in Sz) / len(Sz)
        print(f"   type {n}: |Stab(H_z)| = {len(Sz)}, dim of Stab(H_z)-fixed vectors in V_{wname} = {int(round(np.trace(Pst)))}  (conic sum = projection onto this space)")
    print(f"   K_z-fixed dims: {[(n, fixed[n].shape[1]) for n in fixed]}; tempered bound at 2: |T2+1| <= {7*8}, at 3: <= {7*27}")
    for i in np.argsort(lam2)[::-1]:
        vec = V[:, i] / np.linalg.norm(V[:, i])
        def f_at(x):
            o = cls[x]; fr = bases[o] @ vec[off[o]:off[o+1]]; return R_(Gf[trans[x]].astype(float)) @ fr
        out = []
        for n in (24, 8):
            U = fixed[n]; Sx = sum(f_at(x) for x in conic[n]); raw = np.linalg.norm(np.array([f_at(x) for x in conic[n]])) 
            out.append(f"type {n}: |P_z| = {np.linalg.norm(U @ (U.T @ Sx)):.3f}, |sum f| = {np.linalg.norm(Sx):.3f}, |f on conic| = {raw:.3f}")
        print(f"   T2 = {lam2[i]:9.3f}  T3 = {lam3[i]:9.3f} | " + " ; ".join(out))
