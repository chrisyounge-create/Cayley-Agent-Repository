import numpy as np, itertools, collections, time, glob
exec(open('selection_rule.py').read().split('print("dim V_pi')[0])
# the 64 = character-projector component of 7 (x) 14 over the finite group G2(Z)
R98 = lambda g: np.kron(rho7(g), ad(np.array(g, dtype=float)))
Pr = np.zeros((98, 98))
for g in Gf: Pr += (np.conj(chars(g)["64"]) * R98(g)).real
Pr *= 64 / len(Gf); u, s_, vt = np.linalg.svd(Pr); print("projector rank:", int(np.sum(s_ > 0.5)), "(expect 64); singular values near 1:", np.sum(np.abs(s_ - 1) < 1e-6))
Q64 = u[:, :64]
R64 = lambda g: Q64.T @ R98(g) @ Q64
def R64der(X):
    eps = 1e-6; g = np.eye(8) + eps * np.array(X, dtype=float); return (R64(g) - np.eye(64)) / eps
g1, g2 = Gf[5].astype(float), Gf[77].astype(float); print("R64 anti-multiplicative:", np.allclose(R64(g1 @ g2), R64(g2) @ R64(g1), atol=1e-8))
# level-7 line structures
p = 7
def normalise(W):
    W = W % p; f = (W != 0).argmax(1); l = W[np.arange(len(W)), f]; iv = np.array([pow(int(x), -1, p) for x in l]); return (W * iv[:, None]) % p
coeffs = np.array(list(itertools.product(range(p), repeat=8)), dtype=np.int64); tr = (coeffs @ (G8 @ one)) % p; Vv = coeffs[(tr == 0) & coeffs.any(1)]
norms = np.einsum('ij,jk,ik->i', Vv, G8, Vv) % p; iso = np.unique(normalise(Vv[norms == 0]), axis=0); lidx = {tuple(v): i for i, v in enumerate(iso)}
Gi = np.array(Gf, dtype=np.int64); cls = -np.ones(len(iso), dtype=np.int64); trans = -np.ones(len(iso), dtype=np.int64); reps = []; stabs = []
for i in range(len(iso)):
    if cls[i] >= 0: continue
    imgs = normalise(np.einsum('j,gjk->gk', iso[i], Gi)); c = len(reps); st = []
    for gi, im in enumerate(imgs):
        j = lidx[tuple(im)]
        if cls[j] < 0: cls[j] = c; trans[j] = gi
        if j == i: st.append(gi)
    reps.append(i); stabs.append(st)
conic = {}
for n, (a, b) in zds.items():
    Hs = np.array([a % p, b % p, omul(a, b) % p]); pts = set()
    for c in itertools.product(range(p), repeat=3):
        if not any(c): continue
        v = (np.array(c) @ Hs) % p
        if v.any() and (v @ G8 @ v) % p == 0: pts.add(lidx[tuple(normalise(v[None, :])[0])])
    conic[n] = sorted(pts)
bases = []
for st in stabs:
    Pm = sum(R64(Gf[k].astype(float)) for k in st) / len(st); u, s, _ = np.linalg.svd(Pm); bases.append(u[:, s > 0.5])
dims = [b.shape[1] for b in bases]; off = np.cumsum([0] + dims); print(f"weight 64 at level 7, line level: dims per class {dims}, total {off[-1]}  [{time.time()-t0:.0f}s]")
maps_q = np.load('nbr_maps2.npy'); T = np.zeros((off[-1], off[-1]))
Rm = {}
for o2 in range(len(reps)):
    for k2 in range(dims[o2]):
        v = bases[o2][:, k2]
        for o, r in enumerate(reps):
            val = np.zeros(64)
            for mi, m in enumerate(maps_q):
                Tm = np.rint(np.linalg.inv(m) * 4).astype(np.int64) * pow(4, -1, p) % p
                y = lidx[tuple(normalise(((iso[r] @ Tm) % p)[None, :])[0])]
                if cls[y] != o2: continue
                if mi not in Rm: Rm[mi] = R64(m)
                val += Rm[mi] @ R64(Gf[trans[y]].astype(float)) @ v
            if dims[o]: T[off[o]:off[o+1], off[o2]+k2] = np.linalg.lstsq(bases[o], val, rcond=None)[0]
print(f"T2 built [{time.time()-t0:.0f}s]")
w, V = np.linalg.eig(T); w, V = w.real, V.real; srt = np.argsort(-w); w, V = w[srt], V[:, srt]
def kz_fixed(z):
    a, b = z; A = np.array([np.concatenate([a @ D, b @ D]) for D in Ders]).T; u, s, vt = np.linalg.svd(A, full_matrices=True)
    kz = [sum(c * D for c, D in zip(vt[k], Ders)) for k in range(np.sum(s > 1e-9), 14)]
    Mz = np.vstack([R64der(X) for X in kz]); u, s, vt = np.linalg.svd(Mz, full_matrices=True); return vt[np.sum(s > 1e-3):].T
fixed = {n: kz_fixed(z) for n, z in zds.items()}
res = {n: [] for n in zds}
for i in range(len(w)):
    vec = V[:, i] / np.linalg.norm(V[:, i])
    def f_at(x):
        o = cls[x]; fr = bases[o] @ vec[off[o]:off[o+1]]; return R64(Gf[trans[x]].astype(float)) @ fr
    for n in zds:
        Sx = sum(f_at(x) for x in conic[n]); res[n].append((round(float(w[i]), 3), round(float(np.linalg.norm(Sx)), 4), round(float(np.linalg.norm(fixed[n] @ (fixed[n].T @ Sx))), 4)))
for n in sorted(zds):
    sums = [r[1] for r in res[n]]
    print(f"type {n}: conic sums over {len(sums)} eigenforms: zero for {sum(1 for s_ in sums if s_ < 1e-8)}, nonzero for {sum(1 for s_ in sums if s_ >= 1e-8)} (predicted: {'ALL ZERO' if n == 24 else 'generically nonzero'}); examples: {res[n][:4]}")
print("tempered-bound violators (|T2+1| > 56):", [round(float(x), 3) for x in w if abs(x + 1) > 56])
