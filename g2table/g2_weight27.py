import numpy as np, itertools, time
src = open('g2_level3_parahorics.py').read().split("results = {}")[0]
exec(src)                                    # p = 3, L, Pl, flags, act_flags, orbits(...), G, Gm, inv_mod, S_, E8, ONE
S = S_
# derivations of the octonions (row action): (xy)D = (xD)y + x(yD)
n = 8; eqs = []
for a in range(n):
    for b in range(n):
        for c in range(n):
            row = np.zeros((n, n))
            for c2 in range(n): row[c2, c] += S[a, b, c2]
            for a2 in range(n): row[a, a2] -= S[a2, b, c]
            for b2 in range(n): row[b, b2] -= S[a, b2, c]
            eqs.append(row.ravel())
_, sv, vt = np.linalg.svd(np.array(eqs)); Ders = [v.reshape(n, n) for v in vt[sv < 1e-9]]
print("derivation algebra dimension:", len(Ders), "(expect 14)")
# Sym^2 of row space as symmetric matrices, right action S -> g^T S g; derivative: S -> D^T S + S D
idx = [(i, j) for i in range(8) for j in range(i, 8)]
def to_vec(Sm): return np.array([Sm[i, j] for i, j in idx])
def to_mat(v):
    M = np.zeros((8, 8))
    for k, (i, j) in enumerate(idx): M[i, j] = M[j, i] = v[k]
    return M
basis = [to_mat(e) for e in np.eye(36)]
def rep_mat(g):  return np.array([to_vec(g.T @ B @ g) for B in basis]).T
def der_mat(D): return np.array([to_vec(D.T @ B + B @ D) for B in basis]).T
Bk = np.array([[np.trace(X @ Y) for Y in Ders] for X in Ders]); Bki = np.linalg.inv(Bk)
DM = [der_mat(D) for D in Ders]
Cas = sum(Bki[a, b] * DM[a] @ DM[b] for a in range(14) for b in range(14))
ev = np.linalg.eigvals(Cas).real; vals = sorted(set(np.round(ev, 5))); print("Casimir on Sym²(8):", {v: int(np.sum(np.abs(ev - v) < 1e-4)) for v in vals})
top = max(ev); _, s2, v2 = np.linalg.svd(Cas - top*np.eye(36)); Q = v2[s2 < 1e-6*s2.max()].T; Q, _ = np.linalg.qr(Q); Qp = np.linalg.pinv(Q)
print("top component dimension:", Q.shape[1], "(expect 27)")
def rho(g): return Qp @ rep_mat(g) @ Q
g1, g2 = G[5].astype(float), G[77].astype(float); print("rho anti-multiplicative (right action):", np.allclose(rho(g1 @ g2), rho(g2) @ rho(g1)))
maps = np.load('nbr_maps2.npy')
print("neighbour maps:", maps.shape, "; preserve multiplication:", np.allclose(np.einsum('ia,jb,abc->ijc', maps[0], maps[0], S), np.einsum('ijc,cd->ijd', S, maps[0]), atol=1e-8))
np.save('g2w27_Q.npy', Q)
Gf = [G[i].astype(np.int64) for i in range(len(G))]
def build(level, R_, dimV, inv_conv):
    act = {'line': act_lines, 'plane': act_planes, 'flag': act_flags}[level]
    nX = {'line': len(L), 'plane': len(Pl), 'flag': len(flags)}[level]
    imgs = [act(g % 3) for g in Gf]                                          # x -> x·g for every g in Γ
    orbit_of, trans, reps_ = {}, {}, []
    for x in range(nX):
        if x in orbit_of: continue
        o = len(reps_); reps_.append(x)
        for k, im in enumerate(imgs):
            y = int(im[x])
            if y not in orbit_of: orbit_of[y] = o; trans[y] = k            # x·g_k = y
    stabs_ = [[k for k, im in enumerate(imgs) if int(im[r]) == r] for r in reps_]
    bases = []
    for st in stabs_:
        P = sum(R_(Gf[k].astype(float)) for k in st)/len(st); u, s_, _ = np.linalg.svd(P); bases.append(u[:, s_ > 0.5])
    dims = [b.shape[1] for b in bases]; off = np.cumsum([0] + dims); nW = off[-1]
    Tm = np.zeros((nW, nW)); resid = 0.0
    for o2 in range(len(reps_)):
        for k2 in range(dims[o2]):
            v = bases[o2][:, k2]
            for o, r in enumerate(reps_):
                val = np.zeros(dimV)
                for m in maps:
                    mi = np.linalg.inv(m)
                    y = int(act(inv_mod(np.rint(m*2).astype(np.int64) * pow(2, -1, 3) % 3, 3))[r]) if False else int(act(np.rint((np.linalg.inv(m)*4)).astype(np.int64) * pow(4, -1, 3) % 3)[r])
                    if orbit_of[y] != o2: continue
                    g = Gf[trans[y]].astype(float)
                    val += R_(mi if inv_conv else m) @ R_(g) @ v
                if dims[o]:
                    c = np.linalg.lstsq(bases[o], val, rcond=None)[0]; resid = max(resid, np.linalg.norm(val - bases[o] @ c))
                    Tm[off[o]:off[o+1], off[o2] + k2] = c
    return dims, Tm, resid
triv = lambda g: np.eye(1)
for level in ('line', 'plane', 'flag'):
    d, Tt, _ = build(level, triv, 1, False); print(f"trivial weight, {level}: dims {d}, T2 eigenvalues {np.round(np.sort(np.linalg.eigvals(Tt).real), 4)}")
for level in ('line', 'plane', 'flag'):
    for conv in (False, True):
        d, T27, res = build(level, rho, 27, conv)
        print(f"weight 27, {level}, conv {'m^-1' if conv else 'm'}: dims {d}, residual {res:.2e}, T2 eigenvalues {np.round(np.sort(np.linalg.eigvals(T27).real), 4) if T27.size else '[]'}")
print("PREDICTIONS: 3087 -> T2 = 9 ;  567 -> T2 = -13.5")
