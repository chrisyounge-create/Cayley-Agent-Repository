import sitecustomize_shim
import numpy as np, time, sys, json
exec(open('g2_weight77.py').read().split('print("rho77 anti-multiplicative')[0])   # Ders, rep_mat, der_mat, Bki, Cas, rho (27), ad, ad_der, rho77, act_*, L, Pl, flags, G
log = open('g2_table.log', 'a')
def say(*a): print(*a, flush=True); print(*a, file=log, flush=True)
# weight 7: Casimir-eigenvalue-2 component of Sym^2(8)
_, s7, v7 = np.linalg.svd(Cas - 2.0*np.eye(36)); Q7b = v7[s7 < 1e-6*s7.max()].T; Q7b, _ = np.linalg.qr(Q7b); Q7p = np.linalg.pinv(Q7b)
say("weight-7 component dimension:", Q7b.shape[1], "(expect 7)")
def rho7(g): return Q7p @ rep_mat(g) @ Q7b
def d7(X): return Q7p @ der_mat(X) @ Q7b
# weight 64: top Casimir component of 7 (x) 14
I7, I14 = np.eye(7), np.eye(14)
D98 = [np.kron(d7(X), I14) + np.kron(I7, ad_der(X)) for X in Ders]
Cas98 = sum(Bki[a, b] * D98[a] @ D98[b] for a in range(14) for b in range(14))
ev98 = np.linalg.eigvals(Cas98).real; top = ev98.max(); _, s98, v98 = np.linalg.svd(Cas98 - top*np.eye(98)); Q64 = v98[s98 < 1e-6*s98.max()].T; Q64, _ = np.linalg.qr(Q64); Q64p = np.linalg.pinv(Q64)
def rho64(g): return Q64p @ np.kron(rho7(g), ad(g)) @ Q64
say("weight-64 component dimension:", Q64.shape[1], "(expect 64); Casimir spectrum on 7x14:", sorted({round(float(x), 3) for x in ev98.real}))
g1, g2 = G[5].astype(float), G[77].astype(float)
for nm, R_ in (("7", rho7), ("14", ad), ("64", rho64)):
    say(f"rho{nm} anti-multiplicative:", np.allclose(R_(g1 @ g2), R_(g2) @ R_(g1), atol=1e-8))
maps = {2: (np.load('nbr_maps2.npy'), 4), 5: (np.load('nbr_maps5.npy'), 5), 7: (np.load('nbr_maps7.npy'), None)}
m7 = maps[7][0]
for k in (1, 2):
    if all(np.allclose(np.linalg.inv(m)*7**k, np.rint(np.linalg.inv(m)*7**k), atol=1e-6) for m in m7[:100]): maps[7] = (m7, 7**k); break
say("scale for 7-maps:", maps[7][1])
Gf = [G[i].astype(np.int64) for i in range(len(G))]
def build_q(level, R_, dimV, mp, sc):
    act = {'line': act_lines, 'plane': act_planes, 'flag': act_flags}[level]
    nX = {'line': len(L), 'plane': len(Pl), 'flag': len(flags)}[level]
    imgs = [act(g % 3) for g in Gf]
    orbit_of, trans, reps_ = {}, {}, []
    for x in range(nX):
        if x in orbit_of: continue
        o = len(reps_); reps_.append(x)
        for kk, im in enumerate(imgs):
            y = int(im[x])
            if y not in orbit_of: orbit_of[y] = o; trans[y] = kk
    stabs_ = [[kk for kk, im in enumerate(imgs) if int(im[r]) == r] for r in reps_]
    bases = []
    for st in stabs_:
        P = sum(R_(Gf[kk].astype(float)) for kk in st)/len(st); u, s_, _ = np.linalg.svd(P); bases.append(u[:, s_ > 0.5])
    dims = [b.shape[1] for b in bases]; off = np.cumsum([0] + dims); nW = off[-1]
    if nW == 0: return dims, np.zeros((0, 0)), 0.0
    vals = {(o, o2, k2): np.zeros(dimV) for o in range(len(reps_)) for o2 in range(len(reps_)) for k2 in range(dims[o2])}
    Rg_cache = {}
    for m in mp:
        mi = np.linalg.inv(m); mred = np.rint(mi*sc).astype(np.int64) * pow(sc, -1, 3) % 3
        ys = act(mred); Rm = R_(m)
        for o, r in enumerate(reps_):
            y = int(ys[r]); o2 = orbit_of[y]
            if dims[o2] == 0: continue
            tk = trans[y]
            if tk not in Rg_cache: Rg_cache[tk] = R_(Gf[tk].astype(float))
            RmRg = Rm @ Rg_cache[tk]
            for k2 in range(dims[o2]): vals[(o, o2, k2)] += RmRg @ bases[o2][:, k2]
    Tm = np.zeros((nW, nW)); resid = 0.0
    for (o, o2, k2), val in vals.items():
        if dims[o]:
            c = np.linalg.lstsq(bases[o], val, rcond=None)[0]; resid = max(resid, np.linalg.norm(val - bases[o] @ c))
            Tm[off[o]:off[o+1], off[o2] + k2] = c
    return dims, Tm, resid
weights = [("1", lambda g: np.eye(1), 1), ("7", rho7, 7), ("14", ad, 14), ("27", rho, 27), ("64", rho64, 64), ("77", rho77, 77)]
results = {}
for wname, R_, dimV in weights:
    for level in ('line', 'plane', 'flag'):
        t0 = time.time(); ops = {}; dims = None; resid = 0
        for q in (2, 5, 7):
            mp, sc = maps[q]; dims, Tq, r = build_q(level, R_, dimV, mp, sc); ops[q] = Tq; resid = max(resid, r)
        if ops[2].size == 0: say(f"weight {wname:>2}, {level:5s}: dimension 0"); continue
        X = ops[2] + 0.6180339887*ops[5] + 0.2718281828*ops[7]; evX, U = np.linalg.eig(X)
        systems = []
        for i in range(len(evX)):
            u = U[:, i]; systems.append(tuple(round(float((np.linalg.solve(U, ops[q] @ u) / np.linalg.solve(U, u))[i].real), 6) for q in (2, 5, 7)))
        comm = max(np.linalg.norm(ops[a] @ ops[b] - ops[b] @ ops[a]) for a, b in ((2, 5), (2, 7), (5, 7)))
        results[f"{wname}_{level}"] = {'dims': dims, 'systems': sorted(systems), 'commutator': comm, 'residual': resid}
        say(f"weight {wname:>2}, {level:5s}: dims {dims}, joint (T2,T5,T7) eigen-systems {sorted(systems)}; [T,T'] {comm:.1e}, resid {resid:.1e}  [{time.time()-t0:.0f}s]")
        json.dump(results, open('g2_table.json', 'w'))
