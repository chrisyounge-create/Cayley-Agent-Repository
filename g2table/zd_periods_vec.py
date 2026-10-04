import numpy as np, itertools, sys, pickle, time
exec(open('g2_weight77.py').read().split('print("rho77 anti-multiplicative')[0])     # Ders, rep_mat, der_mat, Q, Qp, rho(27), ad, ad_der, Gf, G, flags, L, Pl, act_*, maps, S, E8, ONE
G8 = np.array(E8, dtype=np.int64); one = np.array(ONE, dtype=np.int64); omul = lambda x, y: np.einsum('i,j,ijk->k', x, y, S)
def my_build(level, R_, dimV, inv_conv):
    act = {'line': act_lines, 'plane': act_planes, 'flag': act_flags}[level]; nX = {'line': len(L), 'plane': len(Pl), 'flag': len(flags)}[level]
    imgs = [act(g % 3) for g in Gf]; orbit_of, trans, reps_ = {}, {}, []
    for x in range(nX):
        if x in orbit_of: continue
        o = len(reps_); reps_.append(x)
        for k, im in enumerate(imgs):
            y = int(im[x])
            if y not in orbit_of: orbit_of[y] = o; trans[y] = k
    stabs_ = [[k for k, im in enumerate(imgs) if int(im[r]) == r] for r in reps_]
    bases = []
    for st in stabs_:
        P = sum(R_(Gf[k].astype(float)) for k in st)/len(st); u, s_, _ = np.linalg.svd(P); bases.append(u[:, s_ > 0.5])
    dims = [b.shape[1] for b in bases]; off = np.cumsum([0] + dims); nW = off[-1]; Tm = np.zeros((nW, nW)); resid = 0.0
    for o2 in range(len(reps_)):
        for k2 in range(dims[o2]):
            v = bases[o2][:, k2]
            for o, r in enumerate(reps_):
                val = np.zeros(dimV)
                for m in maps:
                    mi = np.linalg.inv(m); y = int(act(np.rint((np.linalg.inv(m)*4)).astype(np.int64) * pow(4, -1, 3) % 3)[r])
                    if orbit_of[y] != o2: continue
                    g = Gf[trans[y]].astype(float); val += R_(mi if inv_conv else m) @ R_(g) @ v
                if dims[o]:
                    c = np.linalg.lstsq(bases[o], val, rcond=None)[0]; resid = max(resid, np.linalg.norm(val - bases[o] @ c)); Tm[off[o]:off[o+1], off[o2] + k2] = c
    return dict(dims=dims, T=Tm, resid=resid, bases=bases, reps=reps_, trans=trans, orbit_of=orbit_of, off=off, nX=nX, stabs=[len(s) for s in stabs_])
zd = np.load('/home/claude/zd_choice.npy'); zds = {8: (zd[0][:8], zd[0][8:]), 24: (zd[1][:8], zd[1][8:])}
def line_of(level, x):
    if level == 'line': return np.array(L[x]) % 3
    if level == 'plane': return None
    return np.array(L[flags[x][0]]) % 3 if isinstance(flags[x], (tuple, list, np.ndarray)) and np.ndim(flags[x]) == 1 and len(flags[x]) == 2 else None
def compatible_line(v, z):
    a, b = z; Hs = [a % 3, b % 3, omul(a, b) % 3]
    return any(np.array_equal(sum(c[k] * Hs[k] for k in range(3)) % 3, v) for c in itertools.product(range(3), repeat=3))
def kz_fixed(Rder, z):
    a, b = z; A = np.array([np.concatenate([a @ D, b @ D]) for D in Ders]).T; u, s, vt = np.linalg.svd(A, full_matrices=True)
    kz = [sum(c * D for c, D in zip(vt[k], Ders)) for k in range(np.sum(s > 1e-9), 14)]
    M = np.vstack([Rder(X) for X in kz]); u, s, vt = np.linalg.svd(M, full_matrices=True); return vt[np.sum(s > 1e-8):].T, len(kz)
weights = {'14': (ad, ad_der, 14), '27': (rho, lambda X: Qp @ der_mat(X) @ Q, 27)}
print("flags encoded as:", type(flags[0]).__name__, flags[0] if np.ndim(flags[0]) <= 1 else np.shape(flags[0]))
for wname, (R_, Rder, dimV) in weights.items():
    for level in ('line', 'plane', 'flag'):
        t0 = time.time()
        for conv in (False, True):
            bd = my_build(level, R_, dimV, conv)
            if sum(bd['dims']) == 0: print(f"weight {wname}, {level}: no forms"); break
            ev = np.linalg.eigvals(bd['T']).real
            print(f"weight {wname}, {level}, conv {'m^-1' if conv else 'm'}: dims {bd['dims']}, residual {bd['resid']:.1e}, T2 eigenvalues {np.round(np.sort(ev), 3)}  [{time.time()-t0:.0f}s]")
            if bd['resid'] > 1e-6: continue
            w, V = np.linalg.eig(bd['T']); w, V = w.real, V.real
            lines_ok = {n: [x for x in range(bd['nX']) if line_of(level, x) is not None and compatible_line(line_of(level, x), z)] for n, z in zds.items()}
            fixed = {n: kz_fixed(Rder, z) for n, z in zds.items()}
            print(f"   K_z-fixed subspace dims: {[(n, fixed[n][0].shape[1], 'dim k_z=' + str(fixed[n][1])) for n in fixed]}; compatible structures: {[(n, len(lines_ok[n])) for n in lines_ok]}")
            for k in range(len(w)):
                vec = V[:, k]; vec = vec / np.linalg.norm(vec)
                def f_at(x):
                    o = bd['orbit_of'][x]; comp = vec[bd['off'][o]:bd['off'][o+1]]
                    fr = bd['bases'][o] @ comp; return R_(Gf[bd['trans'][x]].astype(float)) @ fr
                out = []
                for n in (24, 8):
                    if not lines_ok[n]: out.append((n, None)); continue
                    Sx = sum(f_at(x) for x in lines_ok[n]); U = fixed[n][0]; Pz = U @ (U.T @ Sx)
                    out.append((n, round(float(np.linalg.norm(Pz)), 5), round(float(np.linalg.norm(Sx)), 5)))
                print(f"   T2 = {w[k]:9.3f}: period norms (type, |P_z|, |sum f|): {out}")
            break
