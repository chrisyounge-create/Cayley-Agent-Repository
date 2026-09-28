"""Compact G2 at the two maximal parahorics at 3 (null lines, null planes) and the Iwahori: class sets and T_2, T_5."""
import numpy as np, itertools, pickle
from exc_alg import E8
from coxeter_aut import ONE
from neighbors_struct import nullspace_modp, S_
from levelN import inv_mod
from sympy import Matrix
p = 3; G = np.load('gamma_G2Z.npy')
def norm_mod(v): return (np.einsum('...i,ij,...j->...', v, E8, v) // 2) % p
def tr_mod(v): return (v @ E8 @ ONE) % p
base = (p ** np.arange(8)).astype(np.int64); INV = np.array([0] + [pow(k, -1, p) for k in range(1, p)])
def normalise(v):
    v = v % p; nz = (v != 0).argmax(axis=-1); lead = np.take_along_axis(v, nz[..., None], axis=-1)[..., 0]; return (v * INV[lead][..., None]) % p
allv = np.array(list(itertools.product(range(p), repeat=8)), dtype=np.int64); allv = allv[allv.any(axis=1)]
ok = (norm_mod(allv) == 0) & (tr_mod(allv) == 0); L = np.unique(normalise(allv[ok]), axis=0); Lcode = L @ base; print("null lines mod 3:", len(L))
order = np.argsort(Lcode); Lsorted = Lcode[order]
def line_index(v): return order[np.searchsorted(Lsorted, normalise(v) @ base)]
planes = {}; C = np.array(list(itertools.product(range(p), repeat=3)), dtype=np.int64)
for i, u in enumerate(L):
    Lu = np.einsum('a,abc->bc', u, S_); Ru = np.einsum('b,abc->ac', u, S_); ann = np.array(nullspace_modp(np.concatenate([Lu.T, Ru.T], axis=0) % p, p), dtype=np.int64)
    if len(ann) != 3: continue
    W = (C @ ann) % p; okw = W.any(axis=1) & (norm_mod(W) == 0) & (tr_mod(W) == 0); W = np.unique(normalise(W[okw]), axis=0); W = W[line_index(W) != i]
    li = line_index((W[:, None, :] + np.arange(p)[None, :, None]*u[None, None, :]) % p); keys = np.sort(np.concatenate([li, np.full((len(li), 1), i)], axis=1), axis=1)
    for row in np.unique(keys, axis=0): planes.setdefault(tuple(row.tolist()), None)
Pl = np.array(sorted(planes)); Pidx = {tuple(r): j for j, r in enumerate(Pl.tolist())}; print("null planes mod 3:", len(Pl))
flags = [(l, j) for j, k in enumerate(Pl.tolist()) for l in k]; Fidx = {f: i for i, f in enumerate(flags)}
def act_lines(M): return line_index(L @ M)
def act_planes(M):
    li = line_index(L @ M); return np.array([Pidx[tuple(sorted(li[r].tolist()))] for r in Pl.tolist()])
def act_flags(M):
    li = line_index(L @ M); pj = act_planes(M); return np.array([Fidx[(int(li[l]), int(pj[j]))] for (l, j) in flags])
Gm = G % p
def orbits(n, act):
    parent = np.arange(n)
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for gi in (1, 7, 101, 2345, 6000, 11):
        img = act(Gm[gi])
        for i in range(n):
            a, b = find(i), find(int(img[i]))
            if a != b: parent[a] = b
    roots = np.array([find(i) for i in range(n)]); reps, cls, sizes = np.unique(roots, return_inverse=True, return_counts=True); return reps, cls, sizes
results = {}
for name, n, act, kind in (("line3", len(L), act_lines, 'l'), ("plane3", len(Pl), act_planes, 'p'), ("flag3", len(flags), act_flags, 'f')):
    reps, cls, sizes = orbits(n, act); stabs = [12096 // int(s) for s in sizes]
    out = {'sizes': sizes, 'stabs': stabs}
    for q in (2, 5):
        maps = np.load(f'nbr_maps{q}.npy'); gq = np.rint(maps * q).astype(np.int64); qinv = pow(q, -1, p); B = np.zeros((len(reps), len(reps)), dtype=np.int64)
        for g in gq:
            T = inv_mod((g * qinv) % p, p); img = act(T)
            for r_i, r in enumerate(reps): B[r_i, cls[img[r]]] += 1
        out[q] = B
        ev = Matrix(B.tolist()).eigenvals()
        print(f"{name}: h = {len(reps)}, stabilisers {stabs}, Σ1/|Stab| = {sum(__import__('fractions').Fraction(1,s) for s in stabs)};  T_{q}: row sums {set(B.sum(1).tolist())}, eigenvalues {dict((str(k), v) for k, v in ev.items())}")
    results[name] = out
pickle.dump(results, open('g2_level3_parahorics.pkl', 'wb'))
