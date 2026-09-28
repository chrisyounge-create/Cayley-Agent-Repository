"""Instrument v1: weight-2 (trivial-weight) algebraic modular forms on the compact G2 with parahoric level at 3 (null line /
null plane / flag) and optional parahoric level at 2; Hecke operators T_q (q = 5, 7) as Brandt matrices from the stored
q-neighbour maps.  Class set = H-orbits on the mod-3 structures, H = Stab_Gamma(mod-2 structure)."""
import numpy as np, pickle, sys, itertools
from exc_alg import E8
from coxeter_aut import ONE
from neighbors_struct import nullspace_modp, rref_key
G = np.load('gamma_G2Z.npy')
def inv_mod(M, p):
    """inverse of an integer matrix mod p (Gauss-Jordan)"""
    n = len(M); A = np.concatenate([M % p, np.eye(n, dtype=np.int64)], axis=1)
    for c in range(n):
        r = next(i for i in range(c, n) if A[i, c] % p); A[[c, r]] = A[[r, c]]
        A[c] = (A[c] * pow(int(A[c, c]), -1, p)) % p
        for i in range(n):
            if i != c and A[i, c]: A[i] = (A[i] - A[i, c] * A[c]) % p
    return A[:, n:]
def null_lines(p):
    """null vectors mod p (trace 0, norm 0), one representative per line (first nonzero coordinate = 1)"""
    pts = []
    for v in itertools.product(range(p), repeat=8):
        v = np.array(v)
        if not v.any(): continue
        nz = next(i for i in range(8) if v[i]);
        if v[nz] != 1: continue
        if int(v @ E8 @ ONE) % p or (int(v @ E8 @ v) // 2) % p: continue
        pts.append(v)
    return np.array(pts)
def line_key(v, p):
    v = v % p; nz = next(i for i in range(8) if v[i]); return tuple((v * pow(int(v[nz]), -1, p)) % p)
def null_planes_through(lines, p):
    """G2 null planes mod p: for each null line u, the v with uv = vu = 0, isotropic, trace 0 (as in neighbors_struct.neighbours)"""
    from neighbors_struct import S_
    planes = {}
    for u in lines:
        Lu = np.einsum('a,abc->bc', u, S_); Ru = np.einsum('b,abc->ac', u, S_)
        A = np.concatenate([Lu.T, Ru.T], axis=0); ann = nullspace_modp(A, p)
        for coeffs in itertools.product(range(p), repeat=len(ann)):
            v = sum(c * b for c, b in zip(coeffs, ann)) % p
            if not v.any(): continue
            if int(v @ E8 @ ONE) % p or (int(v @ E8 @ v) // 2) % p: continue
            k = rref_key([u, v], p)
            if len(k) < 2: continue
            if k not in planes: planes[k] = (u, v)
    return planes
if __name__ == "__main__":
    p = 3; q = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    lines3 = null_lines(3); print("null lines mod 3:", len(lines3), "(expect 364)")
    planes3 = null_planes_through(lines3, 3); print("null planes mod 3:", len(planes3), "(expect 364)")
    # flags: (line, plane) with line in plane
    flags3 = []
    for k, (u, v) in planes3.items():
        for a, b in itertools.product(range(3), repeat=2):
            w = (a*u + b*v) % 3
            if w.any(): flags3.append((line_key(w, 3), k))
    flags3 = sorted(set(flags3)); print("flags mod 3:", len(flags3), "(expect 1456)")
    Gm3 = G % 3; Gm2 = G % 2
    g2index = {G[i].__mod__(2).tobytes(): i for i in range(len(G))}
    # mod-2 structures and stabilisers
    lines2 = null_lines(2); planes2 = null_planes_through(lines2, 2); print("mod 2: lines", len(lines2), "planes", len(planes2))
    def stab_of(fn):
        return [i for i in range(len(G)) if fn(Gm2[i])]
    l2 = lines2[0]; pk2 = next(iter(planes2)); pu, pv = planes2[pk2]
    H = {'none': list(range(len(G))),
         'line2': stab_of(lambda M: line_key(l2 @ M, 2) == line_key(l2, 2)),
         'plane2': stab_of(lambda M: rref_key([pu @ M, pv @ M], 2) == pk2)}
    # flag at 2: a line inside the chosen plane
    fl = next((a*pu + b*pv) % 2 for a, b in itertools.product(range(2), repeat=2) if ((a*pu + b*pv) % 2).any())
    H['flag2'] = [i for i in H['plane2'] if line_key(fl @ Gm2[i], 2) == line_key(fl, 2)]
    print("stabiliser orders at 2:", {k: len(v) for k, v in H.items()})
    # structures at 3 as canonical keys and action
    def act_line(v, M): return line_key(v @ M, 3)
    def act_plane(k, M): u, v = planes3[k]; return rref_key([u @ M, v @ M], 3)
    def act_flag(f, M): lk, pk = f; return (line_key(np.array(lk) @ M, 3), act_plane(pk, M))
    X = {'line3': ([line_key(v, 3) for v in lines3], act_line, lambda k: np.array(k)),
         'plane3': (list(planes3.keys()), act_plane, lambda k: k),
         'flag3': (flags3, act_flag, lambda f: f)}
    # neighbour maps: transport T = g^{-1} gamma_g, gamma_g in Gamma with gamma_g == g mod 2, reduced mod 3
    maps = np.load(f'nbr_maps{q}.npy'); gq = np.rint(maps * q).astype(np.int64); qinv3 = pow(q, -1, 3)
    Ts = []
    for k in range(len(gq)):
        gm2 = (gq[k] % 2).tobytes(); gi = g2index.get(gm2)
        if gi is None: raise RuntimeError("g mod 2 not in Gamma")
        g3 = (gq[k] * qinv3) % 3; T = (inv_mod(g3, 3) @ Gm3[gi]) % 3; Ts.append(T)
    Ts = np.array(Ts); print(f"T_{q}: {len(Ts)} transports built")
    results = {}
    want = sys.argv[2].split(',') if len(sys.argv) > 2 else None
    for x3name, (elems, act, _) in X.items():
        for h2name, Hidx in H.items():
            if want and f'{x3name}:{h2name}' not in want: continue
            # H-orbits on elems
            orb = {}; reps = []
            for e in elems:
                if e in orb: continue
                oid = len(reps); reps.append(e); orb[e] = oid
                for i in Hidx:
                    orb[act(e, Gm3[i])] = oid
            n = len(reps); B = np.zeros((n, n), dtype=np.int64)
            for T in Ts:
                for r, e in enumerate(reps): B[r, orb[act(e, T)]] += 1
            ev = np.linalg.eigvals(B.astype(float)); ev = np.sort(ev.real)
            sat = np.round((ev + 1) / q**3, 4)
            results[(x3name, h2name)] = (n, sat, B)
            print(f"level ({x3name}, {h2name}): dim {n:3d}; row sums {set(B.sum(1))}; Satake a({q}) of eigenforms: {sat}")
    pickle.dump({k: (v[0], v[2]) for k, v in results.items()}, open(f'levelN_T{q}.pkl', 'wb'))
