"""Compact G2, Iwahori level p: class set (flags: null line ⊂ null plane mod p) under G2(Z), stabilisers, mass check, Hecke T_q.
usage: g2_iwahori.py p [q,...]   (q from stored neighbour maps: 2, 5, 7)"""
import numpy as np, sys, time, itertools, collections, pickle
from exc_alg import E8, STRUCT
from coxeter_aut import ONE
p = int(sys.argv[1]); qs = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [2]
t0 = time.time(); G = np.load('gamma_G2Z.npy').astype(np.int64); one = np.array(ONE, dtype=np.int64); E = np.array(E8, dtype=np.int64)
w = E @ one                                                   # trace functional: tr(x) = x.w ... (tr = (x E one)/? ) use x.w == 0 mod p as trace-zero
PW = [p**k for k in range(8)]
def inv_mod(M, p):
    n = len(M); A = np.concatenate([M % p, np.eye(n, dtype=np.int64)], axis=1)
    for c in range(n):
        r = next(i for i in range(c, n) if A[i, c] % p); A[[c, r]] = A[[r, c]]
        A[c] = (A[c] * pow(int(A[c, c]), -1, p)) % p
        for i in range(n):
            if i != c and A[i, c]: A[i] = (A[i] - A[i, c] * A[c]) % p
    return A[:, n:]
# ---- 1. null lines mod p: trace-zero, norm-zero vectors, one representative per line (first nonzero coordinate 1)
# parametrise: choose a basis of the trace-zero hyperplane H (7-dim over F_p), enumerate projective points of H: 7 coordinates, first nonzero = 1
# solve the quadric by brute force over 6 free coordinates with the last one solved? simpler: enumerate all projective points of P(H) in chunks and filter
# (number of projective points of P^6(F_p) = (p^7-1)/(p-1): p=19 -> 4.7e7, filter to 2.6e6: fine in chunks)
# basis of H: integer kernel of w mod p
def nullspace_modp(A, p):
    A = np.array(A, dtype=np.int64) % p; m, n = A.shape; piv = []; r = 0; A = A.copy()
    for c in range(n):
        pr = next((i for i in range(r, m) if A[i, c] % p), None)
        if pr is None: continue
        A[[r, pr]] = A[[pr, r]]; A[r] = (A[r] * pow(int(A[r, c]), -1, p)) % p
        for i in range(m):
            if i != r and A[i, c]: A[i] = (A[i] - A[i, c] * A[r]) % p
        piv.append(c); r += 1
    free = [c for c in range(n) if c not in piv]; basis = []
    for f in free:
        v = np.zeros(n, dtype=np.int64); v[f] = 1
        for i, pc in enumerate(piv): v[pc] = (-A[i, f]) % p
        basis.append(v)
    return np.array(basis, dtype=np.int64)
H = nullspace_modp(w[None, :] % p, p)                          # 7 x 8
def normalise(V):
    V = V % p; f = (V != 0).argmax(1); l = V[np.arange(len(V)), f]; iv = np.array([0] + [pow(int(x), -1, p) for x in range(1, p)])[l]; return (V * iv[:, None]) % p
def keys(V): return (V % p) @ np.array(PW, dtype=np.int64)
lines = []
# projective points of P(H): coefficient vectors c in F_p^7 with first nonzero = 1
for lead in range(7):
    tail = 7 - lead - 1
    for chunk in itertools.product(range(p), repeat=min(tail, 3)) if tail > 3 else [()]:
        pass
    # enumerate: c = (0,...,0,1, c_{lead+1..6}) ; vectorised over the free coordinates in blocks
    rest = 7 - lead - 1
    size = p ** rest; block = 2_000_000
    for start in range(0, size, block):
        idx = np.arange(start, min(start + block, size), dtype=np.int64); C = np.zeros((len(idx), 7), dtype=np.int64); C[:, lead] = 1
        for k in range(rest): C[:, lead + 1 + k] = (idx // p**k) % p
        V = (C @ H) % p
        norms = np.einsum('ij,jk,ik->i', V, E, V) % p             # x E x = 2 N(x): zero iff N(x) ≡ 0 (p odd)
        lines.append(V[norms == 0])
L = np.vstack(lines); L = normalise(L); K = keys(L); order = np.argsort(K); L = L[order]; K = K[order]; nL = len(L)
assert nL == (p**6 - 1) // (p - 1), (nL, (p**6 - 1) // (p - 1)); print(f"p = {p}: {nL} null lines [{time.time()-t0:.0f}s]", flush=True)
def lookup(V): return np.searchsorted(K, keys(normalise(V)))
# ---- 2. orbits of G2(Z) on lines via union-find over a generating set (use 6 random group elements + verify by closure count)
rng = np.random.default_rng(0); gens = [G[i] for i in rng.choice(len(G), 8, replace=False)]
parent = np.arange(nL); 
def find(a):
    while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
    return a
for g in gens:
    img = lookup(L @ g)
    for a, b in zip(range(nL), img):
        ra, rb = find(a), find(b)
        if ra != rb: parent[max(ra, rb)] = min(ra, rb)
roots = np.array([find(a) for a in range(nL)]); line_reps = np.unique(roots); print(f"line orbits (under 8 random generators): {len(line_reps)} [{time.time()-t0:.0f}s]", flush=True)
# verify generation: full orbit sizes from stabilisers must match counts
sizes = collections.Counter(roots.tolist())
stabs = {}
for r in line_reps:
    img = lookup(np.einsum('j,gjk->gk', L[r], G)); st = np.nonzero(img == r)[0]; stabs[r] = st
    assert 12096 // len(st) == sizes[r], ("generators do not generate", r, 12096 // len(st), sizes[r])
print(f"line-level class set verified: {len(line_reps)} classes, stabilisers {sorted(collections.Counter(len(s) for s in stabs.values()).items())} [{time.time()-t0:.0f}s]", flush=True)
# ---- 3. null planes through a line: span(l, m) with m null, orthogonal to l, l*m = 0 (octonion product), m not in l
def omul(a, b): return np.einsum('i,j,ijk->k', a, b, STRUCT)
Sp = np.array(STRUCT, dtype=np.int64)
def planes_through(l):
    A = np.vstack([w[None, :], (E @ l)[None, :]]) % p; Ml = np.einsum('i,ijk->jk', l, Sp) % p
    cond = np.vstack([A, Ml.T]) % p; B2 = nullspace_modp(cond, p); d = len(B2)
    C = np.array(list(itertools.product(range(p), repeat=d)), dtype=np.int64)[1:]; M = (C @ B2) % p
    M = M[np.einsum('ij,jk,ik->i', M, E, M) % p == 0]
    # discard m on the line l: m is a multiple of l iff normalised(m) == normalised(l)
    M = M[keys(normalise(M)) != keys(l[None, :])[0]]
    pl = {}
    for m in M:
        key = plane_key(l, m)
        if key not in pl: pl[key] = m
    return list(pl.values())
def plane_key(l, m):
    pts = np.array([((a * l + b * m) % p) for a in range(p) for b in range(1, p)]); return tuple(sorted(set(keys(normalise(pts)).tolist())))
nflags = 0; classes = []           # (line rep index, plane rep m, stabiliser order)
for r in line_reps:
    l = L[r]; ms = planes_through(l); assert len(ms) == p + 1, (len(ms), p + 1)
    st = [G[i] for i in stabs[r]]
    # Stab(l)-orbits on the planes through l: plane key = set of its points off l
    pkeys = {plane_key(l, m): m for m in ms}; seen = set()
    for key, m in pkeys.items():
        if key in seen: continue
        orb = set()
        for g in st:
            mg = (m @ g) % p; orb.add(plane_key(l, mg))
        seen |= orb; classes.append((int(r), m, len(st) * 1 // (len(orb)), len(orb)))
        # stabiliser of the flag = |Stab(l)| / |orbit of the plane|
print(f"Iwahori classes: {len(classes)} [{time.time()-t0:.0f}s]", flush=True)
from fractions import Fraction as Fr
mass = sum(Fr(1, len(stabs[r]) // o) for r, m, s, o in classes)
pred = Fr((p**6 - 1) * (p + 1), 12096 * (p - 1))
print(f"mass {mass} vs predicted (p^6-1)(p+1)/((p-1)*12096) = {pred}: {'EXACT' if mass == pred else 'MISMATCH'}")
# ---- 4. transporters: BFS from each line rep over the generators (store parent and generator index)
par = -np.ones(nL, dtype=np.int64); gen_idx = -np.ones(nL, dtype=np.int64); par[line_reps] = line_reps
frontier = list(line_reps); gens_arr = np.array(gens)
while frontier:
    F = np.array(frontier); nxt = []
    for gi, g in enumerate(gens):
        img = lookup(L[F] @ g)
        for a, b in zip(F, img):
            if par[b] < 0: par[b] = a; gen_idx[b] = gi; nxt.append(b)
    frontier = nxt
assert (par >= 0).all(), "BFS did not reach every line"
inv_gens = [np.rint(np.linalg.inv(g.astype(float))).astype(np.int64) for g in gens]
def transporter(i):
    """matrix T with normalise(L[i] @ T) == rep line of i's orbit"""
    T = np.eye(8, dtype=np.int64); cur = i
    while par[cur] != cur: T = T @ inv_gens[gen_idx[cur]]; cur = par[cur]
    return T, cur
# class index of a flag (line vector l, plane vector m)
rep_pos = {int(r): k for k, r in enumerate(line_reps)}
class_of_plane = {}
for ci, (r, m, s_, o) in enumerate(classes):
    l = L[r]; st = [G[i] for i in stabs[r]]
    for g in st: class_of_plane[(int(r), plane_key(l, (m @ g) % p))] = ci
def class_of_flag(lv, mv):
    i = int(lookup(lv[None, :])[0]); T, r = transporter(i); lr = L[r]
    mt = (mv @ T) % p; return class_of_plane[(int(r), plane_key(lr, mt))]
# sanity: every class rep identifies to itself
for ci, (r, m, s_, o) in enumerate(classes): assert class_of_flag(L[r], m) == ci
# ---- 5. Hecke operators T_q from the stored neighbour maps
results = {}
for q in qs:
    maps = np.load(f'nbr_maps{q}.npy'); gq = np.rint(maps * q).astype(np.int64); qinv = pow(q, -1, p)
    h = len(classes); T = np.zeros((h, h), dtype=np.int64)
    for ci, (r, m, s_, o) in enumerate(classes):
        l = L[r]
        for g in gq:
            Tm = inv_mod((g * qinv) % p, p)                        # the neighbour map acts through its inverse mod p (validated convention)
            T[ci, class_of_flag((l @ Tm) % p, (m @ Tm) % p)] += 1
    rs = sorted(set(T.sum(1).tolist())); st_ = np.array([s_ for r, m, s_, o in classes], dtype=float)
    sym = np.max(np.abs(T / st_[:, None] - (T / st_[:, None]).T))
    ev = np.sort(np.linalg.eigvals(T.astype(float)).real)[::-1]
    print(f"T_{q}: row sums {rs}; mass-symmetry defect {sym:.1e}; top eigenvalues {np.round(ev[:6], 3).tolist()} [{time.time()-t0:.0f}s]", flush=True)
    results[q] = T
pickle.dump({'p': p, 'classes': [(r, m.tolist(), len(stabs[r]) // o) for r, m, s, o in classes], 'line_reps': line_reps.tolist()}, open(f'g2_iwahori_{p}.pkl', 'wb')); pickle.dump(results, open(f'g2_iwahori_{p}_T.pkl', 'wb'))
