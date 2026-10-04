"""Proof of Lemma 4.3 of Part IV by finite computation: for distinct same-type quaternion suborders H1, H2 of the Coxeter order,
the bad primes (those p for which the conics of H1 and H2 mod p can share a point) divide the saturation index of Im H1 + Im H2
or the norm of a common imaginary direction. We compute these over all pairs and collect their prime factors."""
import numpy as np, itertools, collections, cypari2, time
from math import gcd
from fractions import Fraction
src = open('g2_level3_parahorics.py').read().split("results = {}")[0]; exec(src)
pari = cypari2.Pari(); G8 = np.array(E8, dtype=np.int64); one = np.array(ONE, dtype=np.int64); omul = lambda x, y: np.einsum('i,j,ijk->k', x, y, S_)
q = pari.qfminim(pari.matrix(8, 8, [int(x) for x in G8.flatten()]), 2); M = q[2]; nc = int(pari.matsize(M)[1])
units = [np.array([int(M[i, j]) for i in range(8)]) for j in range(nc)]; units = units + [-u for u in units]
imag = [u for u in units if u @ G8 @ one == 0]
def sat_basis(vecs):
    """Z-basis of (Q-span of vecs) ∩ Z^8, via PARI (matkerint of the integer constraint matrix)."""
    Bm = pari.matrix(len(vecs), 8, [int(x) for v in vecs for x in v])
    K = pari.matkerint(Bm)                      # integer kernel of B (columns c with B c = 0): constraints
    if int(pari.matsize(K)[1]) == 0: return np.array([list(map(int, pari(f'Vec({Bm}[{i+1},])'))) for i in range(len(vecs))])
    Kt = pari.mattranspose(K)                   # constraint rows
    S = pari.matkerint(Kt)                      # integer vectors killed by all constraints = saturation
    S = pari.mattranspose(S); r, c = int(pari.matsize(S)[0]), int(pari.matsize(S)[1])
    return np.array([[int(S[i, j]) for j in range(c)] for i in range(r)], dtype=np.int64)
def hnf_key(B):
    H = pari.mathnf(pari.mattranspose(pari.matrix(B.shape[0], B.shape[1], [int(x) for x in B.flatten()])))
    return str(H)
t0 = time.time()
# distinct quaternion suborders from the zero divisors (a,b): saturated Z-span of 1,a,b,ab
subs = {}
for a in imag:
    for b in imag:
        if a @ G8 @ b != 0: continue
        S = sat_basis([one, a, b, omul(a, b)]); key = hnf_key(S)
        if key not in subs:
            Pm = S.T.astype(float) @ np.linalg.pinv(S.T.astype(float)); nunits = sum(1 for u in units if np.allclose(Pm @ u, u))
            subs[key] = (S, nunits)
byt = collections.defaultdict(list)
for key, (S, n) in subs.items(): byt[n].append(S)
print("distinct quaternion suborders:", {n: len(v) for n, v in byt.items()}, f"[{time.time()-t0:.0f}s]")
w = (G8 @ one).astype(np.int64)              # trace functional
def imag_part(S):                            # Z-basis of S ∩ 1^perp: integer kernel of the trace on the lattice S
    # vectors x = c @ S with (c @ S) . w = 0  -> c in integer kernel of the row vector S w
    Sw = pari.matrix(1, S.shape[0], [int(x) for x in (S @ w)]); K = pari.matkerint(Sw); K = pari.mattranspose(K)
    C = np.array([[int(K[i, j]) for j in range(int(pari.matsize(K)[1]))] for i in range(int(pari.matsize(K)[0]))], dtype=np.int64)
    return C @ S
def bareiss_det(Mx):
    n = len(Mx); A = [row[:] for row in Mx]; sign = 1; prev = 1
    for k in range(n - 1):
        if A[k][k] == 0:
            sw = next((i for i in range(k+1, n) if A[i][k] != 0), None)
            if sw is None: return 0
            A[k], A[sw] = A[sw], A[k]; sign = -sign
        for i in range(k+1, n):
            for j in range(k+1, n): A[i][j] = (A[i][j]*A[k][k] - A[i][k]*A[k][j]) // prev
        prev = A[k][k]
    return sign * A[n-1][n-1]
def sat_index(B):
    """index of the lattice spanned by the rows of B (r x 8, rank r) in its saturation = gcd of the r x r minors."""
    r = np.linalg.matrix_rank(B.astype(float)); g = 0
    rows = [list(map(int, row)) for row in B]
    # choose r independent rows first (B may have 6 rows of rank 5)
    if r < B.shape[0]:
        for comb in itertools.combinations(range(B.shape[0]), r):
            if np.linalg.matrix_rank(B[list(comb)].astype(float)) == r: rows = [rows[i] for i in comb]; break
    for cols in itertools.combinations(range(8), r):
        d = bareiss_det([[row[c] for c in cols] for row in rows]); g = gcd(g, abs(d))
    return r, g
def common_direction(M1, M2):
    """primitive integer generator of the rank-1 intersection of the Q-spans, or None."""
    A = np.vstack([M1, -M2]).astype(float).T          # 8 x 6: find c with M1^T c1 = M2^T c2
    u, s, vt = np.linalg.svd(A); ker = vt[np.sum(s > 1e-9):]
    if len(ker) == 0: return None
    v = ker[0][:3] @ M1; v = v / np.max(np.abs(v)); 
    # scale to integers
    fr = [Fraction(x).limit_denominator(10**6) for x in v]; den = 1
    for f in fr: den = den * f.denominator // gcd(den, f.denominator)
    iv = np.array([int(f * den) for f in fr]); g = 0
    for x in iv: g = gcd(g, abs(int(x)))
    return iv // g
def primes_of(n):
    n = abs(int(n)); ps = set(); d = 2
    while d * d <= n:
        while n % d == 0: ps.add(d); n //= d
        d += 1
    if n > 1: ps.add(n)
    return ps
for ntype, Slist in sorted(byt.items()):
    Ms = [imag_part(S) for S in Slist]; bad = set(); ranks = collections.Counter(); idxs = collections.Counter(); norms = collections.Counter()
    for i in range(len(Ms)):
        for j in range(i+1, len(Ms)):
            r, g = sat_index(np.vstack([Ms[i], Ms[j]])); ranks[r] += 1; idxs[g] += 1; bad |= primes_of(g)
            if r == 5:
                v = common_direction(Ms[i], Ms[j]); Nv = int(v @ G8 @ v) // 2; norms[Nv] += 1; bad |= primes_of(Nv)
    print(f"type with {ntype} units ({len(Ms)} suborders): pairs by rank of M1+M2: {dict(ranks)}; saturation indices: {dict(idxs)}; common-direction norms (rank-5 pairs): {dict(norms)}")
    print(f"   => bad primes for this type: {sorted(bad)}   [{time.time()-t0:.0f}s]")
