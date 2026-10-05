"""Level-5 operator sampler (Path B for T(omega_1)(2)): random 2-neighbours of (J, S), transported six-space as an ambient lattice,
fingerprint, orbit identification. Ambient = J_Z coordinates; J_E = rows of LamE_lll/4."""
import numpy as np, json, time, sys, collections
import albert, f4_p5 as F, pathB as P
from f4_nbrs_head import hnf_rows, hensel, neighbour, classify, kernel_mod, cross_matrix
import f4_nbrs_head as NB
from neighbors_struct import nullspace_modp
rng = np.random.default_rng(int(sys.argv[-1]) if len(sys.argv) > 1 and sys.argv[-1].isdigit() else 0); NB.rng = rng
L4 = np.load('LamE_lll.npy').astype(np.int64)            # 4 * (J_E basis) in ambient coordinates
GRAMZ = albert.GRAM.astype(np.int64)
def ambient_gram(B4):                                     # B4 = 4 * basis rows (integer) -> Gram of the lattice (integer, exact)
    G = B4 @ GRAMZ @ B4.T; assert not (G % 16).any(); return G // 16
def rank_one_mod2(J='Z'):
    """random nonzero rank-one, trace-zero point mod 2 of J (in J's own coordinates)"""
    sharp, trace = (albert.sharp, albert.trace)
    while True:
        v = rng.integers(0, 2, 27)
        if v.any() and not (sharp(v) % 2).any() and trace(v) % 2 == 0: return v
def neighbour_lattice(v0):
    vb = hensel(v0)
    if vb is None: return None
    return neighbour(vb)                                  # 4 * basis of the neighbour J' (in J_Z coordinates)
def transport(S, Lam4):
    """six-space S of J_Z (mod 5) -> lattice M' = Lambda_S[1/2] ∩ J' ; returns 4 * basis of M' (integer rows, ambient coords)"""
    G0 = P.G0_of(S) % 5                                   # 21 x 27 over F_5: Lambda_S = G0 + 5 J
    # coefficient vectors c (mod 5) with c . Lam4 ∈ G0 (mod 5): kernel of c -> (c Lam4 mod 5) modulo G0
    Q = nullspace_modp(G0, 5)                             # 6 x 27: functionals vanishing on G0  (rows q with G0 q^T = 0)
    Q = np.array(Q, dtype=np.int64) % 5
    Acond = (Lam4 % 5) @ Q.T % 5                          # 27 x 6: c . Acond = 0 mod 5
    K = np.array(nullspace_modp(Acond.T % 5, 5), dtype=np.int64) % 5   # 21 x 27 coefficient vectors
    rows = [k @ Lam4 for k in K] + [5 * r for r in Lam4]
    return hnf_rows(rows)                                 # 4 * basis of M'
if __name__ == "__main__":
    recs = [json.loads(l) for f in __import__('glob').glob('../results/36658053553/**/*.jsonl', recursive=True) for l in open(f) if l.strip()]
    recs = [r for r in recs if r.get('stab')]
    keyZ = collections.Counter(tuple(r['inv']) for r in recs); print("J_Z records:", len(recs), "| distinct inv:", len(keyZ))
    recsE = [json.loads(l) for f in __import__('glob').glob('../results/3709*/**/*.jsonl', recursive=True) + __import__('glob').glob('../results/3717*/**/*.jsonl', recursive=True) for l in open(f) if l.strip()]
    keyE = collections.Counter(tuple(r['inv']) for r in recsE if r.get('stab')); print("J_E distinct inv:", len(keyE))
    r0 = min(recs, key=lambda r: r['stab']); S = np.array(r0['S'], dtype=np.int64); print("test class: J_Z, stab 1, inv", r0['inv'])
    t0 = time.time(); stats = collections.Counter(); n = 0
    while n < int(sys.argv[1]) and time.time() - t0 < 200:
        Lam4 = neighbour_lattice(rank_one_mod2())
        if Lam4 is None: stats['hensel fail'] += 1; continue
        typ = classify(Lam4); M4 = transport(S, Lam4); G = ambient_gram(M4)
        inv = tuple(int(x) for x in P.pari.qfrep(P.gp_mat(G), 4, 1))
        known = 'Z-key' if inv in keyZ else ('E-key' if inv in keyE else 'UNKNOWN')
        stats[(typ, known)] += 1; n += 1
    print(f"{n} neighbours in {time.time()-t0:.0f}s:", dict(stats))
