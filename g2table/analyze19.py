import pickle, numpy as np, collections, sys, cypari2
from fractions import Fraction as Fr
p = int(sys.argv[1]) if len(sys.argv) > 1 else 19
d = pickle.load(open(f'g2_iwahori_{p}.pkl', 'rb')); T = pickle.load(open(f'g2_iwahori_{p}_T.pkl', 'rb'))[2]
h = T.shape[0]; stabs = [c[2] for c in d['classes']]
print(f"level {p}: h = {h}; mass {sum(Fr(1, s) for s in stabs)}; row sums {sorted(set(T.sum(1).tolist()))}")
ev = np.sort(np.linalg.eigvals(T.astype(float)).real)[::-1]; mult = collections.Counter(np.round(ev, 3))
pred = {21: (3, 'lift of a2=-2 (carries 5 and 7)'), 9: (2, 'lift of a2=-6 (no congruence)'), -3.456: (3, 'quadratic lift (7)'), 36.456: (3, 'quadratic lift (7)'),
        32.549: (2, 'quartic lift (127)'), 54.085: (2, 'quartic lift (127)'), 49.131: (2, 'quartic lift (127)'), -0.764: (2, 'quartic lift (127)')}
print("\nBLIND PREDICTION vs COMPUTATION (T2 eigenvalue: predicted multiplicity -> observed multiplicity):")
ok = True
for v, (m, name) in sorted(pred.items(), reverse=True):
    obs = sum(c for w, c in mult.items() if abs(w - v) < 2e-3); flag = 'OK' if obs == m else 'MISMATCH'; ok &= obs == m
    print(f"   {v:>8}: predicted x{m}, observed x{obs}  {flag}   {name}")
print("all lift predictions correct:", ok)
# joint eigen-systems (T2, T5) when T5 is available; predicted lift pairs
try:
    T5 = pickle.load(open(f'g2_iwahori_{p}_T.pkl', 'rb'))[5]
    M = T.astype(float) + 0.3719 * T5.astype(float); w, V = np.linalg.eig(M); V = V.real
    l2 = np.array([(V[:, i] @ T @ V[:, i]) / (V[:, i] @ V[:, i]) for i in range(h)]); l5 = np.array([(V[:, i] @ T5 @ V[:, i]) / (V[:, i] @ V[:, i]) for i in range(h)])
    pred5 = {(21, 630): 3, (9, 1098): 2, (-3.456, 175.438): 3, (36.456, 574.562): 3, (32.549, 835.663): 2, (54.085, 330.978): 2, (49.131, 1425.087): 2, (-0.764, 1044.272): 2}
    print("\nJOINT (T2, T5): predicted lift pairs vs observed multiplicities:")
    for (a, b), m in pred5.items():
        obs = int(np.sum((np.abs(l2 - a) < 2e-3) & (np.abs(l5 - b) < 2e-2))); print(f"   ({a}, {b}): predicted x{m}, observed x{obs}  {'OK' if obs == m else 'MISMATCH'}")
    print("T5 row sums:", sorted(set(T5.sum(1).tolist())), "| tempered-bound violations at 5 (|T5+1| > 875):", int(np.sum(np.abs(l5 + 1) > 875 + 1e-6)) - 1)
    # Eisenstein component mod ell: h - rank over F_ell of [T2 - 126 ; T5 - 19530]
    pari = cypari2.Pari(); pari.allocatemem(6 * 10**9)
    for ell in (5, 7, 127):
        A = np.vstack([(T - 126 * np.eye(h, dtype=np.int64)) % ell, (T5 - 19530 * np.eye(h, dtype=np.int64)) % ell])
        Mp = pari.matrix(2 * h, h, [int(x) for x in A.flatten()]); rk = int(pari.matrank(Mp * pari.Mod(1, ell)))
        print(f"   mod {ell}: dimension of the Eisenstein eigenspace (joint kernel of T2-126, T5-19530 over F_{ell}) = {h - rk}")
except KeyError: print("(no T5 in this run)")
