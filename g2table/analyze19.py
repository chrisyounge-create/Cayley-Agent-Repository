import pickle, numpy as np, collections, sys, cypari2
from fractions import Fraction as Fr
p = int(sys.argv[1]) if len(sys.argv) > 1 else 19
d = pickle.load(open(f'g2_iwahori_{p}.pkl', 'rb')); import scipy.sparse as sp, os
if os.path.exists(f'g2_iwahori_{p}_T.npz'):
    _z = np.load(f'g2_iwahori_{p}_T.npz'); _T = {2: _z['T2'].astype(np.int64), 5: _z['T5'].astype(np.int64)}
else:
    _T = pickle.load(open(f'g2_iwahori_{p}_T.pkl', 'rb')); _T = {q: (M.toarray() if sp.issparse(M) else M) for q, M in _T.items()}
T = _T[2]
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
    T5 = _T[5]
    M = T.astype(float) + 0.3719 * T5.astype(float); w, V = np.linalg.eig(M); V = V.real
    l2 = np.array([(V[:, i] @ T @ V[:, i]) / (V[:, i] @ V[:, i]) for i in range(h)]); l5 = np.array([(V[:, i] @ T5 @ V[:, i]) / (V[:, i] @ V[:, i]) for i in range(h)])
    pred5 = {(21, 630): 3, (9, 1098): 2, (-3.456, 175.438): 3, (36.456, 574.562): 3, (32.549, 835.663): 2, (54.085, 330.978): 2, (49.131, 1425.087): 2, (-0.764, 1044.272): 2}
    print("\nJOINT (T2, T5): predicted lift pairs vs observed multiplicities:")
    for (a, b), m in pred5.items():
        obs = int(np.sum((np.abs(l2 - a) < 2e-3) & (np.abs(l5 - b) < 2e-2))); print(f"   ({a}, {b}): predicted x{m}, observed x{obs}  {'OK' if obs == m else 'MISMATCH'}")
    print("T5 row sums:", sorted(set(T5.sum(1).tolist())), "| tempered-bound violations at 5 (|T5+1| > 875; Eisenstein 19530):", int(np.sum(np.abs(l5 + 1) > 875 + 1e-6)) - 1)
    # Eisenstein component mod ell: h - rank over F_ell of [T2 - 126 ; T5 - 19530]  (numpy elimination)
    from modrank import rank_mod
    for ell in (127, 7, 5):
        A = np.vstack([(T - 126 * np.eye(h, dtype=np.int64)), (T5 - 19530 * np.eye(h, dtype=np.int64))])
        rk = rank_mod(A, ell); print(f"   mod {ell}: dimension of the joint Eisenstein eigenspace (ker of T2-126 and T5-19530 over F_{ell}) = {h - rk}", flush=True)
    print("   predictions: mod 127 -> 1 + 8 (quartic lifts) = 9 (+ ~1/127^2 chance per other form); mod 7 -> 1 + 3 + 6 (lifts 1 and 3) = 10 plus ~h/49 ≈ 90 chance coincidences (two operators cannot isolate it); mod 5 -> 1 + 3 (lift 1) plus ~h/25 ≈ 177 chance")
except KeyError: print("(no T5 in this run)")

# ---- carriers by eigen-system: minimal polynomials of (T2, T5) eigenvalues and congruence to the Eisenstein values modulo ell
try:
    import cypari2; pari = cypari2.Pari(); pari.allocatemem(2 * 10**9)
    E2, E5 = 126, 19530
    seen = set(); carriers = {5: [], 7: [], 127: []}
    for i in range(h):
        a, b = l2[i], l5[i]; key = (round(float(a), 4), round(float(b), 4))
        if key in seen: continue
        seen.add(key)
        if abs(a - E2) < 1e-6 and abs(b - E5) < 1e-3: continue
        for d in (1, 2, 3, 4, 6, 8):
            pol = pari.algdep(pari(repr(float(a))), d, 20)
            if int(pari.poldegree(pol)) <= d and abs(complex(pari.subst(pol, 'x', pari(repr(float(a)))))) < 1e-6 * (1 + abs(a))**d:
                break
        mult = int(np.sum((np.abs(l2 - a) < 2e-3) & (np.abs(l5 - b) < 2e-2)))
        # congruence test on the norm of (lambda2 - 126): N(lambda2 - 126) = pol(126) up to sign for monic pol
        try: n2 = int(pari.subst(pol, 'x', E2)); 
        except Exception: continue
        for ell in (5, 7, 127):
            if n2 % ell == 0:
                carriers[ell].append((key, f"deg {int(pari.poldegree(pol))}", f"x{mult}"))
    for ell in (5, 7, 127):
        print(f"\neigen-systems with N(T2 - 126) ≡ 0 mod {ell} (degree of the T2 minimal polynomial, multiplicity): {carriers[ell][:25]}")
except Exception as e: print("carrier identification failed:", str(e)[:200])
