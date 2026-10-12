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
# tempered bound at 2 and the Eisenstein congruences via the characteristic polynomial (exact integer arithmetic)
print("eigenvalues violating |T2+1| <= 56 (excluding 126):", [round(float(w), 3) for w in ev if abs(w + 1) > 56 + 1e-6 and abs(w - 126) > 1e-6])
pari = cypari2.Pari(); pari.allocatemem(4*10**9)
M = pari.matrix(h, h, [int(x) for x in T.flatten()]); cp = pari.charpoly(M); fac = pari.factor(cp)
print(f"charpoly factored: {int(pari.matsize(fac)[0])} irreducible factors; degrees {sorted(collections.Counter(int(pari.poldegree(fac[i,0])) for i in range(int(pari.matsize(fac)[0]))).items())}")
for ell in (5, 7, 127):
    carriers = []
    for i in range(int(pari.matsize(fac)[0])):
        f = fac[i, 0]; mlt = int(fac[i, 1])
        if int(pari.poldegree(f)) == 1 and int(-pari.polcoef(f, 0)) == 126: continue
        if int(pari.subst(f, pari.variable(f), 126)) % ell == 0: carriers.append((str(f)[:60], mlt))
    print(f"mod {ell}: factors of the characteristic polynomial congruent to the Eisenstein value 126 (degree-level test): {carriers}")
