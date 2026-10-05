"""Assemble the sampled level-5 operator: counts per class -> row-normalised transition estimates -> integer-degree matrix
T_hat (row sums 139230) -> denoise with mass symmetry -> eigenvalues vs the 5a prediction list."""
import json, glob, sys, numpy as np, collections
from fractions import Fraction as Fr
reps = json.load(open('reps_ambient.json')); n = len(reps); DEG = 139230
files = glob.glob(sys.argv[1] + '/**/opsample_*.json', recursive=True) if len(sys.argv) > 1 else glob.glob('opsample_*.json')
rows = {}
for f in files:
    for rec in json.load(open(f)):
        i = rec['class']; c = rows.setdefault(i, collections.Counter())
        for k, v in rec['counts'].items(): c[k] += v
print("classes with sampled rows:", len(rows), "of", n)
T = np.zeros((n, n)); N = np.zeros(n); unid = np.zeros(n)
for i, c in rows.items():
    tot = sum(c.values()); N[i] = tot; unid[i] = c.get('unidentified', 0)
    for k, v in c.items():
        if k != 'unidentified': T[i, int(k)] = DEG * v / tot
print(f"neighbours per class: min {N[N>0].min():.0f}, median {np.median(N[N>0]):.0f}; unidentified fraction overall {unid.sum()/max(N.sum(),1):.3f}")
# mass weights m_i = 1/|Stab_i| (classes without a stabiliser get weight from their sampled row via symmetry later)
m = np.array([1.0 / r['stab'] if r.get('stab') else np.nan for r in reps])
# symmetrise: T_ij m_i = T_ji m_j  ->  combine the two estimates of the symmetric quantity S_ij = T_ij m_i weighted by sample counts
S = np.full((n, n), np.nan)
for i in range(n):
    for j in range(n):
        est = []
        if N[i] > 0 and not np.isnan(m[i]): est.append((T[i, j] * m[i], N[i]))
        if N[j] > 0 and not np.isnan(m[j]): est.append((T[j, i] * m[j], N[j]))
        if est: S[i, j] = sum(e * w for e, w in est) / sum(w for e, w in est)
Tsym = np.where(np.isnan(S), T, S / np.where(np.isnan(m), 1, m)[:, None])
Tsym = np.nan_to_num(Tsym); rs = Tsym.sum(1); Tsym = Tsym * (DEG / np.where(rs > 0, rs, DEG))[:, None]      # restore exact row sums
w = np.sort(np.linalg.eigvals(Tsym).real)[::-1]
pred = {139230: 'Eisenstein', 14157.94: 'F4xPGL2 lift wt12 quad (601)', 12285: 'F4xPGL2 lift wt12 rational (31)', 8631: 'Theta(Delta) (691)', 6668.08: 'Sp6xSL2 lift wt8 quad (313)', 4868.06: 'F4xPGL2 lift wt12 quad (601)', 1935: '207-type lift wt6 (31)', 1297.92: 'Sp6xSL2 lift wt8 quad (313)', 91: 'Sp6xSL2 lift wt8 rational (13)'}
print("largest eigenvalues of the symmetrised estimate:", np.round(w[:12], 1))
for p, name in pred.items():
    d = np.min(np.abs(w - p)); print(f"   predicted {p:>9}: nearest eigenvalue {w[np.argmin(np.abs(w - p))]:10.1f} (distance {d:7.1f}, {100*d/p:5.1f}%)  {name}")
np.save('T_hat.npy', Tsym)
