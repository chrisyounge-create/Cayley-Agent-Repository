"""Assemble the sampled level-5 operator from all passes: counts -> rates -> integer-degree estimates (row sums 139230) ->
mass-symmetric reconciliation -> eigenvalues vs the 5a prediction list."""
import json, glob, sys, numpy as np, collections
reps = json.load(open('reps_ambient.json')); n = len(reps); DEG = 139230
dirs = sys.argv[1:] if len(sys.argv) > 1 else ['../results/37386453034']
rows = collections.defaultdict(collections.Counter)
for d in dirs:
    for f in glob.glob(d + '/**/opsample_*.json', recursive=True):
        for rec in json.load(open(f)):
            for k, v in rec['counts'].items(): rows[rec['class']][k] += v
N = np.zeros(n); T = np.zeros((n, n)); U = np.zeros(n)
for i, c in rows.items():
    ident = sum(v for k, v in c.items() if k != 'unidentified'); U[i] = c.get('unidentified', 0); N[i] = ident
    for k, v in c.items():
        if k != 'unidentified': T[i, int(k)] = DEG * v / ident
have = N > 0; print(f"rows available: {have.sum()} of {n} (Z {sum(1 for i in range(n) if have[i] and reps[i]['side']=='Z')}/53, E {sum(1 for i in range(n) if have[i] and reps[i]['side']=='E')}/205); identified neighbours {int(N.sum())}, unidentified {int(U.sum())}")
m = np.array([1.0 / r['stab'] if r.get('stab') else np.nan for r in reps])
# mass-symmetric reconciliation: S_ij = T_ij m_i = T_ji m_j; weight each estimate by its sample count
S = np.full((n, n), np.nan)
for i in range(n):
    for j in range(n):
        est = []
        if have[i] and not np.isnan(m[i]): est.append((T[i, j] * m[i], N[i]))
        if have[j] and not np.isnan(m[j]): est.append((T[j, i] * m[j], N[j]))
        if est: S[i, j] = sum(e * w for e, w in est) / sum(w for e, w in est)
Trec = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if not np.isnan(S[i, j]) and not np.isnan(m[i]): Trec[i, j] = S[i, j] / m[i]
        elif have[i]: Trec[i, j] = T[i, j]
rs = Trec.sum(1); filled = rs > 0; Trec[filled] *= (DEG / rs[filled])[:, None]
print(f"rows after symmetric fill: {filled.sum()} of {n}; rows still empty: {[(i, reps[i]['side'], reps[i]['stab']) for i in range(n) if not filled[i]][:8]}")
idx = np.where(filled)[0]; Tsub = Trec[np.ix_(idx, idx)]; rs2 = Tsub.sum(1); Tsub *= (DEG / rs2)[:, None]
w = np.linalg.eigvals(Tsub); w = np.sort(w.real)[::-1]
pred = [(139230, 'Eisenstein'), (14157.94, 'F4xPGL2 lift wt12 quad (601)'), (12285, 'F4xPGL2 lift wt12 rational (31)'), (8631, 'Theta(Delta) (691)'), (6668.08, 'Sp6xSL2 lift wt8 quad (313)'), (4868.06, 'F4xPGL2 lift wt12 quad (601)'), (1935, '207-type lift wt6 (31)'), (1297.92, 'Sp6xSL2 lift wt8 quad (313)'), (91, 'Sp6xSL2 lift wt8 rational (13)')]
print("largest eigenvalues of the reconciled estimate (restricted to filled rows):", np.round(w[:14], 1).tolist())
for p, name in pred:
    k = np.argmin(np.abs(w - p)); print(f"   predicted {p:>9}: nearest {w[k]:10.1f}  (off by {100*(w[k]-p)/p:+6.1f}%)   {name}")
np.save('T_hat.npy', Trec); json.dump({'filled': idx.tolist()}, open('T_hat_meta.json', 'w'))
