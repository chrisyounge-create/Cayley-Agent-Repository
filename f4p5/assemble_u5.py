import json, glob, sys, numpy as np, collections
from fractions import Fraction as Fr
reps = json.load(open('reps_ambient.json')); n = len(reps)
rows = {}
for f in glob.glob((sys.argv[1] if len(sys.argv) > 1 else '../results/u5') + '/**/u5_*.json', recursive=True):
    for rec in json.load(open(f)): rows[rec['class']] = rec
U = np.zeros((n, n), dtype=np.int64); unid = 0; missing = [i for i in range(n) if i not in rows]
for i, rec in rows.items():
    for k, v in rec['counts'].items():
        if k == 'unidentified': unid += v
        else: U[i, int(k)] += v
print(f"rows computed: {len(rows)} of {n}; unidentified lifts: {unid}; rows with errors: {sum(1 for r in rows.values() if r['note'])}; missing rows: {missing[:10]}")
rs = U.sum(1); print("row sums (should all be 5):", dict(collections.Counter(rs[list(rows)].tolist())))
m = [Fr(1, r['stab']) if r['stab'] else None for r in reps]; bad = 0; checked = 0
for i in rows:
    for j in rows:
        if m[i] is None or m[j] is None: continue
        checked += 1
        if U[i, j] * m[i] != U[j, i] * m[j]: bad += 1
print(f"mass symmetry U_ij m_i = U_ji m_j: {bad} violations in {checked} ordered pairs")
idx = [i for i in rows if rs[i] == 5]; Us = U[np.ix_(idx, idx)].astype(float)
w = np.linalg.eigvals(Us); w = np.sort(w.real)[::-1]
print(f"spectrum of U on the {len(idx)} complete rows: top {np.round(w[:8], 4).tolist()}; distinct values (rounded 1e-6): {len(set(np.round(w, 6)))}")
print("eigenvalue multiplicities:", sorted(collections.Counter(np.round(w, 6)).items(), key=lambda kv: -kv[0])[:16])
np.save('U5.npy', U); json.dump(idx, open('U5_idx.json', 'w'))
