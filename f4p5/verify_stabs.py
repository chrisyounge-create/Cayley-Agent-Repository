import json, glob, collections, time, sys, numpy as np, pathB as P, f4_p5 as F
from f4_nbrs_head import hnf_rows
E = np.eye(27, dtype=np.int64)
def is_jordan_exact(A):
    if not np.array_equal(P.I27 @ A, P.I27): return False
    for i in range(27):
        if not np.array_equal(F.sharp(E[i] @ A), F.sharp(E[i]) @ A): return False
    for i in range(27):
        for j in range(i+1, 27):
            x = E[i] + E[j]
            if not np.array_equal(F.sharp(x @ A), F.sharp(x) @ A): return False
    return True
rows = [json.loads(l) for f in sorted(glob.glob('/home/claude/gh_pathB2/**/*.jsonl', recursive=True)) for l in open(f) if l.strip()]
groups = collections.OrderedDict()
for r in rows:
    if 'error' in r or r.get('big'): continue
    groups.setdefault((tuple(r['inv']), r['stab'], r['order_OM'], tuple(sorted(r['lift_stab']))), r)
keys = list(groups); lo, hi = int(sys.argv[1]), int(sys.argv[2]); t = time.time()
for idx in range(lo, min(hi, len(keys))):
    k = keys[idx]; r = groups[k]
    S = np.array(r['S'], dtype=np.int64); G0 = P.G0_of(S)
    B = hnf_rows([list(map(int, g)) for g in G0] + [list(map(int, 5*e)) for e in F.E]); Gram = B @ F.GRAM @ B.T
    res = P.pari.qfauto(P.gp_mat(Gram)); OM = P.closure([P.np_mat(g) for g in res[1]])
    n = sum(1 for A in P.jordan_stab(B, OM) if is_jordan_exact(A))
    print(f"{idx+1}/{len(keys)} stab recorded {k[1]:>3}, exact {n:>3}  {'OK' if n == k[1] else 'MISMATCH'}  [{time.time()-t:.0f}s]", flush=True)
