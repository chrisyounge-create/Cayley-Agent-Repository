"""Representative database in ambient (J_Z) coordinates: one record per distinct (inv, stab, lift_stab, types) key for now
(multi-orbit keys get their extra representatives from the splitreps runs later). Writes reps_ambient.json."""
import json, glob, sys, numpy as np, collections
side = sys.argv[1]
if side == 'E': import je_setup
import f4_p5 as F, pathB as P
from f4_nbrs_head import hnf_rows
L4 = np.load('LamE_lll.npy').astype(np.int64)
files = glob.glob('../results/36658053553/**/*.jsonl', recursive=True) if side == 'Z' else glob.glob('../results/3709*/**/*.jsonl', recursive=True) + glob.glob('../results/3717*/**/*.jsonl', recursive=True)
recs = [json.loads(l) for f in files for l in open(f) if l.strip()]; recs = [r for r in recs if r.get('stab')]
key = lambda r: (tuple(r['inv']), r['stab'], tuple(sorted(r['lift_stab'])), tuple(sorted(r['types'])))
seen = {}; out = []
for r in recs:
    k = key(r)
    if k in seen: continue
    seen[k] = 1; S = np.array(r['S'], dtype=np.int64); G0 = P.G0_of(S)
    B = hnf_rows([list(map(int, g)) for g in G0] + [list(map(int, 5*e)) for e in F.E])     # basis of M in J's own coordinates
    B4 = 4 * B if side == 'Z' else B @ L4                                                    # 4 * basis in ambient coordinates
    out.append({'side': side, 'inv': list(k[0]), 'stab': k[1], 'lift_stab': list(k[2]), 'types': list(k[3]), 'B4': B4.tolist(), 'hits': sum(1 for q in recs if key(q) == k)})
json.dump(out, open(f'reps_ambient_{side}.json', 'w')); print(side, "keys:", len(out))
