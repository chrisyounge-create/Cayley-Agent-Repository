"""Exact degree-5 operator at level Q5: U f(J,S) = sum over the five lifts (J_i, S_i) of S. usage: u5_operator.py SIDE j njobs budget"""
import sys, json, time, collections, numpy as np
SIDE, j, njobs, budget = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
if SIDE == 'E': import je_setup
import f4_p5 as F, pathB as P
from identify import setup_amb, same_orbit_amb
P.pari.allocatemem(2*10**9)
L4 = np.load('LamE_lll.npy').astype(np.int64)
reps = json.load(open('reps_ambient.json')); mine = [r for r in reps if r['side'] == SIDE][j::njobs]
J80 = {'Z': 80 * np.eye(27, dtype=np.int64), 'E': 20 * L4}                       # 80 * ambient basis of J_Z, J_E
byinv = collections.defaultdict(list)
for idx, r in enumerate(reps): byinv[tuple(r['inv'])].append(idx)
cache = {}
def rep_setup(idx):
    if idx not in cache: cache[idx] = setup_amb(5 * np.array(reps[idx]['B16'], dtype=np.int64), J80[reps[idx]['side']], scale=80)
    return cache[idx]
lift_amb80 = (lambda Lam: 16 * Lam) if SIDE == 'Z' else (lambda Lam: 4 * (Lam @ L4))   # Lam = 5 * basis of J' in J's coordinates
t0 = time.time(); out = []
for r in mine:
    S = np.array(r['S'], dtype=np.int64); M80 = 5 * np.array(r['B16'], dtype=np.int64); counts = collections.Counter(); note = ''
    try:
        Ls = F.neighbours_of_S(S)
        if Ls is None: raise RuntimeError('no lifts')
        for Lam in Ls:
            b = setup_amb(M80, lift_amb80(Lam), scale=80)
            hit = next((idx for idx in byinv.get(b['inv'], []) if same_orbit_amb(rep_setup(idx), b)), None)
            counts['unidentified' if hit is None else hit] += 1
    except Exception as e: note = str(e)[:80]
    out.append({'class': r['id'], 'side': SIDE, 'counts': {str(k): v for k, v in counts.items()}, 'note': note})
    json.dump(out, open(f'u5_{SIDE}_{j}.json', 'w'))
    print(f"class {r['id']} (stab {r['stab']}): lifts -> {dict(counts)} {note}  [{time.time()-t0:.0f}s]", flush=True)
    if time.time() - t0 > budget: break
print("done")
