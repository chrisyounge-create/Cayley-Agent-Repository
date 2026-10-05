"""Production sampler for T(omega_1)(2) at level 5. usage: opsample.py SIDE j njobs quota budget seed
Classes of SIDE (Z or E) from reps_ambient.json (entries with S in the side's own coordinates and B16 ambient);
for each class: random 2-neighbours, transported six-space (ambient, 16-scaled), fingerprint, exact identification."""
import sys, json, time, collections, glob, numpy as np
SIDE, j, njobs, quota, budget, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
if SIDE == 'E': import je_setup
import albert, f4_p5 as F, pathB as P, f4_nbrs_head as NB
from f4_nbrs_head import hnf_rows
from neighbors_struct import nullspace_modp
from identify import setup_amb, same_orbit_amb
rng = np.random.default_rng(seed); NB.rng = rng; P.pari.allocatemem(2*10**9)
L4 = np.load('LamE_lll.npy').astype(np.int64)
if SIDE == 'E': NB.GRAM, NB.sharp, NB.trace = je_setup.GRAM_E, je_setup.sharp_E, je_setup.trace_E
def rank_one_mod2():
    while True:
        v = rng.integers(0, 2, 27)
        if v.any() and not (NB.sharp(v) % 2).any() and NB.trace(v) % 2 == 0: return v
def transport(S, Lam4):
    G0 = P.G0_of(S) % 5; Q = np.array(nullspace_modp(G0, 5), dtype=np.int64) % 5
    Acond = (Lam4 % 5) @ Q.T % 5; K = np.array(nullspace_modp(Acond.T % 5, 5), dtype=np.int64) % 5
    return hnf_rows([k @ Lam4 for k in K] + [5 * r for r in Lam4])                    # 4 * basis of M' in the side's coordinates
to_amb16 = (lambda M4: 4 * M4) if SIDE == 'Z' else (lambda M4: M4 @ L4)                # 16 * ambient basis
J16_of = {'Z': 16 * np.eye(27, dtype=np.int64), 'E': 4 * L4}
reps = json.load(open('reps_ambient.json')); mine = [r for r in reps if r['side'] == SIDE][j::njobs]
byinv = collections.defaultdict(list)
for idx, r in enumerate(reps): byinv[tuple(r['inv'])].append(idx)
cache = {}
def rep_setup(idx):
    if idx not in cache: cache[idx] = setup_amb(reps[idx]['B16'], J16_of[reps[idx]['side']])
    return cache[idx]
t0 = time.time(); out = []
for r in mine:
    S = np.array(r['S'], dtype=np.int64); counts = collections.Counter(); n = 0; unid = []
    while n < quota and time.time() - t0 < budget:
        vb = NB.hensel(rank_one_mod2())
        if vb is None: continue
        Lam4 = NB.neighbour(vb); M16 = to_amb16(transport(S, Lam4)); b = setup_amb(M16, to_amb16(Lam4)); n += 1
        hit = next((idx for idx in byinv.get(b['inv'], []) if same_orbit_amb(rep_setup(idx), b)), None)
        if hit is None: counts['unidentified'] += 1; unid.append({'inv': list(b['inv']), 'M16': M16.tolist()}) if len(unid) < 20 else None
        else: counts[hit] += 1
    out.append({'class': r['id'], 'side': SIDE, 'n': n, 'counts': {str(k): v for k, v in counts.items()}, 'unidentified_examples': unid})
    print(f"class {r['id']}: {n} neighbours, {len(counts)} distinct targets, unidentified {counts['unidentified']}  [{time.time()-t0:.0f}s]", flush=True)
    if time.time() - t0 > budget: break
json.dump(out, open(f'opsample_{SIDE}_{j}.json', 'w')); print("done")
