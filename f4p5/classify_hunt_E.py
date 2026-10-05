"""Classify J_E hunt candidates: cheap fingerprint (vector counts of M = G0 + 5 J_E), skip known, analyse new representatives exactly."""
import je_setup, json, glob, sys, time, collections, numpy as np, pathB as P, f4_p5 as F
from pathB import pari, gp_mat
j, njobs, budget = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]); t0 = time.time()
known = {tuple(x) for x in json.load(open('known_invs_E.json'))}
files = sorted(glob.glob('../results/37216164333/**/fixE_*.json', recursive=True)); cands = []
for f in files: cands += json.load(open(f))
cands.sort(key=lambda x: (-x['RM'], str(x['S'])[:40])); cands = cands[j::njobs]
print(f"job {j}: {len(cands)} candidates; known fingerprints {len(known)}")
def cheap_fp(S):
    G0 = P.G0_of(S); B = np.vstack([G0, 5 * np.eye(27, dtype=np.int64)])
    H = pari.mathnf(pari.mattranspose(gp_mat(B))); Hm = np.array([[int(H[i, k]) for k in range(27)] for i in range(27)], dtype=object)
    M = Hm.T                                                        # rows: basis of M in J_E coordinates
    Gram = np.array(M.dot(F.GRAM.astype(object)).dot(M.T), dtype=object)
    return tuple(int(x) for x in pari.qfrep(gp_mat(Gram), 4, 1))
seen = collections.Counter(); reps = {}; out = {}
for c in cands:
    if time.time() - t0 > budget * 0.6: break
    S = np.array(c['S'], dtype=np.int64)
    try: fp = cheap_fp(S)
    except Exception: continue
    seen[fp] += 1
    if fp not in known and fp not in reps: reps[fp] = (S, c['RM'])
print(f"fingerprinted {sum(seen.values())} candidates ({time.time()-t0:.0f}s): {len(seen)} fingerprints, {len(reps)} NEW")
for fp, (S, rm) in reps.items():
    if time.time() - t0 > budget: break
    try: r = P.analyse(S)
    except Exception as e: out[str(fp)] = dict(error=str(e)[:80], RM=rm); continue
    out[str(fp)] = dict(stab=r.get('stab'), big=r.get('big'), types=r.get('types'), lift_stab=r.get('lift_stab'), inv=r.get('inv'), RM=rm, S=S.tolist(), seen=seen[fp])
    print(f"  NEW {fp}: stab {r.get('stab')} big {r.get('big')} types {r.get('types')} |R_M| {rm} seen {seen[fp]}  [{time.time()-t0:.0f}s]")
json.dump(out, open(f'huntclass_{j}.json', 'w')); print("done", len(out))
