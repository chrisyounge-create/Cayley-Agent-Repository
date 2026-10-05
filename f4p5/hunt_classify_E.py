"""Classify the J_E fixed-point-sampling candidates: fingerprint each, analyse one representative per NEW fingerprint (not in the records),
report stabilisers; candidates sorted by |R_M| descending."""
import je_setup, json, glob, collections, sys, time, numpy as np, pathB as P
recs = json.load(open('/home/claude/pathBE_runs12_records.json'))
known = {tuple(r['inv']) for r in recs}
cands = []
for f in glob.glob('../results/37216164333/**/fixE_*.json', recursive=True):
    cands += json.load(open(f))
cands.sort(key=lambda x: -x['RM']); print("candidates:", len(cands), "| known fingerprints:", len(known))
budget = float(sys.argv[1]) if len(sys.argv) > 1 else 240; t0 = time.time(); seen = {}; new = {}
for c in cands:
    if time.time() - t0 > budget: break
    S = np.array(c['S'], dtype=np.int64)
    try: inv = tuple(P.invariants(P.G0_of(S)[1]) if isinstance(P.G0_of(S), tuple) else P.invariants(P.G0_of(S)))
    except Exception:
        try: r = P.analyse(S); inv = tuple(r['inv'])
        except Exception as e: continue
    if inv in known or inv in seen: seen[inv] = seen.get(inv, 0) + 1; continue
    try: r = P.analyse(S)
    except Exception as e: continue
    seen[inv] = 1; new[inv] = (r.get('stab'), r.get('big'), r.get('types'), c['RM'])
    print(f"  NEW fingerprint {inv}: stab {r.get('stab')}, big {r.get('big')}, types {r.get('types')}, |R_M| {c['RM']}  [{time.time()-t0:.0f}s]")
print(f"processed {sum(seen.values())} candidates in {time.time()-t0:.0f}s; new fingerprints: {len(new)}")
json.dump({str(k): v for k, v in new.items()}, open('hunt_new_E.json', 'w'))
