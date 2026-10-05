"""Classify fixed-point-sampling candidates on the J_E side: fingerprint + exact stabiliser via pathB.analyse, with a per-candidate time limit."""
import json, glob, sys, time, signal, collections, numpy as np
sys.argv = ['x', '0', '1', '/tmp/none.json'] + sys.argv[1:]
import je_setup, pathB as P
job, njobs, budget, out = int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), sys.argv[7]
cands = [c for f in sorted(glob.glob('hunt/**/fixE_*.json', recursive=True)) for c in json.load(open(f))]
cands = sorted(cands, key=lambda c: -c['RM'])[job::njobs]
class TO(Exception): pass
def handler(s, f): raise TO()
signal.signal(signal.SIGALRM, handler)
t0 = time.time(); res = []; n_big = 0
for c in cands:
    if time.time() - t0 > budget: break
    S = np.array(c['S'], dtype=np.int64); signal.alarm(120)
    try: r = P.analyse(S); signal.alarm(0)
    except TO: res.append({'RM': c['RM'], 'cls': c['cls'], 'big': True, 'S': c['S']}); n_big += 1; continue
    except Exception as e: signal.alarm(0); continue
    if r and r.get('stab'): res.append({'RM': c['RM'], 'cls': c['cls'], 'inv': r['inv'], 'stab': r['stab'], 'types': r.get('types'), 'lift_stab': r.get('lift_stab'), 'S': c['S']})
json.dump(res, open(out, 'w')); print(f"job {job}: classified {len(res)-n_big}, big (timed out) {n_big}, in {time.time()-t0:.0f}s")
