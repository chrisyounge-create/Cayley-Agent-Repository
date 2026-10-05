"""orbit_split with one representative six-space saved per orbit class (needed for the level-5 operator)."""
import json, glob, sys, time, numpy as np
src = open('orbit_split.py').read().split('if __name__ == "__main__":')[0]
exec(src)
def split_reps(fp, files, maxn, budget):
    rows = [json.loads(l) for f in files for l in open(f) if l.strip()]
    grp = [r for r in rows if 'error' not in r and not r.get('big') and tuple(r['inv']) == fp]
    t0 = time.time(); classes = []; reps = []
    for r in grp[:maxn]:
        a = setup(r)
        for k, c in enumerate(classes):
            if same_orbit(a, c[0]): c.append(a); break
        else: classes.append([a]); reps.append({'S': r['S'], 'stab': r.get('stab'), 'lift_stab': r.get('lift_stab'), 'types': r.get('types'), 'inv': r['inv']})
        if time.time() - t0 > budget: break
    return [len(c) for c in classes], len(grp), reps
if __name__ == "__main__":
    files = [sys.argv[1]] if sys.argv[1].endswith('.jsonl') else glob.glob(sys.argv[1] + '/**/*.jsonl', recursive=True)
    for fp in json.loads(sys.argv[2]):
        sizes, tot, reps = split_reps(tuple(fp), files, int(sys.argv[3]), float(sys.argv[4]))
        print(f"fingerprint {tuple(fp)} ({tot} hits): {sum(sizes)} samples -> {len(sizes)} orbit(s), sizes {sizes}", flush=True)
        with open(f"splitreps_{'_'.join(map(str, fp))}.json", 'w') as fh: json.dump({'fingerprint': fp, 'hits': tot, 'examined': sum(sizes), 'orbit_sizes': sizes, 'reps': reps}, fh)
