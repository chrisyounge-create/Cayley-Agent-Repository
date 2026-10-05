"""Full representative database for the level-5 operator. usage: build_reps_db.py SIDE  (writes reps_db_SIDE.json)"""
import json, glob, sys, numpy as np, collections, re
side = sys.argv[1]
if side == 'E': import je_setup
import f4_p5 as F, pathB as P
from f4_nbrs_head import hnf_rows
L4 = np.load('LamE_lll.npy').astype(np.int64)
key = lambda r: (tuple(r['inv']), r['stab'], tuple(sorted(r['lift_stab'] or [])), tuple(sorted(r['types'] or [])))
def B16_of(S):
    G0 = P.G0_of(np.array(S, dtype=np.int64)); B = hnf_rows([list(map(int, g)) for g in G0] + [list(map(int, 5*e)) for e in F.E])
    return (16 * B).tolist() if side == 'Z' else (4 * (B @ L4)).tolist()
out = []
if side == 'Z':
    recs = [json.loads(l) for f in glob.glob('../results/36658053553/**/*.jsonl', recursive=True) for l in open(f) if l.strip()]
    good = [r for r in recs if r.get('stab')]; firsts = {}
    for r in good:
        if key(r) not in firsts: firsts[key(r)] = r
    doubled = {tuple(json.load(open(f))['fingerprint']): json.load(open(f))['reps'] for f in glob.glob('../results/37270653395/**/splitreps_*.json', recursive=True)}
    for k, v in json.load(open('jz_doubled_reps_local.json')).items(): doubled[tuple(int(x) for x in k.strip('()').split(','))] = v
    for k, r in firsts.items():
        if k[0] in doubled: continue
        out.append({'side': 'Z', 'inv': list(k[0]), 'stab': k[1], 'S': r['S'], 'origin': 'record'})
    for fp, reps in doubled.items():
        for r in reps: out.append({'side': 'Z', 'inv': list(fp), 'stab': r.get('stab'), 'S': r['S'], 'origin': 'doubled split'})
    big = json.load(open('jz_big_reps.json'))
    seen_big = {}
    for S in big['big']:
        G0 = P.G0_of(np.array(S, dtype=np.int64)); B = hnf_rows([list(map(int, g)) for g in G0] + [list(map(int, 5*e)) for e in F.E]); G = B @ F.GRAM @ B.T
        inv = tuple(int(x) for x in P.pari.qfrep(P.gp_mat(G), 4, 1))
        if inv not in seen_big: seen_big[inv] = S; out.append({'side': 'Z', 'inv': list(inv), 'stab': None, 'S': S, 'origin': 'big (stab via GAP pending)'})
    S = big['n90'][0]; G0 = P.G0_of(np.array(S, dtype=np.int64)); B = hnf_rows([list(map(int, g)) for g in G0] + [list(map(int, 5*e)) for e in F.E]); G = B @ F.GRAM @ B.T
    out.append({'side': 'Z', 'inv': [int(x) for x in P.pari.qfrep(P.gp_mat(G), 4, 1)], 'stab': 110592, 'S': S, 'origin': 'rare (fixed-point)'})
else:
    recs = [json.loads(l) for f in glob.glob('../results/3709*/**/*.jsonl', recursive=True) + glob.glob('../results/3717*/**/*.jsonl', recursive=True) for l in open(f) if l.strip()]
    good = [r for r in recs if r.get('stab')]; firsts = {}
    for r in good:
        if key(r) not in firsts: firsts[key(r)] = r
    e2 = {}; e1 = {}
    for f in glob.glob('../results/37268678406/**/splitreps_*.json', recursive=True):
        d = json.load(open(f)); fp = tuple(d['fingerprint']); e2.setdefault(fp[:4], []).extend(d['reps'])
    for f in glob.glob('../results/37268676996/**/splitreps_*.json', recursive=True):
        d = json.load(open(f)); e1[tuple(d['fingerprint'])] = d['reps']
    split_invs = set(e2) | set(e1)
    for k, r in firsts.items():
        if k[0] in split_invs: continue
        out.append({'side': 'E', 'inv': list(k[0]), 'stab': k[1], 'S': r['S'], 'origin': 'record'})
    for inv in split_invs:
        reps = e2[inv] if inv in e2 else e1[inv]
        # a shared inv split only by run 1 (by inv) lists all its orbits; keys of run 2 are per (inv, stab): include both runs' reps when run 2 covered only some stabs
        if inv in e2 and inv in e1:
            stabs2 = {r.get('stab') for r in e2[inv]}; reps = e2[inv] + [r for r in e1[inv] if r.get('stab') not in stabs2]
        for r in reps: out.append({'side': 'E', 'inv': list(inv), 'stab': r.get('stab'), 'S': r['S'], 'origin': 'split'})
        have = {r.get('stab') for r in reps}
        for k, r in firsts.items():                     # record keys of this inv whose stabiliser no split rep carries
            if k[0] == inv and k[1] not in have: out.append({'side': 'E', 'inv': list(inv), 'stab': k[1], 'S': r['S'], 'origin': 'record (shared inv)'})
for i, r in enumerate(out): r['B16'] = B16_of(r['S'])
json.dump(out, open(f'reps_db_{side}.json', 'w')); print(side, "representatives:", len(out), dict(collections.Counter(r['origin'] for r in out)))
