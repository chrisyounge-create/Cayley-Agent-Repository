import json, glob, sys, time, numpy as np, sympy, pathB as P, f4_p5 as F
from f4_nbrs_head import hnf_rows
E = np.eye(27, dtype=np.int64); I27 = P.I27
def is_jordan_exact(A):
    if not np.array_equal(I27 @ A, I27): return False
    for i in range(27):
        if not np.array_equal(F.sharp(E[i] @ A), F.sharp(E[i]) @ A): return False
    for i in range(27):
        for j in range(i+1, 27):
            x = E[i] + E[j]
            if not np.array_equal(F.sharp(x @ A), F.sharp(x) @ A): return False
    return True
def setup(r):
    S = np.array(r['S'], dtype=np.int64); G0 = P.G0_of(S)
    B = hnf_rows([list(map(int, g)) for g in G0] + [list(map(int, 5*e)) for e in F.E]); G = B @ F.GRAM @ B.T
    res = P.pari.qfauto(P.gp_mat(G)); OM = [np.array(g, dtype=np.int64).T for g in P.closure([P.np_mat(g) for g in res[1]])]
    Bs = sympy.Matrix(B.tolist()); D = int(Bs.det())
    return dict(B=B, G=G, OM=OM, Badj=np.array((Bs.inv() * D).tolist(), dtype=object), D=D)
def same_orbit(a, b):
    f = P.pari.qfisom(P.gp_mat(a['G']), P.gp_mat(b['G']))
    if f == 0: return False
    Fm = np.array([[int(f[i, j]) for j in range(27)] for i in range(27)], dtype=np.int64)
    Y = Fm.T if np.array_equal(Fm.T @ b['G'] @ Fm, a['G']) else Fm
    for X in a['OM']:
        A5 = a['Badj'].dot((X @ Y).astype(object)).dot(b['B'].astype(object))
        if any(int(z) % a['D'] for z in A5.flatten()): continue
        if is_jordan_exact((A5 // a['D']).astype(np.int64)): return True
    return False
def split(fp, files, maxn, budget):
    rows = [json.loads(l) for f in files for l in open(f) if l.strip()]
    grp = [r for r in rows if 'error' not in r and not r.get('big') and tuple(r['inv']) == fp]
    t0 = time.time(); classes = []
    for r in grp[:maxn]:
        a = setup(r)
        for c in classes:
            if same_orbit(a, c[0]): c.append(a); break
        else: classes.append([a])
        if time.time() - t0 > budget: break
    return [len(c) for c in classes], len(grp)
if __name__ == "__main__":
    files = glob.glob(sys.argv[1] + '/**/*.jsonl', recursive=True)
    for fp in json.loads(sys.argv[2]):
        sizes, tot = split(tuple(fp), files, int(sys.argv[3]), float(sys.argv[4]))
        print(f"fingerprint {tuple(fp)} ({tot} hits): {sum(sizes)} samples -> {len(sizes)} orbit(s), sizes {sizes}", flush=True)
