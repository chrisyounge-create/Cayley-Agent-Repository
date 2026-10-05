"""Exact orbit identification in ambient coordinates for the level-5 operator."""
import numpy as np, sympy, json
import pathB as P, importlib.util
_spec = importlib.util.spec_from_file_location('albert_pristine', 'albert.py'); albert_p = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(albert_p)
E27 = np.eye(27, dtype=np.int64); I27 = albert_p.I27.astype(np.int64)
GRAMZ = albert_p.GRAM.astype(np.int64); SH = albert_p.sharp
from fractions import Fraction
from math import lcm
def jordan_ok_scaled(As, s):
    """A = As/s (ambient map, As integer). Checks I*A = I and sharp(xA) = sharp(x)A for x = e_i, e_i+e_j."""
    if not np.array_equal(I27 @ As, s * I27): return False
    for i in range(27):
        if not np.array_equal(SH(E27[i] @ As), s * (SH(E27[i]) @ As)): return False
    for i in range(27):
        for j in range(i + 1, 27):
            x = E27[i] + E27[j]
            if not np.array_equal(SH(x @ As), s * (SH(x) @ As)): return False
    return True
def setup_amb(B16, J16=None):
    """B16 = 16 * basis rows of an ambient lattice (integer)."""
    B4 = np.array(B16, dtype=np.int64); G = (B4 @ GRAMZ @ B4.T) // 256
    T = P.pari.qflllgram(P.gp_mat(G)); U = np.array([[int(T[i, j]) for j in range(27)] for i in range(27)], dtype=np.int64).T   # rows transform
    B4 = U @ B4; G = (B4 @ GRAMZ @ B4.T) // 256
    res = P.pari.qfauto(P.gp_mat(G)); OM = [np.array(g, dtype=np.int64).T for g in P.closure([P.np_mat(g) for g in res[1]])]
    Bs = sympy.Matrix(B4.tolist()); D = int(Bs.det())
    return dict(B4=B4, G=G, OM=OM, Badj=np.array((Bs.inv() * D).tolist(), dtype=object), D=D, inv=tuple(int(x) for x in P.pari.qfrep(P.gp_mat(G), 4, 1)), J16=(None if J16 is None else np.array(J16, dtype=np.int64)))
def same_orbit_amb(a, b):
    f = P.pari.qfisom(P.gp_mat(a['G']), P.gp_mat(b['G']))
    if f == 0: return False
    Fm = np.array([[int(f[i, j]) for j in range(27)] for i in range(27)], dtype=np.int64)
    Y = Fm.T if np.array_equal(Fm.T @ b['G'] @ Fm, a['G']) else Fm
    for X in a['OM']:
        AD = a['Badj'].dot((X @ Y).astype(object)).dot(b['B4'].astype(object))        # A = AD / D exactly
        fr = [Fraction(int(z), a['D']) for z in AD.flatten()]
        d = 1
        for q in fr: d = lcm(d, q.denominator)
        if d > 64: continue
        As = np.array([int(q * d) for q in fr], dtype=np.int64).reshape(27, 27)
        if not jordan_ok_scaled(As, d): continue
        if a['J16'] is None or b['J16'] is None: return True
        # the pair condition: A must map the ambient Albert lattice of a onto that of b (J_a A ⊂ J_b, same determinant => equality)
        JA = a['J16'].astype(object).dot(As.astype(object))                       # 16*basis of J_a, mapped, times d
        Jb = sympy.Matrix(b['J16'].tolist()); sol = sympy.Matrix(JA.tolist()) * Jb.inv()        # coefficients of J_a A in the basis of J_b, times d
        if all((z / d).q == 1 for z in sol): return True
    return False
