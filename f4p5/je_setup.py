"""Switch the F4 pipeline (f4_p5, pathB) to J_E coordinates. Basis of J_E: rows of LamE_lll / 4 (in J coordinates)."""
import numpy as np, pickle, sympy
from fractions import Fraction as Fr
import f4_p5 as F, pathB as P
import albert
L4 = np.load('LamE_lll.npy').astype(np.int64)                     # = 4 * basis of J_E
L4inv = sympy.Matrix(L4.tolist()).inv()
L4adj = np.array((L4inv * sympy.Matrix(L4.tolist()).det()).tolist(), dtype=object); D4 = int(sympy.Matrix(L4.tolist()).det())
GRAM_E = (L4 @ albert.GRAM @ L4.T) // 16
_sharpJ = albert.sharp
_den = 1
for _v in L4inv: _den = int(np.lcm(_den, sympy.Rational(_v).q))
W_int = np.array((L4inv * _den).tolist(), dtype=np.int64)               # L4^{-1} * den, small integers
def sharp_E(X):
    """sharp in J_E coordinates, vectorised: element = x_E L4/4; sharp(element) in J_E coords = sharp(x_E L4) L4^{-1} / 4."""
    X = np.asarray(X, dtype=np.int64); single = X.ndim == 1; X2 = np.atleast_2d(X)
    S = _sharpJ(X2 @ L4) @ W_int                                           # = 4*den * (J_E coords of sharp)
    q = 4 * _den
    assert not (S % q).any(), "sharp not integral in J_E coordinates"
    res = S // q
    return res[0] if single else res
Ivec = np.array(F.I27, dtype=np.int64).reshape(-1)[:27] if np.ndim(F.I27) > 1 else np.array(F.I27, dtype=np.int64)
I_E = np.array([int(v) for v in (sympy.Matrix([list(map(int, Ivec))]) * 4 * L4inv)], dtype=np.int64)   # identity in J_E coords
TRV = GRAM_E @ I_E                                                        # trace functional: tr(x) = T(x, I)
def trace_E(X): return np.asarray(X, dtype=np.int64) @ TRV
def det_E(x):                                                             # N(x) = T(x, x#)/3
    x = np.asarray(x, dtype=np.int64); return int(x @ GRAM_E @ sharp_E(x)) // 3
gens = pickle.load(open('f4_autE_gens.pkl', 'rb'))
B_E = sympy.Matrix(L4.tolist()) / 4; B_Ei = B_E.inv()
GENS_E = [np.array((B_E * sympy.Matrix([[sympy.Rational(str(x)) for x in row] for row in np.array(A)]) * B_Ei).tolist(), dtype=np.int64) for A in gens]
# patch the pipeline
for mod in (F, P):
    mod.GRAM = GRAM_E
F.sharp = sharp_E; F.trace = trace_E; F.albdet = det_E; F.I27 = I_E; P.I27 = I_E
