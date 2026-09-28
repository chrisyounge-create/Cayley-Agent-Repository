"""
Exact arithmetic for Coxeter's integral octonions and J = H_3(O), mirroring
the conventions of Pollack's g2_motives.sage (so results can be checked
against his published numbers).

Octonions: Cayley-Dickson pairs (x, y) of Hamilton quaternions with gamma = -1:
    (x1,y1)(x2,y2) = (x1 x2 + gamma * conj(y2) y1 ,  y2 x1 + y1 conj(x2))
Quaternions over Q(t), t^2 = -1, are 4-tuples of Gaussian rationals.

Gaussian rationals are pairs (re, im) of Fractions.
"""

from fractions import Fraction as Fr
import numpy as np

# ------------------------------------------------------------ Gaussian rationals
class G:
    __slots__ = ("re", "im")
    def __init__(self, re=0, im=0):
        self.re = Fr(re); self.im = Fr(im)
    def __add__(s, o): o = G.c(o); return G(s.re + o.re, s.im + o.im)
    __radd__ = __add__
    def __sub__(s, o): o = G.c(o); return G(s.re - o.re, s.im - o.im)
    def __rsub__(s, o): o = G.c(o); return G(o.re - s.re, o.im - s.im)
    def __mul__(s, o): o = G.c(o); return G(s.re*o.re - s.im*o.im, s.re*o.im + s.im*o.re)
    __rmul__ = __mul__
    def __neg__(s): return G(-s.re, -s.im)
    def __truediv__(s, o):
        o = G.c(o); d = o.re*o.re + o.im*o.im
        return G((s.re*o.re + s.im*o.im)/d, (s.im*o.re - s.re*o.im)/d)
    def __eq__(s, o): o = G.c(o); return s.re == o.re and s.im == o.im
    def __repr__(s): return f"({s.re}+{s.im}t)"
    @staticmethod
    def c(o): return o if isinstance(o, G) else G(o, 0)

T = G(0, 1)   # sqrt(-1) of the coefficient field K

# ------------------------------------------------------------ quaternions (-1,-1)
class Q:
    """a + b i + c j + d k with i^2=j^2=k^2=-1, ij=k."""
    __slots__ = ("v",)
    def __init__(self, a=0, b=0, c=0, d=0):
        self.v = [G.c(a), G.c(b), G.c(c), G.c(d)]
    def __add__(s, o): return Q(*[x + y for x, y in zip(s.v, o.v)])
    def __sub__(s, o): return Q(*[x - y for x, y in zip(s.v, o.v)])
    def __neg__(s): return Q(*[-x for x in s.v])
    def scale(s, l): return Q(*[l * x for x in s.v])
    def __mul__(s, o):
        a1, b1, c1, d1 = s.v; a2, b2, c2, d2 = o.v
        return Q(a1*a2 - b1*b2 - c1*c2 - d1*d2,
                 a1*b2 + b1*a2 + c1*d2 - d1*c2,
                 a1*c2 - b1*d2 + c1*a2 + d1*b2,
                 a1*d2 + b1*c2 - c1*b2 + d1*a2)
    def conj(s): a, b, c, d = s.v; return Q(a, -b, -c, -d)
    def trace(s): return s.v[0] * 2
    def norm(s): return sum((x * x for x in s.v), G(0))

ONE, I_, J_, K_ = Q(1), Q(0, 1), Q(0, 0, 1), Q(0, 0, 0, 1)
ZERO = Q()

# ------------------------------------------------------------ octonions, gamma=-1
def omul(o1, o2):
    x1, y1 = o1; x2, y2 = o2
    return (x1 * x2 - (y2.conj() * y1), y2 * x1 + y1 * x2.conj())
def oconj(o): return (o[0].conj(), -o[1])
def otrace(o): return o[0].trace()
def onorm(o): return o[0].norm() + o[1].norm()
def oadd(o1, o2): return (o1[0] + o2[0], o1[1] + o2[1])
def osub(o1, o2): return (o1[0] - o2[0], o1[1] - o2[1])
def oscale(o, l): return (o[0].scale(l), o[1].scale(l))
def opair(o1, o2): return otrace(omul(o1, oconj(o2)))          # (x,y) = n(x+y)-n(x)-n(y)
def otri(o1, o2, o3): return otrace(omul(omul(o1, o2), o3))    # (x,y,z) = tr((xy)z)

# ------------------------------------------------------------ Coxeter basis
e = (ZERO, ONE)
h = (Q(0, Fr(1,2), Fr(1,2), Fr(1,2)), Q(Fr(1,2)))
COX = [
    omul((J_, ZERO), h),
    e,
    (-h[0], -h[1]),
    (J_, ZERO),
    omul((I_, ZERO), h),
    (ONE, ZERO),
    omul(e, h),
    omul((K_, ZERO), e),
]

def gram(basis):
    return [[opair(basis[j], basis[k]) for k in range(8)] for j in range(8)]

def inv_matrix(M):
    """Exact inverse over Gaussian rationals (Gauss-Jordan)."""
    n = len(M)
    A = [[G.c(M[r][c]) for c in range(n)] + [G(1 if r == c else 0) for c in range(n)] for r in range(n)]
    for col in range(n):
        piv = next(r for r in range(col, n) if not (A[r][col] == 0))
        A[col], A[piv] = A[piv], A[col]
        p = A[col][col]
        A[col] = [x / p for x in A[col]]
        for r in range(n):
            if r != col and not (A[r][col] == 0):
                f = A[r][col]
                A[r] = [x - f * y for x, y in zip(A[r], A[col])]
    return [row[n:] for row in A]

GRAM = gram(COX)
GRAM_INV = inv_matrix(GRAM)

def oct_to_vec(o, basis=COX):
    ip = [opair(o, basis[j]) for j in range(8)]
    return [sum((GRAM_INV[j][k] * ip[k] for k in range(8)), G(0)) for j in range(8)]

def vec_to_oct(v, basis=COX):
    out = (ZERO, ZERO)
    for j in range(8):
        out = oadd(out, oscale(basis[j], G.c(v[j])))
    return out

# integer data: E8 gram, trilinear tensor, structure constants of the Coxeter ring
E8 = np.array([[int(GRAM[j][k].re) for k in range(8)] for j in range(8)], dtype=np.int64)
TRI = np.array([[[int(otri(COX[a], COX[b], COX[c]).re) for c in range(8)]
                 for b in range(8)] for a in range(8)], dtype=np.int64)
# STRUCT[a, b, :] = coordinates of e_a * e_b in the Coxeter basis (must be integers)
STRUCT = np.zeros((8, 8, 8), dtype=np.int64)
for a in range(8):
    for b in range(8):
        v = oct_to_vec(omul(COX[a], COX[b]))
        for c in range(8):
            assert v[c].im == 0 and v[c].re.denominator == 1
            STRUCT[a, b, c] = int(v[c].re)
# conjugation matrix: CONJ[a, :] = coordinates of conj(e_a)
CONJ = np.zeros((8, 8), dtype=np.int64)
for a in range(8):
    v = oct_to_vec(oconj(COX[a]))
    for c in range(8):
        assert v[c].im == 0 and v[c].re.denominator == 1
        CONJ[a, c] = int(v[c].re)

# ------------------------------------------------------------ Pollack's singular pair
half = Fr(1, 2)
r1 = (ZERO, Q(half, -half * T if False else G(0, -half), 0, 0))   # 1/2 (0, 1 - t i)
r1 = (ZERO, Q(G(half), G(0, -half)))
r2 = (ZERO, Q(G(half), G(0, -half)))
r3 = (Q(0, G(0, -1)), ZERO)                                       # -t (i, 0)
s1 = (ZERO, Q(0, G(0, -1)))                                       # -t (0, i)
s2 = (ZERO, Q(G(half), G(0, half)))                               # 1/2 (0, 1 + t i)
s3 = (Q(G(-half), G(0, -half)), ZERO)                             # -1/2 (1 + t i, 0)
XOCT = [G(1), G(-1), G(0), r1, r2, r3]
YOCT = [G(0), G(-1), G(1), s1, s2, s3]
XX = [XOCT[0], XOCT[1], XOCT[2], oct_to_vec(r1), oct_to_vec(r2), oct_to_vec(r3)]
YY = [YOCT[0], YOCT[1], YOCT[2], oct_to_vec(s1), oct_to_vec(s2), oct_to_vec(s3)]

if __name__ == "__main__":
    print("E8 Gram matrix:\n", E8)
    print("Coxeter ring closed under multiplication: structure constants integral (asserted)")
    print("Xoct vector coordinates:")
    for v in XX[3:]:
        print("  ", v)
    print("Yoct vector coordinates:")
    for v in YY[3:]:
        print("  ", v)
    # denominators
    den = 1
    for v in XX[3:] + YY[3:]:
        for g in v:
            den = max(den, g.re.denominator, g.im.denominator)
    print("max denominator in X,Y coordinates:", den)


# ------------------------------------------------------------ Jordan algebra over K
def jsharp(Tj):
    c1, c2, c3, x1, x2, x3 = Tj
    b1 = c1 * 0 + c2 * c3 - onorm(x1)
    b2 = c3 * c1 - onorm(x2)
    b3 = c1 * c2 - onorm(x3)
    y1 = osub(omul(oconj(x3), oconj(x2)), oscale(x1, c1))
    y2 = osub(omul(oconj(x1), oconj(x3)), oscale(x2, c2))
    y3 = osub(omul(oconj(x2), oconj(x1)), oscale(x3, c3))
    return [b1, b2, b3, y1, y2, y3]

def jadd(A, B):
    return [A[k] + B[k] for k in range(3)] + [oadd(A[k], B[k]) for k in range(3, 6)]

def jsub(A, B):
    return [A[k] - B[k] for k in range(3)] + [osub(A[k], B[k]) for k in range(3, 6)]

def jscale(A, l):
    return [G.c(l) * A[k] for k in range(3)] + [oscale(A[k], G.c(l)) for k in range(3, 6)]

def jcross(A, B):
    return jsub(jsub(jsharp(jadd(A, B)), jsharp(A)), jsharp(B))

def jpair(A, B):
    return A[0]*B[0] + A[1]*B[1] + A[2]*B[2] + opair(A[3], B[3]) + opair(A[4], B[4]) + opair(A[5], B[5])

def act_Phi(c, b, z):
    V = jcross(c, jcross(b, z))
    p = jpair(c, z)
    q = jpair(b, c)
    return jadd(jsub(jscale(b, p), V), jscale(z, q))

def act_wedge_Phi(c, b, z):
    return jsub(act_Phi(c, b, z), act_Phi(b, c, z))

def g_action2(u, v, T0):
    T1 = act_wedge_Phi(u, v, T0)
    T2 = act_wedge_Phi(u, v, T1)
    return jadd(jadd(T0, T1), jscale(T2, Fr(1, 2)))

OZ = (ZERO, ZERO)
def diag(c1, c2, c3):
    return [G.c(c1), G.c(c2), G.c(c3), OZ, OZ, OZ]

# split basis of O tensor K, ordered as Pollack: [ep1, e1, e2, e3, e1*, e2*, e3*, ep2]
def _qK(a=0, b=0, c=0, d=0):
    return Q(G.c(a), G.c(b), G.c(c), G.c(d))
hf = Fr(1, 2)
e2s   = (ZERO, Q(G(hf), G(0, -hf)))
e3st  = (ZERO, Q(0, 0, G(hf), G(0, -hf)))
e3s   = (ZERO, Q(0, 0, G(-hf), G(0, -hf)))
e2st  = (ZERO, Q(G(-hf), G(0, -hf)))
ep1   = (Q(G(hf), G(0, -hf)), ZERO)
ep2   = (Q(G(hf), G(0, hf)), ZERO)
e1s   = (Q(0, 0, G(hf), G(0, -hf)), ZERO)
e1st  = (Q(0, 0, G(-hf), G(0, -hf)), ZERO)
SPLIT = [ep1, e1s, e2s, e3s, e1st, e2st, e3st, ep2]

def Vnum(i, j, num):
    o = oscale(SPLIT[j], G.c(num))
    out = diag(0, 0, 0)
    out[2 + i] = o
    return out

def new_singular_pair(list_1_2, list_1_3, list_2_1):
    """Pollack's list_to_oct_pair: move (Xoct, Yoct) by unipotent elements of F4^c(K)."""
    uv = []
    e11, e22 = diag(1, 0, 0), diag(0, 1, 0)
    for j, r in enumerate(list_1_2): uv.append((e11, Vnum(2, j, r)))
    for j, r in enumerate(list_1_3): uv.append((e11, Vnum(3, j, r)))
    for j, r in enumerate(list_2_1): uv.append((e22, Vnum(1, j, r)))
    X, Y = XOCT, YOCT
    for (u, v) in uv:
        X = g_action2(u, v, X)
        Y = g_action2(u, v, Y)
    Xv = [X[0], X[1], X[2], oct_to_vec(X[3]), oct_to_vec(X[4]), oct_to_vec(X[5])]
    Yv = [Y[0], Y[1], Y[2], oct_to_vec(Y[3]), oct_to_vec(Y[4]), oct_to_vec(Y[5])]
    return Xv, Yv
