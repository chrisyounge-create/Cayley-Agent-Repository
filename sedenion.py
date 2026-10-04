import numpy as np
from exc_alg import STRUCT, CONJ, E8
def omul(x, y): return np.einsum('i,j,ijk->k', x, y, STRUCT)
def oconj(x): return x @ CONJ
def smul(p, q):
    """Cayley–Dickson: (a,b)(c,d) = (ac - conj(d) b, d a + b conj(c)); sedenions as 16-vectors in the doubled Coxeter basis."""
    a, b = p[:8], p[8:]; c, d = q[:8], q[8:]
    return np.concatenate([omul(a, c) - omul(oconj(d), b), omul(d, a) + omul(b, oconj(c))])
G16 = np.zeros((16, 16), dtype=np.int64); G16[:8, :8] = E8; G16[8:, 8:] = E8
def N(x): return float(x @ G16 @ x) / 2            # octonion/sedenion norm: Gram = 2 * norm form
ONE = np.zeros(16); ONE[5] = 1
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    oct_ok = all(abs(N(np.concatenate([omul(x, y), np.zeros(8)])) - N(np.concatenate([x, np.zeros(8)])) * N(np.concatenate([y, np.zeros(8)]))) < 1e-9 for x, y in (rng.normal(size=(2, 8)) for _ in range(20)))
    print("octonions (Coxeter basis): norm multiplicative:", oct_ok, "| identity = basis vector 5:", bool(np.allclose(smul(ONE, rng.normal(size=16)), smul(ONE, rng.normal(size=16)) )) if False else "")
    X = rng.normal(size=(200, 16)); Y = rng.normal(size=(200, 16)); Z = rng.normal(size=(200, 16))
    def allc(f): return all(np.allclose(*f(x, y, z), atol=1e-8) for x, y, z in zip(X, Y, Z))
    print("sedenions:")
    print("  identity works              :", all(np.allclose(smul(ONE, x), x) and np.allclose(smul(x, ONE), x) for x in X[:20]))
    print("  commutative  xy = yx        :", allc(lambda x, y, z: (smul(x, y), smul(y, x))))
    print("  associative (xy)z = x(yz)   :", allc(lambda x, y, z: (smul(smul(x, y), z), smul(x, smul(y, z)))))
    print("  alternative (xx)y = x(xy)   :", allc(lambda x, y, z: (smul(smul(x, x), y), smul(x, smul(x, y)))))
    print("  flexible    (xy)x = x(yx)   :", allc(lambda x, y, z: (smul(smul(x, y), x), smul(x, smul(y, x)))))
    print("  power-assoc (xx)x = x(xx)   :", allc(lambda x, y, z: (smul(smul(x, x), x), smul(x, smul(x, x)))))
    r = np.array([N(smul(x, y)) / (N(x) * N(y)) for x, y in zip(X, Y)])
    print(f"  norm multiplicative         : False  — N(xy)/(N(x)N(y)) ranges over [{r.min():.3f}, {r.max():.3f}] on 200 random pairs (octonions: always 1)")
