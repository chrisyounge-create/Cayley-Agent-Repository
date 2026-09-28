"""Integral Albert algebra J_Z = Her3(Coxeter order) in integer coordinates (c1,c2,c3 | x1,x2,x3), x_i in Z^8 (Coxeter basis).
sharp(T) = (c2c3 - N(x1), c3c1 - N(x2), c1c2 - N(x3) | conj(x3)conj(x2) - c1 x1, conj(x1)conj(x3) - c2 x2, conj(x2)conj(x1) - c3 x3)."""
import numpy as np
from exc_alg import E8, STRUCT, CONJ
from coxeter_aut import ONE
S_ = STRUCT.astype(np.int64)
def omul(x, y):  return np.einsum('...a,...b,abc->...c', x, y, S_)
def oconj(x):    return x @ CONJ if np.array_equal(ONE @ CONJ, ONE) else CONJ @ x
def onorm(x):    return np.einsum('...i,ij,...j->...', x, E8, x) // 2
def opair(x, y): return np.einsum('...i,ij,...j->...', x, E8, y)
def sharp(T):
    """T: (..., 27) integer array -> sharp(T) (..., 27)"""
    c = T[..., :3]; x1, x2, x3 = T[..., 3:11], T[..., 11:19], T[..., 19:27]
    b1 = c[..., 1]*c[..., 2] - onorm(x1); b2 = c[..., 2]*c[..., 0] - onorm(x2); b3 = c[..., 0]*c[..., 1] - onorm(x3)
    y1 = omul(oconj(x3), oconj(x2)) - c[..., 0:1]*x1; y2 = omul(oconj(x1), oconj(x3)) - c[..., 1:2]*x2; y3 = omul(oconj(x2), oconj(x1)) - c[..., 2:3]*x3
    return np.concatenate([np.stack([b1, b2, b3], axis=-1), y1, y2, y3], axis=-1)
def trace(T): return T[..., 0] + T[..., 1] + T[..., 2]
def pair(A, B): return A[..., 0]*B[..., 0] + A[..., 1]*B[..., 1] + A[..., 2]*B[..., 2] + opair(A[..., 3:11], B[..., 3:11]) + opair(A[..., 11:19], B[..., 11:19]) + opair(A[..., 19:27], B[..., 19:27])
def det(T):
    c = T[..., :3]; x1, x2, x3 = T[..., 3:11], T[..., 11:19], T[..., 19:27]
    tr_xyz = opair(omul(x1, x2), oconj(x3))          # Tr((x1 x2) x3) = <x1 x2, conj(x3)>
    return c[..., 0]*c[..., 1]*c[..., 2] + tr_xyz - c[..., 0]*onorm(x1) - c[..., 1]*onorm(x2) - c[..., 2]*onorm(x3)
GRAM = np.zeros((27, 27), dtype=np.int64); GRAM[:3, :3] = np.eye(3, dtype=np.int64)
for k in range(3): GRAM[3+8*k:11+8*k, 3+8*k:11+8*k] = E8
I27 = np.zeros(27, dtype=np.int64); I27[:3] = 1
if __name__ == "__main__":
    E1 = np.zeros(27, dtype=np.int64); E1[0] = 1
    print("sanity: sharp(I) = I?", np.array_equal(sharp(I27), I27), "; sharp(E1) = 0?", not sharp(E1).any(), "; det(I) =", det(I27), "; conj(ONE)=ONE:", np.array_equal(oconj(ONE), ONE))
    # rank-1 trace-2 element [1,1,0; 0,0,z], N(z)=1
    z = ONE.copy(); T = np.zeros(27, dtype=np.int64); T[0] = T[1] = 1; T[19:27] = z
    print("T=[1,1,0;0,0,1]: sharp =", sharp(T).tolist(), "det =", det(T), "trace =", trace(T))
    # count rank-1 elements mod 2 (2^27 elements, chunked)
    import time; t0 = time.time(); n1 = 0; n1tr0 = 0
    idx = np.arange(1 << 20, dtype=np.int64)
    for hi in range(1 << 7):
        v = ((idx[:, None] >> np.arange(27)) & 1).astype(np.int64); v[:, 20:27] = (hi >> np.arange(7)) & 1
        sh = sharp(v) % 2; r1 = ~(sh.any(axis=1)) & v.any(axis=1)
        n1 += int(r1.sum()); n1tr0 += int((r1 & (trace(v) % 2 == 0)).sum())
    print(f"rank-1 elements of J/2J: {n1} (E6/P1 count (2^12-1)(2^9-1)/((2^4-1)(2-1)) = {4095*511//15});  trace-0 among them: {n1tr0} (F4/Q count (2^4+1)(2^12-1)/(2-1) = {17*4095})  [{time.time()-t0:.0f}s]")
