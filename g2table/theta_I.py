"""
Fourier coefficients of the exceptional theta lift Theta_I(X, Y; m) on split
G2 (Pollack, arXiv:2211.05280 Thm 4.1; arXiv:2401.02922 Thm 4.1):

    a(f) = sum_{w in W_J(Z), rank 1, pr_I(w) = f} sigma_4(d_w) * P_{m,I}(w; X, Y),
    P_{m,I}(w; X, Y) = ( (b,X)_I (c,Y) - (b,Y)_I (c,X) )^m,   w = (a, b, c, d).

Rank-one w with first coordinate a != 0 are exactly w = a (1, T, T#, N(T)),
T = S/a, so with S = aT in J_R:

    w = (a, S, S#/a, N(S)/a^2),
    pr_I(w) = (a, (S,I), (S#,I)/a, N(S)/a^2) = (a, b', c', d),
    (S,S)_I = b'^2 - 2 a c'        (identity  (S,S)_I = (S,I)^2 - 2 (S#,I)),
    integrality: S# in a J_R^vee,  a^2 | N(S),
    P = val(S)^m / a^m,   val(S) = (S,X)_I (S#,Y) - (S,Y)_I (S#,X).

For a = 1 this is Pollack's Lemma 4.3 / his G2_FC_dict_I, which is used as the
check.  J_R^vee is identified with J_R through ( , )_I (E8 is unimodular).

Coordinates: S = [u1,u2,u3 ; v1,v2,v3], u_i in Z, v_i in E8 (Coxeter basis).
X, Y coordinates have denominator 2; we scale them by 2 and divide val^m by
4^m at the end.  All sums are exact (Gaussian integers, Python ints across
chunks, int64 inside chunks).
"""

import math
import itertools
import numpy as np
from fractions import Fraction as Fr
from exc_alg import E8, TRI, STRUCT, CONJ, XX, YY

# ------------------------------------------------------------- E8 short vectors

def e8_vectors_by_norm(nmax):
    """dict k -> int64 array of all v in Z^8 with (1/2) v E8 v^T = k, k <= nmax."""
    # Fincke-Pohst on the Gram matrix E8 (Q(v) = v E8 v^T / 2)
    A = E8.astype(float)
    L = np.linalg.cholesky(A)          # A = L L^T
    Linv = np.linalg.inv(L)
    # Q(v) = |L^T v|^2 / 2 ; bound |L^T v|^2 <= 2 nmax
    R = 2 * nmax + 1e-9
    # enumerate using the upper-triangular factor U = L^T:  Q = sum_i (sum_{j>=i} U_ij v_j)^2
    U = L.T
    out = []
    v = np.zeros(8, dtype=np.int64)
    def rec(i, partial):
        # partial = sum over rows > i of (U v)_r^2
        if i < 0:
            out.append(v.copy()); return
        # (U v)_i = U_ii v_i + sum_{j>i} U_ij v_j
        s = sum(U[i, j] * v[j] for j in range(i + 1, 8))
        rem = R - partial
        if rem < 0:
            return
        r = math.sqrt(rem)
        lo = math.ceil((-r - s) / U[i, i] - 1e-9)
        hi = math.floor((r - s) / U[i, i] + 1e-9)
        for vi in range(lo, hi + 1):
            v[i] = vi
            t = U[i, i] * vi + s
            rec(i - 1, partial + t * t)
        v[i] = 0
    rec(7, 0.0)
    V = np.array(out, dtype=np.int64)
    norms = np.einsum("ij,jk,ik->i", V, E8, V) // 2
    assert np.all(np.einsum("ij,jk,ik->i", V, E8, V) % 2 == 0)
    return {k: V[norms == k] for k in range(nmax + 1)}


# ---------------------------------------------------------------- scaled X, Y

def _split(vecs):
    """Gaussian rational vectors -> (re, im) int64 arrays after scaling by 2."""
    re = np.array([[int(2 * g.re) for g in v] for v in vecs], dtype=np.int64)
    im = np.array([[int(2 * g.im) for g in v] for v in vecs], dtype=np.int64)
    return re, im

XR3, XI3 = _split(XX[3:])       # shape (3, 8): coordinates of 2*x_3, 2*x_4, 2*x_5
YR3, YI3 = _split(YY[3:])
X0 = np.array([[int(2 * g.re), int(2 * g.im)] for g in XX[:3]], dtype=np.int64)  # 2*x_0..2
Y0 = np.array([[int(2 * g.re), int(2 * g.im)] for g in YY[:3]], dtype=np.int64)


def gauss_mul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


def gauss_pow(ar, ai, m):
    rr, ri = np.ones_like(ar), np.zeros_like(ar)
    for _ in range(m):
        rr, ri = gauss_mul(rr, ri, ar, ai)
    return rr, ri


# ------------------------------------------------------------ the main routine

def theta_I_coefficient(a, bp, cp, d, m, E8V=None, verbose=False):
    """Fourier coefficient of Theta_I(X, Y; m) at the binary cubic
    a u^3 + bp u^2 v + cp u v^2 + d v^3  (weight 4 + m), returned as a
    Gaussian rational (Fraction pair) and the number of w summed."""
    N = bp * bp - 2 * a * cp
    if N < 0:
        return (Fr(0), Fr(0)), 0
    if E8V is None:
        E8V = e8_vectors_by_norm(N // 2)
    tot_r, tot_i, count = 0, 0, 0

    # candidate (n1,n2,n3) and integer triples u with sum u = bp, sum u^2 = N - 2 sum n
    for n1 in range(N // 2 + 1):
        for n2 in range((N - 2 * n1) // 2 + 1):
            for n3 in range((N - 2 * n1 - 2 * n2) // 2 + 1):
                q = N - 2 * (n1 + n2 + n3)
                us = []
                r = int(math.isqrt(q))
                for u1 in range(-r, r + 1):
                    for u2 in range(-r, r + 1):
                        u3 = bp - u1 - u2
                        if u1 * u1 + u2 * u2 + u3 * u3 == q:
                            us.append((u1, u2, u3))
                if not us:
                    continue
                V1, V2, V3 = E8V[n1], E8V[n2], E8V[n3]
                if verbose:
                    print(f"  (n1,n2,n3)=({n1},{n2},{n3})  u-triples={len(us)}  "
                          f"E8 triples={len(V1)*len(V2)*len(V3):,}")
                # precompute pairings of V3 with the x_5, y_5 and TRI contractions
                V3E = V3 @ E8                                   # (K3, 8)
                p3x = (V3E * XR3[2]).sum(1), (V3E * XI3[2]).sum(1)   # (v3, 2x5)
                p3y = (V3E * YR3[2]).sum(1), (V3E * YI3[2]).sum(1)
                for v1 in V1:
                    v1E = v1 @ E8
                    p1x = (v1E * XR3[0]).sum(), (v1E * XI3[0]).sum()          # (v1, 2x3)
                    p1y = (v1E * YR3[0]).sum(), (v1E * YI3[0]).sum()
                    # trilinear pieces with v1 fixed: (2x4, v3, v1) and (2x5, v1, v2)
                    T_x4_v3_v1 = V3 @ np.einsum("abc,a,c->b", TRI, XR3[1], v1), V3 @ np.einsum("abc,a,c->b", TRI, XI3[1], v1)
                    T_y4_v3_v1 = V3 @ np.einsum("abc,a,c->b", TRI, YR3[1], v1), V3 @ np.einsum("abc,a,c->b", TRI, YI3[1], v1)
                    M_x5_v1 = np.einsum("abc,a,b->c", TRI, XR3[2], v1), np.einsum("abc,a,b->c", TRI, XI3[2], v1)  # dot with v2
                    M_y5_v1 = np.einsum("abc,a,b->c", TRI, YR3[2], v1), np.einsum("abc,a,b->c", TRI, YI3[2], v1)
                    M_v1 = np.einsum("abc,a->bc", TRI, v1)      # (v1, v2, v3) = v2 M_v1 v3
                    for v2 in V2:
                        v2E = v2 @ E8
                        p2x = (v2E * XR3[1]).sum(), (v2E * XI3[1]).sum()      # (v2, 2x4)
                        p2y = (v2E * YR3[1]).sum(), (v2E * YI3[1]).sum()
                        tri123 = V3 @ (v2 @ M_v1)                               # (v1,v2,v3) for each v3
                        # (2x3, v2, v3) and (2y3, v2, v3)
                        T_x3_v2_v3 = V3 @ np.einsum("abc,a,b->c", TRI, XR3[0], v2), V3 @ np.einsum("abc,a,b->c", TRI, XI3[0], v2)
                        T_y3_v2_v3 = V3 @ np.einsum("abc,a,b->c", TRI, YR3[0], v2), V3 @ np.einsum("abc,a,b->c", TRI, YI3[0], v2)
                        c_x5_v1_v2 = int(M_x5_v1[0] @ v2), int(M_x5_v1[1] @ v2)
                        c_y5_v1_v2 = int(M_y5_v1[0] @ v2), int(M_y5_v1[1] @ v2)
                        A3x = (T_x3_v2_v3[0] + T_x4_v3_v1[0] + c_x5_v1_v2[0],
                               T_x3_v2_v3[1] + T_x4_v3_v1[1] + c_x5_v1_v2[1])
                        A3y = (T_y3_v2_v3[0] + T_y4_v3_v1[0] + c_y5_v1_v2[0],
                               T_y3_v2_v3[1] + T_y4_v3_v1[1] + c_y5_v1_v2[1])
                        for (u1, u2, u3) in us:
                            # constraints on S = [u; v]: (S#, I) = a cp, N(S) = a^2 d
                            csharp = u2 * u3 + u3 * u1 + u1 * u2 - (n1 + n2 + n3)
                            if csharp != a * cp:
                                continue
                            det = u1 * u2 * u3 - u1 * n1 - u2 * n2 - u3 * n3 + tri123
                            ok = det == a * a * d
                            if a != 1:
                                # S# must lie in a J_R: diagonal parts and octonion parts
                                b1, b2, b3 = u2 * u3 - n1, u3 * u1 - n2, u1 * u2 - n3
                                if b1 % a or b2 % a or b3 % a:
                                    continue
                                # y1 = conj(v3) conj(v2) - u1 v1 ; y2 = conj(v1) conj(v3) - u2 v2 ;
                                # y3 = conj(v2) conj(v1) - u3 v3   (all as E8 coordinate vectors)
                                cv1, cv2 = v1 @ CONJ, v2 @ CONJ
                                CV3 = V3 @ CONJ
                                y1 = np.einsum("ka,abc,b->kc", CV3, STRUCT, cv2) - u1 * v1[None, :]
                                y2 = np.einsum("a,abc,kb->kc", cv1, STRUCT, CV3) - u2 * v2[None, :]
                                y3 = (cv2 @ np.einsum("abc,a->bc", STRUCT, cv1))[None, :] - u3 * V3
                                ok = ok & ~((y1 % a).any(1)) & ~((y2 % a).any(1)) & ~((y3 % a).any(1))
                            if not np.any(ok):
                                continue
                            idx = np.nonzero(ok)[0]
                            # (S, 2X)_I and (S#, 2X), same for Y
                            Tx_r = u1 * X0[0, 0] + u2 * X0[1, 0] + u3 * X0[2, 0] + p1x[0] + p2x[0] + p3x[0][idx]
                            Tx_i = u1 * X0[0, 1] + u2 * X0[1, 1] + u3 * X0[2, 1] + p1x[1] + p2x[1] + p3x[1][idx]
                            Ty_r = u1 * Y0[0, 0] + u2 * Y0[1, 0] + u3 * Y0[2, 0] + p1y[0] + p2y[0] + p3y[0][idx]
                            Ty_i = u1 * Y0[0, 1] + u2 * Y0[1, 1] + u3 * Y0[2, 1] + p1y[1] + p2y[1] + p3y[1][idx]
                            A1x_r = (u2*u3 - n1) * X0[0, 0] + (u3*u1 - n2) * X0[1, 0] + (u1*u2 - n3) * X0[2, 0]
                            A1x_i = (u2*u3 - n1) * X0[0, 1] + (u3*u1 - n2) * X0[1, 1] + (u1*u2 - n3) * X0[2, 1]
                            A1y_r = (u2*u3 - n1) * Y0[0, 0] + (u3*u1 - n2) * Y0[1, 0] + (u1*u2 - n3) * Y0[2, 0]
                            A1y_i = (u2*u3 - n1) * Y0[0, 1] + (u3*u1 - n2) * Y0[1, 1] + (u1*u2 - n3) * Y0[2, 1]
                            A2x_r = u1 * p1x[0] + u2 * p2x[0] + u3 * p3x[0][idx]
                            A2x_i = u1 * p1x[1] + u2 * p2x[1] + u3 * p3x[1][idx]
                            A2y_r = u1 * p1y[0] + u2 * p2y[0] + u3 * p3y[0][idx]
                            A2y_i = u1 * p1y[1] + u2 * p2y[1] + u3 * p3y[1][idx]
                            Sx_r = A1x_r - A2x_r + A3x[0][idx]
                            Sx_i = A1x_i - A2x_i + A3x[1][idx]
                            Sy_r = A1y_r - A2y_r + A3y[0][idx]
                            Sy_i = A1y_i - A2y_i + A3y[1][idx]
                            v_r, v_i = gauss_mul(Tx_r, Tx_i, Sy_r, Sy_i)
                            w_r, w_i = gauss_mul(Ty_r, Ty_i, Sx_r, Sx_i)
                            val_r, val_i = v_r - w_r, v_i - w_i          # = 4 * val(S)
                            pr, pi = gauss_pow(val_r, val_i, m)
                            # sigma_4(d_w): d_w = gcd of (a, S, S#/a, d)
                            if a == 1:
                                tot_r += int(pr.sum()); tot_i += int(pi.sum()); count += len(idx)
                            else:
                                for jj, k in enumerate(idx):
                                    g = math.gcd(a, u1, u2, u3, d)
                                    g = math.gcd(g, *[int(t) for t in v1], *[int(t) for t in v2], *[int(t) for t in V3[k]])
                                    if g > 1:
                                        g = math.gcd(g, (u2*u3 - n1)//a, (u3*u1 - n2)//a, (u1*u2 - n3)//a)
                                        g = math.gcd(g, *[int(t) for t in y1[k]//a], *[int(t) for t in y2[k]//a], *[int(t) for t in y3[k]//a])
                                    s4 = sum(dd ** 4 for dd in range(1, g + 1) if g % dd == 0)
                                    tot_r += s4 * int(pr[jj]); tot_i += s4 * int(pi[jj]); count += 1
    scale = Fr(1, 4 ** m * a ** m)
    return (Fr(tot_r) * scale, Fr(tot_i) * scale), count


if __name__ == "__main__":
    import time
    E8V = e8_vectors_by_norm(3)
    print({k: len(v) for k, v in E8V.items()})
    # Pollack's check values (weight 6, m = 2), his notebook: (0,-1,0) -> -336, (1,-1,0) -> 720
    t0 = time.time()
    for (b, c, d) in [(0, -1, 0), (1, -1, 0)]:
        val, cnt = theta_I_coefficient(1, b, c, d, 2, E8V)
        print(f"a(u^3 + {b}u^2v + {c}uv^2 + {d}v^3) = {val}   from {cnt} rank-one w   [{time.time()-t0:.1f}s]")
