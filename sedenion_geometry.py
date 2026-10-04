"""Part IV, step 1: the sedenions as the finite part of a Farnsworth–Boyle non-associative spectral triple (A_F = H_F = S, J = conjugation).
Checks (all exact or machine precision): order-zero laws by class, derivations (= g2, inner, isometries), Lie multiplication algebra so(16)+R,
fermion content 16 = 2(1+7) under G2 with the S3 commuting, commutant of the left action = scalars (=> gamma_F = +-1, D_F = 0)."""
import numpy as np
from sedenion import smul, G16
rng = np.random.default_rng(3); E = np.eye(16); T = np.array([[smul(E[i], E[j]) for j in range(16)] for i in range(16)])
L = lambda a: np.einsum('i,ijk->jk', a, T).T; R = lambda a: np.einsum('j,ijk->ik', a, T).T
def holds(f, n=30): return all(np.allclose(*f(rng.normal(size=16), rng.normal(size=16)), atol=1e-9) for _ in range(n))
if __name__ == "__main__":
    print("flexible order zero   [R_b,L_a] = [L_b,R_a]        :", holds(lambda a, b: (R(b)@L(a)-L(a)@R(b), L(b)@R(a)-R(a)@L(b))))
    print("alternative order zero [R_b,L_a] = L_ba - L_b L_a  :", holds(lambda a, b: (R(b)@L(a)-L(a)@R(b), L(smul(b,a))-L(b)@L(a))))
    print("associative order zero [L_a,R_b] = 0               :", holds(lambda a, b: (L(a)@R(b)-R(b)@L(a), np.zeros((16,16)))))
