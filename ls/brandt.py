import cypari2, math
from fractions import Fraction as Fr
from sympy import Matrix
pari = cypari2.Pari(); pari.allocatemem(2*10**9)
pari('ssroots(P, p) = my(w = ffgen(p^2, \'w), F = factor(P * w^0), R = List()); for(i = 1, #F~, if(poldegree(F[i,1]) == 1, for(m = 1, F[i,2], listput(R, -polcoef(F[i,1], 0) / polcoef(F[i,1], 1))))); Vec(R)')
SS = pari('ssroots')
def brandt(p, ells=(2, 3)):
    D = next(d for d in (3, 4, 7, 8, 11, 19, 43, 67, 163) if int(pari.kronecker(-d, p)) == -1)
    start = SS(pari.polclass(-D), p)[0]
    J = [start]; key = {str(start): 0}; frontier = [start]
    Phi2 = pari("polmodular(2)")
    while frontier:
        nxt = []
        for j in frontier:
            for r in SS(pari.subst(Phi2, 'y', j), p):
                if str(r) not in key: key[str(r)] = len(J); J.append(r); nxt.append(r)
        frontier = nxt
    n = len(J); w = []
    for j in J:
        s = str(j); w.append(3 if s in ('0',) and p > 3 else (2 if s == str(pari(f"1728 * ffgen({p}^2,'w)^0")) and p > 3 else 1))
    T = {}
    for ell in ells:
        Phi = pari(f"polmodular({ell})"); A = [[0]*n for _ in range(n)]
        for i, j in enumerate(J):
            for r in SS(pari.subst(Phi, 'y', j), p): A[i][key[str(r)]] += 1
        T[ell] = A
    return J, key, w, T
def e_vector(D, p, key, n):
    v = [0]*n
    for r in SS(pari.polclass(-D), p): v[key[str(r)]] += 1
    return v
if __name__ == "__main__":
    for p in (11, 19, 37):
        J, key, w, T = brandt(p)
        A = T[2]; n = len(J)
        sym = all(A[i][k]*w[k] == A[k][i]*w[i] for i in range(n) for k in range(n))
        ev = Matrix(A).eigenvals()
        print(f"N={p}: class number {n} (expect (N-1)/12 + ... ), weights {w}, A_ik w_k = A_ki w_i: {sym}, T2 eigenvalues {ev}")
