from brandt import *
import math, sys, flint, json, subprocess
p = int(sys.argv[1])
J, key, w, T = brandt(p, ells=(2,)); n = len(J)
A2 = flint.fmpz_mat(T[2]); x = flint.fmpz_poly([0, 1]); cusp = A2.charpoly() // (x - 3)
D0 = next(D for D in range(3, 10**6) if int(pari.isfundamental(-D)) == 1 and int(pari.kronecker(-D, p)) == -1)
e = e_vector(D0, p, key, n)
best = None
for f, m in cusp.factor()[1]:
    P = pari(str(f).replace('x', 'y'))
    M = pari.matrix(n, n, [pari(f"Mod({T[2][j][i]} - (({i})=={j})*y, {P})") if i == j else T[2][j][i] for i in range(n) for j in range(n)])
    K = pari.matker(M); v = [pari.lift(K[i, 0]) for i in range(n)]
    c = sum(e[i]*w[i]*v[i] for i in range(n))
    if pari(f"Mod({c}, {P})") == 0: continue                    # odd-sign orbit (c = 0)
    if best is None or f.degree() > best[0]: best = (f.degree(), P, v, c)
d, P, v, c = best
Fd = pari.factor(pari.poldisc(P), 10**5)
extra = []
for q_, ex_ in zip(Fd[0], Fd[1]):
    if int(q_) > 10**5:
        G = pari.factor(q_)
        extra += [G[0][i] for i in range(len(G[0]))]
small = [Fd[0][i] for i in range(len(Fd[0])) if int(Fd[0][i]) <= 10**5]
nf = pari.nfinit(pari([P, pari(small + extra)]))                  # maximal at EVERY prime dividing disc
I = None
for cc in v:
    if cc != 0: I = pari.idealhnf(nf, cc) if I is None else pari.idealadd(nf, I, cc)
NI = pari.idealnorm(nf, I); h = sum(w[i]*v[i]*v[i] for i in range(n))
Nh = abs(pari.norm(pari(f"Mod({h}, {P})")))
lg = lambda q: math.log(int(pari.numerator(q))) - math.log(int(pari.denominator(q)))
cost = (lg(Nh) - 2*lg(abs(NI)))/d
unc = pari.nfcertify(nf); fdisc = pari.factor(pari.poldisc(P), 10**5)
big = [int(q) for q in fdisc[0] if int(q) > 10**5]
print(json.dumps({"N": p, "d": d, "cost": cost, "kappa": cost - math.log(p), "uncertified_primes": [str(q)[:30] for q in unc], "large_disc_cofactors_digits": [len(str(q)) for q in big]}))
