import numpy as np, pickle, sys
exec(open('zd_periods_level7.py').read().split("for wname, (R_, Rder, dimV) in weights.items():")[0])
weights = {'7': (R7, R7der, 7), '14': (ad, ad_der, 14), '27': (rho, lambda X: Qp @ der_mat(X) @ Q, 27)}
spectra = {}
# trivial weight at the maximal parahoric: Brandt matrices
one_rep = lambda g: np.eye(1)
bases = [np.eye(1) for _ in stabs]; off = np.cumsum([0] + [1]*len(stabs))
T2t = hecke(2, one_rep, 1, bases, off); T3t = hecke(3, one_rep, 1, bases, off)
spectra['1'] = (np.linalg.eigvals(T2t).real, np.linalg.eigvals(T3t).real)
for wname, (R_, Rder, dimV) in weights.items():
    bases = []
    for st in stabs:
        Pm = sum(R_(Gf[k].astype(float)) for k in st) / len(st); u, s, _ = np.linalg.svd(Pm); bases.append(u[:, s > 0.5])
    off = np.cumsum([0] + [b.shape[1] for b in bases])
    T2 = hecke(2, R_, dimV, bases, off); T3 = hecke(3, R_, dimV, bases, off)
    spectra[wname] = (np.linalg.eigvals(T2).real, np.linalg.eigvals(T3).real)
# Iwahori level 7, trivial weight, from Part I data
d = pickle.load(open('/home/claude/g2_level7.pkl', 'rb')); B = d['B']; spectra['Iwahori, 1'] = (np.linalg.eigvals(np.array(B[2], float)).real, np.linalg.eigvals(np.array(B[3], float)).real)
print(f"{'family (level 7)':22s} {'#forms':>6s} | {'mean T2':>8s} {'mean T2^2':>9s} | {'mean T3':>8s} {'mean T3^2':>9s}")
print(f"{'Plancherel prediction':22s} {'inf':>6s} | {0:8.1f} {126:9.1f} | {0:8.1f} {1092:9.1f}")
print(f"{'Sato-Tate prediction':22s} {'inf':>6s} | {-1:8.1f} {65:9.1f} | {-1:8.1f} {730:9.1f}")
for k, (e2, e3) in spectra.items():
    # drop the Eisenstein eigenvalue (126 / 1092) where present
    m = ~((np.abs(e2 - 126) < 1e-6)); e2c, e3c = e2[m], e3[~(np.abs(e3 - 1092) < 1e-6)]
    print(f"{('weight ' + k) if 'Iwahori' not in k else k:22s} {len(e2c):6d} | {e2c.mean():8.2f} {np.mean(e2c**2):9.1f} | {e3c.mean():8.2f} {np.mean(e3c**2):9.1f}")
pooled2 = np.concatenate([spectra[k][0][~(np.abs(spectra[k][0] - 126) < 1e-6)] for k in ('1', '7', '14', '27')]); pooled3 = np.concatenate([spectra[k][1][~(np.abs(spectra[k][1] - 1092) < 1e-6)] for k in ('1', '7', '14', '27')])
print(f"{'pooled weights 1-27':22s} {len(pooled2):6d} | {pooled2.mean():8.2f} {np.mean(pooled2**2):9.1f} | {pooled3.mean():8.2f} {np.mean(pooled3**2):9.1f}")
# distribution shape: histogram of chi7 = (T2+1)/8 for the weight-27 family vs Sato-Tate moments of chi7 (mean 0, var 1) and Plancherel (mean 1/8, var ~ 126/64)
c7 = (spectra['27'][0] + 1) / 8; print(f"\nweight 27: chi7(s_2) mean {c7.mean():.3f} (Plancherel 1/8 = 0.125, Sato-Tate 0), variance {c7.var():.3f} (Plancherel 126/64 = 1.97, Sato-Tate 1.0)")
np.save('/home/claude/level7_spectra.npy', np.array([spectra['27'][0], spectra['27'][1]]))
