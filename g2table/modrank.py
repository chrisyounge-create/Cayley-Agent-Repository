import numpy as np
def rank_mod(A, p):
    """rank of an integer matrix over F_p by Gaussian elimination (vectorised row updates)."""
    A = np.array(A, dtype=np.int64) % p; m, n = A.shape; r = 0
    for c in range(n):
        if r == m: break
        col = A[r:, c]; nz = np.nonzero(col)[0]
        if len(nz) == 0: continue
        piv = r + nz[0]
        if piv != r: A[[r, piv]] = A[[piv, r]]
        inv = pow(int(A[r, c]), -1, p); A[r] = (A[r] * inv) % p
        rows = np.nonzero(A[r+1:, c])[0] + r + 1
        if len(rows): A[rows] = (A[rows] - np.outer(A[rows, c], A[r])) % p
        r += 1
    return r
