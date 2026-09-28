import numpy as _np, os as _os
_orig = _np.load
def _load(f, *a, **k):
    if isinstance(f, str) and f.startswith('nbr_maps') and f.endswith('.npy'):
        z = _orig(f.replace('.npy', '_int.npz')); return z['maps'].astype(float) / float(z['scale'])
    if isinstance(f, str) and f == 'gamma_G2Z.npy':
        return _orig('gamma_G2Z_int.npz')['G'].astype(_np.int64)
    return _orig(f, *a, **k)
_np.load = _load
