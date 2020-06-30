# required packages
import numpy as np
# definitions for shared tools

def nan_gaussian_filter(mat, sigma, keep_nan=False):
    from scipy.ndimage import gaussian_filter
    U = np.array(mat)
    Unan = np.isnan(U)
    V = U.copy()
    V[U != U] = 0
    VV = gaussian_filter(V, sigma=sigma)

    W = 0*U.copy()+1
    W[U != U] = 0
    WW = gaussian_filter(W, sigma=sigma)

    Z = VV/WW
    if keep_nan:
        Z[Unan] = np.nan
    return Z

def interp1dnan(A):
    A_ = np.array(A)
    ok = np.isnan(A) == False
    xp = ok.nonzero()[0]
    fp = A[ok]
    x = np.isnan(A).nonzero()[0]
    A_[np.isnan(A)] = np.interp(x, xp, fp)
    return A_

def interpolate_chr(_chr, gaussian=0):
    """linear interpolate chromosome coordinates"""
    _chr = np.array(_chr).copy()
    for i in range(_chr.shape[-1]):
        if gaussian > 0:
            _chr[:, i] = nan_gaussian_filter(_chr[:, i], gaussian)
    # interpolate
    from scipy.interpolate import interp1d
    not_nan_inds = np.where(np.isnan(_chr).sum(1) == 0)[0]
    if len(not_nan_inds) == 0:
        return _chr
    else:
        f = interp1d(np.arange(len(_chr))[not_nan_inds], _chr[not_nan_inds],
                     kind='linear', axis=0, bounds_error=False,
                     fill_value='extrapolate')
        _interpolated_chr = f(np.arange(len(_chr)))
        return _interpolated_chr

def extract_sequences(zxy, domain_starts):
    """Function to extract sequences of zxy coordinates given domain start indices"""
    _dm_starts = np.array(domain_starts, dtype=np.int)
    _dm_ends = np.array(list(domain_starts[1:])+[len(zxy)], dtype=np.int)
    _zxy = np.array(zxy)
    _seqs = []
    for _start, _end in zip(_dm_starts, _dm_ends):
        _seqs.append(_zxy[_start:_end])
    return _seqs

## sub packages
# domain calling functions
from . import calling
# domain distances
from . import distance
# manual tools for domain selection
from . import manual
# domain interactions
from . import interaction