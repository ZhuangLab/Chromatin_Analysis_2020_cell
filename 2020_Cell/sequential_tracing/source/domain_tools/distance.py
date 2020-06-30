import numpy as np
import matplotlib.pylab as plt
from scipy.stats import normaltest, ks_2samp, ttest_ind
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree, is_valid_linkage

def _local_distances(_zxy, dom_sz=5):
    """Calculate local distances to determine local_domain_calling"""
    _dists = []
    for i in range(len(_zxy)):
        if i >= int(dom_sz/2) and i < len(_zxy)-int(dom_sz/2):
            cm1 = np.nanmean(_zxy[max(i-dom_sz, 0):i], axis=0)
            cm2 = np.nanmean(_zxy[i:min(i+dom_sz, len(_zxy))], axis=0)
            dist = np.linalg.norm(cm1-cm2)
            _dists.append(dist)
    return _dists

# Calculate distance given distance_matrix, window_size and metric type
def _sliding_window_dist(_mat, _wd, _dist_metric='median'):
    """Function to calculate sliding-window distance across one distance-map of chromosome"""
    dists = np.zeros(len(_mat))
    for _i in range(len(_mat)):
        if _i - int(_wd/2) < 0 or _i + int(_wd/2) >= len(_mat):
            dists[_i] = 0
        else:
            # get slices
            _left_slice = slice(max(0, _i-_wd), _i)
            _right_slice = slice(_i, min(_i+_wd, len(_mat)))
            # slice matrix
            _intra1 = np.triu(_mat[_left_slice,_left_slice], 1)
            _intra1 = _intra1[np.isnan(_intra1)==False]
            _intra2 = np.triu(_mat[_right_slice,_right_slice], 1)
            _intra2 = _intra2[np.isnan(_intra2)==False]
            _intra_dist = np.concatenate([_intra1[_intra1 > 0],
                                          _intra2[_intra2 > 0]])
            _inter_dist = _mat[_left_slice,_right_slice]
            _inter_dist = _inter_dist[np.isnan(_inter_dist) == False]
            if len(_intra_dist) == 0 or len(_inter_dist) == 0:
                # return zero distance if one dist list is empty
                dists[_i] = 0
                continue
            # add dist info
            if _dist_metric == 'ks':
                if 'ks_2samp' not in locals():
                    from scipy.stats import ks_2samp
                _f = np.sign((np.median(_inter_dist) - np.median(_intra_dist)) )
                dists[_i] = _f * ks_2samp(_intra_dist, _inter_dist)[0]
            elif _dist_metric == 'median':
                m_inter, m_intra = np.median(_inter_dist), np.median(_intra_dist)
                v_inter, v_intra = np.median((_inter_dist-m_inter)**2),\
                                   np.median((_intra_dist-m_intra)**2)
                dists[_i] = (m_inter-m_intra) / np.sqrt(v_inter+v_intra)
            elif _dist_metric == 'mean':
                m_inter, m_intra = np.mean(_inter_dist), np.mean(_intra_dist)
                v_inter, v_intra = np.var(_inter_dist), np.var(_intra_dist)
                dists[_i] = (m_inter-m_intra) / np.sqrt(v_inter+v_intra)
            elif _dist_metric == 'normed_insulation':
                dists[_i] = (np.nanmean(_intra_dist) - np.nanmean(_inter_dist)) / (np.nanmean(_intra_dist) + np.nanmean(_inter_dist))
            elif _dist_metric == 'insulation':
                m_inter, m_intra = np.mean(_inter_dist), np.mean(_intra_dist)
                dists[_i] = m_inter / m_intra
            else:
                raise ValueError(f"Wrong input _dist_metric")

    #dists[dists<0] = 0

    return dists

def domain_distance(coordinates, _dom1_bds, _dom2_bds,
                    _metric='median', _normalization_mat=None,
                    _allow_minus_distance=True):
    """Function to measure domain distances between two zxy arrays
    use KS-statistic as a distance:
        citation: https://arxiv.org/abs/1711.00761"""
    ## check inputs
    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)
    _metric = str(_metric).lower()
    _dom1_bds = [int(_b) for _b in _dom1_bds]
    _dom2_bds = [int(_b) for _b in _dom2_bds]

    # based on coordinates given, get intra/inter distances
    if len(np.shape(coordinates)) != 2:
        raise ValueError(f"Wrong input shape for coordinates, should be 2d but {len(np.shape(coordinates))} is given")
    elif np.shape(coordinates)[0] == np.shape(coordinates)[1]:
        _mat = coordinates
        _intra1 = _mat[_dom1_bds[0]:_dom1_bds[1], _dom1_bds[0]:_dom1_bds[1]]
        _intra2 = _mat[_dom2_bds[0]:_dom2_bds[1], _dom2_bds[0]:_dom2_bds[1]]
        _intra1 = _intra1[np.triu_indices(len(_intra1),1)]
        _intra2 = _intra2[np.triu_indices(len(_intra2), 1)]

        _intra_dist = [_intra1, _intra2]
        _inter_dist = np.ravel(_mat[_dom1_bds[0]:_dom1_bds[1], _dom2_bds[0]:_dom2_bds[1]])

    elif np.shape(coordinates)[1] == 3:
        # extract sequence
        zxy1 = coordinates[_dom1_bds[0]:_dom1_bds[1]]
        zxy2 = coordinates[_dom2_bds[0]:_dom2_bds[1]]
        # get distances
        _intra_dist = [pdist(zxy1), pdist(zxy2)]
        _inter_dist = np.ravel(cdist(zxy1, zxy2))
    else:
        raise ValueError(f"Input coordinates should be distance-matrix or 3d-coordinates!")

    # normalization
    if _normalization_mat is not None:
        # check other inputs
        if _dom1_bds is None or _dom2_bds is None:
            raise TypeError(
                f"Domain boundaries not fully given while normalization specified, skip normalization!")
        # normalize!
        else:
            _intra_dist[0] = _intra_dist[0] / squareform(
                _normalization_mat[_dom1_bds[0]:_dom1_bds[1], _dom1_bds[0]:_dom1_bds[1]])
            _intra_dist[1] = _intra_dist[1] / squareform(
                _normalization_mat[_dom2_bds[0]:_dom2_bds[1], _dom2_bds[0]:_dom2_bds[1]])
            _intra_dist = np.concatenate(_intra_dist)
            _inter_dist = _inter_dist / \
                np.ravel(
                    _normalization_mat[_dom1_bds[0]:_dom1_bds[1], _dom2_bds[0]:_dom2_bds[1]])
    else:
        # not normalize? directly concatenate
        _intra_dist = np.concatenate(_intra_dist)

    _kept_inter = _inter_dist[np.isnan(_inter_dist) == False]
    _kept_intra = _intra_dist[np.isnan(_intra_dist) == False]

    if len(_kept_inter) == 0 or len(_kept_intra) == 0:
        return 0

    if _metric == 'median':
        m_inter, m_intra = np.nanmedian(_inter_dist), np.nanmedian(_intra_dist)
        v_inter, v_intra = np.nanmedian(
            (_inter_dist-m_inter)**2), np.nanmedian((_intra_dist-m_intra)**2)
        _final_dist = (m_inter-m_intra)/np.sqrt(v_inter+v_intra)
    elif _metric == 'mean':
        m_inter, m_intra = np.nanmean(_inter_dist), np.nanmean(_intra_dist)
        v_inter, v_intra = np.nanvar(_inter_dist), np.nanvar(_intra_dist)
        _final_dist = (m_inter-m_intra)/np.sqrt(v_inter+v_intra)
    elif _metric == 'ks':
        if 'ks_2samp' not in locals():
            from scipy.stats import ks_2samp
        _f = np.sign(np.nanmedian(_inter_dist) - np.nanmedian(_intra_dist))
        _dist, _pval =  ks_2samp(_kept_inter, _kept_intra)
        _final_dist =  _f * _dist
    elif _metric == 'insulation':
        m_inter, m_intra = np.nanmedian(_inter_dist), np.nanmedian(_intra_dist)
        _final_dist = m_inter / m_intra
    elif _metric == 'absolute_median':
        m_inter, m_intra = np.nanmedian(_inter_dist), np.nanmedian(_intra_dist)
        _final_dist = (m_inter-m_intra)
    else:
        raise ValueError(f"Wrong input _metric type:{_metric}")
    
    if _allow_minus_distance:
        return _final_dist
    else:
        return max(_final_dist, 0)
    
# function to call domain pairwise distance as scipy.spatial.distance.pdist
def domain_pdists(coordinates, domain_starts, metric='median', 
                  normalization_mat=None, allow_minus_dist=False):
    """Calculate domain pair-wise distances, return a vector as the same order as
    scipy.spatial.distance.pdist 
    Inputs:
        coordinates:
        """
    ## check inputs
    coordinates = np.array(coordinates).copy()
    if len(np.shape(coordinates)) != 2:
        raise ValueError(
            f"Wrong input shape for coordinates, should be 2d but {len(np.shape(coordinates))} is given")
    elif np.shape(coordinates)[0] == np.shape(coordinates)[1]:
        _mat = coordinates
    elif np.shape(coordinates)[1] == 3:
        _mat = squareform(pdist(coordinates))
    else:
        raise ValueError(
            f"Input coordinates should be distance-matrix or 3d-coordinates!")

    domain_starts = np.array(domain_starts, dtype=np.int)
    domain_ends = np.zeros(np.shape(domain_starts))
    domain_ends[:-1] = domain_starts[1:]
    domain_ends[-1] = len(coordinates)

    # first check whether do normalzation
    if normalization_mat is not None:
        if normalization_mat.shape[0] != _mat.shape[0] or normalization_mat.shape[1] != _mat.shape[1]:
            raise ValueError(
                f"Wrong shape of normalization:{normalization_mat.shape}, should be equal to {_mat.shape}")

    dom_pdists = []
    for _i in range(len(domain_starts)):
        for _j in range(_i+1, len(domain_starts)):
            dom_pdists.append(domain_distance(coordinates,
                                              [domain_starts[_i], domain_ends[_i]],
                                              [domain_starts[_j], domain_ends[_j]],
                                              _metric=metric,
                                              _normalization_mat=normalization_mat,
                                              _allow_minus_distance=allow_minus_dist))

    dom_pdists = np.array(dom_pdists)

    return dom_pdists

# pdist in correlation format
def domain_correlation_pdists(coordinates, domain_starts):
    """Function to calculate domain correlation pair-wise distances"""
    _coordinates = np.array(coordinates)
    if len(_coordinates.shape) != 2:
        raise IndexError(f"Wrong dimension of _coordinates:{_coordinates.shape}, should be 2darray")
    if _coordinates.shape[1] == 3:
        from . import interpolate_chr
        # calculate corrlation matrix
        _coef_map = np.corrcoef(squareform(pdist(interpolate_chr( _coordinates ))))
    elif _coordinates.shape[0] == _coordinates.shape[1] :
        _coef_map = np.ma.corrcoef(np.ma.masked_invalid(_coordinates))
    else:
        raise IndexError(f"Wrong dimension of _coordinates:{_coordinates.shape}, should be zxy or distmap")

    # get domain starts and ends
    _starts = np.array(domain_starts, dtype=np.int)
    if 0 not in _starts:
        _starts = np.concatenate([np.array([0]), _starts])
    _ends = np.concatenate([_starts[1:],np.array([len(_coordinates)])])
    # domain vectors
    _corr_vectors = [np.nanmean(_coef_map[_s:_e], axis=0) for _s,_e in zip(_starts, _ends)]
    # domain corr pdists
    return pdist(_corr_vectors) / len(_coordinates)

def domain_neighboring_dists(coordinates, domain_starts, metric='median', 
                             use_local=True, min_dom_sz=5, 
                             normalization_mat=None, allow_minus_dist=True):
    """Function to calculate neighboring domain distances for domain calling
    Inputs:
        coordnates: n-by-3 coordinates for a chromosome, or n-by-n distance matrix, np.ndarray
        domain_starts: start coordinates for each domain, list/np.ndarray of ints
        metric: distance metric between domains in func:'domain_distance', {'median'|'mean'|'ks'}
        use_local: whether use distances only around local boundary rather than whole domain, bool (default: True)
        min_dom_sz: minimum domain size allowed in this analysis, int (default: 5)
        normalization_mat: normalization matrix, similar shape as coordinates, np.ndarray (default: None. no correction)
        allow_minus_dist: whether distances can be minus, bool (default: True)
    Outputs:
        _neighboring_dists: distances bewteen neighboring regions, np.ndarray, shape=len(domain_starts)-1
        """
    ## check inputs
    coordinates = np.array(coordinates).copy()
    if len(np.shape(coordinates)) != 2:
        raise ValueError(
            f"Wrong input shape for coordinates, should be 2d but {len(np.shape(coordinates))} is given")
    elif np.shape(coordinates)[0] != np.shape(coordinates)[1] and np.shape(coordinates)[1] != 3:
        raise ValueError(
            f"Input coordinates should be distance-matrix or 3d-coordinates!")

    domain_starts = np.array(domain_starts, dtype=np.int)
    domain_ends = np.zeros(np.shape(domain_starts))
    domain_ends[:-1] = domain_starts[1:]
    domain_ends[-1] = len(coordinates)
    
    # first check whether do normalzation
    if normalization_mat is not None:
        if normalization_mat.shape[0] != np.shape(coordinates)[0] or normalization_mat.shape[1] != normalization_mat.shape[0]:
            raise ValueError(
                f"Wrong shape of normalization:{normalization_mat.shape}, should be equal to {np.shape(coordinates)[0]}")

    # initialize
    _neighboring_dists = []
    for _i in range(len(domain_starts)-1):
        _s1, _e1 = domain_starts[_i], domain_ends[_i]
        _s2, _e2 = domain_starts[_i+1], domain_ends[_i+1]
        # change to local if only use local distances
        if use_local:
            _ns1 = max(_s1, _e1 - 2*max(_e2-_s2, min_dom_sz))
            _ne2 = min(_e2, _s2 + 2*max(_e1-_s1, min_dom_sz))
            _s1,_e2 = _ns1, _ne2
        # calculate distance of this pair
        _neighboring_dists.append(domain_distance(coordinates, 
                                                  [_s1,_e1], [_s2,_e2], _metric=metric,
                                                  _normalization_mat=normalization_mat,
                                                  _allow_minus_distance=allow_minus_dist))

    return np.array(_neighboring_dists)


def domain_stat(coordinates, _dom1_bds, _dom2_bds,
                _method='ks', _normalization_mat=None,
                _allow_minus_distance=True, 
                _make_plot=False, _return_pval=True):
    """Function to measure domain difference statistics
    use KS-statistic as a distance:
        citation: https://arxiv.org/abs/1711.00761"""
    ## check inputs
    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)
    _method = str(_method).lower()
    _allowed_methods = ['ks', 'ttest']
    if _method not in _allowed_methods:
        raise ValueError(f"Wrong input kwd _method:{_method}, should be within:{_allowed_methods}")
    # standardize domain boundaries
    _dom1_bds = [int(_b) for _b in _dom1_bds]
    _dom2_bds = [int(_b) for _b in _dom2_bds]
 
    # based on coordinates given, get intra/inter distances
    if len(np.shape(coordinates)) != 2:
        raise ValueError(f"Wrong input shape for coordinates, should be 2d but {len(np.shape(coordinates))} is given")
    elif np.shape(coordinates)[0] == np.shape(coordinates)[1]:
        _mat = coordinates
        _intra1 = _mat[_dom1_bds[0]:_dom1_bds[1], _dom1_bds[0]:_dom1_bds[1]]
        _intra2 = _mat[_dom2_bds[0]:_dom2_bds[1], _dom2_bds[0]:_dom2_bds[1]]
        _intra1 = _intra1[np.triu_indices(len(_intra1),1)]
        _intra2 = _intra2[np.triu_indices(len(_intra2), 1)]

        _intra_dist = [_intra1, _intra2]
        _inter_dist = np.ravel(_mat[_dom1_bds[0]:_dom1_bds[1], _dom2_bds[0]:_dom2_bds[1]])

    elif np.shape(coordinates)[1] == 3:
        # extract sequence
        zxy1 = coordinates[_dom1_bds[0]:_dom1_bds[1]]
        zxy2 = coordinates[_dom2_bds[0]:_dom2_bds[1]]
        # get distances
        _intra_dist = [pdist(zxy1), pdist(zxy2)]
        _inter_dist = np.ravel(cdist(zxy1, zxy2))
    else:
        raise ValueError(f"Input coordinates should be distance-matrix or 3d-coordinates!")

    # normalization
    if _normalization_mat is not None:
        # check other inputs
        if _dom1_bds is None or _dom2_bds is None:
            raise TypeError(
                f"Domain boundaries not fully given while normalization specified, skip normalization!")
        # normalize!
        else:
            _intra_dist[0] = _intra_dist[0] / squareform(
                _normalization_mat[_dom1_bds[0]:_dom1_bds[1], _dom1_bds[0]:_dom1_bds[1]])
            _intra_dist[1] = _intra_dist[1] / squareform(
                _normalization_mat[_dom2_bds[0]:_dom2_bds[1], _dom2_bds[0]:_dom2_bds[1]])
            _intra_dist = np.concatenate(_intra_dist)
            _inter_dist = _inter_dist / \
                np.ravel(
                    _normalization_mat[_dom1_bds[0]:_dom1_bds[1], _dom2_bds[0]:_dom2_bds[1]])
    else:
        # not normalize? directly concatenate
        _intra_dist = np.concatenate(_intra_dist)

    _kept_inter = _inter_dist[np.isnan(_inter_dist) == False]
    _kept_intra = _intra_dist[np.isnan(_intra_dist) == False]

    if len(_kept_inter) == 0 or len(_kept_intra) == 0:
        if _return_pval:
            return 0, 1
        else:
            return 0

    if _make_plot:
        plt.figure()
        plt.hist(_kept_inter, density=True, alpha=0.5, label='inter-domain')
        plt.hist(_kept_intra, density=True, alpha=0.5, label='intra-domain')
        plt.legend()
        plt.show()

    if _method == 'ks':
        if 'ks_2samp' not in locals():
            from scipy.stats import ks_2samp
        _f = np.sign(np.nanmedian(_inter_dist) - np.nanmedian(_intra_dist))
        _dist, _pval =  ks_2samp(_kept_inter, _kept_intra)
        _final_dist =  _f * _dist

    if _method == 'ttest':
        if 'ttest_ind' not in locals():
            from scipy.stats import ttest_ind

        # normalization factor
        _norm_factor = np.mean([np.nanmedian(_kept_inter), np.nanmedian(_kept_intra)])
        _final_dist, _pval =  ttest_ind(_kept_inter/_norm_factor, _kept_intra/_norm_factor)

    if _return_pval:
        return _final_dist, _pval
    else:
        return _final_dist

def domain_neighboring_stats(coordinates, domain_starts, method='ks', 
                             use_local=True, min_dom_sz=5, 
                             normalization_mat=None, allow_minus_dist=True,
                             return_pval=True):
    """Function to calculate neighboring domain diff-statistics for domain calling
    Inputs:
        coordnates: n-by-3 coordinates for a chromosome, or n-by-n distance matrix, np.ndarray
        domain_starts: start coordinates for each domain, list/np.ndarray of ints
        metric: distance metric between domains in func:'domain_distance', {'median'|'mean'|'ks'}
        use_local: whether use distances only around local boundary rather than whole domain, bool (default: True)
        min_dom_sz: minimum domain size allowed in this analysis, int (default: 5)
        normalization_mat: normalization matrix, similar shape as coordinates, np.ndarray (default: None. no correction)
        allow_minus_dist: whether distances can be minus, bool (default: True)
        return_pval: whether return p-values in statistics, bool (default: True)
    Outputs:
        _neighboring_dists: distances bewteen neighboring regions, np.ndarray, shape=len(domain_starts)-1
        """
    ## check inputs
    coordinates = np.array(coordinates).copy()
    if len(np.shape(coordinates)) != 2:
        raise ValueError(
            f"Wrong input shape for coordinates, should be 2d but {len(np.shape(coordinates))} is given")
    elif np.shape(coordinates)[0] != np.shape(coordinates)[1] and np.shape(coordinates)[1] != 3:
        raise ValueError(
            f"Input coordinates should be distance-matrix or 3d-coordinates!")

    domain_starts = np.array(domain_starts, dtype=np.int)
    domain_ends = np.zeros(np.shape(domain_starts))
    domain_ends[:-1] = domain_starts[1:]
    domain_ends[-1] = len(coordinates)
    
    # first check whether do normalzation
    if normalization_mat is not None:
        if normalization_mat.shape[0] != np.shape(coordinates)[0] or normalization_mat.shape[1] != normalization_mat.shape[0]:
            raise ValueError(
                f"Wrong shape of normalization:{normalization_mat.shape}, should be equal to {np.shape(coordinates)[0]}")

    # initialize
    _neighboring_stats = []
    for _i in range(len(domain_starts)-1):
        _s1, _e1 = domain_starts[_i], domain_ends[_i]
        _s2, _e2 = domain_starts[_i+1], domain_ends[_i+1]
        # change to local if only use local distances
        if use_local:
            _ns1 = max(_s1, _e1 - 2*max(_e2-_s2, min_dom_sz))
            _ne2 = min(_e2, _s2 + 2*max(_e1-_s1, min_dom_sz))
            _s1,_e2 = _ns1, _ne2
        # calculate distance of this pair
        _neighboring_stats.append(domain_stat(coordinates, 
                                    [_s1,_e1], [_s2,_e2], _method=method,
                                    _normalization_mat=normalization_mat,
                                    _allow_minus_distance=allow_minus_dist,
                                    _return_pval=return_pval))
    # summarize
    if return_pval:
        _stats = np.array([_s[0] for _s in _neighboring_stats])
        _pvals = np.array([_s[1] for _s in _neighboring_stats])
        return _stats, _pvals
    else:
        return np.array(_neighboring_stats)


def _neighboring_distance(zxys, spot_ids=None, radius=5):
    # zxys
    zxys = np.array(zxys)
    # spot_ids
    if spot_ids is None:
        spot_ids = np.arange(len(zxys))
    else:
        spot_ids = np.array(spot_ids, dtype=np.int)
    # radius
    _radius = int(radius)
    # init local dists
    _neighboring_dists = []
    for _i, _zxy in enumerate(zxys):
        _neighbor_inds = np.intersect1d(np.delete(np.arange(_i-_radius, _i+_radius+1),_radius), spot_ids)
        if len(_neighbor_inds) == 0:
            _neighboring_dists.append(np.nan)
        else:
            _neighboring_dists.append(np.linalg.norm(np.nanmean(zxys[_neighbor_inds], axis=0) - _zxy))
    return np.array(_neighboring_dists)


def _domain_contact_freq(_coordinates, _domain_starts, _contact_th=400):
    """Function to convert spot coordinates and domain starts into domain interaction frequency matrix"""
    ## check inputs
    # convert _coordinates into matrix
    _coordinates = np.array(_coordinates)
    if _coordinates.shape[0] != _coordinates.shape[1]:
        _coordinates = squareform(pdist(_coordinates))
    # _domain_starts
    _domain_starts = np.sort(_domain_starts).astype(np.int)
    if 0 not in _domain_starts:
        _domain_starts = np.concatenate([np.array([0]), _domain_starts])
    if len(_coordinates) in _domain_starts:
        _domain_starts = np.setdiff1d(_domain_starts, [len(_coordinates)])
    ## start conversion
    # get contact map
    _concact = _coordinates <= _contact_th
    # initialize contact mat
    _dm_contact_mat = np.zeros([len(_domain_starts), len(_domain_starts)])
    # slice
    _domain_ends = np.concatenate([_domain_starts[1:], np.array([len(_coordinates)])])
    _dm_slices = [slice(_s,_e) for _s,_e in zip(_domain_starts, _domain_ends)]
    for _i, _si in enumerate(_dm_slices):
        for _j, _sj in enumerate(_dm_slices[:_i+1]):
            _dm_contact_mat[_i,_j] = np.mean(_concact[_si,_sj])
            _dm_contact_mat[_j,_i] = np.mean(_concact[_si,_sj])
    return _dm_contact_mat