import numpy as np
import time 
from . import _distance_zxy
# Function to give scores
def distance_score(dist, ref_dist, weight=1., 
                   metric='linear', distance_limits=[-np.inf, np.inf],
                   nan_mask=-1000, inf_mask=-1000.):
    """Function to calculate distance score given a reference number
    weight is dominating here, which means if weight is zero then everything is zero"""
    # localize variables
    _dist = np.array(dist)
    # weight
    _w = float(weight)
    # metric
    metric = metric.lower()
    # Scores
    _scores = np.zeros(np.shape(_dist))
    # if weight is zero, return zeros!
    if _w == 0.:
        return _scores
    # calculate score based on metric types
    if metric == 'linear':
        # ref_dist is one number if metric is linear
        _ref_dist = float(ref_dist)
        # calculate score
        _scores = -1 * _w * (_dist / _ref_dist) # this score mimic log-pval
        # extra penalty for distance larger than limit
        _scores[_dist>max(distance_limits)] -= \
            _w * (_dist[_dist>max(distance_limits)]-max(distance_limits)) / _ref_dist
    elif metric == 'cdf':
        # ref_dist should be an array of numbers
        _ref_dist = np.ravel(np.array(ref_dist))
        _ref_dist = _ref_dist[np.isnan(_ref_dist)==False] # exclude nans
        if len(_ref_dist) == 0:
            raise ValueError(f"Wrong input ref_dist, not enough values to calculate cdf")
        _cdf = 1 - _cum_prob(_ref_dist, dist,
                             vmin=min(distance_limits), 
                             vmax=max(distance_limits))
        # calculate scores
        _scores[_cdf > 0] = np.log(_cdf[_cdf > 0]) * _w
        _scores[_cdf <= 0] = - np.inf
    else:
        raise ValueError(f"metric type:{metric} has not been supported yet!")
    # remove nan spots
    _scores = np.array(_scores)
    _scores[np.isnan(_dist)] = nan_mask
    # NOTE
    # this doesnt matter because if lc_dist is nan, all spots has comparable scores
    # if ct_dist is nan, which means this is a bad spota
    return _scores

def intensity_score(intensity, ref_intensities, weight=1.,
                    metric='linear', intensity_th=0.,
                    nan_mask=0., inf_mask=-1000.):
    # input intensites
    _int = np.array(intensity)
    # weight
    _w = float(weight)
    # scores
    _scores = np.zeros(np.shape(_int))
    # calculate scores in different metrices
    if metric == 'linear':
        _ref = float(ref_intensities)
        _scores[_int <= 0] = - np.inf
        _scores[_int > 0] = np.log(_int[_int > 0] / (_int[_int > 0] + _ref) ) * _w
    elif metric == 'cdf':
        if isinstance(ref_intensities, int) or isinstance(ref_intensities, float):
            raise ValueError(f"ref_intensity in {metric} mode should be an array rather than one value")
        _ref = np.array(ref_intensities)            
        _cdf = _cum_prob(_ref, _int, vmin=intensity_th)
        _scores[_cdf > 0] = np.log(_cdf[_cdf > 0]) * _w
        _scores[_cdf <= 0] = - np.inf
    # apply masks
    _scores[np.isnan(_scores)] = nan_mask
    _scores[np.isinf(_scores)] = inf_mask
    
    return _scores

# accumulative prob.
def _cum_prob(data, target_value, vmin=-np.inf, vmax=np.inf):
    """Function to calculate CDF from a dataset"""
    data = np.array(data, dtype=np.float)
    data = data[np.isnan(data)==False]
    if len(data) == 0:
        raise ValueError(f"Wrong input data, no valid points at all.")
    target_value = np.array(target_value, dtype=np.float)
    if len(target_value.shape) == 0:
        target_value = np.array([target_value], dtype=np.float)
    target_value[np.isnan(target_value)] = np.inf
    target_shape = np.shape(target_value)
    target_value = target_value.reshape(-1)
    cprob = np.array([np.nansum(data[np.isnan(data)==False] <= _v) / np.sum(np.isnan(data)==False) 
                      for _v in target_value])
    # rescale
    min_p = np.nansum(data[np.isnan(data)==False] <= vmin) / np.sum(np.isnan(data)==False)
    max_p = np.nansum(data[np.isnan(data)==False] <= vmax) / np.sum(np.isnan(data)==False)
    if max_p <= min_p:
        cprob = (cprob-min_p)
    else:
        cprob = (cprob-min_p) / (max_p - min_p)
    # remove extreme values
    cprob[cprob <= 0] = 0
    cprob[np.isnan(cprob)] = 1. / np.nansum(np.isnan(data)==False)
    cprob[cprob >= 1] = 1
    cprob = cprob.reshape(target_shape)
    return cprob

def _center_distance(spot_zxys, center=None):
    """Function to calculate center distance"""
    _zxys = np.array(spot_zxys)
    if center is None:
        _ref_center = np.nanmean(_zxys, axis=0)
    else:
        _ref_center = center
    if len(_zxys.shape) == 2:
        _ct_dist = np.linalg.norm(_zxys-_ref_center, axis=1)
    elif len(_zxys.shape) == 1:
        _ct_dist = np.linalg.norm(_zxys-_ref_center)
    else:
        raise IndexError(f"Wrong shape for spot_zxys:{_zxys.shape}")
    return _ct_dist

def _local_distance(spot_zxys, spot_ids, sel_zxys, sel_ids, 
                    local_size=5, invalid_dist=np.nan):
    """Function to caluclate local distance"""
    _spot_zxys = np.array(spot_zxys)
    _spot_ids = np.array(spot_ids, dtype=np.int)
    
    _sel_zxys = np.array(sel_zxys)
    _sel_ids = np.array(sel_ids, dtype=np.int)
    
    _half_size = int((local_size-1)/2)
    # generate inds based on pt_ids
    _spot_inds = [np.delete(np.arange(_id-_half_size, _id+1+_half_size), _half_size) for _id in _spot_ids]
    # initialize local_distances
    _local_dists = []
    
    for _zxy, _ind in zip(_spot_zxys, _spot_inds):
        _local_zxys = []
        for _i in _ind:
            if _i in _sel_ids:
                _local_zxys += list(_sel_zxys[np.where(_sel_ids==_i)[0]])
        _local_zxys = np.array(_local_zxys)
        # no local spot founded
        if len(_local_zxys) == 0:
            _local_dists.append(invalid_dist)
        # no valid local_spot founded
        elif np.sum(np.isnan(_local_zxys).sum(1)>0) == len(_local_zxys):
            _local_dists.append(invalid_dist)
        # save
        else:
            _local_dists.append(np.linalg.norm(np.nanmean(_local_zxys, axis=0) - _zxy))
    
    return np.array(_local_dists)

def _neighboring_distance(spot_zxys, spot_ids=None, 
                          neighbor_step=1, invalid_dist=np.nan, 
                          use_median=True):
    """Function to calculate neighboring distances between list of spot_zxys"""
    _zxys = np.array(spot_zxys)
    if spot_ids is None:
        _ids = np.arange(len(_zxys)).astype(np.int)
    else:
        _ids = np.array(spot_ids, dtype=np.int)
    
    _nb_dists = []
    for _i, (_zxy, _id) in enumerate(zip(_zxys, _ids)):
        if _id+neighbor_step in _ids:
            _nzxy = _zxys[np.where(_id+neighbor_step==_ids)[0]]
            if use_median:
                _nb_dists.append(np.nanmedian((np.linalg.norm(_nzxy-_zxy, axis=1))))
            else:
                _nb_dists += list(np.linalg.norm(_nzxy-_zxy, axis=1))
        else:
            _nb_dists.append(invalid_dist)
    
    return np.array(_nb_dists)

def neighboring_distances(spot_zxys, spot_ids=None, 
                          neighbor_step=1, invalid_dist=np.nan, 
                          use_median=True):
    """Function to calculate neighboring distances between list of spot_zxys"""
    _zxys = np.array(spot_zxys)
    if spot_ids is None:
        _ids = np.arange(len(_zxys)).astype(np.int)
    else:
        _ids = np.array(spot_ids, dtype=np.int)
    
    _fwd_nb_dists, _rev_nb_dists = [], []
    for _i, (_zxy, _id) in enumerate(zip(_zxys, _ids)):
        if _id+neighbor_step in _ids:
            _fzxy = _zxys[np.where(_id + neighbor_step == _ids)[0]]
            _rzxy = _zxys[np.where(_id - neighbor_step == _ids)[0]]
            if use_median:
                _fwd_nb_dists.append(np.nanmedian((np.linalg.norm(_fzxy-_zxy, axis=1))))
                _rev_nb_dists.append(np.nanmedian((np.linalg.norm(_rzxy-_zxy, axis=1))))
            else:
                _fwd_nb_dists += list(np.linalg.norm(_fzxy-_zxy, axis=1))
                _rev_nb_dists += list(np.linalg.norm(_rzxy-_zxy, axis=1))
        else:
            _fwd_nb_dists.append(invalid_dist)
            _rev_nb_dists.append(invalid_dist)
    
    return np.array(_fwd_nb_dists), np.array(_rev_nb_dists)


def _filter_intensities(spots, intensity_th=0., invalid_dist=np.nan):
    """Function to filter intensities from spots"""
    _spots = np.array(spots)
    _intensities = _spots[:,0]
    _intensities[_intensities <= intensity_th] = invalid_dist

    return _intensities


def generate_ref_from_chromosome(sel_spots, sel_ids=None, distance_zxy=_distance_zxy, 
                                 chr_center=None, intensity_th=0., 
                                 local_size=5, ref_dist_metric='median', ignore_nan=True):
    """Generate all reference scoring info from pre-selected chromosomes
    Inputs:
        sel_spots: selected spots for one specific chromosome, np.ndarray or list of 1d-array
        sel_ids: selected spot ids for corresponding to sel_ids, np.1d-array or list of ints
        distance_zxy: distance in nm for z,x,y pixels, array of 3 (defualt:[200,106,106])
        chr_center: reference center of selected spots, array of 3 (default: None, which is current center of sel_spots)
        local_size: window size of local distance calculation, int (default:5)
        ref_dist_metric: metric of reference distance, str {'median'|'rg'|'cdf'|...} (default: 'median')
        ignore_nan: whether ignore nan spots in these metrics, bool (default: True)
    Outputs:
        _chr_center_dist, _ref_local_dist, _ref_neighbor_dist, _ref_intensities
    """
    # localize spots
    _spots = np.array(sel_spots)
    # sel_ids
    if sel_ids is None:
        _ids = np.arange(len(_spots)).astype(np.int)
    else:
        if len(sel_ids) != len(sel_spots):
            raise IndexError(f"Wrong input ")
        _ids = np.array(sel_ids, dtype=np.int)
    #print(f"** shape of ref spots{_spots.shape}, {distance_zxy.shape}")
    # zxys
    _zxys = _spots[:,1:4] * distance_zxy[np.newaxis,:]
    # ref_dist_metric
    ref_dist_metric = ref_dist_metric.lower()
    
    # calculate center distance list
    if chr_center is None:
        _chr_center = np.nanmean(_zxys, axis=0)
    else:
        _chr_center = chr_center * distance_zxy
    _ct_dist = _center_distance(_zxys, _chr_center)
    #
    if ignore_nan:
        _ct_dist = _ct_dist[np.isnan(_ct_dist)==False]
        if len(_ct_dist) == 0:
            print(f"_ct_dist has no valid values in this chromosome")
            _ct_dist = [1000]
    # calculate local distance list
    _lc_dist = _local_distance(_zxys, _ids, _zxys, _ids, local_size=local_size)
    if ignore_nan:
        _lc_dist = _lc_dist[np.isnan(_lc_dist)==False]
        if len(_lc_dist) == 0:
            print(f"_lc_dist has no valid values in this chromosome", end=', ')
            _lc_dist = [np.inf]
    # caluclate neighboring distances
    _nb_dist = _neighboring_distance(_zxys, _ids)
    if ignore_nan:
        _nb_dist = _nb_dist[np.isnan(_nb_dist)==False]
        if len(_nb_dist) == 0:
            print(f"_nb_dist has no valid values in this chromosome", end=', ')
            _nb_dist = [np.inf]
    # intensities
    _intensities = _filter_intensities(_spots, intensity_th=intensity_th)
    if ignore_nan:
        _intensities = _intensities[np.isnan(_intensities)==False]
        if len(_intensities) == 0:
            print(f"_intensities has no valid values in this chromosome")
            _intensities = np.ones(1)
    # determine ref-metric type
    if ref_dist_metric == 'median':
        _ref_center_dist = np.nanmedian(_ct_dist)
        _ref_local_dist = np.nanmedian(_lc_dist)
        _ref_neighbor_dist = np.nanmedian(_nb_dist)
        _ref_intensities = np.nanmedian(_intensities)
    elif ref_dist_metric == 'mean':
        _ref_center_dist = np.nanmean(_ct_dist)
        _ref_local_dist = np.nanmean(_lc_dist)
        _ref_neighbor_dist = np.nanmean(_nb_dist)
        _ref_intensities = np.nanmean(_intensities)
    elif ref_dist_metric == 'rg':
        _ref_center_dist = radius_of_gyration(_zxys)
        _ref_local_dist = radius_of_gyration(_zxys)
        _ref_neighbor_dist = radius_of_gyration(_zxys)
        _ref_intensities = _intensities
    elif ref_dist_metric == 'cdf':
        _ref_center_dist = _ct_dist
        _ref_local_dist = _lc_dist
        _ref_neighbor_dist = _nb_dist
        _ref_intensities = _intensities
    else:
        raise ValueError(f"Not supported yet!")
    return _ref_center_dist, _ref_local_dist, _ref_neighbor_dist, _ref_intensities


def spot_score_in_chromosome(spots, spot_ids, sel_spots, 
                             sel_ids=None, chr_center=None,
                             ref_center_dist=None, ref_local_dist=None, ref_neighbor_dist=None, 
                             ref_intensities=None, ref_dist_metric='median', 
                             ignore_nan=True, nan_mask=0., inf_mask=-1000., 
                             distance_zxy=_distance_zxy, distance_limits=[0,np.inf], metric='linear',
                             intensity_th=0., local_size=5, w_ctdist=1, w_lcdist=0.1, w_int=1, verbose=True):
    """Function to generate spot_scores in a given chromosomes selected spots
    Inputs:
        spots: spots requires score calculation, np.ndarray of list of 1d-array
        spot_ids: region ids corresponding to spots provided, np.1d-array or list of ints
        sel_spots: selected spots for a chromosome, np.ndarray of list of 1d-array
        sel_ids: region ids corresponding to sel_spots, np.1d-array or list of ints 
            (default: None, 0 to length of sel_spots)
        chr_center: chromosome center coordinate, array-like of 3 (default: None, median of sel_spots)
        ref_center_dist: reference of distances to chromosome center, np.1d-array or float 
            (default: calculate from sel_spots)
        ref_local_dist: reference of distances to chromosome center, np.1d-array or float 
            (default: calculate from sel_spots)
        ref_neighbor_dist: reference of distances to neighboring region, np.1d-array or float 
            (default: calculate from sel_spots)
        ref_intensities: reference of intensities, np.1d-array or float 
            (default: calculate from sel_spots)
        ref_dist_metric: metric for reference distance calculation, str 
            (default: 'median', required if calculation of reference is required)
        ignore_nan: whether ignore nan reference points, bool (default: True)
        distance_zxy: distance in nm for z,x,y pixels, array of 3 (defualt:[200,106,106])
        distance_limits: limits for distance scores, None / low_limit / low_high_limit
        metric: metric of scoring, str {'linear'|'cdf'|...} (default: 'linear')
        intensity_th: minimum threshold for intensiites, float (default: 0., only exclude bad spots)
        local_size: window size of local distance calculation, int (default:5)
        w_ctdist: weight for distance to chr-center, float (default: 1)
        w_lcdist: weight for distance to local-center, float (default: 0.1)
        w_int: weight for intensity, float (default: 1)
    Outputs:
        _scores: spot scores corresponding to spots, np.1d-array, same length as spots
    """
    ## check inputs
    # target spots and ids
    _spots = np.array(spots)
    _zxys = _spots[:,1:4] * distance_zxy[np.newaxis, :]
    # ids 
    if isinstance(spot_ids, list) or isinstance(spot_ids, np.ndarray):
        _ids = np.array(spot_ids, dtype=np.int)
    elif isinstance(spot_ids, int) or isinstance(spot_ids, float) or isinstance(spot_ids, np.int32):
        _ids = np.ones(len(_zxys),dtype=np.int) * spot_ids
    else:
        raise TypeError(f"Wrong input type for spot_ids, should be either np.ndarray/list or int/np.int32")
    if len(_spots) != len(_ids):
        raise IndexError(f"Wrong input length of _spots and _ids, length should match!")
    # selected spots and ids
    _sel_spots = np.array(sel_spots)
    if sel_ids is None:
        _sel_ids = np.arange(len(sel_spots))
    else:
        if len(_sel_spots) != len(sel_ids):
            raise IndexError(f"Wrong input length of _sel_spots and sel_ids, length should match!")
        _sel_ids = np.array(sel_ids, dtype=np.int)
    _sel_zxys = _sel_spots[:,1:4] * distance_zxy[np.newaxis, :]
    # chr_center
    if chr_center is None:
        _chr_center = np.nanmean(_sel_zxys, axis=0)
    else:
        _chr_center = chr_center * distance_zxy

    if ref_center_dist is None or ref_local_dist is None or ref_intensities is None:
        if metric == 'cdf' and ref_dist_metric != metric:
            ref_dist_metric = metric # match two metrics
            print(f"-- adjusted ref_dist_metric to {metric} to match performance")
        if verbose:
            print(f"-- generate reference from selected spots.")
        ref_center_dist, ref_local_dist, ref_neighbor_dist, ref_intensities = generate_ref_from_chromosome(
            _sel_spots, _sel_ids, distance_zxy=distance_zxy, chr_center=chr_center, 
            intensity_th=intensity_th, local_size=local_size, 
            ref_dist_metric=ref_dist_metric, ignore_nan=ignore_nan,
        )

    # calculate scores
    _spot_center_scores = _center_distance(_zxys, _chr_center)
    _spot_local_scores = _local_distance(_zxys, _ids, _sel_zxys, _sel_ids, local_size=local_size)
    _spot_intensities = _spots[:,0]
    _scores = distance_score(_spot_center_scores, ref_center_dist, weight=w_ctdist, 
                             metric=metric, distance_limits=distance_limits,
                             nan_mask=nan_mask, inf_mask=inf_mask) \
            + distance_score(_spot_local_scores, ref_local_dist, weight=w_lcdist, 
                             metric=metric, distance_limits=distance_limits,
                             nan_mask=nan_mask, inf_mask=inf_mask) \
            + intensity_score(_spot_intensities, ref_intensities, weight=w_int, 
                              metric=metric, intensity_th=intensity_th,
                              nan_mask=nan_mask, inf_mask=inf_mask)
    # For debugging:
    #print('ct', distance_score(_spot_center_scores, ref_center_dist, weight=w_ctdist, 
    #                         metric=metric, distance_limits=distance_limits, 
    #                         nan_mask=nan_mask, inf_mask=inf_mask))
    #print('lc', distance_score(_spot_local_scores, ref_local_dist, weight=w_lcdist, 
    #                         metric=metric, distance_limits=distance_limits,
    #                         nan_mask=nan_mask, inf_mask=inf_mask))
    #print('in',intensity_score(_spot_intensities, ref_intensities, weight=w_int, 
    #                          metric=metric, intensity_th=intensity_th,
    #                          nan_mask=nan_mask, inf_mask=inf_mask))
    
    return _scores



def radius_of_gyration(zxys):
    """Function to calculate radius of gyration"""
    zxys = np.array(zxys)
    if len(np.shape(zxys)) != 2:
        return IndexError(f"zxys should be 2d-array!")
    # calculate Rs for each spots
    _rs = np.linalg.norm(zxys - np.nanmean(zxys, axis=0), axis=1)
    # calculate radius of gyration
    _rg = np.sqrt(np.nanmean(_rs**2))
    return _rg


def chromosomal_spot_scores(spots, region_ids, 
                            sel_spots, sel_ids=None, score_metric='cdf',
                            intensity_th=1., local_size=5, 
                            w_ctdist=1, w_lcdist=1, w_int=1, w_nbdist=1,
                            chr_center=None,
                            distance_zxy=_distance_zxy, distance_limits=[0,np.inf],
                            ref_center_dists=None, ref_local_dists=None, 
                            ref_neighbor_dists=None, ref_intensities=None, 
                            nan_mask=-1000, inf_mask=-1000, ignore_nan=True,
                            return_separate_scores=True, verbose=True):
    """NEW function to calculate chromosomal spot scores used for EM spot picking.
    Inputs:
        
    Outputs:
        
    """
    ## check inputs
    # spots
    _spots = np.array(spots)
    if len(np.shape(_spots)) != 2:
        raise IndexError(f"Wrong shape of _spots, should be 2d array!")
    # region ids 
    if isinstance(region_ids, list) or isinstance(region_ids, np.ndarray):
        _region_ids = np.array(region_ids, dtype=np.int)
    elif isinstance(region_ids, int) or isinstance(region_ids, float) or isinstance(region_ids, np.int32):
        _region_ids = np.ones(len(_spots),dtype=np.int) * int(region_ids)
    else:
        raise TypeError(f"Wrong input type for region_ids, should be either np.ndarray/list or int/np.int32")
    # sel_spots
    _sel_spots = np.array(sel_spots)
    if len(np.shape(_sel_spots)) != 2:
        raise IndexError(f"Wrong shape of _sel_spots, should be 2d array!")
    # sel_ids
    if sel_ids is None:
        _sel_ids = np.arange(len(_sel_spots))
    else:
        _sel_ids = np.array(sel_ids)
        
    ## generate references
    if ref_center_dists is None or ref_local_dists is None \
        or ref_neighbor_dists is None or ref_intensities is None:
        if verbose:
            _ref_time = time.time()
            print(f"-- generate spot score reference from sel_spots in", end=' ')
        ref_center_dists, ref_local_dists, ref_neighbor_dists, ref_intensities = generate_ref_from_chromosome(
            _sel_spots, _sel_ids, distance_zxy=distance_zxy, chr_center=chr_center, 
            intensity_th=intensity_th, local_size=local_size, 
            ref_dist_metric=score_metric, ignore_nan=ignore_nan,
        )
        if verbose:
            print(f"in {time.time()-_ref_time:.2f}s.")
    else:
        if verbose:
            print(f"-- use given spot score reference.")
    ## calculate corresponding distances
    if verbose:
        _score_time = time.time()
        print(f"-- calculate {len(_spots)} spot scores in", end=' ')
    _zxys = _spots[:,1:4] * distance_zxy
    _sel_zxys = _sel_spots[:,1:4] * distance_zxy
    if chr_center is None:
        _chr_center = np.nanmean(_sel_zxys, axis=0)
    else:
        _chr_center = chr_center * distance_zxy
    # spot to chr center distances
    _spot_center_dists = _center_distance(_zxys, _chr_center) ##
    # spot to local center distances
    _spot_local_dists = _local_distance(_zxys, _region_ids, _sel_zxys, _sel_ids, local_size=local_size) ##
    # spot to neighbor spot distances
    _fwd_neighbor_dists, _rev_neighbor_dists = neighboring_distances(_zxys, _region_ids, use_median=True)
    # spot intensities
    _spot_intensities = _spots[:,0]

    # calculate scores
    _ct_scores = distance_score(_spot_center_dists, ref_center_dists, weight=w_ctdist, 
                             metric=score_metric, distance_limits=distance_limits,
                             nan_mask=nan_mask, inf_mask=inf_mask) 
    _lc_scores = distance_score(_spot_local_dists, ref_local_dists, weight=w_lcdist, 
                             metric=score_metric, distance_limits=distance_limits,
                             nan_mask=nan_mask, inf_mask=inf_mask) 
    _nb_scores = distance_score(np.nanmean([_fwd_neighbor_dists,_rev_neighbor_dists], axis=0), 
                             ref_neighbor_dists, 
                             weight=w_nbdist, 
                             metric=score_metric, distance_limits=distance_limits,
                             nan_mask=nan_mask, inf_mask=inf_mask) 
    _int_scores = intensity_score(_spot_intensities, ref_intensities, weight=w_int, 
                              metric=score_metric, intensity_th=intensity_th,
                              nan_mask=nan_mask, inf_mask=inf_mask)
    # summarize
    _scores = _ct_scores + _lc_scores + _nb_scores + _int_scores
    if verbose:
        print(f"in {time.time()-_score_time:.2f}s.")
    if return_separate_scores:
        return _ct_scores, _lc_scores, _nb_scores, _int_scores
    else:
        return _scores