import sys,glob,os, time
import numpy as np
import pickle as pickle
import matplotlib.pyplot as plt
import multiprocessing
from .. import get_img_info, corrections, visual_tools
from . import _correction_folder,_temp_folder,_distance_zxy,_sigma_zxy
from . import scoring, checking
from scipy.spatial.distance import pdist, squareform


def naive_pick_spots(cand_spots, region_ids, use_chrom_coord=True, chrom_id=None, 
                     return_indices=False, verbose=True):
    """Naive pick spots simply by intensity"""
    ## check inputs
    if len(cand_spots) != len(region_ids):
        raise ValueError(
            "cand_spots and region_ids should have the same length!")
    if chrom_id is None and use_chrom_coord:
        raise ValueError(
            f"chrom_id should be given if use_chrom_coord is True!")
    elif chrom_id is not None and not isinstance(chrom_id, int):
        chrom_id = int(chrom_id)

    ## For now only support use_chrom_coord mode.
    _selected_spots = []
    _selected_indices = []

    if use_chrom_coord:
        for _i, (_spots, _id) in enumerate(zip(cand_spots, region_ids)):
            # check chrom_id
            if len(_spots) <= chrom_id:
                raise IndexError(
                    f" spots:{_spots} for region:{_id} doesn't have spots for chromosome {chrom_id}")
            # extract points
            _pts = np.array(_spots[chrom_id])
            if len(_pts) == 0:
                _selected_spots.append(np.nan * np.ones(11))
                _selected_indices.append(-1)
            else:
                _selected_spots.append(_pts[np.argsort(_pts[:, 0])[-1]])
                _selected_indices.append(np.argsort(_pts[:, 0])[-1])
        
    ## for not use_chrom_coord
    else:
        for _i, (_spots, _id) in enumerate(zip(cand_spots, region_ids)):
            # extract points
            _pts = np.array(_spots)
            if len(_pts) == 0:
                _bad_pt = np.nan*np.ones(11)
                _bad_pt[0] = 0 # set bad_pt intensity=0
                _selected_spots.append(_bad_pt)
                _selected_indices.append(-1)
            else:
                _selected_spots.append(_pts[np.argsort(_pts[:, 0])[-1]])
                _selected_indices.append(np.argsort(_pts[:, 0])[-1])
    
    # return 
    if return_indices:
        return np.array(_selected_spots), np.array(_selected_indices, dtype=np.int)
    else:
        return np.array(_selected_spots)



def old_spot_score_in_chromosome(spots, reg_id, sel_spots, sel_ids=None, _chr_center=None,
                             _cc_dists=None, _lc_dists=None, _intensities=None,
                             distance_zxy=_distance_zxy, local_size=5, distance_limits=[0,3000],
                             w_ctdist=1, w_ccdist=1, w_lcdist=0.1, w_int=1, ignore_nan=True):
    """Function to calculate log-score for given spot in selected chr_pts from candidiate_points
    Inputs:
        spots: given fitted spots info, list of spots or one spot
        reg_id: region id for these given spots, int
        sel_spots: currently selected spots for chromosome tracing, list of spots / 2darray
        distance_zxy: transform from pixel to nm for z,x,y axes
        local_size: window size to calculate local distance, int (default: 5)
        w_ccdist: weight for distance to chr-center, float (default: 1)
        w_lcdist: weight for distance to local-center, float (default: 1)
        w_int: weight for intensity, float (default: 1)
    Output:
        _log_score: log score for this given spot, float 
    """
    # get chr coordinates
    _zxy = np.array(sel_spots)[:, 1:4]*np.array(distance_zxy)[np.newaxis, :]
    if sel_ids is None:
        sel_ids = np.arange(len(sel_spots))
    else:
        sel_ids = np.array(sel_ids, dtype=np.int)
        if len(sel_ids) != len(sel_spots):
            raise IndexError(f"Wrong input length of ref_ids:{len(sel_ids)}, should match ref_zxys:{len(sel_spots)}")
    # local chr_center
    _local_chr_center = np.nanmean(_zxy, axis=0)
    if _chr_center is not None:
        _ref_center = _chr_center.copy() * distance_zxy
    else:
        _ref_center = _local_chr_center
    _chr_drift = np.linalg.norm(_ref_center-_local_chr_center)
    #print(_chr_drift)
    # get pt coordinates
    _pts = np.array(spots)
    if len(np.shape(_pts)) == 1:
        _pts = _pts[np.newaxis, :]
    _pt_zxy = _pts[:, 1:4] * np.array(distance_zxy)[np.newaxis, :]
    if isinstance(reg_id, int) or isinstance(reg_id, np.int32) or len(reg_id) == 1:
        _rids = reg_id * np.ones(len(_pts), dtype=np.int)
    elif len(reg_id) == len(_pts):
        _rids = np.array(reg_id, dtype=np.int)
    else:
        raise ValueError(f"Input reg_id should be either a int or list of ints aligned with spots!")# get chr statistics
    # if not given, generate from existing chrom_data

    if _cc_dists is None:
        _cc_dists = np.linalg.norm(_zxy - _local_chr_center, axis=1)
    if _lc_dists is None:
        _lc_dists = _local_distance(_zxy, _zxy, np.arange(len(_zxy)))
    if _intensities is None:
        _intensities = _pts[:, 0]
    # get pt statistics
    _pt_ct_dist = np.linalg.norm(_pt_zxy - _ref_center, axis=1)
    _pt_cc_dist = np.linalg.norm(_pt_zxy - _local_chr_center, axis=1)
    _pt_lc_dist = _local_distance(_pt_zxy, _zxy, _rids)
    _pt_intensity = _pts[:, 0]
    # distance score
    if distance_limits is None:
        _lim = [-np.inf, np.inf]
    elif isinstance(distance_limits, int) or isinstance(distance_limits, float):
        _lim = [distance_limits, np.inf]
    elif isinstance(distance_limits, list) or isinstance(distance_limits, np.ndarray) and len(distance_limits) > 1:
        _lim = [min(distance_limits), max(distance_limits)]
    else:
        raise ValueError(f"Wrong inpuf format for distance_limits")
    # get score
    _log_score = np.log(1-_cum_prob(_cc_dists, _pt_ct_dist, vmin=_lim[0], vmax=_lim[1]) )*w_ctdist \
        + np.log(1-_cum_prob(_cc_dists, _pt_cc_dist, vmin=_lim[0], vmax=_lim[1]))*w_ccdist \
        + np.log(1-_cum_prob(_lc_dists, _pt_lc_dist, vmin=_lim[0], vmax=_lim[1]))*w_lcdist \
        + np.log(_cum_prob(_intensities, _pt_intensity))*w_int
    if ignore_nan:
        _nan_flags = np.isnan(_pts).sum(1)        
        _log_score[_nan_flags > 0] = - np.inf
    return _log_score


def distance_score_in_chromosome(dists, sel_spots=None, _nb_dists=None, 
                                 distance_zxy=_distance_zxy, distance_limits=[0,3000], 
                                 w_dist=1):
    """Function to calculate log-score for given spot in selected chr_pts from candidiate_points
    Inputs:
        spots: given fitted spots info, list of spots or one spot
        sel_spots: currently selected spots for chromosome tracing, list of spots / 2darray
        distance_zxy: transform from pixel to nm for z,x,y axes
        distance_limits: limits for distance scores, None / low_limit / low_high_limit
        w_dist: weight for distances, float (default: 1)
    Output:
        _log_score: log score for this given spot, float 
    """
    if _nb_dists is None:
        if sel_spots is None:
            raise ValueError(f"sel_spots should be given if _nb_dists is not given.")
        else:
            _zxy = np.array(sel_spots)[:, 1:4] * \
                np.array(distance_zxy)[np.newaxis, :]
            _nb_dists = np.linalg.norm(_zxy[1:]-_zxy[:-1], axis=1)
            _nb_dists = _nb_dists[np.isnan(_nb_dists)==False]
    # calculate cdf as direct score


    _direct_scores = 1-_cum_prob(_nb_dists, dists)
    if distance_limits is None:
        _direct_scores = 1-_cum_prob(_nb_dists, dists)
    elif isinstance(distance_limits, int) or isinstance(distance_limits, float):
        _direct_scores = 1-_cum_prob(_nb_dists, dists, vmin=distance_limits)
    elif isinstance(distance_limits, list) or isinstance(distance_limits, np.ndarray) and len(distance_limits) > 1:
        _direct_scores = 1-_cum_prob(_nb_dists, dists, 
                                     vmin=min(distance_limits),
                                     vmax=max(distance_limits))
    else:
        raise ValueError(f"Wrong inpuf format for distance_limits")
    # convert to log score
    _scores  = np.log( _direct_scores ) * w_dist



    return _scores

def generate_distance_score_pool(all_spots, distance_zxy=_distance_zxy):
    """Generate distance score pool from sel_spots"""
    if isinstance(all_spots, np.ndarray):
        _zxy = all_spots[:,1:4] * np.array(distance_zxy)[np.newaxis,:]
    elif isinstance(all_spots[0], np.ndarray) or len(all_spots[0].shape)==1:
        _zxy =  np.stack(all_spots)[:,1:4] * np.array(distance_zxy)[np.newaxis,:]
    elif  isinstance(all_spots[0], list) or len(all_spots[0].shape)==2:
        _spots = np.concatenate([np.array(_pts) for _pts in all_spots], axis=0)
        _zxy =  np.array(_spots)[:,1:4] * np.array(distance_zxy)[np.newaxis,:]
    else:
        raise TypeError("Wrong input datatype for all_spots, should be list of spots or list of list of spots!")
    _nb_dists = np.linalg.norm(_zxy[1:]-_zxy[:-1], axis=1)
    _nb_dists = _nb_dists[np.isnan(_nb_dists) == False]
    return _nb_dists


def _local_distance(spot_zxys, spot_ids, ref_zxys, 
                    ref_ids=None, size=5, minimal_dist=0.5):
    """Function to caluclate local distance from spot_zxys to ref_zxys
    Inputs:
        spot_zxys
        spot_ids
        ref_zxys
        ref_ids
    Outputs:

        """
    ## inputs
    # spots and ids
    _spot_zxys = np.array(spot_zxys)
    _ref_zxys = np.array(ref_zxys)
    _spot_ids = np.array(spot_ids, dtype=np.int)
    if ref_ids is None:
        _ref_ids = np.arange(len(ref_zxys))
    else:
        _ref_ids = np.array(ref_ids, dtype=np.int)
        if len(_ref_ids) != len(ref_zxys):
            raise IndexError(f"Wrong input length of ref_ids:{len(_ref_ids)}, should match ref_zxys:{len(_ref_zxys)}")
    # indices
    _half_size = int((size-1)/2)
    _inds = []
    for _id in spot_ids:
        _cand_ids = [_i for _i in range(_id-_half_size, _id+_half_size+1) if _i != _id and _i in _ref_ids]
        _inds.append(np.array(_cand_ids, dtype=np.int))
    _local_dists = []
    for _zxy, _ind in zip(_spot_zxys,_inds):
        if len(_ind) == 0:
            _local_dists.append(minimal_dist)
        else:
            _local_zxy = np.where(np.in1d(_ref_ids, _ind ) )[0]
            if np.sum(np.isnan(_local_zxy).sum(1)>0) == len(_local_zxy):
                _local_dists.append(0)
            else:
                _local_dists.append(np.linalg.norm(np.nanmean(_local_zxy, axis=0) - _zxy))
    
    return np.array(_local_dists)

# accumulative prob.
def _cum_prob(data, target_value, vmin=-np.inf, vmax=np.inf):
    """Function to calculate CDF from a dataset"""
    data = np.array(data, dtype=np.float)
    data = data[np.isnan(data)==False]
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
    cprob[cprob <= 0] = 1. / np.nansum(np.isnan(data)==False)
    cprob[np.isnan(cprob)] = 1. / np.nansum(np.isnan(data)==False)
    cprob[cprob >= 1] = 1. - 1. / np.nansum(np.isnan(data)==False)
    cprob = cprob.reshape(target_shape)
    return cprob

# generate spot score pool 
def generate_spot_score_pool(all_spots, distance_zxy=_distance_zxy,
                             local_size=5, verbose=False):
    """Generate pool for spot_scores
    Inputs:
        all_spots: list of spots, or np.2drray, or list of list of spots
        distane_zxy: distance in nm for z,x,y pixels, array of 3 (defualt:[200,106,106])
        local_size: window size of local distance calculation, int (default:5)
        verbose: say something!, bool (default:False)
    """
    if isinstance(all_spots, np.ndarray):
        _zxy = all_spots[:,1:4] * np.array(distance_zxy)[np.newaxis,:]
        _intensities = all_spots[:,0]
    elif isinstance(all_spots[0], np.ndarray) or len(all_spots[0].shape)==1:
        _spots = np.concatenate(all_spots)
        _zxy =  _spots[:,1:4] * np.array(distance_zxy)[np.newaxis,:]
        _intensities = _spots[:, 0]
    elif  isinstance(all_spots[0], list) or len(all_spots[0].shape)==2:
        _spots = np.concatenate([np.array(_pts) for _pts in all_spots], axis=0)
        _zxy =  np.array(_spots)[:,1:4] * np.array(distance_zxy)[np.newaxis,:]
        _intensities = np.array(_spots)[:,0]
    else:
        raise TypeError("Wrong input datatype for all_spots, should be list of spots or list of list of spots!")
    _chr_center = np.nanmean(_zxy, axis=0)
    _cc_dists = np.linalg.norm(_zxy - _chr_center, axis=1)
    _lc_dists = _local_distance(_zxy, _zxy, np.arange(len(_zxy)))
    # remove bad points
    _cc_dists = _cc_dists[np.isnan(_cc_dists)==False]
    _lc_dists = _lc_dists[np.isnan(_lc_dists) == False]
    _intensities = _intensities[_intensities > 0]

    return _cc_dists, _lc_dists, _intensities


# Pick spots by dynamic-programming
def dynamic_pick_spots(chrom_cand_spots, unique_ids, cand_spot_scores, nb_dists,
                       w_nbdist=1, distance_zxy=_distance_zxy, distance_th=200,
                       return_indices=False, verbose=True):
    """Function to dynamic-programming pick spots
    The idea is to use dynamic progamming to pick spots to get GLOBAL maximum
    for both spot_score (likelihood) and neighboring spot distance (continuity)
    ----------------------------------------------------------------------------
    Inputs:
        chrom_cand_spots: candidate spots for cenrtain chromosome, list of list of spots
        unique_ids: region uid for candidate spots, list/array of ints
        cand_spot_scores: scores for candidate spots corresponding to chrom_cand_spots, list of array of scores
        nb_dists: previous neighboring distance references, could come from different sources, 1darray
        w_nbdist: weight for neighboring distance score, float (default: 1)
        distance_zxy: translate pixel to nm, array of 3 (default: [200,106,106])
        return_indices: whether return indices for picked spots, bool (default: False)
        verbose: say something!, bool (default: True)
    Outputs:
        _sel_spots: list of selected spots, list
    optional outputs:
        _sel_indices: list of indices for picked spots, list of ints        
    """
    from scipy.spatial.distance import cdist
    # extract zxy coordiates
    unique_ids = list(np.array(unique_ids, dtype=np.int))
    _zxy_list = [np.array(_spots)[:, 1:4]*np.array(distance_zxy)[np.newaxis, :]
                 for _spots in chrom_cand_spots if len(_spots) > 0]
    _ids = [_id for _id, _spots in zip(
        unique_ids, chrom_cand_spots) if len(_spots) > 0]
    # initialize dynamic score and pointers
    _dy_scores = [_scores for _scores, _spots in zip(
        cand_spot_scores, chrom_cand_spots) if len(_spots) > 0]
    _dy_pointers = [-np.ones(len(_spots), dtype=np.int)
                    for _spots in chrom_cand_spots if len(_spots) > 0]

    # if there are any spots:
    if len(_dy_scores) > 0:
        # forward
        for _i, (_zxys, _id) in enumerate(zip(_zxy_list[1:], _ids[1:])):
            # notice: i is actually 1 smaller than real indices
            # calculate min_distance and give score
            # real pair-wise distances
            _dists = cdist(_zxy_list[_i], _zxy_list[_i+1])
            # add distance score, which is normalized by how far exactly these two regions are
            _measure = distance_score_in_chromosome(_dists, _nb_dists=nb_dists, 
                w_dist=w_nbdist, distance_limits=distance_th ) / (_ids[_i+1] - _ids[_i])
            _measure += _dy_scores[_i][:,np.newaxis]  # get previous maximum

            # update maximum values and pointers
            _dy_scores[_i+1] += np.max(_measure, axis=0)  # update maximum
            _dy_pointers[_i+1] = np.argmax(_measure, axis=0)  # update pointer

        # backward
        _dy_indices = [np.argmax(_dy_scores[-1])]
        _dy_spots = [
            chrom_cand_spots[unique_ids.index(_ids[-1])][_dy_indices[-1]]]
        for _id, _pointers in zip(_ids[:-1][::-1], _dy_pointers[1:][::-1]):
            _dy_indices.append(_pointers[_dy_indices[-1]])
            _dy_spots.append(
                chrom_cand_spots[unique_ids.index(_id)][_dy_indices[-1]])
        # inverse _sel_indices and _sel_spots
        _dy_indices.reverse()
        _dy_spots.reverse()

    _sel_spots, _sel_indices = [], []
    for _uid in unique_ids:
        if _uid in _ids:
            _sel_spots.append(_dy_spots[_ids.index(_uid)])
            _sel_indices.append(_dy_indices[_ids.index(_uid)])
        else:
            if len(_dy_spots) > 0:
                _bad_pt = np.nan*np.ones(len(_dy_spots[-1]))
                _bad_pt[0] = 0 # set bad_pt intensity=0
            else:
                _bad_pt = np.nan*np.ones(11)
                _bad_pt[0] = 0 # set bad_pt intensity=0
            _sel_spots.append(_bad_pt)
            _sel_indices.append(-1)
    if return_indices:
        return np.array(_sel_spots), np.array(_sel_indices, dtype=np.int)
    else:
        return np.array(_sel_spots)

# Pick spots by EM algorithm
def EM_pick_spots(chrom_cand_spots, unique_ids, _chrom_coord=None,
                  num_iters=np.inf, terminate_th=0.002, intensity_th=1,
                  distance_zxy=_distance_zxy, local_size=5, spot_num_th=200,
                  w_ccdist=1, w_lcdist=0.1, w_int=1, w_nbdist=3,
                  check_spots=True, check_th=-3., check_percentile=10., 
                  distance_th=200, ignore_nan=True, make_plot=False, 
                  save_plot=False, save_path=None, save_filename='',
                  return_indices=False, return_scores=False, 
                  return_other_scores=False, verbose=True):
    """Function to achieve EM spot picking
    -------------------------------------------------------------------------------------
    E-step: calculate spot score based on:
        distance to chromosome center (consistency): w_ctdist
        distance to local center (smoothing): w_lcdist
        intensity (spot confidence): w_int
    M-step: pick spots from candidate spots to maximize spot score + neighboring distances
        distance to neighbor (continuity): w_nbdist
    Iterate till:
        a. iteration exceed num_iters
        b. picked point change percentage lower than terminate_th
        c. current result is stable and oscilliating around miminum
    -------------------------------------------------------------------------------------
    Inputs:
        chrom_cand_spots: candidate spots for cenrtain chromosome, list of list of spots
        unique_ids: region uid for candidate spots, list/array of ints
        _chrom_coord: specify 3d chromosome coordinate (in pixel) for reference,
            * otherwise chrom_center will be generated by sel_spots
        num_iters: maximum allowed number of iterations, int (default: np.inf, i.e. no limit)
        terminate_th: termination threshold for change percentage of spot-picking, float (default: 0.005)
        intensity_th: threshold for intensity that keep to try EM, float (default: 1)
            * threshold=1 means SNR=1, which is a pretty generous threshold
        distance_zxy: translate pixel to nm, array of 3 (default: [200,106,106])
        local_size: size to calculate local distance, int (default: 5)
        spot_num_th: minimum number of spots needed for calculate_spot_score, int (default:200)
        w_ccdist: weight for distance_to_chromosome_center, float (default: 1)
        w_lcdist: weight for distance_to_local_center, float (default: 1)
        w_int: weight for spot intensity, float (default: 2)
        w_nbdist:  weight for distance_to_neighbor_region, float (default: 1)
        check_spots: whether apply stringency check for selected spots, bool (default: True)
        check_th: the relative threshold for stringency check, 
            * which will multiply the sum of all weights to estimate threshold, bool (default: -3)
        check_percentile: another percentile threshold that may apply to data, float (default: 1.)
        make_plot: make plot for each iteration, bool (default: False)
        return_indices: whether return indices for picked spots, bool (default: False)
        return_scores: whether return scores for picked spots, bool (default: False)
        return_other_scores: whether return Other scores for cand_spots, bool (default: False)
        verbose: say something!, bool (default: True)
    Outputs:
        _sel_spots: list of selected spots, list
    optional outputs:
        _sel_indices: list of indices for picked spots, list of ints
    """
    ## check inputs
    # check candidate spot and unique id length
    if len(chrom_cand_spots) != len(unique_ids):
        raise ValueError(f"Length of chrom_cand_spots should match unique_ids, \
            while {len(chrom_cand_spots)} and {len(unique_ids)} received!")
    unique_ids = np.array(unique_ids, dtype=np.int)
    # check termination flags
    if num_iters == np.inf and terminate_th < 0:
        raise ValueError(f"At least one valid termination flag required!")
    # check other inputs
    local_size = int(local_size)

    ## initialize
    if verbose:
        print(f"- EM picking spots for {len(unique_ids)} regions.")
    # filter spots
    if verbose:
        print(f"-- filtering spots by intensity threshold = {intensity_th}.")
    for _i, _spots in enumerate(chrom_cand_spots):
        chrom_cand_spots[_i] = _spots[np.array(_spots)[:, 0] > intensity_th]
    # select spots by naive
    if verbose:
        print(f"-- initialize EM by naively picking spots!")
        
    _sel_spots, _sel_indices = naive_pick_spots(chrom_cand_spots, unique_ids,
                                                use_chrom_coord=False, return_indices=True)
    # make plot for initialized
    if make_plot:
        from scipy.spatial.distance import pdist, squareform
        _distmap_list = []
        _distmap = squareform(pdist(_sel_spots[:,1:4] * distance_zxy[np.newaxis,:]))
        _distmap[_distmap == np.inf] = np.nan
        _distmap_list.append(_distmap)

    # initialize flags to finish EM
    _iter = 0  # a counter for iteration
    _change_ratio = 1  # keep record of how much picked-points are changed
    _previous_ratios = []
    ## get into EM loops if
    # not exceeding num_iters and
    # picked point change percentage lower than terminate_th
    while(_iter < num_iters and _change_ratio >= terminate_th):
        if verbose:
            print(f"-- EM iter:{_iter}")

        ## E-step
        # distributions for spot-score
        _estart = time.time()
        if len(_sel_spots) < spot_num_th:
            _cc_dists, _lc_dists, _intensities = generate_spot_score_pool(chrom_cand_spots, distance_zxy=distance_zxy,
                                                                          local_size=local_size, verbose=verbose)
        else:
            _cc_dists, _lc_dists, _intensities = generate_spot_score_pool(_sel_spots, distance_zxy=distance_zxy,
                                                                          local_size=local_size, verbose=verbose)
        # distribution for neighbor distance
        _nb_dists = generate_distance_score_pool(_sel_spots)
        if verbose:
            print(f"--- E time: {np.round(time.time()-_estart, 4)} s,")

        ## M-step
        _mstart = time.time()
        # calcualte spot score
        _spot_scores = [old_spot_score_in_chromosome(_spots, _uid-1, _sel_spots, _chrom_coord, 
                        _cc_dists=_cc_dists, _lc_dists=_lc_dists, _intensities=_intensities, 
                        distance_zxy=distance_zxy, local_size=local_size, 
                        w_ccdist=w_ccdist, w_lcdist=w_lcdist, w_int=w_int, 
                        ignore_nan=ignore_nan) for _spots, _uid in zip(chrom_cand_spots, unique_ids)]
        # special modification for 
        _spot_scores[-1] += old_spot_score_in_chromosome(chrom_cand_spots[-1], len(unique_ids)-1, _sel_spots,
                                                     _chrom_coord,
                                                     _cc_dists=_cc_dists, _lc_dists=_lc_dists,
                                                     _intensities=_intensities, distance_zxy=distance_zxy,
                                                     local_size=local_size, w_ccdist=2, w_lcdist=0,
                                                     w_int=0, ignore_nan=ignore_nan)
        # pick spots by dynamic programming
        _sel_spots, _new_indices = dynamic_pick_spots(chrom_cand_spots, unique_ids, _spot_scores, _nb_dists,
                                                      w_nbdist=w_nbdist, distance_zxy=distance_zxy, 
                                                      distance_th=distance_th,
                                                      return_indices=True, verbose=verbose)
        if verbose:
            print(f"--- M time: {np.round(time.time()-_mstart, 4)} s.")
        # make plot for initialized
        if make_plot:
            _distmap = squareform(pdist(_sel_spots[:,1:4] * distance_zxy[np.newaxis,:] ) )
            _distmap[_distmap == np.inf] = np.nan
            _distmap_list.append(_distmap)
        # update exit checking flags
        _iter += 1
        _change_ratio = sum(np.array(_new_indices, dtype=np.int) -
                            np.array(_sel_indices, dtype=np.int) != 0) / len(_sel_indices)
        _previous_ratios.append(_change_ratio)
        if verbose:
            print(f"--- change_ratio: {_change_ratio}")
        # update sel_indices
        _sel_indices = _new_indices

        # special exit for long term oscillation around minimum
        if len(_previous_ratios) > 5 and np.mean(_previous_ratios[-5:]) <= 2 * terminate_th:
            if verbose:
                print("- exit loop because of long oscillation around minimum.")
            break
    ## check spots if specified
    if check_spots:
        from scipy.stats import scoreatpercentile
        # weight for intensity now +1
        _sel_scores = old_spot_score_in_chromosome(_sel_spots, 
                            np.array(unique_ids, dtype=np.int)-min(unique_ids), 
                            _sel_spots, _chrom_coord, _cc_dists=_cc_dists, 
                            _lc_dists=_lc_dists, _intensities=_intensities,
                            distance_zxy=distance_zxy, local_size=local_size, 
                            w_ccdist=w_ccdist, w_lcdist=w_lcdist,
            w_int=w_int+1, ignore_nan=ignore_nan)
        _other_scores = []
        for _scs, _sel_i in zip(_spot_scores, _sel_indices):
            _other_cs = list(_scs)
            if len(_other_cs) > 0:
                _other_cs.pop(_sel_i)
                _other_scores += list(_other_cs)

        _th_sel = scoreatpercentile(_sel_scores, check_percentile)
        _th_other = scoreatpercentile(_other_scores, 100-check_percentile)
        _th_weight = check_th * (w_ccdist + w_lcdist + w_int + 1)
        if check_percentile > 0 and check_percentile < 100:
            _final_check_th = max(_th_sel, _th_other, _th_weight)
        else:
            _final_check_th = _th_weight
        if verbose:
            print(f"-- applying stringency cehck for spots, theshold={_final_check_th}")
        # remove bad spots
        if np.sum(_sel_scores < _final_check_th) > 0:
            _inds = np.where(_sel_scores < _final_check_th)[0]
            for _i in _inds:
                _sel_spots[_i] = np.nan
                _sel_spots[_i,0] = 0.
            if verbose:
                print(f"--- {len(_inds)} spots didn't pass stringent quality check.")
    ## make plot
    if make_plot:
        _num_im = len(_distmap_list)
        _plot_limits = [0,2000]
        _font_size = 14
        _dpi = 300
        _single_im_size = 5
        _fig,_axes = plt.subplots(1, _num_im, figsize=(_single_im_size*_num_im, _single_im_size*1.2), dpi=_dpi)
        _fig.subplots_adjust(left=0.02, bottom=0, right=0.98, top=1, wspace=0.08, hspace=0)
        for _i, (ax,_distmap) in enumerate(zip(_axes.ravel(), _distmap_list)):
            # plot
            im = ax.imshow(_distmap, interpolation='nearest',  cmap='seismic_r', 
                           vmin=min(_plot_limits), vmax=max(_plot_limits))
            ax.tick_params(left=False, labelsize=_font_size, length=2)
            ax.yaxis.set_ticklabels([])
            # title
            if _i == 0:
                ax.set_title('Initialized by naive', fontsize=_font_size+2)
            else:
                ax.set_title(f"EM iter:{_i-1}", fontsize=_font_size+2)
            # add colorbar
            cb = plt.colorbar(im, ax=ax, ticks=np.arange(0,2200,200), shrink=0.6)
            cb.ax.tick_params(labelsize=_font_size, width=0.6, length=1)
        # save filename
        if save_plot and save_path is not None:
            if not os.path.exists(save_path):
                if verbose:
                    print(f"-- create folder for image: {save_path}")
                os.makedirs(save_path)
            if save_filename == '':
                save_filename = 'EM_iterations.png'
            else:
                save_filename = 'EM_iterations_'+save_filename
                if '.png' not in save_filename:
                    save_filename += '.png'
            _plot_filename = os.path.join(save_path, save_filename)
            if verbose:
                print(f"-- saving image to file: {_plot_filename}")
            _fig.savefig(_plot_filename, transparent=True)
        elif save_plot:
            print("Save path for plot is not given, skip!")
        # plot show if only in main stream
        if __name__ == '__main__':
            plt.show()

    # Return!
    # case 1: simple return selected spots
    if not return_indices and not return_scores and not return_other_scores:
        return np.array(_sel_spots)
    # return spots combined with other info
    else:
        _return_args = (np.array(_sel_spots),)
        if return_indices:
            _return_args += (np.array(_sel_indices, dtype=np.int),)
        if return_scores:
            # if not check_spots, generate a new score set
            _cc_dists, _lc_dists, _intensities = generate_spot_score_pool(_sel_spots, distance_zxy=distance_zxy,
                                                                        local_size=local_size, verbose=verbose)
            _sel_scores = old_spot_score_in_chromosome(_sel_spots, 
                                np.array(unique_ids, dtype=np.int)-min(unique_ids), 
                                _sel_spots, _chrom_coord, _cc_dists=_cc_dists, 
                                _lc_dists=_lc_dists, _intensities=_intensities,
                                distance_zxy=distance_zxy, local_size=local_size, 
                                w_ccdist=w_ccdist, w_lcdist=w_lcdist,
                                w_int=w_int+1)
            _sel_scores = np.array(_sel_scores)
            if ignore_nan:
                _sel_scores = _sel_scores[np.isnan(_sel_scores) == False]
            _return_args += (_sel_scores,)

        if return_other_scores:
            _other_scores = []
            for _scs, _sel_i in zip(_spot_scores, _sel_indices):
                _other_cs = list(_scs)
                if len(_other_cs) > 0:
                    _other_cs.pop(_sel_i)
                    _other_scores += list(_other_cs)
            _other_scores = np.array(_other_scores)
            if ignore_nan:
                _other_scores = _other_scores[np.isnan(_other_scores)==False]
            _return_args += (_other_scores,)
        # return!
        return _return_args


# check and remove replicated spots
def merge_spot_list(spot_list, dist_th=0.1, dist_norm=2, 
                    intensity_th=0., hard_intensity_th=True, 
                    append_nan_spots=False, chrom_coords=None):
    """Function to check replicated spots in a spot list
        - if spots seen in multiple list, assign it to the closest chrom_coord
        - return a corresponding list with boolean type ndarray
    Inputs:
        spot_list: list of spots, list of np.ndarray
        dist_th: threshold for distances considered as same spot, float (default: 0.1 pixel)
        dist_norm: norm for calculating distances, float (default: 2, Euclidean)
        intensity_th: threshold for intensity, float (default: 0, no threshold)
        append_nan_spots: whether append bad spots into merged spots to keep consistency, bool (default: True)
        chrom_coords: center of picked chromosomes, required if allowing append_nan_spots, np.ndarray or None
    Outputs:
        _kept_spots: list of spots that kept, np.ndarray
    """
    # check inputs
    if append_nan_spots and chrom_coords is None:
        raise ValueError(f"chrom_coords should be given if append_nan_spots is True")
    # acquire all candidate spots by loop through spot_list
    _cand_spots = []
    for _i, _spots in enumerate(spot_list):
        # case 1: spots is empty
        if len(_spots) == 0:
            if append_nan_spots:
                if chrom_coords is None:
                    raise ValueError(f"chrom_coords should be given if allowing append_nan_spots")
                _spot_obj_len = [np.array(_s).shape[1] for _s in spot_list if len(_s)>0]
                if len(np.unique(_spot_obj_len)) == 1:
                    _spot_obj_len = np.unique(_spot_obj_len)[0]
                elif len(np.unique(_spot_obj_len)) == 0:
                    _spot_obj_len = 11
                else:
                    raise ValueError(f"_spot object length is not unique, exit")
                _bad_spot = np.ones(_spot_obj_len) * np.nan
                _bad_spot[0] = 0 # set intensity
                _bad_spot[1:4] = np.array(chrom_coords[_i])
                _spots = _bad_spot[np.newaxis, :]
            else:
                continue                 
        # case 2: there are actually spots
        else:
            _spots = np.array(_spots)
        # append
        _cand_spots += list(_spots)
    # convert datatype into numpy
    _cand_spots = np.array(_cand_spots) # candidate spots
    if len(_cand_spots) == 0:
        _kept_spots = np.array([])
        _spot_obj_len=11
    else:
        _kept_flag = np.ones(len(_cand_spots), dtype=np.bool) # flag to decide keep spots
        # exclude if spot didn't pass intensity threshold
        if intensity_th is not None:
            _ints = _cand_spots[:,0]
            if hard_intensity_th:
                _kept_flag *= (_ints >= intensity_th)
            else:
                _int_inds = np.argsort(_ints)[-max(len(spot_list), sum(_ints>=intensity_th)):]
                for _i in range(len(_kept_flag)):
                    if _i not in _int_inds:
                        _kept_flag[_i] = False
        if append_nan_spots:
            _spot_chrom_flag = -1 * np.ones(len(_cand_spots), dtype=np.int) # match spot to chromosomes
            
        for _i, (_spot, _flg) in enumerate(zip(_cand_spots, _kept_flag)):
            # if currently this spot is kept:
            if _flg:
                _coord = _spot[1:4]
                _dists = np.linalg.norm(_cand_spots[:,1:4]-_coord, axis=1, ord=dist_norm)
                _dist_flag = (_dists < dist_th) # pick close spots
                _dist_flag[_i] = False # exclude itself
                #_dist_flag *= _kept_flag # intersect with kept_flag
                _kept_flag[_dist_flag] = False # remove these spots
                
                if append_nan_spots:
                    _chrom_dists = np.linalg.norm(chrom_coords-_coord, axis=1)
                    _spot_chrom_flag[_i] = np.argmin(_chrom_dists)
            # if this spot already being dumped, skip
            else:
                continue
        # extract kept spots
        _kept_spots = list(_cand_spots[_kept_flag])
    
    # append nan spots if specified.
    if append_nan_spots:
        for _chrid, _chrom_coord in enumerate(chrom_coords):
            if _chrid not in _spot_chrom_flag:
                if len(_cand_spots)== 0:
                    _spot_obj_len = 11
                else:
                    _spot_obj_len = [np.array(_s).shape[1] for _s in spot_list if len(_s)>0]
                if len(np.unique(_spot_obj_len)) == 1:
                    _spot_obj_len = np.unique(_spot_obj_len)[0]
                elif len(np.unique(_spot_obj_len)) == 0:
                    _spot_obj_len = 11
                else:
                    raise ValueError(f"_spot object length is not unique, exit")
                _bad_spot = np.ones(_spot_obj_len) * np.nan
                _bad_spot[0] = 0 # set intensity
                _bad_spot[1:4] = np.array(_chrom_coord)
                _kept_spots.append(_bad_spot)
                
    return np.array(_kept_spots)

def assign_spots_to_chromosomes(spots, chrom_coords, distance_zxy=_distance_zxy, dist_norm=2):
    """Function to assign spots to nearest chromosomes
    Inputs:
        spots: list of spots, list of np.ndarray or np.ndarray
        chrom_coords: center coordinate in pixel for all chromosomes within this cell, lst of np.array
        distance_zxy: pixel size in nm for z,x,y axies, np.array (default: [200,106,106] for STORM6)
        dist_norm: norm for calculating distances, float (default: 2, Euclidean)
    Output:
        _spot_list: list of spots that assigned to corresponding chrom_coords
    """
    # input
    _chrom_zxys = np.array(chrom_coords) * np.array(distance_zxy)

    _spots = np.array(spots)
    if len(_spots) == 0:
        return [[] for _i, _chrom_zxy in enumerate(_chrom_zxys)]
    _zxys = _spots[:,1:4] * np.array(distance_zxy) 
     
    
    # calculate distance
    from scipy.spatial.distance import cdist
    _dists = cdist(_zxys, _chrom_zxys) # distance from spots to chromosome centers
    _assign_flags = np.argmin(_dists, axis=1) # which chromosomes to be assigned
    
    # assign spots
    _spot_list = [_spots[np.where(_assign_flags==_i)] for _i, _chrom_zxy in enumerate(_chrom_zxys)]
    
    return _spot_list



# Pick spots for multiple chromosomes by intensity
def naive_pick_spots_for_chromosomes(cell_cand_spots, region_ids, chrom_coords=None, 
                                     intensity_th=0., hard_intensity_th=True, 
                                     chrom_share_spots=False, distance_zxy=_distance_zxy, 
                                     return_indices=False, verbose=True):
    """Function to simply pick spots based on intensity
    by default: 
    - there is a hard spot intensity limit, default limit is 0. 
    - every spot only belongs to one chromosome
    Inputs:
        cell_cand_spots: all candidate spots required by multi-fitting for a given cell, 
            list of list of spots-ndarray
        region_ids: region uid for candidate spots, list/array of ints
        chrom_coords: chromosome coordinates in 3D, list of np.ndarray of 3
        intensity_th: threshold for spot intensity, float (default: 0.)
        hard_intensity_th: whether remove spot even its the brightest in this round, bool (default: True)
        chrom_share_spots: whether different chromosomes share spots, bool (default: False)
        distance_zxy: translate pixel to nm, array of 3 (default: [200,106,106])
        return_indices: whether return indices for picked spots, bool (default: False)
        verbose: say something!, bool (default: True)
    Outputs:
        _sel_spot_list: list of selected spots, list of np.ndarray
    optional outputs:
        _sel_ind_list: list of indices for picked spots, list of np.1d-array with ints
    """
    ## check inputs
    # merge spots together
    if chrom_coords is not None:
        _merged_spot_list = [merge_spot_list(_spot_list, intensity_th=intensity_th,
                append_nan_spots=True, chrom_coords=chrom_coords, hard_intensity_th=hard_intensity_th,
            ) for _spot_list in cell_cand_spots]
    else:
        _merged_spot_list = [merge_spot_list(_spot_list, 
                             intensity_th=intensity_th, hard_intensity_th=hard_intensity_th)
                             for _spot_list in cell_cand_spots]
    # localize region_ids
    _region_ids = np.array(region_ids, dtype=np.int)
    
    # number of chromosomes
    if chrom_coords is not None:
        _num_chroms = len(chrom_coords)
    else:
        _num_chroms = len(cell_cand_spots[0])
    if _num_chroms == 0:
        if return_indices:
            return [], []
        else:
            return []
    # length of spot object, to add bad spots
    _spot_obj_len = [np.array(_s[0]).shape[1] for _s in cell_cand_spots if len(_s) >0 and len(_s[0])>0]
    if len(_spot_obj_len) == 0:
        _spot_obj_len = 11
    elif len(np.unique(_spot_obj_len)) == 1:
        _spot_obj_len = np.unique(_spot_obj_len)[0]
    else:
        raise ValueError(f"_spot object length is not unique, exit")
    _bad_spot = np.ones(_spot_obj_len) * np.nan
    _bad_spot[0] = 0 # set intensity
    # case 1, no chromosome coordnates specifed or share spots
    if chrom_coords is None or chrom_share_spots:
        _sel_spot_list = [naive_pick_spots(cell_cand_spots, _region_ids, 
                                  use_chrom_coord=True, chrom_id=_i) 
                 for _i in range(_num_chroms)]
        _sel_spot_inds = [-1 * np.ones(len(_merged_spot_list), dtype=np.int)
                          for _i in range(_num_chroms)]
    else:
        _sel_spot_list = [[] for _i in range(_num_chroms)]
        _sel_spot_inds = [[] for _i in range(_num_chroms)]
        for _spots in _merged_spot_list:
            if len(_spots) == 0:
                for _chrom_id in range(_num_chroms):
                    _sel_spot_list[_chrom_id].append(_bad_spot.copy())
                    _sel_spot_inds[_chrom_id].append(-1)
            else:
                _coords = _spots[:,1:4] * distance_zxy[np.newaxis,:]
                # assign spots to chromosomes first
                _spot_chrom_flags = -1 * np.ones(len(_coords), dtype=np.int)
                for _i, _coord in enumerate(_coords):
                    _chrom_dists = np.linalg.norm(chrom_coords*distance_zxy[np.newaxis,:]\
                                                  -_coord, axis=1)
                    _spot_chrom_flags[_i] = np.argmin(_chrom_dists)
                # select spot for each chromosome
                for _chrom_id in range(_num_chroms):
                    _inds = np.where(_spot_chrom_flags==_chrom_id)[0]
                    # if no spots assigned to this chr, add bad spot
                    if len(_inds) == 0:
                        _sel_spot_list[_chrom_id].append(_bad_spot.copy())
                        _sel_spot_inds[_chrom_id].append(-1)
                    else:
                        _chrom_spots = _spots[_inds]
                        _chrom_ind = np.argmax(_chrom_spots[:,0])
                        _sel_spot_list[_chrom_id].append(_chrom_spots[_chrom_ind])
                        _sel_spot_inds[_chrom_id].append(_chrom_ind)
    
    # convert to numpy array
    _sel_spot_list = [np.array(_spots) for _spots in _sel_spot_list]
    _sel_spot_inds = [np.array(_inds) for _inds in _sel_spot_inds]
    
    
    # return
    if return_indices:
        return _sel_spot_list, _sel_spot_inds
    else:
        return _sel_spot_list

# Pick spots for multiple chromosomes by dynamic-programming
def dynamic_pick_spots_for_chromosomes(cell_cand_spots, region_ids, 
                                       chrom_coords=None, sel_spot_list=None, 
                                       ref_spot_list=None, ref_spot_ids=None, 
                                       ref_dist_metric='median', nb_dist_list=None, spot_num_th=100,
                                       intensity_th=0., hard_intensity_th=True, 
                                       ignore_region_ids=False, score_metric='linear',
                                       local_size=5, w_ctdist=2, w_lcdist=1, w_int=1, w_nbdist=2,
                                       ignore_nan=True, nan_mask=0., inf_mask=-1000., 
                                       update_chrom_coords=False, chrom_share_spots=False,
                                       distance_zxy=_distance_zxy, distance_limits=[200,3000],
                                       return_indices=False, verbose=True):
    """Function to dynamic-programming pick spots
    The idea is to use dynamic progamming to pick spots to get GLOBAL maximum
    for both spot_score (likelihood) and neighboring spot distance (continuity)
    This version will optimize multiple chromosomes at the same time to get maxima for the cell
    ----------------------------------------------------------------------------
    Inputs:
        cell_cand_spots: all candidate spots required by multi-fitting for a given cell, 
            list of list of spots-ndarray
        region_ids: region uid for candidate spots, list/array of ints
        chrom_coords: chromosome coordinates in 3D, list of np.ndarray of 3
        sel_spot_list: list of previous selected spots, list of np.ndarray or list of list of np.1d-array (default: None, which genererated by naive_pick_spots_for_chromosomes)
        ref_spot_list: list of reference spots for chromosomes, list of np.ndarray or list of list of np.1d-array (default: None, which means sel_spot_list)
        ref_spot_ids: list of reference spots for chromosomes, list/array of ints (default: None, which is directly match ref_spot_list indices)
        ref_dist_metric
        spot_num_th: threshold of number of spots considered as reference if ref_spot_list is not specified, int (default: 100)
        intensity_th: threshold for spot intensity, float (default: 0.)
        hard_intensity_th: whether remove spot even its the brightest in this round, bool (default: True) 
        nb_dist_list: neighboring distances within their own chromosomes, list of np.ndarray (default: None)
        local_size: window size to calculate local-distances, int (default: 5)
        w_ctdist: weight for distance_to_chromosome_center, float (default: 1)
        w_lcdist: weight for distance_to_local_center, float (default: 1)
        w_int: weight for spot intensity, float (default: 1)
        w_nbdist:  weight for distance_to_neighbor_region, float (default: 3)
        ignore_nan: whether ignore np.nan in calculating scores, bool (default: True)
        update_chrom_coords: whether update chromosome coordinates during EM, bool (default: False)
        chrom_share_spots: whether different chromosomes share spots, bool (default: False)
        distance_zxy: translate pixel to nm, array of 3 (default: [200,106,106])
        distance_limits: threshold for distance score CDF calculation, either None, int, list of 2
        return_indices: whether return indices for picked spots, bool (default: False)
        verbose: say something!, bool (default: True)
    Outputs:
        _sel_spot_list: list of selected spots, list of np.ndarray
    optional outputs:
        _sel_ind_list: list of indices for picked spots, list of np.1d-array with ints
    """
    from scipy.spatial.distance import cdist
    # merge spots together
    if chrom_coords is not None:
        _merged_spot_list = [merge_spot_list(_spot_list, intensity_th=intensity_th,
                append_nan_spots=True, chrom_coords=chrom_coords, hard_intensity_th=hard_intensity_th,
            ) for _spot_list in cell_cand_spots]
    else:
        _merged_spot_list = [merge_spot_list(_spot_list, 
                             intensity_th=intensity_th, hard_intensity_th=hard_intensity_th)
                             for _spot_list in cell_cand_spots]
    # sort by region_ids   
    _merged_spot_list = [_merged_spot_list[_id] for _id in np.argsort(region_ids)]
    #print(np.array([len(_s) for _s in _merged_spot_list]))
    _sorted_region_ids = [int(region_ids[_id]) for _id in np.argsort(region_ids)]

    # number of chromosomes
    if chrom_coords is not None:
        _num_chroms = len(chrom_coords)
    else:
        _num_chroms = len(cell_cand_spots[0])
    
    if _num_chroms == 0:
        if return_indices:
            return [], []
        else:
            return []

    # select spots if not given
    if sel_spot_list is None:
        if verbose:
            print(f"-- initiate dynamic picking by naive picking spots.")
        # do naive picking
        sel_spot_list, _sel_ind_list = naive_pick_spots_for_chromosomes(
                                        cell_cand_spots, region_ids, chrom_coords=chrom_coords,
                                        intensity_th=intensity_th, hard_intensity_th=hard_intensity_th,
                                        chrom_share_spots=chrom_share_spots, distance_zxy=distance_zxy, 
                                        return_indices=True, verbose=verbose)
        # sort sel_spot_list:
        for _chrom_id in range(_num_chroms):
            sel_spot_list[_chrom_id] = np.array([sel_spot_list[_chrom_id][_id] for _id in np.argsort(region_ids)])

    # calculate scores for all spots as reference
    spot_score_list = []
    # check nb_dist_list, if None then create a temp list ref_nb_list
    if nb_dist_list is None:
        _ref_nb_list = []
    else:
        if len(nb_dist_list) != sel_spot_list:
            raise IndexError(f"Length of nb_dist_list{len(nb_dist_list)} doesn't match length of sel_spot_list:{len(sel_spot_list)}")
    
    ## decide reference spots and reference id
    ref_id_list = []
    if ref_spot_list is None:
        ref_spot_list = sel_spot_list
    # loop through chromosome and check
    for _chrom_id, _ref_spots in enumerate(ref_spot_list):
        # get chromosome candidate spots
        if len(_ref_spots) >= spot_num_th:
            # no change for spots
            # change for ids
            if ref_spot_ids is None:
                if len(_ref_spots) == len(region_ids):
                    ref_id_list.append(region_ids)
                else:
                    ref_id_list.append(np.arange(len(_ref_spots)))
            else:
                ref_id_list.append(np.array(ref_spot_ids, dtype=np.int))
        else:
            _chrom_cand_spots = [_spot_list[_chrom_id] for _spot_list in cell_cand_spots]
            # change for spots
            ref_spot_list[_chrom_id] = np.concatenate(_chrom_cand_spots)
            # change for ids
            if ref_spot_ids is None:
                ref_id_list.append(np.concatenate([np.ones(len(_spots))*_id for _spots,_id in zip(_chrom_cand_spots, region_ids) ]) )
            elif len(ref_spot_ids) == len(_chrom_cand_spots):
                ref_id_list.append(np.concatenate([np.ones(len(_spots))*_id for _spots,_id in zip(_chrom_cand_spots, ref_spot_ids) ]) )
            else:
                ref_id_list.append(np.concatenate([np.ones(len(_spots))*_id for _id, _spots in enumerate(_chrom_cand_spots) ]) )

    ## loop through chromosomes and selected spots, calculate scores and references
    for _chrom_id, (_ref_spots, _ref_ids) in enumerate(zip(ref_spot_list, ref_id_list)):
        # get chromosome candidate spots
        _chrom_cand_spots = [_spot_list[_chrom_id] for _spot_list in cell_cand_spots]
        # get chrom_coord
        if chrom_coords is not None and not update_chrom_coords:
            _chrom_coord = chrom_coords[_chrom_id]
        else:
            _chrom_coord = None
        if len(_ref_ids) != len(_ref_spots):
            raise IndexError(f"chr:{_chrom_id}, Length of _ref_ids:{len(_ref_ids)} doesn't match length of _ref_spots:{len(_ref_spots)}")
        #print(f"* length of ref spots{len(_ref_spots)}")
        # v2
        _ref_ct_dist, _ref_lc_dist, ref_nb_dist, _ref_ints = scoring.generate_ref_from_chromosome(
            _ref_spots, _ref_ids, distance_zxy, _chrom_coord, intensity_th,
            local_size, ref_dist_metric)
        #print(_ref_ct_dist, _ref_lc_dist, ref_nb_dist, _ref_ints)
        # append nb_dist reference
        if nb_dist_list is None:
            _ref_nb_list.append(ref_nb_dist)
        
        _chrom_scores = [scoring.spot_score_in_chromosome(_spots, _uid, _ref_spots, _ref_ids,
                            _chrom_coord, _ref_ct_dist, _ref_lc_dist, ref_nb_dist, _ref_ints,
                            ref_dist_metric, ignore_nan, nan_mask, inf_mask, 
                            distance_zxy, distance_limits, 
                            score_metric, intensity_th, local_size, w_ctdist, w_lcdist, w_int)
                         for _spots, _uid in zip(_merged_spot_list, _sorted_region_ids)]        
        # append
        spot_score_list.append(_chrom_scores)

    # update nb_dist_list
    nb_dist_list = _ref_nb_list
    
    ## get dynamic zxy coordiantes, ids and indices and spot_scores
    _zxy_list, _ids, _id_indices = [],[],[] # coord, ids, indices
    # Dynamic picking non-overlapping spots
    _dy_score_list = [[] for _chrom_id in range(_num_chroms)]
    _dy_pointer_list = [[] for _chrom_id in range(_num_chroms)]
    # start iteration
    for _i,(_id, _spots) in enumerate(zip(_sorted_region_ids, _merged_spot_list)):
        if len(_spots) == 0:
            continue
        else:
            _valid_flag = (np.isnan(_spots).sum(1) == 0) # no np.nan in spots
            if np.sum(_valid_flag) == 0: # no vaild spots
                continue
            else: # append
                #_valid_spots = np.array(_spots)[_valid_flag]
                _zxy_list.append( _spots[:,1:4]*np.array(distance_zxy)[np.newaxis, :] )
                _ids.append(_id)
                _id_indices.append(_i)
                # append dynamic picked non-overlapping spots
                for _chrom_id in range(_num_chroms):
                    # scores
                    _scores = spot_score_list[_chrom_id][_i]
                    _dy_score_list[_chrom_id].append(_scores)
                    # pointers
                    _pointers = -np.ones(len(_spots), dtype=np.int)
                    _dy_pointer_list[_chrom_id].append(_pointers)
                #if _i == 86:
                #    raise ValueError()
    # result stores at here
    _dy_ind_list, _dy_spot_list = [], []
    ## Start dynamic if there are spots kept
    if len(_zxy_list) > 0 and len(_ids) > 0:
        ## forward
        for _i, (_zxys, _id) in enumerate(zip(_zxy_list[1:], _ids[1:])):
            #print(_i, end=' ')
            # notice: i is actually 1 smaller than real indices
            # calculate min_distance and give score
            # real pair-wise distances
            _dists = cdist(_zxy_list[_i], _zxy_list[_i+1])
            # add distance score, which is normalized by how far exactly these two regions are
            if np.abs(_ids[_i+1] - _ids[_i]) > 0:
                _measure_list = [scoring.distance_score(_dists, ref_dist=_nb_dist, 
                                    weight=w_nbdist, metric=score_metric,
                                    nan_mask=nan_mask, inf_mask=inf_mask, 
                                    distance_limits=distance_limits ) \
                                    / (_ids[_i+1] - _ids[_i]) + _dy_scores[_i][:,np.newaxis]
                                    for _nb_dist, _dy_scores in zip(nb_dist_list, _dy_score_list)]
            else:
                _measure_list = [scoring.distance_score(_dists, ref_dist=_nb_dist, 
                                    weight=w_nbdist, metric=score_metric,
                                    nan_mask=nan_mask, inf_mask=inf_mask, 
                                    distance_limits=distance_limits ) \
                                    + _dy_scores[_i][:,np.newaxis]
                                    for _nb_dist, _dy_scores in zip(nb_dist_list, _dy_score_list)]

            # pick from this measure_list
            # generate shared iteration
            if len(_zxy_list[_i]) <= _num_chroms:
                _inds = _all_score_combinations([_dy_scores[_i] for _dy_scores in _dy_score_list], 
                                          chrom_share_spots=chrom_share_spots)
            else:
                _inds = _optimized_score_combinations([_dy_scores[_i] for _dy_scores in _dy_score_list], 
                                                chrom_share_spots=chrom_share_spots)

            #print(_i, _id, _id_indices[_i], len(_inds), #_zxy_list[_i], 
            #      np.array([_dy_score_list[_cid][_i] for _cid in range(_num_chroms)]))
            
            #raise ValueError
            # enumerate through spots in _i+1 round
            for _nid in range(len(_zxy_list[_i+1])):
                # version 1.0 optimize sum of scores
                _global_scores = [np.nansum([_measure[_ind[_chrom_id], _nid] 
                                       for _chrom_id, _measure in enumerate(_measure_list)])
                                  for _ind in _inds]
                # version 1.1 optimize sum of order
                _measure_order_list = [np.argsort(_measure[:,_nid]) for _measure in _measure_list]
                _global_orders = [[_m[_ind[_chrom_id]] 
                                       for _chrom_id, _m in enumerate(_measure_order_list)]
                                  for _ind in _inds]
                #_global_scores = np.array([sum(_order) for _order in _global_orders])
                _global_maxs = np.array([max(_order) for _order in _global_orders])
                if len(_global_scores) == 0:
                    print(_i, _inds, len(_zxy_list[_i]), len(_zxy_list[_i+1]), )
                    raise ValueError(f"Not enough _inds combinations")
                _sids = np.where(_global_scores==np.nanmax(_global_scores))[0]
                if len(_sids) == 1:
                    _sel_ind = _inds[_sids[0]]
                else:
                    #print(_sids, _global_scores, _global_maxs)
                    _maxid = np.argmax(_global_maxs[_sids])
                    _sel_ind = _inds[_sids[_maxid]]
                # update corresponding _dy_score and pointer based on previous selected ind
                for _chrom_id in range(_num_chroms):
                    # update dy_score and _dy_pointer
                    _dy_score_list[_chrom_id][_i+1][_nid] += _measure_list[_chrom_id][_sel_ind[_chrom_id], _nid]
                    _dy_pointer_list[_chrom_id][_i+1][_nid] = _sel_ind[_chrom_id]
            #print(_dy_pointer_list[0][_i+1], _dy_pointer_list[1][_i+1])
        ## backward
        for _chrom_id, (_dy_scores, _dy_pointers) in enumerate(zip(_dy_score_list, _dy_pointer_list)):
            _dy_indices = [np.argmax(_dy_scores[-1])]
            _dy_spots = [_merged_spot_list[_id_indices[-1]][_dy_indices[-1]]]
            for _reg_index, _id, _pointers in zip(_id_indices[:-1][::-1], _ids[:-1][::-1], _dy_pointers[1:][::-1]):
                _dy_indices.append(_pointers[_dy_indices[-1]])
                _dy_spots.append(_merged_spot_list[_reg_index][_dy_indices[-1]])
            # inverse _sel_indices and _sel_spots
            _dy_indices.reverse() # this is an in-object inverse!
            _dy_spots.reverse()
           # append
            _dy_ind_list.append(_dy_indices)
            _dy_spot_list.append(_dy_spots)
    # append bad spots as well
    _sel_spot_list = [np.zeros([len(_merged_spot_list), np.shape(_merged_spot_list[0])[1]]) for _i in range(_num_chroms)]
    _sel_ind_list = [-1 * np.ones(len(_merged_spot_list), dtype=np.int) for _i in range(_num_chroms)]
    
    for _chrom_id, (_dy_spots, _dy_indices) in enumerate(zip(_dy_spot_list, _dy_ind_list)):        
        # sort as orignial region_ids order
        for _j, _order in enumerate(np.argsort(region_ids)):
            if region_ids[_order] in _ids and _j in _id_indices:
                # if there's any bad spot, just append empty one
                _picked_spot = _dy_spots[np.where(np.array(_id_indices)==_j)[0][0]]
                # this step revoke temp spot coordinates
                if np.isnan(_picked_spot).any():
                    _picked_spot = np.ones(len(_picked_spot)) * np.nan
                    _picked_spot[0] = 0
                _sel_spot_list[_chrom_id][_order] = _picked_spot
                _sel_ind_list[_chrom_id][_order] = _dy_indices[np.where(np.array(_id_indices)==_j)[0][0]]
            else:
                if len(_dy_spots) > 0:
                    _bad_pt = np.nan*np.ones(len(_dy_spots[-1]))
                    _bad_pt[0] = 0 # set bad_pt intensity=0
                else:
                    _bad_pt = np.nan*np.ones(11)
                    _bad_pt[0] = 0 # set bad_pt intensity=0
                _sel_spot_list[_chrom_id][_order] = _bad_pt
                _sel_ind_list[_chrom_id][_order] = -1 

    
    if return_indices:
        return _sel_spot_list, _sel_ind_list
    else:
        return _sel_spot_list

# Pick spots for multiple chromosomes by EM
# Pick spots by EM algorithm
def EM_pick_spots_for_chromosomes(cell_cand_spots, region_ids, 
                                  chrom_coords=None, sel_spot_list=None, 
                                  ref_spot_list=None, ref_spot_ids=None, 
                                  ref_dist_metric='median', nb_dist_list=None, spot_num_th=100,
                                  num_iters=10, terminate_th=0.0025, intensity_th=0.,
                                  hard_intensity_th=True, score_metric='linear',
                                  local_size=5, w_ctdist=2, w_lcdist=1, w_int=1, w_nbdist=2,
                                  distance_limits=[0,3000], ignore_nan=True, 
                                  nan_mask=0., inf_mask=-1000., update_chrom_coords=False, 
                                  chrom_share_spots=False, distance_zxy=_distance_zxy, 
                                  check_spots=True, check_th=-2., check_percentile=10.,hard_dist_th=8000, 
                                  make_plot=False, save_plot=False, save_path=None, save_filename='',
                                  return_indices=False, return_sel_scores=False, return_other_scores=False, 
                                  verbose=True):
    """Function to achieve EM spot picking for multiple chromosomes
    -------------------------------------------------------------------------------------
    E-step: calculate spot score based on:
        distance to chromosome center (consistency): w_ctdist
        distance to local center (smoothing): w_lcdist
        intensity (spot confidence): w_int
    M-step: pick spots from candidate spots to maximize spot score + neighboring distances
        distance to neighbor (continuity): w_nbdist
    Iterate till:
        a. iteration exceed num_iters
        b. picked point change percentage lower than terminate_th
        c. current result is stable and oscilliating around miminum
    -------------------------------------------------------------------------------------
    Inputs:
        cell_cand_spots: all candidate spots required by multi-fitting for a given cell, 
            list of list of spots-ndarray
        region_ids: region uid for candidate spots, list/array of ints
        chrom_coords: chromosome coordinates in 3D, list of np.ndarray of 3
        sel_spot_list: list of previous selected spots, list of np.ndarray or list of list of np.1d-array
        ref_spot_list: list of reference spots for chromosomes, list of np.ndarray or list of list of np.1d-array (default: None, which means sel_spot_list)
        ref_spot_ids: list of reference spots for chromosomes, list/array of ints (default: None, which is directly match ref_spot_list indices)
        ref_dist_metric: metric for generating reference distances, string         
        spot_num_th: threshold of number of spots considered as reference if ref_spot_list is not specified, int (default: 100)
        num_iters: maximum number of iterations allowed in EM, int (default: 10)
        terminate_th: termination threshold for change percentage of spot-picking, float (default: 0.0025)
        intensity_th: threshold for intensity that keep to try EM, float (default: 0.8)
            * threshold=1 means SNR=1, which is a pretty generous threshold
        hard_intensity_th: whether remove spot even its the brightest in this round, bool (default: True)
        nb_dist_list: neighboring distances within their own chromosomes, list of np.ndarray (default: None)
        score_metric: metrhic type for scoring spots, string ({'linear'|'cdf'})
        local_size: size to calculate local distance, int (default: 5)
        w_ctdist: weight for distance_to_chromosome_center, float (default: 1)
        w_lcdist: weight for distance_to_local_center, float (default: 1)
        w_int: weight for spot intensity, float (default: 2)
        w_nbdist:  weight for distance_to_neighbor_region, float (default: 1)
        distance_limits: limit for neighboring limit scoring, None | int,float | list of two
        ignore_nan: whether ignore nan during scoring, bool (default: True)
        update_chrom_coords: whether update chromosome coordinates during EM, bool (default: False)
        chrom_share_spots: whether chromosomes are allowed to share spots, bool (default: False)
        distance_zxy: translate pixel to nm, array of 3 (default: [200,106,106])
        check_spots: whether apply stringency check for selected spots, bool (default: True)
        check_th: the relative threshold for stringency check, 
            * which will multiply the sum of all weights to estimate threshold, bool (default: -2)
        check_percentile: another percentile threshold that may apply to data, float (default: 5.)
        make_plot: make plot for each iteration, bool (default: False)
        return_indices: whether return indices for picked spots, bool (default: False)
        return_sel_scores: whether return scores for picked spots, bool (default: False)
        return_other_scores: whether return Other scores for cand_spots, bool (default: False)
        verbose: say something!, bool (default: True)
    Outputs:
        sel_spot_list: list of selected spots, list of np.ndarrays(spots)
    optional outputs:
        sel_ind_list: list of indices for picked spots, list of array of ints
        sel_score_list: list of scores of picked spots, list of array of floats
        other_score_list: list of scores of spots not picked, list of array of floats
    """
    ## check inputs
    region_ids = np.array(region_ids, dtype=np.int)
    if verbose:
        print(f"- EM picking spots for {len(region_ids)} regions, use chrom_coords:{(chrom_coords is not None)}")
    # check candidate spot and unique id length
    if len(cell_cand_spots) != len(region_ids):
        raise ValueError(f"Length of cell_cand_spots should match region_ids, while {len(cell_cand_spots)} and {len(region_ids)} received!")
    # filter spots
    if verbose:
        print(f"-- filtering spots by intensity threshold={intensity_th}.")
    for _i, _spot_list in enumerate(cell_cand_spots):
        for _j, _spots in enumerate(_spot_list):
            if len(_spots) > 0:
                cell_cand_spots[_i][_j] = _spots[np.array(_spots)[:, 0] >= min(intensity_th, max(np.array(_spots)[:, 0]))]
    
    # merge spots
    if chrom_coords is not None:
        _merged_spot_list = [merge_spot_list(_spot_list, intensity_th=intensity_th,
                                append_nan_spots=True, chrom_coords=chrom_coords, 
                                hard_intensity_th=hard_intensity_th,
                                ) for _spot_list in cell_cand_spots]
    else:
        _merged_spot_list = [merge_spot_list(_spot_list, intensity_th=intensity_th,
                                             hard_intensity_th=hard_intensity_th,)
                             for _spot_list in cell_cand_spots]    
    # number of chromosomes
    if chrom_coords is not None:
        _num_chroms = len(chrom_coords)
    else:
        _num_chroms = len(cell_cand_spots[0])
    
    if _num_chroms == 0:
        if verbose:
            print("-- exit for no-chromosome case.")
        if not return_indices and not return_sel_scores and not return_other_scores:
            return []
        else:
            _return_args = ([],)
            if return_indices:
                _return_args += ([],)
            if return_sel_scores:
                _return_args += ([],)
            if return_other_scores:
                _return_args += ([],)
            # return!
            return _return_args
    else:
        # select chromosome to initiate
        if sel_spot_list is None:
            if verbose:
                print(f"-- initialize EM by naively picking spots!")
            # select spots by naive
            sel_spot_list, sel_ind_list = naive_pick_spots_for_chromosomes(
                cell_cand_spots, region_ids, chrom_coords=chrom_coords,
                intensity_th=intensity_th, hard_intensity_th=hard_intensity_th,
                chrom_share_spots=chrom_share_spots, distance_zxy=distance_zxy, 
                return_indices=True, verbose=verbose) # naive picking doesnt change order
            # make plot for initialized
            if make_plot:
                from scipy.spatial.distance import pdist, squareform
                _distmap_list = [[] for _i in range(_num_chroms)]
                for _chrom_id, _sel_spots in enumerate(sel_spot_list):
                    _distmap = squareform(pdist(_sel_spots[np.argsort(region_ids), 1:4] * distance_zxy[np.newaxis,:]))
                    _distmap[_distmap == np.inf] = np.nan
                    _distmap_list[_chrom_id].append(_distmap)
        # check termination flags
        if num_iters == np.inf and terminate_th < 0:
            raise ValueError(f"At least one valid termination flag required!")
        # check other inputs
        local_size = int(local_size)
        
        ## initialize select_ind for EM
        if 'sel_ind_list' not in locals():
            sel_ind_list = [[] for _chrom_id in range(_num_chroms)]
        # initialize flags to finish EM
        _iter = 0  # a counter for iteration
        _change_ratio = 1  # keep record of how much picked-points are changed
        _previous_ratios = []
        ## get into EM loops if
        # not exceeding num_iters and
        # picked point change percentage lower than terminate_th
        while(_iter < num_iters and _change_ratio >= terminate_th):
            if verbose:
                print(f"--- EM iter:{_iter}", end=', ')
                _step_start = time.time()
            # do dynamic, spot order will be sorted back to original region_ids
            # note: its critical to pass over new_ind_list, which is essentially E-step
            # if there are too many chromosomes, optimize by themselves
            if _num_chroms > 6:
                new_spot_list, new_ind_list = [], []
                for _i in range(_num_chroms):
                    sel_spots, new_inds = dynamic_pick_spots_for_chromosomes(cell_cand_spots,
                                            region_ids, chrom_coords=[chrom_coords[_i]], sel_spot_list=[sel_spot_list[_i]],
                                            ref_spot_list=None, ref_spot_ids=ref_spot_ids,
                                            ref_dist_metric=ref_dist_metric, nb_dist_list=None, spot_num_th=spot_num_th,
                                            intensity_th=intensity_th, hard_intensity_th=hard_intensity_th,
                                            score_metric=score_metric, local_size=local_size, w_ctdist=w_ctdist,
                                            w_lcdist=w_lcdist, w_int=w_int, w_nbdist=w_nbdist,
                                            ignore_nan=ignore_nan, nan_mask=nan_mask, inf_mask=inf_mask,
                                            update_chrom_coords=update_chrom_coords, 
                                            chrom_share_spots=chrom_share_spots,
                                            distance_zxy=distance_zxy, distance_limits=distance_limits,
                                            return_indices=True, verbose=verbose)
                    new_spot_list += sel_spots
                    new_ind_list += new_inds
                sel_spot_list = new_spot_list
            else:
                sel_spot_list, new_ind_list = dynamic_pick_spots_for_chromosomes(cell_cand_spots,
                                                region_ids, chrom_coords=chrom_coords, sel_spot_list=sel_spot_list,
                                                ref_spot_list=ref_spot_list, ref_spot_ids=ref_spot_ids,
                                                ref_dist_metric=ref_dist_metric, nb_dist_list=nb_dist_list, spot_num_th=spot_num_th,
                                                intensity_th=intensity_th, hard_intensity_th=hard_intensity_th,
                                                score_metric=score_metric, local_size=local_size, w_ctdist=w_ctdist,
                                                w_lcdist=w_lcdist, w_int=w_int, w_nbdist=w_nbdist,
                                                ignore_nan=ignore_nan, nan_mask=nan_mask, inf_mask=inf_mask,
                                                update_chrom_coords=update_chrom_coords, 
                                                chrom_share_spots=chrom_share_spots,
                                                distance_zxy=distance_zxy, distance_limits=distance_limits,
                                                return_indices=True, verbose=verbose)

            # make plot for initialized
            if make_plot:
                for _chrom_id, _sel_spots in enumerate(sel_spot_list):
                    _distmap = squareform(pdist(_sel_spots[np.argsort(region_ids), 1:4] * distance_zxy[np.newaxis,:] ) )
                    _distmap[_distmap == np.inf] = np.nan
                    _distmap_list[_chrom_id].append(_distmap)
            # update exit checking flags
            _iter += 1
            _change_num, _total_num = 0, 0
            for _new_indices, _sel_indices in zip(new_ind_list, sel_ind_list):
                # number of changed indices
                if len(_sel_indices) == 0 and len(_new_indices) != 0:
                    _change_num += len(_new_indices)
                else:
                    _change_num += sum(np.array(_new_indices, dtype=np.int) - np.array(_sel_indices, dtype=np.int) != 0)
                # total number of selected points
                _total_num += len(_new_indices)
            _change_ratio = _change_num / _total_num
            _previous_ratios.append(_change_ratio)
            if verbose:
                print(f"time: {time.time()-_step_start:.3f}, change_ratio={_change_ratio}")
            # update sel_indices
            for _i, _new_indices in enumerate(new_ind_list):
                sel_ind_list[_i] = _new_indices
                
            # special exit for long term oscillation around minimum
            if len(_previous_ratios) > 5 and np.mean(_previous_ratios[-5:]) <= 2 * terminate_th:
                if verbose:
                    print("-- exit loop because of long oscillation around minimum.")
                break
        
        # calculate final scores if necesary
        if check_spots or return_sel_scores or return_other_scores:
            sel_score_list, other_score_list = [], []
            # loop through chromosomes
            for _chrom_id, _sel_spots in enumerate(sel_spot_list):
                _sel_indices = sel_ind_list[_i]
                if chrom_coords is not None:
                    _chrom_coord = chrom_coords[_i]
                else:
                    _chrom_coord = None
                _filtered_spots, _sel_scores, _other_scores = checking.check_spot_scores(
                    _merged_spot_list, _sel_spots, region_ids, _sel_indices, 
                    chrom_coord=_chrom_coord, distance_zxy=distance_zxy, distance_limits=distance_limits,
                    intensity_th=intensity_th, ref_dist_metric=ref_dist_metric, score_metric=score_metric,
                    local_size=local_size, w_ctdist=w_ctdist, w_lcdist=w_lcdist,
                    w_int=w_int, ignore_nan=ignore_nan, check_th=check_th, 
                    check_percentile=check_percentile, hard_dist_th=hard_dist_th,
                    return_sel_scores=True, return_other_scores=True, verbose=verbose)
                # append
                sel_score_list.append(np.array(_sel_scores))
                other_score_list.append(np.array(_other_scores))
                # update spots if check_spots specified
                if check_spots:
                    sel_spot_list[_chrom_id] = np.array(_filtered_spots)

            # make plot for checks
            if make_plot:
                for _chrom_id, _sel_spots in enumerate(sel_spot_list):
                    _distmap = squareform(pdist(_sel_spots[np.argsort(region_ids), 1:4] * distance_zxy[np.newaxis,:] ) )
                    _distmap[_distmap == np.inf] = np.nan
                    _distmap_list[_chrom_id].append(_distmap)
                
        ## make plot
        if make_plot:
            _num_im = len(_distmap_list[0])
            _plot_limits = [0,1500]
            _font_size = 14
            _dpi = 100
            _single_im_size = 5
            _fig,_axes = plt.subplots(len(_distmap_list), _num_im, 
                                    figsize=(_single_im_size*_num_im, _single_im_size*1.2*len(_distmap_list)), 
                                    dpi=_dpi)
            _fig.subplots_adjust(left=0.02, bottom=0, right=0.98, top=1, wspace=0., hspace=0)

            for _chrom_id, _distmaps in enumerate(_distmap_list):
                for _im_id, _distmap in enumerate(_distmaps):
                    # get axis
                    if len(_distmap_list) == 1:
                        ax = _axes[_im_id]
                    else:
                        ax = _axes[_chrom_id, _im_id]
                    # plot
                    im = ax.imshow(_distmap, interpolation='nearest',  cmap='seismic_r', 
                                vmin=min(_plot_limits), vmax=max(_plot_limits))
                    ax.tick_params(left=False, labelsize=_font_size, length=2)
                    ax.yaxis.set_ticklabels([])
                    # title
                    if _im_id==0:
                        ax.set_title('Initialized by naive', fontsize=_font_size+2)
                    if _im_id == len(_distmaps)-1 and check_spots:
                        ax.set_title('Final result by EM', fontsize=_font_size+2)
                    else:
                        ax.set_title(f"Chr:{_chrom_id}, EM:{_im_id}", fontsize=_font_size+2)
                    # add colorbar
                    cb = plt.colorbar(im, ax=ax, ticks=np.arange(0,2200,200), shrink=0.6)
                    cb.ax.tick_params(labelsize=_font_size, width=0.6, length=1)

            # save filename
            if save_plot and save_path is not None:
                if not os.path.exists(save_path):
                    if verbose:
                        print(f"-- create folder for image: {save_path}")
                    os.makedirs(save_path)
                if save_filename == '':
                    save_filename = 'EM_iterations.png'
                else:
                    save_filename = 'EM_iterations_'+save_filename
                    if '.png' not in save_filename:
                        save_filename += '.png'
                _plot_filename = os.path.join(save_path, save_filename)
                if verbose:
                    print(f"-- saving image to file: {_plot_filename}")
                _fig.savefig(_plot_filename, transparent=True)
            elif save_plot:
                print("Save path for plot is not given, skip!")
            # plot show if only in main stream
            if __name__ == '__main__':
                plt.show()

        # Return!
        # case 1: simple return selected spots
        if not return_indices and not return_sel_scores and not return_other_scores:
            return sel_spot_list
        # return spots combined with other info
        else:
            _return_args = (sel_spot_list,)
            if return_indices:
                _return_args += (sel_ind_list,)
            if return_sel_scores:
                _return_args += (sel_score_list,)
            if return_other_scores:
                _return_args += (other_score_list,)
            # return!
            return _return_args


# Generate spot score combinations for multiple chromosomes, used in dynamic_pick_spots
def _optimized_score_combinations(_score_list, chrom_share_spots=False):
    from itertools import product
    if len(_score_list) == 0:
        raise ValueError(f"_score_list is empty, exit!")
    if len(_score_list[0]) == 0:
        raise ValueError(f"_score_list[0] is empty, no spots, exit!")
    _num_chroms = len(_score_list)
    _num_spots = len(_score_list[0])
    if _num_spots < _num_chroms:
        raise IndexError(f"there should be more spots than chromosomes!")
    _allowed_ind_list = []
    for _scores in _score_list:
        if np.isinf(_scores).all():
            _allowed_ind_list.append( np.arange(_num_spots) )
        else:
            _allowed_ind_list.append(np.argsort(_scores)[-_num_chroms:])
    _inds = list(product(*_allowed_ind_list)) # this star is critical to expand list into multiple args
    if not chrom_share_spots:
        _inds = [_ind for _ind in _inds if len(set(_ind))==len(_ind)]
    return _inds
    
def _all_score_combinations(_score_list, chrom_share_spots=False):
    from itertools import product
    if len(_score_list) == 0:
        raise ValueError(f"_score_list is empty, exit!")
    if len(_score_list[0]) == 0:
        raise ValueError(f"_score_list[0] is empty, no spots, exit!")
    _num_chroms = len(_score_list)
    _num_spots = len(_score_list[0])
    _inds = list(product(np.arange(_num_spots), repeat=_num_chroms))
    if not chrom_share_spots:
        _inds = [_ind for _ind in _inds if len(set(_ind))==len(_ind)]
    return _inds



def extract_intensities(cand_hzxys):
    _ints = []
    for _hzxys in cand_hzxys:
        if len(_hzxys) == 0:
            _ints.append([])
        elif len(np.shape(np.array(_hzxys))) == 1:
            _ints.append(_hzxys[0])
        elif len(np.shape(np.array(_hzxys))) == 2:
            _ints.append(_hzxys[:,0])
    return _ints

def chromosome_center_dists(cand_hzxys, ref_zxy):
    
    _ct_dists = []
    _ref_zxy = np.array(ref_zxy)[-3:]
    # loop through regions
    for _hzxys in cand_hzxys:
        if len(_hzxys) == 0:
            _ct_dists.append(np.array([np.nan]))
        elif len(np.shape(np.array(_hzxys))) == 1:
            _dist = np.linalg.norm(np.array(_hzxys)[-3:] - _ref_zxy)
            _ct_dists.append(_dist)
        elif len(np.shape(np.array(_hzxys))) == 2:
            _dists = np.linalg.norm(np.array(_hzxys)[:,-3:] - _ref_zxy, axis=1)
            _ct_dists.append(_dists)
        else:
            raise IndexError("Wrong input shape for cand_hzxys:_hzxys")
    
    return _ct_dists


def local_center_dists(cand_hzxys, cand_ids, ref_hzxys, 
                       ref_ids=None, neighbor_len=5):
    """Function to calculate neighboring center"""
    from scipy.spatial.distance import cdist
    # candidate ids
    if cand_ids is None:
        cand_ids = np.arange(len(cand_hzxys))
    if isinstance(cand_ids, int) or isinstance(cand_ids, np.int32):
        cand_ids = np.ones(len(cand_hzxys), dtype=np.int) * int(cand_ids)
    if len(cand_hzxys) != len(cand_ids):
        raise IndexError(f"cand_hzxys should have same length as cand_ids")
    # reference ids
    if ref_ids is None:
        ref_ids == np.arange(len(ref_hzxys))
    if len(ref_hzxys) != len(ref_ids):
        raise IndexError(f"ref_hzxys should have same length as ref_ids")
    
    _nc_dists = []
    # loop through zxys
    for _hzxys, _id in zip(cand_hzxys, cand_ids):
        
        if len(_hzxys) == 0:
            _nc_dists.append([])
        else:
            _hzxys = np.array(_hzxys)
            # get reference zxys
            #_nc_hzxys = []
            _start,_end = max(_id-neighbor_len, min(ref_ids)), min(_id+neighbor_len+1, max(ref_ids)+1)
            # select inds
            _sel_local_inds = np.intersect1d(np.arange(_start,_end), ref_ids)
            _nc_ct = np.nanmean(np.array(ref_hzxys)[_sel_local_inds], axis=0)
            #for _ri in range(_start, _end):
            #    if _ri in ref_ids:
            #        _nc_hzxys.append(ref_hzxys[list(ref_ids).index(_ri)])
            # get neighboring center
            #_nc_ct = np.nanmean(_nc_hzxys, axis=0)
            # calculate distances
            if len(np.shape(_hzxys)) == 1:
                _dist = np.linalg.norm(_hzxys[-3:]-_nc_ct[-3:])
                _nc_dists.append(_dist)
            elif len(np.shape(_hzxys)) == 2:
                _dists = np.linalg.norm(_hzxys[:,-3:]- _nc_ct[np.newaxis,-3:], axis=1)
                _nc_dists.append(_dists)
            else:
                raise IndexError(f"_hzxys should be 1d or 2d array!")

    return _nc_dists


def Pick_spots_by_intensity(cand_spot_list, pix_size=np.array([200,106,106])):
    hzxys_list = []
    # loop through chromosomes
    for _spot_list in cand_spot_list:
        # init
        _sel_hzxys = []
        # get coords
        for _ind, _spots in enumerate(_spot_list):
            # if no candidate spots, append a Nan
            if len(_spots) == 0:
                _sel_hzxys.append(np.ones(4)*np.nan)
            # else, pick brightest one
            else:
                _spots = np.array(_spots)
                # if spots are given:
                if np.shape(_spots)[1] == 4:
                    _hzxys = np.array(_spots).copy()
                else:
                    # get hzxys
                    _hzxys = np.array(_spots)[:,:4]
                    _hzxys[:,1:4] = _hzxys[:,1:4] * np.array(pix_size)[np.newaxis,:] # convert to nm
                # append the brightest one
                _sel_hzxys.append(_hzxys[np.argmax(_hzxys[:,0])])
        # append selected chr
        hzxys_list.append(np.array(_sel_hzxys))

    return np.stack(hzxys_list)

def _generate_ref_of_chr(_picked_hzxys, picked_ids,
                         _ref_hzxys, ref_ids, neighbor_len=5):
    _ct_dists = chromosome_center_dists(_picked_hzxys, np.nanmean(_ref_hzxys,axis=0))
    _local_dists = local_center_dists(_picked_hzxys, picked_ids, 
                                                _ref_hzxys, ref_ids, neighbor_len=neighbor_len)
    _ints = np.array(_picked_hzxys)[:,0]
    
    return _ct_dists, _local_dists, _ints

def generate_reference_from_population(picked_hzxys_list, picked_ids=None,
                                       ref_hzxys_list=None, ref_ids=None,
                                       num_threads=12,
                                       neighbor_len=5, collapse_regions=True,
                                       verbose=True):
    """Function to generate reference based on currently picked spots"""
    from tqdm.notebook import tqdm
    if verbose:
        print(f"- generate reference metrics from picked chrs.")
    # picked ids
    if picked_ids is None:
        picked_ids = np.arange(len(picked_hzxys_list[0]))
    # references
    if ref_hzxys_list is None:
        ref_hzxys_list = picked_hzxys_list
    if ref_ids is None:
        ref_ids = picked_ids
    # initialize metrics    
    _ct_dists_list, _local_dists_list, _ints_list = [], [], []
    # mp option 
    if isinstance(num_threads, int) and num_threads > 1:
        import multiprocessing as mp
        if verbose:
            _mp_time = time.time()
            print(f"-- multiprocessing process references with {num_threads} threads", end=', ')
        _args = [(_picked_hzxys, picked_ids, _ref_hzxys, ref_ids, neighbor_len)
                 for _picked_hzxys, _ref_hzxys in zip(picked_hzxys_list, ref_hzxys_list)]
        with mp.Pool(num_threads) as ref_pool:
            ref_results = ref_pool.starmap(_generate_ref_of_chr, _args)
            ref_pool.close()
            ref_pool.join()
            ref_pool.terminate()
        # extract results
        _ct_dists_list = [_r[0] for _r in ref_results]
        _local_dists_list = [_r[1] for _r in ref_results]
        _ints_list = [_r[2] for _r in ref_results]
        if verbose:
            print(f"in {time.time()-_mp_time:.3f}s")
    # single thread option
    else:
        if verbose:
            _ss_time = time.time()
            print(f"-- process references", end=', ')
        # loop through chromosomes
        for _picked_hzxys, _ref_hzxys in tqdm(zip(picked_hzxys_list, ref_hzxys_list)):
            _ct_dists_list.append(chromosome_center_dists(_picked_hzxys, np.nanmean(_ref_hzxys,axis=0)))
            _local_dists_list.append(local_center_dists(_picked_hzxys, picked_ids, 
                                                        _ref_hzxys, ref_ids, neighbor_len=neighbor_len))
            _ints_list.append(np.array(_picked_hzxys)[:,0])
        if verbose:
            print(f"in {time.time()-_ss_time:.3f}s")
    
    # merge
    if collapse_regions:
        if verbose:
            print(f"-- collapse all regions into 1d.")
        _ct_dists_list = np.ravel(_ct_dists_list)
        _local_dists_list = np.ravel(_local_dists_list)
        _ints_list = np.ravel(_ints_list)
    else:
        _ct_dists_list = np.array(_ct_dists_list)
        _local_dists_list = np.array(_local_dists_list)
        _ints_list = np.array(_ints_list)
    
    return _ct_dists_list, _local_dists_list, _ints_list


def cum_val(vals,target):
    """returns the fraction of elements with value < taget. assumes vals is sorted"""
    niter_max = 10
    niter = 0
    m,M = 0,len(vals)-1
    while True:
        mid = int((m+M)/2)
        if vals[mid]<target:
            m = mid
        else:
            M = mid
        niter+=1
        if (M-m)<2:
            break
        if niter > niter_max:
            break

    return mid/float(len(vals))

def _maximize_score_spot_picking_of_chr(_cand_hzxys, cand_ids, _ref_hzxys, ref_ids, 
                                        ref_ct_dists, ref_local_dists, ref_ints,
                                        neighbor_len=5, use_center=True, use_local=True,):
    """single function to process data"""
    
    if use_center:
        _cand_ct_dists = chromosome_center_dists(_cand_hzxys, np.nanmean(_ref_hzxys,axis=0))
    if use_local:
        _cand_local_dists = local_center_dists(_cand_hzxys, cand_ids, 
                                               _ref_hzxys, ref_ids, 
                                               neighbor_len=neighbor_len)
    _cand_ints = extract_intensities(_cand_hzxys)
    # convert to scores
    _scores = []
    _sel_scores = []
    _sel_hzxys = []
    for _rid, _hzxys in enumerate(_cand_hzxys):
        if len(_hzxys) == 0:
            _scores.append([])
            # append a NAN to bad region
            _sel_scores.append(np.nan)
            _sel_hzxys.append(np.ones(4)*np.nan)

        elif len(np.shape(np.array(_hzxys))) == 1:
            if len(np.shape(ref_ints)) == 2:
                _sc = cum_val(ref_ints[:,_rid], _cand_ints[_rid])
            else:
                _sc = cum_val(ref_ints[:], _cand_ints[_rid])
            # center dist
            if use_center:
                if len(np.shape(_cand_ct_dists)) == 2:
                    _sc *= 1 - cum_val(_cand_ct_dists[:,_rid], _cand_ct_dists[_rid])
                else:
                    _sc *= 1 - cum_val(_cand_ct_dists[:], _cand_ct_dists[_rid])
            # local dist
            if use_local:
                if len(np.shape(_cand_local_dists)) == 2:
                    _sc *= 1 - cum_val(_cand_local_dists[:,_rid], _cand_local_dists[_rid])
                else:
                    _sc *= 1 - cum_val(_cand_local_dists[:], _cand_local_dists[_rid])
            _scores.append(_sc)
            # append this only spot
            _sel_scores.append(_sc)
            _sel_hzxys.append(_hzxys)

        elif len(np.shape(np.array(_hzxys))) == 2:
            _scs = []
            for _sid in range(len(_hzxys)):
                if len(np.shape(ref_ints)) == 2:
                    _sc = cum_val(ref_ints[:,_rid], _cand_ints[_rid][_sid])
                else:
                    _sc = cum_val(ref_ints[:], _cand_ints[_rid][_sid])
                # center dist
                if use_center:
                    if len(np.shape(ref_ct_dists)) == 2:
                        _sc *= 1 - cum_val(ref_ct_dists[:,_rid], _cand_ct_dists[_rid][_sid])
                    else:
                        _sc *= 1 - cum_val(ref_ct_dists[:], _cand_ct_dists[_rid][_sid])
                # local dist
                if use_local:
                    if len(np.shape(ref_local_dists)) == 2:
                        _sc *= 1 - cum_val(ref_local_dists[:,_rid], _cand_local_dists[_rid][_sid])
                    else:
                        _sc *= 1 - cum_val(ref_local_dists[:], _cand_local_dists[_rid][_sid])
                _scs.append(_sc)
            _scores.append(np.array(_scs))
            # append this only spot
            _sel_scores.append(np.nanmax(_scs))
            _sel_hzxys.append(_hzxys[np.argmax(_scs)])
    
    return np.array(_sel_hzxys), np.array(_sel_scores), _scores


def EM_pick_scores_in_population(cand_hzxys_list, cand_ids=None, init_hzxys_list=None, 
                                 ref_ct_dists=None, ref_local_dists=None, ref_ints=None, 
                                 ref_hzxys_list=None, ref_ids=None, 
                                 num_threads=12,
                                 neighbor_len=5, use_center=True, use_local=True,
                                 pix_size=[200,106,106], collapse_ref_regions=True,
                                 verbose=True):
    """Function for E-step within EM spot picking"""
    from tqdm.notebook import tqdm
    if verbose:
        print(f"- start EM picking.")
        #print(f"- EM pick spots, \
        #E: estimate with currently picked reference values. \
        #M: pick spots that maximize overall scores")
    # initialize if not give
    if init_hzxys_list is None:
        init_hzxys_list = Pick_spots_by_intensity(cand_hzxys_list, pix_size=pix_size)
    # candidate ids
    if cand_ids is None:
        cand_ids = np.arange(len(cand_hzxys_list[0]))
    # reference hzxys
    if ref_hzxys_list is None:
        ref_hzxys_list = init_hzxys_list
    if ref_ids is None:
        ref_ids = cand_ids
    
    # reference distributions
    if (ref_ints is None) \
        or (use_center and ref_ct_dists is None) \
        or (use_local and ref_local_dists is None):
        if verbose:
            print(f"-- generate reference from initial picked spots.")
        # generate references according to this dataset
        ref_ct_dists, ref_local_dists, ref_ints = generate_reference_from_population(
                init_hzxys_list, cand_ids, ref_hzxys_list, ref_ids, 
                num_threads=num_threads,
                neighbor_len=neighbor_len, collapse_regions=collapse_ref_regions,
            )
    else:
        if verbose:
            print(f"-- directly use given reference.")
    
    all_scores_list = []
    sel_scores_list = []
    sel_hzxys_list = []
    
    if isinstance(num_threads, int) and num_threads > 1:
        if verbose:
            _mp_time = time.time()
            print(f"-- multiprocessing maximization step with {num_threads} threads", end=', ')
        import multiprocessing as mp
        _args = [(_cand_hzxys, cand_ids, _ref_hzxys, ref_ids, 
                  ref_ct_dists, ref_local_dists, ref_ints,
                  neighbor_len, use_center, use_local)
                 for _cand_hzxys, _ref_hzxys in zip(cand_hzxys_list, ref_hzxys_list)]
        with mp.Pool(num_threads) as ref_pool:
            ref_results = ref_pool.starmap(_maximize_score_spot_picking_of_chr, _args)
            ref_pool.close()
            ref_pool.join()
            ref_pool.terminate()
        # extract results
        sel_hzxys_list = [_r[0] for _r in ref_results]
        sel_scores_list = [_r[1] for _r in ref_results]
        all_scores_list = [_r[2] for _r in ref_results]     
        if verbose:
            print(f"in {time.time()-_mp_time:.3f}s")
    else:
        if verbose:
            _ss_time = time.time()
            print(f"-- maximization step", end=', ')
        # loop through chromosomes
        for _cand_hzxys, _init_hzxys, _ref_hzxys in tqdm(zip(cand_hzxys_list, init_hzxys_list, ref_hzxys_list)):
            if use_center:
                _cand_ct_dists = chromosome_center_dists(_cand_hzxys, np.nanmean(_ref_hzxys,axis=0))
            if use_local:
                _cand_local_dists = local_center_dists(_cand_hzxys, cand_ids, 
                                                       _ref_hzxys, ref_ids, 
                                                       neighbor_len=neighbor_len)
            _cand_ints = extract_intensities(_cand_hzxys)
            # convert to scores
            _scores = []
            _sel_scores = []
            _sel_hzxys = []
            for _rid, _hzxys in enumerate(_cand_hzxys):
                if len(_hzxys) == 0:
                    _scores.append([])
                    # append a NAN to bad region
                    _sel_scores.append(np.ones(4)*np.nan)
                    _sel_hzxys.append(np.ones(4)*np.nan)

                elif len(np.shape(np.array(_hzxys))) == 1:
                    if np.shape(ref_ints) == 2:
                        _sc = cum_val(ref_ints[:,_rid], _cand_ints[_rid])
                    else:
                        _sc = cum_val(ref_ints[:], _cand_ints[_rid])
                    # center dist
                    if use_center:
                        if np.shape(_cand_ct_dists) == 2:
                            _sc *= 1 - cum_val(_cand_ct_dists[:,_rid], _cand_ct_dists[_rid])
                        else:
                            _sc *= 1 - cum_val(_cand_ct_dists[:], _cand_ct_dists[_rid])
                    # local dist
                    if use_local:
                        if np.shape(_cand_local_dists) == 2:
                            _sc *= 1 - cum_val(_cand_local_dists[:,_rid], _cand_local_dists[_rid])
                        else:
                            _sc *= 1 - cum_val(_cand_local_dists[:], _cand_local_dists[_rid])
                    _scores.append(_sc)
                    # append this only spot
                    _sel_scores.append(_sc)
                    _sel_hzxys.append(_hzxys)

                elif len(np.shape(np.array(_hzxys))) == 2:
                    _scs = []
                    for _sid in range(len(_hzxys)):
                        if np.shape(ref_ints) == 2:
                            _sc = cum_val(ref_ints[:,_rid], _cand_ints[_rid][_sid])
                        else:
                            _sc = cum_val(ref_ints[:], _cand_ints[_rid][_sid])
                        # center dist
                        if use_center:
                            if np.shape(_cand_ct_dists) == 2:
                                _sc *= 1 - cum_val(ref_ct_dists[:,_rid], _cand_ct_dists[_rid][_sid])
                            else:
                                _sc *= 1 - cum_val(ref_ct_dists[:], _cand_ct_dists[_rid][_sid])
                        # local dist
                        if use_local:
                            if np.shape(_cand_local_dists) == 2:
                                _sc *= 1 - cum_val(ref_local_dists[:,_rid], _cand_local_dists[_rid][_sid])
                            else:
                                _sc *= 1 - cum_val(ref_local_dists[:], _cand_local_dists[_rid][_sid])
                        _scs.append(_sc)
                    _scores.append(np.array(_scs))
                    # append this only spot
                    _sel_scores.append(np.nanmax(_scs))
                    _sel_hzxys.append(_hzxys[np.argmax(_scs)])

            all_scores_list.append(_scores)
            sel_scores_list.append(np.array(_sel_scores))
            sel_hzxys_list.append(np.array(_sel_hzxys))
        if verbose:
            print(f"in {time.time()-_ss_time:.3f}s")
    return sel_hzxys_list, sel_scores_list, all_scores_list
                
def evaluate_differences(old_hzxys_list, new_hzxys_list):
    """Function to evaluate similarity between two picked sets"""
    diffs = np.concatenate([np.linalg.norm(_ohzxys[:,-3:]-_nhzxys[:,-3:], axis=1) 
                            for _ohzxys,_nhzxys in zip(old_hzxys_list, new_hzxys_list)])
    return np.sum(diffs < 0.01) / np.sum(np.isnan(diffs)==False)


def screen_RNA_based_on_refs(rna_cand_hzxys_list, rna_region_ids,
                             ref_hzxys, ref_ids, dist_th=500, keep_no_ref=False):
    """Function to screen RNA spots based on their proximity to reference spots.
    Inputs:
        """
    if len(rna_cand_hzxys_list) != len(rna_region_ids):
        raise IndexError(f"length of rna_cand_hzxys_list should match rna_region_ids")
    if len(ref_hzxys) != len(ref_ids):
        raise IndexError(f"length of ref_hzxys should match ref_ids")
    for _rid in rna_region_ids:
        if _rid not in ref_ids:
            raise ValueError(f"RNA region id: {_rid} doesn't exist in reference ids!")
    _sel_cand_list = []
    for _id, _cand_hzxys in zip(rna_region_ids, rna_cand_hzxys_list):
        if len(_cand_hzxys) == 0:
            _sel_cand_list.append([])
        else:
            _ind = list(ref_ids).index(_id)
            if np.isnan(ref_hzxys[_ind, -3:]).any():
                if keep_no_ref:
                    _sel_cand_list.append(_cand_hzxys)
                else:
                    _sel_cand_list.append([])
            else:
                _dists = np.linalg.norm(_cand_hzxys[:,-3:] - ref_hzxys[_ind, -3:], axis=1)
                if np.sum(_dists <= dist_th) == 0:
                    _sel_cand_list.append([])
                else:
                    _sel_cand_list.append(np.array(_cand_hzxys)[_dists <= dist_th,:])
    return _sel_cand_list