import numpy as np
from scipy.stats import scoreatpercentile
# import other sub-packages
from . import scoring
from . import _distance_zxy


## check spots
def check_spot_scores(all_spot_list, sel_spots, region_ids=None, sel_indices=None, spot_num_th=100,
                      chrom_coord=None, distance_zxy=_distance_zxy, distance_limits=[0, np.inf],
                      intensity_th=0., ref_dist_metric='median', score_metric='linear',
                      local_size=5, w_ctdist=2, w_lcdist=1, w_int=1, 
                      ignore_nan=True, nan_mask=0., inf_mask=-1000.,
                      check_th=-3.5, check_percentile=1., hard_dist_th=6000,
                      return_sel_scores=False, return_other_scores=False, verbose=False):
    """Function to check spots given merged_spot_list and selected_spots
    Inputs:
        all_spot_list: list of all spots to be included into score calculation, list of spots-np.ndarray
        sel_spots: selected spots to be evaluated, np.ndarray or list of spot
        region_ids: region uid for candidate spots, list/array of ints
        sel_indices: index of sel_spots in all_spot_list, they should be of same length, list/array of ints
        chrom_coord: chromosome coordinate of sel_spots, np.ndarray of 3 (default: None, not used)
        update_chrom_coord: whether use updated chromosome coords, bool (default: False)
        distance_zxy: translate pixel to nm, array of 3 (default: [200,106,106])
        local_size: size to calculate local distance, int (default: 5)
        w_ctdist: weight for distance_to_chromosome_center, float (default: 1)
        w_lcdist: weight for distance_to_local_center, float (default: 1)
        w_int: weight for spot intensity, float (default: 2)
        ignore_nan: whether ignore nan during scoring, bool (default: True)
        check_th: the relative threshold for stringency check, 
            * which will multiply the sum of all weights to estimate threshold, bool (default: -2)
        check_percentile: another percentile threshold that may apply to data, float (default: 5.)
        return_sel_scores: whether return selected scores, bool (default: False)
        return_other_scores: whether return other scores, bool (default: False)
        verbose: say something! bool (default: False)
    Outputs:
        filtered_spots: np.ndarray of spots that passes filter, np.ndarray
    optional outputs:
        sel_scores: selected scores, np.ndarray
        other_scores: un-selected spot scores, np.ndarray or list of spots
    """
    ## check inputs
    from .scoring import _center_distance
    if isinstance(all_spot_list, list):
        all_spots = np.concatenate(all_spot_list)
    elif isinstance(all_spot_list, np.ndarray):
        all_spots = all_spot_list
    else:
        raise TypeError(f"Wrong input type for all_spot_list, should be list or np.ndarray.")
    
    if not isinstance(sel_spots, list) and not isinstance(sel_spots, np.ndarray):
        raise TypeError(f"Wrong input type for sel_spots, should be list or np.ndarray.")
    sel_spots = np.array(sel_spots)
    if region_ids is None:
        region_ids = np.arange(len(sel_spots))
    elif len(region_ids) != len(sel_spots):
        raise ValueError(f"Wrong input length for region_ids, not matched with sel_spots")
    else:
        region_ids = np.array(region_ids, dtype=np.int)
    
    if verbose:
        print(f"-- check spot for {len(sel_spots)} spots")
    
    ## calculate spot_scores
    # determine reference spots
    if len(sel_spots) < spot_num_th:
        if verbose:
            print(f"--- use all spots as reference,", end=' ')
        _ref_spots = all_spots
        _ref_ids = np.concatenate([_id*np.ones(len(_spots), dtype=np.int) 
                                   for _spots,_id in zip(all_spot_list, region_ids)])          
    else:
        if verbose:
            print(f"--- use selected spots as reference,", end=' ')
        _ref_spots = sel_spots
        _ref_ids = region_ids

    # generate reference spot pool 
    _ref_center_dist, _ref_local_dist, _ref_neighbor_dist, _ref_intensities = scoring.generate_ref_from_chromosome(
        _ref_spots, _ref_ids, distance_zxy=distance_zxy, chr_center=chrom_coord,
        intensity_th=intensity_th, local_size=local_size, ref_dist_metric=ref_dist_metric,
        ignore_nan=ignore_nan)
    
    # calculate spot scores
    _spot_scores = [scoring.spot_score_in_chromosome(_spots, int(_uid), _ref_spots, _ref_ids, chrom_coord, 
                        _ref_center_dist, _ref_local_dist, _ref_neighbor_dist, _ref_intensities,
                        ref_dist_metric=ref_dist_metric, ignore_nan=ignore_nan, 
                        nan_mask=nan_mask, inf_mask=inf_mask, distance_zxy=distance_zxy,
                        distance_limits=distance_limits, metric=score_metric, intensity_th=intensity_th,
                        local_size=local_size, w_ctdist=w_ctdist, w_lcdist=w_lcdist, w_int=w_int) 
                        for _spots, _uid in zip(all_spot_list, region_ids)]
    _sel_scores = scoring.spot_score_in_chromosome(sel_spots, region_ids, _ref_spots, _ref_ids, chrom_coord, 
                        _ref_center_dist, _ref_local_dist, _ref_neighbor_dist, _ref_intensities,
                        ref_dist_metric=ref_dist_metric, ignore_nan=ignore_nan, 
                        nan_mask=nan_mask, inf_mask=inf_mask, distance_zxy=distance_zxy,
                        distance_limits=distance_limits, metric=score_metric, intensity_th=intensity_th,
                        local_size=local_size, w_ctdist=w_ctdist, w_lcdist=w_lcdist, w_int=w_int)
    _other_scores = []
    if sel_indices is not None:
        for _scs, _sel_i in zip(_spot_scores, sel_indices):
            _other_cs = list(_scs)
            if len(_other_cs) > 0 and _sel_i >= 0:
                _other_cs.pop(_sel_i)
                _other_scores += list(_other_cs)
    else:
        for _scs in _spot_scores:
            _other_scores += list(_scs)
    
    _sel_scores = np.array(_sel_scores)
    _other_scores = np.array(_other_scores)
    ## picking thresholds
    # calculate threshold
    _kept_sel_scores = _sel_scores[_sel_scores > max(inf_mask, - np.inf)]
    _th_sel = scoreatpercentile(_kept_sel_scores, check_percentile)
    _kept_other_scores = _other_scores[_other_scores > max(inf_mask, - np.inf)]
    if len(_kept_other_scores) == 0:
        _kept_other_scores = np.array([-np.inf])
    if sel_indices is None:
        _other_th_per = max(0, 100-100.*len(_kept_sel_scores)/len(_kept_other_scores) - check_percentile )
    else:
        _other_th_per = max(0, 100-100.*len(_kept_sel_scores)/len(_kept_other_scores) - check_percentile )
    
    _th_other = scoreatpercentile(_kept_other_scores, _other_th_per)

    _th_weight = check_th * (w_ctdist + w_lcdist + w_int)
    if verbose:
        print(f"current thresholds: {np.round([_th_sel, _th_other, _th_weight],3)}")
    if check_percentile > 0 and check_percentile < 100:
        _final_check_th = max(min(_th_sel, _th_other), _th_weight)
    else:
        _final_check_th = _th_weight
    ## appliy filtering
    if verbose:
        print(f"--- applying stringency check for {len(sel_spots)} spots, threshold={_final_check_th}")
    _filtered_spots = np.array(sel_spots).copy()

    # seletion condition 
    _condition = (_sel_scores < _final_check_th) + (np.isnan(_filtered_spots).sum(1)>0)
    if hard_dist_th is not None and hard_dist_th is not False:
        _ct_dists = _center_distance(sel_spots[:,1:4]*distance_zxy, 
                                     #center=chrom_coord*distance_zxy)
                                     center=None)
        _condition += _ct_dists > hard_dist_th
    # get failed inds
    _inds = np.where(_condition)[0]
    # delete spots
    _filtered_spots[_inds,0] = 0 # remove intensities
    _filtered_spots[_inds,1:] = np.nan # remove other parameters
    if verbose:
        print(f"--- {len(_inds)} spots didn't pass stringent quality check.")
    
    ## return
    if not return_sel_scores and not return_other_scores:
        return _filtered_spots
    # return spots combined with other info
    else:
        _return_args = (_filtered_spots,)
        if return_sel_scores:
            _return_args += (_sel_scores,)
        if return_other_scores:
            _return_args += (_other_scores,)
        # return!
        return _return_args