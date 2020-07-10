import numpy as np
import os, sys
from . import _distance_zxy
from ..visual_tools import translate_spot_coordinates

def convert_pick_RNA_spots(rna_cell, dna_cell, 
                           rotation_mat=None, rotation_order='forward',
                           intensity_th=1,
                           tss_ref='EM_picked_gene_spots', tss_dist_th=500., 
                           dna_ref='EM_picked_unique_spots', dna_dist_th=500., 
                           chr_ref_attr='chrom_coords', chr_dist_th=np.inf,
                           add_attr=True, attr_name='ts_RNA_spots', verbose=False):
    if rotation_mat is None:
        rotation_mat = np.load(os.path.join(rna_cell.experiment_folder, 'rotation.npy'))
    
    # initiate
    cand_spot_list = []
    sel_spot_list = []
    for _i,_chrom_coord in enumerate(dna_cell.chrom_coords):
        _sel_spots = np.zeros([len(getattr(rna_cell, 'rna-unique_spots')), 11])
        _sel_spots[:,0] = 0
        _sel_spots[:,1:] = np.nan
        sel_spot_list.append(_sel_spots)
        
    for _rid, (_spot_list, _k) in enumerate(zip(getattr(rna_cell, 'rna-unique_spots'), sorted(getattr(rna_cell, 'rna-info_dic').keys())) ):
        _info = getattr(rna_cell, 'rna-info_dic')[_k]
        _cands = []
        # loop through each chromosome
        for _cid, (_spots, _chrom_coord) in enumerate(zip(_spot_list, dna_cell.chrom_coords)):
            
            # check if there are any candidate spots
            if len(_spots) == 0:
                _cands.append([])
                continue
            else:
                _cand_spots = _spots[_spots[:,0] >= intensity_th]
            # if there are no candidate spots, directly continue
            if len(_cand_spots) == 0:
                _cands.append([])
                continue
            else:
                # do translation first
                _ts_spots = translate_spot_coordinates(rna_cell, dna_cell, 
                                                    _cand_spots, 
                                                    rotation_mat=rotation_mat, 
                                                    rotation_order=rotation_order, verbose=False)
                                        
                _cands.append(_ts_spots)  
                                                                   
                ## now find ref_targets
                # if no ref_dna, check distance to center
                if 'DNA_id' not in _info or _k not in dna_cell.gene_dic:
                    _kept_spots = _ts_spots[np.linalg.norm((_ts_spots[:,1:4] - _chrom_coord) \
                                                * _distance_zxy, axis=1) <= chr_dist_th]
                # check if there are ref gene spots
                else:
                    _gene_ind = list(sorted(dna_cell.gene_dic.keys())).index(_k)
                    _tss_ref_spot = getattr(dna_cell, tss_ref)[_cid][_gene_ind]
                    _dna_ref_spot = getattr(dna_cell, dna_ref)[_cid][_info['DNA_id']]
                    
                    _keep_flags = np.zeros(len(_ts_spots),dtype=np.bool)
                    if not np.isnan(_tss_ref_spot).any():
                        _keep_flags += (np.linalg.norm((_ts_spots - _tss_ref_spot)[:,1:4] \
                                                * _distance_zxy, axis=1) < tss_dist_th)
                        #print('tss', sum(_keep_flags))
                    if not np.isnan(_dna_ref_spot).any():
                        _keep_flags += (np.linalg.norm((_ts_spots - _dna_ref_spot)[:,1:4] \
                                                * _distance_zxy, axis=1) < dna_dist_th)
                        #print('dna', sum(_keep_flags))
                    if np.isnan(_tss_ref_spot).any() and np.isnan(_dna_ref_spot).any():
                        _keep_flags += (np.linalg.norm((_ts_spots - _dna_ref_spot)[:,1:4] \
                                                * _distance_zxy, axis=1) < chr_dist_th)
                        #print('chr', sum(_keep_flags))
                    _kept_spots = _ts_spots[_keep_flags]
                
                # then keep the brightest one
                if len(_kept_spots) > 0:
                    _sel_spot = _kept_spots[np.argmax(_kept_spots[:,0])]
                    sel_spot_list[_cid][_rid] = _sel_spot
        cand_spot_list.append(_cands)
    if add_attr:
        if verbose:
            print(f"-- add attribute: {attr_name} to DNA cell")
        setattr(dna_cell, attr_name, sel_spot_list)
        setattr(dna_cell, 'ts_cand_rna_spots', cand_spot_list)

    return sel_spot_list

def fit_matched_centers(im, ref_centers, match_distance_th=3,
                         match_unique=True, 
                         th_seed=100, th_seed_per=98, 
                         use_percentile=False, seeding_kwargs={},
                         fitting_kwargs={},
                         verbose=False):
    """Function to find matched centers given image 
        and rough reference centers
    Inputs:
    
    Outputs:
    
    """
    from scipy.spatial.distance import cdist
    from ..spot_tools.fitting import get_seeds, get_centers
    ## check inputs
    if not isinstance(im, np.ndarray) and not isinstance(im, np.memmap):
        raise TypeError(f"Wrong input data type for im, should be np.ndarray or memmap, {type(im)} given!")
    ref_centers = np.array(ref_centers)[:,:3]
    if verbose:
        print(f"-- finding seeds paired with {len(ref_centers)} centers in given image")
    _seeds = get_seeds(im, th_seed=th_seed,
                       th_seed_per=th_seed_per,
                       use_percentile=use_percentile, 
                       return_h=False, 
                       **seeding_kwargs)
    ## find matched seeds
    dists = cdist(_seeds, ref_centers)
    _seed_inds, _ref_inds = np.where(dists <= match_distance_th)
    # get indices that matched
    _matched_ref_inds, _matched_cts = np.unique(_ref_inds, return_counts=True)
    if len(_matched_ref_inds) == 0:
        print(f"--- no matched pair found, exit.")
        return np.array([]), np.array([]), np.array([])
    else:
        if verbose:
            print(f"--- {len(_matched_ref_inds)} pairs found among {len(ref_centers)} refs.")
        # get indices that not matched
        _nonmatch_ref_inds = np.setdiff1d(np.arange(len(ref_centers)), 
                                        _matched_ref_inds)
        if match_unique:
            _matched_ref_inds = _matched_ref_inds[_matched_cts==1]
        # get matched seed inds
        _matched_seed_inds = []
        for _ri in _matched_ref_inds:
            _matched_seed_inds.append(_seed_inds[np.where(_ref_inds==_ri)[0][0]])
        _matched_seed_inds = np.array(_matched_seed_inds, dtype=np.int)
        # get matched ref_centers
        _matched_ref_cts = ref_centers[_matched_ref_inds]
        # get matched seeds and do fitting
        _matched_seeds = _seeds[_matched_seed_inds]
        
        ## fit centers based on matched seeds
        _matched_cts = get_centers(im, _matched_seeds, 
                                remove_close_pts=False,
                                verbose=verbose,)
        
        return _matched_cts, _matched_ref_cts, _matched_ref_inds

def find_paired_centers(tar_cts, ref_cts, drift=None,
                        cutoff=2, 
                        return_paired_cts=True, 
                        return_kept_inds=False,
                        verbose=False):
    """Function to fast find uniquely paired centers given two lists
        of centers and candidate drifts (tar-ref).
    Inputs:
        tar_cts: target centers, 2d-numpy array
        ref_cts: reference centers, 2d-numpy array, 
            numbers don't have to match but dimension should match tar_cts
        drift: rough drift between tar_cts - ref_cts, 1d-numpy array of dim,
        cutoff: unique matching distance cutoff, float (default: 2)
        return_paired_cts: whether return paired center coordinates, bool (default: True)
        verbose: whether say something! bool (default: True)
    Outputs:
        _mean_shift: mean drift calculated from paired centers, 1d numpy array of dim,
    conditional outputs:
        _paired_tar_cts: paired target centers, 2d numpy arrray of n_spots*dim
        _paired_ref_cts: paired reference centers, 2d numpy arrray of n_spots*dim
    """
    from scipy.spatial.distance import cdist
    _tar_cts = np.array(tar_cts)
    _ref_cts = np.array(ref_cts)
    if drift is None:
        _drift = np.zeros(np.shape(_tar_cts)[1])
    else:
        _drift = np.array(drift, dtype=np.float)[:np.shape(_tar_cts)[1]]
    if verbose:
        print(f"-- aligning {len(_tar_cts)} centers to {len(_ref_cts)} ref_centers, given drift:{np.round(_drift,2)}",
              end=', ')
    # adjust ref centers to match target centers
    _adj_ref_cts = _ref_cts + _drift
    # canculate dists
    _dists = cdist(_tar_cts, _adj_ref_cts)
    _tar_inds, _ref_inds = np.where(_dists <= cutoff)
    # pick only unique ones
    _unique_tar_inds = np.where(np.sum(_dists <= cutoff, axis=1) == 1)[0]
    _unique_ref_inds = np.where(np.sum(_dists <= cutoff, axis=0) == 1)[0]
    # get good pairs
    _unique_pair_inds = [[_t, _r] for _t, _r in zip(_tar_inds, _ref_inds)
                         if _t in _unique_tar_inds \
                         and _r in _unique_ref_inds]
    # acquire paired centers
    _paired_tar_cts = []
    _paired_ref_cts = []
    for _it, _ir in _unique_pair_inds:
        _paired_tar_cts.append(_tar_cts[_it])
        _paired_ref_cts.append(_ref_cts[_ir])
    _paired_tar_cts = np.array(_paired_tar_cts)
    _paired_ref_cts = np.array(_paired_ref_cts)
    
    # calculate mean drift and return
    _new_drift = np.nanmean(_paired_tar_cts - _paired_ref_cts, axis=0)
    if verbose:
        print(f"{len(_paired_tar_cts)} pairs found, updated_drift:{np.round(_new_drift,2)}")

    # return
    _return_args = [_new_drift]
    if return_paired_cts:
        _return_args.append(_paired_tar_cts)
        _return_args.append(_paired_ref_cts)
    if return_kept_inds:
        _paired_tar_inds = np.array(_unique_pair_inds, dtype=np.int)[:,0]
        _paired_ref_inds = np.array(_unique_pair_inds, dtype=np.int)[:,1]
        # append
        _return_args.append(_paired_tar_inds)
        _return_args.append(_paired_ref_inds)
    return tuple(_return_args)

def check_paired_centers(paired_tar_cts, paired_ref_cts, 
                         outlier_sigma=1.5, 
                         return_paired_cts=True,
                         verbose=False):
    """Function to check outlier for paired centers, 
        outlier is judged by whether a drift 
        is significantly different from its neighbors
    Inputs:
        paired_tar_cts: paired target centers, 2d-numpy array, 
        paired_ref_cts: paired reference centers, 2d-numpy array,
            should be exactly same size as paired_tar_cts
        outlier_sigma: cutoff for a drift comparing to neighboring drifts
            (assuming gaussian distribution), float, default: 1.5
        return_paired_cts: whether return paired center coordinates, bool (default: True)
        verbose: whether say something! bool (default: True)
    Outputs:
        _mean_shift: mean drift calculated from paired centers, 1d numpy array of dim,
    conditional outputs:
        _paired_tar_cts: paired target centers, 2d numpy arrray of n_spots*dim
        _paired_ref_cts: paired reference centers, 2d numpy arrray of n_spots*dim
        """
    from scipy.spatial import Delaunay
    _tar_cts = np.array(paired_tar_cts, dtype=np.float)
    _ref_cts = np.array(paired_ref_cts, dtype=np.float)
    _shifts = _tar_cts - _ref_cts
    if verbose:
        print(f"-- check {len(_tar_cts)} pairs of centers", end=', ')
    # initialize new center shifts
    _new_shifts = []
    # use Delaunay to find neighbors for each center pair
    _tri = Delaunay(_ref_cts)
    for _i, (_s, _tc, _rc) in enumerate(zip(_shifts, _tar_cts, _ref_cts)):
        # get neighboring center ids
        _nb_ids = np.array([_simplex for _simplex in _tri.simplices.copy()
                            if _i in _simplex], dtype=np.int)
        _nb_ids = np.unique(_nb_ids)
        # remove itself
        _nb_ids = _nb_ids[(_nb_ids != _i) & (_nb_ids != -1)]
        # get neighbor shifts
        _nb_ref_cts = _ref_cts[_nb_ids]
        _nb_shifts = _shifts[_nb_ids]
        _nb_weights = 1 / np.linalg.norm(_nb_ref_cts-_rc, axis=1)
        # update this shift
        _nb_shift = np.dot(_nb_shifts.T, _nb_weights) / np.sum(_nb_weights)
        _new_shifts.append(_nb_shift)
    _new_shifts = np.array(_new_shifts)
    # select among new_shifts
    _diffs = np.linalg.norm(_new_shifts-_shifts, axis=1)
    _keep_flags = np.array(_diffs < np.mean(_diffs) + np.std(_diffs) * outlier_sigma)
    # cast this keep flags
    _kept_tar_cts = _tar_cts[_keep_flags]
    _kept_ref_cts = _ref_cts[_keep_flags]
    # mean drift
    _new_drift = np.nanmean(_kept_tar_cts-_kept_ref_cts, axis=0)
    if verbose:
        print(f"{len(_kept_tar_cts)} pairs kept. new drift:{np.round(_new_drift,2)}")
    
    # return
    _return_args = [_new_drift]
    if return_paired_cts:
        _return_args.append(_kept_tar_cts)
        _return_args.append(_kept_ref_cts)
    
    return tuple(_return_args)
    