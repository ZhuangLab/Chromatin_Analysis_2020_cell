# from packages
import numpy as np
import multiprocessing as mp
import time,os,pickle,time
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import norm, scoreatpercentile

# from local
from .distance import domain_pdists
from .. import visual_tools

# mark off-diagonal features
def inter_domain_markers(coordiantes, domain_starts, 
                         norm_mat=None, metric='median', 
                         off_diagonal_th=0.4, allow_minus_dist=False,
                         keep_triu=False, 
                         marker_type='center', marker_param=1., keep_intensity=False, 
                         exclude_neighbors=False, exclude_edges=False, 
                         verbose=True):
    """Get off-diagonal markers for a given coordinates
    Inputs:
        
    Outouts:
    
    """
    ## check inputs
    _coordinates = np.array(coordiantes)
    _domain_starts = np.array(domain_starts, dtype=np.int)
    if len(_coordinates.shape) != 2:
        raise IndexError(f"Wrong input shape for coordinates, it should be 2d-array but shape:{_coordinates.shape} is given.")
    
    if len(_domain_starts) >= 1:
        _dm_pds = domain_pdists(_coordinates, _domain_starts, metric=metric, 
                                normalization_mat=norm_mat, 
                                allow_minus_dist=allow_minus_dist)

        _dx, _dy = np.where(squareform(_dm_pds) < off_diagonal_th)
        _dists = squareform(_dm_pds)[_dx,_dy]
        if not keep_triu:
            _dists = _dists[_dx!=_dy]
            _dx, _dy = _dx[_dx!=_dy], _dy[_dx!=_dy]
            _unique_dxy = np.stack([_dx, _dy]).transpose()

        else:
            _dxy = [(min(_x,_y),max(_x,_y)) for _x,_y in zip(_dx, _dy) if min(_x,_y) != max(_x,_y)]
            _unique_dxy = []
            for _xy in _dxy:
                if _xy not in _unique_dxy:
                    _unique_dxy.append(_xy)
            _unique_dxy = np.array(_unique_dxy)
        if exclude_neighbors:
            _kept_dxy = []
            for _dxy in _unique_dxy:
                if np.abs(_dxy[0] - _dxy[1]) > 1:
                    _kept_dxy.append(_dxy)
            _unique_dxy = np.array(_kept_dxy)
        if exclude_edges:
            _kept_dxy = []
            for _dxy in _unique_dxy:
                if 0 not in _dxy and len(_domain_starts)-1 not in _dxy:
                    _kept_dxy.append(_dxy)
            _unique_dxy = np.array(_kept_dxy)
    # re-calculate domain pdists
    _dm_pds = domain_pdists(_coordinates, _domain_starts, metric=metric, 
                            normalization_mat=norm_mat, allow_minus_dist=True)
    # generate markers  
    _marker_map = _generate_inter_domain_markers(_coordinates, _domain_starts, _dm_pds, _unique_dxy,
                                                 _marker_type=marker_type, _marker_param=marker_param, 
                                                 _keep_intensity=keep_intensity)
    return _unique_dxy, _marker_map


def _interdomain_likelihood(_distmap, _domain_starts, _inter_pairs,
                            _w_sel=1, _w_intra=1, valid_count=5, 
                            _normalize=True, _exclude_neighbors=True):
    """Function to evaluate likelihood ratio for interdomain interaction pairs, 
        given contact map, domain starts, called interaction pairs (of domain-ids)"""
    
    ## prepare inputs
    # first of all, extract domain starts
    _dm_starts = np.array(_domain_starts, dtype=np.int)
    _dm_ends = np.concatenate([_dm_starts[1:], np.array([len(_distmap)])])
    # initialize a domain interaction likelihood map
    _dm_inter_lks = np.zeros([len(_dm_starts), len(_dm_starts)])
    
    ## loop through all domains
    for _dm_id in range(len(_dm_starts)):
        # collect all interaction pairs related to this domain
        _called_partners = np.unique([np.array(_p)[np.where(np.array(_p)!=_dm_id)[0]] 
                                      for _p in _inter_pairs if _dm_id in _p ]).astype(np.int)
        _intra = _distmap[_dm_starts[_dm_id]:_dm_ends[_dm_id], _dm_starts[_dm_id]:_dm_ends[_dm_id]]
        # if there are interactions, compare called and nocalled pairs and calculate likelihood ratios
        if len(_called_partners) >= 1 \
            and len(_called_partners) < len(_dm_starts)- 1 \
            and np.sum(np.isnan(_intra)==False) > valid_count:
            _inters = [_distmap[_dm_starts[_i]:_dm_ends[_i], _dm_starts[_dm_id]:_dm_ends[_dm_id]]
                      for _i in range(len(_dm_starts))]
            _pos_inters = [_distmap[_dm_starts[_i]:_dm_ends[_i], _dm_starts[_dm_id]:_dm_ends[_dm_id]]
                           for _i in range(len(_dm_starts)) if _i in _called_partners]
            _neg_inters = [_distmap[_dm_starts[_i]:_dm_ends[_i], _dm_starts[_dm_id]:_dm_ends[_dm_id]]
                           for _i in range(len(_dm_starts)) if _i not in _called_partners and _i != _dm_id]
            _neg_ids = np.array([_i for _i in range(len(_dm_starts)) if _i not in _called_partners and _i != _dm_id], dtype=np.int)
            
            _all_inters = np.concatenate(_inters)
            _all_pos_inters = np.concatenate(_pos_inters)
            _all_neg_inters = np.concatenate(_neg_inters)
            # dom gaussian fitting for each dsitribution curve
            _pos_params = norm.fit(_all_pos_inters[np.isnan(_all_pos_inters)==False]) # mu, sigma
            _neg_params = norm.fit(_all_neg_inters[np.isnan(_all_neg_inters)==False]) # mu, sigma
            _intra_params = norm.fit(_intra[np.isnan(_intra)==False]) # mu, sigma
            # calculate likelihood
            _pos_neg_ratios = []
            _intra_neg_ratios = []
            for _i, _d in enumerate(_inters):
                # remove itself by marking a -inf
                if _i == _dm_id:
                    _pos_neg_ratios.append(-np.inf)
                    _intra_neg_ratios.append(-np.inf)
                # remove neighbors if specified
                elif np.abs(_i - _dm_id) == 1 and _exclude_neighbors:
                    _pos_neg_ratios.append(-np.inf)
                    _intra_neg_ratios.append(-np.inf)
                # other case, calculate
                else:
                    # log prob. for positive distribution
                    _pos_lk = np.sum(np.log(norm.pdf(_d[np.isnan(_d)==False], _pos_params[0], _pos_params[1]))) 
                    # log prob. for negative distribution
                    _neg_lk = np.sum(np.log(norm.pdf(_d[np.isnan(_d)==False], _neg_params[0], _neg_params[1])))
                    # log prob. for intra-domain distribution
                    _intra_lk = np.sum(np.log(norm.pdf(_d[np.isnan(_d)==False], _intra_params[0], _intra_params[1])))
                    # calculate log probability ratio
                    _pnr = _pos_lk - _neg_lk 
                    _inr = _intra_lk - _neg_lk
                    # normalize by region number
                    if _normalize:
                        if np.sum(np.isnan(_d)==False) > 0:
                            _pnr = _pnr / np.sum(np.isnan(_d)==False)
                            _inr = _inr / np.sum(np.isnan(_d)==False)
                        else:
                            _pnr = -np.inf 
                            _inr = -np.inf
                    _pos_neg_ratios.append(_pnr)
                    _intra_neg_ratios.append(_inr)
        # if there's no interaction for this domain, just skip because this domain should be far from the rest
        else:
            _pos_neg_ratios = -np.inf * np.ones(len(_dm_starts))
            _intra_neg_ratios = -np.inf * np.ones(len(_dm_starts))
        # add to positive/negative to total marker
        if _w_sel != 0:
            _dm_inter_lks[_dm_id] += np.array(_pos_neg_ratios) * _w_sel
            _dm_inter_lks[:,_dm_id] += np.array(_pos_neg_ratios) * _w_sel
        # add intra/negative to total marker
        if _w_intra != 0:
            _dm_inter_lks[_dm_id] += np.array(_intra_neg_ratios) * _w_intra
            _dm_inter_lks[:,_dm_id] += np.array(_intra_neg_ratios) * _w_intra

    return _dm_inter_lks

def _exclude_interdomain_by_contact(_coordinates, _domain_starts, _inter_pairs, 
                                    _mean_contact_ratio=0.2, 
                                    _contact_th=400):
    """Function to adjust interdomain by freqency of contacts"""
    # check coordinates
    _coordinates = np.array(_coordinates)
    if len(np.shape(_coordinates)) != 2:
        raise IndexError
    elif np.shape(_coordinates)[0] != np.shape(_coordinates)[1]:
        _coordinates = squareform(pdist(_coordinates))
    # check domain starts
    _domain_starts = np.array(_domain_starts, dtype=np.int)
    if 0 not in _domain_starts:
        _domain_starts = np.concatenate([np.array([0]), _domain_starts])
    # domain ends
    _domain_ends = np.concatenate([_domain_starts[1:], np.array([len(_coordinates)])])
    # loop through inter-pairs and check 
    _contact = _coordinates < _contact_th
    _kept_pairs = []
    for _pair in _inter_pairs:
        _slice0 = slice(_domain_starts[_pair[0]], _domain_ends[_pair[0]])
        _slice1 = slice(_domain_starts[_pair[1]], _domain_ends[_pair[1]])
        _arr = _contact[_slice0, _slice1]
        _freq = np.nansum(_arr) / np.sum(np.isnan(_arr)==False)
        if _freq > _mean_contact_ratio:
            _kept_pairs.append(_pair)
    
    return np.array(_kept_pairs, dtype=np.int)


def _adjust_interdomain_by_likelihood(_inter_pairs, _inter_dm_lk_mat, 
                                      _percentage_th=5, _learning_rate=0.1, 

                                      _make_histogram=False, _verbose=True):
    """Adjust inter-domain pairs by domain interaction likelihood ratio matrix,
    Inputs:
        _inter_pairs: original interaction pairs, array-like of 2-ints, which are domain indices
        _inter_dm_lk_mat: domain interation likelihood ratio matrix, should of same size as (num_domain,num_domain), invalid interactions are np.inf
        _percentage_th: threshold for exchanging precentage (default: 5, ->5%)
        _make_histogram: whether make a histrogram for likelihood ratios for selected and removed pairs, bool (default: False)
    Outputs:
        _new_pairs: updated pairs, similar structure as _inter_pairs
        _removed_num: number of removed pairs, int
        _added_num: number of newly added pairs, int
        """
    _sel_pair_mk = np.zeros(_inter_dm_lk_mat.shape)
    for _p in _inter_pairs:
        _sel_pair_mk[_p[0],_p[1]] = 1
        _sel_pair_mk[_p[1],_p[0]] = 1
    
    _sel_lks = _inter_dm_lk_mat[np.where(_sel_pair_mk==1)]
    _sel_lks = _sel_lks[np.where(np.isinf(_sel_lks)==False)]
    # excluded
    _exc_lks = _inter_dm_lk_mat[np.where(_sel_pair_mk==0)]
    _exc_lks = _exc_lks[np.where(np.isinf(_exc_lks)==False)]

    if _make_histogram:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(_sel_lks, 20, color='r', density=True, alpha=0.5)
        plt.hist(_exc_lks, 20, color='b', density=True, alpha=0.5)
        plt.show()
    # get thresholds
    _sel_low_th = scoreatpercentile(_sel_lks, _percentage_th)
    _exc_high_th = scoreatpercentile(_exc_lks, 100-_percentage_th)
    _low_th = _sel_low_th + _learning_rate * (_exc_high_th-_sel_low_th)
    _high_th = _exc_high_th - _learning_rate * (_exc_high_th-_sel_low_th)
    if _verbose:
        print(f'--- adjust inter-domain calling, low={_low_th:.3f}, high={_high_th:.3f}')
    # do exchange when _sel_low_th smaller than _exc_high_th
    if _sel_low_th < _exc_high_th:
        
        # indices for removed (previously selected, not passing select_low_th)
        _remove_inds = np.where((_sel_pair_mk==1)*(_inter_dm_lk_mat < _low_th ))
        # indices for added (previously not selected, passing excluded_high_th)
        _add_inds = np.where((_sel_pair_mk==0)*(_inter_dm_lk_mat > _high_th ))
        # count
        _removed_num = len(_remove_inds[0])
        _added_num = len(_add_inds[0])
        
        # do adjustment
        _pair_sets = []
        for _p in _inter_pairs:
            if set(_p) not in _pair_sets:
                _pair_sets.append(set(_p))
        # remove
        _rxs,_rys = _remove_inds
        for _x, _y in zip(_rxs,_rys):
            if {_x,_y} in _pair_sets:
                _i = _pair_sets.index({_x, _y})
                _pair_sets.pop(_i)
        # add
        _axs,_ays = _add_inds
        for _x, _y in zip(_axs, _ays):
            if {_x,_y} not in _pair_sets:
                _pair_sets.append({_x, _y})
            
        _pair_sets = sorted(_pair_sets, key=lambda v:(min(v),max(v)))
        _new_pairs = np.array([[min(_p),max(_p)] for _p in _pair_sets], dtype=np.int)
    else:
        _new_pairs = _inter_pairs.copy()
        _removed_num = 0
        _added_num = 0
    if _verbose:
        print(f"--- {len(_new_pairs)} pairs selected, removed:{_removed_num}, added:{_added_num}")  

    return _new_pairs, _removed_num, _added_num

def iterative_interdomain_calling(distmap, domain_starts,  
                                  exclude_neighbors=True, exclude_edges=False,
                                  init_metric='ks', init_th=0.55,
                                  w_sel=1., w_intra=0.05, max_num_iter=10, learning_rate=0.3,
                                  normalize_likelihood=True, adjust_percent_th=1., 
                                  mean_contact_ratio=0.1, contact_th=700, 
                                  keep_triu=True, keep_intensity=False, 
                                  marker_type='area', marker_param=1.,
                                  plot_process=False, plot_kwargs={'plot_limits':[0,2000]}, verbose=True):
    """Function to iteratively call interdomain"""
    # initial domain interaction calling
    _init_pairs, _mk = inter_domain_markers(distmap, domain_starts, off_diagonal_th=init_th,
                                           metric=init_metric, marker_type=marker_type, 
                                           keep_intensity=keep_intensity,
                                           keep_triu=keep_triu,
                                           exclude_neighbors=exclude_neighbors,
                                           exclude_edges=exclude_edges, 
                                           marker_param=marker_param, verbose=verbose)
    # plot porcess if specified
    if plot_process:
        from ..figure_tools.color import transparent_gradient
        from ..figure_tools.domain import plot_boundaries
        import matplotlib.pyplot as plt 
        # plot
        ax = plot_boundaries(distmap, domain_starts, figure_dpi=600, **plot_kwargs)
        ax.imshow(_mk, cmap=transparent_gradient([1,1,0]), vmin=0, vmax=2) # plot marker
        ax.set_title(f'initial calling by {init_metric}, th={init_th}')
        plt.show()
    # initialize EM loops
    _pairs = _init_pairs.copy()
    _removed_num, _added_num = np.inf, np.inf
    _n_iter = 0
    # start loop, exit when no points removed or added
    while (_removed_num > 0 or _added_num > 0):
        if _n_iter >= max_num_iter:
            break
        # calculate interdomain likelihood based on current calling
        _interdomain_lks = _interdomain_likelihood(distmap, domain_starts, _pairs, 
                                                   _w_sel=w_sel, _w_intra=w_intra,
                                                   _normalize=normalize_likelihood, 
                                                   _exclude_neighbors=exclude_neighbors)
        # adjust calling by threshold
        _pairs, _removed_num, _added_num = _adjust_interdomain_by_likelihood(
                                                _pairs, _interdomain_lks, 
                                                _make_histogram=plot_process, 
                                                _percentage_th=adjust_percent_th, 
                                                _learning_rate=learning_rate, 
                                                _verbose=verbose)
        # exclude by contacts
        _pairs = _exclude_interdomain_by_contact(distmap, domain_starts, _pairs, 
                                                _mean_contact_ratio=mean_contact_ratio, 
                                                _contact_th=contact_th, )
        # add counter
        _n_iter += 1
        # plot process if specified
        if plot_process:
            if keep_intensity:
                from ImageAnalysis3.domain_tools.distance import domain_pdists
                _dps = domain_pdists(distmap, domain_starts, metric=init_metric)
            else:
                _dps = None
            # calculate new marker
            _new_mk = _generate_inter_domain_markers(distmap, domain_starts, _dps, _pairs, 
                                                     _marker_type=marker_type, 
                                                     _keep_intensity=keep_intensity)
            # plot 
            ax = plot_boundaries(distmap, domain_starts, figure_dpi=600, **plot_kwargs)
            ax.imshow(_new_mk, cmap=transparent_gradient([1,1,0]), vmin=0, vmax=2) # plot marker
            ax.set_title(f'domain interaction iter={_n_iter}')
            plt.show()

    # finally, calculate final marker
    if not keep_triu:
        _new_pairs = []
        for _p in _pairs:
            if [_p[0], _p[1]] not in _new_pairs:
                _new_pairs.append([_p[0], _p[1]])
            if [_p[1], _p[0]] not in _new_pairs:
                _new_pairs.append([_p[1], _p[0]])
        _new_pairs = np.array(_new_pairs, dtype=np.int)
    else:
        _new_pairs = _pairs.copy()
            
    if keep_intensity:
        from ImageAnalysis3.domain_tools.distance import domain_pdists
        _dps = domain_pdists(distmap, domain_starts, metric=init_metric)
    else:
        _dps = None
    # re-generate mk            
    _new_mk = _generate_inter_domain_markers(distmap, domain_starts, _dps, _new_pairs, 
                                             _marker_type=marker_type, 
                                             _keep_intensity=keep_intensity)
    # return
    return _new_pairs, _new_mk


def _generate_inter_domain_markers(_coordinates, _domain_starts, _domain_pdists, _domain_xy, 
                                   _marker_type='center', _marker_param=1., _keep_intensity=True):
    """transform domain_xy into marker format"""
    # get domain features
    _dm_starts = np.array(_domain_starts)
    _dm_ends = np.concatenate([_dm_starts[1:], np.array([len(_coordinates)])])
    _dm_centers = ((_dm_starts + _dm_ends)/2).astype(np.int)
    _domain_xy = np.array(_domain_xy, dtype=np.int)
    # initialize marker-map
    _marker_map = np.zeros([len(_coordinates), len(_coordinates)])
    if len(_domain_xy) == 0:
        print("empty_map")
        return _marker_map
    else:
        # get intensities
        if _keep_intensity:
            _intensities = np.abs(squareform(_domain_pdists)[_domain_xy[:,0], _domain_xy[:,1]])
        else:
            _intensities = np.ones(len(_domain_xy))

        if _marker_type == 'center':
            for _dxy, _int in zip(_domain_xy, _intensities):
                _marker_map[_dm_centers[_dxy[0]], _dm_centers[_dxy[1]]] = _int
        elif _marker_type == 'gaussian':      
            for _dxy, _int in zip(_domain_xy, _intensities):
                _marker_map = visual_tools.add_source(_marker_map, pos=[_dm_centers[_dxy[0]], _dm_centers[_dxy[1]]], 
                                                      h=_int, sig=[_marker_param,_marker_param])
        elif _marker_type == 'area':
            for _dxy, _int in zip(_domain_xy, _intensities):
                _area_slice = tuple(slice(_dm_starts[_d],_dm_ends[_d]) for _d in _dxy)
                _marker_map[_area_slice] = _int
        else:
            raise ValueError(f"Wrong input for _marker_type:{_marker_type}")
    return _marker_map

def batch_iterative_interdomain(distmap_list, domain_start_list, num_threads=12,
                                exclude_neighbors=False, exclude_edges=False,
                                init_kwargs={'init_metric':'ks',
                                             'init_th':0.5,},
                                iter_kwargs={'w_sel':1.,
                                             'w_intra':0.01,
                                             'max_num_iter':10,
                                             'learning_rate':0.3,
                                             'normalize_likelihood':True,
                                             'adjust_percent_th':1.,
                                             'mean_contact_ratio':0.1, 
                                             'contact_th':700,},
                                marker_kwargs={'keep_triu':False,
                                               'keep_intensity':False,
                                               'marker_type':'area',
                                               'marker_param':1.,},
                                save=True, save_folder=None, 
                                save_name='iterative_interdomain',
                                plot_process=False, verbose=True):
    """Function to batch iterative calling inter-domain"""
    ## check inputs
    if len(distmap_list) != len(domain_start_list):
        raise IndexError(f"length of distmap_list should match domain_start_list!")
    if save and save_folder is None:
        raise ValueError(f"save_folder should be given if save specified.")
    if verbose:
        print(f"- Start batch inter-domain calling with {num_threads} threads")
        _start_time = time.time()
    # generate args
    _inter_args = []
    _kwargs = {}
    for _k,_v in init_kwargs.items():
        if _k not in _kwargs:
            _kwargs[_k] = _v
    for _k,_v in iter_kwargs.items():
        if _k not in _kwargs:
            _kwargs[_k] = _v
    for _k,_v in marker_kwargs.items():
        if _k not in _kwargs:
            _kwargs[_k] = _v
    _args = [[_distmap, _dms, exclude_neighbors, exclude_edges]
             for _distmap, _dms in zip(distmap_list, domain_start_list)]    

    _inter_args = [[iterative_interdomain_calling, _arg, _kwargs] 
                   for _arg in _args]

    with mp.Pool(num_threads) as _inter_pool:
        _results = _inter_pool.starmap(apply_kwargs, _inter_args, chunksize=1)
        _inter_pool.close()
        _inter_pool.join()
        _inter_pool.terminate()
    # extract info
    _pair_list = [_r[0] for _r in _results]
    _marker_list = [_r[1] for _r in _results]
    # save
    if save:
        # save pairs
        _pair_filename = os.path.join(save_folder, save_name+f"_pairs_{init_kwargs['init_metric']}_{init_kwargs['init_th']}.pkl")
        if verbose:
            print(f"-- saving pairs to file:{_pair_filename}")
        pickle.dump(_pair_list, open(_pair_filename,'wb'))
        # save markers
        _marker_filename = os.path.join(save_folder, save_name+f"_markers_{init_kwargs['init_metric']}_{init_kwargs['init_th']}.pkl")
        if verbose:
            print(f"-- saving markers to file:{_marker_filename}")
        pickle.dump(_marker_list, open(_marker_filename,'wb'))
    
    if verbose:
        print(f"-- time spent for interdomain calling: {time.time()-_start_time:.2f}")

    return _pair_list, _marker_list

                                                                             








def _loop_out_metric(_coordinates, _position, _domain_starts, metric='median', 
                     _dm_sz=5, _loop_out_th=0., 
                     _exclude_boundaries=True, _exclude_edges=True):
    _dm_starts = np.array(_domain_starts)
    _dm_ends = np.concatenate([_dm_starts[1:], np.array([len(_coordinates)])])
    # position
    _pos = int(_position)
    # exclude domain boudaries
    if _exclude_boundaries:
        if _pos in _dm_starts or _pos in _dm_ends-1:
            return []
    # exclude edges if specified
    if _exclude_edges:
        if _pos > _dm_starts[-1] and _dm_starts[-1] > len(_coordinates) - _dm_sz:
            return []
        elif _pos < _dm_ends[0] and _dm_ends[0] < _dm_sz:
            return []
    # initialize
    _self_dists = []
    _self_id = -1
    _dist_list = []
    for _start, _end in zip(_dm_starts, _dm_ends):
        # calculate the metric
        _dist_list.append(_coordinates[_start:_end, _pos])
        if _position > _start and _position < _end:
            _self_dists.append(_coordinates[_start:_end, _pos])
            
    if len(_self_dists) != 1:
        return []
    else:
        _loop_out_hits = []
        _self_dists = _self_dists[0]
        for _i, _dists in enumerate(_dist_list):
            if np.isnan(_dists).sum() == np.size(_dists):
                _d = np.inf
            else:
                if metric == 'median':
                    m_dist, m_self = np.nanmedian(_dists), np.nanmedian(_self_dists)
                    v_dist, v_self = np.nanmedian((_dists-m_dist)**2),\
                                    np.nanmedian((_self_dists-m_self)**2)
                    if v_dist+v_self == 0:
                        _d = np.inf
                    else:
                        _d = (m_dist-m_self) / np.sqrt(v_dist+v_self)
                elif metric == 'mean':
                    m_dist, m_self = np.nanmean(_dists), np.nanmean(_self_dists)
                    v_dist, v_self = np.nanvar(_dists), np.nanvar(_self_dists)
                    if v_dist+v_self == 0:
                        _d = np.inf
                    else:
                        _d = (m_dist-m_self) / np.sqrt(v_dist+v_self)                
                elif metric == 'ks':
                    if 'ks_2samp' not in locals():
                        from scipy.stats import ks_2samp
                    _f = np.sign((np.nanmedian(_dists) - np.nanmedian(_self_dists)))
                    _d = _f * ks_2samp(_self_dists, _dists)[0]
                else:
                    raise ValueError(f"unsupported metric:{metric}")
            # decide if its a hit
            if _d < _loop_out_th:
                ## totally skip first domain
                #if _i == 0:
                #    continue

                if _exclude_edges:
                    if _i == 0 and _dm_ends[0] < _dm_sz:
                        continue
                    elif _i==len(_dm_starts)-1 and _dm_starts[-1] > len(_coordinates) - _dm_sz:
                        continue

                _loop_out_hits.append(_i)
                
        return _loop_out_hits
                
def _generate_loop_out_markers(_coordinates, _domain_starts, _loop_region_domain_pairs, 
                               _marker_type='center', _marker_param=1., _keep_triu=True, 
                               _normalize=True, _verbose=True):
    """transform domain_xy into marker format"""
    # get domain features
    _dm_starts = np.array(_domain_starts)
    _dm_ends = np.concatenate([_dm_starts[1:], np.array([len(_coordinates)])])
    _dm_centers = ((_dm_starts + _dm_ends)/2).astype(np.int)
    # initialize marker-map
    _marker_map = np.zeros([len(_coordinates), len(_coordinates)])
    if len(_loop_region_domain_pairs) == 0:
        if _verbose:
            print(f"---- no loop given, return empty marker map.")
        return _marker_map
    else:
        if _verbose:
            print(f"--- generate loop-out marker for {len(_loop_region_domain_pairs)} loops")
        

        if _marker_type == 'center':
            for _reg, _dm in _loop_region_domain_pairs:
                _marker_map[_reg, _dm_centers[_dm]] = 1
                if not _keep_triu:
                    _marker_map[_dm_centers[_dm], _reg] = 1
        elif _marker_type == 'gaussian':      
            if _normalize:
                _intensity = 1/_marker_param
            else:
                _intensity = 1 
            for _reg, _dm in _loop_region_domain_pairs:
                _marker_map = visual_tools.add_source(_marker_map, pos=[_reg, _dm_centers[_dm]], 
                                                          h=_intensity, sig=[_marker_param,_marker_param])
                if not _keep_triu:
                    _marker_map = visual_tools.add_source(_marker_map, pos=[_dm_centers[_dm], _reg], 
                                                          h=_intensity, sig=[_marker_param,_marker_param])
        elif _marker_type == 'area':
            for _reg, _dm in _loop_region_domain_pairs:
                if _normalize:
                    _intensity =  1 / (_dm_ends[_dm] - _dm_starts[_dm])
                else:
                    _intensity = 1 
                _marker_map[_reg, _dm_starts[_dm]:_dm_ends[_dm]] = _intensity
                if not _keep_triu:
                    _marker_map[_dm_starts[_dm]:_dm_ends[_dm], _reg] = _intensity
        else:
            raise ValueError(f"Wrong input _marker_type:{_marker_type}")
    return _marker_map

    
def loop_out_markers(coordinates, domain_starts, norm_mat=None, metric='median',
                     loop_out_th=0., domain_size=5, marker_type='center', 
                     marker_param=1., keep_triu=True,
                     exclude_boundaries=True, exclude_edges=True, 
                     normalize=True, verbose=True):
    ## check inputs
    _coordinates = np.array(coordinates)
    _domain_starts = np.array(domain_starts, dtype=np.int)
    if verbose:
        _start_time = time.time()
        print(f"-- calculate loop-out for {len(_domain_starts)} domains in {len(_coordinates)} regions")
    if len(_coordinates.shape) != 2:
        raise IndexError(f"Wrong input shape for coordinates, it should be 2d-array but shape:{_coordinates.shape} is given.")
    if _coordinates.shape[1] == 3: # in zxy format
        _coordinates = squareform(pdist(_coordinates))
    
    # initialze
    _loop_region_domain_pairs = []
    for _pos in range(len(_coordinates)):
        _loop_dms = _loop_out_metric(_coordinates, _pos, domain_starts, 
                          _loop_out_th=loop_out_th, metric=metric, _dm_sz=domain_size,
                          _exclude_boundaries=exclude_boundaries, _exclude_edges=exclude_edges)
        if len(_loop_dms) > 0:
            for _dm in _loop_dms:
                _loop_region_domain_pairs.append([_pos, _dm])

    # generate marker
    _loop_marker = _generate_loop_out_markers(_coordinates, _domain_starts, 
                                              _loop_region_domain_pairs, _marker_type=marker_type,
                                              _marker_param=marker_param, _keep_triu=keep_triu, 
                                              _normalize=normalize, _verbose=verbose)
    if verbose:
        print(f"--- {len(_loop_region_domain_pairs)} loops identified, time:{time.time()-_start_time:2.3}")
    
    return _loop_region_domain_pairs, _loop_marker

def apply_kwargs(function, args, kwargs):
    return function(*args, **kwargs)


def call_domain_contact(zxys, domain_starts, cutoff=2):
    """Function to call domain pairs"""
    
    bad = np.isnan(zxys[:,0])
    zxys_ = zxys[~bad]
    mat=squareform(pdist(zxys_))
    dom_starts = np.zeros(len(zxys))
    dom_starts[domain_starts[:-1]]=1
    dom_starts = list(np.where(dom_starts[~bad])[0])+[len(zxys_)]
    dom_pairs=[]
    for i in range(len(dom_starts)-1):
        for j in range(i):
            in_i = squareform(mat[dom_starts[i]:dom_starts[i+1],dom_starts[i]:dom_starts[i+1]])
            in_j = squareform(mat[dom_starts[j]:dom_starts[j+1],dom_starts[j]:dom_starts[j+1]])
            in_doms = np.median(np.concatenate([in_i,in_j]))
            out_doms = np.median(mat[dom_starts[i]:dom_starts[i+1],dom_starts[j]:dom_starts[j+1]])
            ins_score = out_doms/in_doms
            if ins_score<cutoff:
                dom_pairs.append([i,j])
                
    return dom_pairs