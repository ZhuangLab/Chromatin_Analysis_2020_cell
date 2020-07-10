import sys,glob,os,time,copy
import numpy as np
import pickle as pickle
import multiprocessing as mp
import psutil
from scipy import ndimage, stats
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree, is_valid_linkage
from scipy.spatial.distance import pdist, cdist, squareform
from functools import partial
from scipy.ndimage.filters import maximum_filter
import matplotlib.pyplot as plt

from . import *
from . import _distance_zxy
from .External import Fitting_v3, DomainTools

# default init to make it a package
def __init__():
    pass

def Calculate_BED_to_Region(data_filename, region_dic, data_format='tagAlign', 
                            stat_type='count', normalize_by_total=False,
                            overwrite=False, save=False, save_folder=None, verbose=True):
    """Function to formulate a BED-like format data alignment result into statistics in given regions
    -------------------------------------------------------------------------------------------------
    Inputs:
        data_filename: full filename for BED-like data, str 
        region_dic: dictionary of regions that have chr,start,end info, dict 
            (For example, dict from get_img_info.Load_Region_Positions)
        data_format: format of this file, str (default: 'tagAlign')
        stat_type: type of statistics to collect, str ({'sum', 'count'}) 
        overwrite: whether overwrite saved stat dict, bool (default: False)
        save: whether save stat_dictionary into a file, bool (default: False)
        save_folder: where to save this result, str (default: None, which means same folder as data)
        verbose: say something!, bool (default: True)
    Outputs:
        _region_stat: region statistics, dict of region id (save from region_dic) 
            to the statistics at this region
    -------------------------------------------------------------------------------------------------
    """
    ## check inputs
    _start_time = time.time()
    # filename
    if not isinstance(data_filename, str):
        raise TypeError(f"Wrong input type for data_filename, should be str, but {type(data_filename)} is given.")
    elif not os.path.isfile(data_filename):
        raise IOError(f"Input data_file: {data_filename} doesn't exist, exit!")
    # region_dic
    if not isinstance(region_dic, dict):
        raise TypeError(f"Wrong input type for region_dic, should be dict, but {type(region_dic)} is given!")
    # data_format:
    _allowed_formats = ['bed', 'tagalign', 'wig']
    if not isinstance(data_format, str):
        raise TypeError(f"Wrong input type for data_format, should be str, but {type(data_format)} is given.")
    elif data_format.lower() not in _allowed_formats:
        raise ValueError(f"data_format:{data_format} should be among {_allowed_formats}")
    # stat_type
    _allowed_stats = ['count', 'sum']
    if not isinstance(stat_type, str):
        raise TypeError(f"Wrong input type for stat_type, should be str, but {type(stat_type)} is given.")
    elif stat_type.lower() not in _allowed_stats:
        raise ValueError(f"data_format:{stat_type} should be among {_allowed_stats}")
    
    # sort region dic
    _region_info = list(sorted(region_dic.items(), key=lambda v:int(v[1]['start'])))
    
    # not overwrite?
    _save_filename = data_filename.replace(data_format, f'_{stat_type}_region.pkl')
    if save_folder is not None:
        _save_filename = os.path.join(save_folder, os.path.basename(_save_filename))
    if os.path.isfile(_save_filename) and not overwrite:
        _region_stat = pickle.load(open(_save_filename,'rb'))
        return _region_stat
    # otherwise initialize with 0
    else:
        if verbose:
            print(f"-- Calculate {os.path.basename(data_filename)} for {len(region_dic)} regions!")
        
        _region_stat = {_reg:0 for _reg,_info in sorted(region_dic.items())}

    # load info
    _content = []
    _reg_index = 0
    if normalize_by_total:
        total_stat = 0.
    # loop through lines and add to stat
    if verbose:
        print(f"--- start iterate through data file.")
    with open(data_filename, 'r') as _handle:
        for line in _handle:
            _c = line.strip().split()
            if data_format.lower() == 'tagalign' or data_format.lower() == 'bed':
                _chr = _c[0]
                _start = int(_c[1])
                _end = int(_c[2])
                _mid = int((_start+_end)/2)
                _count = float(_c[4])
            elif data_format.lower() == 'wig':
                if '#' in line:
                    continue
                _chr = _c[0]
                _start = int(_c[1])
                _end = int(_c[2])
                _count = int(_c[3])
                _mid = int((_start+_end)/2)
            # save info to total_stat
            if normalize_by_total:
                if stat_type == 'sum':
                    total_stat += float(np.abs(min(_end, _region_info[_reg_index][1]['end']-1) - \
                        max(_start, _region_info[_reg_index][1]['start']))) * _count * 1e-10
                elif stat_type == 'count':
                    total_stat += _count * 1e-10
            # directly loop through all possible regions, add region_stats
            for _reg in _region_stat:
                pass
            
            # go to next region if:
            if _chr != _region_info[_reg_index][1]['chr']:
                continue
            if _mid < _region_info[_reg_index][1]['start']:
                continue
            if _mid > _region_info[_reg_index][1]['end'] and _chr == _region_info[_reg_index][1]['chr']:
                _reg_index += 1
            if _reg_index >= len(_region_info):
                break
            # really calculate stats here                
            if _mid >= _region_info[_reg_index][1]['start'] \
                and _mid < _region_info[_reg_index][1]['end'] \
                and _chr == _region_info[_reg_index][1]['chr']:
                if stat_type == 'sum':
                    _region_stat[_region_info[_reg_index][0]] += \
                        float(np.abs(min(_end, _region_info[_reg_index][1]['end']-1) - \
                        max(_start, _region_info[_reg_index][1]['start']))) * _count * 1e-10
                elif stat_type == 'count':
                    _region_stat[_region_info[_reg_index][0]] += _count * 1e-10
    # sort back regions
    _region_stat = {_k:_v for _k,_v in sorted(_region_stat.items())}
    # normalize if applied
    if normalize_by_total:
        if verbose:
            print(f"--- total count of this dataset: {total_stat * 1e10}")
        _region_stat = {_k: float(_v) / float(total_stat) for _k, _v in _region_stat.items()}    
    else:
        _region_stat = {_k: float(_v) *1e10 for _k, _v in _region_stat.items()}
    if save:
        if verbose:
            print(f"--- save result into file: {_save_filename}")
        pickle.dump(_region_stat, open(_save_filename, 'wb'))    
    
    if verbose:
        print(f"--- time spent in {data_format} loading: {time.time()-_start_time}")
        
    return _region_stat
    
    
## Compartment Analysis
#-----------------------------------------------------------------------------------
def is_in_hull(ref_zxys, zxy, remove_self=True):
    """Check if point zxy in ref_zxys
    either zxy or ref_zxys should be 3d ZXY coordinates"""
    if 'ConvexHull' not in locals():
        from scipy.spatial import ConvexHull
    if len(np.shape(zxy)) != 1:
        raise ValueError(f"Wrong input dimension for p, should be 1d")

    # Remove Nan in ref_zxys
    ref_zxys = np.array(ref_zxys) # convert to array
    _kept_rows = np.isnan(ref_zxys).sum(axis=1) == 0
    # remove itself
    if remove_self:
        for _i, _ref_zxy in enumerate(ref_zxys):
            if (_ref_zxy == np.array(zxy)).all():
                _kept_rows[_i] = False
    # apply remove
    _kept_ref_zxys = ref_zxys[_kept_rows]
    if len(_kept_ref_zxys) <= 3:
        print('Not enough points to create convex hull.')
        return False
    # create hull for ref_zxys
    _hull = ConvexHull(_kept_ref_zxys)
    # create hull for ref_zxys + zxy
    _extend_zxys = np.concatenate([_kept_ref_zxys, zxy[np.newaxis,:]])
    _extend_hull = ConvexHull(np.array(_extend_zxys))
    if list(_hull.vertices) == list(_extend_hull.vertices):
        return True
    else:
        return False

# basic function to wrap is_in_hull to do bootstrap
def _bootstrap_region_in_domain(_dm_zxys, _reg_zxy, _sampling_size, 
                                _n_iter=100, _remove_self=True):
    if np.isnan(_reg_zxy).any():
        return np.nan
    else:
        _p = []
        for _i in range(_n_iter):
            _bt_inds = np.random.choice(len(_dm_zxys), _sampling_size, replace=False)
            _bt_zxys = _dm_zxys[np.sort(_bt_inds)]
            # check if inhull
            _p.append( is_in_hull(_bt_zxys, _reg_zxy) )
        return np.nanmean(_p)

def Bootstrap_regions_in_domain(chrom_zxy_list, region_index, domain_indices, 
                               p_bootstrap=0.25, n_iter=100, num_threads=12,
                               verbose=True):
    """Estimate how much a region is enclosed by domain/compartments,
        across all chromosomes in chrom_zxy_list
    Inputs:
        chrom_zxy_list: list of chromosome zxy coordinates, list of 2darray
        region_index: index of region in chromosome to be calculated, int
        domain_indices: array-like region indices for a certain domain/compartment, array-like of ints
        p_bootstrap: subsample percentage for bootstap steps, float (default:0.25)
        n_iter: number of bootstrap iterations, int (default: 100)
        verbose: say something!, bool (default: True)
    Outputs:
        _region_probs: proabilities of region in domain, np.ndarray (1d)
        """
    ## check inputs
    _start_time = time.time()
    if not isinstance(chrom_zxy_list, list):
        raise TypeError(f"Wrong input type for chrom_zxy_list, should be list but {type(chrom_zxy_list)} is given.")
    if not isinstance(region_index, int):
        try:
            region_index = int(region_index)
        except:
            raise TypeError(f"Wrong input region_index:{region_index}, cannot convert to int.")
    # convert domain indices into array with ints
    domain_indices = np.array(domain_indices, dtype=np.int)
    if np.max(domain_indices) > len(chrom_zxy_list[0]):
        raise ValueError(f"Wrong input for domain_indices, no indices should be larger than zxy length")
    if verbose:
        print(f"-- Start boostrap sample for region:{region_index} in regions group of {len(domain_indices)}")
    # check p_bootstrap and n_iter
    if not isinstance(p_bootstrap, float):
        p_bootstrap = float(p_bootstrap)
    elif p_bootstrap <= 0 or p_bootstrap >= 1:
        raise ValueError(f"Wrong p_bootstrap={p_bootstrap}, should be float between 0 and 1")
    _sampling_size = int(np.ceil(len(domain_indices)*p_bootstrap))
    if _sampling_size == len(domain_indices):
        _sampling_size -= 1
    if verbose:
        print(f"--- boostrap sampling p={p_bootstrap}, size={_sampling_size}")
    if not isinstance(n_iter, int):
        try:
            n_iter = int(n_iter)
        except:
            raise TypeError(f"Wrong input n_iter:{n_iter}, cannot convert to int.")
    
    _region_spots = [_zxys[region_index] for _zxys in chrom_zxy_list]
    _region_probs = Bootstrap_spots_in_domain(chrom_zxy_list, _region_spots, domain_indices,
                                              p_bootstrap=p_bootstrap, n_iter=n_iter, num_threads=num_threads,
                                              verbose=False)
    if verbose:
        print(f"--- time spent in boostrap: {time.time()-_start_time}")
    return _region_probs

def Bootstrap_spots_in_domain(chrom_zxy_list, spot_zxy_list, domain_indices, 
                              num_threads=12,
                              p_bootstrap=0.25, n_iter=100, remove_self=True,
                              verbose=True):
    """Estimate how much a region is enclosed by domain/compartments,
        across all chromosomes in chrom_zxy_list
    Inputs:
        chrom_zxy_list: list of chromosome zxy coordinates, list of 2darray
        spot_zxy_list: list of zxy of spots to be calculated in corresponding chromosomes, list of 1darray
        domain_indices: array-like region indices for a certain domain/compartment, array-like of ints
        p_bootstrap: subsample percentage for bootstap steps, float (default:0.25)
        n_iter: number of bootstrap iterations, int (default: 100)
        remove_self: whether remove itself, bool (default: True)
        verbose: say something!, bool (default: True)
    Outputs:
        _region_probs: proabilities of region in domain, np.ndarray (1d)
        """
    ## check inputs
    _start_time = time.time()
    if not isinstance(chrom_zxy_list, list) and not isinstance(chrom_zxy_list, np.ndarray):
        raise TypeError(f"Wrong input type for chrom_zxy_list, should be list/array but {type(chrom_zxy_list)} is given.")
    if not isinstance(spot_zxy_list, list) and not isinstance(spot_zxy_list, np.ndarray):
        raise TypeError(f"Wrong input type for spot_zxy_list, should be list/array but {type(spot_zxy_list)} is given.")
    if len(chrom_zxy_list) != len(spot_zxy_list):
        raise IndexError(f"Length of chrom_zxy_list:{len(chrom_zxy_list)} doesn't match length of spot_zxy_list:{len(spot_zxy_list)}")
    
    # convert domain indices into array with ints
    domain_indices = np.array(domain_indices, dtype=np.int)
    if np.max(domain_indices) > len(chrom_zxy_list[0]):
        raise ValueError(f"Wrong input for domain_indices, no indices should be larger than zxy length")
    if verbose:
        print(f"-- Start boostrap sample for region among {len(domain_indices)} regions")
    # check p_bootstrap and n_iter
    if not isinstance(p_bootstrap, float):
        p_bootstrap = float(p_bootstrap)
    elif p_bootstrap <= 0 or p_bootstrap >= 1:
        raise ValueError(f"Wrong p_bootstrap={p_bootstrap}, should be float between 0 and 1")
    _sampling_size = int(np.ceil(len(domain_indices)*p_bootstrap))
    if _sampling_size == len(domain_indices):
        _sampling_size -= 1
    if verbose:
        print(f"--- boostrap sampling p={p_bootstrap}, size={_sampling_size}")
    if not isinstance(n_iter, int):
        try:
            n_iter = int(n_iter)
        except:
            raise TypeError(f"Wrong input n_iter:{n_iter}, cannot convert to int.")
    
    ## Start iteration
    _boostrap_args = []
    
    # loop through chromosomes to get args
    for _chrom_zxys, _spot_zxy in zip(chrom_zxy_list, spot_zxy_list):
        _dm_zxys = np.array(_chrom_zxys)[domain_indices]
        _spot_zxy = np.array(_spot_zxy)
        _boostrap_args.append( (_dm_zxys, _spot_zxy, _sampling_size, n_iter, remove_self))
    
        #_p = postanalysis._bootstrap_region_in_domain(_dm_zxys,_reg_zxy,_sampling_size,n_iter)
    # calculate through multi-processing
    with mp.Pool(num_threads) as _pool:
        if verbose:
            print(f"--- {len(_boostrap_args)} chromosomes processing by {num_threads} threads.")
        _region_probs = _pool.starmap(_bootstrap_region_in_domain, 
                                      _boostrap_args, chunksize=1)
        _pool.close()
        _pool.join()
        _pool.terminate()    
    if verbose:
        print(f"--- time spent in boostrap: {time.time()-_start_time}")
    return _region_probs


# calculate region genomic_scaling
def region_genomic_scaling(coordinates, inds, 
                           genomic_distance_matrix=r'Z:/References/genomic_positions.npy',
                           gaussian=0, make_plot=True, verbose=True):
    """Calculate scaling with given coordinates and inds"""
    ## check inputs
    coordinates = np.array(coordinates).copy()
    if verbose:
        print(f"-- start calculate region scaling to genomic distance with", end=' ')
    if len(np.shape(coordinates)) != 2:
        raise ValueError(
            f"Wrong input shape for coordinates, should be 2d but {len(np.shape(coordinates))} is given")
    elif np.shape(coordinates)[0] == np.shape(coordinates)[1]:
        if verbose:
            print(f"distance map")
        if gaussian > 0:
            from astropy.convolution import Gaussian2DKernel, convolve
            _kernel = _kernel = Gaussian2DKernel(x_stddev=gaussian)
            coordinates = convolve(coordinates, _kernel)
        _mat = coordinates
    elif np.shape(coordinates)[1] == 3:
        if verbose:
            print(f"3d coordinates")
        if gaussian > 0:
            from domain_tools import interpolate_chr
            coordinates = interpolate_chr(coordinates, gaussian=gaussian)
        _mat = squareform(pdist(coordinates))
    else:
        raise ValueError(
            f"Input coordinates should be distance-matrix or 3d-coordinates!")

    # load genomic_distance_matrix
    if isinstance(genomic_distance_matrix, str) and os.path.isfile(genomic_distance_matrix):
        genomic_distance_matrix = np.load(genomic_distance_matrix)
    elif isinstance(genomic_distance_matrix, np.ndarray) and np.shape(_mat)[0] == np.shape(genomic_distance_matrix)[0]:
        pass
    else:
        raise ValueError(f"Wrong input of genomic_distance_matrix, it should be \
                         either 2d matrix having same size as coordinates, or path to file")
    # inds
    inds = np.array(inds, dtype=np.int)

    # splice used indices
    _sel_mat = _mat[inds][:, inds]
    _sel_genomic = genomic_distance_matrix[inds][:, inds]
    _transform = np.triu_indices(len(_sel_mat), 1)
    _sel_mat = _sel_mat[_transform]
    _sel_genomic = _sel_genomic[_transform]
    _keep = (np.isnan(_sel_mat) == False) * (np.isnan(_sel_genomic) == False)
    _sel_mat, _sel_genomic = _sel_mat[_keep], _sel_genomic[_keep]
    if verbose:
        print(f"--- number of indices to regress: {len(_sel_mat)}")
    # linear regression
    lr = stats.linregress(np.log(_sel_genomic), np.log(_sel_mat))
    # plot
    if make_plot:
        plt.figure()
        plt.plot(_sel_genomic, _sel_mat, '.', alpha=0.3)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
    return lr.slope, lr.intercept, lr.rvalue

# Assign single-cell domain clusters into reference compartments
def assign_domain_cluster_to_compartments(coordinates, domain_starts, compartment_dict,
                                          domain_linkage=None, linkage_method='complete',
                                          distance_metric='median', normalization=None,
                                          min_cluster_size_ratio=0.1, min_cluster_dist_ratio=0.08,
                                          assign_method='binary', return_boundary=True, verbose=True):
    """Function to assign domain clusters to given compartments in compartment_dict
    Idea: 1. find normalized overlap ratio between domain_cluster and reference_compartment, 
          2. assign bestmatch for each cluster
    ------------------------------------------------------------------------------------------
    Inputs:
        coordinates: distance map or zxy coordinates for a chromosome, np.ndarray (or like)
        domain_starts: indices of domain start regions in this chromosome, np.ndarray(1d)
        compartment_dict: dictionary for compartment annotation, dict
            Note: this comaprtment_dict has to be exclusive
        domain_linkage: linkage matrix generated from scipy.cluster.hierarchy.linkage, np.ndarray
            (linkage result, default:None, generate from scratch)
        linkage_method: method for linkage if domain_linkage is not given, str (default: 'complete')
        distance_metric: metric for domain distance calculation, str (default: 'median')
        min_cluster_size_ratio: minimal size of cluster ratio to chromosome size, float (default: 0.1)
        min_cluster_dist_ratio: minimal distance of cluster ratio to number of domains, float (default: 0.05)
        assign_method: method for assigning compartments, str {'binary'|'continuous'}
        verbose: whether say something!, bool (default: True)
    Output:
        _assigned_dict: assigned compartment label -> region id list dictionary, dict
    """
    ## check inputs
    # coordinate
    coordinates = np.array(coordinates)
    if verbose:
        print(f"-- assign domain-clusters to compartments with", end=' ')
    if len(np.shape(coordinates)) != 2:
        raise ValueError(
            f"Wrong input shape for coordinates, should be 2d but {len(np.shape(coordinates))} is given")
    elif np.shape(coordinates)[0] == np.shape(coordinates)[1]:
        if verbose:
            print(f"distance map")
        _mat = coordinates
    elif np.shape(coordinates)[1] == 3:
        if verbose:
            print(f"3d coordinates")
        _mat = squareform(pdist(coordinates))
    else:
        raise ValueError(
            f"Input coordinates should be distance-matrix or 3d-coordinates!")
    # domain_starts
    domain_starts = np.array(domain_starts, dtype=np.int)
    for _s in domain_starts:
        if _s < 0 or _s > _mat.shape[0]:
            raise ValueError(
                f"Wrong input domain_starts: {_s}, should be index of coordinates")
    domain_ends = np.zeros(np.shape(domain_starts))
    domain_ends[:-1] = domain_starts[1:]
    domain_ends[-1] = _mat.shape[0]
    # compartment_dict
    _ref_inds = []
    for _k, _v in compartment_dict.items():
        _ref_inds += list(_v)
    _uids, _ucounts = np.unique(_ref_inds, return_counts=True)
    if (_ucounts > 1).any():
        raise ValueError(
            f"There are non unique ids used in reference:{compartment_dict}")
    elif (_uids > _mat.shape[0]).any():
        raise ValueError(
            f"Wrong ind given in compartment_dict:{compartment_dict}, should be index of coordinates")
    # domain_linkage
    if domain_linkage is not None and not is_valid_linkage(domain_linkage):
        raise ValueError(
            f"domain_liknage should be a linkage type array from scipy.cluster.hierarchy.linkage")
    elif domain_linkage is None:
        _dom_pdists = domain_tools.distance.domain_pdists(coordinates, domain_starts,
                                                 metric=distance_metric, 
                                                 normalization_mat=normalization)
        _cov_mat = np.corrcoef(squareform(_dom_pdists))
        try:
            domain_linkage = linkage(_cov_mat, method=linkage_method)
        except ValueError:
            print(f"failed to build linkage, exit.")
            if return_boundary:
                return None, None
            else:
                return None
    # assign_method
    _allowed_assign_method = ['binary', 'continuous']
    assign_method = str(assign_method).lower()
    if assign_method not in _allowed_assign_method:
        raise ValueError(f"Wrong input assign_method:{assign_method}, should be within {_allowed_assign_method}")
    ## 1. acquire exclusive clusters
    # get all subnodes
    _rootnode, _nodelist = to_tree(domain_linkage, rd=True)
    # get selection threshold
    _dist_th = len(domain_starts) * min_cluster_dist_ratio
    if verbose:
        print(f"--- threshold for cluster distance={_dist_th}")
    # init kept clusters
    _kept_clusters = []
    for _node in _nodelist:
        _kept_leafs = []
        for _n in _kept_clusters:
            _kept_leafs += list(_n.pre_order(lambda x: x.id))
        _left_flag, _right_flag = True, True
        if not _node.is_leaf() and _node.dist > _dist_th:
            for _r in _node.left.pre_order(lambda x: x.id):
                if _r in _kept_leafs:
                    _left_flag = False
                    continue
            for _r in _node.right.pre_order(lambda x: x.id):
                if _r in _kept_leafs:
                    _right_flag = False
                    continue
            # otherwise, keep
            if _left_flag:
                _kept_clusters.append(_node.left)
            if _right_flag:
                _kept_clusters.append(_node.right)
    # convert domain ID to region_id
    _reg_id_list = []
    for _n in _kept_clusters:
        _dom_ids = np.array(_n.pre_order(lambda x: x.id), dtype=np.int)
        _reg_ids = [np.arange(domain_starts[_d], domain_ends[_d]).astype(
            np.int) for _d in _dom_ids]
        _reg_id_list.append(np.concatenate(_reg_ids))

    ## 2. with selected clusters, calculate its overlap with compartments
    # init
    _decision_dict = {_k: np.zeros(len(_reg_id_list))
                      for _k in compartment_dict.keys()}
    for _ckey, _cinds in compartment_dict.items():
        for _j, _rids in enumerate(_reg_id_list):
            _decision_dict[_ckey][_j] = len(np.intersect1d(
                _rids, _cinds)) / len(_rids) / len(_cinds)
    if verbose:
        print("--- decision_dict:", _decision_dict)



    ## summarize to a dict
    _assigned_dict = {_k: np.zeros(_mat.shape[0]) for _k in compartment_dict.keys()}
    _keys = list(compartment_dict.keys())
    if assign_method == 'binary':
        for _j, _rids in enumerate(_reg_id_list):
            _match_ind = np.argmax([_v[_j] for _k, _v in _decision_dict.items()])
            _assigned_dict[_keys[_match_ind]][_rids] = 1
    elif assign_method == 'continuous':
        _norm_mat = np.stack([_v for _k,_v in _decision_dict.items()])
        _norm_mat = _norm_mat / np.sum(_norm_mat, 0)
        _norm_mat[np.isnan(_norm_mat)] = 0
        for _j, _rids in enumerate(_reg_id_list):
            for _i,_k in enumerate(_keys):
                _assigned_dict[_k][_rids] = _norm_mat[_i,_j]
    # return
    if return_boundary:
        # calculate compartment boundaries
        _boundary_dict = {_k:[] for _k in compartment_dict.keys()}
        for _k, _v in _assigned_dict.items():
            _bds = np.where((_v[1:]-_v[:-1])>0)[0]+1
            _boundary_dict[_k] = _bds
        _cluster_bds = np.concatenate(list(_boundary_dict.values()))
        _cluster_bds = np.unique(_cluster_bds)
        return _assigned_dict, _cluster_bds
    else:
        return _assigned_dict

def Batch_Assign_Domain_Clusters_To_Compartments(coordinate_list, domain_start_list, compartment_dict,
                                                 num_threads=12, linkage_list=None,
                                                 linkage_method='complete', distance_metric='median',
                                                 normalization=r'Z:\References\normalization_matrix.npy',
                                                 min_cluster_size_ratio=0.1, min_cluster_dist_ratio=0.08,
                                                 assign_method='binary', return_boundary=True, verbose=True):
    """Function to batch assign domain clusters into annotated compartments
    Inputs:
        coordinate_list: list of coordinates:
            n-by-3 coordinates for a chromosome, or n-by-n distance matrix, np.ndarray
        domain_start_list: list of domain start indices:
            domain starts region-indices, np.ndarray
        compartment_dict: dictionary for compartment annotation, dict
            Note: this comaprtment_dict has to be exclusive
        num_threads: number of threads to multiprocess domain calling, int (default: 12)
        linkage_list: list of linkage matrix generated from scipy.cluster.hierarchy.linkage, np.ndarray
            linkage result, default:None, generate from scratch
        linkage_method: method for linkage if domain_linkage is not given, str (default: 'complete')
        distance_metric: metric for domain distance calculation, str (default: 'median')
        min_cluster_size_ratio: minimal size of cluster ratio to chromosome size, float (default: 0.1)
        min_cluster_dist_ratio: minimal distance of cluster ratio to number of domains, float (default: 0.05)
        assign_method: method for assigning compartments, str {'binary'|'continuous'}
        return_boundary: whether return domain-cluster boundaries, bool (default: True)
        verbose: whether say something!, bool (default: True)
    Output:
        _assigned_list: list of assigned dict,
            compartment_label -> 1d-array of size similar to coordinate, with compartment score at each position
    """
    ## check inputs
    if verbose:
        _start_time = time.time()
        print(
            f"- Start batch assign domain clusters into {len(compartment_dict)} compartments.")
    # check coordinate_list
    if isinstance(coordinate_list, list) and len(np.shape(coordinate_list[0])) == 2:
        pass
    elif isinstance(coordinate_list, np.ndarray) and len(np.shape(coordinate_list)) == 3:
        pass
    else:
        raise ValueError(
            f"Input coordinate_list should be a list of 2darray or 3dim merged coordinates")
    # check other input length
    if len(domain_start_list) != len(coordinate_list):
        raise ValueError(
            f"length of domain_start_list:{len(domain_start_list)} doesn't match coordinate_list!")
    # check other input length
    if linkage_list is not None:
        if len(linkage_list) != len(coordinate_list):
            raise ValueError(
                f"length of linkage_list:{len(linkage_list)} doesn't match coordinate_list!")
    # compartment_dict
    _ref_inds = []
    for _k, _v in compartment_dict.items():
        _ref_inds += list(_v)
    _uids, _ucounts = np.unique(_ref_inds, return_counts=True)
    if (_ucounts > 1).any():
        raise ValueError(
            f"There are non unique ids used in reference:{compartment_dict}")
    elif (_uids > len(coordinate_list[0])).any():
        raise ValueError(
            f"Wrong ind given in compartment_dict:{compartment_dict}, should be index of coordinates")
    # load normalization if specified
    if isinstance(normalization, str) and os.path.isfile(normalization):
        normalization = np.load(normalization)
    elif isinstance(normalization, np.ndarray) and len(coordinate_list[0]) == np.shape(normalization)[0]:
        pass
    else:
        normalization = None

    # get args
    _assign_args = []
    for _i, _coordinates in enumerate(coordinate_list):
        _starts = domain_start_list[_i]
        if linkage_list is None:
            _linkage = None
        else:
            _linkage = linkage_list[_i]
        _assign_args.append((_coordinates, _starts, compartment_dict,
                             _linkage, linkage_method, distance_metric, normalization,
                             min_cluster_size_ratio, min_cluster_dist_ratio,
                             assign_method, True, verbose))

    # multi-processing
    if verbose:
        print(
            f"-- multiprocessing of {len(_assign_args)} domain_cluster assignment with {num_threads} threads")
    with mp.Pool(num_threads) as _assign_pool:
        _results = _assign_pool.starmap(assign_domain_cluster_to_compartments, _assign_args)
        _assign_pool.close()
        _assign_pool.join()
        _assign_pool.terminate()
    # get results
    _assigned_list = [_r[0] for _r in _results]

    if verbose:
        print(
            f"-- time spent in domain_cluster assignment:{time.time()-_start_time}")
    # return
    if return_boundary:
        _boundary_list = [_r[1] for _r in _results]
        return _assigned_list, _boundary_list
    else:
        return _assigned_list


# density map




def score_from_density(density_dict, cutoff_per=50):
    A_im = density_dict['A']
    B_im = density_dict['B']
    A_mask = A_im > stats.scoreatpercentile(A_im[A_im > 0],cutoff_per)
    B_mask = B_im > stats.scoreatpercentile(B_im[B_im > 0],cutoff_per)
    score = np.sqrt((1- np.sum(A_mask*B_mask) / np.sum(A_mask))*(1- np.sum(A_mask*B_mask) / np.sum(B_mask)))
    #print(np.sum(A_mask), np.sum(B_mask), np.sum(A_mask*B_mask))
    #score = np.sum(A_mask*B_mask) / np.sqrt(np.sum(A_mask)) / np.sqrt(np.sum(B_mask))
    #plt.figure()
    #plt.imshow(A_mask.sum(2),cmap=myReds)
    #plt.imshow(B_mask.sum(2),cmap=myBlues,alpha=0.5)
    return score

def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x, edge_order=2) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k, edge_order=2) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian


def local_maximum_in_density(den_dict, seeding_window=10, intensity_ratio=0.25):
    """Calculate local maxima """
    A_density, B_density = den_dict['A'], den_dict['B']

    A_z,A_x,A_y = np.where(maximum_filter(A_density,seeding_window) == A_density)
    A_int = A_density[A_z,A_x,A_y]
    A_hessian = hessian(A_density)[:,:, A_z, A_x, A_y]
    A_eigs = [np.linalg.eigvals(A_hessian[:,:,_i]) for _i in range(len(A_x))]
    A_coords = np.array([[_z,_x,_y] for _z,_x,_y,_e,_int in zip(A_z,A_x,A_y,A_eigs,A_int) \
                            if (_e<0).all() and _int > intensity_ratio*np.max(A_int)], dtype=np.int)
    B_z,B_x,B_y = np.where(maximum_filter(B_density,seeding_window) == B_density)
    B_int = B_density[B_z,B_x,B_y]
    B_hessian = hessian(B_density)[:,:, B_z, B_x, B_y]
    B_eigs = [np.linalg.eigvals(B_hessian[:,:,_i]) for _i in range(len(B_x))]
    B_coords = np.array([[_z,_x,_y] for _z,_x,_y,_e,_int in zip(B_z,B_x,B_y,B_eigs,B_int) \
                          if (_e<0).all() and _int > intensity_ratio*np.max(B_int)], dtype=np.int)
    return A_coords, B_coords