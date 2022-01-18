# import packages
import numpy as np
import glob,sys,os,time
import matplotlib.pyplot as plt
import pickle as pickle
import multiprocessing as mp
# import functions
from sklearn.decomposition import PCA
from scipy.ndimage.interpolation import map_coordinates
from scipy.stats import scoreatpercentile
from scipy.spatial.distance import cdist
# from ImageAnalysis3 import constants
from .. import _distance_zxy
from ..visual_tools import _myCmaps
# from ImageAnalysis3 import functions
from ..visual_tools import add_source
from ..spot_tools.translating import normalize_center_spots

def randomize_index_dict(index_dict, key1='A', key2='B'):
    """Function to randomize indices for two keys in a dict
    Inputs:
        index_dict: index dictionary containing at least two components, dict of indices
        key1: key 1 in index_dict to be shuffled, str
        key2L key 2 in index_dict to be shuffled, str
    Outputs:
        _sf_dict: shuffled dict containing only key1 & key2
    """
    ## check inputs
    if not isinstance(index_dict, dict):
        raise TypeError(f"Wrong input index_dict, should be a dict but {type(index_dict)} is given.")
    if key1 not in index_dict:
        raise KeyError(f"key1:{key1} not in index_dict")
    if key2 not in index_dict:
        raise KeyError(f"key2:{key2} not in index_dict")
    ## shuffling dict
    # retrieve data
    _i1 = np.array(index_dict[key1], dtype=np.int)
    _i2 = np.array(index_dict[key2], dtype=np.int)
    _total_is = np.concatenate([_i1,_i2])
    # shuffle
    np.random.shuffle(_total_is)
    _ni1 = _total_is[:len(_i1)]
    _ni2 = _total_is[-len(_i2):]
    # generate output dict
    _sf_dict = {
        key1: np.sort(_ni1),
        key2: np.sort(_ni2),
    }
    return _sf_dict


# align chromosome with AB-axis
def max_project_AB_compartment(spots, comp_dict, pca_other_2d=True):
    """Do mass projection onto AB (first two keys) axis of chromosome 
    by norm-centered spots
    Inputs:
        spots: list of spots that that generated from Fitting_V3, 2d np.ndarray or list of 1d np.ndarray
        comp_dict: dictionary of compartment reference, dict of compartment_name -> region_ids        
        pca_other_2d: whether do PCA transformation for other two dimensions, bool (default: True)
    Output:
        _spots: same spots after transformation, 2d np.ndarray   
        """
    ## check inputs
    # spots
    _spots = np.array(spots).copy()
    if len(spots.shape) != 2:
        raise ValueError(f"Input spots should be 2d-array like structure, but shape:{spots.shape} is given!")
    # comp_dict
    if not isinstance(comp_dict, dict):
        raise TypeError(f"Wrong input type of comp_dict, should be dict but {type(comp_dict)} is given")
    
    # extract indices for first two keys in comp_dict
    _ind1 = np.array(list(comp_dict.values())[0], dtype=np.int)
    _ind2 = np.array(list(comp_dict.values())[1], dtype=np.int)
    # get original coordinates
    _coords = _spots[:,1:4]
    # get first axis
    _AB_axis = np.nanmean(_coords[_ind1],axis=0) - np.nanmean(_coords[_ind2], axis=0)
    _AB_axis = _AB_axis / np.linalg.norm(_AB_axis)
    # get basic rotation matrix
    _rot_mat1 = np.zeros([3,3])
    _rot_mat1[2,2] = 1
    _rot_mat1[0,0] = _AB_axis[0]
    _rot_mat1[1,1] = _AB_axis[0]
    _rot_mat1[1,0] = np.sqrt(1-_rot_mat1[1,1]**2)
    _rot_mat1[0,1] = - _rot_mat1[1,0]
    _rot_mat2 = np.zeros([3,3])
    _rot_mat2[0,0] = 1
    _rot_mat2[1,1] = _AB_axis[1] / _rot_mat1[1,0]
    _rot_mat2[2,2] = _rot_mat2[1,1]
    _rot_mat2[2,1] = np.sign(_AB_axis[1]) * np.sqrt(1-_rot_mat2[1,1]**2)
    _rot_mat2[1,2] = - _rot_mat2[2,1]
    # summarize rotation matrix
    _r = _rot_mat2 @ _rot_mat1
    # transform coordinates
    _trans_coords = _coords @ _r
    # if PCA the other two axis, do the following:
    if pca_other_2d:
        _clean_inds = np.array([_i for _i,_c in enumerate(_trans_coords) if not np.isnan(_c).any()],dtype=np.int)
        _clean_2d_coords = _trans_coords[_clean_inds][:,1:3]
        if 'PCA' not in locals():
            from sklearn.decomposition import PCA
        _model_2d = PCA(2)
        _model_2d.fit(_clean_2d_coords)
        _trans_coords[_clean_inds,1:3] = _model_2d.fit_transform(_clean_2d_coords)
    # update spots and retrun
    _spots[:, 1:4] = _trans_coords
    
    return _spots

# convert spots to 3d cloud by replacing spots with gaussians
def convert_spots_to_cloud(spots, comp_dict, im_radius=30,
                           distance_zxy=_distance_zxy,
                           spot_variance=None, scaling=1., 
                           center=True, pca_align=False, 
                           max_project_AB=False, use_intensity=False, 
                           normalize_count=False, normalize_pdf=False, 
                           return_plot=False, ax=None, return_scores=True, 
                           verbose=False):
    """Convert spots (have to be centered to zero with correct scaling, 
                      better to be max_projected if only 2 compartment specified)
       into a 3d density map for each compartment specified in comp_dict
    Inputs: 
        spots
       """
    ## check inputs
    #from ImageAnalysis3.visual_tools import add_source
    # spots
    _spots = np.array(spots).copy()
    if len(spots.shape) != 2:
        raise ValueError(f"Input spots should be 2d-array like structure, but shape:{spots.shape} is given!")
    # comp_dict
    if not isinstance(comp_dict, dict):
        print(f"Wrong input type of comp_dict, should be dict but {type(comp_dict)} is given")
        return None
    # spot_variance
    if spot_variance is not None:
        if len(spot_variance) < 3:
            raise ValueError(f"variance should be given for 3d")

    _norm_spots = normalize_center_spots(_spots, distance_zxy=distance_zxy, 
                                         center=center, pca_align=pca_align, 
                                         scaling=scaling)
    if max_project_AB:
        _norm_spots = max_project_AB_compartment(_norm_spots, comp_dict, pca_other_2d=True)
    # create density map dict
    _density_dict = {_k:np.zeros([im_radius*2]*3) for _k in comp_dict.keys()}
    _spot_ct = {_k:0 for _k in comp_dict.keys()}
    for _k, _v in comp_dict.items():
        for _spot in _norm_spots[np.array(_v, dtype=np.int)]:
            if spot_variance is not None:
                _var = np.array(spot_variance[:3]).reshape(-1)
            else:
                _var = _spot[5:8] * scaling
            if use_intensity:
                _int = _spot[0]
            else:
                _int = 1

            if not np.isnan(_spot).any():
                _density_dict[_k] = add_source(_density_dict[_k], 
                                               pos=im_radius+_spot[1:4],
                                               h=_int, sig=_var, 
                                               size_fold=im_radius)
                _spot_ct[_k] += 1
        # normalize spot counts if specified
        if normalize_count:
            _density_dict[_k] = _density_dict[_k] / _spot_ct[_k]
        # normalize as pdf if specified
        if normalize_pdf:
            _density_dict[_k] = _density_dict[_k] / np.sum(_density_dict[_k])
    # calculate scores
    if return_scores:
        _score_dict = spot_cloud_scores(_spots, _spots, comp_dict, 
                                        spot_variance=spot_variance,
                                        normalize_spots=True, 
                                        distance_zxy=distance_zxy,
                                        )

    if return_plot:
        if ax is None:
            _fig = plt.figure(figsize=(8,6),dpi=200)
            ax = _fig.add_subplot(111)
        # estimate limits
        _vmax = min(scoreatpercentile(list(_density_dict.values())[0].sum(2), 98), scoreatpercentile(list(_density_dict.values())[1].sum(2), 98))

        # make density plot
        for _i, (_k,_v) in enumerate(list(_density_dict.items())[:2]):
            _im = ax.imshow(_v.sum(2).T, cmap=_myCmaps[_i],
                            vmin=0, vmax=_vmax, label=_k, alpha=1-0.5*(_i),
                            interpolation='nearest')
            _cb = plt.colorbar(_im, ax=ax, shrink=0.85)

        #_im2 = ax.imshow(_density_dict['B'].mean(0), cmap=myBlues,alpha=0.5, vmin=0, vmax=_vmax)
        #_cb2 = plt.colorbar(_im2, ax=ax, shrink=0.85)
        ax.set_xticks(np.arange(0, 2*im_radius+1, 10))
        ax.set_xticklabels(np.arange(-im_radius, im_radius+1,10))
        ax.set_yticks(np.arange(0, 2*im_radius+1, 10))
        ax.set_yticklabels(np.arange(-im_radius, im_radius+1,10))
        if max_project_AB:
            ax.set_xlabel(f"Max projected AB axis")
        elif pca_align:
            ax.set_xlabel("PC1")
        else:
            ax.set_xlabel(f"X")
        if max_project_AB:
            ax.set_ylabel(f"PC1 orthogonal to max projection")
        elif pca_align:
            ax.set_ylabel("PC2")
        else:
            ax.set_ylabel(f"Y")        
    
    # return
    if return_scores and return_plot:
        return _density_dict, _score_dict, ax
    elif return_scores and not return_plot:
        return _density_dict, _score_dict
    elif not return_scores and return_plot:
        return _density_dict, ax
    else:
        return _density_dict


def spot_cloud_scores(spots, ref_spots, comp_dict, 
                      spot_variance=None, normalize_spots=True,
                      distance_zxy=_distance_zxy, exclude_self=True, 
                      convert_to_nm=True,
                      exclude_dict={}, dist_th=0.001):
    from ..spot_tools.translating import normalize_center_spots
    ## check inputs
    # spot_variance
    if spot_variance is not None:
        if len(spot_variance) < 3:
            raise ValueError(f"variance should be given for 3d")
        else:
            spot_variance=np.array(spot_variance[:3],dtype=np.float).reshape(-1)
    # check exclude_dict:
    if not isinstance(exclude_dict, dict):
        raise TypeError(f"Wrong input type for exclude_dict, should be a dict but {type(exclude_dict)} is given.")
    else:
        _rev_ex_dict = {int(_v): int(_k) for _k,_v in exclude_dict.items()}
    
    if normalize_spots:
        _ref_spots = normalize_center_spots(ref_spots, 
                                            distance_zxy=distance_zxy,
                                            center_zero=True, pca_align=False,
                                            scale_variance=False,scaling=1.
                                            )
        # adjust zxys
        _zxys = _spots[:,1:4] - np.nanmean(ref_spots[:,1:4], axis=0)
        if convert_to_nm:
            _zxys = _zxys * np.array(distance_zxy) / np.min(distance_zxy)
    else:
        _ref_spots = np.array(ref_spots).copy()
        _zxys = spots[:,1:4]
    #print(_zxys)
    _score_dict = {}
    for _key, _inds in comp_dict.items():
        # extract indices
        _inds = np.array(_inds, dtype=np.int)
        # extract coordinates
        _ref_cts = _ref_spots[_inds,1:4]
        _scores = np.zeros(len(_zxys), dtype=np.float)
        
        for _i, _ct in enumerate(_ref_cts):
            if not np.isnan(_ct).any():
                # exclude itself flag
                if exclude_self:
                    _ex_flag = cdist(_zxys, _ct[np.newaxis,:]).reshape(-1) < dist_th
                    if _inds[_i] in _rev_ex_dict:
                        _ex_spot_ind = _rev_ex_dict[_inds[_i]]
                        _ex_flag[_ex_spot_ind] = True
                        
                if spot_variance is None:
                    _std = _ref_spots[_i,5:8]
                else:
                    _std = spot_variance
                _gpdfs = calculate_gaussian_density(_zxys, _ct, _std)
                if exclude_self:
                    _gpdfs[_ex_flag] = 0 # excluded don't contribute density
                _scores += _gpdfs
        # append
        _score_dict[_key] = _scores

    return _score_dict

# batch convert spots to clouds
def Batch_Convert_Spots_to_Cloud(spot_list, comp_dict, im_radius=30, 
                                 num_threads=12, distance_zxy=_distance_zxy, 
                                 spot_variance=None, expand_ratio=1.,
                                 verbose=True):
    """Function to batch convert spot list to cloud dict list
    Inputs:
        spot_list: list of fitted spots of chromosomes, list of 2d-array
        comp_dict: compartment index dictionary, marking identities of indices in spots, dict or list of dicts
        im_radius: image radius in pixel, int (default: 30)
        num_threads: number of threads """
    _start_time = time.time()
    _convert_args = []
    if isinstance(comp_dict, list):
        if len(comp_dict) != len(spot_list):
            raise IndexError(f"Wrong length of comp_dict list, should of same size of spot_list")
    elif isinstance(comp_dict, dict):
        # convert to list of dicts
        comp_dict = [comp_dict for _spots in spot_list]
    else:
        raise TypeError(f"Wrong input type of comp_dict, should be dict / list of dict")
    for _spots, _cdict in zip(spot_list, comp_dict):
        _convert_args.append(
            (_spots, _cdict, im_radius, distance_zxy, 
             spot_variance, expand_ratio, True, False)
        )
    with mp.Pool(num_threads) as _convert_pool:
        if verbose:
            print(f"-- {len(_convert_args)} chromos,omes processing by {num_threads} threads.")
        _results = _convert_pool.starmap(convert_spots_to_cloud, _convert_args, chunksize=1)
        _convert_pool.close()
        _convert_pool.join()
        _convert_pool.terminate()
    # extract
    _density_dicts = [_r[0] for _r in _results]
    _score_dicts = [_r[1] for _r in _results]
    if verbose:
        print(f"--- time spent in converting to cloud: {time.time()-_start_time}")
    
    return _density_dicts, _score_dicts

def Batch_Convert_Control_to_Cloud():
    pass

def density_overlaps(d1, d2, method='geometric'):
    """Function to calculate overlaps based on densities
    Inputs: 
        d1: density array 1
        d2: density array 2, same shape as d1S
        method: type of method to calculate density overlap
    Output:
        overlap_score, a float
    """
    if method == 'geometric':
        return np.nansum(np.sqrt(d1*d2)) / np.sqrt(np.sum(d1) * np.sum(d2))

def calculate_gaussian_density(centers, ref_center, sigma, 
                               intensity=1, background=0):
    sigma = np.array(sigma, dtype=np.float)
    g_pdf = np.exp(-0.5 * np.sum((centers - ref_center)**2 / sigma**2, axis=-1))
    g_pdf = float(intensity) * g_pdf + float(background)
    return g_pdf




def winsorize(scores, l_per=5, u_per=5, normalize=False):
    _scores = np.array(scores, dtype=np.float)
    _llim = scoreatpercentile(_scores, l_per)
    _ulim = scoreatpercentile(_scores, 100-u_per)
    _scores[_scores < _llim] = _llim
    _scores[_scores > _ulim] = _ulim
    if normalize:
        _scores = (_scores - np.nanmin(_scores)) / (np.nanmax(_scores) - np.nanmin(_scores))
    return _scores

def spot_density_scores(hzxys, ref_hzxys, comp_dict, stds=[100,100,100],
                        exclude_self=True, self_th=0.001,):
    """Function to calculate spot scores"""
    
    _hzxys = np.array(hzxys)[:,:4]
    _ref_hzxys = np.array(ref_hzxys)[:,:4]
    _stds = np.array(stds)[-3:]
    
    # initialize
    _score_dict = {_k:np.zeros(len(_hzxys)) for _k in comp_dict}
    # loop through keys
    for _k, _inds in comp_dict.items():
        _sel_ref_hzxys = _ref_hzxys[np.array(_inds, dtype=np.int)]
        # add gaussians
        for _i, _hzxy in enumerate(_hzxys):
            # skip if this spot is nan
            if np.isnan(hzxys[_i]).any():
                _score_dict[_k][_i] = np.nan
                continue
            else:
                _zxy = _hzxy[-3:].copy()
                _r_zxys = _sel_ref_hzxys[(np.isnan(_sel_ref_hzxys).sum(1)==0), -3:].copy()
                if exclude_self:
                    _dists = np.linalg.norm(_r_zxys-_zxy, axis=1)
                    _r_zxys = _r_zxys[_dists > self_th]
                _dens = calculate_gaussian_density(_r_zxys, _zxy, _stds)
                # append
                _score_dict[_k][_i] += np.sum(_dens)

    return _score_dict