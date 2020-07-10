# import from other packages
import numpy as np 
import os
import time
from scipy.stats import scoreatpercentile
from scipy.ndimage.filters import maximum_filter,minimum_filter,gaussian_filter

# import from this package
from . import _seed_th
from .. import _sigma_zxy
from ..External import Fitting_v3
from ..visual_tools import get_seed_points_base

def __init__():
    pass



# integrated function to get seeds
def get_seeds(im, max_num_seeds=None, th_seed=150, 
              th_seed_per=95, use_percentile=False,
              sel_center=None, seed_radius=30,
              gfilt_size=0.75, background_gfilt_size=10,
              filt_size=3, min_edge_distance=2,
              use_dynamic_th=True, dynamic_niters=10, min_dynamic_seeds=1,
              remove_hot_pixel=True, hot_pixel_th=3,
              return_h=False, verbose=False,
              ):
    """Function to fully get seeding pixels given a image and thresholds.
    Inputs:
      im: image given, np.ndarray, 
      num_seeds: number of max seeds number, int default=-1, 
      th_seed: seeding threshold between max_filter - min_filter, float/int, default=150, 
      use_percentile: whether use percentile to determine seed_th, bool, default=False,
      th_seed_per: seeding percentile in intensities, float/int of percentile, default=95, 
      sel_center: selected center coordinate to get seeds, array-like, same dimension as im, 
        default=None (whole image), 
      seed_radius: square frame radius of getting seeds, int, default=30,
      gfilt_size: gaussian filter size for max_filter image, float, default=0.75, 
      background_gfilt_size: gaussian filter size for min_filter image, float, default=10,
      filt_size: filter size for max/min filter, int, default=3, 
      min_edge_distance: minimal allowed distance for seed to image edges, int/float, default=3,
      use_dynamic_th: whetaher use dynamic th_seed, bool, default=True, 
      dynamic_niters: number of iterations used for dynamic th_seed, int, default=10, 
      min_dynamic_seeds: minimal number of seeds to get with dynamic seeding, int, default=1,
      return_h: whether return height of seeds, bool, default=False, 
      verbose: whether say something!, bool, default=False,
    """
    # check inputs
    if not isinstance(im, np.ndarray):
        raise TypeError(f"image given should be a numpy.ndarray")
    if th_seed_per >= 100 or th_seed_per <= 50:
        use_percentile = False
        print(f"th_seed_per should be a percentile > 50, invalid value given ({th_seed_per}), so not use percentile here.")
    # crop image if sel_center is given
    if sel_center is not None:
        if len(sel_center) != len(np.shape(im)):
            raise IndexError(f"num of dimensions should match for selected center and image given.")
        # get selected center and cropping neighbors
        _center = np.array(sel_center, dtype=np.int)
        _llims = np.max([np.zeros(len(im.shape)), _center-seed_radius], axis=0)
        _rlims = np.min([np.array(im.shape), _center+seed_radius], axis=0)
        _lims = np.array(np.transpose(np.stack([_llims, _rlims])), dtype=np.int)
        _lim_crops = tuple([slice(_l,_r) for _l,_r in _lims])
        # crop image
        _im = im[_lim_crops].copy()
        # get local centers for adjustment
        _local_edges = _llims
    else:
        _local_edges = np.zeros(len(np.shape(im)))
        _im = im.copy()
        
    # get threshold
    if use_percentile:
        _th_seed = scoreatpercentile(im, th_seed_per) - scoreatpercentile(im, (100-th_seed_per)/2)
    else:
        _th_seed = th_seed
    if verbose:
        _start_time = time.time()
        if not use_dynamic_th:
            print(f"-- start seeding image with threshold: {_th_seed:.2f}", end='; ')
        else:
            print(f"-- start seeding image, th={_th_seed:.2f}", end='')
    ## do seeding
    if not use_dynamic_th:
        dynamic_niters = 1 # setting only do seeding once
    else:
        dynamic_niters = int(dynamic_niters)
    # front filter:
    if gfilt_size:
        _max_im = gaussian_filter(_im, gfilt_size)
    else:
        _max_im = _im
    _max_ft = np.array(maximum_filter(_max_im, int(filt_size)), dtype=np.int)
    # background filter
    if background_gfilt_size:
        _min_im = gaussian_filter(_im,background_gfilt_size)
    else:
        _min_im = _im
    _min_ft = np.array(minimum_filter(_min_im, int(filt_size)), dtype=np.int)
    
    # iteratively select seeds
    for _iter in range(dynamic_niters):
        # get seed coords
        _current_seed_th = _th_seed * (1-_iter/dynamic_niters)
        #print(_iter, _current_seed_th)
        # should be: local max, not local min, differences large than threshold
        _coords = np.where((_max_ft == _max_im) & (_min_ft != _min_im) & (_max_ft-_min_ft >= _current_seed_th))
        # remove edges
        if min_edge_distance > 0:
            _keep_flags = remove_edge_points(_im, _coords, min_edge_distance)
            _coords = tuple(_cs[_keep_flags] for _cs in _coords)
        # if got enough seeds, proceed.
        if len(_coords[0]) >= min_dynamic_seeds:
            break
    # print current th
    if verbose and use_dynamic_th:
        print(f"->{_current_seed_th:.2f}")
    # hot pixels
    if remove_hot_pixel:
        _,_x,_y = _coords
        _xy_str = [str([np.round(x_,1),np.round(y_,1)]) 
                    for x_,y_ in zip(_x,_y)]
        _unique_xy_str, _cts = np.unique(_xy_str, return_counts=True)
        _keep_hot = np.array([_xy not in _unique_xy_str[_cts>=hot_pixel_th] 
                             for _xy in _xy_str],dtype=bool)
        _coords = tuple(_cs[_keep_hot] for _cs in _coords)
    # get heights
    _hs = (_max_ft - _min_ft)[_coords]
    _final_coords = np.array(_coords) + _local_edges[:, np.newaxis] # adjust to absolute coordinates
    if return_h: # patch heights if returning it
        _final_coords = np.concatenate([_final_coords, _hs[np.newaxis,:]])
    # transpose and sort by intensity decreasing order
    _final_coords = np.transpose(_final_coords)[np.flipud(np.argsort(_hs))]
    if verbose:
        print(f"found {len(_final_coords)} seeds in {time.time()-_start_time:.2f}s")
    # truncate with max_num_seeds
    if max_num_seeds is not None and max_num_seeds > 0 and max_num_seeds <= len(_final_coords):
        _final_coords = _final_coords[:np.int(max_num_seeds)]
        if verbose:
            print(f"--- {max_num_seeds} seeds are kept.")
    
    return _final_coords

def remove_edge_points(im, T_seeds, distance=2):
    
    im_size = np.array(np.shape(im))
    _seeds = np.array(T_seeds)[:len(im_size),:].transpose()
    flags = []
    for _seed in _seeds:
        _f = ((_seed >= distance) * (_seed <= im_size-distance)).all()
        flags.append(_f)
    
    return np.array(flags, dtype=np.bool)


# fit the entire field of view image
def fit_fov_image(im, channel, seeds=None, max_num_seeds=500,
                  th_seed=300, th_seed_per=95, use_percentile=False, 
                  use_dynamic_th=True, 
                  dynamic_niters=10, min_dynamic_seeds=1,
                  remove_hot_pixel=True, seeding_kwargs={}, 
                  fit_radius=5, init_sigma=_sigma_zxy, weight_sigma=0, 
                  normalize_backgroud=False, normalize_local=False, 
                  background_args={}, 
                  fitting_args={}, verbose=True):
    """Function to merge seeding and fitting for the whole fov image"""

    ## check inputs
    _th_seed = float(th_seed)
    if verbose:
        print(f"-- start fitting spots in channel:{channel}, ", end='')
        _fit_time = time.time()
    ## seeding
    if seeds is None:
        _seeds = get_seeds(im, max_num_seeds=max_num_seeds,
                        th_seed=th_seed, th_seed_per=th_seed_per,
                        use_percentile=use_percentile,
                        use_dynamic_th=use_dynamic_th, 
                        dynamic_niters=dynamic_niters,
                        min_dynamic_seeds=min_dynamic_seeds,
                        remove_hot_pixel=remove_hot_pixel,
                        return_h=False, verbose=False,
                        **seeding_kwargs,)
        if verbose:
            print(f"{len(_seeds)} seeded, ", end='')
    else:
        _seeds = np.array(seeds)[:,:len(np.shape(im))]
        if verbose:
            print(f"{len(_seeds)} given, ", end='')

    ## fitting
    _fitter = Fitting_v3.iter_fit_seed_points(
        im, _seeds.T, radius_fit=fit_radius, 
        init_w=init_sigma, weight_sigma=weight_sigma,
        **fitting_args,
    )    
    # fit
    _fitter.firstfit()
    # check
    _fitter.repeatfit()
    # get spots
    _spots = np.array(_fitter.ps)
    _spots = _spots[np.sum(np.isnan(_spots),axis=1)==0] # remove NaNs
    # normalize intensity if applicable
    if normalize_backgroud and not normalize_local:
        from ..io_tools.load import find_image_background 
        _back = find_image_background(im, **background_args)
        if verbose:
            print(f"normalize total background:{_back:.2f}, ", end='')
        _spots[:,0] = _spots[:,0] / _back
    elif normalize_local:
        from ..io_tools.load import find_image_background
        from ..io_tools.crop import generate_neighboring_crop
        _backs = []
        for _pt in _spots:
            _crop = generate_neighboring_crop(_pt[1:4],
                                              crop_size=fit_radius*2,
                                              single_im_size=np.array(np.shape(im)))
            _cropped_im = im[_crop]
            _backs.append(find_image_background(_cropped_im, **background_args))
        if verbose:
            print(f"normalize local background for each spot, ", end='')
        _spots[:,0] = _spots[:,0] / np.array(_backs)

    if verbose:
        print(f"{len(_spots)} fitted in {time.time()-_fit_time:.3f}s.")
    return _spots




# integrated function to do gaussian fitting
## Fit bead centers
def get_centers(im, seeds=None, th_seed=150, 
                th_seed_per=98, use_percentile=False,
                sel_center=None, seed_radius=40,
                max_num_seeds=None, use_dynamic_th=True, 
                min_num_seeds=1,
                remove_hot_pixel=True, hot_pixel_th=3,
                seed_kwargs={}, 
                fit_radius=5, 
                remove_close_pts=True, close_threshold=0.1, 
                verbose=False):
    '''Fit centers for one image:
    Inputs:
        im: image, ndarray
        th_seeds: threshold for seeding, float (default: 150)
        dynamic: whether do dynamic seeding, bool (default:True)
        th_seed_percentile: intensity percentile for seeding, float (default: 95)
        remove_close_pts: whether remove points really close to each other, bool (default:True)
        close_threshold: threshold for removing duplicates within a distance, float (default: 0.01)
        fit_radius: radius of gaussian profile
        verbose: say something!, bool (default: False)
    Outputs:
        centers: fitted spots with information, n by 4 array'''
    from ..External.Fitting_v3 import iter_fit_seed_points
    # seeding
    if seeds is None:
        seeds = get_seeds(im, max_num_seeds=max_num_seeds,
                          th_seed=th_seed, th_seed_per=th_seed_per,
                          use_percentile=use_percentile,
                          sel_center=sel_center, seed_radius=seed_radius,
                          use_dynamic_th=use_dynamic_th,
                          min_dynamic_seeds=min_num_seeds,
                          remove_hot_pixel=remove_hot_pixel, 
                          hot_pixel_th=hot_pixel_th,
                          return_h=False, verbose=verbose, 
                          **seed_kwargs)

    # fitting
    fitter = iter_fit_seed_points(im, seeds.T, radius_fit=fit_radius)
    fitter.firstfit()
    pfits = fitter.ps # get fitted points
    # get coordinates for fitted centers
    if len(pfits) > 0:
        centers = np.array(pfits)[:, 1:4]
        if verbose:
            print(f"-- fitting {len(pfits)} points.")
        # remove very close spots
        if remove_close_pts:
            remove = np.zeros(len(centers), dtype=np.bool)
            for i, bead in enumerate(centers):
                if np.isnan(bead).any() or np.sum(np.sum((centers-bead)**2, axis=1) < close_threshold) > 1:
                    remove[i] = True
                if (bead < 0).any() or (bead > np.array(im.shape)).any():
                    remove[i] = True
            centers = centers[remove==False]
            if verbose:
                print(f"-- {np.sum(remove)} points removed, given miminum distance {close_threshold}.")
    else:
        centers = np.array([])
        if verbose:
            print(f"-- no points fitted, return empty array.")

    return centers


# select sparse centers given candidate centers
def select_sparse_centers(centers, distance_th=9, 
                          distance_norm=np.inf,
                          verbose=False):
    """Select sparse centers from given centers
    Inputs:
        centers: center coordinates (zxy) for all candidates, list of 3-array or nx3 2d-array
        distance_th: threshold for distance between neighboring centers, float (default: 9)
        distance_norm: norm for the distance, int (default: np.inf)
    Output:
        _sel_centers: selected centers, nx3 2d-array"""
    _sel_centers = []
    for ct in centers:
        if len(_sel_centers) == 0:
            _sel_centers.append(ct) # if empty, directly append
        else:
            _sc = np.array(_sel_centers)
            _dists = np.linalg.norm(_sc - ct[np.newaxis,:], axis=1, ord=distance_norm)
            if (_dists <= distance_th).any():
                continue
            else:
                _sel_centers.append(ct)
    
    if verbose:
        print(f"-- {len(_sel_centers)} among {len(centers)} centers are selected by th={distance_th}")

    return np.array(_sel_centers)