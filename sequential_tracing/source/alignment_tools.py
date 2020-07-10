from scipy.spatial.distance import cdist
import cv2
import sys
import glob
import os
import time
import copy
import numpy as np
import pickle as pickle
import multiprocessing as mp
import psutil

from . import get_img_info, corrections, visual_tools
from . import _correction_folder, _temp_folder, _distance_zxy, _sigma_zxy, _image_size, _allowed_colors
from .External import Fitting_v3

from scipy import ndimage, stats
from scipy.spatial.distance import pdist, cdist, squareform

from skimage import morphology
from skimage.segmentation import random_walker

from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt

def __init__():
    pass

## Function for alignment between consequtive experiments

# 1. alignment for manually picked points
def align_manual_points(pos_file_before, pos_file_after,
                        save=True, save_folder=None, save_filename='', verbose=True):
    """Function to align two manually picked position files, 
    they should follow exactly the same order and of same length.
    Inputs:
        pos_file_before: full filename for positions file before translation
        pos_file_after: full filename for positions file after translation
        save: whether save rotation and translation info, bool (default: True)
        save_folder: where to save rotation and translation info, None or string (default: same folder as pos_file_before)
        save_filename: filename specified to save rotation and translation points
        verbose: say something! bool (default: True)
    Outputs:
        R: rotation for positions, 2x2 array
        T: traslation of positions, array of 2
    Here's example for how to translate points
        translated_ps_before = np.dot(ps_before, R) + t
    """
    # load position_before
    if os.path.isfile(pos_file_before):
        ps_before = np.loadtxt(pos_file_before, delimiter=',')
    else:
        raise IOError(
            f"- Position file:{pos_file_before} file doesn't exist, exit!")
    # load position_after
    if os.path.isfile(pos_file_after):
        ps_after = np.loadtxt(pos_file_after, delimiter=',')
    else:
        raise IOError(
            f"- Position file:{pos_file_after} file doesn't exist, exit!")
    # do SVD decomposition to get best fit for rigid-translation
    c_before = np.mean(ps_before, axis=0)
    c_after = np.mean(ps_after, axis=0)
    H = np.dot((ps_before - c_before).T, (ps_after - c_after))
    U, _, V = np.linalg.svd(H)  # do SVD
    # calcluate rotation
    R = np.dot(V, U.T).T
    if np.linalg.det(R) < 0:
        R[:, -1] = -1 * R[:, -1]
    # calculate translation
    t = - np.dot(c_before, R) + c_after
    # here's example for how to translate points
    # translated_ps_before = np.dot(ps_before, R) + t
    if verbose:
        print(
            f"- Manually picked points aligned, rotation:\n{R},\n translation:{t}")
    # save
    if save:
        if save_folder is None:
            save_folder = os.path.dirname(pos_file_before)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if len(save_filename) > 0:
            save_filename += '_'
        rotation_name = os.path.join(save_folder, save_filename+'rotation')
        translation_name = os.path.join(
            save_folder, save_filename+'translation')
        np.save(rotation_name, R)
        np.save(translation_name, t)
        if verbose:
            print(f'-- rotation matrix saved to file:{rotation_name}')
            print(f'-- translation matrix saved to file:{translation_name}')
    return R, t

## Translate images given drift

# align single pair of bead images
def align_single_image(_filename, _selected_crops, _ref_filename=None, _ref_ims=None, _ref_centers=None,
                        _bead_channel='488', _all_channels=_allowed_colors, _single_im_size=_image_size,
                        _num_buffer_frames=10, _num_empty_frames=0, 
                        _ref_seed_per=95, _illumination_corr=True,
                        _correction_folder=_correction_folder, 
                        _match_distance=3, _match_unique=True,
                        _rough_drift_gb=0, _drift_cutoff=1, _verbose=False):
    """Function to align single pair of bead images
    Inputs:
        _filename: filename for target image containing beads, string of filename
        _selected_crops: selected crops for choosing beads, list of 3x2 array
        _ref_filename: filename for reference image, string of filename
        _ref_ims: cropped reference images as replacement for ref_filename, list of 2d images(arrays)
        _ref_centers: fitted center coordinates for ref_ims, list of nx3 arrays
        _bead_channel: channel name for beads, int or str (default:'488')
        _all_channels: all allowed channels for given images, list of str (default: _allowed_colors)
        _single_im_size: size for single 3d image, tuple or list of 3, (default: _image_size)
        _num_buffer_frames: number of buffer frames in zscan, int (default: 10)
        _ref_seed_per: seeding intensity percentile for ref-ims, float (default: 95)
        _illumination_corr: whether do illumination correction for cropped images, bool (default: True)
        _correction_folder: folder for where correction_profile is saved, string of path (default: _correction_folder)
        _drift_cutoff: cutoff for selecting two drifts and combine, float (default: 1 pixel)
        _verbose, whether say something!, bool (default: False)
    Output:
        _final_drift: 3d drift as target_im - ref_im, array of 3
    """
    ## check inputs 
    # check filename file type
    if (isinstance(_filename, str) and '.dax' not in _filename) and not isinstance(_filename, np.ndarray):
        raise IOError(f"Wrong input file type, {_filename} should be .dax file or np.ndarray")
    # check ref_filename
    if _ref_filename is not None and '.dax' not in _ref_filename:
        raise IOError(f"Wrong input reference file type, {_ref_filename} should be .dax file")
    # check ref_centers, should be given if ref_filename is not given
    if _ref_centers is None and _ref_filename is None:
        raise ValueError(f"Either ref_center or ref_filename should be given!")
    if _ref_ims is None and _ref_filename is None:
        raise ValueError(f"Either _ref_ims or ref_filename should be given!")
    # printing info
    if isinstance(_filename, str):
        _print_name = os.path.join(_filename.split(os.sep)[-2], _filename.split(os.sep)[-1])
    else:
        _print_name = 'image'
    if _verbose:
        if _ref_filename is not None:
            _ref_print_name = os.path.join(_ref_filename.split(os.sep)[-2], 
                                           _ref_filename.split(os.sep)[-1])
            if _verbose:
                print(f"- Aligning {_print_name} to {_ref_print_name}")
        else:
            if _verbose:
                print(f"- Aligning {_print_name} to reference images and centers")
    _drifts = []
    # for each target and reference pair, do alignment:
    for _i, _crop in enumerate(_selected_crops):
        if isinstance(_filename, str):
            # load target images
            _tar_im = corrections.correct_single_image(_filename, _bead_channel, crop_limits=_crop,
                                                    single_im_size=_single_im_size,
                                                    all_channels=_all_channels, 
                                                    num_buffer_frames=_num_buffer_frames,
                                                    num_empty_frames=_num_empty_frames,
                                                    correction_folder=_correction_folder,
                                                    illumination_corr=_illumination_corr)
        elif isinstance(_filename, np.ndarray):
            _tar_im = _filename[(slice(_single_im_size[0]),slice(*_crop[-2]),slice(*_crop[-1]) )]
        else:
            raise TypeError(f"Wrong input type for _filename")
        print(_tar_im.shape)
        # get reference images
        if _ref_ims is None:
            _ref_im = corrections.correct_single_image(_ref_filename, _bead_channel, crop_limits=_crop,
                                                       single_im_size=_single_im_size,
                                                       all_channels=_all_channels, 
                                                       num_buffer_frames=_num_buffer_frames,
                                                       num_empty_frames=_num_empty_frames,
                                                       correction_folder=_correction_folder,
                                                       illumination_corr=_illumination_corr,)
        else:
            _ref_im = _ref_ims[_i].copy()
        print(_ref_im.shape)
        # get ref center
        if _ref_centers is None:
            _ref_center = visual_tools.get_STD_centers(
                _ref_im, dynamic=True, th_seed_percentile=_ref_seed_per, verbose=_verbose)
        else:
            _ref_center = np.array(_ref_centers[_i]).copy()
        # rough align ref_im and target_im
        _rough_drift = fft3d_from2d(_ref_im, _tar_im, gb=_rough_drift_gb)
        print(f"rough drift:{_rough_drift}")
        # based on ref_center and rough_drift, find matched_ref_center
        _matched_tar_seeds, _find_pair = visual_tools.find_matched_seeds(_tar_im, 
                                                            _ref_center-_rough_drift,
                                                            dynamic=True, 
                                                            th_seed_percentile=_ref_seed_per,
                                                            search_distance=_match_distance, 
                                                            keep_unique=_match_unique,
                                                            verbose=_verbose)
        print(len(_matched_tar_seeds), _ref_center.shape)
        if len(_matched_tar_seeds) < len(_ref_center) * 0.2:
            _matched_tar_seeds, _find_pair = visual_tools.find_matched_seeds(
                                        _tar_im,
                                        _ref_center-_rough_drift,
                                        dynamic=False,
                                        th_seed_percentile=_ref_seed_per,
                                        search_distance=_match_distance,
                                        keep_unique=_match_unique,
                                        verbose=_verbose)
        #print(len(_matched_tar_seeds), _rough_drift, _print_name)
        _matched_ref_center = _ref_center[_find_pair]
        if len(_matched_ref_center) == 0:
            _drifts.append(np.inf*np.ones(3))
            continue
        # apply drift to ref_center and used as seed to find target centers
        _tar_center = visual_tools.get_STD_centers(_tar_im, seeds=_matched_tar_seeds, remove_close_pts=False)
        # compare and get drift
        _drift = np.nanmean(_tar_center - _matched_ref_center , axis=0)
        _drifts.append(_drift)
        # compare difference and exit if two drifts close enough
        if len(_drifts) > 1:
            # calculate pair-wise distance
            _dists = pdist(_drifts)
            # check whether any two of drifts are close enough
            if (_dists < _drift_cutoff).any():
                _inds = list(combinations(range(len(_drifts)), 2))[np.argmin(_dists)]
                _selected_drifts = np.array(_drifts)[_inds, :]
                _final_drift = np.mean(_selected_drifts, axis=0)
                return _final_drift, 0
  
    # if not exit during loops, pick the optimal one
    if _verbose:
        print(f"Suspecting failure for {_print_name}")
    if len(_drifts) > 1:
    # calculate pair-wise distance
        _dists = pdist(_drifts)
    else:
        raise ValueError(f"Less than 2 drifts are calculated, maybe drift is too large or beads are gone?")
    _inds = list(combinations(range(len(_drifts)), 2))[np.argmin(_dists)]
    _selected_drifts = np.array(_drifts)[_inds, :]
    if _verbose:
        print(
            f"-- selected drifts:{_selected_drifts[0]}, {_selected_drifts[1]}")
    _final_drift = np.mean(_selected_drifts, axis=0)

    return _final_drift, 1


def fast_translate(im, trans):
	shape_ = im.shape
	zmax = shape_[0]
	xmax = shape_[1]
	ymax = shape_[2]
	zmin, xmin, ymin = 0, 0, 0
	trans_ = np.array(np.round(trans), dtype=int)
	zmin -= trans_[0]
	zmax -= trans_[0]
	xmin -= trans_[1]
	xmax -= trans_[1]
	ymin -= trans_[2]
	ymax -= trans_[2]
	im_base_0 = np.zeros([zmax-zmin, xmax-xmin, ymax-ymin])
	im_zmin = min(max(zmin, 0), shape_[0])
	im_zmax = min(max(zmax, 0), shape_[0])
	im_xmin = min(max(xmin, 0), shape_[1])
	im_xmax = min(max(xmax, 0), shape_[1])
	im_ymin = min(max(ymin, 0), shape_[2])
	im_ymax = min(max(ymax, 0), shape_[2])
	im_base_0[(im_zmin-zmin):(im_zmax-zmin), (im_xmin-xmin):(im_xmax-xmin), (im_ymin-ymin):(im_ymax-ymin)] = im[im_zmin:im_zmax, im_xmin:im_xmax, im_ymin:im_ymax]
	return im_base_0

def translate_points(position_file, rotation=None, translation=None, profile_folder=None, profile_filename='',
                     save=True, save_folder=None, save_filename='', verbose=True):
    """Function to translate a position file """

    pass

# function to do 2d-gaussian blur
def blurnorm2d(im, gb):
    """Normalize an input 2d image <im> by dividing by a cv2 gaussian filter of the image"""
    import cv2
    im_ = im.astype(np.float32)
    blurred = cv2.blur(im_, (gb, gb))
    return im_/blurred

# calculate pixel-level drift for 2d by FFT
def fftalign_2d(im1, im2, center=[0, 0], max_disp=150, plt_val=False):
    """
    Inputs: 2 2D images <im1>, <im2>, the expected displacement <center>, the maximum displacement <max_disp> around the expected vector.
    This computes the cross-cor between im1 and im2 using fftconvolve (fast) and determines the maximum
    """
    from scipy.signal import fftconvolve
    im2_ = np.array(im2[::-1, ::-1], dtype=float)
    #im2_ = np.array(im2[:,:], dtype=np.float)
    im2_ -= np.mean(im2_)
    im2_ /= np.std(im2_)
    im1_ = np.array(im1, dtype=np.float)
    im1_ -= np.mean(im1_)
    im1_ /= np.std(im1_)
    im_cor = fftconvolve(im1_, im2_, mode='full')

    sx_cor, sy_cor = im_cor.shape
    center_ = np.array(center)+np.array([sx_cor, sy_cor])/2.
    x_min = int(min(max(center_[0]-max_disp, 0), sx_cor))
    x_max = int(min(max(center_[0]+max_disp, 0), sx_cor))
    y_min = int(min(max(center_[1]-max_disp, 0), sy_cor))
    y_max = int(min(max(center_[1]+max_disp, 0), sy_cor))
    im_cor0 = np.zeros_like(im_cor)
    im_cor0[x_min:x_max, y_min:y_max] = 1
    im_cor = im_cor*im_cor0

    y, x = np.unravel_index(np.argmax(im_cor), im_cor.shape)
    if np.sum(im_cor > 0) > 0:
        im_cor[im_cor == 0] = np.min(im_cor[im_cor > 0])
    else:
        im_cor[im_cor == 0] = 0
    if plt_val:
        plt.figure()
        plt.imshow(im1_, interpolation='nearest')
        plt.show()
        plt.figure()
        plt.imshow(im2_, interpolation='nearest')
        plt.show()
        plt.figure()
        plt.plot([x], [y], 'k+')
        plt.imshow(im_cor, interpolation='nearest')
        plt.show()
    xt, yt = (-np.floor(np.array(im_cor.shape)/2)+[y, x]).astype(int)
    return xt, yt

def fft3d_from2d(im1, im2, gb=5, max_disp=150):
    """Given a refence 3d image <im1> and a target image <im2> 
    this max-projects along the first (z) axis and finds the best tx,ty using fftalign_2d.
    Then it trims and max-projects along the last (y) axis and finds tz.
    Before applying fftalignment we normalize the images using blurnorm2d for stability."""
    if gb > 1:
        im1_ = blurnorm2d(np.max(im1, 0), gb)
        im2_ = blurnorm2d(np.max(im2, 0), gb)
    else:
        im1_, im2_ = np.max(im1, 0), np.max(im2, 0)
    tx, ty = fftalign_2d(im1_, im2_, center=[0, 0], max_disp=max_disp, plt_val=False)
    sx, sy = im1_.shape
    if gb > 1:
        im1_t = blurnorm2d(
            np.max(im1[:, max(tx, 0):sx+tx, max(ty, 0):sy+ty], axis=-1), gb)
        im2_t = blurnorm2d(
            np.max(im2[:, max(-tx, 0):sx-tx, max(-ty, 0):sy-ty], axis=-1), gb)
    else:
        im1_t = np.max(im1[:, max(tx, 0):sx+tx, max(ty, 0):sy+ty], axis=-1)
        im2_t = np.max(
            im2[:, max(-tx, 0):sx-tx, max(-ty, 0):sy-ty], axis=-1)
    tz, _ = fftalign_2d(im1_t, im2_t, center=[
                        0, 0], max_disp=max_disp, plt_val=False)
    return np.array([tz, tx, ty])

# translation alignment of points 
def translation_align_pts(cents_fix, cents_target, cutoff=2., xyz_res=1,
                          plt_val=False, return_pts=False, verbose=False):
    """
    This checks all pairs of points in cents_target for counterparts of same distance (+/- cutoff) in cents_fix
    and adds them as posibilities. Then uses multi-dimensional histogram across txyz with resolution xyz_res.
    Then it finds nearest neighbours and returns the median txyz_b within resolution.
    """
    from itertools import combinations
    from scipy.spatial.distance import pdist
    from scipy.spatial.distance import cdist
    cents = np.array(cents_fix)
    cents_target = np.array(cents_target)
    dists_target = pdist(cents_target)
    dists = pdist(cents_fix)
    all_pairs = np.array(list(combinations(list(range(len(cents))), 2)))
    all_pairs_target = np.array(
        list(combinations(list(range(len(cents_target))), 2)))
    #inds_all = np.arange(len(dists))
    txyzs = []
    for ind_target in range(len(dists_target)):
        keep_cands = np.abs(dists-dists_target[ind_target]) < cutoff
        good_pairs = all_pairs[keep_cands][:]
        p1 = cents[good_pairs[:, 0]]
        p2 = cents[good_pairs[:, 1]]
        p1T = cents_target[all_pairs_target[ind_target, 0]]
        p2T = cents_target[all_pairs_target[ind_target, 1]]
        txyzs.extend(p1[:]-[p1T])
        txyzs.extend(p1[:]-[p2T])
    bin_txyz = np.array(
        (np.max(txyzs, axis=0)-np.min(txyzs, axis=0))/float(xyz_res), dtype=int)

    hst_res = np.histogramdd(np.array(txyzs), bins=bin_txyz)
    ibest = np.unravel_index(np.argmax(hst_res[0]), hst_res[0].shape)
    txyz_f = [hst[ib]for hst, ib in zip(hst_res[1], ibest)]
    txyz_f = np.array(txyz_f)
    inds_closestT = np.argmin(cdist(cents, cents_target + txyz_f), axis=1)
    inds_closestF = np.arange(len(inds_closestT))
    keep = np.sqrt(np.sum(
        (cents_target[inds_closestT] + txyz_f-cents[inds_closestF])**2, axis=-1)) < 2*xyz_res
    inds_closestT = inds_closestT[keep]
    inds_closestF = inds_closestF[keep]
    # check result target len
    if len(cents[inds_closestF]) == 0:
        raise ValueError(f"No matched points exist in cents[inds_closestF]")
    if len(cents_target[inds_closestT]) == 0:
        raise ValueError(
            f"No matched points exist in cents_target[inds_closestT]")
    txyz_b = np.median(
        cents_target[inds_closestT]-cents[inds_closestF], axis=0)
    if plt_val:
        plt.figure()
        plt.plot(cents[inds_closestF].T[1], cents[inds_closestF].T[2], 'go')
        plt.plot(cents_target[inds_closestT].T[1]-txyz_b[1],
                 cents_target[inds_closestT].T[2]-txyz_b[2], 'ro')
        plt.figure()
        dists = np.sqrt(
            np.sum((cents_target[inds_closestT]-cents[inds_closestF])**2, axis=-1))
        plt.hist(dists)
        plt.show()
    if verbose:
        print(f"--- {len(cents[inds_closestF])} points are aligned")
    if return_pts:
        return txyz_b, cents[inds_closestF], cents_target[inds_closestT]
    return txyz_b


#Tzxy = fft3d_from2d(im_ref_sm2, im_sm2, gb=5, max_disp=np.inf)


def sparse_centers(centersh, dist_th=0, brightness_th=0, max_num=np.inf):
    """assuming input = zxyh"""
    all_cents = np.array(centersh).T
    centers = [all_cents[0]]
    from scipy.spatial.distance import cdist
    counter = 0
    while True:
        counter += 1
        if counter > len(all_cents)-1:
            break
        if all_cents[counter][-1] < brightness_th:
            break
        dists = cdist([all_cents[counter][:3]], [c[:3] for c in centers])
        if np.all(dists > dist_th):
            centers.append(all_cents[counter])
        if len(centers) >= max_num:
            break
    return np.array(centers).T



def fast_align_centers(target_centers, ref_centers, cutoff=3., norm=2,
                       keep_unique=True, return_inds=False, verbose=True):
    """Function to fast align two set of centers
    Inputs:
        target_centers: centers from target image, list of 1d-array or np.ndarray
        ref_centers: centers from ref image, list of 1d-array or np.ndarray
        cutoff: threshold to match pairs, float (default: 3.)
        norm: distance norm used for cutoff, float (default: 2, Eucledian)
        keep_unique: whether only keep unique matched pairs, bool (default: True)
        return_inds: whether return indices of kept spots, bool (default: False)
        verbose: say something!, bool (default: True)
    Outputs:
        _aligned_target_centers: aligned target centers, np.ndarray
        _aligned_ref_centers: aligned reference centers, np.ndarray
        _target_inds: whether keep certain center, 
    """
    target_centers = np.array(target_centers)
    ref_centers = np.array(ref_centers)
    if verbose:
        print(f"-- Aligning {len(target_centers)} target_centers to {len(ref_centers)} ref_centers")
    _aligned_target_centers, _aligned_ref_centers = [], []
    _target_inds, _ref_inds = [], []
    _target_mask = np.ones(len(target_centers), dtype=np.bool)
    _ref_mask = np.ones(len(ref_centers), dtype=np.bool)
    if verbose:
        print(f"--- start finding pairs, keep_unique={keep_unique}")
    for _i,_tc in enumerate(target_centers):
        _dists = np.linalg.norm(ref_centers - _tc, axis=1, ord=norm)
        _matches = np.where((_dists < cutoff)*_ref_mask)[0]
        if keep_unique:
            if len(_matches) == 1:
                _match_id = _matches[0]
                _ref_mask[_match_id] = False
                _target_mask[_i] = False
                _aligned_ref_centers.append(ref_centers[_match_id])
                _aligned_target_centers.append(target_centers[_i])
                _target_inds.append(_i)
                _ref_inds.append(_match_id)
        else:
            if len(_matches) > 0:
                _match_id = np.argmin(_dists)
                _ref_mask[_match_id] = False
                _target_mask[_i] = False
                _aligned_ref_centers.append(ref_centers[_match_id])
                _aligned_target_centers.append(target_centers[_i])
                _target_inds.append(_i)
                _ref_inds.append(_match_id)
    if verbose:
        print(f"--- {len(_aligned_target_centers)} pairs founded.")
    _aligned_ref_centers = np.array(_aligned_ref_centers)
    _aligned_target_centers = np.array(_aligned_target_centers)
    if not return_inds:
        return _aligned_target_centers, _aligned_ref_centers
    else:
        _target_inds = np.array(_target_inds, dtype=np.int)
        _ref_inds = np.array(_ref_inds, dtype=np.int)
        return _aligned_target_centers, _aligned_ref_centers, _target_inds, _ref_inds

