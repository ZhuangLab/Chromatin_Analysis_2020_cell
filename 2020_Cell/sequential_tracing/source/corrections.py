from . import get_img_info, visual_tools, alignment_tools, io_tools
#from .classes.batch_functions import killchild
from . import _correction_folder,_temp_folder,_distance_zxy,_sigma_zxy,_image_size,_allowed_colors
from .External import Fitting_v3
import numpy as np
import scipy
import pickle
import matplotlib.pylab as plt
import os, glob 
import sys, time
from scipy.stats import scoreatpercentile
import multiprocessing as mp
import ctypes
from scipy.ndimage.interpolation import map_coordinates

def __init__():
    pass

# merged function to calculate bead drift directly from files

def Calculate_Bead_Drift(folders, fovs, fov_id, num_threads=12, drift_size=500, ref_id=0,
                         sequential_mode=False, bead_channel='488', all_channels=_allowed_colors,
                         illumination_corr=True, correction_folder=_correction_folder,
                         coord_sel=None, single_im_size=_image_size, 
                         num_buffer_frames=10, num_empty_frames=1, 
                         match_distance=3, match_unique=True, rough_drift_gb=0,
                         max_ref_points=500, ref_seed_per=90, drift_cutoff=1,
                         save=True, save_folder=None, save_postfix='_current_cor.pkl',
                         stringent=True, overwrite=False, verbose=True):
    """Function to generate drift profile given a list of corrected bead files
    Inputs:
        folders: hyb-folder names planned for drift-correction, list of folders
        fovs: filenames for all field-of-views, list of filenames
        fov_id: selected fov_id to do drift_corrections, int
        num_threads: number of threads used for drift correction, int (default: 12)
        drift_size: selected sub-figure size for bead-fitting, int (default: 500)
        ref_id: id for reference hyb-folder, int (default: 0)
            (only effective if sequential_mode is False)
        sequential_mode: whether align drifts sequentially, bool (default: False)
            (if sequential_mode is false that means align all ims to ref_id im)
        bead_channel: channel for beads, int or str (default: '488')
        all_channels: all allowed channels, list of str (default: _alowed_colors)
        illumination_corr: whether do illumination correction, bool (default: True)
        correction_folder: where to find correction_profile, str for folder path (default: default)
        coord_sel: selected coordinate to pick windows nearby, array of 2 (default: None, center of image)
        single_im_size: image size of single channel 3d image, array of 3 (default: _image_size)
        num_buffer_frames: number of buffer frames in zscan, int (default: 10)
        max_ref_points: maximum allowed reference points, int (default: 500)
        ref_seed_per: seeding intensity percentile for ref-ims, float (default: 95)
        save: whether save drift result dictionary to pickle file, bool (default: True)
        save_folder: folder to save drift result, required if save is True, string of path (default: None)
        save_postfix: drift file postfix, string (default: '_sequential_current_cor.pkl')
        strigent: whether keep only confirmed results, bool (default:True)
        overwrite: whether overwrite existing drift_dic, bool (default: False)
        verbose: say something during the process! bool (default:True)
    Outputs:
        drift_dic: dictionary for ref_hyb_name -> 3d drift array
        fail_count: number of suspicious failures in drift correction, int
    """
    ## check inputs
    # check folders
    if not isinstance(folders, list):
        raise ValueError("Wrong input type of folders, should be a list")
    if len(folders) == 0:  # check folders length
        raise ValueError("Kwd folders should have at least one element")
    if not isinstance(fovs, list):
        raise ValueError("Wrong input type of fovs, should be a list")
    # check if all images exists
    _fov_name = fovs[fov_id]
    for _fd in folders:
        _filename = os.path.join(_fd, _fov_name)
        if not os.path.isfile(_filename):
            raise IOError(
                f"one of input file:{_filename} doesn't exist, exit!")
    # check ref-id
    if ref_id >= len(folders) or ref_id <= -len(folders):
        raise ValueError(
            f"Ref_id should be valid index of folders, however {ref_id} is given")
    # check save-folder
    if save_folder is None:
        save_folder = os.path.join(
            os.path.dirname(folders[0]), 'Analysis', 'drift')
        print(
            f"No save_folder specified, use default save-folder:{save_folder}")
    elif not os.path.exists(save_folder):
        if verbose:
            print(f"Create drift_folder:{save_folder}")
        os.makedirs(save_folder)
    # check save_name
    if sequential_mode:  # if doing drift-correction in a sequential mode:
        save_postfix = '_sequential'+save_postfix

    # check coord_sel
    if coord_sel is None:
        coord_sel = np.array(
            [int(single_im_size[-2]/2), int(single_im_size[-1]/2)], dtype=np.int)
    else:
        coord_sel = np.array(coord_sel, dtype=np.int)
    # collect crop coordinates (slices)
    crop0 = np.array([[0, single_im_size[0]],
                      [max(coord_sel[-2]-drift_size, 0), coord_sel[-2]],
                      [max(coord_sel[-1]-drift_size, 0), coord_sel[-1]]], dtype=np.int)
    crop1 = np.array([[0, single_im_size[0]],
                      [coord_sel[-2], min(coord_sel[-2] +
                                          drift_size, single_im_size[-2])],
                      [coord_sel[-1], min(coord_sel[-1]+drift_size, single_im_size[-1])]], dtype=np.int)
    crop2 = np.array([[0, single_im_size[0]],
                      [coord_sel[-2], min(coord_sel[-2] +
                                          drift_size, single_im_size[-2])],
                      [max(coord_sel[-1]-drift_size, 0), coord_sel[-1]]], dtype=np.int)
    # merge into one array which is easier to feed into function
    selected_crops = np.stack([crop0, crop1, crop2])
    
    ## start loading existing profile
    _save_name = _fov_name.replace('.dax', save_postfix)
    _save_filename = os.path.join(save_folder, _save_name)
    # try to load existing profiles
    if not overwrite and os.path.isfile(_save_filename):
        if verbose:
            print(f"- loading existing drift info from file:{_save_filename}")
        old_drift_dic = pickle.load(open(_save_filename, 'rb'))
        _exist_keys = [os.path.join(os.path.basename(
            _fd), _fov_name) in old_drift_dic for _fd in folders]
        if sum(_exist_keys) == len(folders):
            if verbose:
                print("-- all frames exists in original drift file, exit.")
            return old_drift_dic, 0

    # no existing profile or force to do de novo correction:
    else:
        if verbose:
            print(
                f"- starting a new drift correction for field-of-view:{_fov_name}")
        old_drift_dic = {}
    ## initialize drift correction
    _start_time = time.time()  # record start time
    # whether drift for reference frame changes
    if len(old_drift_dic) > 0:
        old_ref_frames = [_hyb_name for _hyb_name,
                          _dft in old_drift_dic.items() if (np.array(_dft) == 0).all()]
        if len(old_ref_frames) > 1 or len(old_ref_frames) == 0:
            print("-- ref frame not unique, start over!")
            old_drift_dic = {}
            old_ref_frame = None
        else:
            old_ref_frame = old_ref_frames[0]
            # if ref-frame not overlapping, remove the old one for now
            if old_ref_frame != os.path.join(os.path.basename(folders[ref_id]), _fov_name):
                if verbose:
                    print(
                        f"-- old-ref:{old_ref_frame}, delete old refs because ref doesn't match")
                del old_drift_dic[old_ref_frame]
    else:
        old_ref_frame = None
    if not sequential_mode:
        if verbose:
            print(f"- Start drift-correction with {num_threads} threads, image mapped to image:{ref_id}")
        ## get all reference information
        # ref filename
        _ref_filename = os.path.join(
            folders[ref_id], _fov_name)  # get ref-frame filename
        _ref_keyname = os.path.join(
            os.path.basename(folders[ref_id]), _fov_name)
        _ref_ims = []
        _ref_centers = []
        if verbose:
            print("--- loading reference images and centers")
        for _crop in selected_crops:
            _ref_im = correct_single_image(_ref_filename, bead_channel, crop_limits=_crop,
                                        single_im_size=single_im_size, all_channels=all_channels, 
                                        num_buffer_frames=num_buffer_frames,
                                        num_empty_frames=num_empty_frames,
                                        correction_folder=correction_folder,
                                        illumination_corr=illumination_corr, verbose=verbose)
            _ref_center = visual_tools.get_STD_centers(_ref_im, dynamic=True, th_seed_percentile=ref_seed_per,
                                                       sort_by_h=True, verbose=verbose)
            # limit ref points
            if max_ref_points > 0:
                _ref_center = np.array(_ref_center)[:max(
                    max_ref_points, len(_ref_center)), :]
            # append
            _ref_ims.append(_ref_im)
            _ref_centers.append(_ref_center)
        ## retrieve files requiring drift correction
        args = []
        new_keynames = []
        for _i, _fd in enumerate(folders):
            # extract filename
            _filename = os.path.join(_fd, _fov_name)
            _keyname = os.path.join(os.path.basename(_fd), _fov_name)
            # append all new frames except ref-frame. ref-frame will be assigned to 0
            if _keyname not in old_drift_dic and _keyname != _ref_keyname:
                new_keynames.append(_keyname)
                args.append((_filename, selected_crops, None, _ref_ims, _ref_centers,
                             bead_channel, all_channels, single_im_size,
                             num_buffer_frames, num_empty_frames,
                             ref_seed_per * 0.999**_i, illumination_corr,
                             correction_folder, match_distance, match_unique, 
                             rough_drift_gb, drift_cutoff, verbose))
    
    ## sequential mode
    else:
        # retrieve files requiring drift correction
        args = []
        new_keynames = []
        for _ref_fd, _fd in zip(folders[:-1], folders[1:]):
            # extract filename
            _filename = os.path.join(_fd, _fov_name)  # full filename for image
            _keyname = os.path.join(os.path.basename(
                _fd), _fov_name)  # key name used in dic
            # full filename for ref image
            _ref_filename = os.path.join(_ref_fd, _fov_name)
            if _keyname not in old_drift_dic:
                # append
                new_keynames.append(_keyname)
                args.append((_filename, selected_crops, _ref_filename, None, None,
                             bead_channel, all_channels, single_im_size,
                             num_buffer_frames, num_empty_frames,
                             ref_seed_per, illumination_corr,
                             correction_folder, match_distance, match_unique,
                             rough_drift_gb, drift_cutoff, verbose))

    ## multiprocessing
    if verbose:
        print(
            f"-- Start multi-processing drift correction with {num_threads} threads")
    with mp.Pool(num_threads) as drift_pool:
        align_results = drift_pool.starmap(alignment_tools.align_single_image, args)
        drift_pool.close()
        drift_pool.join()
        drift_pool.terminate()
    # clear
    del(args)
    #killchild()
    # convert to dict
    if not sequential_mode:
        if stringent:
            new_drift_dic = {_ref_name: _ar[0] for _ref_name, _ar in zip(
                new_keynames, align_results) if _ar[1]==0 }
        else:
            new_drift_dic = {_ref_name: _ar[0] for _ref_name, _ar in zip(
                new_keynames, align_results)}
    else:
        _total_dfts = [_ar[0][0] for _ar in zip(align_results)]
        _total_dfts = [sum(_total_dfts[:i+1])
                       for i, _d in enumerate(_total_dfts)]
        new_drift_dic = {_ref_name: _dft for _ref_name,
                         _dft in zip(new_keynames, _total_dfts)}
    # calculate failed count
    fail_count = sum([_ar[1] for _ar in align_results])
    # append ref frame drift
    _ref_keyname = os.path.join(os.path.basename(folders[ref_id]), _fov_name)
    new_drift_dic[_ref_keyname] = np.zeros(3)  # for ref frame, assign to zero
    if old_ref_frame is not None and old_ref_frame in new_drift_dic:
        ref_drift = new_drift_dic[old_ref_frame]
        if verbose:
            print(f"-- drift of reference: {ref_drift}")
        # update drift in old ones
        for _old_name, _old_dft in old_drift_dic.items():
            if _old_name not in new_drift_dic:
                new_drift_dic[_old_name] = _old_dft + ref_drift
    ## save
    if save:
        if verbose:
            print(f"- Saving drift to file:{_save_filename}")
        if not os.path.exists(os.path.dirname(_save_filename)):
            if verbose:
                print(
                    f"-- creating save-folder: {os.path.dirname(_save_filename)}")
            os.makedirs(os.path.dirname(_save_filename))
        pickle.dump(new_drift_dic, open(_save_filename, 'wb'))
    if verbose:
        print(
            f"-- Total time cost in drift correction: {time.time()-_start_time}")
        if fail_count > 0:
            print(f"-- number of failed drifts: {fail_count}")

    return new_drift_dic, fail_count


# illumination correction for one image

def Illumination_correction(im, correction_channel, crop_limits=None, 
                            all_channels=_allowed_colors, single_im_size=_image_size, 
                            cropped_profile=None, correction_folder=_correction_folder,
                            profile_dtype=np.float, image_dtype=np.uint16,
                            ic_profile_name='illumination_correction', correction_power=1, verbose=True):
    """Function to do fast illumination correction in a RAM-efficient manner
    Inputs:
        im: 3d image, np.ndarray or np.memmap
        channel: the color channel for given image, int or string (should be in single_im_size list)
        crop_limits: 2d or 3d crop limits given for this image,
            required if im is already sliced, 2x2 or 3x2 np.ndarray (default: None, no cropping at all)
        all_channels: allowed channels to be corrected, list 
        single_im_size: full image size before any slicing, list of 3 (default:[30,2048,2048])
        correction_folder: correction folder to find correction profile, string of path (default: Z://Corrections/)
        correction_power: power for correction factor, float (default: 1)
        verbose: say something!, bool (default: True)
    Outputs:
        corr_im: corrected image
    """
    ## check inputs
    # im
    if not isinstance(im, np.ndarray) and not isinstance(im, np.memmap):
        raise ValueError(
            f"Wrong input type for im: {type(im)}, np.ndarray or np.memmap expected")
    if verbose:
        print(f"-- correcting illumination for image size:{im.shape} for channel:{correction_channel}")
    # channel
    channel = str(correction_channel)
    if channel not in all_channels:
        raise ValueError(
            f"Input channel:{channel} is not in allowed channels:{all_channels}")
    # check correction profile exists:
    ic_filename = os.path.join(correction_folder,
                               ic_profile_name+'_'+str(channel)+'_'
                               + str(single_im_size[-2])+'x'+str(single_im_size[-1])+'.npy')
    if not os.path.isfile(ic_filename):
        raise IOError(
            f"Illumination correction profile file:{ic_filename} doesn't exist, exit!")
    # image shape and ref-shape
    im_shape = np.array(im.shape, dtype=np.int)
    single_im_size = np.array(single_im_size, dtype=np.int)
    # check crop_limits
    if crop_limits is None:
        if (im_shape[-2:]-single_im_size[-2:]).any():
            raise ValueError(
                f"Test image is not of full image size:{single_im_size}, while crop_limits are not given, exit!")
        else:
            # change crop_limits into full image size
            crop_limits = np.stack([np.zeros(3), im_shape]).T.astype(np.int)
    elif len(crop_limits) <= 1 or len(crop_limits) > 3:
        raise ValueError("crop_limits should have 2 or 3 elements")
    elif len(crop_limits) == 2:
        crop_limits = np.stack(
            [np.array([0, im_shape[0]])] + list(crop_limits)).astype(np.int)
    else:
        crop_limits = np.array(crop_limits, dtype=np.int)
    # convert potential negative values to positive for further calculation
    for _s, _lims in zip(im_shape, crop_limits):
        if _lims[1] < 0:
            _lims[1] += _s
    crop_shape = crop_limits[:, 1] - crop_limits[:, 0]
    # crop image if necessary
    if not (im_shape[-2:]-single_im_size[-2:]).any():
        cim = im[crop_limits[0, 0]:crop_limits[0, 1],
                 crop_limits[1, 0]:crop_limits[1, 1],
                 crop_limits[2, 0]:crop_limits[2, 1] ]
    elif not (im_shape-crop_shape).any():
        cim = im
    else:
        raise IndexError(
            f"Wrong input size for im:{im_shape} compared to crop_limits:{crop_limits}")

    ## do correction
    if cropped_profile is None:
        # get cropped correction profile
        cropped_profile = visual_tools.slice_2d_image(
            ic_filename, single_im_size[-2:], crop_limits[1], crop_limits[2], image_dtype=profile_dtype)
    else:
        if (np.array(cropped_profile.shape) - np.array(cim.shape)[1:]).any():
            raise IndexError(f"Shape of cropped_profile doesnt match cim")
    # correction
    corr_im = (cim / cropped_profile**correction_power).astype(image_dtype)

    return corr_im


def Chromatic_abbrevation_correction(im, correction_channel, target_channel='647', crop_limits=None, 
                                     all_channels=_allowed_colors, single_im_size=_image_size,
                                     drift=np.array([0,0,0]), correction_folder=_correction_folder, 
                                     profile_dtype=np.float, image_dtype=np.uint16,
                                     cc_profile_name='chromatic_correction', verbose=True):
    """Chromatic abbrevation correction for given image and crop
        im: 3d image, np.ndarray or np.memmap
        correction_channel: the color channel for given image, int or string (should be in single_im_size list)
        target_channel: the target channel that image should be correct to, int or string (default: '647')
        crop_limits: 2d or 3d crop limits given for this image,
            required if im is already sliced, 2x2 or 3x2 np.ndarray (default: None, no cropping at all)
        all_channels: allowed channels to be corrected, list 
        single_im_size: full image size before any slicing, list of 3 (default:[30,2048,2048])
        drift: 3d drift of the image, which could be corrected at the same time, list/array of 3
        correction_folder: correction folder to find correction profile, string of path (default: Z://Corrections/)
        profile_dtype: data type for correction profile, numpy datatype (default: np.float)
        image_dtype: image data type, numpy datatype (default: np.uint16)
        cc_profile_name: chromatic correction file basename, str (default: 'chromatic_correction')
        verbose: say something!, bool (default: True)"""
    
    ## check inputs
    # im
    if not isinstance(im, np.ndarray) and not isinstance(im, np.memmap):
        raise ValueError(
            f"Wrong input type for im: {type(im)}, np.ndarray or np.memmap expected")
    if verbose:
        print(f"-- correcting chromatic aberrtion for image size:{im.shape}, channel:{correction_channel}")
    # correction channel
    correction_channel = str(correction_channel)
    if correction_channel not in all_channels:
        raise ValueError(f"Input channel:{correction_channel} is not in allowed channels:{all_channels}")
    # target channel
    target_channel = str(target_channel)
    if target_channel not in all_channels:
        raise ValueError(f"Input channel:{target_channel} is not in allowed channels:{all_channels}")
    # if no correction required, directly return
    if correction_channel == target_channel:
        if verbose:
            print(
                f"--- no chromatic aberrtion required for channel:{correction_channel}")
        return im
    
    # check correction profile exists:
    cc_filename = os.path.join(correction_folder,
                               cc_profile_name+'_'+str(correction_channel)+'_'+str(target_channel)+'_'
                               + str(single_im_size[-2])+'x'+str(single_im_size[-1])+'.npy')
    if not os.path.isfile(cc_filename):
        raise IOError(
            f"Chromatic correction profile file:{cc_filename} doesn't exist, exit!")
    
    # image shape and ref-shape
    im_shape = np.array(im.shape, dtype=np.int)
    single_im_size = np.array(single_im_size, dtype=np.int)

    # check crop_limits
    if crop_limits is None:
        if (im_shape[-2:]-single_im_size[-2:]).any():
            raise ValueError(
                f"Test image is not of full image size:{single_im_size}, while crop_limits are not given, exit!")
        else:
            # change crop_limits into full image size
            crop_limits = np.stack([np.zeros(3), im_shape]).T
    elif len(crop_limits) <= 1 or len(crop_limits) > 3:
        raise ValueError("crop_limits should have 2 or 3 elements")
    elif len(crop_limits) == 2:
        crop_limits = np.stack(
            [np.array([0, im_shape[0]])] + list(crop_limits)).astype(np.int)
    else:
        crop_limits = np.array(crop_limits, dtype=np.int)
    
    # convert potential negative values to positive for further calculation
    for _s, _lims in zip(im_shape, crop_limits):
        if _lims[1] < 0:
            _lims[1] += _s
    crop_shape = crop_limits[:, 1] - crop_limits[:, 0]
    
    # crop image if necessary
    if not (im_shape[-2:]-single_im_size[-2:]).any():
        cim = im[crop_limits[0, 0]:crop_limits[0, 1],
                 crop_limits[1, 0]:crop_limits[1, 1],
                 crop_limits[2, 0]:crop_limits[2, 1] ]
    elif not (im_shape-crop_shape).any():
        cim = im
    else:
        raise IndexError(f"Wrong input size for im:{im_shape} compared to crop_limits:{crop_limits}")
    # check drift
    _drift = np.array(drift[:3])

    ## do correction
    # 1. get coordiates to be mapped
    _coords = np.meshgrid( np.arange(crop_limits[0][1]-crop_limits[0][0]), 
                           np.arange(crop_limits[1][1]-crop_limits[1][0]), 
                           np.arange(crop_limits[2][1]-crop_limits[2][0]))
    _coords = np.stack(_coords).transpose((0, 2, 1, 3)) # transpose is necessary
    # 2. load chromatic profile
    _cropped_cc_profile = visual_tools.slice_image(cc_filename, [3, single_im_size[1], single_im_size[2]],
                                                   [0,3], [crop_limits[1][0],crop_limits[1][1]],
                                                   [crop_limits[2][0],crop_limits[2][1]], image_dtype=profile_dtype)
    # 3. calculate corrected coordinates as a reference
    _corr_coords = _coords + _cropped_cc_profile[:,np.newaxis]
    # 4. add drift if applied
    if _drift.any():
        _corr_coords += _drift[:, np.newaxis,np.newaxis,np.newaxis]
    # 4. map coordinates
    _corr_im = map_coordinates(cim, _corr_coords.reshape(_corr_coords.shape[0], -1), mode='nearest')
    _corr_im = _corr_im.reshape(np.shape(cim))

    return _corr_im

# correct for illumination _shifts across z layers
def Z_Shift_Correction(im, dtype=np.uint16, normalization=False, verbose=False):
    '''Function to correct for each layer in z, to make sure they match in term of intensity'''
    if verbose:
        print("-- correcting Z axis illumination shifts.")
    if not normalization:
        _nim = im / np.median(im, axis=(1, 2))[:,np.newaxis,np.newaxis] * np.median(im)
    else:
        _nim = im / np.median(im, axis=(1, 2))[:,np.newaxis,np.newaxis] * np.median(im)
    return _nim.astype(dtype)

# remove hot pixels
def Remove_Hot_Pixels(im, dtype=np.uint16, hot_pix_th=0.50, hot_th=4, 
                      interpolation_style='nearest', verbose=False):
    '''Function to remove hot pixels by interpolation in each single layer'''
    if verbose:
        print("-- removing hot pixels")
    # create convolution matrix, ignore boundaries for now
    _conv = (np.roll(im,1,1)+np.roll(im,-1,1)+np.roll(im,1,2)+np.roll(im,1,2))/4
    # hot pixels must be have signals higher than average of neighboring pixels by hot_th in more than hot_pix_th*total z-stacks
    _hotmat = im > hot_th * _conv
    _hotmat2D = np.sum(_hotmat,0)
    _hotpix_cand = np.where(_hotmat2D > hot_pix_th*np.shape(im)[0])
    # if no hot pixel detected, directly exit
    if len(_hotpix_cand[0]) == 0:
        return im
    # create new image to interpolate the hot pixels with average of neighboring pixels
    _nim = im.copy()
    if interpolation_style == 'nearest':
        for _x, _y in zip(_hotpix_cand[0],_hotpix_cand[1]):
            if _x > 0 and  _y > 0 and _x < im.shape[1]-1 and  _y < im.shape[2]-1:
                _nim[:,_x,_y] = (_nim[:,_x+1,_y]+_nim[:,_x-1,_y]+_nim[:,_x,_y+1]+_nim[:,_x,_y-1])/4
    return _nim.astype(dtype)


def _mean_xy_profle(im_filename, color, all_colors=_allowed_colors, frame_per_color=30,
                    num_buffer_frames=10, num_empty_frames=1,
                    hot_pixel_remove=True, z_shift_corr=True,   
                    seeding_th_per=80., seeding_crop_size=10, 
                    cap_intensity=True, cap_th_per=99.5, gaussian_sigma=40, return_layers=True):
    """sub-function to calculate mean x-y profile for one specific channel
    Inputs:
        im_filename: image filename, str (*.dax)
        color: targeting color, str or int ({750, 647, 561, 488, 405})
    Outputs:
        mean_profile: mean profile of a given image given channel
    """
    # get image shape
    _im_shape, _num_colors = get_img_info.get_num_frame(im_filename, frame_per_color=frame_per_color,
                                                        buffer_frame=num_buffer_frames, verbose=False)
    _num_frames, _dx, _dy = _im_shape
    # print all_colors
    color = str(color)
    all_colors = [str(_c) for _c in all_colors]
    if color not in all_colors:
        raise ValueError(f"color:{color} must be among all_colors:{all_colors}")
    _color_id = all_colors.index(color)
    _single_im_size = [frame_per_color, _dx, _dy]
    _crop_limits= [[num_buffer_frames, _num_frames-num_buffer_frames], [0, _dx], [0, _dy]]
    
    # slice image
    _im = correct_single_image(im_filename, color, single_im_size=_single_im_size, all_channels=all_colors,
                               num_buffer_frames=num_buffer_frames, num_empty_frames=num_empty_frames, 
                               z_shift_corr=z_shift_corr, hot_pixel_remove=hot_pixel_remove, 
                               illumination_corr=False, chromatic_corr=False,
                               return_limits=False, verbose=False)
    _im = _im.astype(np.float)
    # seeding
    _seeds = visual_tools.get_seed_in_distance(_im, th_seed_percentile=seeding_th_per, dynamic=True)
    for _sd in _seeds:
        l_lim = np.max([np.zeros(len(_sd)), _sd - int(seeding_crop_size/2)], axis=0)
        r_lim = np.min([np.array(_im.shape), _sd + int(seeding_crop_size/2)], axis=0)
        _im[l_lim[0]:r_lim[0], l_lim[1]:r_lim[1], l_lim[2]:r_lim[2]] = np.nan
    if cap_intensity:
        _im = visual_tools.remove_cap(_im, cap_th_per=cap_th_per, fill_nan=True)
    
    if return_layers:
        return _im
    else:
        # calculate mean profile
        _profile = np.nanmedian(_im, axis=0)
        from astropy.convolution import Gaussian2DKernel, convolve
        _profile = convolve(_profile, Gaussian2DKernel(x_stddev=gaussian_sigma))
        
        return _profile

# fast function to generate illumination profiles
def generate_illumination_correction(color, data_folder, correction_folder, num_threads=12,
                                     all_colors=_allowed_colors,  num_of_images=50,
                                     folder_prefix='H', folder_id=0, 
                                     num_buffer_frames=10, num_empty_frames=1, 
                                     frame_per_color=30, gaussian_sigma=40, 
                                     seeding_th_per=90., seeding_crop_size=10,
                                     cap_intensity=False, cap_th_per=99.5,
                                     overwrite=False, save=True, 
                                     save_prefix='illumination_correction_', 
                                     make_plot=False, verbose=True):
    """Function to generate illumination correction profile from hybridization type of image or bead type of image
    Inputs:
        color: color to be corrected, str or int ({750,647,561,488,405}) 
        data_folder: master folder for images, string of path
        correction_folder: folder to save correction_folder, string of path
        num_threads: number of threads to parallelize, int (default: 12)
        all_colors: all allowed colors in images, list of color (default: _allowed_colors)
        num_images: number of images used to do correction, int (default: 50)
        folder_prefix: type of folder to generate this profile, str (default: 'H', which means hyb)
        folder_id: index of folder in the list of folder with given prefix, int (default: 0)
        num_buffer_frames: number of buffer frames before Z-scan, int (default: 10)
        num_empty_frames: number of empty frames before shutter turing on, int (default: 1)
        frame_per_color: number of frames in each color, int (default: 30 for IMR90)
        gaussian_sigma: sigma of gaussian filter used to smooth image, int (default: 40)
        seeding_th_per: percentile of seeding threshold in term of intensity, float (default: 90.)
        seeding_crop_size: size to crop image around seeded local maxima, int (default: 10)
        cap_intensity: whether cap_intensity intensity pixels, bool (default: False)
        cap_th_per: percentile of capping intensity, float (default: 99.5)
        overwrite: whether overwrite existing illumination correction profile, bool (default: False)
        save: whether save profile, bool (default: True)
        save_prefix: save prefix of illumination profile, str (default: 'illumination_correction_{color}')
        make_plot: generate a 2d heatmap for this correction profile, bool (default: False)
        verbose: say something!, bool (default: True)
    Outputs:
        ic_profile: 2d illumination correction profile 
        """
    ## check inputs
    from astropy.convolution import Gaussian2DKernel, convolve
    if verbose:
        print(f"- Generate illumination correction profile for {color}")
    # color
    _color = str(color)
    all_colors = [str(_ch) for _ch in all_colors]
    if _color not in all_colors:
        raise ValueError(f"Wrong color input, {color} is given, color among {all_colors} is expected")
    # image type
    folder_prefix = folder_prefix[0].upper()
    if folder_prefix not in ['H', 'B']:
        raise ValueError(f"Wrong input folder_prefix, {folder_prefix} is given, 'H' or 'B' expected!")
    ## get images
    _folders, _fovs = get_img_info.get_folders(data_folder, feature=folder_prefix,verbose=verbose)
    if len(_folders)==0 or len(_fovs)==0:
        raise IOError(f"No folders or fovs detected with given data_folder:{data_folder} and folder_prefix:{folder_prefix}!")
    ## load image info
    _testfile = os.path.join(_folders[folder_id], _fovs[0])
    _im_shape, _num_colors = get_img_info.get_num_frame(_testfile, frame_per_color=frame_per_color, 
                                                        buffer_frame=num_buffer_frames,
                                                        verbose=verbose)
    _num_frames, _dx, _dy = _im_shape
    if _num_colors != len(all_colors):
        raise ValueError(f"Wrong length of all_colors, should be {_num_colors} but {len(all_colors)} colors are given.")
    # save filename    
    save_filename = os.path.join(correction_folder, save_prefix+str(_color)+'_'+str(_dx)+'x'+str(_dy))
    if os.path.isfile(save_filename+'.npy') and not overwrite:
        if verbose:
            print(f"-- directly loading illumination correction profile from file:{save_filename}.npy")
        _ic_profile = np.load(save_filename+'.npy')
    else:
        _ic_args = []
        for _fov in _fovs:
            # if there are enough images, break
            if len(_ic_args) >= num_of_images:
                break
            # getting args
            _fl = os.path.join(_folders[folder_id], _fov)
            _ic_args.append(
                    (_fl, color, all_colors, frame_per_color, 
                     num_buffer_frames, num_empty_frames,
                     True, True, seeding_th_per, seeding_crop_size, 
                     cap_intensity, cap_th_per,
                     gaussian_sigma, True)
                )
        if verbose:
            print(f"-- {len(_ic_args)} image planned.")
        # multi-processing!
        with mp.Pool(num_threads) as _ic_pool:
            if verbose:
                print(f"-- start multi-processing with {num_threads} threads!")
            _profile_list = _ic_pool.starmap(_mean_xy_profle, _ic_args, chunksize=1)
            _ic_pool.close()
            _ic_pool.join()
            _ic_pool.terminate()
        
        # generate averaged profile
        if verbose:
            print("-- generating averaged profile")
        _ic_profile = np.nanmedian(np.concatenate(_profile_list), axis=0)
        _ic_profile[_ic_profile == 0] = np.nan # remove zeros, which means all images are np.nan
        ## gaussian filter
        if verbose:
            print("-- applying gaussian filter to averaged profile")
        # convolution, which will interpolate any NaN numbers
        _ic_profile = convolve(_ic_profile, Gaussian2DKernel(x_stddev=gaussian_sigma), boundary='extend')
        _ic_profile = _ic_profile / np.max(_ic_profile)
        if save:
            if verbose:
                print(f"-- saving correction profile to file:{save_filename}.npy")
            if not os.path.exists(os.path.dirname(save_filename)):
                os.makedirs(os.path.dirname(save_filename))
            np.save(save_filename, _ic_profile)
    if make_plot:
        plt.figure()
        plt.imshow(_ic_profile)
        plt.colorbar()
        plt.show()

    return _ic_profile

# generate chromatic 
def generate_chromatic_abbrevation_info(ca_filename, ref_filename, ca_channel, ref_channel='647',
                                        single_im_size=_image_size, all_channels=_allowed_colors, 
                                        bead_channel='488', bead_drift_size=300, 
                                        bead_coord_sel=None,
                                        num_buffer_frames=10, num_empty_frames=1,
                                        correction_folder=_correction_folder,
                                        normalization=False, illumination_corr=True, 
                                        th_seed=500, crop_window=9,
                                        remove_boundary_pts=True, rsq_th=0.64, 
                                        save_temp=True, overwrite=False, verbose=True):
    """Generate chromatic_abbrevation coefficient
    Inputs:
        ca_filename: full filename for image having chromatic abbrevation, str of filepath
        ref_filename: full filename for reference image, str of filepath
        ca_channel: channel to calculate chromatic abbrevation, int or str (example: 750)
        ref_channel: channel for reference, int or str (default: 647)
        single_im_size: image size for single channel, array of 3 (default:[30,2048,2048])
        all_channels: all allowed channel list, list of channels (default:[750,647,561,488,405])
        num_buffer_frame: number of frames that is not used in zscan, int (default: 10)
        correction_folder: folder for correction profiles, str(default: _correction_folder)
        normalization" whether do normalization for images, bool (default:False)
        illumination_corr: whether do illumination correction, bool (default: Ture)
        th_seed: seeding threshold for getting candidate centers, int (default: 3000)
        crop_window: window size for cropping, int (default: 9)
        remove_boundary_pts: whether remove points that too close to boundary, bool (default: True)
        rsq_th: threshold for rsquare from linear regression between bldthrough and ref image, float (default: 0.81)
        verbose: say something!, bool (default: True)
    Outputs:
        picked_list: list of dictionary containing all info for chromatic correction"""
    from sklearn.linear_model import LinearRegression
    from .alignment_tools import fast_align_centers
    ## generate ref-image and bleed-through-image
    ref_channel = str(ref_channel)
    ca_channel = str(ca_channel)
    all_channels = [str(ch) for ch in all_channels]
    if ref_channel not in all_channels:
        raise ValueError(f"ref_channel:{ref_channel} should be in all_channels:{all_channels}")
    if ca_channel not in all_channels:
        raise ValueError(f"ca_channel:{ca_channel} should be in all_channels:{all_channels}")
    # bead drift related info
    bead_channel = str(bead_channel)
    if bead_channel not in all_channels:
        raise ValueError(f"bead_channel:{bead_channel} should be in all_channels:{all_channels}")
    if bead_coord_sel is None:
        bead_coord_sel = np.array([single_im_size[-2]/2, single_im_size[-1]/2], dtype=np.int)
    # collect crop coordinates (slices)
    crop0 = np.array([[0, single_im_size[0]],
                      [max(bead_coord_sel[-2]-bead_drift_size, 0), bead_coord_sel[-2]],
                      [max(bead_coord_sel[-1]-bead_drift_size, 0), bead_coord_sel[-1]]], dtype=np.int)
    crop1 = np.array([[0, single_im_size[0]],
                      [bead_coord_sel[-2], min(bead_coord_sel[-2] + bead_drift_size, single_im_size[-2])],
                      [bead_coord_sel[-1], min(bead_coord_sel[-1] + bead_drift_size, single_im_size[-1])]], dtype=np.int)
    crop2 = np.array([[0, single_im_size[0]],
                      [bead_coord_sel[-2], min(bead_coord_sel[-2] + bead_drift_size, single_im_size[-2])],
                      [max(bead_coord_sel[-1] - bead_drift_size, 0), bead_coord_sel[-1]]], dtype=np.int)
    # merge into one array which is easier to feed into function
    selected_crops = np.stack([crop0, crop1, crop2])

    # local parameter, cropping radius
    _radius = int((crop_window-1)/2)
    if _radius < 1:
        raise ValueError(f"Crop radius should be at least 1!")
    # temp_file
    _basename = os.path.basename(ca_filename).replace('.dax', f'_channel_{ca_channel}_ref_{ref_channel}_seed_{th_seed}.pkl')
    _basename = 'chromatic_'+_basename
    temp_filename = os.path.join(os.path.dirname(ca_filename), _basename)
    if os.path.isfile(temp_filename) and not overwrite:
        if verbose:
            print(f"-- directly load from temp_file:{temp_filename}")
        picked_list = pickle.load(open(temp_filename,'rb'))
        return picked_list
    
    if verbose:
        print(f"-- loading reference image: {ref_filename}, channel:{ref_channel}")
    ref_im = correct_single_image(ref_filename, ref_channel, 
                                single_im_size=single_im_size,
                                all_channels=all_channels, num_buffer_frames=num_buffer_frames,
                                num_empty_frames=num_empty_frames,
                                normalization=normalization, correction_folder=correction_folder,
                                z_shift_corr=False, hot_pixel_remove=True,
                                illumination_corr=illumination_corr, chromatic_corr=False,
                                return_limits=False, verbose=verbose)
    if verbose:
        print(f"-- loading chromatic target image: {ca_filename}, channel:{ca_channel}")
    ca_im = correct_single_image(ca_filename, ca_channel, 
                                single_im_size=single_im_size,
                                all_channels=all_channels, num_buffer_frames=num_buffer_frames,
                                num_empty_frames=num_empty_frames,
                                normalization=normalization, correction_folder=correction_folder,
                                z_shift_corr=False, hot_pixel_remove=True,
                                illumination_corr=illumination_corr, chromatic_corr=False,
                                return_limits=False, verbose=verbose)
    # fit centers for both images
    print(ref_filename, ca_filename)
    drift, _drift_flag = alignment_tools.align_single_image(ca_filename, selected_crops, 
                            _ref_filename=ref_filename, _bead_channel=bead_channel,
                            _all_channels=all_channels, _single_im_size=single_im_size,
                            _num_buffer_frames=num_buffer_frames,
                            _num_empty_frames=num_empty_frames,
                            _correction_folder=correction_folder,
                            _illumination_corr=illumination_corr, _verbose=verbose)
    

    # fit centers for ref centers
    ref_centers = visual_tools.get_STD_centers(ref_im, th_seed=th_seed, close_threshold=1., 
                                               save_name=os.path.basename(ref_filename).replace('.dax',f'_{ref_channel}_th{th_seed}.pkl'),
                                               save_folder=os.path.dirname(ref_filename),
                                               verbose=verbose)
    ref_centers = visual_tools.select_sparse_centers(ref_centers, _radius) # pick sparse centers
    # fit centers for chromatic abbreviated centers
    ca_centers = visual_tools.get_STD_centers(ca_im, th_seed=th_seed, close_threshold=1., 
                                              save_name=os.path.basename(ca_filename).replace('.dax',f'_{ca_channel}_th{th_seed}.pkl'),
                                              save_folder=os.path.dirname(ca_filename),
                                              verbose=verbose)
    ca_centers = visual_tools.select_sparse_centers(ca_centers, _radius)
    # correct for drift
    ca_centers -= drift
    # align images
    aligned_ca_centers, aligned_ref_centers = fast_align_centers(ca_centers, ref_centers, 
                                                                 cutoff=_radius, keep_unique=True, return_inds=False)
    # add drift back for ca_centers
    aligned_ca_centers += drift 
        
    ## crop images
    cropped_cas, cropped_refs = [], []

    # loop through all centers
    for ct in aligned_ca_centers:
        if len(ct) != 3:
            raise ValueError(f"Wrong input dimension of centers, only expect [z,x,y] coordinates in center:{ct}")
        crop_l = np.array([np.zeros(3), np.round(ct-_radius)], dtype=np.int).max(0)
        crop_r = np.array([np.array(np.shape(ca_im)), 
                           np.round(ct-_radius)+crop_window], dtype=np.int).min(0)
        cropped_cas.append(ca_im[crop_l[0]:crop_r[0], crop_l[1]:crop_r[1], crop_l[2]:crop_r[2]])
    for ct in aligned_ref_centers:
        if len(ct) != 3:
            raise ValueError(f"Wrong input dimension of centers, only expect [z,x,y] coordinates in center:{ct}")
        crop_l = np.array([np.zeros(3), np.round(ct-_radius)], dtype=np.int).max(0)
        crop_r = np.array([np.array(np.shape(ref_im)), 
                           np.round(ct-_radius)+crop_window], dtype=np.int).min(0)
        cropped_refs.append(ref_im[crop_l[0]:crop_r[0], crop_l[1]:crop_r[1], crop_l[2]:crop_r[2]])
        
        
    # remove centers that too close to boundary
    aligned_ca_centers = list(aligned_ca_centers)
    aligned_ref_centers = list(aligned_ref_centers)

    _keep_flags = np.ones(len(cropped_cas),dtype=np.bool)

    for _i, (_cim, _rim, _cct, _rct) in enumerate(zip(cropped_cas, cropped_refs, aligned_ca_centers, aligned_ref_centers)):
        if remove_boundary_pts:
            if not (np.array(_rim.shape)==crop_window).all() \
                and not (np.array(_cim.shape)==crop_window).all():
                # pop points at boundary
                _keep_flags[_i] = False
        elif np.size(_cim) != np.size(_rim):
            _keep_flags[_i] = False
    
    # pop points at boundaries
    cropped_refs = [_rim for _rim, _flg in zip(cropped_refs, _keep_flags) if _flg]
    cropped_cas = [_cim for _cim, _flg in zip(cropped_cas, _keep_flags) if _flg]
    aligned_ca_centers = [_cct for _cct, _flg in zip(aligned_ca_centers, _keep_flags) if _flg]
    aligned_ref_centers = [_rct for _rct, _flg in zip(aligned_ref_centers, _keep_flags) if _flg]
            
    # check cropped image shape    
    cropped_shape = np.array([np.array(_cim.shape) for _cim in cropped_refs]).max(0)
    if (cropped_shape > crop_window).any():
        raise ValueError(f"Wrong dimension for cropped images:{cropped_shape}, should be of crop_window={crop_window} size")

    if verbose:
        print(f"-- {len(aligned_ca_centers)} internal centers are kept")

    ## final picked list
    picked_list = []
    for _i, (_cim, _rim, _cct, _rct) in enumerate(zip(cropped_cas, cropped_refs, aligned_ca_centers, aligned_ref_centers)):
        
        _x = np.ravel(_rim)[:,np.newaxis]
        _y = np.ravel(_cim)
        if len(_x) != len(_y):
            continue
        _reg = LinearRegression().fit(_x,_y)
        if _reg.score(_x,_y) > rsq_th:
            _pair_dic = {'ref_zxy': _rct,
                         'ca_zxy': _cct - drift,
                         'ref_im': _rim,
                         'ca_im': _cim,
                         'rsquare': _reg.score(_x,_y),
                         'slope': _reg.coef_,
                         'intercept': _reg.intercept_,
                         'ca_file':ca_filename,
                         'ref_file':ref_filename}
            picked_list.append(_pair_dic)
    if verbose:
        print(f"-- {len(picked_list)} pairs kept by rsquare > {rsq_th}")
    if save_temp:
        if verbose:
            print(f"--- saving {len(picked_list)} points to file:{temp_filename}")
        pickle.dump(picked_list, open(temp_filename, 'wb'))
    
    return picked_list

def generate_chromatic_abbrevation_from_spots(corr_spots, ref_spots, 
                                              corr_channel, ref_channel, 
                                              image_size=_image_size, fitting_order=2,
                                              correction_folder=_correction_folder, make_plot=False, 
                                              save=True, save_name='chromatic_correction_',force=False, verbose=True):
    """Code to generate chromatic abbrevation from fitted and matched spots"""
    ## check inputs
    if len(corr_spots) != len(ref_spots):
        raise ValueError("corr_spots and ref_spots are of different length, so not matched")
    # color 
    _allowed_colors = ['750', '647', '561', '488', '405']
    if str(corr_channel) not in _allowed_colors:
        raise ValueError(f"corr_channel given:{corr_channel} is not valid, {_allowed_colors} are expected")
    if str(ref_channel) not in _allowed_colors:
        raise ValueError(f"corr_channel given:{ref_channel} is not valid, {_allowed_colors} are expected")
    # fitting_order
    if isinstance(fitting_order, int):
        fitting_order = [fitting_order]*3
    elif isinstance(fitting_order, list):
        if len(fitting_order) != 3:
            raise ValueError(f"Wrong length of fitting_order, should be 3")
    else:
        raise TypeError(f"Wrong input type of fitting order, should be list or int")
    ## savefile
    filename_base = save_name + str(corr_channel)+'_'+str(ref_channel)+'_'+str(image_size[1])+'x'+str(image_size[2])
    saved_profile_filename = os.path.join(correction_folder, filename_base+'.npy')
    saved_const_filename = os.path.join(correction_folder, filename_base+'_const.npy')
    # whether have existing profile
    if os.path.isfile(saved_profile_filename) and os.path.isfile(saved_const_filename) and not force:
        _cac_profiles = np.load(saved_profile_filename)
        _cac_consts = np.load(saved_const_filename)
        if make_plot:
            for _i,_grid_shifts in enumerate(_cac_profiles):
                plt.figure()
                plt.imshow(_grid_shifts)
                plt.colorbar()
                plt.title(f"chromatic-abbrevation {corr_channel} to {ref_channel}, axis-{_i}")
                plt.show()
    else:
        ## start correction
        _cac_profiles = []
        _cac_consts = []
        # variables used in polyfit
        ref_spots, corr_spots = np.array(ref_spots), np.array(corr_spots) # convert to array
        _x = ref_spots[:,1]
        _y = ref_spots[:,2]


        for _i in range(3): # 3D
            if verbose:
                print(f"-- fitting chromatic-abbrevation in axis {_i} with order:{fitting_order[_i]}")
            # ref
            _data = [] # variables in polyfit
            for _order in range(fitting_order[_i]+1): # loop through different orders
                for _p in range(_order+1):
                    _data.append(_x**_p * _y**(_order-_p))
            _data = np.array(_data).transpose()
            # corr
            _value =  corr_spots[:,_i] - ref_spots[:,_i] # target-value for polyfit
            _C,_r,_r2,_r3 = scipy.linalg.lstsq(_data, _value)    # coefficients and residues
            _cac_consts.append(_C) # store correction constants
            _rsquare =  1 - np.mean((_data.dot(_C) - _value)**2)/np.var(_value)
            if verbose:
                print(f"--- fitted rsquare:{_rsquare}")

            ## generate correction function
            def _get_shift(coords, forder=fitting_order[_i]):
                # traslate into 2d
                if len(coords.shape) == 1:
                    coords = coords[np.newaxis,:]
                _cx = coords[:,1]
                _cy = coords[:,2]
                _corr_data = []
                for _order in range(forder+1):
                    for _p in range(_order+1):
                        _corr_data.append(_cx**_p * _cy**(_order-_p))
                _shift = np.dot(np.array(_corr_data).transpose(), _C)
                return _shift

            ## generate correction_profile
            _xc_t, _yc_t = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[2])) # initialize meshgrid
            _xc, _yc = _xc_t.transpose(), _yc_t.transpose()
            # transpose to the shape compatible with function
            _grid_coords = np.array([np.zeros(np.size(_xc)), _xc.reshape(-1), _yc.reshape(-1)]).transpose() 
            # calculate shift and trasform back
            _grid_shifts = _get_shift(_grid_coords)
            _grid_shifts = _grid_shifts.reshape(np.shape(_xc))
            _cac_profiles.append(_grid_shifts) # store correction profile across 2d

            ## make plot
            if make_plot:
                plt.figure()
                plt.imshow(_grid_shifts)
                plt.colorbar()
                plt.title(f"chromatic-abbrevation {corr_channel} to {ref_channel}, axis-{_i}")
                plt.show()
        _cac_profiles = np.array(_cac_profiles)
        _cac_consts = np.array(_cac_consts)
        # save 
        if save:
            if verbose:
                print(f"-- save profiles to file:{saved_profile_filename}")
            np.save(saved_profile_filename.split('.npy')[0], _cac_profiles)
            if verbose:
                print(f"-- save shift functions to file:{saved_const_filename}")
            np.save(saved_const_filename.split('.npy')[0], _cac_consts)
    
    def _cac_func(coords, consts=_cac_consts, max_order=fitting_order):
        # traslate into 2d
        if len(coords.shape) == 1:
            coords = coords[np.newaxis,:]
        _cx = coords[:,1]
        _cy = coords[:,2]
        _corr_data = []
        for _order in range(max_order+1):
            for _p in range(_order+1):
                _corr_data.append(_cx**_p * _cy**(_order-_p))
        _shifts = []
        for _c in consts:
            _shifts.append(np.dot(np.array(_corr_data).transpose(), _c))
        _shifts = np.array(_shifts).transpose()
        _corr_coords = coords - _shifts
        return _corr_coords    
    
    return _cac_profiles, _cac_func

def Generate_chromatic_abbrevation(target_folder, ref_folder, target_channel, ref_channel='647', 
                                   num_threads=12, start_fov=0, num_image=40,
                                   single_im_size=_image_size, all_channels=_allowed_colors, 
                                   bead_channel='488', bead_drift_size=500, bead_coord_sel=None,
                                   num_buffer_frames=10, num_empty_frames=0,
                                   correction_folder=_correction_folder,
                                   normalization=False, illumination_corr=True, 
                                   th_seed=500, crop_window=9,
                                   remove_boundary_pts=True, rsq_th=0.81, fitting_order=2,
                                   save_temp=True, make_plot=True, save=True, save_name='chromatic_correction_',
                                   overwrite_info=False, overwrite_profile=False, verbose=True):
    """Function to generate chromatic abbrevation correction profile
    Inputs:

        target_channel: channel to calculate chromatic abbrevation, int or str (example: 750)
        ref_channel: channel for reference, int or str (default: 647)
        num_threads: number of threads to generate chromatic abbrevation info, int (default: 12)
        single_im_size: image size for single channel, array of 3 (default:[30,2048,2048])
        all_channels: all allowed channel list, list of channels (default:[750,647,561,488,405])
        num_buffer_frame: number of frames that is not used in zscan, int (default: 10)
        correction_folder: folder for correction profiles, str(default: _correction_folder)
        normalization" whether do normalization for images, bool (default:False)
        illumination_corr: whether do illumination correction, bool (default: Ture)
        th_seed: seeding threshold for getting candidate centers, int (default: 3000)
        crop_window: window size for cropping, int (default: 9)
        remove_boundary_pts: whether remove points that too close to boundary, bool (default: True)
        rsq_th: threshold for rsquare from linear regression between bldthrough and ref image, float (default: 0.81)
        verbose: say something!, bool (default: True)
    Outputs:
        
    """
    ## check inputs

    if verbose:
        print(f"- Start generating chromatic abbrevation correction from {target_channel} to {ref_channel}")
    _target_fovs = [os.path.basename(_fl) for _fl in glob.glob(os.path.join(target_folder, '*.dax'))]
    _ref_fovs = [os.path.basename(_fl) for _fl in glob.glob(os.path.join(ref_folder, '*.dax'))]
    _fovs = [_fov for _fov in sorted(_target_fovs) if _fov in _ref_fovs]
    if start_fov + num_image > len(_fovs):
        raise ValueError(f"Not enough fovs provided to start with {start_fov} and have {num_image} images")
    start_fov = int(start_fov)
    num_image = int(num_image)
    # init args
    _ca_args = []
    for _i, _fov in enumerate(_fovs[start_fov:start_fov+num_image]):
        _ca_file = os.path.join(target_folder, _fov)
        _ref_file = os.path.join(ref_folder, _fov)
        _ca_args.append((
            _ca_file, _ref_file, target_channel, ref_channel, 
            single_im_size, all_channels, bead_channel, bead_drift_size, bead_coord_sel,
            num_buffer_frames, num_empty_frames, correction_folder, normalization, 
            illumination_corr, th_seed, crop_window, remove_boundary_pts, rsq_th, 
            save_temp, overwrite_info, verbose))
    
    # multi-processing
    with mp.Pool(num_threads) as _ca_pool:
        if verbose:
            print(f"-- generating chromatic info for {len(_ca_args)} images in {num_threads} threads.")
        align_results = _ca_pool.starmap(generate_chromatic_abbrevation_info, _ca_args, chunksize=1)
        _ca_pool.close()
        _ca_pool.join()
        _ca_pool.terminate()
    
    align_list = []
    for _paired_list in align_results:
        align_list += _paired_list
    
    ca_centers = [_pair['ca_zxy'] for _pair in align_list]
    ref_centers = [_pair['ref_zxy'] for _pair in align_list]
    
    return generate_chromatic_abbrevation_from_spots(ca_centers, ref_centers, target_channel, ref_channel,
                                                     image_size=single_im_size, fitting_order=fitting_order,
                                                     correction_folder=correction_folder, make_plot=make_plot,
                                                     save=save, save_name=save_name, 
                                                     force=overwrite_profile, verbose=verbose)


def load_correction_profile(channel, corr_type, correction_folder=_correction_folder, ref_channel='647', 
                            im_size=_image_size, verbose=False):
    """Function to load chromatic/illumination correction profile"""
    ## check inputs
    # type
    _allowed_types = ['chromatic', 'illumination']
    _type = str(corr_type).lower()
    if _type not in _allowed_types:
        raise ValueError(f"Wrong input corr_type, should be one of {_allowed_types}")
    # channel
    _allowed_channels = ['750', '647', '561', '488', '405']
    _channel = str(channel).lower()
    _ref_channel = str(ref_channel).lower()
    if _channel not in _allowed_channels:
        raise ValueError(f"Wrong input channel, should be one of {_allowed_channels}")
    if _ref_channel not in _allowed_channels:
        raise ValueError(f"Wrong input ref_channel, should be one of {_allowed_channels}")
    ## start loading file
    _basename = _type+'_correction_'+_channel+'_'
    if _type == 'chromatic':
        _basename += _ref_channel+'_'
    _basename += str(im_size[1])+'x'+str(im_size[2])+'.npy'
    # filename 
    _corr_filename = os.path.join(correction_folder, _basename)
    if os.path.isfile(_corr_filename):
        _corr_profile = np.load(_corr_filename)
    elif _type == 'chromatic' and _channel == _ref_channel:
        return None
    else:
        raise IOError(f"File {_corr_filename} doesn't exist, exit!")

    return np.array(_corr_profile)

# generate bleedthrough 
def _generate_bleedthrough_info_per_image(filename, ref_channel, bld_channel, 
                               single_im_size=_image_size, all_channels=_allowed_colors, 
                               num_buffer_frames=10, num_empty_frames=1,
                               correction_folder=_correction_folder,
                               normalization=False, illumination_corr=True, 
                               th_seed=1500, crop_window=9,
                               remove_boundary_pts=True, rsq_th=0.9, 
                               save_temp=True, force=False, verbose=True):
    """Generate bleedthrough coefficient
    Inputs:
        filename: full filename for image, str
        ref_channel: channel for labeling, int or str (example: 750)
        bld_channel: channel for checking bleedthrough, int or str (example: 647)
        single_im_size: image size for single channel, array of 3 (default:[30,2048,2048])
        all_channels: all allowed channel list, list of channels (default:[750,647,561,488,405])
        num_buffer_frame: number of frames that is not used in zscan, int (default: 10)
        correction_folder: folder for correction profiles, str(default: _correction_folder)
        normalization" whether do normalization for images, bool (default:False)
        illumination_corr: whether do illumination correction, bool (default: Ture)
        th_seed: seeding threshold for getting candidate centers, int (default: 3000)
        crop_window: window size for cropping, int (default: 9)
        remove_boundary_pts: whether remove points that too close to boundary, bool (default: True)
        rsq_th: threshold for rsquare from linear regression between bldthrough and ref image, float (default: 0.81)
        verbose: say something!, bool (default: True)
    Outputs:
        picked_list: list of dictionary containing all info for bleedthrough correction"""
    from sklearn.linear_model import LinearRegression
    if verbose:
        print(f"- find bleedthrough pairs for image: {filename}")
    ## generate ref-image and bleed-through-image
    ref_channel = str(ref_channel)
    bld_channel = str(bld_channel)
    all_channels = [str(ch) for ch in all_channels]
    if ref_channel not in all_channels:
        raise ValueError(f"ref_channel:{ref_channel} should be in all_channels:{all_channels}")
    if bld_channel not in all_channels:
        raise ValueError(f"bld_channel:{bld_channel} should be in all_channels:{all_channels}")
    # if not force and temp file exist, continue
    save_filename = filename.replace('.dax', 
        f'_bleedthrough_{ref_channel}_{bld_channel}_{th_seed}.pkl')
    if not force and os.path.isfile(save_filename):
        if verbose:
            print(f"-- directly load picked list for image: {filename}")
        picked_list = pickle.load(open(save_filename, 'rb'))
    
    if verbose:
        print(f"-- acquiring ref_im and bld_im from file:{filename}")
    ref_im, bld_im = correct_one_dax(filename, [ref_channel, bld_channel], single_im_size=single_im_size, 
                                     all_channels=all_channels, num_buffer_frames=num_buffer_frames,
                                     num_empty_frames=num_empty_frames,
                                     correction_folder=correction_folder,
                                     normalization=normalization, bleed_corr=False, 
                                     z_shift_corr=True, hot_pixel_remove=True,
                                     illumination_corr=illumination_corr, chromatic_corr=False,
                                     return_limits=False, verbose=verbose)

    # get candidate centers
    centers = visual_tools.get_STD_centers(ref_im, th_seed=th_seed, 
                                        save_name=os.path.basename(filename).replace('.dax',f'_{ref_channel}_th{th_seed}.pkl'),
                                        save_folder=os.path.dirname(filename),
                                        verbose=verbose)
    # pick sparse centers
    sel_centers = visual_tools.select_sparse_centers(centers, crop_window)
    
    ## crop images
    cropped_refs, cropped_blds = [], []
    # local parameter, cropping radius
    _radius = int((crop_window-1)/2)
    if _radius < 1:
        raise ValueError(f"Crop radius should be at least 1!")
    # loop through all centers
    for ct in sel_centers:
        if len(ct) != 3:
            raise ValueError(f"Wrong input dimension of centers, only expect [z,x,y] coordinates in center:{ct}")
        crop_l = np.array([np.zeros(3), np.round(ct-_radius)], dtype=np.int).max(0)
        crop_r = np.array([np.array(np.shape(ref_im)), 
                           np.round(ct-_radius)+crop_window], dtype=np.int).min(0)
        cropped_refs.append(ref_im[crop_l[0]:crop_r[0], crop_l[1]:crop_r[1], crop_l[2]:crop_r[2]])
        cropped_blds.append(bld_im[crop_l[0]:crop_r[0], crop_l[1]:crop_r[1], crop_l[2]:crop_r[2]])
    # remove centers that too close to boundary
    sel_centers = list(sel_centers)
    for _i, (_rim, _bim, _ct) in enumerate(zip(cropped_refs, cropped_blds, sel_centers)):
        if remove_boundary_pts and (_rim.shape-np.array([crop_window, crop_window, crop_window], dtype=np.int)).any():
            # pop points at boundary
            cropped_refs.pop(_i)
            cropped_blds.pop(_i)
            sel_centers.pop(_i)
    # check cropped image shape    
    cropped_shape = np.array([np.array(_cim.shape) for _cim in cropped_refs]).max(0)
    if (cropped_shape > crop_window).any():
        raise ValueError(f"Wrong dimension for cropped images:{cropped_shape}, should be of crop_window={crop_window} size")

    if verbose:
        print(f"-- {len(sel_centers)} centers are selected.")
    ## final picked list
    picked_list = []
    for _i, (_rim, _bim, _ct) in enumerate(zip(cropped_refs, cropped_blds, sel_centers)):
        _x = np.ravel(cropped_refs[_i])[:,np.newaxis]
        _y = np.ravel(cropped_blds[_i])
        _reg = LinearRegression().fit(_x,_y)
        if _reg.score(_x,_y) > rsq_th:
            _pair_dic = {'zxy': _ct,
                         'ref_im': _rim,
                         'bld_im': _bim,
                         'rsquare': _reg.score(_x,_y),
                         'slope': _reg.coef_,
                         'intercept': _reg.intercept_,
                         'input_file':filename}
            picked_list.append(_pair_dic)
    
    if save_temp:
        if verbose:
            print(f"-- save temp file: {save_filename}")
        pickle.dump(picked_list, open(save_filename, 'wb'))

    if verbose:
        print(f"-- {len(picked_list)} pairs are saved.")
    

    return picked_list

def generate_bleedthrough_correction_channel(data_folder, target_channel, ref_channel,
                                             num_threads=12, min_spot_num=100, 
                                             start_fov=0, num_image=40,
                                             single_im_size=_image_size, all_channels=_allowed_colors, 
                                             num_buffer_frames=10, num_empty_frames=1,
                                             correction_folder=_correction_folder,
                                             normalization=False, illumination_corr=True, 
                                             th_seed=1500, crop_window=9,
                                             remove_boundary_pts=True, rsq_th=0.9, 
                                             fitting_order=2, 
                                             save_temp=True, save_profile=True,
                                             save_name='bleedthrough_correction_',
                                             save_folder=None, make_plot=True,
                                             overwrite=False, verbose=True):
    """
    """
    ## check inputs
    if verbose:
        print(f"-- Start generating bleedthrough correction from {ref_channel} to {target_channel}")
    _fovs = sorted([os.path.basename(_fl) for _fl in glob.glob(os.path.join(data_folder, '*.dax'))])
    start_fov = int(start_fov)
    num_image = int(num_image)
    if start_fov + num_image > len(_fovs):
        raise ValueError(f"Not enough fovs provided to start with {start_fov} and have {num_image} images")
    # check save
    if save_folder is None:
        save_folder = correction_folder
    _save_basename = save_name+str(ref_channel)+'_'+str(target_channel)
    _save_filename = os.path.join(save_folder, _save_basename +'.pkl')
    if normalization:
        _save_filename = _save_filename.replace('.pkl','_normalized.pkl')
    # if info exists, load
    if os.path.isfile(_save_filename) and not overwrite:
        if verbose:
            print(f"file:{_save_filename} already exists, direct load from file!")
        _spot_lst = pickle.load(open(_save_filename, 'rb'))
    # otherwise start loading
    else:
        # init args
        _bc_args = []
        if verbose:
            print(f"--- looping through folder:{data_folder} to collect information.")
        for _i, _fov in enumerate(_fovs[start_fov: start_fov+num_image]):
            _bc_file = os.path.join(data_folder, _fov)
            _bc_args.append(
                (_bc_file, ref_channel, target_channel, single_im_size, all_channels, 
                num_buffer_frames, num_empty_frames, correction_folder, normalization, 
                illumination_corr, th_seed, crop_window, remove_boundary_pts, rsq_th, 
                save_temp, overwrite, verbose)
            )
        # multi-processing
        with mp.Pool(num_threads) as _bc_pool:
            if verbose:
                print(f"--- generating bleedthrough info for {len(_bc_args)} images in {num_threads} threads.")
            align_results = _bc_pool.starmap(_generate_bleedthrough_info_per_image, 
                                             _bc_args, chunksize=1)
            _bc_pool.close()
            _bc_pool.join()
            _bc_pool.terminate()
        #killchild()  
        # summarize results  
        _spot_lst = []
        for _lst in align_results:
            _spot_lst += _lst
        # save
        if save_temp:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            if verbose:
                print(f"--- saving to file: {_save_filename}")
            pickle.dump(_spot_lst, open(_save_filename, 'wb'))
    
    if verbose:
        print(f"--- {len(_spot_lst)} spots are found as bleedthrough pairs")
    if len(_spot_lst) < min_spot_num:
        print(f"-- Too few spots ({len(_spot_lst)}) are found for interpolation, exit!")
        # return zero matrices
        return np.zeros(single_im_size[1:3]), np.zeros(single_im_size[1:3])

    ## extract information
    _zxys = np.array([_d['zxy'] for _d in _spot_lst])
    _slopes = np.array([_d['slope'] for _d in _spot_lst])
    _intercepts = np.array([_d['intercept'] for _d in _spot_lst])

    ## extract information
    _zxys = np.array([_d['zxy'] for _d in _spot_lst])
    _slopes = np.array([_d['slope'] for _d in _spot_lst])
    _intercepts = np.array([_d['intercept'] for _d in _spot_lst])
    
    ## interpolate to get profile
    # check savefiles
    _slope_savename = os.path.join(save_folder, _save_basename+'_slope')
    _intercept_savename = os.path.join(save_folder, _save_basename+'_intercept')
    if normalization:
        _slope_savename += '_normalized'
        _intercept_savename += '_normalized'
    if os.path.isfile(_slope_savename) and os.path.isfile(_intercept_savename) and not overwrite:
        if verbose:
            print(f"-- directly load profiles from files: {_slope_savename},\n {_intercept_savename}")
        _slope_profile = np.load(_slope_savename+'.npy')
        _intercept_profile = np.load(_intercept_savename+'.npy')
    else:
        if verbose:
            print(f"-- apply polynomial interpolation to get profiles")
        # get data for interpolation
        _x = _zxys[:,1]
        _y = _zxys[:,2]
        _data = [] # variables in polyfit
        for _order in range(fitting_order+1): # loop through different orders
            for _p in range(_order+1):
                _data.append(_x**_p * _y**(_order-_p))
        _data = np.array(_data).transpose()
        # fitting
        if verbose:
            print(f"-- poly-fitting bleedthrough profile with order:{fitting_order}")
        # polyfit for slope
        _C_slope,_,_,_ = scipy.linalg.lstsq(_data, _slopes)    # coefficients and residues
        _rsq_slope =  1 - np.var(_data.dot(_C_slope) - _slopes)/np.var(_slopes)
        if verbose:
            print(f"--- fitted rsquare for slope:{_rsq_slope}")
        # polyfit for intercepts
        _C_intercept,_,_,_ = scipy.linalg.lstsq(_data, _intercepts)    # coefficients and residues
        _rsq_intercept =  1 - np.var(_data.dot(_C_intercept) - _intercepts)/np.var(_intercepts)
        if verbose:
            print(f"--- fitted rsquare for intercepts:{_rsq_intercept}")

        ## generate correction function
        def _get_shift(coords, _C, fitting_order=fitting_order):
            # traslate into 2d
            if len(coords.shape) == 1:
                coords = coords[np.newaxis,:]
            _cx = coords[:,1]
            _cy = coords[:,2]
            _corr_data = []
            for _order in range(fitting_order+1):
                for _p in range(_order+1):
                    _corr_data.append(_cx**_p * _cy**(_order-_p))
            _shift = np.dot(np.array(_corr_data).transpose(), _C)
            return _shift

        ## generate correction_profile
        if verbose:
            print("-- generating polynomial interpolated profiles!")
        # initialize meshgrid
        _xc_t, _yc_t = np.meshgrid(np.arange(single_im_size[1]), np.arange(single_im_size[2])) 
        _xc, _yc = _xc_t.transpose(), _yc_t.transpose()
        # transpose to the shape compatible with function
        _grid_coords = np.array([np.zeros(np.size(_xc)), _xc.reshape(-1), _yc.reshape(-1)]).transpose() 
        # calculate shift and trasform back
        _slope_profile = np.reshape(_get_shift(_grid_coords, _C_slope, fitting_order), np.shape(_xc))
        _intercept_profile = np.reshape(_get_shift(_grid_coords, _C_intercept, fitting_order), np.shape(_xc))
        # save profile 
        if save_profile:
            if not os.path.exists(save_folder):
                if verbose:
                    print(f"-- create save folder:{save_folder}")
                os.makedirs(save_folder)
            if verbose:
                print(f"--- saving slope profile to:{_slope_savename}.npy, \
                        intercept_profile to:{_intercept_savename}.npy")
            np.save(_slope_savename, _slope_profile)
            np.save(_intercept_savename, _intercept_profile)
        
    if make_plot:
        plt.figure()
        plt.imshow(_slope_profile)
        plt.colorbar()
        plt.title(f"-- Slope profile, {ref_channel} to {target_channel}")
        plt.show()
        plt.figure()
        plt.imshow(_intercept_profile)
        plt.colorbar()
        plt.title(f"-- Intercept profile, {ref_channel} to {target_channel}")
        plt.show()

    return _slope_profile, _intercept_profile

def Generate_bleedthrough_correction(folder_list, channel_list, 
                                     num_threads=12, min_spot_num=300, 
                                     start_fov=0, num_image=40,
                                     single_im_size=_image_size, all_channels=_allowed_colors, 
                                     num_buffer_frames=10, num_empty_frames=1,
                                     correction_folder=_correction_folder,
                                     normalization=False, illumination_corr=True, 
                                     th_seed=500, crop_window=9,
                                     remove_boundary_pts=True, rsq_th=0.81, 
                                     fitting_order=2, 
                                     save_temp=True, save_profile=True,
                                     save_name='bleedthrough_correction_',
                                     save_folder=None, make_plot=True,
                                     overwrite=False, verbose=True):
    """Master function to generate bleedthrough correction
    Inputs:
        folder_list: list of folder containing color_swap information, one channel labeling per folder, list
        channel_list: list of channels having label; corresponding to folder_list, list
        num_threads: number of threads to generate profile
        """
    if len(folder_list) != len(channel_list):
        raise ValueError(f"Wrong length of folder_list:{len(folder_list)} compared with channel_list:{len(channel_list)}")
    # check length
    channel_list = [str(ch) for ch in channel_list]
    for _ch in channel_list:
        if _ch not in all_channels:
            raise ValueError(f"Wrong input for channel:{channel_list}, should be among {all_channels}")
    num_threads = int(num_threads)
    # check save_folder
    if save_folder is None:
        save_folder = correction_folder
    # profile name
    _profile_basename = save_name \
                        + '_'.join(sorted(channel_list, key=lambda v:-int(v))) \
                        + f"_{single_im_size[-2]}x{single_im_size[-1]}"
    _profile_filename = os.path.join(save_folder, _profile_basename+'.npy')
    if os.path.isfile(_profile_filename):
        if verbose:
            print(f"-- directly load from file:{_profile_filename}")
        reshaped_profile = np.load(_profile_filename)
    else:
        # initialize bld_corr_profile
        bld_corr_profile = np.zeros([3,3,single_im_size[-2], single_im_size[-1]])
        for _i_ref,_c_ref in enumerate(channel_list):
            for _i_tar, _c_tar in enumerate(channel_list):
                if _c_ref == _c_tar:
                    bld_corr_profile[_i_ref, _i_tar] = np.ones([single_im_size[-2], single_im_size[-1]])
                else:
                    _slope_pf, _intercept_pf = generate_bleedthrough_correction_channel(
                        folder_list[_i_ref], _c_tar, _c_ref, num_threads=num_threads, 
                        min_spot_num=min_spot_num, start_fov=start_fov,
                        num_image=num_image, single_im_size=single_im_size, 
                        all_channels=all_channels, num_buffer_frames=num_buffer_frames, 
                        num_empty_frames=num_empty_frames, correction_folder=correction_folder, 
                        normalization=normalization, illumination_corr=illumination_corr, 
                        th_seed=th_seed, crop_window=crop_window, 
                        remove_boundary_pts=remove_boundary_pts, rsq_th=rsq_th, 
                        fitting_order=fitting_order,
                        save_temp=save_temp, save_profile=save_profile, save_name=save_name, 
                        save_folder=save_folder, make_plot=make_plot, 
                        overwrite=overwrite, verbose=verbose)
                    bld_corr_profile[_i_ref, _i_tar] = _slope_pf

        # transpose first two axes
        bld_corr_profile = bld_corr_profile.transpose((1,0,2,3))
        inv_corr_profile = np.zeros(np.shape(bld_corr_profile), dtype=np.float)
        for _i in range(np.shape(bld_corr_profile)[-2]):
            for _j in range(np.shape(bld_corr_profile)[-1]):
                inv_corr_profile[:,:,_i,_j] = np.linalg.inv(bld_corr_profile[:,:,_i,_j])
        # save 
        reshaped_profile = inv_corr_profile.reshape((9, inv_corr_profile.shape[-2], inv_corr_profile.shape[-1]))
        if verbose:
            print(f"-- saving to file:{_profile_filename}")
        np.save(_profile_filename, reshaped_profile)
        
    return reshaped_profile

# bleedthrough correction
def Bleedthrough_correction(input_im, crop_limits=None, all_channels=_allowed_colors,
                            correction_channels=None, single_im_size=_image_size,
                            num_buffer_frames=10, num_empty_frames=1,
                            drift=np.array([0, 0, 0]), 
                            correction_folder=_correction_folder,
                            normalization=False,
                            z_shift_corr=True, hot_pixel_remove=True,
                            profile_basename='bleedthrough_correction_',
                            profile_dtype=np.float, image_dtype=np.uint16,
                            return_limits=False, verbose=True):
    """Bleedthrough correction for a composite image
    Inputs:
        input_im: input image filename or list of images, str or list
        crop_limits: 2d or 3d crop limits given for this image,
            required if im is already sliced, 2x2 or 3x2 np.ndarray (default: None, no cropping at all)
        all_channels: allowed channels to be corrected, list 
        correction_channels: 3 channels that is going to be corrected, list of strs
        single_im_size: full image size before any slicing, list of 3 (default:[30,2048,2048])
        num_buffer_frames: number of buffer frame in front and back of image, int (default:10)
        drift: 3d drift vector for this image, 1d-array (default:np.array([0,0,0]))
        correction_folder: correction folder to find correction profile, string of path (default: Z://Corrections/)
        profile_basename: base filename for bleedthrough correction profile file, str (default: 'Bleedthrough_correction_matrix')
        profile_dtype: data type for correction profile, numpy datatype (default: np.float)
        image_dtype: image data type, numpy datatype (default: np.uint16)
        return_limits: return modified limits, 3x2 np.ndarray
        verbose: say something!, bool (default: True)
    Outputs:
        _corr_ims: list of corrected images
    """
    ## check inputs
    # input_im
    _start_time = time.time()
    if isinstance(input_im, str):
        if not os.path.isfile(input_im):
            raise IOError(f"input image file:{input_im} doesn't exist, exit!")
    elif isinstance(input_im, list):
        if len(input_im) != 3:
            raise ValueError(
                f"input images should be 3, but {len(input_im)} are given!")
    else:
        raise ValueError(
            f"input_im should be str or list, but {type(input_im)} is given!")
    # crop_limits:
    if crop_limits is None:
        crop_limits = np.array([np.zeros(len(single_im_size)), np.array(
            single_im_size)], dtype=np.int).transpose()[:3]
    if len(crop_limits) != 2 and len(crop_limits) != 3:
        raise ValueError(
            f"crop_limits should be 2d or 3d, but {len(crop_limits)} is given!")
    # correction_channels:
    if correction_channels is None:
        correction_channels = all_channels[:3]
    elif len(correction_channels) != 3:
        raise ValueError("correction_channels should have 3 elements!")
    # correction profile
    _profile_basename = profile_basename \
                        + '_'.join(sorted(correction_channels, key=lambda v:-int(v))) \
                        + f"_{single_im_size[-2]}x{single_im_size[-1]}"
    profile_filename = os.path.join(correction_folder, _profile_basename+'.npy')
    if not os.path.isfile(profile_filename):
        raise IOError(f"Bleedthrough correction profile:{profile_filename} doesn't exist, exit!")\

    ## start correction
    # load image if necessary
    if isinstance(input_im, str):
        _ims, _dft_limits = visual_tools.crop_multi_channel_image(input_im, correction_channels, 
                                                        crop_limits,
                                                        num_buffer_frames, num_empty_frames, 
                                                        all_channels, single_im_size,
                                                        drift, return_limits=True, 
                                                        verbose=verbose)
        # do zshift and hot-pixel correction
        # correct for z axis shift
        _ims = [Z_Shift_Correction(_cim, normalization=normalization,
                    verbose=False) for _cim in _ims]
        # correct for hot pixels
        _ims = [Remove_Hot_Pixels(_cim, hot_th=3, verbose=False) for _cim in _ims]
    elif isinstance(input_im, list):
        if verbose:
            print(f"-- correcting bleedthrough for images")
        _ims = input_im
        _dft_limits = crop_limits
    _ims = [_im.astype(np.float) for _im in _ims]
    #print(time.time()-_start_time)
    # load profile
    if verbose:
        print("--- loading bleedthrough profile")
    _bld_profile = visual_tools.slice_image(profile_filename, [9, single_im_size[-2], single_im_size[-1]],
                                            [0, 9], crop_limits[-2], crop_limits[-1], image_dtype=profile_dtype)
    _bld_profile = _bld_profile.reshape(
        (3, 3, _bld_profile.shape[-2], _bld_profile.shape[-1]))
    #print(time.time()-_start_time)
    # do bleedthrough correction
    if verbose:
        print("--- applying bleedthrough correction")
    _corr_ims = []
    for _i in range(3):
        _nim = _ims[0]*_bld_profile[_i, 0] + _ims[1] * \
            _bld_profile[_i, 1] + _ims[2]*_bld_profile[_i, 2]
        _nim[_nim > np.iinfo(image_dtype).max] = np.iinfo(image_dtype).max
        _nim[_nim < np.iinfo(image_dtype).min] = np.iinfo(image_dtype).min
        _corr_ims.append(_nim.astype(image_dtype))
    #print(time.time()-_start_time)
    if return_limits:
        return _corr_ims, _dft_limits
    else:
        return _corr_ims

## merged function to crop and correct single image
def correct_single_image(filename, channel, crop_limits=None, seg_label=None, extend_dim=20,
                         single_im_size=_image_size, all_channels=_allowed_colors, 
                         num_buffer_frames=10, num_empty_frames=1, 
                         drift=np.array([0, 0, 0]), correction_folder=_correction_folder, normalization=False,
                         z_shift_corr=True, hot_pixel_remove=True, illumination_corr=True, chromatic_corr=True,
                         return_limits=False, verbose=False):
    """wrapper for all correction steps to one image, used for multi-processing
    Inputs:
        filename: full filename of a dax_file or npy_file for one image, string
        channel: channel to be extracted, str (for example. '647')
        crop_limits: 2d or 3d crop limits given for this image,
            required if im is already sliced, 2x2 or 3x2 np.ndarray (default: None, no cropping at all)
        seg_label: segmentation label, 2D array
        extend_dim: extension pixel number if doing cropping, int (default: 20)
        single_im_size: z-x-y size of the image, list of 3
        all_channels: all_channels used in this image, list of str(for example, ['750','647','561'])
        num_buffer_frames: number of buffer frame in front and back of image, int (default:10)
        drift: 3d drift vector for this image, 1d-array
        correction_folder: path to find correction files
        z_shift_corr: whether do z-shift correction, bool (default: True)
        hot_pixel_remove: whether remove hot-pixels, bool (default: True)
        illumination_corr: whether do illumination correction, bool (default: True)
        chromatic_corr: whether do chromatic abbrevation correction, bool (default: True)
        return_limits: whether return cropping limits
        verbose: whether say something!, bool (default:True)
        """
    ## check inputs
    # color channel
    channel = str(channel)
    all_channels = [str(ch) for ch in all_channels]
    if channel not in all_channels:
        raise ValueError(
            f"Target channel {channel} doesn't exist in all_channels:{all_channels}")
    # only 3 all_channels requires chromatic correction
    if channel not in ['750', '647', '561']:
        chromatic_corr = False
    # check filename
    if not isinstance(filename, str):
        raise ValueError(
            f"Wrong input of filename {filename}, should be a string!")
    if not os.path.isfile(filename):
        raise IOError(f"Input filename:{filename} doesn't exist, exit!")
    elif '.dax' not in filename:
        raise IOError("Input filename should be .dax format!")
    # decide crop_limits
    if crop_limits is not None:
        _limits = np.array(crop_limits, dtype=np.int)
    elif seg_label is not None:
        _limits = visual_tools.Extract_crop_from_segmentation(seg_label, extend_dim=extend_dim,
                                                              single_im_size=single_im_size)
    else: # no crop-limit specified
        _limits = None
    # check drift
    if len(drift) != 3:
        raise ValueError(f"Wrong input drift:{drift}, should be an array of 3")

    ## load image
    _ref_name = os.path.join(filename.split(
        os.sep)[-2], filename.split(os.sep)[-1])
    if verbose:
        print(f"- Correcting {_ref_name}, channel:{channel}, params:{num_buffer_frames},{num_empty_frames}")
    _full_im_shape, _num_color = get_img_info.get_num_frame(filename,
                                                            frame_per_color=single_im_size[0],
                                                            buffer_frame=num_buffer_frames)
    # crop image
    _cropped_im, _dft_limits = visual_tools.crop_single_image(filename, channel, crop_limits=_limits,
                                                          all_channels=all_channels,
                                                          drift=drift, single_im_size=single_im_size,
                                                          num_buffer_frames=num_buffer_frames,
                                                          num_empty_frames=num_empty_frames,
                                                          return_limits=True, verbose=verbose)
    ## corrections
    _corr_im = _cropped_im.copy()
    if z_shift_corr:
        # correct for z axis shift
        _corr_im = Z_Shift_Correction(
            _corr_im,  normalization=normalization, verbose=verbose)
    elif not z_shift_corr and normalization:
        # normalize if not doing z_shift_corr
        _corr_im = _corr_im / np.median(_corr_im)
    if hot_pixel_remove:
        # correct for hot pixels
        _corr_im = Remove_Hot_Pixels(_corr_im, hot_th=3, verbose=verbose)
    if illumination_corr:
        # illumination correction
        _corr_im = Illumination_correction(_corr_im, channel,
                                           crop_limits=_dft_limits,
                                           correction_folder=correction_folder,
                                           single_im_size=single_im_size,
                                           verbose=verbose)
    if chromatic_corr:
        # chromatic correction
        _corr_im = Chromatic_abbrevation_correction(_corr_im, channel,
                                                    single_im_size=single_im_size,
                                                    crop_limits=_dft_limits,
                                                    correction_folder=correction_folder,
                                                    verbose=verbose)
    ## return
    if return_limits:
        return _corr_im, _dft_limits
    else:
        return _corr_im

## correct selected channels from one dax file
def correct_one_dax(filename, sel_channels=None, crop_limits=None, seg_label=None,
                    extend_dim=20, single_im_size=_image_size, all_channels=_allowed_colors,
                    num_buffer_frames=10, num_empty_frames=1,
                    drift=np.array([0, 0, 0]),
                    correction_folder=_correction_folder, normalization=False, 
                    bleed_corr=True, z_shift_corr=True, hot_pixel_remove=True, 
                    illumination_corr=True, chromatic_corr=True,
                    return_limits=False, verbose=False):
    """wrapper for all correction steps to one image, used for multi-processing
    Inputs:
        filename: full filename of a dax_file or npy_file for one image, string
        sel_channels: selected channels to be extracted, list of str (for example. ['647'])
        crop_limits: 2d or 3d crop limits given for this image,
            required if im is already sliced, 2x2 or 3x2 np.ndarray (default: None, no cropping at all)
        seg_label: segmentation label, 2D array
        extend_dim: extension pixel number if doing cropping, int (default: 20)
        single_im_size: z-x-y size of the image, list of 3
        all_channels: all_channels used in this image, list of str(for example, ['750','647','561'])
        num_buffer_frames: number of buffer frame in front and back of image, int (default:10)
        drift: 3d drift vector for this image, 1d-array
        correction_folder: path to find correction files
        z_shift_corr: whether do z-shift correction, bool (default: True)
        hot_pixel_remove: whether remove hot-pixels, bool (default: True)
        illumination_corr: whether do illumination correction, bool (default: True)
        chromatic_corr: whether do chromatic abbrevation correction, bool (default: True)
        return_limits: whether return cropping limits, bool (default: False)
        verbose: whether say something!, bool (default:True)
    Outputs:
        _corr_ims: list of corrected images, list
        """
    ## check inputs
    # all_channels:
    all_channels = [str(_ch) for _ch in all_channels]
    # sel_channels:
    if sel_channels is None:
        sel_channels = all_channels[:3]
    else:
        sel_channels = [str(_ch) for _ch in sel_channels]
        for _ch in sel_channels:
            if _ch not in all_channels:
                raise ValueError(
                    f"All channels in selected channels should be in all_channels, but {_ch} is given")
    # correction_channels:
    if len(sel_channels) < 3 and bleed_corr:
        correction_channels = all_channels[:3]
    elif len(sel_channels) > 3:
        raise ValueError("correction_channels should have 3 elements!")
    else:
        correction_channels = sel_channels
    # decide crop_limits
    if crop_limits is not None:
        _limits = np.array(crop_limits, dtype=np.int)
    elif seg_label is not None:  # if segmentation_label is provided, then use this info
        _limits = visual_tools.Extract_crop_from_segmentation(seg_label, extend_dim=extend_dim,
                                                              single_im_size=single_im_size)
    else:  # no crop-limit specified
        _limits = None
    # check drift
    if len(drift) != 3:
        raise ValueError(f"Wrong input drift:{drift}, should be an array of 3")

    ## Start correction
    _ref_name = os.path.join(filename.split(os.sep)[-2],filename.split(os.sep)[-1])
    if verbose:
        print(f"- Start correct one dax file: {_ref_name}")
    # load image by bleedthrough correction
    if bleed_corr:
        _corr_ims, _dft_limits = Bleedthrough_correction(filename, _limits, all_channels=all_channels,
                                                         correction_channels=correction_channels, single_im_size=single_im_size,
                                                         num_buffer_frames=num_buffer_frames, 
                                                         num_empty_frames=num_empty_frames,
                                                         drift=drift,
                                                         correction_folder=correction_folder,
                                                         normalization=normalization,
                                                         z_shift_corr=z_shift_corr, hot_pixel_remove=hot_pixel_remove,
                                                         return_limits=True, verbose=verbose)
    else:
        if verbose:
            print(f"- loading image from {_ref_name} for channels:{correction_channels}")
        _corr_ims, _dft_limits = visual_tools.crop_multi_channel_image(filename, correction_channels, 
                                                                       _limits, num_buffer_frames, 
                                                                       num_empty_frames,
                                                                       all_channels, single_im_size,
                                                                       drift, return_limits=True, 
                                                                       verbose=verbose)


    # proceed with only selected channels
    _corr_ims = [_im for _im, _ch in zip(
        _corr_ims, correction_channels) if str(_ch) in sel_channels]
    # do z-shift and hot-pixel correction
    if not bleed_corr:
        # correct for z axis shift
        if z_shift_corr:
            # correct for z axis shift
            _corr_ims = [Z_Shift_Correction(_cim, normalization=normalization,
                            verbose=verbose) for _cim in _corr_ims]
        elif normalization:
            _corr_ims = [_cim/np.median(_cim) for _cim in _corr_ims]

        if hot_pixel_remove:
            # correct for hot pixels
            _corr_ims = [Remove_Hot_Pixels(
                _cim, hot_th=3, verbose=verbose) for _cim in _corr_ims]
    # illumination and chromatic correction
    if illumination_corr:
        # illumination correction
        _corr_ims = [Illumination_correction(_cim, _ch,
                                             crop_limits=_dft_limits,
                                             correction_folder=correction_folder,
                                             single_im_size=single_im_size,
                                             verbose=verbose) for _cim, _ch in zip(_corr_ims, sel_channels)]
    if chromatic_corr:
        # chromatic correction
        _corr_ims = [Chromatic_abbrevation_correction(_cim, _ch,
                                                      single_im_size=single_im_size,
                                                      crop_limits=_dft_limits,
                                                      correction_folder=correction_folder,
                                                      verbose=verbose) for _cim, _ch in zip(_corr_ims, sel_channels)]
    if return_limits:
        return _corr_ims, _dft_limits
    else:
        return _corr_ims

def multi_correct_one_dax(filename, sel_channels=None, crop_limit_list=None, 
                          seg_label=None, extend_dim=20, 
                          single_im_size=_image_size, all_channels=_allowed_colors,
                          num_buffer_frames=10, num_empty_frames=0,
                          drift=np.array([0, 0, 0]), shift_order=1,
                          bleed_channels=None, 
                          correction_folder=_correction_folder, normalization=False, 
                          bleed_corr=True, z_shift_corr=True, hot_pixel_remove=True, 
                          illumination_corr=True, chromatic_corr=True,
                          return_limits=False, verbose=True):
    """Function to correct multiple-cropped image,
        provided the .dax filename, selected channels, list of 3d-crop limits
    *******************************************************************************
    Inputs:
        filename: image filename of .dax format, string of file path
        sel_channels: selected channels to be cropped, str or list of str
        crop_limit_list: 
    Outputs:

    """
    ## check inputs
    # all_channels:
    all_channels = [str(_ch) for _ch in all_channels]
    # bleed_channels
    if bleed_channels is None:
        bleed_channels = all_channels[:3]
    # sel_channels:
    if sel_channels is None:
        sel_channels = all_channels[:3]
    elif isinstance(sel_channels, str) or isinstance(sel_channels, int) or isinstance(sel_channels, np.int):
        sel_channels = [str(sel_channels)]
    elif isinstance(sel_channels, list):
        sel_channels = [str(_ch) for _ch in sel_channels]
        for _ch in sel_channels:
            if _ch not in all_channels:
                raise ValueError(f"All channels in selected channels should be in all_channels, but {_ch} is given")
    else:
        raise TypeError(f"Wrong input types for sel_channels, should be a string or int or list, but {type(sel_channels)} is given.")
    # all channels requires corrections
    if bleed_corr:
        correction_channels = bleed_channels
    else:
        correction_channels = []
    for _ch in sel_channels:
        if _ch not in correction_channels:
            correction_channels.append(_ch)

    # decide crop_limits
    if crop_limit_list is not None:
        _limit_list = crop_limit_list
    elif seg_label is not None:  # if segmentation_label is provided, then use this info
        _limit_list = []
        for _l in np.unique(seg_label):
            if _l > 0:
                _limits = visual_tools.Extract_crop_from_segmentation((seg_label==_l), 
                                                                    extend_dim=extend_dim,
                                                                    single_im_size=single_im_size)
                _limit_list.append(_limits)
    else:  # no crop-limit specified, crop the whole image
        _limit_list = [np.stack([np.zeros(len(single_im_size)), single_im_size]).T.astype(np.int)]
    # check drift
    if len(drift) != 3:
        raise ValueError(f"Wrong input drift:{drift}, should be an array of 3")

    ## 1. Crop image based on drift
    _cropped_im_list, _drift_limit_list = io_tools.load.multi_crop_image_fov(
        filename, correction_channels, _limit_list, all_channels=all_channels,
        single_im_size=single_im_size, num_buffer_frames=num_buffer_frames,
        num_empty_frames=num_empty_frames, drift=drift, shift_order=shift_order,
        return_limits=True, verbose=verbose
        )

    ## 2. Corrections
    if bleed_corr:
        for _i, (_cims, _limits) in enumerate(zip(_cropped_im_list, _drift_limit_list)):
            _bc_ims = Bleedthrough_correction(
                _cims[:3], _limits, all_channels=all_channels,
                correction_channels=bleed_channels, 
                single_im_size=single_im_size,
                num_buffer_frames=num_buffer_frames, 
                num_empty_frames=num_empty_frames,
                drift=drift,
                correction_folder=correction_folder,
                normalization=normalization,
                z_shift_corr=z_shift_corr, hot_pixel_remove=hot_pixel_remove,
                return_limits=False, verbose=verbose)
            for _im,_ch in zip(_bc_ims, bleed_channels):
                if _ch in correction_channels:
                    _cropped_im_list[_i][correction_channels.index(_ch)] = _im
                
    # come back to sel_channels:
    if verbose:
        print(f"-- select {sel_channels} from {correction_channels}")
    for _i, _ims in enumerate(_cropped_im_list):
        _sel_ims = [_im for _im, _ch in zip(_ims, correction_channels) if _ch in sel_channels]
        # append
        _cropped_im_list[_i] = _sel_ims
    
    # correct for z axis shift
    if z_shift_corr:
        for _i, _cims in enumerate(_cropped_im_list):
            # correct for z axis shift
            _cropped_im_list[_i] = [Z_Shift_Correction(_cim, normalization=normalization,
                                    verbose=verbose) for _cim in _cims]
    elif normalization:
        for _i, _cims in enumerate(_cropped_im_list):
            _cropped_im_list[_i] = [_cim/np.median(_cim) for _cim in _cims]

    if hot_pixel_remove:
        # correct for hot pixels
        for _i, _cims in enumerate(_cropped_im_list):
            _cropped_im_list[_i] = [Remove_Hot_Pixels(_cim, hot_th=3, 
                                    verbose=verbose) for _cim in _cims]
    
    # illumination and chromatic correction
    if illumination_corr:
        # illumination correction
        for _i, (_ims, _dft_limits) in enumerate(zip(_cropped_im_list, _drift_limit_list)):
            _corr_ims = [Illumination_correction(_cim, _ch,
                                                crop_limits=_dft_limits,
                                                correction_folder=correction_folder,
                                                single_im_size=single_im_size,
                                                verbose=verbose) for _cim, _ch in zip(_ims, sel_channels)]
            # append
            _cropped_im_list[_i] = _corr_ims
    if chromatic_corr:
        # chromatic correction
        for _i, (_ims, _dft_limits) in enumerate(zip(_cropped_im_list, _drift_limit_list)):
            _corr_ims = [Chromatic_abbrevation_correction(_cim, _ch,
                                                        single_im_size=single_im_size,
                                                        crop_limits=_dft_limits,
                                                        correction_folder=correction_folder,
                                                        verbose=verbose) for _cim, _ch in zip(_ims, sel_channels)]
            # append
            _cropped_im_list[_i] = _corr_ims
    
    if return_limits:
        return _cropped_im_list, _drift_limit_list
    else:
        return _cropped_im_list            