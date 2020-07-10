## Load other sub-packages
from .. import visual_tools, get_img_info

## Load shared parameters
from . import _distance_zxy, _image_size, _allowed_colors 
from . import _num_buffer_frames, _num_empty_frames

## load packages
import numpy as np
import scipy
import os, sys, glob, time


def decide_starting_frames(channels, num_channels=None, all_channels=_allowed_colors, 
                           num_buffer_frames=10, num_empty_frames=0, verbose=False):
    """Function to decide starting frame ids given channels"""
    if num_channels is None:
        if verbose:
            print(f"num_channels is not given, thus use default: {len(all_channels)}")
        num_channels = len(all_channels)
    else:
        all_channels = all_channels[:num_channels]
    # check channels
    if not isinstance(channels, list):
        raise TypeError(f"channels should be a list but {type(channels)} is given.")
    _channels = [str(_ch) for _ch in channels]
    for _ch in _channels:
        if _ch not in all_channels:
            raise ValueError(f"Wrong input channel:{_ch}, should be among {all_channels}")\
    # check num_buffer_frames and num_empty_frames
    num_buffer_frames = int(num_buffer_frames)
    num_empty_frames = int(num_empty_frames)
    
    # get starting frames
    _start_frames = [(all_channels.index(_ch)-num_buffer_frames+num_empty_frames)%num_channels\
                     + num_buffer_frames for _ch in _channels]
    
    return _start_frames
    

    
def translate_crop_by_drift(crop3d, drift3d=np.array([0,0,0]), single_im_size=_image_size):
    
    crop3d = np.array(crop3d, dtype=np.int)
    drift3d = np.array(drift3d)
    single_im_size = np.array(single_im_size, dtype=np.int)
    # deal with negative upper limits    
    for _i, (_lims, _s) in enumerate(zip(crop3d, single_im_size)):
        if _lims[1] < 0:
            crop3d[_i,1] += _s
    _drift_limits = np.zeros(crop3d.shape, dtype=np.int)
    # generate drifted crops
    for _i, (_d, _lims) in enumerate(zip(drift3d, crop3d)):
        _drift_limits[_i, 0] = max(_lims[0]-np.ceil(np.abs(_d)), 0)
        _drift_limits[_i, 1] = min(_lims[1]+np.ceil(np.abs(_d)), single_im_size[_i])
    return _drift_limits

def generate_neighboring_crop(zxy, crop_size=5, 
                              single_im_size=_image_size,
                              sub_pixel_precision=False):
    """Function to generate crop given zxy coordinate and crop size
    Inputs:
    Output:
    """
    ## check inputs
    _zxy =  np.array(zxy)[:len(single_im_size)]
    _crop_size = int(crop_size)
    _single_image_size = np.array(single_im_size, dtype=np.int)
    # find limits for this crop
    if sub_pixel_precision:
        pass
    else:
        _left_lims = np.max([np.round(_zxy-_crop_size), 
                            np.zeros(len(_single_image_size))], axis=0)
        _right_lims = np.min([np.round(_zxy+_crop_size), 
                              _single_image_size], axis=0)
        _crop = tuple([slice(int(_l), int(_r)) 
                       for _l,_r in zip(_left_lims,_right_lims) ])
    
    return _crop