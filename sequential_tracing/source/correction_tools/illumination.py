# import functions from packages
import os
import time 
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import scoreatpercentile
from scipy.ndimage import gaussian_filter

# import local variables
from .. import _allowed_colors, _distance_zxy, _image_size, _correction_folder
from ..io_tools.load import correct_fov_image

def Generate_Illumination_Correction(data_folder, sel_channels=None, 
                                     num_loaded_files=40, 
                                     single_im_size=_image_size, all_channels=_allowed_colors,
                                     num_buffer_frames=10, num_empty_frames=0, 
                                     correction_folder=_correction_folder,
                                     hot_pixel_corr=True, hot_pixel_th=4, z_shift_corr=False,
                                     remove_cap=True, cap_th_per=[5, 90],
                                     gaussian_filter_size=60, 
                                     save=True, overwrite=False, save_folder=None,
                                     save_prefix='illumination_correction_', 
                                     make_plot=True, verbose=True):
    """Function to generate illumination corrections for given channels
    Inputs:
    
    Outputs:
        """
    
    ## check inputs
    _total_start = time.time()
    if sel_channels is None:
        sel_channels = all_channels
    # check save folders
    if save_folder is None:
        save_folder = os.path.join(data_folder, 'Corrections')
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    _save_filenames = [os.path.join(save_folder, f"{save_prefix}{_ch}_{single_im_size[-2]}x{single_im_size[-1]}.npy")
                       for _ch in sel_channels]
    
    # directly load these channels
    _loaded_pfs = [np.load(_fl) for _fl in _save_filenames if os.path.isfile(_fl) and not overwrite]
    _loaded_channels = [_ch for _ch, _fl in zip(sel_channels, _save_filenames) if os.path.isfile(_fl) and not overwrite]
    if verbose:
        print(f"-- directly load:{_loaded_channels} illumination profiles for files")
    # get channels to be loaded
    _sel_channels = [_ch for _ch, _fl in zip(sel_channels, _save_filenames) if not (os.path.isfile(_fl) and not overwrite)]
    _sel_filenames = [_fl for _ch, _fl in zip(sel_channels, _save_filenames) if not (os.path.isfile(_fl) and not overwrite)]
    # start load images if any channels selected
    if len(_sel_channels) > 0:
        if verbose:
            print(f"-- start calculating {_sel_channels} illumination profiles")

        ## detect dax files
        _fovs = [_fl for _fl in os.listdir(data_folder) if _fl.split('.')[-1]=='dax']
        _num_load = min(num_loaded_files, len(_fovs))
        if verbose:
            print(f"-- {_num_load} among {len(_fovs)} dax files will be loaded in data_folder: {data_folder}")
        # get input daxfiles
        _input_fls = [os.path.join(data_folder, _fl) for _fl in _fovs[:_num_load]]
        # load images
        _signal_sums = [np.zeros([single_im_size[-2], single_im_size[-1]]) for _c in _sel_channels]
        _layer_cts = [np.zeros([single_im_size[-2], single_im_size[-1]]) for _c in _sel_channels]

        # loop through files
        for _fl in _input_fls:
            if verbose:
                print(f"--- {os.path.basename(_fl)}: correction, ", end='')
            _ims = correct_fov_image(_fl, _sel_channels,
                                    single_im_size=single_im_size, all_channels=all_channels,
                                    num_buffer_frames=num_buffer_frames, num_empty_frames=num_empty_frames,
                                    calculate_drift=False, corr_channels=_sel_channels,
                                    hot_pixel_corr=hot_pixel_corr, hot_pixel_th=hot_pixel_th,
                                    z_shift_corr=z_shift_corr,
                                    illumination_corr=False, chromatic_corr=False, bleed_corr=False,
                                    return_drift=False, verbose=False)
            _start_time = time.time()
            for _i, (_ch, _im) in enumerate(zip(_sel_channels, _ims)):
                if verbose:
                    print(_ch, end=' ')
                _limits = [scoreatpercentile(_im, min(cap_th_per)), scoreatpercentile(_im, max(cap_th_per))]
                _nim = np.clip(_im, min(_limits), max(_limits))
                _signal_sums[_i] += gaussian_filter(np.sum(_nim, axis=0), gaussian_filter_size)
                #_layer_cts[_i] += np.sum((_im > min(_limits)) & (_im < max(_limits)), axis=0)
                #_im[(_im < min(_limits)) | (_im > max(_limits))] = 0
                #_signal_sums[_i] += np.sum(_im, axis=0)
            if verbose:
                print(f" in {time.time()-_start_time:.2f}s")
        if verbose:
            print(f"-- calculating mean profiles")
        # do gaussian filter
        _sel_pfs = [gaussian_filter(_s/len(_input_fls), gaussian_filter_size) for _s in _signal_sums]
        # normalize
        _sel_pfs = [_pf / np.max(_pf) for _pf in _sel_pfs]
        
        # save
        if save:
            if verbose:
                print(f"-- saving updated profiles")
            for _ch, _pf, _fl in zip(_sel_channels, _sel_pfs, _sel_filenames):
                if verbose:
                    print(f"--- saving {_ch} profile into file: {_fl}")
                np.save(_fl.split('.npy')[0], _pf)
            
    # merge illumination profiles:
    _illumination_pfs = []
    for _ch in sel_channels:
        if _ch in _sel_channels:
            _illumination_pfs.append(_sel_pfs[_sel_channels.index(_ch)])
        elif _ch in _loaded_channels:
            _illumination_pfs.append(_loaded_pfs[_loaded_channels.index(_ch)])
        else:
            raise IndexError(f"channel: {_ch} doesn't exist in either _sel_channels or _loaded_channels!")
    
    if make_plot:
        for _ch, _pf, _fl in zip(sel_channels, _illumination_pfs, _save_filenames):
            plt.figure(dpi=150, figsize=(4,3))
            plt.imshow(_pf, )
            plt.colorbar()
            plt.title(f"illumination, channel:{_ch}")
            if save:
                plt.savefig(_fl.replace('.npy', '.png'), transparent=True)
            plt.show()
    if verbose:
        print(f"-- finish generating illumination profiles, time:{time.time()-_total_start:.2f}s")
    return _illumination_pfs