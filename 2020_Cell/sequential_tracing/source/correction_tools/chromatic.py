# import functions from packages
import os
import time 
import pickle 
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import scoreatpercentile
from scipy.ndimage import gaussian_filter

# import local variables
from .. import _allowed_colors, _distance_zxy, _image_size, _correction_folder
from ..io_tools.load import correct_fov_image

def generate_warp_function(chromatic_consts=None, 
                            drift=None,
                            n_dim=3,
                            verbose=True):
    """Function to generate a spot translating function"""
    ## check inputs
    if chromatic_consts is None:
        _ch_consts = np.zeros([n_dim,1])
    else:
        _ch_consts = chromatic_consts
    if drift is None:
        _drift = np.zeros(n_dim)
    else:
        _drift = drift[:n_dim]

    def _shift_function(coords, _ch_consts=_ch_consts, 
                        _drift=_drift):
        if np.shape(coords)[1] == n_dim:
            _coords = np.array(coords).copy()
        elif np.shape(coords)[1] == 11: # this means 3d fitting result
            _coords = np.array(coords).copy()[:,4-n_dim:4]
        else:
            raise ValueError(f"Wrong input coords")
        _shifts = []
        for _idim in range(n_dim):
            _consts = np.array(_ch_consts[_idim])
            _ft_order = np.int(np.sqrt(len(_consts)*2+0.25)-1.5) # only dependent on 2d
            _corr_data = []
            for _order in range(_ft_order+1):
                for _p in range(_order+1):
                    _corr_data.append(_coords[:,-2]**_p \
                                      * _coords[:,-1]**(_order-_p))
            _shifts.append(np.dot(np.array(_corr_data).transpose(), _consts))
        # generate corrected coordinates
        _corr_coords = _coords - np.stack(_shifts).transpose() - _drift
        
        # return as input
        if np.shape(coords)[1] == n_dim:
            _output_coords = _corr_coords
        elif np.shape(coords)[1] == 11: # this means 3d fitting result
            _output_coords = np.array(coords).copy()
            _output_coords[:,4-n_dim:4] = _corr_coords
        return _output_coords
    
    # return function
    return _shift_function

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