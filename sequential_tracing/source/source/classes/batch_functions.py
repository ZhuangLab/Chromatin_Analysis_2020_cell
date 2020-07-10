# Functions used in batch processing
import os, h5py, pickle, psutil, time
import numpy as np

from . import _allowed_kwds, _image_dtype
from ..io_tools.load import correct_fov_image
from ..spot_tools.fitting import fit_fov_image, get_centers

_seed_th={
    '750': 400,
    '647': 600,
    '561': 500,
}

## Process managing
def killtree(pid, including_parent=False, verbose=False):
    """Function to kill all children of a given process"""
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        if verbose:
            print ("child", child)
        child.kill()
    if including_parent:
        parent.kill()

def killchild(verbose=False):
    """Easy function to kill children of current process"""
    _pid = os.getpid()
    killtree(_pid, False, verbose)

## Parsing existing files
def _color_dic_stat(color_dic, channels, _type_dic=_allowed_kwds):
    """Extract number of targeted datatype images in color_dic"""
    _include_types = {}
    for _name, _k in _type_dic.items():
        for _fd, _infos in color_dic.items():
            for _ch, _info in zip(channels, _infos):
                if len(_info) > 0 and _info[0] == _k:
                    if _name not in _include_types:
                        _include_types[_name] = {'ids':[], 'channels':[]}
                    # append
                    _include_types[_name]['ids'].append(int(_info.split(_k)[1]))
                    _include_types[_name]['channels'].append(_ch)
    # sort
    for _name, _dict in _include_types.items():
        _ids = _dict['ids']
        _chs = _dict['channels']
        _sorted_ids = [_id for _id in sorted(_ids)]
        _sorted_chs = [_ch for _id,_ch in sorted(zip(_ids, _chs))]
        _include_types[_name]['ids'] = _sorted_ids
        _include_types[_name]['channels'] = _sorted_chs
        
    return _include_types



def batch_process_image_to_spots(dax_filename, sel_channels, ref_filename, 
                                 load_file_lock=None, 
                                 correction_args={}, 
                                 save_image=False, save_filename=None, 
                                 data_type=None, region_ids=None,
                                 warp_image=True,
                                 image_file_lock=None, 
                                 overwrite_image=False, 
                                 drift_args={}, save_drift=True, 
                                 drift_filename=None, 
                                 drift_file_lock=None, 
                                 overwrite_drift=False, 
                                 fitting_args={}, save_spots=True, 
                                 spot_file_lock=None, overwrite_spot=False, 
                                 return_drift=False,
                                 verbose=False):
    """run by multi-processing to batch process images to spots
    Inputs:

    Outputs:
        _spots: fitted spots for this image
    """
    ## check inputs
    # dax_filename
    if not os.path.isfile(dax_filename):
        raise IOError(f"Dax file: {dax_filename} is not a file, exit!")
    if not isinstance(dax_filename, str) or dax_filename[-4:] != '.dax':
        raise IOError(f"Dax file: {dax_filename} has wrong data type, exit!")
    # selected channels
    sel_channels = [str(ch) for ch in sel_channels]
    if verbose:
        print(f"+ batch process image: {dax_filename} for channels:{sel_channels}")
    # ref_Filename 
    if not os.path.isfile(ref_filename):
        raise IOError(f"Dax file: {ref_filename} is not a file, exit!")
    if not isinstance(ref_filename, str) or ref_filename[-4:] != '.dax':
        raise IOError(f"Dax file: {ref_filename} has wrong data type, exit!")
    ## judge if drift correction is required
    if drift_filename is None:
        drift_folder = os.path.join(os.path.dirname(os.path.dirname(dax_filename)),
                            'Analysis', 'drift')
        drift_filename = os.path.join(drift_folder, 
                            os.path.basename(dax_filename).replace('.dax', '_current_cor.pkl'))
    _key = os.path.join(os.path.basename(os.path.dirname(dax_filename)),
                        os.path.basename(dax_filename))
    # try to load drift
    if os.path.isfile(drift_filename):
        _drift_dict = pickle.load(open(drift_filename, 'rb'))
    else:
        _drift_dict = {}
    if _key in _drift_dict and not overwrite_drift:
        if verbose:
            print(f"-- load drift from drift_dict: {drift_filename}")
        _drift = _drift_dict[_key]
        _corr_drift = False 
    else:
        if verbose:
            print(f"-- no existing drift loaded, initialize drift.")
        _drift = np.array([0.,0.,0.])
        if ref_filename == dax_filename:
            _corr_drift = False
        else:
            _corr_drift = True
    # check save_image parameters
    if save_image:
        if data_type not in _allowed_kwds:
            raise ValueError(f"Wrong input data_type:{data_type}, should be among {_allowed_kwds}.")
        if save_filename is None:
            raise ValueError(f"Input save_filename:{save_filename} should be given.")
        if region_ids is None:
            raise ValueError(f"Input region_ids:{region_ids} should be given.")
        if len(region_ids) != len(sel_channels):
            raise ValueError(f"Wrong input region_ids:{region_ids}, should of same length as sel_channels:{sel_channels}.")
        region_ids = [int(_id) for _id in region_ids] # convert to ints
    ## correct images
    if warp_image:
        _sel_ims, _drift = correct_fov_image(dax_filename, 
                                sel_channels, 
                                load_file_lock=load_file_lock,
                                calculate_drift=_corr_drift, 
                                drift=_drift,
                                ref_filename=ref_filename, 
                                warp_image=warp_image,
                                return_drift=True, verbose=verbose, 
                                **correction_args, **drift_args)
    else:
        _sel_ims, _warp_funcs, _drift = correct_fov_image(
                                dax_filename, 
                                sel_channels, 
                                load_file_lock=load_file_lock,
                                calculate_drift=_corr_drift, 
                                drift=_drift,
                                ref_filename=ref_filename, 
                                warp_image=warp_image,
                                return_drift=True, verbose=verbose, 
                                **correction_args, **drift_args)                
    ## save image if specified
    if save_image:
        # initiate lock
        if 'image_file_lock' in locals() and image_file_lock is not None:
            image_file_lock.acquire()
        # run saving
        _save_img_success = save_image_to_fov_file(
            save_filename, _sel_ims, data_type, region_ids, 
            warp_image, _drift, overwrite_image, verbose)
        # release lock
        if 'image_file_lock' in locals() and image_file_lock is not None:
            image_file_lock.release()

    ## save drift if specified
    if save_drift:
        # initiate lock
        if 'drift_file_lock' in locals() and drift_file_lock is not None:
            drift_file_lock.acquire()
        # run saving
        _save_drift_success = save_drift_to_file(drift_filename,
                                                dax_filename, _drift, 
                                                overwrite_drift, verbose)
        # release lock
        if 'drift_file_lock' in locals() and drift_file_lock is not None:
            drift_file_lock.release()

    ## multi-fitting
    _spot_list = []
    for _ich, (_im, _ch) in enumerate(zip(_sel_ims, sel_channels)):
        _spots = fit_fov_image(
            _im, _ch, verbose=verbose, 
            **fitting_args,
        )
        if not warp_image:
            # update spot coordinates given warp functions, if image was not warpped.
            _func = _warp_funcs[_ich]
            _spots = _func(_spots)
        # append 
        _spot_list.append(_spots)
    ## save fitted_spots if specified
    if save_spots:
        # initiate lock
        if spot_file_lock is not None:
            spot_file_lock.acquire()
        # run saving
        _save_spt_success = save_spots_to_fov_file(
            save_filename, _spot_list, data_type, region_ids, 
            overwrite_spot, verbose)
        # release lock
        if spot_file_lock is not None:
            spot_file_lock.release()
    if return_drift:
        return _spot_list, _drift
    else:
        return _spot_list
    
# save image to fov file
def save_image_to_fov_file(filename, ims, data_type, region_ids, 
                           warp_image=False, drift=None,
                           overwrite=False, verbose=True):
    """Function to save image to fov-standard savefile(hdf5)
    Inputs:
        filename: fov class hdf5 saving filename, string of file path
        ims: images to be saved, list of np.ndarray 
        data_type: data type used to load, string
        region_ids: corresponding region ids of given data_type, 
            should match length of ims, list of ints
        warp_image: whether image was warpped or not, bool (default: False)
        drift: whether drift exist and whether we are going to save it, bool (default: None, not saving)
        overwrite: whether overwrite existing data, bool (default: False)
        verbose: say something!, bool (default: True)
    Outputs:
    """
    ## check inputs
    if not os.path.isfile(filename):
        raise IOError(f"save file: {filename} doesn't exist!")
    if data_type not in _allowed_kwds:
        raise ValueError(f"Wrong input data_type:{data_type}, should be among {_allowed_kwds}.")
    if len(ims) != len(region_ids):
        raise ValueError(f"Wrong input region_ids:{region_ids}, should of same length as ims, len={len(ims)}.")
    if drift is not None:
        if len(np.shape(drift)) == 1:
            _all_drifts = [drift for _im in ims]
        elif len(drift) == len(ims):
            _all_drifts = drift
        else:
            raise IndexError(f"Length of drift should match ims")
    if verbose:
        print(f"- writting {data_type} info to file:{filename}")
        _save_start = time.time()
    _updated_ims = []
    _updated_drifts = []
    _saving_flag = False 
    ## start saving
    with h5py.File(filename, "a", libver='latest') as _f:
        _grp = _f[data_type]
        for _i, (_id, _im) in enumerate(zip(region_ids, ims)):
            _index = list(_grp['ids'][:]).index(_id)
            _flag = _grp['flags'][_index]
            # if not been written or overwrite:
            if _flag == 0 or overwrite:
                _saving_flag = True 
                _grp['ims'][_index] = _im
                # warpping image flag
                if not warp_image:
                    _grp['flags'][_index] = 1
                else:
                    _grp['flags'][_index] = 2 # 2 as markers of warpped images
                _updated_ims.append(_id)
                if drift is not None:
                    _grp['drifts'][_index] = _all_drifts[_i]
                    _updated_drifts.append(_id)
                
    if verbose:
        if _saving_flag:
            print(f"-- updated ims for id:{_updated_ims}, drifts for id:{_updated_drifts} in {time.time()-_save_start:.3f}s")
        else:
            print(f"-- images and drifts already exist, skip.")

    # return success flag    
    return _saving_flag

# load image from fov file
def load_image_from_fov_file(filename, data_type, region_ids,
                             image_dtype=_image_dtype, verbose=True):
    """Function to load images from fov class file
    Inputs:
        filename: fov class hdf5 saving filename, string of file path
        data_type: data type used to load, string
        region_ids: corresponding region ids of given data_type, list of ints
        verbose: say something!, bool (default: True)
    Outputs:
        _ims: images in the order of region_ids provided, list of np.ndarray
        _flags: whether these images were warpped (==2), list of ints
    """
    ## check inputs
    if not os.path.isfile(filename):
        raise IOError(f"load file: {filename} doesn't exist!")
    if data_type not in _allowed_kwds:
        raise ValueError(f"Wrong input data_type:{data_type}, should be among {_allowed_kwds}.")
    if isinstance(region_ids, int) or isinstance(region_ids, np.int):
        _region_ids = [int(region_ids)]
    elif isinstance(region_ids, list) or isinstance(region_ids, np.ndarray):
        _region_ids = [int(_id) for _id in region_ids]
    else:
        raise TypeError(f"Wrong input type for region_ids:{region_ids}")
    
    if verbose:
        print(f"- loading {data_type} info from file:{filename}", end=' ')
        _load_start = time.time()
    ## start loading
    _ims = []
    _flags = []
    with h5py.File(filename, "a", libver='latest') as _f:
        # get the group
        _grp = _f[data_type]
        # get index
        for _i, _id in enumerate(_region_ids):
            _index = list(_grp['ids'][:]).index(_id)
            # extract images and flag
            _ims.append(_grp['ims'][_index])
            _flags.append(_grp['flags'][_index])
    if verbose:
        print(f"in {time.time()-_load_start:.3f}s.")
        
    return _ims, _flags

# save image to fov file
def save_spots_to_fov_file(filename, spot_list, data_type, region_ids, 
                           overwrite=False, verbose=True):
    """Function to save image to fov-standard savefile(hdf5)
    Inputs:
    
    Outputs:
    """
    ## check inputs
    if not os.path.isfile(filename):
        raise IOError(f"save file: {filename} doesn't exist!")
    if data_type not in _allowed_kwds:
        raise ValueError(f"Wrong input data_type:{data_type}, should be among {_allowed_kwds}.")
    if len(spot_list) != len(region_ids):
        raise ValueError(f"Wrong input region_ids:{region_ids}, should of same length as spots, len={len(spot_list)}.")

    if verbose:
        print(f"- writting {data_type} spots into file:{filename}")
        _save_start = time.time()
    _updated_spots = []
    ## start saving
    with h5py.File(filename, "a", libver='latest') as _f:
        _grp = _f[data_type]
        for _i, (_id, _spots) in enumerate(zip(region_ids, spot_list)):
            _index = list(_grp['ids'][:]).index(_id)
            if np.sum(_grp['spots'][_index])==0 or overwrite:
                _grp['spots'][_index, :len(_spots), :] = _spots
                _updated_spots.append(_id)

    if verbose:
        print(f"-- updated spots for id:{_updated_spots} in {time.time()-_save_start:.3f}s")
    # return success flag    
    return True

# save drift to file
def save_drift_to_file(drift_filename, image_filename, drift, overwrite=False, verbose=True):
    """Save drift for one image to drift file"""
    ## check inputs
    if os.path.isfile(drift_filename):
        drift_dict = pickle.load(open(drift_filename, 'rb'))
    else:
        drift_dict = {}
    # update drift
    _update = False
    _key = os.path.join(os.path.basename(os.path.dirname(image_filename)),
                        os.path.basename(image_filename))
    if _key not in drift_dict or overwrite:
        drift_dict[_key] = drift
        _update = True
    # save
    if _update:
        if verbose:
            print(f"-- update drift of {_key} into file:{drift_filename}")
        pickle.dump(drift_dict, open(drift_filename, 'wb'))
    else:
        if verbose:
            print(f"-- no updates in drift, skip.")
    # return success flag
    return True
    

# create drift file
def create_drift_file(drift_filename, ref_filename, 
                      n_dim=3,
                      overwrite=False, verbose=True):
    """Function to create drift save file"""
    ## check inputs
    if os.path.isfile(drift_filename) and not overwrite:
        drift_dict = pickle.load(open(drift_filename, 'rb'))
    else:
        drift_dict = {}
    _ref_key = os.path.join(os.path.basename(os.path.dirname(ref_filename)),
                            os.path.basename(ref_filename))
    if _ref_key not in drift_dict:
        drift_dict[_ref_key] = np.zeros(n_dim)
        _update = True
    else:
        _update = False
    if _update:
        # create folder 
        if not os.path.isdir(os.path.dirname(drift_filename)):
            if verbose:
                print(f"--- creating folder:{os.path.dirname(drift_filename)}")
            os.makedirs(os.path.dirname(drift_filename))
        # save initialized drift_dict
        if verbose:
            print(f"-- create drift file:{drift_filename} with reference:{_ref_key}")
        pickle.dump(drift_dict, open(drift_filename, 'wb'))
    else:
        if verbose:
            print(f"-- no updates in drift file:{drift_filename}, skip.")
    
    return True 

