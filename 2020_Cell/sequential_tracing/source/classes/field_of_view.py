import sys
import glob
import os
import time
import copy
import numpy as np
import pickle as pickle
import multiprocessing as mp
# saving
import h5py
import ast
# plotting
import matplotlib
import matplotlib.pyplot as plt

# import other sub-packages
# import package parameters
from .. import _correction_folder, _corr_channels, _temp_folder,_distance_zxy,\
    _sigma_zxy,_image_size, _allowed_colors, _num_buffer_frames, _num_empty_frames, _image_dtype
from . import _allowed_kwds, _max_num_seeds, _min_num_seeds, _spot_seeding_th

def __init__():
    print(f"Loading field of view class")
    pass

class Field_of_View():
    """Class of field-of-view of a certain sample, which includes all possible files across hybs and parameters"""
    
    def __init__(self, parameters, 
                 _fov_id=None, _fov_name=None,
                 _load_references=True, _color_info_kwargs={},
                 _create_savefile=True, _save_filename=None,
                 _savefile_kwargs={},
                 _segmentation_kwargs={},
                 _load_all_attrs=True,
                 _overwrite_attrs=False,
                 _verbose=True,
                 ):
        ## Initialize key attributes:
        #: attributes for unprocessed images:

        # correction profiles 
        self.correction_profiles = {'bleed':None,
                                    'chromatic':None,
                                    'illumination':None,}
        # drifts
        self.drift = {}
        # rotations
        self.rotation = {}
        # segmentation
        if 'segmentation_dim' not in _segmentation_kwargs:
            self.segmentation_dim = 2 # default is 2d segmentation
        else:
            self.segmentation_dim = int(_segmentation_kwargs['segmentation_dim'])

        #: attributes for processed images:
        # splitted processed images
        self.im_dict = {}
        # channel dict corresponding to im_dict
        self.channel_dict = {}

        ## check input datatype
        if not isinstance(parameters, dict):
            raise TypeError(f'wrong input type of parameters, should be dict containing essential info, but {type(parameters)} is given!')

        ## required parameters: 
        # data_folder: str of folder or list of str of folders
        if 'data_folder' not in parameters:
            raise KeyError(f"data_folder is required in parameters.")
        if isinstance(parameters['data_folder'], list):
            self.data_folder = [str(_fd) for _fd in parameters['data_folder']]
        else:
            self.data_folder = [str(parameters['data_folder'])]
        ## extract hybe folders and field-of-view names
        self.folders = []
        for _fd in self.data_folder:
            from ..get_img_info import get_folders
            _hyb_fds, _fovs = get_folders(_fd, feature='H', verbose=True)
            self.folders += _hyb_fds # here only extract folders not fovs

        if _fov_name is None and _fov_id is None:
            raise ValueError(f"either _fov_name or _fov_id should be given!")
        elif _fov_id is not None:
            _fov_id = int(_fov_id)
            # define fov_name
            _fov_name = _fovs[_fov_id]
        else:
            _fov_name = str(_fov_name)
            if _fov_name not in _fovs:
                raise ValueError(f"_fov_name:{_fov_name} should be within fovs:{_fovs}")
            _fov_id = _fovs.index(_fov_name)
        # append fov information 
        self.fov_id = _fov_id
        self.fov_name = _fov_name

        # experiment_folder
        if 'experiment_folder'  in parameters:
            self.experiment_folder = parameters['experiment_folder']
        else:
            self.experiment_folder = os.path.join(self.data_folder[0], 'Experiment')
        ## analysis_folder, segmentation_folder, save_folder, correction_folder,map_folder
        if 'analysis_folder'  in parameters:
            self.analysis_folder = str(parameters['analysis_folder'])
        else:
            self.analysis_folder = os.path.join(self.data_folder[0], 'Analysis')
        if 'segmentation_folder' in parameters:
            self.segmentation_folder = parameters['segmentation_folder']
        else:
            self.segmentation_folder = os.path.join(self.analysis_folder, 'segmentation')
        # save folder
        if 'save_folder' in parameters:
            self.save_folder = parameters['save_folder']
        else:
            self.save_folder = os.path.join(self.analysis_folder,'save')
        if 'correction_folder' in parameters:
            self.correction_folder = parameters['correction_folder']
        else:
            self.correction_folder = _correction_folder
        if 'drift_folder' in parameters:
            self.drift_folder = parameters['drift_folder']
        else:
            self.drift_folder =  os.path.join(self.analysis_folder, 'drift')
        if 'map_folder' in parameters:
            self.map_folder = parameters['map_folder']
        else:
            self.map_folder = os.path.join(self.analysis_folder, 'distmap')
        # number of num_threads
        if 'num_threads' in parameters:
            self.num_threads = parameters['num_threads']
        else:
            self.num_threads = int(os.cpu_count() / 4) # default: use one third of cpus.
        # ref_id
        if 'ref_id' in parameters:
            self.ref_id = int(parameters['ref_id'])
        else:
            self.ref_id = 0
        ## shared_parameters
        # initialize
        if 'shared_parameters' in parameters:
            self.shared_parameters = parameters['shared_parameters']
        else:
            self.shared_parameters = {}
        # add parameter keys:
        if 'image_dtype' not in self.shared_parameters:    
            self.shared_parameters['image_dtype'] = _image_dtype        
        if 'distance_zxy' not in self.shared_parameters:    
            self.shared_parameters['distance_zxy'] = _distance_zxy
        if 'sigma_zxy' not in self.shared_parameters:
            self.shared_parameters['sigma_zxy'] = _sigma_zxy
        if 'single_im_size' not in self.shared_parameters:
            self.shared_parameters['single_im_size'] = _image_size
        if 'num_buffer_frames' not in self.shared_parameters:
            self.shared_parameters['num_buffer_frames'] = _num_buffer_frames
        if 'num_empty_frames' not in self.shared_parameters:
            self.shared_parameters['num_empty_frames'] = _num_empty_frames
        if 'normalization' not in self.shared_parameters:
            self.shared_parameters['normalization'] = False
        if 'corr_channels' not in self.shared_parameters:
            self.shared_parameters['corr_channels'] = _corr_channels
        if 'corr_bleed' not in self.shared_parameters:
            self.shared_parameters['corr_bleed'] = True
        if 'corr_Z_shift' not in self.shared_parameters:
            self.shared_parameters['corr_Z_shift'] = True
        if 'corr_hot_pixel' not in self.shared_parameters:
            self.shared_parameters['corr_hot_pixel'] = True
        if 'corr_illumination' not in self.shared_parameters:
            self.shared_parameters['corr_illumination'] = True
        if 'corr_chromatic' not in self.shared_parameters:
            self.shared_parameters['corr_chromatic'] = True
        if 'allowed_kwds' not in self.shared_parameters:
            self.shared_parameters['allowed_data_types'] = _allowed_kwds
        # params for drift
        if 'max_num_seeds' not in self.shared_parameters:
            self.shared_parameters['max_num_seeds'] = _max_num_seeds
        if 'min_num_seeds' not in self.shared_parameters:
            self.shared_parameters['min_num_seeds'] = _min_num_seeds
        if 'drift_size' not in self.shared_parameters:
            self.shared_parameters['drift_size'] = 600
        if 'drift_use_fft' not in self.shared_parameters:
            self.shared_parameters['drift_use_fft'] = True 
        if 'drift_sequential' not in self.shared_parameters:
            self.shared_parameters['drift_sequential'] = False 
        if 'good_drift_th' not in self.shared_parameters:
            self.shared_parameters['good_drift_th'] = 1. 
        # param for spot_finding
        if 'spot_seeding_th' not in self.shared_parameters:
            self.shared_parameters['spot_seeding_th'] = _spot_seeding_th
        if 'normalize_intensity_local' not in self.shared_parameters:
            self.shared_parameters['normalize_intensity_local'] = True
    
        ## load experimental info
        if _load_references:
            if '_color_filename' not in _color_info_kwargs:
                self.color_filename = 'Color_Usage'
                _color_info_kwargs['_color_filename'] = self.color_filename
            else:
                self.color_filename = _color_info_kwargs['_color_filename']
            if '_color_format' not in _color_info_kwargs:
                self.color_format = 'csv'
                _color_info_kwargs['_color_format'] = self.color_format
            else:
                self.color_format = _color_info_kwargs['_color_format']
            _color_dic = self._load_color_info(_annotate_folders=True, **_color_info_kwargs)
        
        ## Drift
        # update ref_filename
        self.ref_filename = os.path.join(self.annotated_folders[self.ref_id], self.fov_name)
        # update drift filename
        _dft_fl_postfix = '_current_cor.pkl'
        if self.shared_parameters['drift_sequential']:
            _dft_fl_postfix = '_sequential'+_dft_fl_postfix
        self.drift_filename = os.path.join(self.drift_folder,
                                            self.fov_name.replace('.dax', _dft_fl_postfix))
        # generate drift crops
        from ..correction_tools.alignment import generate_drift_crops
        self.drift_crops = generate_drift_crops(
                                drift_size=self.shared_parameters['drift_size'],
                                single_im_size=self.shared_parameters['single_im_size'],
                            )

        ## Create savefile
        # save filename
        if _save_filename is None:
            _save_filename = os.path.join(self.save_folder, self.fov_name.replace('.dax', '.hdf5'))
        # set save_filename attr
        self.save_filename = _save_filename

        # initialize save file
        if _create_savefile:
            self._init_save_file(_save_filename=_save_filename, 
                                 _overwrite=_overwrite_attrs,
                                 **_savefile_kwargs)


    ## Load basic info
    def _load_color_info(self, _color_filename=None, _color_format=None, 
                         _save_color_dic=True, _annotate_folders=False):
        """Function to load color usage representing experimental info"""
        ## check inputs
        if _color_filename is None:
            _color_filename = self.color_filename
        if _color_format is None:
            _color_format = self.color_format
        from ..get_img_info import Load_Color_Usage, find_bead_channel, find_dapi_channel
        _color_dic, _use_dapi, _channels = Load_Color_Usage(self.analysis_folder,
                                                            color_filename=_color_filename,
                                                            color_format=_color_format,
                                                            return_color=True)
        
        # need-based store color_dic
        if _save_color_dic:
            self.color_dic = _color_dic
        # store other info
        self.use_dapi = _use_dapi
        self.channels = [str(ch) for ch in _channels]
        # channel for beads
        _bead_channel = find_bead_channel(_color_dic)
        self.bead_channel_index = _bead_channel
        _dapi_channel = find_dapi_channel(_color_dic)
        self.dapi_channel_index = _dapi_channel

        # get annotated folders by color usage
        if _annotate_folders:
            self.annotated_folders = []
            for _hyb_fd, _info in self.color_dic.items():
                _matches = [_fd for _fd in self.folders if _hyb_fd == _fd.split(os.sep)[-1]]
                if len(_matches)==1:
                    self.annotated_folders.append(_matches[0])
            print(f"- {len(self.annotated_folders)} folders are found according to color-usage annotation.")

        return _color_dic
    
    ### Here are some initialization functions
    def _init_save_file(self, _save_filename=None, 
                        _overwrite=False, _verbose=True):
        """Function to initialize save file for FOV object
        Inputs:
            _save_filename: full path for filename saving this dataset.
            _overwrite: whether overwrite existing info within save_file, bool (default: False)
            _verbose: say something!, bool (default: True)
        Outputs:
            save_file created, current info saved.
        """
        if _save_filename is None:
            _save_filename = getattr(self, 'save_filename')
        # set save_filename attr
        setattr(self, 'save_filename', _save_filename)
        if _verbose: 
            if not os.path.exists(_save_filename):
                print(f"- Creating save file for fov:{self.fov_name}: {_save_filename}.")
            else:
                print(f"- Initialize save file for fov:{self.fov_name}: {_save_filename}.")

        ## initialize fov_info, segmentation and correction
        for _type in ['fov_info', 'segmentation', 'correction']:
            self._save_to_file(_type, _overwrite=_overwrite, _verbose=_verbose)
        
        ## initialize image data types
        from .batch_functions import _color_dic_stat
        # acquire valid types
        _type_dic = _color_dic_stat(self.color_dic, 
                                    self.channels, 
                                    self.shared_parameters['allowed_data_types']
                                   )
        # create
        for _type, _dict in _type_dic.items():
            self._save_to_file(_type, _overwrite=_overwrite, _verbose=_verbose)
        
        return

    def _old_init_save_file(self, _save_filename=None, 
                        _overwrite=False, _verbose=True):
        """Function to initialize save file for FOV object"""
        if _save_filename is None:
            _save_filename = getattr(self, 'save_filename')
        # set save_filename attr
        setattr(self, 'save_filename', _save_filename)
        if _verbose and not os.path.exists(_save_filename):
            print(f"- Creating save file for fov:{self.fov_name}: {_save_filename}")
        with h5py.File(_save_filename, "a", libver='latest') as _f:
            if _verbose:
                print(f"- Updating info for fov:{self.fov_name}: {_save_filename}")
            ## self specific attributes stored directly in attributes:
            _base_attrs = []
            for _attr_name in dir(self):
                # exclude all default attrs and functions
                if _attr_name[0] != '_' and getattr(self, _attr_name) is not None:
                    # set default to be save
                    _info_attr_flag = True
                    # if included into data_type, not save here
                    for _name in self.shared_parameters['allowed_data_types'].keys():
                        # give some criteria
                        if _name in _attr_name:
                            _info_attr_flag = False
                            break
                    # if its image dict, exclude
                    if 'im_dict' in _attr_name or 'channel_dict' in _attr_name:
                        _info_attr_flag = False
                    # if its segmentation, exclude
                    if 'segmentation' in _attr_name:
                        _info_attr_flag = False
                    # if its related to correction, exclude
                    if 'correction' in _attr_name:
                        _info_attr_flag = False
                    ## all the rest attrs saved to here: 
                    # save here:
                    if _info_attr_flag:
                        # extract the attribute
                        _attr = getattr(self, _attr_name)
                        # convert dict if necessary
                        if isinstance(_attr, dict):
                            _attr = str(_attr)
                        # save
                        if _attr_name not in _f.attrs or _overwrite:
                            _f.attrs[_attr_name] = _attr
                            _base_attrs.append(_attr_name)
            if _verbose:
                print(f"-- base attributes updated:{_base_attrs}")
                        
            ## segmentation
            if 'segmentation' not in _f.keys():
                _grp = _f.create_group('segmentation') # create segmentation group
            else:
                _grp = _f['segmentation']
            
            # directly create segmentation label dataset
            if 'segmentation_label' not in _grp:
                _seg = _grp.create_dataset('segmentation_label', 
                                        self.shared_parameters['single_im_size'][-self.segmentation_dim:], 
                                        dtype='i8')
                if hasattr(self, 'segmentation_label'):
                    _grp['segmentation_label'] = getattr(self, 'segmentation_label')
            # create other segmentation related datasets
            for _attr_name in dir(self):
                if _attr_name[0] != '_' and 'segmentation' in _attr_name and _attr_name not in _grp.keys():
                    _grp[_attr_name] = getattr(self, _attr_name)
            
            # create label for each datatype
            from .batch_functions import _color_dic_stat
            _type_dic = _color_dic_stat(self.color_dic, self.channels, self.shared_parameters['allowed_data_types'])
            for _data_type, _dict in _type_dic.items():
                if _data_type not in _f.keys():
                    _grp = _f.create_group(_data_type) # create data_type group
                else:
                    _grp = _f[_data_type]
                # record updated data_type related attrs
                _data_attrs = []
                # save images, ids, channels
                # calculate image shape and chunk shape
                _im_shape = np.concatenate([np.array([len(_dict['ids'])]), 
                                            self.shared_parameters['single_im_size']])
                _chunk_shape = np.concatenate([np.array([1]), 
                                            self.shared_parameters['single_im_size']])                              
                # change size
                _change_size_flag = []
                # if missing any of these features, create new ones
                # ids
                if 'ids' not in _grp:
                    _ids = _grp.create_dataset('ids', (len(_dict['ids']),), dtype='i', data=_dict['ids'])
                    _ids = np.array(_dict['ids'], dtype=np.int) # save ids
                    _data_attrs.append('ids')
                elif len(_dict['ids']) != len(_grp['ids']):
                    _change_size_flag.append('id')
                    _old_size=len(_grp['ids'])
                # channels
                if 'channels' not in _grp:
                    _channels = [_ch.encode('utf8') for _ch in _dict['channels']]
                    _chs = _grp.create_dataset('channels', (len(_dict['channels']),), dtype='S3', data=_channels)
                    _chs = np.array(_dict['channels'], dtype=str) # save ids
                    _data_attrs.append('channels')
                elif len(_dict['channels']) != len(_grp['channels']):
                    _change_size_flag.append('channels')
                    _old_size=len(_grp['channels'])
                # images
                if 'ims' not in _grp:
                    _ims = _grp.create_dataset('ims', tuple(_im_shape), dtype='u16', chunks=tuple(_chunk_shape))
                    _data_attrs.append('ims')
                elif len(_im_shape) != len(_grp['ims'].shape) or (_im_shape != (_grp['ims']).shape).any():
                    _change_size_flag.append('ims')
                    _old_size=len(_grp['ims'])

                # spots
                if 'spots' not in _grp:
                    _spots = _grp.create_dataset('spots', 
                                (_im_shape[0], self.shared_parameters['max_num_seeds'], 11), 
                                dtype='f')
                    _data_attrs.append('spots')
                elif _im_shape[0] != len(_grp['spots']):
                    _change_size_flag.append('spots')
                    _old_size=len(_grp['spots'])
                # drift
                if 'drifts' not in _grp:
                    _drift = _grp.create_dataset('drifts', (_im_shape[0], 3), dtype='f')
                    _data_attrs.append('drifts')
                elif _im_shape[0] != len(_grp['drifts']):
                    _change_size_flag.append('drifts')
                    _old_size=len(_grp['drifts'])
                # flags for whether it's been written
                if 'flags' not in _grp:
                    _filenames = _grp.create_dataset('flags', (_im_shape[0], ), dtype='u8')
                    _data_attrs.append('flags')
                elif _im_shape[0] != len(_grp['flags']):
                    _change_size_flag.append('flags')
                    _old_size=len(_grp['flags'])

                # if change size, update these features:
                if len(_change_size_flag) > 0:
                    print(f"* data size of {_data_type} is changing from {_old_size} to {len(_dict['ids'])} because of {_change_size_flag}")
                    ###UNDER CONSTRUCTION################
                    pass
                # elsif size don't change, also load other related dtypes
                else:
                    for _attr_name in dir(self):
                        if _attr_name[0] != '_' and _data_type in _attr_name:
                            if _attr_name not in _grp.keys() or _overwrite:
                                _grp[_attr_name] = getattr(self, _attr_name)
                                _data_attrs.append(_attr_name)
                # summarize
                if _verbose:
                    print(f"-- {_data_type} attributes updated:{_data_attrs}")

    def _DAPI_segmentation(self):
        pass

    def _load_correction_profiles(self, _correction_folder=None, 
                                  _corr_channels=['750','647','561'],
                                  _chromatic_target='647',
                                  _profile_postfix='.npy', _verbose=True):
        """Function to laod correction profiles in RAM"""
        from ..io_tools.load import load_correction_profile
        # determine correction folder
        if _correction_folder is None:
            _correction_folder = self.correction_folder
        # loading bleedthrough
        if self.shared_parameters['corr_bleed']:
            self.correction_profiles['bleed'] = load_correction_profile('bleedthrough', self.shared_parameters['corr_channels'], 
                                        self.correction_folder, all_channels=self.channels, 
                                        im_size=self.shared_parameters['single_im_size'],
                                        verbose=_verbose)
        # loading chromatic
        if self.shared_parameters['corr_chromatic']:
            self.correction_profiles['chromatic']= load_correction_profile('chromatic', self.shared_parameters['corr_channels'], 
                                        self.correction_folder, all_channels=self.channels, 
                                        im_size=self.shared_parameters['single_im_size'],
                                        verbose=_verbose)
            self.correction_profiles['chromatic_constants']= load_correction_profile(
                'chromatic_constants', 
                self.shared_parameters['corr_channels'], 
                self.correction_folder, all_channels=self.channels, 
                im_size=self.shared_parameters['single_im_size'],
                verbose=_verbose)
            
        # load illumination
        if self.shared_parameters['corr_illumination']:
            bead_channel = str(self.channels[self.bead_channel_index])
            self.correction_profiles['illumination'] = load_correction_profile('illumination', self.shared_parameters['corr_channels']+[bead_channel], 
                                                self.correction_folder, all_channels=self.channels, 
                                                im_size=self.shared_parameters['single_im_size'],
                                                verbose=_verbose)
        return
    
    ## load existing drift info
    def _load_drift_file(self , _drift_basename=None, _drift_postfix='_current_cor.pkl', 
                         _sequential_mode=False, _verbose=False):
        """Function to simply load drift file"""
        if _verbose:
            print(f"-- loading drift for fov: {self.fov_name}")
        if _drift_basename is None:
            _postfix = _drift_postfix
            if _sequential_mode:
                _postfix = '_sequential' + _postfix
            _drift_basename = self.fov_name.replace('.dax', _postfix)
        # get filename
        _drift_filename = os.path.join(self.drift_folder, _drift_basename)
        if os.path.isfile(_drift_filename):
            if _verbose:
                print(f"--- from file: {_drift_filename}")
            self.drift = pickle.load(open(_drift_filename, 'rb'))
            return True
        else:
            if _verbose:
                print(f"--- file {_drift_filename} not exist, exit.")
            return False

    ## generate reference images and reference centers
    def _prepare_dirft_references(self,
        _ref_filename = None,
        _drift_crops = None,
        _drift_size=None,
        _single_im_size = None,
        _all_channels = None,
        _num_buffer_frames = None,
        _num_empty_frames = None,
        _bead_channel=None,
        _seeding_th=150,
        _dynamic_seeding=False,
        _min_num_seeds=50,
        _seeding_percentile=95,
        _match_distance_th=3, 
        _overwrite=False,
        _verbose=True):
        """Function to calculate ref_centers in given drift_crops"""
        from ..correction_tools.alignment import generate_drift_crops
        from ..io_tools.load import split_im_by_channels
        from ..visual_tools import DaxReader, get_STD_centers
        from ..get_img_info import get_num_frame, split_channels
        # set default params
        if _drift_size is None:
            _drift_size = self.shared_parameters['drift_size']
        if _single_im_size is None:
            _single_im_size = self.shared_parameters['single_im_size']
        if _all_channels is None:
            _all_channels = self.channels
        if _num_buffer_frames is None:
            _num_buffer_frames = self.shared_parameters['num_buffer_frames']
        if _num_empty_frames is None:
            _num_empty_frames = self.shared_parameters['num_empty_frames']
        if _bead_channel is None:
            _bead_channel=self.channels[self.bead_channel_index]

        # check ref_filename
        if _ref_filename is None:
            _ref_filename = self.ref_filename
        if _verbose:
            print(f"++ acquire reference images and centers from file:{_ref_filename}")
        if '.dax' not in _ref_filename:
            raise TypeError(f"Wrong ref_filename type, should be .dax file!")
        # Check drift file
        if not os.path.isfile(self.drift_filename):
            from .batch_functions import create_drift_file
            create_drift_file(self.drift_filename, self.ref_filename,
                              overwrite=_overwrite, verbose=_verbose)
        # check drift_crops
        if not hasattr(self, 'drift_crops') or _overwrite:
            if _drift_crops is not None:
                pass
            else:
                _drift_crops = generate_drift_crops(drift_size=_drift_size,
                                                     single_im_size=_single_im_size,)
            if _verbose:
                print(f"+++ updating drift crop for this fov to: {_drift_crops}")
            setattr(self, 'drift_crops', _drift_crops)
        else:
            _drift_crops = getattr(self, 'drift_crops')

        # first check if attribute already exists:
        _load_ref_flag = _overwrite
        if hasattr(self, 'ref_ims') and hasattr(self, 'ref_cts') and not _overwrite:
            #
            _ref_cts = getattr(self, 'ref_cts')
            _ref_ims = getattr(self, 'ref_ims')
            for _im, _ct, _crop in zip(_ref_ims, _ref_cts, _drift_crops):
                if _im.shape != tuple(_crop[:,1]-_crop[:,0]):
                    if _verbose:
                        print(f"+++ image shape:{np.shape(_im)} doesn't match crop:{_crop}, reload ref_ims!")
                    _load_ref_flag = True 
                    break
                if len(_ct) < self.shared_parameters['min_num_seeds']:
                    if _verbose:
                        print(f"+++ not enough reference centers, {len(_ct)} given, {self.shared_parameters['min_num_seeds']} expected, reload ref_ims!")
                    _load_ref_flag = True 
                    break
        else:
            _load_ref_flag = True 
        # if reload im, do loading here:
        if _load_ref_flag:
            # load image
            if _verbose:
                print(f"+++ loading image from .dax file")
            _reader = DaxReader(_ref_filename, verbose=_verbose)
            _full_im = _reader.loadAll()
            _reader.close()
            _full_im_shape, _num_color = get_num_frame(_ref_filename,
                                                    frame_per_color=_single_im_size[0],
                                                    buffer_frame=_num_buffer_frames)
            _bead_im = split_im_by_channels(_full_im, [_bead_channel], _all_channels[:_num_color], 
                                        single_im_size=_single_im_size, 
                                        num_buffer_frames=_num_buffer_frames,
                                        num_empty_frames=_num_empty_frames)[0]
            # crop images
            if _verbose:
                print(f"+++ cropping {len(_drift_crops)} images")
            _ref_ims = [_bead_im[_c[0,0]:_c[0,1],
                                _c[1,0]:_c[1,1],
                                _c[2,0]:_c[2,1]] for _c in _drift_crops]
            
            # fit centers
            if _verbose:
                print(f"+++ get ref_centers by gaussian fitting")
            # collect ref_cts
            from ..spot_tools.fitting import select_sparse_centers, get_centers
            _ref_cts = []
            for _im in _ref_ims:
                _cand_cts = get_centers(_im, th_seed=_seeding_th,
                                        th_seed_per=_seeding_percentile, 
                                        use_percentile=_dynamic_seeding,
                                        max_num_seeds=self.shared_parameters['max_num_seeds'],
                                        min_num_seeds=self.shared_parameters['min_num_seeds'],
                                        verbose=_verbose,
                                        )
                _ref_cts.append(select_sparse_centers(_cand_cts, 
                                    distance_th=_match_distance_th,
                                    verbose=_verbose))
            # append
            self.ref_ims = _ref_ims
            self.ref_cts = _ref_cts
        # direct load if not necesary
        else:
            if _verbose:
                print(f"+++ direct load from attributes.")
            _ref_ims = getattr(self, 'ref_ims')
            _ref_cts = getattr(self, 'ref_cts')
        # return
        return _ref_ims, _ref_cts
    
    ## check drift info
    def _check_drift(self, _load_drift_kwargs={}, 
                     _load_info_kwargs={}, _verbose=False):
        """Check whether drift exists and whether all keys required for images exists"""
        ## try to load drift if not exist
        if not hasattr(self, 'drift') or len(self.drift) == 0:
            if _verbose:
                print("-- No drift attribute detected, try to load from file")
            _flag = self._load_drift_file(_verbose=_verbose, **_load_drift_kwargs)
            if not _flag:
                return False
        ## drift exist, do check
        # load color_dic as a reference
        if not hasattr(self, 'color_dic'):
            self._load_color_info(**_load_info_kwargs)
            # check every folder in color_dic whether exists in drift
            for _hyb_fd, _info in self.color_dic.items():
                _drift_query = os.path.join(_hyb_fd, self.fov_name)
                if _drift_query not in self.drift:
                    if _verbose:
                        print(f"-- drift info for {_drift_query} was not found")
                    return False
        # if everything is fine return True
        return True

    def _bead_drift(self, _sequential_mode=True, _load_annotated_only=True, 
                    _size=500, _ref_id=0, _drift_postfix='_current_cor.pkl', 
                    _num_threads=12, _coord_sel=None, _force=False, _dynamic=True, 
                    _stringent=True, _verbose=True):
        # num-threads
        if hasattr(self, 'num_threads'):
            _num_threads = min(_num_threads, self.num_threads)
        # if drift meets requirements:
        if self._check_drift(_verbose=False) and not _force:
            if _verbose:
                print(f"- drift already exists for fov:{self.fov_name}, skip")
            return getattr(self,'drift')
        else:
            # load color usage if not given
            if not hasattr(self, 'channels'):
                self._load_color_info()
            # check whether load annotated only
            if _load_annotated_only:
                _folders = self.annotated_folders
            else:
                _folders = self.folders
            # load existing drift file 
            _drift_filename = os.path.join(self.drift_folder, self.fov_name.replace('.dax', _drift_postfix))
            _sequential_drift_filename = os.path.join(self.drift_folder, self.fov_name.replace('.dax', '_sequential'+_drift_postfix))
            # check drift filename and sequential file name:
            # whether with sequential mode determines the order to load files
            if _sequential_mode:
                _check_dft_files = [_sequential_drift_filename, _drift_filename]
            else:
                _check_dft_files = [_drift_filename, _sequential_drift_filename]
            for _dft_filename in _check_dft_files:
                # check one drift file
                if os.path.isfile(_dft_filename):
                    _drift = pickle.load(open(_dft_filename, 'rb'))
                    _exist = [os.path.join(os.path.basename(_fd),self.fov_name) for _fd in _folders \
                            if os.path.join(os.path.basename(_fd),self.fov_name) in _drift]
                    if len(_exist) == len(_folders):
                        if _verbose:
                            print(f"- directly load drift from file:{_dft_filename}")
                        self.drift = _drift
                        return self.drift
            # if non-of existing files fulfills requirements, initialize
            if _verbose:
                print("- start a new drift correction!")

            ## proceed to amend drift correction
            from corrections import Calculate_Bead_Drift
            _drift, _failed_count = Calculate_Bead_Drift(_folders, [self.fov_name], 0, 
                                        num_threads=_num_threads,sequential_mode=_sequential_mode, 
                                        ref_id=_ref_id, drift_size=_size, coord_sel=_coord_sel,
                                        single_im_size=self.shared_parameters['single_im_size'], 
                                        all_channels=self.channels,
                                        num_buffer_frames=self.shared_parameters['num_buffer_frames'], 
                                        num_empty_frames=self.shared_parameters['num_empty_frames'], 
                                        illumination_corr=self.shared_parameters['corr_illumination'],
                                        save_postfix=_drift_postfix,
                                        save_folder=self.drift_folder, stringent=_stringent,
                                        overwrite=_force, verbose=_verbose)
            if _verbose:
                print(f"- drift correction for {len(_drift)} frames has been generated.")
            _exist = [os.path.join(os.path.basename(_fd),self.fov_name) for _fd in _folders \
                if os.path.join(os.path.basename(_fd),self.fov_name) in _drift]
            # print if some are failed
            if _failed_count > 0:
                print(f"-- failed number: {_failed_count}"
                )
            if len(_exist) == len(_folders):
                self.drift = _drift
                return self.drift
            else:
                raise ValueError("length of _drift doesn't match _folders!")


    def _process_image_to_spots(self, _data_type, _sel_folders=[], _sel_ids=[], 
                                _load_common_correction_profiles=True, 
                                _load_common_reference=True, 
                                _load_with_multiple=True, 
                                _use_exist_images=False, 
                                _warp_images=True, 
                                _save_images=True, 
                                _save_drift=True,
                                _save_fitted_spots=True, 
                                _correction_args={},
                                _drift_args={}, 
                                _fitting_args={},
                                _overwrite_drift=False, 
                                _overwrite_image=False, 
                                _overwrite_spot=False,
                                _verbose=True):
        ## check inputs
        if _data_type not in self.shared_parameters['allowed_data_types']:
            raise ValueError(f"Wrong input for _data_type:{_data_type}, should be within {self.shared_parameters['allowed_data_types'].keys()}")
        # extract datatype marker
        _dtype_mk = self.shared_parameters['allowed_data_types'][_data_type]
        # get color_dic data-type
        from .batch_functions import _color_dic_stat
        _type_dic = _color_dic_stat(self.color_dic, 
                                    self.channels, 
                                    self.shared_parameters['allowed_data_types'])
        ## select folders
        _input_fds = []
        if _sel_folders is not None and len(_sel_folders) > 0:
            # load folders
            for _fd in _sel_folders:
                if _fd in self.annotated_folders:
                    _input_fds.append(_fd)
            if _verbose:
                print(f"-- {len(_sel_folders)} folders given, {len(_input_fds)} folders selected.")
        else:
            _sel_folders = self.annotated_folders
            if _verbose:
                print(f"-- No folder selected, allow processing all {len(self.annotated_folders)} folders")
        # check selected ids
        if _sel_ids is not None and len(_sel_ids) > 0:
            _sel_ids = [int(_id) for _id in _sel_ids] # convert to list of ints
            for _id in _sel_ids:
                if _id not in _type_dic[_data_type]['ids']:
                    print(f"id: {_id} not allowed in color_dic!")
            _sel_ids = [_id for _id in _sel_ids if _id in _type_dic[_data_type]['ids']]
        else:
            # if not given, process all ids for this data_type
            _sel_ids = [int(_id) for _id in _type_dic[_data_type]['ids']]
        
        ## load correction profiles if necessary:
        if _load_common_correction_profiles and not hasattr(self, 'correction_profiles'):
            self._load_correction_profiles()

        ## load shared drift references
        if _load_common_reference and (not hasattr(self, 'ref_ims') or not hasattr(self, 'ref_cts')) :
            self._prepare_dirft_references(_verbose=_verbose)

        ## multi-processing for correct_splice_images
        # prepare common params
        _correction_args.update({
            'single_im_size': self.shared_parameters['single_im_size'],
            'all_channels':self.channels,
            'num_buffer_frames':self.shared_parameters['num_buffer_frames'],
            'num_empty_frames':self.shared_parameters['num_empty_frames'],
            'bead_channel': self.channels[self.bead_channel_index],
            'correction_folder':self.correction_folder,
            'hot_pixel_corr':self.shared_parameters['corr_hot_pixel'], 
            'z_shift_corr':self.shared_parameters['corr_Z_shift'], 
            'bleed_corr':self.shared_parameters['corr_bleed'],
            'bleed_profile':self.correction_profiles['bleed'],
            'illumination_corr':self.shared_parameters['corr_illumination'],
            'illumination_profile':self.correction_profiles['illumination'],
            'chromatic_corr':self.shared_parameters['corr_chromatic'],
            'normalization':self.shared_parameters['normalization'],
        })
        # specifically, chromatic profile loaded is based on whether warp image
        if _warp_images:
            _correction_args.update({
                'chromatic_profile':self.correction_profiles['chromatic'],})
        else:
            _correction_args.update({
                'chromatic_profile':self.correction_profiles['chromatic_constants'],})
        _drift_args.update({
            'drift_size':self.shared_parameters['drift_size'],
            'use_fft': self.shared_parameters['drift_use_fft'],
            'drift_crops':self.drift_crops,
            'ref_beads':self.ref_cts,
            'ref_ims':self.ref_ims,
            'max_num_seeds' : self.shared_parameters['max_num_seeds'],
            'min_num_seeds' : self.shared_parameters['min_num_seeds'],
            'good_drift_th': self.shared_parameters['good_drift_th'],
        })
        _fitting_args.update({
            'max_num_seeds' : self.shared_parameters['max_num_seeds'],
            'th_seed': self.shared_parameters['spot_seeding_th'],
            'init_sigma': self.shared_parameters['sigma_zxy'],
            'min_dynamic_seeds': self.shared_parameters['min_num_seeds'],
            'remove_hot_pixel': self.shared_parameters['corr_hot_pixel'],
            'normalize_local': self.shared_parameters['normalize_intensity_local'],
        })
        
        # initiate locks
        _manager = mp.Manager()
        # recursive lock so each thread can acquire multiple times
        # lock to make sure only one thread is reading from hard drive.
        if _load_with_multiple:
            _load_file_lock = None
        else:
            _load_file_lock = _manager.RLock()
        # lock to write to save_file
        _image_file_lock = _manager.RLock() 
        # lock to write to drift file
        _drift_file_lock = _manager.RLock() 

        ## initialize reported ids and spots
        _final_ids = []
        _final_spots = []
        # prepare kwargs to be processed.
        _processing_arg_list = []
        _processing_id_list = []
        
        # loop through annotated folders
        for _fd in _sel_folders:
            _dax_filename = os.path.join(_fd, self.fov_name)
            # get selected channels
            _info = self.color_dic[os.path.basename(_fd)]
            _sel_channels = []
            _reg_ids = []
            # loop through color_dic to collect selected channels and ids
            for _mk, _ch in zip(_info, self.channels):
                if _dtype_mk in _mk:
                    _id = int(_mk.split(_dtype_mk)[1])
                    if _id in _sel_ids:
                        # append sel_channels and reg_ids for now
                        _sel_channels.append(_ch)
                        _reg_ids.append(_id)
            # check existence of these candidate selected ids
            if len(_sel_channels) > 0:
                # Case 1: if trying to use existing images 
                if _use_exist_images:
                    _exist_spots, _exist_drifts, _exist_flags, _exist_ims =  self._check_exist_data(_data_type, _reg_ids, _check_im=True, _verbose=_verbose)
                    _sel_channels = [_ch for _ch, _es, _ei in zip(_sel_channels, _exist_spots, _exist_ims)
                                    if not _es or not _ei] # if spot or im not exist, process sel_channel
                    _reg_ids = [_id for _id, _es, _ei in zip(_reg_ids, _exist_spots, _exist_ims)
                                if not _es or not _ei] # if spot or im not exist, process reg_id
                # Case 2: image saving is not required
                else:
                    _exist_spots, _exist_drifts, _exist_flags =  self._check_exist_data(_data_type, _reg_ids, _check_im=False, _verbose=_verbose)
                    _sel_channels = [_ch for _ch, _es in zip(_sel_channels, _exist_spots)
                                    if not _es] # if spot not exist, process sel_channel
                    _reg_ids = [_id for _id, _es in zip(_reg_ids, _exist_spots)
                                if not _es] # if spot not exist, process reg_id

            # append if any channels selected
            if len(_sel_channels) > 0:
                _args = (_dax_filename, _sel_channels, self.ref_filename,
                        _load_file_lock,
                        _correction_args, 
                        _save_images, self.save_filename,
                        _data_type, _reg_ids, 
                        _warp_images,
                        _image_file_lock, 
                        _overwrite_image, 
                        _drift_args, 
                        _save_drift, 
                        self.drift_filename,
                        _drift_file_lock, 
                        _overwrite_drift,
                        _fitting_args, _save_fitted_spots,
                        _image_file_lock, _overwrite_spot, 
                        False, 
                        _verbose,)
                _processing_arg_list.append(_args)
                _processing_id_list.append(_reg_ids)

        # multi-processing
        ## multi-processing for translating segmentation
        if len(_processing_arg_list) > 0:
            from .batch_functions import batch_process_image_to_spots, killchild
            with mp.Pool(self.num_threads,) as _processing_pool:
                if _verbose:
                    print(f"+ Start multi-processing of pre-processing for {len(_processing_arg_list)} images!")
                    print(f"++ processed {_data_type} ids: {np.sort(np.concatenate(_processing_id_list))}", end=' ')
                    _start_time = time.time()
                # Multi-proessing!
                _spot_results = _processing_pool.starmap(
                    batch_process_image_to_spots,
                    _processing_arg_list, 
                    chunksize=1)
                # close multiprocessing
                _processing_pool.close()
                _processing_pool.join()
                _processing_pool.terminate()
            # clear
            killchild()        
            if _verbose:
                print(f"in {time.time()-_start_time:.2f}s.")
        else:
            return [], []
        # unravel and append process

        for _ids, _spot_list in zip(_processing_id_list, _spot_results):
            _final_ids += list(_ids)
            _final_spots += list(_spot_list)
        # sort
        _ps_ids = [_id for _id,_spots in sorted(zip(_final_ids, _final_spots))]
        _ps_spots = [_spots for _id,_spots in sorted(zip(_final_ids, _final_spots))]
  
        return _ps_ids, _ps_spots

        

    def _save_to_file(self, _type, _save_attr_list=[], 
                      _overwrite=False, _verbose=True):
        """Function to save attributes into standard HDF5 save_file of fov class.
        """
        ## check inputs:
        _type = str(_type).lower()
        # only allows saving the following types:
        _allowed_save_types = ['fov_info', 'segmentation', 'correction'] \
                              + list(self.shared_parameters['allowed_data_types'].keys())
        if _type not in _allowed_save_types:
            raise ValueError(f"Wrong input for _type:{_type}, should be within:{_allowed_save_types}.")
        # save file should exist
        if not os.path.isfile(self.save_filename):
            print(f"* create savefile: {self.save_filename}")
        
        ## start saving here:
        if _verbose:
            print(f"-- saving {_type} to file: {self.save_filename}")
            _save_start = time.time()
        
        with h5py.File(self.save_filename, "a", libver='latest') as _f:

            ## segmentation
            if _type == 'segmentation':
                # create segmentation group if not exist 
                if 'segmentation' not in _f.keys():
                    _grp = _f.create_group('segmentation') # create segmentation group
                else:
                    _grp = _f['segmentation']
                # directly create segmentation label dataset
                if 'segmentation_label' not in _grp:
                    _seg = _grp.create_dataset('segmentation_label', 
                                            self.shared_parameters['single_im_size'][-self.segmentation_dim:], 
                                            dtype='i8')
                if hasattr(self, 'segmentation_label'):
                    if len(_save_attr_list) > 0 and _save_attr_list is not None:
                        if 'segmentation_label' in _save_attr_list:
                            _grp['segmentation_label'] = getattr(self, 'segmentation_label')
                    else:
                        _grp['segmentation_label'] = getattr(self, 'segmentation_label')
                # create other segmentation related datasets
                for _attr_name in dir(self):
                    if _attr_name[0] != '_' and 'segmentation' in _attr_name and _attr_name not in _grp.keys():
                        if len(_save_attr_list) > 0 and _save_attr_list is not None:
                            # if save_attr_list is given validly and this attr not in it, skip.
                            if _attr_name not in _save_attr_list:
                                continue 
                        _grp[_attr_name] = getattr(self, _attr_name)

            elif _type == 'correction':
                pass 

            ## save basic attributes as info
            elif _type == 'fov_info':
                # initialize attributes as _info_attrs
                _info_attrs = []
                for _attr_name in dir(self):
                    # exclude all default attrs and functions
                    if _attr_name[0] != '_' and getattr(self, _attr_name) is not None:
                        # check within save_attr_list:
                        if len(_save_attr_list) > 0 and _save_attr_list is not None:
                            # if save_attr_list is given validly and this attr not in it, skip.
                            if _attr_name not in _save_attr_list:
                                continue 
                        # set default to be save
                        _info_attr_flag = True
                        # exclude attributes belongs to other categories
                        for _save_type in _allowed_save_types:
                            if _save_type in _attr_name:
                                _info_attr_flag = False 
                                break
                        # if its image dict, exclude
                        if 'im_dict' in _attr_name or 'channel_dict' in _attr_name:
                            _info_attr_flag = False

                        # save here:
                        if _info_attr_flag:
                            # extract the attribute
                            _attr = getattr(self, _attr_name)
                            # convert dict if necessary
                            if isinstance(_attr, dict):
                                _attr = str(_attr)
                            # save
                            if _attr_name not in _f.attrs or _overwrite:
                                _f.attrs[_attr_name] = _attr
                                _info_attrs.append(_attr_name)
                if _verbose:
                    print(f"--- base attributes updated:{_info_attrs} in {time.time()-_save_start:.3f}s.")

            ## images and spots for a specific data type 
            elif _type in self.shared_parameters['allowed_data_types']:
                from .batch_functions import _color_dic_stat
                _type_dic = _color_dic_stat(self.color_dic, self.channels, self.shared_parameters['allowed_data_types'])
                if _type not in _type_dic:
                    print(f"--- given save type:{_type} doesn't exist in this dataset, skip.") 
                else:
                    # extract info dict for this data_type
                    _dict = _type_dic[_type]

                    # create data_type group if not exist 
                    if _type not in _f.keys():
                        _grp = _f.create_group(_type) 
                    else:
                        _grp = _f[_type]
                    # record updated data_type related attrs
                    _data_attrs = []
                    ## save images, ids, channels, save_flags
                    # calculate image shape and chunk shape
                    _im_shape = np.concatenate([np.array([len(_dict['ids'])]), 
                                                self.shared_parameters['single_im_size']])
                    _chunk_shape = np.concatenate([np.array([1]), 
                                                self.shared_parameters['single_im_size']])                              
                    # change size
                    _change_size_flag = []
                    # if missing any of these features, create new ones
                    # ids
                    if 'ids' not in _grp:
                        _ids = _grp.create_dataset('ids', (len(_dict['ids']),), dtype='i', data=_dict['ids'])
                        _ids = np.array(_dict['ids'], dtype=np.int) # save ids
                        _data_attrs.append('ids')
                    elif len(_dict['ids']) != len(_grp['ids']):
                        _change_size_flag.append('id')
                        _old_size=len(_grp['ids'])

                    # channels
                    if 'channels' not in _grp:
                        _channels = [_ch.encode('utf8') for _ch in _dict['channels']]
                        _chs = _grp.create_dataset('channels', (len(_dict['channels']),), dtype='S3', data=_channels)
                        _chs = np.array(_dict['channels'], dtype=str) # save ids
                        _data_attrs.append('channels')
                    elif len(_dict['channels']) != len(_grp['channels']):
                        _change_size_flag.append('channels')
                        _old_size=len(_grp['channels'])

                    # images
                    if 'ims' not in _grp:
                        _ims = _grp.create_dataset('ims', tuple(_im_shape), 
                                                   dtype='u2',  # uint16
                                                   chunks=tuple(_chunk_shape))
                        _data_attrs.append('ims')
                    elif len(_im_shape) != len(_grp['ims'].shape) or (_im_shape != (_grp['ims']).shape).any():
                        _change_size_flag.append('ims')
                        _old_size=len(_grp['ims'])

                    # spots
                    if 'spots' not in _grp:
                        _spots = _grp.create_dataset('spots', 
                                    (_im_shape[0], self.shared_parameters['max_num_seeds'], 11), 
                                    dtype='f')
                        _data_attrs.append('spots')
                    elif _im_shape[0] != len(_grp['spots']):
                        _change_size_flag.append('spots')
                        _old_size=len(_grp['spots'])

                    # drift
                    if 'drifts' not in _grp:
                        _drift = _grp.create_dataset('drifts', (_im_shape[0], 3), dtype='f')
                        _data_attrs.append('drifts')
                    elif _im_shape[0] != len(_grp['drifts']):
                        _change_size_flag.append('drifts')
                        _old_size=len(_grp['drifts'])

                    # flags for whether it's been written
                    if 'flags' not in _grp:
                        _filenames = _grp.create_dataset('flags', (_im_shape[0], ), dtype='u1')
                        _data_attrs.append('flags')
                    elif _im_shape[0] != len(_grp['flags']):
                        _change_size_flag.append('flags')
                        _old_size=len(_grp['flags'])

                    # Create other features
                    for _attr_name in dir(self):
                        if _attr_name[0] != '_' and _type in _attr_name:
                            _attr_feature = _attr_name.split(_type)[1][1:]

                            if _attr_feature not in _grp.keys():
                                _grp[_attr_name] = getattr(self, _attr_name)
                                _data_attrs.append(_attr_name)
                    
                    # if change size, update these features:
                    if len(_change_size_flag) > 0:
                        print(f"* data size of {_type} is changing from {_old_size} to {len(_dict['ids'])} because of {_change_size_flag}")
                        ###UNDER CONSTRUCTION################
                        pass
                    # summarize
                    if _verbose:
                        print(f"--- {_type} attributes updated:{_data_attrs} in {time.time()-_save_start:.3f}s.")
                        _save_mid = time.time()

        ## save ims, spots, drifts, flags
        if _type in self.im_dict:
            _image_info = self.im_dict[_type]
            if 'ids' not in _image_info:
                print(f"--- ids for type:{_type} not given, skip.")
            else:
                _ids = _image_info['ids']
                # save images
                if 'ims' in _image_info and len(_image_info['ims']) == len(_ids):
                    from .batch_functions import save_image_to_fov_file
                    if 'drifts' in _image_info:
                        _drifts = _image_info['drifts']
                    else:
                        _drifts = None 
                    _ims_flag = save_image_to_fov_file(self.save_filename, 
                                                        _image_info['ims'],
                                                        _type, _ids, 
                                                        warp_image=self.shared_parameters['_warp_images'], 
                                                        drift=_drifts, 
                                                        overwrite=_overwrite,
                                                        verbose=_verbose)
                # save spots
                if 'spots' in _image_info and len(_image_info['spots']) == len(_ids):
                    from .batch_functions import save_spots_to_fov_file
                    _spots_flag = save_spots_to_fov_file(self.save_filename, 
                                                        _image_info['spots'],
                                                        _type, _ids, 
                                                        overwrite=_overwrite,
                                                        verbose=_verbose)
            if _verbose:
                print(f"--- save images and spots for {_type} in {time.time()-_save_mid:.3f}s.")

            

    def _check_exist_data(self, _data_type, _region_ids=None,
                          _check_im=False,
                          empty_value=0, _verbose=False):
        """function to check within a specific data_type, 
            a specific region_id, does image and spot exists
        Inputs:
            
        Outputs:
            _exist_im, 
            _exist_spots,
            _exist_drift,
            _exist_flag
        """
        if _data_type not in self.shared_parameters['allowed_data_types']:
            raise ValueError(f"Wrong input data_type: {_data_type}, should be among:{self.shared_parameters['allowed_data_types']}")
        if _region_ids is None:
            # get color_dic data-type
            from .batch_functions import _color_dic_stat
            _type_dic = _color_dic_stat(self.color_dic, 
                                        self.channels, 
                                        self.shared_parameters['allowed_data_types'])
            _region_ids = _type_dic[_data_type]
        else:  
            if isinstance(_region_ids, int) or isinstance(_region_ids, np.int):
                _region_ids = [_region_ids]   
            elif not isinstance(_region_ids, list) and not isinstance(_region_ids, np.ndarray):
                raise TypeError(f"Wrong input type for region_ids:{_region_ids}")
            _region_ids = np.array([int(_i) for _i in _region_ids],dtype=np.int)
        # print
        if _verbose:
            _check_time = time.time()
            print(f"-- checking {_data_type}, region:{_region_ids}", end=' ')
            if _check_im:
                print("including images", end=' ')
        with h5py.File(self.save_filename, "a", libver='latest') as _f:
            if _data_type not in _f.keys():
                raise ValueError(f"input data type doesn't exist in this save_file:{self.save_filename}")
            _grp = _f[_data_type]
            _ids = list(_grp['ids'][:])
            for _region_id in _region_ids:
                if _region_id not in _ids:
                    raise ValueError(f"region_id:{_region_id} not in {_data_type} ids.")

            # initialize
            _exist_im, _exist_spots, _exist_drift, _exist_flag = [],[],[],[]

            for _region_id in _region_ids:
                # get indices
                _ind = _ids.index(_region_id)

                if _check_im:
                    _exist_im.append((np.array(_grp['ims'][_ind]) != empty_value).any())

                _exist_spots.append((np.array(_grp['spots'][_ind]) != empty_value).any())
                _exist_drift.append((np.array(_grp['drifts'][_ind]) != empty_value).any())
                _exist_flag.append( np.array(_grp['flags'][_ind]))

        # convert to array
        _exist_spots = np.array(_exist_spots) 
        _exist_drift = np.array(_exist_drift) 
        _exist_flag = np.array(_exist_flag) 
        if _verbose:
            print(f"in {time.time()-_check_time:.3f}s.")
        if _check_im:
            _exist_im = np.array(_exist_im) 
            return _exist_spots, _exist_drift, _exist_flag, _exist_im
        else:
            return _exist_spots, _exist_drift, _exist_flag


    def _load_from_file(self, _type):
        pass

    def _delete_save_file(self, _type):
        pass

    def _convert_to_cell_list(self):
        pass

    def _spot_finding(self):
        pass

