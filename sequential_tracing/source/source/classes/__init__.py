import sys,glob,os,time,copy
import numpy as np
import pickle as pickle
import multiprocessing as mp

from .. import get_img_info, corrections, visual_tools, spot_tools, domain_tools
from .. import _correction_folder, _corr_channels, _temp_folder,_distance_zxy,\
    _sigma_zxy,_image_size, _allowed_colors, _num_buffer_frames, _num_empty_frames, _image_dtype
from ..External import Fitting_v3
from scipy import ndimage, stats
from scipy.spatial.distance import pdist,cdist,squareform
from skimage import morphology
from skimage.segmentation import random_walker
from functools import partial
import matplotlib
import matplotlib.pyplot as plt

import h5py
import ast

_allowed_kwds = {'combo': 'c', 
                'decoded':'d',
                'unique': 'u', 
                'merfish': 'm', 
                'rna-unique':'r', 
                'rna': 'r', # long term used label, because "-" is creating issue in python
                'gene':'g'}
_max_num_seeds = 300 
_min_num_seeds = 50 
_spot_seeding_th = 300
from . import batch_functions
from . import field_of_view

# initialize pool
init_dic = {}
def _init_unique_pool(_ic_profile_dic, _cac_profile_dic, _ic_shape, _cac_shape):
    """initialize pool, function used to put data into shared memory"""
    print(f"- Initialize core with illumination correction profiles for {list(_ic_profile_dic.keys())}")
    init_dic['illumination'] = _ic_profile_dic
    print(f"- Initialize core with chromatic correction profiles for {list(_cac_profile_dic.keys())}")
    init_dic['chromatic'] = _cac_profile_dic
    init_dic['ic_shape'] = _ic_shape
    init_dic['cac_shape'] = _cac_shape

def _fit_single_image(_im, _id, _chrom_coords, _seeding_args, _fitting_args, _check_fitting=True, 
                      _normalization=True, _verbose=False):
    if _verbose:
        print(f"+++ fitting for region:{_id}")
    _spots_for_chrom = []
    if _normalization:
        _norm_cst = np.nanmedian(_im)
    for _chrom_coord in _chrom_coords:
        if _im is None:
            _spots_for_chrom.append(np.array([]))
        else:
            # seeding
            _seeds = visual_tools.get_seed_in_distance(_im, _chrom_coord, *_seeding_args)
            if len(_seeds) == 0:
                # no seed found, return empty array
                _spots_for_chrom.append(np.array([]))
                continue
            # fit
            _fitter = Fitting_v3.iter_fit_seed_points(
                _im, _seeds.T, *_fitting_args)
            _fitter.firstfit()
            # if check-fitting
            if _check_fitting:
                _fitter.repeatfit()
            
            #_fits = visual_tools.fit_multi_gaussian(_im, _seeds, *_fitting_args)
            _spots = np.array(_fitter.ps)
            if _normalization:
                _spots[:,0] = _spots[:,0] / _norm_cst
            _spots_for_chrom.append(_spots)    

    return _spots_for_chrom

# function to allow multi-processing pick spots
def _pick_spot_in_batch(_cell, _data_type='unique', _pick_type='EM', 
                        _use_chrom_coords=True,
                        _sel_ids=None, _num_iters=10, _terminate_th=0.003, 
                        _intensity_th=1, _hard_intensity_th=True, _spot_num_th=100,
                        _ref_spot_list=None, _ref_spot_ids=None, _ref_pick_type='EM',
                        _ignore_ids=False, _ref_dist_metric='median', _score_metric='linear',
                        _local_size=5, _w_ctdist=2, _w_lcdist=1, _w_int=1, _w_nbdist=2,
                        _save_inter_plot=False, _save_to_info=True, _save_plot=True, 
                        _check_spots=True, _check_th=-3.5, _check_percentile=10.,
                        _hard_dist_th=6000, _distance_limits=[0, np.inf], 
                        _ignore_nan=True, _nan_mask=0., _inf_mask=-1000., 
                        _chrom_share_spots=False, _plot_limits=[0, 1500], _cmap='seismic_r', _fig_dpi=300, _fig_size=4,
                        _overwrite=False, _verbose=True):
    """_cell: Cell_Data class"""
    if _verbose:
        print(f"-- {_pick_type} pick spots for fov:{_cell.fov_id}, cell:{_cell.cell_id}")

    # notice: always load in attributes, never return indices in batch format
    _picked_spots = _cell._pick_spots(_data_type=_data_type, 
                                      _pick_type=_pick_type, 
                                      _use_chrom_coords=_use_chrom_coords, _sel_ids=_sel_ids, 
                                      _num_iters=_num_iters, _terminate_th=_terminate_th, 
                                      _intensity_th=_intensity_th, _hard_intensity_th=_hard_intensity_th,
                                      _spot_num_th=_spot_num_th, _ref_spot_list=_ref_spot_list, 
                                      _ref_spot_ids=_ref_spot_ids, _ref_pick_type=_ref_pick_type,
                                      _ignore_ids=_ignore_ids, 
                                      _ref_dist_metric=_ref_dist_metric, _score_metric=_score_metric,
                                      _local_size=_local_size, _w_ctdist=_w_ctdist, _w_lcdist=_w_lcdist,  
                                      _w_int=_w_int, _w_nbdist=_w_nbdist, 
                                      _distance_limits=_distance_limits, _ignore_nan=_ignore_nan,
                                      _nan_mask=_nan_mask, _inf_mask=_inf_mask, _chrom_share_spots=_chrom_share_spots, 
                                      _check_spots=_check_spots, _check_th=_check_th, 
                                      _check_percentile=_check_percentile, _hard_dist_th=_hard_dist_th,
                                      _save_inter_plot=_save_inter_plot, _save_to_attr=True, _save_to_info=_save_to_info,
                                      _return_indices=False, _overwrite=_overwrite, 
                                      _verbose=_verbose)

    _distmaps = _cell._generate_distance_map(_data_type=_data_type, 
                                             _pick_type=_pick_type, 
                                             _sel_ids=_sel_ids, _save_info=_save_to_info, _save_plot=_save_plot,
                                             _limits=_plot_limits, _cmap=_cmap, 
                                             _fig_dpi=_fig_dpi, _fig_size=_fig_size, 
                                             _overwrite=_overwrite, _verbose=_verbose)
    return _cell


def _load_cell_in_batch(_cell, _data_type='all', _save_folder=None,
                        _decoded_flag=None, _distmap_data='unique', _distmap_pick='EM',
                        _load_attrs=[], _exclude_attrs=[], _overwrite=False, _verbose=True):
    """Function to allow batch loading"""                   
    _cell._load_from_file(_data_type=_data_type, _save_folder=_save_folder, 
                          _decoded_flag=_decoded_flag, 
                          _distmap_data=_distmap_data, _distmap_pick=_distmap_pick,  
                          _load_attrs=_load_attrs, _exclude_attrs=_exclude_attrs,
                          _overwrite=_overwrite, _verbose=_verbose)
    return _cell

def _save_cell_in_batch(_cell, _data_type='cell_info', _save_dic={}, _save_folder=None, 
                      _unsaved_attrs=None, _clear_old_attrs=False, 
                      _overwrite=False, _verbose=True):
    """Function to allow batch saving"""
    _cell._save_to_file(_data_type=_data_type, _save_dic=_save_dic, _save_folder=_save_folder, 
                        _unsaved_attrs=_unsaved_attrs, _clear_old_attrs=_clear_old_attrs, 
                        _overwrite=_overwrite, _verbose=_verbose)

# batch merge cells
def _merge_RNA_to_DNA_in_batch(_cell, _source_cell_data, _merge_type='cell_info',
                               _attr_feature='rna-',
                               _load_in_ram=True, _save_to_file=True, 
                               _overwrite=False, _verbose=True):
    """Function to allow batch cell_data merging"""
    _cell._merge_RNA_to_DNA(_source_cell_data=_source_cell_data, 
                            _merge_type=_merge_type,
                            _attr_feature=_attr_feature,
                            _load_in_ram=_load_in_ram, _save_to_file=_save_to_file, 
                            _overwrite=_overwrite, _verbose=_verbose)
    return _cell

class Cell_List():
    """
    Class Cell_List:
    this is a typical data structure of cells within one chromosome with images in multiple independent color-channels and decoding-groups.

    """
    # initialize
    def __init__(self, parameters, _chosen_fovs=[], _exclude_fovs=[], 
                 _load_all_attr=False, _load_reference_info=True,
                 _color_filename='Color_Usage'):
        if not isinstance(parameters, dict):
            raise TypeError('wrong input type of parameters, should be a dictionary containing essential info.')

        ## required parameters: data folder (list)
        if isinstance(parameters['data_folder'], list):
            self.data_folder = [str(_fd) for _fd in parameters['data_folder']]
        else:
            self.data_folder = [str(parameters['data_folder'])]

        ## extract hybe folders and field-of-view names
        self.folders = []
        for _fd in self.data_folder:
            _hyb_fds, _fovs = get_img_info.get_folders(_fd, feature='H', verbose=True)
            self.folders += _hyb_fds
            self.fovs = _fovs
        
        ## experiment_type, default is DNA
        if 'experiment_type' in parameters:
            setattr(self, 'experiment_type', parameters['experiment_type'])
        else:
            setattr(self, 'experiment_type', 'DNA')

        # experiment_folder
        if 'experiment_folder'  in parameters:
            self.experiment_folder = parameters['experiment_folder']
        else:
            self.experiment_folder = os.path.join(self.data_folder[0], 'Experiment')
        # experiment type
        if 'experiment_type'  in parameters:
            self.experiment_type = parameters['experiment_type']
        else:
            self.experiment_type = 'DNA'
        ## analysis_folder, segmentation_folder, save_folder, correction_folder,map_folder
        if 'analysis_folder'  in parameters:
            self.analysis_folder = str(parameters['analysis_folder'])
        else:
            self.analysis_folder = self.data_folder[0]+os.sep+'Analysis'
        if 'segmentation_folder' in parameters:
            self.segmentation_folder = parameters['segmentation_folder']
        else:
            self.segmentation_folder = self.analysis_folder+os.sep+'segmentation'
        if 'save_folder' in parameters:
            self.save_folder = parameters['save_folder']
        else:
            self.save_folder = self.analysis_folder+os.sep+'5x10'
        if 'correction_folder' in parameters:
            self.correction_folder = parameters['correction_folder']
        else:
            self.correction_folder = _correction_folder
        if 'drift_folder' in parameters:
            self.drift_folder = parameters['drift_folder']
        else:
            self.drift_folder =  self.analysis_folder+os.sep+'drift'
        if 'map_folder' in parameters:
            self.map_folder = parameters['map_folder']
        else:
            self.map_folder = self.analysis_folder+os.sep+'distmap'

        # number of num_threads
        if 'num_threads' in parameters:
            self.num_threads = parameters['num_threads']
        else:
            self.num_threads = int(os.cpu_count() / 4) # default: use one third of cpus.
        # other shared_parameters for imaging processing, etc
        if "shared_parameters" in parameters:
            self.shared_parameters = parameters['shared_parameters']
        else:
            self.shared_parameters = {}

        ## if loading all remaining attr in parameter
        if _load_all_attr:
            for _key, _value in parameters.items():
                if not hasattr(self, _key):
                    setattr(self, _key, _value)

        ## list to store Cell_data
        self.cells = []
        # distance from pixel to nm:
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

        ## chosen field of views
        if len(_chosen_fovs) == 0: # no specification
            _chosen_fovs = np.arange(len(_fovs))
        if len(_chosen_fovs) > 0: # there are specifications
            _chosen_fovs = [_i for _i in _chosen_fovs if _i <= len(_fovs)]
            _chosen_fovs = list(np.array(np.unique(_chosen_fovs), dtype=np.int))
        # exclude fovs
        if len(_exclude_fovs) > 0: #exclude any fov:
            for _i in _exclude_fovs:
                if _i in _chosen_fovs:
                    _chosen_fovs.pop(_chosen_fovs.index(_i))
        # save values to the class
        self.fov_ids = _chosen_fovs
        self.chosen_fovs = list(np.array(self.fovs)[np.array(self.fov_ids, dtype=np.int)])
        # read color-usage and encodding-scheme
        if not hasattr(self, 'color_dic') or not hasattr(self, 'channels'):
            self._load_color_info(_color_filename=_color_filename)
        # load extra info for DNA / RNA 
        if _load_reference_info:
            if getattr(self, 'experiment_type') == 'RNA' and not hasattr(self, 'rna-info_dic'):
                self._load_rna_info()
            elif getattr(self, 'experiment_type') == 'DNA' and not hasattr(self, 'region_dic'):
                self._load_genomic_regions()        

        # get annotated folders by color usage
        self.annotated_folders = []
        for _hyb_fd, _info in self.color_dic.items():
            _matches = [_fd for _fd in self.folders if _hyb_fd == _fd.split(os.sep)[-1]]
            if len(_matches)==1:
                self.annotated_folders.append(_matches[0])
        print(f"{len(self.annotated_folders)} folders are found according to color-usage annotation.")
        # tool for iteration
        self.index = 0


    # allow print info of Cell_List
    def __str__(self):
        if hasattr(self, 'data_folder'):
            print("Data folder:", self.data_folder)
        if hasattr(self, 'cells'):
            print("Number of cells in this list:", len(self.cells))
        return 'test'
    # allow iteration of Cell_List
    def __iter__(self):
        return self.cells
    def __next__(self):
        if not hasattr(self, 'cells') or not not hasattr(self, 'index'):
            raise StopIteration
        elif self.index == 0:
            raise StopIteration
        else:
            self.index -= 1
        return self.cells[self.index]

    ## Load basic info
    def _load_color_info(self, _color_filename='Color_Usage', _color_format='csv', _save_color_dic=True):
        _color_dic, _use_dapi, _channels = get_img_info.Load_Color_Usage(self.analysis_folder,
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
        _bead_channel = get_img_info.find_bead_channel(_color_dic)
        self.bead_channel_index = _bead_channel
        _dapi_channel = get_img_info.find_dapi_channel(_color_dic)
        self.dapi_channel_index = _dapi_channel

        return _color_dic

    ## load RNA
    def _load_rna_info(self, _filename='RNA_Info', _table_format='csv', 
                       _match_to_genomic_region=True, _verbose=True):
        """Load RNA information"""
        _rna_dic = get_img_info.Load_RNA_Info(self.analysis_folder, filename=_filename,
                                              table_format=_table_format, verbose=_verbose)
        if _match_to_genomic_region:
            _region_dic = self._load_genomic_regions(_verbose=_verbose)
            _rna_dic = get_img_info.match_RNA_to_DNA(_rna_dic, _region_dic)
        # set to attribute
        setattr(self, 'rna-info_dic', _rna_dic)

        return _rna_dic

    ## load Gene
    def _load_gene_info(self, _filename='Gene_Info', _table_format='csv', 
                       _match_to_genomic_region=True, _verbose=True):
        """Load RNA information"""
        _gene_dic = get_img_info.Load_Gene_Info(self.analysis_folder, filename=_filename,
                                              table_format=_table_format, verbose=_verbose)
        if _match_to_genomic_region:
            _region_dic = self._load_genomic_regions(_verbose=_verbose)
            _gene_dic = get_img_info.match_Gene_to_DNA(_gene_dic, _region_dic)
        # set to attribute
        setattr(self, 'gene_dic', _gene_dic)

        return _gene_dic


    ## load genomic regions
    def _load_genomic_regions(self, _filename='Region_Positions', _table_format='csv', _verbose=True):
        """Function to load Genomic Positions etc."""
        _region_dic = get_img_info.Load_Region_Positions(self.analysis_folder, filename=_filename,
                                                      table_format=_table_format, verbose=_verbose)
        setattr(self, 'region_dic', _region_dic)
        return _region_dic

    def _load_encoding_scheme(self, _encoding_filename='Encoding_Scheme', _encoding_format='csv', _save_encoding_scheme=True):
        _encoding_scheme, self.hyb_per_group, self.reg_per_group, \
        self.encoding_colors, self.encoding_group_nums \
            = get_img_info.Load_Encoding_Scheme(self.analysis_folder,
                                                   encoding_filename=_encoding_filename,
                                                   encoding_format=_encoding_format,
                                                   return_info=True)
        # need-based encoding scheme saving
        if _save_encoding_scheme:
            self.encoding_scheme = _encoding_scheme

        return _encoding_scheme

    ## Pick segmentations info for all fovs 
    def _pick_cell_segmentations(self, _num_threads=None, _allow_manual=True,
                                 _min_shape_ratio=0.036, _signal_cap_ratio=0.2, _denoise_window=5,
                                 _shrink_percent=13, _max_conv_th=0, _min_boundary_th=0.48,
                                 _load_in_ram=True, _save=True, _save_npy=True, _save_postfix='_segmentation',
                                 _cell_coord_fl='cell_coords.pkl', _overwrite=False, _verbose=True):
        ## load segmentation
        # check attributes
        if not hasattr(self, 'channels') or not hasattr(self, 'color_dic'):
            self._load_color_info()
        if _num_threads is None:
            if not hasattr(self, 'num_threads'):
                raise AttributeError('No num_threads given in funtion kwds and class attributes')
            else:
                _num_threads = self.num_threads
        # find the folder name for dapi
        _select_dapi = False # not select dapi fd yet
        for _fd, _info in self.color_dic.items():
            if len(_info) >= self.dapi_channel_index+1 and _info[self.dapi_channel_index] == 'DAPI':
                _dapi_fd = [_full_fd for _full_fd in self.annotated_folders if os.path.basename(_full_fd) == _fd]
                if len(_dapi_fd) == 1:
                    if _verbose:
                        print(f"-- choose dapi images from folder: {_dapi_fd[0]}.")
                    _dapi_fd = _dapi_fd[0]
                    _select_dapi = True # successfully selected dapi
        if not _select_dapi:
            raise ValueError("No DAPI folder detected in annotated_folders, stop!")
        # prepare filenames for images to do segmentation
        if _verbose:
            print(f"{len(self.chosen_fovs)} of field-of-views are selected to load segmentation.")
        _chosen_files = [os.path.join(_dapi_fd, _fov) for _fov in self.chosen_fovs]
        # do segmentation
        _segmentation_labels, _dapi_ims = visual_tools.DAPI_convoluted_segmentation(
            _chosen_files, self.channels[self.dapi_channel_index], num_threads=_num_threads,
            single_im_size=self.shared_parameters['single_im_size'], 
            all_channels=self.channels,
            num_buffer_frames=self.shared_parameters['num_buffer_frames'], 
            num_empty_frames=self.shared_parameters['num_empty_frames'], 
            correction_folder=self.correction_folder, 
            illumination_correction=self.shared_parameters['corr_illumination'],
            min_shape_ratio=_min_shape_ratio, signal_cap_ratio=_signal_cap_ratio,
            denoise_window=_denoise_window, shrink_percent=_shrink_percent,
            max_conv_th=_max_conv_th, min_boundary_th=_min_boundary_th,
            make_plot=False, return_images=True, 
            save=_save, save_npy=_save_npy, save_folder=self.segmentation_folder, 
            save_postfix=_save_postfix, force=_overwrite, verbose=_verbose)
        ## pick(exclude) cells from previous result
        if _allow_manual:
            # generate coordinates
            _coord_list, _index_list = [],[]
            for _i, _label in enumerate(_segmentation_labels):
                for _j in range(np.max(_label)):
                    _center = np.round(ndimage.measurements.center_of_mass(_label==_j+1))
                    _center = list(np.flipud(_center))
                    _center.append(_dapi_ims[0].shape[0]/2)
                    _coord_list.append(_center)
                    _index_list.append(_i)
            # wrap into a dic
            _cell_coord_dic = {'coords': _coord_list,
                          'class_ids': _index_list,
                          'pfits':{},
                          'dec_text':{},
                          }
            self.cell_coord_dic = copy.deepcopy(_cell_coord_dic)
            # use visual tools to pick
            _cell_coord_savefile = self.segmentation_folder + os.sep + _cell_coord_fl

            _cell_viewer = visual_tools.imshow_mark_3d_v2(_dapi_ims, image_names=self.chosen_fovs,
                                                          save_file=_cell_coord_savefile,
                                                          given_dic=_cell_coord_dic)

            return _cell_viewer
        else:
            return _segmentation_labels, _dapi_ims

    def _update_cell_segmentations(self, _cell_coord_fl='cell_coords.pkl',
                                  _overwrite_segmentation=False,
                                  _marker_displace_th = 50,
                                  _append_new=True, _append_radius=100,
                                  _overlap_percent=60,
                                  _save_npy=True, _save_postfix="_segmentation",
                                  _make_plot=True, _return_all=False, _verbose=True):
        """Function to update cell segmentation info from saved file,
            - usually do this after automatic segmentation
        Inputs:
            _cell_coord_fl: cell coordinate file generated by _pick_cell_segmentations, str
            _overwrite_segmentation: whether overwrite previous segmentation files, bool (default: True)
            _marker_displace_th: overall displacement of picked cellcenter to previous ones, int (default:300)
            _append_new: whether append manually picked spots, bool (default: True)
            _append_radius: the radii of circled-shape label appended manually, int (default:90)
            _overlap_percent: percentage of manual labels allowed to overlap with existing labels, float (default:60)
            _save_npy: whether save .npy file or .pkl file, bool (default: True)
            _save_postfix: filename postfix for saved segmentation files, str
            _make_plot: whether make plots for new segmentation labels, bool (default: True)
            _return_all: whether return all info, bool (default: False)
            _verbose: say something!, bool (default: True)
        Outputs:
            _new_seg_labels, _remove_cts, _append_cts"""
        ## decide save_handle
        if _save_npy:
            _file_type = '.npy'
        else:
            _file_type = '.pkl'
        print(f"- Update segmentation information for file type: {_file_type}")

        ## check saved cell_coord.pkl file, which was generated by _pick_cell_segmentations
        _cell_coord_savefile = self.segmentation_folder + os.sep + _cell_coord_fl
        if not os.path.exists(_cell_coord_savefile):
            raise IOError(f'{_cell_coord_savefile} doesnot exist, exit')
        # open cell_coord.pkl
        with open(_cell_coord_savefile, 'rb') as handle:
            _new_cell_coord_dic = pickle.load(handle)
        # parse
        _new_ccd = visual_tools.partition_map(_new_cell_coord_dic['coords'], _new_cell_coord_dic['class_ids'])

        ## check if cell_coord for automatic file existed, otherwise load 
        if not hasattr(self, 'cell_coord_dic'):
            # check if all segmentation files exists
            _segmentation_filenames = [os.path.join(self.segmentation_folder, _fov.replace('.dax', _save_postfix + _file_type)) for _fov in self.chosen_fovs]
            _missed_segmentation_files = [_fl for _fl in _segmentation_filenames if not os.path.isfile(_fl)]
            if len(_missed_segmentation_files) > 0:
                raise IOError(f"Not full segmentation results were found, {_missed_segmentation_files} are missing!")
            else:
                # generate coordinates
                _coord_list, _index_list = [],[]
                for _i, _label_file in enumerate(_segmentation_filenames):
                    # load segmentation labels
                    _label = np.load(_label_file)
                    # get centers
                    for _j in range(np.max(_label)):
                        _center = np.round(ndimage.measurements.center_of_mass(_label==_j+1))
                        _center = list(np.flipud(_center)) 
                        _center.append(_image_size[0]/2)
                        _coord_list.append(_center)
                        _index_list.append(_i)
                # wrap into a dic
                _cell_coord_dic = {'coords': _coord_list,
                            'class_ids': _index_list,
                            'pfits':{},
                            'dec_text':{},
                            }
                # save to cell-list
                self.cell_coord_dic = _cell_coord_dic
        # parse
        _ccd = visual_tools.partition_map(self.cell_coord_dic['coords'], self.cell_coord_dic['class_ids'])

        # initialize
        _new_seg_labels, _dapi_ims = [], []
        _remove_cts, _append_cts = [], []
        for _i, (_cell_coords, _new_cell_coords) in enumerate(zip(_ccd, _new_ccd)):
            # now we are taking care of one specific field of view
            if _verbose:
                print(f"-- fov-{_i}, match manually picked cell with sgementation ")

            # load fov image
            _seg_file = os.path.join(self.segmentation_folder, self.chosen_fovs[_i].replace('.dax', _save_postfix+_file_type))
            if _save_npy:
                _seg_label = np.load(_seg_file)
                if not _overwrite_segmentation:
                    # save original seg label into another file
                    _old_seg_folder = os.path.join(os.path.dirname(_seg_file), 'old')
                    if not os.path.exists(_old_seg_folder):
                        os.makedirs(_old_seg_folder)
                    _old_seg_file = os.path.join(os.path.dirname(_seg_file), 
                                                 'old', 
                                                 os.path.basename(_seg_file).replace(_save_postfix+_file_type, _save_postfix))
                    # notice: _file_type .npy was not added to _old_seg_file because np.save automatically adds postfix
                    np.save(_old_seg_file, _seg_label)
            else:
                _seg_label, _dapi_im = pickle.load(open(_seg_file, 'rb'))
                if not _overwrite_segmentation:
                    # save original seg label into another file
                    _old_seg_file = _seg_file.replace(_save_postfix+_file_type, _save_postfix+'_old'+_file_type)
                    pickle.dump([_seg_label, _dapi_im], open(_old_seg_file, 'wb'))

            # keep record of removed labels 
            _remove = 0
            # keep cells in original segmentation with markers
            for _l, _coord in enumerate(_cell_coords):
                _dist = [np.sum((_c-_coord)**2) for _c in _new_cell_coords]
                _match = [_d < _marker_displace_th for _d in _dist]
                if sum(_match) == 0:
                    _seg_label[_seg_label==_l+1-_remove] = -1
                    _seg_label[_seg_label >_l+1-_remove] -= 1
                    _remove += 1
            if _append_new:
                _append = 0
                if _verbose:
                    print(f"-- Appending manually added markers with radius={_append_radius}")
                # local function used to add a new marker to label 
                def _add_round_marker(_label, _center, _radius, _overlap_percent=60, overwrite_marker=False):
                    """Function to add round-marker with given center and radius"""
                    if len(_label.shape) != len(_center):
                        raise ValueError(
                            "Dimension of label and center doesn't match")
                    # convert format
                    _center = np.array(_center, dtype=np.int)
                    _radius = np.int(_radius)
                    # generate mask
                    _shape_lst = (list(range(_label.shape[i]))
                                for i in range(len(_label.shape)))
                    _coord_lst = np.meshgrid(*_shape_lst, indexing='ij')
                    _dist = np.sqrt(np.sum(np.stack(
                        [(_coords - _ct)**2 for _coords, _ct in zip(_coord_lst, _center)]), axis=0))
                    _new_mask = np.array(_dist <= _radius, dtype=np.int)
                    if not overwrite_marker:
                        _new_mask *= np.array(_label <= 0, dtype=np.int)

                    # check overlap percentage of new mask to previous ones
                    _overlap = np.array(_new_mask * (_label > 0), dtype=np.int)

                    if np.float(np.sum(_overlap)) / np.sum(_new_mask) > _overlap_percent / 100.0:
                        print(np.float(np.sum(_overlap)) / np.sum(_new_mask))
                        return _label
                    else:
                        # create new label
                        _new_label = _label.copy()
                        _new_label[_new_mask > 0] = int(np.max(_label))+1
                        return _new_label
                for _l, _new_coord in enumerate(_new_cell_coords):
                    _dist = [np.sum((_c-_new_coord)**2) for _c in _cell_coords]
                    _match = [_d < _marker_displace_th for _d in _dist]
                    if sum(_match) == 0:
                        if _verbose:
                            print(f"--- adding manually picked new label in {_i}, label={np.max(_seg_label)+1} ")
                        _seg_label = _add_round_marker(_seg_label, np.flipud(_new_coord)[-len(_seg_label.shape):], 
                                                       _append_radius, _overlap_percent=_overlap_percent)
                        _append += 1
                _append_cts.append(_append)

            if _verbose:
                print(f"--- {_remove} label(s) got removed!")
            _new_seg_labels.append(_seg_label)
            #_dapi_ims.append(_dapi_im)
            _remove_cts.append(_remove)
            if _make_plot:
                plt.figure()
                plt.imshow(_seg_label)
                plt.colorbar()
                plt.title(f"Updated segmentation: {os.path.basename(_seg_file)}")
                plt.show()
            # save
            if _verbose:
                print(f"--- save updated segmentation to {os.path.basename(_seg_file)}")
            if _save_npy:
                np.save(_seg_file.replace(_save_postfix+_file_type, _save_postfix), _seg_label)
            else:
                pickle.dump([_seg_label, _dapi_im], open(_seg_file, 'wb'))

        #return _new_seg_labels, _dapi_ims, _remove_cts, _append_cts
        if _return_all:
            return _new_seg_labels, _remove_cts, _append_cts
        else:
            # just return numbers of removed and append cells
            return _remove_cts, _append_cts

    ## translate from a previous segmentation
    def _translate_old_segmentations(self, old_segmentation_folder, old_dapi_folder, rotation_mat,
                                    _old_correction_folder=_correction_folder,
                                    _new_correction_folder=_correction_folder,
                                    _num_threads=12, _fft_gb=0, _fft_max_disp=200,
                                    _save=True, _save_postfix='_segmentation',
                                    _save_npy=True, _return_all=False, _force=False, _verbose=True):
        """Function to translate segmenation from a previous experiment 
        given old_segmentation_folder and rotation matrix"""
        # number of threads
        if hasattr(self, 'num_threads'):
            _num_threads = max(_num_threads, self.num_threads)
        # decide filetype
        if _save_npy:
            _file_postfix = '.npy'
        else:
            _file_postfix = '.pkl'
        if _verbose:
            print(
                f"+ Start translating {_file_postfix} segmentation labels from folder:{old_segmentation_folder}")
        # find old segmentation files
        if not os.path.isdir(old_segmentation_folder):
            raise IOError(
                f"old_segmentation_folder:{old_segmentation_folder} doesn't exist, exit!")
        old_seg_filenames = glob.glob(os.path.join(
            old_segmentation_folder, '*' + _file_postfix))
        # find old_dapi_folder
        if not os.path.isdir(old_dapi_folder):
            raise IOError(
                f"old_dapi_folder:{old_dapi_folder} doesn't exist, exit!")
        # create new segmentation folder if necessary
        if not os.path.exists(self.segmentation_folder):
            os.makedirs(self.segmentation_folder)
        # find the folder name for dapi
        _select_dapi = False  # not select dapi fd yet
        for _fd, _info in self.color_dic.items():
            if len(_info) >= self.dapi_channel_index+1 and _info[self.dapi_channel_index] == 'DAPI':
                _dapi_fd = [_full_fd for _full_fd in self.annotated_folders if os.path.basename(
                    _full_fd) == _fd]
                if len(_dapi_fd) == 1:
                    if _verbose:
                        print(f"-- choose dapi images from folder: {_dapi_fd[0]}.")
                    _dapi_fd = _dapi_fd[0]
                    _select_dapi = True  # successfully selected dapi
        if not _select_dapi:
            raise ValueError("No DAPI folder detected in annotated_folders, stop!")

        # translate segmentation file
        _seg_args, _seg_fls = [], [] # list for multi-processing
        _new_filenames, _new_labels, _dapi_ims = [], [], [] # list for final results
        for _old_fl in old_seg_filenames:
            _new_fl = os.path.join(self.segmentation_folder,
                                os.path.basename(_old_fl))
            _dapi_im_name = os.path.basename(_old_fl).replace(
                _save_postfix+_file_postfix, '.dax')
            # translate new segmentation if it doesn't exists or force to generate new ones
            if _force or not os.path.exists(_new_fl):
                if _verbose:
                    print(f"++ prepare translating segmentation label:{_old_fl}")
                # prepare args for multi-processing
                _arg = (_old_fl, os.path.join(old_dapi_folder, _dapi_im_name), os.path.join(_dapi_fd, _dapi_im_name),
                        rotation_mat, None, '405', self.channels, 
                        self.shared_parameters['num_buffer_frames'], 
                        self.shared_parameters['num_empty_frames'], 
                        _old_correction_folder, _new_correction_folder,
                        _fft_gb, _fft_max_disp, _return_all, _verbose)
                _seg_args.append(_arg)
                _seg_fls.append(_new_fl)
            else:
                if _verbose:
                    print(f"++ directly loading segmentation label:{_new_fl}")
                if _save_npy:
                    _new_label = np.load(_new_fl)
                    if _return_all:
                        _dapi_im = corrections.correct_single_image(os.path.join(
                            _dapi_fd, _dapi_im_name), self.channels[self.dapi_channel_index],
                            correction_folder=self.correction_folder,
                            single_im_size=self.shared_parameters['single_im_size'], 
                            all_channels=self.channels,
                            num_buffer_frames=self.shared_parameters['num_buffer_frames'], 
                            num_empty_frames=self.shared_parameters['num_empty_frames'], 
                            )
                else:
                    _new_label, _dapi_im = pickle.load(open(_new_fl, 'rb'))
                _new_labels.append(_new_label)
                _dapi_ims.append(_dapi_im)

        ## multi-processing for translating segmentation
        with mp.Pool(_num_threads,) as _seg_pool:
            if _verbose:
                print(f"+ Start multi-processing of translate_segmentation for {len(_seg_args)} fovs!")
            # Multi-proessing!
            _seg_result = _seg_pool.starmap(visual_tools.translate_segmentation, _seg_args, chunksize=1)
            # close multiprocessing
            _seg_pool.close()
            _seg_pool.join()
            _seg_pool.terminate()
        # clear
        batch_functions.killchild()
        del(_seg_args) 
        # extract result
        _new_filenames += _seg_fls # filenames
        if _return_all:
            _new_labels += [_r[0] for _r in _seg_result] # segmentation_label
            _dapi_ims += [_r[1] for _r in _seg_result] # dapi_im
        else:
            _new_labels += _seg_result # segmentation_label only
        
        ## save
        if _save:
            if _verbose:
                print(f"++ saving segmentation result to file:{_new_fl}")
            if _save_npy or not _return_all:
                [np.save(_new_fl.replace('.npy', ''), _new_label) 
                 for _new_fl, _new_label in zip(_new_filenames, _new_labels)]
            else:
                [pickle.dump([_new_label, _dapi_im], open(_new_fl, 'wb'))
                 for _new_fl, _new_label in zip(_new_filenames, _new_labels)]

        # return
        if _return_all:
            return _new_labels, _dapi_ims
        else:
            return True

    def _create_cell(self, _parameter, _load_info=True, _color_filename='Color_Usage',
                     _load_segmentation=True, _load_drift=True, _drift_size=500, _drift_ref=0, 
                     _drift_postfix='_sequential_current_cor.pkl', _dynamic=True, 
                     _load_cell=True, _exclude_attrs=[],
                     _save=False, _append_cell_list=False, _verbose=True):
        """Function to create one cell_data object"""
        if _verbose:
            print(f"+ creating cell for fov:{_parameter['fov_id']}, cell:{_parameter['cell_id']}")
        _cell = Cell_Data(_parameter, _load_all_attr=True, _load_reference_info=False)
        if _load_info:
            if not hasattr(_cell, 'color_dic') or not hasattr(_cell, 'channels'):
                _cell._load_color_info(_color_filename=_color_filename)
        # load segmentation
        if _load_segmentation and (not hasattr(_cell, 'segmentation_label') or not hasattr(_cell, 'segmentation_crop')):
            _cell._load_segmentation(_load_in_ram=True)
        # load drift  v
        if _load_drift and not _cell._check_drift(_verbose=False):
            _cell._load_drift(_num_threads=self.num_threads, _size=_drift_size, _ref_id=_drift_ref, _drift_postfix=_drift_postfix,
                               _dynamic=_dynamic, _force=False, _verbose=_verbose)
        # load cell_info
        if _load_cell and os.path.exists(os.path.join(_cell.save_folder, 'cell_info.pkl')):
            _cell._load_from_file('cell_info', _exclude_attrs=_exclude_attrs,
                                  _overwrite=False, _verbose=_verbose)
        if _save:
            _cell._save_to_file('cell_info')
        # whether directly store
        if _append_cell_list:
            self.cells.append(_cell)
        return _cell

    def _create_cells_fov(self, _fov_ids, _num_threads=None, _sequential_mode=False, _plot_segmentation=True, 
                          _load_segmentation=True, _load_exist_info=True, _exclude_attrs=[],
                          _color_filename='Color_Usage', _load_annotated_only=True,
                          _drift_size=500, _drift_ref=0, _drift_postfix='_current_cor.pkl', _coord_sel=None,
                          _dynamic=True, _save=False, _force_drift=False, _stringent=True, _verbose=True):
        """Create Cele_data objects for one field of view"""
        if not _num_threads:
            _num_threads = int(self.num_threads)
        if isinstance(_fov_ids, int):
            _fov_ids = [_fov_ids]
        for _fov_id in _fov_ids:
            if _fov_id not in self.fov_ids:
                raise ValueError("Wrong fov_id kwd given! \
                    this should be real fov-number that allowed during intiation of class.")
        if _verbose:
            print(f"+ Create Cell_Data objects for field of view: {_fov_ids}")
            print("++ preparing variables")
        # whether load annotated hybs only
        if _load_annotated_only:
            _folders = self.annotated_folders
        else:
            _folders = self.folders
        # check attributes
        if not hasattr(self, 'channels') or not hasattr(self, 'color_dic'):
            self._load_color_info(_color_filename=_color_filename)
        # find the folder name for dapi
        _select_dapi = False  # not select dapi fd yet
        for _fd, _info in self.color_dic.items():
            if len(_info) >= self.dapi_channel_index+1 and _info[self.dapi_channel_index] == 'DAPI':
                _dapi_fd = [_full_fd for _full_fd in _folders if os.path.basename(_full_fd) == _fd]
                if len(_dapi_fd) == 1:
                    if _verbose:
                        print(f"++ choose dapi images from folder: {_dapi_fd[0]}.")
                    _dapi_fd = _dapi_fd[0]
                    _select_dapi = True  # successfully selected dapi
        if not _select_dapi:
            raise ValueError("No DAPI folder detected in annotated_folders, stop!")
        ## load segmentation for this fov
        _args = []
        for _fov_id in _fov_ids:
            if _verbose:
                print("+ Load segmentation for fov", _fov_id)
            # do segmentation if necessary, or just load existing segmentation file
            _fov_segmentation_labels = visual_tools.DAPI_convoluted_segmentation(
                os.path.join(_dapi_fd, self.fovs[_fov_id]), self.channels[self.dapi_channel_index],
                single_im_size=self.shared_parameters['single_im_size'], 
                all_channels=self.channels,
                num_buffer_frames=self.shared_parameters['num_buffer_frames'], 
                num_empty_frames=self.shared_parameters['num_empty_frames'], 
                illumination_correction=self.shared_parameters['corr_illumination'],
                correction_folder=self.correction_folder, 
                num_threads=_num_threads, make_plot=_plot_segmentation, return_images=False,
                save=_save, save_npy=True, save_folder=self.segmentation_folder, force=False,verbose=_verbose)
            # extract result segmentation and image
            _fov_segmentation_label = _fov_segmentation_labels[0]
            # make plot if necesary
            if _plot_segmentation:
                plt.figure()
                plt.imshow(_fov_segmentation_label)
                plt.colorbar()
                plt.title(f"Segmentation result for fov:{self.fovs[_fov_id]}")
                plt.show()
            # check whether can directly load drift
            _direct_load_drift = False
            _drift_filename = os.path.join(self.drift_folder, self.fovs[_fov_id].replace('.dax', _drift_postfix))
            if os.path.isfile(_drift_filename):
                _drift = pickle.load(open(_drift_filename, 'rb'))
                _exist = [os.path.join(os.path.basename(_fd),self.fovs[_fov_id]) for _fd in _folders \
                        if os.path.join(os.path.basename(_fd),self.fovs[_fov_id]) in _drift]
                if len(_exist) == len(self.annotated_folders):
                    _direct_load_drift = True
            if not _direct_load_drift:
                if _verbose:
                    print(f"+ Generate drift correction profile for fov:{self.fovs[_fov_id]}")
                _drift, _failed_count = corrections.Calculate_Bead_Drift(_folders, self.fovs, _fov_id, 
                                            num_threads=_num_threads, sequential_mode=_sequential_mode, 
                                            single_im_size=self.shared_parameters['single_im_size'], 
                                            all_channels=self.channels,
                                            num_buffer_frames=self.shared_parameters['num_buffer_frames'], 
                                            num_empty_frames=self.shared_parameters['num_empty_frames'], 
                                            illumination_corr=self.shared_parameters['corr_illumination'],
                                            correction_folder=self.correction_folder,
                                            ref_id=_drift_ref, drift_size=_drift_size, save_postfix=_drift_postfix, 
                                            coord_sel=_coord_sel, stringent=_stringent,
                                            ref_seed_per=90,
                                            overwrite=_force_drift, verbose=_verbose)

            # create cells in parallel
            _cell_ids = np.array(np.unique(_fov_segmentation_label[_fov_segmentation_label>0])-1, dtype=np.int)
            if _verbose:
                print(f"+ Create cell_data objects, num_of_cell:{len(_cell_ids)}")
            _params = [{'fov_id': _fov_id,
                      'cell_id': _cell_id,
                      'folders': self.folders,
                      'fovs': self.fovs,
                      'data_folder': self.data_folder,
                      'color_dic': self.color_dic,
                      'use_dapi': self.use_dapi,
                      'channels': self.channels,
                      'bead_channel_index': self.bead_channel_index,
                      'dapi_channel_index': self.dapi_channel_index,
                      'annotated_folders': self.annotated_folders,
                      'experiment_folder': self.experiment_folder,
                      'analysis_folder':self.analysis_folder,
                      'save_folder': self.save_folder,
                      'segmentation_folder': self.segmentation_folder,
                      'correction_folder': self.correction_folder,
                      'drift_folder': self.drift_folder,
                      'map_folder': self.map_folder,
                      'shared_parameters': self.shared_parameters,
                      'experiment_type': self.experiment_type, 
                      } for _cell_id in _cell_ids]
            if not _direct_load_drift:
                for _p in _params:
                    _p['drift'] = _drift
            if self.experiment_type == 'RNA':
                for _p in _params:
                    _p['rna-info_dic'] = getattr(self, 'rna-info_dic')
            _args += [(_p, True, _color_filename, _load_segmentation,
                       _direct_load_drift, _drift_size, _drift_ref, 
                       _drift_postfix, _dynamic, _load_exist_info, 
                       _exclude_attrs, _save, 
                       False, _verbose) for _p in _params]
            del(_fov_segmentation_label, _params, _cell_ids)

        ## do multi-processing to create cells!
        if _verbose:
            print(f"+ Creating {len(_args)} cells with {_num_threads} threads.")
        _cell_pool = mp.Pool(_num_threads)
        _cells = _cell_pool.starmap(self._create_cell, _args, chunksize=1)
        _cell_pool.close()
        _cell_pool.terminate()
        _cell_pool.join()
        # clear
        batch_functions.killchild()
        del(_args, _cell_pool)
        # load
        self.cells += _cells

        ## If not directly load drift, do them here:
        for _cell in self.cells:
            if not hasattr(_cell, 'drift'):
                _cell._load_drift(_num_threads=self.num_threads, _size=_drift_size, _ref_id=_drift_ref, 
                                  _drift_postfix=_drift_postfix,_load_annotated_only=_load_annotated_only,
                                  _sequential_mode=_sequential_mode,
                                  _force=_force_drift, _dynamic=_dynamic, _verbose=_verbose)
            if _save:
                _cell._save_to_file('cell_info', _verbose=_verbose)

    # function to do cropping
    def _crop_image_for_cells(self, _data_type='unique', _load_in_ram=False, 
                              _load_annotated_only=True,
                              _extend_dim=20, _corr_drift=True, _save=True, 
                              _force=False, _overwrite_cell_info=False, _verbose=True):
        """Load images for all cells in this cell_list
        Inputs:
            _data_type: loading type for this """
        ## check inputs
        # check whether cells and segmentation,drift info exists
        if _verbose:
            print (f"+ Load images for {len(self.cells)} cells in this cell list")
        if not hasattr(self, 'cells'):
            raise ValueError("No cells loaded in cell_list")
        if len(self.cells) == 0:
            print("cell_list is empty, exit.")
        # check type
        _data_type = _data_type.lower()
        if _data_type not in self.shared_parameters['allowed_data_types']:
            raise ValueError(f"Wrong _data_type kwd, {_data_type} is given, {self.shared_parameters['allowed_data_types']} are expected")
        # whether load annotated hybs only
        if _load_annotated_only:
            _folders = self.annotated_folders
        else:
            _folders = self.folders

        ## Start to generate temp_files
        # collect field of views
        _used_fov_ids = []
        for _cell in self.cells:
            if _cell.fov_id not in _used_fov_ids:
                _used_fov_ids.append(_cell.fov_id)

        if _data_type in self.shared_parameters['allowed_data_types']:
            if _verbose:
                print(f"+ generating unique images for field-of-view:{_used_fov_ids}")
            for _fov_id in _used_fov_ids:
                _fov_cells = [_cell for _cell in self.cells if _cell.fov_id==_fov_id]
                for _cell in _fov_cells:
                    # if not all unique exists for this cell:
                    if not _cell._check_full_set(_data_type) or _force:
                        if _verbose:
                            print(f"+ Crop unique images for fov:{_cell.fov_id}, cell:{_cell.cell_id}")
                        _cell._crop_images(_data_type, _num_threads=self.num_threads, 
                                           _load_in_ram=_load_in_ram, _extend_dim=_extend_dim,
                                           _save=_save, _overwrite=_force, 
                                           _overwrite_cell_info=_overwrite_cell_info,
                                           _verbose=_verbose)
                    else:
                        if _verbose:
                            print(f"+ unique info exists for fov:{_cell.fov_id}, cell:{_cell.cell_id}, skip")


    def _crop_image_by_fov(self, _data_type='unique', _num_threads=None,
                            _load_in_ram=False, 
                            _load_annotated_only=True, _sel_folders=None,
                            _extend_dim=20, _corr_drift=True, _shift_order=1,
                            _save=True, 
                            _force=False, _overwrite_cell_info=False, _verbose=True):
        """Function to crop image for the whole cell_list by field-of-view
        Inputs:
        
        Outputs:
            """
        ## check inputs
        from ..corrections import multi_correct_one_dax
        # check whether cells and segmentation,drift info exists
        if _verbose:
            print (f"+ Load images for {len(self.cells)} cells in this cell list")
        if not hasattr(self, 'cells'):
            raise ValueError("No cells loaded in cell_list")
        if len(self.cells) == 0:
            print("+ cell_list is empty, exit.")
            return 
        # check type
        _data_type = _data_type.lower()
        if _data_type not in self.shared_parameters['allowed_data_types']:
            raise ValueError(f"Wrong _data_type kwd, {_data_type} is given, {self.shared_parameters['allowed_data_types']} are expected")
        else:
            # generate attribute names
            _im_attr = _data_type + '_' + 'ims'
            _id_attr = _data_type + '_' + 'ids'
            _channel_attr = _data_type + '_' + 'channels'
            
        # whether load annotated hybs only
        if _load_annotated_only:
            _folders = self.annotated_folders
        else:
            _folders = self.folders
        # if specified selected folders
        if _sel_folders is not None:
            if not isinstance(_sel_folders, list):
                raise TypeError(f"_sel_folders should be a list but {type(_sel_folders)} is given.")

        # collect field of views
        _used_fov_ids = []
        for _cell in self.cells:
            if _cell.fov_id not in _used_fov_ids:
                _used_fov_ids.append(_cell.fov_id)

        ## Start to loop through field-of-view
        if _verbose:
            print(f"++ generating unique images for field-of-view:{_used_fov_ids}")
            _fov_start = time.time()
        for _fov_id in _used_fov_ids:
            # load data_type data for this fov
            self._load_cells_from_files(_data_type=_data_type, _sel_fovs=[_fov_id])
            # first get cells
            _fov_cells = [_cell for _cell in self.cells if _cell.fov_id==_fov_id]
            print('fov', _fov_id, len(_fov_cells))
            _fov_name = self.fovs[_fov_cells[0].fov_id]
            # get segmentation crops for cells
            for _cell in _fov_cells:
                if not hasattr(_cell, 'segmentation_crop'): # check segmentation crops
                    _cell._load_segmentation()
            _crops = [_cell.segmentation_crop for _cell in _fov_cells]
            # get corresponding ids for fov_cells
            _reg_id_list = []
            for _cell in _fov_cells:
                _reg_id_list.append(getattr(_cell, _id_attr, []))
            # for all corresponding folders and colors, pick the ones to be corrected
            _corr_colors = []
            _corr_ids = []
            _corr_folders = []
            for _fd in _folders:
                # skip if sel_folders not including this folder
                if _sel_folders is not None and _fd not in _sel_folders:
                    continue
                # loop through colors
                _base_fd = os.path.basename(_fd)
                _colors_to_process = []
                _ids_to_process = []
                for _c, _info in zip(self.channels, self.color_dic[_base_fd]):
                    if self.shared_parameters['allowed_data_types'][_data_type] in _info:
                        _rid = int( _info.split(self.shared_parameters['allowed_data_types'][_data_type])[1] )
                        # check whether this exist or not in all cells, if not exists in all, append
                        _exist_in_cells = [_rid in _ids for _ids in _reg_id_list]
                        if np.sum(_exist_in_cells) < len(_exist_in_cells): # not exist in all cells
                            _colors_to_process.append(_c)
                            _ids_to_process.append(_rid)
                    
                # after loop through colors, if there's any color to process, append
                if len(_colors_to_process) > 0:
                    _corr_folders.append(_fd)
                    _corr_colors.append(_colors_to_process)
                    _corr_ids.append(_ids_to_process)
            
            # start multi-processing
            _corr_args = []
            for _fd, _colors in zip(_corr_folders, _corr_colors):
                _dft = _fov_cells[0].drift
                _dft_q = os.path.join(os.path.basename(_fd), _fov_name)
                if _dft_q not in _dft:
                    raise KeyError(f"drift for {_dft_q} doesn't exist in fov:{_fov_cells[0].fov_id}")
                _corr_args.append(
                    (os.path.join(_fd, _fov_name), _colors, _crops, None, _extend_dim,
                    self.shared_parameters['single_im_size'],self.channels,
                    self.shared_parameters['num_buffer_frames'], self.shared_parameters['num_empty_frames'],
                    _dft[_dft_q], _shift_order,
                    None, self.correction_folder, 
                    self.shared_parameters['normalization'], 
                    self.shared_parameters['corr_bleed'],
                    self.shared_parameters['corr_Z_shift'],
                    self.shared_parameters['corr_hot_pixel'], 
                    self.shared_parameters['corr_illumination'], 
                    self.shared_parameters['corr_chromatic'],
                    False, _verbose,)
                )
            if _num_threads is None:
                _num_threads = self.num_threads
            with mp.Pool(_num_threads) as _corr_pool:
                if _verbose:
                    print(f"++ start multi-processing with {_num_threads} threads for {len(_fov_cells)} cells in fov:{_fov_cells[0].fov_id}")
                    print(f"++ number of jobs: {len(_corr_args)}")
                    _start_time = time.time()
                _cropped_ims_for_cells = _corr_pool.starmap(multi_correct_one_dax, _corr_args, chunksize=1)
                _corr_pool.close()
                _corr_pool.join()
                _corr_pool.terminate()
            if _verbose:
                print(f"+++ time spent in multiprocessing cropping:{time.time()-_start_time}s")
            print('result length:', len(_cropped_ims_for_cells))
            # summarize for each of the cells
            if _verbose:
                print(f"++ summarize cropped {_data_type} images for cells in fov:{_fov_id}")
                _summ_start = time.time()
            
            for _cell_id, _cell in enumerate(_fov_cells):
                # initialize
                _cell_ims = getattr(_fov_cells[_cell_id], _im_attr, [])
                _cell_ids = list(getattr(_fov_cells[_cell_id], _id_attr, []))
                _cell_channels = getattr(_fov_cells[_cell_id], _channel_attr, [])
                # loop through result and append
                for _im_cells_list, _id_list, _color_list in zip(_cropped_ims_for_cells, _corr_ids, _corr_colors):
                    _im_list = _im_cells_list[_cell_id]
                    # loop through multiple colors
                    for _im, _id, _color in zip(_im_list, _id_list, _color_list):
                        # case1, completely new id, append
                        if _id not in list(_cell_ids):
                            _cell_ims.append(_im)
                            _cell_ids.append(_id)
                            _cell_channels.append(_color)
                            print("append_id", _id)
                        # case2, exist and not force, skip
                        elif _id in list(_cell_ids) and not _force:
                            print("skip id", _id)
                            continue
                            
                        # case3, exist and force, overwrite
                        else:
                            _index = list(_cell_ids).index(_id)
                            _cell_ims[_index] = _im
                            _cell_channels[_index] = _color
                            print("replace_id", _id)
                print('cell_size', len(_cell_ids), len(_cell_ims), len(_cell_channels))
                # sort
                _tp = [(_id, _im, _ch) for _id, _im, _ch in sorted(zip(_cell_ids, _cell_ims, _cell_channels))]
                _sorted_ids = [_t[0] for _t in _tp]
                _sorted_ims = [_t[1] for _t in _tp]
                _sorted_channels = [_t[2] for _t in _tp]
                print('size', len(_sorted_ids), len(_sorted_ims), len(_sorted_channels))
                # append to attributes
                setattr(_fov_cells[_cell_id], _im_attr, _sorted_ims)
                setattr(_fov_cells[_cell_id], _id_attr, _sorted_ids)
                setattr(_fov_cells[_cell_id], _channel_attr, _sorted_channels)
                # replace the one in cell_list
                for _old_id, _old_cell in enumerate(self.cells):
                    if _old_cell.fov_id == _fov_cells[_cell_id].fov_id \
                        and _old_cell.cell_id == _fov_cells[_cell_id].cell_id:
                        # update cell_list 
                        self.cells[_old_id] = _fov_cells[_cell_id]
            if _verbose:
                print(f"+++ time spent in summarizing: {time.time()-_summ_start}s")
            # save to file if specified:
            if _save:
                if _verbose:
                    print(f"++ save result to {_data_type} file and cell_info:")
                    _save_start = time.time()
                self._save_cells_to_files(_data_type=_data_type, _sel_fovs=[_fov_id],
                                         _overwrite=_force, _verbose=_verbose)
                self._save_cells_to_files('cell_info', _sel_fovs=[_fov_id],
                                         _overwrite=_overwrite_cell_info, _verbose=_verbose)
                if _verbose:
                    print(f"+++ time spent in saving:{time.time()-_save_start}s")
            # remove if load_in_ram is false
            if not _load_in_ram:
                for _cell in _fov_cells:
                    delattr(_cell, _im_attr)
            if _verbose:
                print(f"++ time spent for fov:{_fov_id}: {time.time()-_fov_start}s")




    # load processed cell info/unique/decoded/merfish from files
    def _load_cells_from_files(self, _data_type='cell_info', _num_threads=None, 
                               _sel_fovs=None, _save_folder=None,
                               _decoded_flag=None, _distmap_data='unique', _distmap_pick='EM',
                               _load_attrs=[], _exclude_attrs=[], _overwrite=False, _verbose=True):
        """Function to load cells from existing files"""
        if _num_threads is None:
            _num_threads = getattr(self, 'num_threads')
        if _verbose:
            print(f"+ Load {_data_type} for cells from existing files.")
        if not hasattr(self, 'cells') or len(self.cells) == 0:
            raise ValueError(
                'No cell information provided, should create cells first!')
        # check fov_id input
        _loading_args = []
        # prepare args
        for _cell in self.cells:
            if _sel_fovs is None or (_sel_fovs is not None and _cell.fov_id in _sel_fovs):
                _loading_args.append((_cell, _data_type, _save_folder,
                                    _decoded_flag, _distmap_data, _distmap_pick,
                                    _load_attrs, _exclude_attrs, _overwrite, _verbose))
        if _verbose:
            print(f"++ {len(_loading_args)} of {_data_type} loading jobs planned.")
            _start_time = time.time()
        # load info by multi-processing!
        with mp.Pool(_num_threads) as _loading_pool:
            # Multi-proessing!
            _updated_cells = _loading_pool.starmap(_load_cell_in_batch,
                                                   _loading_args, chunksize=1)
            # close multiprocessing
            _loading_pool.close()
            _loading_pool.join()
            _loading_pool.terminate()
        # update
        _updated_cell_infos = [(_cell.fov_id, _cell.cell_id) for _cell in _updated_cells]
        for _cid, _cell in enumerate(self.cells):
            if (_cell.fov_id, _cell.cell_id) in _updated_cell_infos:
                self.cells[_cid] = _updated_cells[_updated_cell_infos.index((_cell.fov_id, _cell.cell_id))]

        if _verbose:
            print(f"+++ time spent in loading: {time.time()-_start_time}s")

    # load processed cell info/unique/decoded/merfish from files
    def _save_cells_to_files(self, _data_type='cell_info', _num_threads=None, 
                             _sel_fovs=None, _save_folder=None,
                             _save_list=[], _unsaved_attrs=None, _clear_old_attrs=False,
                             _overwrite=False, _verbose=True):
        """Function to load cells from existing files"""
        if _num_threads is None:
            _num_threads = getattr(self, 'num_threads')
        if _verbose:
            print(f"+ Save {_data_type} for cells from existing files.")
        if not hasattr(self, 'cells') or len(self.cells) == 0:
            raise ValueError(
                'No cell information provided, should create cells first!')
        # check fov_id input
        _saving_args = []
        # prepare args
        for _cell in self.cells:
            if _sel_fovs is None or (_sel_fovs is not None and _cell.fov_id in _sel_fovs):
                # generate a temp save_dic
                _save_dic = {_k: getattr(_cell, _k) for _k in _save_list if hasattr(_cell, _k)}
                # append save_arg
                _saving_args.append((_cell, _data_type, _save_dic, _save_folder,
                                    _unsaved_attrs, _clear_old_attrs, 
                                    _overwrite, _verbose))
        if _verbose:
            print(f"++ {len(_saving_args)} of {_data_type} loading jobs submitted to {_num_threads} threads.")
        # load info by multi-processing!
        with mp.Pool(_num_threads) as _saving_pool:
            # Multi-proessing!
            _updated_cells = _saving_pool.starmap(_save_cell_in_batch,
                                                   _saving_args, chunksize=1)
            # close multiprocessing
            _saving_pool.close()
            _saving_pool.join()
            _saving_pool.terminate()

    # generate chromosome coordinates
    def _get_chromosomes_for_cells(self, _source='unique', _max_count= 90,
                                   _gaussian_size=2, _cap_percentile=1, _seed_dim=3,
                                   _th_percentile=99.5, _min_obj_size=125,
                                   _coord_filename='chrom_coords.pkl', _overwrite=False, _verbose=True):
        """Function to generate chromosome and chromosome coordinates, open a picker to correct for it
        Inputs:
            _source: image source to generate chromosome image, combo requires "combo_gorups",
                unique requires 'unique_ims', 'combo'/'unique' (default: 'combo')
            _max_count: maximum image count to generate chromosome profile, int (default:30)
        Outputs:
            _chrom_viewer: chromosome viewer object, used to click"""
        # check attribute
        if not hasattr(self, 'cells') or len(self.cells) == 0:
            raise ValueError('No cells are generated in this cell list!')
        if _verbose:
            print("+ Generate chromosomes for cells.")
        # chromsome savefile #
        _fov_ids = [_cell.fov_id for _cell in self.cells]
        _fov_ids = np.unique(_fov_ids)
        _filename = '_'+str(min(_fov_ids)) + '-' + str(max(_fov_ids))+'.pkl'
        _chrom_savefile = os.path.join(self.save_folder, _coord_filename.replace('.pkl', _filename))
        # loop through cells to generate chromosome
        _chrom_ims = []
        _chrom_dims = []
        _coord_dic = {'coords': [],
                      'class_ids': [],
                      'pfits':{},
                      'dec_text':{},
                      } # initialize _coord_dic for picking
        for _i, _cell in enumerate(self.cells):
            # first try to load chrom_im if not exist right now
            if not hasattr(_cell, 'chrom_im') and not _overwrite:
                _cell._load_from_file('cell_info', _load_attrs=['chrom_im'], _verbose=_verbose)
            # directly use chrom_im in cell_data
            if hasattr(_cell, 'chrom_im') and not _overwrite:
                _cim = _cell.chrom_im
            # else create a new chrom_im
            else:
                _cim = _cell._generate_chromosome_image(_source=_source, _max_count=_max_count, _verbose=_verbose)
                _cell.chrom_im = _cim
            _chrom_ims.append(_cim)
            _chrom_dims.append(np.array(np.shape(_cim)))
            if not hasattr(_cell, 'chrom_coords') and not _overwrite:
                 _cell._load_from_file('cell_info', _load_attrs=['chrom_coords'], 
                                       _verbose=_verbose)
            # directly use chrom_coords in cell_data
            if hasattr(_cell, 'chrom_coords') and not _overwrite:
                _chrom_coords = _cell.chrom_coords
            # else try to generate automatically
            else:
                _chrom_coords = _cell._identify_chromosomes(_gaussian_size=_gaussian_size, _cap_percentile=_cap_percentile,
                                                            _seed_dim=_seed_dim, _th_percentile=_th_percentile,
                                                            _min_obj_size=_min_obj_size,_verbose=_verbose)
            # build chrom_coord_dic
            _coord_dic['coords'] += [np.flipud(_coord) for _coord in _chrom_coords]
            _coord_dic['class_ids'] += list(np.ones(len(_chrom_coords),dtype=np.uint8)*int(_i))
        # create existing coord_dic file
        if _verbose:
            print("++ dumping existing info to file:", _chrom_savefile)
        pickle.dump(_coord_dic, open(_chrom_savefile, 'wb'))
        # convert to the same dimension
        _max_dim = np.max(np.concatenate([_d[np.newaxis,:] for _d in _chrom_dims]), axis=0)
        if _verbose:
            print("Maximum dimension for these images:", _max_dim)
        _converted_ims = [np.ones(_max_dim) * np.min(_cim) for _cim in _chrom_ims]
        for _im, _d, _cim in zip(_converted_ims, _chrom_dims, _chrom_ims):
            _im[:_d[0], :_d[1],:_d[2]] = _cim

        _chrom_viewer = visual_tools.imshow_mark_3d_v2(_converted_ims, image_names=[f"fov:{_cell.fov_id}, cell:{_cell.cell_id}" for _cell in self.cells],
                                                       save_file=_chrom_savefile)
        _chrom_viewer.load_coords()

        return _chrom_viewer

    def _update_chromosomes_for_cells(self, _coord_filename='chrom_coords.pkl', 
                                      _force_save_to_combo=False, _save=True, _verbose=True):
        # check attribute
        if not hasattr(self, 'cells') or len(self.cells) == 0:
            raise ValueError('No cells are generated in this cell list!')
        if _verbose:
            print("+ Update manually picked chromosomes to cells")
        # chromsome savefile #
        _fov_ids = [_cell.fov_id for _cell in self.cells]
        _fov_ids = np.unique(_fov_ids)
        _filename = '_'+str(min(_fov_ids)) + '-' + str(max(_fov_ids))+'.pkl'
        _chrom_savefile = os.path.join(
            self.save_folder, _coord_filename.replace('.pkl', _filename))
        # load from chrom-coord and partition it
        _coord_dic = pickle.load(open(_chrom_savefile, 'rb'))
        _coord_list = visual_tools.partition_map(_coord_dic['coords'], _coord_dic['class_ids'], enumerate_all=True)
        if len(_coord_list) > len(self.cells):
            raise ValueError(f'Number of cells doesnot match between cell-list and {_chrom_savefile}')
        elif len(_coord_list) < len(self.cells):
            print("++ fewer picked chromosome sets discovered than number of cells, append with empty lists.")
            for _i in range(len(self.cells) - len(_coord_list)):
                _coord_list.append([])
        # save to attribute first
        for _cell, _coords in zip(self.cells, _coord_list):
            _chrom_coords = [np.flipud(_coord) for _coord in _coords]
            _cell.chrom_coords = _chrom_coords
            if _verbose:
                print(f"++ matching {len(_chrom_coords)} chromosomes for fov:{_cell.fov_id}, cell:{_cell.cell_id}")
        # then update files if specified
            if _save:
                _cell._save_to_file('cell_info', 
                                    _save_dic={'chrom_coords': _cell.chrom_coords,
                                               'chrom_im':_cell.chrom_im,},  
                                    _verbose=_verbose)
                if hasattr(_cell, 'combo_groups') or _force_save_to_combo:
                    if _cell._check_full_set('combo'):
                        if not hasattr(_cell, 'combo_groups'):
                            _cell._load_from_file('combo', _verbose=_verbose)
                            _load_mk = True
                        else:
                            _load_mk = False
                        _cell._save_to_file('combo', _overwrite=True, _verbose=_verbose)
                        # remove temporarily loaded combo_groups
                        if _load_mk:
                            delattr(_cell, 'combo_groups')
                    else:
                        if _verbose:
                            print(f"++ Combo info not complete for fov:{_cell.fov_id}, cell:{_cell.cell_id}, skip")

    def _translate_chromosome_coords(self, _source_cell_list, _num_threads=12,
                                    _rotation_mat=None, _rotation_ref_file=None,
                                    _rotation_order='reverse', _border_lim=10, 
                                    _save=True, _overwrite=False, _verbose=True):
        """Function to translate chromosome coordinates from source_cell_list
        Inputs:
            _source_cell_list
            _num_threads
            _rotation_mat: rotation matrix, if provided, np.2darray (default:None)
            _rotation_ref_file: file for rotation matrix, string (default:None)
            _rotation_order: whether rotation_mat is forward or reverse, (default:'reverse')
            _border_lim: limit to judge whether close to border, int (default:10)
            _overwrite: whether overwrite existing chrom_coords in this cell_list, bool (default:False)
            _verbose: say something!, bool (default:True)
        """
        from copy import copy
        if _verbose:
            print(f"+ Start translating chromosome coordinates from other cell_list:{_source_cell_list}")
        # load rotation matrix
        if _rotation_mat is None or len(_rotation_mat.shape())!=2 or np.array(_rotation_mat.shape()-2).any():
            if _rotation_ref_file is None:
                _rotation_ref_file = os.path.join(self.experiment_folder, 'rotation.npy')
                _rotation_mat = np.load(_rotation_ref_file)
                if _verbose:
                    print(f"++ neither rotation_mat and rotation_ref_file are given, load from default:\n\t{_rotation_ref_file}")
            else:
                if not os.path.isfile(_rotation_ref_file):
                    raise IOError(f"Wrong input rotation_ref_file:{_rotation_ref_file}")
        # start recording args
        _trans_args = []
        _trans_ids = []
        for _i, _cell in enumerate(self.cells):
            # find matched cell_data 
            _matched_cell = [_src_cell for _src_cell in _source_cell_list.cells 
                            if getattr(_src_cell,'fov_id')==getattr(_cell,'fov_id') 
                            and getattr(_src_cell,'cell_id')==getattr(_cell,'cell_id') ]
            # unique match
            if len(_matched_cell) == 1: 
                _trans_args.append((copy(_matched_cell[0]), copy(_cell), _rotation_mat, 
                                    None, _rotation_order, self.shared_parameters['single_im_size'], 
                                    _border_lim, _overwrite, True, _verbose))
                _trans_ids.append(_i)
            else:
                if _verbose:
                    print(f"++ cell from fov:{_cell.fov_id}, cell:{_cell.cell_id} \
                            doesn't have uniquely matched source cell, skip")
        # multiprocessing for translating chrom_coords
        if hasattr(self, 'num_threads'):
            _num_threads = getattr(self, 'num_threads')
        if _verbose:
            print(
                f"++ start translating chromosomes for {len(_trans_args)} cells with {_num_threads} threads")
        with mp.Pool(_num_threads) as _trans_pool:
            _new_coords = _trans_pool.starmap(visual_tools.translate_chromosome_coordinates, _trans_args)
            _trans_pool.close()
            _trans_pool.join()
            _trans_pool.terminate()
        batch_functions.killchild()

        # save
        for _i, _cell in enumerate(self.cells):
            if _i in _trans_ids:
                _coords = _new_coords[_trans_ids.index(_i)]
                if _overwrite or not hasattr(_cell, 'chrom_coords'):
                    setattr(_cell, 'chrom_coords', _coords)
            if _save:
                if _overwrite:
                    _cell._save_to_file('cell_info',_save_dic={'chrom_coords':_coords}, _verbose=_verbose)
                else:
                    _cell._save_to_file('cell_info',_save_dic={'chrom_coords':_coords}, _verbose=_verbose)

    # multi-gaussian fitting
    def _spot_finding_for_cells(self, _data_type='unique', _decoded_flag='diff', 
                                _max_fitting_threads=12, 
                                _clear_image=False, _normalize_image=True, 
                                _use_chrom_coords=True, 
                                _seed_by_per=True, _th_seed_percentile=90,
                                _max_filt_size=3,
                                _max_seed_count=6, _min_seed_count=3, _fit_window=40,
                                _expect_weight=1000, _min_height=100, _max_iter=10, _th_to_end=1e-6,
                                _save=True, _overwrite=False, _verbose=True):
        """Function to allow multi-fitting in cell_list"""
        ## Check attributes
        for _cell_id, _cell in enumerate(self.cells):
            _clear_image_for_cell = _clear_image # whether clear image for this cell
            _im_attr = _data_type + '_' + 'ims'
            _id_attr = _data_type + '_' + 'ids'
            _result_attr = _data_type + '_' + 'spots'
            if not hasattr(self, _im_attr) or not hasattr(self, _id_attr):
                _clear_image_for_cell = True
                try:
                    _cell._load_from_file(_data_type, _verbose=_verbose)
                except:
                    raise IOError(f"Cannot load {_data_type} files")

            # do multi_fitting
            _cell._multi_fitting_for_chromosome(_data_type=_data_type, _decoded_flag=_decoded_flag, 
                                                _normalization=_normalize_image, _use_chrom_coords=_use_chrom_coords,
                                                _num_threads=max(_max_fitting_threads, self.num_threads),
                                                _seed_by_per=_seed_by_per, 
                                                _th_seed_percentile=_th_seed_percentile,_max_filt_size=_max_filt_size, 
                                                _max_seed_count=_max_seed_count, _min_seed_count=_min_seed_count, 
                                                _fit_window=_fit_window, _expect_weight=_expect_weight, 
                                                _min_height=_min_height, _max_iter=_max_iter,
                                                _save=_save, _overwrite=_overwrite, _verbose=_verbose)
            if _clear_image_for_cell:
                if _verbose:
                    print(f"++ clear images for {_data_type} in fov:{_cell.fov_id}, cell:{_cell.cell_id}")
                delattr(_cell, _im_attr)

    # new version for batch pick spots
    def _pick_spots_for_cells(self, _data_type='unique', 
                              _pick_type='EM', decoded_flag='diff',
                              _num_threads=12, _use_chrom_coords=True,  _sel_ids=None, 
                              _num_iters=10, _terminate_th=0.0025, 
                              _intensity_th=1., _hard_intensity_th=True,
                              _spot_num_th=100, _ref_spot_list=None, _ref_spot_ids=None, _ref_pick_type='EM',
                              _ignore_ids=False, _ref_dist_metric='median', _score_metric='linear',
                              _local_size=5, _w_ctdist=1, _w_lcdist=0.1, _w_int=1, _w_nbdist=3,
                              _save_inter_plot=False, _save_to_info=True, _save_plot=True,
                              _check_spots=True, _check_th=-1.5, 
                              _check_percentile=10., _hard_dist_th=6000,
                              _distance_limits=[0,np.inf], _ignore_nan=True, 
                              _nan_mask=0., _inf_mask=-1000., _chrom_share_spots=False,
                              _plot_limits=[0, 1500], _cmap='seismic_r', _fig_dpi=100, _fig_size=4,
                              _release_ram=False, _overwrite=False, _verbose=True):
        """Function to pick spots given candidates in batch"""
        ## Check Inputs
        if _verbose:
            print(f"+ Pick spots and convert to distmap, use_chrom_coords:{_use_chrom_coords}")
            _start_time = time.time()
        if _pick_type not in ['dynamic', 'naive', 'EM']:
            raise ValueError(
                f"Wrong _pick_type kwd given ({_pick_type}), should be dynamic or naive.")
        # check num_threads
        if _num_threads is None:
            _num_threads = self.num_threads
        if _save_inter_plot:
            print(
                "++ _save_inter_plot is ON for now, which may requires long time to finish.")
        # decide references
        if _ref_spot_list is None or isinstance(_ref_spot_list, str):
            _ref_spot_list = [_ref_spot_list for _cell in self.cells]
            _ref_id_list = [None for _cell in self.cells]
        elif isinstance(_ref_spot_list, list):
            if len(_ref_spot_list) != len(self.cells):
                raise IndexError(f"Wrong length of _ref_spot_list as list:{len(_ref_spot_list)}, should be same as number of cells:{len(self.cells)} ")
            if _ref_spot_ids is None:
                _ref_id_list = [None for _cell in self.cells]
            else:
                _ref_id_list = [_ref_spot_ids for _cell in self.cells]
        ## start generate multi-processing args
        _pick_args = []
        for _i, _cell in enumerate(self.cells):
            # extract references
            _ref_spots = _ref_spot_list[_i]
            _ref_ids = _ref_id_list[_i]
            _pick_args.append((_cell, _data_type, _pick_type, _use_chrom_coords,
                               _sel_ids, _num_iters, _terminate_th, 
                               _intensity_th, _hard_intensity_th, _spot_num_th,
                               _ref_spots, _ref_ids, _ref_pick_type,
                               _ignore_ids,
                               _ref_dist_metric, _score_metric,
                               _local_size, _w_ctdist, _w_lcdist, _w_int, _w_nbdist,
                               _save_inter_plot, _save_to_info, _save_plot,
                               _check_spots, _check_th, _check_percentile, _hard_dist_th,
                               _distance_limits, _ignore_nan, _nan_mask, _inf_mask,
                               _chrom_share_spots, _plot_limits, _cmap, _fig_dpi, _fig_size,
                               _overwrite, _verbose))
            # create folder to save distmaps ahead
            if _save_plot:
                _distmap_fd = os.path.join(_cell.map_folder, _cell.fovs[_cell.fov_id].replace('.dax',''))
                if not os.path.exists(_distmap_fd):
                    if _verbose:
                        print(f"+++ create distance map folder:{_distmap_fd}")
                    os.makedirs(_distmap_fd)

        with mp.Pool(_num_threads) as _pick_pool:
            _pick_start = time.time()
            if _verbose:
                print(f"++ start multi-processing picking spots by {_pick_type} for {len(self.cells)} cells")
            # feed in args
            _updated_cells = _pick_pool.starmap(_pick_spot_in_batch,
                                                _pick_args, chunksize=1)
            # close multi-processing
            _pick_pool.close()
            _pick_pool.join()
            _pick_pool.terminate()
        # clear
        batch_functions.killchild()
        del(_pick_args)
        if not _release_ram or not _save_to_info:
            if _verbose:
                print("")
            self.cells = _updated_cells
        else:
            for _cell in _updated_cells:
                for _attr in dir(_cell):
                    if _attr[0] != '_' and 'distance_map' in _attr:
                        delattr(_cell, _attr)
            self.cells = _updated_cells
        
        if _verbose:
            print(f"+++ finish in {time.time()-_start_time:.2f}s.")

    # Calculate population median / contact map
    def _calculate_population_map(self, _data_type='unique', 
                                  _pick_type='EM', _stat_type='median',
                                  _max_loss_prob=0.2, _pick_flag=None, _contact_th=200,
                                  _make_plot=True, _save_plot=True, _save_name='distance_map',
                                  _cmap='seismic', _fig_dpi=300, 
                                  _fig_size=4, _gfilt_size=0.75, 
                                  _plot_limits=[0,2000],
                                  _release_ram=False, _return_all_maps=False, _verbose=True):
        """Calculate 'averaged' map for all cells in this list
        Inputs:
            _data_type: unique or decoded
            _max_loss_prob: maximum """
        ## check inputs:
        if _data_type not in self.shared_parameters['allowed_data_types']:
            raise ValueError(f"Wrong _data_type kwd given, should be {self.shared_parameters['allowed_data_types'].keys()}, {_data_type} is given!")
        _allowed_pick_types = ['EM', 'dynamic', 'naive']
        if _pick_type not in _allowed_pick_types:
            raise ValueError(
                f"Wrong _pick_type kwd given ({_pick_type}), should be among {_allowed_pick_types}.")
        if _stat_type not in ['median', 'mean', 'contact']:
            raise ValueError(f"Wrong _stat_type({_stat_type}) kwd is given!")
        if _cmap not in ['seismic', 'Reds']:
            raise ValueError(f"Wrong imnut _cmap:{_cmap}, exit!")
        # check _pick_flag
        if _pick_flag is not None:
            if len(_pick_flag) != len(self.cells):
                raise ValueError(f"_pick_flag should have exactly same length as cells!")
        
        # get distmap attr
        if _pick_type != '':
            _distmap_attr = str(_pick_type) + '_' + str(_data_type) + '_' + 'distance_map'
        else:
            _distmap_attr = str(_data_type) + '_' + 'distance_map'    
        # detect distmap shape
        _distmap_shape=[]
        for _cell_id, _cell in enumerate(self.cells):
            if hasattr(_cell, _distmap_attr):
                # loop through distmaps to get shape
                for _distmap in getattr(_cell, _distmap_attr):
                    if np.shape(_distmap)[0] not in _distmap_shape:
                        _distmap_shape.append(np.shape(_distmap)[0])
            else:
                # try to load distmap
                _cell._load_from_file('distance_map', _distmap_data=_data_type, _distmap_pick=_pick_type, _verbose=False)
                # then do the same: loop through distmaps to get shape
                if hasattr(_cell, _distmap_attr):
                    for _distmap in getattr(_cell, _distmap_attr):
                        if np.shape(_distmap)[0] not in _distmap_shape:
                            _distmap_shape.append(np.shape(_distmap)[0])
            # check if _pick_flag is fully given
            if _pick_flag is not None and len(_pick_flag[_cell_id]) < len(getattr(_cell, _distmap_attr)):
                raise IndexError(f"Wrong _pick_flag for cell-num:{_cell_id}, not enough flags are given")

        if len(_distmap_shape) == 0:
            print("No distant map loaded, return.")
            return None, 0
        if _verbose:
            print(f"+++ maximum distance-map size is {max(_distmap_shape)}")

        _cand_distmaps = []
        ## check and collect distance maps
        for _cell_id, _cell in enumerate(self.cells):
            if not hasattr(_cell, _distmap_attr):
                if _verbose:
                    print(f"+++ fov:{_cell.fov_id}, cell:{_cell.cell_id} doesn't have {_distmap_attr}, skip!")
            else:
                for _chrom_id, _distmap in enumerate(getattr(_cell, _distmap_attr)):
                    # calculate failed entries
                    _chr_failure_rate = np.sum(np.isnan(_distmap).sum(0) == len(_distmap)-1)/len(_distmap)
                    # screen out by flag
                    if _pick_flag is not None and np.max(_pick_flag[_cell_id][_chrom_id]) < 1:
                        if _verbose:
                            print(
                                f"+++ filtered out by pick_flag, fov:{_cell.fov_id}, cell:{_cell.cell_id}, chrom:{_chrom_id}")
                    # screen out by failure rate
                    elif _chr_failure_rate > _max_loss_prob:
                        if _verbose:
                            print(f"+++ filtered out by loss probability, fov:{_cell.fov_id}, cell:{_cell.cell_id}, chrom:{_chrom_id}")
                        continue
                    # screen out by shape
                    elif np.shape(_distmap)[0] != max(_distmap_shape):
                        if _verbose:
                            print(f"+++ filtered out by dist-map shape, fov:{_cell.fov_id}, cell:{_cell.cell_id}, chrom:{_chrom_id}")
                        continue
                    else:
                        _cand_distmaps.append(_distmap.astype(np.float))

        ## calculate averaged map
        # acquire total map
        _total_map = np.array(_cand_distmaps, dtype=np.float)
        _region_failure_rate = np.sum(np.sum(np.isnan(_total_map),axis=1) >= \
                                 np.shape(_total_map)[2]-1, axis=0) / len(_total_map)
        # calculate averaged map
        if _stat_type == 'median':
            _averaged_map = np.nanmedian(_total_map, axis=0)
            _cmap+= '_r'
        elif _stat_type == 'mean':
            _averaged_map = np.nanmean(_total_map, axis=0)
            _cmap += '_r'
        elif _stat_type == 'contact':
            _averaged_map = np.nansum(_total_map < _contact_th, axis=0) / \
                (np.nansum(_total_map < _contact_th, axis=0)+np.nansum(_total_map > _contact_th, axis=0))
            
        # add gaussian filter (to make it nicer!)
        if _gfilt_size:
            from astropy.convolution import Gaussian2DKernel
            from astropy.convolution import convolve
            # remove smoothing artifacts caused by diagonal
            for _i in range(_averaged_map.shape[0]):
                _averaged_map[_i,_i] = np.nan
            _kernel = Gaussian2DKernel(x_stddev=_gfilt_size)
            _averaged_map = convolve(_averaged_map, _kernel)
        
        # change plot_limits for contact map
        if _stat_type == 'contact' and  max(_plot_limits) > 1:
            _plot_limits=[stats.scoreatpercentile(_averaged_map, 1),
                          stats.scoreatpercentile(_averaged_map, 99)]
        
        ## make plots
        if _make_plot:
            if _verbose:
                print(f"++ generating distance map for {len(_cand_distmaps)} chromosomes.")
            _used_fovs = []
            for _cell in self.cells:
                if _cell.fov_id not in _used_fovs:
                    _used_fovs.append(_cell.fov_id)
            _used_fovs = sorted(_used_fovs)
            plt.figure(figsize=(1.25*_fig_size, _fig_size), dpi=_fig_dpi)
            plt.title(f"{_stat_type} {_save_name}, N={len(_cand_distmaps)}")
            
            if _stat_type == 'contact':
                from matplotlib.colors import LogNorm
                _plot_map = _averaged_map.copy()
                _plot_map[_plot_map==0] = np.min(_plot_map[_plot_map!=0])
                plt.imshow(_plot_map, interpolation='nearest', cmap=_cmap,
                           norm=LogNorm(vmin=min(_plot_limits), 
                                        vmax=max(_plot_limits)),
                            )
                plt.colorbar(ticks=np.arange(min(_plot_limits), max(_plot_limits)+0.05,
                             0.05), label='contact prob.')
            else:
                plt.imshow(_averaged_map, interpolation='nearest', cmap=_cmap,
                       vmin=min(_plot_limits), vmax=max(_plot_limits))
                plt.colorbar(ticks=np.arange(min(_plot_limits), max(_plot_limits)+2,
                                             200), label='distance (nm)')

            if _save_plot:
                if _verbose:
                    print(f"++ saving {_stat_type} distance map.")
                if len(_used_fovs) > 10:
                    _filename = os.path.join(self.map_folder, f"{_stat_type}_{_save_name}_fov{min(_used_fovs)}-{max(_used_fovs)}_{_cmap}.png")
                else:
                    _filename = os.path.join(self.map_folder, f"{_stat_type}_{_save_name}_fov{_used_fovs}_{_cmap}.png")

                # add gaussian info if given
                if _gfilt_size:
                    _filename = _filename.replace('.png', f'_g{_gfilt_size}.png')
                if not os.path.exists(self.map_folder):
                    os.makedirs(self.map_folder)
                plt.savefig(_filename, transparent=True)
            if __name__=='__main__':
                plt.show()

        # release ram if specified
        if _release_ram:
            for _cell in self.cells:
                for _attr in dir(_cell):
                    if _attr[0] != '_' and 'distance_map' in _attr:
                        delattr(_cell, _attr)
        # add to attribute
        _save_attr = 'population_' + str(_pick_type)
        # return
        if _return_all_maps:
            return _averaged_map, _total_map, _region_failure_rate
        else:
            return _averaged_map, _region_failure_rate

    # merge RNA list to DNA list
    def _merge_RNA_to_DNA(self, _source_cell_list, _num_threads=None,
                          _match_dic=None, 
                          _merge_type='cell_info', _attr_feature='rna-',
                          _load_in_ram=True, _save_to_file=True, 
                          _overwrite=False, _verbose=True):
        """Batch merge RNA to DNA in two cell_lists 
        ---------------------------------------------------------------------------------
        Inputs:
        
        
        Outouts:"""
        ## check inputs
        if _verbose:
            _start_time = time.time()
            print(f"+ Start merging RNA {_merge_type} into cell list with cells:{len(self.cells)}")
        if _num_threads is None:
            _num_threads = self.num_threads
        # merge type
        _allowed_merge_types = ['cell_info','unique','distance_map']
        _merge_type = _merge_type.lower()
        if _merge_type not in _allowed_merge_types:
            raise ValueError(f"Wrong input _merge_type:{_merge_type}, should be among {_allowed_merge_types}")
        
        # generate 1-1 match_dic
        if _match_dic is None:
            # default is match exactly with fov_id and cell_id
            _match_dic = {(_cell.fov_id, _cell.cell_id): (_cell.fov_id, _cell.cell_id) for _cell in _source_cell_list.cells}
        elif not isinstance(_match_dic, dict):
            raise TypeError(f"_match_dic should be a dict rather than {type(_match_dic)}!")
        else:
            for _k,_v in _match_dic.items():
                if len(_k) != 2 or len(_v) != 2:
                    raise ValueError(f"Wrong input pair for _match_dic: {_k}:{_v}")
                
        # collect args
        _merging_args = []
        # now pair cell_data according to _match_dic
        for _cell in _source_cell_list.cells:
            # key for source cell_data
            _cell_key = (_cell.fov_id, _cell.cell_id)
            if _cell_key in _match_dic:
                # query key for target cell_data
                _target_cell_key = _match_dic[_cell_key]
                # matched_cells 
                _matched_cells =[_tc for _tc in self.cells if _tc.fov_id==_target_cell_key[0] and _tc.cell_id==_target_cell_key[1]]
                if len(_matched_cells) == 1:
                    _matched_cell = _matched_cells[0]
                    # append arg
                    _arg = (_matched_cell, _cell, _merge_type, _attr_feature,
                            _load_in_ram, _save_to_file, _overwrite, _verbose)
                    _merging_args.append(_arg)
        if _verbose:
            print(f"++ {len(_merging_args)} cells are going to merge with RNA info!")
        # copy cells that are not in _match_dic
        _not_merged_cells = [_cell for _cell in self.cells 
                             if (_cell.fov_id, _cell.cell_id) not in list(_match_dic.values())]
        if _verbose:
            print(f"++ {len(_not_merged_cells)} cells are skipped for merging.")
        # merge by multi-processing!
        with mp.Pool(_num_threads) as _merging_pool:
            if _verbose:
                print(f"++ start multiprocessing merging RNA to DNA.")
            # Multi-proessing!
            _updated_cells = _merging_pool.starmap(_merge_RNA_to_DNA_in_batch,
                                                   _merging_args, chunksize=1)
            # close multiprocessing
            _merging_pool.close()
            _merging_pool.join()
            _merging_pool.terminate()
        
        # update
        if _load_in_ram:
            if _verbose:
                print(f"++ update cell_datas into cell_list")
            self.cells = sorted(_updated_cells + _not_merged_cells, key=lambda c:[c.fov_id, c.cell_id])

        if _verbose:
            print(f"++ total time for merging RNA to DNA: {time.time()-_start_time}")

    # get inteinsity statistics
    def _get_intensity_stats(self, _data_type='unique', _pick_type='EM', 
                             _type_marker_dic=None, _num_ref_species=5, 
                             _gaussian_fitting_th=2,
                             _save=True, _filename='spot_intensity', 
                             _make_plot=False, _overwrite=False, _verbose=True):
        """Get intensity statistics for given datatype"""
        ## check inputs
        if _verbose:
            print(f"+ Get intensity information for {_data_type} given {len(self.cells)} cells")
        # first check data_type input
        _allowed_types = ['unique', 'decoded', 'rna-unique']
        _data_type = _data_type.lower()
        if _data_type not in _allowed_types:
            raise KeyError(f"Wrong input key for _data_type:{_data_type}")
        # color_dic
        if 'rna-' in _data_type:
            _use_RNA = True
        else:
            _use_RNA = False 
        self.cells[0]._load_from_file('cell_info')
        if _use_RNA and hasattr(self.cells[0], 'rna-' + 'color_dic'):
            _color_dic = getattr(self.cells[0], 'rna-'+'color_dic')
        else:
            _color_dic = getattr(self.cells[0], 'color_dic')
        if _type_marker_dic is None:
            _type_marker_dic = {'rna-unique':'r',
                                'unique':'u', 
                                'combo':'c'}
        elif not isinstance(_type_marker_dic, dict):
            raise TypeError(f"Wrong input type for _type_marker_dic, should be a dict, but {type(_type_marker_dic)} is given.")

        # check saved file
        _save_filename = os.path.join(self.save_folder,
                                      str(_data_type)+'_'+_filename+'.pkl')
        if os.path.isfile(_save_filename) and not _overwrite:
            if _verbose:
                print(f"++ directly load {_data_type} intensity from file:\n   {_save_filename}")
            _id_gaussian = pickle.load(open(_save_filename, 'rb'))
            return _id_gaussian
        else:
            if _verbose:
                print(f"++ start acquiring {_data_type} intensities for different regions.")
            # generate attribute name
            _id_attr = _data_type + '_' + 'ids'
            _channel_attr = _data_type + '_' + 'channels'
            _spot_attr = _data_type + '_' + 'spots'

            # load from cell_info to get these attributes
            for _cell in self.cells:
                if not hasattr(_cell, _id_attr) or not hasattr(_cell, _channel_attr) or not hasattr(_cell, _spot_attr):
                    _cell._load_from_file('cell_info', _load_attrs=[_id_attr,_channel_attr,_spot_attr],
                                        _verbose=_verbose)

            _color_indices = {_c:[] for _c in self.channels}
            for _hyb_fd, _info_lst in self.color_dic.items():
                for _info, _c in zip(_info_lst, list(_color_indices.keys())[:len(_info_lst)]):
                    if _type_marker_dic[_data_type] in _info:
                        _uid = int(_info.split(_type_marker_dic[_data_type])[1])

                        if _uid in getattr(self.cells[0], _id_attr):
                            _color_indices[_c].append(getattr(self.cells[0], _id_attr).index(_uid))

            
            # record intensities
            _intensities = {_i:[] for _i in getattr(self.cells[0],_id_attr)}

            for _cell in self.cells:
                if hasattr(_cell, _id_attr) and hasattr(_cell, _spot_attr):
                    for _id, _spot_list in zip(getattr(_cell, _id_attr), getattr(_cell, _spot_attr)):
                        if isinstance(_spot_list, list):
                            for _spots in _spot_list:
                                _intensities[_id].append(_spots[:,0])
                        elif len(np.array(_spot_list)) == 2:
                            _intensities[_id].append(np.array(_spot_list)[:,0])
            # merge
            _intensities = {_k:np.concatenate(_v) for _k,_v in _intensities.items()}

            # pick 5 weakest regions/genes as channel representative
            # fit channel specfic feature gaussian
            _color_gaussians = {}
            for _color, _inds in _color_indices.items():
                if len(_inds) > 0:
                    _color_medians = []
                    for _i in _inds:
                        _color_medians.append(np.nanmedian(_intensities[getattr(self.cells[0],_id_attr)[_i]]))
                    _color_medians = np.array(_color_medians)
                    _order = np.argsort(_color_medians)
                    # select 5 genes with lowest median
                    _selected_inds = np.array(_inds, dtype=np.int)[_order[:min(len(_inds),_num_ref_species)]]
                    _sel_ints = np.concatenate([_intensities[getattr(self.cells[0],_id_attr)[_j]] for _j in _selected_inds])
                    # do gaussian fitting for background peak
                    _mean, _std = stats.norm.fit(_sel_ints[_sel_ints<_gaussian_fitting_th])
                    if _make_plot:
                        plt.figure()
                        plt.title(f"channel:{_color}, mean={_mean:.3f}, std={_std:.3f}")
                        plt.hist(_sel_ints, 60, range=(0,2))
                        plt.show()
                    # append to color_gaussian
                    _color_gaussians[_color] = {'mean':copy.copy(_mean), 'std':copy.copy(_std)} 
                    
            # summarize results
            _id_gaussian = {}
            for _id, _ch in zip(getattr(self.cells[0], _id_attr),getattr(self.cells[0], _channel_attr)):
                _temp_dic = {'intensity': _intensities[_id],
                            'params': [ _color_gaussians[str(_ch)]['mean'], _color_gaussians[str(_ch)]['std'] ]
                            }
                _id_gaussian[_id] = _temp_dic
            # save
            if _save:
                if _verbose:
                    print(f"++ save {_data_type} intensity info into file:{_save_filename}")
                pickle.dump(_id_gaussian, open(_save_filename, 'wb'))

            return _id_gaussian

    # generate a p_val flag matching cell_list.cells
    def _p_value_filter(self, _data_type='unique', _pick_type='EM', _use_chrom_coords=True,
                        _ref_dist='gaussian', _ref_dist_params=None,
                        _pval_th=[1e-6,0.01], _save_info=True, _verbose=True):
        """Function to filter acquired spots and filter by p-value specified"""
        ## check inputs
        if _verbose:
            print(f"+ calculate p-value for {_ref_dist} of {_data_type} given {len(self.cells)} cells")
        # first check data_type input
        _allowed_types = ['unique', 'decoded', 'rna-unique']
        _data_type = _data_type.lower()
        if _data_type not in _allowed_types:
            raise KeyError(f"Wrong input key for _data_type:{_data_type}")
            
        # generate attribute name
        _id_attr = _data_type + '_' + 'ids'
        _channel_attr = _data_type + '_' + 'channels'
        _spot_attr = _data_type + '_' + 'spots'
        _result_attr = _data_type + '_' + 'pval_flags'
        # check ref_dist_params
        if _ref_dist_params is None:
            if _verbose:
                print(f"++ acquire param_dic from _get_intensity_stats:")
            _param_dic = self._get_intensity_stats(_data_type=_data_type, _pick_type=_pick_type,
                                                   _save=False, _make_plot=False, 
                                                   _overwrite=False, _verbose=_verbose)
        elif isinstance(_ref_dist_params, list) or isinstance(_ref_dist_params, list):
            if _verbose:
                print(f"++ generate ref_distribution parameter dict by {_ref_dist_params}")
            self.cells[0]._load_from_file('cell_info', _load_attrs=[_id_attr], _verbose=_verbose)
            _param_dic = {_id:{'params':_ref_dist_params} for _id in getattr(self.cells[0], _id_attr)}
        elif isinstance(_ref_dist_params, dict):
            _param_dic = _ref_dist_params
        else:
            raise TypeError("Wrong input type for _ref_dist_params")  
        _pval_flags_by_id = {_id:[] for _id in getattr(self.cells[0], _id_attr)}
        _pval_flags_by_cell = []
        # loop through cells 
        for _cell_id, _cell in enumerate(self.cells):
            if _verbose and _cell_id % 100 == 0:
                print(f"+++ {_cell_id} has been processed.")
            if not hasattr(_cell, _id_attr) or not hasattr(_cell, _spot_attr) or not hasattr(_cell, 'chrom_coords'):
                _cell._load_from_file('cell_info', _load_attrs=[_id_attr,_channel_attr,_spot_attr, 'chrom_coords'],
                                      _verbose=_verbose)
            if _use_chrom_coords:
                if hasattr(_cell, _id_attr) and hasattr(_cell, _spot_attr) and hasattr(_cell, 'chrom_coords'):
                    # initialize
                    _cell_pval_dics = [{} for _i in range( max(len(getattr(_cell, _spot_attr)[0]), len(getattr(_cell, 'chrom_coords'))) )]
                   
                    for _id, _spot_lst in zip(getattr(_cell, _id_attr), getattr(_cell, _spot_attr)):
                        
                        if isinstance(_spot_lst, list):
                            _cell_id_pval = [[]for _cc in _spot_lst]
                            for _i, _spots in enumerate(_spot_lst):
                                _ints = _spots[:,0]
                                _flags = np.zeros(len(_ints), dtype=np.int)
                                if _ref_dist == 'gaussian':
                                    _z = (_ints - _param_dic[_id]['params'][0]) / _param_dic[_id]['params'][1]
                                    _p_vals = stats.norm.sf(_z) #one-sided 
                                    # save to flags
                                    # for values significantly larger than distribution
                                    _flags[_p_vals < np.min(_pval_th)] = 1
                                    # for values that are not significantly larger than distribution
                                    _flags[_p_vals >= np.max(_pval_th)] = -1 
                                    
                                    # the rest are set to 0 as default
                                _cell_id_pval[_i].append(np.unique(_flags))
                                _cell_pval_dics[_i][_id] = np.unique(_flags)
                        else:
                            _cell_id_pval = [[]
                                             for _cc in getattr(_cell, 'chrom_coords')]
                            # _spot_list is not list, directly append false flags
                            for _i in range(len(getattr(_cell, 'chrom_coords'))):
                                _cell_id_pval[_i].append(np.array([-1],dtype=np.int))
                                _cell_pval_dics[_i][_id] = np.array([-1],dtype=np.int)
                        # append cell_id_pval to _pval_flags_by_id
                        _pval_flags_by_id[_id].append(_cell_id_pval)
                else:
                    print(f"-- no {_id_attr} or {_spot_attr} or chrom_coords are found in fov:{_cell.fov_id}, cell:{_cell.cell_id}, skip")
                # clear bad _cell_pval_dics
                for _j, _dic in enumerate(_cell_pval_dics):
                    if len(_dic) < len(getattr(_cell, _id_attr)):
                        _cell_pval_dics[_j] = {}
                # append _cell_pval to _pval_flags_by_cell
                _pval_flags_by_cell.append(_cell_pval_dics)
                # save attributes
                setattr(_cell, _result_attr, _cell_pval_dics) # this is an empty list if no chrom_coords and spots in this cell
        # save if necessary
        if _save_info:
            self._save_cells_to_files('cell_info', _save_list=[_result_attr], _verbose=_verbose)

        return _pval_flags_by_id, _pval_flags_by_cell
    
    # draw (gene) dependent maps
    def _generate_dependent_maps(self, _flags, _gene_id=None, _name_by_rna=True,
                                 _data_type='unique', _pick_type='EM', _stat_type='median',
                                 _max_loss_prob=0.2, _make_plot=True, _save_plot=True,
                                 _cmap='seismic', _fig_dpi=300, _save_name='distance_map',
                                 _fig_size=4, _gfilt_size=0.75, _plot_limits=[0, 2000],
                                 _verbose=True):
        """Generate dependent distance-maps for given flags
        Inputs:
        Outputs:
        """
        ## check inputs
        if _name_by_rna:
            self._load_rna_info(_verbose=_verbose)
            if _gene_id is not None and _gene_id in getattr(self, 'rna-info_dic'):
                _save_name += '_' + \
                    getattr(self, 'rna-info_dic')[_gene_id]['gene_name']
                if 'DNA_id' in getattr(self, 'rna-info_dic')[_gene_id]:
                    _save_name += f"_r{getattr(self, 'rna-info_dic')[_gene_id]['DNA_id']}"
        # types
        if _data_type not in ['unique', 'decoded']:
            raise ValueError(
                f"Wrong _data_type kwd given, should be unique or decoded, {_data_type} is given!")
        _allowed_pick_types = ['EM', 'dynamic', 'naive']
        if _pick_type not in _allowed_pick_types:
            raise ValueError(
                f"Wrong _pick_type kwd given ({_pick_type}), should be among {_allowed_pick_types}.")
        if _stat_type not in ['median', 'mean', 'contact']:
            raise ValueError(f"Wrong _stat_type({_stat_type}) kwd is given!")
        # check _flags
        if _flags is not None:
            if len(_flags) != len(self.cells):
                raise ValueError(
                    f"_flags should have exactly same length as cells!")

        ## start filtering flags
        _on_flags = []
        _off_flags = []
        _on_count, _off_count = 0, 0
        for _flg_lst, _cell in zip(_flags, self.cells):
            _on_flg_per_cell, _off_flg_per_cell = [], []
            for _chr_flg in _flg_lst:
                if isinstance(_chr_flg, dict) and len(list(_chr_flg.keys())) > 0 :
                    if _gene_id is None or _gene_id not in _chr_flg:
                        print(_cell.fov_id, _cell.cell_id)
                        raise ValueError(
                            f"If you give flags which combines all genes, a gene_id is required.")
                    else:
                        _on_flg_per_cell.append(np.max(_chr_flg[_gene_id]) > 0)
                        _off_flg_per_cell.append(np.max(_chr_flg[_gene_id]) < 0)
                        if (np.max(_chr_flg[_gene_id]) > 0).all():
                            _on_count += 1
                        if (np.max(_chr_flg[_gene_id]) < 0).all():
                            _off_count += 1
                elif isinstance(_chr_flg, np.ndarray) or isinstance(_chr_flg, list):
                    _on_flg_per_cell.append(np.max(_chr_flg) > 0)
                    _off_flg_per_cell.append(np.max(_chr_flg) < 0)
                    if (np.max(_chr_flg) > 0).all():
                        _on_count += 1
                    if (np.max(_chr_flg) < 0).all():
                        _off_count += 1
                else:
                    # append false for both
                    _on_flg_per_cell.append(np.array([0], dtype=np.bool))
                    _off_flg_per_cell.append(np.array([0], dtype=np.bool))
            _on_flags.append(_on_flg_per_cell)
            _off_flags.append(_off_flg_per_cell)

        if _verbose:
            print(f"++ on chromosomes N={_on_count}, off chromosomes N={_off_count}")
            print(f"++ start generating on and off maps")
        ## draw on and off maps
        _on_map, _on_failure_rate = self._calculate_population_map(_data_type=_data_type,
                                                                    _pick_type=_pick_type, _stat_type=_stat_type,
                                                                    _max_loss_prob=_max_loss_prob, _pick_flag=_on_flags, _contact_th=200,
                                                                    _make_plot=_make_plot, _save_plot=_save_plot, _save_name=_save_name+'_on',
                                                                    _cmap=_cmap, _fig_dpi=_fig_dpi,
                                                                    _fig_size=_fig_size, _gfilt_size=_gfilt_size, _plot_limits=_plot_limits,
                                                                    _release_ram=False, _return_all_maps=False, _verbose=_verbose)
        _off_map, _off_failure_rate = self._calculate_population_map(_data_type=_data_type,
                                                                        _pick_type=_pick_type, _stat_type=_stat_type,
                                                                        _max_loss_prob=_max_loss_prob, _pick_flag=_off_flags, _contact_th=200,
                                                                        _make_plot=_make_plot, _save_plot=_save_plot, _save_name=_save_name+'_off',
                                                                        _cmap=_cmap, _fig_dpi=_fig_dpi,
                                                                        _fig_size=_fig_size, _gfilt_size=_gfilt_size, _plot_limits=_plot_limits,
                                                                        _release_ram=False, _return_all_maps=False, _verbose=_verbose)
        ## get ratio map
        _ratio_map = _on_map / _off_map
        # generate_ratio map
        if _make_plot:
            if _verbose:
                print(f"++ generating ratio map for {_save_name}")
            _used_fovs = []
            for _cell in self.cells:
                if _cell.fov_id not in _used_fovs:
                    _used_fovs.append(_cell.fov_id)
            _used_fovs = sorted(_used_fovs)
            # plot 
            plt.figure(figsize=(1.25*_fig_size, _fig_size), dpi=_fig_dpi)
            plt.title(f"{_stat_type} {_save_name} on-off ratio")
            if _stat_type != 'contact':
                _cmap += '_r'
            plt.imshow(_ratio_map, interpolation='nearest', cmap=_cmap, vmin=0.8, vmax=1.2)
            plt.colorbar(ticks=np.arange(0.8, 1.3, 0.1), label=f'{_stat_type} ratio')

            if _save_plot:
                if _verbose:
                    print(f"++ saving {_stat_type} distance map.")
                if len(_used_fovs) > 10:
                    _ratio_filename = os.path.join(self.map_folder, f"{_stat_type}_{_save_name}_ratio_fov{min(_used_fovs)}-{max(_used_fovs)}_{_cmap}.png")
                else:
                    _ratio_filename = os.path.join(self.map_folder, f"{_stat_type}_{_save_name}_ratio_fov{_used_fovs}_{_cmap}.png")
                # add gaussian info if given
                if _gfilt_size:
                    _ratio_filename = _ratio_filename.replace('.png', f'_g{_gfilt_size}.png')
                if not os.path.exists(self.map_folder):
                    os.makedirs(self.map_folder)
                plt.savefig(_ratio_filename, transparent=True)
            if __name__=='__main__':
                plt.show()

        return _on_map, _off_map, _ratio_map

    # run domain calling in batch
    def _batch_domain_calling(self, _data_type='unique', _pick_type='EM', _method='iterative', _num_threads=None,
                            _normalization=True, _domain_size=5, _gfilt_size=0.5,
                            _split_level=1, _num_iter=5, _dist_metric='ks', _cluster_metric='ward',
                            _corr_th=0.6, _dist_th=0.2, _plot_results=False, _dpi=100, _dim=10, _fontsize=18,
                            _save_result_figure=False, save_folder=None, _method_add_to_attr=True,
                            _save_info=True, _overwrite=False, _verbose=True):
        """Function to call domains in batch"""
        ## check inputs
        if _verbose:
            print(f"+ Start {_method} calling {_data_type} domains in batch")
        # check specific attributes and initialize
        _allowed_data_types = ['unique', 'decoded', 'rna-unique']
        if _data_type not in _allowed_data_types:
            raise ValueError(
                f"Wrong input for _data_type:{_data_type}, should be among {_allowed_data_types}")
        # extract attribute name for picked spots
        _allowed_pick_types = ['EM', 'dynamic', 'naive', '']
        if _pick_type not in _allowed_pick_types:
            raise ValueError(
                f"Wrong input for _pick_type:{_pick_type}, should be among {_allowed_pick_types}")
        if _pick_type != '':
            _key_attr = str(_pick_type) + '_picked_' + str(_data_type) + '_spots'
        else:
            _key_attr = 'picked_' + str(_data_type) + '_spots'
        # check method
        _method = str(_method).lower()
        _allowed_methods = ['iterative', 'basic', 'local']
        if _method not in _allowed_methods:
            raise ValueError(
                f"Wrong input for _method:{_method}, should be among {_allowed_methods}")

        # extract result attribute and check if exists
        _result_attr = 'domain_starts'
        if _method_add_to_attr:
            _result_attr = _method + '_' + _result_attr

        # num_threads
        if _num_threads is None:
            _num_threads = getattr(self, 'num_threads')
        # normalization
        if _normalization:
            # temporarily directly load
            _normalization_file = os.path.join(
                self.analysis_folder, 'normalization_matrix.npy')
            if os.path.isfile(_normalization_file):
                _norm_mat = np.load(_normalization_file)
            else:
                _norm_mat = False

        ## loop through cells and acquire spots
        _domain_args = []
        _domain_indices = []
        for _cell in self.cells:
            # try to laod first
            if not hasattr(_cell, _result_attr) or not hasattr(_cell, _key_attr):
                _cell._load_from_file('cell_info', _load_attrs=[
                                    _result_attr, _key_attr], _verbose=False)
            if not hasattr(_cell, _key_attr):
                if _verbose:
                    print(
                        f"+++ skip fov:{_cell.fov_id}, cell:{_cell.cell_id} without {_key_attr}!")
                continue
            # if attribute exists and not overwrite, exit
            elif hasattr(_cell, _result_attr) and \
                    len(getattr(_cell, _result_attr)) == len(getattr(_cell, _key_attr)) \
                    and not _overwrite:
                if _verbose:
                    print(
                        f"+++ skip fov:{_cell.fov_id}, cell:{_cell.cell_id} having {_result_attr}")
            else:
                for _chrom_id, _spots in enumerate(getattr(_cell, _key_attr)):
                    # append args
                    _domain_indices.append(
                        [_cell.fov_id, _cell.cell_id, _chrom_id])
                    _domain_args.append((_spots, _cell.save_folder))

        ## start multi-processing
        # get partially filled function
        if _method == 'iterative':
            _domain_func = partial(domain_tools.calling.iterative_domain_calling,
                                distance_zxy=self.shared_parameters['distance_zxy'], dom_sz=_domain_size,
                                gfilt_size=_gfilt_size, split_level=_split_level,
                                num_iter=_num_iter, normalization_matrix=_norm_mat,
                                domain_dist_metric=_dist_metric,
                                domain_cluster_metric=_cluster_metric,
                                corr_th=_corr_th, dist_th=_dist_th, plot_results=_plot_results,
                                fig_dpi=_dpi, fig_dim=_dim, fig_font_size=_fontsize,
                                save_result_figs=_save_result_figure, verbose=_verbose)
        elif _method == 'basic':
            _domain_func = partial(domain_tools.calling.basic_domain_calling,
                                distance_zxy=self.shared_parameters['distance_zxy'], dom_sz=_domain_size,
                                gfilt_size=_gfilt_size, normalization_matrix=_norm_mat,
                                domain_dist_metric=_dist_metric,
                                domain_cluster_metric=_cluster_metric,
                                corr_th=_corr_th, dist_th=_dist_th, plot_results=_plot_results,
                                fig_dpi=_dpi, fig_dim=_dim, fig_font_size=_fontsize,
                                save_result_figs=_save_result_figure, verbose=_verbose)
        elif _method == 'local':
            _domain_func = partial(domain_tools.calling.local_domain_calling,
                                   distance_zxy=self.shared_parameters['distance_zxy'], dom_sz=_domain_size, 
                                   gfilt_size=_gfilt_size,
                                   cutoff_max=_dist_th, plot_results=_plot_results,
                                   fig_dpi=_dpi, fig_dim=_dim, fig_font_size=_fontsize,
                                   save_result_figs=_save_result_figure, verbose=_verbose)

        with mp.Pool(_num_threads) as _domain_pool:
            if _verbose:
                print(f"++ start multi-processing {_method} calling domains")
                print(
                    f"+++ {len(_domain_args)} jobs submitted to {_num_threads} threads ")

            _domain_starts = _domain_pool.starmap(
                _domain_func, _domain_args, chunksize=1)
            # close multiprocessing
            _domain_pool.close()
            _domain_pool.join()
            _domain_pool.terminate()
        # clear
        if __name__ == '__main__':
            batch_functions.killchild()

        ## save attr
        if _verbose:
            print(f"++ save attribute to cells.")
        for _ind, _starts in zip(_domain_indices, _domain_starts):
            _target_cell = [_cell for _cell in self.cells
                            if _cell.fov_id == _ind[0] and _cell.cell_id == _ind[1]][0]
            if not hasattr(_target_cell, _result_attr):
                setattr(_target_cell, _result_attr, [])
            if len(getattr(_target_cell, _result_attr)) < _ind[2]+1:
                # create slots
                for _cid in range(len(getattr(_target_cell, _result_attr)), _ind[2]+1):
                    getattr(_target_cell, _result_attr).append([])
            getattr(_target_cell, _result_attr)[_ind[2]] = _starts

        # append empty list and save to cell_info
        for _cell in self.cells:
            if not hasattr(_cell, _result_attr):
                setattr(_cell, _result_attr, [])
        if _save_info and len(_domain_args) > 0:
            # save if any new info generated
            self._save_cells_to_files("cell_info", _save_list=[
                                    _result_attr], _verbose=_verbose)
        return _domain_starts


class Cell_Data():
    """
    Class Cell_data:
    data structure of each cell with images in multiple independent color-channels and decoding-groups.
    initialization of cell_data requires:
    """
    # initialize
    def __init__(self, parameters, _load_all_attr=False, _color_filename='Color_Usage',
                 _load_reference_info=False):
        if not isinstance(parameters, dict):
            raise TypeError('wrong input type of parameters, should be a dictionary containing essential info.')
        # necessary parameters
        # data folder (list)
        if isinstance(parameters['data_folder'], list):
            self.data_folder = [str(_fd) for _fd in parameters['data_folder']]
        else:
            self.data_folder = [str(parameters['data_folder'])]
        # experiment_type, default is DNA
        if 'experiment_type' in parameters:
            setattr(self, 'experiment_type', parameters['experiment_type'])
        else:
            setattr(self, 'experiment_type', 'DNA')
        # analysis folder
        if 'analysis_folder'  in parameters:
            self.analysis_folder = str(parameters['analysis_folder'])
        else:
            self.analysis_folder = os.path.join(self.data_folder[0], 'Analysis')
        # experiment_folder
        if 'experiment_folder'  in parameters:
            self.experiment_folder = parameters['experiment_folder']
        else:
            self.experiment_folder = os.path.join(self.data_folder[0], 'Experiment')
        # experiment type
        if 'experiment_type'  in parameters:
            self.experiment_type = parameters['experiment_type']
        else:
            self.experiment_type = 'DNA'
        # extract hybe folders and field-of-view names
        if 'folders' in parameters and 'fovs' in parameters:
            self.folders = parameters['folders']
            self.fovs = parameters['fovs']
        else:
            self.folders = []
            for _fd in self.data_folder:
                _hyb_fds, self.fovs = get_img_info.get_folders(_fd, feature='H', verbose=True)
                self.folders += _hyb_fds
        # fov id and cell id given
        self.fov_id = int(parameters['fov_id'])
        self.cell_id = int(parameters['cell_id'])
        # parameters
        if 'num_threads' in parameters:
            self.num_threads = parameters['num_threads']
        if 'distance_reference' in parameters:
            self.distance_reference = parameters['distance_reference']
        else:
            self.distance_reference = os.path.join(self.analysis_folder, 'distance_ref.npz')

        # segmentation_folder, save_folder, correction_folder,map_folder
        if 'segmentation_folder' in parameters:
            self.segmentation_folder = parameters['segmentation_folder']
        else:
            self.segmentation_folder = self.analysis_folder+os.sep+'segmentation'
        if 'save_folder' in parameters:
            self.save_folder = os.path.join(parameters['save_folder'],
                                            'fov-'+str(self.fov_id),
                                            'cell-'+str(self.cell_id))
        else:
            self.save_folder = os.path.join(self.analysis_folder,
                                            '5x10',
                                            'fov-'+str(self.fov_id),
                                            'cell-'+str(self.cell_id))
        if 'correction_folder' in parameters:
            self.correction_folder = parameters['correction_folder']
        else:
            self.correction_folder = _correction_folder
        if 'drift_folder' in parameters:
            self.drift_folder = parameters['drift_folder']
        else:
            self.drift_folder =  self.analysis_folder+os.sep+'drift'
        if 'map_folder' in parameters:
            self.map_folder = parameters['map_folder']
        else:
            self.map_folder = self.analysis_folder+os.sep+'distmap'
        # other shared_parameters for imaging processing, etc
        if "shared_parameters" in parameters:
            self.shared_parameters = parameters['shared_parameters']
        else:
            self.shared_parameters = {}

        # if loading all remaining attr in parameter
        if _load_all_attr:
            for _key, _value in parameters.items():
                if not hasattr(self, _key):
                    setattr(self, _key, _value)
        
        ## shared parameters
        # distance from pixel to nm:
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

        # load color info
        if not hasattr(self, 'color_dic') or not hasattr(self, 'channels'):
            self._load_color_info(_color_filename=_color_filename)
        if _load_reference_info:
            if getattr(self, 'experiment_type') == 'RNA' and not hasattr(self, 'rna-info_dic'):
                self._load_rna_info()
            elif getattr(self, 'experiment_type') == 'DNA' and not hasattr(self, 'region_dic'):
                self._load_genomic_regions()
        # annotated folders
        if not hasattr(self, 'annotated_folders'):
            self.annotated_folders = []
            for _hyb_fd in self.color_dic:
                _matches = [_fd for _fd in self.folders if _hyb_fd == _fd.split(os.sep)[-1]]
                if len(_matches) == 1:
                    self.annotated_folders.append(_matches[0])
            print(f"-- {len(self.annotated_folders)} folders are found according to color-usage annotation.")


    # allow print info of Cell_List
    def __str__(self):
        if hasattr(self, 'data_folder'):
            print("Cell Data from folder(s):", self.data_folder)
        if hasattr(self, 'analysis_folder'):
            print("\t path for anaylsis result:", self.analysis_folder)
        if hasattr(self, 'fov_id'):
            print("\t from field-of-view:", self.fov_id)
        if hasattr(self, 'cell_id'):
            print("\t with cell_id:", self.cell_id)

        return 'test'

    # allow iteration of Cell_List
    def __iter__(self):
        return self
    def __next__(self):
        return self
    
    ## Load color_usage
    def _load_color_info(self, _color_filename='Color_Usage', _color_format='csv', _save_color_dic=True):
        _color_dic, _use_dapi, _channels = get_img_info.Load_Color_Usage(self.analysis_folder,
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
        _bead_channel = get_img_info.find_bead_channel(_color_dic)
        self.bead_channel_index = _bead_channel
        _dapi_channel = get_img_info.find_dapi_channel(_color_dic)
        self.dapi_channel_index = _dapi_channel

        return _color_dic

    ## load RNA
    def _load_rna_info(self, _filename='RNA_Info', _table_format='csv', 
                       _match_to_genomic_region=True, _verbose=True):
        """Load RNA information"""
        _rna_dic = get_img_info.Load_RNA_Info(self.analysis_folder, filename=_filename,
                                              table_format=_table_format, verbose=_verbose)
        if _match_to_genomic_region:
            _region_dic = self._load_genomic_regions(_verbose=_verbose)
            _rna_dic = get_img_info.match_RNA_to_DNA(_rna_dic, _region_dic)
        # set to attribute
        setattr(self, 'rna-info_dic', _rna_dic)

        return _rna_dic

    ## load Gene
    def _load_gene_info(self, _filename='Gene_Info', _table_format='csv', 
                       _match_to_genomic_region=True, _verbose=True):
        """Load RNA information"""
        _gene_dic = get_img_info.Load_Gene_Info(self.analysis_folder, filename=_filename,
                                              table_format=_table_format, verbose=_verbose)
        if _match_to_genomic_region:
            _region_dic = self._load_genomic_regions(_verbose=_verbose)
            _gene_dic = get_img_info.match_Gene_to_DNA(_gene_dic, _region_dic)
        # set to attribute
        setattr(self, 'gene_dic', _gene_dic)

        return _gene_dic
    
    ## load genomic regions
    def _load_genomic_regions(self, _filename='Region_Positions', _table_format='csv', _verbose=True):
        """Function to load Genomic Positions etc."""
        _region_dic = get_img_info.Load_Region_Positions(self.analysis_folder, filename=_filename,
                                                      table_format=_table_format, verbose=_verbose)
        setattr(self, 'region_dic', _region_dic)
        return _region_dic

    ## Load encoding scheme
    def _load_encoding_scheme(self, _encoding_filename='Encoding_Scheme', _encoding_format='csv', _save_encoding_scheme=True):
        _encoding_scheme, self.hyb_per_group, self.reg_per_group, \
        self.encoding_colors, self.encoding_group_nums \
            = get_img_info.Load_Encoding_Scheme(self.analysis_folder,
                                                   encoding_filename=_encoding_filename,
                                                   encoding_format=_encoding_format,
                                                   return_info=True)
        # need-based encoding scheme saving
        if _save_encoding_scheme:
            self.encoding_scheme = _encoding_scheme

        return _encoding_scheme
    ## Load cell specific info
    def _load_segmentation(self, _min_shape_ratio=0.035, _signal_cap_ratio=0.2, _denoise_window=5,
                           _shrink_percent=15, _max_conv_th=0, _min_boundary_th=0.48,
                           _load_in_ram=True, _save=True, _force=False, _verbose=False):
        # check attributes
        if not hasattr(self, 'channels') or not hasattr(self, 'color_dic'):
            self._load_color_info()
        # find the folder name for dapi
        _select_dapi = False # not select dapi fd yet
        for _fd, _info in self.color_dic.items():
            if len(_info) >= self.dapi_channel_index+1 and _info[self.dapi_channel_index] == 'DAPI':
                _dapi_fd = [_full_fd for _full_fd in self.annotated_folders if os.path.basename(_full_fd) == _fd]
                if len(_dapi_fd) == 1:
                    if _verbose:
                        print(f"-- choose dapi images from folder: {_dapi_fd[0]}.")
                    _dapi_fd = _dapi_fd[0]
                    _select_dapi = True # successfully selected dapi
        if not _select_dapi:
            raise ValueError("No DAPI folder detected in annotated_folders, stop!")
        # do segmentation if necessary, or just load existing segmentation file
        _segmentation_labels = visual_tools.DAPI_convoluted_segmentation(
            os.path.join(_dapi_fd, self.fovs[self.fov_id]), 
            self.channels[self.dapi_channel_index],
            single_im_size=self.shared_parameters['single_im_size'], 
            all_channels=self.channels,
            num_buffer_frames=self.shared_parameters['num_buffer_frames'], 
            num_empty_frames=self.shared_parameters['num_empty_frames'], 
            illumination_correction=self.shared_parameters['corr_illumination'],
            min_shape_ratio=_min_shape_ratio, signal_cap_ratio=_signal_cap_ratio,
            denoise_window=_denoise_window, shrink_percent=_shrink_percent,
            max_conv_th=_max_conv_th, min_boundary_th=_min_boundary_th,
            make_plot=False, return_images=False, 
            save=_save, save_npy=True, save_folder=self.segmentation_folder, force=_force,
            verbose=_verbose
        )
        fov_segmentation_label = _segmentation_labels[0]
        #fov_dapi_im = _dapi_ims[0]
        ## pick corresponding cell
        # exclude special cases
        if not hasattr(self, 'cell_id'):
            raise AttributeError('no cell_id attribute for this cell_data class object!')
        elif self.cell_id+1 not in np.unique(fov_segmentation_label):
            raise ValueError('segmentation label doesnot contain this cell id:', self.cell_id)
        # if everything works, keep segementation_albel and dapi_im for this cell
        else:
            _seg_label = - np.ones(fov_segmentation_label.shape)
            _seg_label[fov_segmentation_label==self.cell_id+1] = 1
            #_dapi_im = visual_tools.crop_cell(fov_dapi_im, _seg_label, drift=None)[0]
            _seg_crop = visual_tools.Extract_crop_from_segmentation(_seg_label)
            if _load_in_ram:
                self.segmentation_label = _seg_label
                #self.dapi_im = _dapi_im
                self.segmentation_crop = _seg_crop
        return _seg_label, _seg_crop
        #return _seg_label, _dapi_im, _seg_crop
    
    def _load_dapi_image(self, _segmentation_kwargs={}, 
                         _save_to_info=True, _force=False, _verbose=True):
        """Function to load dapi_im for Cell_Data"""
        if hasattr(self, 'dapi_im') and not _force:
            _dapi_im = getattr(self, 'dapi_im')
        else:
            if not hasattr(self, 'segmentation_crop'):
                self._load_segmentation(**_segmentation_kwargs)
            for _fd, _info in self.color_dic.items():
                if len(_info) >= self.dapi_channel_index+1 and _info[self.dapi_channel_index] == 'DAPI':
                    _dapi_fd = [_full_fd for _full_fd in self.annotated_folders if os.path.basename(
                        _full_fd) == _fd]
                    if len(_dapi_fd) == 1:
                        if _verbose:
                            print(f"-- choose dapi images from folder: {_dapi_fd[0]}.")
                        _dapi_fd = _dapi_fd[0]
                        _select_dapi = True  # successfully selected dapi
            _dapi_im = corrections.correct_single_image(os.path.join(
                _dapi_fd, self.fovs[self.fov_id]), self.channels[self.dapi_channel_index],
                correction_folder=self.correction_folder,
                crop_limits=self.segmentation_crop,
                single_im_size=self.shared_parameters['single_im_size'], 
                all_channels=self.channels,
                num_buffer_frames=self.shared_parameters['num_buffer_frames'], 
                num_empty_frames=self.shared_parameters['num_empty_frames'], 
                z_shift_corr=self.shared_parameters['corr_Z_shift'], 
                hot_pixel_remove=self.shared_parameters['corr_hot_pixel'], 
                illumination_corr=self.shared_parameters['corr_illumination'], 
                chromatic_corr=self.shared_parameters['corr_chromatic'],
                )
            # save attribute
            setattr(self, 'dapi_im', _dapi_im)
        if _save_to_info:
            self._save_to_file('cell_info', _save_dic={'dapi_im':_dapi_im})
        
        return _dapi_im

    ## check drift info
    def _check_drift(self, _verbose=False):
        """Check whether drift exists and whether all keys required for images exists"""
        if not hasattr(self, 'drift'):
            if _verbose:
                print("-- No drift attribute detected")
            return False
        else:
            # load color_dic as a reference
            if not hasattr(self, 'color_dic'):
                self._load_color_info()
            # check every folder in color_dic whether exists in drift
            for _hyb_fd, _info in self.color_dic.items():
                _drift_query = os.path.join(_hyb_fd, self.fovs[self.fov_id])
                if _drift_query not in self.drift:
                    if _verbose:
                        print(f"-- drift info for {_drift_query} was not found")
                    return False
        # if everything is fine return True
        return True

    ## Load drift (better load, although de novo drift is allowed)
    def _load_drift(self, _sequential_mode=True, _load_annotated_only=True, 
                    _size=500, _ref_id=0, _drift_postfix='_current_cor.pkl', 
                    _num_threads=12, _coord_sel=None, _force=False, _dynamic=True, 
                    _stringent=True, _verbose=True):
        # num-threads
        if hasattr(self, 'num_threads'):
            _num_threads = min(_num_threads, self.num_threads)
        # if drift meets requirements:
        if self._check_drift(_verbose=False) and not _force:
            if _verbose:
                print(f"- drift already exists for cell:{self.cell_id}, skip")
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
            _drift_filename = os.path.join(self.drift_folder, self.fovs[self.fov_id].replace('.dax', _drift_postfix))
            _sequential_drift_filename = os.path.join(self.drift_folder, self.fovs[self.fov_id].replace('.dax', '_sequential'+_drift_postfix))
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
                    _exist = [os.path.join(os.path.basename(_fd),self.fovs[self.fov_id]) for _fd in _folders \
                            if os.path.join(os.path.basename(_fd),self.fovs[self.fov_id]) in _drift]
                    if len(_exist) == len(_folders):
                        if _verbose:
                            print(f"- directly load drift from file:{_dft_filename}")
                        self.drift = _drift
                        return self.drift
            # if non-of existing files fulfills requirements, initialize
            if _verbose:
                print("- start a new drift correction!")

            ## proceed to amend drift correction
            _drift, _failed_count = corrections.Calculate_Bead_Drift(_folders, self.fovs, self.fov_id, 
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
            _exist = [os.path.join(os.path.basename(_fd),self.fovs[self.fov_id]) for _fd in _folders \
                if os.path.join(os.path.basename(_fd),self.fovs[self.fov_id]) in _drift]
            # print if some are failed
            if _failed_count > 0:
                print(f"-- failed number: {_failed_count}"
                )
            if len(_exist) == len(_folders):
                self.drift = _drift
                return self.drift
            else:
                raise ValueError("length of _drift doesn't match _folders!")

    ## crop images given segmentation and images/temp_filenames
    def _crop_images(self, _data_type, _num_threads=12, 
                     _corr_drift=True, _load_in_ram=False, _extend_dim=20, 
                     _save=True, _overwrite=False, _overwrite_cell_info=False, 
                     _verbose=True):
        "Function to crop combo/unique images "
        ## check inputs
        # Num of threads
        if hasattr(self, 'num_threads'):
            _num_threads = max(self.num_threads, _num_threads)
        # load attributes
        if not hasattr(self, 'channels') or not hasattr(self, 'color_dic'):
            self._load_color_info()
        if not hasattr(self, 'segmentation_label'):
            self._load_segmentation()
        if not hasattr(self, 'drift'):
            self._load_drift()
        # check type
        _data_type = _data_type.lower()
        if _data_type not in self.shared_parameters['allowed_data_types']:
            raise ValueError(
                f"Wrong type kwd! {_data_type} is given, {self.shared_parameters['allowed_data_types']} expected.")
        # generate attribute names
        _im_attr = _data_type + '_' + 'ims'
        _id_attr = _data_type + '_' + 'ids'
        _channel_attr = _data_type + '_' + 'channels'
        # generate save_filename:
        _save_filename = os.path.join(self.save_folder, _data_type+'_'+'rounds.npz')
        ### Start crop image
        if _verbose:
            print(f"- Start cropping {_data_type} image")
        _fov_name = self.fovs[self.fov_id]  # name of this field-of-view
        ### unique
        if _data_type in self.shared_parameters['allowed_data_types']:
            # case 1: unique info already loaded in ram
            if hasattr(self, _im_attr) and hasattr(self, _id_attr) and hasattr(self, _channel_attr) \
                and len(getattr(self, _im_attr)) == len(getattr(self, _id_attr)) \
                and len(getattr(self, _id_attr))==len(getattr(self, _channel_attr)):
                _ims = list(getattr(self, _im_attr))
                _ids = list(getattr(self, _id_attr))
                _channels = list(getattr(self, _channel_attr))
                if _verbose:
                    print(f"-- {len(_ims)} {_data_type} image already loaded")
            # case 2: unique info saved, so load from file
            elif os.path.isfile(_save_filename):
                if _verbose:
                    print(f"-- loading {os.path.basename(_save_filename)}", end=', ')
                    _loading_start_time = time.time()
                with np.load(_save_filename) as handle:
                    _ims = list(handle['observation'])
                    _ids = list(handle['ids'])
                    _channels = list(handle['channels'])
                if len(_ims) != len(_ids) or len(_ims) != len(_channels):
                    print(f"length for unique info doesn't match! \n ims:{len(_ims)}, ids:{len(_ids)}, channels:{len(_channels)}\n - start a new crop")
                    _ims, _ids, _channels = [], [], []
                if _verbose:
                    print(f"time:{time.time()-_loading_start_time}")
            # case 3: start from scratch
            else:
                _ims, _ids, _channels = [], [], []

            ## Acquire unique args to be cropped
            _args = []
            # loop through color-dic and find all matched type
            for _hyb_fd, _infos in self.color_dic.items():
                # extract reference name
                _ref_name = os.path.join(_hyb_fd, _fov_name)
                # select channels in this hybe to be corrected:
                _sel_channels = []
                _sel_ids = []
                for _channel, _info in zip(self.channels[:len(_infos)], _infos):
                    # if keyword type matches:
                    if self.shared_parameters['allowed_data_types'][_data_type] in _info:
                        # if this channel requires loading:
                        if _overwrite or int(_info.split(self.shared_parameters['allowed_data_types'][_data_type])[1]) not in _ids:
                            # append _sel_channel
                            _sel_channels.append(_channel)
                            _sel_ids.append(int(_info.split(self.shared_parameters['allowed_data_types'][_data_type])[1]) )
                # do cropping if there are any channels selected:
                if len(_sel_channels) > 0:
                    # match to annotated_folders
                    _matched_folders = [_fd for _fd in self.annotated_folders if _hyb_fd == os.path.basename(_fd)]
                    # if there is uniquely matched folder, append to unique_args
                    if len(_matched_folders) == 1:
                        # get filename for this image
                        _im_filename = os.path.join(_matched_folders[0], _fov_name)
                        # if already exist and going to overwrite, just delete old ones
                        for _uid in _sel_ids:
                            if _overwrite and _uid in _ids:
                                _old_index = _ids.index(_uid)
                                _ids.pop(_old_index)
                                _ims.pop(_old_index)
                                _channels.pop(_old_index)
                                print(f"-- overwrite regions:{_uid}, {len(_ims)} remains")
                        # append ids and channels
                        _ids += _sel_ids
                        _channels += _sel_channels
                        #print('-image length 2:', len(_ims), len(_ids), len(_channels))
                        # add unique_arg
                        _new_arg = (_im_filename, _sel_channels, None, self.segmentation_label,
                                    _extend_dim, self.shared_parameters['single_im_size'], 
                                    self.channels,
                                    self.shared_parameters['num_buffer_frames'], 
                                    self.shared_parameters['num_empty_frames'],
                                    self.drift[_ref_name],
                                    self.correction_folder, 
                                    self.shared_parameters['normalization'], 
                                    self.shared_parameters['corr_bleed'],
                                    self.shared_parameters['corr_Z_shift'],
                                    self.shared_parameters['corr_hot_pixel'], 
                                    self.shared_parameters['corr_illumination'], 
                                    self.shared_parameters['corr_chromatic'],
                                    False, _verbose)
                        _args.append(_new_arg)
                    # if not uniquely-matched, skip
                    else:
                        if len(_matched_folders) > 1:
                            print(f"- ref_name:{_ref_name} has non-unique matches:{_matched_folders}, skip!")
                        if len(_matched_folders) == 0:
                            print(f"- ref_name:{_ref_name} has no corresponding folder, skip.")
                        continue
                # skip the following if already existed & not overwrite
                else:
                    if _verbose:
                        print( f"- all channels in hyb:{_ref_name} don't have new {_data_type} regions, skip!")

            ## Multiprocessing for unique_args
            _start_time = time.time()
            if len(_args) > 0:
                # multi-processing to do cropping
                if _verbose:
                    print(
                        f"-- start cropping {_data_type} for fov:{self.fov_id}, cell:{self.cell_id} with {_num_threads} threads")
                with mp.Pool(_num_threads,
                             maxtasksperchild=int(
                                 np.ceil(len(_args)/_num_threads))+1,
                             ) as _crop_pool:
                    # Multi-proessing!
                    _cropped_results = _crop_pool.starmap(
                        corrections.correct_one_dax, _args, chunksize=1)
                    # close multiprocessing
                    _crop_pool.close()
                    _crop_pool.join()
                    _crop_pool.terminate()
                # clear
                batch_functions.killchild()
                # append (Notice: unique_ids and unique_channels has been appended)
                for _uims in _cropped_results:
                    _ims += _uims
                    #print('image length:', len(_ims), len(_ids), len(_channels))
            # sort
            _tp = [(_id, _im, _ch) for _id, _im, _ch in sorted(
                zip(_ids, _ims, _channels))]
            _sorted_ids = [_t[0] for _t in _tp]
            _sorted_ims = [_t[1] for _t in _tp]
            _sorted_channels = [_t[2] for _t in _tp]
            if _verbose:
                print(f"-- time spent in cropping:{time.time()-_start_time} for {len(_sorted_ims)} images")
            # save unique_ids and unique_channels anyway
            setattr(self, _id_attr, _sorted_ids)
            setattr(self, _channel_attr, _sorted_channels)
            # dict to update unique_ids in cell_info
            _id_info_dict={'unique_ids': _sorted_ids,
                           'unique_channels':_sorted_channels}
            # check if load_in_ram, if true keep images as well
            if _load_in_ram:
                setattr(self, _im_attr, _sorted_ims)
            else:
                _save = True  # not load-in-ram, then save to file
            # save
            if _save and len(_ids) > 0 and len(_args) > 0:
                _dc = {_im_attr: _sorted_ims,
                       _id_attr: _sorted_ids,
                       _channel_attr: _sorted_channels}
                # save to unique
                self._save_to_file(_data_type, _save_dic=_dc,
                                   _overwrite=_overwrite, _verbose=_verbose)
                # update cell_list
                self._save_to_file('cell_info', _save_dic=_id_info_dict, 
                                   _overwrite=_overwrite_cell_info, _verbose=_verbose)

            return _sorted_ims, _sorted_ids, _sorted_channels

    # function to give boolean output of whether a centain type of images are fully generated
    def _check_full_set(self, _data_type, _decoded_flag='diff', _verbose=False):
        """Function to check whether files for a certain type exists"""
        # check inputs
        _data_type = _data_type.lower()
        if _data_type not in self.shared_parameters['allowed_data_types']:
            raise ValueError(
                f"Wrong type kwd! {_data_type} is given, {self.shared_parameters['allowed_data_types']} expected.")
        # start checking 
        if _data_type == 'unique' or _data_type == 'rna-unique':
            # generate attribute names
            _im_attr = _data_type + '_' + 'ims'
            _id_attr = _data_type + '_' + 'ids'
            _channel_attr = _data_type + '_' + 'channels'
            # generate save_filename:
            _save_filename = os.path.join(self.save_folder, _data_type+'_'+'rounds.npz')
            if not os.path.isfile(_save_filename):
                if _verbose:
                    print("-- unique file does not exist")
                return False
            else:
                with np.load(_save_filename) as handle:
                    _keys = handle.keys()
                    if 'observation' not in _keys or 'ids' not in _keys or 'channels' not in _keys:
                        if _verbose:
                            print("-- unique file missing key")
                        return False
                    else:
                        _ids = list(handle['ids'])
                # check unique in color_dic
                if not hasattr(self, 'color_dic') or not hasattr(self,'channels'):
                    self._load_color_info()
                for _hyb_fd, _infos in self.color_dic.items():
                    for _info in _infos:
                        if self.shared_parameters['allowed_data_types'][_data_type] in _info:
                            _uid = int(_info.split(self.shared_parameters['allowed_data_types'][_data_type])[-1])
                            if _uid not in _ids:
                                return False
            # if everything's right, return true
            return True
        if _data_type == 'combo':
            pass
        if _data_type == 'decoded':
            pass
        else:
            if _verbose:
                print("Wrong input data_type!")
            return False

    # saving
    def _save_to_file(self, _data_type='all', _save_dic={}, _save_folder=None, 
                      _unsaved_attrs=None, _clear_old_attrs=False, 
                      _overwrite=False, _verbose=True):
        """Function to save Cell_Data information into storage files."""
        # if not specified attrs not to save, here's a basic list:
        if _unsaved_attrs is None:
            _unsaved_attrs = ['unique_ims', 'combo_groups','bead_ims', 'splitted_ims', 'decoded_ims']
        # create save_folder if needed
        if _save_folder is None:
            _save_folder = self.save_folder
        if not os.path.exists(_save_folder):
            if _verbose:
                print(f"- create save_folder:{_save_folder}\n\t for fov:{self.fov_id}, cell:{self.cell_id}")
            os.makedirs(_save_folder)

        # saving cell_info
        if _data_type=='all' or _data_type =='cell_info':
            
            # get dict for attrs in file
            _savefile = os.path.join(_save_folder, 'cell_info.pkl') # load from saved cell_info
            if _verbose:
                print(f"- saving cell_info to file:{_savefile}")
            if os.path.isfile(_savefile) and not _clear_old_attrs:
                if _verbose:
                    print(f"-- loading existing info from file: {_savefile}")
                _file_dic = pickle.load(open(_savefile, 'rb'))
            else: # no existing file:
                _file_dic = {} # create an empty dic

            # get dict for attrs in RAM
            _existing_attributes = [_attr for _attr in dir(self) if not _attr.startswith('_') and _attr not in _unsaved_attrs]
            
            # compare
            _updated_info = [] # store updated information
            # loop through all existing attributes 
            for _attr in _existing_attributes:
                # screen out special keys and images, which won't be stored in 
                if _attr not in _unsaved_attrs and 'ims' not in _attr and 'distance_map' not in _attr: 
                    # if this attribute is new, or about to overwrite:
                    if _attr not in _file_dic or _overwrite:
                        _file_dic[_attr] = getattr(self, _attr)
                        _updated_info.append(_attr)
            # if specified in save_dic, overwrite with highest priority
            for _k,_v in _save_dic.items():
                _file_dic[_k] = _v # this means overwrite!
                if _k not in _updated_info: 
                    # register in updated_info
                    _updated_info.append(_k)

            if _verbose and len(_updated_info) > 0:
                print(f"-- information updated in cell_info.pkl: {_updated_info}")
            # save info to all
            with open(_savefile, 'wb') as output_handle:
                if _verbose:
                    print(f"- Writing cell data to file: {_savefile}")
                pickle.dump(_file_dic, output_handle)

        # save unique
        if _data_type == 'all' or _data_type in self.shared_parameters['allowed_data_types']:
            # generate attribute names
            _im_attr = _data_type + '_' + 'ims'
            _id_attr = _data_type + '_' + 'ids'
            _channel_attr = _data_type + '_' + 'channels'
            # generate save_filename:
            _save_filename = os.path.join(_save_folder, _data_type+'_'+'rounds.npz')
            _start_time = time.time()
            # check images
            if _im_attr in _save_dic:
                _ims = _save_dic[_im_attr]
            elif hasattr(self, _im_attr):
                _ims = getattr(self, _im_attr)
            else:
                raise ValueError(f'No {_im_attr} information given in fov:{self.fov_id}, cell:{self.cell_id}')
            # check ids
            if _id_attr in _save_dic and len(_save_dic[_id_attr]) == len(_ims):
                _ids = _save_dic[_id_attr]
            elif hasattr(self, _id_attr) and len(getattr(self, _id_attr)) == len(_ims):
                _ids = getattr(self, _id_attr)
            else:
                raise ValueError(f'No {_id_attr} information matched in fov:{self.fov_id}, cell:{self.cell_id}')
            # check channels
            if _channel_attr in _save_dic and len(_save_dic[_channel_attr]) == len(_ims):
                _channels = _save_dic[_channel_attr]
            elif hasattr(self, _channel_attr) and len(getattr(self, _channel_attr)) == len(_ims):
                _channels = getattr(self, _channel_attr)
            else:
                raise ValueError(f'No {_channel_attr} information matched in fov:{self.fov_id}, cell:{self.cell_id}')

            # generate dict to save into npz
            _save_dic = {
                'observation': np.stack(_ims),
                'ids': np.array(_ids),
                'channels': np.array(_channels)
            }
            # save
            if _verbose:
                print(f"-- saving {_data_type} to file: {_save_filename} with {len(_ims)} images" )
            np.savez_compressed(_save_filename, **_save_dic)
            if _verbose:
                print(f"--- time spent in saving:{time.time()-_start_time}")

        # save combo
        if _data_type =='all' or _data_type == 'combo':
            if hasattr(self, 'combo_groups'):
                _combo_groups = self.combo_groups
            elif 'combo_groups' in _save_dic:
                _combo_groups = _save_dic['combo_groups']
            else:
                raise ValueError(f'No combo-groups information given in fov:{self.fov_id}, cell:{self.cell_id}')
            for _group in _combo_groups:
                _combo_savefolder = os.path.join(_save_folder,
                                                'group-'+str(_group.group_id),
                                                'channel-'+str(_group.color)
                                                )
                if not os.path.exists(_combo_savefolder):
                    os.makedirs(_combo_savefolder)
                _combo_savefile = _combo_savefolder+os.sep+'rounds.npz'
                # if file exists and not overwriting, skip
                if os.path.exists(_combo_savefile) and not _overwrite:
                    if _verbose:
                        print("file {:s} already exists, skip.".format(_combo_savefile))
                    continue
                # else, write
                _attrs = [_attr for _attr in dir(_group) if not _attr.startswith('_')]
                _combo_dic = {
                    'observation': np.concatenate([_im[np.newaxis,:] for _im in _group.ims]),
                    'encoding': _group.matrix,
                    'names': np.array(_group.names),
                }
                if hasattr(_group, 'readouts'):
                    _combo_dic['readouts'] = np.array(_group.readouts)
                # append chromosome info if exists
                if hasattr(self, 'chrom_coords'):
                    _combo_dic['chrom_coords'] = self.chrom_coords
                # save
                if _verbose:
                    print("-- saving combo to:", _combo_savefile, end="\t")
                    if 'chrom_coords' in _combo_dic:
                        print("with chrom_coords info.")
                    else:
                        print("")
                np.savez_compressed(_combo_savefile, **_combo_dic)
        
        if _data_type == 'distance_map':
            _start_time = time.time()
            # get save_file:
            _save_filename = os.path.join(_save_folder, 'distance_maps.npz')
            _loaded_dic = {}
            # load saved distmaps if necessary
            if os.path.isfile(_save_filename) and not _clear_old_attrs:
                with np.load(_save_filename) as handle:
                    for _key in handle.iterkeys():
                        _loaded_dic[_key] = handle[_key]
            # get dict for attrs in RAM
            _distance_map_dic = {_attr:getattr(self, _attr) for _attr in dir(self) if not _attr.startswith('_') and 'distance_map' in _attr}
            
            if _verbose:
                print(f"-- saving {_data_type} to file: {_save_filename}")
            # acquire variables from cell_data attributes
            _changed_attrs = []
            for _k, _distmap in _distance_map_dic.items():
                if _k not in _save_dic or _overwrite:
                    _loaded_dic[_k] = _distmap
                    _changed_attrs.append(_k)
            # acquire variables in specified _save_dic
            for _k, _v in _save_dic.items():
                if 'distance_map' in _k:
                    # if the key is distance_map, save with highest priority
                    _loaded_dic[_k] = _v
                    _changed_attrs.append(_k)
            # save
            if len(_changed_attrs):
                np.savez_compressed(_save_filename, **_loaded_dic)
            if _verbose:
                if _changed_attrs:
                    print(f"--- updated attributes: {_changed_attrs}")
                print(f"--- time spent in saving:{time.time()-_start_time}")

    # load attributes or images or distance_maps into Cell_Data class
    def _load_from_file(self, _data_type='all', _save_folder=None, 
                        _decoded_flag=None, 
                        _distmap_data='unique', _distmap_pick='EM',  
                        _load_attrs=[], _exclude_attrs=[],
                        _overwrite=False, _verbose=True):
        """ Function to load cell_data from existing npz and pickle files
        Inputs:
            _data_type: 'all'/'combo'/'unique'/'decoded', string
            _save_folder: where did the files save, None or path string (default: None)
            _decoded_flag
            _distance_map_flag
            _load_attrs: list of additional attributes that want to load to RAM, list of string (default: [])
            _overwrite: whether overwrite existing attributes in class, bool (default: false)
            _verbose: say something!, bool (default: True)
        (everything will be loaded into class attributes, so no return )
        """
        # check input
        _data_type=str(_data_type).lower()
        if _data_type not in ['all', 'cell_info', 'distance_map'] and _data_type not in self.shared_parameters['allowed_data_types']:
            raise ValueError(f"Wrong _data_type kwd ({_data_type}) given!")
        if not _save_folder and not hasattr(self, 'save_folder'):
            raise ValueError('Save folder info not given!')
        elif _save_folder is None:
            _save_folder = self.save_folder

        if _data_type == 'all' or _data_type == 'cell_info':
            _infofile = _save_folder + os.sep + 'cell_info.pkl'
            if os.path.exists(_infofile):
                if _verbose:
                    print(f"-- loading {_data_type} for fov:{self.fov_id}, cell:{self.cell_id}")
                _info_dic = pickle.load(open(_infofile, 'rb'))
                #loading keys from info_dic
                for _key, _value in _info_dic.items():
                    if not hasattr(self, _key) or _overwrite:
                        # if (load_attrs) specified:
                        if len(_load_attrs) > 0 and _key in _load_attrs and _key not in _exclude_attrs:
                            setattr(self, _key, _value)
                        # no load_attr specified
                        elif len(_load_attrs) == 0 and _key not in _exclude_attrs:
                            setattr(self, _key, _value)
            else:
                print(f"No cell-info file found for fov:{self.fov_id}, cell:{self.cell_id}, skip!")

        ## load unique
        if _data_type == 'all' or _data_type in self.shared_parameters['allowed_data_types']:
            # generate attribute names
            _im_attr = _data_type + '_' + 'ims'
            _id_attr = _data_type + '_' + 'ids'
            _channel_attr = _data_type + '_' + 'channels'
            # generate save_filename:
            _save_filename = os.path.join(_save_folder, _data_type+'_'+'rounds.npz')
            _start_time = time.time()
            if not os.path.exists(_save_filename):
                print(f"- savefile {_save_filename} not exists, exit.")
                return False
            if _verbose:
                print("- Loading unique from file:", _save_filename)
            with np.load(_save_filename) as handle:
                _ims = list(handle['observation'])
                _ids = list(handle['ids'])
                _channels = list(handle['channels'])
            # save
            # case1, directly load all
            if not hasattr(self, _im_attr) or not hasattr(self, _id_attr) or not hasattr(self, _channel_attr) \
                    or len(getattr(self, _im_attr)) != len(getattr(self, _id_attr)) \
                    or len(getattr(self, _id_attr)) != len(getattr(self, _channel_attr)):
                setattr(self, _im_attr, _ims)
                setattr(self, _id_attr, _ids)
                setattr(self, _channel_attr, _channels)

                if _verbose:
                    print(f"-- loaded unique images with ids:{_ids}")
            # case 2: only load required ones
            else:  
                for _ct, (_uim, _uid, _channel) in enumerate(zip(_ims, _ids, _channels)):
                    if int(_uid) not in getattr(self, _id_attr):
                        if _verbose:
                            print(f"{_uid},", end=' ')
                            if _ct%10 == -1:
                                print("")
                        getattr(self, _id_attr).append(_uid)
                        getattr(self, _im_attr).append(_uim)
                        getattr(self, _channel_attr).append(_channel)

                    elif int(_uid) in getattr(self, _id_attr) and _overwrite:
                        if _verbose:
                            print(f"overwriting image with unique_id: {_uid}")
                        getattr(self, _im_attr)[list(getattr(self, _id_attr)).index(int(_uid))] = _uim
                if _verbose:
                    print("]")
                    print(f"--- time spent in loading unique images:{time.time()-_start_time}")

        ## load combo
        if _data_type == 'all' or _data_type == 'combo':
            if not hasattr(self, 'combo_groups'):
                self.combo_groups = []
            elif not isinstance(self.combo_groups, list):
                raise TypeError('Wrong datatype for combo_groups attribute for cell data.')

            # load existing combo files
            _raw_combo_fl = "rounds.npz"
            _combo_files = glob.glob(os.path.join(
                _save_folder, "group-*", "channel-*", _raw_combo_fl))
            for _combo_file in _combo_files:
                if _verbose:
                    print("-- loading combo from file:", _combo_file)
                with np.load(_combo_file, mmap_mode='r+') as handle:
                    _ims = list(handle['observation'])
                    _matrix = handle['encoding']
                    _names = handle['names']
                    if 'readout' in handle.keys():
                        _readouts = handle['readouts']
                    if 'chrom_coords' in handle.keys():
                        _chrom_coords = handle['chrom_coords']
                        if not hasattr(self, 'chrom_coords'):
                            self.chrom_coords = _chrom_coords
                _name_info = _combo_file.split(os.sep)
                _fov_id = [int(_i.split('-')[-1])
                           for _i in _name_info if "fov" in _i][0]
                _cell_id = [int(_i.split('-')[-1])
                            for _i in _name_info if "cell" in _i][0]
                _group_id = [int(_i.split('-')[-1])
                             for _i in _name_info if "group" in _i][0]
                _color = [_i.split('-')[-1]
                          for _i in _name_info if "channel" in _i][0]
                # check duplication
                _check_duplicate = [(_g.fov_id == _fov_id) and (_g.cell_id == _cell_id) and (
                    _g.group_id == _group_id) and (_g.color == _color) for _g in self.combo_groups]
                if sum(_check_duplicate) > 0:  # duplicate found:
                    if not _overwrite:
                        if _verbose:
                            print("---", _combo_file.split(_save_folder)
                                  [-1], "already exists in combo_groups, skip")
                        continue
                    else:
                        self.combo_groups.pop(_check_duplicate.index(True))
                # create new group
                if '_readouts' in locals():
                    _group = Encoding_Group(_ims, _names, _matrix, _save_folder,
                                            _fov_id, _cell_id, _color, _group_id, _readouts)
                else:
                    _group = Encoding_Group(_ims, _names, _matrix, _save_folder,
                                            _fov_id, _cell_id, _color, _group_id)
                # append
                self.combo_groups.append(_group)
        ## load decoded
        if _data_type == 'all' or _data_type == 'decoded':
            if _data_type == 'decoded' and not _decoded_flag:
                raise ValueError("Kwd _decoded_flag not given, exit!")
            elif not _decoded_flag:
                print("Kwd _decoded_flag not given, skip this step.")
            else:
                # check combo_groups
                # look for decoded _results
                if not hasattr(self, 'combo_groups'):
                    _temp_flag = True
                    if _verbose:
                        print(f"No combo groups loaded in fov:{self.fov_id}, cell:{self.cell_id}, start loading combo!")
                    self._load_from_file('combo', _overwrite=_overwrite, _verbose=_verbose)
                else:
                    _temp_flag = False
                # scan existing combo files
                _raw_decoded_fl = "regions.npz"
                _decoded_files = glob.glob(os.path.join(_save_folder, "group-*", "channel-*", _decoded_flag, _raw_decoded_fl))
                # intialize list to store images
                _decoded_ims, _decoded_ids = [],[]
                # loop through files and
                for _decoded_file in _decoded_files:
                    if _verbose:
                        print("-- loading decoded result from file:", _decoded_file)
                    with np.load(_decoded_file) as handle:
                        _ims = handle['observation']
                        _ims = _ims.swapaxes(0,3).swapaxes(1,2)
                    _name_info = _decoded_file.split(os.sep)
                    _fov_id = [int(_i.split('-')[-1]) for _i in _name_info if "fov" in _i][0]
                    _cell_id = [int(_i.split('-')[-1]) for _i in _name_info if "cell" in _i][0]
                    _group_id = [int(_i.split('-')[-1]) for _i in _name_info if "group" in _i][0]
                    _color = [_i.split('-')[-1] for _i in _name_info if "channel" in _i][0]
                    # check duplication
                    _matches = [(_g.fov_id==_fov_id) and (_g.cell_id==_cell_id) and (_g.group_id==_group_id) and (_g.color==_color) for _g in self.combo_groups]
                    if sum(_matches) == 1: #duplicate found:
                        if _verbose:
                            print(f"--- decoded result matched for group:{_group_id}, color:{_color}")
                        _matched_group = self.combo_groups[_matches.index(True)]
                        _matrix = _matched_group.matrix
                        _n_hyb, _n_reg = np.shape(_matrix)
                        _ims = _ims[:_n_reg]
                        _ids = [np.unique(_matrix[:,_col_id])[-1] for _col_id in range(_n_reg)]
                        # filter by reg name
                        _kept_ims = [_im for _im,_id in zip(_ims, _ids) if _id >=0]
                        _kept_ids = [_id for _im,_id in zip(_ims, _ids) if _id >=0]
                        # append
                        _decoded_ids += _kept_ids
                        _decoded_ims += _kept_ims
                        print("--- kept ids:",_kept_ids)
                # check with combo groups
                for _group in self.combo_groups:
                    _ids = [np.unique(_group.matrix[:,_col_id])[-1] for _col_id in range(_group.matrix.shape[1])]
                    for _id in _ids:
                        if _id not in _decoded_ids and _id > 0:
                            if _verbose:
                                print(f"--- filling up not decoded region:{_id}")
                            _decoded_ids.append(_id)
                            _decoded_ims.append(None)
                # sort
                self.decoded_ims = [_im for _id,_im in sorted(zip(_decoded_ids, _decoded_ims))]
                self.decoded_ids = [_id for _id in sorted(_decoded_ids)]
                if _temp_flag:
                    delattr(self, 'combo_groups')
        ## load distance_maps
        if _data_type == 'all' or _data_type == 'distance_map':
            _start_time = time.time()
            # get save_file:
            _save_filename = os.path.join(_save_folder, 'distance_maps.npz')
            if not os.path.exists(_save_filename):
                if _verbose:
                    print(f"--- distmap_savefile:{_save_filename} not exist, exit!")
                return None
            else:
                # get loading attr
                if _distmap_data is not None:
                    if not isinstance(_distmap_data, str):
                        raise ValueError("_distmap_data should be a string!")
                if _distmap_pick is not None:
                    if not isinstance(_distmap_pick, str):
                        raise ValueError("_distmap_pick should be a string!")
                if _distmap_data is None or _distmap_pick is None:
                    _load_attr = 'distance_map'
                else:
                    _load_attr = _distmap_pick + '_' + _distmap_data + '_' + 'distance_map'
                # open npz and load
                with np.load(_save_filename) as handle:
                    if _load_attr in handle.keys():
                        if _verbose:
                            print(f"-- loading {_load_attr} for fov:{self.fov_id}, cell:{self.cell_id}")
                        _distmap = handle[_load_attr]
                        # append to cell_data
                        setattr(self, _load_attr, _distmap)
                        return _distmap 
                    elif _distmap_data is None or _distmap_pick is None:
                        if _verbose:
                            print(f"-- loading all {_load_attr} for fov:{self.fov_id}, cell:{self.cell_id}")
                        # initialize a list to store saved distmaps                        
                        _all_maps = []
                        for _distmap_attr in handle.iterkeys():
                            # load to ram if specified
                            if _load_attr in _distmap_attr:
                                setattr(self, _distmap_attr, handle[_distmap_attr])
                            # append to all_maps
                            _all_maps.append(handle[_distmap_attr])
                        return _all_maps
                    else:
                        print(f"--- {_load_attr} doesn't exist in saved file, exit!")
                        return None
            
    # Generate pooled image representing chromosomes
    def _generate_chromosome_image(self, _source='unique', _max_count=90, _verbose=False):
        """Generate chromosome from existing combo / unique images"""
        _source = _source.lower()
        if _source != 'combo' and _source != 'unique' and _source != 'rna-unique':
            raise ValueError('wrong source key given, should be combo or unique. ')
        if _source == 'combo':
            if not hasattr(self, 'combo_groups'):
                _temp_flag = True # this means the combo groups images are temporarily loaded
                print('-- cell_data doesnot have combo images, trying to load now.')
                self._load_from_file('combo', _verbose=False)
            else:
                _temp_flag = False
            # sum up existing Images
            _image_count = 0
            _chrom_im = np.zeros(np.shape(self.combo_groups[0].ims[0]))
            for _group in self.combo_groups:
                _chrom_im += sum(_group.ims)
                _image_count += len(_group.ims)
                if _max_count > 0 and _image_count > _max_count:
                    break
            _chrom_im = _chrom_im / _image_count

        elif _source == 'unique' or _source == 'rna-unique':
            # generate attribute names
            _im_attr = _source + '_' + 'ims'
            _id_attr = _source + '_' + 'ids'
            # generate save_filename:
            _save_filename = os.path.join(self.save_folder, _source+'_'+'rounds.npz')
            if not hasattr(self, _im_attr) or not hasattr(self, _id_attr):
                _temp_flag = True # this means the unique images are temporarily loaded
                print(f'-- cell:{self.cell_id} in fov:{self.fov_id} doesnot have {_im_attr}, trying to load now.')
                self._load_from_file(_source, _verbose=False)
            else:
                _temp_flag = False
            # sum up existing Images
            _picking_freq = int(np.ceil(len(getattr(self, _im_attr))/_max_count))
            _selected_ims = np.array(getattr(self, _im_attr))[::_picking_freq]
            _chrom_im = np.mean(np.stack(_selected_ims), axis=0)

        # final correction
        _chrom_im = corrections.Z_Shift_Correction(_chrom_im)
        _chrom_im = corrections.Remove_Hot_Pixels(_chrom_im)
        self.chrom_im = _chrom_im
        if _temp_flag: # if temp loaded, release
            if _source == 'combo':
                delattr(self, 'combo_groups')
            elif _source == 'unique' or _source == 'rna-unique':
                delattr(self, _im_attr)
        return _chrom_im

    # Identify chromosome(generated by _generate_chromosome_image)
    def _identify_chromosomes(self, _gaussian_size=2, _cap_percentile=1, _seed_dim=3,
                              _th_percentile=99.5, _min_obj_size=125, _verbose=True):
        """Function to identify chromsome automatically first"""
        if not hasattr(self, 'chrom_im'):
            self._generate_chromosome_image()
        _chrom_im = np.zeros(np.shape(self.chrom_im), dtype=_image_dtype) + self.chrom_im
        if not hasattr(self,'chrom_coords'):
            # gaussian filter
            if _gaussian_size:
                _chrom_im = ndimage.filters.gaussian_filter(_chrom_im, _gaussian_size)
            # normalization
            _limit = stats.scoreatpercentile(_chrom_im, (_cap_percentile, 100.-_cap_percentile)).astype(np.float)
            _chrom_im = (_chrom_im-np.min(_limit))/(np.max(_limit)-np.min(_limit))
            # max filter - min filter
            _max_ft = ndimage.filters.maximum_filter(_chrom_im, _seed_dim)
            _min_ft = ndimage.filters.minimum_filter(_chrom_im, _seed_dim)
            _seed_im = 2*_max_ft - _min_ft
            # binarilize
            _binary_im = (_seed_im > stats.scoreatpercentile(_seed_im, _th_percentile))
            # dialation and erosion
            _binary_im = ndimage.binary_dilation(_binary_im, morphology.ball(1))
            _binary_im = ndimage.binary_erosion(_binary_im, morphology.ball(0))
            _binary_im = ndimage.binary_fill_holes(_binary_im, structure=morphology.ball(2))
            # find objects
            _open_objects = morphology.opening(_binary_im, morphology.ball(0))
            _close_objects = morphology.closing(_open_objects, morphology.ball(1))
            _label, _num = ndimage.label(_close_objects)
            _label[_label==0] = -1
            # segmentation
            _seg_label = random_walker(_chrom_im, _label, beta=100, mode='bf')
            # keep object
            _kept_label = -1 * np.ones(_seg_label.shape, dtype=np.int)
            _sizes = [np.sum(_seg_label==_j+1) for _j in range(np.max(_seg_label))]
            # re-label
            _label_ct = 1
            for _i, _size in enumerate(_sizes):
                if _size > _min_obj_size: # then save this label
                    _kept_label[_seg_label == _i+1] = _label_ct
                    _label_ct += 1
            _chrom_coords = [ndimage.measurements.center_of_mass(_kept_label==_j+1) for _j in range(np.max(_kept_label))]
            # store
            self.chrom_segmentation = _kept_label
            self.chrom_coords = _chrom_coords
            return _chrom_coords
        else:
            return self.chrom_coords  
    # manually adjust chromosome pick
    def _pick_chromosome_manual(self, _save_folder=None, _save_fl='chrom_coord.pkl'):
        if not _save_folder:
            if hasattr(self, 'save_folder'):
                _save_folder = self.save_folder
            else:
                raise ValueError('save_folder not given in keys and attributes.')

        _chrom_savefile = os.path.join(_save_folder, _save_fl.replace('.pkl', '_'+str(self.fov_id)+'_'+str(self.cell_id)+'.pkl'))
        if not hasattr(self, 'chrom_coords'):
            raise ValueError("chromosome coordinates doesnot exist in attributes.")
        _coord_dic = {'coords': [np.flipud(_coord) for _coord in self.chrom_coords],
                      'class_ids': list(np.zeros(len(self.chrom_coords),dtype=np.uint8)),
                      'pfits':{},
                      'dec_text':{},
                      }
        #pickle.dump(_coord_dic, open(_chrom_savefile, 'wb'))
        _viewer = visual_tools.imshow_mark_3d_v2([self.chrom_im], image_names=['chromosome'],
                                                 save_file=_chrom_savefile, given_dic=_coord_dic)
        return _viewer
    # update manually picked chromosome info
    def _update_chromosome_from_file(self, _save_folder=None, _save_fl='chrom_coord.pkl', 
                        _save=True, _force_save_combo=False, _force=False, _verbose=True):
        if not _save_folder:
            if hasattr(self, 'save_folder'):
                _save_folder = self.save_folder
            else:
                raise ValueError('save_folder not given in keys and attributes.')
        _chrom_savefile = os.path.join(_save_folder, _save_fl.replace('.pkl', '_'+str(self.fov_id)+'_'+str(self.cell_id)+'.pkl'))
        _coord_dic = pickle.load(open(_chrom_savefile, 'rb'))
        _chrom_coords = [np.flipud(_coord) for _coord in _coord_dic['coords']]
        if _verbose:
            print(f"-- {len(_chrom_coords)} loaded")
        self.chrom_coords = _chrom_coords
        if _save:
            self._save_to_file('cell_info', _save_dic={'chrom_coords':_chrom_coords})
            if hasattr(self,'combo_groups') or _force_save_combo:
                self._save_to_file('combo', _overwrite=_force)
        return _chrom_coords

    # calculate background levels for images
    def _calculate_background(self, _data_type='unique', _function_type='median', 
                              _num_per_channel=20, _verbose=False):
        """Function to get background levels for channels having signal"""
        ## first check Inputs
        # data_type
        _allowed_type_dic = {'unique':'unique_ims', 
                            'combo':'combo_groups',
                            }
        _type_marker_dic = {'rna-unique':'r',
                            'unique':'u', 
                            'combo':'c'}
        _data_type = _data_type.lower()
        if _data_type not in _allowed_type_dic or _data_type not in _type_marker_dic:
            raise KeyError(f"Wrong input key for _data_type:{_data_type}")
        # function_type
        _allowed_function_types = ['median', 'mean']
        _function_type = _function_type.lower()
        if _function_type not in _allowed_function_types:
            raise KeyError(f"Wrong input key for _function_type:{_function_type}")
        # load from color_dic
        if not hasattr(self, 'channels') or not hasattr(self, 'color_dic'):
            self._load_color_info()
        # initialize color_indices dict
        _color_indices = {_c:[] for _c in self.channels}
        for _hyb_fd, _info_lst in self.color_dic.items():
            for _info, _c in zip(_info_lst, list(_color_indices.keys())[:len(_info_lst)]):
                if _type_marker_dic[_data_type] in _info:
                    _uid = int(_info.split(_type_marker_dic[_data_type])[1])
                    _id_lst_name = _allowed_type_dic[_data_type].replace('ims', 'ids')
                    if _uid in getattr(self, _id_lst_name) and len(_color_indices[_c]) < _num_per_channel:
                        _color_indices[_c].append(getattr(self, _id_lst_name).index(_uid))
        # calculate background
        if _verbose:
            print(f"-- calculating background for {[_c for _c,_l in _color_indices.items() if len(_l)>0]}")
        _background_dic = {}
        _ims = np.array(getattr(self, _allowed_type_dic[_data_type]))
        for _c, _ind_lst in _color_indices.items():
            if len(_ind_lst) == 0:
                continue
            else:
                _ind_lst = np.array(_ind_lst, dtype=np.int)
                _ims_in_channel = _ims[_ind_lst]
                if _function_type == 'median':
                    _backgrounds = np.nanmedian(_ims_in_channel, axis=0)
                elif _function_type == 'mean':
                    _backgrounds = np.nanmean(_ims_in_channel, axis=0)
                
                _background_dic[_c] = np.median(_backgrounds)
        return _background_dic

    # multi-gaussian fitting to pick spots
    def _multi_fitting_for_chromosome(self, _data_type='unique', _decoded_flag='diff', 
                       _normalization=True,  _use_chrom_coords=True, _num_threads=12,
                       _gfilt_size=0.75, _background_gfilt_size=10, _max_filt_size=3,
                       _seed_by_per=True, _th_seed_percentile=90,
                       _max_seed_count=10, _min_seed_count=3,
                       _fit_radius=5, _fit_window=40, 
                       _expect_weight=1000, _min_height=100, _max_iter=10, _th_to_end=1e-6,
                       _check_fitting=True, _save=True, _overwrite=False, _verbose=True):
        """Function for multi-fitting for chromosomes in cell_data"""
        # first check Inputs
        _data_type = _data_type.lower()
        if _data_type not in self.shared_parameters['allowed_data_types']:
            raise KeyError(f"Wrong input key for _data_type:{_data_type}")
        # generate attribute names
        _im_attr = _data_type + '_' + 'ims'
        _id_attr = _data_type + '_' + 'ids'
        _channel_attr = _data_type + '_' + 'channels'
        # generate save_filename:
        _save_filename = os.path.join(self.save_folder, _data_type+'_'+'rounds.npz')
        # generate save_attr
        _spot_attr = _data_type + '_' + 'spots'
        # num threads
        if hasattr(self, 'num_threads'):
            _num_threads = max(self.num_threads, _num_threads)
        if _use_chrom_coords:
            if not hasattr(self, 'chrom_coords'):
                self._load_from_file('cell_info')
                if not hasattr(self, 'chrom_coords'):
                    raise AttributeError("No chrom-coords info found in cell-data and saved cell_info.")
        if _verbose:
            print(f"- Start multi-fitting for {_data_type} images")
        # check specific attributes
        if _data_type in self.shared_parameters['allowed_data_types']:
            # check attributes
            if not hasattr(self, _im_attr) or not hasattr(self, _id_attr):
                _temp_flag = True # this means the unique images are temporarily loaded
                print(f"-- no {_im_attr} info loaded to this cell, try loading:")
                self._load_from_file(_data_type, _overwrite=False, _verbose=_verbose)
            else:
                _temp_flag = False # not temporarily loaded
            # in both cases set ims and ids
            _ims = getattr(self, _im_attr)
            _ids = getattr(self, _id_attr)


        # first check wether requires do it denovo
        if hasattr(self, _spot_attr) and not _overwrite \
            and len(getattr(self, _spot_attr)) == len(_ims) \
            and (not _use_chrom_coords or len(getattr(self, 'chrom_coords')) == len(getattr(self, _spot_attr)[0]) ):
            if _verbose:
                print(f"-- {_spot_attr} already exist for fov:{self.fov_id}, cell:{self.cell_id}.")
            return getattr(self, _spot_attr)
        else:
            ## Do the multi-fitting
            if _data_type in self.shared_parameters['allowed_data_types']:
                _seeding_args = (_max_seed_count, _fit_window, _gfilt_size, _background_gfilt_size, _max_filt_size, _seed_by_per, _th_seed_percentile, 150, True, 10, _min_seed_count, 0, False)
                _fitting_args = (_fit_radius, 1, 2.5, _max_iter, 0.1, self.shared_parameters['sigma_zxy'], _expect_weight)
                    # merge arguments
            if _use_chrom_coords:
                _args = [(_im, _id, self.chrom_coords, _seeding_args, _fitting_args, 
                        _check_fitting, _normalization, _verbose) for _im, _id in zip(_ims, _ids)]
            else:
                _args = [(_im, _id, [None], _seeding_args, _fitting_args, 
                        _check_fitting, _normalization, _verbose) for _im, _id in zip(_ims, _ids)]
            # multi-processing for multi-Fitting
            if _verbose:
                print(f"++ start fitting {_data_type} for fov:{self.fov_id}, cell:{self.cell_id} with {_num_threads} threads")
            _start_time = time.time()
            _fitting_pool = mp.Pool(_num_threads)
            _spots = _fitting_pool.starmap(_fit_single_image, _args, chunksize=1)
            _fitting_pool.close()
            _fitting_pool.join()
            _fitting_pool.terminate()
            # release
            _ims = None
            if _temp_flag:
                if _data_type == 'unique' or _data_type  == 'rna-unique':
                    delattr(self, _im_attr)
                elif _data_type == 'decoded':
                    delattr(self, 'decoded_ims')
            if _verbose:
                print(f"++ total time in fitting {_data_type}: {time.time()-_start_time}")
            ## return and save
            setattr(self, _spot_attr, _spots)
            if _save:
                self._save_to_file('cell_info',_save_dic={_spot_attr: getattr(self, _spot_attr),
                                                            _id_attr: _ids}, _verbose=_verbose)

        return _spots

    # an integrated function to pick spots
    def _pick_spots(self, _data_type='unique', _pick_type='EM', 
                    _use_chrom_coords=True,
                    _sel_ids=None, _num_iters=10, _terminate_th=0.003, 
                    _intensity_th=0., _hard_intensity_th=True, _spot_num_th=100,
                    _ref_spot_list=None, _ref_spot_ids=None, _ref_pick_type='EM',
                    _ignore_ids=False, _ref_dist_metric='median', _score_metric='linear',
                    _local_size=5, _w_ctdist=2, _w_lcdist=1, _w_int=1, _w_nbdist=2,
                    _distance_limits=[0,np.inf], _ignore_nan=True,  
                    _nan_mask=0., _inf_mask=-1000., _chrom_share_spots=False,
                    _check_spots=True, _check_th=-3., 
                    _check_percentile=2.5, _hard_dist_th=6000, 
                    _save_inter_plot=False, _save_to_attr=True, _save_to_info=True,
                    _return_indices=False, _overwrite=False, _verbose=True):
        """Function to pick spots from all candidate spots within Cell_Data
        There are three versions allowed for now:
            - naive: pick spots solely based on intensity
            - dynamic: pick spots based on neighboring distance + scores
            - EM: Expectation-Maximization iterations to get optimum spot picking
        -----------------------------------------------------------------------------
        Inputs:
            _pick_type: method for picking spots, str ('EM', 'dynamic' or 'naive')
            _data_type: data type of spots to be picked, str ('unique', 'decoded' etc)
            _use_chrom_coords: whether use chrom_coords in cell_data, bool (default: True)
            _sel_ids: selected ids to pick spots, list/array of ints (default: None, which means all possible spots)
            _num_iters: number of iterations for EM picking spots, int (default: 10)
            _terminate_th: threshold of thermination for spot picking change in EM spots, float (default: 0.003)
            _intensity_th: intensity threshold for fitted spots, float (default: 0., no threshold)
            _hard_intensity_th: whether apply hard intensity threshold, bool (default: True)
            _spot_num_th: min number of spots as reference, int (default: 100)
            _ref_spot_list: list of reference spots, could be data_type or direct spots
            _ref_spot_ids: list/array of reference ids, could be data_type or direct spots
            _ref_pick_type: pick type for reference spots, str
            _ref_dist_metric: metric for reference distance metric, str {'median'|'cdf'|'rg'} 
            _score_metric: 
            _local_size
            _w_ctdist, 
            _w_lcdist, 
            _w_int, 
            _w_nbdist,
            _distance_limits
            _ignore_nan: whether ignore nan spots, bool (default:True)
            _nan_mask
            _inf_mask
            _chrom_share_spots
            _check_spots: whether do statistical check for spots, bool (default: True)
            _check_th: threshold of spot_checking, float (default: -3.)
            _check_percentile: another percentile threshold that may apply to data, float (default: 2.5.)
            _save_inter_plot: whether save intermediate plots, bool (default: False)
                * only useful in EM
            _save_to_attr: whether save picked spots into attributes, bool (default: True)
            _save_to_info: whether save picked spots into cell_info, bool (default: True)
            _return_indices: whether return indices for selected spots, bool (default: False)
            _overwrite: whether overwrite existing info, bool (default: False)
            _verbose: say something!, bool (default: True)
        Outputs:
            _pick_spot_list: list of picked spots, list if use chrom_coord, otherwise array
            (optional)
            _pick_ind_list: list of picked indices, list if use chrom_coord, otherwise array
        """
        ## check inputs
        # pick type
        _allowed_pick_types = ['EM', 'naive', 'dynamic']
        if _pick_type not in _allowed_pick_types:
            raise ValueError(f"Wrong input for _pick_type:{_pick_type}, should be among {_allowed_pick_types}")
        if _ref_pick_type not in _allowed_pick_types:
            raise ValueError(f"Wrong input for _ref_pick_type:{_ref_pick_type}, should be among {_allowed_pick_types}")
        # data type
        _data_type = _data_type.lower()
        if _data_type not in self.shared_parameters['allowed_data_types']:
            raise ValueError(
                f"Wrong input for {_data_type}, should be among {self.shared_parameters['allowed_data_types']}")
        
        # generate attribute names
        _im_attr = _data_type + '_' + 'ims'
        _id_attr = _data_type + '_' + 'ids'
        _spot_attr = _data_type + '_' + 'spots'
        # generate save attribute
        _picked_attr = str(_pick_type) + '_'+ 'picked' + '_' + _spot_attr

        ## ref spots and ids
        # _ref_spot_list is data_type
        if isinstance(_ref_spot_list, str):
            if _ref_spot_list not in self.shared_parameters['allowed_data_types']:
                raise ValueError(f"if _ref_spot_list is specified as string ({_ref_spot_list}), \
                                                                                                  it should be among {self.shared_parameters['allowed_data_types']}")
            # decided attributes
            _ref_picked_attr = str(_ref_pick_type) + '_'+ 'picked' + '_' + _ref_spot_list + '_' + 'spots'
            _ref_id_attr = _ref_spot_list + '_' + 'ids'
            # load if not exist now
            if not hasattr(self, _ref_picked_attr):
                self._load_from_file('cell_info', _load_attrs=[_ref_picked_attr])
            if not hasattr(self, _ref_id_attr):
                self._load_from_file('cell_info', _load_attrs=[_ref_id_attr])
            # get attributes
            _ref_spot_list = getattr(self, _ref_picked_attr, None)
            _ref_ids = getattr(self, _ref_id_attr, None)
        # _ref_spot_list is directly list
        elif isinstance(_ref_spot_list, list):
            if _ref_spot_ids is None:
                _ref_ids = None
            elif isinstance(_ref_spot_ids, list) or isinstance(_ref_spot_ids, np.ndarray):
                if len(_ref_spot_ids) == _ref_spot_list[0]:
                    _ref_ids = _ref_spot_ids
                else:
                    raise IndexError(f"length of _ref_spot_ids{_ref_spot_ids} not matched with length of _ref_spot_list[0]{_ref_spot_list[0]}")
            else:
                raise TypeError(f"Wrong input type for _ref_spot_ids:{type(_ref_spot_ids)}")
        # _ref_spot_list is None
        elif _ref_spot_list is None:
            _ref_ids = None
        else:
            raise TypeError(f"Wrong input type for _ref_spot_list:{_ref_spot_list}")

        # get cand_spots
        if _verbose:
            print(f"- Start {_pick_type} picking {_data_type} spots, fov:{self.fov_id}, cell:{self.cell_id}.")
        # spot_attr
        if not hasattr(self, _spot_attr):
            self._load_from_file('cell_info', _load_attrs=[_spot_attr])
        _all_spots = getattr(self, _spot_attr)
        if _all_spots is None:
            raise ValueError(f"fov:{self.fov_id}, cell:{self.cell_id} doesnt have attribute: {_spot_attr}")
        # id_attr
        if not hasattr(self, _id_attr):
            self._load_from_file('cell_info', _load_attrs=[_id_attr])
        _ids = getattr(self, _id_attr)
        if _ids is None:
            print( _id_attr, hasattr(self, _id_attr) )
            raise ValueError(f"fov:{self.fov_id}, cell:{self.cell_id} doesnt have attribute: {_id_attr}")

        if _sel_ids is not None:
            # order matters, change spot candidates first, then change ids
            _all_spots = [_pts for _i, _pts in zip(_ids, _all_spots) if _i in _sel_ids]
            _ids = [_i for _i in _ids if _i in _sel_ids]

        if _use_chrom_coords and not hasattr(self, 'chrom_coords'):
            self._load_from_file('cell_info', _load_attrs=['chrom_coords'])
            if getattr(self, 'chrom_coords', None) is None:
                raise AttributeError(f"chrom_coords not found in fov:{self.fov_id}, cell:{self.cell_id}.")

        # special ids for RNA, corresponding to DNA regions
        if _data_type == 'rna-unique':
            _gids = []
            if hasattr(self, 'rna-info_dic'):
                _rna_dic = getattr(self, 'rna-info_dic')
            else:
                _rna_dic = self._load_rna_info(_verbose=False)
            for _id in _ids:
                _info = _rna_dic[_id]
                if 'DNA_id' in _info:
                    _gids.append(int(_info['DNA_id']))
                else:
                    _gids.append(-1 * _id)
            _ids = _gids
        if _data_type == 'gene':
            _gids = []
            if hasattr(self, 'gene_dic'):
                _rna_dic = getattr(self, 'gene_dic')
            else:
                _rna_dic = self._load_gene_info(_verbose=False)
            for _id in _ids:
                _info = _rna_dic[_id]
                if 'DNA_id' in _info:
                    _gids.append(int(_info['DNA_id']))
                else:
                    _gids.append(-1 * _id)
            _ids = _gids
        # if ignore ids, directly convert into continuous ids
        if _ignore_ids:
            print("-- ignoring ids")
            _ids = np.arange(len(_all_spots))

        # if not overwrite:
        if not _overwrite:
            if not hasattr(self, _picked_attr):
                self._load_from_file('cell_info', _load_attrs=[_picked_attr])
            if _use_chrom_coords:
                pass
            if hasattr(self, _picked_attr) and len(_ids)==len(getattr(self, _picked_attr)[0]):
                _picked_spot_list = getattr(self, _picked_attr)
                # return if 
                if _verbose:
                    print(f"-- not overwriting {_picked_attr} for fov:{self.fov_id}, cell:{self.cell_id}")
                if not _return_indices:
                    return _picked_spot_list
                else:
                    return _picked_spot_list, []
        
        # check chrom_coords
        if _use_chrom_coords and not hasattr(self, 'chrom_coords'):
            self._load_from_file('cell_info', _load_attrs=['chrom_coords'])
            if not hasattr(self, 'chrom_coords'):
                raise AttributeError(
                    f"No chrom-coords info found for fov:{self.fov_id}, cell:{self.cell_id} in cell-data and saved cell_info.")
            
            # temporary limit
            if len(getattr(self, 'chrom_coords')) > 6:
                if not _return_indices:
                    return []
                else:
                    return [], []

            # check if length of chrom_coords matches all_spots
            if len(_all_spots) > 0 and len(getattr(self, 'chrom_coords')) != len(_all_spots[0]):
                raise ValueError(f"Length of chrom_coords and all_spots for fov:{self.fov_id}, cell:{self.cell_id} doesn't match!")
        elif not _use_chrom_coords and len(_all_spots) != len(_ids):
            if hasattr(self, 'chrom_coords') and len(_all_spots) == len(getattr(self, 'chrom_coords')):
                print("Probably wrong input for _use_chrom_coords?, switch to True")
                _use_chrom_coords = True
            else:
                raise ValueError("Length of ids and candidate_spots doesn't match!")
        if _use_chrom_coords:
            _cand_spot_list = []
            for _chrom_id in range(len(getattr(self, 'chrom_coords'))):
                _cand_spot_list.append([np.array(_spot_lst[_chrom_id]).copy()
                                        for _spot_lst in _all_spots])
        else:
            _cand_spot_list = [_all_spots]
        
        # judge whether use chrom_coords
        if not _use_chrom_coords:
            _chrom_coords = None
        else:
            _chrom_coords = getattr(self, 'chrom_coords')

        ## Initialize
        _picked_spot_list = []
        _plot_folder = os.path.join(self.map_folder, self.fovs[self.fov_id].replace('.dax', ''))
        _plot_filename = f"Steps-{_pick_type}_{_data_type}_{self.cell_id}.png"
        if _save_inter_plot and not os.path.exists(_plot_folder):
            if _verbose:
                print(f"-- create distance_map folder:{_plot_folder} to save picking-step plots")
            os.makedirs(_plot_folder)
        # pick spots according to types
        if _pick_type == 'naive':
            from ..spot_tools.picking import naive_pick_spots_for_chromosomes
            # loop through chromosomes and pick
            _picked_spot_list, _picked_ind_list = naive_pick_spots_for_chromosomes(
                _all_spots, _ids, chrom_coords=_chrom_coords, intensity_th=_intensity_th,
                hard_intensity_th=_hard_intensity_th,chrom_share_spots=_chrom_share_spots,
                distance_zxy=self.shared_parameters['distance_zxy'],
                return_indices=True, verbose=_verbose)

        elif _pick_type == 'dynamic':
            # directly do dynamic picking
            # note: by running this allows default Naive picking as initial condition
            from ..spot_tools.picking import dynamic_pick_spots_for_chromosomes
            _picked_spot_list, _picked_ind_list = dynamic_pick_spots_for_chromosomes(
                _all_spots, _ids, chrom_coords=_chrom_coords, sel_spot_list=None,
                intensity_th=_intensity_th, hard_intensity_th=_hard_intensity_th,
                ref_spot_list=_ref_spot_list, ref_spot_ids=_ref_ids, 
                nb_dist_list=None, spot_num_th=_spot_num_th,
                ref_dist_metric=_ref_dist_metric, score_metric=_score_metric,
                local_size=_local_size, w_ctdist=_w_ctdist, w_lcdist=_w_lcdist,
                w_int=_w_int, w_nbdist=_w_nbdist, ignore_nan=_ignore_nan, 
                nan_mask=_nan_mask, inf_mask=_inf_mask, 
                chrom_share_spots=_chrom_share_spots,
                distance_zxy=self.shared_parameters['distance_zxy'],
                distance_limits=_distance_limits,
                return_indices=True, verbose=_verbose)

        elif _pick_type == 'EM':
            # dirctly do EM
            # note: by running this allows default Naive picking as initial condition
            from ..spot_tools.picking import EM_pick_spots_for_chromosomes
            _picked_spot_list, _picked_ind_list = EM_pick_spots_for_chromosomes(
                _all_spots, _ids, chrom_coords=_chrom_coords, sel_spot_list=None,
                num_iters=_num_iters, terminate_th=_terminate_th,
                intensity_th=_intensity_th, hard_intensity_th=_hard_intensity_th,
                ref_spot_list=_ref_spot_list, ref_spot_ids=_ref_ids, 
                nb_dist_list=None, spot_num_th=_spot_num_th, 
                local_size=_local_size, w_ctdist=_w_ctdist, w_lcdist=_w_lcdist,
                w_int=_w_int, w_nbdist=_w_nbdist, distance_limits=_distance_limits,
                ignore_nan=_ignore_nan, nan_mask=_nan_mask, inf_mask=_inf_mask, 
                chrom_share_spots=_chrom_share_spots,
                distance_zxy=self.shared_parameters['distance_zxy'],
                check_spots=_check_spots, check_th=_check_th, 
                check_percentile=_check_percentile, hard_dist_th=_hard_dist_th,
                save_plot=_save_inter_plot, save_path=_plot_folder,
                save_filename=_plot_filename,
                return_indices=True, verbose=_verbose)
        else:
            raise ValueError(f"Wrong input _pick_type!")

        # convert cases for not use_chrom_coords
        if not _use_chrom_coords:
            _picked_spot_list = _picked_spot_list[0]
            _picked_ind_list = _picked_ind_list[0]

        # add to attribute
        if _save_to_attr:
            setattr(self, _picked_attr, _picked_spot_list)

        # save to info
        if _save_to_info:
            self._save_to_file('cell_info', 
                               _save_dic={_picked_attr: _picked_spot_list}, 
                               _verbose=_verbose)

        # return
        if _return_indices:
            return _picked_spot_list, _picked_ind_list
        else:
            return _picked_spot_list

    # visualize picked spots
    def _visualize_picked_spots(self, _data_type='unique', _pick_type='EM', 
                                _use_chrom_coords=True,
                                _overwrite=False, _verbose=True):
        """Function to visualize picked-fitted spots in corresponding images, 
            generate one window for each chromosome
        Inputs:
            _pick_type: method for picking spots, str ('EM', 'dynamic' or 'naive')
            _data_type: data type of spots to be picked, str ('unique', 'decoded' etc)
            _use_chrom_coords: whether use chrom_coords in cell_data, bool (default: True)
            _overwrite: whether overwrite existing info, bool (default: False)
            _verbose: say something!, bool (default: True)
        Outputs:    
            tuple of viewer items
        """
        ## check inputs
        # pick type
        _allowed_pick_types = ['EM', 'naive', 'dynamic']
        if _pick_type not in _allowed_pick_types:
            raise ValueError(f"Wrong input for _pick_type:{_pick_type}, should be among {_allowed_pick_types}")
        # data type
        _data_type = _data_type.lower()
        if _data_type not in self.shared_parameters['allowed_data_types']:
            raise ValueError(
                f"Wrong input for {_data_type}, should be among {self.shared_parameters['allowed_data_types']}")
        
        ## required attributes
        # generate attribute names
        _im_attr = _data_type + '_' + 'ims'
        _id_attr = _data_type + '_' + 'ids'
        _spot_attr = _data_type + '_' + 'spots'
        # generate save attribute
        _picked_attr = str(_pick_type) + '_'+ 'picked' + '_' + _spot_attr
        # load images and ids if necessary
        if not hasattr(self, _im_attr) or not hasattr(self, _id_attr):
            self._load_from_file(_data_type=_data_type, _overwrite=True, _verbose=_verbose)
        # load picked spots if necessary
        if not hasattr(self, _picked_attr) or not hasattr(self, 'chrom_coords'):
            self._load_from_file(_data_type='cell_info', _load_attrs=[_picked_attr, 'chrom_coords'],
                                 _verbose=_verbose)
        # generate image names to show in viewer
        _image_name_list = []
        _sel_spot_list = []
        _sel_ind_list = []
        if not _use_chrom_coords:
            _im_names = [f"id:{_id}, int:{np.round(_spot[0],2)}, coord:{np.round(_spot[1:4],1)}" 
                for _id, _spot in zip(getattr(self, _id_attr), getattr(self, _picked_attr))
                ]
            _sel_spots = [_spot
                for _id, _spot in zip(getattr(self, _id_attr), getattr(self, _picked_attr))
                if not np.isnan(_spot).any()]
            _sel_inds = [_i
                for _i,(_id, _spot) in enumerate(zip(getattr(self, _id_attr), getattr(self, _picked_attr)))
                if not np.isnan(_spot).any()]
            _image_name_list.append(_im_names)
            _sel_spot_list.append(_sel_spots)
            _sel_ind_list.append(_sel_inds)
        else:
            for _chrom_id, _chrom_coord in enumerate(getattr(self,'chrom_coords')):
                _spots = getattr(self, _picked_attr)[_chrom_id]
                _im_names = [f"chr:{np.round(_chrom_coord,1)}, id:{_id}, int:{np.round(_spot[0],2)}, coord:{np.round(_spot[1:4],1)}" 
                    for _id, _spot in zip(getattr(self, _id_attr), _spots)
                    ]
                _sel_spots = [_spot
                    for _id, _spot in zip(getattr(self, _id_attr), _spots)
                    if not np.isnan(_spot).any()]
                _sel_inds = [_i
                    for _i, (_id, _spot) in enumerate(zip(getattr(self, _id_attr), _spots))
                    if not np.isnan(_spot).any()]
                _image_name_list.append(_im_names)
                _sel_spot_list.append(np.array(_sel_spots))
                _sel_ind_list.append(np.array(_sel_inds, dtype=np.int))
        # generate visualization items
        _vis_objs = []
        for _chrom_id, (_spots, _inds, _names) in enumerate(zip(_sel_spot_list, _sel_ind_list, _image_name_list)):
            _vis_objs.append(visual_tools.visualize_fitted_spot_images(
                    getattr(self, _im_attr), _spots[:,1:4], _inds, save_folder=self.save_folder,
                    image_names=_names,
                    save_name=f"visualize_{_picked_attr}_chr_{_chrom_id}.pkl",
                    overwrite=_overwrite, verbose=_verbose,
                )
            )
        return tuple(_vis_objs)


    def _generate_distance_map(self, _data_type='unique', 
                               _pick_type='EM', _sel_ids=None,
                               _save_info=True, _save_plot=True, _limits=[0, 2000], _cmap='seismic_r',
                               _fig_dpi=300, _fig_size=4, _overwrite=False, _verbose=True):
        """Function to generate distance map"""
        ## check inputs
        # use chrom_coords?
        _use_chrom_coords = True
        if not hasattr(self, 'chrom_coords'):
            self._load_from_file('cell_info', _load_attrs=['chrom_coords'])
            if not hasattr(self, 'chrom_coords'):
                _use_chrom_coords = False
        ## check specific attributes and initialize
        if _data_type not in self.shared_parameters['allowed_data_types']:
            raise ValueError(f"Wrong input for _data_type:{_data_type}, should be among {self.shared_parameters['allowed_data_types']}")
        # extract attribute names
        if _pick_type != '':
            _key_attr = str(_pick_type) + '_picked_' + str(_data_type) + '_spots'
        else:
            _key_attr = 'picked_' + str(_data_type) + '_spots'
        # id attributes
        _id_attr = _data_type + '_' + 'ids'
        _save_attr = str(_pick_type) + '_' + str(_data_type) + '_' + 'distance_map'
        if not hasattr(self, _id_attr):
            self._load_from_file('cell_info', _load_attrs=[_id_attr])

        ## get ids
        _ids = getattr(self, _id_attr)
        # special ids for RNA, corresponding to DNA regions
        if _data_type == 'rna-unique' or _data_type == 'rna':
            _gids = []
            if hasattr(self, 'rna-info_dic'):
                _rna_dic = getattr(self, 'rna-info_dic')
            else:
                _rna_dic = self._load_rna_info(_verbose=False)
            for _id in _ids:
                _info = _rna_dic[_id]
                if 'DNA_id' in _info:
                    _gids.append(int(_info['DNA_id']))
                else:
                    _gids.append(-1 * _id)
            _ids = _gids
        elif _data_type == 'gene':
            _gids = []
            if hasattr(self, 'gene_dic'):
                _rna_dic = getattr(self, 'gene_dic')
            else:
                _rna_dic = self._load_gene_info(_verbose=False)
            for _id in _ids:
                _info = _rna_dic[_id]
                if 'DNA_id' in _info:
                    _gids.append(int(_info['DNA_id']))
                else:
                    _gids.append(-1 * _id)
            _ids = _gids
        else:
            pass 
        
        ## get picked spots
        # check loading of necessary
        if not hasattr(self, _key_attr):
            self._load_from_file('cell_info', _load_attrs=[_key_attr])
            if not hasattr(self, _key_attr):
                raise AttributeError(f"No {_key_attr} info found in cell-data and saved cell_info.")
        _picked_spots = getattr(self, _key_attr)
        if not _use_chrom_coords: 
            _picked_spots = [_picked_spots]  # convert to same order list if not use_chrom_coords
        
        # filter ids and spots given selected ids
        if _sel_ids is not None:
            # order matters, change picked_spots first, then change ids
            if len(_picked_spots) > 0 and len(_picked_spots[0]) != len(_sel_ids):
                _picked_spots = [
                    np.array([_spt for _id, _spt in zip(_ids, _spots) if _id in _sel_ids ]) 
                    for _spots in _picked_spots]
            else:
                if _verbose:
                    print(f"--- spots already picked by given _sel_ids.")
            # screen ids 
            _ids = [_id for _id in _ids if _id in _sel_ids]

        # check existing saved attr
        if not hasattr(self, _save_attr):
            self._load_from_file('distance_map', _distmap_data=_data_type, _distmap_pick=_pick_type)
        if hasattr(self, _save_attr) and not _overwrite:
            if _verbose:
                print(f"-- directly load {_save_attr} from fov:{self.fov_id}, cell:{self.cell_id}")
            # loading
            _distmaps = getattr(self, _save_attr)
            # check size of this distance_map is same as picked_spots, which means picked spots are not updated
            for _distmap,_spots in zip(_distmaps, _picked_spots):
                if len(_spots) != len(_distmap):
                    del(_distmaps) # clear distmaps
                    break

        ## try to generate new ones of distmaps not exists
        if '_distmaps' not in locals():
            # initialize distmaps    
            _distmaps = []
            for _chrom_id, _spots in enumerate(_picked_spots):
                if _verbose:
                    print(f"-- generate {_data_type} dist-map for fov:{self.fov_id}, cell:{self.cell_id}, chrom:{_chrom_id}")
                # get zxy coordinates
                _zxys = np.array(_spots)[:,1:4] * self.shared_parameters['distance_zxy'][np.newaxis,:]
                # sort 
                _order =np.argsort(_ids)
                _zxys = _zxys[_order]
                # generate distmap
                _distmap = squareform(pdist(_zxys))
                # transform inf into NaN
                _distmap[_distmap == np.inf] = np.nan
                # append 
                _distmaps.append(_distmap)
            # do the saving here
            if _save_info:
                self._save_to_file('distance_map', _save_dic={_save_attr: _distmaps}, _verbose=_verbose)
        
        ## make plot         
        for _chrom_id, _distmap in enumerate(_distmaps):
            plt.figure(figsize=(1.25*_fig_size, _fig_size), dpi=_fig_dpi)
            plt.title(f"{_data_type} dist-map for fov:{self.fov_id}, cell:{self.cell_id}, chrom:{_chrom_id}")
            plt.imshow(_distmap, interpolation='nearest', cmap=_cmap, vmin=np.min(_limits), vmax=np.max(_limits))
            plt.colorbar(ticks=range(np.min(_limits),np.max(_limits)+200,200), label='distance (nm)')
            if len(_distmap) > 300:
                plt.xticks(np.arange(0,len(_distmap), 100), np.array(_ids)[::100])
                plt.yticks(np.arange(0,len(_distmap), 100), np.array(_ids)[::100])
            elif len(_distmap) > 100:
                plt.xticks(np.arange(0,len(_distmap), 50), np.array(_ids)[::50])
                plt.yticks(np.arange(0,len(_distmap), 50), np.array(_ids)[::50])
            else:
                plt.xticks(np.arange(0,len(_distmap), 20), np.array(_ids)[::20])
                plt.yticks(np.arange(0,len(_distmap), 20), np.array(_ids)[::20])
            # save plot
            if _save_plot:
                _distmap_fd = os.path.join(self.map_folder, self.fovs[self.fov_id].replace('.dax', ''))
                if not os.path.exists(_distmap_fd):
                    if _verbose:
                        print(f"++ Make directory: {_distmap_fd}")
                    os.makedirs(_distmap_fd)
                _save_fl = os.path.join(_distmap_fd, f"{_pick_type}_distance-map_{_data_type}_{self.cell_id}_{_chrom_id}.png")
                plt.savefig(_save_fl, transparent=True)
            if __name__ == "__main__":
                plt.show()
        
        # append into attribute and save
        setattr(self, _save_attr, _distmaps)

        # return
        return _distmaps

    # merge a RNA cell_data to this DNA cell_data
    def _merge_RNA_to_DNA(self, _source_cell_data, _merge_type='cell_info',
                          _attr_feature='rna-',
                          _load_in_ram=True, _save_to_file=True, 
                          _overwrite=False, _verbose=True):
        """Function to match decoded and unique regions and generate matched ids, spots, distance maps etc.
        Inputs:
            _source_cell_data: Input RNA cell_data with descent objects
        """
        ## check inputs
        if _verbose:
            _start_time = time.time()
            print(
                f"-- start merging {_merge_type} in cell_data for fov:{self.fov_id}, cell:{self.cell_id} ")
        # merge type
        _allowed_merge_types = ['cell_info','unique','distance_map']
        _merge_type = _merge_type.lower()
        if _merge_type not in _allowed_merge_types:
            raise ValueError(f"Wrong input _merge_type:{_merge_type}, should be among {_allowed_merge_types}")
        
        ## merge infos in cell_info, which is the most useful option
        if _merge_type == 'cell_info':
            # load cell_info for _source_cell_data
            _source_cell_data._load_from_file('cell_info', _verbose=_verbose)
            # extract all attributes that are not function
            _all_attrs = [_attr for _attr in dir(_source_cell_data) if not _attr.startswith('_')]
            # modify attributes by given _attr_feature
            _new_attrs = []
            for _i, _attr in enumerate(_all_attrs):
                if _attr_feature not in _attr:
                    _new_attrs.append(_attr_feature + _attr)
                else:
                    _new_attrs.append(_attr)
            # append!
            _updated_dic = {}
            for _i, (_attr, _new_attr) in enumerate(zip(_all_attrs, _new_attrs)):
                if not hasattr(self, _new_attr) or _overwrite:
                    if _load_in_ram:
                        setattr(self, _new_attr, getattr(_source_cell_data, _attr))
                    _updated_dic[_new_attr] = getattr(_source_cell_data, _attr)
            # save
            if _save_to_file:
                self._save_to_file('cell_info', _save_dic=_updated_dic, _verbose=_verbose)
                if _verbose:
                    print(f"--- {len(list(_updated_dic.keys()))} new attributes appended to cell_data")        
        ## merge unique images            
        elif _merge_type == 'unique':
            pass
        ## merge distance maps (although presumbly there's no distance map for RNA?)
        elif _merge_type == 'distance_map':
            pass 

        if _verbose:
            print(f"--- time spent in merging RNA to DNA: {time.time()-_start_time}")       

    # transfer data_type for a cell_data object
    def _transfer_data_type(self, _data_type='unique', _target_type='rna-unique',
                            _load_in_ram=False, _save_to_file=True, _new_savefolder=None, 
                            _verbose=True):
        """Function to transfer a specific data_type into new one
        ----------------------------------------------------------------------------------------------
        Attributes to be transferred:
        (* represents data_type, ~ represents)
        *_images
        *_ids
        *_channels
        *_spots
        ~_picked_*_spots
        ~_*_distance_map
        ----------------------------------------------------------------------------------------------
        Inputs:
            _data_type: original data_type, string (default: 'unique')
            _target_type: target data_type, string (default: 'rna-unique')
            _load_in_ram: whether load new_cell_data into RAM, bool (default: False)
            _save_to_file: whether save transferred cell_data to file, bool (default: True)
            _new_savefolder: specify new save folder, bool (default: 'analysis_folder/transferred')
            
            """
        ## check inputs
        _start_time = time.time()
        if _verbose:
            print(f"-- start transfering fov:{self.fov_id}, cell:{self.cell_id} from {_data_type} to {_target_type}")
        # check data_type and _target_type
        _allowed_types = ['unique', 'decoded', 'rna-unique']
        _data_type = _data_type.lower()
        if _data_type not in _allowed_types:
            raise KeyError(f"Wrong input key for _data_type:{_data_type}")
        _target_type = _target_type.lower()
        if _target_type not in _allowed_types:
            raise KeyError(f"Wrong input key for _target_type:{_target_type}")
        
        ## prepare attributes
        if _verbose:
            print("-- prepare attributes:")
        # load existing datasets
        self._load_from_file('cell_info',  _verbose=_verbose)
        self._load_from_file(_data_type,  _verbose=_verbose)
        self._load_from_file('distance_map', _distmap_data=None, 
                             _distmap_pick=None, _verbose=_verbose)
        
        # generate attribute names
        _im_attr = _data_type + '_' + 'ims'
        _id_attr = _data_type + '_' + 'ids'
        _channel_attr = _data_type + '_' + 'channels'
        _spot_attr = _data_type + '_' + 'spots'
        # get attrs for picked spots
        _picked_attrs = [_attr for _attr in dir(self) if not _attr.startswith('_')\
                                                        and 'picked' in _attr \
                                                        and _spot_attr in _attr ]
        # get attrs for distance maps
        _distmap_attrs = [_attr for _attr in dir(self) if not _attr.startswith('_') \
                                                        and 'distance_map' in _attr \
                                                        and _data_type in _attr.split('_') ]
        # summarize changed attributes
        _changed_attrs = [_im_attr, _id_attr, _channel_attr, _spot_attr]
        _changed_attrs += _picked_attrs
        _changed_attrs += _distmap_attrs
        if _verbose:
            print(f"--- attributes to be converted to new datatype: {_changed_attrs}")
        # unchanged attributes
        _kept_attrs = [_attr for _attr in dir(self) if not _attr.startswith('_') and _attr not in _changed_attrs]
        if _verbose:
            print(f"--- attributes remain unchanged: {_kept_attrs}")
        # generate save_filename:
        _save_filename = os.path.join(self.save_folder, _data_type+'_'+'rounds.npz')        

        ## create a new cell_data object with shared attributes
        if _verbose:
            print("-- creating new Cell_Data ojbect.")
        _init_params = {'data_folder':self.data_folder,
                        'fov_id': self.fov_id,
                        'cell_id': self.cell_id}
        _new_cell_data = Cell_Data(parameters=_init_params)
        # transfer all kept attributes
        for _attr in _kept_attrs:
            setattr(_new_cell_data, _attr, getattr(self, _attr))

        ## convert all related attributes
        for _attr in _changed_attrs:
            _new_attr = _attr.replace(_data_type, _target_type)
            setattr(_new_cell_data, _new_attr, getattr(self, _attr))
        # convert new save_folder:
        if _new_savefolder is None:
            _save_folder = os.path.join(os.path.dirname(self.save_folder.split('fov')[0]),
                                        f'transferred_{_target_type}',
                                        f'fov-{self.fov_id}',
                                        f'cell-{self.cell_id}')
        else:
            _save_folder = os.path.join(_new_savefolder,
                                        f'fov-{self.fov_id}',
                                        f'cell-{self.cell_id}')
        if not os.path.exists(_save_folder):
            os.makedirs(_save_folder)
            if _verbose:
                print(f"--- creating save_folder:{_save_folder}")
        setattr(_new_cell_data, 'save_folder', _save_folder)
        # save all
        if _save_to_file:
            _new_cell_data._save_to_file("cell_info", _verbose=_verbose)
            _new_cell_data._save_to_file(_target_type, _verbose=_verbose)
            _new_cell_data._save_to_file("distance_map", _verbose=_verbose)
        if _verbose:
            print(f"--- time spent in transfering cell_data: {time.time()-_start_time}")
        # return 
        if _load_in_ram:
            return _new_cell_data
        
    def _domain_calling(self, _data_type='unique', _pick_type='EM', method='basic',                        
                        ):
        pass

class Encoding_Group():
    """defined class for each group of encoded images"""
    def __init__(self, ims, hybe_names, encoding_matrix, save_folder,
                 fov_id, cell_id, color, group_id, readouts=None):
        # info for this cell
        self.ims = ims
        self.names = hybe_names
        self.matrix = encoding_matrix
        self.save_folder = save_folder
        # detailed info for this group
        self.fov_id = fov_id
        self.cell_id = cell_id
        self.color = color
        self.group_id = group_id
        if readouts:
            self.readouts = readouts
    def _save_group(self, _overwrite=False, _verbose=True):
        _combo_savefolder = os.path.join(self.save_folder,
                                        'group-'+str(self.group_id),
                                        'channel-'+str(self.color)
                                        )
        if not os.path.exists(_combo_savefolder):
            os.makedirs(_combo_savefolder)
        _combo_savefile = _combo_savefolder+os.sep+'rounds.npz'
        # if file exists and not overwriting, skip
        if os.path.exists(_combo_savefile) and not _overwrite:
            if _verbose:
                print("file {:s} already exists, skip.".format(_combo_savefile))
            return False
        # else, write
        _attrs = [_attr for _attr in dir(self) if not _attr.startswith('_')]
        _combo_dic = {
            'observation': np.stack(self.ims),
            'encoding': self.matrix,
            'names': np.array(self.names),
        }
        if hasattr(self, 'readouts'):
            _combo_dic['readouts'] = np.array(self.readouts)
        # save
        if _verbose:
            print("-- saving combo to:", _combo_savefile)
        np.savez_compressed(_combo_savefile, **_combo_dic)
        return True

class Merfish_Group():
    """Define class for MERFISH type of encoded images"""
    def __init__(self, ims, hyb_names, colors, encoding_matrix, save_folder,
                 fov_id, cell_id, readouts=None):
        """Initalize a merfish-group class"""
        self.ims = ims
        self.hyb_names = hyb_names
        self.colors = colors
        self.encoding_matrix = encoding_matrix
        self.save_folder = save_folder
        self.fov_id = fov_id
        self.cell_id = cell_id
        if readouts is not None:
            self.readouts = readouts

    def _save_group(self, _overwrite=False, _verbose=True):
        _merfish_savefile = os.path.join(self.save_folder,
                                         'merfish_rounds.npz'
                                         )
        if os.path.exists(_merfish_savefile) and not _overwrite:
            if _verbose:
                print(f"file {_merfish_savefile} already exists, skip!")
            return False
        else:
            pass

