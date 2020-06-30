import sys,glob,os
import numpy as np
from . import _correction_folder, _temp_folder, _distance_zxy, _sigma_zxy, _image_size, _allowed_colors

def get_hybe(folder):
    #this sorts by the region number Rx.
    try: return int(os.path.basename(folder).split('H')[-1].split('R')[0])
    except: return np.inf

def get_folders(master_folder, feature='H', verbose=True):
    '''Function to get all subfolders
    given master_folder directory
        feature: 'H' for hybes, 'B' for beads
    returns folders, field_of_views'''
    folders = [folder for folder in glob.glob(master_folder+os.sep+'*') if os.path.basename(folder)[0]==feature] # get folders start with 'H'
    folders = list(np.array(folders)[np.argsort(list(map(get_hybe,folders)))])
    if len(folders) > 0:
        fovs = sorted(list(map(os.path.basename,glob.glob(folders[0]+os.sep+'*.dax'))),key=lambda l:int(l.split('.dax')[0].split('_')[-1]))
    else:
        raise IOError("No folders detected!")
    if verbose:
        print("Get Folder Names: (ia.get_img_info.get_folders)")
        print("- Number of folders:", len(folders))
        print("- Number of field of views:", len(fovs))
    return folders, fovs

def get_img_fov(folders, fovs, fov_id=0, verbose=True):
    '''Function to load a certain field_of_view
    Inputs:
        folders: folder names for each hybe, list of strings
        fovs: field of view names, list of strings
        fov_id: field_of_views id, int
    Outputs:
        List of dax-image items
        List of names compatible to dax-images
    '''
    from . import visual_tools as vis
    if not isinstance(fov_id, int):
        raise ValueError('Wrong fov_id input type!')
    if verbose:
        print("- get images of a fov (ia.get_img_info.get_img_fov)")
    _fov = fovs[fov_id]
    _names, _ims = [],[]
    if verbose:
        print("-- loading field of view:", _fov)   
    # load images
    for _folder in folders[:]:
        _filename= _folder+os.sep+_fov
        if os.path.exists(_filename):
            _names += [os.path.basename(_folder)+os.sep+_fov]
            _ims += [vis.DaxReader(_filename).loadMap()]

    if verbose:
        print("- number of images loaded:", len(_ims))

    return _ims, _names

def get_img_hyb(folders, fovs, hyb_id=0, verbose=True):
    '''Function to load images for a certain hyb
    Inputs:
        folders: folder names for each hybe, list of strings
        fovs: field of view names, list of strings
        hyb_id: field_of_views id, int
    Outputs:
        List of dax-image items
        List of names compatible to dax-images
    '''
    from . import visual_tools as vis
    # check input style
    if not isinstance(hyb_id, int):
        raise ValueError('Wrong hyb_id input type!')
    # initialize
    _names, _ims = [],[]
    _folder = folders[hyb_id]
    print("--loading images from folder:", _folder)
    # read images
    for _fov in fovs:
        _filename = _folder+os.sep+_fov
        if os.path.exists(_filename):
            _names += [os.path.basename(_folder)+os.sep+_fov]
            _ims += [vis.DaxReader(_filename).loadMap()]

    print("-- number of images loaded:", len(_ims))

    return _ims, _names

## Load file Color_Usage in dataset folder
def Load_Color_Usage(master_folder, color_filename='Color_Usage', color_format='csv',
                     DAPI_hyb_name="H0R0", return_color=True):
    '''Function to load standard Color_Usage file:
    Inputs:
        master_folder: master directory of this dataset, path(string)
        color_filename: filename and possible sub-path for color file, string
        color_format: format of color file, csv or txt
    Outputs:
        color_usage: dictionary of color usage, folder_name -> list of region ID
        dapi: whether dapi is used, bool
        '''
    # initialize as default
    _color_usage = {}

    # process with csv format
    if color_format == 'csv':
        _full_name = master_folder+os.sep+color_filename+"."+'csv'
        print("- Importing csv file:", _full_name)
        import csv
        with open(_full_name, 'r') as handle:
            _reader = csv.reader(handle)
            _header = next(_reader)
            print("- header:", _header)
            for _content in _reader:
                while len(_content)>0 and _content[-1] == '':
                    _content = _content[:-1]
                if len(_content) > 1:
                    _hyb = _content.pop(0)
                    _color_usage[_hyb] = _content
    # process with txt format (\t splitted)
    elif color_format == 'txt':
        _full_name = master_folder+os.sep+color_filename+"."+'txt'
        print("- Importing txt file:", _full_name)
        with open(_full_name, 'r') as handle:
            _line = handle.readline().rstrip()
            _header = _line.split('\t')
            print("-- header:", _header)
            for _line in handle.readlines():
                _content = _line.rstrip().split('\t')
                while _content[-1] == '':
                    _content = _content[:-1]
                _hyb = _content.pop(0)
                _color_usage[_hyb] = _content
    # detect dapi
    dapi_hyb_name = []
    for _hyb_name, _info in _color_usage.items():
        if 'DAPI' in _info:
            dapi_hyb_name.append(_hyb_name)
    if DAPI_hyb_name in _color_usage:
        print(f"-- Hyb {DAPI_hyb_name} exists in this data")
        if 'dapi' in _color_usage[DAPI_hyb_name] or 'DAPI' in _color_usage[DAPI_hyb_name]:
            print("-- DAPI exists in hyb:", DAPI_hyb_name)
            _dapi = True
        else:
            _dapi = False
    else:
        _dapi = False
    if return_color:
        _colors = [int(_c) for _c in _header[1:]]
        return _color_usage, _dapi, _colors
    return _color_usage, _dapi

## Load file Region_Positions in dataset folder
def Load_Region_Positions(analysis_folder, filename='Region_Positions', table_format='csv',
                          verbose=True):
    '''Function to load standard Color_Usage file:
    Inputs:
        analysis_folder: analysis directory of this dataset, path(string)
        filename: filename and possible sub-path for color file, string
        table_format: format of color file, csv or txt
    Outputs:
        color_usage: dictionary of color usage, folder_name -> list of region ID
        dapi: whether dapi is used, bool
        '''
    # initialize as default
    _genomic_positions = {}

    # process with csv format
    if table_format == 'csv':
        _full_name = analysis_folder+os.sep+filename+"."+'csv'
        if verbose:
            print("- Importing csv file:", _full_name)
        import csv
        with open(_full_name, 'r') as _handle:
            _reader = csv.reader(_handle)
            _header = next(_reader)
            if verbose:
                print("- header:", _header)
            for _content in _reader:
                while len(_content) > 0 and _content[-1] == '':
                    _content = _content[:-1]
                if len(_content) > 1:
                    _hyb = int(_content.pop(0))
                    _dic = {_h: _c for _h, _c in zip(_header[1:], _content)}
                    for _d, _v in _dic.items():
                            if _d != 'chr':
                                _dic[_d] = int(_v)
                    _genomic_positions[_hyb] = _dic
    # process with txt format (\t splitted)
    elif table_format == 'txt':
        _full_name = analysis_folder+os.sep+filename+"."+'txt'
        if verbose:
            print("- Importing txt file:", _full_name)
        with open(_full_name, 'r') as _handle:
            _line = _handle.readline().rstrip()
            _header = _line.split('\t')
            if verbose:
                print("-- header:", _header)
            for _line in _handle.readlines():
                _content = _line.rstrip().split('\t')
                while _content[-1] == '':
                    _content = _content[:-1]
                if len(_content) > 1:
                    _hyb = int(_content.pop(0))
                    _dic = {_h: _c for _h, _c in zip(_header[1:], _content)}
                    for _d, _v in _dic.items():
                            if _d != 'chr':
                                _dic[_d] = int(_v)
                    _genomic_positions[_hyb] = _dic
    if verbose:
        print(f"-- {len(_genomic_positions)} genomic regions loaded!")
    return _genomic_positions

# load chip-seq peak info
def Load_ChIP_Data(analysis_folder, gene_name, postfix="ChIP-Seq_chr21", table_format='csv',
                   verbose=True):
    '''Function to load standard Color_Usage file:
    Inputs:
        analysis_folder: analysis directory of this dataset, path(string)
        filename: filename and possible sub-path for color file, string
        table_format: format of color file, csv or txt
    Outputs:
        color_usage: dictionary of color usage, folder_name -> list of region ID
        dapi: whether dapi is used, bool
        '''
    # initialize as default
    _chip_peaks = []

    # process with csv format
    if table_format == 'csv':
        _full_name = os.path.join(
            analysis_folder, str(gene_name)+'_'+postfix+".csv")
        if verbose:
            print("- Importing csv file:", _full_name)
        import csv
        with open(_full_name, 'r') as _handle:
            _reader = csv.reader(_handle)
            _header = next(_reader)
            if verbose:
                print("- header:", _header)
            for _content in _reader:
                while len(_content) > 0 and _content[-1] == '':
                    _content = _content[:-1]
                if len(_content) > 1:
                    _dic = {_h: _c for _h, _c in zip(_header, _content)}
                    for _k, _v in _dic.items():
                        if _k in ['start', 'end']:
                            _dic[_k] = int(_v)
                        if _k in ['fold', 'midpoint']:
                            _dic[_k] = float(_v)
                    _chip_peaks.append(_dic)
    # process with txt format (\t splitted)
    elif table_format == 'txt':
        _full_name = os.path.join(analysis_folder, str(gene_name)+'_'+postfix+".txt")
        if verbose:
            print("- Importing txt file:", _full_name)
        with open(_full_name, 'r') as _handle:
            _line = _handle.readline().rstrip()
            _header = _line.split('\t')
            if verbose:
                print("-- header:", _header)
            for _line in _handle.readlines():
                _content = _line.rstrip().split('\t')
                while _content[-1] == '':
                    _content = _content[:-1]
                if len(_content) == len(_header):
                    _dic = {_h: _c for _h, _c in zip(_header, _content)}
                    for _k, _v in _dic.items():
                        if _k in ['start', 'end']:
                            _dic[_k] = int(_v)
                        if _k in ['midpoint', 'fold']:
                            _dic[_k] = float(_v)
                    _chip_peaks.append(_dic)
    if verbose:
        print(f"-- {len(_chip_peaks)} {gene_name} ChIP-Seq peaks loaded!")
    return _chip_peaks

# load RNA information
def Load_RNA_Info(analysis_folder, filename='RNA_Info', table_format='csv',
                  verbose=True):
    '''Function to load standard RNA_Info:
    ------------------------------------------------------------------------------
    table_format:
    RNA_id \t gene_name \t chr \t strand \t start \t end \t midpoint \n
    r13 \t CYP4F29P \t chr21 \t - \t 13848364 \t 13843133 \t 13845748.5 \n
    ------------------------------------------------------------------------------
    Inputs:
        analysis_folder: analysis directory of this dataset, path(string)
        filename: filename and possible sub-path for color file, string
        table_format: table_format of color file, csv or txt
        verbose: say something!, bool (default:True)
    Outputs:
        _rna_info: dictionary of RNAs labelled in experiment, 
            rna_id -> dict of other formation
        '''
    # initialize as default
    _rna_info = {}

    # process with csv table_format
    if table_format == 'csv':
        _full_name = analysis_folder+os.sep+filename+"."+'csv'
        if verbose:
            print("- Importing csv file:", _full_name)
        import csv
        with open(_full_name, 'r') as _handle:
            _reader = csv.reader(_handle)
            _header = next(_reader)
            if verbose:
                print("- header:", _header)
            for _content in _reader:
                while len(_content) > 0 and _content[-1] == '':
                    _content = _content[:-1]
                if len(_content) > 1:
                    _hyb = int(_content.pop(0))
                    _dic = {_h: _c for _h, _c in zip(_header[1:], _content)}
                    for _d, _v in _dic.items():
                            if _d in ['start', 'end']:
                                _dic[_d] = int(_v)
                            if _d == 'midpoint':
                                _dic[_d] = float(_v)
                    _rna_info[_hyb] = _dic
    # process with txt table_format (\t splitted)
    elif table_format == 'txt':
        _full_name = analysis_folder+os.sep+filename+"."+'txt'
        if verbose:
            print("- Importing txt file:", _full_name)
        with open(_full_name, 'r') as _handle:
            _line = _handle.readline().rstrip()
            _header = _line.split('\t')
            if verbose:
                print("-- header:", _header)
            for _line in _handle.readlines():
                _content = _line.rstrip().split('\t')
                while _content[-1] == '':
                    _content = _content[:-1]
                if len(_content) > 1:
                    _hyb = int(_content.pop(0))
                    _dic = {_h: _c for _h, _c in zip(_header[1:], _content)}
                    for _d, _v in _dic.items():
                            if _d in ['start', 'end']:
                                _dic[_d] = int(_v)
                            if _d in['midpoint', 'fpkm']:
                                _dic[_d] = float(_v)
                    _rna_info[_hyb] = _dic
    if verbose:
        print(f"-- {len(_rna_info)} RNA information loaded!")
    return _rna_info

# load gene information
def Load_Gene_Info(analysis_folder, filename='Gene_Info', table_format='csv',
                  verbose=True):
    '''Function to load standard gene_Info:
    ------------------------------------------------------------------------------
    table_format:
    gene_id \t gene_name \t chr \t TSS_posiiton \t 5kb_readout \n
    2 \t HSPA13 \t chr21 \t - \t 14383484 \t NDB_1159 \n
    ------------------------------------------------------------------------------
    Inputs:
        analysis_folder: analysis directory of this dataset, path(string)
        filename: filename and possible sub-path for color file, string
        table_format: table_format of color file, csv or txt
        verbose: say something!, bool (default:True)
    Outputs:
        _gene_info: dictionary of genes labelled in experiment, 
            gene_id -> dict of other formation
        '''
    # initialize as default
    _gene_info = {}

    # process with csv table_format
    if table_format == 'csv':
        _full_name = analysis_folder+os.sep+filename+"."+'csv'
        if verbose:
            print("- Importing csv file:", _full_name)
        import csv
        with open(_full_name, 'r') as _handle:
            _reader = csv.reader(_handle)
            _header = next(_reader)
            if verbose:
                print("- header:", _header)
            for _content in _reader:
                while len(_content) > 0 and _content[-1] == '':
                    _content = _content[:-1]
                if len(_content) > 1:
                    _hyb = int(_content.pop(0))
                    _dic = {_h: _c for _h, _c in zip(_header[1:], _content)}
                    for _d, _v in _dic.items():
                            if _d in ['start', 'end', 'TSS_position']:
                                _dic[_d] = int(_v)
                            if _d == 'midpoint':
                                _dic[_d] = float(_v)
                    _gene_info[_hyb] = _dic
    # process with txt table_format (\t splitted)
    elif table_format == 'txt':
        _full_name = analysis_folder+os.sep+filename+"."+'txt'
        if verbose:
            print("- Importing txt file:", _full_name)
        with open(_full_name, 'r') as _handle:
            _line = _handle.readline().rstrip()
            _header = _line.split('\t')
            if verbose:
                print("-- header:", _header)
            for _line in _handle.readlines():
                _content = _line.rstrip().split('\t')
                while _content[-1] == '':
                    _content = _content[:-1]
                if len(_content) > 1:
                    _hyb = int(_content.pop(0))
                    _dic = {_h: _c for _h, _c in zip(_header[1:], _content)}
                    for _d, _v in _dic.items():
                            if _d in ['start', 'end']:
                                _dic[_d] = int(_v)
                            if _d == 'midpoint':
                                _dic[_d] = float(_v)
                    _gene_info[_hyb] = _dic
    if verbose:
        print(f"-- {len(_gene_info)} gene information loaded!")
    return _gene_info

# function to match results from Load_Region_Positions and Load_ChIP_Data
def match_peak_to_region(region_dic, peak_list, return_list=True):
    """Function to match region_dictionary and a list for peaks, decide which region contain peaks"""
    # initialzie a dict to record peak existence
    _region_records = {_k:0 for _k in region_dic.keys()}
    for _peak in peak_list:
        for _rid, _region in region_dic.items():
            if _peak['midpoint'] >= _region['start'] and _peak['midpoint'] <= _region['end'] and _peak['chr'] == _region['chr']:
                _region_records[_rid] += _peak['fold']
                break
    if not return_list:
        return _region_records
    else:
        _rids = list(_region_records.keys())
        _rx = np.arange(int(min(_rids)), int(max(_rids))+1)
        _ry = np.zeros(len(_rx))
        for _rid, _signal in _region_records.items():
            _ry[np.where(_rid == _rx)[0]] = _signal
        
        return _rx, _ry

# function to match result from Load_Region_Positions to Load_RNA_Info 
def match_RNA_to_DNA(rna_dic, region_dic, max_size_th=100000):
    """Function to match RNA to DNA region and append new information to RNA_dic"""
    # initialize a dict
    _updated_dic = {_k:_v for _k,_v in rna_dic.items()}
    for _k, _rdic in _updated_dic.items():
        for _rid, _region in region_dic.items():
            if abs(_rdic['end'] - _rdic['start']) and \
                _rdic['start'] >= _region['start'] and _rdic['start'] <= _region['end'] \
                    and _rdic['chr'] == _region['chr']:
                _updated_dic[_k]['DNA_id'] = _rid
    return _updated_dic 

# function to match result from Load_Region_Positions to Load_Gene_Info 
def match_Gene_to_DNA(gene_dic, region_dic, max_size_th=100000):
    """Function to match gene to DNA region and append new information to gene_dic"""
    # initialize a dict
    _updated_dic = {_k:_v for _k,_v in gene_dic.items()}
    for _k, _rdic in _updated_dic.items():
        for _rid, _region in region_dic.items():
            if _rdic['TSS_position'] >= _region['start'] and \
               _rdic['TSS_position'] < _region['end'] and \
               _rdic['chr'] == _region['chr']:
                _updated_dic[_k]['DNA_id'] = _rid
    return _updated_dic 

def match_Enhancer_to_DNA(enhancer_dic, region_dic):
    """Assign enhancers into regions"""
    _region_dic = {_k:_v for _k,_v in region_dic.items()}
    for _k in _region_dic:
        _region_dic[_k]['enhancer_count'] = 0.
    for _k, _v in _region_dic.items():
        for _e, _ed in enhancer_dic.items():
            if (_ed['start'] >= _v['start'] and _ed['start'] < _v['end']) or (_ed['end'] >= _v['start'] and _ed['end'] < _v['end']):
                _eh_len = _ed['end'] - _ed['start']
                _eh_overlap = min(_ed['end'], _v['end']) - max(_ed['start'], _v['start'])
                _region_dic[_k]['enhancer_count'] += float(_eh_overlap) / float(_eh_len)
    return _region_dic

# function for finding bead_channel given color_usage profile
def find_bead_channel(__color_dic, __bead_mark='beads'):
    '''Given a color_dic loaded from Color_Usage file, return bead channel if applicable'''
    __bead_channels = []
    for __name, __info in sorted(list(__color_dic.items()), key=lambda k_v:int(k_v[0].split('H')[1].split('R')[0])):
        __bead_channels.append(__info.index(__bead_mark))
    __unique_channel = np.unique(__bead_channels)
    if len(__unique_channel) == 1:
        return __unique_channel[0]
    else:
        raise ValueError("-- bead channel not unique:", __unique_channel)

# function for finding DAPI channel given color_usage profile
def find_dapi_channel(__color_dic, __dapi_mark='DAPI'):
    '''Given a color_dic loaded from Color_Usage file, return bead channel if applicable'''
    __dapi_channels = []
    for __name, __info in sorted(list(__color_dic.items()), key=lambda k_v:int(k_v[0].split('H')[1].split('R')[0])):
        if __dapi_mark in __info:
            __dapi_channels.append(__info.index(__dapi_mark))
    __unique_channel = np.unique(__dapi_channels)
    if len(__unique_channel) == 1:
        return __unique_channel[0]
    else:
        raise ValueError("-- dapi channel not unique:", __unique_channel)
    
    return __unique_channel

# load encoding scheme for decoding
def Load_Encoding_Scheme(master_folder, encoding_filename='Encoding_Scheme', encoding_format='csv',
                         return_info=True, verbose=True):
    '''Load encoding scheme from csv file
    Inputs:
        master_folder: master directory of this dataset, path(string)
        encoding_filename: filename and possible sub-path for color file, string
        encoding_format: format of color file, csv or txt
    Outputs:
        _encoding_scheme: dictionary of encoding scheme, list of folder_name -> encoding matrix
        (optional)
        _num_hyb: number of hybridization per group, int
        _num_reg: number of regions per group, int
        _num_color: number of colors used in parallel, int'''
    # initialize
    _hyb_names = []
    _encodings = []
    _num_hyb,_num_reg,_num_color,_num_group = None,None,None,None
    # process with csv format
    if encoding_format == 'csv':
        _full_name = master_folder+os.sep+encoding_filename+"."+'csv'
        if verbose:
            print("- Importing csv file:", _full_name)
        import csv
        with open(_full_name, 'r') as handle:
            _reader = csv.reader(handle)
            _header = next(_reader)
            for _content in _reader:
                _hyb = _content.pop(0)
                for _i,_n in enumerate(_content):
                    if _n == '':
                        _content[_i] = -1
                if str(_hyb) == 'num_hyb':
                    _num_hyb = int(_content[0])
                elif str(_hyb) == 'num_reg':
                    _num_reg = int(_content[0])
                elif str(_hyb) == 'num_color':
                    _num_color = int(_content[0])
                else:
                    _hyb_names.append(_hyb)
                    _encodings.append(_content)
    # process with txt format (\t splitted)
    elif encoding_format == 'txt':
        _full_name = master_folder+os.sep+encoding_filename+"."+'txt'
        if verbose:
            print("- Importing txt file:", _full_name)
        with open(_full_name, 'r') as handle:
            _line = handle.readline().rstrip()
            _header = _line.split('\t')
            for _line in handle.readlines():
                _content = _line.rstrip().split('\t')
                _hyb = _content.pop(0)
                for _i,_n in enumerate(_content):
                    if _n == '':
                        _content[_i] = -1
                if str(_hyb) == 'num_hyb':
                    _num_hyb = int(_content[0])
                elif str(_hyb) == 'num_reg':
                    _num_reg = int(_content[0])
                elif str(_hyb) == 'num_color':
                    _num_color = int(_content[0])
                else:
                    _hyb_names.append(_hyb)
                    _encodings.append(_content)
    # first, read all possible colors
    _header.pop(0)
    _colors = []
    if len(_header) == _num_reg * _num_color:
        for _j in range(int(len(_header)/_num_reg)):
            if int(_header[_j*_num_reg]):
                _colors.append(_header[_j*_num_reg])
            else:
                raise EOFError('wrong color input'+str(_header[_j*_num_reg]))
    else:
        raise ValueError('Length of header doesnt match num reg and num color')
    # initialze encoding_scheme dictionary
    _encoding_scheme = {_color:{'names':[],'matrices':[]} for _color in _colors}
    # if number of region, number of hybe and number of color defined:
    if _num_hyb and _num_reg and _num_color:
        if len(_hyb_names)% _num_hyb == 0:
            for _i in range(int(len(_hyb_names)/ _num_hyb)):
                _hyb_group = _hyb_names[_i*_num_hyb: (_i+1)*_num_hyb]
                _hyb_matrix = np.array(_encodings[_i*_num_hyb: (_i+1)*_num_hyb], dtype=int)
                if _hyb_matrix.shape[1] == _num_reg * _num_color:
                    for _j in range( int(_hyb_matrix.shape[1]/_num_reg)):
                        _mat = _hyb_matrix[:,_j*_num_reg:(_j+1)*_num_reg]
                        if not (_mat == -1).all():
                            _encoding_scheme[_colors[_j]]['names'].append(_hyb_group)
                            _encoding_scheme[_colors[_j]]['matrices'].append(_mat)
                else:
                    raise ValueError('dimension of hyb matrix doesnt match color and region')
        else:
            raise ValueError('number of hybs doesnot match number of hybs per group.')
    # calculate number of groups per color
    _group_nums = []
    for _color in _colors:
        _group_nums.append(len(_encoding_scheme[_color]['matrices']))
    if verbose:
        print("-- hyb per group:",_num_hyb)
        print("-- region per group:", _num_reg)
        print("-- colors:", _colors)
        print('-- number of groups:', _group_nums)
    # return
    if return_info:
        return _encoding_scheme, _num_hyb, _num_reg, _colors, _group_nums
    return _encoding_scheme

def split_channels(ims, names, num_channel=2, buffer_frames=10, DAPI=False, verbose=True):
    '''Function to load images, and split images according to channels
    Inputs:
        ims: all images that acquired by get_images, list of images
        names: compatible names also from get_images, list of strings
        num_channel: number of channels, int
        buffer frames to be removed, non-negative int
        DAPI: whether DAPI exists in round 0, default is false.
    Outputs:
        image lists in all field_of_views
    '''
    _ims = ims
    if verbose:
        print("Split multi-channel images (ia.get_img_info.split_channels)")
        print("--number of channels:", num_channel)
    # Initialize
    if DAPI:
        print("-- assuming H0R0 has extra DAPI channel")
        _splitted_ims = [[] for _channel in range(num_channel + 1)]
    else:
        _splitted_ims = [[] for _channel in range(num_channel)]
    # if H0 is for DAPI, there will be 1 more channel:
    if DAPI:
        _im = _ims[0]
        for _channel in range(num_channel+1):
            _splitted_ims[(buffer_frames-1+_channel)%(num_channel+1)].append(_im[int(buffer_frames): -int(buffer_frames)][_channel::num_channel+1])
    # loop through images
    for _im in _ims[int(DAPI):]:
        # split each single image
        for _channel in range(num_channel):
            _splitted_ims[(buffer_frames-1+_channel)%(num_channel)].append(_im[int(buffer_frames): -int(buffer_frames)][_channel::num_channel])
    return _splitted_ims

def split_channels_by_image(ims, names, num_channel=4, buffer_frames=10, DAPI=False, verbose=True):
    '''Function to split loaded images into multi channels, save as dict'''
    # initialzie
    _ims = ims
    _im_dic = {}
    if verbose:
        print("Split multi-channel images (ia.get_img_info.split_channels)")
        print("-- number of channels:", num_channel)
    # if dapi applied
    if DAPI:
        if verbose:
            print("-- scanning through images to find images with DAPI:")
        _im_sizes = [_im.shape[0] for _im in _ims]
        if np.max(_im_sizes) == np.min(_im_sizes):
            raise ValueError('image dimension does not match for DAPI')
        if (np.max(_im_sizes) - 2*buffer_frames) / (num_channel + 1) != (np.min(_im_sizes) - 2*buffer_frames) / num_channel:
            raise ValueError('image for DAPI does not have compatible frame numbers for 1 more channel')
        else:
            if verbose:
                print("--- DAPI images discovered, splitting images")
        for _im,_name in zip(_ims, names):
            if _im.shape[0] == np.max(_im_sizes):
                _im_list = [_im[int(buffer_frames): -int(buffer_frames)][(-buffer_frames+1+_channel)%(num_channel+1)::num_channel+1] for _channel in range(num_channel+1)]
            else:
                _im_list = [_im[int(buffer_frames): -int(buffer_frames)][(-buffer_frames+1+_channel)%num_channel::num_channel] for _channel in range(num_channel)]
            _im_dic[_name] = _im_list
    else:
        if verbose:
            print("-- splitting through images without DAPI")
        for _im,_name in zip(_ims, names):
            _im_list = [_im[int(buffer_frames): -int(buffer_frames)][(-buffer_frames+1+_channel)%num_channel::num_channel] for _channel in range(num_channel)]
            _im_dic[_name] = _im_list
    return _im_dic

# match harry's result with raw data_
def decode_match_raw(raw_data_folder, raw_feature, decode_data_folder, decode_feature, fovs, e_type):
    # initialize
    _match_dic = {}
    # get raw data file list
    _raw_list = glob.glob(raw_data_folder+os.sep+'*'+raw_feature)
    _raw_list = [_raw for _raw in _raw_list if str(e_type) in _raw]
    # get decode data file list
    _decode_list = glob.glob(decode_data_folder+os.sep+'*'+decode_feature)

    print(len(_raw_list), len(_decode_list))
    # loop to match!
    for _decode_fl in sorted(_decode_list):
        # fov id
        _fov_id = int(_decode_fl.split('fov-')[1].split('-')[0])
        # search to match
        _matched_raw = [_raw_fl for _raw_fl in _raw_list if fovs[_fov_id].split('.')[0] in _raw_fl]
        # keep unique match
        if len(_matched_raw) == 1:
            _match_dic[_fov_id] = [_matched_raw[0], _decode_fl]

    return _match_dic


def get_num_frame(dax_filename, frame_per_color=_image_size[0], buffer_frame=10, verbose=False):
    """Function to extract image size and number of colors"""
    ## check input
    if '.dax' not in dax_filename:
        raise ValueError(
            f"Wrong input type, .dax file expected for {dax_filename}")
    if not os.path.isfile(dax_filename):
        raise IOError(f"input file:{dax_filename} doesn't exist!")

    _info_filename = dax_filename.replace('.dax', '.inf')
    with open(_info_filename, 'r') as _info_hd:
        _infos = _info_hd.readlines()
    # get frame number and color information
    _num_frame, _num_color = 0, 0
    _dx, _dy = 0, 0
    for _line in _infos:
        _line = _line.rstrip()
        if "number of frames" in _line:
            _num_frame = int(_line.split('=')[1])
            _num_color = (_num_frame - 2*buffer_frame) / frame_per_color
            if _num_color != int(_num_color):
                raise ValueError("Wrong num_color, should be integer!")
            _num_color = int(_num_color)
        if "frame dimensions" in _line:
            _dx = int(_line.split('=')[1].split('x')[0])
            _dy = int(_line.split('=')[1].split('x')[1])
    _im_shape = [_num_frame, _dx, _dy]

    return _im_shape, _num_color


def shuffle_channel_order(im_after, channels_before, channels_after, zlims):
    ## convert and check input
    _ch_before = [str(_ch) for _ch in channels_before]
    _ch_after = [str(_ch) for _ch in channels_after]
    for _ch in _ch_after:
        if _ch not in _ch_before:
            raise ValueError(
                f"All channels in ch_after should be in ch_before, but {_ch} is found!")
    # get original order
    minz, maxz = np.sort(zlims)[:2]
    zstep = len(channels_before)
    _start_layers = minz + \
        np.array([(_z+1-minz) % zstep for _z in np.arange(zstep)], dtype=np.int)
    for _i, _s in enumerate(_start_layers):
        if _s == minz:
            _start_layers[_i] += zstep
    _index_after = [_ch_before.index(_ch) for _ch in _ch_after]
    _start_layers_after = _start_layers[np.array(_index_after)]
    _channel_order_after = np.argsort(_start_layers_after)
    for _i, _od in enumerate(_channel_order_after):
        if _od == 0:
            _channel_order_after[_i] += len(channels_after)
    _layer_order = list(np.arange(minz+1))
    while(max(_layer_order) < maxz):
        _layer_order += list(_channel_order_after+max(_layer_order))
    _layer_order += list(np.arange(max(_layer_order)+1, len(im_after)))

    im_sorted = im_after[np.array(_layer_order)]
    return im_sorted


def Save_Dax(im, filename, dtype=np.uint16, overwrite=False,
             save_info_file=False, source_dax_filename=None, save_other_files=False):
    """Function to save an np.ndarray image file into dax
    im: input image, np.ndarray
    
    """
    _im = np.array(im, dtype=dtype)
    if '.dax' not in filename:
        filename += '.dax'
    if os.path.isfile(filename) and not overwrite:
        print(f"-- file:{filename} already exists, not overwrite dax so skip.")
        return False
    else:
        # save dax
        _im.tofile(filename)
        if save_info_file:
            if source_dax_filename is None:
                raise ValueError(
                    f"If save_info_file is specified, source dax filename should be specified")
            else:
                # load old_info
                _old_info_file = source_dax_filename.replace('.dax', '.inf')
                with open(_old_info_file, 'r') as _info_hd:
                    _infos = _info_hd.readlines()
                # modify number-of-frames
                for _i, _line in enumerate(_infos):
                    if "number of frames" in _line:
                        _infos[_i] = f"{_line.split('=')[0]}= {int(_im.shape[0])}\n"
                    if "frame dimensions" in _line:
                        _infos[_i] = f"{_line.split('=')[0]}= {int(_im.shape[1])} x {int(_im.shape[2])}\n"
                    if "frame size" in _line:
                        _infos[_i] = f"{_line.split('=')[0]}= {int(_im.shape[1])*int(_im.shape[2])}\n"
                # save to new file
                _new_info_file = filename.replace('.dax', '.inf')
                with open(_new_info_file, 'w') as _out_info_hd:
                    _out_info_hd.writelines(_infos)
        if save_other_files:
            if source_dax_filename is None:
                raise ValueError(
                    f"If save_info_file is specified, source dax filename should be specified")
            related_file_list = glob.glob(
                source_dax_filename.replace('.dax', '*'))
            related_file_list = [
                _fl for _fl in related_file_list if '.dax' not in _fl and '.inf' not in _fl]
            from shutil import copyfile
            for _fl in related_file_list:
                _new_fl = filename.replace('.dax', '.'+_fl.split('.')[-1])
                copyfile(_fl, _new_fl)
        return True

