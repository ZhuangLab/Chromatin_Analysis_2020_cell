import os, glob, sys, time
import pickle
import numpy as np

# biopython
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from Bio.SeqRecord import SeqRecord
from Bio.Blast.Applications import NcbiblastnCommandline
from Bio.Blast import NCBIXML

# other packages
from . import LibraryDesigner as ld
from . import LibraryTools as lt 
# shared variables
from . import _rand_seq_generator
from . import _primer_folder, _readout_folder, _genome_folder
from . import quality_check
# Load probes
def _load_probes_in_folder(report_folder, pb_postfix='.pbr', save_folder=None):

    pb_dict = {}
    report_files = glob.glob(os.path.join(report_folder, '*'+pb_postfix))
    for _file in report_files:
        reg_name = os.path.basename(_file).split(pb_postfix)[0]
        pbde = ld.pb_reports_class()
        pbde.load_pbr(_file)
        if save_folder is not None:
            _out_file = os.path.join(save_folder, os.path.basename(_file))
            pbde.save_file = _out_file
        pb_dict[reg_name] = pbde

    return pb_dict

# Screen probes
def Screen_fasta_against_fasta(input_fasta, ref_fasta, word_size=17, allowed_hits=8,
                               check_rc=True, save=True, save_folder=None,
                               overwrite=False, return_kept_flag=False, verbose=True):
    """Function to screen a given fasta against another reference fasta"""
    ## Check inputs
    if verbose:
        print(f"- Screen fasta:{input_fasta} against given fasta file:{ref_fasta}")
    if not os.path.isfile(input_fasta):
        raise IOError(f"Input fasta:{input_fasta} is not a file.")
    if not os.path.isfile(ref_fasta):
        raise IOError(f"Reference fasta:{ref_fasta} is not a file.")
    word_size = int(word_size)
    allowed_hits = int(allowed_hits)
    if save_folder is None:
        save_folder = os.path.dirname(input_fasta)+'_filtered'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        if verbose:
            print(f"-- create {save_folder} to store filter probes")
    
    ## load probes
    with open(input_fasta, 'r') as _handle:
        _input_records = []
        for _record in SeqIO.parse(_handle, "fasta"):
            _input_records.append(_record)

    # construct table for ref_fasta
    if verbose:
        print(f"-- constructing reference table for fasta file")
    _ref_names, _ref_seqs = ld.fastaread(ref_fasta, force_upper=True)
    _ref_table = ld.OTmap(_ref_seqs, word_size, use_kmer=True)

    


def Screen_probe_against_fasta(report_folder, ref_fasta, word_size=17, allowed_hits=8,
                               check_rc=True, save=True, save_folder=None,
                               overwrite=False, return_kept_flag=False, verbose=True):
    """Function to screen probes in one folder against a given fasta file,
    Inputs:
        report_folder: folder for probe reports, str of path
        ref_fasta: filename for reference fasta file to screen against, string of file path
        word_size: word_size used for probe screening, int (default: 17)
        allowed_hits: allowed hits for one probe in the fasta, int (default: 8)
        check_rc: whether check reverse-complement of the probe, bool (default: True)
        save: whether save result probe reports, bool (default: True)
        save_folder: folder to save selected probes, string of path (default: None, which means +'_filtered')
        overwrite: whether overwrite existing result probe reports, bool (default: False)
        return_kept_flag: whether return flags for whether keeping the record, bool (default:False)
        verbose: say something!, bool (default: True)
    """
    ## Check inputs
    if verbose:
        print(f"- Screen probes against given fasta file:{ref_fasta}")
    if not os.path.exists(report_folder):
        raise IOError(f"Inpout report_folder not exist!")
    if not os.path.isfile(ref_fasta):
        raise IOError(f"Reference fasta:{ref_fasta} is not a file.")
    word_size = int(word_size)
    allowed_hits = int(allowed_hits)
    if save_folder is None:
        save_folder = report_folder+'_filtered'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        if verbose:
            print(f"-- create {save_folder} to store filter probes")
    ## load probes
    _pb_dict = _load_probes_in_folder(report_folder)
    # screen savefile
    for _reg_name, _pb_obj in _pb_dict.items():
        if not overwrite and os.path.exists(_pb_obj.save_file):
            del(_pb_dict[_reg_name])

    if len(_pb_dict) == 0:
        if verbose:
            print(f"-- no probe loaded, exit.")
        return {}

    # construct table for ref_fasta
    if verbose:
        print(f"-- constructing reference table for fasta file")
    _ref_names, _ref_seqs = ld.fastaread(ref_fasta, force_upper=True)
    _ref_table = ld.OTmap(_ref_seqs, word_size, use_kmer=True)

    # filter probes
    _filtered_pb_dict = {_reg_name: _pb_obj for _reg_name,
                         _pb_obj in _pb_dict.items()}
    _kept_flag_dict = {_reg_name: [] for _reg_name in _pb_dict}

    for _reg_name, _pb_obj in _filtered_pb_dict.items():
        _seqs = list(_pb_obj.pb_reports_keep.keys())
        _seq_lens = [len(_seq) for _seq in _seqs]
        if check_rc:
            _hits = [_ref_table.get(_seq, rc=True) +
                     _ref_table.get(_seq, rc=False)
                     for _seq in _seqs]
        else:
            _hits = [_ref_table.get(_seq, rc=False) for _seq in _seqs]
        _keep_filter = [_h <= allowed_hits for _h in _hits]
        # save
        _kept_pbs = {_s: _info for (_s, _info), _keep in zip(
            _pb_obj.pb_reports_keep.items(), _keep_filter) if _keep}

        _filtered_pb_dict[_reg_name].pb_reports_keep = _kept_pbs
        _kept_flag_dict[_reg_name] = np.array(_keep_filter, dtype=np.bool)
        if verbose:
            print(
                f"--- {len(_kept_pbs)} / {len(_keep_filter)} probes kept for {_reg_name}")
        if save:
            _filtered_pb_dict[_reg_name].save_pbr()
            _filtered_pb_dict[_reg_name].save_csv()

    if return_kept_flag:
        return _filtered_pb_dict, _kept_flag_dict
    else:
        return _filtered_pb_dict


# load
def load_readouts(_num_readouts, _type='NDB', _num_colors=3,
                  _readout_folder=_readout_folder, _start_id=0, _verbose=True):
    """Function to load readouts into a list"""
    # get target files
    _readout_files = glob.glob(os.path.join(
        _readout_folder, _type) + '_*.fasta')
    if len(_readout_files) < _num_colors:
        raise IOError(
            "Not enough readout files in given readout folder compared to num-colors specified")
    _num_per_color = int(np.ceil(_num_readouts / _num_colors))
    # load readouts
    _multi_readout_lists = []
    for _rd_fl in _readout_files[:_num_colors]:
        _readout_list = []
        with open(_rd_fl, 'r') as _rd_handle:
            for _readout in SeqIO.parse(_rd_handle, "fasta"):
                _readout_list.append(_readout)
        _multi_readout_lists.append(
            _readout_list[_start_id: _start_id+_num_per_color])
    # sort and save
    _selected_list = []
    while len(_selected_list) < _num_readouts:
        _selected_list.append(
            _multi_readout_lists[len(_selected_list) % _num_colors].pop(0))
    # return
    return _selected_list


def load_primers(_picked_sets, _primer_folder=_primer_folder,
                 _primer_file_tag='_keep.fasta', _verbose=True):
    """Function to extract a pair of primers"""
    if not isinstance(_picked_sets, list):
        raise ValueError("kwd _picked_sets should be a list!")
    _primer_files = glob.glob(os.path.join(
        _primer_folder, '*_primers'+_primer_file_tag))
    if len(_primer_files) != 2:
        raise IOError("_primer_files have more or less two hits")
    _primer_sets = []
    for _pr_fl, _picked_id in zip(_primer_files, _picked_sets):
        with open(_pr_fl, 'r') as _pr_handle:
            for _primer in SeqIO.parse(_pr_handle, "fasta"):
                if int(_picked_id) == int(_primer.id.split('_')[-1]):
                    if _verbose:
                        print("- Picked primer:", _primer)
                    _primer_sets.append(_primer)
    return _primer_sets


def _assemble_single_probe(_target, _readout_list, _fwd_primer, _rev_primer,
                           _primer_len=20, _readout_len=20, _target_len=42,
                           _add_random_gap=0):
    """Assemble one probe sequence
    All required inputs are SeqRecords """
    # initialize final sequence with fwd primer
    _seq = _fwd_primer[-_primer_len:]
    # append half of readouts on 5'
    for _i in range(int(len(_readout_list)/2)):
        _seq += _readout_list[_i][-_readout_len:].reverse_complement()
        _seq += _rand_seq_generator(_add_random_gap)
    # append target region
    _seq += _target[-_target_len:]
    _seq += _rand_seq_generator(_add_random_gap)
    # append other half of readouts on 3'
    for _i in range(int(len(_readout_list)/2), len(_readout_list)):
        _seq += _readout_list[_i][-_readout_len:].reverse_complement()
        if _i < len(_readout_list)-1:
            _seq += _rand_seq_generator(_add_random_gap)
    # append reverse_complement
    _seq += _rev_primer[-_primer_len:].reverse_complement()
    _seq.description = ''
    return _seq


def _assemble_single_probename(_pb_info, _readout_name_list, _pb_id):
    """Assemble one probe name by given probe info
    _pb_info is one of values in pb_designer.pb_reports_keep"""
    _name = [_pb_info['reg_name'].split('_')[0],
             'gene_'+_pb_info['reg_name'].split('_')[2],
             'pb_'+str(_pb_id),
             'pos_'+str(_pb_info['pb_index']),
             'readouts_[' + ','.join(_readout_name_list) + ']']
    return '_'.join(_name)

# function to assemble probes in the whole library
def Assemble_probes(library_folder, probe_source, gene_readout_dict, readout_dict, primers,
                    rc_targets=False, add_random_gap=0,
                    primer_len=20, readout_len=20, target_len=42,
                    save=True, save_name='candidate_probes.fasta', save_folder=None,
                    overwrite=True, verbose=True):
    """Function to Assemble_probes by given probe_soruce, gene_readout_dict, readout_dict and primers,
    Inputs:
        library_folder: path to the library, str of path
        probe_source: source for probes, str or list of SeqRecords
        gene_readout_dict: dict of gene/region -> list of readouts used
        readout_dict: dict of readout_type ('u'|'c'|'m') -> list of readout SeqRecords
        primers: list of two SeqRecords for forward and reverse primers, list of SeqRecords
        rc_targets: whether reverse-complement target sequences, bool (default: False)
        add_random_gap: number of random sequence added between readout biding sites, int (default: 0)
        save: whether save result probes as fasta, bool (default: True)
        save_name: file basename of saved fasta file, str (default: 'candidate_probes.fasta')
        save_folder: folder for saved fasta file, str (default: None, which means library_folder)
        overwrite: whether overwrite existing file, bool (default: True)
        verbose: say something!, bool (default: True)
    Outputs:
        cand_probes: list of probes that assembled, list of SeqRecords
        readout_summary: summary dict of readout used in every region, dict of str -> list
    """
    ## Check inputs
    if verbose:
        print(f"- Assemble probes by given target sequences, readouts and primers.")

    if not os.path.isdir(library_folder):
        raise ValueError(
            f"Wrong input :{library_folder}, should be path to library folder")
    # probe source
    if isinstance(probe_source, str):
        if not os.path.isdir(probe_source):
            report_folder = os.path.join(library_folder, probe_source)
        else:
            report_folder = probe_source
        if not os.path.isdir(report_folder):
            raise ValueError(
                f"Wrong input :{report_folder}, should be path to probes")
        # load probes
        _pb_dict = _load_probes_in_folder(report_folder)
    elif isinstance(probe_source, dict):
        _pb_dict = probe_source

    # gene_readout_dict and readout_dict
    if not isinstance(gene_readout_dict, dict):
        raise TypeError(f"Wrong input type for gene_readout_dict, \
                        should be dict but {type(gene_readout_dict)} is given")
    if not isinstance(readout_dict, dict):
        raise TypeError(f"Wrong input type for readout_dict, \
                        should be dict but {type(readout_dict)} is given")
    # check readout types
    _readout_types = []
    for _reg_name, _readout_markers in gene_readout_dict.items():
        for _mk in _readout_markers:
            if _mk[0] not in _readout_types:
                _readout_types.append(_mk[0])
                if _mk[0] not in readout_dict:
                    raise KeyError(
                        f"{_mk[0]} type readout is not included in readout_dict")
    if verbose:
        print(f"-- included readout types: {_readout_types}")
    if save_folder is None:
        save_folder = library_folder
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    if '.fasta' not in save_name:
        save_name += '.fasta'

    ## start asembling probes
    cand_probes = []  # initialize
    readout_summary = {_t: {_reg_name: []
                            for _reg_name in _pb_dict} for _t in _readout_types}
    # primers shared by the library
    fwd_primer, rev_primer = primers
    for _reg_name, _pb_obj in _pb_dict.items():
        _reg_readout_info = gene_readout_dict[_reg_name]
        _reg_readouts = []
        _reg_readout_names = []
        for _mk in _reg_readout_info:
            _type = _mk[0]
            _ind = int(_mk[1:])
            _sel_readout = readout_dict[_type][_ind]
            _reg_readouts.append(_sel_readout)
            _reg_readout_names.append(_sel_readout.id + '_' + _type)
            if _sel_readout not in readout_summary[_type][_reg_name]:
                readout_summary[_type][_reg_name].append(_sel_readout)
        if isinstance(_pb_obj, ld.pb_reports_class):
            if verbose:
                print(
                    f"--- assemblying {len(_pb_obj.pb_reports_keep)} probes in region: {_reg_name}")
            for _i, (_seq, _info) in enumerate(_pb_obj.pb_reports_keep.items()):
                if isinstance(_seq, bytes):
                    _seq = _seq.decode()
                if rc_targets:
                    _target = SeqRecord(
                        Seq(_seq), id=_info['name']).reverse_complement()
                else:
                    _target = SeqRecord(Seq(_seq), id=_info['name'])
                _probe = _assemble_single_probe(_target, _reg_readouts, fwd_primer, rev_primer,
                                                _primer_len=primer_len, _readout_len=readout_len, 
                                                _target_len=target_len, _add_random_gap=add_random_gap)
                _name = _assemble_single_probename(_info, _reg_readout_names, _i)
                _probe.id = _name
                _probe.name, _probe.description = '', ''
                cand_probes.append(_probe)
        elif isinstance(_pb_obj, list):
            if verbose:
                print( f"--- assemblying {len(_pb_obj)} probes in region: {_reg_name}")
            # case that a list is provided
            for _i, _info in enumerate(_pb_obj):
                # target
                if rc_targets:
                    _target = SeqRecord(Seq(_info['sequence'])).reverse_complement()
                else:
                    _target = SeqRecord(Seq(_info['sequence']))
                # probe
                _probe = _assemble_single_probe(_target, _reg_readouts, fwd_primer, rev_primer,
                                                _add_random_gap=add_random_gap)
                _name = _info['region']+'_gene_'+_info['gene']+'_pb_'+str(_i) +\
                       '_pos_'+str(_info['position'])+'_readouts_[' + \
                       ','.join(_reg_readout_names) + ']'
                _probe.id = _name
                _probe.name, _probe.description = '', ''
                cand_probes.append(_probe)
    if verbose:
        print(f"-- {len(cand_probes)} probes assembled in total.")

    if save:
        # save cand_probe
        _save_filename = os.path.join(save_folder, save_name)
        if not os.path.isfile(_save_filename) or overwrite:
            if verbose:
                print(
                    f"-- saving {len(cand_probes)} probes into file:{_save_filename}")
            with open(_save_filename, 'w') as _output_handle:
                SeqIO.write(cand_probes, _output_handle, "fasta")
            # save readout_summary, coupled with save_filename
            _readout_summary_filename = os.path.join(
                save_folder, 'readout_summary.pkl')
            if verbose:
                print(
                    f"-- saving readout_summary into file:{_readout_summary_filename}")
            pickle.dump(readout_summary, open(_readout_summary_filename, 'wb'))
        else:
            print(f"cand_probe file:{_save_filename} already exists, skip.")

    return cand_probes, readout_summary



def Select_subset(input_probes, select_num=None, select_size=None, 
                  region_marker='reg_', position_marker='pos_',
                  select_mode='mid', return_dict=False, 
                  save=True, save_folder=None, save_name=None, verbose=True):
    """Function to select a subset of a given library"""
    ## check inputs
    if verbose:
        print(f"- Start selecting sub_library, n={select_num},size={select_size}")
    if isinstance(input_probes, str):
        if os.path.isfile(input_probes):
            with open(input_probes, 'r') as _handle:
                _pb_records = []
                for _record in SeqIO.parse(_handle, "fasta"):
                    _pb_records.append(_record)
        else:
            raise IOError(f"Input file:{input_probes} not exists!")
    elif isinstance(input_probes, list) and isinstance(input_probes[0], SeqRecord):
        _pb_records = input_probes
    else:
        raise TypeError(f"Wrong input type of input_probes, should be list of SeqRecords or string of file-path")
    if select_num is None and select_size is None:
        raise ValueError(f"At least one of select_num and select_size should be given. ")
    # probe dict
    pb_dic = quality_check.split_probe_by_gene(_pb_records, species_marker=region_marker)
    # keep record of region length
    _reg_size_dic = {}
    # sort probes and also check region size
    for _reg_id, _pbs in pb_dic.items():
        _sorted_pbs = sorted(_pbs, key=lambda v:int(v.id.split(position_marker)[1].split('_')[0]))
        pb_dic[_reg_id] = _sorted_pbs
        _start, _end = _pbs[0].id.split(':')[1].split('_')[0].split('-')
        _start, _end = int(_start), int(_end)
        _reg_size_dic[_reg_id] = abs(_end - _start)
    # select probes
    sel_pb_dic = {_reg_id:[] for _reg_id in pb_dic}
    for _reg_id, _pbs in pb_dic.items():
        _pb_coords = np.array([int(_p.id.split(position_marker)[1].split('_')[0]) for _p in _pbs])
        
        if select_mode == 'mid':
            _metric = np.abs(_pb_coords - _reg_size_dic[_reg_id]/2)
            if select_size is not None:
                _sel_inds = np.where(_metric <= select_size/2)[0]
                if select_num is not None and len(_sel_inds) < select_num:
                    _sel_inds = np.argsort(_metric)[:select_num]
            elif select_num is not None:
                _sel_inds = np.argsort(_metric)[:select_num]

        elif select_mode == 'left':
            _metric = _pb_coords
            if select_size is not None:
                _sel_inds = np.where(_metric <= select_size/2)[0]
                if select_num is not None and len(_sel_inds) < select_num:
                    _sel_inds = np.argsort(_metric)[:select_num]
            elif select_num is not None:
                _sel_inds = np.argsort(_metric)[:select_num]
        
        elif select_mode == 'right':
            _metric = _reg_size_dic[_reg_id] - _pb_coords
            if select_size is not None:
                _sel_inds = np.where(_metric <= select_size/2)[0]
                if select_num is not None and len(_sel_inds) < select_num:
                    _sel_inds = np.argsort(_metric)[:select_num]
            elif select_num is not None:
                _sel_inds = np.argsort(_metric)[:select_num]
                
        else:
            raise ValueError(f"Wrong input select_mode. ")
        # select probes
        sel_pb_dic[_reg_id] = [_p for _i,_p in enumerate(_pbs) if _i in _sel_inds.astype(np.int)]
        if verbose:
            print(f"--- {len(sel_pb_dic[_reg_id])} probes kept for region:{_reg_id}")
    
    # summarize into list of pb_records
    _sel_pb_records = []
    for _reg_id, _pbs in sel_pb_dic.items():
        _sel_pb_records += _pbs
    # save
    if save:
        if isinstance(input_probes, str):
            save_basename = os.path.basename(input_probes)
        else:
            save_basename = 'sub_probes.fasta'
        if isinstance(input_probes, str) and save_folder is None:
            save_folder = os.path.dirname(input_probes)
        elif save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
        else:
            raise ValueError(f"save_folder is not given!")
        # add postfix
        if select_num is not None:
            save_basename = save_basename.replace('.fasta', f'_n_{select_num}.fasta')
        if select_size is not None:
            save_basename = save_basename.replace('.fasta', f'_size_{select_size}.fasta')
        
        save_filename = os.path.join(save_folder, save_basename)

        with open(save_filename, 'w') as _output_handle:
            if verbose:
                print(f"-- saving to file: {save_filename} with {len(_sel_pb_records)} probes")
            SeqIO.write(_sel_pb_records, _output_handle, "fasta")
            
    if return_dict:
        return sel_pb_dic
    else:

        return _sel_pb_records


def Replace_primers(input_probes, primers, primer_len=20, primer_nametag='primer_',
                    save=True, save_folder=None, save_name=None, 
                    verbose=True):
    """Function to replace primer set in given library"""
    if verbose:
        print(f"- Start replacing primers")
    if isinstance(input_probes, str):
        if os.path.isfile(input_probes):
            with open(input_probes, 'r') as _handle:
                _pb_records = []
                for _record in SeqIO.parse(_handle, "fasta"):
                    _pb_records.append(_record)
        else:
            raise IOError(f"Input file:{input_probes} not exists!")
    elif isinstance(input_probes, list) and isinstance(input_probes[0], SeqRecord):
        _pb_records = input_probes
    else:
        raise TypeError(f"Wrong input type of input_probes, should be list of SeqRecords or string of file-path")
    # get primers
    fwd_primer, rev_primer = primers
    fwd_seq = fwd_primer[-primer_len:].seq
    rev_seq = rev_primer[-primer_len:].reverse_complement().seq
    
    # update primers
    _updated_records = []
    for _pb in _pb_records:
        _new_pb = _pb
        _new_pb.seq = fwd_seq + _pb.seq[primer_len:-primer_len] + rev_seq
        _new_pb.name, _new_pb.description = '',''
        _updated_records.append(_new_pb)
        if primer_nametag in _pb.id:
            _new_pb.id = _pb.id.split(primer_nametag)[0]+primer_nametag+\
                f"[{fwd_primer.id.split('_')[-1]},{rev_primer.id.split('_')[-1]}]"+\
                _pb.id.split(_pb.id.split(primer_nametag)[1].split('_')[0])[1]
    if save:
        if isinstance(input_probes, str):
            save_basename = os.path.basename(input_probes)
        else:
            save_basename = 'probes_new.fasta'
        save_basename = save_basename.replace('.fasta', f"_primer_{fwd_primer.id.split('_')[-1]}_{rev_primer.id.split('_')[-1]}.fasta")
        if isinstance(input_probes, str) and save_folder is None:
            save_folder = os.path.dirname(input_probes)
        elif save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
        else:
            raise ValueError(f"save_folder is not given!")

        save_filename = os.path.join(save_folder, save_basename)
        with open(save_filename, 'w') as _output_handle:
            if verbose:
                print(f"-- saving to file: {save_filename} with {len(_updated_records)} probes")
            SeqIO.write(_updated_records, _output_handle, "fasta")

    return _updated_records