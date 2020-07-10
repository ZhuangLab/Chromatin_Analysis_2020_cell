import numpy as np
import os, sys, glob, time
import pickle
# biopython SeqRecord
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
# biopython blast
from Bio.Blast.Applications import NcbiblastnCommandline
from Bio.Blast import NCBIXML
# import library designer
from . import LibraryDesigner as ld
from .LibraryTools import fastaread, fastawrite, fastacombine
from . import _primer_folder, _readout_folder, _genome_folder
def __init__():
    pass


def Screen_probe_by_hit(library_folder, probe_source, num_probes_per_region, 
                        hit_type='genome', verbose=True):
    """Function to screen probes by number of one type of hits in probe designer class"""
    from copy import copy
    
    ## Check inputs
    if verbose:
        print(f"- Screen probes by {hit_type} counts.")
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
        from .probe import _load_probes_in_folder
        _pb_dict = _load_probes_in_folder(report_folder)
    elif isinstance(probe_source, dict):
        _pb_dict = probe_source
    # num_probes_per_region
    n_probes = int(num_probes_per_region)
    
    ## start screening
    _kept_pb_dict = {}
    for _reg_name, _pb_obj in _pb_dict.items():
        if verbose:
            print(f"-- filtering region:{_reg_name}", end=', ')
        if len(_pb_obj.pb_reports_keep) <= n_probes:
            if verbose:
                print(f"directly append {len(_pb_obj.pb_reports_keep)} probes")
            _kept_pb_dict[_reg_name] = _pb_obj
        else:
            if verbose:
                print(f"screen {hit_type} to keep {n_probes} probes")
            _hits =  [int(_pb[hit_type]) for _seq, _pb in _pb_obj.pb_reports_keep.items()]
            _kept_inds = np.argsort(_hits)[:n_probes]
            _kept_reports_dict = {_seq:_pb for _i, (_seq,_pb) \
                                  in enumerate(_pb_obj.pb_reports_keep.items()) \
                                  if _i in _kept_inds}
            _new_pb_obj = copy(_pb_obj)
            _new_pb_obj.pb_reports_keep = _kept_reports_dict
            # append to kept_pb_dict
            _kept_pb_dict[_reg_name] = _new_pb_obj
    
    return _kept_pb_dict

def split_probe_by_gene(pb_records, species_marker='gene_'):
    pb_dic = {}
    for _pb in pb_records:
        _gene = _pb.id.split(species_marker)[1].split('_')[0]
        if _gene not in pb_dic:
            pb_dic[_gene] = [_pb]
        else:
            pb_dic[_gene].append(_pb)
    return pb_dic

# check if all probes use the same primer
def _check_primer_usage(pb_records, fwd_primer, rev_primer, primer_len=20, _verbose=True):
    '''Check whether forward or reverse primer are used in all probes'''
    if _verbose:
        print("-- Checking primer usage, total probes:", len(pb_records))
    fwd_len = len(fwd_primer.seq)
    rev_len = len(rev_primer.seq[-primer_len:].reverse_complement())

    for record in pb_records:
        if record.seq[:fwd_len] != fwd_primer.seq:
            if _verbose:
                print("--- Forward primer incorrect!")
            return False
        if record.seq[-rev_len:] != rev_primer.seq[-primer_len:].reverse_complement():
            if _verbose:
                print("--- Forward primer incorrect!")
            return False
    return True  # if no error applies

# check region size is large enough and generate a dict
def _check_region_size(pb_records, species_marker='gene_', min_size=24, verbose=True):
    '''Generate a dirctionary to species probe number dic '''
    # get region size from probe names
    _probe_num_dic = {}
    _size_marker = True
    for record in pb_records:
        _reg_id = record.id.split(species_marker)[1].split('_')[0]
        if _reg_id not in list(_probe_num_dic.keys()):
            _probe_num_dic[_reg_id] = 1  # if not in key, create
        else:
            _probe_num_dic[_reg_id] += 1  # otherwise, add count
    for _reg_id, _reg_size in _probe_num_dic.items():
        if _reg_size <= min_size:
            _size_marker = False
        if verbose:
            print(f"{species_marker.split('_')[0]}: {_reg_id} -> {_reg_size}")
    return _probe_num_dic, _size_marker

# check number of readouts used for each region/speciespo
def _check_region_to_readouts(pb_records, readout_dict,
                              species_marker='gene_', total_readout_num=3, 
                              verbose=True):
    '''Generate map from region id to barcodes used in this region'''
    import re
    _reg_to_barcode = {}
    for record in pb_records:
        # region id
        reg_id = record.id.split(species_marker)[1].split('_')[0]
        # barcode ids
        stv_matches = re.findall(
            'Stv_([0-9]+?)_[a-z\]|,]', record.id, re.DOTALL)
        ndb_matches = re.findall(
            'NDB_([0-9]+?)_[a-z\]|,]', record.id, re.DOTALL)
        stv_names = ['Stv_'+str(stv_id) for stv_id in stv_matches]
        ndb_names = ['NDB_'+str(ndb_id) for ndb_id in ndb_matches]
        if reg_id not in _reg_to_barcode:
            _reg_to_barcode[reg_id] = stv_names+ndb_names
        else:
            for _name in stv_names+ndb_names:
                if _name not in _reg_to_barcode[reg_id]:
                    _reg_to_barcode[reg_id].append(_name)
    ## barcode check
    _barcode_check = True
    # barcode names
    rd_names = []
    for _type, _readouts in readout_dict.items():
        rd_names += [_r.id for _r in _readouts]
    # search through previous dictionary
    for reg, readouts in _reg_to_barcode.items():
        # sort the names
        _stvs = [_rd for _rd in readouts if 'Stv' in _rd]
        _ndbs = [_rd for _rd in readouts if 'NDB' in _rd]
        _reg_to_barcode[reg] = sorted(_stvs, key=lambda r: int(
            r.split('_')[-1])) + sorted(_ndbs, key=lambda r: int(r.split('_')[-1]))
        # check total number of readout seq used for each species
        for rd in readouts:
            if len(readouts) != total_readout_num:
                print("-- Error in barcode number for region:", reg)
                _barcode_check = False
                break
            if rd not in rd_names:
                print("-- Wrong barcode name for barcode: " +
                      str(rd)+", region: "+str(reg))
                _barcode_check = False
                break
        if verbose:
            print(f"{reg} -> {_reg_to_barcode[reg]}")

    return _reg_to_barcode, _barcode_check

# function to be called, parse probe sequence
def _parsing_probe_sequence(record, add_rand_gap=0, primer_len=20, readout_len=20, target_len=42):
    '''parse a probe sequence to acquire all readout binding sites'''
    # take in a seq record, parse the sequence and return a list of all included readouts (20mer,RC)
    readout_list = []
    _main_seq = record.seq[primer_len:-primer_len]
    # number of readout sequence in this probe
    _readout_num = (len(_main_seq) - target_len) / (primer_len+add_rand_gap)
    if _readout_num != int(_readout_num):
        raise ValueError(
            "Length of probe doesn't match given element length! ")
    # trim readouts in the back
    for i in range(int(np.ceil(_readout_num/2))):
        readout_list.append(_main_seq[-readout_len:])
        _main_seq = _main_seq[:-(readout_len+add_rand_gap)]
    # trim all readouts from the beginning
    while len(_main_seq) > target_len:
        readout_list.append(_main_seq[:readout_len])
        _main_seq = _main_seq[(readout_len+add_rand_gap):]
    return readout_list

def _finding_readout_name(readout_list, readout_dict, readout_len=20, 
                          probe_readout_num=3):
    '''Given readout list generated by parsing probe, return a list of readout names'''
    _name_list = []
    _all_readouts = []
    for _type, _readouts in readout_dict.items():
        _all_readouts += _readouts
    # loop through readouts
    for _site in readout_list:
        for readout in _all_readouts:
            if readout.seq[-readout_len:] == _site.reverse_complement():
                _name_list.append(readout.id)
                break
    # check
    if len(_name_list) != probe_readout_num:
        print("-- Number of readouts on this probe doesn't match")
        return False
    return _name_list

def _check_readout_to_region(reg_to_readout, pb_records, readout_dict, 
                             species_marker='gene_', target_len=42,
                             primer_len=20, probe_readout_num=3,
                             verbose=True):
    '''Generate map from readout id to region id, require a region_to_readout map and probe lists'''
    _readout_to_reg = {}
    prev_reg_id = ''
    if verbose:
        print(f"-- extracting probes from >", end='')
    for record in pb_records:
        reg_id = record.id.split(species_marker)[1].split('_')[0]
        if verbose:
            if prev_reg_id != reg_id:
                print(f"region:{reg_id}", end=', ')
        prev_reg_id = reg_id
        # extract readouts used in this probe
        _readout_list = _parsing_probe_sequence(record, target_len=target_len, primer_len=primer_len)
        _name_list = _finding_readout_name(_readout_list, readout_dict,
            probe_readout_num=probe_readout_num)
        _names, _name_cts = np.unique(_name_list, return_counts=True)
        for _n, _nct in zip(_names, _name_cts):
            if _n not in list(_readout_to_reg.keys()):  # create if not in dic
                _readout_to_reg[_n] = [reg_id] * _nct
            elif reg_id not in _readout_to_reg[_n]:  # otherwise, append
                _readout_to_reg[_n] += [reg_id] * _nct
    if verbose:
        print(" > Done.")
    # sort _readout_to_reg
    _readout_keys = sorted([_r for _r in _readout_to_reg if 'Stv' in _r], key=lambda r:int(r.split('_')[-1]) ) + \
                    sorted([_r for _r in _readout_to_reg if 'NDB' in _r], key=lambda r:int(r.split('_')[-1]) )
    _regions = [_readout_to_reg[_r] for _r in _readout_keys]
    _readout_to_reg = dict(zip(_readout_keys, _regions))
    
    ## check region distribution
    # invert dic from reg_to_readout
    if verbose:
        print(f"-- inverting region_to_readout dict")
    _inv_dic = {}
    for reg, readouts in sorted(reg_to_readout.items()):
        for _readout in readouts:
            if _readout not in list(_inv_dic.keys()):
                _inv_dic[_readout] = [reg]
            else:
                _inv_dic[_readout].append(reg)
    # compare
    _region_check = True
    for _readout, regs in sorted(_inv_dic.items()):
        if _readout not in list(_readout_to_reg.keys()):
            print("-- "+str(_readout)+" not in readout_to_region dic!")
            _region_check = False
            break
        else:
            for _reg in regs:
                if _reg not in _readout_to_reg[_readout]:
                    print("-- "+str(_readout)+" and region"+str(_reg) +
                          " not compatible with readout_to_region dic!")
                    _region_check = False
            if _region_check == False:
                break
        if verbose:
            print(f"{_readout} -> {regs}")
    return _readout_to_reg, _region_check


def _construct_internal_map(pb_records, library_folder, word_size=17, save=True, save_filename='probe_table', verbose=True):
    '''Using functions in LibraryDesign, compute an internal khmer map'''
    # initialize internal map
    _full_savename = os.path.join(
        library_folder, save_filename)+'_'+str(word_size)
    _int_map = ld.countTable(save_file=_full_savename, word=word_size)
    # feed in sequences
    for _record in pb_records:
        _int_map.consume(str(_record.seq))
    _int_map.complete()
    # save
    if save:
        if verbose:
            print(
                f"-- saving internal {word_size}-mer map to file:{_full_savename}.npz")
        _int_map.save()
    return _int_map

# check number of readouts in probes
def _check_readout_in_probes(readout_to_reg, reg_size_dic, int_map,
                             readout_dict, word_size=17,
                             readout_len=20, max_internal_hits=50):
    '''Check readout appearance in probes, whether that match readout_to_region scheme'''
    # load all readouts
    _all_readouts = []
    for _type, _readouts in readout_dict.items():
        _all_readouts += _readouts
    # get
    _readout_in_probes = {}
    for readout_name, regs in sorted(readout_to_reg.items()):
        readout = None
        for _r in _all_readouts:
            if readout_name == _r.id:
                readout = _r
                break
        if readout is None:
            return False
        readout_hits = int_map.get(
            str(readout.seq[-readout_len:].reverse_complement()).upper())
        regs, reg_cts = np.unique(regs, return_counts=True)
        readout_in_probe = 0
        for reg, ct in zip(regs, reg_cts):
            readout_in_probe += reg_size_dic[reg] * ct * (readout_len-word_size+1)

        if readout_hits - readout_in_probe > max_internal_hits:
            print(readout_hits, regs, readout_in_probe, max_internal_hits)
            print("-- readout: "+str(readout.id) +
                    " has more off-target than threshold!")
            return {}, False
        _readout_in_probes[readout_name] = readout_in_probe
    return _readout_in_probes, True

# check target-seq between probes
def _check_between_probes(pb_records, int_map, _max_internal_hits=50, 
                          primer_len=20, target_len=30, readout_len=20, 
                          add_rand_gap=0,
                          _make_plot=False, _verbose=True):
    """Function to check k-mer appreance between probes"""
    def __extract_targeting_sequence(record, primer_len=primer_len, target_len=target_len,
                                     readout_len=readout_len, add_rand_gap=add_rand_gap):
        return record.seq[-(readout_len+add_rand_gap)*2-primer_len-target_len: -(readout_len+add_rand_gap)*2-primer_len]
    _internal_hits = []
    _kept_pb_records = []
    _removed_count = 0
    for record in pb_records:
        target_seq = str(__extract_targeting_sequence(record)).upper()
        _rec_hits = int_map.get(target_seq) + int_map.get(target_seq, rc=True)
        _internal_hits.append(_rec_hits) 
        if _rec_hits <= _max_internal_hits:
            _kept_pb_records.append(record)
        else:
            _removed_count += 1
            print(f"--- Sequence:{record.id} got hits:{_rec_hits}, dumped")

    if _make_plot:
        import matplotlib.pyplot as plt 
        plt.figure()
        plt.hist(_internal_hits)
        plt.show()
    if _verbose:
        print("-- total probes removed by internal screening:", _removed_count)
    return _kept_pb_records, _removed_count


def acquire_blast_counts(blast_record, hard_thres=30, soft_thres=17, verbose=False):
    '''filter for genome blast record
    Input:
        blast_record: xml format of blast result, generated by NCBIWWW.qblast or NCBIblastnCommandLine
        hard_threshold: only one hsp_score match larger than this is allowed, int
        soft_threshold: hsp_score larger than this should be counted, int
        verbose: it says a lot of things!
    Output:
        _hard_count: hits for hard threshold
        _soft_count: hits for soft threshold'''
    _hard_count, _soft_count = 0, 0
    if verbose:
        print('- '+blast_record.query_id, ' hits =',
              len(blast_record.alignments))
    # extract information
    hsp_scores = []
    hsp_aligns = []
    for alignment in blast_record.alignments:
        for hsp in alignment.hsps:
            hsp_scores.append(hsp.score)
            hsp_aligns.append(hsp.align_length)
    # hard threshold
    keep_hard = [hsp_align >= hard_thres and hsp_score >=
                 hard_thres for hsp_score, hsp_align in zip(hsp_scores, hsp_aligns)]
    _hard_count = sum(keep_hard)
    if verbose:
        print("-- hard count:", _hard_count)

    # soft threshold
    keep_soft = [hsp_align >= soft_thres and hsp_score >=
                 soft_thres for hsp_score, hsp_align in zip(hsp_scores, hsp_aligns)]
    _soft_count = sum(keep_soft)
    if verbose:
        print("-- soft count:", _soft_count)

    return _hard_count, _soft_count


def Blast_probes(probes, library_folder, blast_subfolder='blast',
                 num_threads=12, blast_db=os.path.join(_genome_folder, 'hg38'),
                 blast_evalue=500, blast_word_size=10,
                 overwrite=True, verbose=True):
    """Function to blast probes"""
    ## check inputs
    if verbose:
        print(f"- Blast probe against {blast_db}")
    if isinstance(probes, list):
        pb_records = probes
    elif isinstance(probes, str):
        pb_records = []
        if os.path.isfile(probes):
            with open(probes, 'r') as _handle:
                for _record in SeqIO.parse(_handle, "fasta"):
                    pb_records.append(_record)
    if verbose:
        print(f"-- number of probes loaded: {len(pb_records)}")
    # blast folder
    blast_folder = os.path.join(library_folder, blast_subfolder)
    if not os.path.exists(blast_folder):
        os.makedirs(blast_folder)
    # others
    num_threads = int(num_threads)

    ## split probes
    pb_dic = split_probe_by_gene(pb_records)
    # loop through each gene and do blast
    for _gene, _pbs in sorted(pb_dic.items()):
        if verbose:
            print(f"--- processing region: {_gene} with {len(_pbs)} probes")
            _start = time.time()
        _infile = os.path.join(blast_folder, f'probe_gene_{_gene}.fasta')
        _outfile = os.path.join(blast_folder, f'blast_gene_{_gene}.xml')
        if overwrite or not os.path.isfile(_outfile):
            if verbose:
                print(f"--- *writing file: {_infile}")
            # save these number of probes into temp.fasta
            with open(_infile, "w") as _output_handle:
                SeqIO.write(_pbs, _output_handle, "fasta")
            # blast
            if verbose:
                print(f"-- *blasting region: {_gene}")
            # Run BLAST and parse the output as XML
            output = NcbiblastnCommandline(query=_infile,
                                           num_threads=num_threads,
                                           db=blast_db,
                                           evalue=blast_evalue,
                                           word_size=blast_word_size,
                                           out=_outfile,
                                           outfmt=5)()[0]
            if verbose:
                print(
                    f"--- total time for blast {_gene}: {time.time()-_start}")
                    
# screen blast results
def Screening_Probes_by_Blast(library_folder, probe_per_region, keep_mode='center', 
                              blast_subfolder='blast', probe_subfolder='.',                 
                              probe_filename='filtered_full_probes.fasta',
                              hard_thres=40, soft_thres=20, soft_count_th=30,
                              smallest_region_ratio=0.75,
                              save=True, save_filename='blast_full_probes.fasta',
                              verbose=True,):
    '''Read blast results in blast folder and probe in filtered_probes, keep'''
    # folders
    _blast_folder = library_folder + os.sep + blast_subfolder
    _probe_folder = library_folder + os.sep + probe_subfolder
    # load probes
    _probes = []
    with open(_probe_folder+os.sep+probe_filename, 'r') as _handle:
        for _record in SeqIO.parse(_handle, "fasta"):
            _probes.append(_record)
    if verbose:
        print("- Number of probes loaded:", len(_probes))
    # parse loaded probes by region
    _pb_dic = split_probe_by_gene(_probes)
    if verbose:
        print("- Number of regions in this library:", len(_pb_dic))
    # dictionary to store whether keep this probe
    _keep_dic = {}  # whether keep because of blast only
    _kept_pb_dic = {}
    _hard_count_list = []
    _soft_count_list = []
    # loop through all regions
    for _reg, _pbs in _pb_dic.items():
        if verbose:
            print("-- checking probes in region:", _reg)
        _keep_dic[_reg] = np.ones(
            len(_pbs), dtype=np.bool)  # initialize with True
        # parse blast result of this region
        blast_records = NCBIXML.parse(
            open(_blast_folder+os.sep+'blast_gene_'+str(_reg)+'.xml', 'r'))
        # loop through each probe in this region
        _hard_cts, _soft_cts = [], []
        for _pbid, blast_record in enumerate(blast_records):
            _hc, _sc = acquire_blast_counts(blast_record, hard_thres=hard_thres,
                                            soft_thres=soft_thres, verbose=False)
            _hard_cts.append(_hc)
            _soft_cts.append(_sc)
            if _hc > 2 or _hc < 1:  # if this probe has no hit, or more than 2 hits, remove
                _keep_dic[_reg][_pbid] = False
                print(
                    f"--- gene={_reg}, id={_pbid} removed by hard count = {_hc}")
                continue
            # if this probe has too many soft counts (20mer hits)
            if _sc > soft_count_th:
                _keep_dic[_reg][_pbid] = False
                print(
                    f"--- gene={_reg}, id={_pbid} removed by soft count = {_sc}")
                continue
        # after looped through this region, check the hard counts
        _hard_cts = np.array(_hard_cts)
        _soft_cts = np.array(_soft_cts)
        _hard_count_list.append(_hard_cts)
        _soft_count_list.append(_soft_cts)
        if verbose:
            print("--- number of probes:", len(_pbs), ", kept by blast:",
                  sum(_keep_dic[_reg]), ", if remove dups:", sum(_keep_dic[_reg] * (_hard_cts == 1)))
        # check duplicated probes
        if sum(_keep_dic[_reg] * (_hard_cts == 1)) / float(sum(_keep_dic[_reg])) >= smallest_region_ratio and sum(_keep_dic[_reg]) >= smallest_region_ratio*probe_per_region:
            print('--- remove duplicated probes')
            _keep_dic[_reg] = _keep_dic[_reg] * (_hard_cts == 1)

        # generate list of kept probes
        _kept_pbs = [_pb for _pb, _k in zip(_pbs, _keep_dic[_reg]) if _k]

        # keep the center of this region
        if sum(_keep_dic[_reg]) > probe_per_region:
            if keep_mode == 'center':
                if verbose:
                    print("--- keep probes from beginning")
                _start, _end = _pbs[0].id.split(
                    ':')[1].split('_')[0].split('-')
                _start, _end = int(_start), int(_end)
                _reg_len = np.abs(_end - _start)
                _kept_center_pbs = []
                for _pb in sorted(_kept_pbs, key=lambda p: np.abs(int(p.id.split('pb_')[1].split('_')[0])-_reg_len/2)):
                    _kept_center_pbs.append(_pb)
                    if len(_kept_center_pbs) >= probe_per_region:
                        break
                _kept_pb_dic[_reg] = sorted(_kept_center_pbs, key=lambda p: int(
                    p.id.split('pb_')[1].split('_')[0]))
            elif keep_mode == 'front':
                _kept_pbs = _kept_pbs[:probe_per_region]
                _kept_pb_dic[_reg] = _kept_pbs
        else:
            _kept_pb_dic[_reg] = sorted(_kept_pbs, key=lambda p: int(
                p.id.split('pb_')[1].split('_')[0]))
        if verbose:
            print('-- number of probes kept for this region:',
                  len(_kept_pb_dic[_reg]))

    # SUMMARIZE
    _kept_probe_list = []
    if verbose:
        print("- summarize")
    for _reg, _pbs in _kept_pb_dic.items():
        if verbose:
            print("-- region:", _reg, ", number of probes:", len(_pbs))
        _kept_probe_list += _pbs

    print("- Number of probes kept:", len(_kept_probe_list))

    if save:
        if verbose:
            print("- Saving to file:", _probe_folder + os.sep + save_filename)
        with open(_probe_folder + os.sep + save_filename, 'w') as _output_handle:
            SeqIO.write(_kept_probe_list, _output_handle, 'fasta')

    return _kept_probe_list, _keep_dic, _hard_count_list, _soft_count_list
