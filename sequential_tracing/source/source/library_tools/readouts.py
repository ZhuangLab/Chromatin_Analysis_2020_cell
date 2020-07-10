import os, glob, sys,time
import numpy as np
import matplotlib.pyplot as plt
# biopython imports
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from Bio.SeqRecord import SeqRecord
from Bio.Blast.Applications import NcbiblastnCommandline
from Bio.Blast import NCBIXML

# other packages
from . import LibraryDesigner as ld
# shared functions
from . import _adaptor_site_names
from . import _primer_folder, _readout_folder, _genome_folder

def __init__():
    pass
##---------------------------------------------------------------------------##
# Readouts
def Extend_Readout(input_seq, target_len=30, add_5p=True):
    '''Function to extend existing short readout to longer one by generation of random sequence
    Inputs:
        input_seq: input short readout, Bio.Seq.Seq object
        target_len: length of target readouts, int
        add_5p: the direction to patch new random seq, bool, default is patch at 5p
    Output:
        out_seq: extended readout, Bio.Seq.Seq object'''
    # imports
    from random import choice
    # Sanity check
    if len(input_seq) >= target_len:
        raise ValueError('input seq length doesnot match target length!')
        out_seq = input_seq[:target_len]
    # Start processing readout
    dna_alphabet = ['A','C','G','T']
    input_seq_string = str(input_seq)
    # first added base should be A/T
    if add_5p:
        input_seq_string = choice(['A','T'])+input_seq_string
    else:
        input_seq_string += choice(['A','T'])
    # random for the rest
    for i in range(target_len - len(input_seq) - 1):
        if add_5p:
            input_seq_string = choice(dna_alphabet)+input_seq_string
        else:
            input_seq_string += choice(dna_alphabet)
    out_seq = Seq(input_seq_string, IUPAC.unambiguous_dna)
    return out_seq


def Filter_Readout(input_seq, 
                   GC_percent=[0.4, 0.6], 
                   max_consecutive=4, 
                   max_rep=6, 
                   C_percent=[0.22, 0.28], 
                   blast_hsp_thres=10.0,
                   readout_folder=_readout_folder,
                   blast_ref='cand_readouts.fasta',
                   verbose=False):
    '''Filter a readout by defined criteria
    Inputs:
    input_seq: the readout sequence, Bio.Seq.Seq object 
    check_GC: whether check gc content, list of two values or False ([0.4, 0.6])
    max_consecutive: maximum allowed consecutive bases, int (4)
    max_rep: maximum replicated sequence in a readout allowed, int (6)
    C_percent: percentage of base C, list of two values or False([0.22, 0.28])
    blast_hsp_thres: threshold for blast hsp, no hsp larger than this allowed, int (10)
    readout_folder: folder to store readout information, string
    blast_ref: file basename for fasta file of existing readouts in readout_folder, used for blast
    Output:
    keep: whether this readout should be kept.
    '''
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.Alphabet import IUPAC
    input_seq = Seq(str(input_seq).upper(), IUPAC.unambiguous_dna)
    
    def _checking_GC(input_seq=input_seq, check_GC=GC_percent):
        if check_GC:
            if max(check_GC) == min(check_GC):
                raise ValueError('Invalid check_GC input vector!')
            from Bio.SeqUtils import GC
            _gc = GC(input_seq) / 100.0
            if verbose:
                print ("GC:", _gc)
            if _gc >= max(check_GC) or _gc <= min(check_GC): # if gc content doesn't pass filter
                return False
        return True
            
    def _checking_consecutive(input_seq=input_seq, max_consecutive=max_consecutive):
        if max_consecutive:
            _seq_str = str(input_seq).upper()
            # Check Consecutive 
            _mask = ['A'*max_consecutive, 'C'*max_consecutive, 'G'*max_consecutive, 'T'*max_consecutive] # mask for consecutive bases
            for i in range(len(_seq_str) - max_consecutive + 1):
                if _seq_str[i:i+max_consecutive] in _mask:
                    return False
        return True
    
    def _checking_repetitive(input_seq=input_seq, max_rep=max_rep):
        if max_rep:
            _seq_str = str(input_seq).upper()
            # check repetitive
            _rep_dic = {}
            for i in range(len(_seq_str) - max_rep + 1):
                if _seq_str[i:i+max_rep] in _rep_dic.keys():
                    return False
                else:
                    _rep_dic[_seq_str[i:i+max_rep]] = 1
        return True
    
    def _checking_C_percent(input_seq=input_seq, C_percent=C_percent):
        if C_percent:
            # global c percent
            if max(C_percent) == min(C_percent):
                raise ValueError('Invalid C_percent input vector!')
            _c_per = input_seq.count('C') / float(len(input_seq))
            if verbose:
                print ("C:", _c_per)
            if _c_per >= max(C_percent) or _c_per <= min(C_percent):
                return False
            else:
                # constraining C in first 12 bases
                _seq_str = str(input_seq).upper()
                for i in range(12-6):
                    if Seq(_seq_str[i:i+6]).count('C') >= 4:
                        return False
        return True
            
    def _checking_blast(input_seq=input_seq, blast_hsp_thres=blast_hsp_thres, 
                        readout_folder=readout_folder, blast_ref=blast_ref):
        import glob, os
        from Bio.Blast.Applications import NcbiblastnCommandline
        from Bio.Blast import NCBIXML 
        # write input_seq into a temp file
        SeqIO.write(SeqRecord(input_seq),  "temp.fasta", "fasta")
        # Run BLAST and parse the output as XML
        output = NcbiblastnCommandline(query="temp.fasta", 
                                       subject=os.path.join(readout_folder, blast_ref), 
                                       evalue=10,
                                       word_size=7,
                                       out='temp_out.xml',
                                       outfmt=5)()[0]
        _blast_record = NCBIXML.read(open(r'temp_out.xml', 'r'))
        _hsp_scores = []
        # parse xml results
        for _alignment in _blast_record.alignments:
            for _hsp in _alignment.hsps:
                _hsp_scores.append(_hsp.score)
        # clean temp files        
        os.remove("temp.fasta")
        os.remove("temp_out.xml")
        if verbose:
            print ("blast:", sum([_hsp_score > blast_hsp_thres for _hsp_score in _hsp_scores]))
        #False for having hsp_score larger than threshold
        _blast_result = sum([_hsp_score > blast_hsp_thres for _hsp_score in _hsp_scores])
        return _blast_result==0
            
    _keep_GC = _checking_GC()
    if not _keep_GC:
        return False
    #print _keep_GC
    _keep_cons = _checking_consecutive()
    if not _keep_cons:
        return False
    #print _keep_cons
    _keep_rep = _checking_repetitive()
    if not _keep_rep:
        return False
    #print _keep_rep
    _keep_c = _checking_C_percent()
    if not _keep_c:
        return False
    #print _keep_c
    _keep_blast = _checking_blast()
    if not _keep_blast:
        return False
    #print _keep_blast
    
    # Merge all keeps
    _keeps = [_keep_GC, _keep_cons, _keep_rep, _keep_c, _keep_blast]
    if verbose:
        print (_keeps)
        
    return True # All true, return true, otherwise False


def _generate_existing_readouts(filenames, readout_folder=_readout_folder, 
                                save_name=r'cand_readouts.fasta', verbose=True):
    """Function to generate one single fasta file containing all existing readouts
    Inputs:
        filenames: list of filenames that contain existing readouts, list of strings
        readout_folder: folder to store readout information, string
        save_name: filename to save merged readout information, string (default: cand_readouts.fasta)
        verbose: say something!, bool (default: True)"""
    _readout_fls = []
    for _fl in filenames:
        if not os.path.isfile(_fl):
            _full_fl = os.path.join(readout_folder, _fl)
            if os.path.isfile(_full_fl):
                _readout_fls.append(_full_fl)
            else:
                raise IOError(f"Wrong input filename, both {_fl} and {_full_fl} doesn't exist.")
    if verbose:
        print(f"- Start merging existing readouts from files: {_readout_fls}")
    # Loading
    _readout_records = []
    for _fl in _readout_fls:
        with open(_fl, "rU") as _handle:
            for _record in SeqIO.parse(_handle, "fasta"):
                _readout_records.append(_record)
    if verbose:
        print(f"-- {len(_readout_records)} readouts are loaded.")
    
    # save
    _save_filename = os.path.join(readout_folder, save_name)
    if verbose:
        print(f"-- saving to file:{_save_filename}")
    with open(_save_filename, "w") as _output_handle:
        SeqIO.write(_readout_records, _output_handle, "fasta")
    
    return _readout_records    

def Search_Candidates(source_readout_file, total_cand=200, existing_readout_file='cand_readouts.fasta',
                      readout_folder=_readout_folder, GC_percent=[0.4,0.6], max_consecutive=4, 
                      max_rep=6, C_percent=[0.2, 0.28], blast_hsp_thres=10.0, 
                      save_name='selected_candidates.fasta', verbose=True):
    """Function to search readout sequences in a given pool compared with existing readouts
    Inputs:
        source_readout_file: filename for readout sequence pool, string (should be .fasta)
        total_cand: number of candidates we hope to generate, int (default: 1000)
        existing_readout_file: filename for existing readouts, string (should be ,fasta)
        readout_folder: folder to store readout information, string (default: globally given)
        GC_percent: whether check gc content, list of two values or False ([0.4, 0.6])
        max_consecutive: maximum allowed consecutive bases, int (4)
        max_rep: maximum replicated sequence in a readout allowed, int (6)
        C_percent: percentage of base C, list of two values or False([0.22, 0.28])
        blast_hsp_thres: threshold for blast hsp, no hsp larger than this allowed, int (10)
        blast_ref: file basename for fasta file of existing readouts in readout_folder, used for blast
        verbose: say something!, bool (default: True)
    Outputs:
        _cand_readouts: list of Bio.SeqRecord.SeqRecord objects
    """
    ## check input files
    if not os.path.isfile(source_readout_file):
        source_readout_file = os.path.join(readout_folder, source_readout_file)
        if not os.path.isfile(source_readout_file):
            raise IOError(f"Wrong input source readout file:{source_readout_file}, not exist.")
    elif '.fasta' not in source_readout_file:
        raise IOError(f"Wrong input file type for {source_readout_file}")
    if not os.path.isfile(existing_readout_file):
        existing_readout_file = os.path.join(readout_folder, existing_readout_file)
        if not os.path.isfile(existing_readout_file):
            raise IOError(f"Wrong input source readout file:{existing_readout_file}, not exist.")
    elif '.fasta' not in existing_readout_file:
        raise IOError(f"Wrong input file type for {existing_readout_file}")
    # load candidate sequences and filter
    
    # start looping
    if verbose:
        print(f"- Start selecting readout candidates from {source_readout_file},\n\tfiltering with {existing_readout_file} ")
    _cand_records = []
    _ct = 0
    with open(source_readout_file, "rU") as _handle:
        for _record in SeqIO.parse(_handle, "fasta"):
            if len(_cand_records) >= total_cand:
                if verbose:
                    print(f"-- {total_cand} new candidates acquired, stop iteration.")
                break
            
            if verbose:
                print (f"--- processing: {_record.seq}")
                for i in range(32):
                    _new_seq = Extend_Readout(_record.seq)
                    _keep = Filter_Readout(_new_seq,GC_percent=GC_percent, 
                                            max_consecutive=max_consecutive,
                                            max_rep=max_rep, C_percent=C_percent,
                                            blast_hsp_thres=blast_hsp_thres,
                                            readout_folder=readout_folder,
                                            blast_ref=os.path.basename(existing_readout_file),
                                            verbose=False)
                    if _keep:
                        _kept_record = SeqRecord(_new_seq, id='cand_'+str(_ct+1), description='30mer_candidate')
                        _cand_records.append(_kept_record)
                        if verbose:
                            print (f"--- candidate:{_ct} {_new_seq} saved")
                        # Save to candidate records
                        with open(existing_readout_file, "a") as _output_handle:
                            SeqIO.write(_kept_record, _output_handle, "fasta")
                        _ct += 1
                        break
            else:
                break
    # after selection, save selected_candidates
    _save_filename = os.path.join(readout_folder, save_name)
    with open(_save_filename, 'w') as _output_handle:
        if verbose:
            print(f"-- saving candidate readouts into file: {_save_filename}")
        SeqIO.write(_cand_records, _output_handle, "fasta")
    
    return _cand_records


def filter_readouts_by_blast(blast_record, hard_thres=17, soft_thres=14, soft_count=100, verbose=False):
    '''filter for genome blast record
    Input:
        blast_record: xml format of blast result, generated by NCBIWWW.qblast or NCBIblastnCommandLine
        hard_threshold: no hsp_score match larger than this is allowed, int
        soft_threshold: hsp_score larger than this should be counted, int
        soft_count: number of hits larger than soft_threshold shouldnot pass this threshold, int
        verbose: it says a lot of things!
    Output:
        keep the record or not, bool'''
    if verbose:
        print (blast_record.query_id, len(blast_record.alignments))
    # extract information
    hsp_scores = []
    hsp_aligns = []
    for alignment in blast_record.alignments:
        for hsp in alignment.hsps:
            hsp_scores.append(hsp.score) 
            hsp_aligns.append(hsp.align_length)
    # hard threshold
    keep_hard = [hsp_score>hard_thres for hsp_score in hsp_scores]
    if verbose: 
        print( "hard count:", sum(keep_hard))
    if sum(keep_hard) > 0:
        if verbose:
            print ("Filtered out by hard threshold.")
        return False
    # soft threshold 
    keep_soft = [hsp_align>soft_thres for hsp_align in hsp_aligns]
    if verbose: 
        print ("soft count:", sum(keep_soft))
    if sum(keep_soft) >= soft_count:
        if verbose:
            print ("Filtered out by soft threshold count!")
        return False
    return True


def Filter_Readouts_by_Genome(cand_readout_file='selected_candidates.fasta', 
                              genome_db='hg38',
                              readout_folder=_readout_folder, genome_folder=_genome_folder,
                              word_size=10, evalue=1000, save_postfix='genome', 
                              verbose=True):
    """Filter readout candiates by blasting against genome
    Inputs:
    Outputs:
    """
    if not os.path.isfile(cand_readout_file):
        cand_readout_file = os.path.join(readout_folder, cand_readout_file)
        if not os.path.isfile(cand_readout_file):
            raise IOError(f"Wrong input candidate readout file:{cand_readout_file}, not exist.")
    elif '.fasta' not in cand_readout_file:
        raise IOError(f"Wrong input file type for {cand_readout_file}")
    # blast!
    blast_outfile = cand_readout_file.replace('.fasta', f'_{genome_db}.xml')
    output = NcbiblastnCommandline(query=cand_readout_file,
                                    num_threads=12,
                                    db=os.path.join(genome_folder, genome_db),
                                    evalue=500,
                                    word_size=10,
                                    out=blast_outfile,
                                    outfmt=5)()[0]
    # decide which to keep
    genomeblast_keeps = []                
    blast_records = NCBIXML.parse(open(os.path.join(readout_folder, 'selected_candidates_hg38.xml'), 'r'))
    for blast_record in blast_records:
        if verbose:
            print(blast_record.query_id, len(blast_record.alignments))
        keep = filter_readouts_by_blast(blast_record, verbose=verbose)
        genomeblast_keeps.append(keep)

    # save all 
    with open(cand_readout_file, "r") as handle:
        record_keeps = []
        for _i, record in enumerate(SeqIO.parse(handle, "fasta")):
            if genomeblast_keeps[_i]:
                record_keeps.append(record)
    save_filename = cand_readout_file.replace('.fasta', f'_{save_postfix}.fasta')
    with open(save_filename, "w") as output_handle:
        SeqIO.write(record_keeps, output_handle, "fasta")
    if verbose:
        print(f"-- number of readout candidates kept: {len(record_keeps)}")
    
    return record_keeps

def Filter_Readouts_by_RNAfold(cand_readout_file='selected_candidates_genome.fasta',
                               rnafold_exe=r'E:\Shared_Apps\ViennaRNA\RNAfold',
                               energy_th =-6.0, readout_folder=_readout_folder, 
                               make_plot=False, verbose=True):
    """Filter Readouts by energy of secondary structure generated by RNAfold
    Inputs:
    Outputs:
    """
    if not os.path.isfile(cand_readout_file):
        cand_readout_file = os.path.join(readout_folder, cand_readout_file)
        if not os.path.isfile(cand_readout_file):
            raise IOError(f"Wrong input candidate readout file:{cand_readout_file}, not exist.")
    elif '.fasta' not in cand_readout_file:
        raise IOError(f"Wrong input file type for {cand_readout_file}")
    # run RNAfold
    _rnafold_output = cand_readout_file.replace('.fasta', '_structreport.txt')
    os.system(f"{rnafold_exe} < {cand_readout_file} > {_rnafold_output}")
    # load RNAfold result and read it
    import re
    structure_dics = []
    energy_list = []
    with open(_rnafold_output, 'r') as handle:
        structure_reports = handle.read()
        structure_reports = structure_reports.split('>')[1:]
        for structure_report in structure_reports:
            lines = structure_report.split('\n')[:-1]
            barcode_id = lines[0].split(' ')[0]
            barcode_description = lines[0].split(' ')[1]
            re_result = re.match('(\S+)\s\(\s*?([0-9\-\+\.]+)\)', lines[2])
            barcode_energy = float(re_result.group(2))
            structure_dic = {'id':barcode_id, 'description':barcode_description, 'energy':barcode_energy}
            structure_dics.append(structure_dic)
            energy_list.append(barcode_energy)
    # whether keep each record:
    structure_keeps = np.array(energy_list) > energy_th
    # extract kept records
    kept_records = []
    with open(os.path.join(readout_folder, 'selected_candidates_genome.fasta'), "r") as handle:
        for _i, record in enumerate(SeqIO.parse(handle, "fasta")):
            if structure_keeps[_i]:
                kept_records.append(record)
    # save selected records
    _save_filename =   cand_readout_file.replace('.fasta', '_structure.fasta')
    with open(_save_filename, "w") as output_handle:
        SeqIO.write(kept_records, output_handle, "fasta")
    
    if make_plot:
        f1 = plt.figure()
        plt.hist(energy_list)
        plt.show()

    return kept_records


def Save_Readouts(cand_readout_file='selected_candidates_genome_structure.fasta',
                  existing_readout_file='NDBs.fasta', readout_folder=_readout_folder,
                  write_append=False, verbose=True):
    """Rename candidate readouts along with existing readouts and save
    Inputs:
    Output:"""
    # check inputs
    if not os.path.isfile(cand_readout_file):
        cand_readout_file = os.path.join(readout_folder, cand_readout_file)
        if not os.path.isfile(cand_readout_file):
            raise IOError(f"Wrong input candidate readout file:{cand_readout_file}, not exist.")
    elif '.fasta' not in cand_readout_file:
        raise IOError(f"Wrong input file type for {cand_readout_file}")
    if not os.path.isfile(existing_readout_file):
        existing_readout_file = os.path.join(readout_folder, existing_readout_file)
        if not os.path.isfile(existing_readout_file):
            raise IOError(f"Wrong input existing readout file:{existing_readout_file}, not exist.")
    elif '.fasta' not in existing_readout_file:
        raise IOError(f"Wrong input file type for {existing_readout_file}")
    
    # load existing readouts
    with open(existing_readout_file, 'r') as handle:
        existing_readouts = []
        existing_id_num = 0
        for _i, record in enumerate(SeqIO.parse(handle, "fasta")):
            existing_readouts.append(record)    
            _rec_id = int(record.id.split('_')[1])
            if _rec_id > existing_id_num:
                existing_id_num = _rec_id
    # load candidate readouts
    with open(cand_readout_file, 'r') as handle:
        cand_readouts = []
        for _i, record in enumerate(SeqIO.parse(handle, "fasta")):
            cand_readouts.append(record)    
    # modify records
    new_records = [SeqRecord(_record.seq, id='NDB_'+str(existing_id_num+_i+1), description="", name="")
                   for _i, _record in enumerate(cand_readouts)]
    if verbose:
        print(f"-- saving {len(new_records)} new readouts ")
    
    if write_append:
        _save_filename = existing_readout_file
        # save
        with open(_save_filename, "a") as output_handle:
            SeqIO.write(new_records, output_handle, "fasta")
    else:
        _save_filename = existing_readout_file.replace('.fasta', '_new.fasta')
        # save
        with open(_save_filename, "w") as output_handle:
            SeqIO.write(new_records, output_handle, "fasta")
        
    return new_records


def Split_readouts_into_channels(readout_fasta, num_channels=3, start_ind=0,
                                 save=True, save_name=None, save_folder=_readout_folder,
                                 overwrite=True, verbose=True):
    """Function to split readouts into 3 channels and save separate files
    Inputs:
        readout_fasta: fasta filename or list of SeqRecord for readouts, str or list
        
    Outputs:
        splited_readout_list: list of num_channels containing SeqRecords for readouts.
    """
    # readout-fasta input
    if isinstance(readout_fasta, str):
        with open(readout_fasta, 'r') as _handle:
            readout_records = []
            for _record in SeqIO.parse(_handle, "fasta"):
                readout_records.append(_record)
    elif isinstance(readout_fasta, list):
        readout_records = readout_fasta
    # other inputs
    num_channels = int(num_channels)
    start_ind = int(start_ind)
    # save name
    if save and save_name is None and isinstance(readout_fasta, list):
        raise ValueError(f"No save_name is given, exit!")
    else:
        if '.fasta' not in save_name:
            save_name += '.fasta'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # initialize
    if verbose:
        print(f"- Splitting {len(readout_records)} readouts into {num_channels} channels")
    splitted_readout_list = [[] for _i in range(num_channels)]
    # split
    for _i, _r in enumerate(readout_records[start_ind:]):
        splitted_readout_list[_i % num_channels].append(_r)
    # save
    if save:
        for _i, _readouts in enumerate(splitted_readout_list):
            _save_filename = os.path.join(
                save_folder, save_name.replace('.fasta', f'_{_i}.fasta'))
            if not os.path.exists(_save_filename) or overwrite:
                if verbose:
                    print(f"-- saving {len(_readouts)} readouts into file:{_save_filename}")
                with open(_save_filename, 'w') as _output_handle:
                    SeqIO.write(_readouts, _output_handle, "fasta")

    return splitted_readout_list

##---------------------------------------------------------------------------##
# Adaptors

# convert readouts to adaptors
def Generate_adaptors(readout_fasta, adaptor_site_fasta, rc_readout=False, rc_adaptor_site=False,
                      kept_len=20, kept_5=False):
    """Generate adaptors by giving readout fasta and adaptor fasta"""

    if isinstance(readout_fasta, str):
        with open(readout_fasta, 'r') as _handle:
            readout_records = []
            for _record in SeqIO.parse(_handle, "fasta"):
                if rc_readout:
                    readout_records.append(_record.reverse_complement())
                else:
                    readout_records.append(_record)
    elif isinstance(readout_fasta, list):
        readout_records = readout_fasta

    if isinstance(adaptor_site_fasta, str):
        with open(adaptor_site_fasta, 'r') as _handle:
            site_records = []
            for _record in SeqIO.parse(_handle, "fasta"):
                if rc_adaptor_site:
                    site_records.append(_record.reverse_complement())
                else:
                    site_records.append(_record)
    elif isinstance(readout_fasta, list):
        site_records = adaptor_site_fasta

    adaptor_records = []
    for _i, _readout in enumerate(readout_records):
        if kept_5:
            _readout_target = _readout[:kept_len]
        else:
            _readout_target = _readout[-kept_len:]

        _adaptor = _readout_target + \
            site_records[_i % len(site_records)][:kept_len] + \
            site_records[_i % len(site_records)][:kept_len]
        _adaptor.id = f"{_adaptor.id}_2x{site_records[_i % len(site_records)].id}"
        adaptor_records.append(_adaptor)

    return adaptor_records

## Adaptor functions

def Screen_seqs_against_fasta(record_fasta, ref_fasta, word_size=17, allowed_hits=0,
                              exclude_names=_adaptor_site_names, check_rc=True,
                              save=False, save_folder=None, save_name=None,
                              overwrite=False, return_kept_flag=False, verbose=True):
    """Function to screen sequences against a given fasta file
    Inputs:
        record_fasta: fasta filename or list of SeqRecord, str or list
        ref_fasta: filename for reference fasta file to screen against, string of file path
        word_size: word_size used for probe screening, int (default: 17)
        allowed_hits: allowed hits for one probe in the fasta, int (default: 8)
        exclude_names: list of names to be excluded, list (default: _adaptor_site_names)
        check_rc: whether check reverse-complement of the probe, bool (default: True)
        save: whether save result probe reports, bool (default: True)
        save_folder: folder to save selected probes, string of path (default: None, which means +'_filtered')
        overwrite: whether overwrite existing result probe reports, bool (default: False)
        return_kept_flag: whether return flags for whether keeping the record, bool (default:False)
        verbose: say something!, bool (default: True)
    """
    ## Check inputs
    if verbose:
        print(f"- Screen sequences against given fasta file:{ref_fasta}")
    # load record-fasta
    if isinstance(record_fasta, str):
        with open(record_fasta, 'r') as _handle:
            _records = []
            for _record in SeqIO.parse(_handle, "fasta"):
                _records.append(_record)
    elif isinstance(record_fasta, list):
        _records = record_fasta
    if verbose:
        print(f"-- {len(_records)} sequences loaded.")

    if not os.path.isfile(ref_fasta):
        raise IOError(f"Reference fasta:{ref_fasta} is not a file.")
    word_size = int(word_size)
    allowed_hits = int(allowed_hits)
    if save_folder is None:
        if isinstance(record_fasta, str):
            save_folder = os.path.dirname(record_fasta)
        else:
            save_folder = os.path.dirname(ref_fasta)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        if verbose:
            print(f"-- create {save_folder} to store filter probes")

    ## construct table for ref_fasta
    if verbose:
        print(f"-- constructing reference table for fasta file")
    _ref_names, _ref_seqs = ld.fastaread(ref_fasta, force_upper=True)
    # filter sequences by given reference name
    _kept_ref_seqs = []
    for _n, _s in zip(_ref_names, _ref_seqs):
        if _n.split(' ')[0] in exclude_names:
            continue
        else:
            _kept_ref_seqs.append(_s)

    _ref_table = ld.OTmap(_kept_ref_seqs, word_size, use_kmer=True)

    ## filter records
    if check_rc:
        _hits = [_ref_table.get(str(_r.seq), rc=True) +
                 _ref_table.get(str(_r.seq), rc=False)
                 for _r in _records]
    else:
        _hits = [_ref_table.get(str(_r.seq), rc=False) for _r in _records]
    # filter
    _kept_records = [_r for _r, _h in zip(
        _records, _hits) if _h <= allowed_hits]
    if return_kept_flag:
        _kept_flags = [_h <= allowed_hits for _h in _hits]
    if verbose:
        print(
            f"-- {len(_kept_records)} sequences kept by allowing hits:{allowed_hits}")
    ## Save
    if save:
        if save_name is None and not isinstance(record_fasta, str):
            print(f"Save name not given in either save_name kwd and record_fasta, skip.")
        elif save_name is None:
            save_name = os.path.basename(record_fasta)
        if '.fasta' not in save_name:
            save_name += '.fasta'
        save_filename = os.path.join(save_folder, save_name)
        with open(save_filename, 'wb') as _output_handle:
            if verbose:
                print(
                    f"-- saving {len(_kept_records)} kept records in file:{save_filename}")
            SeqIO.write(_kept_records, _output_handle, "fasta")

    if return_kept_flag:
        return _kept_records, np.array(_kept_flags, dtype=np.bool)
    else:
        return _kept_records


def Check_adaptors_against_fasta(readout_fasta, adaptor_site_fasta, ref_fasta, word_size=11, allowed_hits=0,
                                 exclude_names=_adaptor_site_names, check_rc=True,
                                 save=False, save_folder=_readout_folder, save_name=None, save_postfix='_kept',
                                 save_adaptors=False, overwrite=False, verbose=True):
    """Function to check adaptors against a list of fasta files, until get satisfying matches
    Inputs:
    
    Outputs:
        _kept_readouts: list of SeqRecords of which readouts are saved
    """
    ## check inputs
    if verbose:
        print(f"- Check raedouts->adaptors against fasta")
    # readout_Fasta
    if not isinstance(readout_fasta, str):
        raise TypeError(f"Wrong input type of readout_fasta:{readout_fasta}")
    elif not os.path.isfile(readout_fasta):
        raise IOError(
            f"Input file readout_fasta:{readout_fasta} not exist, exit!")
    # adaptor_site_Fasta
    if not isinstance(adaptor_site_fasta, str):
        raise TypeError(
            f"Wrong input type of adaptor_site_fasta:{adaptor_site_fasta}")
    elif not os.path.isfile(adaptor_site_fasta):
        raise IOError(
            f"Input file adaptor_site_fasta:{adaptor_site_fasta} not exist, exit!")
    # ref_fasta
    if isinstance(ref_fasta, str):
        ref_fasta = [ref_fasta]
    if not isinstance(ref_fasta, list):
        raise TypeError(
            f"ref_fasta should be either one filename or list of filenames")
    for _fl in ref_fasta:
        if not os.path.isfile(_fl):
            raise IOError(f"input ref_fasta file:{_fl} not exist, exit.")
    # save etc.
    if save_name is None:
        save_name = os.path.basename(readout_fasta).replace(
            '.fasta', save_postfix+'.fasta')
        save_filename = os.path.join(save_folder, save_name)
    # load readouts
    with open(readout_fasta, 'r') as _handle:
        readouts = []
        for _record in SeqIO.parse(_handle, "fasta"):
            readouts.append(_record)
        if verbose:
            print(f"-- {len(readouts)} readout loaded")
    # initialize adaptor selection flags
    _adaptor_flags = []
    while(len(_adaptor_flags) != len(readouts)):
        # generate current adaptors
        _adaptors = Generate_adaptors(readouts, adaptor_site_fasta)
        # update whether keep the adaptor
        _adaptor_flags = np.ones(len(_adaptors), dtype=np.bool)
        for _fl in ref_fasta:
            _, _fl_kept = Screen_seqs_against_fasta(_adaptors, _fl, word_size=word_size,
                                                    allowed_hits=allowed_hits,
                                                    return_kept_flag=True, verbose=False)
            _adaptor_flags *= _fl_kept

        readouts = [_r for _r, _f in zip(readouts, _adaptor_flags) if _f]
    if verbose:
        print(f"-- {len(readouts)} readous are kept.")
    if save:
        with open(save_filename, 'w') as _output_handle:
            if verbose:
                print(f"-- saving filtered readouts to file: {save_filename}")
            SeqIO.write(readouts, _output_handle, "fasta")

    if save_adaptors:
        _adaptors = Generate_adaptors(readouts, adaptor_site_fasta)
        adaptor_save_filename = save_filename.replace(
            '.fasta', '_adaptor.fasta')
        if verbose:
            print(
                f"-- saving corresponding adaptors to file: {adaptor_save_filename}")
        with open(adaptor_save_filename, 'w') as _output_handle:
            SeqIO.write(readouts, _output_handle, "fasta")

    return readouts
