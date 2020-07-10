import numpy as np
import sys, os, glob, time
from . import LibraryDesigner as ld
from . import LibraryTools as lt
def Batch_Extract_Sequences(library_folder, genome_folder,
                            reg_filename =r'Regions.txt', save_dir='sequences',
                            merge=True, save=True, resolution=50000, flanking=0):
    '''Function to extract sequences for all regions written in a file
    Given:
    
    merge: if merge all regions together and arrange region id, bool'''
    save_folder = os.path.join(library_folder, save_dir)
    if not isinstance(library_folder, str) and not isinstance(reg_filename, str) and not isinstance(genome_folder, str):
        raise ValueError('wrong input format!')

    def read_region_file(library_folder=library_folder, reg_filename=reg_filename):
        '''Sub-function to read region file'''
        # region filename
        _reg_filename = library_folder + os.sep + reg_filename
        _reg_file = open(_reg_filename, 'r')    
        print ('Input region file is: '+ _reg_filename)
        # start reading
        _lines = _reg_file.read().split('\n')
        _titles = _lines[0].split('\t')
        # save a list of dictionaries
        _reg_list = []
        for _line in _lines[1:]:
            _reg_dic = {} # dinctionary to save all informations
            _info = _line.split('\t') # split informations
            if len(_info) != len(_titles): # sanity check to make sure they are of the same size
                continue
            for _i in range(len(_info)): # save info to dic
                _reg_dic[_titles[_i]] = _info[_i]
            _reg_list.append(_reg_dic) # save dic to list
        _reg_file.close()
        return _reg_list

    def parse_region(reg_dic):
        '''given a dictionary of one region, 
        report:
            _chrom: str
            _start: int
            _stop: int'''
        region_str = reg_dic['Region']
        # grab chromosome
        _chrom = region_str.split(':')[0]
        _locus = region_str.split(':')[1]
        # grab start and stop positions
        _start, _stop = _locus.split('-')
        _start = int(_start.replace(',', ''))
        _stop = int(_stop.replace(',', ''))
        # return in this order:
        return _chrom, _start, _stop
    
    def extract_sequence(reg_dic, genome_folder=genome_folder, \
                         resolution=resolution, flanking=flanking, \
                         save_folder=save_folder, save=save, merge=merge):
        from math import ceil
        '''sub-function to extract sequences of one locus
        Given:
        reg_dic: dic for region info, dictionary
        genome_folder: dir for genome files, str
        resolution: resolution of each region in bp, int
        flanking: upstream and downstream included in bp, int
        save: if save as fasta files, bool
        Return:
        dic of sequences of designed regions
        Dependencies:
        ld.fastaread, ld.fastawrite, ./parse_region'''
        # get chromosome, start and stop information
        _chrom, _start, _stop = parse_region(reg_dic)
        # dir to store sequences, this will be returned
        _seq_dic = {}
        _seqs = []
        _names = []
        if not flanking:
            flanking=0
        # read chromosome seq
        _, _wholechr = ld.fastaread(genome_folder+os.sep+_chrom+'.fa')
        _wholechr = _wholechr[0]
        # number of regions
        _n_reg = int(np.ceil( float(_stop+flanking - (_start-flanking)) / resolution))
        # extract all required seq
        _whole_seq = _wholechr[_start-flanking: min(_start-flanking+_n_reg*resolution, len(_wholechr))]
        for _i in range(_n_reg):
            # end location
            _end_loc = min((_i+1)*resolution, len(_wholechr))
            # extract sequence for this region
            _seq = _whole_seq[_i*resolution:_end_loc]
            _name = _chrom+':'+str(_start-flanking+_i*resolution)+'-'+\
                     str(_start-flanking+_end_loc)+'_reg_'+str(_i+1)
            _seq_dic[_name] = _seq
            _seqs.append(_seq)
            _names.append(_name)
            if _end_loc == len(_wholechr):
                break
        # if Save
        if save:                
            # mkdir if not exist for save folder
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
                
            print ('-writing region number:' + str(_n_reg))
            
            if merge: # NOTICE! this will never overwrite!
                save_sub_folder = save_folder+os.sep+'merged'
                if not os.path.exists(save_sub_folder):
                    os.makedirs(save_sub_folder)
                ex_file_num = len(glob.glob(save_sub_folder+os.sep+r'*'))
                # writing files
                for _i, (_name, _seq) in enumerate(zip(_names, _seqs)):
                    _filename = save_sub_folder + os.sep + 'reg_' + str(ex_file_num+_i+1) + '.fasta'
                    # save as fasta
                    if 'Gene' in reg_dic.keys():
                        ld.fastawrite(_filename, [_name+'_gene_'+reg_dic['Gene']], [_seq.upper()])
                    else:
                        ld.fastawrite(_filename, [_name], [_seq.upper()])
                print ('Number of region: '+str(len(glob.glob(save_sub_folder+os.sep+r'*'))))
            else:
                # assign correct name of the sub folder
                if 'Gene' in reg_dic.keys():
                    save_sub_folder = save_folder+os.sep+reg_dic['Gene']
                else:
                    save_sub_folder = save_folder+os.sep+_chrom+str(_start)
                # mkdir if not exist for this region
                if not os.path.exists(save_sub_folder):
                    os.makedirs(save_sub_folder)
                # writing files
                for _i in range(_n_reg):
                    _filename = save_sub_folder + os.sep + 'reg_' + str(_i+1) + '.fasta'
                    # save as fasta
                    ld.fastawrite(_filename, [_names[_i]], [_seqs[_i].upper()])
                
        return _seq_dic
        
    ## read region file
    reg_list = read_region_file()
    
    # extract sequences and save!
    seq_dic_list = []
    for reg_dic in reg_list:
        seqs = extract_sequence(reg_dic, save=save)
        seq_dic_list.append(seqs)
        
    return seq_dic_list, reg_list
        