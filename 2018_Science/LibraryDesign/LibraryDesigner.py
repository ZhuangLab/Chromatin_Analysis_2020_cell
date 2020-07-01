
#Bogdan Bintu
#Copyright Presidents and Fellows of Harvard College, 2017.

#External tools
from Bio.SeqUtils import MeltingTemp as mt
import cPickle as pickle
import numpy as np
import os,subprocess,time

#import tools LibraryTools
from LibraryTools import fastaread
from LibraryTools import fastawrite
from LibraryTools import constant_zero_dict

from LibraryTools import seqrc
from LibraryTools import OTTable

#python functions to convert seq->number
def seq2Int(seq):
    dic_conv = {'C':1,'c':1,'G':2,'g':2,'T':3,'t':3}
    digits = np.array([dic_conv.get(let,0) for let in seq],dtype='uint64')
    word = len(seq)
    base = 4**(np.arange(word,dtype='uint64')[::-1])
    return np.sum(base*digits)
def seq2Int_rc(seq):
    dic_conv = {'G':1,'g':1,'C':2,'c':2,'A':3,'a':3}
    digits = np.array([dic_conv.get(let,0) for let in seq],dtype='uint64')
    word = len(seq)
    base = 4**(np.arange(word,dtype='uint64')[::-1])
    return np.sum(base*digits)
# Faster (500x) cython functions to convert seq->number. Compiled on a Windows machine. 
# Recompile on Linux by running in a cell in a jupyter notebook in the current directory: !python C_Tools\setup.py build_ext --inplace
# If too complicated, comment out line bellow and have patience.
from seqint import seq2Int,seq2Int_rc
import scipy.sparse as ss
import time
def unique_cts(x):
    x_=np.array(x,dtype=np.uint16)
    y = np.bincount(x_)
    ii = np.nonzero(y)[0]
    return ii,y[ii]
class countTable():
    def __init__(self,word=17,save_file=None):
        """
        This constructs sparse array count table using scipy lil_matrix for efficient construction.
        """
        self.word = word
        self.save_file = save_file
        
        self.max_size = 4**word
        self.max_sparse_ind = 2**31#scipy.sparse decided to encoded in indeces in int32. Correct!
        self.nrows = self.max_size/self.max_sparse_ind
        self.matrix = ss.csr_matrix((self.nrows,self.max_sparse_ind), dtype=np.uint16)
        self.ints=[]
    def save(self):
        if self.save_file is not None:
            ss.save_npz(self.save_file, self.matrix)
    def load(self):
        if self.save_file is not None:
            ss.load_npz(self.save_file, self.matrix)
    def complete(self,verbose=False):
        """a np.unique is performed on self.ints and the number of occurences for each unique 17mer (clipped to 2^16-1) is recorded in a sparse array self.matrix"""
        if verbose:
            start = time.time()
        pos,cts = np.unique(self.ints,return_counts=True)
        countTable_values = np.array(np.clip(cts, 0, 2**16-1),dtype='uint16')#clip and recast as uint16
        if verbose:
            end=time.time()
            print 'Time to compute unique and clip:',end-start
        
        if verbose:
            start = time.time()
        self.matrix = self.matrix.tolil()
        pos_col = pos/self.max_sparse_ind
        pos_row = pos-pos_col*self.max_sparse_ind
        self.matrix[pos_col,pos_row] = countTable_values
        self.matrix = self.matrix.tocsr()
        if verbose:
            end=time.time()
            print 'Time to update matrix:',end-start
        
    def consume(self,sq,verbose=False):
        """Given a big sequence sq, this breaks in into all contigous subseqs of size <word> and records each occurence in self.ints"""
        word=self.word
        if len(sq)>=word:
            sq_word = [sq[i:i+word] for i in range(len(sq)-word+1)]
            if verbose:
                start = time.time()
            self.ints.extend(map(seq2Int,sq_word))
            if verbose:
                end=time.time()
                print 'Time to compute seq2Int:',end-start
    def get(self,seq,rc=False):
        """give an oligo, this breaks it into all contigous subseqs of size <word> and returns the sum of the counts"""
        word = self.word
        if len(seq)<len(word):
            return 0
        seqs = [seq[i:i+word] for i in range(len(seq)-word+1)]
        if not rc:
            ints = np.array(map(seq2Int,seqs),dtype='uint64')
        else:
            ints = np.array(map(seq2Int_rc,seqs),dtype='uint64')
        pos_col = ints/self.max_sparse_ind
        pos_row = ints-pos_col*self.max_sparse_ind
        return np.sum(self.matrix[pos_col,pos_row])

def OTmap(seqs,word_size=17,use_kmer=True,progress_report=False,save_file=None):
    """This creates an count table using either a sparse matrix form or a python dictionary.
    For large genome-sized tables we recommend the sparse matrix
    """
    if use_kmer:
        map_ = countTable(word=word_size)
        print "Mapping no. of seqs: "+str(len(seqs))
        for seq in seqs:
            map_.consume(seq.upper(),verbose=progress_report)
        if save_file is not None:
            map_.save(save_file)
    else:
        specTable = OTTable()
        map_ = specTable.computeOTTable(seqs,word_size,progress_report=progress_report)
        if save_file is not None:
            pickle.dump(map_,open(save_file,'wb'),protocol=pickle.HIGHEST_PROTOCOL)
    return map_


class pb_reports_class:
    def __init__(self,sequence_dic={'file':None,'use_revc':False,'use_kmer':False},
                 map_dic={'transcriptome':{'file':None,'use_kmer':False},
                          'genome':{'file':None,'use_revc':True,'use_kmer':False},
                          'rep_transcriptome':{'file':None,'use_revc':True,'use_kmer':False},
                          'rep_genome':{'file':None,'use_revc':True,'use_kmer':False},
                          'isoforms':{'file':None,'force_list':True,'use_kmer':False},
                          'local_genome':{'files':None,'force_list':True,'use_kmer':False}},
                 save_file=None,
                 params_dic={'word_size':17,'pb_len':42,'buffer_len':2,'max_count':2**16-1,'check_on_go':False,'auto':False},
                 dic_check={('transcriptome','isoforms'):3,('genome','local_genome'):20,
                             'rep_transcriptome':0,'rep_genome':0,'gc':(0.25,0.75),'tm':70}):
        """
        Description:
        Example full usage with serial_reque:
        #from: http://localhost:8811/notebooks/Chromatin_Multiplex_Notes/Bogdan/SI10-construction.ipynb
        Example full usage with for loop in interact:
        #from: http://localhost:8811/notebooks/Chromatin_Multiplex_Notes/Bogdan/SI10-construction.ipynb
        
        master_folder  = r'/n/dulacfs2/Users/bbintu/Libraries/SI10/DNA'
        in_files = glob.glob(master_folder+os.sep+'old*.fasta')+glob.glob(master_folder+os.sep+'left*.fasta')
        in_files = in_files[-2:]

        save_files = [r'/n/dulacfs2/Users/bbintu/Libraries/SI10/Reports/'+os.path.basename(in_file).split('.')[0]+'.pbr' 
                      for in_file in in_files]

        transcriptome_fl = r'/n/dulacfs2/Users/bbintu/Transcriptomes/mouse/mm10_rna_word17_.kmer'
        genome_fl = r'/n/dulacfs2/Users/bbintu/Genomes/mouse/mm10/full_word17_.kmer'
        rep_transcriptome_fl = r'/n/dulacfs2/Users/bbintu/Transcriptomes/mouse/rtRNA.fasta'
        rep_genome_fl = r'/n/dulacfs2/Users/bbintu/Genomes/mouse/mm10/repeatSequences.fasta'
        local_genome_fl = in_files



        pb_designer = ld.pb_reports_class(
            sequence_dic={'file':None,'use_revc':True,'use_kmer':True},
            map_dic={'transcriptome':{'file':transcriptome_fl,'use_revc':False,'use_kmer':True},
                  'genome':{'file':genome_fl,'use_revc':True,'use_kmer':True},
                  'rep_transcriptome':{'file':rep_transcriptome_fl,'use_revc':True,'use_kmer':True},
                  'rep_genome':{'file':rep_genome_fl,'use_revc':True,'use_kmer':True},
                  'local_genome':{'file':None,'force_list':True,'use_revc':True,'use_kmer':True}},
            save_file=None,
            params_dic={'word_size':17,'pb_len':42,'buffer_len':2,'max_count':2**16-1,'check_on_go':False,'auto':False},
            dic_check={'transcriptome':3,('genome','local_genome'):20,
                        'rep_transcriptome':0,'rep_genome':0,'gc':[0.25,0.75],'tm':70})

        pb_designer.computeOTmaps()

        for sequence_file,save_file in zip(in_files,save_files):
            #modify input file
            pb_designer.sequence_dic['file'] = sequence_file
            #modify save file
            pb_designer.save_file = save_file
            pb_designer.load_sequence_file_and_paramaters()
            #modify maps
            key='local_genome'
            pb_designer.map_dic[key]['file'] = sequence_file
            pb_designer.files_to_OTmap("map_"+key,pb_designer.map_dic[key])
            #compute
            pb_designer.compute_pb_report()
            pb_designer.perform_check_end()
            pb_designer.plots()
        
        """
        #internalize paramaters
        self.sequence_dic=sequence_dic
        self.map_dic=map_dic
        self.params_dic=params_dic
        self.dic_check = dic_check
        self.save_file = save_file
        #load sequence file
        self.load_sequence_file_and_paramaters()     
      
        if self.auto:
            self.computeOTmaps()
            self.compute_pb_report()
            if not self.check_on_go:
                self.perform_check_end()
    def load_sequence_file_and_paramaters(self):
        sequence_file = self.sequence_dic.get('file',None)
        self.input_names,self.input_seqs = [],[]
        self.input_noSeqsperFile=[]
        if sequence_file is not None:
            sequence_files = self.str_to_list(sequence_file)
            for sequence_file_ in sequence_files:
                names,seqs = fastaread(sequence_file_,force_upper=True)
                self.input_noSeqsperFile.append(len(names))
                self.input_names.extend(names)
                self.input_seqs.extend(seqs)
        #add additional parameters
        self.params_dic['sequence_lens']=map(len,self.input_seqs)
        self.params_dic['input_noSeqsperFile']=self.input_noSeqsperFile
        self.params_dic['input_names']=self.input_names
        #internalize paramaters in params_dic
        for key in self.params_dic.keys():
            setattr(self,key,self.params_dic[key])

    def load_pbr(self,filename):
        "loads report file"
        dic_save = pickle.load(open(filename,'rb'))
        #internalize loaded values
        for key in dic_save.keys():
            setattr(self,key,dic_save[key])
        ### repeat __init__
        #load sequence file
        #self.load_sequence_file_and_paramaters()
        #internalize paramaters in params_dic
        for key in self.params_dic.keys():
            setattr(self,key,self.params_dic[key])    
    def save_pbr(self):
        "saves report file"
        dic_save = {"pb_reports":self.pb_reports,"pb_reports_keep":self.pb_reports_keep,
                   'sequence_dic':self.sequence_dic,'map_dic':self.map_dic,'params_dic':self.params_dic,'dic_check':self.dic_check,'save_file':self.save_file}
        if self.save_file is not None:
            pickle.dump(dic_save,open(self.save_file,'wb'))
    def save_pbrs(self):
        "saves to report files each input sequence. It auto generates save_files if necessary based on inputfiles and folder of save_file provided"
        if not hasattr(self,'save_files'):
            self.savefile_to_savefiles()
        reports_keys = self.pb_reports.keys()
        reports_keep_keys = self.pb_reports_keep.keys()
        region_indexes = [self.pb_reports[key_]['reg_index'] for key_ in reports_keys]
        ks = np.unique(region_indexes)
        for k in ks:
            pb_reports_save = {key:self.pb_reports[key] 
                               for key in reports_keys if self.pb_reports[key]['reg_index']==k}
            pb_reports_keep_save = {key:self.pb_reports_keep[key] 
                                    for key in reports_keep_keys if self.pb_reports_keep[key]['reg_index']==k}
            dic_save = {"pb_reports":pb_reports_save,"pb_reports_keep":pb_reports_keep_save,
                       'sequence_dic':self.sequence_dic,'map_dic':self.map_dic,'params_dic':self.params_dic,'dic_check':self.dic_check,'save_file':self.save_file}
            if self.save_file is not None:
                pickle.dump(dic_save,open(self.save_files[k],'wb'))
    def str_to_list(self,var):
        "Converst a string to a list"
        if type(var) is str:
            return [var]
        else:
            return var
        
    def check_extension(self,files,extensions):
        "Checks if file/(all files in list) have the same extension/extensions"
        files_ = self.str_to_list(files)
        extensions_ = self.str_to_list(extensions)
        return np.prod([os.path.basename(fl).split(os.path.extsep)[-1] in extensions_ for fl in files_])==1

    def load_key(self,key):
        "Loads a sequences from the map_dic if extension of map_dic[key]['file'] is in ['fa','fasta']. self.<key>_names,self.<key>_seqs"
        curr_dic = self.map_dic['key']
        files = curr_dic.get('file',None)
        if files is not None:
            extensions_available = ['fa','fasta']
            if self.check_extension(files,extensions_available):
                files_ = self.str_to_list(files)
                names,seqs=[],[]
                for fl in files_:
                    names_,seqs_=fastaread(fl,force_upper=True)
                    if curr_dic.get('force_list',False):
                        names.append(names_)
                        seqs.append(seqs_)
                    else:
                        names.extend(names_)
                        seqs.extend(seqs_)
                setattr(self,key+'_names',names)
                setattr(self,key+'_seqs',seqs)
            else:
                print "Error, extensions availabe: "+str(extensions_available)
        else:
            print "No files"
                
        
    def computeOTmaps(self):
        """This creates maps:
        Iterates over keys in map_dic and uses self.files_to_OTmap.
        """
        start = time.time()
        for key in self.map_dic.keys():
            self.files_to_OTmap("map_"+key,self.map_dic[key])
        end = time.time()
        print "Time(s): "+str(end-start)
    def files_to_OTmap(self,map_key,curr_dic):
        "This function transforms a file or list of files to an OT map and sets it as an attribute in self."
        print "Setting attribute: "+map_key
        
        files = curr_dic.get('file',None)
        force_list = curr_dic.get('force_list',False)
        expt_table_size = curr_dic.get('table_size',1e7)
        save_file = curr_dic.get('save_file',None)
        use_kmer = curr_dic.get('use_kmer',False)
        
        if files is not None:
            files_=self.str_to_list(files)
            fasta_ext = ['fa','fasta']
            if self.check_extension(files,fasta_ext):
                names,seqs=[],[]
                for fl in files_:
                    names_,seqs_=fastaread(fl,force_upper=True)
                    if force_list:
                        names.append(names_)
                        seqs.append(seqs_)
                    else:
                        names.extend(names_)
                        seqs.extend(seqs_)
                if not force_list:
                    names =[names]
                    seqs = [seqs]
                OTmaps = [OTmap(seq_,word_size=17,use_kmer=use_kmer,progress_report=False,save_file=save_file)
                          for seq_ in seqs]
                setattr(self,map_key,OTmaps)
            elif len(files_)==1 and self.check_extension(files,'.npz'):
                OTMap_ = OTmap(seq_,word_size=17)
                OTMap_.load(files_[0])
                OTmaps = [OTMap_]
                setattr(self,map_key,OTmaps)
            elif len(files_)==1 and self.check_extension(files,'pkl'):
                OTmaps = [pickle.load(open(files_[0],'rb'))]
                setattr(self,map_key,OTmaps)
            else:
                print "Extension error or more than 1 npz/pkl file provided."
        else:
            print "No files"
            setattr(self,map_key,[constant_zero_dict()])
    def coords_to_seq(self,coord,genome_folder=r'/n/dulacfs2/Users/bbintu/Genomes/mouse/mm10',save_file=None):
        "given coords of form chr*:*-* and a genome folder, returns the sequence and saves to file if save_file not None"
        import fnmatch
        if fnmatch.fnmatchcase(coord,'chr*:*-*'):
            chr_=coord.split(':')[0]
            if not hasattr(self,'seq_'+chr_):
                chr_file = genome_folder+os.sep+chr_+'.fa'
                nms,seq = fastaread(chr_file)
                seq = seq[0]
                setattr(self,'seq_'+chr_,seq)
            else:
                seq = getattr(self,'seq_'+chr_)
            
            start = int(coord.split(':')[-1].split('-')[0])
            end = int(coord.split('-')[-1])
            
            sq = seq[start:end]
            if save_file is not None:
                fastawrite(save_file,[coord],[sq])
            return sq
    def create_isoform_fl(self,isoform_fls,transcr_fl=None,isoform_filter_name=None,isoform_filter_seq=None):
        """
        Given a list of filenames and a list of isoform_filter_name or a dictionary isoform_filter_seq
        it will look through the loaded transcriptome and save isoforms to each file.
        
        isoform_filter_name = ['*FLNA*','*(Xist)*',...] 
        (list with length matching seqs_RNAs)
        isoform_filter_seq = {'word_size':17,'n_choice':100,'perc_keep':0.3} 
        (isoforms are selected if a percent of inclusion of higher than 0.3 of a random set of 100 17mers)
        """
        self.isoform_filter_name = isoform_filter_name
        self.isoform_filter_seq = isoform_filter_seq
        self.isoform_fls = isoform_fls
        if transcr_fl is not None:
            self.names_transcr,self.seqs_transcr = fastaread(transcr_fl)
        import fnmatch
        if self.isoform_filter_name is not None:
            if hasattr(self,"names_transcr"):
                #Asume a list of name filters are provided
                list_unfilt = self.names_transcr
                list_filters = self.isoform_filter_name
                #keep only a subset
                self.isoform_names_list,self.isoform_seqs_list = [],[]
                for filter_,isoform_fl in zip(list_filters,isoform_fls):
                    isoform_names = fnmatch.filter(list_unfilt, '*'+filter_+'*')
                    isoform_sequences  = [self.seqs_transcr[self.names_transcr.index(is_name)] for is_name in isoform_names]
                    #make list
                    self.isoform_names_list.append(isoform_names)
                    self.isoform_seqs_list.append(isoform_sequences)
                    print "Writing fasta file "+str(isoform_fl)+" with number of seqs: "+str(len(isoform_names))
                    fastawrite(isoform_fl,isoform_names, isoform_sequences)
            else:
                print "Warning! No isoforms computed because the transciptome is not loaded."
        if self.isoform_filter_seq is not None:
            if hasattr(self,"names_transcr"):
                word_size_ = self.isoform_filter_seq['word_size']
                n_choice_ = self.isoform_filter_seq['n_choice']
                perc_keep_ = self.isoform_filter_seq['perc_keep']
                
                self.isoform_names_list,self.isoform_seqs_list = [],[]
                for seq_ref,isoform_fl in zip(self.seqs_RNAs,isoform_fls):
                    i_choice = np.random.choice(range(len(seq_ref)-word_size_),n_choice_)
                    seq_choice = [seq_ref[i_:i_+word_size_] for i_ in i_choice]
                    specs = [np.sum([sq in seq_transcr for sq in seq_choice])/float(n_choice_) for seq_transcr in self.seqs_transcr]
                    index_keep = np.ravel(np.where(np.array(specs)>perc_keep_))
                    
                    isoform_names = [self.names_transcr[i_] for i_ in index_keep]
                    isoform_seqs = [self.seqs_transcr[i_] for i_ in index_keep]
                    
                    self.isoform_names_list.append(isoform_names)
                    self.isoform_seqs_list.append(isoform_seqs)
                    fastawrite(isoform_fl,isoform_names, isoform_seqs)
            else:
                print "Warning! No isoforms computed because the transciptome is not loaded."
                
    
    def compute_pb_report(self):
        block = self.word_size
        pb_len = self.pb_len
        buffer_len = self.buffer_len
        check_on_go = self.check_on_go
        gen_seqs,gen_names = self.input_seqs,self.input_names
        input_use_revc = self.sequence_dic.get('use_revc',False)
        input_use_kmer = self.sequence_dic.get('use_kmer',False)
        
        pb_reports = {}
        self.pb_reports_keep = {}
        
        #Iterate across regions:
        for k in range(len(gen_seqs)):
            print "Dealing with sequence: " + str(k+1)+" out of "+str(len(gen_seqs))
            start = time.time()
            reg_t = gen_seqs[k].replace('N','A')
            #Iterate through probes:
            checks_ = np.zeros(len(reg_t))
            for i in range(len(reg_t)-pb_len):
                pb_t = reg_t[i:i+pb_len]
                if checks_[i]==0:
                    pb_reports[pb_t] = constant_zero_dict()
                    pb_reports[pb_t].update({'name':gen_names[k]+'_pb_'+str(i),'reg_index':k,'reg_name':gen_names[k],'pb_index':i,'gc':gc(pb_t),'tm':tm(pb_t)})
                    
                    #Iterate through maps:
                    for key in self.map_dic.keys():
                        curr_dic = self.map_dic[key]
                        use_kmer = curr_dic.get('use_kmer',False)
                        use_revc = curr_dic.get('use_revc',False)

                        map_key = 'map_'+key
                        maps=getattr(self,map_key)
                        
                        #for python dictionary:
                        if not use_kmer:
                            #Iterate through block regions:
                            for j in range(pb_len-block+1):
                                blk_t = pb_t[j:j+block]
                                map_=maps[k%len(maps)]
                                blks = [pb_t]
                                if use_revc:
                                    blks.append(seqrc(pb_t))
                                for blk_ in blks:
                                    pb_reports[pb_t][key]+= int(map_.get(blk_int))
                                    
                        #for sparse matrix:
                        if use_kmer:
                            if use_revc:
                                pb_reports[pb_t][key]+= map_.get(pb_t)+map_.get(pb_t,rc=True)
                            else:
                                pb_reports[pb_t][key]+= map_.get(pb_t)
                    if check_on_go:
                        if self.perform_check(pb_reports[pb_t]):
                            checks_[i:i+pb_len+buffer_len]=1
                            self.pb_reports_keep[pb_t] = pb_reports[pb_t]
            self.pb_reports = pb_reports
            self.save_pbr()
            end=time.time()
            print "Time(s): "+str(end-start)    
        

    def perform_check(self,pb_report):
        dic_check = self.dic_check
        off_maps = True
        for key in dic_check.keys():
            if not (key in ['gc','tm']):
                if type(key) is str:
                    off_maps = off_maps and (pb_report[key]<=dic_check[key])
                    if pb_report[key]>=self.max_count-1:
                        off_maps = False
                elif type(key) in [list,tuple]:
                    off_maps = off_maps and ((pb_report[key[0]]-pb_report[key[1]])<=dic_check[key])
                    if pb_report[key[0]]>=self.max_count-1:
                        off_maps = False
        gc_ch = True
        if dic_check['gc'] is not None:
            if type(dic_check['gc']) in [list,tuple]:
                gc_ch=pb_report['gc']>=np.min(dic_check['gc']) and pb_report['gc']<=np.max(dic_check['gc'])
            else:
                gc_ch=pb_report['gc']>=dic_check['gc']
        tm_ch = True
        if dic_check['tm'] is not None:
            if type(dic_check['tm']) in [list,tuple]:
                tm_ch=pb_report['tm']>=np.min(dic_check['tm']) and pb_report['tm']<=np.max(dic_check['tm'])
            else:
                tm_ch=pb_report['tm']>=dic_check['tm']
        return off_maps and gc_ch and tm_ch
    def perform_check_end(self):
        print "Picking non-overlaping sequences."
        start = time.time()
        self.pb_reports_keep = {}
        pb_reports = self.pb_reports
        keys_ = pb_reports.keys()
        keys_seqs = partition_map(keys_,[pb_reports[key_]['reg_index'] for key_ in keys_])
        for keys_seq in keys_seqs:
            index_key = [pb_reports[key_]['pb_index'] for key_ in keys_seq]
            keys_seq_sort = np.array(keys_seq)[np.argsort(index_key)]
            checks_ = np.zeros(np.max(index_key)+1)
            for key in keys_seq_sort:
                pb_report = pb_reports[key]
                i = pb_report['pb_index']
                if checks_[i]==0:
                    if self.perform_check(pb_report):
                        self.pb_reports_keep[key] = pb_report
                        checks_[i:i+self.pb_len+self.buffer_len]=1
        self.save_pbr()
        end = time.time()
        print "Time(s): "+str(end-start)
    def single_plot(self,sz_bin=5000,region = 0,save_file_png=None,show=True):
        "Plot figure with distribution of probes"
        import matplotlib.pyplot as plt
        indexes_keep=[self.pb_reports_keep[key]['pb_index'] for key in self.pb_reports_keep.keys() 
                      if self.pb_reports_keep[key]['reg_index']==region]
        nm,ln_sq=self.input_names[region],self.sequence_lens[region]
        fig = plt.figure()
        plt.hist(range(0,ln_sq,self.pb_len+self.buffer_len),bins=int(np.ceil(ln_sq/sz_bin)),range=(0,ln_sq))
        plt.hist(indexes_keep,bins=int(np.ceil(ln_sq/sz_bin)),range=(0,ln_sq))
        plt.title(nm+'\\n'+'No. probes kept:'+str(len(indexes_keep)))
        if show:
            plt.show()
        if save_file_png is not None:
            fig.savefig(save_file_png)
        plt.close('all')
    def savefile_to_savefiles(self):
        "Analyses the current savefile if folder or does not contain enough files switches."
        if self.sequence_dic['file'] is not None and self.save_file is not None:
            input_files = self.str_to_list(self.sequence_dic['file'])
            input_noSeqsperFile = self.input_noSeqsperFile

            base_saves = []
            for input_file,input_noSeqs in zip(input_files,input_noSeqsperFile):
                base_input_fl = os.path.basename(input_file).split(os.path.extsep)[0]
                for seq_i in range(input_noSeqs):
                    if input_noSeqs>1:
                        base_saves.append(base_input_fl+'_seq'+str(seq_i)+'.pbr')
                    else:
                        base_saves.append(base_input_fl+'.pbr')
            curr_save_files=self.str_to_list(self.save_file)
            if len(curr_save_files)<len(base_saves):
                if not os.path.isdir(curr_save_files[0]):
                    save_folder = os.path.dirname(curr_save_files[0])
                else:
                    save_folder = curr_save_files[0]
                self.save_files = [save_folder+os.sep+base_save for base_save in base_saves]
            else:
                self.save_files = curr_save_files
            
    def plots(self,sz_bin=5000.,show=False):
        if hasattr(self,'plot_bp'):
            sz_bin_ = self.plot_bp
        else:
            sz_bin_ = sz_bin
        regions = np.unique([self.pb_reports_keep[key]['reg_index'] for key in self.pb_reports_keep.keys()])
        if not hasattr(self,'savefiles'):
            self.savefile_to_savefiles()
        for region in regions:
            if self.save_file is not None:
                save_file_png= self.save_files[region].replace('.pbr','.png')
            else:
                save_file_png=None
            self.single_plot(sz_bin=sz_bin_,region = region,save_file_png=save_file_png,show=show)

## Functions for research computing (cluster) machines operating under SLURM
def slurm_python(python_file,n=1,N=1,t='0-03:00',p='serial_requeue',mem=32000,
                 err_file=None,out_file=None,slurm_script=None,job_index=False):
    "Given python file and slurm specs this launches a python file in sbatch."
    base_name = python_file.replace('.py','')
    if job_index: job_str = '_%j'
    else: job_str = ''
    if err_file is None: err_file=base_name
    if out_file is None: out_file=base_name
    if slurm_script is None: slurm_script = base_name+'.sh'
    string = """#!/bin/bash
#SBATCH -n """+str(n)+"""                    # Number of cores
#SBATCH -N """+str(N)+"""                    # Ensure that all cores are on one machine
#SBATCH -t """+str(t)+"""              # Runtime in D-HH:MM
#SBATCH -p """+str(p)+"""       # Partition to submit to
#SBATCH --mem="""+str(mem)+"""               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o """+out_file+job_str+""".out      # File to which STDOUT will be written
#SBATCH -e """+err_file+job_str+""".err      # File to which STDERR will be written


. new-modules.sh && module load python/2.7.6-fasrc01 && source activate python27rc

python """+python_file+"""
"""
    fid = open(slurm_script,'w')
    fid.write(string)
    fid.close()
    return subprocess.check_output("sbatch "+slurm_script,shell=True)
def python_scripts(notebook_string,params,save_folder,file_base='script'):
    "Creates a set of python files in the save_folder with text: notebook_string(params[i])"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_files=[]
    for i,param in enumerate(params):
        save_file = save_folder+os.sep+file_base+'_param'+str(i)+'.py'
        save_files.append(save_file)
        python_script(notebook_string(param),save_file)
    return save_files
def python_script(notebook_string_,save_file):
    fid = open(save_file,'w')
    fid.write(notebook_string_)
    fid.close()
def get_map(map_):
    map__=[]
    for elem in map_:
        if elem not in map__:
            map__.append(elem)
    map__=np.array(map__)
    return map__
def partition_map(list_,map_,mapU=None,return_map=False):
    """
    Inputs
    takes a list [e1,e2,e3,e4,e5,e6] and a map (a list of indices [0,0,1,0,1,2]).  map can be a list of symbols too. ['aa','aa','bb','aa','bb','cc']
    Output
    returns a sorted list of lists, e.g. [[e1, e2,e4],[e3,e5],[e6]]
    """
    list__=np.array(list_)
    map__=np.array(map_)
    if mapU is None:
        mapU = np.unique(map__)
    if type(mapU)==str:
        if mapU=='ordered':
            mapU=get_map(map_)
    if return_map:
        return [list(list__[map__==element]) for element in mapU],list(mapU)
    return [list(list__[map__==element]) for element in mapU]
def tm(string):
    return mt.Tm_NN(string, nn_table=mt.DNA_NN4,Na=330)#330 for 2xSSC
def gc(string):
    return float(string.count('g')+string.count('G')+string.count('c')+string.count('C'))/len(string)

def notebook_string(in_file):
    return """
#This is an auto generated string

##imports
import time,os,sys,glob
import cPickle as pickle
import numpy as np

import khmer
sys.path.append(r'/n/dulacfs2/Users/bbintu/python-functions/python-functions-library')
from LibraryConstruction import fastaread,fastawrite
import LibraryDesigner as ld

in_file = r'"""+in_file+"""'
save_file = r'/n/dulacfs2/Users/bbintu/Libraries/SI10/Reports'+os.sep+os.path.basename(in_file).replace('.fasta','.pbr')

transcriptome_fl = r'/n/dulacfs2/Users/bbintu/Transcriptomes/mouse/mm10_rna_word17_.kmer'
genome_fl = r'/n/dulacfs2/Users/bbintu/Genomes/mouse/mm10/full_word17_.kmer'
rep_transcriptome_fl = r'/n/dulacfs2/Users/bbintu/Transcriptomes/mouse/rtRNA.fasta'
rep_genome_fl = r'/n/dulacfs2/Users/bbintu/Genomes/mouse/mm10/repeatSequences.fasta'
local_genome_fl = r'/n/dulacfs2/Users/bbintu/Genomes/mouse/mm10/chrX_word17_.kmer'

pb_designer = ld.pb_reports_class(
    sequence_dic={'file':in_file,'use_revc':True,'use_kmer':True},
    map_dic={'transcriptome':{'file':transcriptome_fl,'use_revc':False,'use_kmer':True},
          'genome':{'file':genome_fl,'use_revc':True,'use_kmer':True},
          'rep_transcriptome':{'file':rep_transcriptome_fl,'use_revc':True,'use_kmer':True},
          'rep_genome':{'file':rep_genome_fl,'use_revc':True,'use_kmer':True},
          'local_genome':{'file':local_genome_fl,'force_list':True,'use_revc':True,'use_kmer':True}},
    save_file=save_file,
    params_dic={'word_size':17,'pb_len':42,'buffer_len':2,'max_count':2**16-1,'check_on_go':False,'auto':False},
    dic_check={'transcriptome':3,('genome','local_genome'):20,
                'rep_transcriptome':0,'rep_genome':0,'gc':[0.25,0.75],'tm':70})

pb_designer.computeOTmaps()
pb_designer.compute_pb_report()
pb_designer.perform_check_end()
pb_designer.plots()

"""
