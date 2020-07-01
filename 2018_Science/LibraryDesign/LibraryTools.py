import numpy as np
import os

class constant_key_dict (dict):
    def __missing__ (self, key):
        return key
def seqrc(string):
    "returns the reverse complement of a sequence"
    dic_rc = constant_key_dict({'a':'t','c':'g','g':'c','t':'a','A':'T','C':'G','G':'C','T':'A'})
    string_rc = "".join([dic_rc[c] for c in string][::-1])
    return string_rc
def getFastaWeb(chrom,pos1,pos2):
    "For mouse genome this impots from the infojax website the chr:pos1:pos2 genomic sequence"
    import urllib2
    site_name = "http://www.informatics.jax.org/seqfetch/tofasta.cgi?seq1=mousegenome%21%21"+str(chrom)+"%21"+str(pos1)+"%21"+str(pos2)+"%21%21&flank1=0"
    html = str(urllib2.urlopen(site_name).read())
    sequence = "".join(html.split("\n")[1:-1])
    return sequence
def getGeneWeb(gene_id):
    "For mouse genome this impots from the infojax website a given gene its genomic sequence"
    import urllib2
    site_name = "http://useast.ensembl.org/Mus_musculus/Export/Output/Gene?db=core;flank3_display=0;flank5_display=0;g="+gene_id+";output=fasta;strand=feature;genomic=unmasked;_format=Text"
    html = str(urllib2.urlopen(site_name).read())
    sequence = "".join(html.split("\r\n")[2:-1])
    return sequence
def split_pieces(save_fl,coords,gen_seq,name_seq,piece_length=int(10E3)):
    num_pieces = round(len(gen_seq)/piece_length)
    gen_seqs = []
    names_seqs = []
    for i in range(int(num_pieces)):
        start = i*piece_length
        end = min((i+1)*piece_length,len(gen_seq))
        gen_seqs.append(gen_seq[start:end])
        names_seqs.append("chr"+str(coords[0])+":"+str(int(coords[1])+start)+"-"+str(int(coords[1])+end)+" region="+name_seq+"_pt_"+str(i))    
    fastawrite(save_fl,names_seqs,gen_seqs)
    return names_seqs,gen_seqs
def fastawrite(file_name,names,seqs,append=False):
    if append:
        print "do something"
    f=open(file_name,'w')
    for i in range(len(names)):
        name = names[i]
        seq = seqs[i]
        f.write('>'+name+'\n'+seq+'\n')
    f.close()
def deal_with_masks0(seq_test,masks = ["GGGGGG","CCCC","TTTTTT","AAAA"],dicRepMask = {"G":"C","C":"G","A":"T","T":"A","g":"c","c":"g","a":"t","t":"a"}):
    dicMask = {mask[0]:len(mask) for mask in masks}
    
    seq_t=seq_test.upper()
    contigs=[]
    prev_let="N"
    count=1
    for i_let in range(len(seq_t)):
        let = seq_t[i_let]
        if prev_let==let:
            count+=1
        else:
            contigs.append([prev_let,i_let,count])
            prev_let=let
            count=1
    contigs.append([prev_let,i_let,count])
    contigs = contigs[1:]
    contig_danger = [contig for contig in contigs if dicMask[contig[0]]<=contig[-1]]
    replace_indx = [0]
    replace_indx.extend([item for sublist in [range(contig[1]-contig[2],contig[1])[dicMask[contig[0]]/2::dicMask[contig[0]]] for contig in contig_danger] for item in sublist])
    replace_indx.append(len(seq_t))
    seqs_split = [dicRepMask[seq_test[replace_indx[i]]]+seq_test[replace_indx[i]+1:replace_indx[i+1]] for i in range(len(replace_indx)-1)]
    seq_final = seq_test[0]+"".join(seqs_split)[1:]
    return seq_final,len(replace_indx)-2
def deal_with_masks(seq_test,masks = ["GGGGGG","CCCC","TTTTTT","AAAA"],dicRepMask = {"G":"C","C":"G","A":"T","T":"A","g":"c","c":"g","a":"t","t":"a"}):
    lenrep_t=0
    seq_final_,lenrep = deal_with_masks0(seq_test,masks=masks,dicRepMask=dicRepMask)
    lenrep_t+=lenrep
    while(sum([seq_final_.upper().count(mask) for mask in masks])>0):
        seq_final_,lenrep = deal_with_masks0(seq_final_,masks=masks,dicRepMask=dicRepMask)
        lenrep_t+=lenrep
    return seq_final_,lenrep_t
def up_down_split(seq):
    splits = ['']
    c0 = seq[0]
    for c in seq:
        if c.islower()==c0.islower():
            splits[-1]+=c
        else:
            c0 = c
            splits.append(c)
    return splits
def up_down(seqs):
    count=0
    seqs_ = []
    for seq in seqs:
        count+=1
        if np.mod(count,2):
            seqs_.append(seq.upper())
        else:
            seqs_.append(seq.lower())
    return "".join(seqs_)
def fastaread(fl,force_upper=False):
    fid = open(fl,'r')
    names = []
    seqs = []
    lines = []
    while True:
        line = fid.readline()
        if not line:
            seq = "".join(lines)
            if force_upper:
                seq=seq.upper()
            seqs.append(seq)
            break
        if line[0]=='>':
            name = line[1:-1]
            names.append(name)
            seq = "".join(lines)
            if force_upper:
                seq=seq.upper()
            seqs.append(seq)
            lines = []
        else:
            lines.append(line[:-1])
    fid.close()
    return [names,seqs[1:]]
def nt2intblock(gen_seq):
    int_gen_seq = nt2int(gen_seq)
    block = len(gen_seq)
    base = 5**np.arange(block,dtype="int64")
    hit = np.dot(base,int_gen_seq)
    return hit
def nt2intblock_rc(gen_seq):
    int_gen_seq = nt2int(gen_seq)
    int_gen_seq = 4-int_gen_seq[::-1]
    block = len(gen_seq)
    base = 5**np.arange(block,dtype="int64")
    hit = np.dot(base,int_gen_seq)
    return hit
def nt2int(gen_seq):
    nt2int_dic=constant_minus_one_dict({'a':0,'c':1,'g':2,'t':3,'A':0,'C':1,'G':2,'T':3})
    int_gen_seq = [nt2int_dic[ch] for ch in gen_seq]
    int_gen_seq = np.array(int_gen_seq)+1
    return int_gen_seq
def single_report(pb_t,maps,block):
    pb_t_counts = np.zeros(len(maps))
    for j in range(len(pb_t)-block+1):
        blk_t = pb_t[j:j+block]
        ##Deal with maps:
        blk_t_counts = np.array([OTTableMap[blk_t] for OTTableMap in maps])
        pb_t_counts+=blk_t_counts
    return pb_t_counts
def fastacombine(files_,new_file = None):
    #This combines multiple fasta files and returns names,seqs
    names_f,seqs_f = [],[]
    for file_ in files_:
        names_,seqs_ = fastaread(file_)
        names_f.extend(names_)
        seqs_f.extend(seqs_)
    if new_file!=None:
        fastawrite(new_file,names_f,seqs_f)
    return names_f,seqs_f
def pad_seqs(seqs_,max_len=None,pad=None):
    #The pad comes from the last 10 orthogonal primers in my old list
    if pad is None:
        pad = 'CACGGGCCGGATTGGTTGAGATGGCGACGGGTCACTTGCCCGATGAAGCGCCTGTTTGCGTACGGGTCATCGCCCTTGGCTTCGTGCCTCTTGGGTTGGGTGCCCGTTCCCGATCTTGTCATCGGGTCGCCGCGATTTAGTATGCATTGGCCGCGTTTCCTTCGGAGGCGTCACGTTTCGTGACACGCGACCGACTTTGG'
    #pad="".join(primers_all[-((max_len-min_len)/20+1):])
    if max_len is None:
        max_len = max(map(len,seqs_))
    seqs_pad = []
    for seq in seqs_:
        padlen = max_len-len(seq)
        pad_left = min(5,padlen)
        pad_right = padlen-pad_left
        if seq[0].islower():
            padL = pad[:pad_left].upper()
        else:
            padL = pad[:pad_left].lower()
        if seq[-1].islower():
            padR = pad[-pad_right:].upper()
        else:
            padR = pad[-pad_right:].lower()
        if pad_right==0:
            padR=""
        seqs_pad.append(padL+seq+padR)
    return seqs_pad
def combine_lib(files_,deal_mask=False,masks = ["GGGGGG","CCCC","TTTTTT","AAAA"]):
    print "Combining library"
    names_,seqs_=[],[]
    for file_ in files_:
        names_t,seqs_t = fastaread(file_)
        names_.extend(names_t)
        seqs_.extend(seqs_t)
    primersA = [seq[:20] for seq in seqs_]
    primersB = [seq[-20:] for seq in seqs_]
    max_len = max([len(seq) for seq in seqs_])
    min_len = min([len(seq) for seq in seqs_])
    seqs_pad = pad_seqs(seqs_,max_len)
    if deal_mask:
        print "Dealing with masks"
        seqs_mask=[deal_with_masks(seq,masks=masks) for seq in seqs_pad]
        seqs_f = [seq[0] for seq in seqs_mask]
        changes = [seq[1] for seq in seqs_mask]
        print "Changes:"+str(np.unique(changes,return_counts=True))
        
    #Rough test mispriming
    print "Performing rough test for mispriming"
    maps_seqs_12 = [computeOTTable(seq,12) for seq in seqs_f]
    maps_seqs_19 = [computeOTTable(seq,19) for seq in seqs_f]
    primersAB = ["".join(zipAB) for zipAB in zip(primersA,primersB)]
    primersAB_unq,primersA_inv = np.unique(primersAB,return_inverse=True)
    primersA_unq = np.array([prm[:20] for prm in primersAB_unq])
    primersB_unq = np.array([prm[-20:] for prm in primersAB_unq])
    mis_primed,mis_useA,mis_useB = [],[],[]
    for index_prm in range(len(primersA_unq)):
        primer_tA = primersA_unq[index_prm]
        primer_tB = primersB_unq[index_prm]
        vals0=np.zeros(len(seqs_f))
        vals0[primersA_inv==index_prm]=len(primer_tA)-12
        mis_primed.append(np.sum(((single_report(primer_tA,maps_seqs_12,12)-vals0)>0)&((single_report(primer_tB,maps_seqs_12,12)-vals0)>0)))
        vals0=np.zeros(len(seqs_f))
        vals0[primersA_inv==index_prm]=len(primer_tA)-19
        mis_useA.append(np.sum((single_report(primer_tA,maps_seqs_19,19)-vals0)>0))
        mis_useB.append(np.sum((single_report(primer_tB,maps_seqs_19,19)-vals0)>0))
    print "[Mis-primed(12bp both fwd and rev),Mis-used_A,Mis-used_B]:"+str([sum(mis_primed),sum(mis_useA),sum(mis_useB)])
    return names_,seqs_f
def computeOTTable(gen_seq_,block,FPKM = None,verbose = False,progress_report=False):
    #This takes a sequence <gen_seq_> (or a list of sequences) and computes the specificity table as a modified dictionary with values spanning up to 5^block
    OTTableDic = constant_zero_dict()
    count_gen = 0
    recount_gen = 0
    if type(gen_seq_)==list:
        gen_seqs_ = gen_seq_
    else:
        gen_seqs_ = [gen_seq_]
    for gen_seq in gen_seqs_:
        if len(gen_seq)>block:
            arr = [gen_seq[i:i+block] for i in range(len(gen_seq)-block+1)]
            arr = np.array(arr)
            if verbose: print "Computing dictionary"
            if FPKM is None:
                for key in arr:
                    OTTableDic[key]+=1
            else:
                FPKM_ = FPKM[count_gen]
                for key in arr:
                    OTTableDic[key]+=FPKM_
        count_gen += 1
        recount_gen += 1
        if progress_report:
            if recount_gen>len(gen_seq_)/100:
                recount_gen = 0
                print str(count_gen*100/len(gen_seq_))+"% complete"
    return OTTableDic
def computeIndexTable(gen_seq_,block,verbose = False,progress_report=False):
    #This takes a sequence <gen_seq_> (or a list of sequences) and computes the specificity table as a modified dictionary with values spanning up to 5^block
    IndexTableDic = constant_list_dict()
    count_gen = 0
    recount_gen = 0
    if type(gen_seq_)==list:
        gen_seqs_ = gen_seq_
    else:
        gen_seqs_ = [gen_seq_]
    for gen_seq in gen_seqs_:
        if len(gen_seq)>block:
            if verbose: print "Computing conversion nt2int excluding bad symbols"
            nt2int=constant_minus_one_dict({'a':0,'c':1,'g':2,'t':3,'A':0,'C':1,'G':2,'T':3})
            int_gen_seq =[nt2int[ch] for ch in gen_seq]
            int_gen_seq = np.array(int_gen_seq)+1
            #int_gen_seq = int_gen_seq[int_gen_seq>=0]
        
            if verbose: print "Computing block matrix version of nt2int "
            len_int_gen_seq = len(int_gen_seq)
            arr = [int_gen_seq[i:len_int_gen_seq+i-block+1] for i in range(block)]
            arr = np.array(arr)
        
            if verbose: print "Computing hits"
            base = 5**np.arange(block,dtype="int64")
            hits = np.dot(base,arr)
            if verbose: print "Computing dictionary"
            for key in hits:
                IndexTableDic[key]+=[count_gen]
        count_gen += 1
        recount_gen += 1
        if progress_report:
            if recount_gen>len(gen_seq_)/100:
                recount_gen = 0
                print str(count_gen*100/len(gen_seq_))+"% complete"
    return IndexTableDic
class OTTable (dict):
    """ run this as:
    specTable = OTTable()
    specTableMAP = specTable.computeOTTable(gen_seq,block)
    specTable.Save(filename)

    OR:
    specTable = OTTable()
    specTable.Load(filename)
    specTableMAP = specTable.Map()
    """
    def Load(self,filename):
        import cPickle as pickle
        dic = pickle.load(open(filename, 'rb'))
        self.keys,self.values,self.block,self.filename = dic["keys"],dic["values"],dic["block"],dic["filename"]
    def Save(self,filename):
        import cPickle as pickle
        self.filename = filename
		
        print "Saving keys/values"
        dic = {"keys":self.keys,"values":self.values,"block":self.block,"filename":self.filename}
        pickle.dump(dic,open(filename, 'wb'),protocol=pickle.HIGHEST_PROTOCOL)
    def Map(self):
        # This computes the map of hash table
        keys,values = self.keys,self.values
        self.OTTableDic = constant_zero_dict()
        print "Computing dictionary"
        for i in range(len(keys)):
            self.OTTableDic[keys[i]]=values[i]
        return self.OTTableDic
    def computeOTTable(self,gen_seq_,block,FPKM = None,verbose = False,progress_report=False):
        #This takes a sequence <gen_seq_> (or a list of sequences) and computes the specificity table as a modified dictionary with values spanning up to 5^block
        OTTableDic = constant_zero_dict()
        count_gen = 0
        recount_gen = 0
        if type(gen_seq_)==list:
            gen_seqs_ = gen_seq_
        else:
            gen_seqs_ = [gen_seq_]
        for gen_seq in gen_seqs_:
            if len(gen_seq)>block:
                arr = [gen_seq[i:i+block] for i in range(len(gen_seq)-block+1)]
                arr = np.array(arr)
                if verbose: print "Computing dictionary"
                if FPKM is None:
                    for key in arr:
                        OTTableDic[key]+=1
                else:
                    FPKM_ = FPKM[count_gen]
                    for key in arr:
                        OTTableDic[key]+=FPKM_
            count_gen += 1
            recount_gen += 1
            if progress_report:
                if recount_gen>len(gen_seq_)/100:
                    recount_gen = 0
                    print str(count_gen*100/len(gen_seq_))+"% complete"
        self.OTTableDic = OTTableDic
        return self.OTTableDic

def tm(string):
    from Bio.SeqUtils import MeltingTemp as mt
    return mt.Tm_NN(string, nn_table=mt.DNA_NN4)
def gc(string):
    return float(string.count('g')+string.count('G')+string.count('c')+string.count('C'))/len(string)
def find_value_dic(dic,key):
    try: 
        return dic[key]
    except:
        return 0
def specificity_report(probes,names,maps,block=17):
    pb_report = []
    for i in range(len(probes)):
        pb_t,name_curr = probes[i],names[i]
        pb_t_counts = np.zeros(len(maps),dtype=int) 
        #Iterate through block regions:
        for j in range(len(pb_t)-block):
            blk_t = pb_t[j:j+block]
            blk_t_rc = seqrc(blk_t)
            ##Deal with maps:
            blk_t_counts = np.array([find_value_dic(OTTableMap,blk_t)+find_value_dic(OTTableMap,blk_t_rc) for OTTableMap in maps])
            pb_t_counts+=blk_t_counts
        ls0 = [pb_t,i,name_curr,tm(pb_t),gc(pb_t)]
        ls0.extend(pb_t_counts)
        pb_report.append(ls0)
    return pb_report

def compute_pair_probes(gene_names,gene_seqs,folder_save="",blk=17,pb_len=30,map_rep=None,map_noGenes=None,maps_Genes=None,FPKM_cuttoff1=2,FPKM_cuttoff2=10,min_gc=0.30,max_gc=0.70,gene_cutoff1=3,gene_cutoff2=5):
    """This is intended to return to you pairs of probes for bTree
    Require paramaters:
    @gene_names : a list of names for the genes you require probes.
    @gene_seqs : a list of sequences (A,C,G,T,N) for the genes you require probes. For exon junctions or masking use N.
    @blk : word size of the maps
    @pb_len : desired probe length
    @map_noGenes : if dictionary: the OT map of FPKM excluding the genes interested in. 
                                It is not necessary to exclude the genes or to use FPKM vs just sequences, but then use the cutoffs appropriately.
                   if None: ignore this test 
                   if string: the path of a fasta file and compute the map on the spot using blk word size
    @maps_Genes : if dictionary: the Index map for the gese interested in. 
                  if None: ignore this test 
                  if string: the path of a fasta file and compute the map on the spot using blk word size
    @map_rep : if dictionary: the OT maps of higly abundant genes. No crosstalk with these genes will be accepted
                  if None: ignore this test 
                  if string: the path of a fasta file and compute the map on the spot using blk word size
    @FPKM_cuttoff1/FPKM_cuttoff2 : min/max cut for the map_noGenes
    @gene_cutoff1/gene_cutoff2 : min/max cut for the maps_Genes
    @pb_len : probe length
    @min_gc/max_gc
    Returns:
    @pb_names_f : names of the probe in the form "<gene name>_pb:<location>_pb-pair:[<pair index>, <index in pair>]"
    @pb_seqs_f : sequences of the probes
    """
    ### Check for variable pop:
    if type(map_noGenes)==str:
        print "Computing the map for transcriptome! Please wait!"
        names_,seqs_ = fastaread(map_noGenes)
        #construct specificity table (dictionary) for all the sequences of the transcripts in MOE.
        specTable = OTTable()
        print "Warning: FPKM might not be computed correctly!"
        FPKM_ = [float(nm.split('_')[-3]) for nm in names_] #change this for generic FPKM
        map_noGenes = specTable.computeOTTable(seqs_,blk,FPKM=FPKM_)

    if type(maps_Genes)==str:
        print "Computing the maps for genes!"
        names_,seqs_ = fastaread(maps_Genes)
        maps_Genes = computeIndexTable(seqs_,blk)
            
    if type(map_rep)==str:
        print "Computing the map for repetitive RNA!"
        names_,seqs_ = fastaread(map_rep)
        specTable = OTTable()
        map_rep = specTable.computeOTTable(seqs_,blk)
    ###Construct the probe scoring function
    def map_seq(seq,map_OT,blk):
        if type(map_OT[0])==list:
            return_val = []
            for i in range(len(seq)-blk+1):
                return_val += map_OT[seq[i:i+blk]]
            gns,cts = np.unique(return_val,return_counts=True)
            return zip(gns,cts)
        else:
            return sum([map_OT[seq[i:i+blk]] for i in range(len(seq)-blk+1)])
        
    pb_names_f,pb_seqs_f=[],[]
    num_pairs_f = []
    ###Iterate through genes:
    for current_gene in range(len(gene_seqs)):
        gene_seq = gene_seqs[current_gene]
        gene_name = gene_names[current_gene]

        locations_recorded = []
        location,pair_ind,pb_pair_ind=0,0,0

        pb_names,pb_seqs=[],[]
        num_pairs = 0
        while True:
            pb1,pb2 = gene_seq[location:location+pb_len],gene_seq[location+pb_len:location+2*pb_len]
            #check where this probe is already in the set
            pb1_verified = False
            if locations_recorded.count(location): pb1_verified = True
            noNs = pb1.count('N') + pb2.count('N') == 0
            if noNs:
                seq_cut = gc(pb1)>=min_gc and gc(pb2)>=min_gc and gc(pb1)<=max_gc and gc(pb2)<=max_gc
                if seq_cut:
                    if map_rep is not None:
                        #Deal with cross-reactivity
                        #--To very abundant mRNA
                        score_rep1 = map_seq(pb1,map_rep,blk)
                        score_rep2 = map_seq(pb2,map_rep,blk)
                        rep_cut = score_rep1==0 and score_rep2==0
                    else:
                        rep_cut=True
                    if rep_cut: 
                        if map_noGenes is not None:
                            #--To all RNA's except the olfrs (using FPKMcuttoff)
                            score_RNA1 = map_seq(pb1,map_noGenes,blk)
                            score_RNA2 = map_seq(pb2,map_noGenes,blk)
                            RNA_cut = min(score_RNA1,score_RNA2)<min(FPKM_cuttoff1,FPKM_cuttoff2) and max(score_RNA1,score_RNA2)<max(FPKM_cuttoff1,FPKM_cuttoff2)
                        else:
                            RNA_cut=True
                        if RNA_cut:
                            if maps_Genes is not None:
                                #--To all olfrs
                                #--To all olfrs
                                scores_Gene1 = map_seq(pb1,maps_Genes,blk)
                                scores_Gene1.pop(np.argmax([sc[1] for sc in scores_Gene1])) # pop the maximum
                                scores_Gene2 = map_seq(pb2,maps_Genes,blk)
                                scores_Gene2.pop(np.argmax([sc[1] for sc in scores_Gene2])) # pop the maximum
                                
                                geneIdx_offGene1 = [score[0] for score in scores_Gene1 if score[1]>gene_cutoff1]
                                geneIdx_offGene2 = [score[0] for score in scores_Gene2 if score[1]>gene_cutoff2]

                                geneIdx_offGene1_ = [score[0] for score in scores_Gene2 if score[1]>gene_cutoff1]
                                geneIdx_offGene2_ = [score[0] for score in scores_Gene1 if score[1]>gene_cutoff2]
                                
                                gene_int = min(len(np.intersect1d(geneIdx_offGene1,geneIdx_offGene2)),len(np.intersect1d(geneIdx_offGene1_,geneIdx_offGene2_)))
                            else:
                                gene_int=0
                            if gene_int==0:
                                # record poisitions:
                                ## create name:
                                if pb1_verified: 
                                    pb_pair_ind+=1
                                    pb_name = gene_name+"_pb:"+str(location+pb_len)+'_pb-pair:'+str([pair_ind,pb_pair_ind])
                                    pb_names.append(pb_name)
                                    pb_seqs.append(pb2)
                                    locations_recorded.append(location+pb_len)
                                    num_pairs+=1
                                else:
                                    pair_ind+=1
                                    pb_pair_ind=1
                                    pb_name = gene_name+"_pb:"+str(location)+'_pb-pair:'+str([pair_ind,pb_pair_ind])
                                    pb_names.append(pb_name)
                                    pb_seqs.append(pb1)
                                    locations_recorded.append(location)
                                    pb_pair_ind+=1
                                    pb_name = gene_name+"_pb:"+str(location+pb_len)+'_pb-pair:'+str([pair_ind,pb_pair_ind])
                                    pb_names.append(pb_name)
                                    pb_seqs.append(pb2)
                                    locations_recorded.append(location+pb_len)
                                    num_pairs+=1
                                gene_seq = gene_seq[:location]+"".join(['N' for k in range(pb_len)])+gene_seq[location+pb_len:]
            location+=1
            if location+2*pb_len>len(gene_seq):
                break
        print gene_name+" (pairs: "+str(num_pairs)+") done!"
        fastawrite(folder_save+os.sep+gene_name+'_probes.fasta',pb_names,pb_seqs)
        pb_names_f.append(pb_names)
        pb_seqs_f.append(pb_seqs)
        num_pairs_f+=[num_pairs]
    return pb_names_f,pb_seqs_f,num_pairs_f
    
def index_no_overlap(starts,ends):
    # This returns the index of probes ensuring that they do not overlap
    starts_i = np.argsort(starts)
    #starts_inv = np.argsort(starts_i)
    starts_ = np.array(starts)[starts_i]
    ends_ = np.array(ends)[starts_i]
    starts_t = list(starts_)
    ends_t = list(ends_)

    danger_tag = True
    while danger_tag:
        danger_tag = False
        for i in range(len(starts_t)-1): 
            if starts_t[i+1]<ends_t[i]:
                danger_tag = True
                danger = i
                del starts_t[i]
                del ends_t[i]
                break
    starts_ = list(starts)
    index_keep = [starts_.index(st) for st in starts_t]
    return index_keep

def map_pair(nms_,sqs_,code,cts_icode,cts_vcode,tails,pair_limit_per_bit):
    nms_new,sqs_new=[],[]
    for nm,sq in zip(nms_,sqs_):
        if nm is nms_[0]:
            str_l = 'i_'
            str_r = 'v_'
            l = code[np.argmin(cts_icode)]
            cts_icode[code.index(l)]+=1
            r = code[np.argmin(cts_vcode)]
        elif nm is nms_[-1]:
            str_l = 'v_'
            str_r = 'i_'
            l = r
            cts_vcode[code.index(l)]+=1
            r = code[np.argmin(cts_icode)]
            cts_icode[code.index(r)]+=1
        else:
            str_l = 'v_'
            str_r = 'v_'
            l = r
            cts_vcode[code.index(l)]+=1
            r = code[np.argmin(cts_vcode)]
        #name modif
        nm_lr=str([str_l+str(l),str_r+str(r)])
        nm+='_code:'+str(code)+'_left-right-new:'+nm_lr
        nms_new.append(nm)
        #seq modif
        l_seq=seqrc(tails[l][15:])
        r_seq=seqrc(tails[r][:15])
        sqs_new.append(up_down([l_seq,seqrc(sq),r_seq]))
        if min(cts_vcode)>=pair_limit_per_bit: break
    return nms_new,sqs_new
def map_gene(pairs_nms_,pairs_sqs_,code,tails,pair_limit_per_bit=10):
    """ Use as:
    map_gene([['name_pb-pair:[1,1]','name_pb-pair:[1,2]','name_pb-pair:[1,3]'],['name_pb-pair:[2,1]','name_pb-pair:[2,2]']],
            [['sq1','sq2','sq3'],['sq4','sq5']],[0,1,2],
            ['0000','1111','2222','3333','4444','5555'],pair_limit_per_bit=10)
    """
    nms_gene,sqs_gene = [],[]
    cts_icode = [0 for i in range(len(code))]
    cts_vcode = [0 for i in range(len(code))]
    for nms_,sqs_ in zip(pairs_nms_,pairs_sqs_):
        nms_new,sqs_new = map_pair(nms_,sqs_,code,cts_icode,cts_vcode,tails,pair_limit_per_bit)
        nms_gene.extend(nms_new)
        sqs_gene.extend(sqs_new)
        if min(cts_vcode)>=pair_limit_per_bit: break
    return nms_gene,sqs_gene
def simmilar(str1,str2,val=10):
    str1_=str1.lower();
    str2_=str2.lower();
    str2_rc=seqrc(str2_);
    sim=sum([str2_.count(str1_[i:i+val]) for i in range(len(str1_)-val+1)])
    sim_rc=sum([str2_rc.count(str1_[i:i+val]) for i in range(len(str1_)-val+1)])
    return sim+sim_rc
def simmilar_list(lst,names=None,seq_len=12):
    problems=[]
    for i in range(len(lst)):
        for j in range(len(lst)):
            if i>j:
                sim=simmilar(lst[i],lst[j],val=seq_len)
                if sim:
                    if names is not None:
                        problems.append([names[i],names[j],sim])
                    else:
                        problems.append([i,j,sim])
    return problems
class constant_zero_dict (dict):
    def __missing__ (self, key):
        return 0
    def get(self,k,d=0):
        return dict.get(self,k,d)
class constant_minus_one_dict (dict):
    def __missing__ (self, key):
        return -1
class constant_nan_dict (dict):
    def __missing__ (self, key):
        return np.nan
class constant_list_dict (dict):
    def __missing__ (self, key):
        return []
def valid_sequence(pb_seq):
    pb_valid = sum([pb_seq.count(let) for let in 'aAtTgGcC'])==len(pb_seq)
    return pb_valid   
def tail_to_lr(seq):
    n_=len(seq)
    return map(seqrc,[seq[n_/2:],seq[:n_/2]])
def lr_to_tail(seqs):
    l,r=seqs
    return seqrc(r)+seqrc(l)
