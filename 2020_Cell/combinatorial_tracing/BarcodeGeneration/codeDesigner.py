"""
Authors: Seon Kinrot, Bogdan Bintu
These are the functions for creation of encodings for TADs and chromosomes
"""

#Dependencies
import itertools
import numpy as np
from scipy.spatial.distance import pdist

def flatten(list_):
    ### flattens one layer of a list. For example, [[0,1],[2,3],[4,[5,6]]] will become [0,1,2,3,4,[5,6]]
    return [item for sublist in list_ for item in sublist]

def fastaread(fl,force_upper=False):
    ### reads fasta files and returns a list of names and sequences
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
    
def combs_to_code(combs_eq_sp,nchr=None):
    """Construct from combs list to code binary array
    For example changes:
    combs_eq_sp = [(0,1,2,3),(0,1,2,4)] to codes = [[1,1,1,1,0],[1,1,1,0,1]]
    """
    if nchr is None:
        nchr = np.max(combs_eq_sp)+1
    codes = np.zeros([len(combs_eq_sp),nchr],dtype=int)
    for i,comb in enumerate(combs_eq_sp):
        codes[i][list(comb)] = 1
    return codes
    
def hybes_to_TADcode(code,hybes):
    """
    Takes code and hybes and converts them to a direct correspondence of codes and TADs
    Returns: a list of the same length as code, containing the TAD id encoded as chr_id*nTADs+TAD_id, or 0 if the code is unassigned
    """
    tads_assign=np.zeros(len(code),dtype=int)
    for icode,code_ in enumerate(code):
        code_hybes=np.where(code_)[0]
        tad_=[tad_id+1 for tad_id in hybes[code_hybes[0]] if tad_id in hybes[code_hybes[1]]]
        if tad_:
            tads_assign[icode]=int(tad_[0])
    return tads_assign
    
def code_encoder_v2(nTADs=None,refine_chr=100000,refine_TAD=50000,nchr=23,nhybes=80,buffer_=100):
    """
    nTADs - a list of the number of TADs selected from each chromosome
    Master function for the encoder
    refine_chr - number of switches attempted in refinement of chr code (pick_choose2_code)
    refine_TAD - number of switche sattempted in refinement of TAD assignment (assign_TADs_choose2)
    nchr - the number of *unique* - i.e. non-homologous - chromosomes
    nhybes - the numbr of hybes in the experiment
    buffer_ - a multiplier from chr number in identifying TADs. TAD IDs in the output are (buffer_)x(chr_id-1)+(TAD_ID)
    
    Return: 
    hybes - a list of lists with TAD ids per hybe (see choose2_order_to_hybes)
    tads_assign - assignment of TADs to each code, not including homologues (see hybes_to_TADcode)
    """
    if nTADs is None:
        nTADs=[100]*nchr
    elif len(nTADs)!=nchr:
        print 'error - invalid list of TAD numbers'
        return [],[]
    code, chrs_assign = init_choose2_code_v2(nTADs,nhybes)
    chrs_assign_0 = pick_choose2_code_v2(code=code,chrs_assign=chrs_assign,nTADs=nTADs,iter=refine_chr)
    chr_tad_order = assign_TADs_choose2_v2(code=code,chrs_assign_0=chrs_assign_0,nTADs=nTADs,iterations=refine_TAD)
    hybes = choose2_order_to_hybes_v2(code=code,chrs_assign_0=chrs_assign_0,chr_tad_order=chr_tad_order,nTADs=nTADs,chr_buff=buffer_)
    tads_assign = hybes_to_TADcode(code=code,hybes=hybes)
    return hybes, tads_assign
   
def init_choose2_code_v2(nTADs, nhybes):
    """
    construct all possibilities for bit codes with 2 1-s in nhybes rounds, with number of TADs per chromosome 
    given by nTADs and assign random starting code
    Returns:
    code - a (nhybes-1)*nhybes*0.5 by nhybes array with numbers. Each row corresponds to a unique possible pair of hybes 
    in which a TAD can appear
    chrs_assign_0 - a list of number in the range [0,nchr] corresponding to which chromosome (if any) is assigned which
    code. Each chromosome (except 0, which means no assignment) gets nTADs codes
    """
    combs = list(itertools.combinations(range(nhybes),2))
    code = combs_to_code(combs)
    chrs_assign = flatten([[i+1]*nTADs[i] for i in range(len(nTADs))]) #(chr_ID)x(TADs _in_chr) for every chromosome
    n_tot = nhybes*(nhybes-1)/2
    chrs_assign = chrs_assign+[0]*(n_tot-len(chrs_assign))
    chrs_assign_0 = np.random.permutation(chrs_assign)
    print 'code initialized'
    return code,chrs_assign_0

def pick_choose2_code_v2(code,chrs_assign,nTADs,iter=100000):
    """
    This function performs pairwise switches of repeat codes to minimize the std for the number of TADs imaged 
    simultaneously from each chromosome (i.e. make it uniform)

    Inputs:
    code - all possible codes for pairs (see init_choose2_code)
    chrs_assign - assignment of codes to chromosomes (see output of init_choose2_code)
    nTADs - a list of the number of TADs selected from each chromosome
    iter - the number of pairwise flip attempts the function will make

    Returns:
    chrs_assign_0 - the final assignment after minimization of the appearance of each chr in the same hybe
    """
    nhybes=code.shape[1]
    n_tot = nhybes*(nhybes-1)/2
    chrs_assign_0=chrs_assign
    nchr = len(nTADs)
    stds = [np.std(np.sum(code[chrs_assign_0==(i+1)],axis=0)) for i in range(nchr)] 
    #list of stds of number of TADs from each chromosome in each hybe
    std_min = np.sum(stds)
    std_hybe = np.std(np.sum(code[chrs_assign_0!=0],axis=0)) #std of total number of spots per hybe
    n0=n_tot-np.sum(nTADs) #number of codes assigned to "chromosome 0"
    n_frac = n0/float(n_tot) #n0's fraction of the total possible codes
    for iter_ in range(iter):
        weights = np.cumsum(stds) #assign weights to help choose the "worst" chromosomes
        weights = weights/float(np.max(weights))
        weights = np.concatenate([[n_frac],weights*(1-n_frac)+n_frac])
        chr1,chr2=0,0
        while(chr1==chr2): 
            #make sure the switch is between different chromosomes - TAD identities within each chromosome are arbitrary
            chr1 = np.sum(np.random.rand()-weights>0)
            chr2 = np.sum(np.random.rand()-weights>0)
        #choose TADs at these chromosomes and try to switch their codes
        tad1_p = np.random.choice(np.where(chrs_assign_0==chr1)[0])
        tad2_p = np.random.choice(np.where(chrs_assign_0==chr2)[0])
        chrs_assign_0_t = np.array(chrs_assign_0)
        chrs_assign_0_t[tad1_p],chrs_assign_0_t[tad2_p]=chr2,chr1
        stds_local = [np.std(np.sum(code[chrs_assign_0_t==(i+1)],axis=0)) for i in range(nchr)]
        std_local = np.sum(stds_local)
        std_hybe_loc = np.std(np.sum(code[chrs_assign_0_t!=0],axis=0))
        if std_local<std_min: #accept switch if std is decreased
            chrs_assign_0=np.array(chrs_assign_0_t)
            std_min = std_local
            stds = stds_local
            std_hybe = std_hybe_loc
        elif std_local==std_min and std_hybe_loc<std_hybe: #if per chromosome there is no change, check if hybes more even overall
            chrs_assign_0=np.array(chrs_assign_0_t)
            std_min = std_local
            stds = stds_local
            std_hybe = std_hybe_loc
    print 'assigned hybes to chromosomes'
    return chrs_assign_0
    
def assign_TADs_choose2_v2(code,chrs_assign_0,nTADs,iterations=50000):
    """
    Takes a choose2 code and chr assignment and assigns TADs to each of the codes. Initially assigns the TADs
    randomly, then uses a metric that maximizes the minimal distance, as well as the mean distance and minimizes
    the std of the linear genomic distance between TADs appearing in the same hybe

    Inputs:
    code - a chromosome choose2 code (see init_choose2_code)
    chrs_assign_0 - an assignment of codes to each chromosome (see pick_choose2_code)
    nTADs - a list of the number of TADs selected from each chromosome 
    iterations - number of iterations for refinement

    Return:
    chr_tad_order - a list of nchr by nTADs, indicating the correspondence between codes spicified in chrs_assign_0
    and the TADs
    """
    nchr = len(nTADs)
    nhybes = code.shape[1]
    
    chr_tad_order=[]
    for ichr_ in range(nchr):
        tads_chr = nTADs[ichr_]
        if tads_chr==0: #trivial case 1 - no TADs in this chromosome
            print 'chr'+str(ichr_)+' has no TADs'
            chr_tad_order.append([])
            continue
        prev_metric=0
        codes_chr=code[chrs_assign_0==ichr_+1]
        tad_asigments_save = np.random.permutation(range(tads_chr))
        prev_min_dist=0
        if tads_chr<3: #trivial case 2 - no meaning in changing order
            print 'chr'+str(ichr_)+' has less than 3 regions. No need to optimize'
            chr_tad_order.append(tad_asigments_save)
            continue
        for iter_ in range(iterations):
            tad_asigments=np.array(tad_asigments_save)
            tad1=0
            tad2=0
            while tad2==tad1: #choose 2 unique TADs
                tad1,tad2=np.random.randint(tads_chr),np.random.randint(tads_chr)
            tad_asigments[tad1],tad_asigments[tad2]=tad_asigments[tad2],tad_asigments[tad1]
            tads,hybes = np.where(codes_chr[tad_asigments])
            tad_loc = [tads[hybes==i] for i in range(nhybes)]
            tad_dists = map(lambda x: pdist(np.array([x]).T,'minkowski', 1),tad_loc)
            tad_dists_ = flatten(tad_dists)
            # metric for choosing configuration maximizing distance between TADs in same hybe. Below see example for 6 TADS.
            # cosider three possibilities: [(1,2),(3,4),(5,6)], [(1,6),(2,5),(3,4)], [(1,4),(2,5),(3,6)]. We would like to 
            # choose the last one, but what metric points to it? The mean of the distance between pairs is natural, but is 
            # the same between options 2,3. The std tells them apart, but is the same for 1,3. The difference or ratio 
            # between the two, though, is a unique predictor (at least in this simple example, although the ratio is infinite)
            if len(tad_dists_):
                metric = np.mean(tad_dists_)/np.std(tad_dists_)
                min_dist = np.min(tad_dists_)
            else:
                metric, min_dist = np.inf, np.inf
            if min_dist>prev_min_dist:
                prev_min_dist=min_dist
                prev_metric=metric
                tad_asigments_save=tad_asigments
            elif min_dist==prev_min_dist:
                if metric>prev_metric:
                    prev_min_dist=min_dist
                    prev_metric=metric
                    tad_asigments_save=tad_asigments
        chr_tad_order.append(tad_asigments_save)
        print 'chr'+str(ichr_+1)+' - min dist: %d ; mean dist: %3.2f'%(prev_min_dist,np.mean(tad_dists_))
    return chr_tad_order
    
def choose2_order_to_hybes_v2(code,chrs_assign_0,chr_tad_order,nTADs,chr_buff=100):
    """
    Takes the order of TADs, chromosome codes and total set of codes to create a list of TAD IDs per hybe
    
    Inputs:
    code - list of all codes available (see init_choose2_code)
    chrs_assign_0 - the codes assigned to each chromosome (see pick_choose2_code)
    chr_tad_order - the order of correspondence between codes and TADs for each chr (see assign_TADs_choose2)
    chr_buff - the number which multiplies each chr number in identifying its spots. should be bigger than max(nTADs)
    
    Returns:
    hybes - a list of nhybes lists. Each entry x=chr_buff*chr_id+TAD_id, all ids in Python notation. For example, TAD 3 in chr 5 will be 402
    """
    nchr = len(nTADs)
    nhybes = code.shape[1]
    hybes=[[] for ih in range(nhybes)]
    for ichr_ in range(nchr):
        codes_chr=code[chrs_assign_0==ichr_+1]
        chr_order = chr_tad_order[ichr_]
        #print chr_order
        if not len(codes_chr):
            continue
        tads,hybes_ = np.where(codes_chr[chr_order])
        tad_loc = [tads[hybes_==i]+chr_buff*ichr_ for i in range(nhybes)]
        hybes=[np.concatenate([hybes[i],tad_loc[i]],axis=0) for i in range(nhybes)]
    return hybes
    
def split_code(code,hybes,buff_=100):
    """
    Takes a code and splits it to a sparser code in twice the number of hybes.
    Returns the new code, the new hybes and the new tad assignments
    """
    nhybes = code.shape[1]
    new_code = combs_to_code(list(itertools.combinations(range(nhybes*2),2)))
    new_hybes = [[] for h_ in range(nhybes*2)]
    new_tads_assign = []
    shift = nhybes

    for ihb_,hb_ in enumerate(hybes):
        def_ = np.random.randint(2) #default "half" to add an extra point to
        for ichr_ in range(23):
            tads_chr_hb = np.array([t_ for t_ in hybe_ if ichr_*buff_<=t_<(ichr_+1)*buff_])
            for itd_,tad_ in enumerate(tads_chr_hb):
                new_hybes[ihb_+(itd_%2!=def_)*shift].extend([tad_])
            parity = len(tads_chr_hb)%2 #is there an even number of TADs from this chromosome in this hybe
            if parity:
                def_ = 1-def_
    new_hybes = np.array(new_hybes)
    new_tads_assign = hybes_to_TADcode(new_code,new_hybes)
    print 'split code in two'
    return new_code, new_hybes, new_tads_assign