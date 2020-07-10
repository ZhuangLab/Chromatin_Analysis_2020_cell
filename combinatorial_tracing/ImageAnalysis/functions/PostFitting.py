from __future__ import print_function
import sys,os,glob
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
#import scipy.ndimage as ndi
from copy import deepcopy
import Fitting_v4 as ft
from tqdm import tqdm_notebook as tqdm
try:
    import cPickle as pickle
except:
    import pickle
   
def get_real_tags(htag,flips=['R3,R28','R4,R29','R8,R33','R14,R39','R21,R46','R67,R92','R72,R97']):
    for ilet,let in enumerate(htag[1:]):
        if not let.isdigit():
            break
    htags_real = htag[ilet+1:].split('_')[0]
    flip = -1
    if htags_real in flips or 'Q' in htags_real or 'b' in htags_real:
        flip=1
    htags_real = htags_real.upper()
    htags_real = htags_real.split(',')[::flip]
    return htags_real
def htags_to_Rs(htags):
    htags = np.array(htags)[np.argsort(map(getH,htags))]
    keep_tags = set()
    htags__all = {}
    for htag in htags:
        htags_ = get_real_tags(htag)
        htags__=[]
        for htg in htags_:
            rep = 0
            htg_ = htg
            while True:
                if htg_ not in keep_tags:
                    keep_tags.add(htg_)
                    htags__.append(htg_)
                    break
                else:
                    rep+=1
                    htg_ = htg+'_rep'+str(rep)

        htags__all[htag]=htags__
    return htags__all
    
from scipy.spatial import cKDTree as KDTree
def reduce_matching(all_zn1,all_zo1,all_zn2,all_zo2,distance_upper_bound = 0.01):
    dists,neighs = KDTree(all_zo1).query(all_zo2,distance_upper_bound=distance_upper_bound)
    i2 = np.where(dists<distance_upper_bound*2)[0]
    i1 = neighs[i2]
    return all_zn1[i1].copy(),all_zn2[i2].copy()
def robust_m(cs1_,cs2_,th=70,order=1,no_z=False,ransac=False):
    """Given two matching Nx4 (z,x,y,h) set of fits for beads this returns the m matrix 
    using ft.calc_color_matrix(points1,points2).
    <th> is the percentile threshold for brightness and brightness mismatches.
    the rest of the paramters get passed to ft.calc_color_matrix
    Note: the dominant z layer of beads is removed for robust multiorder fitting.
    """
    cs1 = np.array(cs1_)
    cs2 = np.array(cs2_)
    cs1[:,-1]=np.log(cs1[:,-1])
    cs2[:,-1]=np.log(cs2[:,-1])
    h1,h2 = cs1[:,-1],cs2[:,-1]
    keep = (h1>=np.percentile(h1,th))&(h2>=np.percentile(h2,th))

    cts_,lims_ = np.histogram(cs2[:,0],bins=np.arange(0,np.max(cs2[:,0]),3))
    bad = (cs2[:,0]>lims_[np.argmax(cts_)])&(cs2[:,0]<lims_[np.argmax(cts_)+1])
    keep = keep&(~bad)

    cs2t = cs2 - np.median(cs2[keep]-cs1[keep],axis=0)
    difs = cs2t[keep,:]-cs1[keep,:]
    keeph = difs[:,-1]<=np.percentile(difs[:,-1],100-th)

    zxy1,zxy2 =  cs1[keep,:3][keeph],cs2[keep,:3][keeph]
    m = ft.calc_color_matrix(zxy1,zxy2,order = order,no_z=no_z,ransac=ransac)
    return m


### Deal with repeats and select the best ones
def get_Rkeys_v2(dic,ncells=100,exclude_B=True):
    """
    Given a dictionary this goes through the first ncells and returns the keys.
    """
    cells = np.sort(list(dic.keys()))[:ncells]
    Rkeys = np.unique([list(dic[cell].keys()) for cell in cells])
    return [key for key in Rkeys if exclude_B and key[0]!='B']
def get_repeats_v2(dic):
    """Given a dictionary, this cleans them and identifies and groups repeats.
    It assumes repeats come in form R1, R1_rep1 ...
    """
    Rkeys = get_Rkeys_v2(dic)
    urepRs = np.unique([key.split('_rep')[0] for key in Rkeys if '_rep' in key])
    return [[r for r in Rkeys if r==uR or uR+'_rep' in r] for uR in urepRs]
def best2(dic,r1,r2,ncells = 1000,npts = 30):
    cells = np.sort(list(dic.keys()))[:ncells]
    Rhs = []
    for r in [r1,r2]:
        rhs = []
        for cell in cells:
            ps = dic[cell].get(r,[])
            if len(ps)>0:
                h = dic[cell][r][:npts,0]
                hsn = dic[cell][r][:npts,0]/dic[cell][r][:npts,4]
                rhs.extend(h)
        Rhs.append(rhs)
        
    lens = [len(Rhs[0]),len(Rhs[1])]
    
    if np.min(lens)/float(np.max(lens))<0.8:
        return [r1,r2][np.argmax(lens)]
    else:
        min_len = np.min(lens)
        Rhs = [np.sort(h)[-min_len:] for h in Rhs]
        fr = np.sum(Rhs[0]<Rhs[1])/float(len(Rhs[0]))
        if fr>0.5:
            return r2
        else:
            return r1
def best_from_set(dic,repeat_set,ncells = 500,npts = 30):
    rmax = repeat_set[0]
    for r in repeat_set:
        rmax = best2(dic,rmax,r,ncells = ncells,npts = npts)
    return rmax
from copy import deepcopy
def replace_repeats_v2(dic,forced=[]):
    repeat_sets = get_repeats_v2(dic)
    bestRs = [best_from_set(dic,set_) for set_ in repeat_sets]
    #deal with forced
    bestRs_ = []
    forced_ = [r.split('_rep')[0] for r in forced]
    for Rb in bestRs:
        Rb_ = Rb
        r = Rb.split('_rep')[0]
        if r in forced_: Rb_ = forced[forced_.index(r)]
        bestRs_.append(Rb_)
    bestRs = bestRs_
    print('Best picks:',bestRs)
    flip_pairs = []
    for r in bestRs:
        if '_rep' in r:
            R = r.split('_rep')[0]
            #print(r,R)
            flip_pairs.append((r,R))
            for cell in dic.keys():
                dic[cell][R],dic[cell][r] = dic[cell][r],dic[cell][R]

    return dic,flip_pairs,bestRs,repeat_sets

### Deal with repeats - Deprecated ######################################
def get_Rkeys(dic__):
    """Given a dictionary of the form of fitted dictionary, this returns the Rkeys."""
    Rkeys = []
    for cell in dic__:
        dic_cell = dic__[cell]
        Rkeys.extend(dic_cell.keys())
    Rkeys = np.unique(Rkeys)
    return Rkeys
def get_repeats(Rkeys):
    """Given a set of Rkeys, this cleans them and identifies and groups repeats."""
    rinds = [r[1:] for r in Rkeys if r[0] in ['R','r']]
    rinds_,cts_ = np.unique(rinds,return_counts=True)
    return [['R'+r,'r'+r]for r in rinds_[cts_>1]]

def quantify_repeats(dic__,href,pair_reps):
    """Given a fit-dictionary, a reference set of brightnesses 
    and a list of pairs of repeats this quantifies the repeats."""
    rep_stats = []
    rkeys = [r for pair in pair_reps for r in pair]
    #print rkeys
    for cell in dic__:
        if 'cell' in cell:
            dic_cell = dic__[cell]
            dic_mc = get_mc_h(dic_cell,href,rkeys=rkeys)
            rep_stats.append([[dic_mc[r][0] for r in pair] for pair in pair_reps])
    rep_stats = np.median(rep_stats,0)
    return rep_stats

def replace_repeats(dic__,referenceR='R3'):
    """Given a fit-dictionary this swaps the repeats to """
    dic___ = deepcopy(dic__)
    Rkeys = get_Rkeys(dic__)
    pair_reps = get_repeats(Rkeys)
    cell = [key for key in dic__.keys() if 'cell' in key][0]
    pair_quant = quantify_repeats(dic__,dic__[cell][referenceR][:,0],pair_reps)
    if pair_quant.size>0:        
        for ibest,pair in zip(np.argmin(pair_quant,-1),pair_reps):
            print("Picking "+pair[ibest]+" as the best.")
            for cell in dic__:
                if 'cell' in cell:
                    dic___[cell][pair[ibest]],dic___[cell][pair[1-ibest]] = dic__[cell][pair[0]],dic__[cell][pair[1]]
    return dic___
### End ---- Deal with repeats - Deprecated ######################################

### Flatten brightness of colors and readouts ########################################
def get_mc(x,y):
    x_,y_ = np.array(x),np.array(y)
    keep = (~np.isnan(x))&(~np.isnan(y))
    x_,y_ = x_[keep],y_[keep]
    A = np.vstack([x_, np.ones(len(x_))]).T
    m, c = np.linalg.lstsq(A, y_)[0]
    return (m,c)
def flatten_colors(dic,col_dic,coltag = 'Q',ncells=100,npts = 30,replace_dic = True,plt_val=False):
    """Itterates through cells and colors and flattens the field and equivalets the colors"""
    keep_perc = np.arange(0,1,0.05)

    
    dic_mc_colpercell={}

    cells = np.sort(list(dic.keys()))
    Rkeys = get_Rkeys_v2(dic)

    #construct reference
    Rcol = [R for R in Rkeys if coltag in R if col_dic[R]==0]
    ncols = np.max(col_dic.values())+1
    h0 = []
    for cell in cells[:ncells]:
        for R in Rcol:
            p_ = dic[cell][R]
            if len(p_)>0:
                h0.extend(p_[:npts,0])
    href = np.sort(h0)[(keep_perc*len(h0)).astype(int)]

    #run through cells

    for cell in tqdm(cells):
        hs = []
        for coli in range(ncols):
            Rcol = [R for R in Rkeys if coltag in R if col_dic[R]==coli]
            h0 = []
            for R in Rcol:
                p_ = dic[cell][R]
                if len(p_)>0:
                    h0.extend(p_[:npts,0])
            if len(h0)>0:
                hs.append(h0)
        hs = np.array([np.sort(hs_)[(keep_perc*len(hs_)).astype(int)] for hs_ in hs])

        dic_mc_colpercell[cell] = {}
        for i in range(len(hs)):
            m,c = get_mc(hs[i],href)
            dic_mc_colpercell[cell][i]=(m,c)
            if plt_val:
                plt.figure()
                plt.title(cell+' - Col:'+str([0,i])+' - '+str([m,c]))
                plt.plot(hs[i],href,'o')
                plt.plot(hs[i],hs[i]*m+c)
        if replace_dic:
            for r in dic[cell].keys():
                m,c = dic_mc_colpercell[cell][col_dic.get(r,0)]
                ps = dic[cell][r]
                if len(ps)>0:
                    dic[cell][r][:,[0,4]]=ps[:,[0,4]]*m+c
    return dic_mc_colpercell
def flatten_hybes(dic,ncells=1000,npts = 30,Rref = None,replace_dic = True,plt_val=False):
    """Itterates through cells and colors and flattens the field and equivalets the colors"""
    keep_perc = np.arange(0,1,0.05)

    dic_mc_Rs={}

    cells = np.sort(list(dic.keys()))
    Rkeys = get_Rkeys_v2(dic)

    #construct reference
    h0 = []
    if Rref is None: Rref = Rkeys[0]
    for cell in cells[:ncells]:
        p_ = dic[cell].get(Rref,[])
        if len(p_)>0: h0.extend(p_[:npts,0])
    href = np.sort(h0)[(keep_perc*len(h0)).astype(int)]

    #run through cells
    count_figures=0
    for R in Rkeys:
        h0 = []
        for cell in cells[:ncells]:
            p_ = dic[cell].get(R,[])
            if len(p_)>0: h0.extend(p_[:npts,0])
        htemp = np.sort(h0)[(keep_perc*len(h0)).astype(int)]
        m,c = get_mc(htemp,href)
        dic_mc_Rs[R] =(m,c)
        if plt_val>0:
            if count_figures<plt_val:
                if 'R' in R:
                    count_figures+=1
                    plt.figure()
                    plt.title(R+'='+Rref+' - '+str([m,c]))
                    plt.plot(htemp,href,'o')
                    plt.plot(htemp,htemp*m+c)
    if replace_dic:
        for cell in tqdm(cells):
            for r in dic[cell].keys():
                m,c = dic_mc_Rs.get(r,(1,0))
                ps = dic[cell][r]
                if len(ps)>0:
                    dic[cell][r][:,[0,4]]=ps[:,[0,4]]*m+c
    return dic_mc_Rs
def get_mc_h(dic_cell,href,rkeys=None,npts=50):####depricated
    """
    Given a dictionary with hybe keys and pfits values, this computes the m,c such that brightnes of reference ~ m* brightness + c 
    Returns a dic such that (m,c) is reported for each hybe key.
    """
    dic_cell_mc = {}
    if rkeys is None: rkeys = list(dic_cell.keys())
    y = np.sort(href)[-npts:]
    for rkey in rkeys:
        dic_cell_mc[rkey] = (np.nan,np.nan)
        if len(dic_cell[rkey])>0:
            x = np.sort(dic_cell[rkey][:,0])[-npts:]
            l = np.min([len(x),len(y)])
            x_,y_ = x[-l:],y[-l:]
            A = np.vstack([x_, np.ones(len(x_))]).T
            m, c = np.linalg.lstsq(A, y_)[0]
            dic_cell_mc[rkey] = (m,c)
        
        #print rkey
    return dic_cell_mc

def load_double_dic(fl):
    dic_arr,cells,Rs = np.load(fl)
    dic={}
    for icell,cell in enumerate(cells):
        dic[cell] = {}
        for iR,R in enumerate(Rs):
            dic[cell][R]=dic_arr[icell][iR]
    return dic
def save_double_dic(dic,fl):
    cells = np.sort(list(dic.keys()))
    Rs = np.unique([key for cell in cells for key in dic[cell]])
    dic_arr = np.array([[dic[cell].get(R,[]) for R in Rs]for cell in cells])
    np.save(fl,[dic_arr,cells,Rs])

    
def get_metrics_candidates(dic__,tad_to_PR=None,cutoffs = [2,1.5,1.5],show_progress=False,shape=False):
    """returns dic_comp,dic_metrics"""
    read_fl = None
    if type(dic__) is not dict:
        read_fl = dic__
        dic__ = load_double_dic(read_fl)
    
    dic_comp,dic_metrics = {},{}
    if tad_to_PR is None:
        tad_to_PR,PR_to_tad = get_code_v2()
        tad_to_PR,PR_to_tad = extend_tad_to_PR_v2(tad_to_PR,PR_to_tad,num_blank=np.inf)

    cells = [key for key in dic__.keys() if 'cell' in key]
    if show_progress: cells = tqdm(cells)
    for cell in cells:
        dic_cell = dic__[cell]
        #dic_mc = get_mc_h(dic_cell,href)  ### this is noew done before the dictionary is fed

        dic_comp[cell],dic_metrics[cell] = {},{}

        for chr_ in tad_to_PR.keys():
            R,P = tad_to_PR[chr_]
            pPs,pRs = dic_cell[P],dic_cell[R]
            pP,pR = [],[]
            if len(pPs)>0 and len(pRs)>0:
                M = cdist(pPs[:,1:4]/cutoffs,pRs[:,1:4]/cutoffs,metric='chebyshev')
                Pg,Rg = np.where(M<1)
                pP,pR = pPs[Pg],pRs[Rg]
            if len(pP)>0:
                zxy_delta = np.abs(pP[:,1:4]-pR[:,1:4])
                #(mR,cR),(mP,cP) = dic_mc[R],dic_mc[P]
                hP = np.log(pP[:,0])#hP = np.log(pP[:,0]*mP+cP)
                hR = np.log(pR[:,0])#hR = np.log(pR[:,0]*mR+cR)
                if len(pP[0])>11:
                    incell = (pP[:,-2]+pR[:,-2])/2.
                    othercell = (pP[:,-1]+pR[:,-1])/2.
                else: 
                    incell,othercell=np.ones(len(pP)),np.zeros(len(pP))
                def KL_dif_null(p1,p2): return 0 #ft.KL_dif
                KL_dif = ft.KL_dif if shape else KL_dif_null

                Ds = np.array([(hP+hR)/2.,zxy_delta[:,0],zxy_delta[:,1],zxy_delta[:,2],
                          np.abs(hP-hR),[KL_dif(p_,r_) for p_,r_ in zip(pP,pR)],incell,othercell]).T

                dic_comp[cell][chr_]=(pP,pR)
                dic_metrics[cell][chr_]=Ds
    if read_fl is not None:
        save_fl = read_fl.replace('.npy','_comp.npy')
        save_double_dic(dic_comp,save_fl)
        save_fl = read_fl.replace('.npy','_metrics.npy')
        save_double_dic(dic_metrics,save_fl)
    else:
        return dic_comp,dic_metrics
    
def flatten_filter_sort(scores):
    scores_ = np.array(flatten(scores))
    keep = (np.isnan(scores_)==False)&(np.isinf(scores_)==False)
    return np.sort(scores_[keep])
def return_best_th(scores_good_,scores_bad_,plt_val=True):
    M_,m_ = np.max(scores_good_),np.min(scores_bad_)
    th_prev=-np.inf
    while True:
        th_ = (M_+m_)/2.
        p_g,p_b=cum_val(scores_good_,th_,False),cum_val(scores_bad_,th_,True)
        if p_g>p_b: m_ = th_
        else: M_=th_
        if th_==th_prev: break
        else: th_prev=th_
    th_gb = th_
    if plt_val:
        plt.figure()
        plt.title('bleed:'+str(1-p_b))
        vls_,edgs_,_ =plt.hist(scores_good_,normed=True,alpha=0.5,bins=100)
        plt.hist(scores_bad_,normed=True,alpha=0.5,bins=100)
        plt.plot([th_gb,th_gb],[0,np.max(vls_)],'k')
        plt.show()
    return th_gb,1-p_b
def Ds_to_scores(Ds,Ds_keep,biggers = [True,False,False,False,False,False,True,False],weigths=[1.,1./5,1./5,1./5,1./5,1./5,1,1./2,1./2]):
    
    if len(Ds_keep)==0:
        return Ds[:,0]
    weigths = np.array(weigths)/np.sum(weigths)
    return np.array([np.sum([np.log(cum_val(Ds_kp,d_,bigger))*w_ for d_,Ds_kp,bigger,w_ in zip(Ds_,Ds_keep.T,biggers,weigths)]) 
                     for Ds_ in Ds])
def DsC_to_scores(Ds,Ds_keep,biggers = [True,False,False,False,False,False,False,True,False],
                  weigths=[1.,1./5,1./5,1./5,1./5,1./5,1,1./2,1./2]):
    if len(Ds_keep)==0:
        return Ds[:,0]
    weigths = np.array(weigths)/np.sum(weigths)
    return np.array([np.sum([np.log(cum_val(Ds_kp,d_,bigger))*w_ for d_,Ds_kp,bigger,w_ in zip(Ds_,Ds_keep.T,biggers,weigths)]) 
                     for Ds_ in Ds])
                     
def EMStep(dic_metrics,dic_comp,Ds_keep=[],ncells=1000,niter=5):
    cells = np.sort([key for key in dic_metrics.keys() if 'cell' in key])[:ncells]
    #np.unique(flatten([dic_metrics[cell].keys() for cell in cells]))
    selected_inds_s=[]
    for i in tqdm(range(niter)):
        Ds_keep_,scores_good,scores_all = [],[],[]
        selected_inds = []
        dic_comp_={}
        for cell in cells:
            dic_comp_[cell]={}
            for chr_ in dic_metrics[cell].keys():
                if 'chr' in chr_:
                    dic_comp_[cell][chr_]={}
                    Ds = dic_metrics[cell][chr_]
                    if len(Ds)>0:
                        scores = Ds_to_scores(Ds,Ds_keep)
                        keep_i = np.argsort(scores)[::-1][:2]
                        #if chr_=='chr1_1': print keep_i#Ds-Ds[keep_i]
                        selected_inds.extend(keep_i)
                        #print(len(dic_comp[cell][chr_][0]),len(dic_comp[cell][chr_][1]),scores)
                        dic_comp_[cell][chr_] = (dic_comp[cell][chr_][0][keep_i][:,1:4]+dic_comp[cell][chr_][1][keep_i][:,1:4])/2.
                        Ds_keep_.extend(Ds[keep_i])
                        scores_good.append(scores[keep_i])
                        scores_all.append(scores)
        Ds_keep_ = np.array(Ds_keep_)
        Ds_keep_ = Ds_keep_[np.any(np.isnan(Ds_keep_),axis=-1)==False]
        Ds_keep_ = np.sort(Ds_keep_,axis=0)
        perc_overlap = 0
        if len(selected_inds_s)==len(selected_inds):
            perc_overlap = np.sum(np.array(selected_inds)==selected_inds_s)/float(len(selected_inds))
        selected_inds_s = selected_inds
        print("Number interation, fraction overlap:",i,perc_overlap)
        Ds_keep=Ds_keep_
    
    #add the distribution for illegal pairs.
    scores_bad=[]
    for cell in cells:
        for chr_ in dic_metrics[cell].keys():
            if 'chr' not in chr_:
                Ds = dic_metrics[cell][chr_]
                scores = Ds_to_scores(Ds,Ds_keep)
                scores_bad.append(scores)
                
    scores_good_ = flatten_filter_sort(scores_good)
    scores_bad_ = flatten_filter_sort(scores_bad)
    return dic_comp_,scores_good_,scores_bad_,Ds_keep

def filter_th(dic_metrics,dic_comp,Ds_keep,th_gb,keep_pts=2):
    cells = np.sort([key for key in dic_metrics.keys() if 'cell' in key])
    dic_comp_={}
    for cell in tqdm(cells):
        dic_comp_[cell]={}
        for chr_ in dic_metrics[cell].keys():
            if 'chr' in chr_:
                dic_comp_[cell][chr_]={}
                Ds = dic_metrics[cell][chr_]
                if len(Ds):
                    scores = Ds_to_scores(Ds,Ds_keep)
                    keep_i = np.argsort(scores)[::-1][:keep_pts]
                    keep_i = keep_i[scores[keep_i]>th_gb]
                    dic_comp_[cell][chr_] = (dic_comp[cell][chr_][0][keep_i][:,1:4]+dic_comp[cell][chr_][1][keep_i][:,1:4])/2.
    return dic_comp_
def key_to_chreregi(key):
    chrs = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 
            'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 
            'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']
    chr_nm,reg_nm = key.split('_')
    chr_i = chrs.index(chr_nm)
    reg_i = int(reg_nm)
    return (chr_i,reg_i)
def get_ordered_keys(dic_comp_f):
    cell_ = list(dic_comp_f.keys())[0]
    keys = list(dic_comp_f[cell_].keys())
    chr_reg_i = np.array([key_to_chreregi(key) for key in keys], dtype=[('chr_i', '<i4'), ('reg_i', '<i4')])
    return np.array(keys)[np.argsort(chr_reg_i, order=('chr_i','reg_i'))]
def get_dic_centers(dic_comp__,show_progress=True):
    chrs = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 
            'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 
            'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']
    dic_centers={}
    cells = [key for key in dic_comp__.keys() if 'cell' in key]
    if show_progress: cells = tqdm(cells)
    for cell in cells:
        dic_cell = dic_comp__[cell]
        dic_centers[cell] ={}
        for chr_ in chrs:
            tads = [tad for tad in dic_cell.keys() if chr_+'_' in tad]
            c1,c2 = homolog_separator_centers(dic_cell,tads,plt_val=False,save_fl=None)
            dic_centers[cell][chr_]=(c1,c2)
    return dic_centers


def partition_map(list_,map_,umap=None):
    """
    Inputs
    takes a list [e1,e2,e3,e4,e5,e6] and a map (a list of indices [0,0,1,0,1,2]).  map can be a list of symbols too. ['aa','aa','bb','aa','bb','cc']
    Output
    returns a sorted list of lists, e.g. [[e1, e2,e4],[e3,e5],[e6]]
    """
    list__=np.array(list_)
    map__=np.array(map_)
    if umap is None:
        umap = np.unique(map__)
    return [list(list__[map__==element]) for element in umap]
    
def update_dic_centers(dic_centers_,dic_comp_):
    cells = np.sort([key for key in dic_comp_.keys() if 'cell' in key])
    chrs = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']
    tads = np.unique([list(dic_comp_[cell].keys()) for cell in cells])
    tads_groups = partition_map(tads,[tad.split('_')[0] for tad in tads],chrs)
    for cell in cells:
        for tg,chr_ in zip(tads_groups,chrs):
            c1 = np.nanmedian([dic_comp_[cell][tad][0] for tad in tg if len(dic_comp_[cell][tad])>0],0)
            c2 = np.nanmedian([dic_comp_[cell][tad][1] for tad in tg if len(dic_comp_[cell][tad])>0],0)
            dic_centers_[cell][chr_]=(c1,c2)
    return dic_centers_
def extend_Ds_keep(Ds_keep,dic_centers_,dic_comp_):
    cells = np.sort([key for key in dic_comp_.keys() if 'cell' in key])
    dists = []
    for cell in cells:
        for chr_ in dic_comp_[cell]:
            if 'chr' in chr_:
                pts = dic_comp_[cell][chr_]
                if len(pts)>0:
                    chr_c = chr_.split('_')[0]
                    c1,c2 =  dic_centers_[cell][chr_c]
                    d1s = np.linalg.norm(pts-c1,axis=-1)
                    d2s = np.linalg.norm(pts-c2,axis=-1)
                    dists.extend(np.min([d1s,d2s],axis=0))
    dists = np.sort(dists)
    dists = dists[np.isnan(dists)==False]
    dmin = min(Ds_keep.shape[0],len(dists))
    return np.concatenate([Ds_keep[:dmin],np.array([dists[:dmin]]).T],axis=-1)
    
def EMStepCenter(dic_metrics,dic_comp,dic_centers,Ds_keep=[],niter=5):
    cells = np.sort([key for key in dic_metrics.keys() if 'cell' in key])
    #np.unique(flatten([dic_metrics[cell].keys() for cell in cells]))
    selected_inds_s=[]
    
    dic_centers_ = deepcopy(dic_centers)
    for i in range(niter):
        Ds_keep_,scores_good = [],[]
        selected_inds = []
        dic_comp_={}
        scores_good_raw={}
        for cell in cells:
            dic_comp_[cell]={}
            scores_good_raw[cell]={}
            for chr_ in dic_metrics[cell].keys():
                if 'chr' in chr_:
                    dic_comp_[cell][chr_]={}
                    Ds = dic_metrics[cell][chr_]
                    if len(Ds):
                        chr_c = chr_.split('_')[0]
                        c1,c2 = dic_centers_[cell][chr_c]
                        pts = (dic_comp[cell][chr_][0][:,1:4]+dic_comp[cell][chr_][1][:,1:4])/2.
                        dists1 = np.linalg.norm(pts-c1,axis=-1)
                        dists2 = np.linalg.norm(pts-c2,axis=-1)
                        Ds1 = np.concatenate([Ds,np.array([dists1]).T],axis=-1)
                        Ds2 = np.concatenate([Ds,np.array([dists2]).T],axis=-1)
                        scores_c1 = DsC_to_scores(Ds1,Ds_keep)
                        scores_c2 = DsC_to_scores(Ds2,Ds_keep)
                        
                        icopy = np.argmin([dists1,dists2],axis=0)
                        scores_c1[icopy==1]=-np.inf
                        scores_c2[icopy==0]=-np.inf
                        
                        
                        keep_i = [np.argmax(scores_c1),np.argmax(scores_c2)]
                        
                        scores_keep = [scores_c1[keep_i[0]],scores_c2[keep_i[1]]]
                        Dss = [Ds1[keep_i[0]],Ds2[keep_i[1]]]
                        pts = np.concatenate([pts,[[np.nan]*3]])
                        if np.isinf(scores_keep[0]): 
                            keep_i[0]=-1
                            Dss[0]+=np.nan
                        if np.isinf(scores_keep[1]): 
                            keep_i[1]=-1
                            Dss[1]+=np.nan
                        selected_inds.extend(keep_i)
                        dic_comp_[cell][chr_] = pts[keep_i]
                        Ds_keep_.extend(Dss)
                        scores_good.append(scores_keep)
                    else:
                        dic_comp_[cell][chr_] = [[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan]]
                        scores_good.append([-np.inf,-np.inf])
                        #scores_all.append(scores)
                    scores_good_raw[cell][chr_]=scores_good[-1]
        #update centers
        dic_centers_ = update_dic_centers(dic_centers_,dic_comp_)
        #update distributions
        Ds_keep_ = np.array(Ds_keep_)
        Ds_keep_ = Ds_keep_[np.any(np.isnan(Ds_keep_),axis=-1)==False]
        Ds_keep_ = np.sort(Ds_keep_,axis=0)
        Ds_keep=Ds_keep_
        #check convergence
        perc_overlap = 0
        if len(selected_inds_s)==len(selected_inds):
            perc_overlap = np.sum(np.array(selected_inds)==selected_inds_s)/float(len(selected_inds))
        selected_inds_s = selected_inds
        print("Number interation, fraction overlap:",i,perc_overlap)
        
    chrs = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 
            'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']
    #add the distribution for illegal pairs.
    scores_bad=[]
    for cell in cells:
        for chr_ in dic_metrics[cell].keys():
            if 'chr' not in chr_:
                Ds = dic_metrics[cell][chr_]
                if len(Ds):
                    chr__,ichr = np.random.choice(chrs),np.random.choice([0,1]) #pick randomly a chromosome
                    c= dic_centers_[cell][chr__][ichr]
                    pts = (dic_comp[cell][chr_][0][:,1:4]+dic_comp[cell][chr_][1][:,1:4])/2.
                    dists = np.linalg.norm(pts-c,axis=-1)
                    Ds = np.concatenate([Ds,np.array([dists]).T],axis=-1)
                    scores = DsC_to_scores(Ds,Ds_keep)
                    scores_bad.append(scores)
                
    scores_good_ = flatten_filter_sort(scores_good)
    scores_bad_ = flatten_filter_sort(scores_bad)
    return dic_comp_,scores_good_,scores_bad_,Ds_keep,dic_centers_,scores_good_raw
    
    
def plot_bk(dic_cell,**kargs):
    zxys_all = []
    for chr_ in dic_cell:
        zxys_all.extend(dic_cell.get(chr_,[[np.nan,np.nan,np.nan]]))
    z,x,y = np.array(zxys_all).T
    plt.plot(x,y,'.',**kargs)
def plot_chr(dic_cell,chr_tag='chr1_',**kargs):
    zxys_chr = []
    for chr_ in dic_cell:
        if chr_tag in chr_:
            zxys_chr.extend(dic_cell.get(chr_,[[np.nan,np.nan,np.nan]]))
    z,x,y = np.array(zxys_chr).T
    plt.plot(x,y,'.',**kargs)
def plot_chr_split(dic_cell,chr_tag='chr1_',**kargs):
    zxys_chr1,zxys_chr2 = [],[]
    tads = np.array([tad for tad in dic_cell.keys() if chr_tag in tad])
    inds = np.argsort([int(tad.split('_')[-1]) for tad in tads])
    tads =  tads[inds]
    for tad in tads:
        zxyspr = dic_cell.get(tad)
        if len(zxyspr):
            zxys_chr1.append(zxyspr[0])
            zxys_chr2.append(zxyspr[1])
    z,x,y = np.array(zxys_chr1).T
    plt.plot(x,y,'r-.',**kargs)
    z,x,y = np.array(zxys_chr2).T
    plt.plot(x,y,'b-.',**kargs)
### homolog separator centers

def compute_inds(Xs,c1,c2):
    Xp=[[],[]]
    inds=[]
    for X in Xs:
        if len(X)>0:
            X1d = X-[c1]
            d1 = np.sum(X1d*X1d,1)
            X2d = X-[c2]
            d2=np.sum(X2d*X2d,1)
            ind=np.argmin([d1,d2],0)
            Xp[0].extend(X[ind==0])
            Xp[1].extend(X[ind==1])
            inds.append(ind)
        else:
            inds.append([])
    return inds,list(map(np.array,Xp))
from scipy.spatial.distance import pdist,squareform
def get_homolog_centers(xyz_all,chr_all):
    ### usefull inner functions
    def coverage(inds):
        return np.sum([(0 in ind) and (1 in ind) for ind in inds])
    def get_rgsum(XpT,c1,c2):
        if (len(XpT[0])>0) and (len(XpT[1])>0):
            X1d = XpT[0]-[c1]
            X2d = XpT[1]-[c2]
            rg1 = np.mean(np.sum(X1d*X1d,-1))
            rg2 = np.mean(np.sum(X2d*X2d,-1))
            rgsumT = rg1+rg2
            return rgsumT
        return np.inf
    def get_cap(dists,cap):
        if len(dists)>cap:
            return dists
        else:
            return np.sort(dists)[:cap]
    def compute_cov_rg_fast(xyz_all,chr_all,i1,i2,dist_mat,chr_inds,cap=None):
        keep = dist_mat[i1]>dist_mat[i2]
        nonkeep = ~keep
        #inds = np.array(keep,dtype=int)
        #Xp = [xyz_all[nonkeep],xyz_all[keep]]
        dist1 = dist_mat[i1]
        dist2 = dist_mat[i2]
        if cap is not None:
            inds2 = chr_inds&keep
            inds1 = chr_inds&nonkeep
            rg_mean2 = np.mean([item for ind_ in inds2 for item in get_cap(dist2[ind_],cap)])
            rg_mean1 = np.mean([item for ind_ in inds1 for item in get_cap(dist1[ind_],cap)])
            rg_mean = (rg_mean1+rg_mean2)/2
            cov = np.sum(np.sum(inds1,-1)&np.sum(inds2,-1))
        else:
            rg_mean = np.mean(dist1[nonkeep])+np.mean(dist2[keep])
            cov = np.sum(np.dot(chr_inds,nonkeep)&np.dot(chr_inds,keep))
        return cov,rg_mean
    ### Pool all candidates together

    chr_all = np.array(chr_all)
    chr_all_u = np.unique(chr_all)
    chr_inds = np.array([chr_all==chr_u for chr_u in chr_all_u])
    
    dist_mat = squareform(pdist(xyz_all))
    ### If not enough data return empty
    if len(xyz_all)<2:
        return [np.nan]*3, [np.nan]*3
        #return c1,c2
        
    ### Restructure some information
    Xs=[]
    for tad in tads:
        xg,yg,zg=[],[],[]
        if len(dic_cand.get(tad,[])):
            (xg,yg,zg)=dic_cand[tad].T
        X=np.array([xg,yg,zg]).T
        Xs.append(X)

    ### Compute best centers that first maximize coverage and then minimize radius of gyration
    rg_mean = np.inf
    cov=0
    for i1 in range(len(xyz_all)):
        for i2 in range(i1):
            covT,rg_meanT = compute_cov_rg_fast(xyz_all,chr_all,i1,i2,dist_mat,chr_inds,cap=None)
            if (covT==cov and rg_meanT<rg_mean) or covT>cov:
                c1=xyz_all[i1]
                c2=xyz_all[i2]
                cov=covT
                rg_mean = rg_meanT
    
    
    inds,Xp = compute_inds(Xs,c1,c2)
    #minc = coverage(inds)
    c1=np.median(Xp[0],0)
    c2=np.median(Xp[1],0)
    

        
    return c1,c2
def homolog_separator_centers(dic_cand,tads,plt_val=False,save_fl=None):
    """This perfors the separation between 2 homologs and returns the 2 best centers. Could easily be extended for more but not implemented yet.
    This aims to find two centers c1 and c2 such that the points closest to each maximise the converage of each homolog.
    For the same values of coverage, we minimized for the dispersion (similar to Kmeans clustering).
    After the separation a refinement is performed to keep only the brightest spot for each TAD within the chrT size.
    If that does not exist, just grab the closest point to the center.
    
    dic_cand is a dictionary with keys of the form: chr1_1,chr3_5... and values (xg,yg,zg).T for the intersections
    tads is a list of subkeys from the dic_cand corresponding to the desired chromosome
    """
    ### usefull inner functions
    def coverage(inds):
        return np.sum([(0 in ind) and (1 in ind) for ind in inds])
    def get_rgsum(XpT,c1,c2):
        if (len(XpT[0])>0) and (len(XpT[1])>0):
            X1d = XpT[0]-[c1]
            X2d = XpT[1]-[c2]
            rg1 = np.mean(np.sum(X1d*X1d,-1))
            rg2 = np.mean(np.sum(X2d*X2d,-1))
            rgsumT = rg1+rg2
            return rgsumT
        return np.inf
    def get_cap(dists,cap):
        if len(dists)>cap:
            return dists
        else:
            return np.sort(dists)[:cap]
    def compute_cov_rg_fast(xyz_all,chr_all,i1,i2,dist_mat,chr_inds,cap=None):
        keep = dist_mat[i1]>dist_mat[i2]
        nonkeep = ~keep
        #inds = np.array(keep,dtype=int)
        #Xp = [xyz_all[nonkeep],xyz_all[keep]]
        dist1 = dist_mat[i1]
        dist2 = dist_mat[i2]
        if cap is not None:
            inds2 = chr_inds&keep
            inds1 = chr_inds&nonkeep
            rg_mean2 = np.mean([item for ind_ in inds2 for item in get_cap(dist2[ind_],cap)])
            rg_mean1 = np.mean([item for ind_ in inds1 for item in get_cap(dist1[ind_],cap)])
            rg_mean = (rg_mean1+rg_mean2)/2
            cov = np.sum(np.sum(inds1,-1)&np.sum(inds2,-1))
        else:
            rg_mean = np.mean(dist1[nonkeep])+np.mean(dist2[keep])
            cov = np.sum(np.dot(chr_inds,nonkeep)&np.dot(chr_inds,keep))
        return cov,rg_mean
    ### Pool all candidates together
    xyz_all,chr_all=[[],[],[]],[]
    for itad,tad in enumerate(tads):
        if len(dic_cand.get(tad,[])):
            xg,yg,zg =dic_cand[tad].T
            xyz_all[0].extend(xg)
            xyz_all[1].extend(yg)
            xyz_all[2].extend(zg)
            chr_all.extend([itad]*len(xg))
    xyz_all = np.array(xyz_all).T
    chr_all = np.array(chr_all)
    chr_all_u = np.unique(chr_all)
    chr_inds = np.array([chr_all==chr_u for chr_u in chr_all_u])
    from scipy.spatial.distance import pdist,squareform
    dist_mat = squareform(pdist(xyz_all))
    ### If not enough data return empty
    if len(xyz_all)<2:
        return [np.nan]*3, [np.nan]*3
        #return c1,c2
        
    ### Restructure some information
    Xs=[]
    for tad in tads:
        xg,yg,zg=[],[],[]
        if len(dic_cand.get(tad,[])):
            (xg,yg,zg)=dic_cand[tad].T
        X=np.array([xg,yg,zg]).T
        Xs.append(X)

    ### Compute best centers that first maximize coverage and then minimize radius of gyration
    rg_mean = np.inf
    cov=0
    for i1 in range(len(xyz_all)):
        for i2 in range(i1):
            covT,rg_meanT = compute_cov_rg_fast(xyz_all,chr_all,i1,i2,dist_mat,chr_inds,cap=None)
            if (covT==cov and rg_meanT<rg_mean) or covT>cov:
                c1=xyz_all[i1]
                c2=xyz_all[i2]
                cov=covT
                rg_mean = rg_meanT
    
    
    inds,Xp = compute_inds(Xs,c1,c2)
    #minc = coverage(inds)
    c1=np.median(Xp[0],0)
    c2=np.median(Xp[1],0)
    

    ### Plotting
    if plt_val:
        f=plt.figure()
        ix,iy=1,2
        cols = ['red','blue']
        xyz_all=[[],[],[]]
        for tad in dic_cand.keys():
            if 'chr' in tad:
                if len(dic_cand[tad]):
                    (xg,yg,zg)=dic_cand[tad].T
                    xyz_all[0].extend(xg)
                    xyz_all[1].extend(yg)
                    xyz_all[2].extend(zg)
        plt.plot(xyz_all[ix],xyz_all[iy],'o',color='gray',alpha=1)
        plt.axis('equal')
        for X,ind in zip(Xs,inds):#iterate through candidate positions
            for ind_ in [0,1]:#iterate through homologs
                if len(ind)>0:
                    Xt = X[ind==ind_]#candidate positions for current homolog varibalex3
                    if len(Xt)>0:
                        xt,yt = Xt.T[[ix,iy]]
                        for x_,y_ in zip(xt,yt):
                            plt.plot(x_,y_,'o',color=cols[ind_]) #all the candidates
        plt.plot(c1[ix],c1[iy],'ko')
        plt.plot(c2[ix],c2[iy],'ko')
        #plot all the other points in black with alpha set.
        
        if save_fl is not None:
            f.savefig(save_fl)
        
    return c1,c2

################


def flatten(l):
    return [item for sublist in l for item in sublist]
def cum_val(vals,target,bigger=True):
    """returns the fraction of elements with value < taget. assumes vals is sorted"""
    if vals[0]==vals[-1]:
        return 1
    niter = 0
    m,M = 0,len(vals)-1
    while True:
        mid = int((m+M)/2)
        if vals[mid]<target:
            m = mid
        else:
            M = mid
        niter+=1
        if M-m<2:
            break
    if vals[mid]<target:
        while mid<len(vals)-1 and vals[mid+1]<target:
            mid+=1
    else:
        while mid>0 and vals[mid]>target:
            mid-=1
    mid+=1
    ratio = mid/float(len(vals))
    if bigger:
        return ratio
    return 1-ratio



def folder_to_readout(folder):
    tag = os.path.basename(folder)
    for ilet,let in enumerate(tag[1:]):
        if not let.isdigit():
            break
    return tag[ilet+1:].split(',')
def getH(folder):
    tag = os.path.basename(folder)
    for ilet,let in enumerate(tag[1:]):
        if not let.isdigit():
            break
    try: return int(tag[1:ilet+1])
    except: return np.inf
def nm_to_nms(folder):
    """Given a folder this parse the readouts name and splits the colors. """
    numbers,tags = [],[]
    nm_process = os.path.basename(folder).replace('B','B0').replace(',','')+'_'
    prev_num=''
    prev_tag=''
    for ichar,char in enumerate(nm_process):
        if not char.isdigit():
            prev_tag+=char
            if len(prev_num):
                numbers.append(prev_num)
            prev_num=''
        if char.isdigit():
            prev_num+=char
            if len(prev_tag):
                tags.append(prev_tag)
            prev_tag=''
    base = tags[0]+numbers[0]
    tags_ = [base+tag+num for num,tag in zip(numbers,tags)[1:]][::-1]
    return tags_



from scipy.spatial.distance import cdist
#from sklearn.cluster import KMeans
def compute_inds(Xs,c1,c2):
    Xp=[[],[]]
    inds=[]
    for X in Xs:
        if len(X)>0:
            X1d = X-[c1]
            d1 = np.sum(X1d*X1d,1)
            X2d = X-[c2]
            d2=np.sum(X2d*X2d,1)
            ind=np.argmin([d1,d2],0)
            Xp[0].extend(X[ind==0])
            Xp[1].extend(X[ind==1])
            inds.append(ind)
        else:
            inds.append([])
    return inds,list(map(np.array,Xp))
def homolog_separator(dic_cand,dic_cand_err,tads,plt_val=False,save_fl=None,bkpoints_dic={}):
    """This perfors the separation between 2 homologs. COuld easily be extended for more but not implemented yet.
    This aims to find two centers c1 and c2 such that the points closest to each maximise the converage of each homolog.
    For the same values of coverage, we minimized for the dispersion (similar to Kmeans clustering).
    After the separation a refinement is performed to keep only the brightest spot for each TAD within the chrT size.
    If that does not exist, just grab the closest point to the center.
    
    dic_cand is a dictionary with keys of the form: chr1_1,chr3_5... and values (xg,yg,zg,hg) for the intersections
    tads is a list of subkeys from the dic_cand corresponding to the desired chromosome
    """
    ### usefull inner functions
    def coverage(inds):
        return np.sum([(0 in ind) and (1 in ind) for ind in inds])
    def get_rgsum(XpT,c1,c2):
        if (len(XpT[0])>0) and (len(XpT[1])>0):
            X1d = XpT[0]-[c1]
            X2d = XpT[1]-[c2]
            rg1 = np.mean(np.sum(X1d*X1d,-1))
            rg2 = np.mean(np.sum(X2d*X2d,-1))
            rgsumT = rg1+rg2
            return rgsumT
        return np.inf
    def get_cap(dists,cap):
        if len(dists)>cap:
            return dists
        else:
            return np.sort(dists)[:cap]
    def compute_cov_rg_fast(xyz_all,chr_all,i1,i2,dist_mat,chr_inds,cap=None):
        keep = dist_mat[i1]>dist_mat[i2]
        nonkeep = ~keep
        #inds = np.array(keep,dtype=int)
        #Xp = [xyz_all[nonkeep],xyz_all[keep]]
        dist1 = dist_mat[i1]
        dist2 = dist_mat[i2]
        if cap is not None:
            inds2 = chr_inds&keep
            inds1 = chr_inds&nonkeep
            rg_mean2 = np.mean([item for ind_ in inds2 for item in get_cap(dist2[ind_],cap)])
            rg_mean1 = np.mean([item for ind_ in inds1 for item in get_cap(dist1[ind_],cap)])
            rg_mean = (rg_mean1+rg_mean2)/2
            cov = np.sum(np.sum(inds1,-1)&np.sum(inds2,-1))
        else:
            rg_mean = np.mean(dist1[nonkeep])+np.mean(dist2[keep])
            cov = np.sum(np.dot(chr_inds,nonkeep)&np.dot(chr_inds,keep))
        return cov,rg_mean
    ### Pool all candidates together
    xyz_all,chr_all=[[],[],[]],[]
    for itad,tad in enumerate(tads):
        (xg,yg,zg,hg)=dic_cand[tad]
        xyz_all[0].extend(xg)
        xyz_all[1].extend(yg)
        xyz_all[2].extend(zg)
        chr_all.extend([itad]*len(xg))
    xyz_all = np.array(xyz_all).T
    chr_all = np.array(chr_all)
    chr_all_u = np.unique(chr_all)
    chr_inds = np.array([chr_all==chr_u for chr_u in chr_all_u])
    from scipy.spatial.distance import pdist,squareform
    dist_mat = squareform(pdist(xyz_all))
    ### If not enough data return empty
    if len(xyz_all)<2:
        return [], [], [], [], [np.nan]*3, [np.nan]*3
        #return Xs,Xerrs,Hs,inds,c1,c2
        
    ### Restructure some information
    Xs=[]
    Xerrs=[]
    Hs=[]
    for tad in tads:
        (xg,yg,zg,hg)=dic_cand.get(tad,([],[],[],[]))
        X=np.array([xg,yg,zg]).T
        Xerrs.append(np.array(dic_cand_err.get(tad,[[]]*4)).T)
        Xs.append(X)
        Hs.append(hg)

    ### Compute best centers that first maximize coverage and then minimize radius of gyration
    rg_mean = np.inf
    cov=0
    for i1 in range(len(xyz_all)):
        for i2 in range(i1):
            covT,rg_meanT = compute_cov_rg_fast(xyz_all,chr_all,i1,i2,dist_mat,chr_inds,cap=None)
            if (covT==cov and rg_meanT<rg_mean) or covT>cov:
                c1=xyz_all[i1]
                c2=xyz_all[i2]
                cov=covT
                rg_mean = rg_meanT
    
    
    inds,Xp = compute_inds(Xs,c1,c2)
    #minc = coverage(inds)
    #c1=np.median(Xp[0],0)
    #c2=np.median(Xp[1],0)
    

    ### Plotting
    if plt_val:
        f=plt.figure()
        ix,iy=0,1
        cols = ['red','blue']
        for X,ind,h in zip(Xs,inds,Hs):#iterate through candidate positions
            for ind_ in [0,1]:#iterate through homologs
                if len(ind)>0:
                    Xt = X[ind==ind_]#candidate positions for current homolog varibalex3
                    ht = h[ind==ind_]
                    if len(ht)>0:
                        xt,yt,zt = Xt.T
                        maxh=np.max(ht)
                        for x_,y_,h_ in zip(xt,yt,ht):
                            plt.plot(x_,y_,'o',color=cols[ind_],alpha=h_/maxh) #all the candidates    
        #plot all the other points in black with alpha set.
        xyz_all=[[],[],[]]
        for tad in bkpoints_dic.keys():
            (xg,yg,zg,hg)=bkpoints_dic[tad]
            xyz_all[0].extend(xg)
            xyz_all[1].extend(yg)
            xyz_all[2].extend(zg)
        plt.plot(xyz_all[ix],xyz_all[iy],'o',color='black',alpha=0.05)
        plt.axis('equal')
        if save_fl is not None:
            f.savefig(save_fl)
        
    return Xs,Xerrs,Hs,inds,c1,c2
def check_color(dic,key,colors=['647', '647']):
    return [dic['Channels'][rd__] for rd__ in tad_to_PR[key]]==colors
def std_plots(base_folder,dic_cand,dic_cand_err,tad_to_PR,title_plots):
    labels = ['Interdistance - x axis (pixel)','Interdistance - y axis (pixel)','Interdistance - z axis (pixel)','Interdistance - brightness axis (median units)']
    label_names = ['x_axis','y_axis','z_axis','b_axis']
    for i in range(4):
        fig = plt.figure(figsize=(10,10))
        plt.hist(flatten([dic_cand_err[key][i]for key in dic_cand_err.keys() 
                              if ('chr' in key)]),bins=20)
        plt.hist(flatten([dic_cand_err[key][i]for key in dic_cand_err.keys() 
                              if ('chr' not in key)]),bins=20)
        plt.xlabel(labels[i])
        plt.ylabel('Number of decoded regions/cell')
        plt.title(title_plots)
        plt.show()
        fig.savefig(base_folder+os.sep+'Precission__'+label_names[i]+'.png')
    # Number of spots/region    
    fig = plt.figure(figsize=(10,10))
    val_cts,cts=np.unique([len(dic_cand[key].T) for key in dic_cand.keys() if 'chr' in key],return_counts=True)
    plt.plot(val_cts,cts,'o-')
    val_cts,cts=np.unique([len(dic_cand[key].T) for key in dic_cand.keys() if 'chr' not in key],return_counts=True)
    plt.plot(val_cts,cts,'o-')
    plt.xlabel('Number of spots/cell')
    plt.ylabel('Probability')
    plt.title(title_plots)
    plt.show()
    fig.savefig(base_folder+os.sep+'No_spots_per_cell.png')

    # Number of spots/region
    l=[[int(il[1:]) for il in tad_to_PR.get(key,[])] 
       for key in [key for key in dic_cand.keys() if (len(dic_cand[key][0])>1 and 'chr' in key)]]
    lR=[[int(il[1:]) for il in tad_to_PR.get(key,[])] 
        for key in [key for key in dic_cand.keys() if ('chr' in key)]]
    flat_list = flatten(l)
    flat_listR = flatten(lR)
    nmsR,ctsR = np.unique(flat_listR,return_counts=True)
    num_R_dic = dict(zip(nmsR,ctsR))
    nms,cts = np.unique(flat_list,return_counts=True)
    num_dic = dict(zip(nms,cts))
    probs = [num_dic.get(nm,0)/float(num_R_dic[nm])for nm in nmsR]

    fig = plt.figure(figsize=(10,10))
    plt.plot(nmsR,probs,'-o')
    plt.xlabel('Readout index')
    plt.ylabel('Probability of decoding >2 spots/cell')
    plt.ylim([0,1])
    plt.title(title_plots)
    plt.show()
    fig.savefig(base_folder+os.sep+'Readout_coverage.png')
def refinement(Xs,Xerrs,Hs,inds,c1,c2,bkpoints_dic={},chrT=20,plt_val=True,save_fl=None):
    """
    Inputs: Xs is all the candidates unsplited of size 10xvariablex3
    Hs is the brightnesses associated with the candidates Xs , size 10xvarible
    inds gives the homolog index for the Xs with values 0 or 1, size 10xvarible
    c1 and c2 are the two homolog centers
    
    Output: an aray xys_selected of size 2x10x3 and hs 2x10 with brightnesses
    """
    xys_selected=[[],[]]
    xys_errors_selected=[[],[]]
    hs_selected = [[],[]]
    
    inds,Xp = compute_inds(Xs,c1,c2)
    if not (len(Xp[0]) and len(Xp[1])):
        return np.array(xys_selected), np.array(xys_errors_selected), np.array(hs_selected)
    c1=np.median(Xp[0],0)
    c2=np.median(Xp[1],0)
    hom1_dists = cdist([c1],Xp[0])
    hom2_dists = cdist([c2],Xp[1])
    medians = [np.median(hom1_dists), np.median(hom2_dists)]
    ctr_dist = np.sqrt(np.sum((c1-c2)*(c1-c2)))
    if ctr_dist>np.sum(medians):
        inds,Xp = compute_inds(Xs,c1,c2)
    
    for X,ind,h,Xe in zip(Xs,inds,Hs,Xerrs):#iterate through candidate positions
        for ind_ in [0,1]:#iterate through homologs
            if len(ind)==0:
                xys_selected[ind_].append([np.nan]*3)
                xys_errors_selected[ind_].append([np.nan]*4)
                hs_selected[ind_].append(np.nan)
            else:
                Xt = X[ind==ind_]#candidate positions for current homolog varibalex3
                ht = h[ind==ind_]
                Xet = Xe[ind==ind_]
                if len(Xt)==0:
                    xys_selected[ind_].append([np.nan]*3)
                    xys_errors_selected[ind_].append([np.nan]*4)
                    hs_selected[ind_].append(np.nan)
                else:
                    ct = [c1,c2][ind_]
                    Xtd = Xt-[ct]
                    dt = np.sqrt(np.sum(Xtd*Xtd,-1))#distances from the center
                    keep = dt<chrT #consider first points within radius chrT from center
                    if np.sum(dt<chrT)>0: # if have any
                        ind_keep = np.arange(len(keep))[keep][np.argmax(ht[keep])] #index of brightest
                    else:# if do not
                        ind_keep=np.argmin(dt)#get the closest to the center
                    xys_selected[ind_].append(Xt[ind_keep])
                    xys_errors_selected[ind_].append(Xet[ind_keep])
                    hs_selected[ind_].append(ht[ind_keep])
    xys_selected = np.array(xys_selected)
    xys_errors_selected = np.array(xys_errors_selected)
    hs_selected = np.array(hs_selected)
    
    #Plotting
    if plt_val>0:
        f=plt.figure(figsize=(10,10))
        cols = ['orange','cyan']
        for i_,(X,ind,h) in enumerate(zip(Xs,inds,Hs)):#iterate through candidate positions
            for ind_ in range(len(xys_selected)):#iterate through homologs
                if len(ind)>0:
                    Xt = X[ind==ind_]#candidate positions for current homolog varibalex3
                    ht = h[ind==ind_]
                    if len(Xt)>0:
                        xt,yt,zt = Xt.T
                        maxh=np.max(ht)
                        for x_,y_,h_ in zip(xt,yt,ht):
                            #plt.plot(x_,y_,'.',color=cols[ind_],alpha=h_/maxh) #all the candidates
                            #plt.scatter(x_,y_,facecolor=cols[ind_],edgecolor='none',alpha=h_/maxh) #all the candidates
                            plt.scatter(x_,y_,facecolor=cols[ind_],alpha=h_/maxh) #all the candidates
                            if plt_val>2:
                                plt.text(x_,y_,str(i_+1),color=cols[ind_],alpha=h_/maxh)
        #plot the selected spots per homolog
        cols = ['red','blue']
        for ind_hom in range(len(xys_selected)):
            if len(xys_selected[ind_hom])==0:
                continue
            xt,yt,zt =xys_selected[ind_hom].T
            ht = hs_selected[ind_hom]
            if len(ht):
                for i_,(x_,y_,h_) in enumerate(zip(xt,yt,ht)):
                    #plt.plot(x_,y_,'o',color=cols[ind_hom],alpha=h_/maxh) #selected
                    plt.scatter(x_,y_,facecolor=cols[ind_hom],edgecolor='black') #selected
                    if plt_val>1:
                        plt.text(x_,y_,str(i_+1),color=cols[ind_hom],alpha=h_/maxh) #selected
        #plot all the other points in black with alpha set.
        xyz_all=[[],[],[]]
        for tad in bkpoints_dic.keys():
            (xg,yg,zg,hg)=bkpoints_dic[tad]
            xyz_all[0].extend(xg)
            xyz_all[1].extend(yg)
            xyz_all[2].extend(zg)
        plt.plot(xyz_all[0],xyz_all[1],'.',color='black',alpha=0.1)
        plt.axis('equal')
        if save_fl is not None:
            f.savefig(save_fl)
        #plt.show()
    # final output is xys_selected of size 2x10x3
    # hs selected is 2x10 brightnesses for the selected spots
    # Xs is all the candidates unsplited of size 10xvariablex3
    # Hs is the brightnesses associated with the candidates Xs , size 10xvarible
    # inds gives the homolog index for the Xs with values 0 or 1, size 10xvarible
    # c1 and c2 are the two homolog centers
    return xys_selected,xys_errors_selected,hs_selected
import copy
def apply_drfitcor(dic_):
    dic__ = copy.deepcopy(dic_)
    for cell in dic__.keys():
        if 'cell' in cell:
            drift_key = cell.split('cell')[0]+'Drift'
            for R in dic__[cell].keys():
                if len(dic__[cell][R]):
                    dic__[cell][R][:,1:4] = dic_[cell][R][:,1:4]-dic_[drift_key][R][0]
    return dic__
def get_zxyh(dic,r_key='P1',cell='cell_3',nomr_h=False,norm_n=np.inf,keep_n=np.inf):
    """Given an analysis dictionary this drift corrects and extracts fits for a specific readout"""
    
    drift_key = 'Drift'
    if 'fov' in cell: drift_key = cell.split('cell')[0]+'Drift'
    drift = dic[drift_key][r_key][0]
    
    pfits = dic[cell][r_key]
        
    if type(pfits) is str:
        return np.array([[np.nan]*4]).T
    
    keep_n = int(np.min([keep_n,len(pfits)]))
    zxy = pfits[:keep_n,1:4]
    #print drift_key,r_key,drift
    zxy_c = zxy - [drift]
    h = pfits[:keep_n,0]
    if nomr_h:
        norm_n = int(np.min([norm_n,len(h)]))
        h=h/np.median(h[:norm_n])
    return np.concatenate([zxy_c.T,[h]])
def get_intersection(dic,R,P,cell,cutoffs=[1,0.75,0.75,np.inf],norm_n=np.inf,keep_n=np.inf,return_inds=False):
    """This is a bit too simplistic, does not deal with bleadthroughs or caryovers"""
    nomr_h = False
    if norm_n>0:
        nomr_h=True
    zR,xR,yR,hR = get_zxyh(dic,r_key=R,cell=cell,nomr_h=nomr_h,norm_n=norm_n,keep_n=keep_n)
    zP,xP,yP,hP = get_zxyh(dic,r_key=P,cell=cell,nomr_h=nomr_h,norm_n=norm_n,keep_n=keep_n)
    if len(xP)==0  or len(xR)==0:
        return np.array([[[],[]],[[],[]],[[],[]],[[],[]]])
    matz = cdist(np.expand_dims(zR,-1),np.expand_dims(zP,-1))
    matx = cdist(np.expand_dims(xR,-1),np.expand_dims(xP,-1))
    maty = cdist(np.expand_dims(yR,-1),np.expand_dims(yP,-1))
    math = cdist(np.expand_dims(hR,-1),np.expand_dims(hP,-1))
    #dist = np.sqrt(matz*matz+matx*matx+maty*maty)
    good = (matz<cutoffs[0])&(matx<cutoffs[1])&(maty<cutoffs[2])#cutoff on distances
    Rg,Pg = np.where(good)
    hRg,hPg=hR[Rg],hP[Pg]
    hdiff = np.abs(hRg-hPg)/(hRg+hPg)*2
    keeph = hdiff<cutoffs[3]
    Rg,Pg =Rg[keeph],Pg[keeph]
    if return_inds:
        return [[xR[Rg],xP[Pg]],[yR[Rg],yP[Pg]],[zR[Rg],zP[Pg]],[hR[Rg],hP[Pg]]],Rg,Pg
    return [[xR[Rg],xP[Pg]],[yR[Rg],yP[Pg]],[zR[Rg],zP[Pg]],[hR[Rg],hP[Pg]]]
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
def get_code(fl=r'C:\Users\bogdan\code_seurat\WholeGenome_MERFISH\SI11.fasta'):
    """
    Given a fasta file in the format of SI11 this returns 2 dictionaries: tad_to_PR,PR_to_tad
    """
    names,seqs = fastaread(fl)
    tad_to_PR = {}
    PR_to_tad = {}
    for nm in names: 
        if 'ChrTAD' in nm:
            tad,iP,iR=(nm.split('_ChrTAD')[-1].split('__')[0],nm.split('_iP')[-1].split('__')[0],nm.split('_iR')[-1].split('__')[0])
            P = 'P'+str(int(iP)+1)
            R = 'R'+str(int(iR)+1)
            tad_to_PR[tad]=(P,R)
            PR_to_tad[P+R]=tad
    return tad_to_PR,PR_to_tad
def extend_tad_to_PR(tad_to_PR,PR_to_tad,num_blank=np.inf):
    """Extend the encoding dictionaries with the blank controls up to num_blank"""
    counter=0
    for Pi in range(16):
        for Ri in range(16):
            P='P'+str(Pi+1)
            R='R'+str(Ri+1)
            if P+R not in PR_to_tad.keys():
                counter+=1
                tad_to_PR['blank'+str(counter)]=(P,R)
                if counter>num_blank:
                    return tad_to_PR,PR_to_tad
    return tad_to_PR,PR_to_tad
def Rs_to_Rnm(Rs):
    ints = np.sort([int(R[1:]) for R in Rs])
    return 'R'+'R'.join(map(str,ints))
def get_code_v2(fl = r'C:\Users\bogdan\code_seurat\WholeGenome_MERFISH\Coordinates_code_1000region.csv'):
    """
    Given a .csv file with header this returns 2 dictionaries: tad_to_PR,PR_to_tad
    """
    lst = [(ln[:-1].split(',')[0].replace('__','_'),['R'+R for R in ln[:-1].split(',')[3].split('--')])
     for ln in open(fl,'r')][1:]
    tad_to_PR = dict(lst)
    PR_to_tad = {Rs_to_Rnm(Rs):nm for nm,Rs in lst}
    return tad_to_PR,PR_to_tad
def extend_tad_to_PR_v2(tad_to_PR,PR_to_tad,num_blank=np.inf):
    """Extend the encoding dictionaries with the blank controls up to num_blank"""
    counter=0
    Rs_all = np.unique(tad_to_PR.values())
    for ri in range(len(Rs_all)):
        for rj in range(ri):
            Ri=Rs_all[ri]
            Rj=Rs_all[rj]
            if Rs_to_Rnm([Ri,Rj]) not in PR_to_tad.keys():
                counter+=1
                tad_to_PR['blank'+str(counter)]=[Ri,Rj]
                if counter>num_blank:
                    return tad_to_PR,PR_to_tad
    return tad_to_PR,PR_to_tad
def apply_colcor(dic,dic_chr={}):
    dic_ = copy.deepcopy(dic)
    cells = [key for key in dic_.keys() if 'cell_' in key]
    channels = [key for key in dic_.keys() if 'Channels' in key][0]
    #cells = ['cell_'+str(int_) for int_ in np.sort([int(key.replace('cell_',''))for key in cells])]
    # perform chromatic correction
    for cell in cells:
        rds = list(dic_[cell].keys())
        for rd_ in rds:
            m_col = dic_chr.get('m_647_'+dic_[channels][rd_],None)
            if type(dic_[cell][rd_]) is str:
                dic_[cell][rd_]=np.array([])
            if len(dic_[cell][rd_])>0:
                dic_[cell][rd_][:,1:4]=apply_colorcor(dic_[cell][rd_][:,1:4],m_col)#647->750
    return dic_
def get_dics_cand_simple(dic,cell,tad_to_PR,PR_to_tad,cutoffs=1.5*np.array([1,0.75,0.75,2]),keep_n=60,norm_n=30):
    dic_cand_err = {}
    dic_cand = {}
    dic_cand_o = {}
    chrs = list(tad_to_PR.keys())
    for chr__,(P,R) in zip(chrs,[tad_to_PR[chr_] for chr_ in chrs]):
        (xs,ys,zs,hs),Rg,Pg = get_intersection(dic,R,P,cell,cutoffs=cutoffs,keep_n=keep_n,norm_n=norm_n,return_inds=True)
        xg,yg,zg,hg = [np.mean(val_,0) for val_ in [xs,ys,zs,hs]]
        errors = []
        if len(xs)>0: errors = [np.diff(val_,axis=0)[0] for val_ in [xs,ys,zs,hs]]
        dic_cand_err[chr_]=np.array(errors)
        dic_cand[chr__]=np.array((xg,yg,zg,hg))
        dic_cand_o[chr__] ={}
        for L,Lg in [(R,Rg),(P,Pg)]:
            dic_cand_o[chr__][L] = dic[cell][L][Lg,1:4]
    return dic_cand,dic_cand_err,dic_cand_o
def get_dics_cand(dic,cell,tad_to_PR,PR_to_tad,cutoffs=1.5*np.array([1,0.75,0.75,2]),keep_n=60,norm_n=30):
    """Given 
    a dictionary of fits <dic> from _analyzed.pkl, 
    a cell id <cell> in dic,
    the encoding dics from PRs to Tads and viceversa <tad_to_PR>,<PR_to_tad>
    
    the cutoff distance for z,x,y and brightness for colocalization, 
    the number of fits to consider per cell keep_n
    the number of fits to median for the brightness renormalization
    
    Returns
    <dic_cand> dictionary which for a given tad returns the mean z,x,y and brightness
    <dic_cand_err> dictionary which for a given tad returns the difference for z,x,y and brightness
    <dic_cand_o> dictionary which for a given tad returns a dictionary with the readouts for each tad and the z,x,y in the original dax
    """
    #1. First pass through at decoding
    dic_rep = {}
    partner_dic = {}
    chrs = list(tad_to_PR.keys())
    for chr__,(P,R) in zip(chrs,[tad_to_PR[chr_] for chr_ in chrs]):
        (xs,ys,zs,hs),Rg,Pg = get_intersection(dic,R,P,cell,cutoffs=cutoffs,keep_n=keep_n,norm_n=norm_n,return_inds=True)
        xg,yg,zg,hg = [np.mean(val_,0) for val_ in [xs,ys,zs,hs]]

        errors = []
        if len(xs)>0: errors = [np.diff(val_,axis=0)[0] for val_ in [xs,ys,zs,hs]]
        if not dic_rep.has_key(P): dic_rep[P]=[]
        if not dic_rep.has_key(R): dic_rep[R]=[]
        dic_rep[P].append((Pg,R,Rg,np.array(errors).T,np.array((xg,yg,zg,hg)).T,dic[cell][P][Pg,1:4],dic[cell][R][Rg,1:4],chr__))
        dic_rep[R].append((Rg,P,Pg,np.array(errors).T,np.array((xg,yg,zg,hg)).T,dic[cell][R][Rg,1:4],dic[cell][P][Pg,1:4],chr__))
    
    #2. Get a list of all the errors, positions, spot ids and decoded ids
    errors_tot,pos_tot,poso_tot,id_tot,partner_tot,pospartner_tot,chr_tot = [],[],[],[],[],[],[]
    errors_tot_ = []
    for key in dic_rep.keys():
        for indRep,partner_Rd,partner_ind,errors,pos,pos_o,pos_o_part,chr_ in dic_rep[key]:
            errors_=errors
            if 'blank' in chr_:
                errors_ = np.zeros_like(errors)+np.inf
            errors_tot.extend(errors)
            errors_tot_.extend(errors_)
            pos_tot.extend(pos)
            poso_tot.extend(pos_o)
            pospartner_tot.extend(pos_o_part)
            chr_tot.extend([chr_]*len(pos))
            for iRep,ipart in zip(indRep,partner_ind):
                id_tot.append(key+'_'+str(iRep))
                partner_tot.append(partner_Rd+'_'+str(ipart))
    errors_tot = np.array(errors_tot)
    errors_tot_ = np.array(errors_tot_)
    stds = np.std(errors_tot,0)
    # normalize by std of each distance ind 4d space (x,y,z and normalized brightness) and sort by best precision
    inds_sort = np.argsort(np.sum((errors_tot_/[stds])**2,1))
    #ids,cts = np.unique(id_tot,return_counts=True)
    #np.unique(cts,return_counts=True)

    # construct candidate dictionary
    dic_cand={}
    dic_cand_err={}
    dic_cand_o={}
    keep_track=[]

    for iS in inds_sort:
        id_ = id_tot[iS]
        chr_=chr_tot[iS]
        pos_=pos_tot[iS]
        poso_ = poso_tot[iS]
        error_=errors_tot[iS]
        partner_ = partner_tot[iS]
        poso_part = pospartner_tot[iS]
        if (id_ not in keep_track) and (partner_ not in keep_track):
            #exclude the added element and its partner
            keep_track.append(id_)
            keep_track.append(partner_)
            #initiate if empty
            if not dic_cand.has_key(chr_):
                dic_cand[chr_]=[]
                dic_cand_err[chr_]=[]
                dic_cand_o[chr_]={}
            dic_cand[chr_].append(pos_)
            dic_cand_err[chr_].append(error_)
            PR_1 = id_.split('_')[0]
            PR_2 = partner_.split('_')[0]
            #print PR_1,PR_2
            dic_cand_o[chr_][PR_1]=dic_cand_o[chr_].get(PR_1,[])+[poso_]
            dic_cand_o[chr_][PR_2]=dic_cand_o[chr_].get(PR_2,[])+[poso_part]

    dic_cand={key:np.array(value).T for key,value in zip(dic_cand.keys(),dic_cand.values())}
    dic_cand_err={key:np.array(value).T for key,value in zip(dic_cand_err.keys(),dic_cand_err.values())}
    for key in tad_to_PR.keys():
        if not dic_cand.has_key(key):
            dic_cand[key]=np.array([[]]*4)
    return dic_cand,dic_cand_err,dic_cand_o
def sort_str(lst,sep='_'):
    return np.array(lst)[np.argsort([int(item.split('_')[-1]) for item in lst])]
def get_approx_cov_error(dic_cand,verbose=True):
    ###get the approximate coverage
    lens = [len(dic_cand.get(key,np.array([])).T) for key in dic_cand.keys() if 'chr' in key]
    lens_blank = [len(dic_cand.get(key,np.array([])).T) for key in dic_cand.keys() if 'blank' in key]
    approx_cov = 1.*np.sum(np.array(lens)>1)/len(lens)
    mean_blank = np.mean(lens_blank)
    if verbose:
        print('Mean number of blank/frame: '+str(mean_blank))
        print("Coverage: "+str(approx_cov))
    return approx_cov,mean_blank
