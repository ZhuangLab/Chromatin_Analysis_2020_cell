import os,glob,sys
import numpy as np
import cPickle as pickle
import PostFitting as pf
import Fitting_v4 as ft
import IOTools as io
from tqdm import tqdm_notebook as tqdm
#from tqdm import tqdm as tqdm
import scipy.ndimage as ndi
from scipy.spatial.distance import cdist
import matplotlib.pylab as plt

import cPickle as pickle


from scipy.spatial import cKDTree as KDTree
from copy import deepcopy
from scipy.spatial.distance import cdist,pdist,squareform

import cv2
def normalzie_im(im,sz=20):
    im_ = np.array(im,dtype=np.float32)
    im_blur = np.array([cv2.blur(im__,(sz,sz)) for im__ in im_],dtype=np.float32)
    im_ =im_-im_blur
    return im_
    
    
def get_standard_seeds(im_sm,sz_blur=20,filt_size=4,th_stds=4,npts=150):
    """Get the seed points in an image.
    #1 perform a gaussian filter
    #2 find local maxima within a radius <filt_size> with <th_seed> above local maximum (in log space)
    """
    if sz_blur!=0: im_norm = normalzie_im(im_sm,sz_blur)
    else: im_norm=im_sm
    std_ = np.std(im_norm)
    hcutoff = std_*th_stds
    z,x,y = np.where(im_norm>hcutoff)
    h_im = im_norm[z,x,y]
    sz,sx,sy = im_norm.shape
    keep = h_im>0
    pix = int(filt_size/2)
    deltas = range(-pix,pix+1)
    for deltax in deltas:
        for deltay in deltas:
            for deltaz in deltas:
                #if (np.abs(deltax)+np.abs(deltay)+np.abs(deltaz))>0:
                keep &= (h_im>=im_norm[(z+deltaz)%sz,(x+deltax)%sx,(y+deltay)%sy])
    zf,xf,yf = z[keep],x[keep],y[keep]
    hf = im_norm[zf,xf,yf]
    inds = np.argsort(hf)[::-1]
    centers_zxy = np.array([zf[inds],xf[inds],yf[inds]]).T
    if npts is not None:
        centers_zxy = centers_zxy[:npts]
    return centers_zxy

def get_dax_im(dax_fl,icol,num_cols,zxym,zxyM,pad=10,im_meds=None,no_med=False,remove_first=0):
    im = io.DaxReader(dax_fl).loadMap().swapaxes(1,2)[remove_first:]
    if pad!=0: im = im[:][icol:-pad:num_cols][int(pad/num_cols)+1:]
    else: im = im[:][icol::num_cols]
    im = np.array(im[zxym[0]:zxyM[0],zxym[1]:zxyM[1],zxym[2]:zxyM[2]],dtype=np.float32)
    if im_meds is None:
        if not no_med:
            im = im/np.median(im)
    else:
        im_med = im_meds[icol]
        norm_ = np.median(im_med)
        im_med = im_med[zxym[1]:zxyM[1],zxym[2]:zxyM[2]]
        im = (im-im_med)/norm_
    return im
    


def get_homolog_centers_(xyz_all,chr_all):
    ### usefull inner functions
    def compute_cov_rg_fast(xyz_all,chr_all,i1,i2,dist_mat,chr_inds):
        dist1 = dist_mat[i1]
        dist2 = dist_mat[i2]
        keep = dist1>dist2
        nonkeep = ~keep
        rg_mean = np.mean(dist1[nonkeep])+np.mean(dist2[keep])
        cov = np.sum(np.dot(chr_inds,nonkeep)&np.dot(chr_inds,keep))
        return cov,rg_mean
    def get_cum_sums(ar_):
        ar__=[]
        count=1
        for i in range(len(ar_)):
            change = ar_[i]!=ar_[i+1] if i+1<len(ar_) else True
            if change:
                ar__.extend([i+1]*count)
                count=1
            else:
                count+=1
        return np.array(ar__)/float(ar__[-1])
    def get_cumprobs(vals,bigger=True):
        invert = 1 if bigger else -1
        asort_ = np.argsort(vals)[::invert]
        cumsums = get_cum_sums(vals[asort_])[np.argsort(asort_)]
        return cumsums


    ### Pool all candidates together
    chr_all = np.array(chr_all)
    chr_all_u = np.unique(chr_all)
    chr_inds = np.array([chr_all==chr_u for chr_u in chr_all_u])
    
    dist_mat = squareform(pdist(xyz_all))
    ### If not enough data return empty
    if len(xyz_all)<2:
        return [np.nan]*3, [np.nan]*3
        #return c1,c2
    
    ### Compute best centers that first maximize coverage and then minimize radius of gyration
    rgs,covs,pairs=[],[],[]
    for i1 in range(len(xyz_all)):
        for i2 in range(i1):
            covT,rg_meanT = compute_cov_rg_fast(xyz_all,chr_all,i1,i2,dist_mat,chr_inds)
            covs.append(covT)
            rgs.append(rg_meanT)
            pairs.append([i1,i2])
    rgs,covs = np.array(rgs,dtype=np.float32),np.array(covs,dtype=np.float32)
    ibest = np.argmax(get_cumprobs(rgs,bigger=False)*get_cumprobs(covs,bigger=True))
    i1,i2 = pairs[ibest]
    c1=xyz_all[i1]
    c2=xyz_all[i2]
    return c1,c2
def update_color(cell_obj,ms_chr):
    multi_ms = cell_obj.dic_noncells[cell_obj.fov]['multi_ms']
    for R in cell_obj.dic_cell.keys():
        ps = cell_obj.dic_cell[R]
        if len(ps)>0:
            col = int(ps[0,-3])
            zxy = ps[:,-6:-3].copy()
            ms_ = list(multi_ms[R])
            ms_[0]=ms_chr[col]
            for m in ms_:
                zxy = ft.apply_colorcor(zxy,m)
            cell_obj.dic_cell[R][:,1:4]=zxy
            #if col==2:
            #    cell_obj.dic_cell[R][:,1]+=1
    return cell_obj
class cell_analysis():
    def __init__(self,cellfiles,flip_R90R65=False):
        #Load data
        self.dic_ = {}
        for fl in cellfiles:
            dic_T = pf.load_double_dic(fl)
            #"""
            if flip_R90R65:
                for key in dic_T.keys():
                    for r in dic_T[key].keys():
                        if len(dic_T[key][r])>0:
                            dic_T[key]['R90'],dic_T[key]['R65']=dic_T[key]['R65'],dic_T[key]['R90']
                            pass
            #"""
            self.dic_.update(dic_T)
        #print self.dic_['Conv_zscan_06__cell87']['R33'][:,0]
        self.dic_noncells ={}
        for fl in cellfiles:
            fov = os.path.basename(fl).split('__cell')[0]
            self.dic_noncells[fov] = pickle.load(open(fl.replace('cells.npy','noncells.pkl')))
            if flip_R90R65:
                self.dic_noncells[fov]['R_to_htag']['R90'],self.dic_noncells[fov]['R_to_htag']['R65']=self.dic_noncells[fov]['R_to_htag']['R65'],self.dic_noncells[fov]['R_to_htag']['R90']
        fovs = list(self.dic_noncells.keys())
        R_to_htag = self.dic_noncells[fovs[0]]['R_to_htag']
        col_dic = {R:int(R_to_htag.get(R,'_col-1').split('_col')[-1]) for R in R_to_htag if 'B'!=R[0]}
        #if flip_R90R65:
        #    col_dic['R90'],col_dic['R65']=col_dic['R65'],col_dic['R90']
        self.col_dic = col_dic
        #get cell dic
        self.cells = np.sort([key for key in self.dic_.keys()])
        self.dic_centers = {}
        self.load_DNA_code()
    def normalize_cell_dic(self,ms_chr=None,forced=[],coltag='Q',normalize_color_hybe=True):
        #if not hasattr(self,'dic_original'):
        #    self.dic_original = deepcopy(self.dic_)
        #dic_ = deepcopy(self.dic_original)
        #extend with original positions
        """
        if ms_chr is not None:
            for cell in dic_:
                dic_cell = dic_[cell]
                fov = cell.split('__cell')[0]
                multi_ms = self.dic_noncells[fov]['multi_ms']
                for Rkey in dic_cell:
                    ps = dic_cell[Rkey]
                    if len(ps)>0:
                        col = int(ps[0,-3])
                        zxy = ps[:,-6:-3].copy()
                        ms_ = list(multi_ms[Rkey])
                        ms_[0]=ms_chr[col]
                        for m in ms_:
                            zxy = ft.apply_colorcor(zxy,m)
                        dic_[cell][Rkey][:,1:4] = zxy#np.concatenate([ps,ps[:,[0]],zxyo,cols],axis=-1)
        """               
        #replace repeats
        dic_ = self.dic_
        dic_,flips,bestRs,repeat_sets = pf.replace_repeats_v2(dic_,forced=forced)
        fov= self.dic_noncells.keys()[0]
        print("Picks")
        for set_R,R in zip(repeat_sets,bestRs):
            print R,self.dic_noncells[fov]['R_to_htag'][R]
            print [(r,self.dic_noncells[fov]['R_to_htag'][r]) for r in set_R]
        for fov in self.dic_noncells:
            for r,R in flips:
                for key in self.dic_noncells[fov].keys():
                    if R in self.dic_noncells[fov][key] and r in self.dic_noncells[fov][key]:
                        self.dic_noncells[fov][key][R],self.dic_noncells[fov][key][r]=self.dic_noncells[fov][key][r],self.dic_noncells[fov][key][R]
        for r,R in flips:
            self.col_dic[r],self.col_dic[R]=self.col_dic[R],self.col_dic[r]
        col_dic = self.col_dic
        if normalize_color_hybe:
            dic_mc_colpercell = pf.flatten_colors(dic_,col_dic,coltag = coltag,ncells=100,npts = 30,replace_dic = True,plt_val=False)
            dic_mc_Rs = pf.flatten_hybes(dic_,ncells=1000,npts = 30,Rref = 'R2',replace_dic = True)
        self.dic_ = dic_
    def set_cell(self,cell_index):
        if type(cell_index) is int:
            self.cell_index = cell_index
            self.cell = self.cells[self.cell_index]
        else:
            self.cell = cell_index
        self.fov = self.cell.split('__cell')[0]
        self.dic_cell =deepcopy(self.dic_[self.cell])
    def load_DNA_code(self):
        tad_to_PR,PR_to_tad = pf.get_code_v2()
        Rs = np.array(['R'+str(Ri+1)for Ri in range(100)])
        
        self.tad_to_PR,self.PR_to_tad =tad_to_PR,PR_to_tad
        self.Rs = Rs
        
        Rkeys=Rs
        Rkey_to_ind = {R:iR for iR,R in enumerate(Rkeys)}
        valid_keys = tad_to_PR.keys()
        valid_pairs = [tad_to_PR[key] for key in  valid_keys]
        self.valid_colors = [[self.col_dic[r1],self.col_dic[r2]]for r1,r2 in valid_pairs]
        
        valid_pairs = [[Rkey_to_ind[r1],Rkey_to_ind[r2]]for r1,r2 in valid_pairs]
        self.valid_pairs = valid_pairs
        self.valid_keys = valid_keys
        
        valid_set = set(map(tuple,valid_pairs))
        allset = set([(jR,iR) for iR in range(len(Rkeys)) for jR in range(iR)])
        invalid_pairs = list(allset.difference(valid_set))
        self.invalid_pairs = invalid_pairs
        
        valid_chr = np.array([key.split('_')[0]for key in valid_keys])
        chrs = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 
                    'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 
                    'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']      
        self.chrs=chrs
        chr_ivs = []
        iregs = np.array([int(vk.split('_')[-1])for vk in valid_keys])
        for chr_ in chrs:
            ichrs = np.where(valid_chr==chr_)[0]
            ichrs = ichrs[np.argsort(iregs[ichrs])]
            chr_ivs.append(ichrs)
        
        iv_to_chr = {iv:chr_ for chr_,ivs in enumerate(chr_ivs) for iv in ivs}
        
        self.chr_ivs = chr_ivs
        self.iv_to_chr = iv_to_chr
        
    def get_candidates(self,cutoff_candidate = [1.75,1.75,1.75],cap_invalid = 2000,random =True):
        """Based on interdistance cut_off select the valid/invalid pairs of spots
        Stores info in self.ps_pairs_valid and self.ps_pairs_invalid
        """
        Rkeys = self.Rs
        dic_cell = self.dic_cell
        valid_pairs = self.valid_pairs
        invalid_pairs = self.invalid_pairs
        
        Rinds,pts,ps = [],[],[]
        cutoff=cutoff_candidate
        for iR,R in enumerate(Rkeys):
            ps_ = dic_cell.get(R,[])
            if len(ps_)>0:
                Rinds.extend([iR]*len(ps_))
                pts.extend(ps_[:,1:4]/cutoff)
                ps.extend(ps_)
        
        ps = np.array(ps)
        tree = KDTree(pts)
        pairs = tree.query_pairs(1,p=np.inf,output_type='ndarray')
        Rinds = np.array(Rinds)
        Rind_pairs = Rinds[pairs]
        ps_pairs = ps[pairs]
        
        #Valid candidates
        ps_pairs_valid = []
        pairs_valid = {}
        for iv,(r1,r2) in enumerate(valid_pairs):
            keep = (Rind_pairs[:,0]==r1)&(Rind_pairs[:,1]==r2)
            ps_pairs_ = ps_pairs[keep]
            ps_pairs_valid.append(ps_pairs_)
            # add to dictionary to allow for uniqueness constraint
            pairs_ = pairs[keep]
            for ipair,pair_ in enumerate(pairs_):
                pp1,pp2 = pair_
                pairs_valid[pp1]=pairs_valid.get(pp1,[])+[[iv,ipair]]
                pairs_valid[pp2]=pairs_valid.get(pp2,[])+[[iv,ipair]]
        
        self.pairs_valid = pairs_valid
        self.ps_pairs_valid = ps_pairs_valid
            
            
        #Invalid candidates due to random intersections
        ps_pairs_invalid = []
        num_points=0
        if random:
            invalid_pairs = np.random.permutation(invalid_pairs)
        for r1,r2 in invalid_pairs:
            keep = (Rind_pairs[:,0]==r1)&(Rind_pairs[:,1]==r2)
            ps_pairs_ = ps_pairs[keep]
            ps_pairs_invalid.append(ps_pairs_)
            num_points+=np.sum(keep)
            if num_points>cap_invalid:
                break
        self.ps_pairs_invalid = ps_pairs_invalid
    def get_Ds(self,ps_pairs,ih=0):
        Ds_all = []
        for ps_pairs_ in ps_pairs:
            if len(ps_pairs_)>0:
                p1s,p2s = ps_pairs_[:,0],ps_pairs_[:,1]
                DH = (p1s[:,ih]+p2s[:,ih])/2.
                Dh = np.abs(p1s[:,ih]-p2s[:,ih])
                Dh*=0
                Dzxy = np.abs(p1s[:,1:4]-p2s[:,1:4])
                #Dshape = np.mean(np.abs(p1s[:,5:11]-p2s[:,5:11]),axis=-1)
                Dshape = np.zeros(len(p1s))
                Ds = np.array([DH,Dh,Dzxy[:,0],Dzxy[:,1],Dzxy[:,2],Dshape]).T
                Ds_all.append(Ds)
            else:
                Ds_all.append([])
        Ds_distr = np.array([D_ for Ds_ in Ds_all for D_ in Ds_])
        Ds_distr = np.sort(Ds_distr,0).T
        return Ds_all,Ds_distr
    def get_scores(self,Ds_all,Ds_distr,biggers = [True,False,False,False,False,False,False,False]):
        scores_all = [[np.sum(np.log([pf.cum_val(D_distr,d_,bigger)for d_,D_distr,bigger in zip(D_,Ds_distr,biggers)])) 
               for D_ in Ds_]
              for Ds_ in Ds_all]
        return scores_all
    def to_unique_scores(self,scores_valid,pairs_valid):
        scores_valid_unique = deepcopy(scores_valid)
        for ipoint in pairs_valid:
            iv_ips = np.array(pairs_valid[ipoint])
            if len(iv_ips)>0:
                scores_valid_nu = [scores_valid[iv][ip] for iv,ip in iv_ips]
                bad_iv_ips =  iv_ips[np.argsort(scores_valid_nu)[:-1]]
                for iv,ip in bad_iv_ips:
                    scores_valid_unique[iv][ip]=-np.inf
        return scores_valid_unique

    def get_main_scores(self):
        #valid
        self.Ds_valid,self.Ds_distr_valid = self.get_Ds(self.ps_pairs_valid)
        self.scores_valid = self.get_scores(self.Ds_valid,self.Ds_distr_valid)
        self.scores_valid_unique = self.to_unique_scores(self.scores_valid,self.pairs_valid)
        #invalid
        self.Ds_invalid,self.Ds_distr_invalid = self.get_Ds(self.ps_pairs_invalid)
        self.scores_invalid = self.get_scores(self.Ds_invalid,self.Ds_distr_valid)
        
    def get_th_bleed(self,nkeep=2,plt_val=False):
        scores_valid,scores_invalid = self.scores_valid,self.scores_invalid
        scores_invalid_ = np.array([vl_ for vl in scores_invalid for vl_ in vl])
        scores_invalid_ = scores_invalid_[~np.isinf(scores_invalid_)]
        scores_valid_ = np.array([vl_ for vl in scores_valid for vl_ in np.sort(vl)[-nkeep:]])#scores_valid_unique
        scores_valid_ = scores_valid_[~np.isinf(scores_valid_)]
        th,bleed = pf.return_best_th(np.sort(scores_valid_),np.sort(scores_invalid_),plt_val=plt_val)
        self.th_score,self.bleed_score = th,bleed
        
    def enhnace_Ds_1(self,Ds_all,ps_pairs_all,nneigh=10,valid=True):
        #Check distances and add that as scores
        chr_trees = self.chr_trees
        DsC_all = deepcopy(Ds_all)
        for iv,ps_ in enumerate(ps_pairs_all):
            if len(ps_)>0:
                centers = np.mean(ps_[:,:,1:4],axis=1)
                tree = chr_trees[self.iv_to_chr[iv]] if valid else chr_trees[np.random.randint(len(chr_trees))]
                dcs,_=tree.query(centers,k=nneigh)
                dcs = np.mean(dcs,axis=-1)
                DsC_all[iv] = np.concatenate([Ds_all[iv],dcs[:,np.newaxis]],axis=-1)
        Ds_distr = np.array([D_ for Ds_ in DsC_all for D_ in Ds_])
        Ds_distr = np.sort(Ds_distr,0).T
        return DsC_all,Ds_distr
    def enhance_scores_1(self,nneigh=10):
        self.DsC_valid,self.DsC_distr_valid = self.enhnace_Ds_1(self.Ds_valid,self.ps_pairs_valid,nneigh=nneigh,valid=True)
        self.scores_valid = self.get_scores(self.DsC_valid,self.DsC_distr_valid)
        self.scores_valid_unique = self.to_unique_scores(self.scores_valid,self.pairs_valid)
        self.DsC_invalid,self.DsC_distr_invalid = self.enhnace_Ds_1(self.Ds_invalid,self.ps_pairs_invalid,nneigh=nneigh,valid=False)
        self.scores_invalid = self.get_scores(self.DsC_invalid,self.DsC_distr_valid)
    def enhnace_Ds_2(self,Ds_all,ps_pairs_all,ihom=0,valid=True):
        #enhance with distance from the centers
        DsC_all = deepcopy(Ds_all)
        for iv,ps_ in enumerate(ps_pairs_all):
            if len(ps_)>0:
                centers = np.mean(ps_[:,:,1:4],axis=1)
                i_chr = self.iv_to_chr[iv] if valid else np.random.randint(len(self.dic_centers))
                chom = self.dic_centers[i_chr][ihom]
                dcs = cdist(centers,[chom])
                DsC_all[iv] = np.concatenate([Ds_all[iv],dcs],axis=-1)
                #DsC_all[iv][:,-1] =dcs[:,0]
        Ds_distr = np.array([D_ for Ds_ in DsC_all for D_ in Ds_])
        Ds_distr = np.sort(Ds_distr,0).T
        
        #base = np.zeros([Ds_distr.shape[0],1])
        #Ds_distr = np.concatenate([base-np.inf,Ds_distr,base+np.inf],axis=-1)
        
        return DsC_all,Ds_distr
    def enhance_scores_2(self,ihom=0,Distrib = None):
        self.DsCC_valid,self.DsCC_distr_valid = self.enhnace_Ds_2(self.DsC_valid,self.ps_pairs_valid,ihom=ihom,valid=True)
        if Distrib is None: Distrib = self.DsCC_distr_valid
        self.scores_valid = self.get_scores(self.DsCC_valid,Distrib)
        self.scores_valid_unique = self.to_unique_scores(self.scores_valid,self.pairs_valid)
        self.DsCC_invalid,self.DsCC_distr_invalid = self.enhnace_Ds_2(self.DsC_invalid,self.ps_pairs_invalid,ihom=ihom,valid=False)
        self.scores_invalid = self.get_scores(self.DsCC_invalid,Distrib)
    def keep_valid(self,unique=False,nkeep=None,th=None):
        ps_pairs_valid = self.ps_pairs_valid
        scores_valid = self.scores_valid
        if unique: scores_valid = self.scores_valid_unique

        if th is None: th = self.th_score

        ps_pairs_valid_keep = []
        for iv,ps_pairs_ in enumerate(ps_pairs_valid):
            if len(ps_pairs_)>0:
                scores_ = np.array(scores_valid[iv])
                keep = np.argsort(scores_)[:]
                if nkeep is not None: keep = keep[-nkeep:]
                keep = keep[scores_[keep]>th]
                ps_pairs_valid_keep.append(ps_pairs_[keep])
            else:
                ps_pairs_valid_keep.append([])
        centers_keep = []
        for ps_pairs_ in ps_pairs_valid_keep:
            if len(ps_pairs_)>0:
                centers_keep.append(np.mean(ps_pairs_[:,:,1:4],axis=1))
            else:
                centers_keep.append([])
        centers_keep = np.array(centers_keep)
        self.centers_keep = centers_keep
        self.ps_pairs_valid_keep = ps_pairs_valid_keep
        #coverage
        kp = np.array(map(len,centers_keep))
        kp[kp>2]=2
        self.coverage = float(np.sum(kp))/(len(kp)*2)
    def get_chr_points(self,nkeep=2):
        self.keep_valid()
        chr_ivs = self.chr_ivs
        centers_keep = self.centers_keep
        points = []
        points_ids = []
        for chr_iv in chr_ivs:
            points_chr,points_chr_ids=[],[]
            for iv in chr_iv:
                pts_ = centers_keep[iv][-nkeep:]
                points_chr.extend(pts_)
                points_chr_ids.extend([iv]*len(pts_))
            points_chr = np.array(points_chr)
            points.append(points_chr)
            points_ids.append(points_chr_ids)
        self.points_chr = points
        self.points_chr_ids = points_ids
    def get_chr_trees(self):
        self.get_chr_points()
        points = self.points_chr
        chr_trees = []
        for points_chr in points:
            if len(points_chr)>0:
                chr_trees.append(KDTree(points_chr))
            else:
                chr_trees.append(None)
        self.chr_trees = chr_trees
    def get_homolog_centers_single(self,ichr_ = 0,plt_val=False):
        xyz_all,chr_all=self.points_chr[ichr_],self.points_chr_ids[ichr_]
        c1,c2 = get_homolog_centers_(xyz_all,chr_all)
        self.dic_centers[ichr_]=[c1,c2]
        if plt_val:
            in1 = cdist([c1],xyz_all)[0]<cdist([c2],xyz_all)[0]
            in2 = ~in1
            all_pts = np.array([pt for pts_set in self.points_chr for pt in pts_set])
            cs = np.array([c1,c2])
            
            plt.figure()
            ax1_,ax2_ = 1,2
            plt.plot(all_pts[:,ax1_],all_pts[:,ax2_],'o',color='gray')
            plt.plot(xyz_all[in1,ax1_],xyz_all[in1,ax2_],'bo')
            plt.plot(xyz_all[in2,ax1_],xyz_all[in2,ax2_],'ro')
            plt.plot(cs[:,ax1_],cs[:,ax2_],'ko')
            plt.axis('equal')
    def get_homolog_centers(self):
        for ichr_ in range(len(self.points_chr)):
            self.get_homolog_centers_single(ichr_ = ichr_,plt_val=False)

    def get_chr_points_homologs(self):
        ps_pairs_valid = self.ps_pairs_valid
        scores_valid1,scores_valid2 = self.scores_valid1,self.scores_valid2
        th_score1,th_score2 = self.th_score1,self.th_score2
        chr_ivs = self.chr_ivs
        ps_pairs_valid_keep1,ps_pairs_valid_keep2 = [],[]
        
        self.refined_Distr = []
        
        for iv,ps_pairs_ in enumerate(ps_pairs_valid):
            if len(ps_pairs_)>0:
                scores_1,scores_2 = np.array(scores_valid1[iv]),np.array(scores_valid2[iv])
                ihoms = np.argmax([scores_1,scores_2],0)
                #1
                keep = (ihoms==0)&(scores_1>th_score1)
                scores_1[~keep]=-np.inf
                if np.sum(keep)>0:
                    ikeep = np.argmax(scores_1)
                    ps_pairs_valid_keep1.append(ps_pairs_[ikeep])
                    self.refined_Distr.append(self.DsCC1[iv][ikeep])
                else:
                    ps_pairs_valid_keep1.append([])
                #2
                keep = (ihoms==1)&(scores_2>th_score2)
                scores_2[~keep]=-np.inf
                if np.sum(keep)>0:
                    ikeep = np.argmax(scores_2)
                    ps_pairs_valid_keep2.append(ps_pairs_[ikeep])
                    self.refined_Distr.append(self.DsCC2[iv][ikeep])
                else:
                    ps_pairs_valid_keep2.append([])
                
            else:
                ps_pairs_valid_keep1.append([])
                ps_pairs_valid_keep2.append([])
        self.refined_Distr = np.sort(self.refined_Distr,0).T
        self.ps_pairs_valid_keep1 = ps_pairs_valid_keep1
        self.ps_pairs_valid_keep2 = ps_pairs_valid_keep2
        points1,points2 = [],[]
        for chr_iv in chr_ivs:
            points_chr1,points_chr2=[],[]
            for iv in chr_iv:
                ps_pairs_ = ps_pairs_valid_keep1[iv]
                if len(ps_pairs_)>0: points_chr1.append(np.mean(ps_pairs_[:,1:4],axis=0)) 
                else: points_chr1.append([np.nan]*3) 
                ps_pairs_ = ps_pairs_valid_keep2[iv]
                if len(ps_pairs_)>0: points_chr2.append(np.mean(ps_pairs_[:,1:4],axis=0)) 
                else: points_chr2.append([np.nan]*3) 
            points1.append(points_chr1)
            points2.append(points_chr2)
        self.points1 = points1
        self.points2 = points2
    def plot_chr(self,chri=0,ax1=1,ax2=2):
        zxy_chr1 = np.array(self.points1[chri])
        zxy_chr2 = np.array(self.points2[chri])
        zxy_all = np.array([pt for pts_hom in [self.points1,self.points2] for pts_chr in pts_hom for pt in pts_chr])
        fig = plt.figure()
        plt.title(self.cell+' - '+self.chrs[chri])
        plt.plot(zxy_all[:,ax1],zxy_all[:,ax2],'o',color='gray')
        plt.plot(zxy_chr1[:,ax1],zxy_chr1[:,ax2],'ro')
        plt.plot(zxy_chr2[:,ax1],zxy_chr2[:,ax2],'bo')
        plt.axis('equal')
        #plt.show()
        return fig
    def load_RNA_code(self):
        coords_txt = np.loadtxt(r'C:\Users\bogdan\code_seurat\WholeGenome_MERFISH\1041Coordinates_hg38.txt',dtype=str)
        dic_coords_centers = {}
        for txt in coords_txt:
            chr_ = txt.split(':')[0]
            if chr_ not in dic_coords_centers: dic_coords_centers[chr_] = []
            dic_coords_centers[chr_].append(np.mean(np.array([txt.split(':')[-1].split('-')],dtype=float)))


        names,seqs = pf.fastaread(r'C:\Users\bogdan\code_seurat\WholeGenome_MERFISH\SI16.fasta')
        names_probes = [nm for nm in names if 'regintrons' in nm]
        codes_coords = {tuple(eval(nm.split('__code:')[-1].split('_')[0])):nm.split('_coord:')[-1].split('_')[0] for nm in names_probes}
        codes_nms = {tuple(eval(nm.split('__code:')[-1].split('_')[0])):nm.split('_pb')[0].split('_')[-1] for nm in names_probes}
        chrs = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 
                            'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 
                            'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']  
        codes_reg = {}
        for code in codes_coords:
            txt = codes_coords[code]
            chr_ = txt.split(':')[0]
            coord_ = np.mean(np.array([txt.split(':')[-1].split('-')],dtype=float))
            ireg = np.argmin(np.abs(dic_coords_centers[chr_]-coord_))
            codes_reg[tuple(np.sort(code))]=[chrs.index(chr_),ireg,codes_nms[code]]+list(code)
        tad_to_QR = {}
        Qs = ['Q'+str(iQ+1) for iQ in range(np.max(np.array(codes_reg.keys()))+1)]
        for code in codes_reg:
            chri,regi=codes_reg[code][0],codes_reg[code][1]
            tad_to_QR[chrs[chri]+'_'+str(regi+1)]=(Qs[code[0]],Qs[code[1]])
        invalid_pairs_RNA,valid_pairs_RNA = [],[]
        for iQ in range(np.max(np.array(codes_reg.keys()))+1):
            for jQ in range(iQ):
                if (jQ,iQ) in codes_reg:
                    valid_pairs_RNA.append((jQ,iQ))
                else:
                    invalid_pairs_RNA.append((jQ,iQ))

        
        self.iv_to_chrreg = {ik:codes_reg[key] for ik,key in enumerate(valid_pairs_RNA)}
        self.Rs = Qs
        self.valid_pairs = valid_pairs_RNA
        self.invalid_pairs = invalid_pairs_RNA
        self.tad_to_QR = tad_to_QR
        self.codes_reg = codes_reg
    def clean_Qs(self,ih=-7,h_cutoff=4.,zxy_cutoff=1.5):
        dic_cell_clean = deepcopy(self.dic_cell)    
        dic_cell = self.dic_cell
        ks = [k for k in dic_cell.keys() if 'Q'in k and '_rep' not in k]
        for ir1 in range(len(ks)):
            r1 = 'Q'+str(ir1+1)
            r2 = 'Q'+str(ir1+4)
            if r1 in ks and r2 in ks:
                h1,zxy1 = dic_cell[r1][:,ih],dic_cell[r1][:,1:4]
                h2,zxy2 = dic_cell[r2][:,ih],dic_cell[r2][:,1:4]
                i1,i2 = np.where(cdist(zxy1/zxy_cutoff,zxy2/zxy_cutoff,metric='chebyshev')<1)
                bad = (h1[i1]/h2[i2])>h_cutoff
                p2 = dic_cell_clean[r2]
                keep = np.setdiff1d(np.arange(len(p2)),i2[bad])
                dic_cell_clean[r2] = p2[keep]
        self.dic_cell = dic_cell_clean
    def plot_cell_chromatic_difference(self,col_check = [0,1]):
        zxy_dif=[]
        for iv,cols in enumerate(self.valid_colors):
            for pairs in [self.ps_pairs_valid_keep1[iv],self.ps_pairs_valid_keep2[iv]]:
                if len(pairs)>0:
                    if cols == col_check:
                        zxy_dif.append(pairs[0][1:4]-pairs[1][1:4])
        zxy_dif = np.array(zxy_dif)
        med_dif = np.median(np.abs(zxy_dif),0)
        plt.figure()
        for iz in range(3):
            plt.hist(zxy_dif[:,iz],label = 'zxy'[iz]+' - '+str(np.round(med_dif[iz],2))+' pix',alpha=0.75);
        plt.legend()
        return np.median(zxy_dif,0),med_dif
    def get_dna_rna_pts(self,drift=[0,0,0],cutoff_dist=15,plt_val=True):
        ps_pairs_valid = self.ps_pairs_valid
        scores_valid = self.scores_valid
        points1 = np.array(self.points1)
        points2 = np.array(self.points2)
        rna_zxy = []
        ps_rna = []
        rna_names = []
        rna_iqs = []
        #rna_scores = []
        dna_zxy = []
        ds = []
        dna_loc = []
        for iv in range(len(ps_pairs_valid)):
            chr_,regi,genenm,q1,q2 = self.iv_to_chrreg[iv]
            ps_ = ps_pairs_valid[iv]
            if len(ps_)>0:# and chr_==chr_i:
                #scores_ = np.array(scores_valid[iv])
                #keep = np.argsort(scores_)
                #keep = keep[scores_[keep]>th_score][-2:]
                ctrs = np.mean(ps_[:,:,1:4],axis=1)-drift #RNA centers
                hctrs = np.mean(ps_[:,:,0],axis=1) # RNA brightness
                chrpt_1 = points1[chr_][regi]
                chrpt_2 = points2[chr_][regi]
                if len(ctrs)>0:
                    M = cdist([chrpt_1,chrpt_2],ctrs)
                    ds_min = np.nanmin(M,0)
                    ds.extend(ds_min)
                    keep = ds_min<cutoff_dist
                    ctrs = ctrs[keep]
                    ps_RNA = ps_[keep]
                    M = M[:,keep]
                    if len(ctrs)>0:
                        #M = cdist([chrpt_1,chrpt_2],ctrs)
                        ichrhoms = np.nanargmin(M,0)
                        rna_zxy.extend(ctrs)
                        ps_rna.extend(ps_RNA)
                        #rna_scores.append(scores_[ictr[itmp]])
                        dna_zxy.extend(np.array([chrpt_1,chrpt_2])[ichrhoms])
                        dna_loc.extend(np.array([[chr_]*len(ichrhoms),[regi]*len(ichrhoms),ichrhoms]).T)
                        rna_names.extend([genenm]*len(ichrhoms))
                        rna_iqs.extend([[q1,q2]]*len(ichrhoms))
        dna_zxy = np.array(dna_zxy)
        rna_zxy = np.array(rna_zxy)
        fig = None
        if plt_val:
            fig = plt.figure()
            plt.title(str(len(rna_zxy))+' RNA')
            plt.scatter(rna_zxy[:,1],rna_zxy[:,2])#,color=colors)
            plt.plot(dna_zxy[:,1],dna_zxy[:,2],'x',color='k')
            plt.axis('equal')
            #plt.show()
        return dna_zxy,rna_zxy,ps_rna,dna_loc,rna_names,rna_iqs,fig#,rna_scores
    def redefine_RNA(self,cutoff=[2.25,2,2],drift=[0,0,0],th_dist=10,plt_val=False):
        tad_to_PR = self.tad_to_PR
        chrs = self.chrs
        iv_to_chrreg = self.iv_to_chrreg
        dic_cell = self.dic_cell
        dna_keys = list(self.valid_keys)
        ps_pairs_DNA = self.DNA_ps_pairs_valid

        dks = []
        difs = []
        dna_zxy,rna_zxy,dna_loc = [],[],[]
        ps_rna,rna_names = [],[]
        
        
        for iv in iv_to_chrreg:
            #get info for current iv in RNA
            chri,regi,nm,q1,q2 = iv_to_chrreg[iv]
            q1,q2='Q'+str(q1+1),'Q'+str(q2+1)
            #get DNA counterpart
            key = chrs[chri]+'_'+str(regi+1)
            r1,r2 = tad_to_PR[key]
            iv_dna = dna_keys.index(key)
            ps_DNA = ps_pairs_DNA[iv_dna]

            pts_DNA = np.mean(ps_DNA[:,:,1:4],axis=1) if len(ps_DNA)>0 else []

            scores1,th1 = self.scores_valid1[iv_dna],self.th_score1
            scores2,th2 =  self.scores_valid2[iv_dna],self.th_score2

            #cutoff = [2.25,2,2]
            #pts1,pts2 = dic_cell[r1][:,1:4],dic_cell[r2][:,1:4]
            #ir1,ir2 = np.where(cdist(pts1/cutoff,pts2/cutoff,metric='chebyshev')<1)
            #pts_DNA = (pts1[ir1]+pts2[ir2])/2

            pts1,pts2 = dic_cell[q1][:,1:4],dic_cell[q2][:,1:4]
            iq1,iq2 = np.where(cdist(pts1/cutoff,pts2/cutoff,metric='chebyshev')<1)
            pts_RNA = (pts1[iq1]+pts2[iq2])/2

            ihoms = np.argmax([scores1,scores2],axis=0)
            for ihom in range(2):
                keep = ihoms==ihom
                scores = np.array([scores1,scores2])[ihom][keep]
                th = [th1,th2][ihom]
                keep_DNA = scores>th#-np.inf
                #keep_DNA = ~np.isnan(scores>th)
                if np.sum(keep_DNA)>0 and len(pts_RNA)>0:
                    M = cdist(pts_DNA[keep][keep_DNA],pts_RNA-drift,metric='chebyshev')
                    ik1,ik2 = np.unravel_index(np.argmin(M),M.shape)

                    if M[ik1,ik2]<th_dist:
                        ps_rna.append([dic_cell[q1][iq1[ik2]],dic_cell[q2][iq2[ik2]]])
                        rna_names.append(nm)
                        dks.append(M[ik1,ik2])
                        difs.append(pts_DNA[ik1]-pts_RNA[ik2])
                        if ihom==0:
                            self.ps_pairs_valid_keep1[iv_dna]=ps_DNA[keep][keep_DNA][ik1]
                            self.points1[chri][regi] = pts_DNA[keep][keep_DNA][ik1]
                        else:
                            self.ps_pairs_valid_keep2[iv_dna]=ps_DNA[keep][keep_DNA][ik1]
                            self.points2[chri][regi] = pts_DNA[keep][keep_DNA][ik1]
                        dna_zxy.append(pts_DNA[keep][keep_DNA][ik1])
                        rna_zxy.append(pts_RNA[ik2])
                        dna_loc.append([chri,regi,ihom])
        ps_rna = np.array(ps_rna)
        rna_names = np.array(rna_names)
        rna_zxy,dna_zxy = np.array(rna_zxy),np.array(dna_zxy)
        fig = None
        if plt_val:
            fig = plt.figure()
            plt.title(str(len(rna_zxy))+' RNA')
            plt.scatter(rna_zxy[:,1],rna_zxy[:,2])#,color=colors)
            plt.plot(dna_zxy[:,1],dna_zxy[:,2],'x',color='k')
            plt.axis('equal')
        return ps_rna,rna_names,dna_loc,rna_zxy,dna_zxy,fig

def map_RNA(self,R='Q1',cell = 'Conv_zscan_01__cell1',dna_zxy=[],rna_ps_pairs=None,rna_names=None,
            fln_non_cells = r'E:\Bogdan_Temp\7_27_2019_IMR90RNA-Analysis-refitted\cell_dics\Conv_zscan_01__noncells.pkl',
           master_folder = r'\\dolly\Raw_data\Bogdan\7_27_2019_IMR90RNA',extra_pts=[],vmax=5):
    
    dic_non_cells = np.load(fln_non_cells)
    R_to_htag = dic_non_cells['R_to_htag']

    htag_ = R_to_htag[R]
    htag,icol = htag_.split('__col')
    icol = int(icol)

    fov = cell.split('__cell')[0]
    dax_fl = master_folder+os.sep+htag+os.sep+fov+'.dax'
    num_cols=htag.count(',')+2
    
    codes = [self.codes_nms[nm] for nm in rna_names]
    dic_ps = {}
    dic_cds = {}

    for cd,ps_ in zip(codes,rna_ps_pairs):
        for cd_,ps__ in zip(cd,ps_):
            R_ = 'Q'+str(cd_+1)
            dic_ps[R_]=dic_ps.get(R_,[])+[ps__]
            dic_cds[R_]=dic_cds.get(R_,[])+[cd]
    ps = np.array(dic_ps.get(R,[]))
    codes = dic_cds.get(R,[])
       
    drift =[0,0,0]
    if len(ps)>0:
        if len(ps[0])==18:
            coords = ps[:,12:15]
        else:
            coords = ps[:,13:16]

        drift = np.median(coords-ps[:,1:4],axis=0)
        
    coordsi = np.array([reg for chr_hom in dna_zxy for chr_r in chr_hom for reg in chr_r],dtype=float)+drift#coords.astype(int)
    pd=10
    
    zxym,zxyM = np.nanmin(coordsi,0).astype(int)-pd,np.nanmax(coordsi,0).astype(int)+pd
    zxym[zxym<0]=0
    
    #import workers_cells_v3 as wkc
    im = get_dax_im(dax_fl,icol,num_cols,zxym,zxyM,pad=11)


    plt.figure(figsize=(10,10))
    plt.title(htag_)
    plt.imshow(np.max(im,0),vmax=vmax,cmap='gray')
    if len(ps)>0:
        coords_ = coords-zxym
        plt.plot(coords_[:,2],coords_[:,1],'rx')
    for cd,zxy in zip(codes,coords_):
        plt.text(zxy[2],zxy[1],str(cd),color='w')
    for pts in extra_pts:
        pts_ = np.array(pts)-zxym+drift
        plt.plot(pts_[:,2],pts_[:,1],'x')
    #plt.imshow(np.max(im,1),vmax=5)
    #plt.plot(coords_[:,2],coords_[:,0],'rx')
    
    
def map_DNA(self,R='R8',cell = 'Conv_zscan_01__cell1',dna_zxy=[],ps_pairs_final=None,
            fln_non_cells = r'E:\Bogdan_Temp\7_27_2019_IMR90RNA-Analysis-refitted\cell_dics\Conv_zscan_01__noncells.pkl',
           master_folder = r'\\dolly\Raw_data\Bogdan\7_27_2019_IMR90RNA',extra_pts=[],vmax=5,remove_first=0):
    
    dic_non_cells = np.load(fln_non_cells)
    R_to_htag = dic_non_cells['R_to_htag']

    htag_ = R_to_htag[R]
    htag,icol = htag_.split('__col')
    icol = int(icol)

    fov = cell.split('__cell')[0]
    dax_fl = master_folder+os.sep+htag+os.sep+fov+'.dax'
    num_cols=htag.count(',')+2
    
    #codes = [self.codes_nms[nm] for nm in rna_names]
    dic_ps = {}
    dic_cds = {}
    chrs = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']
    for ichr,ps_chr in enumerate(ps_pairs_final):
        for ireg,ps_reg in enumerate(ps_chr):
            for ihom,ps__ in enumerate(ps_reg):
                if len(ps__)>0:
                    key = chrs[ichr]+'_'+str(ireg+1)
                    Rs = self.tad_to_PR[key]
                    cd = tuple([int(R_[1:]) for R_ in Rs])
                    for iR,R_ in enumerate(Rs):
                        dic_ps[R_]=dic_ps.get(R_,[])+[ps__[iR]]
                        dic_cds[R_]=dic_cds.get(R_,[])+[cd]
    ps = np.array(dic_ps.get(R,[]))
    codes = dic_cds.get(R,[])
       
    drift =[0,0,0]
    if len(ps)>0:
        if len(ps[0])==18:
            coords = ps[:,12:15]
        else:
            coords = ps[:,13:16]

        drift = np.median(coords-ps[:,1:4],axis=0)
        
    coordsi = np.array([reg for chr_hom in dna_zxy for chr_r in chr_hom for reg in chr_r],dtype=float)+drift#coords.astype(int)
    pd=10
    
    zxym,zxyM = np.nanmin(coordsi,0).astype(int)-pd,np.nanmax(coordsi,0).astype(int)+pd
    zxym[zxym<0]=0
    
    #import workers_cells_v3 as wkc
    im = get_dax_im(dax_fl,icol,num_cols,zxym,zxyM,pad=11,remove_first=remove_first)


    plt.figure(figsize=(10,10))
    plt.title(htag_)
    plt.imshow(np.max(im,0),vmax=vmax,cmap='gray')
    if len(ps)>0:
        coords_ = coords-zxym
        plt.plot(coords_[:,2],coords_[:,1],'rx')
    for cd,zxy in zip(codes,coords_):
        plt.text(zxy[2],zxy[1],str(cd),color='w')
    for pts in extra_pts:
        pts_ = np.array(pts)-zxym+drift
        plt.plot(pts_[:,2],pts_[:,1],'x')
    #plt.imshow(np.max(im,1),vmax=5)
    #plt.plot(coords_[:,2],coords_[:,0],'rx')
def get_chromatic_ms(fl_750_647,fl_750_561):
    #load chromatic aberation files
    dic_chr = pickle.load(open(fl_750_647,'rb'))
    m1 = dic_chr['m_750_647'].copy()
    dic_chr = pickle.load(open(fl_750_561,'rb'))
    m2 = dic_chr['m_750_561'].copy()
    ms_chr = [None,m1,m2]
    return ms_chr
def pfits_to_htag(pfits_fl):
    return os.path.basename(pfits_fl).split('__')[0]
def pfits_to_col(pfits_fl):
    return int(os.path.basename(pfits_fl).split('_col')[-1].split('_')[0])
class cell_focus():
    def __init__(self,index_fov=0,dataset_tag='5_20_2019_GM_SI16_RNA',
                   parent_dapiSeg=r'\\mendel\Chromatin5',
                  parent_fits_analysis=r'E:\Bogdan_Temp',
                  parent_save = r'E:\Bogdan_Temp',
                  fl_750_647 = r'C:\Users\bogdan\Temp\6_1_2019_GM_SI14-Amanitin_repeat-Analysis\750_647_2col_chromatic.pkl',
                  fl_750_561 = r'C:\Users\bogdan\Temp\6_1_2019_GM_SI14-Amanitin_repeat-Analysis\750_561_3col_chromatic.pkl',
                  RNA_dapi_tag = 'H19B,B',
                  master_folder=None,
                  overwrite=False,refit=False):
        #info where to find the dapi segmentation
        self.refit = refit
        self.master_folder=master_folder
        self.overwrite=overwrite
        self.RNA_dapi_tag = RNA_dapi_tag
        self.fl_750_647 = fl_750_647
        self.fl_750_561 = fl_750_561
        self.analysis_folder_dapiseg = parent_dapiSeg
        self.im_dapiseg_fls = glob.glob(self.analysis_folder_dapiseg+r'\cnn_segmentation\*_imcells3d.npy')
        self.good_index = index_fov<len(self.im_dapiseg_fls)
        if self.good_index:
            self.cnn_dapi_fl = self.im_dapiseg_fls[index_fov]
            self.cnn_lims_fl = self.cnn_dapi_fl.replace('.npy','_lims.npy')
            fov = os.path.basename(self.cnn_dapi_fl).split('__imcells')[0]
            self.fov = fov
            #info where to find the fits
            analysis_folder_ = parent_fits_analysis#+os.sep+dataset_tag+'-Analysis'
            self.fits_folder = analysis_folder_+os.sep+'Fits'
            self.pfits_fls = glob.glob(self.fits_folder+os.sep+'*_'+fov+'__col*'+'__fits.npy')
            if len(self.pfits_fls)==0:
                folders = glob.glob(self.master_folder+os.sep+'H*')
                tags = [os.path.basename(fld) for fld in folders]
                tags = [tag for tag in tags if 'R' in tag or 'Q' in tag or 'r' in tag]# or 'U' in tag]
                self.pfits_fls = [self.fits_folder+os.sep+tag+'___'+fov+'__col'+str(icol)+'__fits.npy' for tag in tags 
                      for icol in range(tag.count(',')+1)]
            #info where to find the drifts
            self.drift_fls = [pfits_fl.replace('Fits','beads').split('__col')[0]+'___drift.npy' 
                        for pfits_fl in self.pfits_fls]
            #info for RNA_to_DNA dapi alignment
            
            self.prepostFISH_drift_file = None
            cand_fls = glob.glob(os.path.dirname(os.path.dirname(self.cnn_dapi_fl))+os.sep+'dapi_prepostFISH_drifts'+os.sep+'*'+fov+'.npy')
            if len(cand_fls)>0:
                self.prepostFISH_drift_file = cand_fls[0]
            
            #info where to save
            self.cell_folder = parent_save+os.sep+'cell_dics'
            if not os.path.exists(self.cell_folder): os.makedirs(self.cell_folder)
            self.cells_fl = self.cell_folder+os.sep+fov+'__cells.npy'
            self.noncells_fl = self.cell_folder+os.sep+fov+'__noncells.pkl'
            #where to store data
            self.dic_save = {'completed':{},'multi_ms':{},'m_invs':{}}
            self.dic_cells = {}
            self.dic_temp = {}
    def is_complete(self):
        if not self.good_index:
            return True
        
        if os.path.exists(self.noncells_fl):
            self.dic_save = pickle.load(open(self.noncells_fl,'rb'))
        is_complete_ = True
        for pfits_fl in self.pfits_fls:
            pfits_tag = os.path.basename(pfits_fl)
            if not self.dic_save['completed'].get(pfits_tag,False):
                is_complete_=False
        return is_complete_
    def load_previous(self):
        cells_fl,noncells_fl = self.cells_fl,self.noncells_fl
        if os.path.exists(cells_fl) and not self.overwrite: 
            self.dic_save = pickle.load(open(noncells_fl,'rb'))
            self.dic_cells = pf.load_double_dic(cells_fl)
    def drift_correct_prepost(self):
        if self.prepostFISH_drift_file is not None:
            prepostFISH_drift_file = self.prepostFISH_drift_file
            m_prepost,zxy2,zxy1,centers_ref,info_dapis = np.load(prepostFISH_drift_file)
            #fix degenerate z in pre-post fish
            zxy2_add = zxy2*[0,1,1]
            zxy2_add[:,0]=(zxy2-zxy1)[:,0]
            zxy2_ = np.concatenate([zxy2,zxy2_add])
            zxy1_ = np.concatenate([zxy1,zxy1*[0,1,1]])
            m_prepost = ft.calc_color_matrix(zxy1_,zxy2_,order=1,no_z=True,ransac=False)
            error = np.median(np.abs(ft.apply_colorcor(zxy2_,m_prepost)-zxy1_),0)
            print('DAPI pre-post accuracy:',error)
        else:
            m_prepost,error=None,None
        self.m_prepost = m_prepost
        self.dic_save['prepostDAPI'] = m_prepost,error
    def load_chromatic_ms(self):
        self.ms_chr = get_chromatic_ms(self.fl_750_647,self.fl_750_561)
        self.dic_save['ms_chr']=self.ms_chr
    def drift_correct_fov(self,drift_refs_DNA_RNA = [['R50','R60','R40','R80','R20'],['Q30','Q40','Q20','Q50','Q10']],plt_val=False):
        print("Performing drift correction")
        if 'prepostDAPI' not in self.dic_save: self.drift_correct_prepost()
        else: self.m_prepost,_ = self.dic_save['prepostDAPI']
        pfits_fls = self.pfits_fls
        drift_fls = self.drift_fls
        RNA_tag = self.RNA_dapi_tag 
        
        htags = np.unique([pfits_to_htag(fl) for fl in pfits_fls])
        dic_htags = pf.htags_to_Rs(htags)
        self.dic_save['htags_to_Rs']=dic_htags
        Rs = [dic_htags[pfits_to_htag(pfits_fl)][pfits_to_col(pfits_fl)%len(dic_htags[pfits_to_htag(pfits_fl)])] 
                for pfits_fl in pfits_fls]
        dic_zozn = {}
        self.dic_isRNA = {}
        for drift_fl,R in zip(drift_fls,Rs):
            [all_zn,all_zo,zns,zos,m,dax_fl_ref] = np.load(drift_fl)
            assert(all_zn.shape[-1]==4)
            z_cutoff = getattr(self,'z_cutoff',0)
            keep = (all_zn[:,0]>z_cutoff)&(all_zo[:,0]>z_cutoff)
            all_zn = all_zn[keep]
            all_zo = all_zo[keep]
            dic_zozn[R]=(all_zo,all_zn)
            #self.dic_isRNA[R]=RNA_tag in dax_fl_ref
            self.dic_isRNA[R]=RNA_tag in R
        m_refs = []
        for drift_refs in drift_refs_DNA_RNA:
            m_refs.append([])
            for R_ref in drift_refs:
                m_ref_0=None
                if R_ref in dic_zozn:
                    all_zo1,all_zn1 = dic_zozn[R_ref]
                    m_ref_0 = pf.robust_m(all_zo1,all_zn1, th=70, order=1, no_z=False, ransac=False)
                m_refs[-1].append(m_ref_0)
        dic_drift = {}
        allRs = np.sort(list(dic_zozn.keys()))[::-1]

        for R in tqdm(allRs):
            RNA_ = int(self.dic_isRNA[R])
            drift_refs = drift_refs_DNA_RNA[RNA_]
            m_refs_ = m_refs[RNA_]
            dic_drift[R] = []
            for m_ref,R_ref in zip(m_refs_,drift_refs):
                m=None
                if (R_ref in dic_zozn) and (R in dic_zozn):
                    all_zo1,all_zn1 = dic_zozn[R_ref]
                    all_zo2,all_zn2 = dic_zozn[R]
                    cs1,cs2 = pf.reduce_matching(all_zn1, all_zo1, all_zn2, all_zo2)
                    m = pf.robust_m(cs1,cs2, th=70, order=1, no_z=False, ransac=False)                
                dic_drift[R].append((m,m_ref))
        Ms_DNA_RNA=[[],[]]
        test_pts = np.array([[k,j,i] for i in range(0,2048,400) for j in range(0,2048,400) for k in range(0,60,10)])
        self.dic_test_points={}
        for R in allRs:
            test_pts_all = []
            for mm_refs in dic_drift[R]:
                m,m_ref = mm_refs
                test_pts_ = test_pts.copy()
                test_pts_ = ft.apply_colorcor(ft.apply_colorcor(test_pts_,m),m_ref)
                test_pts_all.append(test_pts_)
            self.dic_test_points[R] = test_pts_all
            
            M = np.array([[np.mean(np.abs(t1-t2),0) for t2 in test_pts_all] for t1 in test_pts_all])
            M = np.median(M,-1)
            RNA_ = int(self.dic_isRNA[R])
            Ms_DNA_RNA[RNA_].append(M)
        best_refs =[]
        self.best_refs_DNA_RNA = []
        #self.Ms_DNA_RNA = Ms_DNA_RNA
        for Ms_ in Ms_DNA_RNA:
            if len(Ms_)>0:
                M_ = np.median(Ms_,0)
                M_[M_==0]=np.inf
                best_pair = np.unravel_index(np.argmin(M_),M_.shape)
                best_refs.append(best_pair[0])
                self.best_refs_DNA_RNA.append(best_pair)
            else:
                self.best_refs_DNA_RNA.append((0,0))
        
        #Quantify drift errors
        self.drift_errors = {}
        for R in allRs:
            RNA_ = int(self.dic_isRNA[R])
            i1,i2 = self.best_refs_DNA_RNA[RNA_]
            test_pts_all = self.dic_test_points[R]
            t1,t2 = test_pts_all[i1],test_pts_all[i2]
            self.drift_errors[R] = np.median(np.abs(t1-t2),axis=0)
        
        #save
        dic_drift_final = {}
        for R in allRs:
            RNA_ = int(self.dic_isRNA[R])
            best_ref = best_refs[RNA_]
            m_prepost_ = self.m_prepost if not self.dic_isRNA[R] else None
            dic_drift_final[R] = list(dic_drift[R][best_ref])+[m_prepost_]
        self.dic_drift_final = dic_drift_final
        self.dic_drift = dic_drift
        self.dic_save['dic_drift_final']=dic_drift_final
        self.dic_save['dic_drift']=dic_drift
        self.dic_save['drift_errors']=self.drift_errors
        if plt_val:
            for Ms_ in Ms_DNA_RNA: 
                plt.figure()
                plt.imshow(np.median(Ms_,0),cmap='seismic')
                plt.colorbar()
            errors = np.array([self.drift_errors[R]for R in allRs])
            plt.figure()
            for error in errors.T:
                plt.plot(error,'o-')
            plt.xticks(np.arange(len(errors)),allRs,rotation='vertical')
    def remove_redundancy(self,ps):
        #remove redundancy - no longer needed in newer versions.
        from scipy.spatial import cKDTree
        tree = cKDTree(ps[:,1:4])
        bad_pairs = tree.query_pairs(1.,output_type='ndarray');
        keep = np.ones(len(ps),dtype=bool)
        keep[bad_pairs[:,0]]=False
        return ps[keep]
    def deal_with_cells(self,nkeep = 125,delta_dapi=4,rescale_xy=2):
        #Load dapi segmentation if not already loaded
        if 'imcells' not in self.dic_temp: 
            imcells = np.load(self.cnn_dapi_fl)
            cell_lims = np.load(self.cnn_lims_fl)
        else: 
            imcells=self.dic_temp['imcells']
            cell_lims=self.dic_temp['cell_lims']
        
        
        for icl in range(len(cell_lims)):
            cell_tag = self.fov+'__cell'+str(icl)
            zmin,zmax,ymin,ymax,xmin,xmax = cell_lims[icl]
            #zmin,zmax = 0,100#imcells.shape[0]####added
            #process the dapi images for inside and other cells only once######
            if cell_tag not in self.dic_temp:
                imred = imcells[zmin:zmax,ymin:ymax,xmin:xmax].swapaxes(1,2)
                #compute cell info based on dapi segmentation
                #outputs: imcell/imothercells of real shape, from 0->1  and 
                imcell = imred==(icl+1)
                imothercells = (imred>0).astype(np.uint8)-imcell
                imcellgrad = imcell.astype(np.float32)
                delta = delta_dapi
                for it in range(delta):
                    imcellnext = ndi.morphology.binary_dilation(imcell,iterations=1)
                    imcellgrad[imcell!=imcellnext]=np.exp(-(it+1)**2/2./(delta/2)**2)
                    imcell = imcellnext
                imcell = ndi.zoom(imcellgrad,[1,rescale_xy,rescale_xy],order=0)

                imothercellsgrad = imothercells.astype(np.float32)
                for it in range(delta):
                    imcellnext = ndi.morphology.binary_dilation(imothercells,iterations=1)
                    imothercellsgrad[imothercells!=imcellnext]=np.exp(-(it+1)**2/2./(delta/2)**2)
                    imothercells = imcellnext
                imothercells = ndi.zoom(imothercellsgrad,[1,rescale_xy,rescale_xy],order=0)
                self.dic_temp[cell_tag]=(imcell,imothercells)
            else:
                imcell,imothercells=self.dic_temp[cell_tag]
            
            
            #Keep only the points in the cell ##########################
            zxy_min = (np.array([zmin,xmin*rescale_xy,ymin*rescale_xy]))
            
            zxy_max = (zxy_min+imcell.shape)
            zxy_max[0]=100################################hack
            if self.refit:
                try:
                    #get the minimum/ maximum box capturing the cell zxym,zxyM 
                    zxy_min_,zxy_max_ = ft.apply_colorcor(np.array([zxy_min,zxy_max]),self.inv_m)
                    zxy_min_,zxy_max_ = zxy_min_.astype(int),zxy_max_.astype(int)
                    zxym,zxyM=zxy_min_.copy(),zxy_max_.copy()
                    zxym[zxym<0] = 0
                
                    #get the image info and load the image
                    self.dax_fl = self.master_folder+os.sep+self.htag+os.sep+self.fov+'.dax'
                    self.num_cols=self.htag.count(',')+2
                    pad = getattr(self,'pad',10)
                    self.remove_first = getattr(self,'remove_first',0)
                    #self.icol=0
                    im = get_dax_im(self.dax_fl,self.icol,self.num_cols,zxym,zxyM,pad=pad,im_meds = self.im_meds,remove_first=self.remove_first)#pisici
                    im_cpy = im.copy().astype(np.float32)
                    #print(np.mean(im_cpy))
                    im_cpy = im_cpy-np.min(im_cpy)
                    im = np.array([im__/np.median(im__) for im__ in im_cpy])
                    
                    real_zmax = im.shape[0]
                    imcell = np.array([imcell[0]]*real_zmax)################################hack
                    imothercells = np.array([imothercells[0]]*real_zmax)################################hack
                    # get candidate seeds for the image
                    centers_zxy = get_standard_seeds(im,th_stds = 4,sz_blur=20,npts=None)
                    ind_array = (centers_zxy-zxy_min_+zxym).astype(int)
                    keep = np.all(ind_array<np.array(imcell.shape),axis=-1)
                    centers_zxy,ind_array = centers_zxy[keep],ind_array[keep]
                    psk_final=[]
                    #print len(centers_zxy)
                    if len(centers_zxy):
                        cell_score = imcell[ind_array[:,0],ind_array[:,1],ind_array[:,2]]
                        other_cell_score = imothercells[ind_array[:,0],ind_array[:,1],ind_array[:,2]]
                        keep = (cell_score>0)#&(other_cell_score<cell_score)   
                        cell_score = cell_score[keep,np.newaxis][:nkeep]
                        other_cell_score = other_cell_score[keep,np.newaxis][:nkeep]
                        centers_zxy = centers_zxy[keep][:nkeep]
                        if len(centers_zxy):
                            psk = ft.fast_fit_big_image(im,centers_zxy,radius_fit = 4,avoid_neigbors=True,
                                          recenter=False,verbose = False,better_fit=True,troubleshoot=False)
                            psk[:,1:4] += zxym
                            psk = np.concatenate([psk,psk[:,:4],[[self.icol]]*len(psk),cell_score,other_cell_score],axis=-1)
                            for m in self.ms:
                                psk[:,1:4] = ft.apply_colorcor(psk[:,1:4],m)
                            psk_final = psk
                    if False:
                        ax=1
                        for ax in [0,1]:
                            fig = plt.figure()#figsize=(15,15))
                            plt.title(self.htag)
                            if len(psk_final)>0:
                                #print psk_final[0,:]
                                zxy = psk_final[:,12:15]-zxym
                                #print zxy
                                
                                plt.plot(zxy[:,2],zxy[:,1-ax],'gx')
                            plt.imshow(np.max(im,axis=ax),vmax=np.percentile(im,99.9),cmap='gray')
                            plt.show()
                except:
                    psk_final=[]
                
            #if do not refit then used the fits already computed.
            else:
                psc = self.psc
                zxy_drft = psc[:,1:4]
                keep = np.all((zxy_drft>zxy_min)&(zxy_drft<zxy_max),-1)
                psk = psc[keep]
                psk_final = psk
                if len(psk)>0:
                    #keep within cell and compute other cell score
                    inds_array = (psk[:,1:4]-zxy_min).astype(int)
                    cell_score = imcell[inds_array[:,0],inds_array[:,1],inds_array[:,2]]
                    other_cell_score = imothercells[inds_array[:,0],inds_array[:,1],inds_array[:,2]]
                    keep = (cell_score>0)#&(other_cell_score<cell_score)
                    psk_final = np.concatenate([psk[keep],cell_score[keep,np.newaxis],other_cell_score[keep,np.newaxis]],axis=-1)
                    if len(psk_final)>0: 
                        keep = np.argsort(psk_final[:,0])[::-1][:nkeep]
                        psk_final = psk_final[keep]
            if cell_tag not in self.dic_cells: self.dic_cells[cell_tag]={}
            self.dic_cells[cell_tag][self.R]=psk_final
            
    def apply_to_pfits(self):
        #make sure the chromatic and drift corrections are computed
        
        if True:#'htags_to_Rs' not in self.dic_save:######HACK
            htags = np.unique([pfits_to_htag(fl) for fl in self.pfits_fls])
            dic_htags = pf.htags_to_Rs(htags)
            self.dic_save['htags_to_Rs']=dic_htags
        else: dic_htags = self.dic_save['htags_to_Rs']
        if 'R_to_htag' not in self.dic_save: self.dic_save['R_to_htag']={}
        Rs = [elem for key in self.dic_save['htags_to_Rs'] for elem in self.dic_save['htags_to_Rs'][key]]
        
        if 'ms_chr' not in self.dic_save: self.load_chromatic_ms()
        else: self.ms_chr = self.dic_save['ms_chr']
        if 'dic_drift_final' not in self.dic_save: 
            self.drift_correct_fov()
        else: 
            if np.any([R not in self.dic_save['dic_drift_final'] for R in Rs]):
                self.drift_correct_fov()
            else:
                self.dic_drift_final = self.dic_save['dic_drift_final']
        
        htags = np.unique([pfits_to_htag(fl) for fl in self.pfits_fls])
        dic_htags = pf.htags_to_Rs(htags)

        
        for pfits_fl in tqdm(self.pfits_fls):
            pfits_tag = os.path.basename(pfits_fl)
            #Extract information from the filename
            htag = pfits_to_htag(pfits_fl)
            icol = pfits_to_col(pfits_fl)
            R = dic_htags[htag][icol%len(dic_htags[htag])]
            self.R,self.htag,self.icol=R,htag,icol
            self.dic_save['R_to_htag'][R]=htag+'__col'+str(icol)
            if not self.dic_save['completed'].get(pfits_tag,False):
                m_chr = self.ms_chr[icol]
                ms_drift = self.dic_drift_final[R]
                ms = [m_chr]+list(ms_drift)
                self.ms = ms
                self.inv_m = ft.calc_inverse_m(ms)
                self.dic_save['multi_ms'][R] = ms
                self.dic_save['m_invs'][R] = None
                
                if not self.refit:
                    ps = np.load(pfits_fl)
                    if len(ps)>0:
                        #ps = self.remove_redundancy(ps)
                        psc = ps.copy()
                        psc = np.concatenate([psc,psc[:,:4],[[icol]]*len(psc)],axis=-1)
                        for m in ms:
                            psc[:,1:4] = ft.apply_colorcor(psc[:,1:4],m)
                        z_cutoff = getattr(self,'z_cutoff',0)
                        z_max = getattr(self,'z_max',np.max(ps[:,1]))
                        psc = psc[(ps[:,1]>z_cutoff)&(ps[:,1]<(z_max-z_cutoff))]
                        self.psc = psc
                else:
                    fl = os.path.dirname(self.fits_folder)+os.sep+'im_meds.npy'
                    self.im_meds = np.load(fl)
                self.deal_with_cells()
                self.dic_save['completed'][pfits_tag]=True
    def save(self):
        pf.save_double_dic(self.dic_cells,self.cells_fl)
        pickle.dump(self.dic_save,open(self.noncells_fl,'wb'))