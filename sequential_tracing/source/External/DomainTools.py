#Bogdan Bintu and Pu Zheng
#Copyright Presidents and Fellows of Harvard College, 2017.

#external packages
import sys,os,glob
import numpy as np

import pickle as pickle

import matplotlib.pyplot as plt

## Plotting tools
def fig_no_axis(**kwargs):
    """Get figure with no right/upper axis and which allows for saving text in pdf and larger font."""
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['font.size']=22
    fig1, ax1 = plt.subplots(facecolor='white',**kwargs)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()
    return fig1
## Tad insulation tools defined as in Nagano et al., Nature, 2017
def get_b(i,j):
    """returns the in-tad coordinates"""
    i,j = np.sort([i,j])
    bx,by=[],[]
    for y_ in range(i,j+1):
        for x_ in range(i,y_):
            bx.append(x_)
            by.append(y_)
    return bx,by
def get_a(i,j,k):
    """returns between tad coordinates"""
    i,j,k = np.sort([i,j,k])
    ax,ay=[],[]
    for x_ in range(i,j+1):
        for y_ in range(j+1,k+1):
            ax.append(x_)
            ay.append(y_)
    return ax,ay

def insulation(i,j,k,mat,func = np.nanmedian,return_dists=False):
    """Given 3 coordinates for the boundaries of two domains i,j,k and a distance matrix
    Return the boundary score (as defined in  Nagano et al., Nature, 2017) and the distributions of mat_ within each domains and beween domains"""
    mat_ = np.array(mat,dtype=float)
    xmax,ymax = mat_.shape
    max_ = min(xmax,ymax)-1
    i,j,k = min(i,max_),min(j,max_),min(k,max_)
    b1x,b1y = get_b(i,j-1)
    b2x,b2y = get_b(j,k-1)
    ax,ay = get_a(i,j-1,k-1)
    b1 = mat_[b1x,b1y]
    b2 = mat_[b2x,b2y]
    a = mat_[ax,ay]
    B1 = func(b1)
    B2 = func(b2)
    A = func(a)
    score = np.log(1+A/B1+A/B2)/np.log(2)
    if return_dists:
        return score,b1,b2,a
    return score
##
from scipy.ndimage.filters import maximum_filter,minimum_filter,median_filter,gaussian_filter
from scipy.spatial.distance import cdist,pdist,squareform
from scipy.optimize import leastsq
def get_ind_loc_max(dists,cutoff_max=1.,valley=3, remove_edge=True, make_plot=False):
    """get local maximum within valley size bigger than cut-off"""
    _local_max_inds = []
    for _id in range(len(dists)):
        l = np.max([0,_id-valley])
        r = np.min([len(dists),_id+valley])
        if dists[_id]==np.nanmax(dists[l:r]) and dists[_id]>cutoff_max:
            if remove_edge:
                if _id != 0 and _id !=len(dists)-1:
                    _local_max_inds.append(_id)
            else:
                _local_max_inds.append(_id)
    if make_plot:
        plt.figure()
        plt.plot(np.arange(len(dists)), dists)
        plt.plot(_local_max_inds, np.array(dists)[_local_max_inds], 'ro')
        plt.show()
    return np.array(_local_max_inds, dtype=np.int)

def get_boundaries_old(im,su=5,sl=5,valley=5,cutoff_max=1.,plt_val=False):
    """Bintu et al 2018 candidate boundary calling"""
    im_=np.array(im)
    ratio,ration,center,centern=[],[],[],[]
    for i in range(len(im)):
        x_im_l,y_im_l = [],[]
        x_im_r,y_im_r = [],[]

        xn_im_l,yn_im_l = [],[]
        xn_im_r,yn_im_r = [],[]

        for j in range(sl):
            xn_im_l.extend(i+j-np.arange(su)-1)
            yn_im_l.extend([i+j]*su)
            xn_im_r.extend(i+j+sl-np.arange(su)-1)
            yn_im_r.extend([i+j+sl]*su)
            
            x_im_l.extend(i+j+np.arange(su)+1)
            y_im_l.extend([i+j]*su)
            x_im_r.extend(i+j+sl+np.arange(su)+1)
            y_im_r.extend([i+j+sl]*su)
            

        x_im_l,y_im_l,x_im_r,y_im_r = list(map(np.array,[x_im_l,y_im_l,x_im_r,y_im_r]))
        xn_im_l,yn_im_l,xn_im_r,yn_im_r = list(map(np.array,[xn_im_l,yn_im_l,xn_im_r,yn_im_r]))

        in_image = np.all(x_im_l>=0) and np.all(x_im_r>=0) and np.all(y_im_l>=0) and np.all(y_im_r>=0)
        in_image = in_image and np.all(x_im_l<len(im)) and np.all(x_im_r<len(im)) and np.all(y_im_l<len(im)) and np.all(y_im_r<len(im))

        in_imagen = np.all(xn_im_l>=0) and np.all(xn_im_r>=0) and np.all(yn_im_l>=0) and np.all(yn_im_r>=0)
        in_imagen = in_imagen and np.all(xn_im_l<len(im)) and np.all(xn_im_r<len(im)) and np.all(yn_im_l<len(im)) and np.all(yn_im_r<len(im))
        if in_image:
            val_l,val_r = np.nanmean(im_[x_im_l,y_im_l]),np.nanmean(im_[x_im_r,y_im_r])
            ratio.append(val_l/val_r)
            center.append(i+sl)
        if in_imagen:
            val_l,val_r = np.nanmean(im_[xn_im_l,yn_im_l]),np.nanmean(im_[xn_im_r,yn_im_r])
            ration.append(val_r/val_l)
            centern.append(i+sl)
        if False:#i==9:
            plt.figure(figsize=(20,20))
            plt.plot(xn_im_l,yn_im_l,'mo')
            plt.plot(xn_im_r,yn_im_r,'go')
            plt.plot(x_im_l,y_im_l,'ro')
            plt.plot(x_im_r,y_im_r,'bo')
            plt.imshow(im,interpolation='nearest',cmap='seismic_r',vmax=1000)
            plt.show()
            print(x_im_l,y_im_l,x_im_r,y_im_r)

    center,ratio=np.array(center),np.array(ratio)
    centern,ration=np.array(centern),np.array(ration)
    max_ratio = np.zeros(len(im))+np.nan
    max_ratio[center]=ratio
    max_ratio[centern]=np.nanmax([max_ratio[centern],ration],axis=0)
    
    local_max_good = get_ind_loc_max(max_ratio,cutoff_max=cutoff_max,valley=valley)
    #local_max_goodn = get_ind_loc_max(ration,cutoff_max=cutoff_max,valley=valley)
    ###Plotting
    if plt_val:
        #plt.close('all')
        plt.figure(figsize=(12,7))
        #print local_max_good,local_max_goodn
        plt.plot(center,np.log(ratio),'o-')
        plt.plot(centern,np.log(ration),'o-')
        plt.plot(np.log(max_ratio),'k-')
        if len(local_max_good)>0:
            plt.plot(local_max_good,np.log(max_ratio[local_max_good]),'o')
        
        plt.show()
        fig, ax = plt.subplots(figsize=(12,7))
        if len(local_max_good)>0:
                ax.plot(local_max_good[:],local_max_good[:],'go',ms=10,mec='k',mew=2)
        #cax = ax.imshow(set_diag(img,np.nanmax(img)),interpolation='nearest',cmap='bwr')#,vmax=1000,vmin=0)
        cax = ax.imshow(im,interpolation='nearest',cmap='seismic_r',vmax=1000,vmin=0)
        cbar = fig.colorbar(cax)
        plt.show()
    return local_max_good,max_ratio[local_max_good]
def fuse_doms(mat,dom_starts,tag='median',cut_off=1, hard_cutoff=2, 
              use_local=True, min_dom_sz=5):
    from ..domain_tools.distance import domain_neighboring_dists, domain_neighboring_stats

    dom_starts = np.array(dom_starts, dtype=np.int)
    dom_ends = np.zeros(np.shape(dom_starts))
    dom_ends[:-1] = dom_starts[1:]
    dom_ends[-1] = len(mat)
    # calculate initial seps
    #init_seps = np.array(calc_seps(mat,dom_starts,func=tag,plt_val=False))
    init_seps = domain_neighboring_dists(mat, dom_starts, metric=tag, 
                                         use_local=False, min_dom_sz=min_dom_sz)
    old_seps = np.copy(init_seps)
    # use initial seps to determine changable boundaries
    changeable_labels = (init_seps <= hard_cutoff)
    dom_size_check = (dom_ends-dom_starts) <= min_dom_sz
    dom_size_check = dom_size_check[1:] + dom_size_check[:-1]
    changeable_labels += dom_size_check
                        
    # now iterate until no seps can be modified
    while len(dom_starts)>1:
        #seps = calc_seps(mat,dom_starts,func=tag,plt_val=False)
        seps = domain_neighboring_dists(mat, dom_starts, metric=tag, 
                                        use_local=use_local, min_dom_sz=min_dom_sz)
        # this step gives seps some buffer zone
        seps = (seps +  old_seps) / 2
        
        #print(calc_seps(mat,dom_starts,func=tag,plt_val=False))
        #print(changeable_labels)
        
        # keep very good seps in this iteration
        #changeable_labels *= (seps <= hard_cutoff ) 
        # changeable seps
        ch_seps = np.array(seps)[changeable_labels]
        if len(ch_seps) == 0:
            break
        #imin = np.argmin(seps)
        if np.min(ch_seps) < cut_off:
            # remove the boundary with highest pvalue
            sep_stats, sep_pvals = domain_neighboring_stats(mat, dom_starts, method='ttest', 
                                        use_local=use_local, min_dom_sz=min_dom_sz,
                                        return_pval=True)
            #remove_inds = np.where(sep_stats==np.min(sep_stats))[0]

            remove_inds = np.where(sep_pvals==np.max(sep_pvals))[0]
            # delete elements in dom_starts
            if 0 in dom_starts:
                dom_starts = np.delete(dom_starts, remove_inds+1)
            else:
                dom_starts = np.delete(dom_starts, remove_inds)
            # delete elements in changeable_labels
            changeable_labels = np.delete(changeable_labels, remove_inds)
            # delete old seps
            old_seps = np.delete(seps.copy(), remove_inds)
            sep_pvals = np.delete(sep_pvals, remove_inds)
        else:
            #print(seps)
            #print(-np.log(sep_pvals))
            #print(len(seps), len(sep_pvals))
            break
    return dom_starts,seps
def calc_seps(mat,dom_starts,func = 'mean',plt_val=False):
    dom_edges = list(dom_starts)+[len(mat)]
    ijks = [dom_edges[i:i+3]for i in range(len(dom_edges)-2)]
    seps = []
    for i,j,k in ijks:
        score,b1,b2,a = insulation(i,j,k,mat,func = np.nanmean,return_dists=True)
        b = np.concatenate([b1,b2])
        if func is 'mean':
            sep_score = (np.nanmedian(a)-np.nanmedian(b))/np.sqrt(np.nanvar(a)+np.nanvar(b))
        if func is 'median':
            am,bm = np.nanmedian(a),np.nanmedian(b)
            ad,bd = np.nanmedian((a-am)**2),np.nanmedian((b-bm)**2)
            sep_score = (am-bm)/np.sqrt(ad+bd)
        if func is 'median2':
            am,bm = np.nanmedian(a),np.nanmedian(b)
            ad,bd = np.nanmedian(np.abs(a-am)),np.nanmedian(np.abs(b-bm))
            sep_score = (am-bm)/float(ad+bd)
        seps.append(sep_score)
        if plt_val:
            plt.figure()
            plt.title(str(sep_score))
            plt.hist(a[np.isnan(a)==False],alpha=0.75,color = 'm')
            plt.hist(b[np.isnan(b)==False],alpha=0.75,color = 'g')
            plt.show()
    return seps
def plot_dom(zxy,dom_starts,starters=None):
    dom_edges = list(dom_starts)+[len(zxy)]
    zxy_ = interpolate_chr(zxy,gaussian=0)
    nan = np.where(np.isnan(zxy[:,0]))[0]
    mat = squareform(pdist(zxy_))
    mat[nan,nan]=np.nan
    plt.figure()
    for i in range(len(dom_edges)-1):
        e1,e2 = dom_edges[i],dom_edges[i+1]
        plt.plot([e1,e1,e2],[e1,e2,e2],'green',lw=3)
    if starters is not None:
        plt.plot(starters,starters,'go',ms=10,mew=2,mec='k')
    plt.imshow(mat,cmap='seismic_r',vmax=1000)
    plt.colorbar()
    #plt.show()
def standard_domain_calling_old(zxy,gaussian=None,su=7,sl=4,valley=4,dom_sz=5,cutoff_max=1):
    zxy_ = np.array(zxy)
    if gaussian is not None:
        zxy_ = interpolate_chr(zxy_,gaussian=gaussian)
    
    mat = squareform(pdist(zxy))
    bds_candidates,scores = get_boundaries_old(mat,su=su,sl=sl,valley=valley,
                                       cutoff_max=cutoff_max,plt_val=False)

    dom_starts= [0]+[dm for dm in bds_candidates 
                     if dm>int(dom_sz/2) and dm<len(zxy_)-int(dom_sz/2)]
    dom_starts,seps = fuse_doms(mat,dom_starts,tag='median',cut_off=1.)
    
    return dom_starts
    

def standard_domain_calling_new(zxy, gaussian=None, 
                                metric='median', dom_sz=5, 
                                cutoff_max=1., hard_cutoff=2., 
                                use_local=True, remove_edge=True):
    zxy_ = np.array(zxy)
    if gaussian is not None:
        zxy_ = interpolate_chr(zxy_,gaussian=gaussian)

    dists = []
    for i in range(len(zxy_)):
        if i >= int(dom_sz/2) and i < len(zxy_)-int(dom_sz/2):
            cm1 = np.nanmean(zxy_[max(i-dom_sz, 0):i], axis=0)
            cm2 = np.nanmean(zxy_[i:min(i+dom_sz, len(zxy_))], axis=0)
            dist = np.linalg.norm(cm1-cm2)
            dists.append(dist)
    
    bds_candidates = get_ind_loc_max(dists,cutoff_max=0,
                                     valley=dom_sz, remove_edge=remove_edge) + int(dom_sz/2)
    
    mat = squareform(pdist(zxy))
    
    dom_starts= [0]+[dm for dm in bds_candidates 
                     if dm>int(dom_sz/2) and dm<len(zxy_)-int(dom_sz/2)]
    dom_starts, seps = fuse_doms(mat, dom_starts, tag=metric, 
                                 cut_off=cutoff_max, hard_cutoff=hard_cutoff,
                                use_local=use_local, min_dom_sz=dom_sz+3)

    return dom_starts



def nan_gaussian_filter(mat,sigma,keep_nan=False):
    from scipy.ndimage import gaussian_filter
    U=np.array(mat)
    Unan = np.isnan(U)
    V=U.copy()
    V[U!=U]=0
    VV=gaussian_filter(V,sigma=sigma)

    W=0*U.copy()+1
    W[U!=U]=0
    WW=gaussian_filter(W,sigma=sigma)

    Z=VV/WW
    if keep_nan:
        Z[Unan]=np.nan
    return Z
def interp1dnan(A):
    A_=np.array(A)
    ok = np.isnan(A)==False
    xp = ok.nonzero()[0]
    fp = A[ok]
    x  = np.isnan(A).nonzero()[0]
    A_[np.isnan(A)] = np.interp(x, xp, fp)
    return A_
def chromosome_segment_RG(_chr, _group):
    """Calculate radius of gyration given chr coordinates and selected segment group"""
    _segment = _chr[_group]
    return np.nanmean(np.nanvar(_segment, 0))

def best_combine_step(_chr, _groups):
    """Find the combination to minimize radius of gyration in the combination"""
    _combined_groups = [_groups[_i]+_groups[_i+1] for _i in range(len(_groups)-1)]
    _combined_grs = [chromosome_segment_RG(_chr, _g) \
                     for _i,_g in enumerate(_combined_groups)]
    _combined_id = np.argmin(_combined_grs)
    _new_groups = list(_groups)
    _new_groups[_combined_id] = _combined_groups[_combined_id]
    _new_groups.pop(_combined_id+1)
    return _new_groups
    
def interpolate_chr(_chr,gaussian=0):
    """linear interpolate chromosome coordinates"""
    _chr = np.array(_chr).copy()
    for i in range(_chr.shape[-1]):
        if gaussian > 0:
            _chr[:, i] = nan_gaussian_filter(_chr[:, i], gaussian)
    # interpolate
    from scipy.interpolate import interp1d
    not_nan_inds = np.where(np.isnan(_chr).sum(1) == 0)[0]
    if len(not_nan_inds) == 0:
        return _chr
    else:
        f = interp1d(np.arange(len(_chr))[not_nan_inds], _chr[not_nan_inds], 
                    kind='linear', axis=0, bounds_error=False, 
                    fill_value='extrapolate')
        _interpolated_chr = f(np.arange(len(_chr)))
        return _interpolated_chr

def Radius_of_Gyration_Segmentation(chromosome,interpolate=True,gaussian=0):
    """get segmentation info from radius of gyration information
    (Adpoted from Tan et al., 2018, Science)
    Inputs:
        chromosome: 3d positions of a chromosome"""
    _chr = np.array(chromosome)
    if interpolate:
        _chr = interpolate_chr(chromosome,gaussian=gaussian)
    ## initialize each segment as a group
    _groups = [[_i] for _i in np.arange(len(_chr))]
    _group_in_steps = [_groups]
    while len(_groups) > 1:
        _groups = best_combine_step(_chr, _groups)
        _group_in_steps.append(_groups)
    #_splits = [[_g[0] for _g in _gs] for _gs in _group_in_steps]
    return _group_in_steps
def Generate_ArrowHead(zxy, _make_plot=True, _normalization=False, _scale=[200,1000], _gaussian_size=0):
    """Function to transfer normal distance map to arrow head map
    Inputs:
        distmap: n-by-n array for pair-wise distance, 2d-array
        _make_plot: whether make plot for arrowhead result, bool
        _normalization: whether do normalization for orignial iamage, bool
        _scale: if _normalization is given, give lower and upper limits which will be transformed into 0 and 1, list of 2
        _gaussian_size: sigma for gaussian blur the original distance map, int > 0
    Outputs:
        _arrowmap: arrowhead map for given distance map, 2d-array"""
    distmap = squareform(pdist(zxy))
    _distmap_shape = distmap.shape
    _arrowmap = np.zeros(_distmap_shape)
    ## normalization
    if _normalization:
        _normed_map = (distmap-min(_scale))/(max(_scale)-min(_scale))
        _normed_map[_normed_map>1] = 1
        _normed_map[_normed_map<0] = 0
    else:
        _normed_map = distmap
    ## gaussian convolve
    if _gaussian_size > 0:
        pass
    ## Do arrowhead transformation here
    for i in range(_distmap_shape[0]):
        for j in range(1, _distmap_shape[1]-i):
            _arrowmap[i, i+j] = (_normed_map[i,i-j]-_normed_map[i,i+j])/(_normed_map[i,i-j]+_normed_map[i,i+j])
            _arrowmap[i+j, i] = _arrowmap[i, i+j]
    if _gaussian_size > 0:
        #_arrowmap = convolve(_arrowmap, _kernel)
        pass

    if _make_plot:
        plt.figure()
        plt.imshow(_normed_map, cmap='seismic_r', vmin=0, vmax=1000)
        plt.colorbar()
        plt.title("input distance map")
        plt.show()
        plt.figure()
        plt.imshow(_arrowmap, cmap='seismic_r')
        plt.colorbar()
        plt.title("Arrowhead")
        plt.show()
    return _arrowmap
    
def ArrowHead_Feature(_arrowmap, _make_plot=True, _verbose=True):
    """From Arrowhead map generate three feature matrices
    Inputs:
        _arrowmap: input arrowhead map, 2d-array
        _make_plot: whether make plot for arrowhead result, bool
        _verbose: say something!, bool
    Outputs:
        _S_sign: sum of signs of upper triangle - lower triangle, 2d matrix
        _S_sum: sum of values of upper triangle - lower triangle, 2d matrix
        _S_var: variance among values in upper and lower triangle, 2d matrix"""
    # get shape
    _arrow_shape = np.shape(_arrowmap)
    _dim = _arrow_shape[0]
    # initialize three feature matrices
    _S_sign = np.zeros(_arrow_shape)
    _S_sum = np.zeros(_arrow_shape)
    _S_var = np.zeros(_arrow_shape)
    # loop through entries
    for _i in range(1, _dim):
        #for _j in range(_i+1, min(int(np.ceil((_dim+2*_i)/3)), int(_i*2))):
        for _j in range(_i+1, _dim):
            _crop_dim = _j - _i
            # get limits
            _ulim = max(_i-_crop_dim,0) # upper and left limits
            _rlim = min(_j+2*_crop_dim, _dim) # right limit
            # if not cropped as a whole, crop again:
            if _j-_ulim != 2*_crop_dim or _rlim-_ulim != 4*_crop_dim:
                _crop_dim = min(_i, int((_dim-_j)/2))
                if _crop_dim < 1:
                    continue
                else:
                    _crop = np.copy(_arrowmap[_i-_crop_dim:_i+_crop_dim, _j-2*_crop_dim:_j+2*_crop_dim])
            else:
                # crop feature triangles
                _crop = np.copy(_arrowmap[_ulim:_j, _ulim:_rlim])
            for _c in range(2*_crop_dim):
                _crop[np.ceil(_c/2).astype(np.int):,_c] = np.nan # remove lower-left triangle
                _crop[:_crop_dim+int((_c+1)/2), _c+_crop.shape[0]] = np.nan # remote upper-right triangle
            # get sign sum var for this (i,j) pair
            _sign = np.nansum(_crop[:, :_crop.shape[0]]>0) - np.nansum(_crop[:, :_crop.shape[0]]<0) \
                    - np.nansum(_crop[:, _crop.shape[0]:]>0) + np.nansum(_crop[:, _crop.shape[0]:]<0)
            _sum = np.nansum(_crop[:, :_crop.shape[0]]) - np.nansum(_crop[:, _crop.shape[0]:])
            _num_elem = _crop[:, :_crop.shape[0]]
            _var = np.nanvar(_crop)
            # save
            _S_sign[_i,_j] = _sign
            _S_sum[_i,_j] = _sum
            _S_var[_i,_j] = _var
            _S_sign[_j,_i] = _S_sign[_i,_j]
            _S_sum[_j,_i] = _S_sum[_i,_j] 
            _S_var[_j,_i] = _S_var[_i,_j]
            
    if _make_plot:
        plt.figure()
        plt.imshow(_S_sign, cmap='seismic')
        plt.colorbar()
        plt.title("sign")
        plt.show()
        plt.figure()
        plt.imshow(_S_sum, cmap='seismic')
        plt.colorbar()
        plt.title("sum")
        plt.show()
        plt.figure()
        plt.imshow(_S_var, cmap='seismic')
        plt.colorbar()
        plt.title("var")
        plt.show()
                
    return _S_sign, _S_sum, _S_var
    
def Find_Boundaries(distmap, S_features, gaussian_size=0.25, lower_ind_thres=-5, make_plot=True):
    """Primary algorithm to find domain boundaries
    Inputs:
        distmap: distance map for a chromosome, 2d-array
        S_features: tuple or list of features, list or tuple of 2d-array
        gaussian_size: sigma for gaussian filter applied to features to better call local maximum, float
        lower_ind_thres: lower boundary for accepted indices along off-diagonal lines, int
        make_plot: whether make plots, bool
    Outputs:
        selected_pk_coords: selected peaks in feature maps, which corresponds to domain boundaries, 1d-array
    """
    from scipy.ndimage.interpolation import map_coordinates
    from scipy.signal import find_peaks
    #from astropy.convolution import Gaussian2DKernel,convolve
    
    dim = np.shape(distmap)[0]
    # genrate coordinates for line i+x, i+x/2 which arrow edges align:
    start_ind = np.arange(-int(dim/2),dim)
    coord_list = [np.stack([np.arange(np.abs(i),dim), max(0,i)/2+np.arange(max(0,i),dim+min(0,i))/2]) for i in start_ind] 
    # set gaussian kernel
    #kernel = Gaussian2DKernel(x_stddev=gaussian_size)
    # initialize feature ids
    feature_list = []
    for feature_id in range(2):
        # gaussian filter this map
        if gaussian_size:
            feature_map = convolve(S_features[feature_id], kernel)
        else:
            feature_map = S_features[feature_id]
        # extract arrow lines
        arrow_lines = [map_coordinates(feature_map, _coords) for _coords in coord_list]
        # calculate mean to find local maximum
        arrow_line_means = np.array([np.mean(arrline) for arrline in arrow_lines])
        # calculate peaks for meean behavior line 
        feature_line_ids = find_peaks(arrow_line_means, distance=3, width=2)[0] # this step is better to be more rigorious
        feature_line_ids = feature_line_ids[start_ind[feature_line_ids]>lower_ind_thres]
        feature_list.append(feature_line_ids)
        # plot selected lines
        #plt.figure()
        #plt.plot(start_ind, arrow_line_means)
        #plt.plot(start_ind[feature_line_ids], arrow_line_means[feature_line_ids], 'ro')
        #plt.show()
    # select shared feature_ids
    selected_ids = []
    for _id in feature_list[0]:
        if sum(np.abs(feature_list[1]-_id) <= 1) > 0:
            _local_ids = feature_list[1][np.abs(feature_list[1]-_id) <= 1]
            _local_ids = np.concatenate([[_id], _local_ids])
            selected_ids.append(np.min(_local_ids))
    selected_ids = np.array(selected_ids)
    if len(selected_ids) == 0:
        return np.array([])
    # selected ids plus +-1 lines
    feature_map = convolve(S_features[1], kernel)
    selected_coords = [coord_list[_i] for _i in np.unique([selected_ids, selected_ids-1, selected_ids+1])]
    selected_lines = [map_coordinates(feature_map, _coords) for _coords in selected_coords]
    # call peaks
    pks = [find_peaks(_line, distance=2, width=2)[0] for _line in selected_lines]
    pk_coords = np.sort(np.concatenate([_coord[0,_pk] for _coord, _pk in zip(selected_coords, pks)]))
    # select into connected groups
    selected_groups = []
    _group = []
    for _i,_c in enumerate(pk_coords):
        if len(_group) == 0:
            _group.append(_c)
        elif sum(np.abs(np.array(_group)-_c)<=1) >= 1:
            _group.append(_c)
            np.delete(pk_coords, _i)
        else:
            if len(_group) > 1:
                selected_groups.append(_group)
            _group = []
    # pick from connected groups
    group_size_th = 2
    selected_pk_coords = np.sort([int(np.round(np.mean(_group))) for _group in selected_groups if len(_group) >= group_size_th])
    if make_plot:
        plt.figure()
        plt.imshow(distmap, cmap='seismic_r', vmin=0, vmax=1000)
        plt.colorbar()
        plt.title("input distance map")
        edges = [0] + list(selected_pk_coords)+[dim]
        for _i,_c in enumerate(edges[:-1]):
            plt.plot(np.arange(_c, edges[_i+1]), np.ones(edges[_i+1]-_c)*_c, color='y', linewidth=3.0)
            plt.plot(np.ones(edges[_i+1]-_c)*edges[_i+1], np.arange(_c, edges[_i+1]), color='y', linewidth=3.0)
        plt.xlim([0,dim])
        plt.show()
        
    return selected_pk_coords
