import matplotlib.pylab as plt
import numpy as np
import pickle,os
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist,cdist,squareform
from sklearn.cluster import DBSCAN
def rg_med(zxy):
    zxy_ = np.array(zxy)
    zxy_ = zxy_[~np.isnan(zxy_[:,0])]
    zxy_ = zxy_ - np.median(zxy_,0)
    return np.median(np.linalg.norm(zxy_,axis=-1))
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

def resize(im__,scale_percent = 100):
    import cv2
    width = int(im__.shape[1] * scale_percent / 100)
    height = int(im__.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(im__, dim, interpolation = cv2.INTER_NEAREST)
    return resized
def rotate_bound(image, angle):
    import cv2
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH),cv2.INTER_NEAREST)
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
            

        x_im_l,y_im_l,x_im_r,y_im_r = map(np.array,[x_im_l,y_im_l,x_im_r,y_im_r])
        xn_im_l,yn_im_l,xn_im_r,yn_im_r = map(np.array,[xn_im_l,yn_im_l,xn_im_r,yn_im_r])

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
            #print x_im_l,y_im_l,x_im_r,y_im_r

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
        cax = ax.imshow(im,interpolation='nearest',cmap='seismic')#,vmax=1000,vmin=0)
        cbar = fig.colorbar(cax)
        plt.show()
    return max_ratio,local_max_good
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
def interpolate_chr(_chr):
    """linear interpolate chromosome coordinates"""
    _new_chr = np.array(_chr)
    for i in range(_new_chr.shape[-1]):
        _new_chr[:,i]=interp1dnan(_new_chr[:,i])
    return _new_chr

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


def Generate_ArrowHead(distmap, _make_plot=True, _normalization=False, _scale=[200,1000], _gaussian_size=0):
    """Function to transfer normal distance map to arrow head map
    Inputs:
        distmap: n-by-n array for pair-wise distance, 2d-array
        _make_plot: whether make plot for arrowhead result, bool
        _normalization: whether do normalization for orignial iamage, bool
        _scale: if _normalization is given, give lower and upper limits which will be transformed into 0 and 1, list of 2
        _gaussian_size: sigma for gaussian blur the original distance map, int > 0
    Outputs:
        _arrowmap: arrowhead map for given distance map, 2d-array"""
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
        plt.imshow(_normed_map, cmap='seismic')
        plt.colorbar()
        plt.title("input distance map")
        plt.show()
        plt.figure()
        plt.imshow(_arrowmap, cmap='seismic_r')
        plt.colorbar()
        plt.title("Arrowhead")
        plt.show()
    return _arrowmap

def get_domain_fusion_rg(zxy,dom_starts_f,percR=90):
    """ zxy is Nx3 in nm (with nans) and dom_starts_f are the edges of all domains
    This algoritm checks to see if any two domains "fuse" by checking when applying 
    DBSCAN whether they are primarily in one class
    """
    #zxy = pts[ichr]
    #dom_starts_f = dom_starts_fs[ichr]
    bad = np.isnan(zxy[:,0])
    zxy_ = zxy[~bad]
    #mat=squareform(pdist(zxy_))

    dom_starts = np.zeros(len(zxy))
    dom_starts[dom_starts_f[:-1]]=1
    dom_starts = list(np.where(dom_starts[~bad])[0])+[len(zxy_)]
    dices = []
    dices_f = []
    for i in range(len(dom_starts)-1):
        for j in range(i):
            zxy1 = zxy_[dom_starts[i]:dom_starts[i+1]]
            zxy2 = zxy_[dom_starts[j]:dom_starts[j+1]]
            cm1 = np.median(zxy1,0)
            cm2 = np.median(zxy2,0)
            
            r1 = np.percentile(np.linalg.norm(zxy1-cm1,axis = -1),percR)
            r2 = np.percentile(np.linalg.norm(zxy2-cm2,axis = -1),percR)
            d12 = np.linalg.norm(cm1-cm2)
            dice = d12<(r1+r2)
            if dice:
                dices.append([dom_starts[i],dom_starts[i+1],dom_starts[j],dom_starts[j+1],dice])
                dices_f.append([dom_starts_f[i],dom_starts_f[i+1],dom_starts_f[j],dom_starts_f[j+1]])
    dices_f = np.array(dices_f)
    return dices_f
def get_domain_fusion(zxy,dom_starts_f,thglob=0.5,eps=500,min_samples=2):
    """ zxy is Nx3 in nm (with nans) and dom_starts_f are the edges of all domains
    This algoritm checks to see if any two domains "fuse" by checking when applying 
    DBSCAN whether they are primarily in one class
    """
    #zxy = pts[ichr]
    #dom_starts_f = dom_starts_fs[ichr]
    bad = np.isnan(zxy[:,0])
    zxy_ = zxy[~bad]
    #mat=squareform(pdist(zxy_))

    dom_starts = np.zeros(len(zxy))
    dom_starts[dom_starts_f[:-1]]=1
    dom_starts = list(np.where(dom_starts[~bad])[0])+[len(zxy_)]
    dices = []
    dices_f = []
    for i in range(len(dom_starts)-1):
        for j in range(i):
            zxy1 = zxy_[dom_starts[i]:dom_starts[i+1]]
            zxy2 = zxy_[dom_starts[j]:dom_starts[j+1]]

            X = np.concatenate([zxy1,zxy2])
            #X = zxy_
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
            #db = SpectralClustering(n_clusters=2).fit(X)
            lab1,lab2 = db.labels_[:len(zxy1)],db.labels_[len(zxy1):]

            lab1 = lab1[lab1!=-1]
            lab2 = lab2[lab2!=-1]
            dice = False
            if len(lab1)>0 and len(lab2)>0:
                e1,cts1 = np.unique(lab1,return_counts=True)
                e1=e1[np.argmax(cts1)]
                e2,cts2 = np.unique(lab2,return_counts=True)
                e2=e2[np.argmax(cts2)]
                #dice = 1.*(np.sum(np.in1d(lab1,lab2))+np.sum(np.in1d(lab2,lab1)))/(len(lab1)+len(lab2))
                dice = (e1==e2) and (np.max(cts1)/len(lab1)>thglob) and (np.max(cts2)/len(lab2)>thglob)
            if dice:
                dices.append([dom_starts[i],dom_starts[i+1],dom_starts[j],dom_starts[j+1]])
                dices_f.append([dom_starts_f[i],dom_starts_f[i+1],dom_starts_f[j],dom_starts_f[j+1]])
    dices_f = np.array(dices_f)
    return dices_f
    
def get_ind_loc_max(ratio,cutoff_max=1.,valley=3):
    """get local maximum within valley size bigger than cut-off"""
    local_max_good_ = []
    for id_ in range(len(ratio)):
        l = np.max([0,id_-valley])
        r = np.min([len(ratio),id_+valley])
        if ratio[id_]==np.nanmax(ratio[l:r]) and ratio[id_]>cutoff_max:
            local_max_good_.append(id_)
    return np.array(local_max_good_)
def calc_sep_cor(mat,a,b,c,context=75):
    #a,b,c = dom_starts[i],dom_starts[i+1],dom_starts[i+2]
    bst,bend = b-context,b+context
    if bst<0:bst=0
    if bend>len(mat):bend=len(mat)
    dom1_cont = mat[a:b,bst:bend]
    dom2_cont = mat[b:c,bst:bend]
    func = np.mean
    dom1_cont = func(dom1_cont,0)
    dom2_cont = func(dom2_cont,0)
    dom1_cont-=func(dom1_cont)
    dom2_cont-=func(dom2_cont)
    #cor_coef = np.median(dom1_cont*dom2_cont)/np.sqrt(np.median(dom1_cont*dom1_cont)*np.median(dom2_cont*dom2_cont))
    cor_coef = func(dom1_cont*dom2_cont)/np.sqrt(func(dom1_cont*dom1_cont)*func(dom2_cont*dom2_cont))
    return cor_coef
def calc_seps_cor(mat,dom_starts,context=75):
    return np.array([calc_sep_cor(mat,dom_starts[i],dom_starts[i+1],dom_starts[i+2],context=context) 
                     for i in range(len(dom_starts)-2)])
def get_dom_starts_cor(zxy,dom_sz=5,context=1000,cut_off=0.66,dist_cutoff=750):
    """
    This is the main function for domain calling.
    dom_sz is the minimum domains size - this is usesd to caclulate candidate boundaries.
    context is how much of the off diagonal to consider for correlation between domains.
    Prior to computing the correlations to decide whtehr domains fuse, dist_cutoff is used to threshold
    the distance matrix.
    Use as:
    #zxy,zxy_,mat,dom_starts_,dom_starts,dom_starts_f= 
    get_dom_starts_cor(pts[99],dom_sz=5,context=1000,cut_off=0.66,dist_cutoff=750)
    """

    #throw away nans
    zxy_ = np.array(zxy)
    bad = np.isnan(zxy_[:,0])
    zxy_ = zxy_[~bad]
    
    
    #get candidate boundaries
    dists = []
    
    for i in range(len(zxy_)):
        cm1= np.nanmean(zxy_[max(i-dom_sz,0):i],axis=0)
        cm2= np.nanmean(zxy_[i:i+dom_sz],axis=0)
        dist = np.linalg.norm(cm1-cm2)
        dists.append(dist)

    bds_candidates = get_ind_loc_max(dists,cutoff_max=0,valley=dom_sz)
    dom_starts= [0]+[dm for dm in bds_candidates if dm>dom_sz and dm<len(zxy_)-dom_sz]+[len(zxy_)]
    
    mat = squareform(pdist(zxy_))
    mat_ = mat<dist_cutoff
    #mat_ = np.exp(-mat*mat/(2*dist_cutoff**2))
    dom_starts_ = list(dom_starts)
    dom_starts = list(dom_starts)
    
    while len(dom_starts)>2:
        seps = calc_seps_cor(mat_,dom_starts,context=context)
        imin = np.argmax(seps)
        if seps[imin]>cut_off:
            dom_starts.pop(imin+1)
            seps = list(seps)
            seps.pop(imin)
        else:
            break
            
    dom_starts_f = np.concatenate([np.arange(len(zxy))[~bad],[len(zxy)]])
    dom_starts_f = dom_starts_f[dom_starts]#
    return zxy,zxy_,mat,dom_starts_,dom_starts,dom_starts_f
