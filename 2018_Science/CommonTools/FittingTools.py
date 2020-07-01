#Bogdan Bintu
#Copyright Presidents and Fellows of Harvard College, 2017.

#external packages
import sys,os,glob
import numpy as np

import cPickle as pickle

import matplotlib.pyplot as plt

from scipy.ndimage.filters import maximum_filter,minimum_filter,median_filter,gaussian_filter
from scipy.spatial.distance import cdist,pdist,squareform
from scipy.optimize import leastsq

from AlignmentTools import *

def partition_map(list_,map_):
    """
    Inputs
    takes a list [e1,e2,e3,e4,e5,e6] and a map (a list of indices [0,0,1,0,1,2]).  map can be a list of symbols too. ['aa','aa','bb','aa','bb','cc']
    Output
    returns a sorted list of lists, e.g. [[e1, e2,e4],[e3,e5],[e6]]
    """
    list__=np.array(list_)
    map__=np.array(map_)
    return [list(list__[map__==element]) for element in np.unique(map__)]
def gaussian(height,center_z, center_x, center_y, width_z, width_x, width_y, bk=0):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    width_z = float(width_z)
    def gauss(z,x,y):
        g = np.abs(bk)+np.abs(height)*np.exp(
            -(((center_z-z)/width_z)**2+((center_x-x)/width_x)**2+
              ((center_y-y)/width_y)**2)/2.)
        return g
    return gauss
def sphere(center,radius,imshape=None):
    """Returns an int array (size: n x len(center)) with the xyz... coords of a sphere(elipsoid) of radius in imshape"""
    radius_=np.array(radius,dtype=int)
    if len(radius_.shape)==0:
        radius_ = np.array([radius]*len(center),dtype=int)
    xyz = np.array(np.indices(2*radius_+1),dtype=float)
    radius__=np.array(radius_,dtype=float)
    for i in range(len(xyz.shape)-1):
        radius__=np.expand_dims(radius__,axis=-1)
    xyz_keep = np.array(np.where(np.sum((xyz/radius__-1)**2,axis=0)<1))
    xyz_keep = xyz_keep-np.expand_dims(np.array(radius_,dtype=int),axis=-1)+np.expand_dims(np.array(center,dtype=int),axis=-1)
    xyz_keep = xyz_keep.T
    if imshape is not None:
        xyz_keep=xyz_keep[np.all((xyz_keep>=0)&(xyz_keep<np.expand_dims(imshape,axis=0)),axis=-1)]
    return xyz_keep

def grab_block(im,center,block_sizes):
    """Given an n-dim image <im>, a position <center> and a list of sizez <block_sizes>,
    retuns a block of the image of size <block_sizes> from <im> centered at <center>"""
    dims = im.shape
    slices = []
    def in_dim(c,dim):
        c_ = c
        if c_<0: c_=0
        if c_>dim: c_=dim
        return c_
    for c,block,dim in zip(center,block_sizes,dims):
        block_ = int(block/2)
        c=int(c)
        c_min,c_max = in_dim(c-block_,dim),in_dim(c+block-block_,dim)
        slices.append(slice(c_min,c_max))
    slices.append(Ellipsis)
    return im[slices]
   
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def fitmultigaussian(data,centers,radius=10,n_approx=10,width_zxy=[1.,1.,1.],min_width=0.5,fix_width=False):
    """Returns (height, x, y, z, width_x, width_y,width_z,background)
    for the 3D gaussian fit parameters (unconstrained, except for widths>min_width, height>0,background>0) for each point in <centers>
    A spherical neighbourhood of <radius> from the 3d image <data> is used.
    <n_approx> is the list of points in the neighbourhood for which to estimate the paramaters before optimizing for fitting.
    
    Warning: In practice this loosely constrained version is only used to estimate the widths. fitsinglegaussian_fixed_width behaves more robustly
    """
    data_=np.array(data,dtype=float)
    dims = np.array(data_.shape)
    xyz_unq=set()
    params=[]
    for center in centers:
        xyz = sphere(center,radius,imshape=dims)
        for xyz_ in xyz:
            xyz_unq.add(tuple(xyz_))
        data__=data_[xyz.T[0],xyz.T[1],xyz.T[2]]
        bk = np.median(np.sort(np.ravel(data__))[:n_approx])
        height = (np.median(np.sort(np.ravel(data__))[-n_approx:])-bk)
        center_z,center_x,center_y = center
        width_z,width_x,width_y = np.array(width_zxy)-min_width
        if fix_width:
            params_ = (height,center_z,center_x,center_y,bk)
        else:
            params_ = (height,center_z,center_x,center_y,width_z,width_x,width_y,bk)
        params.append(params_)
    params = np.array(params)
    xyz_unq = np.array([val for val in xyz_unq]).T
    dist_bk = cdist(xyz_unq.T,centers)
    dist_bk[dist_bk<1]=1
    weigh_bk = dist_bk/np.expand_dims(np.sum(dist_bk,axis=-1),-1)
    data_=data_[xyz_unq[0],xyz_unq[1],xyz_unq[2]]
    def gaussian(height,center_z, center_x, center_y, width_z=width_zxy[0]-min_width, 
                 width_x=width_zxy[1]-min_width, 
                 width_y=width_zxy[2]-min_width, 
                 bk=0,min_width=min_width):
        """Returns a gaussian function with the given parameters"""
        
        width_x_ = np.abs(width_x)+float(min_width)
        width_y_ = np.abs(width_y)+float(min_width)
        width_z_ = np.abs(width_z)+float(min_width)
        height_ = np.abs(height)
        bk_ = np.abs(bk)
        def gauss(z,x,y):
            g = bk_+height_*np.exp(
                -(((center_z-z)/width_z_)**2+((center_x-x)/width_x_)**2+
                  ((center_y-y)/width_y_)**2)/2.)
            return g
        return gauss
    def errorfunction(p):
        p_ = np.reshape(p,[len(centers),-1])
        bk_map = np.dot(weigh_bk,np.abs(p_[:,-1]))
        f=bk_map+np.sum([gaussian(*p__)(*xyz_unq) for p__ in p_[:,:-1]],0)
        g=data_
        #err=np.ravel(f-g-g*np.log(f/g))
        err=np.ravel(f-g)
        #print np.mean(err**2)
        return err
    p, success = leastsq(errorfunction, params)
    p = np.reshape(p,[len(centers),-1])
    p=np.abs(p)
    #p[:1:4]+=0.5
    if fix_width:
        p = np.concatenate([p[:,:-1],[width_zxy]*len(p),np.expand_dims(p[:,-1],-1)],axis=-1)
    else:
        p[:,4:7]+=min_width
    return  np.reshape(p,[len(centers),-1]),success

def gauss_ker(sig_xyz=[2,2,2],sxyz=16,xyz_disp=[0,0,0]):
    dim = len(xyz_disp)
    xyz=np.indices([sxyz+1]*dim)
    for i in range(len(xyz.shape)-1):
        sig_xyz=np.expand_dims(sig_xyz,axis=-1)
        xyz_disp=np.expand_dims(xyz_disp,axis=-1)
    im_ker = np.exp(-np.sum(((xyz-xyz_disp-sxyz/2.)/sig_xyz**2)**2,axis=0)/2.)
    return im_ker
def add_source(im_,pos=[0,0,0],h=200,sig=[2,2,2]):
    pos = np.array(pos)+0.5
    im=np.array(im_,dtype=float)
    pos_int = np.array(pos,dtype=int)
    xyz_disp = -pos_int+pos
    im_ker = gauss_ker(sig_xyz=sig,sxyz=int(np.max(sig)*5),xyz_disp=xyz_disp)
    im_ker_sz = np.array(im_ker.shape,dtype=int)
    pos_min = pos_int-im_ker_sz/2
    pos_max = pos_min+im_ker_sz
    im_shape = np.array(im.shape)
    def in_im(pos__):
        pos_=np.array(pos__,dtype=int)
        pos_[pos_>=im_shape]=im_shape[pos_>=im_shape]#-1
        pos_[pos_<0]=0
        return pos_
    pos_min_ = in_im(pos_min)
    pos_max_ = in_im(pos_max)
    pos_min_ker = pos_min_-pos_min
    pos_max_ker = im_ker_sz+pos_max_-pos_max
    #print zip(pos_min_ker,pos_max_ker),zip(pos_min_,pos_max_),zip(pos_min,pos_max)
    slices_ker = [slice(pm,pM)for pm,pM in zip(pos_min_ker,pos_max_ker)]
    slices_im = [slice(pm,pM)for pm,pM in zip(pos_min_,pos_max_)]
    im[slices_im]+=im_ker[slices_ker]*h
    return im
def subtract_source(im,pfit):
    return add_source(im,pos=pfit[1:4],h=-pfit[0],sig=pfit[-3:])
def plus_source(im,pfit):
    return add_source(im,pos=pfit[1:4],h=pfit[0],sig=pfit[-3:])
def fitsinglegaussian_fixed_width(data,center,radius=5,n_approx=10,width_zxy=[1.,1.,1.]):
    """Returns (height, x, y,z, width_x, width_y,width_z,bk)
    for the 3D gaussian fit for <radius> around a 3Dpoint <center> in the 3Dimage <data>
    <width_zxy> are the widths of the gaussian
    """
    data_=data
    dims = np.array(data_.shape)
    if center is  not None:
        center_z,center_x,center_y = center
    else:
        xyz = np.array(map(np.ravel,np.indices(data_.shape)))
        data__=data_[xyz[0],xyz[1],xyz[2]]
        args_high = np.argsort(data__)[-n_approx:]
        center_z,center_x,center_y = np.median(xyz[:,args_high],axis=-1)
    
    xyz = sphere([center_z,center_x,center_y],radius,imshape=dims).T
    if len(xyz[0])>0:
        data__=data_[xyz[0],xyz[1],xyz[2]]
        sorted_data = np.sort(data__)#np.sort(np.ravel(data__))
        bk = np.median(sorted_data[:n_approx])
        height = (np.median(sorted_data[-n_approx:])-bk)
            
        width_z,width_x,width_y = np.array(width_zxy)
        params_ = (height,center_z,center_x,center_y,bk)
        
        def gaussian(height,center_z, center_x, center_y,
                     bk=0,
                     width_z=width_zxy[0], 
                     width_x=width_zxy[1], 
                     width_y=width_zxy[2]):
            """Returns a gaussian function with the given parameters"""
            width_x_ = np.abs(width_x)
            width_y_ = np.abs(width_y)
            width_z_ = np.abs(width_z)
            height_ = np.abs(height)
            bk_ = np.abs(bk)
            def gauss(z,x,y):
                g = bk_+height_*np.exp(
                    -(((center_z-z)/width_z_)**2+((center_x-x)/width_x_)**2+
                      ((center_y-y)/width_y_)**2)/2.)
                return g
            return gauss
        def errorfunction(p):
            f=gaussian(*p)(*xyz)
            g=data__
            #err=np.ravel(f-g-g*np.log(f/g))
            err=np.ravel(f-g)
            return err
        p, success = leastsq(errorfunction, params_)
        p=np.abs(p)
        p = np.concatenate([p,width_zxy])
        return  p,success
    else:
        return None,None
def fast_local_fit(im,centers,radius=7,width_zxy=[1,1,1]):
    """
    Given a set of seeds <centers> in a 3d image <im> iteratively 3d gaussian fit around the seeds for <radius> and with fixed <width_zxy>
    Retruns a numpy array of size Nx(height, x, y, z, width_x, width_y,width_z,background) where N~len(centers). Bad fits are disregarded.
    """
    ps=[]
    im_=np.array(im)
    for center in centers:
        p,success = fitsinglegaussian_fixed_width(im_,center,radius=radius,n_approx=5,width_zxy=width_zxy)
        if p is not None:
            if np.max(np.abs(p[1:4]-center))<radius:
                ps.append(p)
    return np.array(ps)
def fit_seed_points_base(im,centers,width_zxy=[1.,1.,1.],radius_fit=5,n_max_iter = 10,max_dist_th=0.25):
    """
    Given a set of seeds <centers> in a 3d image <im> iteratively 3d gaussian fit around the seeds (in order of brightness) and subtract the gaussian signal.
    Retruns a numpy array of size Nx(height, x, y, z, width_x, width_y,width_z,background) where N~len(centers). Bad fits are disregarded.
    Warning: Generally a bit slow. In practice, the faster version fast_local_fit is used.
    """
    #print "Fitting:" +str(len(centers[0]))+" points"
    z,x,y = centers
    
    
    if len(x)>0:
        #get height of the points and order by brightness
        h = [im[int(z_),int(x_),int(y_)] for z_,x_,y_ in zip(z,x,y)]
        inds = np.argsort(h)[::-1]
        z,x,y = z[inds],x[inds],y[inds]
        zxy = np.array([z,x,y],dtype=int).T
        
        #fit the points in order of brightness and at each fit subtract the gaussian signal
        ps = []
        im_subtr = np.array(im,dtype=float)
        for center in zxy:
            p,success = fitsinglegaussian_fixed_width(im_subtr,center,radius=radius_fit,n_approx=5,width_zxy=width_zxy)
            if p is not None:
                ps.append(p)
                im_subtr = subtract_source(im_subtr,p)

        im_add = np.array(im_subtr)

        max_dist=np.inf
        n_iter = 0
        while max_dist>max_dist_th:
            ps_1=np.array(ps)
            ps_1=ps_1[np.argsort(ps_1[:,0])[::-1]]
            ps = []
            ps_1_rem=[]
            for p_1 in ps_1:
                center = p_1[1:4]
                im_add = plus_source(im_add,p_1)
                p,success = fitsinglegaussian_fixed_width(im_add,center,radius=radius_fit,n_approx=5,width_zxy=width_zxy)
                if p is not None:
                    ps.append(p)
                    ps_1_rem.append(p_1)
                    im_add = subtract_source(im_add,p)
            ps_2=np.array(ps)
            ps_1_rem=np.array(ps_1_rem)
            dif = ps_1_rem[:,1:4]-ps_2[:,1:4]
            max_dist = np.max(np.sum(dif**2,axis=-1))
            n_iter+=1
            if n_iter>n_max_iter:
                break
        return ps_2
    else:
        return np.array([])

def get_seed_points_base(im,gfilt_size=0.,filt_size=3,th_seed=150.,hot_pix_th=0,return_h=False):
    """Get the seed points in an image.
    #1 perform a gaussian filter
    #2 find local maxima within a radius 3 above th_seed form the minimum
    #3 remove hot pixels (high multiplicity of xy positions with different z)
    """
    im_plt = np.array(im)
    if gfilt_size>0:
        im_plt = gaussian_filter(im,gfilt_size)
    max_filt = maximum_filter(im_plt,filt_size)
    min_filt = minimum_filter(im_plt,filt_size)
    im_plt2 = (max_filt==im_plt)&(min_filt!=im_plt)
    z,x,y = np.where(im_plt2)
    
    keep = (max_filt[z,x,y]-min_filt[z,x,y])>th_seed
    x,y,z = x[keep],y[keep],z[keep]
    h = max_filt[z,x,y]-min_filt[z,x,y]

    #get rid of hot pixels
    if hot_pix_th>0 and len(x)>0:
        xy = y*np.max(x)+x
        #xy_str = [str([x_,y_]) for x_,y_ in zip(x,y)]
        xy_,cts_ = np.unique(xy,return_counts=True)
        bad_xy = xy_[cts_>hot_pix_th]
        keep = np.array([xy_ not in bad_xy for xy_ in xy],dtype=bool)
        x,y,z = x[keep],y[keep],z[keep]
        h = h[keep]
    centers = np.array([z,x,y])
    hind = np.argsort(h)[::-1]
    centers = np.array([z[hind],x[hind],y[hind]])
    if return_h:
        centers = np.array([z[hind],x[hind],y[hind],h[hind]])
    return centers

def update_candid_spot(im,low_limit,prefered_limit,chr_pts_,txyz_,tag,cutoff_window=20,cutoff_chr=10,candid_spot={},fl_cands=None):
    """im -> 3dimage of the current hybe
    chr_pts_ -> the postions of the selected chromosomes
    txyz_ -> drift from current hybe to reference frame where chromosome was selected 
             and to which the drift was computed to
    tag -> the tag on how to update candid_spot dictionary
    candid_spot -> the dictionary where to save data
    fl_cands -> the file where to save data
    """
    [z,x,y,h]=get_seed_points_base(im,gfilt_size=0.75,filt_size=3,th_seed=low_limit,hot_pix_th=4,return_h=True)
    centersh = np.array([z,x,y,h]).T
    centers = np.array([z,x,y]).T
    #compare with prefered spots
    dists = cdist(chr_pts_,centers-[txyz_])#distances to chromosomes should be computed in the reference where the chromosomes where selected (ref=0)
    mindist = np.min(dists,0)
    keep = mindist<cutoff_window #keep only points within a distance cutoff_window from a chromosome
    centersh_close = centersh[keep] 
    closest_chr = np.argmin(dists[:,keep],0) #vector with ids of closest chromosomes
    centersh_close = np.concatenate([centersh_close.T,[mindist[keep]]],axis=0).T #extend with distance from closest chromosome
    chroms_id = np.unique(closest_chr)
    partitioned_info = partition_map(centersh_close,closest_chr) #partition to chromosome
    #iterate through the crhomosomes in the filed of view
    for pts_info,chr_id in zip(partitioned_info,chroms_id):
        z_,x_,y_,h_,dist_=np.array(pts_info).T
        centes_fit=np.array([z_,x_,y_]).T
        keep_close=dist_<cutoff_chr
        if np.sum(keep_close)>0:
            if np.max(h_[keep_close])>=prefered_limit:
                centes_fit=centes_fit[h_>=prefered_limit]
        pp = fast_local_fit(im,centes_fit)
        pp=np.array(pp)
        if len(pp)>0:
            pp[:,1:4]=pp[:,1:4]-[txyz_]#apply drift correction
        #record in dictionary in the old format
        candid_dic,_ = candid_spot.get(chr_id,({},None))
        candid_dic[tag]=pp#drif corrected
        candid_spot[chr_id]=(candid_dic,chr_pts_[chr_id])
    if fl_cands is not None:
        pickle.dump(candid_spot,open(fl_cands,'wb'))

## Bead drift correction functions
        
def get_STD_centers(im,hseed=150,nseed=100,plt_val=False):
    """Given an image <im> a min height <hseed> and a number of seeds <nseed> this returns  zxy positions of the  brightest ~<nseed> fits.
    This is intended for beads.
    """
    im_=np.array(im,dtype=float)
    centers = get_seed_points_base(im_,gfilt_size=0.75,filt_size=3,th_seed=hseed,hot_pix_th=5)
    centers = np.array(centers).T
    centers = centers[:nseed]
    
    #pfits = fit_seed_points_base(im,centers,width_z=1.8,width_xy=1.,radius_fit=5,n_max_iter=3,max_dist_th=0.25)
    pfits = fast_local_fit(im,centers,radius=7,width_zxy=[1,1,1])
    if len(pfits)==0:
        cents = np.array([[np.inf]*3])
    else:
        cents = pfits[:,1:4]
    #cents = centers.T
    if plt_val:
        plt.figure()
        plt.imshow(np.max(im,0),interpolation='nearest')
        plt.plot(cents[:,2],cents[:,1],'o')
        plt.show()
    return cents
    
def simplify_txyz(txyz_both):
    """
    ensure that the final output is Nx(tx,ty,tz) for the list of drifts
    """
    if len(np.array(txyz_both).shape)==3:
        txyz = np.mean(txyz_both,1)
    else:
        txyz = txyz_both
    return txyz

def get_STD_beaddrift(ims_beads,coord_sel=None,sz_ex=50,desired_nbeads = 5, desired_displ = 0.4, hseed=150,nseed=100,ref=None,force=False,save=True,save_file='temp.pkl'):
    """Given a list of bead images <ims_beads> this handles the fine bead drift correction.
    For each 3d image in <ims_beads> the beads for subimages of size <sz_ex>,
    centered at [center,center,center],[center,center,center]+[0,2*sz_ex,0] are fitted using #get_STD_centers with paramaters <hseed>,<nseed>.
    Beads for each of the two subimages are aligned with the corresponding beads for the reference image of index <ref> (default = len(ims_beads)/2) in ims_beads.
    """
    repeat = True
    if save:
        save_cor = save_file
        if os.path.exists(save_cor):
            txyzs_both = pickle.load(open(save_cor,'rb'))
            txyzs = simplify_txyz(txyzs_both)
            if len(txyzs)==len(ims_beads):
                repeat=False
    repeat = repeat or force
    if repeat:
        #get txyz
        if ref is None: ref=len(ims_beads)/2
        im_ref = ims_beads[ref]
        coord_sel = np.array(im_ref.shape)/2
        coord_sel1 =coord_sel
        im_ref_sm = grab_block(im_ref,coord_sel1,[sz_ex]*3)
        cents_ref1 = get_STD_centers(im_ref_sm,hseed=hseed,nseed=nseed)#list of fits of beads in the ref cube 1
        coord_sel2 = np.array([0,2*sz_ex,0])+coord_sel
        im_ref_sm = grab_block(im_ref,coord_sel2,[sz_ex]*3)
        cents_ref2 = get_STD_centers(im_ref_sm,hseed=hseed,nseed=nseed)#list of fits of beads in the ref cube 2
        txyzs = []
        txyzs_both = []
        cutoff_=2
        xyz_res_=1
        for iim,im in enumerate(ims_beads):
            im_sm = grab_block(im,coord_sel1,[sz_ex]*3)
            cents1 = get_STD_centers(im_sm,hseed=hseed,nseed=nseed)#list of fits of beads in the cube 1
            im_sm = grab_block(im,coord_sel2,[sz_ex]*3)
            cents2 = get_STD_centers(im_sm,hseed=hseed,nseed=nseed)#list of fits of beads in the cube 2
            print "Aligning "+str(iim)
            #print "Ver 1",len(cents_ref1),len(cents1)
            txyz1,c11,c12 = translation_aling_pts(cents_ref1,cents1,cutoff=cutoff_,xyz_res=xyz_res_,plt_val=False,return_pts=True)
            #print "Ver 2",len(cents_ref2),len(cents2)
            txyz2,c21,c22 = translation_aling_pts(cents_ref2,cents2,cutoff=cutoff_,xyz_res=xyz_res_,plt_val=False,return_pts=True)
            #print "Ended"
            txyz = (txyz1+txyz2)/2.
            print txyz1,txyz2,len(c11),len(c21)
            displ = np.max(np.abs(txyz1-txyz2))
            if (displ>desired_displ) or (len(c11)<desired_nbeads) or (len(c21)<desired_nbeads):
                print "Suspecting failure."
                #sz_ex+=10
                coord_sel3 = np.array([0,-2*sz_ex,0])+coord_sel
                im_ref_sm = grab_block(im_ref,coord_sel3,[sz_ex]*3)
                cents_ref3 = get_STD_centers(im_ref_sm,hseed=hseed,nseed=nseed)#list of fits of beads in the ref cube 3
                im_sm = grab_block(im,coord_sel3,[sz_ex]*3)
                cents3 = get_STD_centers(im_sm,hseed=hseed,nseed=nseed)#list of fits of beads in the cube 2
                txyz3 = translation_aling_pts(cents_ref3,cents3,cutoff=cutoff_,xyz_res=xyz_res_,plt_val=False)
                measures = map(np.sum,map(np.abs,[txyz3-txyz1,txyz3-txyz2,txyz1-txyz2]))
                imeasure = np.argmin(measures)
                variants = [[txyz3,txyz1],[txyz3,txyz2],[txyz1,txyz2]]
                txyz = np.mean(variants[imeasure],0)
                txyzs_both.append(variants[imeasure])
                print variants[imeasure]
            else:
                txyzs_both.append([txyz1,txyz2])
            txyzs.append(txyz)
        if save:
            save_cor = save_file
            pickle.dump(txyzs_both,open(save_cor,'wb'))
    return txyzs_both,repeat
