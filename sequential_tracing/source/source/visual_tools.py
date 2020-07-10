import os,re,time,glob
import numpy as np
import pickle as pickle
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import scipy
from scipy.signal import fftconvolve
from scipy.ndimage.filters import maximum_filter,minimum_filter,median_filter,gaussian_filter
from scipy import ndimage, stats
from skimage import morphology, restoration, measure
from skimage.segmentation import random_walker
from scipy.ndimage import gaussian_laplace
import cv2
import multiprocessing as mp
from sklearn.decomposition import PCA
from scipy.ndimage.interpolation import map_coordinates

from . import get_img_info, corrections, alignment_tools
from .External import Fitting_v3
from . import _correction_folder,_temp_folder,_distance_zxy,_sigma_zxy,_image_size, _allowed_colors

# generate common colors
# generate my colors
from matplotlib.colors import ListedColormap
# red
Red_colors = np.ones([256,4])
Red_colors[:,1] = np.linspace(1,0,256)
Red_colors[:,2] = np.linspace(1,0,256)
myReds = ListedColormap(Red_colors)
# blue
Blue_colors = np.ones([256,4])
Blue_colors[:,0] = np.linspace(1,0,256)
Blue_colors[:,1] = np.linspace(1,0,256)
myBlues = ListedColormap(Blue_colors)
# green
Green_colors = np.ones([256,4])
Green_colors[:,0] = np.linspace(1,0,256)
Green_colors[:,2] = np.linspace(1,0,256)
myGreens = ListedColormap(Green_colors)
_myCmaps = [myReds, myBlues, myGreens]


def partition_map(list_,map_, enumerate_all=False):
    """
    Inputs
    takes a list [e1,e2,e3,e4,e5,e6] and a map (a list of indices [0,0,1,0,1,2]).  map can be a list of symbols too. ['aa','aa','bb','aa','bb','cc']
    Output
    returns a sorted list of lists, e.g. [[e1, e2,e4],[e3,e5],[e6]]
    """
    list__=np.array(list_)
    map__=np.array(map_)
    if enumerate_all:
        return [list(list__[map__==_i]) for _i in np.arange(0, np.max(map__)+1)]
    else:
        return [list(list__[map__==element]) for element in np.unique(map__)]

def old_gauss_ker(sig_xyz=[2,2,2],sxyz=16,xyz_disp=[0,0,0]):
    '''Create a gaussian kernal, return standard gaussian level within sxyz size and sigma 2,2,2'''
    dim = len(xyz_disp)
    xyz=np.indices([sxyz+1]*dim)
    print(sxyz)
    for i in range(len(xyz.shape)-1):
        sig_xyz=np.expand_dims(sig_xyz,axis=-1)
        xyz_disp=np.expand_dims(xyz_disp,axis=-1)
    im_ker = np.exp(-np.sum(((xyz-xyz_disp-sxyz/2.)/sig_xyz**2)**2,axis=0)/2.)
    return im_ker

def gauss_ker(sig_xyz=[2,2,2],sxyz=16,xyz_disp=[0,0,0]):
    """Faster version of gaussian kernel"""
    dim = len(xyz_disp)
    xyz=np.swapaxes(np.indices([sxyz+1]*dim), 0,dim)
    return np.exp(-np.sum(((xyz-np.array(xyz_disp)-sxyz/2.)/np.array(sig_xyz)**2)**2,axis=dim)/2.)

def gaussian_kernel_2d(center_xy, sigma_xy=[2,2], radius=8):
    """Function to generate gaussian kernel in 2d space"""
    ## check inputs
    if len(center_xy) != 2:
        raise IndexError(f"center_xy should be length=2 list or array")
    if len(sigma_xy) != 2:
        raise IndexError(f"sigma_xy should be length=2 list or array")
    radius = int(radius)
    if radius < 3 * max(sigma_xy): # if radius is smaller than 3-sigma, expand
        radius = 3*max(sigma_xy)
    xy_coords=np.swapaxes(np.indices([radius*2+1]*2), 0, 2)
    return np.exp(-np.sum(((xy_coords-np.array(center_xy)-radius)/np.array(sigma_xy)**2)**2,axis=2)/2.)

def add_source(im_,pos=[0,0,0],h=200,sig=[2,2,2],size_fold=10):
    '''Impose a guassian distribution with given position, height and sigma, onto an existing figure'''
    im=np.array(im_,dtype=float)
    pos = np.array(pos)
    pos_int = np.array(pos,dtype=int)
    xyz_disp = -pos_int+pos
    im_ker = gauss_ker(sig, int(max(sig)*size_fold), xyz_disp)
    im_ker_sz = np.array(im_ker.shape,dtype=int)
    pos_min = np.array(pos_int-im_ker_sz/2, dtype=np.int)
    pos_max = np.array(pos_min+im_ker_sz, dtype=np.int)
    im_shape = np.array(im.shape)
    def in_im(pos__):
        pos_=np.array(pos__,dtype=np.int)
        pos_[pos_>=im_shape]=im_shape[pos_>=im_shape]-1
        pos_[pos_<0]=0
        return pos_
    pos_min_ = in_im(pos_min)
    pos_max_ = in_im(pos_max)
    pos_min_ker = pos_min_-pos_min
    pos_max_ker = im_ker_sz+pos_max_-pos_max
    slices_ker = tuple(slice(pm,pM) for pm,pM in zip(pos_min_ker,pos_max_ker))
    slices_im = tuple(slice(pm,pM) for pm,pM in zip(pos_min_,pos_max_))
    im[slices_im] += im_ker[slices_ker]*h
    return im

def subtract_source(im,pfit):
    return add_source(im,pos=pfit[1:4],h=-pfit[0],sig=pfit[-3:])

def plus_source(im,pfit):
    return add_source(im,pos=pfit[1:4],h=pfit[0],sig=pfit[-3:])

def sphere(center,radius,imshape=None):
    """Returns an int array (size: n x len(center)) with the xyz... coords of a sphere(elipsoid) of radius in imshape"""
    radius_=np.array(radius,dtype=float)
    if len(radius_.shape)==0:
        radius_ = np.array([radius]*len(center),dtype=np.int)
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

# fit single gaussian
def fitsinglegaussian_fixed_width(data,center,radius=10,n_approx=10,width_zxy=_sigma_zxy):
    """Returns (height, x, y,z, width_x, width_y,width_z,bk)
    the gaussian parameters of a 2D distribution found by a fit"""
    data_=np.array(data,dtype=float)
    dims = np.array(data_.shape)
    if center is  not None:
        center_z,center_x,center_y = center
    else:
        xyz = np.array(list(map(np.ravel,np.indices(data_.shape))))
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
        p, success = scipy.optimize.leastsq(errorfunction, params_)
        p=np.abs(p)
        p = np.concatenate([p,width_zxy])
        #p[:1:4]+=0.5
        return  p,success
    else:
        return None,None
def fit_seed_points_base(im, centers, width_z=_sigma_zxy[0], width_xy=_sigma_zxy[1],
                         radius_fit=5, n_max_iter = 10, max_dist_th=0.25):
    '''Basic function used for multiple gaussian fitting, given image:im, seeding_result:centers '''
    print("Fitting:" +str(len(centers[0]))+" points")
    z,x,y = centers # fitting kernels provided by previous seeding
    if len(x)>0:
        #estimate height
        #gfilt_size=0.75
        #filt_size=3
        #im_plt = gaussian_filter(im,gfilt_size)
        #max_filt = maximum_filter(im_plt,filt_size)
        #min_filt = minimum_filter(im_plt,filt_size)
        #h = max_filt[z,x,y]-min_filt[z,x,y]

        #inds = np.argsort(h)[::-1]
        #z,x,y = z[inds],x[inds],y[inds]
        zxy = np.array([z,x,y],dtype=int).T

        ps = []
        im_subtr = np.array(im,dtype=float)
        for center in zxy:
            p,success = fitsinglegaussian_fixed_width(im_subtr,center,radius=radius_fit,n_approx=10,width_zxy=[width_z,width_xy,width_xy])
            if p is not None: # If got any successful fitting, substract fitted profile
                ps.append(p)
                im_subtr = subtract_source(im_subtr,p)

        im_add = np.array(im_subtr)

        max_dist=np.inf
        n_iter = 0
        while max_dist > max_dist_th:
            ps_1=np.array(ps)
            ps_1=ps_1[np.argsort(ps_1[:,0])[::-1]]
            ps = []
            ps_1_rem=[]
            for p_1 in ps_1:
                center = p_1[1:4]
                im_add = plus_source(im_add,p_1)
                p,success = fitsinglegaussian_fixed_width(im_add,center,radius=radius_fit,n_approx=10,width_zxy=[width_z,width_xy,width_xy])
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


## Fit bead centers
def get_STD_centers(im, seeds=None, th_seed=150, 
                    dynamic=False, seed_by_per=False, th_seed_percentile=95,
                    min_num_seeds=1,
                    remove_close_pts=True, close_threshold=0.1, fit_radius=5,
                    sort_by_h=False, save=False, save_folder='', save_name='',
                    plt_val=False, force=False, verbose=False):
    '''Fit beads for one image:
    Inputs:
        im: image, ndarray
        th_seeds: threshold for seeding, float (default: 150)
        dynamic: whether do dynamic seeding, bool (default:True)
        th_seed_percentile: intensity percentile for seeding, float (default: 95)
        remove_close_pts: whether remove points really close to each other, bool (default:True)
        close_threshold: threshold for removing duplicates within a distance, float (default: 0.01)
        fit_radius
        sort_by_h: whether sort fitted points by height, bool (default:False)
        plt_val: whether making plot, bool (default: False)
        save: whether save fitting result, bool (default: False)
        save_folder: full path of save folder, str (default: None)
        save_name: full name of save file, str (default: None)
        force: whether force fitting despite of saved file, bool (default: False)
        verbose: say something!, bool (default: False)
    Outputs:
        beads: fitted spots with information, n by 4 array'''
    import os
    import pickle as pickle
    if not force and os.path.exists(save_folder+os.sep+save_name) and save_name != '':
        if verbose:
            print("- loading file:,", save_folder+os.sep+save_name)
        beads = pickle.load(open(save_folder+os.sep+save_name, 'rb'))
        if verbose:
            print("--", len(beads), " of beads loaded.")
        return beads
    else:
        # seeding
        if seeds is None:
            seeds = get_seed_in_distance(im, center=None, dynamic=dynamic,
                                        th_seed_percentile=th_seed_percentile,
                                        seed_by_per=seed_by_per,
                                        min_dynamic_seeds=min_num_seeds,
                                        gfilt_size=0.75, filt_size=3, 
                                        th_seed=th_seed, hot_pix_th=4, verbose=verbose)
        # fitting
        fitter = Fitting_v3.iter_fit_seed_points(im, seeds.T, radius_fit=5)
        fitter.firstfit()
        pfits = fitter.ps
        #pfits = visual_tools.fit_seed_points_base_fast(im,seeds.T,width_z=1.8*1.5/2,width_xy=1.,radius_fit=5,n_max_iter=3,max_dist_th=0.25,quiet=not verbose)
        # get coordinates for fitted beads
        
        if len(pfits) > 0:
            if sort_by_h:
                _intensity_order = np.argsort(np.array(pfits)[:,0])
                beads = np.array(pfits)[np.flipud(_intensity_order), 1:4]
            else:
                beads = np.array(pfits)[:, 1:4]
            # remove very close spots
            if remove_close_pts:
                remove = np.zeros(len(beads), dtype=np.bool)
                for i, bead in enumerate(beads):
                    if np.isnan(bead).any() or np.sum(np.sum((beads-bead)**2, axis=1) < close_threshold) > 1:
                        remove[i] = True
                    if (bead < 0).any() or (bead > np.array(im.shape)).any():
                        remove[i] = True
                beads = beads[remove==False]
        else:
            beads = None
        if verbose:
            print(f"- fitting {len(pfits)} points")
            print(
                f"-- {np.sum(remove)} points removed given smallest distance {close_threshold}")
        # make plot if required
        if plt_val:
            plt.figure()
            plt.imshow(np.max(im, 0), interpolation='nearest')
            plt.plot(beads[:, -1], beads[:, -2], 'or')
            plt.show()
        # save to pickle if specified
        if save:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            if verbose:
                print("-- saving fitted spots to",
                      save_folder+os.sep+save_name)
            pickle.dump(beads[:,-3:], open(save_folder+os.sep+save_name, 'wb'))

        return beads


def get_seed_points_base(im, gfilt_size=0.75, background_gfilt_size=10, filt_size=3,
                         th_seed=300, hot_pix_th=0, return_h=False):
    """Base function to do seeding"""
    # gaussian-filter + max-filter
    if gfilt_size:
        max_im = gaussian_filter(im,gfilt_size)
    else:
        max_im = im
    # gaussian_filter (large) + min_filter
    if background_gfilt_size:
        min_im = gaussian_filter(im,background_gfilt_size)
    else:
        min_im = im
        
    max_filt = np.array(maximum_filter(max_im,filt_size), dtype=np.int64)
    min_filt = np.array(minimum_filter(min_im,filt_size), dtype=np.int64)
    # get candidate seed points
    im_plt2 = (max_filt==max_im) & (min_filt!=min_im) & (min_filt!=0)
    z,x,y = np.where(im_plt2)
    keep = (max_filt[z,x,y]-min_filt[z,x,y])>th_seed#/np.array(max_filt[z,x,y],dtype=float)>0.5
    x,y,z = x[keep],y[keep],z[keep]
    h = max_filt[z,x,y]-min_filt[z,x,y]

    #get rid of hot pixels
    if hot_pix_th>0:
        xy_str = [str([x_,y_]) for x_,y_ in zip(x,y)]
        xy_str_,cts_ = np.unique(xy_str,return_counts=True)
        keep = np.array([xy_str__ not in xy_str_[cts_>hot_pix_th] for xy_str__ in xy_str],dtype=bool)
        x,y,z = x[keep],y[keep],z[keep]
        h = h[keep]
    centers = np.array([z,x,y])
    if return_h:
        centers = np.array([z,x,y,h])
    return centers

def fit_seed_points_base_fast(im,centers,width_z=_sigma_zxy[0],width_xy=_sigma_zxy[1],radius_fit=5,n_max_iter = 10,max_dist_th=0.25, quiet=False):
    if not quiet:
        print("Fitting:" +str(len(centers[0]))+" points")
    z,x,y = centers
    if len(x)>0:

        zxy = np.array([z,x,y],dtype=int).T

        ps = []
        im_subtr = np.array(im,dtype=float)
        for center in zxy:
            p,success = fitsinglegaussian_fixed_width(im_subtr,center,radius=radius_fit,n_approx=5,width_zxy=[width_z,width_xy,width_xy])
            if p is not None:
                ps.append(p)

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
                p,success = fitsinglegaussian_fixed_width(im_add,center,radius=5,n_approx=10,width_zxy=[1.8,1.,1.])
                if p is not None:
                    ps.append(p)
                    ps_1_rem.append(p_1)
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


# fast alignment of fitted items which are bright and sparse (like beads)
def beads_alignment_fast(beads, ref_beads, unique_cutoff=2., check_outlier=True, outlier_sigma=1., verbose=True):
    '''beads_alignment_fast, for finding pairs of beads when they are sparse
    Inputs:
        beads: ndarray of beads coordnates, num_beads by [z,x,y], n-by-3 numpy ndarray
        ref_beads: similar coorndiates for beads in reference frame, n-by-3 numpy ndarray
        unique_cutoff: a threshold that assuming there are only unique pairs within it, float
        check_outlier: whether using Delaunay triangulation neighbors to check
        outlier_sigma: times for sigma that determine threshold in checking outlier, positive float
        verbose: whether say something during alignment, bool
    Outputs:
        _paired_beads: beads that find their pairs in ref frame, n-by-3 numpy array
        _paired_ref_beads: ref_beads that find their pairs (sorted), n-by-3 numpy array
        _shifts: 3d shift of beads (bead - ref_bead), n-by-3 numpy array
        '''
    # initialize
    _paired_beads, _paired_ref_beads, _shifts = [], [], []
    # loop through all beads in ref frame
    for _rb in ref_beads:
        _competing_ref_beads = ref_beads[np.sqrt(np.sum((ref_beads - _rb)**2,1)) < unique_cutoff]
        if len(_competing_ref_beads) > 1: # in this case, other ref_bead exist within cutoff
            continue
        else:
            _candidate_beads = beads[np.sqrt(np.sum((beads - _rb)**2,1)) < unique_cutoff]
            if len(_candidate_beads) == 1: # if unique pairs identified
                _paired_beads.append(_candidate_beads[0])
                _paired_ref_beads.append(_rb)
                _shifts.append(_candidate_beads[0] - _rb)
    # covert to numpy array
    _paired_beads = np.array(_paired_beads)
    _paired_ref_beads = np.array(_paired_ref_beads)
    _shifts = np.array(_shifts)
    # remove suspicious shifts
    for _j in range(np.shape(_shifts)[1]):
        _shift_keeps = np.abs(_shifts)[:,_j] < np.mean(np.abs(_shifts)[:,_j])+outlier_sigma*np.std(np.abs(_shifts)[:,_j])
        # filter beads and shifts
        _paired_beads = _paired_beads[_shift_keeps]
        _paired_ref_beads = _paired_ref_beads[_shift_keeps]
        _shifts = _shifts[_shift_keeps]



    # check outlier
    if check_outlier:
        from scipy.spatial import Delaunay
        from mpl_toolkits.mplot3d import Axes3D
        # initialize list for shifts calculated by neighboring points
        _alter_shifts = []
        # calculate Delaunay triangulation for ref_beads
        _tri = Delaunay(_paired_ref_beads)
        # loop through all beads
        for _i in range(_paired_ref_beads.shape[0]):
            # initialize diff, which used to judge whether keep this
            _keep = True
            # extract shift
            _shift = _shifts[_i]
            # initialize neighboring point ids
            _neighbor_ids = []
            # find neighbors for this point
            for _simplex in _tri.simplices.copy():
                if _i in _simplex:
                    _neighbor_ids.append(_simplex)
            _neighbor_ids = np.array(np.unique(_neighbor_ids).astype(np.int))
            _neighbor_ids = _neighbor_ids[_neighbor_ids != _i] # remove itself
            _neighbor_ids = _neighbor_ids[_neighbor_ids != -1] # remove error
            # calculate alternative shift
            _neighbors = _paired_ref_beads[_neighbor_ids,:]
            _neighbor_shifts = _shifts[_neighbor_ids,:]
            _neighbor_weights = 1/np.sqrt(np.sum((_neighbors-_paired_ref_beads[_i])**2,1))
            _alter_shift = np.dot(_neighbor_shifts.T, _neighbor_weights) / np.sum(_neighbor_weights)
            _alter_shifts.append(_alter_shift)
            #print _i,  _alter_shift, _shift
        # differences between shifts and alternative shifts
        _diff = [np.linalg.norm(_shift-_alter_shift) for _shift,_alter_shift in zip(_shifts, _alter_shifts)]
        # determine whether keep this:
        print('-- differences in original drift and neighboring dirft:', _diff, np.mean(_diff), np.std(_diff))
        _keeps = np.array(_diff < np.mean(_diff)+np.std(_diff)*outlier_sigma, dtype=np.bool)
        # filter beads and shifts
        _paired_beads = _paired_beads[_keeps]
        _paired_ref_beads = _paired_ref_beads[_keeps]
        _shifts = _shifts[_keeps]

    return np.array(_paired_beads), np.array(_paired_ref_beads), np.array(_shifts)


class imshow_mark_3d_v2:
    def master_reset(self):
        #self.dic_min_max = {}
        self.class_ids = []
        self.draw_x,self.draw_y,self.draw_z=[],[],[]
        self.coords = list(zip(self.draw_x,self.draw_y,self.draw_z))
        #load vars
        self.load_coords()
        self.set_image()
    def __init__(self,ims,fig=None,image_names=None,rescz=1.,min_max_default = [None,None], given_dic=None,save_file=None,paramaters={}):
        #internalize
        #seeding paramaters
        self.gfilt_size = paramaters.get('gfilt_size',0.75)#first gaussian blur with radius # to avoid false local max from camera fluc
        self.filt_size = paramaters.get('filt_size',3)#local maxima and minima are computed on blocks of size #
        self.th_seed = paramaters.get('th_seed',300.)#keep points when difference between local minima and maxima is more than #
        self.hot_pix_th = paramaters.get('hot_pix_th',0)
        #fitting paramaters
        self.width_z = paramaters.get('width_z',1.8*1.5)#fixed width in z # 1.8 presuposes isotropic pixel size
        self.width_xy = paramaters.get('width_xy',1.)#fixed width in xy
        self.radius_fit = paramaters.get('radius_fit',5)#neibouring of fitting for each seed point

        self.paramaters=paramaters

        self.ims=ims
        self.rescz = rescz
        if image_names is None:
            self.image_names = ['Image '+str(i+1) for i in range(len(ims))]
        else:
            self.image_names = image_names
        self.save_file = save_file
        #define extra vars
        self.dic_min_max = {}
        self.class_ids = []
        self.draw_x,self.draw_y,self.draw_z=[],[],[]
        self.coords = list(zip(self.draw_x,self.draw_y,self.draw_z))
        self.delete_mode = False
        #load vars
        self.load_coords(_given_dic=given_dic)
        #construct images
        self.index_im = 0
        self.im_ = self.ims[self.index_im]
        self.im_xy = np.max(self.im_,axis=0)
        self.im_z = np.max(self.im_,axis=1)
        im_z_len = self.im_z.shape[0]
        indz=np.array(np.round(np.arange(0,im_z_len,self.rescz)),dtype=int)
        self.im_z = self.im_z[indz[indz<im_z_len],...]
        #setup plots
        if fig is None:
            self.f=plt.figure()
        else:
            self.f=fig
        self.ax1,self.ax2 = ImageGrid(self.f, 111, nrows_ncols=(2, 1), axes_pad=0.1)
        self.lxy,=self.ax1.plot(self.draw_x, self.draw_y, 'o',
                              markersize=12,markeredgewidth=1,markeredgecolor='y',markerfacecolor='None')
        self.lz,=self.ax2.plot(self.draw_x, self.draw_z, 'o',
                      markersize=12,markeredgewidth=1,markeredgecolor='y',markerfacecolor='None')
        self.imshow_xy = self.ax1.imshow(self.im_xy,interpolation='nearest',cmap='gray')
        self.imshow_z = self.ax2.imshow(self.im_z,interpolation='nearest',cmap='gray')

        self.min_,self.max_ = min_max_default
        if self.min_ is None: self.min_ = np.min(self.im_)
        if self.max_ is None: self.max_ = np.max(self.im_)
        self.imshow_xy.set_clim(self.min_,self.max_)
        self.imshow_z.set_clim(self.min_,self.max_)

        self.ax1.callbacks.connect('ylim_changed', self.xy_on_lims_change)
        self.ax2.callbacks.connect('ylim_changed', self.z_on_lims_change)
        self.f.suptitle(self.image_names[self.index_im])
        #connect mouse and keyboard
        cid = self.f.canvas.mpl_connect('button_press_event', self.onclick)
        cid2 = self.f.canvas.mpl_connect('key_press_event', self.press)
        cid3 = self.f.canvas.mpl_connect('key_release_event', self.release)
        self.set_image()
        if fig is None:
            plt.show()
    def onclick(self,event):
        if event.button==3:
            #print "click"
            if event.inaxes is self.ax1:
                if self.delete_mode:
                    z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
                    x_,y_,z_ = list(map(np.array,[self.draw_x,self.draw_y,self.draw_z]))
                    #print x_min,x_max,y_min,y_max,z_min,z_max
                    #print x_,y_,z_
                    keep_in_window = (x_>y_min)&(x_<y_max)&(y_>x_min)&(y_<x_max)&(z_>z_min)&(z_<z_max)
                    keep_class = (np.array(self.class_ids)==self.index_im)&(np.isnan(self.draw_x)==False)
                    keep = keep_in_window&keep_class
                    if np.sum(keep)>0:
                        keep_ind = np.arange(len(keep))[keep]
                        coords_xy_class = list(zip(np.array(self.draw_x)[keep],
                                              np.array(self.draw_y)[keep]))
                        difs = np.array(coords_xy_class)-np.array([[event.xdata,event.ydata]])
                        ind_= np.argmin(np.sum(np.abs(difs),axis=-1))
                        self.draw_x.pop(keep_ind[ind_])
                        self.draw_y.pop(keep_ind[ind_])
                        self.draw_z.pop(keep_ind[ind_])
                        self.class_ids.pop(keep_ind[ind_])
                        print(ind_)
                    else:
                        print('test')
                else:
                    if event.xdata is not None and event.ydata is not None:
                        self.draw_x.append(event.xdata)
                        self.draw_y.append(event.ydata)
                        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
                        self.draw_z.append((z_min+z_max)/2.)
                        self.class_ids.append(self.index_im)
            if event.inaxes is self.ax2:
                if event.xdata is not None and event.ydata is not None:
                    z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
                    x_,y_,z_ = list(map(np.array,[self.draw_x,self.draw_y,self.draw_z]))
                    keep_in_window = (x_>y_min)&(x_<y_max)&(y_>x_min)&(y_<x_max)&(z_>z_min)&(z_<z_max)
                    keep_class = (np.array(self.class_ids)==self.index_im)&(np.isnan(self.draw_x)==False)
                    keep = keep_in_window&keep_class
                    if np.sum(keep)>0:
                        keep_ind = np.arange(len(keep))[keep]
                        coords_x = np.array(self.draw_x)[keep]
                        ind_ = np.argmin(np.abs(coords_x-event.xdata))
                        self.draw_z[keep_ind[ind_]]=event.ydata
            self.update_point_plot()
    def press(self,event):
        if event.key== 'd':
            self.index_im = (self.index_im+1)%len(self.ims)
            self.set_image()
        if event.key== 'a':
            self.index_im = (self.index_im-1)%len(self.ims)
            self.set_image()
        if event.key=='s':
            self.save_ims()
        if event.key== 'x':
            self.auto_scale()
        if event.key== 't':
            self.get_seed_points()
        if event.key== 'n':
            self.handle_in_nucleus()
        if event.key== 'q':
            prev_im = self.index_im
            for self.index_im in range(len(self.ims)):
                self.set_image()
                self.get_seed_points()
                self.fit_seed_points()
            self.index_im = prev_im
            self.set_image()
        if event.key== 'y':
            self.fit_seed_points()
        if event.key == 'delete':
            self.draw_x.pop(-1)
            self.draw_y.pop(-1)
            self.draw_z.pop(-1)
            self.class_ids.pop(-1)
            self.update_point_plot()
        if event.key == 'shift':
            self.delete_mode = True
    def release(self, event):
        if event.key == 'shift':
            self.delete_mode = False
    def populate_draw_xyz(self,flip=False):
        if len(self.coords)>0:
            self.draw_x,self.draw_y,self.draw_z = list(zip(*self.coords))
            if flip: self.draw_x,self.draw_y,self.draw_z =  list(map(list,[self.draw_y,self.draw_x,self.draw_z]))
            else: self.draw_x,self.draw_y,self.draw_z =  list(map(list,[self.draw_x,self.draw_y,self.draw_z]))
        else:
            self.draw_x,self.draw_y,self.draw_z = [],[],[]
    def create_text(self):
        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
        self.texts = []
        i_ims = np.zeros(len(self.ims),dtype=int)
        for (xyz,c_id) in zip(self.coords,self.class_ids):
            i_ims[c_id]+=1
            if c_id==self.index_im:
                if not np.isnan(xyz[0]):
                    if z_min<xyz[2] and z_max>xyz[2] and y_min<xyz[0] and y_max>xyz[0] and x_min<xyz[1] and x_max>xyz[1]:
                        text_ = str(i_ims[c_id])
                        color_='r'
                        if hasattr(self,'dec_text'):
                            key_dec = tuple(list(np.array(xyz,dtype=int))+[c_id])
                            if key_dec in self.dec_text:
                                text_=self.dec_text[key_dec]['text']
                                color_='b'
                        self.texts.append(self.ax1.text(xyz[0],xyz[1],text_,color=color_))
                        self.texts.append(self.ax2.text(xyz[0],xyz[2],text_,color=color_))
    def update_point_plot(self):
        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()

        self.coords = list(zip(self.draw_x,self.draw_y,self.draw_z))
        x_,y_,z_ = list(map(np.array,[self.draw_x,self.draw_y,self.draw_z]))
        #print x_min,x_max,y_min,y_max,z_min,z_max
        #print x_,y_,z_
        keep_class = np.array(self.class_ids)==self.index_im
        keep_in_window = (x_>y_min)&(x_<y_max)&(y_>x_min)&(y_<x_max)&(z_>z_min)&(z_<z_max)
        keep = keep_class&keep_in_window
        self.lxy.set_xdata(x_[keep])
        self.lxy.set_ydata(y_[keep])
        self.lz.set_xdata(x_[keep])
        self.lz.set_ydata(z_[keep])
        self.save_coords()
        self.remove_text()
        self.create_text()
        self.f.canvas.draw()
    def remove_text(self):
        if not hasattr(self,'texts'): self.texts = []
        for txt in self.texts:
            txt.remove()
    def load_coords(self, _given_dic=None):
        save_file = self.save_file
        if _given_dic:
            save_dic = _given_dic
        elif save_file is not None and os.path.exists(save_file):
            with open(save_file,'rb') as fid:
                save_dic = pickle.load(fid)
        else:
            return False
        # load information from save_dic
        self.coords,self.class_ids = save_dic['coords'],save_dic['class_ids']
        if 'pfits' in save_dic:
            self.pfits_save = save_dic['pfits']
        if 'dec_text' in save_dic:
            self.dec_text=save_dic['dec_text']
        self.populate_draw_xyz()#coords to plot list

    def save_coords(self):
        save_file = self.save_file
        if save_file is not None:
            if not os.path.exists(os.path.dirname(save_file)):
                os.makedirs(os.path.dirname(save_file))
            fid = open(save_file,'wb')
            self.pfits_save = getattr(self,'pfits_save',{})
            self.dec_text = getattr(self,'dec_text',{})
            save_dic = {'coords':self.coords,'class_ids':self.class_ids,'pfits':self.pfits_save,'dec_text':self.dec_text}
            pickle.dump(save_dic,fid)
            fid.close()
    def auto_scale(self):
        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
        im_chop = self.im_[z_min:z_max,x_min:x_max,y_min:y_max,...]
        min_,max_ = np.min(im_chop),np.max(im_chop)
        self.imshow_xy.set_clim(min_,max_)
        self.imshow_z.set_clim(min_,max_)
        self.dic_min_max[self.index_im] = [min_,max_]
        self.f.canvas.draw()
    def del_ext(self,str_):
        "Deletes extention"
        if os.path.basename(str_).count('.')>0:
            return '.'.join(str_.split('.')[:-1])
        else:
            return str_
    def save_ims(self):
        import scipy.misc
        save_file = self.save_file
        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
        for index_im,im_ in enumerate(self.ims):
            im_chop = im_[self.get_z_ind(),x_min:x_max,y_min:y_max,...]
            im_xy = np.max(im_chop,axis=0)
            im_z = np.max(im_chop,axis=1)

            if index_im in self.dic_min_max:
                min_,max_ = self.dic_min_max[index_im]
                im_xy = minmax(im_xy,min_=min_,max_=max_)
                im_z = minmax(im_z,min_=min_,max_=max_)
            else:
                min_,max_ = self.min_,self.max_
                im_xy = minmax(im_xy,min_=min_,max_=max_)
                im_z = minmax(im_z,min_=min_,max_=max_)
            if save_file is not None:
                if not os.path.exists(os.path.dirname(save_file)):
                    os.makedirs(os.path.dirname(save_file))
                save_image = self.del_ext(save_file)+'_'+self.image_names[index_im]
                scipy.misc.imsave(save_image+'_xy.png', im_xy)
                scipy.misc.imsave(save_image+'_z.png', im_z)

    def set_image(self):
        self.im_ = self.ims[self.index_im]
        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
        self.im_sm = self.im_[z_min:z_max,x_min:x_max,y_min:y_max]
        self.im_xy = np.max(self.im_[z_min:z_max,:,...],axis=0)
        self.imshow_xy.set_data(self.im_xy)

        self.im_z = np.max(self.im_[:,x_min:x_max,...],axis=1)
        self.im_z = self.im_z[self.get_z_ind(),:]
        self.imshow_z.set_data(self.im_z)

        if self.index_im in self.dic_min_max:
            min_,max_ = self.dic_min_max[self.index_im]
            self.imshow_xy.set_clim(min_,max_)
            self.imshow_z.set_clim(min_,max_)
        self.update_point_plot()
        self.f.suptitle(self.image_names[self.index_im])
        self.f.canvas.draw()
    def get_limits(self):
        y_min,y_max = self.ax1.get_xlim()
        x_min,x_max = self.ax1.get_ylim()[::-1]
        x_min = max(int(x_min),0)
        x_max = min(int(x_max),self.im_.shape[1])
        y_min = max(int(y_min),0)
        y_max = min(int(y_max),self.im_.shape[2])

        z_min,z_max = np.array(self.ax2.get_ylim()[::-1])*self.rescz
        z_min = max(int(z_min),0)
        z_max = min(int(z_max),self.im_.shape[0])
        return z_min,z_max,x_min,x_max,y_min,y_max
    def get_z_ind(self):
        im_z_len = self.im_z.shape[0]
        indz=np.array(np.round(np.arange(0,im_z_len,self.rescz)),dtype=int)
        return indz[indz<im_z_len]
    def xy_on_lims_change(self,ax):
        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
        self.im_sm = self.im_[z_min:z_max,x_min:x_max,y_min:y_max]
        self.im_z = np.max(self.im_[:,x_min:x_max,...],axis=1)
        self.im_z = self.im_z[self.get_z_ind(),:]
        self.imshow_z.set_data(self.im_z)
        self.update_point_plot()
    def z_on_lims_change(self,ax):
        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
        self.im_sm = self.im_[z_min:z_max,x_min:x_max,y_min:y_max]
        self.im_xy = np.max(self.im_[z_min:z_max,:,...],axis=0)
        self.imshow_xy.set_data(self.im_xy)
        self.update_point_plot()
    def fit_seed_points(self):
        #get default paramaters from self
        width_z = self.width_z
        width_xy = self.width_xy
        radius_fit = self.radius_fit
        im = self.im_sm
        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
        y_,x_,z_ = list(map(np.array,[self.draw_x,self.draw_y,self.draw_z]))
        keep_class = np.array(self.class_ids)==self.index_im
        keep_in_window = (x_>x_min)&(x_<x_max)&(y_>y_min)&(y_<y_max)&(z_>z_min)&(z_<z_max)
        keep = keep_class&keep_in_window
        xyzguess = np.array([z_[keep]-z_min,x_[keep]-x_min,y_[keep]-y_min],dtype=int)

        self.pfits = fit_seed_points_base(im,xyzguess,width_z=width_z,width_xy=width_xy,radius_fit=3,n_max_iter = 15,max_dist_th=0.25)
        if len(self.pfits>0):
            self.pfits[:,1:4]+=[[z_min,x_min,y_min]]
            #update graph and points
            keep = np.array(self.class_ids)!=self.index_im
            self.class_ids,self.draw_z,self.draw_x,self.draw_y = [list(np.array(x)[keep]) for x in [self.class_ids,self.draw_z,self.draw_x,self.draw_y]]
            if not hasattr(self,'pfits_save'):
                self.pfits_save={}
            self.pfits_save[self.index_im]=self.pfits
            centers_0,centers_1,centers_2 = self.pfits[:,1:4].T
            self.draw_z.extend(centers_0)
            self.draw_x.extend(centers_2)
            self.draw_y.extend(centers_1)
            self.class_ids.extend([self.index_im]*len(centers_0))
        self.update_point_plot()
    def get_seed_points(self):
        #get default paramaters from self
        gfilt_size = self.gfilt_size
        filt_size = self.filt_size
        th_seed = self.th_seed
        hot_pix_th = self.hot_pix_th

        im = self.im_sm

        centers = get_seed_points_base(im,gfilt_size=gfilt_size,filt_size=filt_size,th_seed=th_seed,hot_pix_th=hot_pix_th)

        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
        keep = np.array(self.class_ids)!=self.index_im
        self.class_ids,self.draw_z,self.draw_x,self.draw_y = [list(np.array(x)[keep]) for x in [self.class_ids,self.draw_z,self.draw_x,self.draw_y]]
        self.draw_z.extend(centers[0]+z_min)
        self.draw_x.extend(centers[2]+y_min)
        self.draw_y.extend(centers[1]+x_min)
        self.class_ids.extend([self.index_im]*len(centers[0]))
        self.update_point_plot()
    def handle_in_nucleus(self):
        if hasattr(self,'nucl_x'):
            i_im = self.index_im
            class_ids = np.array(self.class_ids)
            Y,X,Z = np.array(self.draw_x,dtype=int),np.array(self.draw_y,dtype=int),np.array(self.draw_z,dtype=int)
            keep = class_ids==i_im
            Y,X,Z=Y[keep],X[keep],Z[keep]
            nucl_ = np.array([self.nucl_x,self.nucl_y,self.nucl_z],dtype=int).T
            draw_x,draw_y,draw_z=[],[],[]
            for x,y,z in zip(X,Y,Z):
                if np.any(np.sum(np.abs(nucl_-[[x,y,z]]),axis=-1)==0):
                    draw_z.append(z)
                    draw_x.append(y)
                    draw_y.append(x)
            keep = np.array(self.class_ids)!=self.index_im
            self.class_ids,self.draw_z,self.draw_x,self.draw_y = [list(np.array(x)[keep]) for x in [self.class_ids,self.draw_z,self.draw_x,self.draw_y]]
            self.draw_z.extend(draw_z)
            self.draw_x.extend(draw_x)
            self.draw_y.extend(draw_y)
            self.class_ids.extend([self.index_im]*len(draw_x))
            self.update_point_plot()

class Reader:

    # Close the file on cleanup.
    def __del__(self):
        if self.fileptr:
            self.fileptr.close()

    def __enter__(self):
        return self

    def __exit__(self, etype, value, traceback):
        if self.fileptr:
            self.fileptr.close()

    # Average multiple frames in a movie.
    def averageFrames(self, start = False, end = False, verbose = False):
        if (not start):
            start = 0
        if (not end):
            end = self.number_frames

        length = end - start
        average = np.zeros((self.image_width, self.image_height), np.float)
        for i in range(length):
            if verbose and ((i%10)==0):
                print(" processing frame:", i, " of", self.number_frames)
            average += self.loadAFrame(i + start)

        average = average/float(length)
        return average

    # returns the film name
    def filmFilename(self):
        return self.filename

    # returns the film size
    def filmSize(self):
        return [self.image_width, self.image_height, self.number_frames]

    # returns the picture x,y location, if available
    def filmLocation(self):
        if hasattr(self, "stage_x"):
            return [self.stage_x, self.stage_y]
        else:
            return [0.0, 0.0]

    # returns the film focus lock target
    def lockTarget(self):
        if hasattr(self, "lock_target"):
            return self.lock_target
        else:
            return 0.0

    # returns the scale used to display the film when
    # the picture was taken.
    def filmScale(self):
        if hasattr(self, "scalemin") and hasattr(self, "scalemax"):
            return [self.scalemin, self.scalemax]
        else:
            return [100, 2000]

# Dax reader class. This is a Zhuang lab custom format.
#

def batch_load_dax(filename):
    _im = DaxReader(filename).loadAll()
    return _im

class DaxReader(Reader):
    # dax specific initialization
    def __init__(self, filename, swap_axis=False, verbose = 0):
        import os,re
        # save the filenames
        self.filename = filename
        dirname = os.path.dirname(filename)
        if (len(dirname) > 0):
            dirname = dirname + "/"
        self.inf_filename = dirname + os.path.splitext(os.path.basename(filename))[0] + ".inf"
        # swap_axis
        self.swap_axis = swap_axis

        # defaults
        self.image_height = None
        self.image_width = None

        # extract the movie information from the associated inf file
        size_re = re.compile(r'frame dimensions = ([\d]+) x ([\d]+)')
        length_re = re.compile(r'number of frames = ([\d]+)')
        endian_re = re.compile(r' (big|little) endian')
        stagex_re = re.compile(r'Stage X = ([\d\.\-]+)')
        stagey_re = re.compile(r'Stage Y = ([\d\.\-]+)')
        lock_target_re = re.compile(r'Lock Target = ([\d\.\-]+)')
        scalemax_re = re.compile(r'scalemax = ([\d\.\-]+)')
        scalemin_re = re.compile(r'scalemin = ([\d\.\-]+)')

        inf_file = open(self.inf_filename, "r")
        while 1:
            line = inf_file.readline()
            if not line: break
            m = size_re.match(line)
            if m:
                self.image_height = int(m.group(1))
                self.image_width = int(m.group(2))
            m = length_re.match(line)
            if m:
                self.number_frames = int(m.group(1))
            m = endian_re.search(line)
            if m:
                if m.group(1) == "big":
                    self.bigendian = 1
                else:
                    self.bigendian = 0
            m = stagex_re.match(line)
            if m:
                self.stage_x = float(m.group(1))
            m = stagey_re.match(line)
            if m:
                self.stage_y = float(m.group(1))
            m = lock_target_re.match(line)
            if m:
                self.lock_target = float(m.group(1))
            m = scalemax_re.match(line)
            if m:
                self.scalemax = int(m.group(1))
            m = scalemin_re.match(line)
            if m:
                self.scalemin = int(m.group(1))

        inf_file.close()

        # set defaults, probably correct, but warn the user
        # that they couldn't be determined from the inf file.
        if not self.image_height:
            print("Could not determine image size, assuming 256x256.")
            self.image_height = 256
            self.image_width = 256

        # open the dax file
        if os.path.exists(filename):
            self.fileptr = open(filename, "rb")
        else:
            self.fileptr = 0
            if verbose:
                print("dax data not found", filename)

    # Create and return a memory map the dax file
    def loadMap(self):
        if os.path.exists(self.filename):
            if self.bigendian:
                self.image_map = np.memmap(self.filename, dtype='>u2', mode='r', shape=(self.number_frames,self.image_width, self.image_height))
            else:
                self.image_map = np.memmap(self.filename, dtype='uint16', mode='r', shape=(self.number_frames,self.image_width, self.image_height))
        return self.image_map

    # load a frame & return it as a np array
    def loadAFrame(self, frame_number):
        if self.fileptr:
            assert frame_number >= 0, "frame_number must be greater than or equal to 0"
            assert frame_number < self.number_frames, "frame number must be less than " + str(self.number_frames)
            self.fileptr.seek(frame_number * self.image_height * self.image_width * 2)
            image_data = np.fromfile(self.fileptr, dtype='uint16', count = self.image_height * self.image_width)
            if self.swap_axis:
                image_data = np.transpose(np.reshape(image_data, [self.image_width, self.image_height]))
            else:
                image_data = np.reshape(image_data, [self.image_width, self.image_height])
            if self.bigendian:
                image_data.byteswap(True)
            return image_data
    # load full movie and retun it as a np array
    def loadAll(self):
        image_data = np.fromfile(self.fileptr, dtype='uint16', count = -1)
        if self.swap_axis:
            image_data = np.swapaxes(np.reshape(image_data, [self.number_frames,self.image_width, self.image_height]),1,2)
        else:
            image_data = np.reshape(image_data, [self.number_frames,self.image_width, self.image_height])
        if self.bigendian:
            image_data.byteswap(True)
        return image_data
    
    def close(self):
        if self.fileptr.closed:
            print(f"file {self.filename} has been closed.")
        else:
            self.fileptr.close()

## segmentation with DAPI
def DAPI_segmentation(ims, names,
                      cap_percentile=0.5,
                      illumination_correction=True,
                      illumination_correction_channel=405,
                      correction_folder=_correction_folder,
                      merge_layer_num = 11,
                      denoise_window = 5,
                      log_window = 13,
                      signal_cap_ratio = 0.15,
                      cell_min_size=1000,
                      shape_ratio_threshold = 0.030,
                      remove_fov_boundary = 40,
                      make_plot=False,
                      verbose=True):
    """cell segmentation for DAPI images with pooling and convolution layers
    Inputs:
        ims: list of images
        names: list of names, same length as ims
        cap_percentile: removing top and bottom percentile in each image, float from 0-100 (default: 0.5)
        illumination_correction: whether correct illumination for each field of view, bool (default: True)
        illumination_correction_channel: which color channel to correct illumination for each field of view, int or str (default: 405)
        correction_folder: full directory that contains such correction files, string (default: )
        merge_layer_num: number of z-stack layers to merge, int (default: 11)
        denoise_window: window size used for billateral denoising method, int (default: 31)
        log_window: window size for laplacian-gaussian filter, int (default: 13)
        signal_cap_ratio: intensity ratio that considered as signal if intensity over max intensity larger than this, float between 0-1, (default: 0.15)
        cell_min_size: smallest object size allowed as nucleus, int (default:1000 for 2D)
        shape_ratio_threshold: min threshold for: areasize of one label / (contour length of a label)^2, float (default: 0.15)
        remove_fov_boundary: if certain label is too close to fov boundary within this number of pixels, remove, int (default: 50)
        make_plot: whether making plots for checking purpose, bool
        verbose: whether say something during the process, bool
    Output:
        _ft_seg_labels: list of labels, same dimension as ims, list of bool matrix"""
    # imports
    from scipy import ndimage
    from skimage import morphology
    from scipy import stats
    from skimage import restoration, measure
    from ImageAnalysis3.corrections import Illumination_correction
    from skimage.segmentation import random_walker
    from scipy.ndimage import gaussian_laplace

    # check whether input is a list of images or just one image
    if isinstance(ims, list):
        if verbose:
            print("Start segmenting list of images")
        _ims = ims
        _names = names
    else:
        if verbose:
            print("Start segmenting one image")
        _ims = [ims]
        _names = [names]

    # check input length
    if len(_names) != len(_ims):
        raise ValueError('input images and names length not compatible!')

    # illumination correction
    if illumination_correction:
        _ims = [corrections.Illumination_correction(_im, illumination_correction_channel, 
                                                    correction_folder=correction_folder,
                                                    verbose=verbose) for _im in _ims]

    # rescale image to 0-1 gray scale
    _limits = [stats.scoreatpercentile(_im, (cap_percentile, 100.-cap_percentile)).astype(np.float) for _im in _ims]
    _norm_ims = [(_im-np.min(_limit))/(np.max(_limit)-np.min(_limit)) for _im,_limit in zip(_ims, _limits)]
    for _im in _norm_ims:
        _im[_im < 0] = 0
        _im[_im > 1] = 1

    # find the layer that on focus
    _focus_layers = [np.argmin(np.array([np.sum(_layer > signal_cap_ratio) for _layer in _im])) for _im in _norm_ims]

    # stack images close to this focal layer
    if verbose:
        print('- find focal plane and slice')
    _stack_ims = []
    for _im, _layer in zip(_norm_ims, _focus_layers):
        if _im.shape[0] - _layer < np.ceil((merge_layer_num-1)/2):
            _stack_lims = [_im.shape[0]-merge_layer_num, _im.shape[0]]
        elif _layer < np.floor((merge_layer_num-1)/2):
            _stack_lims = [0, merge_layer_num]
        else:
            _stack_lims = [_layer-np.ceil((merge_layer_num-1)/2), _layer+np.floor((merge_layer_num-1)/2)]
        _stack_lims = np.array(_stack_lims, dtype=np.int)
        # extract image
        _stack_im = np.zeros([np.max(_stack_lims)-np.min(_stack_lims), np.shape(_im)[1], np.shape(_im)[2]])
        # denoise and merge
        if denoise_window:
            for _i,_l in enumerate(range(np.min(_stack_lims), np.max(_stack_lims))):
                _stack_im[_i] = restoration.denoise_bilateral(_im[_l], win_size=int(denoise_window), mode='edge', multichannel=False)
        else:
            for _i,_l in enumerate(range(np.min(_stack_lims), np.max(_stack_lims))):
                _stack_im[_i] = _im[_l]

        _stack_im = np.mean(_stack_im, axis=0)
        _stack_ims.append(_stack_im)

    # laplace of gaussian filter
    if verbose:
        print("- apply by laplace-of-gaussian filter")
    _conv_ims = [gaussian_laplace(_im, log_window) for _im in _stack_ims]

    # binarilize the image
    _supercell_masks = [(_cim < -1e-6) *( _sim > signal_cap_ratio) for _cim, _sim in zip(_conv_ims, _stack_ims)]
    _supercell_masks = [ndimage.binary_dilation(_im, structure=morphology.disk(4)) for _im in _supercell_masks]
    _supercell_masks = [ndimage.binary_erosion(_im, structure=morphology.disk(12)) for _im in _supercell_masks]
    _supercell_masks = [ndimage.binary_fill_holes(_im, structure=morphology.disk(3)) for _im in _supercell_masks]

    # acquire labels
    if verbose:
        print("- acquire labels")
    _open_objects = [morphology.opening(_im, morphology.disk(3)) for _im in _supercell_masks]
    _close_objects = [morphology.closing(_open, morphology.disk(3)) for _open in _open_objects]
    _close_objects = [morphology.remove_small_objects(_close, 2000) for _close in _close_objects]
    _bboxes = [ndimage.find_objects(_close) for _close in _close_objects]
    _masks = [_close[_bbox[0]] for _bbox, _close in zip(_bboxes, _close_objects)]
    _labels = []
    for _close,_sim in zip(_close_objects,_stack_ims):
        _label, _num = ndimage.label(_close)
        _label[(_sim > signal_cap_ratio)*(_label==0)] = 0
        _label[(_sim <= signal_cap_ratio)*(_label==0)] = -1
        _labels.append(_label)

    # random walker segmentation
    if verbose:
        print ("- random walker segmentation!")
    _seg_labels = [random_walker(_im, _label, beta=1, mode='bf') for _im, _label in zip(_stack_ims, _labels)]

    # remove bad labels by shape ratio: A(x)/I(x)^2
    if verbose:
        print ("- remove failed labels by shape ratio: A(x)/I(x)^2")
    _ft_seg_labels = []
    _contours = []
    for _i, _seg_label in enumerate(_seg_labels):
        if verbose:
            print ("- screen labels in field of view:", names[_i])
        _failed_labels = []
        for _l in range(np.max(_seg_label)):
            _contour = measure.find_contours(np.array(_seg_label==_l+1, dtype=np.int), 0)[0]
            _length = np.sum(np.sqrt(np.sum((_contour[1:] - _contour[:-1])**2, axis=1)))
            _size = np.sum(_seg_label==_l+1)
            _center = np.round(ndimage.measurements.center_of_mass(_seg_label==_l+1))
            _shape_ratio = _size/_length**2
            if _shape_ratio < shape_ratio_threshold:

                _seg_label[_seg_label==_l+1] = -1
                _failed_labels.append(_l+1)
                if verbose:
                    print("-- fail by shape_ratio, label", _l+1, 'contour length:', _length, 'size:', _size, 'shape_ratio:',_size/_length**2)
                continue
            for _coord,_dim in zip(_center[-2:], _seg_label.shape[-2:]):
                if _coord < remove_fov_boundary or _coord > _dim - remove_fov_boundary:
                    _seg_label[_seg_label==_l+1] = -1
                    _failed_labels.append(_l+1)
                    if verbose:
                        print("-- fail by center_coordinate, label:", _l+1, "center of this nucleus:", _center[-2:])
                    break

        _lb = 1
        while _lb <= np.max(_seg_label):
            if np.sum(_seg_label == _lb) == 0:
                print ("-- remove", _lb)
                _seg_label[_seg_label>_lb] -= 1
            else:
                print ("-- pass", _lb)
                _lb += 1

        _ft_seg_labels.append(_seg_label)
    # plot
    if make_plot:
        for _seg_label, _name in zip(_ft_seg_labels, _names):
            plt.figure()
            plt.imshow(_seg_label)
            plt.title(_name)
            plt.colorbar()
            plt.show()

    # return segmentation results
    return _ft_seg_labels


# segmentation with convolution of DAPI images

def DAPI_convoluted_segmentation(filenames, correction_channel=405, 
        num_threads=12, cap_percentile=1,
        single_im_size=_image_size, all_channels=_allowed_colors, 
        num_buffer_frames=10, num_empty_frames=1, 
        illumination_correction=True, illumination_correction_channel=405, 
        correction_folder=_correction_folder,
        merge_layer_num=11, denoise_window=5, mft_size=25, glft_size=30,
        max_conv_th=0, min_boundary_th=0.48, signal_cap_ratio=0.20,
        max_cell_size=40000, min_cell_size=5000, min_shape_ratio=0.035,
        max_iter=4, shrink_percent=15,
        dialation_dim=4, random_walker_beta=0.1, remove_fov_boundary=50,
        save=True, save_folder=None, force=False, 
        save_npy=True, save_postfix="_segmentation",
        make_plot=False, return_images=False, verbose=True):
    """cell segmentation for DAPI images with pooling and convolution layers
    Inputs:
        ims: list of images
        names: list of names, same length as ims
        cap_percentile: removing top and bottom percentile in each image, float from 0-100 (default: 0.5)
        num_buffer_frames: number of buffer frames, int
        num_empty_frames: num of empty frames, int
        illumination_correction: whether correct illumination for each field of view, bool (default: True)
        illumination_correction_channel: which color channel to correct illumination for each field of view, int or str (default: 405)
        correction_folder: full directory that contains such correction files, string (default: )
        merge_layer_num: number of z-stack layers to merge, int (default: 11)
        denoise_window: window size used for billateral denoising method, int (default: 31)
        mft_size: size of max-min filters to get cell boundaries, int (default: 25)
        glft_size: window size for laplacian-gaussian filter, int (default: 35)
        binarilize image:
        max_conv_th: maximum convolution threshold, float(default: -5e-5)
        min_boundary_th: minimal boundary im threshold, float(default: 0.55)
        signal_cap_ratio: intensity ratio that considered as signal if intensity over max intensity larger than this, float between 0-1, (default: 0.15)
        max_cell_size: upper limit for object otherwise undergoes extra screening, int(default: 30000)
        min_cell_size: smallest object size allowed as nucleus, int (default:5000 for 2D)
        min_shape_ratio: min threshold for: areasize of one label / (contour length of a label)^2, float (default: 0.15)
        max_iter: maximum iterations allowed in splitting shapes, int (default:3)
        shrink_percent: percentage of label areas removed during splitting, float (0-100, default: 13)
        dialation_dim: dimension for dialation after splitting objects, int (default:4)
        random_walker_beta: beta used for random walker segementation algorithm, float (default: 0.1)
        remove_fov_boundary: if certain label is too close to fov boundary within this number of pixels, remove, int (default: 50)
        make_plot: whether making plots for checking purpose, bool
        verbose: whether say something during the process, bool
    Output:
        _seg_labels: list of labels, same dimension as ims, list of bool matrix"""
    ## import images
    if not isinstance(filenames, list):
        filenames = [filenames]
    ## load segmentation if already existed:
    if save_folder is None:
        save_folder = os.path.dirname(os.path.dirname(filenames[0]))
        save_folder = os.path.join(save_folder, 'Analysis', 'segmentation')
    if not os.path.exists(save_folder): # create folder if not exists
        os.makedirs(save_folder)
    if save_npy:
        save_filenames = [os.path.join(save_folder, os.path.basename(_fl).replace('.dax', save_postfix +'.npy')) for _fl in filenames]    
    else:
        save_filenames = [os.path.join(save_folder, os.path.basename(_fl).replace('.dax', save_postfix +'.pkl')) for _fl in filenames]
    # decide if directly load
    _direct_load_flags = [True for _fl in save_filenames if os.path.exists(_fl) and not force]
    if len(_direct_load_flags) == len(filenames) and not force:
        if verbose:
            if len(filenames) == 1:
                print(f"-- directly load segmentation result from:{save_filenames[0]}")
            else:
                print(f"-- directly load segmentation result from folder:{save_folder}, load_npy:{save_npy}")
        # load segmentation labels
        if save_npy:
            _seg_labels = [np.load(_fl) for _fl in save_filenames]
        else:
            _seg_labels = [pickle.load(open(_fl, 'rb')) for _fl in save_filenames]
        # return    
        if return_images:
            if verbose:
                print(f"- loading {len(filenames)} images for output")
            _load_args = [(_fl, correction_channel, None, None, 20,
                           single_im_size, all_channels, 
                           num_buffer_frames,num_empty_frames,
                           np.zeros(3), correction_folder) for _fl in filenames]
            _load_pool = mp.Pool(num_threads)
            _ims = _load_pool.starmap(corrections.correct_single_image, _load_args, chunksize=1)
            _load_pool.close()
            _load_pool.join()
            _load_pool.terminate()
            return _seg_labels, _ims
        else:
            return _seg_labels
    else:
        if verbose:
            print(f"- loading {len(filenames)} images for segmentation")
        _load_args = [(_fl, correction_channel, None, None, 20,
                       single_im_size, all_channels, 
                       num_buffer_frames,num_empty_frames,
                       np.zeros(3), correction_folder) for _fl in filenames]        
        _load_pool = mp.Pool(num_threads)
        _ims = _load_pool.starmap(corrections.correct_single_image, _load_args, chunksize=1)
        _load_pool.close()
        _load_pool.join()
        _load_pool.terminate()
    ## rescaling and stack
    # rescale image to 0-1 gray scale
    _limits = [stats.scoreatpercentile(_im, (cap_percentile, 100.-cap_percentile)).astype(np.float) for _im in _ims]
    _norm_ims = [(_im-np.min(_limit))/(np.max(_limit)-np.min(_limit)) for _im,_limit in zip(_ims, _limits)]
    for _im in _norm_ims:
        _im[_im < 0] = 0
        _im[_im > 1] = 1
    # find the layer that on focus
    _focus_layers = [np.argmin(np.array([np.sum(_layer > signal_cap_ratio) for _layer in _im])) for _im in _norm_ims]

    # stack images close to this focal layer
    if verbose:
        print('-- find focal plane and slice')
    _stack_ims = []
    for _im, _layer in zip(_norm_ims, _focus_layers):
        if _im.shape[0] - _layer < np.ceil((merge_layer_num-1)/2):
            _stack_lims = [_im.shape[0]-merge_layer_num, _im.shape[0]]
        elif _layer < np.floor((merge_layer_num-1)/2):
            _stack_lims = [0, merge_layer_num]
        else:
            _stack_lims = [_layer-np.ceil((merge_layer_num-1)/2), _layer+np.floor((merge_layer_num-1)/2)]
        _stack_lims = np.array(_stack_lims, dtype=np.int)
        # extract image
        _stack_im = np.zeros([np.max(_stack_lims)-np.min(_stack_lims), np.shape(_im)[1], np.shape(_im)[2]])
        # denoise and merge
        if denoise_window:
            for _i,_l in enumerate(range(np.min(_stack_lims), np.max(_stack_lims))):
                _stack_im[_i] = restoration.denoise_bilateral(_im[_l], win_size=int(denoise_window), mode='edge', multichannel=False)
        else:
            for _i,_l in enumerate(range(np.min(_stack_lims), np.max(_stack_lims))):
                _stack_im[_i] = _im[_l]
        _stack_im = np.mean(_stack_im, axis=0)
        _stack_ims.append(_stack_im)

    ## Get boundaries of cells and apply Gaussian-Laplacian filter
    # get boundaries of cells
    _diff_ims = [2*ndimage.filters.maximum_filter(_stack_im, mft_size)-ndimage.filters.minimum_filter(_stack_im, mft_size) for _stack_im in _stack_ims]
    # laplace of gaussian filter
    if verbose:
        print("- apply by laplace-of-gaussian filter")
    _conv_ims = [gaussian_laplace(_im, glft_size) for _im in _diff_ims]
        
    ## get rough labels
    # binarilize the image
    _supercell_masks = [(_cim < max_conv_th) *( _sim > min_boundary_th) for _cim, _sim in zip(_conv_ims, _diff_ims)]
    # erosion and dialation
    _supercell_masks = [ndimage.binary_erosion(_im, structure=morphology.disk(3)) for _im in _supercell_masks]
    _supercell_masks = [ndimage.binary_dilation(_im, structure=morphology.disk(5)) for _im in _supercell_masks]
    # filling holes
    _supercell_masks = [ndimage.binary_fill_holes(_im, structure=morphology.disk(4)) for _im in _supercell_masks]
    # acquire labels
    if verbose:
        print("- acquire labels")
    _open_objects = [morphology.opening(_im, morphology.disk(3)) for _im in _supercell_masks]
    _close_objects = [morphology.closing(_open, morphology.disk(3)) for _open in _open_objects]
    _close_objects = [morphology.remove_small_objects(_close, min_cell_size) for _close in _close_objects]
    # labeling
    _labels = [ np.array(ndimage.label(_close)[0], dtype=np.int) for _close in _close_objects]
    
    ## Tuning labels
    def _label_binary_im(_im, obj_size=3):
        '''Given an binary image, find labels for all isolated objects with given size'''
        # make sure image is binary
        _bim = np.array(_im > 0, dtype=np.int)
        # find objects
        _open = morphology.opening(_bim, morphology.disk(obj_size))
        _close = morphology.closing(_open, morphology.disk(obj_size))
        # label objects
        _label, _num = ndimage.label(_close.astype(bool))
        # return
        return _label, _num

    def _check_label(_label, _id, _min_shape_ratio, _max_size, verbose=False):
        """Check whether the label is qualified as a cell"""
        # get features
        _length,_size,_center,_ratio = _get_label_features(_label, _id)
        if _ratio < _min_shape_ratio:
            if verbose:
                print(f"--- {_ratio} is smaller than minimum shape ratio, failed")
            return False
        if _size > _max_size:
            if verbose:
                print(f"--- {_size} is larger than maximum shape size, failed")
            return False
        return True

    def _get_label_features(_label, _id):
        """Given a label and corresponding label id, return four features of this label"""
        # get features
        _contours = measure.find_contours(np.array(_label==_id, dtype=np.int), 0)
        if len(_contours) > 0:
            _length = np.sum(np.sqrt(np.sum((np.roll(_contours[0],1,axis=0) - _contours[0])**2, axis=1)))
        else:
            _length = 0
        _size = np.sum(_label==_id)
        _center = np.round(ndimage.measurements.center_of_mass(_label==_id))
        _shape_ratio = _size/_length**2
        return _length, _size, _center, _shape_ratio

    def _split_single_label(_stack_im, _conv_im, _label, _id,
                     min_size=min_cell_size, shrink_percent=shrink_percent,
                     erosion_dim=2, dialation_dim=dialation_dim):
        """Function to split suspicious labels and validate"""
        if shrink_percent > 50 or shrink_percent < 0:
            raise ValueError(f"Wrong shrink_percent kwd ({shrink_percent}) is given, should be in [0,50]")
        # get features
        _length,_size,_center,_ratio = _get_label_features(_label, _id)
        if _size < 2*min_size: # adjust shrink percentage if shape is small
            shrink_percent = shrink_percent * 0.8
        _mask = np.array(_label == _id, dtype=np.int)
        _mask *= np.array(_stack_im > stats.scoreatpercentile(_stack_im[_label==_id], shrink_percent), dtype=int)
        #_mask *= np.array(_conv_im < stats.scoreatpercentile(_conv_im[_label==_id], 100-2*shrink_percent), dtype=int)
        _mask = ndimage.binary_erosion(_mask, structure=morphology.disk(erosion_dim))
        _mask = morphology.remove_small_objects(_mask.astype(bool), min_size)
        _new_label, _num = _label_binary_im(_mask, 3)
        for _l in range(_num):
            _single_label = np.array(_new_label==_l+1, dtype=np.int)
            _single_label = ndimage.binary_dilation(_single_label, structure=morphology.disk(int(dialation_dim/2)))
            _new_label[_single_label>0] = _l+1
        return _new_label, _num

    def _iterative_split_labels(_stack_im, _conv_im, _label, max_iter=3,
                                min_shape_ratio=min_shape_ratio, max_size=max_cell_size,
                                min_size=min_cell_size, shrink_percent=15,
                                erosion_dim=2, dialation_dim=10,
                                verbose=False):
        """Function to iteratively split labels within one fov"""
        _single_labels = [np.array(_label==_i+1,dtype=np.int) for _i in range(int(np.max(_label))) if np.sum(np.array(_label==_i+1,dtype=np.int))>0]
        _iter_counts = [0 for _i in range(len(_single_labels))]
        _final_label = np.zeros(np.shape(_label), dtype=np.int)
        # start selecting labels
        while(len(_single_labels)) > 0:
            _sg_label = _single_labels.pop(0)
            _iter_ct = _iter_counts.pop(0)
            if verbose:
                print(f"- Remaining labels:{len(_single_labels)}, iter_num:{_iter_ct}")
            # if this cell passes the filter
            if _check_label(_sg_label, 1, min_shape_ratio, max_size, verbose=verbose):
                if verbose:
                    print(f"-- saving label: {np.max(_final_label)+1}")
                _save_label = ndimage.binary_dilation(_sg_label, structure=morphology.disk(int(dialation_dim/2)))
                _save_label = ndimage.binary_fill_holes(_save_label, structure=morphology.disk(int(dialation_dim/2)))
                
                if np.sum(_save_label==1) > min_size:
                    if verbose:
                        print('save1', _get_label_features(_save_label, 1))
                    _final_label[_save_label==1] = np.max(_final_label)+1
                continue
            # not pass, try to split
            else:
                _new_label, _num = _split_single_label(_stack_im, _conv_im, _sg_label, 1,
                                                       min_size=min_size*(1-shrink_percent/100)**_iter_ct,
                                                       shrink_percent=shrink_percent,
                                                       erosion_dim=erosion_dim, dialation_dim=dialation_dim)
                for _i in range(_num):
                    _cand_label = np.array(_new_label==_i+1, dtype=np.int)
                    if _check_label(_cand_label, 1, min_shape_ratio*0.9**_iter_ct, max_size, verbose=verbose):
                        if verbose:
                            print(f"-- saving label: {np.max(_final_label)+1}")
                        _save_label = ndimage.binary_dilation(_cand_label, structure=morphology.disk(int(dialation_dim/2+1)))
                        _save_label = ndimage.binary_fill_holes(_save_label, structure=morphology.disk(int(dialation_dim/2)))
                        if np.sum(_save_label == 1) > min_size:
                            if verbose:
                                print('save2', _get_label_features(_save_label, 1))
                            _final_label[_save_label==1] = np.max(_final_label)+1
                    elif _iter_ct > max_iter:
                        if verbose:
                            print("--- Exceeding max-iteration count, skip.")
                        continue
                    else:
                        if verbose:
                            print("--- Append this cell back to pool")
                        _single_labels.append(_cand_label)
                        _iter_counts.append(_iter_ct+1)
        return _final_label

    # initialize updated labels and call functions
    if verbose:
        print("- start iterative segmentation")
    _seg_labels = []
    for _i, (_sim, _cim, _label) in enumerate(zip(_stack_ims, _conv_ims, _labels)):
        _updated_label = _iterative_split_labels(_sim, _cim, _label, max_iter=max_iter,
                                                 min_shape_ratio=min_shape_ratio, shrink_percent=shrink_percent,
                                                 max_size=max_cell_size, min_size=min_cell_size,
                                                 dialation_dim=dialation_dim, verbose=verbose)
        for _l in range(int(np.max(_updated_label))):
            _, _, _center, _ = _get_label_features(_updated_label, _l+1)
            if _center[0] < remove_fov_boundary or _center[1] < remove_fov_boundary or _center[0] >= _updated_label.shape[0]-remove_fov_boundary or _center[1] >= _updated_label.shape[1]-remove_fov_boundary:
                if verbose:
                    print(f"-- Remove im:{_i}, label {_l+1} for center coordiate too close to edge.")
                _updated_label[_updated_label==_l+1] = 0
        # relabel
        _relabel_id = 1
        _seg_label = np.zeros(np.shape(_updated_label), dtype=np.int)
        for _l in range(int(np.max(_updated_label))):
            if np.sum(np.array(_updated_label == _l+1,dtype=np.int)) > 0:
                _seg_label[_updated_label==_l+1] = _relabel_id
                _relabel_id += 1
        # label background
        _dialated_mask = ndimage.binary_dilation(np.array(_seg_label>0, dtype=np.int), structure=morphology.disk(int(dialation_dim/2)))
        _seg_label[(_seg_label==0)*(_dialated_mask==0)] = -1
        # save
        _seg_labels.append(_seg_label)

    ## random walker segmentation
    if random_walker_beta:
        if verbose:
            print ("- random walker segmentation!")
        _seg_labels = [random_walker(_im, _label, beta=random_walker_beta, mode='bf') for _im, _label in zip(_stack_ims, _seg_labels)]

    ## plot
    if make_plot:
        for _seg_label, _name in zip(_seg_labels, filenames):
            plt.figure()
            plt.imshow(_seg_label)
            plt.title(_name)
            plt.colorbar()
            plt.show()
    ## save 
    if save:
        if save_npy:
            for _fl, _lb in zip(save_filenames, _seg_labels):
                np.save(_fl, _lb)
        else:
            for _fl, _lb in zip(save_filenames, _seg_labels):
                 pickle.dump(_lb, open(_fl, 'wb'))
    if return_images:
        return _seg_labels, _ims
    else:
        return _seg_labels

# merge images to generate "chromosome"
def generate_chromosome_from_dic(im_dic, merging_channel, color_dic,  bead_label='beads',
                                 merge_num=10, ref_frame=0, fft_dim=125, verbose=True):
    '''Function to generate "chromosomes" by merging first several regions
    Given:
        im_dic: dictionary of images loaded by get_img_info.split_channels_by_image, dic
        merging_channel: use which channel to merge as chromosome, -1 means all channels except beads, int
        color_dic: dictionary of color usage loaded by get_img_info.Load_Color_Usage, dic
        merge_num: number of images to be merged, int (default: 10)
        ref_frame: which frame is used as reference, non-negative int (default: 0)
        fft_dim: dimension for FFT, positive int (default: 125)
        verbose: say something!, bool (default: True)
    Return:
        _mean_im: merged image, 3d-array
        _rough_dfts: drifts calculated by FFT, list of 1d-arrays
    '''
    import numpy as np
    import os
    from ImageAnalysis3.corrections import fast_translate, fftalign
    # initialize mean_image as chromosome
    _mean_im=[]
    _rough_dfts = []
    # get ref frame
    _ref_name = sorted(list(im_dic.items()), key=lambda k_v: int(k_v[0].split('H')[1].split('R')[0]))[ref_frame][0]
    _ref_ims = sorted(list(im_dic.items()), key=lambda k_v1: int(k_v1[0].split('H')[1].split('R')[0]))[ref_frame][1]
    if bead_label not in color_dic[_ref_name.split(os.sep)[0]]:
        raise ValueError('wrong ref frame, no beads exist in this hybe.')
    for _i, _label in enumerate(color_dic[_ref_name.split(os.sep)[0]]):
        # check bead label
        if bead_label == _label:
            _ref_bead = _ref_ims[_i]
            break
    # loop through all images for this field of view
    for _name, _ims in sorted(list(im_dic.items()), key=lambda k_v2: int(k_v2[0].split('H')[1].split('R')[0])):
        if len(_rough_dfts) >= merge_num: # stop if images more than merge_num are already calclulated.
            break
        if _name == _ref_name: # pass the ref frame
            continue
        if bead_label in color_dic[_name.split(os.sep)[0]]:
            #if verbose:
            #    print "processing image:", _name
            # extract bead image
            for _i, _label in enumerate(color_dic[_name.split(os.sep)[0]]):
                # check bead label
                if bead_label == _label:
                    _bead = _ims[_i]
                    break
            # calculate drift fastly with FFT
            _rough_dft = fftalign(_ref_bead, _bead)
            _rough_dfts.append(_rough_dft)
            # roughly align image and save
            if merging_channel >=0 and merging_channel < len(_ims): # if merging_channel is provided properly
                _corr_im = fast_translate(_ims[merging_channel],-_rough_dft)
                _mean_im.append(_corr_im)
            else: # if merging_channel is not provided etc:
                for _i, _label in enumerate(color_dic[_name.split(os.sep)[0]]):
                    if bead_label != _label and _label != '':
                        _corr_im = fast_translate(_ims[_i],-_rough_dft)
                        _mean_im.append(_corr_im)
    if verbose:
        print('- number of images to calculate mean: '+str(len(_mean_im))+'\n- number of FFT drift corrections: '+str(len(_rough_dfts)))
        print("- drifts are: \n", _rough_dfts)
    _mean_im = np.mean(_mean_im,0)

    return _mean_im, _rough_dfts

# crop cells based on DAPI segmentation result
def crop_cell(im, segmentation_label, drift=None, extend_dim=20, overlap_threshold = 0.1, verbose=True):
    '''basic function to crop image into small ones according to segmentation_label
    Inputs:
        im: single nd-image, numpy.ndarray
        segmentation_label: 2D or 3D segmentaiton label, each cell has unique id, numpy.ndarray (if None, no drift applied)
        drift: whether applying drift to the cropping, should be relative drift to frame with DAPI, 1darray (default: None)
        extend_dim: dimension that expand for cropping, int (default: 30)
        overlap_threshold: upper limit of how much the cropped image include other labels, float<1 (default: 0.1)
        verbose: say something during processing!, bool (default: True)
    Outputs:
        _crop_ims: list of images that has been cropped
    '''
    # imports
    from scipy.ndimage.interpolation import shift
    # check dimension
    _im_dim = np.shape(im)
    _label_dim = np.shape(segmentation_label)
    if drift is not None:
        if len(drift) != len(im.shape):
            raise ValueError('drift dimension and image dimension doesnt match!')
    # initialize cropped image list
    _crop_ims = []

    for _l in range(int(np.max(segmentation_label))):
        #print _l
        if len(_label_dim) == 3: # 3D
            _limits = np.zeros([len(_label_dim),2]) # initialize matrix to save cropping limit
            _binary_label = segmentation_label == _l+1 # extract binary image
            for _m in range(len(_label_dim)):
                _1d_label = _binary_label.sum(_m) > 0
                _has_label=False
                for _n in range(len(_1d_label)):
                    if _1d_label[_n] and not _has_label:
                        _limits[_m,0] = max(_n-extend_dim, 0)
                        _has_label = True
                    elif not _1d_label[_n] and _has_label:
                        _limits[_m,1] = min(_n+extend_dim, _im_dim[_m])
                        _has_label = False
                if _has_label:
                    _limits[_m,1] = _im_dim[_m]
            # crop image and save to _crop_ims
            if drift is None:
                _crop_ims.append(im[_limits[0,0]:_limits[0,1], _limits[2,0]:_limits[2,1], _limits[1,0]:_limits[1,1]])
            else: # do drift correction first and crop
                # define a new drift limits to do cropping
                _drift_limits = np.zeros(_limits.shape)
                for _m, _dim, _d in zip(list(range(len(_label_dim))), _im_dim[-len(_label_dim):], drift[[0,2,1]]):
                    _drift_limits[_m, 0] = max(_limits[_m, 0]-np.ceil(np.max(np.abs(_d))), 0)
                    _drift_limits[_m, 1] = min(_limits[_m, 1]+np.ceil(np.max(np.abs(_d))), _dim)
                #print _drift_limits
                # crop image for pre-correction
                _pre_im = im[_drift_limits[0,0]:_drift_limits[0,1],_drift_limits[2,0]:_drift_limits[2,1],_drift_limits[1,0]:_drift_limits[1,1]]
                # drift correction
                _post_im = shift(_pre_im, - drift)
                # re-crop
                _limit_diffs = _limits - _drift_limits
                for _m in range(len(_label_dim)):
                    if _limit_diffs[_m,1] == 0:
                        _limit_diffs[_m,1] = _limits[_m,1] - _limits[_m,0]
                _limit_diffs = _limit_diffs.astype(np.int)
                #print _limit_diffs
                _crop_ims.append(_post_im[_limit_diffs[0,0]:_limit_diffs[0,0]+_limits[0,1]-_limits[0,0],\
                                          _limit_diffs[2,0]:_limit_diffs[2,0]+_limits[2,1]-_limits[2,0],\
                                          _limit_diffs[1,0]:_limit_diffs[1,0]+_limits[1,1]-_limits[1,0]])

        else: # 2D
            _limits = np.zeros([len(_label_dim),2], dtype=np.int) # initialize matrix to save cropping limit
            _binary_label = segmentation_label == _l+1 # extract binary image
            for _m in range(len(_label_dim)):
                _1d_label = _binary_label.sum(_m) > 0
                _has_label=False
                for _n in range(len(_1d_label)):
                    if _1d_label[_n] and not _has_label:
                        _limits[_m,0] = max(_n-extend_dim, 0)
                        _has_label = True
                    elif not _1d_label[_n] and _has_label:
                        _limits[_m,1] = min(_n+extend_dim, _im_dim[1+_m])
                        _has_label = False
                if _has_label: # if label touch boundary
                    _limits[_m,1] = _im_dim[1+_m]
            #print _limits
            # crop image and save to _crop_ims
            if drift is None:
                _crop_ims.append(im[:,_limits[1,0]:_limits[1,1],_limits[0,0]:_limits[0,1]])
            else: # do drift correction first and crop
                # define a new drift limits to do cropping
                _drift_limits = np.zeros(_limits.shape, dtype=np.int)
                for _m, _dim in zip(list(range(len(_label_dim))), _im_dim[-len(_label_dim):]):
                    _drift_limits[_m, 0] = max(_limits[_m, 0]-np.ceil(np.abs(drift[2-_m])), 0)
                    _drift_limits[_m, 1] = min(_limits[_m, 1]+np.ceil(np.abs(drift[2-_m])), _dim)
                #print _drift_limits
                # crop image for pre-correction
                _pre_im = im[:,_drift_limits[1,0]:_drift_limits[1,1],_drift_limits[0,0]:_drift_limits[0,1]]
                # drift correction
                _post_im = shift(_pre_im, -drift)
                # re-crop
                _limit_diffs = (_limits - _drift_limits).astype(np.int)
                #print _limit_diffs
                _crop_ims.append(_post_im[:,_limit_diffs[1,0]:_limit_diffs[1,0]+_limits[1,1]-_limits[1,0],_limit_diffs[0,0]:_limit_diffs[0,0]+_limits[0,1]-_limits[0,0]])
    return _crop_ims

# get limitied points of seed within radius of a center
def get_seed_in_distance(im, center=None, num_seeds=0, seed_radius=30,
                         gfilt_size=0.75, background_gfilt_size=10, filt_size=3, 
                         seed_by_per=False, th_seed_percentile=95, 
                         th_seed=300,
                         dynamic=True, dynamic_iters=10, min_dynamic_seeds=2, 
                         distance_to_edge=1, hot_pix_th=4, 
                         return_h=False, verbose=False):
    '''Get seed points with in a distance to a center coordinate
    Inputs:
        im: image, 3D-array
        center: center coordinate to get seeds nearby, 1d array / list of 3
        num_seed: maximum number of seeds kept within radius, 0 means keep all, int (default: -1)
        seed_radius: distance of seed points to center, float (default: 15)
        gfilt_size: sigma of gaussian filter applied to image before seeding, float (default: 0.5)
        filt_size: getting local maximum and minimum within in size, int (default: 3)
        th_seed_percentile: intensity percentile of whole image that used as seeding threshold, float (default: 90.)
        hot_pix_th: thereshold for hot pixels, int (default: 0, not removing hot pixel)
        return_h: whether return height of seeds, bool (default: False)
    Outputs:
        _seeds: z,x,y coordinates of seeds, 3 by n matrix
            n = num_seed
            if return height is true, return h,z,x,y instead.
        '''
    from scipy.stats import scoreatpercentile
    from scipy.spatial.distance import cdist

    # check input
    if center is not None and len(center) != 3:
        raise ValueError('wrong input dimension of center!')
    _dim = np.shape(im)
    _im = im.copy()
    # seeding threshold
    if seed_by_per:
        _im_ints = _im[np.isnan(_im)==False].astype(np.float)
        _th_seed = scoreatpercentile(_im_ints, th_seed_percentile) - \
                    scoreatpercentile(_im_ints, 100-th_seed_percentile)
    else:
        _th_seed = th_seed

    if verbose:
        print(f"-- seeding with threshold: {_th_seed}, per={th_seed_percentile}")
    # start seeding 
    if center is not None:
        _center = np.array(center, dtype=np.float)
        _limits = np.zeros([2, 3], dtype=np.int)
        _limits[0, 1:] = np.array([np.max([x, y]) for x, y in zip(
            np.zeros(2), _center[1:]-seed_radius)], dtype=np.int)
        _limits[0, 0] = np.array(
            np.max([0, _center[0]-seed_radius/2]), dtype=np.int)
        _limits[1, 1:] = np.array([np.min([x, y]) for x, y in zip(
            _dim[1:], _center[1:]+seed_radius)], dtype=np.int)
        _limits[1, 0] = np.array(
            np.min([_dim[0], _center[0]+seed_radius/2]), dtype=np.int)
        _local_center = _center - _limits[0]
        # crop im
        _cim = _im[_limits[0, 0]:_limits[1, 0], _limits[0, 1]:_limits[1, 1], _limits[0, 2]:_limits[1, 2]]
        if dynamic:
            _dynamic_range = np.linspace(1, 1 / dynamic_iters, dynamic_iters)
            for _dy_ratio in _dynamic_range:
                _dynamic_th = _th_seed * _dy_ratio
                #print(_dynamic_th)
                # get candidate seeds
                _cand_seeds = get_seed_points_base(_cim, gfilt_size=gfilt_size, background_gfilt_size=background_gfilt_size,
                                                   filt_size=filt_size, th_seed=_dynamic_th, hot_pix_th=hot_pix_th, return_h=True)
                # keep seed within distance
                _distance = cdist(_cand_seeds[:3].transpose(), _local_center[np.newaxis, :3]).transpose()[0]
                _keep = _distance < seed_radius
                _seeds = _cand_seeds[:, _keep]
                _seeds[:3, :] += _limits[0][:, np.newaxis]
                if len(_seeds.shape) == 2:
                    if num_seeds > 0 and _seeds.shape[1] >= min(num_seeds, min_dynamic_seeds):
                        break
                    elif num_seeds == 0 and _seeds.shape[1] >= min_dynamic_seeds:
                        break
        else:
            # get candidate seeds
            _seeds = get_seed_points_base(_cim, gfilt_size=gfilt_size, filt_size=filt_size,
                                          th_seed=th_seed, hot_pix_th=hot_pix_th, return_h=True)

    else:
        # get candidate seeds
        _seeds = get_seed_points_base(_im, gfilt_size=gfilt_size, filt_size=filt_size,
                                      th_seed=_th_seed, hot_pix_th=hot_pix_th, return_h=True)
    # remove seeds out of boundary
    #_keep = np.sum(, axis=0)

    # if limited seeds reported, report top n
    if _seeds.shape[1] > 1:
        _intensity_order = np.argsort(_seeds[-1])
        _seeds = _seeds[:, np.flipud(_intensity_order[-num_seeds:])]
    # if not return height, remove height
    if not return_h:
        _seeds = _seeds[:3].transpose()
    else:
        _seeds = _seeds[:4].transpose()
    return _seeds

# fit single gaussian with varying width given prior


def fit_single_gaussian(im, center_zxy, counted_indices=None,
                        width_zxy=[1.35, 1.9, 1.9], fit_radius=5, n_approx=10,
                        height_sensitivity=100., expect_intensity=800.,
                        weight_sigma=1000.,
                        th_to_end=1e-6):
    """ Function to fit single gaussian with given prior
    Inputs:
        im: image, 3d-array
        center_zxy: center coordinate of seed, 1d-array or list of 3
        counted_indices: z,x,y indices for pixels to be counted, np.ndarray, length=3
        width_zxy: prior width of gaussian fit, 1darray or list of 3 (default: [1.35,1,1])
        fit_radius: fit_radius that allowed for fitting, float (default: 10)
        n_approx: number of pixels used for approximation, int (default: 10)
        height_sensitivity: grant height parameter extra sensitivity compared to others, float (default: 100)
        expect_intensity: lower limit of penalty function applied to fitting, float (default: 1000)
        weight_sigma: L1 norm penalty function applied to widths, float (default: 1000)
    Outputs:
        p.x, p.success: parameters and whether success
        Returns (height, x, y,z, width_x, width_y,width_z,bk)
        the gaussian parameters of a 2D distribution found by a fit"""

    _im = np.array(im, dtype=np.float32)
    dims = np.array(_im.shape)
    # dynamic adjust height_sensitivity
    if np.max(_im) < height_sensitivity:
        height_sensitivity = np.ceil(np.max(_im)) * 0.5
    if np.max(_im) < expect_intensity:
        expect_intensity = np.max(_im) * 0.1
    if len(center_zxy) == 3:
        center_z, center_x, center_y = center_zxy
    else:
        raise ValueError(
            "Wrong input for kwd center_zxy, should be of length=3")
    if counted_indices is not None and len(counted_indices) != 3:
        raise ValueError(
            "Length of counted_indices should be 3, for z,x,y coordinates")
    elif counted_indices is not None:
        zxy = counted_indices
    else:  # get affected coordinates de novo
        total_zxy = (np.indices([2*fit_radius+1]*3) + center_zxy[:,
                                                                 np.newaxis, np.newaxis, np.newaxis] - fit_radius).reshape(3, -1)
        keep = (total_zxy >= 0).all(0) * (total_zxy[0] < _im.shape[0]) * (
            total_zxy[1] < _im.shape[1]) * (total_zxy[2] < _im.shape[2])
        zxy = total_zxy[:, keep]
    if len(zxy[0]) > 0:
        _used_im = _im[zxy[0], zxy[1], zxy[2]]
        sorted_im = np.sort(_used_im)  # np.sort(np.ravel(_used_im))
        bk = np.median(sorted_im[:n_approx])
        if bk < 0:
            bk = 0
        height = (np.median(sorted_im[-n_approx:])-bk) / height_sensitivity
        if height < 0:
            height = 0
        width_z, width_x, width_y = np.array(width_zxy)
        params_ = (height, center_z, center_x, center_y,
                   bk, width_z, width_x, width_y)

        def gaussian(height, center_z, center_x, center_y,
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

            def gauss(z, x, y):
                g = bk_ + height_ * height_sensitivity * np.exp(
                    -(((center_z-z)/width_z_)**2 +
                      ((center_x-x)/width_x_)**2 +
                      ((center_y-y)/width_y_)**2)/2.)
                return g
            return gauss

        def errorfunction(p):
            f = gaussian(*p)(*zxy)
            g = _used_im
            #err=np.ravel(f-g-g*np.log(f/g))
            err = np.ravel(f-g) \
                + weight_sigma * np.linalg.norm(p[-3:]-width_zxy, 1)
            return err

        p = scipy.optimize.least_squares(errorfunction,  params_, bounds=(
            0, np.inf), ftol=th_to_end, xtol=th_to_end, gtol=th_to_end/10.)
        p.x[0] *= height_sensitivity

        return p.x, p.success
    else:
        return None, None

# Multi gaussian fitting
def fit_multi_gaussian(im, seeds, width_zxy = [1.5, 2, 2], fit_radius=5,
                       height_sensitivity=100., expect_intensity=500., expect_weight=1000.,
                       th_to_end=1e-7,
                       n_max_iter=10, max_dist_th=0.25, min_height=100.0,
                       return_im=False, verbose=True):
    """ Function to fit multiple gaussians (with given prior)
    Inputs:
        im: image, 3d-array
        center_zxy: center coordinate of seed, 1darray or list of 3
        width_zxy: prior width of gaussian fit, 1darray or list of 3 (default: [1.35,1,1])
        fit_radius: radius that allowed for fitting, float (default: 10)
        height_sensitivity: grant height parameter extra sensitivity compared to others, float (default: 100)
        expect_intensity: lower limit of penalty function applied to fitting, float (default: 1000)
        expect_weight: L1 norm penalty function applied to widths, float (default: 1000)
        n_max_iter: max iteration count for re-fit existing points, int (default: 10)
        max_dist_th: maximum allowed distance between original fit and re-fit, float (default: 0.25)
        min_height: miminal heights required for fitted spots, float (default: 100.)
        return_im: whether return images of every single fitting, bool (default: False)
        verbose: whether say something, bool (default: True)
    Outputs:
        p: parameters
        Returns (height, x, y,z, width_x, width_y,width_z,bk)
        the gaussian parameters of a 2D distribution found by a fit"""
    if verbose:
        print(f"-- Multi-Fitting:{len(seeds)} points")
    # adjust min_height:
    if np.max(im) * 0.1 < min_height:
        min_height = np.max(im)*0.05
    # seeds
    _seeds = seeds
    if len(_seeds) > 0:
        # initialize
        ps = []
        sub_ims = []
        im_subtr = np.array(im,dtype=np.float)

        # loop through seeds
        for _seed in _seeds:
            p, success = fit_single_gaussian(im_subtr,_seed[:3],
                                          height_sensitivity=height_sensitivity,
                                          expect_intensity=expect_intensity,
                                          weight_sigma=expect_weight,
                                          fit_radius=fit_radius,
                                          width_zxy=width_zxy,
                                          th_to_end=th_to_end)
            if p is not None and success: # If got any successful fitting, substract fitted profile
                ps.append(p)
                sub_ims.append(im_subtr)
                im_subtr = subtract_source(im_subtr,p)

        return np.array(ps)
        print("do something")
        # recheck fitting
        im_add = np.array(im_subtr)
        max_dist=np.inf
        n_iter = 0
        while max_dist > max_dist_th:
            ps_1=np.array(ps)
            if len(ps_1)>0:
                ps_1=ps_1[np.argsort(ps_1[:,0])[::-1]]
            else:
                return np.array([])
            ps = []
            ps_1_rem=[]
            for p_1 in ps_1:
                _seed = p_1[1:4]
                im_add = plus_source(im_add, p_1)
                p,success = fit_single_gaussian(im_add,_seed,
                                              height_sensitivity=height_sensitivity,
                                              expect_intensity=expect_intensity,
                                              weight_sigma=expect_weight,
                                              fit_radius=fit_radius,
                                              width_zxy=width_zxy,
                                              th_to_end=th_to_end)
                if p is not None:
                    #print('recheck',p[1:4], success)
                    im_add = subtract_source(im_add, p)
                    ps.append(p)
                    ps_1_rem.append(p_1)
            ps_2=np.array(ps)
            ps_1_rem=np.array(ps_1_rem)
            #print(len(ps_2), len(ps_1_rem))
            dif = ps_1_rem[:,1:4]-ps_2[:,1:4]
            max_dist = np.max(np.sum(dif**2,axis=-1))
            n_iter+=1
            if n_iter>n_max_iter:
                break
        _kept_fits = ps_2
        if len(_kept_fits) > 1:
            _intensity_order = np.argsort(_kept_fits[:,0])
            _kept_fits = _kept_fits[np.flipud(_intensity_order),:]
        if len(_kept_fits) > 0 and sum([_ft[0]>min_height for _ft in _kept_fits]) > 0:
            _kept_fits = np.array([_ft for _ft in _kept_fits if _ft[0]>min_height])
        elif len(_kept_fits) > 0:
            _kept_fits = np.array([_kept_fits[0]])
        if return_im:
            return _kept_fits, sub_ims
        else:
            return _kept_fits

    else:
        return np.array([])

# slice 3d image
def slice_image(fl, sizes, zlims, xlims, ylims, zstep=1, zstart=0, empty_frame=1,
                npy_start=128, image_dtype=np.uint16, verbose=False):
    """
    Slice image in a memory-efficient manner.
    Inputs:
        fl: filename of a binary np.uint16 image or matrix. 
            Notice: .dax is binary file stored from data.tofile("temp.dax"), is a header-less nd-array. 
            However '.npy' file has some header in the beginning, which is ususally 128 Bytes, string(path)
        sizes: size of raw image in z-x-y order, array-like struct of 3 example:[30,2048,2048]
        zlims: limits in z axis (axis0), array-like struct of 2
        xlims: limits in x axis (axis1), array-like struct of 2
        ylims: limits in y axis (axis2), array-like struct of 2
        zstep: number of steps to take one z-stack image, positive int (default: 1)
        zstart: channel id(s), non-negative int or list of nn-int (default: 0)
        empty_frame: number of empty_frames 
        npy_start: starting bytes for npy format, int (default: 64)
    Output:
        data: cropped 3D image
    Usage:
        fl = 'Z:\\20181022-IMR90_whole-chr21-unique\\H0R0\\Conv_zscan_00.dax'
        im = slice_image(fl, [170, 2048, 2048],[10, 160], [100, 300], [1028, 2048],5,4)
    """
    if isinstance(zstart, int):
        zs = [zstart]
    elif isinstance(zstart, list):
        _zs, _inds = np.unique(zstart, return_index=True)
        zs = list(np.array(zstart)[np.sort(_inds)])
    else:
        raise TypeError(
            f"Wrong input type for zstart, should be int or list of int, {type(zstart)} is given!")
    for _z in zs:
        if _z >= zstep or _z < 0:
            raise Warning(
                f"Wrong z-start input:{_z}, should be non-negeative integer < {zstep}")
    if zstep <= 0:
        raise ValueError(
            f"Wrong z-step input:{zstep}, should be positive integer.")
    # image dimension
    sz, sx, sy = np.array(sizes, dtype=np.int)[:3]
    # acquire min-max indices
    minz, maxz = np.sort(np.array(zlims, dtype=np.int))[:2]
    minx, maxx = np.sort(np.array(xlims, dtype=np.int))[:2]
    miny, maxy = np.sort(np.array(ylims, dtype=np.int))[:2]
    # acquire dimension
    dz = int((maxz-minz)/zstep)
    dx = int(maxx-minx)
    dy = int(maxy-miny)
    # acquire element size
    element_size = np.dtype(image_dtype).itemsize

    if dx <= 0 or dy <= 0 or dz <= 0:
        print("-- slicing result is empty.")
        return np.array([])
    # initialize
    _ims = [np.zeros([dz, dx, dy], dtype=image_dtype) for _z in zs]
    # file handle
    f = open(fl, "rb")
    # starting point
    if fl.split('.')[-1] == 'npy':
        if verbose:
            print(f"- slicing .npy file, start with {npy_start}")
        pt_pos = int(npy_start / element_size)
    else:
        pt_pos = 0
    # initialize pointers
    pt_pos += int(minx*sy + miny)

    # start layer
    _start_layer = minz

    if fl.split('.')[-1] == 'dax' and zstep > 1:
        _lims = [minz + (_z + empty_frame - minz) % zstep for _z in zs]
        _new_lims = []
        for _i, _l in enumerate(_lims):
            if _l - minz == 0:
                _lims[_i] += zstep
        _start_layer = min(_lims)
    else:
        _start_layer = minz + min([(_z + empty_frame - minz) % zstep for _z in zs])
    # get data
    _data_cts = [0 for _z in zs]
    _res = [(_z + empty_frame) % zstep for _z in zs]
    for iz in range(sz):
        if (np.array(_data_cts) >= dz).all():
            # stop if all data image filled
            break
        elif iz >= _start_layer and iz % zstep in _res:
            _data_index = _res.index(iz % zstep)
            _data_layer = np.zeros([dx, dy], dtype=image_dtype)
            for ix in range(dx):
                # record dy data
                f.seek(pt_pos * element_size, 0)
                _data_layer[ix, :] = np.fromfile(
                    f, dtype=image_dtype, count=dy)
                # skip to next line
                pt_pos += int(sy)
            _ims[_data_index][_data_cts[_data_index], :, :] = _data_layer
            _data_cts[_data_index] += 1
            # skip to next layer
            pt_pos += int((sx-dx) * sy)
        else:
            # skip the whole layer
            pt_pos += int(sx * sy)
    # close and return
    f.close()
    if isinstance(zstart, int):
        return _ims[0]
    else:
        return _ims

def slice_image_remove_channel(fl, sizes, zstep, remove_zstarts, empty_frame=1,
                               zlims=None, xlims=None, ylims=None, 
                               npy_start=128, image_dtype=np.uint16, 
                               verbose=False):
    """
    Slice image in a memory-efficient manner.
    Inputs:
        fl: filename of a binary np.uint16 image or matrix. 
            Notice: .dax is binary file stored from data.tofile("temp.dax"), is a header-less nd-array. 
            However '.npy' file has some header in the beginning, which is ususally 128 Bytes, string(path)
        sizes: size of raw image in z-x-y order, array-like struct of 3 example:[30,2048,2048]
        zstep: number of steps to take one z-stack image, positive int 
        remove_zstarts: channel id(s), non-negative int or list of nn-int 
        zlims: limits in z axis (axis0), array-like struct of 2
        xlims: limits in x axis (axis1), array-like struct of 2
        ylims: limits in y axis (axis2), array-like struct of 2
        npy_start: starting bytes for npy format, int (default: 64)
    Output:
        data: cropped 3D image
    Usage:
        fl = 'Z:\\20181022-IMR90_whole-chr21-unique\\H0R0\\Conv_zscan_00.dax'
        im = slice_image(fl, [170, 2048, 2048],[10, 160], [100, 300], [1028, 2048],5,4)
    """
    if isinstance(remove_zstarts, int):
        zs = [remove_zstarts]
    elif isinstance(remove_zstarts, list):
        _zs, _inds = np.unique(remove_zstarts, return_index=True)
        zs = list(np.array(remove_zstarts)[np.sort(_inds)])
    else:
        raise TypeError(
            f"Wrong input type for remove_zstarts, should be int or list of int, {type(remove_zstarts)} is given!")
    for _z in zs:
        if _z >= zstep or _z < 0:
            raise Warning(
                f"Wrong z-start input:{_z}, should be non-negeative integer < {zstep}")
    if zstep <= 0:
        raise ValueError(
            f"Wrong z-step input:{zstep}, should be positive integer.")
    # image dimension
    sz, sx, sy = sizes[:3]
    # set default limits to be size of image
    if zlims is None:
        zlims = [0, sz]
    if xlims is None:
        xlims = [0, sx]
    if ylims is None:
        ylims = [0, sy]
    # acquire min-max indices
    minz, maxz = np.sort(zlims)[:2]
    minx, maxx = np.sort(xlims)[:2]
    miny, maxy = np.sort(ylims)[:2]
    # acquire dimension
    dz = int((maxz-minz)/zstep)
    dx = int(maxx-minx)
    dy = int(maxy-miny)
    # acquire element size
    element_size = np.dtype(image_dtype).itemsize

    if dx <= 0 or dy <= 0 or dz <= 0:
        print("-- slicing result is empty.")
        return np.array([])
    # initialize
    _kept_layers = []
    # file handle
    f = open(fl, "rb")
    # starting point
    if fl.split('.')[-1] == 'npy':
        if verbose:
            print(f"- slicing .npy file, start with {npy_start}")
        pt_pos = int(npy_start / element_size)
    else:
        pt_pos = 0
    # initialize pointers
    pt_pos += int(minx*sy + miny)

    # start layer
    _start_layer = minz + empty_frame

    # get data
    _res = [(_z+empty_frame) % zstep for _z in zs]
    for iz in range(sz):
        # if this layer to be removed, skip
        if iz >= _start_layer and iz <= maxz and iz % zstep in _res:
            # skip the whole layer
            pt_pos += int(sx * sy)
        # else, keep
        else:
            _data_layer = np.zeros([dx, dy], dtype=image_dtype)
            for ix in range(dx):
                # record dy data
                f.seek(pt_pos * element_size, 0)
                _data_layer[ix, :] = np.fromfile(
                    f, dtype=image_dtype, count=dy)
                # skip to next line
                pt_pos += int(sy)
            _kept_layers.append(_data_layer)
            # skip to next layer
            pt_pos += int((sx-dx) * sy)
            
    # close and return
    f.close()
    
    return np.array(_kept_layers, dtype=image_dtype)

def slice_2d_image(fl, im_shape, xlims, ylims, npy_start=128, image_dtype=np.uint16, verbose=False):
    """Function to slice 2d image directly by avoiding loading in RAM
    Inputs:
        fl: filename of 2d image, string
        im_shape: shape of 2d image, list of 2 or array of 2
        xlims: slice limits along x-axis, list of 2 or array of 2
        ylims: slice limits along y-axis, list of 2 or array of 2
        npy_start: number of bytes for prefix for npy filetype (may vary across systems), int (default: 128)
        image_dtype: data_type for this image, datatype from numpy (default: np.uint16)
        verbose: say something! bool (default: False)
    Outputs:
        data: sliced 2d image
    """
    # check input filename
    if not isinstance(fl, str):
        raise ValueError(
            f"Wrong input for fl:{fl}, a string of filename is expected!")
    elif not os.path.isfile(fl):
        raise IOError(f"Wrong input for fl:{fl}, input file not exists!")
    else:
        _file_postfix = fl.split('.')[-1]
        if _file_postfix != 'dax' and _file_postfix != 'npy':
            raise IOError("Wrong fl filetype, should be dax or npy!")
    # acquire sizes
    sx, sy = im_shape[:2]
    # acquire limits
    minx, maxx = np.sort(xlims)[:2]
    miny, maxy = np.sort(ylims)[:2]
    ## start acquiring image
    # initialize data
    data = np.zeros([maxx-minx, maxy-miny], dtype=image_dtype)
    element_size = np.dtype(image_dtype).itemsize
    # open handle
    f = open(fl, 'rb')
    if _file_postfix == 'npy':
        pt_pos = np.int(npy_start/element_size)
    else:
        pt_pos = 0
    if verbose:
        print(
            f"- start slicing filetype {_file_postfix}, start with {pt_pos*element_size}, element_size:{element_size}")
    # skip lines before min_x
    pt_pos += minx * sy + miny
    # start acquiring images
    for ix in range(maxx-minx):
        f.seek(pt_pos * element_size, 0)
        data[ix, :] = np.fromfile(f, dtype=image_dtype, count=(maxy-miny))
        pt_pos += sy
    # close file handle
    f.close()
    return data

# function to crop one image given filename, color_channel and crop_limits
def crop_single_image(filename, channel, crop_limits=None, num_buffer_frames=10,
                      all_channels=_allowed_colors, single_im_size=_image_size,
                      drift=np.array([0, 0, 0]), num_empty_frames=1,
                      return_limits=False, verbose=False):
    """Function to crop one image given filename, color_channel and crop_limits
    Inputs:
        filename: .dax filename for given image, string of filename
        channel: color_channel for the specific data, int or str
        crop_limits: 2x2 or 3x2 array specifying where to crop, 
            np.ndarray (default: None, i.e. the whole image)
        num_buffer_frame: number of frames before z-scan starts, int (default:10)
        all_channels: all allowed colors in given data, list (default: _allowed_colors)
        single_im_size: image size for single color full image, list/array of 3 (default:[30,2048,2048])
        drift: drift to ref-frame of this image, np.array of 3 (default:[0,0,0])
        verbose: say something!, bool (default:False)
    Output:
        _crp_im: cropped image
        """
    ## 0. check inputs
    # filename
    if not os.path.isfile(filename):
        raise ValueError(f"file {filename} doesn't exist!")
    # check input channel
    channel = str(channel)
    if channel not in all_channels:
        raise ValueError(
            f"Input channel:{channel} is not included in all_channels{all_channels}, exit!")
    _channel_id = all_channels.index(channel)
    # drift
    drift = np.array(drift)
    if len(drift) != 3:
        raise ValueError("dimension of drift should be 3!")
    # crop_limits
    if crop_limits is None:
        crop_limits = np.stack([np.zeros(3), single_im_size]).T.astype(np.int)
    elif len(crop_limits) == 2:
        crop_limits = np.array(
            [np.array([0, single_im_size[0]])]+list(crop_limits), dtype=np.int)
    elif len(crop_limits) == 3:
        crop_limits = np.array(crop_limits, dtype=np.int)
    else:
        raise IndexError(
            f"Wrong shape for crop_limits:{np.shape(crop_limits)}")
    # convert negative slices into positives
    for _lims, _s in zip(crop_limits, single_im_size):
        if _lims[1] < 0:
            _lims[1] += _s
    # convert crop_limits into drift_limits
    _drift_limits = np.zeros(crop_limits.shape, dtype=np.int)
    for _i, (_d, _lims) in enumerate(zip(drift, crop_limits)):
        # expand drift_limits a bit for shifting
        _drift_limits[_i, 0] = max(_lims[0]-np.ceil(np.abs(_d)), 0)
        _drift_limits[_i, 1] = min(
            _lims[1]+np.ceil(np.abs(_d)), single_im_size[_i])

    ## 1. load image
    # extract image info
    _full_im_shape, _num_color = get_img_info.get_num_frame(filename,
                                                            frame_per_color=single_im_size[0],
                                                            buffer_frame=num_buffer_frames)
    _zlims = [num_buffer_frames+_drift_limits[0, 0]*_num_color,
              num_buffer_frames+_drift_limits[0, 1]*_num_color]
    # slice image
    _crp_im = slice_image(filename, _full_im_shape, _zlims, _drift_limits[1],
                          _drift_limits[2], zstep=_num_color, zstart=_channel_id,
                          empty_frame=num_empty_frames)
    # do shift if drift exists
    if drift.any():
        _crp_im = ndimage.interpolation.shift(_crp_im, -drift, mode='nearest')

    ## 2. second crop to adjust drift
    # get differences between two limits
    _limit_diffs = (crop_limits - _drift_limits).astype(np.int)

    # do second crop
    _crp_im = _crp_im[_limit_diffs[0, 0]: _limit_diffs[0, 0]+crop_limits[0, 1]-crop_limits[0, 0],
                      _limit_diffs[1, 0]: _limit_diffs[1, 0]+crop_limits[1, 1]-crop_limits[1, 0],
                      _limit_diffs[2, 0]: _limit_diffs[2, 0]+crop_limits[2, 1]-crop_limits[2, 0]]
    _final_limits = np.array([_drift_limits[:, 0]+_limit_diffs[:, 0],
                              _drift_limits[:, 0]+_limit_diffs[:, 0]+crop_limits[:, 1]-crop_limits[:, 0]]).T
    if return_limits:
        return _crp_im, _final_limits
    else:
        return _crp_im


# function to crop multi-channel images from one dax file given filename, color_channel and crop_limits
def crop_multi_channel_image(filename, channels, crop_limits=None, 
                             num_buffer_frames=10, num_empty_frames=1, 
                             all_channels=_allowed_colors, single_im_size=_image_size,
                             drift=np.array([0, 0, 0]), 
                             return_limits=False, verbose=False):
    """Function to crop one image given filename, color_channel and crop_limits
    Inputs:
        filename: .dax filename for given image, string of filename
        channel: color_channel for the specific data, int or str
        crop_limits: 2x2 or 3x2 array specifying where to crop, 
            np.ndarray (default: None, i.e. the whole image)
        num_buffer_frame: number of frames before z-scan starts, int (default:10)
        all_channels: all allowed colors in given data, list (default: _allowed_colors)
        single_im_size: image size for single color full image, list/array of 3 (default:[30,2048,2048])
        drift: drift to ref-frame of this image, np.array of 3 (default:[0,0,0])
        verbose: say something!, bool (default:False)
    Output:
        _crp_im: cropped image
        """
    ## 0. check inputs
    # filename
    if not os.path.isfile(filename):
        raise ValueError(f"file {filename} doesn't exist!")
    # check input channel
    channels = [str(_ch) for _ch in channels]
    for _ch in channels:
        if _ch not in all_channels:
            raise ValueError(
                f"Input channel:{_ch} is not included in all_channels{all_channels}, exit!")
    _channel_ids = [all_channels.index(_ch) for _ch in channels]
    # drift
    drift = np.array(drift)
    if len(drift) != 3:
        raise ValueError("dimension of drift should be 3!")
    # crop_limits
    if crop_limits is None:
        crop_limits = np.stack([np.zeros(3), single_im_size]).T.astype(np.int)
    elif len(crop_limits) == 2:
        crop_limits = np.array(
            [np.array([0, single_im_size[0]])]+list(crop_limits), dtype=np.int)
    elif len(crop_limits) == 3:
        crop_limits = np.array(crop_limits, dtype=np.int)
    else:
        raise IndexError(
            f"Wrong shape for crop_limits:{np.shape(crop_limits)}")
    # convert negative slices into positives
    for _lims, _s in zip(crop_limits, single_im_size):
        if _lims[1] < 0:
            _lims[1] += _s
    # convert crop_limits into drift_limits
    _drift_limits = np.zeros(crop_limits.shape, dtype=np.int)
    for _i, (_d, _lims) in enumerate(zip(drift, crop_limits)):
        # expand drift_limits a bit for shifting
        _drift_limits[_i, 0] = max(_lims[0]-np.ceil(np.abs(_d)), 0)
        _drift_limits[_i, 1] = min(_lims[1]+np.ceil(np.abs(_d)), single_im_size[_i])

    ## 1. load image
    # extract image info
    _full_im_shape, _num_color = get_img_info.get_num_frame(filename,
                                                            frame_per_color=single_im_size[0],
                                                            buffer_frame=num_buffer_frames)
    _zlims = [num_buffer_frames+_drift_limits[0, 0]*_num_color,
              num_buffer_frames+_drift_limits[0, 1]*_num_color]
    # slice image
    _crp_ims = slice_image(filename, _full_im_shape, _zlims, _drift_limits[1],
                           _drift_limits[2], zstep=_num_color, zstart=_channel_ids,
                           empty_frame=num_empty_frames)
    # do shift if drift exists
    if drift.any():
        _crp_ims = [ndimage.interpolation.shift(_im, -drift, mode='nearest') for _im in _crp_ims]

    ## 2. second crop to adjust drift
    # get differences between two limits
    _limit_diffs = (crop_limits - _drift_limits).astype(np.int)

    # do second crop
    _crp_ims = [_im[_limit_diffs[0, 0]: _limit_diffs[0, 0]+crop_limits[0, 1]-crop_limits[0, 0],
                    _limit_diffs[1, 0]: _limit_diffs[1, 0]+crop_limits[1, 1]-crop_limits[1, 0],
                    _limit_diffs[2, 0]: _limit_diffs[2, 0]+crop_limits[2, 1]-crop_limits[2, 0]]
                for _im in _crp_ims]
    _final_limits = np.array([_drift_limits[:, 0]+_limit_diffs[:, 0],
                              _drift_limits[:, 0]+_limit_diffs[:, 0]+crop_limits[:, 1]-crop_limits[:, 0]]).T
    if return_limits:
        return _crp_ims, _final_limits
    else:
        return _crp_ims

def crop_multi_channel_image_v2(filename, channels, crop_limits=None, 
                             num_buffer_frames=10, num_empty_frames=0, 
                             all_channels=_allowed_colors, single_im_size=_image_size,
                             drift=np.array([0, 0, 0]), 
                             return_limits=False, verbose=False):
    """Function to crop one image given filename, color_channel and crop_limits
    Inputs:
        filename: .dax filename for given image, string of filename
        channel: color_channel for the specific data, int or str
        crop_limits: 2x2 or 3x2 array specifying where to crop, 
            np.ndarray (default: None, i.e. the whole image)
        num_buffer_frame: number of frames before z-scan starts, int (default:10)
        all_channels: all allowed colors in given data, list (default: _allowed_colors)
        single_im_size: image size for single color full image, list/array of 3 (default:[30,2048,2048])
        drift: drift to ref-frame of this image, np.array of 3 (default:[0,0,0])
        verbose: say something!, bool (default:False)
    Output:
        _crp_im: cropped image
        """
    ## 0. check inputs
    from scipy.ndimage.interpolation import map_coordinates
    # filename
    if not os.path.isfile(filename):
        raise ValueError(f"file {filename} doesn't exist!")
    # check input channel
    channels = [str(_ch) for _ch in channels]
    for _ch in channels:
        if _ch not in all_channels:
            raise ValueError(
                f"Input channel:{_ch} is not included in all_channels{all_channels}, exit!")
    _channel_ids = [all_channels.index(_ch) for _ch in channels]
    # drift
    drift = np.array(drift)
    if len(drift) != 3:
        raise ValueError("dimension of drift should be 3!")
    # crop_limits
    if crop_limits is None:
        crop_limits = np.stack([np.zeros(3), single_im_size]).T.astype(np.int)
    elif len(crop_limits) == 2:
        crop_limits = np.array(
            [np.array([0, single_im_size[0]])]+list(crop_limits), dtype=np.int)
    elif len(crop_limits) == 3:
        crop_limits = np.array(crop_limits, dtype=np.int)
    else:
        raise IndexError(
            f"Wrong shape for crop_limits:{np.shape(crop_limits)}")
    # convert negative slices into positives
    for _lims, _s in zip(crop_limits, single_im_size):
        if _lims[1] < 0:
            _lims[1] += _s
    # convert crop_limits into drift_limits
    _drift_limits = np.zeros(crop_limits.shape, dtype=np.int)
    for _i, (_d, _lims) in enumerate(zip(drift, crop_limits)):
        # expand drift_limits a bit for shifting
        _drift_limits[_i, 0] = max(_lims[0]-np.ceil(np.abs(_d)), 0)
        _drift_limits[_i, 1] = min(_lims[1]+np.ceil(np.abs(_d)), single_im_size[_i])

    ## 1. load image
    # extract image info
    _full_im_shape, _num_color = get_img_info.get_num_frame(filename,
                                                            frame_per_color=single_im_size[0],
                                                            buffer_frame=num_buffer_frames)
    _zlims = [num_buffer_frames+_drift_limits[0, 0]*_num_color,
              num_buffer_frames+_drift_limits[0, 1]*_num_color]
    # slice image
    _crp_ims = slice_image(filename, _full_im_shape, _zlims, _drift_limits[1],
                           _drift_limits[2], zstep=_num_color, zstart=_channel_ids,
                           empty_frame=num_empty_frames)
    ## 2. do shift if drift exists
    # get differences between two limits
    _limit_diffs = (crop_limits - _drift_limits).astype(np.int)
    if drift.any():    
        # 2.1 get coordiates to be mapped
        _coords = np.meshgrid( np.arange(crop_limits[0][1]-crop_limits[0][0]), 
                               np.arange(crop_limits[1][1]-crop_limits[1][0]), 
                               np.arange(crop_limits[2][1]-crop_limits[2][0]))
        _coords = np.stack(_coords).transpose((0, 2, 1, 3)).astype(np.float) # transpose is necessary
        # 2.2 adjust coordinates based on drift_limits and limits
        _coords += _limit_diffs[:,0,np.newaxis,np.newaxis,np.newaxis]
        # 2.3 adjust coordinates based on drift
        _coords += drift[:, np.newaxis,np.newaxis,np.newaxis]
        # 2.4 map coordinates
        _start_shift = time.time()
        _crp_ims = [
            map_coordinates(_im, _coords.reshape(_coords.shape[0], -1), mode='nearest').reshape(_coords.shape[1:])
            for _im in _crp_ims]
        print(time.time()-_start_shift)
    else:
        pass
    ## 3. determine final limits
    _final_limits = np.array([_drift_limits[:, 0]+_limit_diffs[:, 0],
                              _drift_limits[:, 0]+_limit_diffs[:, 0]+crop_limits[:, 1]-crop_limits[:, 0]]).T
    
    if return_limits:
        return _crp_ims, _final_limits
    else:
        return _crp_ims



# visualize fitted spot crops
def visualize_fitted_spot_crops(im, centers, center_inds, radius=10):
    """Function to visualize fitted spots within a given images and fitted centers"""
    center_inds = [_id for _id,ct in zip(center_inds, centers) if (np.isnan(ct)==False).all()]
    centers = [ct for ct in centers if (np.isnan(ct)==False).all()]
    if len(centers) == 0:  # no center given
        return
    if isinstance(im, np.ndarray) and len(im.shape) != 3:
        raise ValueError("Input image should be 3D!")
    elif isinstance(im, list) and len(im[0].shape)!= 3:
        raise ValueError("Input images should be 3D!")

    cropped_ims = []
    if isinstance(im, np.ndarray):
        # iterate through centers
        for ct in centers:
            if len(ct) != 3:
                raise ValueError(
                    f"Wrong input dimension of centers, only expect [z,x,y] coordinates in center:{ct}")
            crop_l = np.array([np.zeros(3), np.round(ct-radius)], dtype=np.int).max(0)
            crop_r = np.array([np.array(np.shape(im)), 
                            np.round(ct+radius+1)], dtype=np.int).min(0)
            _cim = im[crop_l[0]:crop_r[0], crop_l[1]:crop_r[1], crop_l[2]:crop_r[2]]
            _nim = np.ones([radius*2+1]*3) * np.median(_cim)
            _im_l = np.round(ct - crop_l + radius).astype(np.int)
            _im_r = np.round(crop_r - ct + radius).astype(np.int)
            _nim[_im_l[0],_im_r[0],
                 _im_l[1],_im_r[1],
                 _im_l[2],_im_r[2]] = _cim

            cropped_ims.append(_nim)
    elif isinstance(im, list):
        _ims = im
        center_list = partition_map(centers, center_inds, enumerate_all=True)
        for _i, _list in zip(np.arange(np.min(center_inds), np.max(center_inds)+1), center_list):
            # extract image
            _im = _ims[_i]
            for ct in _list:
                if len(ct) != 3:
                    raise ValueError(f"Wrong input dimension of centers, only expect [z,x,y] coordinates in center:{ct}")
                crop_l = np.array([np.zeros(3), np.round(ct-radius)], dtype=np.int).max(0)
                crop_r = np.array([np.array(np.shape(_im)), np.round(ct+radius+1)], dtype=np.int).min(0)
                _cim = _im[crop_l[0]:crop_r[0], crop_l[1]:crop_r[1], crop_l[2]:crop_r[2]]
                _nim = np.ones([radius*2+1]*3) * np.median(_cim)
                _im_l = np.round(crop_l - ct + radius).astype(np.int)
                _im_r = np.round(crop_r - ct + radius).astype(np.int)
                _nim[_im_l[0]:_im_r[0],
                     _im_l[1]:_im_r[1],
                     _im_l[2]:_im_r[2]] = _cim
                cropped_ims.append(_nim)
    else:
        raise TypeError(f"Wrong input type for im")

    cropped_shape = np.array([np.array(_cim.shape) for _cim in cropped_ims]).max(0)
    image_names = [f"im:{_ind}, ct:{np.round(_ct,2)}" for _ct, _ind in zip(centers, center_inds)]
    if sum([(np.array(_cim.shape) == cropped_shape).all() for _cim in cropped_ims]) == len(cropped_ims):
        return imshow_mark_3d_v2(cropped_ims, image_names=image_names)
    else:
        amended_cropped_ims = [
            np.ones(cropped_shape)*np.mean(_cim) for _cim in cropped_ims]
        for _cim, _acim in zip(cropped_ims, amended_cropped_ims):
            _cs = list(_cim.shape)
            _acim[:_cs[0], :_cs[1], :_cs[2]] += _cim
        return imshow_mark_3d_v2(amended_cropped_ims, image_names=image_names)

def visualize_fitted_spot_images(ims, centers, center_inds, 
                                 save_folder=None, image_names=None,
                                 save_name='fitted_image.pkl', 
                                 overwrite=True, verbose=True):
    """Visualize fitted spots in original image shape"""
    ## Inputs
    # images
    _ims = list(ims)
    if len(_ims[0].shape)!= 3:
        raise ValueError("Input images should be 3D!")
    # centers
    center_inds = [_id for _id,_ct in zip(center_inds, centers) if (np.isnan(_ct)==False).all()]
    centers = [_ct for _ct in centers if (np.isnan(_ct)==False).all()]
    if len(centers) == 0:  # no center given
        return
    if image_names is None:
        image_names = [str(_i) for _i in range(len(_ims))]
    _coord_dic = {'coords': [np.flipud(_ct) for _ct,_id in zip(centers, center_inds)],
                  'class_ids': [int(_id) for _ct,_id in zip(centers, center_inds)],
                  'pfits':{},
                  'dec_text':{},
                  } # initialize _coord_dic for picking
    if save_folder is None:
        save_folder = '.'
    save_filename = os.path.join(save_folder, save_name)
    if not os.path.isfile(save_filename) or overwrite:
        if verbose:
            print(f"--- dump coordinate information into {save_filename}")
        pickle.dump(_coord_dic, open(save_filename, 'wb'))
    _im_viewer = imshow_mark_3d_v2(_ims, 
                                   image_names=image_names,
                                   save_file=save_filename)
    return _im_viewer

def Extract_crop_from_segmentation(segmentation_label, extend_dim=20, single_im_size=_image_size):
    """Function to extract a crop matrix from given segmentation label and extend_dim
    Inputs:
        segmentation_label: 2d segmentation label for single cell, 2d-array
        extend_dim: num of pixels to extend for cropping, int (default: 20)
        single_im_size: expected image size for one color channel, list/array of 3 (default:[30,2048,2048])
    Outputs
        _limits: 3-by-2 matrix for segmentation crop limits
        """
    # convert to local vars
    _seg_label = np.array(segmentation_label > 0, dtype=np.int)
    _limits = np.zeros([3, 2], dtype=np.int)
    _limits[0, 1] = single_im_size[0]
    # calculate _limits
    for _dim in range(len(_seg_label.shape)):
        _1d_label = np.array(np.sum(_seg_label, axis=tuple(i for i in range(len(_seg_label.shape)) if i != _dim)) > 0,
                             dtype=np.int)
        _1d_indices = np.where(_1d_label)[0]
        # update limits
        _limits[_dim+1, 0] = max(_1d_indices[0]-extend_dim, 0)
        _limits[_dim+1, 1] = min(
            _1d_indices[-1]+extend_dim, _seg_label.shape[_dim])
    return _limits

# translate segmentation from previous experiment
def translate_segmentation(old_segmentation, old_dapi_im, new_dapi_im, 
                           rotation_mat=None, rotation_ref_file=None,
                           dapi_channel='405', all_channels=_allowed_colors, 
                           num_buffer_frames=10, num_empty_frames=1,
                           old_correction_folder=_correction_folder, 
                           new_correction_folder=_correction_folder, 
                           fft_gb=0, fft_max_disp=200, 
                           return_new_dapi=False, verbose=True):
    """Function to translate segmentation to another given both dapi_images 
    (rotation_matrix may be provided as well)
    Inputs:
        old_segmentation: reference segmentation to be translated, filename or 2d-np.array
        old_dapi_im: DAPI image from original experiment, 3d-array
        new_dapi_im: DAPI image from new experiment, 3d-array
        rotation_mat: rotation matrix, adopted from align_manual_points, 2x2 ndarray or None (default by loading from file)
        dapi_channel: channel used for dapi, int or str (default: '405')
        all_channels: all allowed channels, list
        old_correction_folder: correction folder for old dapi
        new_correction_folder: correction folder for new dapi
        return_new_dapi: whether return new dapi image as well, bool
        verbose: say something!, bool (default: True)
    Outputs:
        _cleaned_rot_seg_label: new segmentation label after rotation/translation/cleanup
    """
    from math import pi
    ## check inputs
    # old_segmentation
    if not isinstance(old_segmentation, str) and not isinstance(old_segmentation, np.ndarray) and not isinstance(old_segmentation, np.memmap):
        raise TypeError(f"Wrong data type for old_segmentation:{type(old_segmentation)}, np.ndarray or np.memmap expected")
    elif isinstance(old_segmentation, str):
        if verbose:
            print(f"-- loading segmentation from file:{old_segmentation}")
        old_segmentation = np.load(old_segmentation)
    # old_dapi_im
    if not isinstance(old_dapi_im, str) and not isinstance(old_dapi_im, np.ndarray) and not isinstance(old_dapi_im, np.memmap):
        raise TypeError(f"Wrong data type for old_dapi_im:{type(old_dapi_im)}, np.ndarray or np.memmap expected")
    elif isinstance(old_dapi_im, str):
        if verbose:
            print(f"-- loading old_dapi_im from file:{old_dapi_im}")
        old_dapi_im = corrections.correct_single_image(old_dapi_im, dapi_channel, all_channels=all_channels,
                                               correction_folder=old_correction_folder, 
                                               num_buffer_frames=num_buffer_frames, 
                                               num_empty_frames=num_empty_frames,
                                               verbose=verbose) 
    # new_dapi_im
    if not isinstance(new_dapi_im, str) and not isinstance(new_dapi_im, np.ndarray) and not isinstance(new_dapi_im, np.memmap):
        raise TypeError(f"Wrong data type for new_dapi_im:{type(new_dapi_im)}, np.ndarray or np.memmap expected")
    elif isinstance(new_dapi_im, str):
        if verbose:
            print(f"-- loading new_dapi_im from file:{new_dapi_im}")
        new_dapi_im = corrections.correct_single_image(new_dapi_im, dapi_channel,
                                               all_channels=all_channels,
                                               correction_folder=new_correction_folder, 
                                               num_buffer_frames=num_buffer_frames, 
                                               num_empty_frames=num_empty_frames,
                                               verbose=verbose) 
    # check dimension of rotation_matrix
    if rotation_mat is not None:
        if len(rotation_mat.shape) != 2:
            raise ValueError(f"rotation_mat should be 2d array!")
        d1,d2 = np.shape(rotation_mat)
        if d1 != 2 or d2 != 2:
            raise ValueError(f"Wrong dimension for rotation_mat:{rotation_mat.shape}, should be 2x2 array!")
    elif rotation_ref_file is None:
        raise ValueError(f"rotation_mat and rotation_ref_file should at least have one given!")
    else:
        if os.path.isfile(rotation_ref_file):
            rotation_mat = np.load(rotation_ref_file)
        else:
            raise IOError(f"File:{rotation_ref_file} for rotation reference doesnot exist, exit.")
    # dapi channel should be in all channels
    if dapi_channel not in all_channels:
        raise ValueError(f"dapi_channel:{dapi_channel} is not in all_channels:{all_channels}")
    
    ## 1. calculate translational drift
    if verbose:
        print(f"-- start calculating drift between DAPI images")
    # get dimensions
    _dz,_dx,_dy = np.shape(old_dapi_im)
    # calculate cv2 rotation inputs from given rotation_mat
    _rotation_angle = np.arcsin(rotation_mat[0,1])/pi*180
    _rotation_M = cv2.getRotationMatrix2D((_dx/2, _dy/2), _rotation_angle, 1) # cv2 input rotation_M
    # generated rotated image by rotation at the x-y center
    _rot_new_im = np.array([cv2.warpAffine(_lyr, _rotation_M, _lyr.shape, borderMode=cv2.BORDER_DEFAULT) for _lyr in new_dapi_im], dtype=np.uint16)
    # calculate drift by FFT
    _drift = alignment_tools.fft3d_from2d(old_dapi_im, _rot_new_im, max_disp=fft_max_disp, gb=fft_gb)
    if verbose:
        print(f"--- drift between DAPI images:{_drift}")
    ## 2. rotate segmentation
    print(f"-- start generating translated segmentation labels")
    # define mat to translate old mat into new ones
    _rot_old_M = cv2.getRotationMatrix2D((_dx/2, _dy/2), -_rotation_angle, 1)
    _rot_old_M[:,2] -= np.flipud(_drift[-2:])
    # rotate old image
    #_rot_old_im = np.array([cv2.warpAffine(_lyr, _rot_old_M, _lyr.shape, borderMode=cv2.BORDER_DEFAULT) for _lyr in old_dapi_im], dtype=np.uint16)
    # rotate segmentation
    _rot_seg_label = np.array(cv2.warpAffine(old_segmentation.astype(np.float), 
                                             _rot_old_M, old_segmentation.shape, 
                                             borderMode=cv2.BORDER_CONSTANT), dtype=np.int)
    
    ## 3. generate cleaned_segmentation_label
    _cleaned_rot_seg_label = -1 * np.ones(np.shape(_rot_seg_label))
    for _i in range(np.max(_rot_seg_label)):
        # extract cell_label
        _cell_label = np.array(_rot_seg_label==_i+1)
        # clean up cell label
        _cell_label = ndimage.binary_erosion(_cell_label, structure=morphology.disk(1))
        _cell_label = ndimage.binary_dilation(_cell_label, structure=morphology.disk(1))
        # save to cleaned labels
        _cleaned_rot_seg_label[_cell_label] = _i+1
    
    if return_new_dapi:
        return _cleaned_rot_seg_label, new_dapi_im
    else:
        return _cleaned_rot_seg_label

# translate chromosome coordinates

def translate_chromosome_coordinates(source_cell_data, target_cell_data, rotation_mat=None,
                                     rotation_ref_file=None, rotation_order='reverse', 
                                     single_im_size=_image_size, border_lim=10, 
                                     force=True, add_attribute=True, verbose=True):
    """Function to translate chromosome coordinate given cell_data object before and after translation
    Inputs:
        source_cell_data: cell_data providing chromosome_coordinates, classes.Cell_Data,
        target_cell_data: cell_data to receive translated chromosomes, classes.Cell_Data, 
        rotation_mat: rotation matrix, if provided, np.2darray (default:None), 
        rotation_ref_file file for rotation matrix, string (default:None), 
        rotation_order: whether rotation_mat is forward or reverse, (default:'reverse')
        border_lim: limit to judge whether close to border, int (default:10)
        verbose: say something!, bool (default:True)
    Outputs:
        tar_coords: list of translated chromosome coordinates, list of array-3"""
    ## check input attributes
    if verbose:
        print(f"-- start translating chrom_coord for fov:{source_cell_data.fov_id}, cell:{source_cell_data.cell_id}")
    # check chrom_coords
    if not hasattr(source_cell_data, 'chrom_coords'):
        raise AttributeError(f"Cell_Data:{source_cell_data} doesn't have chromosome coordinates, exit!")
    # load segmentation crop
    if not hasattr(source_cell_data, 'segmentation_crop'):
        source_cell_data._load_segmentation(_load_in_ram=True)
    if not hasattr(target_cell_data, 'segmentation_crop'):
        target_cell_data._load_segmentation(_load_in_ram=True)   
    # check rotation matrix
    if rotation_mat is None and rotation_ref_file is None:
        raise ValueError(f"rotation_mat and rotation_ref_file should at least be given 1!")
    # load rotation_mat
    elif rotation_ref_file is not None:
        if verbose:
            print(f"--- loading rotation matrix from file:{rotation_ref_file}")
        rotation_mat = np.load(rotation_ref_file)
    if len(rotation_mat.shape) != 2 or (np.array(rotation_mat.shape)!=2).any():
        raise ValueError(f"Rotation_mat should be 2x2 array, but {rotation_mat.shape} is given!")
    # check rotation_order
    rotation_order = str(rotation_order).lower()
    if rotation_order not in ['forward','reverse']:
        raise ValueError(f"Wrong input for rotation_order:{rotation_order}, should be forward or reverse!")
    if rotation_order == 'reverse':
        rotation_mat = np.transpose(rotation_mat)
        if verbose:
            print(f"--- {rotation_order}-ing rotation_mat")
    
    # if chrom_coord already exist and not force:
    if not force and hasattr(target_cell_data, 'chrom_coords'):
        return target_cell_data.chrom_coords
    
    ## start rotation!
    
    ## get rotation centers
    # extract crop information
    _src_crop = source_cell_data.segmentation_crop
    _tar_crop = target_cell_data.segmentation_crop
    # init centers
    _src_center = [np.mean(_src_crop[0])]
    _tar_center = [np.mean(_tar_crop[0])]
    # get x,y
    for _fov_lim, _src_lim, _tar_lim in zip(single_im_size[-2:], _src_crop[-2:], _tar_crop[-2:]):
        # now we are assuming a cell cannot go across the whole fov
        if _src_lim[0] < border_lim and _tar_lim[0] < border_lim:
            if verbose:
                print(f"---- both cells are out of fov")
            _ct = max(np.mean(_src_lim) - _src_lim[0], np.mean(_tar_lim)- _tar_lim[0])
            _src_center.append(_src_lim[1] - _src_lim[0] - _ct)
            _tar_center.append(_tar_lim[1] - _tar_lim[0] - _ct)
        elif _src_lim[0] < border_lim:
            _ct = np.mean(_tar_lim) - _tar_lim[0]
            _src_center.append(_src_lim[1] - _src_lim[0] - _ct)
            _tar_center.append(_tar_lim[1] - _tar_lim[0] - _ct)
        elif _tar_lim[0] < border_lim:
            _ct = np.mean(_src_lim)- _src_lim[0]
            _src_center.append(_src_lim[1] - _src_lim[0] - _ct)
            _tar_center.append(_tar_lim[1] - _tar_lim[0] - _ct)
        elif _src_lim[1] > _fov_lim - border_lim and _tar_lim[1] > _fov_lim - border_lim:
            if verbose:
                print(f"---- both cells are out of fov")
            _ct = max(np.mean(_src_lim) - _src_lim[0], np.mean(_tar_lim) - _tar_lim[0])
            _src_center.append(_ct)
            _tar_center.append(_ct)
        elif _src_lim[1] > _fov_lim - border_lim:
            _ct = np.mean(_tar_lim) - _tar_lim[0]
            _src_center.append(_ct)
            _tar_center.append(_ct)
        elif _tar_lim[1] > _fov_lim - border_lim:
            _ct = np.mean(_src_lim) - _src_lim[0]
            _src_center.append(_ct)
            _tar_center.append(_ct)
        else:
            _src_center.append(np.mean(_src_lim) - _src_lim[0])
            _tar_center.append(np.mean(_tar_lim) - _tar_lim[0])
    # transform 
    _src_center, _tar_center = np.array(_src_center), np.array(_tar_center)
    _src_ref_coords = [
        _coord - _src_center for _coord in source_cell_data.chrom_coords]
    _tar_ref_coords = [[_rc[0]]+list(np.dot(rotation_mat, _rc[-2:])) for _rc in _src_ref_coords]
    # get final coord
    _tar_coords  = [np.array(_tc) + _tar_center for _tc in _tar_ref_coords]

    if verbose:
        print(f"--- translated chromsome coordinates:\n{_tar_coords}")
    if add_attribute:
        setattr(target_cell_data, 'chrom_coords', _tar_coords)
        if verbose:
            print(f"--- appended translated chromosomes to target_cell_data")
    return _tar_coords


def translate_spot_coordinates(source_cell_data, target_cell_data, spots, 
                               rotation_mat=None, rotation_ref_file=None, rotation_order='reverse', 
                               single_im_size=_image_size, border_lim=10, 
                               force=True, add_attribute=True, save_attr=None, verbose=True):
    """Function to translate chromosome coordinate given cell_data object before and after translation
    Inputs:
        source_cell_data: cell_data providing chromosome_coordinates, classes.Cell_Data,
        target_cell_data: cell_data to receive translated spots, classes.Cell_Data, 
        rotation_mat: rotation matrix, if provided, np.2darray (default:None), 
        rotation_ref_file file for rotation matrix, string (default:None), 
        rotation_order: whether rotation_mat is forward or reverse, (default:'reverse')
        border_lim: limit to judge whether close to border, int (default:10)
        verbose: say something!, bool (default:True)
    Outputs:
        tar_coords: list of translated chromosome coordinates, list of array-3"""
    ## check input attributes
    if verbose:
        print(f"-- start translating spots for fov:{source_cell_data.fov_id}, cell:{source_cell_data.cell_id}")
    # check chrom_coords
    if not hasattr(source_cell_data, 'chrom_coords'):
        raise AttributeError(f"Cell_Data:{source_cell_data} doesn't have chromosome coordinates, exit!")
    # load segmentation crop
    if not hasattr(source_cell_data, 'segmentation_crop'):
        source_cell_data._load_segmentation(_load_in_ram=True)
    if not hasattr(target_cell_data, 'segmentation_crop'):
        target_cell_data._load_segmentation(_load_in_ram=True)   
    # check rotation matrix
    if rotation_mat is None and rotation_ref_file is None:
        raise ValueError(f"rotation_mat and rotation_ref_file should at least be given 1!")
    # load rotation_mat
    elif rotation_ref_file is not None:
        if verbose:
            print(f"--- loading rotation matrix from file:{rotation_ref_file}")
        rotation_mat = np.load(rotation_ref_file)
    if len(rotation_mat.shape) != 2 or (np.array(rotation_mat.shape)!=2).any():
        raise ValueError(f"Rotation_mat should be 2x2 array, but {rotation_mat.shape} is given!")
    # check rotation_order
    rotation_order = str(rotation_order).lower()
    if rotation_order not in ['forward','reverse']:
        raise ValueError(f"Wrong input for rotation_order:{rotation_order}, should be forward or reverse!")
    if rotation_order == 'reverse':
        rotation_mat = np.transpose(rotation_mat)
        if verbose:
            print(f"--- {rotation_order}-ing rotation_mat")
    
    # if chrom_coord already exist and not force:
    if not force and save_attr is not None and hasattr(target_cell_data, save_attr):
        return getattr(target_cell_data, save_attr)
    
    ## start rotation!
    _spots = np.array(spots)
    _spot_coords = _spots[:, 1:4]
    
    ## get rotation centers
    # extract crop information
    _src_crop = source_cell_data.segmentation_crop
    _tar_crop = target_cell_data.segmentation_crop
    # init centers
    _src_center = [np.mean(_src_crop[0])]
    _tar_center = [np.mean(_tar_crop[0])]
    # get x,y
    for _fov_lim, _src_lim, _tar_lim in zip(single_im_size[-2:], _src_crop[-2:], _tar_crop[-2:]):
        # now we are assuming a cell cannot go across the whole fov
        if _src_lim[0] < border_lim and _tar_lim[0] < border_lim:
            if verbose:
                print(f"---- both cells are out of fov")
            _ct = max(np.mean(_src_lim) - _src_lim[0], np.mean(_tar_lim)- _tar_lim[0])
            _src_center.append(_src_lim[1] - _src_lim[0] - _ct)
            _tar_center.append(_tar_lim[1] - _tar_lim[0] - _ct)
        elif _src_lim[0] < border_lim:
            _ct = np.mean(_tar_lim) - _tar_lim[0]
            _src_center.append(_src_lim[1] - _src_lim[0] - _ct)
            _tar_center.append(_tar_lim[1] - _tar_lim[0] - _ct)
        elif _tar_lim[0] < border_lim:
            _ct = np.mean(_src_lim)- _src_lim[0]
            _src_center.append(_src_lim[1] - _src_lim[0] - _ct)
            _tar_center.append(_tar_lim[1] - _tar_lim[0] - _ct)
        elif _src_lim[1] > _fov_lim - border_lim and _tar_lim[1] > _fov_lim - border_lim:
            if verbose:
                print(f"---- both cells are out of fov")
            _ct = max(np.mean(_src_lim) - _src_lim[0], np.mean(_tar_lim) - _tar_lim[0])
            _src_center.append(_ct)
            _tar_center.append(_ct)
        elif _src_lim[1] > _fov_lim - border_lim:
            _ct = np.mean(_tar_lim) - _tar_lim[0]
            _src_center.append(_ct)
            _tar_center.append(_ct)
        elif _tar_lim[1] > _fov_lim - border_lim:
            _ct = np.mean(_src_lim) - _src_lim[0]
            _src_center.append(_ct)
            _tar_center.append(_ct)
        else:
            _src_center.append(np.mean(_src_lim) - _src_lim[0])
            _tar_center.append(np.mean(_tar_lim) - _tar_lim[0])
    # transform 
    _src_center, _tar_center = np.array(_src_center), np.array(_tar_center)
    _src_ref_coords = [_coord - _src_center for _coord in _spot_coords]
    _tar_ref_coords = [[_rc[0]]+list(np.dot(rotation_mat, _rc[-2:])) for _rc in _src_ref_coords]
    # get final coord
    _tar_coords  = [np.array(_tc) + _tar_center for _tc in _tar_ref_coords]

    if verbose:
        print(f"--- {len(_tar_coords)} spots translated")
    if add_attribute and save_attr is not None:
        setattr(target_cell_data, save_attr, _tar_coords)
        if verbose:
            print(f"--- appended translated chromosomes to target_cell_data")
    # get back to translated spots
    _trans_spots = _spots.copy()
    _trans_spots[:,1:4] = np.array(_tar_coords)
    
    return _trans_spots


# find nearby seeds for given center references, used in bead-drift
def find_matched_seeds(im, ref_centers, search_distance=3, 
                       gfilt_size=0.75, background_gfilt_size=10, filt_size=3, 
                       dynamic=False, th_seed_percentile=95, th_seed=200, 
                       keep_unique=False, verbose=True):
    """Find nearby seeds for on given image for given ref_centers
    Inputs:
        im: image, np.ndarray or np.memmap
        ref_centers: centers to find seeds nearby, list of array-3 or nx3 array
        search_distance: allowed distance between any ref-center to acquired seeds, int (default: 3)
        gilt_size: gaussian_filter size locally for seeding, float (default: 0.75)
        background_gfilt_size: background gaussian_filter size, float (default: 10)
        filt_size: maximum filter to identify local maximum, int (default: 3)
        dynamic: whether do dynamic seeding, bool (default: False)
        th_seed_percentile: intensity percentile for dynamic seeding, float (default: 95)
        th_seed: seeding threshold if not doing dynamic seeding, float (default: 150)
        keep_unique: whether keep only uniquely paired seeds to ref_centers, bool (default: False)
        verbose: say something!, bool (default: True)
    Outputs
        _matched_seeds: seeds found to have match in ref_center, np.nx3 array
        _find_pair: whether each ref_center finds corresponding seed, np.1d-array
    """
    ## check inputs
    if not isinstance(im, np.ndarray) and not isinstance(im, np.memmap):
        raise TypeError(f"Wrong input data type for im, should be np.ndarray or memmap, {type(im)} given!")
    ref_centers = np.array(ref_centers)[:,:3]
    if verbose:
        print(f"- find seeds paired with {len(ref_centers)} centers in given image")
    ## start seeding
    _seeds = get_seed_in_distance(im, center=None, gfilt_size=gfilt_size, 
                                  background_gfilt_size=background_gfilt_size, 
                                  filt_size=filt_size, dynamic=dynamic, 
                                  th_seed_percentile=th_seed_percentile, 
                                  th_seed=th_seed, return_h=True)
    ## find seed match
    _matched_seeds = []
    _find_pair = []
    for _ct in ref_centers:
        _dist = np.linalg.norm(_seeds[:,:3]-_ct[np.newaxis,:], axis=1)
        _matched_inds, = np.where(_dist < search_distance)
        # if unique matched, save and proceed
        if len(_matched_inds) == 1:
            _matched_seeds.append(_seeds[_matched_inds[0], :3])
            _find_pair.append(True)
        # keep the brightest one if multiple matches detected and not keeping unique
        elif len(_matched_inds) > 1 and not keep_unique:
            _cand_seeds = _seeds[_matched_inds, :]
            _intensity_order = np.argsort(_cand_seeds[:,-1]) # sort to find the brightest one
            _matched_seeds.append(_cand_seeds[_intensity_order[-1], :3])
            _find_pair.append(True)
        # otherwise dont record cand_seeds and record find-pair failed
        else:
            _find_pair.append(False)
    
    # return
    _matched_seeds = np.array(_matched_seeds)
    _find_pair = np.array(_find_pair, dtype=np.bool)
    if verbose:
        print(f"-- {len(_matched_seeds)} paired seeds are found. ")
    return _matched_seeds, _find_pair

# select sparse centers given candidate centers
def select_sparse_centers(centers, distance_th=9, distance_norm=np.inf):
    """Select sparse centers from given centers
    Inputs:
        centers: center coordinates (zxy) for all candidates, list of 3-array or nx3 2d-array
        distance_th: threshold for distance between neighboring centers, float (default: 9)
        distance_norm: norm for the distance, int (default: np.inf)
    Output:
        _sel_centers: selected centers, nx3 2d-array"""
    _sel_centers = []
    for ct in centers:
        if len(_sel_centers) == 0:
            _sel_centers.append(ct) # if empty, directly append
        else:
            _sc = np.array(_sel_centers)
            _dists = np.linalg.norm(_sc - ct[np.newaxis,:], axis=1, ord=distance_norm)
            if (_dists <= distance_th).any():
                continue
            else:
                _sel_centers.append(ct)
    
    return np.array(_sel_centers)

# remove a channel in dax file and resave
def Remove_Dax_Channel(source_filename, target_filename, source_channels,
                       keep_channels=_allowed_colors, image_dtype=np.uint16,
                       num_buffer_frames=10, single_im_size=_image_size, num_empty_frames=1,
                       save_dax=True, save_info=True, save_other=True,
                       return_image=True, overwrite=False, verbose=False):
    """Function to remove some channels from a Dax file and re-save
    Inputs:
        source_filename: input dax filename
        target_filename: output dax filename
        source_channels: input dax file-channels
        keep_channels: kept channels in output dax file, which will be converted to subset of source_channels
        image_dtype: datatype for image, datatype from numpy (default: np.uint16)
        num_buffer_frames: number of frames that are at beginning and end of z-scan, int (default: 10)
        single_im_size: 3d image shape for image in one channel, array-like of 3 (default: [30,2048,2048])
        save_dax: whether save .dax file to target_filename, bool (default: True)
        save_info: whether save .inf file together with .dax, bool (default: True)
        save_other: whether save other files together with .dax, bool (default: True)
        return_image: whether return image, bool (default: True)
        overwrite: whether overwrite existing dax files, bool (default: False)
        verbose: say something! bool (default: True)
    """
    ## convert and check input
    # filenames
    if '.dax' not in source_filename or not os.path.isfile(source_filename):
        raise ValueError(
            f"Wrong input source_filename, no correct .dax file exists!")
    # channels
    _ch_before = [str(_ch) for _ch in source_channels]
    _ch_after = [str(_ch) for _ch in keep_channels if _ch in _ch_before]
    if verbose:
        print(f"-- start remove channel from:{source_filename}, \n \
            original channels:{len(_ch_before)}, kept channels:{len(_ch_after)}")
    # get ch-ids to be removed
    _remove_cids = [_i for _i, _ch in enumerate(
        _ch_before) if _ch not in _ch_after]
    # get image ids
    _im_shape, _num_channels = get_img_info.get_num_frame(source_filename, single_im_size[0],
                                                          buffer_frame=num_buffer_frames,
                                                          verbose=verbose)
    if _num_channels != len(source_channels):
        raise ValueError(
            f"Number of channels from info file:{_num_channels} doesn't match given source_channels:{source_channels}, exit!")
    _kept_im = slice_image_remove_channel(source_filename, _im_shape, _num_channels,
                                        remove_zstarts=_remove_cids, empty_frame=num_empty_frames,
                                        zlims=[num_buffer_frames, _im_shape[0]-num_buffer_frames],
                                        image_dtype=image_dtype, verbose=verbose)
    # shuffle frame order
    _sorted_im = get_img_info.shuffle_channel_order(_kept_im, _ch_before, _ch_after,
                                                    zlims=[num_buffer_frames, _kept_im.shape[0]-num_buffer_frames])
    if save_dax:
        if verbose:
            print(f"--- saving spliced images to file:{target_filename}")
        _save_flag = get_img_info.Save_Dax(_sorted_im, target_filename,
                                           source_dax_filename=source_filename,
                                           overwrite=overwrite, save_info_file=save_info,
                                           save_other_files=save_other)
    else:
        _save_flag = False
    if return_image:
        return _sorted_im, _save_flag
    else:
        return _save_flag

def Batch_Remove_Dax_Channel(source_folder, target_folder, source_channels,
                             num_threads=12, keep_channels=_allowed_colors, image_dtype=np.uint16,
                             num_buffer_frames=10, single_im_size=_image_size,
                             save_dax=True, save_info=True, save_other=True,
                             overwrite=False, verbose=False):
    """Batch to remove some channels from a Dax file and re-save, please refer to Remove_Dax_Channel"""
    source_dax_files = glob.glob(os.path.join(source_folder, "*.dax"))
    target_dax_files = [_sfl.replace(source_folder, target_folder) for _sfl in source_dax_files]
    _args = [(_sfl, _tfl, source_channels, keep_channels, 
              image_dtype, num_buffer_frames, single_im_size,
              save_dax, save_info, save_other, False, overwrite, 
              verbose) for _sfl, _tfl in zip(source_dax_files, target_dax_files)]
    if verbose:
        print(f"- Start batch convert dax from {source_folder} to {target_folder}")
    with mp.Pool(num_threads) as _pool:
        if verbose:
            print(f"-- {len(_args)} images processing by {num_threads} threads.")
        _flags = _pool.starmap(Remove_Dax_Channel, _args, chunksize=1)
        _pool.close()
        _pool.join()
        _pool.terminate()
    
    if verbose:
        print(f"-- {np.sum(_flags)} / {len(_args)} iamges been saved.")
    
    return _flags
  

# align chromosome with AB-axis
def max_project_AB_compartment(spots, comp_dict, pca_other_2d=True):
    """Do mass projection onto AB (first two keys) axis of chromosome 
    by norm-centered spots
    Inputs:
        spots: list of spots that that generated from Fitting_V3, 2d np.ndarray or list of 1d np.ndarray
        comp_dict: dictionary of compartment reference, dict of compartment_name -> region_ids        
        pca_other_2d: whether do PCA transformation for other two dimensions, bool (default: True)
    Output:
        _spots: same spots after transformation, 2d np.ndarray   
        """
    ## check inputs
    # spots
    _spots = np.array(spots).copy()
    if len(spots.shape) != 2:
        raise ValueError(f"Input spots should be 2d-array like structure, but shape:{spots.shape} is given!")
    # comp_dict
    if not isinstance(comp_dict, dict):
        raise TypeError(f"Wrong input type of comp_dict, should be dict but {type(comp_dict)} is given")
    
    # extract indices for first two keys in comp_dict
    _ind1 = np.array(list(comp_dict.values())[0], dtype=np.int)
    _ind2 = np.array(list(comp_dict.values())[1], dtype=np.int)
    # get original coordinates
    _coords = _spots[:,1:4]
    # get first axis
    _AB_axis = np.nanmean(_coords[_ind1],axis=0) - np.nanmean(_coords[_ind2], axis=0)
    _AB_axis = _AB_axis / np.linalg.norm(_AB_axis)
    # get basic rotation matrix
    _rot_mat1 = np.zeros([3,3])
    _rot_mat1[2,2] = 1
    _rot_mat1[0,0] = _AB_axis[0]
    _rot_mat1[1,1] = _AB_axis[0]
    _rot_mat1[1,0] = np.sqrt(1-_rot_mat1[1,1]**2)
    _rot_mat1[0,1] = - _rot_mat1[1,0]
    _rot_mat2 = np.zeros([3,3])
    _rot_mat2[0,0] = 1
    _rot_mat2[1,1] = _AB_axis[1] / _rot_mat1[1,0]
    _rot_mat2[2,2] = _rot_mat2[1,1]
    _rot_mat2[2,1] = np.sign(_AB_axis[1]) * np.sqrt(1-_rot_mat2[1,1]**2)
    _rot_mat2[1,2] = - _rot_mat2[2,1]
    # summarize rotation matrix
    _r = _rot_mat2 @ _rot_mat1
    # transform coordinates
    _trans_coords = _coords @ _r
    # if PCA the other two axis, do the following:
    if pca_other_2d:
        _clean_inds = np.array([_i for _i,_c in enumerate(_trans_coords) if not np.isnan(_c).any()],dtype=np.int)
        _clean_2d_coords = _trans_coords[_clean_inds][:,1:3]
        if 'PCA' not in locals():
            from sklearn.decomposition import PCA
        _model_2d = PCA(2)
        _model_2d.fit(_clean_2d_coords)
        _trans_coords[_clean_inds,1:3] = _model_2d.fit_transform(_clean_2d_coords)
    # update spots and retrun
    _spots[:, 1:4] = _trans_coords
    
    return _spots



def remove_cap(im, cap_th_per=99.5, fill_nan=True):
    """Given image and cap percentile, remove top intensity pixels."""
    _corr_im = im.copy()
    if cap_th_per > 0 and cap_th_per < 100:
        if 'scoreatpercentile' not in locals():
            from scipy.stats import scoreatpercentile
        _cap_th = scoreatpercentile(_corr_im[np.isnan(_corr_im)==False], cap_th_per)
        if fill_nan:
            _corr_im[_corr_im > _cap_th] = np.nan
        else:
            _corr_im[_corr_im > _cap_th] = _cap_th
    return _corr_im