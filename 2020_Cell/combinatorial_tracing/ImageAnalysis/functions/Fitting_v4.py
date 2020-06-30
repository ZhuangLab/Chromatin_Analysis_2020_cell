#Author: Bogdan Bintu
#Date: 6/3/2019
# Fitting tools
import numpy as np

#gaussian functions
from scipy import signal
def gker(gaus=[3,3,3],exp=8):
    szgaus = [int(gaus_*exp) for gaus_ in gaus]
    gker_ = np.outer(np.outer(signal.gaussian(szgaus[0], gaus[0]), signal.gaussian(szgaus[1], gaus[1])),
             signal.gaussian(szgaus[2], gaus[2])).reshape(szgaus)
    gker_ = gker_/np.sum(gker_)
    return gker_.astype(np.float32)
def reflect(arr,sizes=[10,10,10]):
    arr_ = arr.copy()
    for axis,size in zip(range(len(arr.shape)),sizes):
        arr_ = arr_.swapaxes(axis,0)
        arr_ = np.concatenate([arr_[:size][::-1],arr_,arr_[-size:][::-1]])
        arr_ = arr_.swapaxes(axis,0)
    return arr_[:-1,:-1,:-1]
def fft_gaussian(arr,gaus=[3,3,3],exp=8):
    gk3d = gker(gaus,exp)
    szs = (np.array(gk3d.shape)/2).astype(int)
    arr_ = reflect(arr,sizes=szs)
    return signal.fftconvolve(arr_,gk3d,mode='valid')
def best_rotation(zns_zxy,zos_zxy):
    """Returns drift and rotation matrix. Use as: np.dot(R,zns_zxy)+t==zos_zxy"""
    PA = np.array(zns_zxy)
    PB = np.array(zos_zxy)
    cA = np.mean(PA,0)
    cB = np.mean(PB,0)
    H=np.dot((PA-cA).T,(PB-cB))
    U,S,V = np.linalg.svd(H)
    R = np.dot(U,V)
    if np.linalg.det(R)<0:
        R[:,-1]*=-1
    t = -np.dot(cA,R)+cB
    #np.dot(R,zns_zxy)+t==zos_zxy
    return t,R

import pyfftw
from pyfftw.interfaces.numpy_fft import rfftn,irfftn
from scipy import fftpack
def _centered(arr, newshape):
    # Return the center newshape portion of the array.
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]
def fftconvolvefast(in1,in2):
    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)

    shape = s1 + s2 - 1

    fshape = [fftpack.helper.next_fast_len(int(d)) for d in shape]
    fslice = tuple([slice(0, int(sz)) for sz in shape])

    sp1 = rfftn(in1, fshape)
    sp2 = rfftn(in2, fshape)
    ret = (irfftn(sp1 * sp2, fshape)[fslice].copy())
    return _centered(ret, s1 - s2 + 1)
def fft_gaussian_fast(arr,gaus=[3,3,3],exp=8):
    gk3d = gker(gaus,exp)
    szs = (np.array(gk3d.shape)/2).astype(int)
    arr_ = reflect(arr,sizes=szs)
    return fftconvolvefast(arr_,gk3d)
from scipy.ndimage.filters import maximum_filter
def get_seed_points_base(im_sm,gfilt_size=5,filt_size=3,th_seed=3.,max_num=None):
    """Get the seed points in an image.
    #1 perform a gaussian filter
    #2 find local maxima within a radius <filt_size> with <th_seed> above local maximum (in log space)
    """
    im_diff = np.log(im_sm)-np.log(fft_gaussian_fast(im_sm,gaus=[gfilt_size]*3))
    im_tomax = im_diff
    im_max = maximum_filter(im_tomax,filt_size)
    z,x,y = np.where(im_tomax == im_max)
    std_ = np.std(im_diff)
    h = im_diff[z,x,y]
    kp = h>th_seed*std_
    z,x,y,h = z[kp],x[kp],y[kp],h[kp]
    centers = np.array([z,x,y])
    #sort by relative brightness
    ind = np.argsort(h)[::-1]
    centers = np.array([z[ind],x[ind],y[ind],h[ind]])
    if max_num is not None:
        centers = centers[:,:max_num]
    return centers,std_

import cv2
def normalzie_im(im,sz=20):
    im_ = np.array(im,dtype=np.float32)
    im_blur = np.array([cv2.blur(im__,(sz,sz)) for im__ in im_],dtype=np.float32)
    im_ =im_-im_blur
    return im_
    
def get_seed_points_base_v2(im_sm,gfilt_size=5,filt_size=3,th_seed=3.,max_num=None):
    """Get the seed points in an image.
    #1 perform a gaussian filter
    #2 find local maxima within a radius <filt_size> with <th_seed> above local maximum (in log space)
    """
    if gfilt_size!=0: im_norm = normalzie_im(im_sm,gfilt_size)
    else: im_norm=im_sm
    std_ = np.std(im_norm)
    hcutoff = std_*th_seed
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
    centers_zxyh = np.array([zf[inds],xf[inds],yf[inds],hf[inds]])
    if max_num is not None:
        centers_zxyh = centers_zxyh[:,:max_num]
    return centers_zxyh,std_




def gauss_ker(sig_xyz=[2,2,2],sxyz=16,xyz_disp=[0,0,0]):
    dim = len(xyz_disp)
    xyz=np.indices([sxyz+1]*dim)
    for i in range(len(xyz.shape)-1):
        sig_xyz=np.expand_dims(sig_xyz,axis=-1)
        xyz_disp=np.expand_dims(xyz_disp,axis=-1)
    im_ker = np.exp(-np.sum(((xyz-xyz_disp-sxyz/2.)/sig_xyz)**2,axis=0)/2.)
    return im_ker
def add_source(im_,pos=[0,0,0],h=200,sig=[2,2,2]):
    im=np.array(im_,dtype=float)
    pos_int = np.array(pos,dtype=int)
    xyz_disp = -pos_int+pos
    im_ker = gauss_ker(sig_xyz=sig,sxyz=int(np.max(sig)*20),xyz_disp=xyz_disp)
    im_ker_sz = np.array(im_ker.shape,dtype=int)
    pos_min = (pos_int-im_ker_sz/2).astype(int)
    pos_max = pos_min+im_ker_sz
    im_shape = np.array(im.shape)
    def in_im(pos__):
        pos_=np.array(pos__,dtype=int)
        pos_[pos_>=im_shape]=im_shape[pos_>=im_shape]-1
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
    

    
from scipy.optimize import leastsq
class GaussianFit():
    def __init__(self,im,X,center=None,n_aprox=10,min_w=0.5,max_w=4.,delta_center=3.):
        self.min_w = min_w*min_w
        self.max_w = max_w*max_w
        
        self.delta_center = delta_center
        self.im = np.array(im,dtype=np.float32)
        self.x,self.y,self.z = np.array(X,dtype=np.float32)
        #get estimates
        argsort_im = np.argsort(im)
        if center is None:
            center = np.median(X[:,argsort_im][:,-n_aprox:],-1)
        self.center_est = center
        sorted_im = im[argsort_im]
        eps =  np.exp(-10.)
        bk_guess = np.log(np.max([np.mean(sorted_im[:n_aprox]),eps]))
        h_guess = np.log(np.max([np.mean(sorted_im[-n_aprox:]),eps]))
        wsq = 1.5**2
        wg = np.log((self.max_w - wsq)/(wsq-self.min_w))
        self.p_ = np.array([bk_guess,h_guess,0,0,0,wg,wg,wg,0,0],dtype=np.float32)
        self.to_natural_paramaters()
        self.success = False
    def to_center(self,c0_,c1_,c2_):
        """constrains via sigmoidal function close to local center"""
        delta = self.delta_center
        c0 = 2.*delta/(1.+np.exp(c0_))-delta+self.center_est[0]
        c1 = 2.*delta/(1.+np.exp(c1_))-delta+self.center_est[1]
        c2 = 2.*delta/(1.+np.exp(c2_))-delta+self.center_est[2]
        return c0,c1,c2
    def to_sine(self,t_):
        """constrain sin-angles to -1,1"""
        #eps =  10E-5
        #self.sine_eps = eps
        #return 2.*(1-eps)/(1+np.exp(t_))-1.+eps
        return 2./(1+np.exp(t_))-1.
    def to_ws(self,w_):
        """constrain widths"""
        min_ws = self.min_w
        delta_ws = self.max_w - min_ws
        ws = delta_ws/(1.+np.exp(w_))+min_ws
        return ws
    def to_natural_paramaters(self,parms=None):
        """
        Convert from constrained paramaters to [hf,xc,yc,zc,bkf,w1f,w2f,w3f,t,p,eps]
        """
        if parms is None:
            parms = self.p_
        bk,h,xp,yp,zp,w1,w2,w3,pp,tp = parms
        bkf,hf=np.exp(bk),np.exp(h)
        t,p = self.to_sine(tp),self.to_sine(pp)
        w1f,w2f,w3f = np.sqrt(self.to_ws(w1)),np.sqrt(self.to_ws(w2)),np.sqrt(self.to_ws(w3))
        xc,yc,zc = self.to_center(xp,yp,zp)
        eps = self.calc_eps(parms)
        eps = np.mean(np.abs(eps))
        self.p = np.array([hf,xc,yc,zc,bkf,w1f,w2f,w3f,t,p,eps],dtype=np.float32)
        return self.p
    def calc_f(self,parms):
        self.p_ = parms
        
        bk,h,xp,yp,zp,w1,w2,w3,pp,tp = parms
        t,p = self.to_sine(tp),self.to_sine(pp)
        ws1,ws2,ws3 = self.to_ws(w1),self.to_ws(w2),self.to_ws(w3)
        xc,yc,zc = self.to_center(xp,yp,zp)
        xt,yt,zt = self.x-xc,self.y-yc,self.z-zc
        
        p2 = p*p
        t2 = t*t
        tc2 = 1-t2
        pc2 = 1-p2
        tc= np.sqrt(tc2)
        pc= np.sqrt(pc2)
        s1,s2,s3 = 1./ws1,1./ws2,1./ws3
        x2c = pc2*tc2*s1 + t2*s2 + p2*tc2*s3
        y2c = pc2*t2*s1 + tc2*s2 + p2*t2*s3
        z2c = p2*s1 + pc2*s3
        xyc = 2*tc*t*(pc2*s1 - s2 + p2*s3)
        xzc = 2*p*pc*tc*(s3 - s1)
        yzc = 2*p*pc*t*(s3 - s1)
        
        
        
        xsigmax = x2c*xt*xt+y2c*yt*yt+z2c*zt*zt+xyc*xt*yt+xzc*xt*zt+yzc*yt*zt
        self.f0 = np.exp(h-0.5*xsigmax)
        self.f = np.exp(bk)+self.f0
        
        return self.f
    
    def frac_conv(self,x1,x2):return 2*np.abs(x1-x2)/(x1+x2)<self.eps_frac
    def dist_conv(self,x1,x2):return np.abs(x1-x2)<self.eps_dist
    def angle_conv(self,x1,x2):return np.abs(x1-x2)<self.eps_angle
    def calc_eps(self,parms):
        """
        calculate the loss function
        """
        #Decided not to include this extra step of convergence
        """
        if self.p_old is not None:
            if np.any((parms-self.parms_old)!=0):
                p_new = self.to_natural_paramaters(parms)
                h1,x1,y1,z1,bk1,wx1,wy1,wz1,t1,p1 = self.p_old[:10]
                h2,x2,y2,z2,bk2,wx2,wy2,wz2,t2,p2 = p_new[:10]
                #print self.p_old,p_new
                self.p_old = p_new
                
                converged = self.frac_conv(h1,h2) and self.frac_conv(bk1,bk2) 
                converged = converged and self.dist_conv(x1,x2) and self.dist_conv(y1,y2) and self.dist_conv(z1,z2)
                converged = converged and self.dist_conv(wx1,wx2) and self.dist_conv(wy1,wy2) and self.dist_conv(wz1,wz2)
                converged = converged and self.angle_conv(t1,t2) and self.angle_conv(p1,p2)
                if converged:
                    self.converged = True
                    return np.zeros(len(self.im),dtype=np.float32)
        else:
            self.p_old = self.to_natural_paramaters(parms)
            self.parms_old = parms
        """
        return self.calc_f(parms)-self.im
    def calc_jac(self,parms):
        bk,h,xp,yp,zp,w1,w2,w3,pp,tp = parms
        t,p = self.to_sine(tp),self.to_sine(pp)
        ws1,ws2,ws3 = self.to_ws(w1),self.to_ws(w2),self.to_ws(w3)
        xc,yc,zc = self.to_center(xp,yp,zp)
        xt,yt,zt = self.x-xc,self.y-yc,self.z-zc
        p2 = p*p
        t2 = t*t
        tc2 = 1-t2
        pc2 = 1-p2
        tc= np.sqrt(tc2)
        pc= np.sqrt(pc2)
        s1,s2,s3 = 1./ws1,1./ws2,1./ws3
        x2c = pc2*tc2*s1 + t2*s2 + p2*tc2*s3
        y2c = pc2*t2*s1 + tc2*s2 + p2*t2*s3
        z2c = p2*s1 + pc2*s3
        xyc = 2*tc*t*(pc2*s1 - s2 + p2*s3)
        xzc = 2*p*pc*tc*(s3 - s1)
        yzc = 2*p*pc*t*(s3 - s1)
        xt2,xtyt,xtzt,yt2,ytzt,zt2 = xt*xt,xt*yt,xt*zt,yt*yt,yt*zt,zt*zt
        xsigmax = x2c*xt2+y2c*yt2+z2c*zt2+xyc*xtyt+xzc*xtzt+yzc*ytzt
        
        d,minw,maxw = self.delta_center,self.min_w,self.max_w
        
        
        f2 = np.exp(h-0.5*xsigmax)
        f1 =  np.exp(bk)+np.zeros(len(f2))
        e_xp,e_yp,e_zp = np.exp(-np.abs(xp)),np.exp(-np.abs(yp)),np.exp(-np.abs(zp))
        norm_xp = -d*e_xp/((1 + e_xp)*(1 + e_xp))
        norm_yp = -d*e_yp/((1 + e_yp)*(1 + e_yp))
        norm_zp = -d*e_zp/((1 + e_zp)*(1 + e_zp))
        f3 = (f2*(2*x2c*xt + xyc*yt + xzc*zt))*norm_xp
        f4 = (f2*(xt*xyc + 2*y2c*yt + yzc*zt))*norm_yp
        f5 = (f2*(xt*xzc + yt*yzc + 2*z2c*zt))*norm_zp
        f6 = (f2*(-pc2*tc2*xt2 - 2*pc2*t*tc*xtyt - pc2*t2*yt2 + 2*p*pc*tc*xtzt + 2*p*pc*t*ytzt - p2*zt2))*self.norm_w(w1,minw,maxw)
        f7 = (f2*(-t2*xt2 + 2*t*tc*xtyt - tc2*yt2))*self.norm_w(w2,minw,maxw)
        f8 = (f2*(-p2*tc2*xt2 - 2*p2*t*tc*xtyt - p2*t2*yt2 - 2*p*pc*tc*xtzt - 2*p*pc*t*ytzt - pc2*zt2))*self.norm_w(w3,minw,maxw)
        e_p = np.exp(-np.abs(pp)/2)
        norm_p = e_p/(1+e_p*e_p)
        f9 = f2*(s3-s1)*((2*pc2-1.)*(tc*xtzt + t*ytzt) + p*pc*(tc2*xt2 + 2*t*tc*xtyt + t2*yt2 - zt2))*norm_p
        e_t = np.exp(-np.abs(tp)/2)
        norm_t = e_t/(1+e_t*e_t)
        f10 = f2*((pc2 *s1 - s2 +  p2*s3)*(t *tc*(yt2 - xt2) - (t2 - tc2)*xtyt) + p* pc *(s1 - s3)* (t *xtzt - tc*ytzt))*norm_t
        
        self.jac = np.array([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10],np.float32).T
        
        return self.jac

    def norm_w(self,w,minw,maxw):
        if w>0:
            e_w = np.exp(-w)
            return 0.5*(maxw - minw)*e_w/(maxw*e_w+minw)**2
        else:
            e_w = np.exp(w)
            return 0.5*(maxw - minw)*e_w/(minw*e_w+maxw)**2
    def fit(self,eps_frac=10E-3,eps_dist=10E-3,eps_angle=10E-3):
        """
        This implements the Levenberg-Marquardt algorithm for 3D gaussian fitting.
        Stores the results in [height,x,y,z,background,width_1,width_2,width_3,sin_theta,sin_phi,error] = self.p
        """
        if len(self.p_)>len(self.im):
            self.success = False
        else:
            self.eps_frac,self.eps_dist,self.eps_angle = eps_frac,eps_dist,eps_angle
            parms0 = self.p_
            self.p_old = None
            parmsf,_ = leastsq(self.calc_eps,parms0,Dfun=self.calc_jac)
            #parmsf=parms0#####pisici
            self.p_ = parmsf
            self.to_natural_paramaters()
            self.center = self.p[1:4]
            self.success = True
    def get_im(self):
        self.calc_f(self.p_)
        return self.f0
        


def in_dim(x,y,z,xmax,ymax,zmax):
    keep = ((x>=0)&(x<xmax)&(y>=0)&(y<ymax)&(z>=0)&(z<zmax))>0
    return x[keep],y[keep],z[keep]
from scipy.spatial.distance import cdist
def closest(x,y,z,ic,all_centers):
    xyz = np.array([x,y,z]).T
    dists = cdist(xyz,all_centers)
    closest_point_id = np.argmin(dists,axis=-1)
    keep = closest_point_id==ic
    xyz_keep = xyz[keep,:]
    return xyz_keep.T
def closest_fast(x,y,z,ic,all_centers,rsearch =6):
    xyz = np.array([x,y,z]).T
    dists = cdist([all_centers[ic]],all_centers,'chebyshev')[0,:]
    close = dists<rsearch
    close_inds = np.arange(len(all_centers))[close]
    close_centers = all_centers[close]
    dists = cdist(xyz,close_centers)
    closest_point_id = np.argmin(dists,axis=-1)
    keep = close_inds[closest_point_id]==ic
    xyz_keep = xyz[keep,:]
    return xyz_keep.T
from scipy.spatial import cKDTree as KDTree
def closest_faster(xyz,ic,tree,rsearch = 6):
    dists_,nns_ = tree.query(xyz,distance_upper_bound=rsearch)
    return xyz[nns_==ic].T

def inv_sigma(sigma):
    [[a, d, e], [d, b, f], [e, f, c]] = sigma
    
    sigma_inv = np.array([[(b*c - f ** 2)/(a*b*c - c*d ** 2 - b*e ** 2 + 2*d*e*f - a*f ** 2), (-(c*d) + e*f)/(a*b*c - c*d ** 2 - b*e ** 2 + 2*d*e*f - a*f ** 2), (-(b*e) + d*f)/(a*b*c - c*d ** 2 - b*e ** 2 + 2*d*e*f - a*f ** 2)], 
                      [(-(c*d) + e*f)/(a*b*c - c*d ** 2 - b*e ** 2 + 2*d*e*f - a*f ** 2), (a*c - e ** 2)/(a*b*c - c*d ** 2 - b*e ** 2 + 2*d*e*f - a*f ** 2), (d*e - a*f)/(a*b*c - c*d ** 2 - b*e ** 2 + 2*d*e*f - a*f ** 2)], 
                      [(-(b*e) + d*f)/(a*b*c - c*d ** 2 - b*e ** 2 + 2*d*e*f - a*f ** 2), (d*e - a*f)/(a*b*c - c*d ** 2 - b*e ** 2 + 2*d*e*f - a*f ** 2), (a*b - d ** 2)/(a*b*c - c*d ** 2 - b*e ** 2 + 2*d*e*f - a*f ** 2)]])
    return sigma_inv
def gfit_fast(im_,X_,bk_f=0.1,reconstruct=False,plt_val=False,compare_with_fitting=False):
    pfinal = np.array([np.nan]*12)
    if len(im_)>0:
        #background, height and new weights
        nim = len(im_)
        bk = np.sort(im_)[int(nim*bk_f)]
        weights = im_-bk
        weights[weights<0]=0
        h = np.max(weights)
        weights = weights/np.sum(weights)
        
        #approximate position
        zxy_ = np.sum(X_*weights,-1)
        z_,x_,y_ = zxy_
        #approximate shape
        X_c = X_.T-zxy_
        Cov = np.array([[ X_c[:,i]*X_c[:,j]for i in range(3)] for j in range(3)])
        Cov = np.sum(Cov*weights,-1)
        [[a, d, e], [d, b, f], [e, f, c]] = Cov
        #compute reconstruction
        eps = np.nan
        if reconstruct:
            iCov = inv_sigma(Cov)
            im_fit = h*np.exp(-np.sum(np.dot(X_c,iCov)*X_c,-1)*0.5)+bk
            eps = np.mean(np.abs(im_-im_fit))
        pfinal = np.array([h,z_,x_,y_,bk,a,b,c,d,e,f,eps])
    if plt_val:
        import matplotlib.pylab as plt
        zs,xs,ys = np.min(X_,1) 
        
        X_im = X_.T-[zs,xs,ys]
        size_im = np.max(X_im.T,1)+1
        im__ = np.zeros(size_im)+np.nan
        im__[X_im[:,0],X_im[:,1],X_im[:,2]]=im_
        plt.figure()
        plt.imshow(np.nanmax(im__,0))
        plt.plot(y_-ys,x_-xs,'bx')
        # plot reconstructed image and fit
        im__ = np.zeros(size_im)+np.nan
        im__[X_im[:,0],X_im[:,1],X_im[:,2]]=im_fit
        plt.figure()
        plt.imshow(np.nanmax(im__,0))
        plt.plot(y_-ys,x_-xs,'bx')
        plt.figure()
        plt.plot(im_,im_fit,'o')
        if compare_with_fitting:
            center = X_.T[np.argmax(im_)]
            obj = GaussianFit(im_,X_,center=center,delta_center=1.5)
            obj.fit()
            im_fit2 = obj.get_im()+obj.p[4]
            im__ = np.zeros(size_im)+np.nan
            im__[X_im[:,0],X_im[:,1],X_im[:,2]]=im_fit2
            plt.figure()
            plt.imshow(np.nanmax(im__,0))
            plt.plot(y_-ys,x_-xs,'bx')
            plt.plot(obj.p[3]-ys,obj.p[2]-xs,'rx')
            #print obj.p[1:3]-pfinal[1:3]
    return pfinal
def in_dim(x,y,z,xmax,ymax,zmax):
    keep = ((x>=0)&(x<xmax)&(y>=0)&(y<ymax)&(z>=0)&(z<zmax))>0
    return x[keep],y[keep],z[keep]

from scipy.spatial import cKDTree as KDTree
def fast_fit_big_image(im,centers_zxy,radius_fit = 4,avoid_neigbors=True,recenter=False ,verbose = True,better_fit=False,troubleshoot=False):
    ps = []
    if len(centers_zxy)>0:
        if avoid_neigbors:
            centers_tree = KDTree(centers_zxy)
            list_inters = centers_tree.query_ball_tree(centers_tree,radius_fit*2)
        centers_ = centers_zxy
        if verbose:
            from tqdm import tqdm_notebook as tqdm
            centers_ = tqdm(centers_zxy)
        zb,xb,yb = np.reshape(np.indices([radius_fit*2]*3)-radius_fit,[3,-1]).astype(int)
        keep = zb*zb+xb*xb+yb*yb<=radius_fit**2
        zb,xb,yb = zb[keep],xb[keep],yb[keep]
        sz,sx,sy = im.shape
        X_c = np.array([zb,xb,yb]).T

        for ic,(zc,xc,yc) in enumerate(centers_):
            if avoid_neigbors:
                common_pts_ids = list_inters[ic]
                index_ic = list_inters[ic].index(ic)
                centers__ = centers_zxy[common_pts_ids]-[zc,xc,yc]
                nns_ = np.argmin(cdist(centers__,X_c),0)
                zb_,xb_,yb_ = X_c[nns_==index_ic].T
            else:
                zb_,xb_,yb_ = zb,xb,yb
            
            z_keep,x_keep,y_keep = int(zc)+zb_,int(xc)+xb_,int(yc)+yb_
            z_keep,x_keep,y_keep = in_dim(z_keep,x_keep,y_keep,sz,sx,sy)

            X_ = np.array([z_keep,x_keep,y_keep]).T
            im_ = im[z_keep,x_keep,y_keep]
            
            #recenter to max in the imge - generally indicated if using seeds from a different image
            if recenter:
                if len(im_)>0:
                    #while True:
                    im_c = np.argmax(im_)
                    zcn,xcn,ycn = z_keep[im_c],x_keep[im_c],y_keep[im_c]
                    #if zcn==zc and xcn==xc and ycn==yc:
                    #    break
                    zc,xc,yc = zcn,xcn,ycn
                    z_keep,x_keep,y_keep = int(zc)+zb_,int(xc)+xb_,int(yc)+yb_
                    z_keep,x_keep,y_keep = in_dim(z_keep,x_keep,y_keep,sz,sx,sy)

                    X_ = np.array([z_keep,x_keep,y_keep]).T
                    im_ = im[z_keep,x_keep,y_keep]
                        
            if not better_fit:
                reconstruct,plt_val,compare_with_fitting=False,False,False
                if troubleshoot:
                    reconstruct,plt_val,compare_with_fitting=True,True,True
                p_ = gfit_fast(im_,X_.T,bk_f=0.1,
                                reconstruct=reconstruct,plt_val=plt_val,compare_with_fitting=compare_with_fitting)
            else:
                p_ = np.array([np.nan]*11)
                if len(im_)>0:
                    center = X_[np.argmax(im_)]
                    obj = GaussianFit(im_,X_.T,center=center,delta_center=2.5)
                    obj.fit()
                    p_  = obj.p
            ps.append(p_)
    ps = np.array(ps)
    return ps
class iter_fit_seed_points():
    def __init__(self,im,centers,radius_fit=5,min_delta_center=1.,max_delta_center=2.5,n_max_iter = 10,max_dist_th=0.1):
        """
        Given a set of seeds <centers> in a 3d image <im> iteratively 3d gaussian fit around the seeds (in order of brightness) 
        and subtract the gaussian signal.
        Retruns a numpy array of size Nx(height, x, y, z, width_x, width_y,width_z,background) where N~len(centers). Bad fits are disregarded.
        Warning: Generally a bit slow. In practice, the faster version fast_local_fit is used.
        """
        
        #internalize
        self.im = im
        self.radius_fit = radius_fit
        self.n_max_iter = n_max_iter
        self.max_dist_th = max_dist_th
        self.min_delta_center = min_delta_center
        self.max_delta_center = max_delta_center
        
        self.centers = centers.T
        self.z,self.x,self.y = centers
        self.zb,self.xb,self.yb = np.reshape(np.indices([self.radius_fit*2]*3)-self.radius_fit,[3,-1])
        keep = self.zb*self.zb+self.xb*self.xb+self.yb*self.yb<=self.radius_fit**2
        self.zb,self.xb,self.yb = self.zb[keep],self.xb[keep],self.yb[keep]
        self.zxyb = np.array([self.zb,self.xb,self.yb]).T
        self.sz,self.sx,self.sy = im.shape
        


                    
    def firstfit(self):
        """
        Perform a first fit on the sample with the gaussian constrained close to the local maximum
        """
        if len(self.centers)>0:
            #fit the points in order of brightness and at each fit subtract the fitted signal
            self.ps = []
            self.ims_rec=[]
            self.im_subtr = np.array(self.im,dtype=float)
            self.centers_fit = []
            self.success=[]
            self.centers_tree = KDTree(self.centers)
            from tqdm import tqdm_notebook as tqdm
            centers_ = tqdm(self.centers)
            self.gparms = []
            for ic,(zc,xc,yc) in enumerate(centers_):
                z_keep,x_keep,y_keep = int(zc)+self.zb,int(xc)+self.xb,int(yc)+self.yb
                z_keep,x_keep,y_keep = in_dim(z_keep,x_keep,y_keep,self.sz,self.sx,self.sy)
                
                X_full = np.array([z_keep,x_keep,y_keep],dtype=int)
                center = [zc,xc,yc]
                z_keep,x_keep,y_keep = closest_faster(X_full.T,ic,self.centers_tree,rsearch = self.radius_fit*2)
                
                X = np.array([z_keep,x_keep,y_keep])
                im_ = self.im[z_keep,x_keep,y_keep]
                
                self.gparms
                
                obj = GaussianFit(im_,X,center=center,delta_center=self.min_delta_center)
                obj.fit()
                
                self.gparms.append([im_,X,center])
                n_p = len(obj.p)
                self.success.append(obj.success)
                if obj.success:
                    self.ps.append(obj.p)
                    self.centers_fit.append(obj.center)
                    z_keep,x_keep,y_keep = X_full
                    obj.x,obj.y,obj.z = X_full
                    im_rec = obj.get_im()
                    self.ims_rec.append(im_rec)
                    self.im_subtr[z_keep,x_keep,y_keep] -= im_rec
                else:
                    self.ims_rec.append(np.nan)
                    self.ps.append([np.nan]*n_p)
                    self.centers_fit.append([np.nan]*3)
                
        self.im_add = np.array(self.im_subtr)
        
    def repeatfit(self):
            self.n_iter = 0
            
            self.converged = np.zeros(len(self.centers),dtype=bool)
            self.dists = np.zeros(len(self.centers))+np.inf
            converged = np.all(self.converged)
            while not converged:
                
                self.success_old,self.centers_fit_old=np.array(self.success),np.array(self.centers_fit)
                #self.ps_old=np.array(self.ps)
                for ic,(zc,xc,yc) in enumerate(self.centers):
                    if not self.converged[ic]:
                        #get modified image positions
                        z_keep,x_keep,y_keep = int(zc)+self.zb,int(xc)+self.xb,int(yc)+self.yb
                        z_keep,x_keep,y_keep = in_dim(z_keep,x_keep,y_keep,self.sz,self.sx,self.sy)
                        X = np.array([z_keep,x_keep,y_keep])
                        #get image, adding back the fit
                        im_ = self.im_add[z_keep,x_keep,y_keep]
                        
                        if self.success_old[ic]:
                            im_rec = self.ims_rec[ic]
                            im_=im_rec+im_ #add back the image
                            

                        delta_center = self.max_delta_center
                        obj = GaussianFit(im_,X,center=[zc,xc,yc],delta_center=delta_center)
                        obj.fit()
                        self.success[ic] = obj.success
                        if obj.success:
                            im_rec = obj.get_im()
                            self.ps[ic]=obj.p
                            self.centers_fit[ic]=obj.center
                            self.ims_rec[ic]=im_rec
                            self.im_add[z_keep,x_keep,y_keep] = im_-im_rec
                
                keep = (np.array(self.success)&np.array(self.success_old))>0
                self.dists[~keep]=0
                self.dists[keep]=np.sum((np.array(self.centers_fit_old)[keep]-np.array(self.centers_fit)[keep])**2,axis=-1)
                self.converged = self.dists<self.max_dist_th**2
                converged = np.all(self.converged)
                self.n_iter+=1
                converged = converged or (self.n_iter>self.n_max_iter)
                
def to_sigmas(t,p,w_1,w_2,w_3):
    """Given t = sin(theta), p = sin(phi), and the stds this computes the covariance matrix and its inverse"""
    p2 = p*p
    t2 = t*t
    tc2 = 1-t2
    pc2 = 1-p2
    tc= np.sqrt(tc2)
    pc= np.sqrt(pc2)
    s1,s2,s3 = 1./(w_1*w_1),1./(w_2*w_2),1./(w_3*w_3)
    a = pc2*tc2*s1 + t2*s2 + p2*tc2*s3
    b = pc2*t2*s1 + tc2*s2 + p2*t2*s3
    c = p2*s1 + pc2*s3
    d = tc*t*(pc2*s1 - s2 + p2*s3)
    e = p*pc*tc*(s3 - s1)
    f = p*pc*t*(s3 - s1)
    sigma_inv = np.array([[a, d, e], [d, b, f], [e, f, c]])
    sigma = np.array([[(b*c - f ** 2)/(a*b*c - c*d ** 2 - b*e ** 2 + 2*d*e*f - a*f ** 2), (-(c*d) + e*f)/(a*b*c - c*d ** 2 - b*e ** 2 + 2*d*e*f - a*f ** 2), (-(b*e) + d*f)/(a*b*c - c*d ** 2 - b*e ** 2 + 2*d*e*f - a*f ** 2)], 
                      [(-(c*d) + e*f)/(a*b*c - c*d ** 2 - b*e ** 2 + 2*d*e*f - a*f ** 2), (a*c - e ** 2)/(a*b*c - c*d ** 2 - b*e ** 2 + 2*d*e*f - a*f ** 2), (d*e - a*f)/(a*b*c - c*d ** 2 - b*e ** 2 + 2*d*e*f - a*f ** 2)], 
                      [(-(b*e) + d*f)/(a*b*c - c*d ** 2 - b*e ** 2 + 2*d*e*f - a*f ** 2), (d*e - a*f)/(a*b*c - c*d ** 2 - b*e ** 2 + 2*d*e*f - a*f ** 2), (a*b - d ** 2)/(a*b*c - c*d ** 2 - b*e ** 2 + 2*d*e*f - a*f ** 2)]])
    return sigma,sigma_inv
def to_sigmas_abc(a,b,c,d,e,f):
    sigma = np.array([[a, d, e], [d, b, f], [e, f, c]])
    sigma_inv = np.array([[b*c - f*f,-(c*d) + e*f,-(b*e) + d*f],[-(c*d) + e*f,a*c - e*e,d*e - a*f],[-(b*e) + d*f,d*e - a*f,a*b - d*d]])
    det_sigma = a*b*c - c*d**2 - b*e**2 + 2*d*e*f - a*f**2
    sigma_inv/=det_sigma
    return sigma,sigma_inv,det_sigma
def KL_dif(p1,p2):
    """Given two gaussian paramters p1 and p2 this computes their KL divergence."""
    if len(p1)<11:
        return 0
    elif len(p1)==11:
        [height1,center_01,center_11,center_21,background1,widths_01,widths_11,widths_21,t_1,p_1,e1]=p1
        [height2,center_02,center_12,center_22,background2,widths_02,widths_12,widths_22,t_2,p_2,e2]=p2
        sigma1,sigma_inv1 = to_sigmas(*[t_1,p_1,widths_01,widths_11,widths_21])
        sigma2,sigma_inv2 = to_sigmas(*[t_2,p_2,widths_02,widths_12,widths_22])
        k = len(sigma1)
        inds = np.arange(k)
        return 0.5*(np.sum(np.dot(sigma_inv2,sigma1)[inds,inds])-k+np.log(np.linalg.det(sigma2))-np.log(np.linalg.det(sigma1)))
    else:
        sigma1,sigma_inv1,det1 =to_sigmas_abc(*p1[5:11])
        sigma2,sigma_inv2,det2 =to_sigmas_abc(*p2[5:11])
        k = len(sigma1)
        inds = np.arange(k)
        #
        #return 0.5*(np.sum(np.dot(sigma_inv2,sigma1)[inds,inds])-k+np.log(det2)-np.log(det1))
        return 0
##2d images
import cv2
def blurnorm2d(im,gb):
    """Normalize an input 2d image <im> by dividing by a cv2 gaussian filter of the image"""
    im_ = im.astype(np.float32)
    blurred = cv2.blur(im_,(gb,gb))
    return im_-blurred
def fft3d_from2d(im1,im2,gb=5,max_disp=150,plt_val=False,return_cor=False):
    """Given a refence 3d image <im1> and a target image <im2> 
    this max-projects along the first (z) axis and finds the best tx,ty using fftalign_2d.
    Then it trims and max-projects along the last (y) axis and finds tz.
    Before applying fftalignment we normalize the images using blurnorm2d for stability."""
    im1_ = blurnorm2d(np.max(im1,0),gb)
    im2_ = blurnorm2d(np.max(im2,0),gb)
    tx,ty,cor_xy = fftalign_2d(im1_, im2_, center=[0, 0], max_disp=max_disp, plt_val=plt_val,return_cor=True)
    sx,sy = im1_.shape
    im1_t = blurnorm2d(np.max(im1[:,max(tx,0):sx+tx,max(ty,0):sy+ty],axis=-1),gb)
    im2_t = blurnorm2d(np.max(im2[:,max(-tx,0):sx-tx,max(-ty,0):sy-ty],axis=-1),gb)
    tz,_,cor_z = fftalign_2d(im1_t, im2_t, center=[0, 0], max_disp=max_disp, plt_val=plt_val,return_cor=True)
    if return_cor:
        return np.array([tz,tx,ty]),cor_xy,cor_z
    return np.array([tz,tx,ty])
import matplotlib.pylab as plt
def fftalign_2d(im1,im2,center=[0,0],max_disp=50,plt_val=False,return_cor=False):
    """
    Inputs: 2 2D images <im1>, <im2>, the expected displacement <center>, the maximum displacement <max_disp> around the expected vector.
    This computes the cross-cor between im1 and im2 using fftconvolve (fast) and determines the maximum
    """    
    from scipy.signal import fftconvolve
    im2_=np.array(im2[::-1,::-1],dtype=float)
    im2_-=np.mean(im2_)
    im2_/=np.std(im2_)
    im1_=np.array(im1,dtype=float)
    im1_-=np.mean(im1_)
    im1_/=np.std(im1_)
    im_cor = fftconvolve(im1_,im2_, mode='full')

    sx_cor,sy_cor = im_cor.shape
    center_ = np.array(center)+np.array([sx_cor,sy_cor])/2.
    
    x_min = int(min(max(center_[0]-max_disp,0),sx_cor))
    x_max = int(min(max(center_[0]+max_disp,0),sx_cor))
    y_min = int(min(max(center_[1]-max_disp,0),sy_cor))
    y_max = int(min(max(center_[1]+max_disp,0),sy_cor))
    
    im_cor0=np.zeros_like(im_cor)
    im_cor0[x_min:x_max,y_min:y_max]=1
    im_cor = im_cor*im_cor0
       
    y, x = np.unravel_index(np.argmax(im_cor), im_cor.shape)
    xt,yt=np.array([y,x],dtype=int)-im2_.shape+1
    if np.sum(im_cor>0)>0:
        im_cor[im_cor==0]=np.min(im_cor[im_cor>0])
    else:
        im_cor[im_cor==0]=0
    if plt_val:
        plt.figure()
        plt.plot([x],[y],'k+')
        plt.imshow(im_cor,interpolation='nearest')
        plt.show()
        std_2 = np.std(im2)
        std_1 = np.std(im1)
        im2t = translate(minmax(im2,min_=-2*std_2,max_=std_2*2),[-xt,-yt])
        im1t = minmax(im1,min_=-2*std_1,max_=std_1*2)
        #print im1.shape,im2.shape,im1t.shape,im2t.shape
        #print im_cor.shape,y, x
        szx,szy = np.min([im2t.shape,im1t.shape],axis=0)
        plt.figure()
        plt.imshow(im1t[:szx,:szy])
        plt.figure()
        plt.imshow(im2t[:szx,:szy])
        plt.figure()
        plt.imshow(np.dstack([im2t[:szx,:szy],im1t[:szx,:szy],im2t[:szx,:szy]]))
        plt.show()
    if return_cor:
        return xt,yt,np.max(im_cor)/np.prod(np.min([im1_.shape,im2_.shape],0))
    return xt,yt
def minmax(im,min_=None,max_=None):
    if min_ is None: min_=np.min(im)
    if max_ is None: max_=np.max(im)
    return (np.array(im,dtype=np.float32)-min_)/(max_-min_)
def translate(im,trans):
    """translates ND images"""
    shape_ = im.shape
    trans_=np.array(np.round(trans),dtype=int)
    im_slices = tuple([slice(max(t_,0),min(sh_,sh_+t_)) for t_,sh_ in zip(trans_,shape_)])
    im_slices0 = tuple([slice(max(-t_,0),min(sh_,sh_-t_)) for t_,sh_ in zip(trans_,shape_)])
    im_base_0 = np.zeros_like(im)+np.median(im)
    im_base_0[im_slices0]=im[im_slices]
    return im_base_0
from sklearn import linear_model
def calc_color_matrix(x,y,order=2,ransac=False):
    """This gives a quadratic color transformation (in matrix form)
    x is Nx3 vector of positions in the reference channel (typically cy5)
    y is the Nx3 vector of positions in another channel (i.e. cy7)
    return m_ a 3x7 matrix which when multipled with x,x**2,1 returns y-x
    This m_ is indended to be used with apply_colorcor
    """ 
    x_ = np.array(y)# ref zxy
    y_ = np.array(x)-x_# dif zxy
    # get a list of exponents
    exps = []
    for p in range(order+1):
        for i in range(p+1):
            for j in range(p+1):
                if i+j<=p:
                    exps.append([i,j,p-i-j])
    # construct A matrix
    A = np.zeros([len(x_),len(exps)])
    for iA,(ix,iy,iz) in enumerate(exps):
        s = (x_[:,0]**ix*x_[:,1]**iy*x_[:,2]**iz)
        A[:,iA]=s
    if ransac:
        ransac = linear_model.RANSACRegressor()
        ransac.base_estimator = linear_model.LinearRegression(fit_intercept=False)
        m_ = []
        for iy in range(len(x_[0])):
            ransac.fit(A, y_[:,iy])
            m_.append(ransac.estimator_.coef_)
        m_=np.array(m_)
    else:
        m_ = [np.linalg.lstsq(A, y_[:,iy])[0] for iy in range(len(x_[0]))]
    m_=np.array(m_)
    return m_
def apply_colorcor(x,m=None):
    """This applies chromatic abberation correction to order 2
    x is a Nx3 vector of positions (typically 750(-->647))
    m is a matrix computed by function calc_color_matrix
    y is the corrected vector in another channel"""
    if m is None:
        return x
    exps = []
    order_max=10
    for p in range(order_max+1):
        for i in range(p+1):
            for j in range(p+1):
                if i+j<=p:
                    exps.append([i,j,p-i-j])
    #find the order
    mx,my = m.shape
    order = int((my-1)/mx)
    assert(my<len(exps))
    x_ = np.array(x)
    # construct A matrix
    exps = exps[:my]
    A = np.zeros([len(x_),len(exps)])
    for iA,(ix,iy,iz) in enumerate(exps):
        s = (x_[:,0]**ix*x_[:,1]**iy*x_[:,2]**iz)
        A[:,iA]=s
    diff = [np.dot(A,m_) for m_ in m]
    return x_+np.array(diff).T