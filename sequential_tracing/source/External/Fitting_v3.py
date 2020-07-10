import numpy as np
from . import _sigma_zxy

def __init__():
    pass

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
    
from scipy.spatial.distance import cdist
def closest(x,y,z,center,all_centers):
    xyz = np.array([x,y,z]).T
    dists = cdist(xyz,all_centers)
    closest_point_id = np.argmin(dists,axis=-1)
    center_id = np.argmin(cdist([center],all_centers)[0,:])
    keep = closest_point_id==center_id
    xyz_keep = xyz[keep,:]
    return xyz_keep.T

from scipy.optimize import leastsq
class GaussianFit():
    def __init__(self,im,X,center=None,n_aprox=10,min_w=0.5,max_w=4.,delta_center=3., init_w=_sigma_zxy, weight_sigma=0):
        # store parameters
        self.min_w = min_w*min_w
        self.max_w = max_w*max_w
        self.delta_center = delta_center
        self.im = np.array(im,dtype=np.float32)
        self.x,self.y,self.z = np.array(X,dtype=np.float32)
        self.weight_sigma = weight_sigma

        #get estimates
        argsort_im = np.argsort(im)
        if center is None:
            center = np.median(X[:,argsort_im][:,-n_aprox:],-1)
        self.center_est = center
        sorted_im = im[argsort_im]
        eps =  np.exp(-10.)
        bk_guess = np.log(np.max([np.mean(sorted_im[:n_aprox]),eps]))
        h_guess = np.log(np.max([np.mean(sorted_im[-n_aprox:]),eps]))
        # take care of initialzed w
        init_w = np.array(init_w[:3]).copy()
        self.init_w = init_w
        for _i, _iw in enumerate(init_w):
            if _iw**2 > max_w or _iw**2 < min_w: # if extreme values applied, adjust
                init_w[_i] = 1.5**2
            self.init_w[_i] = np.log((self.max_w - init_w[_i]**2)/(init_w[_i]**2-self.min_w))
        #wsq = 1.5**2
        #wg = np.log((self.max_w - wsq)/(wsq-self.min_w))
        self.p_ = np.array([bk_guess,h_guess,0,0,0,init_w[0],init_w[1],init_w[2],0,0],dtype=np.float32)
        self.to_natural_paramaters()
        self.success = False
    def to_center(self,c0_,c1_,c2_):
        """constrains via sigmoidal function close to local center"""
        delta = self.delta_center
        c0 = 2.*delta*np.exp(-c0_)/(1.+np.exp(-c0_))-delta+self.center_est[0]
        c1 = 2.*delta*np.exp(-c1_)/(1.+np.exp(-c1_))-delta+self.center_est[1]
        c2 = 2.*delta*np.exp(-c1_)/(1.+np.exp(-c2_))-delta+self.center_est[2]
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
        _raw_eps = self.calc_f(parms)-self.im
        if hasattr(self, 'weight_sigma') and getattr(self, 'weight_sigma') > 0:
            bk,h,xp,yp,zp,w1,w2,w3,pp,tp = parms
            #ws1,ws2,ws3 = self.to_ws(w1),self.to_ws(w2),self.to_ws(w3) # convert to natural parameters
            _curr_w = np.array([w1, w2, w3])
            _eps = _raw_eps + self.weight_sigma * np.linalg.norm(self.init_w - _curr_w)            
        else:
            _eps = _raw_eps
        return _eps
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
        f6 = (f2*(-pc2*tc2*xt2 - 2*pc2*t*tc*xtyt - pc2*t2*yt2 + 2*p*pc*tc*xtzt + 2*p*pc*t*ytzt - p2*zt2))*self.norm_w(w1,minw,maxw) + \
            int(self.init_w[0]>w1) * self.weight_sigma - int(self.init_w[0]<w1) * self.weight_sigma
        f7 = (f2*(-t2*xt2 + 2*t*tc*xtyt - tc2*yt2))*self.norm_w(w2,minw,maxw) + \
            int(self.init_w[1]>w2) * self.weight_sigma - int(self.init_w[1]<w2) * self.weight_sigma
        f8 = (f2*(-p2*tc2*xt2 - 2*p2*t*tc*xtyt - p2*t2*yt2 - 2*p*pc*tc*xtzt - 2*p*pc*t*ytzt - pc2*zt2))*self.norm_w(w3,minw,maxw) +\
            int(self.init_w[2]>w3) * self.weight_sigma - int(self.init_w[2]<w3) * self.weight_sigma
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
            self.p_ = parmsf
            self.to_natural_paramaters()
            self.center = self.p[1:4]
            self.success = True
    def get_im(self):
        self.calc_f(self.p_)
        return self.f0
        
from scipy.ndimage.filters import maximum_filter,minimum_filter,median_filter,gaussian_filter

def get_seed_points_base(im,gfilt_size_min=1,gfilt_size_max=3,filt_size=3,th_seed=0.,max_num=None,
                         use_snr=False,hot_pix_th=0,return_h=False):
    """Get the seed points in an image.
    #1 perform a gaussian filter
    #2 find local maxima within a radius 3 above th_seed form the minimum
    #3 remove hot pixels (high multiplicity of xy positions with different z)
    """
    im_plt=np.array(im)
    #if gfilt_size>0:
    #    im_plt = gaussian_filter(im,gfilt_size)
    max_filt = maximum_filter(im_plt,filt_size)
    min_filt = minimum_filter(im_plt,filt_size)
    g_filt = gaussian_filter(im_plt,gfilt_size_max)
    g_filt_sm = gaussian_filter(im_plt,gfilt_size_min)
    im_plt2 = (max_filt==im_plt)&(min_filt!=im_plt)
    z,x,y = np.where(im_plt2)
    h = g_filt_sm[z,x,y]-g_filt[z,x,y]
    snr = 1.*g_filt_sm[z,x,y]/g_filt[z,x,y]
    
    if use_snr:
        keep = snr>th_seed
    else:
        keep = h>th_seed
    x,y,z = x[keep],y[keep],z[keep]
    h,snr = h[keep],snr[keep]
    #get rid of hot pixels
    if hot_pix_th>0 and len(x)>0:
        xy = y*np.max(x)+x
        xy_,cts_ = np.unique(xy,return_counts=True)
        bad_xy = xy_[cts_>hot_pix_th]
        keep = np.array([xy_ not in bad_xy for xy_ in xy],dtype=bool)
        x,y,z = x[keep],y[keep],z[keep]
        snr=snr[keep]
        h = h[keep]
    centers = np.array([z,x,y])
    #sort by absolute brightness or signal to noise ratio (snr)
    if not use_snr:
        ind = np.argsort(h)[::-1]
    else:
        ind = np.argsort(snr)[::-1]
    centers = np.array([z[ind],x[ind],y[ind]])
    if return_h:
        centers = np.array([z,x,y,h])
    if max_num is not None:
        centers = centers[:,:max_num]
    return centers

def in_dim(x,y,z,xmax,ymax,zmax):
    keep = ((x>=0)&(x<xmax)&(y>=0)&(y<ymax)&(z>=0)&(z<zmax))>0
    return x[keep],y[keep],z[keep]

class iter_fit_seed_points():
    def __init__(self,im,centers,radius_fit=5,min_delta_center=1.,max_delta_center=2.5,n_max_iter = 10,max_dist_th=0.1, init_w=_sigma_zxy, weight_sigma=0):
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
        self.sz,self.sx,self.sy = im.shape
        self.init_w = init_w
        self.weight_sigma = weight_sigma
         
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
            for zc,xc,yc in self.centers:
                z_keep,x_keep,y_keep = int(zc)+self.zb,int(xc)+self.xb,int(yc)+self.yb
                z_keep,x_keep,y_keep = in_dim(z_keep,x_keep,y_keep,self.sz,self.sx,self.sy)
                X_full = np.array([z_keep,x_keep,y_keep],dtype=int)
                center = [zc,xc,yc]
                z_keep,x_keep,y_keep = closest(z_keep,x_keep,y_keep,center,self.centers)
                X = np.array([z_keep,x_keep,y_keep])
                im_ = self.im[z_keep,x_keep,y_keep]
                obj = GaussianFit(im_,X,center=center,delta_center=self.min_delta_center, 
                                    init_w=self.init_w, weight_sigma=self.weight_sigma)
                n_p = len(obj.p)
                obj.fit()
                
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
            # set default self.add
            self.im_add = np.array(self.im_subtr)
        else:
            raise ValueError(f"{len(self.centers)} points have been seeded, exit.")
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
                        obj = GaussianFit(im_,X,center=[zc,xc,yc],delta_center=delta_center, 
                                            init_w=self.init_w, weight_sigma=self.weight_sigma)
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
                #print(np.sum(self.converged))
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

def KL_dif(p1,p2):
    """Given two gaussian paramters p1 and p2 this computes their KL divergence."""
    [height1,center_01,center_11,center_21,background1,widths_01,widths_11,widths_21,t_1,p_1]=set1
    [height2,center_02,center_12,center_22,background2,widths_02,widths_12,widths_22,t_2,p_2]=set2
    sigma1,sigma_inv1 = to_sigmas(*[t_1,p_1,widths_01,widths_11,widths_21])
    sigma2,sigma_inv2 = to_sigmas(*[t_2,p_2,widths_02,widths_12,widths_22])
    k = len(sigma1)
    inds = np.arange(k)
    return 0.5*(np.sum(np.dot(sigma_inv2,sigma1)[inds,inds])-k+np.log(np.linalg.det(sigma2))-np.log(np.linalg.det(sigma1)))
