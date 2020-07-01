#Bogdan Bintu
#Copyright Presidents and Fellows of Harvard College, 2017.


from FittingTools import *

import scipy.ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.segmentation import random_walker

class imshow_DAPIsegmentation:

    def __init__(self,ims,ims_c=[],fig=None,image_names=None,rescz=1.,min_max_default = [None,None],save_file=None,dapi_save_files=None):
        #internalize
        self.dapi_save_files=dapi_save_files
        self.coloron=False
        self.ims=ims
        self.ims_c=ims_c
        self.rescz = rescz
        if image_names is None:
            self.image_names = ['Image '+str(i+1) for i in range(len(ims))]
        else:
            self.image_names = image_names
        self.save_file = save_file
        #define extra vars
        self.dic_min_max = {}
        self.dic_min_max_c = {}
        self.class_ids = []
        self.draw_x,self.draw_y,self.draw_z=[],[],[]
        self.coords = zip(self.draw_x,self.draw_y,self.draw_z)
        self.delete_mode = False
        #load vars
        self.load_coords()
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
        self.imshow_xy = self.ax1.imshow(self.im_xy,interpolation='nearest',cmap=cm.gray)
        self.imshow_z = self.ax2.imshow(self.im_z,interpolation='nearest',cmap=cm.gray)
        
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
            
    def master_reset(self):
        #self.dic_min_max = {}
        self.class_ids = []
        self.draw_x,self.draw_y,self.draw_z=[],[],[]
        self.coords = zip(self.draw_x,self.draw_y,self.draw_z)
        #load vars
        self.load_coords()
        self.set_image()
    def get_ims_c(self):
        if self.ims_c is None:
            return None
        try:
            return self.ims_c[self.index_im%len(self.ims_c)]
        except:
            return self.ims_c(self.index_im)
    def onclick(self,event):
        if event.button==3:
            #print "click"
            if event.inaxes is self.ax1:
                if self.delete_mode:
                    z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
                    x_,y_,z_ = map(np.array,[self.draw_x,self.draw_y,self.draw_z])
                    #print x_min,x_max,y_min,y_max,z_min,z_max
                    #print x_,y_,z_
                    keep_in_window = (x_>y_min)&(x_<y_max)&(y_>x_min)&(y_<x_max)&(z_>z_min)&(z_<z_max)
                    keep_class = (np.array(self.class_ids)==self.index_im)&(np.isnan(self.draw_x)==False)
                    keep = keep_in_window&keep_class
                    if np.sum(keep)>0:
                        keep_ind = np.arange(len(keep))[keep]
                        coords_xy_class = zip(np.array(self.draw_x)[keep],
                                              np.array(self.draw_y)[keep])
                        difs = np.array(coords_xy_class)-np.array([[event.xdata,event.ydata]])
                        ind_= np.argmin(np.sum(np.abs(difs),axis=-1))
                        self.draw_x.pop(keep_ind[ind_])
                        self.draw_y.pop(keep_ind[ind_])
                        self.draw_z.pop(keep_ind[ind_])
                        self.class_ids.pop(keep_ind[ind_])
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
                    x_,y_,z_ = map(np.array,[self.draw_x,self.draw_y,self.draw_z])
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
            self.remove_contours()
            self.set_image()
        if event.key== 'w':
            self.coloron=(self.coloron==False)
            self.set_image()
        if event.key== 'a':
            self.index_im = (self.index_im-1)%len(self.ims)
            self.remove_contours()
            self.set_image()
        if event.key=='s':
            self.save_ims()
        if event.key== 'x':
            self.auto_scale()
        if event.key== 't':
            self.nuc_watershed_get_seed(self.ims[self.index_im])
        if event.key== 'y':
            self.nuc_watershed()
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
            self.draw_x,self.draw_y,self.draw_z = zip(*self.coords)
            if flip: self.draw_x,self.draw_y,self.draw_z =  map(list,[self.draw_y,self.draw_x,self.draw_z])
            else: self.draw_x,self.draw_y,self.draw_z =  map(list,[self.draw_x,self.draw_y,self.draw_z])
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
                            if self.dec_text.has_key(key_dec):
                                text_=self.dec_text[key_dec]['text']
                                color_='b'
                        self.texts.append(self.ax1.text(xyz[0],xyz[1],text_,color=color_))
                        self.texts.append(self.ax2.text(xyz[0],xyz[2],text_,color=color_))
    def update_point_plot(self):
        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
        
        self.coords = zip(self.draw_x,self.draw_y,self.draw_z)
        x_,y_,z_ = map(np.array,[self.draw_x,self.draw_y,self.draw_z])
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
    def load_coords(self):
        save_file = self.save_file
        if save_file is not None and os.path.exists(save_file):
            fid = open(save_file,'r')
            save_dic = pickle.load(fid)
            self.coords,self.class_ids = save_dic['coords'],save_dic['class_ids']
            if save_dic.has_key('pfits'):
                self.pfits_save = save_dic['pfits']
            if save_dic.has_key('dec_text'):
                self.dec_text=save_dic['dec_text']
            fid.close()
            self.populate_draw_xyz()#coords to plot list
    def save_coords(self):
        save_file = self.save_file
        if save_file is not None:
            if not os.path.exists(os.path.dirname(save_file)):
                os.makedirs(os.path.dirname(save_file))
            fid = open(save_file,'w')
            save_dic = {'coords':self.coords,'class_ids':self.class_ids}
            save_dic['names']=self.image_names
            if hasattr(self,'dec_text'):
                save_dic['dec_text']=self.dec_text
            if hasattr(self,'pfits_save'):
                save_dic['pfits']=self.pfits_save
            pickle.dump(save_dic,fid)
            fid.close()
    def auto_scale(self):
        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
        if self.coloron and self.ims_c is not None:
            self.im_ = self.get_ims_c()
            im_chop = self.im_[z_min:z_max,x_min:x_max,y_min:y_max,...]
            im_chop = np.array(im_chop,dtype=float)
            min_ = np.array([[[np.min(im_chop[...,i]) for i in range(im_chop.shape[-1])]]])
            max_ = np.array([[[np.max(im_chop[...,i]) for i in range(im_chop.shape[-1])]]])
            self.im_ = (self.im_-min_)/(max_-min_)
            self.im_[self.im_>1]=1
            self.im_[self.im_<0]=0
            self.finish_set_image()
            self.imshow_xy.set_clim([0,1])
            self.imshow_z.set_clim([0,1])
            self.dic_min_max_c[self.index_im] = [min_,max_]
        else:
            im_chop = np.array(self.im_[z_min:z_max,x_min:x_max,y_min:y_max,...])
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
            
            if self.dic_min_max.has_key(index_im):
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
    
    def finish_set_image(self):
        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
        
        if self.dic_min_max_c.has_key(self.index_im) and self.coloron and self.ims_c is not None:
            min_,max_ = self.dic_min_max_c[self.index_im]
            self.im_ = self.get_ims_c()
            self.im_ = (self.im_-min_)/(max_-min_)
            self.im_[self.im_>1]=1
            self.im_[self.im_<0]=0
            self.imshow_xy.set_clim([0,1])
            self.imshow_z.set_clim([0,1])
        
        self.im_sm = self.im_[z_min:z_max,x_min:x_max,y_min:y_max]
        self.im_xy = np.max(self.im_[z_min:z_max,:,...],axis=0)
        self.imshow_xy.set_data(self.im_xy)
        
        self.im_z = np.max(self.im_[:,x_min:x_max,...],axis=1)
        self.im_z = self.im_z[self.get_z_ind(),:]
        self.imshow_z.set_data(self.im_z)
        self.imshow_xy.autoscale()
        self.imshow_z.autoscale()
        if self.dic_min_max.has_key(self.index_im) and not self.coloron:
            min_,max_ = self.dic_min_max[self.index_im]
            self.imshow_xy.set_clim(min_,max_)
            self.imshow_z.set_clim(min_,max_)

        self.update_point_plot()
        self.f.suptitle(self.image_names[self.index_im])
        self.f.canvas.draw()
    def set_image(self):
        self.im_ = np.array(self.ims[self.index_im])
        if self.coloron and self.ims_c is not None:
            self.im_ = self.get_ims_c()
        self.finish_set_image()
        
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
    def nuc_watershed_get_seed(self,dapi):
        """
        Input:
        dapi - a 3d image of the DAPI channel
        watershed - boolean determining whether to attempt a separation of "touching" nuclei by watershed
        plot - boolean determining whether to plot DAPI image and cell contours
        min_nuc_area - the minimal area in pixel^2 for a region to be called a nucleus. Default is 90 um^2
        big_filt = size in pixels of large gaussian filter (for local signal levels). Default 7.5 um 
        sml_filt = size in pixels of small gaussian filter (for smoothing). Default 750 nm
        threh = how high the small blur should be above local background
        thperc90 = post segmentation how high the top 10% brightest dapi pixels should be above the background
        delta = disntace from edge to get rid of cell
        
        Returns:
        labeled_nuc - a 2d array with 0 where there is no dapi signal and a number (1,2...) where a nucleus is observed
        dapi_int - the total intensity of DAPI in each nucles, in the same order of te labels
        """
        #get default paramaters but allow for remodification
        self.dic_DAPI = getattr(self,'dic_DAPI',{'min_nuc_area':4000, 'big_filt':100, 'sml_filt':5,'threh':1.2,'thperc90':0,'delta':3})
        min_nuc_area = self.dic_DAPI['min_nuc_area']
        big_filt = self.dic_DAPI['big_filt']
        sml_filt = self.dic_DAPI['sml_filt']
        threh = self.dic_DAPI['threh']
        thperc90 = self.dic_DAPI['thperc90']
        delta = self.dic_DAPI['delta']
        #process dapi
        dapi_int = []
        dapim = np.max(dapi,0) #median x-y projection
        self.dapim=dapim
        dapi_blur = ndi.gaussian_filter(dapim, big_filt)
        dapi_sblur = ndi.gaussian_filter(dapim, sml_filt)
        dapi_val = 1.*dapi_sblur/dapi_blur
        th = dapi_val>threh #small blur is 75% above local background
        th = ndi.binary_fill_holes(th)
        labeled_dapi, num_feat = ndi.label(th)
        #remove things too small or too dim to be nuclei
        for i in range(num_feat):
            nuc_im = labeled_dapi==i+1
            area = np.sum(nuc_im)
            tot_int = np.sum(dapim[nuc_im])
            perc90 = np.percentile(dapi_val[nuc_im],90)
            if (area<min_nuc_area) or (perc90<thperc90): #too small or too dim- liekly not a nucleus
                labeled_dapi[labeled_dapi==i+1] = 0
            else: #keep total intensity
                dapi_int.append(tot_int)
        dapi_in_nuc = labeled_dapi>0
        dapi_in_nuc=ndi.morphology.binary_fill_holes(dapi_in_nuc)
        self.dapi_in_nuc=dapi_in_nuc
        
        #autowatersheding
        dist = ndi.distance_transform_edt(dapi_in_nuc) #distance transformation for watershed
        local_max = peak_local_max(dist, indices = False, min_distance=10)
        #plt.figure()
        self.pisici = np.sum(local_max)
        x,y = np.where(local_max)
        X=np.array([x,y]).T
        distM = cdist(X,X)
        distM[range(len(X)),range(len(X))]=np.inf
        xp,yp = np.where(distM<35)
        ik=np.setdiff1d(np.arange(len(x)),xp[xp>yp])
        x,y = x[ik],y[ik]
        
        #Update self
        #contour graphs
        self.remove_contours()
        self.tent_cells = self.ax1.contour(self.dapi_in_nuc,[0.5],colors=['r'])
        self.f.canvas.draw()
        
        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
        keep = np.array(self.class_ids)!=self.index_im
        self.class_ids,self.draw_z,self.draw_x,self.draw_y = map(lambda x: list(np.array(x)[keep]),[self.class_ids,self.draw_z,self.draw_x,self.draw_y])
        self.draw_z.extend([(z_max+z_min)/2]*len(x))#+z_min)
        self.draw_x.extend(y)#+y_min)
        self.draw_y.extend(x)#+x_min)
        self.class_ids.extend([self.index_im]*len(x))
        self.update_point_plot()
        #self.nuc_watershed()
        return x,y
    def remove_contours(self):
        if hasattr(self,'tent_cells'): 
            self.tent_cells.set_alpha(0)#.remove()
        if hasattr(self,'good_cells'): 
            for good_cell in self.good_cells:
                good_cell.set_alpha(0)#.remove()
        self.f.canvas.draw()
    def nuc_watershed(self):
        
        x,y=np.array(self.draw_y)[np.array(self.class_ids)==self.index_im],np.array(self.draw_x)[np.array(self.class_ids)==self.index_im]
        dapi_in_nuc = self.dapi_in_nuc
        ###gets dapi signal in identified nuclei and attempts to separate touching ones by watershed
        dist = ndi.distance_transform_edt(dapi_in_nuc)
        local_max = 0*dapi_in_nuc
        local_max[np.array(x,dtype=int),np.array(y,dtype=int)]=1
        
        #local_max = ndi.morphology.binary_dilation(local_max,iterations=5)
        markers = ndi.label(local_max)[0] #local maxima locations
        #labeled_dapi = watershed(-dist,markers, mask = dapi_in_nuc)
        
        markers[~dapi_in_nuc] = -1
        self.markers=markers
        labeled_dapi = random_walker(dapi_in_nuc, markers)
        labeled_dapi[labeled_dapi==-1]=0
        self.labeled_dapi=labeled_dapi
        
        
        num_feat = len(np.unique(labeled_dapi))-1
        
        #remove border cells
        delta = self.dic_DAPI['delta']
        im = np.array(labeled_dapi)
        xs,ys = np.where(im>0)
        bad = (xs<delta)|(xs>im.shape[0]-delta)|(ys<delta)|(ys>im.shape[1]-delta)
        bad_inds = np.unique(im[xs[bad],ys[bad]])
        for ind in bad_inds:
            im[im==ind]=0
         
        #labeled_nuc, num_nuc = ndi.label(im)
        inds = np.unique(im)[1:]
        im_ = np.array(im)
        for iind,ind in enumerate(inds):
            im[im_==ind]=iind+1
      
        labeled_nuc = np.array(im,dtype=np.uint8)
        self.labeled_nuc=labeled_nuc
        #save intensities
        dapi_int =[]
        #areas = []
        #for i in range(len(inds)):
        #    nuc_im = labeled_nuc==i+1
        #    #area = np.sum(nuc_im)
        #    tot_int = np.sum(dapim[nuc_im])
        #    dapi_int.append(tot_int)
        
        self.remove_contours()
        self.good_cells=[]
        for i in np.unique(labeled_nuc)[1:]:
            self.good_cells.append(self.ax1.contour(labeled_nuc==i,[0.5],colors=['g']))
        self.f.canvas.draw()
        dic_save = {'labeled_nuc':labeled_nuc,'dapi_int':dapi_int}
        if self.dapi_save_files is not None:#False:#save_file is not None:
            save_file = self.dapi_save_files[self.index_im]
            parent_folder= os.path.dirname(save_file)
            if not os.path.exists(parent_folder):
                os.makedirs(parent_folder)
            fid = open(save_file,'wb')
            pickle.dump(dic_save,fid)
            fid.close()
            
            fig = plt.figure(figsize=(16,8))
            plt.subplot(121)
            plt.imshow(self.labeled_nuc,cmap='jet')
            plt.subplot(122)
            #plt.imshow(dapi_val,cmap='jet')
            plt.imshow(np.max(self.ims[self.index_im],0),cmap='jet')
            plt.contour(self.labeled_nuc,[0.5],)
            plt.show()
            fig.savefig(parent_folder+os.sep+os.path.basename(save_file).split('.')[0]+'.png')
        
        return labeled_dapi, num_feat
        
        
        
        
        
        
        

