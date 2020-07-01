#Bogdan Bintu
#Copyright Presidents and Fellows of Harvard College, 2017.


from FittingTools import *


class imshow_mark_3d:
    def __init__(self,ims,ims_c=[],fig=None,image_names=None,rescz=1.,min_max_default = [None,None],save_file=None):
        """
        This is a class which controls an interactive maplotlib figure.
        Intended for navigating and interacting with 'spot'-like data that is spread across multiple images <ims>.
        Two max projection images are shown: xy and xz. By zooming into a region, the zoomed in region gets re-maxprojected.
        
        Right click to add a 'seed' point.
        Shift+Right click  to remove the closest 'seed' point
        
        Press 'a' or 'd' to navigate between images.
        
        Press 't' to automatically find seed points.
        Press 'y' to fit the current seed points, including the manually selected/curated ones.
        
        Press 'x' to automatically adjust contrast to min-max in the zoomed in region.
        
        Optional features:
        Can provide a list of color 3d images (or functions that produce color 3d images) as markers (i.e. DAPI or geminin)
        
        """
        #internalize
        self.coloron=False
        self.ims=ims #gray scale images
        self.ims_c=ims_c #color images
        self.rescz = rescz #the resolution in z sometimes is different than in xy
        if image_names is None:
            self.image_names = ['Image '+str(i+1) for i in range(len(ims))]
        else:
            self.image_names = image_names
        self.save_file = save_file
        
        #define extra vars
        self.dic_min_max = {} #kees record of the min-max for adjusting contrast for the grayscale images
        self.dic_min_max_c = {} #kees record of the min-max for adjusting contrast for the color images
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
        self.imshow_xy = self.ax1.imshow(self.im_xy,interpolation='nearest',cmap='gray')
        self.imshow_z = self.ax2.imshow(self.im_z,interpolation='nearest',cmap='gray')
        
        self.min_,self.max_ = min_max_default
        if self.min_ is None: self.min_ = np.min(self.im_)
        if self.max_ is None: self.max_ = np.max(self.im_)
        self.imshow_xy.set_clim(self.min_,self.max_)
        self.imshow_z.set_clim(self.min_,self.max_)
        #connect zoom/pan
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
            self.set_image()
        if event.key== 'w':
            self.coloron=(self.coloron==False)
            self.set_image()
        if event.key== 'a':
            self.index_im = (self.index_im-1)%len(self.ims)
            self.set_image()
        if event.key== 'x':
            self.auto_scale()
        if event.key== 't':
            self.get_seed_points()
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
    def fit_seed_points(self):
        #get default paramaters
        if hasattr(self,'paramaters'):
            paramaters=self.paramaters
        else:
            paramaters=self
        width_z = getattr(paramaters,'width_z',1.)
        width_xy = getattr(paramaters,'width_xy',1.)
        radius_fit = getattr(paramaters,'radius_fit',5)
        radius_group = getattr(paramaters,'radius_group',7)
        
        im = self.im_sm
        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
        y_,x_,z_ = map(np.array,[self.draw_x,self.draw_y,self.draw_z])
        keep_class = np.array(self.class_ids)==self.index_im
        keep_in_window = (x_>x_min)&(x_<x_max)&(y_>y_min)&(y_<y_max)&(z_>z_min)&(z_<z_max)
        keep = keep_class&keep_in_window
        xyzguess = np.array([z_[keep]-z_min,x_[keep]-x_min,y_[keep]-y_min],dtype=int)
        
        multifit = True
        #self.fast = True
        if hasattr(self,"fast"):
            if self.fast:
                self.pfits = fast_local_fit(im,xyzguess.T,radius=radius_fit,width_zxy=[width_z,width_xy,width_xy])
                multifit = False
        if multifit:
            self.pfits = fit_seed_points_base(im,xyzguess,width_zxy=[width_z,width_xy,width_xy],radius_fit=radius_fit,n_max_iter = 10,max_dist_th=0.25)
        #pisici
        if len(self.pfits>0):
            self.pfits[:,1:4]+=[[z_min,x_min,y_min]]
            #update graph and points
            keep = np.array(self.class_ids)!=self.index_im
            self.class_ids,self.draw_z,self.draw_x,self.draw_y = map(lambda x: list(np.array(x)[keep]),[self.class_ids,self.draw_z,self.draw_x,self.draw_y])
            if not hasattr(self,'pfits_save'):
                self.pfits_save={}
            self.pfits_save[self.index_im]=self.pfits
            centers_0,centers_1,centers_2 = self.pfits[:,1:4].T
            self.draw_z.extend(centers_0)
            self.draw_x.extend(centers_2)
            self.draw_y.extend(centers_1)
            self.class_ids.extend([self.index_im]*len(centers_0))
        else:
            if not hasattr(self,'pfits_save'):
                self.pfits_save={}
            if self.pfits_save.has_key(self.index_im):
                del self.pfits_save[self.index_im]
        self.update_point_plot()
    def get_seed_points(self):
        #get default paramaters
        if hasattr(self,'paramaters'):
            paramaters=self.paramaters
        else:
            paramaters=self
        gfilt_size = getattr(paramaters,'gfilt_size',0.)
        filt_size = getattr(paramaters,'filt_size',3)
        th_seed = getattr(paramaters,'th_seed',150.)
        hot_pix_th = getattr(paramaters,'hot_pix_th',0)
        
        im = self.im_sm
        
        centers = get_seed_points_base(im,gfilt_size=gfilt_size,filt_size=filt_size,th_seed=th_seed,hot_pix_th=hot_pix_th)
        
        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
        keep = np.array(self.class_ids)!=self.index_im
        self.class_ids,self.draw_z,self.draw_x,self.draw_y = map(lambda x: list(np.array(x)[keep]),[self.class_ids,self.draw_z,self.draw_x,self.draw_y])
        self.draw_z.extend(centers[0]+z_min)
        self.draw_x.extend(centers[2]+y_min)
        self.draw_y.extend(centers[1]+x_min)
        self.class_ids.extend([self.index_im]*len(centers[0]))
        self.update_point_plot()
            
