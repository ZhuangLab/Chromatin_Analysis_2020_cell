import sys, os, time, glob, re
import numpy as np
import pickle as pickle
import matplotlib.pylab as plt
import scipy
import multiprocessing as mp
# local
from .. import visual_tools
from ..External import DomainTools



class mark_boundaries:
    """
    #Bogdan Bintu
    #Copyright Presidents and Fellows of Harvard College, 2017.
    """
    def __init__(self,ims,fig=None,image_names=None,plot_limits=[0,1000],
                 save_file=None, scale_percentile=95):
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

        self.ims=ims #gray scale images
        if image_names is None:
            self.image_names = ['Image '+str(i+1) for i in range(len(ims))]
        else:
            self.image_names = image_names
        self.save_file = save_file
        self.scale_percentile = scale_percentile

        #define extra vars
        self.dic_min_max = {} #kees record of the min-max for adjusting contrast for the images
        self.class_ids = []
        self.draw_x=[]
        self.delete_mode = False
        #load vars
        self.load_coords()
        #construct images
        self.index_im = 0
        self.im_ = self.ims[self.index_im]
        #setup plots
        if fig is None:
            self.f,self.ax1 = plt.subplots()
        else:
            self.f,self.ax1=fig
        
        
        self.lxy,=self.ax1.plot([],[], 'g-',linewidth=2.5)
        self.imshow_xy = self.ax1.imshow(self.im_,interpolation='nearest',cmap='seismic_r')
        
        self.min_,self.max_ = min(plot_limits), max(plot_limits)
        if self.min_ is None: self.min_ = np.min(self.im_)
        if self.max_ is None: self.max_ = np.max(self.im_)
        self.imshow_xy.set_clim(self.min_,self.max_)
        #connect zoom/pan
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
        self.draw_x=[]
        #load vars
        self.load_coords()
        self.set_image()
    def onclick(self,event):
        if event.button==3:
            if event.inaxes is self.ax1:
                if self.delete_mode:
                    self.mouse_pos = [event.xdata,event.ydata]
                    x_min,x_max,y_min,y_max = self.get_limits()
                    x_= np.array(self.draw_x)
                    keep_in_window = (x_>y_min)&(x_<y_max)&(x_>x_min)&(x_<x_max)
                    keep_class = (np.array(self.class_ids)==self.index_im)&(np.isnan(self.draw_x)==False)
                    keep = keep_in_window&keep_class
                    if np.sum(keep)>0:
                        keep_ind = np.arange(len(keep))[keep]
                        coords_xy_class = list(zip(np.array(self.draw_x)[keep],
                                              np.array(self.draw_x)[keep]))
                        difs = np.array(coords_xy_class)-np.array([[event.xdata,event.ydata]])
                        ind_= np.argmin(np.sum(np.abs(difs),axis=-1))
                        self.draw_x.pop(keep_ind[ind_])
                        self.class_ids.pop(keep_ind[ind_])
                else:
                    if event.xdata is not None and event.ydata is not None:
                        self.draw_x.append((event.xdata+event.ydata)/2.)
                        self.class_ids.append(self.index_im)
            self.update_point_plot()
    def press(self,event):
        if event.key== 'd':
            self.index_im = (self.index_im+1)%len(self.ims)
            self.set_image()
        if event.key== 'a':
            self.index_im = (self.index_im-1)%len(self.ims)
            self.set_image()
        if event.key== 'e':
            self.index_im = (self.index_im+20)%len(self.ims)
            self.set_image()
        if event.key== 'w':
            self.index_im = (self.index_im-20)%len(self.ims)
            self.set_image()
        if event.key== 'x':
            self.auto_scale()
        if event.key== 'z':
            self.scale_up()
        if event.key== 'c':
            self.scale_down()    
        if event.key== 'f':
            self.fit()
        if event.key == 'delete':
            self.draw_x.pop(-1)
            self.class_ids.pop(-1)
            self.update_point_plot()
        if event.key == 'shift':
            self.delete_mode = True
    def release(self, event):
        if event.key == 'shift':
            self.delete_mode = False
    def update_point_plot(self):
        keep_class = np.array(self.class_ids)==self.index_im
        x_ = np.array(self.draw_x)[keep_class]
        
        x_ = np.sort([0]+list(x_)+[len(self.im_)])
        x__,y__ = [],[]
        for i in range(len(x_)):
            x__.append(x_[i])
            y__.append(x_[i])
            if i+1<len(x_):
                x__.append(x_[i])
                y__.append(x_[i+1])
        self.lxy.set_xdata(x__)
        self.lxy.set_ydata(y__)
        self.save_coords()
        self.f.canvas.draw()
    def load_coords(self):
        save_file = self.save_file
        if save_file is not None and os.path.exists(save_file):
            fid = open(save_file,'rb')
            save_dic = pickle.load(fid)
            self.draw_x,self.class_ids = save_dic['coords'],save_dic['class_ids']
            fid.close()
    def save_coords(self):
        save_file = self.save_file
        if save_file is not None:
            #if not os.path.exists(os.path.dirname(save_file)):os.makedirs(os.path.dirname(save_file))
            fid = open(save_file,'wb')
            save_dic = {'coords':self.draw_x,'class_ids':self.class_ids}
            save_dic['names']=self.image_names
            pickle.dump(save_dic,fid)
            fid.close()
    def save_copy(self):
        save_copy_file = self.save_file.replace('.npy', '_picked.npy')
        from shutil import copyfile
        copyfile(save_file, save_copy_file)
            
    def auto_scale(self, percentile=95):
        from scipy.stats import scoreatpercentile
        x_min,x_max,y_min,y_max = self.get_limits()
        im_chop = np.array(self.im_[x_min:x_max,y_min:y_max])

        min_ = scoreatpercentile(im_chop[np.isnan(im_chop)==False], 100-self.scale_percentile)
        max_ = scoreatpercentile(im_chop[np.isnan(im_chop)==False], self.scale_percentile)
        #min_,max_ = np.nanmin(im_chop),np.nanmax(im_chop)
        _range = max_ - min_
        self.imshow_xy.set_clim(min_,max_)
        self.dic_min_max[self.index_im] = [min_, max_]
        self.f.canvas.draw()
    
    def scale_up(self, factor=1.1):
        if self.index_im in self.dic_min_max:
            min_, max_ = self.dic_min_max[self.index_im]
        else:
            min_, max_ = self.min_, self.max_
        new_min_, new_max_ = min_*factor, max_*factor
        self.imshow_xy.set_clim(new_min_,new_max_)
        self.dic_min_max[self.index_im] = [new_min_, new_max_]
        self.f.canvas.draw()

    def scale_down(self, factor=1.1):
        if self.index_im in self.dic_min_max:
            min_, max_ = self.dic_min_max[self.index_im]
        else:
            min_, max_ = self.min_, self.max_
        new_min_, new_max_ = min_/factor, max_/factor
        self.imshow_xy.set_clim(new_min_,new_max_)
        self.dic_min_max[self.index_im] = [new_min_, new_max_]
        self.f.canvas.draw()

    def set_image(self):
        self.im_ = np.array(self.ims[self.index_im])
        self.imshow_xy.set_data(self.im_)
        if self.index_im in self.dic_min_max:
            min_,max_ = self.dic_min_max[self.index_im]
            self.imshow_xy.set_clim(min_,max_)
        self.f.suptitle(self.image_names[self.index_im])
        self.update_point_plot()
        self.f.canvas.draw()
    def get_limits(self):
        y_min,y_max = self.ax1.get_xlim()
        x_min,x_max = self.ax1.get_ylim()[::-1]
        x_min = max(int(x_min),0)
        x_max = min(int(x_max),self.im_.shape[0])
        y_min = max(int(y_min),0)
        y_max = min(int(y_max),self.im_.shape[1])
        
        return x_min,x_max,y_min,y_max
    def fit(self):
        pass


def fit_manual_boundaries(save_file, zxys, num_chroms=None, dom_sz=5, cutoff_max=1.):
    """Function to fit manual picked boundaries to match local minimum"""

    from .distance import _local_distances
    manual_dict = np.load(save_file.replace('.npy', '_picked.npy'))
    if num_chroms is None:
        manual_starts = visual_tools.partition_map(manual_dict['coords'], manual_dict['class_ids'])
    else:
        manual_starts = visual_tools.partition_map(manual_dict['coords'], manual_dict['class_ids'])[:int(num_chroms)]
    
    if len(zxys) < len(manual_starts):
        raise IndexError(f"length of zxy_list:{len(zxys)} should be larger than manual starts:{len(manual_starts)}")
    manual_starts = manual_starts[:len(zxys)]
    # initialize fitted_start_list
    fitted_start_list = []
    for _zxy, _starts in zip(zxys, manual_starts):
        _dists = _local_distances(_zxy, dom_sz=dom_sz)
        _loc_max = int(dom_sz/2) + DomainTools.get_ind_loc_max(_dists, cutoff_max=cutoff_max,
                                                               valley=int((dom_sz+1)/2))
        # append a zero for first domain
        if 0 not in _loc_max:
            _loc_max = np.concatenate([_loc_max, np.zeros(1,dtype=np.int)])
        # find nearest match 
        _ft_starts = [_loc_max[np.argmin(np.abs(_s-_loc_max))] for _s in _starts]
        fitted_start_list.append(np.sort(_ft_starts).astype(np.int))
        
    return fitted_start_list

def find_matched_starts(starts, ref_starts, dom_sz=5, ignore_multi_match=True, ignore_zero=True):
    """Function to find matched domain starts"""
    # convert data_types
    _ref_starts = np.array(ref_starts, dtype=np.int)
    # initialize
    _matched_starts = []
    
    for _start in starts:
        _match = np.where(np.abs(_ref_starts - _start) <= int(dom_sz/2))[0]
        if len(_match) == 0:
            continue
        elif len(_match) == 1:
            _matched_starts.append(_ref_starts[_match[0]])
        else:
            if ignore_multi_match:
                continue
            else:
                _matched_starts.append(_ref_starts[_match[0]])
    
    return np.array(_matched_starts, dtype=np.int)