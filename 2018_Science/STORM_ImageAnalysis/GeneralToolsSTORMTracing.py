import numpy as np
import StormAnalysisAdditions as saa
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os,sys
from scipy import ndimage
import cPickle as pickle
import scipy

def extract_flag(string,flag1,flag2):
    """Get the substring between flag1 and flag2 in string"""
    return string.split(flag1)[-1].split(flag2)[0]
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
def flatten(list_):
    return [item for sublist in list_ for item in sublist]
def minmax(im,max_=None,min_=None):
    im_=np.array(im,dtype=float)
    if max_ is None:
        max_= np.max(im_)
    if min_ is None:
        min_= np.min(im_)
    if max_-min_==0:
        return im_*0
    im_=(im-float(min_))/(float(max_)-float(min_))
    im_[im_>1]=1.
    im_[im_<0]=0.
    return im_
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
    
def listDax(folder,name='*',mask=[]):
    """
    This is intended to get the dax files in the folder.
    listDax(folder,name ='set*Conv*',mask=['Toehold','bubble'])
    returns a list of filenames of the conventionals in the sets which not including in the toeholds or bubble.
    """
    import fnmatch,os
    import numpy as np
    #load all files:
    list_unfilt=[]
    for root, dirs, files in os.walk(folder, topdown=False):
        for nm in files:
            list_unfilt.append(os.path.join(root, nm))
    #keep only a subset
    imList = []
    
    #list_unmasked = fnmatch.filter(list_unfilt, '*'+name+'*.dax')
    
    #####Patch
    #'[' and ']' are unfriendly for fnmatch. Will convert to '(--(' and ')--)'
    list_unfilt_patch = [fl.replace('[','(--(').replace(']',')--)') for fl in list_unfilt]
    list_unmasked_patch = fnmatch.filter(list_unfilt_patch, '*'+name.replace('[','(--(').replace(']',')--)')+'*.dax')
    list_unmasked = [fl.replace('(--(','[').replace(')--)',']') for fl in list_unmasked_patch]
    #######
    
    
    if len(mask)==0:
        return list_unmasked
    # disregard the ones not containing a mask member
    for fl in list_unmasked:
        maskBool = np.sum([fl.count(subStr) > 0 for subStr in  mask])
        if not maskBool:
            imList.append(fl)
    return imList
    
def save_name_dic(dax_fl,save_folder=None):
    if save_folder is None: dax_fl_base = dax_fl
    else: dax_fl_base = save_folder+os.sep+"--".join(dax_fl.split(os.sep)[-3:])
    dic_names={}
    dax_fl_base_ = os.path.dirname(dax_fl_base)+os.sep+'selectedSpots'+os.sep+os.path.basename(dax_fl_base)
    dic_names["coords_file"]=dax_fl_base_.replace('.dax','_tempcoords.pkl')
    dic_names["png_figure"]=dax_fl_base_.replace('.dax','_refspots.png')
    return dic_names
def STORM_STD_pointSelect(dax_fl,rep_fr = 50,save_folder=None,cy3_tag = 'maxfull'):
    """
    Given file name dax_fl this function will pop-up a gui to select the spots. 
    It returns a dictionary with a name linked to an instance of imshow_mark class <fig>. Use fig.coords to access coords selected.
    """
    #This makes a dax object
    dax_obj = saa.datareader.DaxReader(dax_fl)
    #dimensions:
    sx,sy,sz = dax_obj.filmSize()
    #Create memory map
    im_ = dax_obj.loadMap()
    im_=np.swapaxes(im_,1,2)
    #Storm mean of first 10 frames
    im_STORM_ff = np.mean(im_[1:10],axis=0)
    #Get only the first cy3 in the storm zscan (faster)
    
    
    cy3_frames = np.ravel(zip(np.arange(rep_fr,sz,rep_fr)-1,np.arange(rep_fr,sz,rep_fr)))
    cy3_frames_fast = cy3_frames
    #Big storm movies are slow.
    if len(cy3_frames) > 100:
        cy3_frames_fast = get_cy3CenterFrames(dax_fl,rep_fr=rep_fr)
        if cy3_frames_fast is None:
            cy3_frames_fast = [cy3_frames[0],cy3_frames[1],cy3_frames[len(cy3_frames)/2],cy3_frames[len(cy3_frames)/2+1],cy3_frames[-2],cy3_frames[-1]]
        
    #Decide which cy3 frames to get
    if cy3_tag == 'first':
        im_cy3 = np.mean(im_[rep_fr-1:rep_fr+1],axis=0)
    elif cy3_tag == 'meanfull':
        im_cy3 = np.mean(im_[cy3_frames_fast],axis=0)
    elif cy3_tag == 'maxfull':
        im_cy3 = np.max(im_[cy3_frames_fast],axis=0)
    #Plot image for spot selection
    #im_plot = np.dstack(map(minmax,[im_cy3,im_STORM_ff,np.zeros_like(im_cy3)]))
    im_plot = im_cy3
    #Pop-up GUI
    dic_name = save_name_dic(dax_fl,save_folder=save_folder)
    coords_file = dic_name["coords_file"]
    fig = imshow_mark(im_plot,coords_file,title="--".join(dax_fl.split(os.sep)[-3:]))
    return {dax_fl:fig}
class imshow_mark:
    """
    Right click to add points on image, press delete to delete last added point.
    Returns the list of points.
    if <savefile> is not None it will save in <savefile>  the coords after every change.
    """
    def __init__(self,im,savefile=None,title=''):
        self.fig=plt.figure()
        self.ax = self.fig.add_subplot(111)
        #initialize variables
        self.im = np.array(im)
        self.min_im = np.min(im)
        self.max_im = np.max(im)
        self.dif_im = self.max_im-self.min_im
        self.savefile = savefile
        self.draw_x,self.draw_y=[],[]
        self.coords = zip(self.draw_x,self.draw_y)
        self.load_coords()
        #initialize plot
        self.l,=self.ax.plot(self.draw_x, self.draw_y, 'o',markersize=12,markeredgewidth=1,markeredgecolor='y',markerfacecolor='None')
        self.imshow_ = self.ax.imshow(self.im,interpolation='nearest',cmap=cm.gray)
        #connect mouse and keyboard
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        cid2 = self.fig.canvas.mpl_connect('key_press_event', self.press)
        cid3 = self.fig.canvas.mpl_connect('key_release_event', self.release)
        #show plot
        self.update_plot()
        plt.title(title)
        plt.show()
    def adjust_contrast(self):
        self.imshow_.set_clim(vmin=self.min_im, vmax=self.max_im)
        self.fig.canvas.draw()
    def load_coords(self):
        save_file = self.savefile
        if save_file is not None and os.path.exists(save_file):
            fid = open(save_file,'r')
            self.coords = pickle.load(fid)
            fid.close()
            if len(self.coords)>0: 
                self.draw_x,self.draw_y = zip(*self.coords)
                self.draw_x,self.draw_y =  map(list,[self.draw_x,self.draw_y])
            else: 
                self.draw_x,self.draw_y = [],[]
    def save_coords(self):
        save_file = self.savefile
        if save_file is not None:
            if not os.path.exists(os.path.dirname(save_file)):
                os.makedirs(os.path.dirname(save_file))
            fid = open(save_file,'w')
            pickle.dump(self.coords,fid)
            fid.close()
    def threshold(self):
        #default parameters:
        if not hasattr(self,"med_size"): self.med_size = 20
        if not hasattr(self,"med_height"): self.med_height=1.5
        if not hasattr(self,"im_minmax"): self.im_minmax = minmax(self.im)
        if not hasattr(self,"med_im"): self.med_im = ndimage.filters.median_filter(self.im_minmax,self.med_size)
        if not hasattr(self,"perc_height"): self.perc_height = 99
        if not hasattr(self,"min_max"): self.min_max = [np.min(self.im),np.max(self.im)]
        if not hasattr(self,"abs_height"): self.abs_height = np.mean(self.im)#np.percentile(self.im_minmax-self.med_im,self.perc_height)*(self.min_max[1]-self.min_max[0])+self.min_max[0]
        #handy local variables
        med_im = self.med_im
        med_height = self.med_height
        im = self.im_minmax
        abs_height_ = (float(self.abs_height)-self.min_max[0])/(self.min_max[1]-self.min_max[0])
        im__ = (im>(med_height*med_im))*(im>(med_im+abs_height_))
        return im__
    def centroids_bw(self,im,small_th=2,edge_crop=4):
        from skimage.measure import label as sklabel
        from skimage.measure import regionprops as skregionprops
        centroids = []
        sx,sy = im.shape
        label_ = sklabel(im)
        for region in skregionprops(label_):
            if region.area>small_th:
                cent = region.centroid
                if cent[0]>edge_crop and cent[1]>edge_crop and sx-cent[0]>edge_crop and sy-cent[1]>edge_crop:
                    centroids.append(cent)
        return centroids
    def populate_draw_xy(self,flip=False):
        if len(self.coords)>0: 
            self.draw_x,self.draw_y = zip(*self.coords)
            if flip: self.draw_x,self.draw_y =  map(list,[self.draw_y,self.draw_x])
            else: self.draw_x,self.draw_y =  map(list,[self.draw_x,self.draw_y])
        else: 
            self.draw_x,self.draw_y = [],[]
    def create_text(self):
        self.texts = []
        for i,xy in enumerate(self.coords):
            self.texts.append(self.ax.text(xy[0],xy[1],str(i),color='r'))
    def remove_text(self):
        if not hasattr(self,'texts'): self.texts = []
        for txt in self.texts:
            txt.remove()
    def update_plot(self):
        self.l.set_xdata(self.draw_x)
        self.l.set_ydata(self.draw_y)
        self.coords = zip(self.draw_x,self.draw_y)
        self.save_coords()
        self.remove_text()
        self.create_text()
        self.fig.canvas.draw()
    def press(self,event):
        if event.key == 'delete':
            self.draw_x.pop(-1)
            self.draw_y.pop(-1)
            self.update_plot()
        if event.key == ',':
            self.max_im = self.min_im+(self.max_im - self.min_im)*(1.-1./10.)
            self.adjust_contrast()
        if event.key == '.':
            self.max_im = self.min_im+(self.max_im - self.min_im)*(1.+1./10.)
            self.adjust_contrast()
        if event.key == 'shift':
            self.delete_mode = True
        if event.key in ['[','a',']']:
            if hasattr(self,'min_max'):
                if event.key == '[': self.abs_height-=0.05*self.min_max[-1]
                if event.key == ']': self.abs_height+=0.05*self.min_max[-1]
            self.im_bw = self.threshold()
            self.coords = self.centroids_bw(self.im_bw)
            self.class_ids = ['' for _ in range(len(self.coords))]
            self.populate_draw_xy(flip=True)
            self.update_plot()
    def release(self, event):
        self.class_id = ''
        if event.key == 'shift':
            self.delete_mode = False
            
    def onclick(self,event):
        if event.button==3:
            #print "click"
            if not hasattr(self,'delete_mode'): self.delete_mode = False
            if self.delete_mode:
                ind_ = np.argmin(np.sum(np.abs(np.array(self.coords)-np.array([[event.xdata,event.ydata]])),axis=-1))
                self.draw_x.pop(ind_)
                self.draw_y.pop(ind_)
            else:
                if event.xdata is not None and event.ydata is not None:
                    self.draw_x.append(event.xdata)
                    self.draw_y.append(event.ydata)
            self.update_plot()
def STORM_STD_crop(dax_fl,coords_good,save_folder=None,s_good=40,rep_fr = 50,memmap=True,tag=None,overwrite=False):
        """
        Given file name dax_fl and coordinates coords_good this function will save cropped Storm and Cy3 files.
        It will also save cy3 and cy5 image using the first few frames.
        """
        #This makes a dax object
        dax_obj = saa.datareader.DaxReader(dax_fl)
        #dimensions:
        sx,sy,sz = dax_obj.filmSize()
        
        # Decide on what files to save and whether to save them
        if save_folder is None:
            dax_fl_base = dax_fl
        else:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            dax_fl_base = save_folder+os.sep+"--".join(dax_fl.split(os.sep)[-3:])
        
        new_dax_cy3s,new_dax_storms = [],[]
        for count_pos,good_coord in enumerate(coords_good):
            im_sm_simulate = grab_block(np.zeros([sx,sy]),list(good_coord[::-1]),[s_good,s_good])
            if len(np.ravel(im_sm_simulate))>0:
                tagOut = tag
                if type(tag) in [tuple,list,np.ndarray]:
                    tagOut = str(tag[count_pos])
                if tag is None:
                    tagOut = str(count_pos)
                new_dax_cy3s.append(dax_fl_base.replace('.dax','_tag'+tagOut+'_coord'+str(list(np.array(good_coord,dtype=int)))+'_cy3.dax'))
                new_dax_storms.append(dax_fl_base.replace('.dax','_tag'+tagOut+'_coord'+str(list(np.array(good_coord,dtype=int)))+'_storm.dax'))
        dax_fullcy3 = dax_fl_base.replace('.dax','_fullcy3.dax')
        dax_fullcy5 = dax_fl_base.replace('.dax','_fullcy5.dax')
        dax_fullcy3_frames = dax_fullcy3.replace('.dax','_frames.pkl')
        dax_fullcy5_frames = dax_fullcy5.replace('.dax','_frames.pkl')
        #Decide whether to loadData
        loadData = False
        if overwrite or not os.path.exists(dax_fullcy3) or not os.path.exists(dax_fullcy3_frames):
            loadData = True
        if overwrite or not os.path.exists(dax_fullcy5) or not os.path.exists(dax_fullcy5_frames):
            loadData = True
        for dax_cy3,dax_strom in zip(new_dax_cy3s,new_dax_storms):
            if overwrite or not os.path.exists(dax_cy3) or not os.path.exists(dax_strom):
                loadData = True
        if loadData:
            #Load big dax data
            if not memmap:
                #load full dax
                im_ = dax_obj.loadAll()
            else:
                #Create memory map
                im_ = dax_obj.loadMap()
                im_=np.swapaxes(im_,1,2) 
            #Full Cy3 and STORM frames
            cy3_frames = np.ravel(zip(np.arange(rep_fr,sz,rep_fr)-1,np.arange(rep_fr,sz,rep_fr)))
            cy3_frames_extra = np.ravel(zip(np.arange(rep_fr,sz+1,rep_fr)-1,np.arange(rep_fr,sz+1,rep_fr)))
            cy3_bleed = np.ravel(zip(np.arange(rep_fr,sz,rep_fr)-2,np.arange(rep_fr,sz,rep_fr)+1))
            storm_frames = np.setdiff1d(np.setdiff1d(np.arange(sz),cy3_frames_extra),cy3_bleed)            
            #Storm first 10 frames
            im_STORM_ff = im_[1:10]
            #Deal with cy3 frames
            cy3_frames_fast = cy3_frames
            if len(cy3_frames)>100:
                cy3_frames_fast = get_cy3CenterFrames(dax_fl,rep_fr=rep_fr,sz_max=sz)
                if cy3_frames_fast is None:
                    cy3_frames_fast = [cy3_frames[0],cy3_frames[1],cy3_frames[len(cy3_frames)/2],cy3_frames[len(cy3_frames)/2+1],cy3_frames[-2],cy3_frames[-1]]
                    
            im_cy3s = im_[cy3_frames_fast]
            #Save STORM ff and Cy3
            if overwrite or not os.path.exists(dax_fullcy3) or not os.path.exists(dax_fullcy3_frames):
                pickle.dump(cy3_frames_fast,open(dax_fullcy3_frames,'w'))
                saa.writeDax(dax_fullcy3, im_cy3s)
            
            if overwrite or not os.path.exists(dax_fullcy5) or not os.path.exists(dax_fullcy5_frames):
                pickle.dump(np.arange(10),open(dax_fullcy5_frames,'w'))
                saa.writeDax(dax_fullcy5, im_STORM_ff)
            
            
            it_cy3,it_storm = iter(new_dax_cy3s),iter(new_dax_storms)
            if list(coords_good): # if coords is empty, just leave off with extracted cy3 and cy5 data. 
                #Loop through positions selected in the GUI
                for count_pos,good_coord in enumerate(coords_good):
                    print "Looping through position "+str(count_pos+1)+" out of "+str(len(coords_good))
                    im_sm = grab_block(im_,[0]+list(good_coord[::-1]),[2*sz,s_good,s_good])
                    if len(np.ravel(im_sm))>0:
                        im_sm_cy3 = im_sm[cy3_frames]
                        im_sm_storm = im_sm[storm_frames]
                        new_dax_cy3 = next(it_cy3)
                        new_dax_storm = next(it_storm)
                        if overwrite or not os.path.exists(new_dax_cy3):
                            pickle.dump(cy3_frames,open(new_dax_cy3.replace('.dax','_frames.pkl'),'w'))
                            saa.writeDax(new_dax_cy3, np.array(im_sm_cy3))
                        if overwrite or not os.path.exists(new_dax_storm):
                            pickle.dump(storm_frames,open(new_dax_storm.replace('.dax','_frames.pkl'),'w'))
                            saa.writeDax(new_dax_storm, np.array(im_sm_storm))
def load_cy3(dax_fl,rep_fr=50,func=np.max):
    """Initially Returns last cy3 image. Expect this to be the most stable in z.
    Bad idea for Zstacks
    
    """
    dax_obj = saa.datareader.DaxReader(dax_fl)
    sx,sy,sz = dax_obj.filmSize()
    im_ = dax_obj.loadMap()
    im_ = np.swapaxes(im_,1,2) 
    cy3_frames = np.ravel(zip(np.arange(rep_fr,sz,rep_fr)-1,np.arange(rep_fr,sz,rep_fr)))
    #Big storm movies are slow.
    cy3_frames_fast = cy3_frames
    if len(cy3_frames)>100:
        cy3_frames_fast = get_cy3CenterFrames(dax_fl,rep_fr=rep_fr)
        if cy3_frames_fast is None:
            cy3_frames_fast = [cy3_frames[0],cy3_frames[1],cy3_frames[len(cy3_frames)/2],cy3_frames[len(cy3_frames)/2+1],cy3_frames[-2],cy3_frames[-1]]
    cy3_frames_fast = np.array(cy3_frames_fast)
    cy3_frames_fast = cy3_frames_fast[cy3_frames_fast<len(im_)]
    im_cy3 = func(im_[cy3_frames_fast],axis=0)
    im_cy3 = im_cy3[1:-1,1:-1]
    del(im_)
    return im_cy3
def fftalign_guess(im1,im2,center=[0,0],max_disp=50,use_fft=False,normalized=False,plt_val=False):
        """
        Inputs: 2 images im1, im2 and a maximum displacement max_disp.
        This computes the cross-cor between im1 and im2 using fftconvolve (fast) and determines the maximum
        """
        
        if not use_fft:
            from scipy.signal import fftconvolve
            im2_=np.array(im2[::-1,::-1],dtype=float)
            im2_-=np.mean(im2_)
            im2_/=np.std(im2_)
            im1_=np.array(im1,dtype=float)
            im1_-=np.mean(im1_)
            im1_/=np.std(im1_)
            im_cor = fftconvolve(im1_,im2_, mode='full')
        else:
            from numpy import fft
            im2_=np.array(im2,dtype=float)
            im2_-=np.mean(im2_)
            im1_=np.array(im1,dtype=float)
            im1_-=np.mean(im1_)
            f0, f1 = [fft.fft2(arr) for arr in (im1_,im2_)]
            # spectrum can be filtered, so we take precaution against dividing by 0
            eps = abs(f1).max() * 1e-15
            # cps == cross-power spectrum of im0 and im2
            #cps = abs(fft.ifft2((f0 * f1.conjugate()) ))
            if normalized:
                cps = abs(fft.ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1) + eps)))
            else:
                cps = abs(fft.ifft2((f0 * f1.conjugate()) ))
            # scps = shifted cps
            im_cor = fft.fftshift(cps)

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
        if np.sum(im_cor>0)>0:
            im_cor[im_cor==0]=np.min(im_cor[im_cor>0])
        else:
            im_cor[im_cor==0]=0
        if plt_val:
            plt.figure()
            plt.plot([x],[y],'k+')
            plt.imshow(im_cor,interpolation='nearest')
            plt.show()
        xt,yt=np.round(-np.array(im_cor.shape)/2.+[y,x]).astype(int)
        return xt,yt
def threshold(im_,med_size = 20, med_height=1.5,perc_height=99,abs_height=200,plot_val=False):
    im = minmax(im_)
    med_im = ndimage.filters.median_filter(im,med_size)
    if abs_height is None:
        im__ = (im>(med_height*med_im))*(im>np.percentile(im,perc_height))
    else:
        abs_height_ = (float(abs_height)-np.min(im_))/(np.max(im_)-np.min(im_))
        im__ = (im>(med_height*med_im))*(im>(med_im+abs_height_))
    if plot_val:
        plt.figure()
        plt.imshow(im__)
        plt.show()
    return np.array(im__,dtype=np.uint8)

def get_frame_z_scan(dax_fl,start_buffer_frames =10,end_buffer_frames=1000):
    "reads the paramater file and returns z_frames such that z_frames[i] is the z of stage in nm in frame i"
    from xml.etree import ElementTree
    parameters = ElementTree.parse(dax_fl.replace('.dax','.xml')).getroot()
    zscan_stop = float(parameters.find('focuslock').find('zscan_stop').text)
    zscan_step = float(parameters.find('focuslock').find('zscan_step').text)
    zscan_frames_to_pause = int(parameters.find('focuslock').find('zscan_frames_to_pause').text)
    z_frames = np.array(list(np.zeros(start_buffer_frames))+flatten([np.zeros(zscan_frames_to_pause)+_ for i_,_ in enumerate(np.arange(-zscan_stop,zscan_stop,zscan_step))])+list(np.zeros(end_buffer_frames)))
    return z_frames*1000
    
def get_cy3CenterFrames(dax_fl,rep_fr=50,start_buffer_frames =10,sz_max=np.inf):
    from xml.etree import ElementTree
    parameters = ElementTree.parse(dax_fl.replace('.dax','.xml')).getroot()
    zscan_stop = float(parameters.find('focuslock').find('zscan_stop').text)
    zscan_step = float(parameters.find('focuslock').find('zscan_step').text)
    zscan_frames_to_pause = int(parameters.find('focuslock').find('zscan_frames_to_pause').text)
    cy3_final_frames = []
    if zscan_step==0:
        i_frames = None
        cy3_final_frames = None
    else:
        i_frames = np.array(list(np.zeros(start_buffer_frames))+flatten([np.zeros(zscan_frames_to_pause)+i_+1 for i_,_ in enumerate(np.arange(-zscan_stop,zscan_stop,zscan_step))]))
    if i_frames is not None:
        sz = len(i_frames)
        #assuming 2 frame repetition
        cy3_frames = np.ravel(zip(np.arange(rep_fr,sz,rep_fr)-1,np.arange(rep_fr,sz,rep_fr)))
        for iframe in np.unique(i_frames):
            step_frames = np.ravel(np.where(i_frames==iframe))
            step_frames_cy3_frames = np.intersect1d(step_frames,cy3_frames)
            len_ = len(step_frames_cy3_frames)
            if len_>2:
                cy3_final_frames.extend([step_frames_cy3_frames[len_/2],step_frames_cy3_frames[len_/2+1]])
            else: 
                cy3_final_frames.extend(step_frames_cy3_frames)
    cy3_final_frames=np.array(cy3_final_frames)
    cy3_final_frames = cy3_final_frames[cy3_final_frames<sz_max]
    return cy3_final_frames