######### imports
import numpy as np
import glob,os,sys
import matplotlib.pyplot as plt
import Fitting_v4 as ft
import time

######### Base functions

def get_ims(dax_fl,num_cols=3,sx=2048,sy=2048,remove_first=1,pad=10):
    """
    returns list of 3D images sorted by color in the order of shutters (typically cy7,cy5,cy3)
    """
    im = np.fromfile(dax_fl,dtype=np.uint16).reshape([-1,sx,sy]).swapaxes(1,2)
    if pad==0:
        ims = [im[remove_first:][icol::num_cols] for icol in range(num_cols)]
    else:
        ims = [im[remove_first:][icol:-pad:num_cols][int(pad/num_cols)+1:] for icol in range(num_cols)]
    lens = [len(im_)for im_ in ims]
    min_len = np.min(lens)
    ims = [im_[:min_len] for im_ in ims]
    return ims
def get_frame(dax_fl,ind_z=1,sx=2048,sy=2048):
    "returns single frame of a dax file"
    f = open(dax_fl, "rb")
    bytes_frame = sx*sy*2
    f.seek(bytes_frame*ind_z)
    im_ = np.fromfile(f,dtype=np.uint16,count=sx*sy).reshape([sx,sy]).swapaxes(0,1)
    f.close()
    return im_
def getH(folder):
    nm = os.path.basename(folder)
    let_ = ''
    for let in nm[1:]:
        if not let.isdigit():
            break
        let_+=let
    return int(let_)
def get_real_tags(htag,flips=['R3,R28','R4,R29','R8,R33','R14,R39','R21,R46','R67,R92','R72,R97']):
    for ilet,let in enumerate(htag[1:]):
        if not let.isdigit():
            break
    htags_real = htag[ilet+1:].split('_')[0]
    flip = -1
    if htags_real in flips or 'Q' in htags_real or 'b' in htags_real:
        flip=1
    htags_real = htags_real.upper()
    htags_real = htags_real.split(',')[::flip]
    return htags_real
def htags_to_Rs(htags):
    htags = np.array(htags)[np.argsort(map(getH,htags))]
    keep_tags = set()
    htags__all = {}
    for htag in htags:
        htags_ = get_real_tags(htag)
        htags__=[]
        for htg in htags_:
            rep = 0
            htg_ = htg
            while True:
                if htg_ not in keep_tags:
                    keep_tags.add(htg_)
                    htags__.append(htg_)
                    break
                else:
                    rep+=1
                    htg_ = htg+'_rep'+str(rep)

        htags__all[htag]=htags__
    return htags__all
def num_range_bead_colors(dax_fl):
    num_cols = 3
    tag = os.path.basename(os.path.dirname(dax_fl))
    range_colors = range(num_cols)[:-1]
    bead_col = num_cols-1
    if 'Q' in tag:
        num_cols = 4
        range_colors = range(num_cols)[:-1]
        bead_col = num_cols-1
    if 'B,B' in tag and '3_26_2019_HCT116_6hAuxin_SI14' not in dax_fl:
        num_cols = 5
        range_colors = []
        bead_col = num_cols-2
    if 'b,' in tag:
        num_cols = 4
        range_colors = [2]
        bead_col = num_cols-1
    return num_cols,range_colors,bead_col
############# Main program
def run_fitting_analysis(params,remove_first=0,im_pad=0,fit_colors=True,analysis_folder=None,Tzxy_tag=None):
    verbose = False
    try:
    #if True:
        dax_fl,dax_fl_ref,overwrite_fits,overwrite_drift=params
        hfolder = os.path.dirname(dax_fl)
        fov = os.path.basename(dax_fl)
        master_folder = os.path.dirname(hfolder)
        
        #overwrite_fits = False
        #overwrite_drift = False
        ######### Get the info for the current project
        
        #analysis_folder = r'I:\Bogdan_temp'+os.sep+os.path.basename(master_folder)+'-Analysis'
        if analysis_folder is None:
            analysis_folder = master_folder+'-Analysis'
        if not os.path.exists(analysis_folder): os.makedirs(analysis_folder)
        
        
        
        start_program = time.time()
        
        ## where to save
        base = os.path.basename(os.path.dirname(dax_fl))+'___'+os.path.basename(dax_fl).replace('.dax','')
        
        #mark/check for completeness
        finished_folder = analysis_folder+os.sep+'finished'
        if not os.path.exists(finished_folder): os.makedirs(finished_folder)
        finished_fl = finished_folder+os.sep+base+'.txt'

        if not os.path.exists(finished_fl) or overwrite_fits or overwrite_drift:
        
            ############################Loading image##############
            print "Loading image:",dax_fl
            start = time.time()
            im_meds = np.load(analysis_folder+os.sep+'im_meds.npy')
            num_cols,range_colors,bead_col = num_range_bead_colors(dax_fl)
            ims = get_ims(dax_fl,num_cols=num_cols,remove_first=remove_first,pad=im_pad)
            end = time.time()
            print "time:",end-start
            real_colors = range(num_cols)[:-1]

            fits_folder = analysis_folder+os.sep+'Fits'
            if not os.path.exists(fits_folder): os.makedirs(fits_folder)
            if fit_colors:
                for icol in range_colors:
                    print "Dealing with color "+str(icol)
                    start = time.time()
                    pfits_fl = fits_folder+os.sep+base+'__col'+str(icol)+'__fits.npy'
                    if not os.path.exists(pfits_fl) or overwrite_fits:
                        #details for tiles. tiles are used to conserve RAM
                        nsq=4
                        delta = int(2048/nsq)
                        sz = int(delta/2)
                        centers_ref = np.array([[int(delta/2)+delta*i,int(delta/2)+delta*j] for i in range(nsq) for j in range(nsq)])
                        pad=10
                        
                        ps=[]

                        for center in centers_ref:
                            min_x = int(np.max([center[0]-sz-pad,0]))
                            max_x = int(np.min([center[0]+sz+pad,im_meds[icol].shape[0]]))
                            min_y = int(np.max([center[1]-sz-pad,0]))
                            max_y = int(np.min([center[1]+sz+pad,im_meds[icol].shape[1]]))
                            im_ = ims[icol][:,min_x:max_x,min_y:max_y].astype(np.float32)
                            im_med = im_meds[icol][min_x:max_x,min_y:max_y]
                            im_ = im_-im_med
                            im_ = im_/np.std(im_)
                            centers,std_im = ft.get_seed_points_base_v2(im_, gfilt_size=15, filt_size=5, th_seed=8., max_num=None)
                            ps_ = []
                            if len(centers[0])>0:
                                sz_full = ims[icol].shape
                                keep = (centers[1]>=(pad if min_x>0 else 0))
                                keep = keep&(centers[1]<(im_.shape[1]-pad if max_x<sz_full[1] else im_.shape[1]))
                                keep = keep&(centers[2]>=(pad if min_y>0 else 0))
                                keep = keep&(centers[2]<(im_.shape[2]-pad if max_y<sz_full[2] else im_.shape[2]))
                                centers_zxy = centers[:3,keep].T
                                ps_ = ft.fast_fit_big_image(im_,centers_zxy,radius_fit = 4, verbose = False)
                                if len(ps_)>0:
                                    ps_[:,2:4]+=[min_x,min_y]
                            ps.extend(ps_)
                            #print(len(ps))
                        ps = np.array(ps,dtype=np.float32)
                        print "Number of points:",len(ps)
                        np.save(pfits_fl,ps)
                    end = time.time()
                    print "time:",end-start
                
            print "Dealing with the drift"
            
            start =  time.time()
            ####### paramaters ####################################
            
            nsq = 4 # There will be nsq*nsq tiles of images
            delta = int(2048/nsq)
            centers_ref = np.array([[int(delta/2)+delta*i,int(delta/2)+delta*j] for i in range(nsq) for j in range(nsq)])
            sz =int(delta/2)
            
            nmax = 1000 # The number of fitted points per tile
            
            base_ref = os.path.basename(os.path.dirname(dax_fl_ref))+'___'+os.path.basename(dax_fl_ref).replace('.dax','')
            beads_folder = analysis_folder+os.sep+'beads'
            if not os.path.exists(beads_folder): os.makedirs(beads_folder)
            im_meds_beads = im_meds[bead_col]
            #print dax_fl_ref,base_ref,glob.glob(beads_folder+os.sep+base_ref+'__reg*.npy'),len(glob.glob(beads_folder+os.sep+base_ref+'__reg*.npy')),len(centers_ref)
            #Load and tile the reference image
            if len(glob.glob(beads_folder+os.sep+base_ref+'__reg*.npy'))!=len(centers_ref):
                print "Loading reference image..."
                num_cols_ref,range_colors_ref,bead_col_ref = num_range_bead_colors(dax_fl_ref)
                ims_ref = get_ims(dax_fl_ref,num_cols=num_cols_ref,remove_first=remove_first,pad=im_pad)
                for ic,center in enumerate(centers_ref):
                    beads_savefl = beads_folder+os.sep+base_ref+'__reg'+str(ic)+'.npy'
                    if not os.path.exists(beads_savefl):
                        print "Making file:",beads_savefl
                        min_x = int(np.max([center[0]-sz,0]))
                        max_x = int(np.min([center[0]+sz,im_meds_beads.shape[0]]))
                        min_y = int(np.max([center[1]-sz,0]))
                        max_y = int(np.min([center[1]+sz,im_meds_beads.shape[1]]))
                        im_beads_ref = ims_ref[bead_col_ref][:,min_x:max_x,min_y:max_y]-im_meds_beads[min_x:max_x,min_y:max_y]
                        im_beads_ref = im_beads_ref/np.std(im_beads_ref)
                        seeds_ref,std_ref = ft.get_seed_points_base_v2(im_beads_ref,gfilt_size=15, filt_size=5, th_seed=7., max_num=None)
                        bead_ps = ft.fast_fit_big_image(im_beads_ref,seeds_ref[:3,:nmax].T,verbose=False)
                        np.save(beads_savefl,[im_beads_ref,bead_ps])
            # Register
            drift_file_save = beads_folder+os.sep+base+'___drift.npy'
            if not os.path.exists(drift_file_save) or overwrite_drift:
                zns,zos=[],[]
                all_zn,all_zo=[],[]
                for ic,center in enumerate(centers_ref):
                    beads_savefl = beads_folder+os.sep+base_ref+'__reg'+str(ic)+'.npy'
                    [im_beads_ref,ref_bead_ps] = np.load(beads_savefl)
                    ref_bead_ps = ft.fast_fit_big_image(im_beads_ref,ref_bead_ps[:,1:4],radius_fit = 4,avoid_neigbors=False,verbose=False,recenter=True,better_fit=False)
                    
                    min_x = int(np.max([center[0]-sz,0]))
                    max_x = int(np.min([center[0]+sz,im_meds_beads.shape[0]]))
                    min_y = int(np.max([center[1]-sz,0]))
                    max_y = int(np.min([center[1]+sz,im_meds_beads.shape[1]]))
                    im_beads = ims[bead_col][:,min_x:max_x,min_y:max_y]-im_meds_beads[min_x:max_x,min_y:max_y]
                    im_beads = im_beads/np.std(im_beads)
                    if Tzxy_tag is None: Tzxy = ft.fft3d_from2d(im_beads_ref, im_beads,max_disp=sz)
                    else: Tzxy=0
                    beads_seeds = ref_bead_ps[:,1:4]-Tzxy
                    print "Rough drift:",Tzxy
                    bead_ps = ft.fast_fit_big_image(im_beads,beads_seeds,radius_fit = 4,avoid_neigbors=False,verbose=False,recenter=True,better_fit=False)
                    
                    zxy_new = bead_ps[:,[1,2,3,0]]+[0,min_x,min_y,0]
                    zxy_old = ref_bead_ps[:,[1,2,3,0]]+[0,min_x,min_y,0]
                    kp = (~np.isnan(zxy_new[:,0]))&(~np.isnan(zxy_old[:,0]))
                    zxy_new = zxy_new[kp]
                    zxy_old = zxy_old[kp]
                    print "Fine drift:",np.median(zxy_new-zxy_old,0)
                    all_zn.extend(zxy_new)
                    all_zo.extend(zxy_old)
                    zns.append(np.mean(zxy_new[:,:-1],axis=0))
                    zos.append(np.mean(zxy_old[:,:-1],axis=0))
                all_zn,all_zo,zns,zos = np.array(all_zn),np.array(all_zo),np.array(zns),np.array(zos)
                m__ = ft.calc_color_matrix(all_zo[:,:3],all_zn[:,:3], order=1)
                np.save(drift_file_save,[all_zn,all_zo,zns,zos,m__,dax_fl_ref])
                print "Expected drift error (z,x,y) in pixels: ",np.median(np.abs(zos-ft.apply_colorcor(zns,m__)),0)
                end=time.time()
                print "time:",end-start
                end_program = time.time()
            print "Total time:",end_program-start_program
            fid = open(finished_fl,'w')
            fid.close()
        return True
    except Exception as e: 
        print(e)
        return False