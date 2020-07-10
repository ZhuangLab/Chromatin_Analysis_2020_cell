from . import visual_tools as vis
from . import drift_corr as dft
import numpy as np
import pickle as pickle
import matplotlib.pylab as plt
import os, glob, sys
sys.path.append(r'C:\Users\puzheng\Documents\python-functions\python-functions-library')

def select_candidate_spots(folders, fovs, analysis_folder, buffer_frames=10, crop_dm=25, ref=0, dft_corr_size=120, bead_th=150):
    '''Function for picking candidate spots
    Given:
    folders: directories for each hyb, list of strings
    fov: field of views names, list of strings
    analysis folder: the path to store analysis data, string
    buffer_frame: setting by z-scan, non-negative int
    crop_dm: half length in pixels of cropped region around each chromosome, positive int
    ref: frame number for reference, non-negative int
    dft_corr_size: size of square cropped for bead-fitting, positive int
    bead_th: beads fitting threshold, positive real number
    Save:
    A series of pickle files, categorized by fov
    '''
    # load selected spot information
    _dic = pickle.load(open(analysis_folder+os.sep+'Selected_Spot.pkl','r'))
    # convert dictionary into two parts:
    _chr_pts = vis.partition_map(_dic['coords'],_dic['class_ids'])
    _fov_ids = np.unique(_dic['class_ids'])
    # count failures
    _total_fails = 0
    # Use GUI to run across all field-of-views:
    for _ind in range(len(_fov_ids)):
        #try:
        plt.close('all')
        #_im_means = [] # initialize
        _fov_id = _fov_ids[_ind] # fov id
        _chr_pt = _chr_pts[_ind] # chromosome points in this fov
        _filename = fovs[_fov_id] # filename
        print("--"+_filename)
        _ims_cy5, _ims_cy3, _names=[],[],[]
        for _folder in folders[:]:
            _file = _folder + os.sep + _filename
            if os.path.exists(_file):
                _names += [os.path.basename(_folder)]
                _im = vis.DaxReader(_file).loadMap()
                _ims_cy3 += [_im[2::2][int(buffer_frames/2): -int(buffer_frames/2)]]
                _ims_cy5 += [_im[1::2][int(buffer_frames/2): -int(buffer_frames/2)]]
        # Generate a chromosome view object:
        _chrom_view = vis.imshow_mark_3d_v2(_ims_cy5, image_names=folders)
        # Setting up fitting parameters
        _chrom_view.th_seed = 300
        _chrom_view.width_z = 1.8 * 1.5 / 2
        _chrom_view.hot_pix_th = 4

        from scipy.spatial.distance import cdist    

        # Compute the drift:
        _drift,_overwrite,_fail_ct = dft.get_STD_beaddrift(_ims_cy3, analysis_folder=analysis_folder, fovs=fovs, fov_id=_fov_id, \
                                                  th_seed=bead_th, sz_ex=dft_corr_size,ref=ref,force=False,quiet=True)
        # initialize candidate spots
        _cand_spot = {}
        _cand_savefile = analysis_folder+os.sep+fovs[_fov_id].replace('.dax','__current_cand.pkl')
        if os.path.exists(_cand_savefile) and not _overwrite:
            _cand_spot = pickle.load(open(_cand_savefile,'rb'))

        # Now, loop through all possible chromosomes in certain field of view
        for _chr_id, _chr_coord in enumerate(_chr_pt): # chromosome id, chromosome coordinate
            if _chr_id not in _cand_spot:
                # select the surrounding (zoom in)
                _chrom_view.ax2.set_ylim(_chr_coord[2]+crop_dm+crop_dm/2,_chr_coord[2]-crop_dm-crop_dm/2) # z-axis
                # initialize
                _new_ids, _new_coords=[],[]
            #for each chr you have a candid_dic. the candid_dic dictionary goes across hybes and reports a list of fits
                _candid_dic={} #this will be saved to file

                sel_cl_dic={}
                sel_max_dic={}

                for _chrom_view.index_im in range(len(_chrom_view.ims)):
                    # retrive drift info
                    drift_x,drift_y,drift_z = _drift[_chrom_view.index_im][::-1]
                    drift_x,drift_y = int(drift_x),int(drift_y)
                    # set limit of image, do fitting
                    _chrom_view.set_image()
                    _chrom_view.ax1.set_xlim(_chr_coord[0]+drift_x-crop_dm,_chr_coord[0]+drift_x+crop_dm) # x-axis
                    _chrom_view.ax1.set_ylim(_chr_coord[1]+drift_y+crop_dm,_chr_coord[1]+drift_y-crop_dm) # y-axis
                    _chrom_view.auto_scale()
                    _chrom_view.get_seed_points()
                    _chrom_view.fit_seed_points()
                    # results of fitting:
                    _ids=np.array(_chrom_view.class_ids)
                    _coords=np.array(_chrom_view.coords)

                    # Proceed if there is any fitting
                    if len(_chrom_view.pfits)>0:
                        # Acquire tag
                        _tag = os.path.basename(_chrom_view.image_names[_chrom_view.index_im]) # example: H1R1_bigstep
                        # Apply drift correction
                        _candid_dic[_tag] = np.array(_chrom_view.pfits)
                        _candid_dic[_tag][:,1:4]=_candid_dic[_tag][:,1:4] - [_drift[_chrom_view.index_im]]#drift correction
                        # match fitted points in this hybe to chromosomes
                        _all_pts_hybe = _candid_dic[_tag][:,1:4]#zxy for candidate fits
                        
                        #split associate points with chr/fov
                        _chr_pts_comp =[_pt[::-1]for _pt in _chr_pt]#transform to pfits conversion
                        ### Key point, could be changed
                        _inds = np.argmin(cdist(_all_pts_hybe,_chr_pts_comp),-1)#indices to nearest chr

                        #print _candid_dic[_tag][:,1:4]
                        #print _drift[_chrom_view.index_im]
                        # keep candidates with matched chromosome id
                        _keep = _inds==_chr_id
                        _candid_dic[_tag] = _candid_dic[_tag][_keep]#keep only the points closese to the current chromosome
                        print(_inds)
                        print(_chr_id)
                        # Update Plot
                        if np.sum(_keep)>0:
                            _new_coords.append(_chrom_view.pfits[_keep][0][1:4][::-1])
                            _new_ids.append(_chrom_view.index_im)

                        #print _new_coords
                        _chrom_view.coords=_new_coords
                        _chrom_view.class_ids=_new_ids
                        _chrom_view.populate_draw_xyz()
                        _chrom_view.set_image()
                _cand_spot[_chr_id]=(_candid_dic,np.array(_chr_coord[::-1]))
                pickle.dump(_cand_spot,open(_cand_savefile,'wb'))
        #except:
        #    print "Failed"
        #    _total_fails += 1;
