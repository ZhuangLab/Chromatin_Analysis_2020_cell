import sys, os, time, glob, re
import numpy as np
import pickle as pickle
import matplotlib.pylab as plt
import scipy
import multiprocessing as mp

# functions from other sub-packages
from .. import get_img_info, corrections, visual_tools, alignment_tools, classes
from ..External import Fitting_v3, DomainTools
from .. import _correction_folder, _temp_folder, _distance_zxy, _sigma_zxy, _image_size, _allowed_colors
from ..figure_tools.domain import plot_boundaries
from .distance import domain_distance, domain_pdists, domain_correlation_pdists
from . import interpolate_chr

# 
from scipy.signal import find_peaks, fftconvolve
from scipy.stats import normaltest, ks_2samp, ttest_ind
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree, is_valid_linkage


def generate_candidate_domain_boundary(coordinates, dm_sz=5, 
                                       match_boundary_dist=1, adjust_corr_bd=False,
                                       make_plot=False):
    """Generate candidate domain boundaries"""
    from .distance import _sliding_window_dist
    from scipy.signal import find_peaks
    # initialize coordinates and convert to squared array
    _coordinates = np.array(coordinates)
    if len(np.shape(_coordinates))!= 2:
        raise IndexError(f"Wrong input shape for _coordinates, should be 2d-array but {_coordinates.shape} is given.")
    if np.shape(_coordinates)[1] == 3:
        _coordinates = squareform(pdist(_coordinates))
        
    # get sliding window peaks
    _slide_dists = _sliding_window_dist(_coordinates, dm_sz)
    _slide_peaks = find_peaks(_slide_dists, distance=dm_sz)[0]
    
    # get correlation corr peaks
    _corr_map = np.ma.corrcoef(np.ma.masked_invalid(_coordinates))
    _corr_dists = []
    for _i in range(dm_sz, len(_coordinates)-dm_sz):
        _corr_dists.append(np.linalg.norm(_corr_map[_i-dm_sz:_i] - _corr_map[_i:_i+dm_sz]))
    _corr_dists = np.array(_corr_dists)
    _corr_peaks = find_peaks(_corr_dists, distance=dm_sz)[0]+dm_sz
    #print(_slide_peaks, _slide_dists[_slide_peaks])
    #print(_corr_peaks, _corr_dists[_corr_peaks-dm_sz])
    if make_plot:
        plt.figure(figsize=(9,3))
        plt.plot(np.arange(len(_coordinates)), _slide_dists*10, label='window')
        plt.plot(np.arange(dm_sz, len(_coordinates)-dm_sz), _corr_dists, label='corr')
        plt.legend()
        plt.title(f"distance measures")
        plt.show()
    # match corr_peaks and slide peaks
    _kept_peaks = [0]
    for _p in _corr_peaks:
        if (np.abs(_slide_peaks-_p) <= match_boundary_dist).any():
            if adjust_corr_bd:
                _matched_peak = _slide_peaks[np.argmin(np.abs(_slide_peaks-_p))]
                _kept_peaks.append( int((_matched_peak+_p)/2) )
            else:
                _kept_peaks.append(_p)
    return np.array(_kept_peaks, dtype=np.int)


def merge_domains(coordinates, cand_bd_starts, norm_mat=None, 
                  corr_th=0.05, dist_th=0.8, flexible_rate=0.2, 
                  domain_dist_metric='median', plot_steps=False, verbose=True):
    """Function to merge domains given zxy coordinates and candidate_boundaries"""
    cand_bd_starts = np.array(cand_bd_starts, dtype=np.int)
    _merge_inds = np.array([-1])
    if verbose:
        print(
            f"-- start iterate to merge domains, num of candidates:{len(cand_bd_starts)}")
    # start iteration:
    while len(_merge_inds) > 0 and len(cand_bd_starts) > 1:
        # calculate domain pairwise-distances
        _dm_pdists = domain_pdists(coordinates, cand_bd_starts,
                                   metric=domain_dist_metric, 
                                   normalization_mat=norm_mat)
        _dm_dist_mat = squareform(_dm_pdists)
        # remove domain candidates that are purely nans
        _keep_inds = np.isnan(_dm_dist_mat).sum(1) < len(_dm_dist_mat)-1
        if np.sum(_keep_inds) != len(cand_bd_starts) and verbose:
            print(f"---** remove {len(cand_bd_starts)-np.sum(_keep_inds)} domains because of NaNs")
        _dm_dist_mat = _dm_dist_mat[_keep_inds][:,_keep_inds]
        cand_bd_starts = cand_bd_starts[_keep_inds]
        # calculate correlation coefficient pdist
        _dm_corr_pdists = domain_correlation_pdists(coordinates, cand_bd_starts)
        _dm_corr_mat = squareform(_dm_corr_pdists)
        # plot step if specified
        if plot_steps:
            _fig, _axes = plt.subplots(1,2, figsize=(8,3), sharey=True)
            _axes[0].imshow(_dm_dist_mat, cmap='seismic_r')
            _axes[1].imshow(_dm_corr_mat, cmap='seismic')
            _axes[0].set_title(f"Domain distances")
            _axes[1].set_title(f"Domain correlations")
            plt.show()
        # get nb_dists
        _nb_dists = np.diag(_dm_dist_mat, 1)
        _nb_corrs = np.diag(_dm_corr_mat, 1)
        # update domain_id to be merged (domain 0 will never be merged)
        # generate conditions
        _merge_ind_list = [np.where((_nb_dists<=dist_th)* (_nb_corrs<=corr_th))[0]+1] # condition1: all pass
        if flexible_rate > 0 and flexible_rate < 1:
            _merge_ind_list.append(
                np.where((_nb_dists<=dist_th*(1+flexible_rate)) * (_nb_corrs<=corr_th*(1-flexible_rate)))[0]+1
            )
            _merge_ind_list.append(
                np.where((_nb_dists<=dist_th*(1-flexible_rate)) * (_nb_corrs<=corr_th*(1+flexible_rate)))[0]+1
            )
        # summarize merge_inds
        _merge_inds = np.unique(np.concatenate(_merge_ind_list)).astype(np.int)
        # if there are any domain to be merged:
        if len(_merge_inds) > 0:
            # find the index with minimum distance bewteen neighboring domains
            # first merge neighbors with high corr and low dist
            _merge_dists = _nb_dists / dist_th + _nb_corrs / corr_th 
            # filter
            _merge_dists = _merge_dists[np.array(_merge_inds, dtype=np.int)-1]
            _picked_ind = _merge_inds[np.argmin(_merge_dists)]
            if verbose:
                print(f"---* merge domain:{_picked_ind} starting with region:{cand_bd_starts[_picked_ind]}, dist={np.diag(_dm_corr_mat, 1)[_picked_ind-1].round(4), np.diag(_dm_dist_mat, 1)[_picked_ind-1].round(4)}")
            # remove this domain from domain_starts (candidates)
            cand_bd_starts = np.delete(cand_bd_starts, _picked_ind)
    
    if verbose and len(cand_bd_starts) == 1:
            print(f"--- only 1 domain left, skip plotting.")
    elif verbose:
        print(
            f"--- final neighbor domain dists:{np.diag(_dm_dist_mat,1).round(4)}")
        print(f"--- final neighbor domain corr:{np.diag(_dm_corr_mat,1).round(4)}")
        print(f"--- num of domain kept:{len(cand_bd_starts)}")

    return cand_bd_starts


def basic_domain_calling(spots, save_folder=None,
                         distance_zxy=_distance_zxy, gfilt_size=0.,
                         normalization_matrix=None, min_domain_size=5, 
                         match_boundary_dist=3, adjust_corr_bd=False, 
                         domain_dist_metric='median', domain_cluster_metric='average',
                         corr_th=0.03, dist_th=0.65, flexible_rate=0.25,
                         plot_steps=False, plot_results=True,
                         fig_dpi=150,  fig_dim=4, fig_font_size=12,
                         save_result_figs=False, save_name='', verbose=True):
    """Function to call single-cell domains by thresholding correlation matrices.
    --------------------------------------------------------------------------------
    The idea for subcompartment calling:
    1. call rough domain candidates by maximize local distances.
    2. calculate 'distances' between domain candidates
    3. merge neighboring domains by correlation between domain distance vectors
    4. repeat step 2 and 3 until converge
    --------------------------------------------------------------------------------
    Inputs:
        spots: all sepected spots for this chromosome, np.ndarray
        distance_zxy: transform pixels in spots into nm, np.ndarray-like of 3 (default: [200,106,106])
        min_domain_size: domain window size of the first rough domain candidate calling, int (default: 5)
        gfilt_size: filter-size to gaussian smooth picked coordinates, float (default: 1.)
        normalization_matrix: either path or matrix for normalizing polymer effect, str or np.ndarray 
            * if specified, allow normalization, otherwise put a None. (default:str of path)        
        corr_th: lower threshold for correlations to merge neighboring vectors, float (default: 0.6)
        dist_th: upper threshold for distance to merge neighboring vectors, float (default: 1.)
        plot_steps: whether make plots during intermediate steps, bool (default: False)
        plot_results: whether make plots for results, bool (default: True)
        fig_dpi: dpi of image, int (default: 200)  
        fig_dim: dimension of subplot of image, int (default: 10)  
        fig_font_size: font size of titles in image, int (default: 12)
        save_result_figs: whether save result image, bool (default: False)
        save_folder: folder to save image, str (default: None, which means not save)
        save_name: filename of saved image, str (default: '', which will add default postfixs)
        verbose: say something!, bool (default: True)
    Output:
        cand_bd_starts: candidate boundary start region ids, list / np.1darray
        """
    ## check inputs
    if not isinstance(distance_zxy, np.ndarray):
        distance_zxy = np.array(distance_zxy)
    if len(distance_zxy) != 3:
        raise ValueError(
            f"size of distance_zxy should be 3, however {len(distance_zxy)} was given!")
    # load normalization if specified
    if isinstance(normalization_matrix, str) and os.path.isfile(normalization_matrix):
        norm_mat = np.load(normalization_matrix)
        _normalization = True
    elif isinstance(normalization_matrix, np.ndarray):
        norm_mat = normalization_matrix.copy()
    else:
        norm_mat = None
        _normalization = False

    ## 0. prepare coordinates
    if verbose:
        print(f"-- start basic domain calling")
        _start_time = time.time()
    # get zxy
    _spots = np.array(spots)
    if _spots.shape[1] == 3:
        _zxy = _spots
    else:
        _zxy = np.array(_spots)[:, 1:4] * distance_zxy[np.newaxis, :]
    # smooth
    if gfilt_size is not None and gfilt_size >= 0:
        if verbose:
            print(f"--- gaussian interpolate chromosome, sigma={gfilt_size}")
        _zxy = interpolate_chr(_zxy, gaussian=gfilt_size)

    ## 1. call candidate domains
    if verbose:
        print(f"--- call initial candidate boundaries")
    cand_bd_starts = generate_candidate_domain_boundary(_zxy, min_domain_size, 
                        match_boundary_dist=match_boundary_dist, 
                        adjust_corr_bd=adjust_corr_bd)

    ## 2. get zxy sequences
    cand_bd_starts = merge_domains(_zxy, cand_bd_starts=cand_bd_starts,
                                   norm_mat=norm_mat, corr_th=corr_th, dist_th=dist_th,
                                   flexible_rate=flexible_rate,
                                   domain_dist_metric=domain_dist_metric,
                                   plot_steps=plot_steps, verbose=verbose)

    ## 3. finish up and make plot
    if plot_results and len(cand_bd_starts) > 1:
        from .distance import domain_pdists
        import matplotlib.gridspec as gridspec

        _dm_pdists = domain_pdists(_zxy, cand_bd_starts,
                                   metric=domain_dist_metric,
                                   normalization_mat=norm_mat,
                                   allow_minus_dist=False)
        _dm_corr_pdists = domain_correlation_pdists(_zxy, cand_bd_starts)
        if verbose:
            print(
                f"-- make plot for results with {len(cand_bd_starts)} domains")
        
        _fig = plt.figure(figsize=(3*(fig_dim+1), 2*fig_dim), dpi=fig_dpi)
        gs = gridspec.GridSpec(2, 5, figure=_fig)
        gs.update(wspace=0.1, hspace=0)
        #ax1 = plt.subplot2grid((2, 5), (0, 0), colspan=1)
        ax1 = plt.subplot(gs[0, 0])
        ax1.set_title('Domain noramlized distances',
                      fontsize=fig_font_size)
        im1 = ax1.imshow(squareform(_dm_pdists), cmap='seismic_r',)
        cb1 = plt.colorbar(im1, ax=ax1, shrink=0.6)
        cb1.ax.tick_params(labelsize=fig_font_size-2, width=0.6, length=1)

        #ax2 = plt.subplot2grid((2, 5), (0, 1), colspan=1)
        ax2 = plt.subplot(gs[0, 1])
        ax2.set_title('Domain correlation distances',
                      fontsize=fig_font_size)
        im2 = ax2.imshow(squareform(_dm_corr_pdists), cmap='seismic_r')
        cb2 = plt.colorbar(im2, ax=ax2, shrink=0.6)
        cb2.ax.tick_params(labelsize=fig_font_size-2, width=0.6, length=1)

        #ax4 = plt.subplot2grid((2, 5), (1, 0), colspan=2, rowspan=1)
        ax4 = plt.subplot(gs[1, 0:2])
        ax4.set_title('Hierarchal clustering of domains',
                      fontsize=fig_font_size)
        # calculate hierarchy clusters
        _hierarchy_clusters = linkage(_dm_pdists, method=domain_cluster_metric)
        # draw dendrogram
        dn = dendrogram(_hierarchy_clusters, ax=ax4)
        ax4.tick_params(labelsize=fig_font_size)

        #ax3 = plt.subplot2grid((2, 5), (0, 2), colspan=3, rowspan=2)
        ax3 = plt.subplot(gs[:, 2:])
        plot_boundaries(squareform(pdist(_zxy)), cand_bd_starts, input_ax=ax3,
                        figure_dpi=fig_dpi, figure_fontsize=fig_font_size, 
                        line_width=1.5, save=save_result_figs, plot_limits=[0,1000],
                        save_folder=save_folder, save_name=save_name)
        # save result figure
        if save_result_figs and save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            if save_name == '':
                _result_save_name = 'basic_domain_calling.png'
            else:
                _result_save_name = save_name + '_basic_domain_calling.png'
            _full_result_filename = os.path.join(
                save_folder, _result_save_name)
            if verbose:
                print(f"--- save result image into file:{_full_result_filename}")
            plt.savefig(_full_result_filename, transparent=True)
        if __name__ == '__main__':
            plt.show()

    if verbose:
        print(
            f"--- total time spent in basic domain calling: {time.time()-_start_time}")

    return cand_bd_starts
def iterative_domain_calling(spots, save_folder=None,
                             distance_zxy=_distance_zxy, dom_sz=5, gfilt_size=0.5,
                             split_level=1, num_iter=5, corr_th_scaling=1., dist_th_scaling=1.,
                             normalization_matrix=r'Z:\References\normalization_matrix.npy',
                             domain_dist_metric='ks', domain_cluster_metric='ward',
                             corr_th=0.6, dist_th=0.2, plot_steps=False, plot_results=True,
                             fig_dpi=100,  fig_dim=10, fig_font_size=18,
                             save_result_figs=False, save_name='', verbose=True):
    """Function to call 'sub-compartments' by thresholding correlation matrices.
    --------------------------------------------------------------------------------
    The idea for subcompartment calling:
    1. call rough domain candidates by maximize local distances.
    2. calculate 'distances' between domain candidates
    3. merge neighboring domains by correlation between domain distance vectors
    4. repeat step 2 and 3 until converge
    --------------------------------------------------------------------------------
    Inputs:
        spots: all sepected spots for this chromosome, np.ndarray
        distance_zxy: transform pixels in spots into nm, np.ndarray-like of 3 (default: [200,106,106])
        dom_sz: domain window size of the first rough domain candidate calling, int (default: 5)
        gfilt_size: filter-size to gaussian smooth picked coordinates, float (default: 1.)
        split_level: number of iterative split candidate domains, int (default: 1)
        num_iter: number of iterations for split-merge domains, int (default: 5)
        corr_th_scaling: threshold scaling for corr_th during split-merge iteration, float (default: 1.)
        dist_th_scaling: threshold scaling for dist_th during split-merge iteration, float (default: 1.)
        normalization_matrix: either path or matrix for normalizing polymer effect, str or np.ndarray 
            * if specified, allow normalization, otherwise put a None. (default:str of path)
        corr_th: lower threshold for correlations to merge neighboring vectors, float (default: 0.64)
        dist_th: upper threshold for distance to merge neighboring vectors, float (default: 0.2)
        plot_steps: whether make plots during intermediate steps, bool (default: False)
        plot_results: whether make plots for results, bool (default: True)
        fig_dpi: dpi of image, int (default: 200)  
        fig_dim: dimension of subplot of image, int (default: 10)  
        fig_font_size: font size of titles in image, int (default: 18)
        save_result_figs: whether save result image, bool (default: False)
        save_folder: folder to save image, str (default: None, which means not save)
        save_name: filename of saved image, str (default: '', which will add default postfixs)
        verbose: say something!, bool (default: True)
    Output:
        cand_bd_starts: candidate boundary start region ids, list / np.1darray
        """
    ## check inputs
    if not isinstance(distance_zxy, np.ndarray):
        distance_zxy = np.array(distance_zxy)
    if len(distance_zxy) != 3:
        raise ValueError(
            f"size of distance_zxy should be 3, however {len(distance_zxy)} was given!")
    # load normalization if specified
    if isinstance(normalization_matrix, str) and os.path.isfile(normalization_matrix):
        norm_mat = np.load(normalization_matrix)
        _normalization = True
    elif isinstance(normalization_matrix, np.ndarray):
        _normalization = True
        norm_mat = normalization_matrix.copy()
    else:
        norm_mat = None 
        _normalization = False
    if corr_th_scaling <= 0:
        raise ValueError(
            f"corr_th_scaling should be float number in [0,1], while {corr_th_scaling} given!")
    if dist_th_scaling <= 0:
        raise ValueError(
            f"dist_th_scaling should be float number in [0,1], while {dist_th_scaling} given!")
    ## 0. prepare coordinates
    if verbose:
        print(f"- start iterative domain calling")
        _start_time = time.time()
    # get zxy
    _spots = np.array(spots)
    if _spots.shape[1] == 3:
        _zxy = _spots
    else:
        _zxy = np.array(_spots)[:, 1:4] * distance_zxy[np.newaxis, :]
    # smooth
    if gfilt_size is not None and gfilt_size > 0:
        if verbose:
            print(f"--- gaussian interpolate chromosome, sigma={gfilt_size}")
        _zxy = interpolate_chr(_zxy, gaussian=gfilt_size)

    ## 1. do one round of basic domain calling
    cand_bd_starts = basic_domain_calling(spots=spots, distance_zxy=_distance_zxy,
                                          dom_sz=dom_sz, gfilt_size=gfilt_size,
                                          normalization_matrix=normalization_matrix,
                                          domain_dist_metric=domain_dist_metric,
                                          domain_cluster_metric=domain_cluster_metric,
                                          corr_th=corr_th, dist_th=dist_th, plot_steps=False,
                                          plot_results=False, save_result_figs=False,
                                          save_folder=None, save_name='', verbose=verbose)
    ## 2. iteratively update domains
    for _i in range(int(num_iter)):
        cand_bd_ends = np.concatenate([cand_bd_starts[1:], np.array([len(spots)])])
        splitted_starts = cand_bd_starts.copy()
        for _j in range(int(split_level)):
            splitted_starts = list(splitted_starts)
            for _start, _end in zip(splitted_starts, cand_bd_ends):
                if (_end - _start) > 2 * dom_sz:
                    if _normalization:
                        new_bds = basic_domain_calling(spots[_start:_end],
                                                    normalization_matrix=norm_mat[_start:_end,
                                                                                    _start:_end],
                                                    dist_th=dist_th,
                                                    corr_th=corr_th*corr_th_scaling,
                                                    gfilt_size=gfilt_size,
                                                    domain_dist_metric=domain_dist_metric,
                                                    plot_results=False, verbose=False)
                    else:
                        new_bds = basic_domain_calling(spots[_start:_end],
                                                    normalization_matrix=None,
                                                    dist_th=dist_th,
                                                    corr_th=corr_th*corr_th_scaling,
                                                    gfilt_size=gfilt_size,
                                                    domain_dist_metric=domain_dist_metric,
                                                    plot_results=False, verbose=False)
                    # save new boundaries
                    splitted_starts += list(_start+new_bds)
            # summarize new boundaries
            splitted_starts = np.unique(splitted_starts).astype(np.int)
        # merge
        if _normalization:
            # no scaling for dist_th
            new_starts = merge_domains(_zxy, splitted_starts, norm_mat=norm_mat,
                                       corr_th=corr_th,
                                       dist_th=dist_th*dist_th_scaling, 
                                       domain_dist_metric=domain_dist_metric,
                                       plot_steps=False, verbose=verbose)
        else:
            new_starts = merge_domains(_zxy, splitted_starts, norm_mat=None,
                                       corr_th=corr_th,
                                       dist_th=dist_th*dist_th_scaling,
                                       domain_dist_metric=domain_dist_metric,
                                       plot_steps=False, verbose=verbose)
        # check if there is no change at all
        if len(new_starts) == len(cand_bd_starts) and (new_starts == cand_bd_starts).all():
            if verbose:
                print(f"-- iter {_i} finished, all boundaries are kept, exit!")
            break
        # else, update
        else:
            cand_bd_starts = new_starts
            if verbose:
                print(
                    f"-- iter {_i} finished, num of updated boundaries: {len(new_starts)}")

    ## 3. plot results
    if plot_results and len(cand_bd_starts) > 1:
        _dm_pdists = domain_pdists(_zxy, cand_bd_starts,
                                   metric=domain_dist_metric,
                                   normalization_mat=norm_mat,
                                   allow_minus_dist=False)
        _coef_mat = np.corrcoef(squareform(_dm_pdists))
        if verbose:
            print(
                f"-- make plot for results with {len(cand_bd_starts)} domains")
        import matplotlib.gridspec as gridspec
        _fig = plt.figure(figsize=(3*(fig_dim+1), 2*fig_dim), dpi=fig_dpi)
        gs = gridspec.GridSpec(2, 5, figure=_fig)
        gs.update(wspace=0.1, hspace=0)
        #ax1 = plt.subplot2grid((2, 5), (0, 0), colspan=1)
        ax1 = plt.subplot(gs[0, 0])
        ax1.set_title('Noramlized distances between domains',
                      fontsize=fig_font_size)
        im1 = ax1.imshow(squareform(_dm_pdists), cmap='seismic_r',
                         vmin=0, vmax=1)
        cb1 = plt.colorbar(
            im1, ax=ax1, ticks=np.arange(0, 1.2, 0.2), shrink=0.6)
        cb1.ax.tick_params(labelsize=fig_font_size-2, width=0.6, length=1)

        #ax2 = plt.subplot2grid((2, 5), (0, 1), colspan=1)
        ax2 = plt.subplot(gs[0, 1])
        ax2.set_title('Correlation matrix between domains',
                      fontsize=fig_font_size)
        im2 = ax2.imshow(_coef_mat, cmap='seismic')
        cb2 = plt.colorbar(im2, ax=ax2, shrink=0.6)
        cb2.ax.tick_params(labelsize=fig_font_size-2, width=0.6, length=1)

        #ax4 = plt.subplot2grid((2, 5), (1, 0), colspan=2, rowspan=1)
        ax4 = plt.subplot(gs[1, 0:2])
        ax4.set_title('Hierarchal clustering of domains',
                      fontsize=fig_font_size)
        # calculate hierarchy clusters
        _hierarchy_clusters = linkage(_dm_pdists, method=domain_cluster_metric)
        # draw dendrogram
        dn = dendrogram(_hierarchy_clusters, ax=ax4)
        ax4.tick_params(labelsize=fig_font_size)

        #ax3 = plt.subplot2grid((2, 5), (0, 2), colspan=3, rowspan=2)
        ax3 = plt.subplot(gs[:, 2:])
        plot_boundaries(squareform(pdist(_zxy)), cand_bd_starts, input_ax=ax3,
                        figure_dpi=fig_dpi,
                        figure_fontsize=fig_font_size, save=save_result_figs,
                        save_folder=save_folder, save_name=save_name)
        # save result figure
        if save_result_figs and save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            if save_name == '':
                _result_save_name = 'iterative_domain_calling.png'
            else:
                _result_save_name = save_name + '_iterative_domain_calling.png'
            _full_result_filename = os.path.join(
                save_folder, _result_save_name)
            if os.path.isfile(_full_result_filename):
                _full_result_filename.replace('.png', '_.png')
            if verbose:
                print(
                    f"--- save result image into file:{_full_result_filename}")
            plt.savefig(_full_result_filename, transparent=True)
        if __name__ == '__main__':
            plt.show()
    if verbose:
        print(
            f"-- total time for iterative domain calling: {time.time()-_start_time}")

    return cand_bd_starts


def local_domain_calling(spots, save_folder=None, metric='median',
                         distance_zxy=_distance_zxy, dom_sz=5, gfilt_size=0.5,
                         cutoff_max=0.5, hard_cutoff=2., plot_results=True,
                         fig_dpi=100,  fig_dim=10, fig_font_size=18,
                         save_result_figs=False, save_name='', verbose=True):
    """Wrapper for local domain calling in bogdan's code"""
    from ..External.DomainTools import standard_domain_calling_new
    ## 0. prepare coordinates
    if verbose:
        print(f"- start local domain calling")
        _start_time = time.time()
    _spots = np.array(spots)
    if _spots.shape[1] == 3:
        _zxy = _spots
    else:
        _zxy = np.array(_spots)[:, 1:4] * distance_zxy[np.newaxis, :]
    # smooth
    if gfilt_size is not None and gfilt_size > 0:
        if verbose:
            print(f"--- gaussian interpolate chromosome, sigma={gfilt_size}")
        _zxy = interpolate_chr(_zxy, gaussian=gfilt_size)
    # call bogdan's function
    cand_bd_starts =  standard_domain_calling_new(_zxy, gaussian=0., 
                                                  metric=metric,
                                                  dom_sz=dom_sz, 
                                                  cutoff_max=cutoff_max,
                                                  hard_cutoff=hard_cutoff)

    return cand_bd_starts




def Domain_Calling_Sliding_Window(coordinates, window_size=5, distance_metric='median',
                                  gaussian=0, normalization=r'Z:\References\normalization_matrix.npy',
                                  min_domain_size=4, min_prominence=0.25, reproduce_ratio=0.6,
                                  merge_candidates=True, corr_th=0.6, dist_th=0.2,
                                  merge_strength_th=1., return_strength=False,
                                  verbose=False):
    """Function to call domain candidates by sliding window across chromosome
    Inputs:
        coordnates: n-by-3 coordinates for a chromosome, or n-by-n distance matrix, np.ndarray
        window_size: size of sliding window for each half, the exact windows will be 1x to 2x of size, int
        distance_metric: type in distance metric in each sliding window, 
        gaussian: size of gaussian filter applied to coordinates, float (default: 0, no gaussian)
        normalization: normalization matrix / path to normalization matrix, np.ndarray or str
        min_domain_size: minimal domain size allowed in calling, int (default: 4)
        min_prominence: minimum prominence of peaks in distances called by sliding window, float (default: 0.25)
        reproduce_ratio: ratio of peaks found near the candidates across different window size, float (default: 0.6)
        merge_candidates: wheather merge candidate domains, bool (default:True)
        corr_th: min corrcoef threshold to merge domains, float (default: 0.6)
        dist_th: max distance threshold to merge domains, float (defaul: 0.2)
        merge_strength_th: min strength to not merge at all, float (default: 1.)     
        return_strength: return boundary strength generated sliding_window, bool (default: False)
        verbose: say something!, bool (default: False)
    Outputs:
        kept_domains: domain starts region-indices, np.ndarray
        kept_strengths (optional): kept domain boundary strength, np.ndarray
    """
    ## check inputs
    from .distance import _sliding_window_dist
    coordinates = np.array(coordinates).copy()
    if verbose:
        print(f"-- start sliding-window domain calling with", end=' ')
    if len(np.shape(coordinates)) != 2:
        raise ValueError(
            f"Wrong input shape for coordinates, should be 2d but {len(np.shape(coordinates))} is given")
    elif np.shape(coordinates)[0] == np.shape(coordinates)[1]:
        if verbose:
            print(f"distance map")
        if gaussian > 0:
            from astropy.convolution import Gaussian2DKernel, convolve
            _kernel = _kernel = Gaussian2DKernel(x_stddev=gaussian)
            coordinates = convolve(coordinates, _kernel)
        _mat = coordinates
    elif np.shape(coordinates)[1] == 3:
        if verbose:
            print(f"3d coordinates")
        if gaussian > 0:
            coordinates = interpolate_chr(coordinates, gaussian=gaussian)
        _mat = squareform(pdist(coordinates))
    else:
        raise ValueError(
            f"Input coordinates should be distance-matrix or 3d-coordinates!")
    window_size = int(window_size)

    # load normalization if specified
    if isinstance(normalization, str) and os.path.isfile(normalization):
        normalization = np.load(normalization)
    elif isinstance(normalization, np.ndarray) and np.shape(_mat)[0] == np.shape(normalization)[0]:
        pass
    else:
        normalization = None
    # do normalization if satisfied
    if normalization is not None:
        if verbose:
            print(f"--- applying normalization")
        _mat = _mat / normalization
    ## Start slide window to generate a vector for distance
    dist_list = []
    if verbose:
        print(
            f"--- calcualte distances with sliding window from {window_size} to {2*window_size-1}")
    # loop through real window_size between 1x to 2x of window_size
    for _wd in np.arange(window_size, 2*window_size):
        dist_list.append(_sliding_window_dist(_mat, _wd, distance_metric))
    #plt.figure(figsize=(15,5))
    #for dist in dist_list:
    #    plt.plot(np.arange(len(dist)),dist,alpha=0.5)
    #plt.show()
    ## call peaks
    if verbose:
        print(
            f"--- call peaks with minimum domain size={min_domain_size}, prominence={min_prominence}")
    peak_list = [scipy.signal.find_peaks(_dists, distance=min_domain_size,
                                         prominence=(min_prominence, None))[0] for _dists in dist_list]

    ## summarize peaks
    if verbose:
        print(
            f"--- summarize domains with {reproduce_ratio} reproducing rate.")
    cand_peaks = peak_list[0]
    _peak_coords = np.ones([len(peak_list), len(cand_peaks)]) * np.nan
    _peak_coords[0] = peak_list[0]
    _r = int(np.ceil(min_domain_size/2))
    for _i, _peaks in enumerate(peak_list[1:]):
        for _j, _p in enumerate(cand_peaks):
            _matched_index = np.where((_peaks >= _p-_r) * (_peaks <= _p+_r))[0]
            if len(_matched_index) > 0:
                # record whether have corresponding peaks in other window-size cases
                _peak_coords[_i+1, _j] = _peaks[_matched_index[0]]

    # only select peaks which showed up in more than reproduce_ratio*number_of_window cases
    _keep_flag = np.sum((np.isnan(_peak_coords) == False).astype(
        np.int), axis=0) >= reproduce_ratio*len(peak_list)
    # summarize selected peaks by mean of all summarized peaks
    sel_peaks = np.round(np.nanmean(_peak_coords, axis=0)
                         ).astype(np.int)[_keep_flag]
    # concatenate a zero
    domain_starts = np.concatenate([np.array([0]), sel_peaks])
    # calculate strength
    _strengths = np.nanmean([_dists[domain_starts]
                             for _dists in dist_list], axis=0)
    if verbose:
        print(f"--- domain called by sliding-window: {len(domain_starts)}")

    if merge_candidates:
        merged_starts = merge_domains(coordinates, domain_starts, 
                                      norm_mat=normalization, corr_th=corr_th,
                                      dist_th=dist_th, domain_dist_metric=distance_metric,
                                      plot_steps=False, verbose=False)    
        kept_domains = np.array([_d for _i,_d in enumerate(domain_starts)  
                                if _d in merged_starts or _strengths[_i] > merge_strength_th])
        if verbose:
            print(f"--- domain after merging: {len(kept_domains)}")
    else:
        kept_domains = domain_starts
        merged_starts = domain_starts

    # return_strength
    if return_strength:
        kept_strengths = np.array([_s for _i, _s in enumerate(_strengths)
                                    if domain_starts[_i] in merged_starts or _s > merge_strength_th])
        return kept_domains.astype(np.int), kept_strengths
    else:
        return kept_domains.astype(np.int)


def Batch_Domain_Calling_Sliding_Window(coordinate_list, window_size=5, distance_metric='median',
                                        num_threads=12, gaussian=0,
                                        normalization=r'Z:\References\normalization_matrix.npy',
                                        min_domain_size=4, min_prominence=0.25, reproduce_ratio=0.6,
                                        merge_candidates=True, corr_th=0.8, dist_th=0.2,
                                        merge_strength_th=1., return_strength=False,
                                        verbose=False):
    """Function to call domain candidates by sliding window across chromosome
    Inputs:
        coordinate_list: list of coordinates:
            n-by-3 coordinates for a chromosome, or n-by-n distance matrix, np.ndarray
        window_size: size of sliding window for each half, the exact windows will be 1x to 2x of size, int
        distance_metric: type in distance metric in each sliding window, 
        num_threads: number of threads to multiprocess domain calling, int (default: 12)
        gaussian: size of gaussian filter applied to coordinates, float (default: 0, no gaussian)
        normalization: normalization matrix / path to normalization matrix, np.ndarray or str
        min_domain_size: minimal domain size allowed in calling, int (default: 4)
        min_prominence: minimum prominence of peaks in distances called by sliding window, float (default: 0.25)
        reproduce_ratio: ratio of peaks found near the candidates across different window size, float (default: 0.6)
        merge_candidates: wheather merge candidate domains, bool (default:True)
        corr_th: min corrcoef threshold to merge domains, float (default: 0.6)
        dist_th: max distance threshold to merge domains, float (defaul: 0.2)
        merge_strength_th: min strength to not merge at all, float (default: 1.)     
        return_strength: return boundary strength generated sliding_window, bool (default: False)
        verbose: say something!, bool (default: False)
    Outputs:
        domain_start_list: list of domain start indices:
            domain starts region-indices, np.ndarray
        strength_list: list of strengths:
            (optional): kept domain boundary strength, np.ndarray
    """
    ## inputs
    if verbose:
        _start_time = time.time()
        print(f"- Start batch domain calling with sliding window.")
    # check coordinate_list
    if isinstance(coordinate_list, list) and len(np.shape(coordinate_list[0]))==2:
        pass
    elif isinstance(coordinate_list, np.ndarray) and len(np.shape(coordinate_list)) == 3:
        pass
    else:
        raise ValueError(f"Input coordinate_list should be a list of 2darray or 3dim merged coordinates")
    # load normalization if specified
    if isinstance(normalization, str) and os.path.isfile(normalization):
        normalization = np.load(normalization)
    elif isinstance(normalization, np.ndarray) and len(coordinate_list[0]) == np.shape(normalization)[0]:
        pass
    else:
        normalization = None
    # num_threads
    num_threads = int(num_threads)
    ## init
    domain_args = []
    # loop through coordinates
    for coordinates in coordinate_list:
        domain_args.append((coordinates, window_size, distance_metric,
                            gaussian, normalization,
                            min_domain_size, min_prominence, reproduce_ratio,
                            merge_candidates, corr_th, dist_th,
                            merge_strength_th, True, verbose))
    # multi-processing
    if verbose:
        print(
            f"-- multiprocessing of {len(domain_args)} domain calling with {num_threads} threads")
    with mp.Pool(num_threads) as domain_pool:
        results = domain_pool.starmap(
            Domain_Calling_Sliding_Window, domain_args)
        domain_pool.close()
        domain_pool.join()
        domain_pool.terminate()
    domain_start_list = [_r[0] for _r in results]

    if verbose:
        print(f"-- time spent in domain-calling:{time.time()-_start_time}")

    if return_strength:
        strength_list = [_r[1] for _r in results]
        return domain_start_list, strength_list
    else:
        return domain_start_list


def insulation_domain_calling(distmap, min_domain_size=5, window_size=None, 
                              use_distance=None, peak_kwargs={'prominence':0.03}, 
                              make_plot=False, verbose=True):
    """Function to use insulation to call domains, 
        * specifically designed for chr2 250kb DNA-FISH, probably called sub-compartments for these data
    Inputs:
        distmap: distance map for a chromosomal region, could be either distance or contact, np.ndarray(2d, symetrical)
        min_domain_size: minimal domain size in calling, int (default: 5)
        window_size: window size for sliding-window domain calling, int or None (if None, 2x min_domain_size)
        use_distance: whether using distance map or contact, bool or None (if None, automatically determined by insulation calculated)
        
    Outputs:
    """
    from .distance import _sliding_window_dist
    
    ## initalize parameters
    _dm_sz = int(min_domain_size)
    # window size should be 2xdomain-size
    if window_size is None:
        _wd_size = 2 * _dm_sz
    else:
        _wd_size = int(window_size)
    # peak key args defaults
    if 'distance' not in peak_kwargs:
        peak_kwargs['distance'] = _dm_sz-1
    if 'width' not in peak_kwargs:
        peak_kwargs['width'] = 1
    ## call local distances
    _dists = _sliding_window_dist(distmap, _wd=_wd_size, _dist_metric='insulation')
    # judge whether this is a distance matrix use_distance
    if use_distance is None:
        if np.nanmedian(_dists) < 0:
            use_distance = True
        else:
            use_distance = False
    # then call peaks
    if use_distance:
        _peaks = find_peaks(-_dists, **peak_kwargs)
    else:
        _peaks = find_peaks(_dists, **peak_kwargs)
    # extract peak locations and append a zero
    _domain_starts = np.concatenate([np.array([0]), _peaks[0]])
    _domain_starts = np.array(_domain_starts, dtype=np.int)
    # make plot
    if make_plot:
        from ..figure_tools import _dpi, _double_col_width, _single_col_width
        fig, ax = plt.subplots(figsize=(_double_col_width, _single_col_width),dpi=_dpi)
        if use_distance:
            ax.plot(-_dists, linewidth=1, color='b')
            ax.plot(_peaks[0], -_dists[_peaks[0]], 'r.')
        else:
            ax.plot(_dists, linewidth=1, color='b')
            ax.plot(_peaks[0], _dists[_peaks[0]], 'r.')
        ax.set_xlim([0, len(_dists)])
        plt.show()
    
    return _domain_starts


def merge_domain_by_contact_correlation(coordinates, domain_starts, contact_th=500, corr_th=0.8,
                                        plot_steps=False,):
    """Function to iteratively merge domains by contact correlation"""
    ## check inputs
    from .distance import _domain_contact_freq
    # convert _coordinates into matrix
    _coordinates = np.array(coordinates)
    if _coordinates.shape[0] != _coordinates.shape[1]:
        _coordinates = squareform(pdist(_coordinates))
    # _domain_starts
    _domain_starts = np.sort(domain_starts).astype(np.int)
    if 0 not in _domain_starts:
        _domain_starts = np.concatenate([np.array([0]), _domain_starts])
    if len(_coordinates) in _domain_starts:
        _domain_starts = np.setdiff1d(_domain_starts, [len(_coordinates)])
    ## merge domains
    
    _dm_cfreq = _domain_contact_freq(_coordinates, _domain_starts, contact_th)
    _dm_corrs = np.diag(_dm_cfreq,1)
    _ct = 0
    
    while (_dm_corrs>corr_th).any():
        _ct += 1
        _sel_id = np.argmax(_dm_corrs)+1
        _domain_starts = np.delete(_domain_starts, _sel_id)
        # update domain contact correlation
        _dm_cfreq = _domain_contact_freq(_coordinates, _domain_starts, contact_th)
        _dm_corrs = np.diag(_dm_cfreq,1)
        if plot_steps:
            plt.figure(figsize=(4,3),dpi=100)
            plt.imshow(_dm_cfreq, cmap='seismic')
            plt.colorbar()
            plt.title(f"i={_ct}")
            plt.show()
    return _domain_starts

def contact_correlation_domain_calling(zxys, remove_outlier_th=750, domain_size=5, 
                                       cand_domain_th=0.3, contact_th=500, corr_th=0.5):
    """"""
    ## check inputs
    from .distance import _neighboring_distance, _sliding_window_dist      
    _zxys = np.array(zxys)
    # remove nans
    _good_inds = np.where(np.isnan(_zxys).sum(1)==0)[0]
    _good_zxys = _zxys[_good_inds]
    # remove outliers
    _neighboring_dists = _neighboring_distance(_good_zxys)
    _outlier_inds = scipy.signal.find_peaks(_neighboring_dists, prominence=remove_outlier_th)[0]
    _kept_inds = np.setdiff1d(np.arange(len(_good_zxys)), _outlier_inds)
    _kept_zxys = _good_zxys[_kept_inds]
    
    # call candidate domains
    _med_dists = _sliding_window_dist(squareform(pdist(_kept_zxys)), domain_size)
    _cand_dm_starts = scipy.signal.find_peaks(_med_dists, prominence=cand_domain_th,
                                              distance=domain_size/2)[0]
    # merge domain with contact correlation
    
    _kept_starts = merge_domain_by_contact_correlation(_kept_zxys, _cand_dm_starts, 
                                                       contact_th=contact_th, corr_th=corr_th)
    _good_starts = _kept_inds[_kept_starts]
    _dm_starts = _good_inds[_good_starts]
    
    return _dm_starts
