import sys,os,re,time,glob
import numpy as np
import pickle as pickle
import matplotlib 
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pylab as plt
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
from matplotlib import cm 
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

from . import _distance_zxy,_sigma_zxy,_allowed_colors
from . import _ticklabel_size, _ticklabel_width, _font_size,_dpi, _single_col_width, _single_row_height,_double_col_width
from scipy.stats import linregress
#from astropy.convolution import Gaussian2DKernel,convolve



## Plotting function
def plot_boundary_probability(region_ids, domain_start_list, figure_kwargs={}, plot_kwargs={},
                              xlabel="region_ids", ylabel="probability", fontsize=16,
                              save=False, save_folder='.', save_name=''):
    """Wrapper function to plot boundary probability given domain_start list"""
    if 'plt' not in locals():
        import matplotlib.pyplot as plt
    # summarize
    _x = np.array(region_ids, dtype=np.int)
    _y = np.zeros(np.shape(_x), dtype=np.float)
    for _dm_starts in domain_start_list:
        for _d in _dm_starts:
            if _d > 0 and _d in _x:
                _y[np.where(_x == _d)[0]] += 1
    _y = _y / len(domain_start_list)
    _fig, _ax = plt.subplots(figsize=(15, 5), dpi=200, **figure_kwargs)
    _ax.plot(_x, _y, label=ylabel, **plot_kwargs)
    _ax.set_xlim([0, len(_x)])
    _ax.set_xlabel(xlabel, fontsize=fontsize)
    _ax.set_ylabel(ylabel, fontsize=fontsize)
    plt.legend()
    if save:
        _filename = 'boundary_prob.png'
        if save_name != '':
            _filename = save_name + '_' + _filename
        plt.savefig(os.path.join(save_folder, _filename), transparent=True)
    return _ax

def plot_boundaries(distance_map, boundaries, input_ax=None, plot_limits=[0, 1500],
                    line_width=1.5, figure_dpi=200, figure_fontsize=20, figure_cmap='seismic_r', title='',
                    save=False, save_folder=None, save_name=''):
    boundaries = list(boundaries)
    if 0 not in boundaries:
        boundaries = [0] + boundaries
    if len(distance_map) not in boundaries:
        boundaries += [len(distance_map)]
    # sort
    boundaries = sorted([int(_b) for _b in boundaries])
    if input_ax is None:
        fig = plt.figure(dpi=figure_dpi)
        ax = plt.subplot(1, 1, 1)
    else:
        ax = input_ax
    im = ax.imshow(distance_map, cmap=figure_cmap,
                   vmin=min(plot_limits), vmax=max(plot_limits))
    plt.subplots_adjust(left=0.02, bottom=0.06,
                        right=0.95, top=0.94, wspace=0.05)
    if input_ax is None:
        cb = plt.colorbar(im, ax=ax)
    else:
        cb = plt.colorbar(im, ax=ax, shrink=0.75)
        cb.ax.tick_params(labelsize=figure_fontsize)
        ax.tick_params(labelsize=figure_fontsize)
        ax.yaxis.set_ticklabels([])
        line_width *= 2
    for _i in range(len(boundaries)-1):
        ax.plot(np.arange(boundaries[_i], boundaries[_i+1]), boundaries[_i]*np.ones(
            boundaries[_i+1]-boundaries[_i]), 'y', linewidth=line_width)
        ax.plot(boundaries[_i]*np.ones(boundaries[_i+1]-boundaries[_i]),
                np.arange(boundaries[_i], boundaries[_i+1]), 'y', linewidth=line_width)
        ax.plot(np.arange(boundaries[_i], boundaries[_i+1]), boundaries[_i+1]*np.ones(
            boundaries[_i+1]-boundaries[_i]), 'y', linewidth=line_width)
        ax.plot(boundaries[_i+1]*np.ones(boundaries[_i+1]-boundaries[_i]),
                np.arange(boundaries[_i], boundaries[_i+1]), 'y', linewidth=line_width)
    ax.set_xlim([0, distance_map.shape[0]])
    ax.set_ylim([distance_map.shape[1], 0])
    if title != '':
        ax.set_title(title, pad=1)
    if save:
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            if save_name == '':
                save_name = 'boundaries.png'
            else:
                if '.png' not in save_name:
                    save_name += '_boundaries.png'
            fig.savefig(os.path.join(save_folder, save_name), transparent=True)

    return ax

def plot_domain_in_distmap(distmap, domain_starts, ax=None, 
                           cmap='seismic_r', color_limits=[0,1500], color_norm=None, imshow_kwargs={},
                           domain_color=[1,1,0], domain_line_width=0.75, ticks=None, tick_labels=None, 
                           tick_label_length=_ticklabel_size, tick_label_width=_ticklabel_width, 
                           font_size=_font_size, ax_label=None,
                           add_colorbar=True, colorbar_labels=None,
                           figure_width=_single_col_width, figure_dpi=_dpi, 
                           save=False, save_folder='.', save_basename='', verbose=True):
    """Function to plot domains in distance map"""
    
    ## check inputs
    # distmap
    if np.shape(distmap)[0] != np.shape(distmap)[1]:
        raise IndexError(f"Wrong input dimension for distmap, should be nxn matrix but {distmap.shape} is given")
    _distmap = distmap.copy()
    _distmap[_distmap<min(color_limits)] = min(color_limits)
    # domain starts
    domain_starts = np.array(domain_starts, dtype=np.int)
    if 0 not in domain_starts:
        domain_starts = np.concatenate([np.array([0]), domain_starts]).astype(np.int)
    # domain ends
    domain_ends = np.concatenate([domain_starts[1:], np.array([len(distmap)])]).astype(np.int)

    
    ## create image
    if ax is None:
        fig, ax = plt.subplots(figsize=(figure_width, figure_width),
                               dpi=figure_dpi)
    
    # plot background distmap
    from .distmap import plot_distance_map
    ax = plot_distance_map(_distmap, ax=ax, cmap=cmap, 
                           color_limits=color_limits, color_norm=color_norm, imshow_kwargs=imshow_kwargs,
                           ticks=ticks, tick_labels=tick_labels,
                           tick_label_length=tick_label_length, tick_label_width=tick_label_width,
                           font_size=font_size, ax_label=ax_label,
                           add_colorbar=add_colorbar, colorbar_labels=colorbar_labels, 
                           figure_width=figure_width, figure_dpi=figure_dpi,
                           save=False, verbose=verbose)
    for _start, _end in zip(domain_starts, domain_ends):
        ax.plot(np.arange(_start, _end+1), _start*np.ones(
            _end+1-_start), color=domain_color, linewidth=domain_line_width)
        ax.plot(_start*np.ones(_end+1-_start),
                np.arange(_start, _end+1), color=domain_color, linewidth=domain_line_width)
        ax.plot(np.arange(_start, _end+1), _end*np.ones(
            _end+1-_start), color=domain_color, linewidth=domain_line_width)
        ax.plot(_end*np.ones(_end+1-_start),
                np.arange(_start, _end+1), color=domain_color, linewidth=domain_line_width)
    ax.set_xlim([0, distmap.shape[0]-0.5])
    ax.set_ylim([distmap.shape[1]-0.5, 0])
    
    if save:
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            if save_basename == '':
                save_basename = 'boundaries.png'
            else:
                if '.png' not in save_basename and '.pdf' not in save_basename:
                    save_basename += '_boundaries.png'
            fig.savefig(os.path.join(save_folder, save_basename), transparent=True)

    return ax